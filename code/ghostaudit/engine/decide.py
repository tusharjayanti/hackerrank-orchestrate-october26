"""Per-case pipeline: resolve -> link -> scope -> state_at -> policy_diff -> build_steps."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from .. import DATASET_DIR
from ..agent.loop import agent_trace, review_case
from ..identity import IdentityIndex
from ..ingest.activity import Activity
from ..ingest.manifest import Manifest, parse_ts
from ..ingest.observations import ObservationTable, build_table, category_for
from ..intent import Classified
from ..linking import CaseContext, Linker
from ..llm import LLM, credentials_present
from ..obs import get_logger
from ..schemas import DataCategory, Decision, DeletionStatus, Flag, FlagSet, LinkLayer, RequiredAction
from .policy import Scoped, load_rules, policy_diff, retention_expired, shared_attachment
from .state import out_of_scope, state_at
from .steps import build_steps

log = get_logger(__name__)


@dataclass
class World:
    identity: IdentityIndex
    manifest: Manifest
    table: ObservationTable
    activity: Activity
    linker: Linker
    rules: dict[DataCategory, RequiredAction]
    llm: LLM | None = None
    requests: dict[str, dict] = field(default_factory=dict)  # every participant request by case_id

    @classmethod
    def load(cls, dataset_dir: Path = DATASET_DIR, formats: set[str] | None = None, run_id: str = "") -> World:
        identity = IdentityIndex.load(dataset_dir)
        manifest = Manifest.load(dataset_dir)
        activity = Activity.load(dataset_dir)
        # Screenshots need the model; without credentials they are deferred
        # and the run is engine-only for them (PDFs are read without a model).
        llm = LLM(run_id=run_id) if credentials_present() else None
        table = build_table(manifest, identity, formats, activity=activity, llm=llm)
        linker = Linker(table, activity)
        # Layers 5-6 eliminate across cases, so they are solved once over every
        # participant-facing request (evaluation and sample sets share batches).
        requests: dict[str, dict] = {}
        for name in ("deletion_requests.csv", "sample_deletion_requests.csv"):
            with open(dataset_dir / name, newline="", encoding="utf-8") as f:
                requests.update((r["case_id"], r) for r in csv.DictReader(f))
        linker.prepare([c for c in (case_context(identity, r) for r in requests.values()) if c])
        return cls(identity, manifest, table, activity, linker, load_rules(dataset_dir), llm, requests)


def case_context(identity: IdentityIndex, request: dict) -> CaseContext | None:
    uid = identity.resolve_email(request["email"]).user_id
    if uid is None:
        return None
    return CaseContext(request["case_id"], uid, parse_ts(request["requested_at"]), parse_ts(request["reported_at"]),
                       parse_ts(request["audit_at"]), request["reported_status"])


def decide_case(world: World, request: dict, max_layer: LinkLayer = LinkLayer.SIBLING,
                classified: Classified | None = None) -> tuple[Decision, dict]:
    case_id = request["case_id"]
    audit_at = parse_ts(request["audit_at"])
    flags = FlagSet()
    notes: list[str] = []
    trace: dict = {"case_id": case_id, "audit_at": request["audit_at"], "notes": notes, "agent": agent_trace()}

    def done(status: DeletionStatus, steps: tuple = ()) -> tuple[Decision, dict]:
        trace.update(flags.trace())
        return Decision(case_id=case_id, status=status, steps=tuple(steps), flags=flags.hits), trace

    _run_flags(world, audit_at, flags)
    if classified is not None and not _intent_gate(classified, flags, trace):
        return done(DeletionStatus.NOT_APPLICABLE)

    resolution = world.identity.resolve_email(request["email"])
    trace["identity"] = {"user_id": resolution.user_id, "match": resolution.kind.value}
    if resolution.user_id is None:
        # Fallback: an unresolvable subject has no evidence to report on; the
        # row is COMPLETE with no steps and marked for review (no invented
        # finding, no missing row).
        log.warning("case_identity_unresolved", case_id=case_id)
        flags.add(Flag.AMBIGUOUS_IDENTIFIER if world.identity.is_ambiguous(request["email"])
                  else Flag.UNRESOLVED_IDENTITY, request["case_id"])
        flags.add(Flag.NEEDS_REVIEW, "no subject")
        return done(DeletionStatus.COMPLETE)

    resources = _scoped_resources(world, request, resolution.user_id, max_layer, flags, notes, trace)
    treatments, unmapped = policy_diff(resources, world.rules)
    for u in unmapped:
        flags.add(Flag.UNMAPPED_CATEGORY, u)
    for r in resources:
        if r.shared:
            flags.add(Flag.SHARED_ATTACHMENT, f"{r.system_id}/{r.resource_id} left in place")
    trace["treatments"] = [_treatment_trace(t) for t in treatments]
    status, steps = build_steps(treatments)
    if not treatments:
        notes.append("no resource with a policy treatment")
    if status is DeletionStatus.COMPLETE and any(t.state is None for t in treatments):
        notes.append("linked resource without citable state left out of COMPLETE verification")
    if status is DeletionStatus.COMPLETE and not steps:
        # Agreed fallback: nothing citable was found and no finding is
        # invented; the row is COMPLETE with no steps and marked for review.
        flags.add(Flag.NEEDS_REVIEW, "no evidence-backed step")
    case_obs = world.table.case_observations(resolution.user_id, audit_at)
    for o in case_obs:
        if o.source_detail == "png:disagreed":
            flags.add(Flag.VISION_DISAGREED, f"{o.file_id}:{o.resource_id}")
    trace["observations"] = [o.model_dump(mode="json") for o in case_obs]
    # L2: gated cases go to the investigator agent, which may only choose
    # between engine-generated candidates; trace["agent"] is set either way.
    status, steps = review_case(world, request, max_layer, status, steps, flags, trace)
    return done(status, steps)


def _run_flags(world: World, audit_at, flags: FlagSet) -> None:
    """Conditions of the run that weaken any case audited after them."""
    for st in world.table.stats:
        f = world.manifest.by_id.get(st.file_id)
        if f is None or f.captured_at > audit_at:
            continue
        if st.status == "failed":
            flags.add(Flag.PARSE_FAILED, st.file_id)
        elif st.status == "deferred" and st.format == "png":
            flags.add(Flag.VISION_UNAVAILABLE, st.file_id)
    if world.llm is not None and world.llm.budget.tripped:
        flags.add(Flag.BUDGET_BREACHED)


def _intent_gate(classified: Classified, flags: FlagSet, trace: dict) -> bool:
    """Record the intent; True when the request authorizes deletion."""
    flags.extend(classified.flags)
    trace["intent"] = {"label": classified.intent.label.value, "raw_label": classified.raw_label,
                       "second_label": classified.second_label, "confidence": classified.intent.confidence,
                       "language": classified.language, "injection_suspected": classified.intent.injection_suspected}
    return classified.intent.authorizes_deletion


def _scoped_resources(world: World, request: dict, uid: str, max_layer: LinkLayer, flags: FlagSet,
                      notes: list[str], trace: dict) -> list[Scoped]:
    """Linked resources in this request's scope, each with its state at audit_at
    and the two policy conditions read from evidence (retention date passed,
    attachment shared with another subject). Sorted by (system_id, resource_id)."""
    requested_at, audit_at = parse_ts(request["requested_at"]), parse_ts(request["audit_at"])
    links = world.linker.links(case_context(world.identity, request), max_layer)
    trace["links"] = [lk.model_dump(mode="json") for lk in sorted(links.values(),
                                                                   key=lambda x: (x.system_id, x.resource_id))]
    for o in world.table.held_for_user(uid, audit_at):
        flags.add(Flag.VISION_ONLY_PENDING, f"{o.file_id}:{o.resource_id}")
    notes.extend(f"inferred link L{int(lk.layer)} {lk.resource_id} via {lk.source_id}"
                 for lk in sorted(links.values(), key=lambda x: x.resource_id) if lk.layer >= LinkLayer.PROXIMITY)
    resources = []
    trace["out_of_scope"] = []
    trace["state_conflicts"] = []
    for res in sorted(links):
        if out_of_scope(res, world.table, requested_at, audit_at, links[res].linked_at):
            trace["out_of_scope"].append(f"{res[0]}/{res[1]}")
            continue
        st = state_at(res, world.table, audit_at)
        category = st.data_category if st and st.data_category else category_for(res[0])
        eligible = world.table.eligible(res, audit_at)
        expired = retention_expired(eligible, audit_at)
        shared = category is DataCategory.SUPPORT_ATTACHMENT and shared_attachment(eligible,
                                                                                  world.identity.owner_of)
        if expired:
            notes.append(f"retention date before audit_at: {res[0]}/{res[1]}")
        resources.append(Scoped(res[0], res[1], category, st, expired, shared))
        if st and st.conflicting:
            trace["state_conflicts"].append(f"{res[0]}/{res[1]}")
    return resources


def _treatment_trace(t) -> dict:
    return {"resource": f"{t.system_id}/{t.resource_id}", "category": str(t.category),
            "state": t.state.state.value if t.state else None, "evidence": t.state.evidence_id if t.state else None,
            "finding": t.finding, "remediation": t.remediation, "verification": t.verification, "note": t.note}
