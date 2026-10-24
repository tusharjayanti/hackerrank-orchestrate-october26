"""Per-case pipeline: resolve -> link -> scope -> state_at -> policy_diff -> build_steps."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .. import DATASET_DIR
from ..identity import IdentityIndex
from ..ingest.activity import Activity
from ..ingest.manifest import Manifest, parse_ts
from ..ingest.observations import ObservationTable, build_table, category_for
from ..linking import CaseContext, Linker
from ..llm import LLM, credentials_present
from ..obs import get_logger
from ..schemas import DataCategory, Decision, DeletionStatus, LinkLayer, RequiredAction
from .policy import load_rules, policy_diff
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

    @classmethod
    def load(cls, dataset_dir: Path = DATASET_DIR, formats: set[str] | None = None) -> World:
        identity = IdentityIndex.load(dataset_dir)
        manifest = Manifest.load(dataset_dir)
        activity = Activity.load(dataset_dir)
        # Screenshots need the model; without credentials they are deferred
        # and the run is engine-only for them (PDFs are read without a model).
        llm = LLM() if credentials_present() else None
        table = build_table(manifest, identity, formats, activity=activity, llm=llm)
        linker = Linker(table, activity)
        # Layers 5-6 eliminate across cases, so they are solved once over every
        # participant-facing request (evaluation and sample sets share batches).
        contexts = []
        for name in ("deletion_requests.csv", "sample_deletion_requests.csv"):
            with open(dataset_dir / name, newline="", encoding="utf-8") as f:
                contexts.extend(c for c in (case_context(identity, r) for r in csv.DictReader(f)) if c)
        linker.prepare(contexts)
        return cls(identity, manifest, table, activity, linker, load_rules(dataset_dir))


def case_context(identity: IdentityIndex, request: dict) -> CaseContext | None:
    uid = identity.resolve_email(request["email"]).user_id
    if uid is None:
        return None
    return CaseContext(request["case_id"], uid, parse_ts(request["requested_at"]), parse_ts(request["reported_at"]),
                       parse_ts(request["audit_at"]), request["reported_status"])


def decide_case(world: World, request: dict, max_layer: LinkLayer = LinkLayer.SIBLING) -> tuple[Decision, dict]:
    case_id = request["case_id"]
    requested_at = parse_ts(request["requested_at"])
    reported_at = parse_ts(request["reported_at"])
    audit_at = parse_ts(request["audit_at"])
    flags: list[str] = []
    trace: dict = {"case_id": case_id, "audit_at": request["audit_at"]}

    resolution = world.identity.resolve_email(request["email"])
    trace["identity"] = {"user_id": resolution.user_id, "match": resolution.kind.value}
    if resolution.user_id is None:
        # Fallback: an unresolvable subject has nothing provable; report an
        # empty COMPLETE would claim verification we do not have.
        log.warning("case_identity_unresolved", case_id=case_id)
        flags.append("IDENTITY_UNRESOLVED")
        return Decision(case_id=case_id, status=DeletionStatus.INCOMPLETE, steps=(), flags=tuple(flags)), trace

    uid = resolution.user_id
    links = world.linker.links(case_context(world.identity, request), max_layer)
    trace["links"] = [lk.model_dump(mode="json") for lk in sorted(links.values(), key=lambda x: (x.system_id,
                                                                                                    x.resource_id))]
    for lk in links.values():
        if lk.layer >= LinkLayer.PROXIMITY:
            flags.append(f"INFERRED_LINK_L{int(lk.layer)}:{lk.resource_id}")

    resources = []
    trace["out_of_scope"] = []
    for res in sorted(links):
        if out_of_scope(res, world.table, requested_at, audit_at, links[res].linked_at):
            trace["out_of_scope"].append(f"{res[0]}/{res[1]}")
            continue
        st = state_at(res, world.table, audit_at)
        category = st.data_category if st and st.data_category else category_for(res[0])
        resources.append((res[0], res[1], category, st))
        if st and st.conflicting:
            flags.append(f"STATE_CONFLICT:{res[1]}")
    treatments = policy_diff(resources, world.rules)
    trace["treatments"] = [
        {"resource": f"{t.system_id}/{t.resource_id}", "category": str(t.category),
         "state": t.state.state.value if t.state else None, "evidence": t.state.evidence_id if t.state else None,
         "finding": t.finding, "remediation": t.remediation, "verification": t.verification, "note": t.note}
        for t in treatments]
    if not treatments:
        flags.append("NO_RESOURCES")
    status, steps = build_steps(treatments)
    if status is DeletionStatus.COMPLETE and any(t.state is None for t in treatments):
        flags.append("UNVERIFIABLE_RESOURCE")
    trace["observations"] = [o.model_dump(mode="json") for o in world.table.case_observations(uid, audit_at)]
    trace["flags"] = flags
    return Decision(case_id=case_id, status=status, steps=tuple(steps), flags=tuple(flags)), trace
