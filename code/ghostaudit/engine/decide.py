"""Per-case pipeline: resolve -> link -> scope -> state_at -> policy_diff -> build_steps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .. import DATASET_DIR
from ..identity import IdentityIndex
from ..ingest.activity import Activity
from ..ingest.manifest import Manifest, parse_ts
from ..ingest.observations import ObservationTable, build_table, category_for
from ..linking import Linker
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
        table = build_table(manifest, identity, formats)
        activity = Activity.load(dataset_dir)
        return cls(identity, manifest, table, activity, Linker(table, activity), load_rules(dataset_dir))


def decide_case(world: World, request: dict, max_layer: LinkLayer = LinkLayer.PROXIMITY) -> tuple[Decision, dict]:
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
    links = world.linker.links(uid, audit_at, reported_at, max_layer)
    trace["links"] = [lk.model_dump(mode="json") for lk in sorted(links.values(), key=lambda x: (x.system_id,
                                                                                                    x.resource_id))]
    if any(lk.layer is LinkLayer.PROXIMITY for lk in links.values()):
        flags.append("PROXIMITY_LINK")

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
