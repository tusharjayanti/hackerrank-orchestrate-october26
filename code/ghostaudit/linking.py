"""Resource-to-user linking ladder.

Each resource keeps the strongest layer that linked it. Layers 1-4 are facts
found in evidence; layer 5 is an inference and is only adopted when exactly
one candidate survives elimination.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from .ingest.activity import Activity
from .ingest.observations import ObservationTable
from .obs import get_logger
from .schemas import LINK_CONFIDENCE, Link, LinkLayer, case_sort_key

log = get_logger(__name__)

PROXIMITY_WINDOWS = (timedelta(minutes=90),)
SIBLING_WINDOW = timedelta(minutes=180)
# The worker result a case's own account delete carries, by reported_status
# (16/16 SUCCESS for COMPLETE, 10/10 FAILED for NOT_STARTED on the samples).
# reported_status tells us which worker run belonged to this request, and the
# evidence tells us what state that run left behind.
RESULT_FOR_STATUS: dict[str, frozenset[str]] = {
    "COMPLETE": frozenset({"SUCCESS"}),
    "NOT_STARTED": frozenset({"FAILED"}),
    "PARTIALLY_COMPLETE": frozenset({"PARTIAL", "SUCCESS"}),
}


@dataclass(frozen=True)
class CaseContext:
    case_id: str
    user_id: str
    requested_at: datetime
    reported_at: datetime
    audit_at: datetime
    reported_status: str

Res = tuple[str, str]


class Linker:
    def __init__(self, table: ObservationTable, activity: Activity, vision_links: list[Link] | None = None,
                 enable_sibling: bool = False):
        # Layer 6 measured 1/3 on the samples (it attaches analytics records to
        # cases that have none), so it is off unless asked for explicitly.
        self.enable_sibling = enable_sibling
        self.table = table
        self.activity = activity
        self.vision_links = vision_links or []
        self._deletes = activity.primary_profile_deletes()
        self._anonymizations = [e for e in activity.events if e.event_type == "ANONYMIZE_RECORD"
                                and e.source_system == "deletion_worker" and e.target_system == "analytics_platform"]

    def _link(self, uid: str, res: Res, layer: LinkLayer, source_id: str, at: datetime) -> Link:
        return Link(user_id=uid, system_id=res[0], resource_id=res[1], layer=layer,
                    confidence=LINK_CONFIDENCE[layer], source_id=source_id, linked_at=at)

    def fact_links(self, uid: str, audit_at: datetime) -> dict[Res, Link]:
        """Layers 1-4 for one user at audit_at."""
        found: dict[Res, Link] = {}

        def keep(link: Link) -> None:
            res = (link.system_id, link.resource_id)
            cur = found.get(res)
            # Ties broken on the source id, never on iteration order.
            if cur is None or (link.layer, link.linked_at, link.source_id) < (cur.layer, cur.linked_at,
                                                                             cur.source_id):
                found[res] = link

        for o in self.table.by_user.get(uid, ()):
            res = (o.system_id, o.resource_id)
            if o.captured_at <= audit_at:
                layer = LinkLayer.VISION if o.vision else LinkLayer.IDENTIFIER
                keep(self._link(uid, res, layer, o.file_id, o.observed_at))
            elif o.observed_at <= audit_at:
                # The snapshot existed at audit time; only its export is late.
                # Usable to link, never to cite.
                keep(self._link(uid, res, LinkLayer.INNER_SNAPSHOT, o.file_id, o.observed_at))
        for lf in self.activity.links:
            if lf.user_id == uid and lf.event_time <= audit_at:
                keep(self._link(uid, (lf.system_id, lf.resource_id), LinkLayer.CATALOG, lf.activity_id,
                                lf.event_time))
        for vl in self.vision_links:
            if vl.user_id == uid and vl.linked_at <= audit_at:
                keep(vl)
        return found

    def owners(self, res: Res, audit_at: datetime) -> set[str]:
        """Users that layers 1-4 tie to a resource at audit_at."""
        out = set()
        for o in self.table.by_resource.get(res, ()):
            if o.observed_at <= audit_at:
                out |= self.table.owners(o)
        out |= {lf.user_id for lf in self.activity.links
                if (lf.system_id, lf.resource_id) == res and lf.event_time <= audit_at}
        out |= {vl.user_id for vl in self.vision_links
                if (vl.system_id, vl.resource_id) == res and vl.linked_at <= audit_at}
        return out

    def prepare(self, cases: list[CaseContext]) -> None:
        """Solve layers 5 and 6 jointly over every known request, so that an
        inference made for one case can eliminate a candidate for another."""
        self._inferred = {}
        ctx = {c.case_id: c for c in cases}
        facts = {c.case_id: self.fact_links(c.user_id, c.audit_at) for c in cases}
        taken: dict[Res, str] = {}  # resource -> user it was inferred for

        def free(c: CaseContext, res: Res) -> bool:
            if self.owners(res, c.audit_at) - {c.user_id}:
                return False
            return taken.get(res, c.user_id) == c.user_id

        changed = True
        while changed:
            changed = False
            for cid in sorted(ctx, key=case_sort_key):
                c = ctx[cid]
                got = self._inferred.setdefault(cid, {})
                has_account = any(r[0] == "account_service" for r in (*facts[cid], *got))
                if not has_account:
                    link = self._proximity(c, lambda res: free(c, res))
                    if link:
                        got[(link.system_id, link.resource_id)] = link
                        taken[(link.system_id, link.resource_id)] = c.user_id
                        changed = True
                        has_account = True
                has_analytics = any(r[0] == "analytics_platform" for r in (*facts[cid], *got))
                if self.enable_sibling and has_account and not has_analytics:
                    account = min(r for r in (*facts[cid], *got) if r[0] == "account_service")
                    link = self._sibling(c, account[1], lambda res: free(c, res))
                    if link:
                        got[(link.system_id, link.resource_id)] = link
                        taken[(link.system_id, link.resource_id)] = c.user_id
                        changed = True

    def _proximity(self, c: CaseContext, is_free) -> Link | None:
        """Layer 5: the primary-profile delete whose worker result matches the
        case's reported status and that nobody else owns. The narrow window is
        tried first; the wide one only when the narrow one has no candidate."""
        results = RESULT_FOR_STATUS.get(c.reported_status)
        for window in PROXIMITY_WINDOWS:
            near = [e for e in self._deletes
                    if abs(e.event_time - c.reported_at) <= window and e.event_time <= c.audit_at
                    and (results is None or e.reported_result in results)]
            cands = sorted({e.resource_id for e in near if is_free(("account_service", e.resource_id))})
            if not cands:
                continue
            if len(cands) > 1:
                return None
            event = min((e for e in near if e.resource_id == cands[0]),
                        key=lambda e: (abs(e.event_time - c.reported_at), e.activity_id))
            return self._link(c.user_id, ("account_service", cands[0]), LinkLayer.PROXIMITY, event.activity_id,
                              event.event_time)
        return None

    def _sibling(self, c: CaseContext, account_id: str, is_free) -> Link | None:
        """Layer 6: the analytics anonymization from the same worker run as the
        user's account deletion, when exactly one free candidate exists."""
        deletes = [e for e in self._deletes if e.resource_id == account_id
                   and abs(e.event_time - c.reported_at) <= PROXIMITY_WINDOWS[-1]]
        if not deletes:
            return None
        start = min(deletes, key=lambda e: (abs(e.event_time - c.reported_at), e.activity_id)).event_time
        cands = sorted({e.resource_id for e in self._anonymizations
                        if start <= e.event_time <= start + SIBLING_WINDOW and e.event_time <= c.audit_at
                        and is_free(("analytics_platform", e.resource_id))})
        if len(cands) != 1:
            return None
        event = min((e for e in self._anonymizations if e.resource_id == cands[0] and e.event_time >= start),
                    key=lambda e: (e.event_time, e.activity_id))
        return self._link(c.user_id, ("analytics_platform", cands[0]), LinkLayer.SIBLING, event.activity_id,
                          event.event_time)

    def links(self, c: CaseContext, max_layer: LinkLayer = LinkLayer.SIBLING) -> dict[Res, Link]:
        found = {r: lk for r, lk in self.fact_links(c.user_id, c.audit_at).items() if lk.layer <= max_layer}
        for res, lk in getattr(self, "_inferred", {}).get(c.case_id, {}).items():
            if lk.layer <= max_layer and res not in found:
                found[res] = lk
        return found
