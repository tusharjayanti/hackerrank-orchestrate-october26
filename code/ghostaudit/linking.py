"""Resource-to-user linking ladder.

Each resource keeps the strongest layer that linked it. Layers 1-4 are facts
found in evidence; layer 5 is an inference and is only adopted when exactly
one candidate survives elimination.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from .ingest.activity import Activity
from .ingest.observations import ObservationTable
from .obs import get_logger
from .schemas import LINK_CONFIDENCE, Link, LinkLayer

log = get_logger(__name__)

PROXIMITY_WINDOW = timedelta(minutes=10)

Res = tuple[str, str]


class Linker:
    def __init__(self, table: ObservationTable, activity: Activity, vision_links: list[Link] | None = None):
        self.table = table
        self.activity = activity
        self.vision_links = vision_links or []
        self._deletes = activity.primary_profile_deletes()

    def _link(self, uid: str, res: Res, layer: LinkLayer, source_id: str, at: datetime) -> Link:
        return Link(user_id=uid, system_id=res[0], resource_id=res[1], layer=layer,
                    confidence=LINK_CONFIDENCE[layer], source_id=source_id, linked_at=at)

    def fact_links(self, uid: str, audit_at: datetime) -> dict[Res, Link]:
        """Layers 1-4 for one user at audit_at."""
        found: dict[Res, Link] = {}

        def keep(link: Link) -> None:
            res = (link.system_id, link.resource_id)
            cur = found.get(res)
            if cur is None or (link.layer, link.linked_at) < (cur.layer, cur.linked_at):
                found[res] = link

        for o in self.table.by_user.get(uid, ()):
            res = (o.system_id, o.resource_id)
            if o.captured_at <= audit_at:
                keep(self._link(uid, res, LinkLayer.IDENTIFIER, o.file_id, o.observed_at))
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

    def proximity_link(self, uid: str, audit_at: datetime, reported_at: datetime,
                       facts: dict[Res, Link]) -> tuple[Link | None, list[str]]:
        """Layer 5: the primary-profile delete near reported_at that no other
        user owns. Returns (link or None, candidate resource ids considered)."""
        if any(sys == "account_service" for sys, _ in facts):
            return None, []
        near = [e for e in self._deletes
                if abs(e.event_time - reported_at) <= PROXIMITY_WINDOW and e.event_time <= audit_at]
        cands = sorted({e.resource_id for e in near})
        free = [rid for rid in cands if not (self.owners(("account_service", rid), audit_at) - {uid})]
        if len(free) != 1:
            return None, cands
        event = min((e for e in near if e.resource_id == free[0]), key=lambda e: abs(e.event_time - reported_at))
        return self._link(uid, ("account_service", free[0]), LinkLayer.PROXIMITY, event.activity_id,
                          event.event_time), cands

    def links(self, uid: str, audit_at: datetime, reported_at: datetime,
              max_layer: LinkLayer = LinkLayer.PROXIMITY) -> dict[Res, Link]:
        found = {r: lk for r, lk in self.fact_links(uid, audit_at).items() if lk.layer <= max_layer}
        if max_layer >= LinkLayer.PROXIMITY:
            link, _ = self.proximity_link(uid, audit_at, reported_at, found)
            if link:
                found[(link.system_id, link.resource_id)] = link
        return found
