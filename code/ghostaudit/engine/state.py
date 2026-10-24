"""Resource state at audit_at.

A fact is an explicit observation or an absence implied by a complete
snapshot that omits the resource. Only facts from evidence captured at or
before audit_at count. Facts are grouped per evidence file; within one file a
copy in any table wins over its absence elsewhere (a row surviving in a
failover replica is still a remaining record). The latest file wins.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ..ingest.observations import ObservationTable
from ..schemas import EXPOSURE, Coverage, DataCategory, Observation, ObservedState, ResourceState



@dataclass(frozen=True)
class Fact:
    file_id: str
    observed_at: datetime
    state: ObservedState
    explicit: bool
    category: DataCategory | None
    detail: str


def facts_for(res: tuple[str, str], table: ObservationTable, cutoff: datetime) -> list[Fact]:
    """Facts about a resource from evidence captured at or before cutoff."""
    system, rid = res
    seen = table.by_resource.get(res, ())
    out = [_from_obs(o) for o in seen if o.captured_at <= cutoff]
    explicit_files = {f.file_id for f in out}
    if not out and any(o.observed_at <= cutoff for o in seen):
        # Observed before the cutoff but only in a file exported later: the
        # state is unknown at audit time, not absent by omission.
        return []
    for c in table.coverage:
        if c.system_id != system or c.captured_at > cutoff or rid in c.resource_ids:
            continue
        # Snapshots are not reliably complete: a later bundle routinely omits
        # retained audit records it never meant to list. Omission therefore
        # counts only inside a file that observes the resource explicitly
        # (replica drift), or when no explicit observation exists at all.
        if not explicit_files or c.file_id in explicit_files:
            out.append(_from_coverage(c))
    return sorted(out, key=lambda f: (f.observed_at, f.file_id, f.detail))


def _from_obs(o: Observation) -> Fact:
    return Fact(o.file_id, o.observed_at, o.state, True, o.data_category, o.source_detail)


def _from_coverage(c: Coverage) -> Fact:
    return Fact(c.file_id, c.observed_at, ObservedState.ABSENT, False, None, f"omitted from {c.source_detail}")


def state_at(res: tuple[str, str], table: ObservationTable, audit_at: datetime) -> ResourceState | None:
    facts = facts_for(res, table, audit_at)
    if not facts:
        return None
    by_file: dict[str, list[Fact]] = {}
    for f in facts:
        by_file.setdefault(f.file_id, []).append(f)
    summaries = []
    for fid, fs in by_file.items():
        latest = max(f.observed_at for f in fs)
        at_latest = [f for f in fs if f.observed_at == latest]
        best = max(at_latest, key=lambda f: (EXPOSURE[f.state], f.explicit))
        summaries.append((latest, best.explicit, fid, best))
    # Latest file wins; at equal time an explicit observation beats an omission.
    summaries.sort(key=lambda s: (s[0], s[1], s[2]))
    latest_time = summaries[-1][0]
    winner = summaries[-1][3]
    same_time_states = {s[3].state for s in summaries if s[0] == latest_time}
    category = next((f.category for f in reversed(facts) if f.category is not None), None)
    return ResourceState(
        system_id=res[0], resource_id=res[1], data_category=category, state=winner.state,
        evidence_id=winner.file_id, as_of=winner.observed_at, conflicting=len(same_time_states) > 1,
        sources=tuple(sorted({s[2] for s in summaries})),
    )


def out_of_scope(res: tuple[str, str], table: ObservationTable, requested_at: datetime, audit_at: datetime,
                 linked_at: datetime | None = None) -> bool:
    """A record not touched by this request's period is not part of it:
    either its last explicit observation before requested_at is ABSENT (it
    was already gone), or it has no explicit observation at all and was linked
    before requested_at (it belongs to an earlier request). In both cases
    nothing explicit after requested_at shows it."""
    explicit = [f for f in facts_for(res, table, audit_at) if f.explicit]
    before = [f for f in explicit if f.observed_at < requested_at]
    after = [f for f in explicit if f.observed_at >= requested_at]
    if after:
        return False
    if before:
        return before[-1].state is ObservedState.ABSENT
    return linked_at is not None and linked_at < requested_at
