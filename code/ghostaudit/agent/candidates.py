"""When the agent runs (the gate) and what it may choose from (candidates).

The gate fires on exactly three conditions, each a case the deterministic
engine cannot settle from its own evidence:
  a. a resolved user with zero citable observations at audit_at;
  b. one of the case's resources has conflicting states in evidence captured
     at the same captured_at;
  c. a violation whose only support is a held vision_only observation.

Candidates are complete decisions the engine generated for the case under
explicit, named variations of its own inputs. The agent picks one by id; it
never writes a resource id or a step. Every candidate goes through the same
state_at -> policy_diff -> build_steps path as the engine default.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from ..engine.policy import policy_diff, treat
from ..engine.state import state_at
from ..engine.steps import build_steps
from ..ingest.manifest import parse_ts
from ..ingest.observations import ObservationTable, category_for
from ..linking import Linker
from ..schemas import Decision, LinkLayer, Observation, ObservedState, ResourceState, serialize_steps

if TYPE_CHECKING:
    from ..engine.decide import World

VIOLATING_STATES = frozenset({ObservedState.PRESENT, ObservedState.IDENTIFIABLE, ObservedState.ACTIVE})
ENGINE_DEFAULT = "engine_default"


@dataclass(frozen=True)
class Candidate:
    id: str
    description: str
    status: str
    steps_json: str

    def summary(self) -> dict:
        return {"id": self.id, "description": self.description, "status": self.status,
                "steps": self.steps_json.count('"step_no"')}


def _uid(trace: dict) -> str | None:
    return (trace.get("identity") or {}).get("user_id")


def scoped_resources(trace: dict) -> list[tuple[str, str]]:
    """The case's linked resources that the scope rule kept."""
    out = set(trace.get("out_of_scope") or [])
    return sorted({(lk["system_id"], lk["resource_id"]) for lk in trace.get("links") or []
                   if f"{lk['system_id']}/{lk['resource_id']}" not in out})


def captured_conflicts(world: World, resources, audit_at) -> dict[tuple[str, str], dict]:
    """Resources with more than one state in evidence captured at the same
    captured_at (<= audit_at): {res: {captured_at: sorted states}}."""
    out = {}
    for res in resources:
        groups: dict = defaultdict(set)
        for o in world.table.by_resource.get(res, ()):
            if o.captured_at <= audit_at:
                groups[o.captured_at].add(o.state)
        clash = {at: sorted(s.value for s in states) for at, states in groups.items() if len(states) > 1}
        if clash:
            out[res] = clash
    return out


def vision_only_violations(world: World, uid: str, audit_at) -> list[Observation]:
    """Held vision_only rows that would be a violation and are the only
    citable observation of their resource. Policy decides what a violation
    is; for a category policy cannot judge, a violating state counts."""
    out = []
    for o in world.table.held_for_user(uid, audit_at):
        res = (o.system_id, o.resource_id)
        if any(x.captured_at <= audit_at for x in world.table.by_resource.get(res, ())):
            continue
        category = o.data_category or category_for(o.system_id)
        state = ResourceState(system_id=o.system_id, resource_id=o.resource_id, data_category=category,
                              state=o.state, evidence_id=o.file_id, as_of=o.observed_at)
        t = treat(o.system_id, o.resource_id, category, state, world.rules)
        if (t.finding is not None) if t is not None else (o.state in VIOLATING_STATES):
            out.append(o)
    return out


def needs_agent(world: World, request: dict, decision: Decision, trace: dict) -> list[str]:
    """Reasons the case goes to the agent; empty means it never does."""
    uid = _uid(trace)
    if uid is None or decision.status.value == "NOT_APPLICABLE":
        return []
    audit_at = parse_ts(request["audit_at"])
    reasons = []
    if not world.table.case_observations(uid, audit_at):
        reasons.append("zero_observations: no observation of the user's resources captured <= audit_at")
    for (system, rid), clash in sorted(captured_conflicts(world, scoped_resources(trace), audit_at).items()):
        for at, states in sorted(clash.items()):
            reasons.append(f"state_conflict: {system}/{rid} is {'+'.join(states)} at captured_at "
                           f"{at.isoformat()}")
    for o in vision_only_violations(world, uid, audit_at):
        reasons.append(f"vision_only_violation: {o.system_id}/{o.resource_id} {o.state.value} only in "
                       f"{o.file_id}")
    return reasons


# --- candidate generation ---------------------------------------------------

def _engine_only(world: World, **changes) -> World:
    """A copy of the world with no model, so a variant run never re-enters
    the agent."""
    return replace(world, llm=None, **changes)


def _table(world: World, observations: list[Observation]) -> ObservationTable:
    t = world.table
    return ObservationTable(observations, t.coverage, t.stats, t.identity)


def _linker_over(world: World, table: ObservationTable, enable_sibling: bool | None = None) -> Linker:
    """A linker over a variant table that keeps the run's layer 5-6
    inferences (solved jointly over every request) unless asked to re-solve
    with layer 6 on."""
    base = world.linker
    sibling = base.enable_sibling if enable_sibling is None else enable_sibling
    lk = Linker(table, world.activity, vision_links=base.vision_links, enable_sibling=sibling)
    if sibling != base.enable_sibling:
        from ..engine.decide import case_context

        lk.prepare([c for c in (case_context(world.identity, r) for r in world.requests.values()) if c])
    else:
        lk._inferred = getattr(base, "_inferred", {})
    return lk


def _via_engine(world: World, request: dict, max_layer: LinkLayer) -> tuple[str, str]:
    from ..engine.decide import decide_case

    d, _ = decide_case(world, request, max_layer)
    return d.status.value, serialize_steps(list(d.steps))


def _all_links_in_scope(world: World, request: dict, max_layer: LinkLayer) -> tuple[str, str]:
    """The engine pipeline with the scope rule switched off: every linked
    resource is judged, including ones last seen before requested_at."""
    from ..engine.decide import case_context

    audit_at = parse_ts(request["audit_at"])
    links = world.linker.links(case_context(world.identity, request), max_layer)
    resources = []
    for res in sorted(links):
        st = state_at(res, world.table, audit_at)
        category = st.data_category if st and st.data_category else category_for(res[0])
        resources.append((res[0], res[1], category, st))
    status, steps = build_steps(policy_diff(resources, world.rules)[0])
    return status.value, serialize_steps(steps)


def generate_candidates(world: World, request: dict, decision: Decision, trace: dict,
                        max_layer: LinkLayer) -> list[Candidate]:
    """Engine decisions for the case under each applicable variation,
    deduplicated on (status, steps_json), engine default first and the rest
    sorted by id."""
    uid = _uid(trace)
    audit_at = parse_ts(request["audit_at"])
    raw: list[tuple[str, str, tuple[str, str]]] = [
        (ENGINE_DEFAULT, "the engine's own decision", (decision.status.value, serialize_steps(list(decision.steps))))]

    def add(cid: str, description: str, make) -> None:
        try:
            raw.append((cid, description, make()))
        except Exception as e:  # noqa: BLE001 - a variant that cannot run is not offered
            from ..obs import get_logger

            get_logger(__name__).warning("agent_candidate_failed", candidate=cid, error=repr(e))

    held = world.table.held_for_user(uid, audit_at) if uid else []
    if held:
        def admitted():
            extra = [o.model_copy(update={"source_detail": o.source_detail.replace(":vision_only", ":admitted")})
                     for o in held]
            table = _table(world, world.table.observations + extra)
            return _via_engine(_engine_only(world, table=table, linker=_linker_over(world, table)), request,
                               max_layer)
        add("admit_vision_only", "held screenshot-only rows (" + ", ".join(
            sorted({f"{o.file_id}:{o.resource_id}" for o in held})) + ") admitted as evidence", admitted)

    for (system, rid), clash in sorted(captured_conflicts(world, scoped_resources(trace), audit_at).items()):
        for at, states in sorted(clash.items()):
            for keep in states:
                def side(system=system, rid=rid, at=at, keep=keep):
                    obs = [o for o in world.table.observations
                           if not ((o.system_id, o.resource_id) == (system, rid) and o.captured_at == at
                                   and o.state.value != keep)]
                    table = _table(world, obs)
                    return _via_engine(_engine_only(world, table=table, linker=_linker_over(world, table)),
                                       request, max_layer)
                add(f"conflict:{system}/{rid}@{at.strftime('%Y%m%dT%H%M')}:{keep}",
                    f"{system}/{rid} taken as {keep} in the evidence captured at {at.isoformat()}", side)

    add("all_links_in_scope", "scope rule off: every linked resource is judged, including ones last seen "
        "before requested_at", lambda: _all_links_in_scope(world, request, max_layer))
    if max_layer >= LinkLayer.PROXIMITY:
        add("fact_links_only", "only evidence and catalog links (layers 1-4); no inferred proximity or sibling "
            "links", lambda: _via_engine(_engine_only(world), request, LinkLayer.VISION))
    if max_layer >= LinkLayer.SIBLING and not world.linker.enable_sibling:
        add("with_sibling_links", "layer 6 on: the sole free analytics anonymization in the account deletion's "
            "worker run is linked", lambda: _via_engine(
                _engine_only(world, linker=_linker_over(world, world.table, enable_sibling=True)), request,
                max_layer))

    seen: dict[tuple[str, str], int] = {}
    out: list[Candidate] = []
    for cid, desc, key in raw:
        if key in seen:
            prev = out[seen[key]]
            out[seen[key]] = replace(prev, description=f"{prev.description}; same result as {cid}")
            continue
        seen[key] = len(out)
        out.append(Candidate(cid, desc, key[0], key[1]))
    return [out[0]] + sorted(out[1:], key=lambda c: c.id)
