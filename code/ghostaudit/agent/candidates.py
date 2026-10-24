"""When the agent runs (the gate) and what it may choose from (candidates).

The gate fires on exactly three conditions, each a case the deterministic
engine cannot settle from its own evidence:
  a. a resolved user with zero citable observations at audit_at;
  b. one of the case's resources has conflicting states in evidence captured
     at the same captured_at;
  c. a violation whose only support is a held vision_only observation.

Candidates are complete decisions the engine generated for the case under
explicit, named variations of its own inputs. The agent picks one by id; it
never writes a step. Every candidate goes through the same state_at ->
policy_diff -> build_steps path as the engine default.

The action space is monotone toward caution: a candidate is admissible only
if it keeps every finding of the engine default, never turns an INCOMPLETE
default into COMPLETE, and never drops a resource the default covers. The
agent can make an audit stricter, never looser. (Measured: before this rule
the agent chose a less safe reading on 4 cases and the guards reverted all 4.)
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from ..engine.policy import policy_diff, treat
from ..engine.state import state_at
from ..engine.steps import build_steps
from ..ingest.manifest import parse_ts
from ..ingest.observations import ObservationTable, category_for
from ..linking import Linker
from ..schemas import EXPOSURE, Decision, LinkLayer, Observation, ObservedState, ResourceState, serialize_steps

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
    # Agent-verified observations this candidate relies on (apply_policy).
    observations: tuple[dict, ...] = ()

    def summary(self) -> dict:
        return {"id": self.id, "description": self.description, "status": self.status,
                "steps": self.steps_json.count('"step_no"')}


def _findings(steps_json: str) -> set[tuple[str, str, str]]:
    return {(s["operation"], s["system_id"], s["resource_id"]) for s in json.loads(steps_json)
            if s["phase"] == "FINDING"}


def _resources(steps_json: str) -> set[tuple[str, str]]:
    return {(s["system_id"], s["resource_id"]) for s in json.loads(steps_json)}


def inadmissible(candidate: Candidate, default: Candidate) -> str | None:
    """Why a candidate would make the audit looser than the engine default,
    or None if it is admissible."""
    if default.status == "INCOMPLETE" and candidate.status == "COMPLETE":
        return "turns the engine's INCOMPLETE into COMPLETE"
    lost = _findings(default.steps_json) - _findings(candidate.steps_json)
    if lost:
        return "drops findings " + ", ".join(f"{op} {sy}/{r}" for op, sy, r in sorted(lost))
    dropped = _resources(default.steps_json) - _resources(candidate.steps_json)
    if dropped:
        return "drops resources " + ", ".join(f"{sy}/{r}" for sy, r in sorted(dropped))
    return None


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


def generate_candidates(world: World, request: dict, decision: Decision, trace: dict,
                        max_layer: LinkLayer, dropped: list[dict] | None = None) -> list[Candidate]:
    """Admissible engine decisions for the case under each applicable
    variation, deduplicated on (status, steps_json), engine default first and
    the rest sorted by id. Inadmissible variations are left out and, when
    `dropped` is given, recorded there with the reason."""
    uid = _uid(trace)
    audit_at = parse_ts(request["audit_at"])
    raw: list[tuple[str, str, tuple[str, str]]] = [
        (ENGINE_DEFAULT, "the engine's own decision", (decision.status.value, serialize_steps(list(decision.steps))))]

    def add(cid: str, description: str, make) -> None:
        try:
            raw.append((cid, description, make()))
        except Exception as e:  # noqa: BLE001 - a variant that cannot run is not offered
            from ..telemetry import get_logger

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
            # Unresolved conflict: only the safer (more exposed) reading is
            # offered; the less safe one would loosen the audit.
            safer = max(states, key=lambda st: (EXPOSURE[ObservedState(st)], st))
            for keep in [safer]:
                def side(system=system, rid=rid, at=at, keep=keep):
                    obs = [o for o in world.table.observations
                           if not ((o.system_id, o.resource_id) == (system, rid) and o.captured_at == at
                                   and o.state.value != keep)]
                    table = _table(world, obs)
                    return _via_engine(_engine_only(world, table=table, linker=_linker_over(world, table)),
                                       request, max_layer)
                add(f"conflict:{system}/{rid}@{at.strftime('%Y%m%dT%H%M')}:{keep}",
                    f"{system}/{rid} taken as {keep} in the evidence captured at {at.isoformat()}", side)

    # No candidate switches a measured rule off (scope rule, L5 links) or a
    # rejected one on (L6 sibling links, 1/3 on the samples): the agent adds
    # evidence or takes the safer side of a conflict, nothing else.

    seen: dict[tuple[str, str], int] = {}
    out: list[Candidate] = []
    for cid, desc, key in raw:
        if key in seen:
            prev = out[seen[key]]
            out[seen[key]] = replace(prev, description=f"{prev.description}; same result as {cid}")
            continue
        seen[key] = len(out)
        out.append(Candidate(cid, desc, key[0], key[1]))
    default, kept = out[0], []
    for c in sorted(out[1:], key=lambda c: c.id):
        why = inadmissible(c, default)
        if why is None:
            kept.append(c)
        elif dropped is not None:
            dropped.append({"id": c.id, "reason": why})
    return [default] + kept


def build_with_observation(world: World, request: dict, trace: dict, obs: Observation) -> Candidate:
    """The engine decision for the case with one agent-verified record added
    (the record replaces nothing the engine already knows: it is judged with
    the same state -> policy -> steps path, citing the file it came from)."""
    audit_at = parse_ts(request["audit_at"])
    res = (obs.system_id, obs.resource_id)
    resources = []
    for r in scoped_resources(trace):
        if r == res:
            continue
        st = state_at(r, world.table, audit_at)
        category = st.data_category if st and st.data_category else category_for(r[0])
        resources.append((r[0], r[1], category, st))
    st = ResourceState(system_id=obs.system_id, resource_id=obs.resource_id, data_category=obs.data_category,
                       state=obs.state, evidence_id=obs.file_id, as_of=obs.observed_at)
    resources.append((obs.system_id, obs.resource_id, obs.data_category, st))
    status, steps = build_steps(policy_diff(sorted(resources, key=lambda x: (x[0], x[1])), world.rules)[0])
    return Candidate(f"agent_observation:{obs.system_id}/{obs.resource_id}:{obs.state.value}",
                     f"engine decision with {obs.system_id}/{obs.resource_id} {obs.state.value} as read in "
                     f"{obs.file_id} (a record the parsers did not extract)", status.value, serialize_steps(steps),
                     (obs.model_dump(mode="json"),))
