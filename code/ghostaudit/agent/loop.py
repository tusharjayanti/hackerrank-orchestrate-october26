"""Investigator agent loop (L2) and its single entry point from the engine.

review_case runs after the engine has decided a case. If the gate
(candidates.needs_agent) does not fire, nothing changes. If it fires, the
engine generates candidate decisions and a manual tool-use loop on
model_for("AGENT") chooses one of them, by id, within MAX_TURNS model calls.
Every way the loop can fail ends in the engine default with a flag:
  model call fails / refuses / budget reached -> AGENT_UNAVAILABLE
  two rejected submissions                     -> AGENT_REJECTED_SUBMISSION
  MAX_TURNS without an accepted submission     -> AGENT_MAX_TURNS
The steps written are always exactly one candidate's steps.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .. import llm as llm_mod
from ..budget import BudgetExceeded
from ..llm import LLMUnavailable, image_block, model_for
from ..obs import get_logger
from ..schemas import Decision, DeletionStatus, Flag, FlagSet, LinkLayer, ObservedState, Step
from . import prompts
from .candidates import ENGINE_DEFAULT, Candidate, generate_candidates, needs_agent
from .tools import ALLOWED, Session, run_tool, tool_specs

if TYPE_CHECKING:
    from ..engine.decide import World

log = get_logger(__name__)

MAX_TURNS = 6
MAX_TOKENS = 16000
EFFORT = "medium"
SUMMARY_CHARS = 240


def agent_trace(**kw) -> dict:
    """trace["agent"], recorded for every case whether or not the agent ran."""
    base = {"gated": False, "reasons": [], "ran": False, "turns": 0, "tool_calls": [], "candidates": [],
            "inadmissible": [], "chosen": None, "outcome": "not_gated", "model": None, "cached": False,
            "observations": []}
    base.update(kw)
    return base


@dataclass
class AgentResult:
    outcome: str  # accepted | rejected_submission | max_turns | unavailable
    chosen: Candidate | None
    turns: int
    tool_calls: list[dict] = field(default_factory=list)
    detail: str = ""
    model: str = ""
    cached: bool = False


def _summary(result: dict) -> str:
    s = json.dumps(result, sort_keys=True, default=str, ensure_ascii=False)
    return s if len(s) <= SUMMARY_CHARS else s[:SUMMARY_CHARS] + "..."


def _tool_result_content(name: str, result: dict) -> str | list[dict]:
    text = json.dumps(result, sort_keys=True, default=str, ensure_ascii=False)
    if name == "read_evidence_file" and result.get("format") == "png" and "image_path" in result:
        from pathlib import Path

        return [{"type": "text", "text": text}, image_block(Path(result["image_path"]), "image/png")]
    return text


def _cache_path(model: str, first_message: str):
    h = hashlib.sha256()
    for part in (model, prompts.PROMPT_VERSION, prompts.SYSTEM, json.dumps(tool_specs(), sort_keys=True),
                 first_message):
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    return llm_mod.CACHE_DIR / "agent" / f"{h.hexdigest()}.json"


def run_agent(world: World, request: dict, trace: dict, reasons: list[str], candidates: list[Candidate],
              client=None) -> AgentResult:
    """The manual tool-use loop for one gated case. Never raises."""
    llm = world.llm
    model = model_for("AGENT")
    session = Session(request=request, trace=trace, candidates=candidates, world=world)
    first = prompts.case_message(request, reasons, [{**c.summary(), "steps_json": c.steps_json}
                                                     for c in candidates])
    cache = _cache_path(model, first)
    base_row = {"purpose": f"agent:{request['case_id']}", "prompt_version": prompts.PROMPT_VERSION}
    if cache.is_file():
        hit = json.loads(cache.read_text(encoding="utf-8"))
        chosen = next((c for c in candidates if c.id == hit["chosen"]), None)
        if chosen is None and hit.get("observation"):
            chosen = _replay_agent_candidate(world, request, trace, candidates, hit["observation"])
        if chosen is not None:
            llm._log({**base_row, "model": hit["model"], "cached": True, "input_tokens": 0, "output_tokens": 0,
                      "usd": 0.0, "cache_key": cache.stem})
            return AgentResult("accepted", chosen, hit["turns"], hit["tool_calls"], model=hit["model"],
                               cached=True)
    try:
        client = client or llm._client_or_raise()
    except LLMUnavailable as e:
        return AgentResult("unavailable", None, 0, detail=str(e), model=model)

    specs = tool_specs()
    messages: list[dict] = [{"role": "user", "content": first}]
    calls: list[dict] = []
    rejections = 0
    served_by = model
    for turn in range(1, MAX_TURNS + 1):
        try:
            llm.budget.check()
        except BudgetExceeded as e:
            return AgentResult("unavailable", None, turn - 1, calls, f"budget: {e}", served_by)
        start = time.monotonic()
        try:
            resp = client.messages.create(model=model, max_tokens=MAX_TOKENS, system=prompts.SYSTEM, tools=specs,
                                          tool_choice={"type": "auto"}, messages=messages,
                                          output_config={"effort": EFFORT})
        except Exception as e:  # noqa: BLE001 - any SDK/API failure ends in the engine default
            log.error("agent_call_failed", case_id=request["case_id"], turn=turn, error=type(e).__name__)
            return AgentResult("unavailable", None, turn - 1, calls, f"model call failed: {type(e).__name__}",
                               served_by)
        served_by = getattr(resp, "model", model)
        usage = resp.usage
        llm._log({**base_row, "model": served_by, "cached": False, "turn": turn, "stop_reason": resp.stop_reason,
                  "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
                  "usd": round(llm.budget.record(served_by, usage.input_tokens, usage.output_tokens), 6),
                  "latency_s": round(time.monotonic() - start, 2), "request_id": getattr(resp, "_request_id", None)})
        if resp.stop_reason in ("refusal", "max_tokens"):
            return AgentResult("unavailable", None, turn, calls, f"model stopped with {resp.stop_reason}", served_by)
        messages.append({"role": "assistant", "content": resp.content})
        uses = [b for b in resp.content if b.type == "tool_use"]
        if not uses:
            messages.append({"role": "user", "content": prompts.NUDGE})
            continue
        results: list[dict] = []
        repair: str | None = None
        for b in uses:
            if session.accepted is not None:
                result = {"error": "a decision was already accepted; this call was not run"}
            elif b.name not in ALLOWED:
                result = {"error": f"tool {b.name!r} is not available; allowed: {list(ALLOWED)}"}
            else:
                result = run_tool(world, b.name, dict(b.input), session)
            calls.append({"turn": turn, "tool": b.name, "args": dict(b.input), "result_summary": _summary(result)})
            if b.name == "submit_decision" and session.accepted is None and "error" in result:
                rejections += 1
                session.rejections.append(result["error"])
                if rejections >= 2:
                    return AgentResult("rejected_submission", None, turn, calls, result["error"], served_by)
                repair = prompts.REPAIR.format(reason=result["error"], ids=[c.id for c in candidates])
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": _tool_result_content(b.name, result),
                            "is_error": "error" in result})
        if session.accepted is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"chosen": session.accepted.id, "turns": turn, "tool_calls": calls,
                                         "model": served_by,
                                         "observation": (session.accepted.observations or [None])[0]},
                                        sort_keys=True, indent=1), encoding="utf-8")
            return AgentResult("accepted", session.accepted, turn, calls, model=served_by)
        if repair:
            results.append({"type": "text", "text": repair})
        messages.append({"role": "user", "content": results})
    return AgentResult("max_turns", None, MAX_TURNS, calls, f"no accepted submission in {MAX_TURNS} turns",
                       served_by)


def _replay_agent_candidate(world: World, request: dict, trace: dict, candidates: list[Candidate],
                            stored: dict) -> Candidate | None:
    """Rebuild an agent-built candidate from the cache by re-running the same
    verification and engine path (the cache never vouches for evidence)."""
    from .candidates import build_with_observation, inadmissible
    from .tools import ToolError, verified_observation

    uid = (trace.get("identity") or {}).get("user_id")
    try:
        obs = verified_observation(world, request, uid, stored["system_id"], stored["resource_id"],
                                   ObservedState(stored["state"]), stored["file_id"])
    except (ToolError, KeyError, ValueError):
        return None
    cand = build_with_observation(world, request, trace, obs)
    return cand if inadmissible(cand, candidates[0]) is None else None


_OUTCOME_FLAG = {"unavailable": Flag.AGENT_UNAVAILABLE, "rejected_submission": Flag.AGENT_REJECTED_SUBMISSION,
                 "max_turns": Flag.AGENT_MAX_TURNS}


def review_case(world: World, request: dict, max_layer: LinkLayer, status: DeletionStatus, steps,
                flags: FlagSet, trace: dict, client=None) -> tuple[DeletionStatus, list[Step]]:
    """The engine's call site: gate, then (if gated) candidates and the loop.
    Returns the status and steps to write, which are the engine's unless the
    agent chose another engine candidate."""
    steps = list(steps)
    engine = Decision(case_id=request["case_id"], status=status, steps=tuple(steps), flags=flags.hits)
    reasons = needs_agent(world, request, engine, trace)
    if not reasons:
        trace["agent"] = agent_trace()
        return status, steps
    if world.llm is None:
        flags.add(Flag.AGENT_UNAVAILABLE, "no model")
        trace["agent"] = agent_trace(gated=True, reasons=reasons, outcome="unavailable")
        return status, steps
    dropped: list[dict] = []
    candidates = generate_candidates(world, request, engine, trace, max_layer, dropped)
    flags.add(Flag.AGENT_INVOKED, "; ".join(r.split(":", 1)[0] for r in reasons))
    result = run_agent(world, request, trace, reasons, candidates, client)
    trace["agent"] = agent_trace(
        gated=True, reasons=reasons, ran=True, turns=result.turns, tool_calls=result.tool_calls,
        candidates=[c.summary() for c in candidates], inadmissible=dropped,
        chosen=result.chosen.id if result.chosen else None,
        observations=list(result.chosen.observations) if result.chosen else [],
        outcome=result.outcome, detail=result.detail, model=result.model, cached=result.cached)
    if result.outcome in _OUTCOME_FLAG:
        flags.add(_OUTCOME_FLAG[result.outcome], result.detail)
        return status, steps
    chosen = result.chosen
    if chosen.id == ENGINE_DEFAULT:
        return status, steps
    trace["notes"].append(f"agent chose {chosen.id}: {chosen.description}")
    return DeletionStatus(chosen.status), [Step.model_validate(s) for s in json.loads(chosen.steps_json)]
