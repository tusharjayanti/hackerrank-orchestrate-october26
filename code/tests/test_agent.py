"""The investigator agent loop against a stubbed model (no network)."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ghostaudit import llm as llm_mod
from ghostaudit.agent import loop, tools
from ghostaudit.agent.candidates import ENGINE_DEFAULT, needs_agent
from ghostaudit.engine.decide import World, decide_case
from ghostaudit.intent import Classified
from ghostaudit.llm import LLM
from ghostaudit.schemas import Flag, Intent, IntentLabel

BASE = World.load()
GATED = "case_01"  # offline: resolved user with no citable observation (zero_observations)
UNGATED = "case_05"  # offline: the account row is in evidence_54


def tool_use(name, args, i=1):
    return SimpleNamespace(type="tool_use", id=f"toolu_{i}", name=name, input=args)


def response(*blocks, stop="tool_use"):
    return SimpleNamespace(content=list(blocks), stop_reason=stop, model="claude-opus-5-5", _request_id="req_test",
                           usage=SimpleNamespace(input_tokens=100, output_tokens=20))


class Stub:
    """A client whose messages.create replays scripted turns."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.requests = []
        self.messages = self

    def create(self, **kw):
        self.requests.append(json.loads(json.dumps(kw["messages"], default=lambda o: o.__dict__)))
        nxt = self.turns.pop(0) if self.turns else self.turns_default()
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    def turns_default(self):
        return response(tool_use("list_observations", {"case_id": GATED}))


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_mod, "CALL_LOG", tmp_path / "llm_calls.jsonl")
    monkeypatch.setattr(llm_mod, "CACHE_DIR", tmp_path / "cache")
    return replace(BASE, llm=LLM(run_id="test"))


def run(world, case_id, stub):
    world.llm._client = stub
    return decide_case(world, world.requests[case_id])


def engine_only(case_id):
    d, _ = decide_case(replace(BASE, llm=None), BASE.requests[case_id])
    return d


def submit(cid, steps_json=None, i=1):
    return tool_use("submit_decision", {"case_id": GATED, "candidate_id": cid, "steps_json": steps_json,
                                        "rationale": "test"}, i)


FAKE_STEPS = json.dumps([{"step_no": 1, "phase": "FINDING", "system_id": "account_service",
                          "resource_id": "account_999", "operation": "REPORT_REMAINING_RECORD",
                          "state": "PRESENT", "evidence_id": "evidence_01"}])


def test_case_is_gated_offline():
    d, trace = decide_case(replace(BASE, llm=None), BASE.requests[GATED])
    assert any(r.startswith("zero_observations") for r in needs_agent(BASE, BASE.requests[GATED], d, trace))
    assert trace["agent"]["gated"] and Flag.AGENT_UNAVAILABLE.value in trace["flags_fired"]


def test_unflagged_case_never_invokes_agent(world):
    stub = Stub([RuntimeError("the model must not be called")])
    d, trace = run(world, UNGATED, stub)
    assert stub.requests == []
    assert trace["agent"] == loop.agent_trace()
    assert Flag.AGENT_INVOKED.value not in trace["flags_fired"]


def test_not_applicable_case_never_reaches_agent(world):
    stub = Stub([RuntimeError("the model must not be called")])
    world.llm._client = stub
    na = Classified(Intent(label=IntentLabel.ACCESS, confidence="high"), "en", ())
    d, trace = decide_case(world, world.requests[GATED], classified=na)
    assert d.status.value == "NOT_APPLICABLE" and stub.requests == []
    assert trace["agent"]["gated"] is False


def test_accepted_submission_writes_exactly_a_candidate(world):
    stub = Stub([response(tool_use("list_observations", {"case_id": GATED})), None])
    stub.turns[1] = response(submit(ENGINE_DEFAULT, i=2))
    d, trace = run(world, GATED, stub)
    ag = trace["agent"]
    assert ag["ran"] and ag["outcome"] == "accepted" and ag["chosen"] == ENGINE_DEFAULT and ag["turns"] == 2
    assert [c["tool"] for c in ag["tool_calls"]] == ["list_observations", "submit_decision"]
    assert Flag.AGENT_INVOKED.value in trace["flags_fired"]
    assert d.to_row() == engine_only(GATED).to_row()
    rows = [json.loads(x) for x in (llm_mod.CALL_LOG).read_text().splitlines()]
    assert [r["turn"] for r in rows] == [1, 2] and all(r["run_id"] == "test" and "usd" in r for r in rows)


def test_non_default_candidate_steps_are_copied_byte_for_byte(world):
    stub = Stub([])
    world.llm._client = stub
    req = world.requests[GATED]
    d0, t0 = decide_case(replace(BASE, llm=None), req)
    from ghostaudit.agent.candidates import generate_candidates

    cands = generate_candidates(world, req, d0, t0, max_layer=__import__("ghostaudit").schemas.LinkLayer.SIBLING)
    pick = cands[-1]
    stub.turns = [response(submit(pick.id, pick.steps_json))]
    d, trace = run(world, GATED, stub)
    assert trace["agent"]["chosen"] == pick.id
    assert d.to_row()["steps_json"] == pick.steps_json and d.status.value == pick.status


def test_fabricated_step_list_is_rejected_then_falls_back(world):
    stub = Stub([response(submit(ENGINE_DEFAULT, FAKE_STEPS)), response(submit("my_own_plan", None, 2))])
    d, trace = run(world, GATED, stub)
    ag = trace["agent"]
    assert ag["outcome"] == "rejected_submission" and ag["chosen"] is None and ag["turns"] == 2
    assert "does not match" in ag["tool_calls"][0]["result_summary"]
    # the one repair message went back with the rejection
    repair = stub.requests[1][-1]["content"]
    assert repair[0]["is_error"] is True and "Submit again" in repair[-1]["text"]
    assert Flag.AGENT_REJECTED_SUBMISSION.value in trace["flags_fired"]
    assert d.to_row() == engine_only(GATED).to_row()


def test_max_turns_falls_back(world):
    stub = Stub([])
    d, trace = run(world, GATED, stub)
    assert len(stub.requests) == loop.MAX_TURNS
    assert trace["agent"]["outcome"] == "max_turns" and trace["agent"]["turns"] == loop.MAX_TURNS
    assert Flag.AGENT_MAX_TURNS.value in trace["flags_fired"]
    assert d.to_row() == engine_only(GATED).to_row()


def test_unknown_tool_burns_a_turn(world):
    stub = Stub([response(tool_use("drop_table", {"table": "accounts"})), response(submit(ENGINE_DEFAULT, i=2))])
    d, trace = run(world, GATED, stub)
    ag = trace["agent"]
    assert ag["turns"] == 2 and ag["outcome"] == "accepted"
    assert ag["tool_calls"][0]["tool"] == "drop_table" and "not available" in ag["tool_calls"][0]["result_summary"]
    assert stub.requests[1][-1]["content"][0]["is_error"] is True


def test_tool_exception_comes_back_as_error_result(world, monkeypatch):
    def boom(world, inp, session=None):
        raise RuntimeError("disk on fire")

    monkeypatch.setitem(tools.TOOLS, "list_observations", (tools.ListObservationsInput, boom))
    assert "disk on fire" in tools.run_tool(world, "list_observations", {"case_id": GATED})["error"]
    stub = Stub([response(tool_use("list_observations", {"case_id": GATED})), response(submit(ENGINE_DEFAULT, i=2))])
    d, trace = run(world, GATED, stub)
    result = stub.requests[1][-1]["content"][0]
    assert result["is_error"] is True and "disk on fire" in result["content"]
    assert trace["agent"]["outcome"] == "accepted"


@pytest.mark.parametrize("failure", [RuntimeError("connection reset"), response(stop="refusal")])
def test_model_failure_gives_agent_unavailable(world, failure):
    stub = Stub([response(tool_use("list_observations", {"case_id": GATED})), failure])
    d, trace = run(world, GATED, stub)
    assert trace["agent"]["outcome"] == "unavailable" and trace["agent"]["ran"]
    assert {Flag.AGENT_INVOKED.value, Flag.AGENT_UNAVAILABLE.value} <= set(trace["flags_fired"])
    assert d.to_row() == engine_only(GATED).to_row()


def test_budget_breach_gives_agent_unavailable(world):
    world.llm.budget.max_usd = 0.0
    d, trace = run(world, GATED, Stub([]))
    assert trace["agent"]["outcome"] == "unavailable" and "budget" in trace["agent"]["detail"]


def test_accepted_decision_replays_from_cache(world):
    run(world, GATED, Stub([response(submit(ENGINE_DEFAULT))]))
    stub = Stub([RuntimeError("cached; the model must not be called")])
    d, trace = run(world, GATED, stub)
    assert stub.requests == [] and trace["agent"]["cached"] and trace["agent"]["chosen"] == ENGINE_DEFAULT
