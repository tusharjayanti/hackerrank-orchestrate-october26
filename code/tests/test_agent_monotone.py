"""The agent's action space is monotone toward caution: it can make an audit
stricter, never looser."""

from dataclasses import replace

import pytest

from ghostaudit.agent import tools
from ghostaudit.agent.candidates import (
    ENGINE_DEFAULT,
    Candidate,
    build_with_observation,
    generate_candidates,
    inadmissible,
)
from ghostaudit.engine.decide import World, decide_case
from ghostaudit.schemas import LinkLayer, ObservedState

WORLD = replace(World.load(), llm=None)


def finding(op, rid, ev="evidence_01", system="account_service"):
    return {"evidence_id": ev, "operation": op, "phase": "FINDING", "resource_id": rid, "state": "PRESENT",
            "step_no": 1, "system_id": system}


def verify(op, state, rid, ev="", system="account_service"):
    return {"evidence_id": ev, "operation": op, "phase": "VERIFICATION", "resource_id": rid, "state": state,
            "step_no": 2, "system_id": system}


def cand(cid, status, steps):
    import json
    return Candidate(cid, cid, status, json.dumps(steps, sort_keys=True, separators=(",", ":")))


def candidates_for(case_id):
    req = WORLD.requests[case_id]
    decision, trace = decide_case(WORLD, req, use_agent=False)
    dropped = []
    return generate_candidates(WORLD, req, decision, trace, LinkLayer.SIBLING, dropped), dropped


@pytest.mark.parametrize("case_id", ["case_01", "case_02", "case_135", "case_383"])
def test_the_four_reverted_cases_have_no_admissible_alternative(case_id):
    # Before the monotone rule the agent chose fact_links_only on 01 and 02
    # and the ABSENT side of a same-file conflict on 135 and 383; the guards
    # reverted all four. Now the engine default is the only choice.
    got, dropped = candidates_for(case_id)
    assert [c.id for c in got] == [ENGINE_DEFAULT]
    assert all(not d["id"].startswith("fact_links_only") for d in dropped)


def test_fact_links_only_is_never_generated():
    for case_id in ("case_01", "case_02", "case_04", "case_61", "case_63"):
        got, dropped = candidates_for(case_id)
        assert not any(c.id == "fact_links_only" for c in got + [Candidate(d["id"], "", "", "") for d in dropped])


def test_inadmissible_reasons():
    default = cand(ENGINE_DEFAULT, "INCOMPLETE", [finding("REPORT_REMAINING_RECORD", "account_1"),
                                                  verify("VERIFY_RECORD_ABSENT", "ABSENT", "account_1")])
    looser = cand("x", "COMPLETE", [verify("VERIFY_RECORD_ABSENT", "ABSENT", "account_1", "evidence_02")])
    assert "COMPLETE" in inadmissible(looser, default)
    fewer = cand("y", "INCOMPLETE", [finding("REPORT_REMAINING_RECORD", "account_2"),
                                     verify("VERIFY_RECORD_ABSENT", "ABSENT", "account_2")])
    assert "drops findings" in inadmissible(fewer, default)
    stricter = cand("z", "INCOMPLETE", [finding("REPORT_REMAINING_RECORD", "account_1"),
                                        finding("REPORT_REMAINING_RECORD", "search_9", system="search_service"),
                                        verify("VERIFY_RECORD_ABSENT", "ABSENT", "account_1"),
                                        verify("VERIFY_RECORD_ABSENT", "ABSENT", "search_9", system="search_service")])
    assert inadmissible(stricter, default) is None
    complete = cand(ENGINE_DEFAULT, "COMPLETE", [verify("VERIFY_RECORD_ABSENT", "ABSENT", "account_1", "e")])
    empty = cand("w", "COMPLETE", [])
    assert "drops resources" in inadmissible(empty, complete)


def test_submit_rejects_an_inadmissible_candidate():
    default = cand(ENGINE_DEFAULT, "INCOMPLETE", [finding("REPORT_REMAINING_RECORD", "account_1"),
                                                  verify("VERIFY_RECORD_ABSENT", "ABSENT", "account_1")])
    looser = cand("looser", "COMPLETE", [verify("VERIFY_RECORD_ABSENT", "ABSENT", "account_1", "evidence_02")])
    session = tools.Session(request=WORLD.requests["case_24"], trace={}, candidates=[default, looser], world=WORLD)
    out = tools.run_tool(WORLD, "submit_decision", {"case_id": "case_24", "candidate_id": "looser",
                                                    "steps_json": None, "rationale": "x"}, session)
    assert "not admissible" in out["error"] and session.accepted is None


def test_apply_policy_needs_the_resource_and_the_subject_in_the_file():
    req = WORLD.requests["case_05"]
    uid = WORLD.identity.resolve_email(req["email"]).user_id
    with pytest.raises(tools.ToolError, match="does not appear"):
        tools.verified_observation(WORLD, req, uid, "account_service", "account_999999", ObservedState.PRESENT,
                                   "evidence_54")
    other = WORLD.identity.resolve_email(WORLD.requests["case_01"]["email"]).user_id
    with pytest.raises(tools.ToolError, match="none of the subject's identifiers"):
        tools.verified_observation(WORLD, req, other, "account_service", "account_005", ObservedState.PRESENT,
                                   "evidence_54")
    obs = tools.verified_observation(WORLD, req, uid, "account_service", "account_005", ObservedState.PRESENT,
                                     "evidence_54")
    assert obs.source_detail == "agent:apply_policy" and "leila.haddad@yahoo.com" in obs.identifiers


def test_a_verified_missed_record_builds_a_stricter_candidate():
    # case_01's subject has no record in evidence_54; suppose the agent found
    # one the parsers missed: the candidate keeps the engine's verification
    # and adds a finding for the new record.
    req = WORLD.requests["case_05"]
    decision, trace = decide_case(WORLD, req, use_agent=False)
    uid = trace["identity"]["user_id"]
    obs = tools.verified_observation(WORLD, req, uid, "account_service", "account_005", ObservedState.PRESENT,
                                     "evidence_54")
    obs = obs.model_copy(update={"resource_id": "account_005", "system_id": "account_service"})
    built = build_with_observation(WORLD, req, trace, obs)
    default = generate_candidates(WORLD, req, decision, trace, LinkLayer.SIBLING)[0]
    assert inadmissible(built, default) is None
