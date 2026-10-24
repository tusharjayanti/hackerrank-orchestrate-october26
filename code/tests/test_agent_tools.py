import json

import pytest
from pydantic import ValidationError

from ghostaudit.agent import tools as t
from ghostaudit.engine.decide import World

WORLD = World.load()


def test_inputs_are_strict_and_closed():
    with pytest.raises(ValidationError):
        t.ListObservationsInput(case_id=61)  # no coercion
    with pytest.raises(ValidationError):
        t.ReadEvidenceFileInput(case_id="case_61", file_id="evidence_01", path="/etc/passwd")


def test_tool_specs_are_strict_with_closed_schemas():
    specs = t.tool_specs()
    assert {s["name"] for s in specs} == set(t.TOOLS)
    assert all(s["strict"] and s["input_schema"]["additionalProperties"] is False for s in specs)
    assert all("May not" in s["description"] for s in specs)


def test_read_refuses_evidence_after_audit_at():
    # case_01 is audited 2026-09-03; evidence_92 was captured 2026-09-15.
    with pytest.raises(t.ToolError):
        t.read_evidence_file(WORLD, t.ReadEvidenceFileInput(case_id="case_01", file_id="evidence_92"))
    out = t.read_evidence_file(WORLD, t.ReadEvidenceFileInput(case_id="case_05", file_id="evidence_54"))
    assert "account_005" in out["content"]


def test_search_is_exact_and_time_bounded():
    hits = t.search_raw_evidence(WORLD, t.SearchRawEvidenceInput(case_id="case_05",
                                                                  identifier="LEILA.HADDAD@yahoo.com"))["hits"]
    assert any(h["file_id"] == "evidence_54" for h in hits)
    decoy = t.search_raw_evidence(WORLD, t.SearchRawEvidenceInput(case_id="case_06", identifier="jonas.berg@outlook"))
    assert decoy["hits"] == []


def test_apply_policy_reports_the_rule_without_deciding():
    out = t.run_tool(WORLD, "apply_policy", {"case_id": "case_32", "system_id": "identity_service",
                                             "resource_id": "tombstone_9178", "observed_state": "IDENTIFIABLE",
                                             "evidence_id": "evidence_51"})
    assert (out["finding"], out["remediation"]) == ("REPORT_EXPOSED_PII", "REDACT_RECORD")


def test_tool_errors_come_back_as_results_not_exceptions():
    out = t.run_tool(WORLD, "apply_policy", {"case_id": "case_32", "system_id": "identity_service",
                                             "resource_id": "tombstone_9178", "observed_state": "MAYBE",
                                             "evidence_id": "evidence_51"})
    assert "error" in out
    assert "error" in t.run_tool(WORLD, "drop_table", {})


def test_submit_decision_only_runs_inside_an_investigation():
    out = t.run_tool(WORLD, "submit_decision", {"case_id": "case_01", "candidate_id": "engine_default",
                                                "steps_json": None, "rationale": "x"})
    assert "only available inside" in out["error"]
    # the old observation-list shape is not accepted
    out = t.run_tool(WORLD, "submit_decision", {"case_id": "case_01", "outcome": "with_observations",
                                                "observations": [], "rationale": "x"})
    assert "error" in out


def test_strict_schemas_carry_no_unsupported_keywords():
    blob = json.dumps(t.tool_specs())
    assert all(k not in blob for k in ('"maxLength"', '"minLength"', '"default"'))
    for s in t.tool_specs():
        sch = s["input_schema"]
        assert sorted(sch["required"]) == sorted(sch["properties"])
