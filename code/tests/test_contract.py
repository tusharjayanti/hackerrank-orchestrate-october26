import json
from datetime import UTC, datetime

from ghostaudit.guards import check_row
from ghostaudit.ingest.manifest import CitationRegistry

AUDIT_AT = datetime(2026, 9, 5, 12, 0, tzinfo=UTC)
REGISTRY = CitationRegistry({
    "evidence_01": datetime(2026, 9, 1, tzinfo=UTC),
    "evidence_02": AUDIT_AT,  # captured exactly at audit_at: citable
    "evidence_99": datetime(2026, 9, 5, 12, 1, tzinfo=UTC),  # one minute late
})


def s(no, phase, op, state, system, resource, ev=""):
    return {"step_no": no, "phase": phase, "operation": op, "state": state,
            "system_id": system, "resource_id": resource, "evidence_id": ev}


def good_incomplete():
    return [
        s(1, "FINDING", "REPORT_REMAINING_RECORD", "PRESENT", "search_service", "search_1", "evidence_01"),
        s(2, "REMEDIATION", "DELETE_RECORD", "ABSENT", "search_service", "search_1"),
        s(3, "VERIFICATION", "VERIFY_RECORD_ABSENT", "ABSENT", "account_service", "account_1"),
        s(4, "VERIFICATION", "VERIFY_RECORD_ABSENT", "ABSENT", "search_service", "search_1"),
    ]


def codes(status, steps):
    return {i.code for i in check_row("case_x", status, json.dumps(steps), AUDIT_AT, REGISTRY)}


def test_good_rows_pass():
    assert codes("INCOMPLETE", good_incomplete()) == set()
    assert codes("NOT_APPLICABLE", []) == set()
    assert codes("COMPLETE", [s(1, "VERIFICATION", "VERIFY_RECORD_ABSENT", "ABSENT",
                                "account_service", "account_1", "evidence_02")]) == set()


def test_wrong_order():
    steps = good_incomplete()
    steps[2], steps[3] = steps[3], steps[2]
    for i, x in enumerate(steps, 1):
        x["step_no"] = i
    assert "step_order" in codes("INCOMPLETE", steps)


def test_phase_order():
    steps = good_incomplete()
    steps[0], steps[1] = steps[1], steps[0]
    for i, x in enumerate(steps, 1):
        x["step_no"] = i
    assert "phase_order" in codes("INCOMPLETE", steps)


def test_bad_numbering():
    steps = good_incomplete()
    steps[1]["step_no"] = 3
    assert "numbering" in codes("INCOMPLETE", steps)


def test_missing_finding_evidence():
    steps = good_incomplete()
    steps[0]["evidence_id"] = ""
    assert "finding_missing_evidence" in codes("INCOMPLETE", steps)


def test_complete_requires_evidence_and_only_verification():
    assert "complete_missing_evidence" in codes("COMPLETE", [s(1, "VERIFICATION", "VERIFY_RECORD_ABSENT", "ABSENT",
                                                                 "account_service", "account_1")])
    assert "complete_non_verification" in codes("COMPLETE", good_incomplete())


def test_future_steps_must_not_cite():
    steps = good_incomplete()
    steps[1]["evidence_id"] = "evidence_01"
    assert "future_step_has_evidence" in codes("INCOMPLETE", steps)


def test_bad_triple():
    steps = good_incomplete()
    steps[0]["state"] = "ACTIVE"
    assert "triple" in codes("INCOMPLETE", steps)
    steps = good_incomplete()
    steps[1]["phase"] = "FINDING"
    assert "triple" in codes("INCOMPLETE", steps)


def test_extra_or_missing_field():
    steps = good_incomplete()
    steps[0]["note"] = "x"
    assert "step_fields" in codes("INCOMPLETE", steps)
    steps = good_incomplete()
    del steps[0]["evidence_id"]
    assert "step_fields" in codes("INCOMPLETE", steps)


def test_not_applicable_must_be_empty():
    assert "na_with_steps" in codes("NOT_APPLICABLE", good_incomplete())


def test_finding_needs_remediation_and_verification():
    steps = [good_incomplete()[0], s(2, "VERIFICATION", "VERIFY_RECORD_ABSENT", "ABSENT", "search_service", "search_1")]
    assert "finding_without_remediation" in codes("INCOMPLETE", steps)


def test_verification_state_must_match_remediation():
    steps = good_incomplete()
    steps[3] = s(4, "VERIFICATION", "VERIFY_RECORD_REDACTED", "REDACTED", "search_service", "search_1")
    assert "verification_state_mismatch" in codes("INCOMPLETE", steps)


def test_resource_twice_in_phase():
    steps = good_incomplete()
    steps.insert(1, s(2, "FINDING", "REPORT_EXPOSED_PII", "PRESENT", "search_service", "search_1", "evidence_01"))
    for i, x in enumerate(steps, 1):
        x["step_no"] = i
    assert "resource_repeated" in codes("INCOMPLETE", steps)


def test_citations_unknown_and_after_audit():
    steps = good_incomplete()
    steps[0]["evidence_id"] = "evidence_77"
    assert "citation_unknown" in codes("INCOMPLETE", steps)
    steps[0]["evidence_id"] = "evidence_99"
    assert "citation_after_audit" in codes("INCOMPLETE", steps)


def test_invalid_json_and_status():
    assert "json_invalid" in {i.code for i in check_row("c", "INCOMPLETE", "[{", AUDIT_AT, REGISTRY)}
    assert "status_invalid" in codes("DONE", [])
