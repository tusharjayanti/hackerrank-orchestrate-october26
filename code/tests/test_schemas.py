import pytest
from pydantic import ValidationError

from ghostaudit.schemas import (
    ALLOWED_TRIPLES,
    TRIPLE_SET,
    Decision,
    DeletionStatus,
    Operation,
    Phase,
    Step,
    StepState,
    case_sort_key,
    number_steps,
)


def step(phase, op, state, system="account_service", resource="account_1", ev=""):
    return Step(step_no=1, phase=phase, system_id=system, resource_id=resource,
                operation=op, state=state, evidence_id=ev)


def test_triple_table_has_fifteen_unique_rows():
    assert len(ALLOWED_TRIPLES) == 15
    assert len(TRIPLE_SET) == 15
    assert len({op for _, op, _ in ALLOWED_TRIPLES}) == 15
    assert {op for _, op, _ in ALLOWED_TRIPLES} == set(Operation)


def test_triple_table_phases_are_contiguous_and_ordered():
    phases = [ph for ph, _, _ in ALLOWED_TRIPLES]
    assert phases == sorted(phases, key=list(Phase).index)


def test_step_rejects_disallowed_triple():
    with pytest.raises(ValidationError):
        step(Phase.FINDING, Operation.DELETE_RECORD, StepState.ABSENT)
    with pytest.raises(ValidationError):
        step(Phase.VERIFICATION, Operation.VERIFY_RECORD_ABSENT, StepState.PRESENT)


def test_step_rejects_extra_field():
    with pytest.raises(ValidationError):
        Step(step_no=1, phase="FINDING", system_id="s", resource_id="r",
             operation="REPORT_REMAINING_RECORD", state="PRESENT", evidence_id="e", note="x")


def test_sort_key_orders_by_phase_then_operation_then_system_then_resource():
    # Mirrors sample case_11: findings by operation then system_id.
    steps = [
        step(Phase.VERIFICATION, Operation.VERIFY_RECORD_REDACTED, StepState.REDACTED, "support_platform", "support_864"),
        step(Phase.REMEDIATION, Operation.ANONYMIZE_RECORD, StepState.ANONYMIZED, "analytics_platform", "analytics_2034"),
        step(Phase.REMEDIATION, Operation.DISABLE_DATA_FLOW, StepState.DISABLED, "integration_service", "data_flow_9522"),
        step(Phase.FINDING, Operation.REPORT_EXPOSED_PII, StepState.PRESENT, "support_platform", "support_864", "e"),
        step(Phase.FINDING, Operation.REPORT_EXPOSED_PII, StepState.PRESENT, "analytics_platform", "analytics_2034", "e"),
        step(Phase.FINDING, Operation.REPORT_REMAINING_RECORD, StepState.PRESENT, "search_service", "search_052", "e"),
        step(Phase.VERIFICATION, Operation.VERIFY_RECORD_ABSENT, StepState.ABSENT, "search_service", "search_052"),
        step(Phase.VERIFICATION, Operation.VERIFY_RECORD_ABSENT, StepState.ABSENT, "account_service", "account_4248"),
    ]
    ordered = number_steps(steps)
    assert [s.step_no for s in ordered] == list(range(1, 9))
    assert [(s.operation.value, s.resource_id) for s in ordered] == [
        ("REPORT_REMAINING_RECORD", "search_052"),
        ("REPORT_EXPOSED_PII", "analytics_2034"),
        ("REPORT_EXPOSED_PII", "support_864"),
        ("DISABLE_DATA_FLOW", "data_flow_9522"),
        ("ANONYMIZE_RECORD", "analytics_2034"),
        ("VERIFY_RECORD_ABSENT", "account_4248"),
        ("VERIFY_RECORD_ABSENT", "search_052"),
        ("VERIFY_RECORD_REDACTED", "support_864"),
    ]


def test_serialization_matches_sample_format():
    d = Decision(case_id="case_01", status=DeletionStatus.COMPLETE, steps=tuple(number_steps([
        step(Phase.VERIFICATION, Operation.VERIFY_RECORD_ABSENT, StepState.ABSENT, "account_service",
             "account_001", "evidence_54")])))
    assert d.to_row()["steps_json"] == (
        '[{"evidence_id":"evidence_54","operation":"VERIFY_RECORD_ABSENT","phase":"VERIFICATION",'
        '"resource_id":"account_001","state":"ABSENT","step_no":1,"system_id":"account_service"}]')


def test_case_sort_is_natural():
    ids = ["case_100", "case_33", "case_9", "case_550"]
    assert sorted(ids, key=case_sort_key) == ["case_9", "case_33", "case_100", "case_550"]


def test_flag_enum_is_the_agreed_list():
    from ghostaudit.schemas import Flag
    assert [f.value for f in Flag] == [
        "UNRESOLVED_IDENTITY", "AMBIGUOUS_IDENTIFIER", "PARSE_FAILED", "UNMAPPED_CATEGORY", "INTENT_REVIEW",
        "INTENT_UNAVAILABLE", "INJECTION_SUSPECTED", "VISION_DISAGREED", "VISION_ONLY_PENDING",
        "VISION_UNAVAILABLE", "BUDGET_BREACHED", "AGENT_INVOKED", "AGENT_MAX_TURNS", "AGENT_REJECTED_SUBMISSION",
        "AGENT_UNAVAILABLE", "CONTRACT_REPAIRED", "COMPLETE_DOWNGRADED", "SHARED_ATTACHMENT", "ENGINE_ERROR",
        "NEEDS_REVIEW"]


def test_flagset_dedupes_and_orders_by_enum():
    from ghostaudit.schemas import Flag, FlagSet
    fs = FlagSet()
    fs.add(Flag.NEEDS_REVIEW, "x")
    fs.add(Flag.INTENT_REVIEW, "y")
    fs.add(Flag.NEEDS_REVIEW, "x")
    assert fs.fired() == ["INTENT_REVIEW", "NEEDS_REVIEW"] and len(fs.hits) == 2
