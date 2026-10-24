import csv
from datetime import datetime, timezone

from ghostaudit import DATASET_DIR
from ghostaudit.engine.decide import World, decide_case
from ghostaudit.engine.policy import load_rules, treat
from ghostaudit.engine.state import out_of_scope, state_at
from ghostaudit.ingest.manifest import parse_ts
from ghostaudit.schemas import DataCategory, ObservedState, Operation, ResourceState

WORLD = World.load()
REQUESTS = {r["case_id"]: r for r in csv.DictReader(open(DATASET_DIR / "sample_deletion_requests.csv"))}


def ts(s):
    return parse_ts(s)


def test_record_absent_before_request_is_out_of_scope():
    # case_17: account_6523 was a 404 in March (an earlier request); this
    # request was made in August.
    r = REQUESTS["case_17"]
    res = ("account_service", "account_6523")
    assert out_of_scope(res, WORLD.table, ts(r["requested_at"]), ts(r["audit_at"]))
    decision, trace = decide_case(WORLD, r)
    assert "account_service/account_6523" in trace["out_of_scope"]
    assert all(s.resource_id != "account_6523" for s in decision.steps)


def test_record_present_after_request_stays_in_scope():
    r = REQUESTS["case_24"]
    assert not out_of_scope(("account_service", "account_5334"), WORLD.table, ts(r["requested_at"]),
                            ts(r["audit_at"]))


def test_old_link_without_later_evidence_is_out_of_scope():
    r = REQUESTS["case_17"]
    linked = datetime(2026, 3, 10, 6, 7, 1, tzinfo=timezone.utc)
    assert out_of_scope(("identity_service", "tombstone_17"), WORLD.table, ts(r["requested_at"]), ts(r["audit_at"]),
                        linked)


def test_replica_copy_beats_primary_absence():
    st = state_at(("account_service", "account_005"), WORLD.table, ts(REQUESTS["case_05"]["audit_at"]))
    assert st.state is ObservedState.PRESENT and st.evidence_id == "evidence_54"


def test_late_export_is_unknown_not_absent():
    # audit_record_4902 is only in evidence_06, exported after case_09's audit_at.
    assert state_at(("security_service", "audit_record_4902"), WORLD.table,
                    ts(REQUESTS["case_09"]["audit_at"])) is None


def test_policy_treatments():
    rules = load_rules()

    def st(state):
        return ResourceState(system_id="s", resource_id="r", data_category=None, state=state, evidence_id="e",
                             as_of=ts("2026-01-01T00:00:00Z"))

    t = treat("identity_service", "tombstone_1", DataCategory.ACCOUNT_TOMBSTONE, st(ObservedState.IDENTIFIABLE), rules)
    assert (t.finding, t.remediation, t.verification) == (
        Operation.REPORT_EXPOSED_PII, Operation.REDACT_RECORD, Operation.VERIFY_RECORD_REDACTED)
    t = treat("backup_service", "backup_1", DataCategory.BACKUP_ARCHIVE, st(ObservedState.ACTIVE), rules)
    assert t.finding is Operation.REPORT_RESTORE_RISK and t.remediation is Operation.BLOCK_RESTORE
    t = treat("security_service", "audit_1", DataCategory.SECURITY_AUDIT_LOG, st(ObservedState.PRESENT), rules)
    assert t.finding is None and t.verification is Operation.VERIFY_RECORD_PRESENT
    t = treat("search_service", "search_1", DataCategory.SEARCH_INDEX_ENTRY, None, rules)
    assert t.state is None and t.verification is Operation.VERIFY_RECORD_ABSENT
