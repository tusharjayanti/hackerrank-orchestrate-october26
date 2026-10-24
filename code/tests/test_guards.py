"""Evidence guards, the policy contract, conditional retention and shared
attachments. Offline: screenshots are deferred and intent is not classified,
so every request is audited as a deletion."""

import csv
import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ghostaudit import CODE_ROOT, DATASET_DIR
from ghostaudit.engine.decide import World, decide_case
from ghostaudit.engine.policy import (
    RETENTION_EXPIRED,
    Scoped,
    load_rules,
    policy_diff,
    retention_expired,
    shared_attachment,
    treat,
)
from ghostaudit.guards import (
    GuardContext,
    categories_from_trace,
    check_row,
    citation_issues,
    complete_check,
    cross_row_check,
    guard_decision,
    policy_issues,
    rederive,
    sample_mix,
)
from ghostaudit.ingest.manifest import CitationRegistry, parse_ts
from ghostaudit.ingest.observations import retain_until_in
from ghostaudit.schemas import (
    DataCategory,
    Decision,
    DeletionStatus,
    Flag,
    FlagSet,
    Observation,
    ObservedState,
    Operation,
    ResourceState,
    Step,
    number_steps,
)

WORLD = World.load()
CTX = GuardContext(WORLD, CitationRegistry.load())


def _requests(name):
    with open(DATASET_DIR / name, newline="", encoding="utf-8") as f:
        return {r["case_id"]: r for r in csv.DictReader(f)}


SAMPLES = _requests("sample_deletion_requests.csv")
EVAL = _requests("deletion_requests.csv")
UTC = UTC


def step(phase, op, state, system, resource, ev=""):
    return Step(step_no=1, phase=phase, system_id=system, resource_id=resource, operation=op, state=state,
                evidence_id=ev)


def decision(case_id, status, steps, *flags):
    fs = FlagSet()
    for f in flags:
        fs.add(f)
    return Decision(case_id=case_id, status=status, steps=tuple(number_steps(steps)), flags=fs.hits)


def as_dicts(d: Decision):
    return json.loads(d.to_row()["steps_json"])


# ---------------------------------------------------------------- engine rows pass every guard

def test_every_engine_sample_row_passes_the_guards_unchanged():
    for cid, req in SAMPLES.items():
        d, t = decide_case(WORLD, req)
        g, gt = guard_decision(d, t, req, CTX, regenerate=lambda cid=cid: (_ for _ in ()).throw(AssertionError(cid)))
        assert g == d, cid
        assert gt["guards"]["action"] == "none", cid


def test_answer_key_complete_rows_rederive_from_their_citations():
    # The same code that guards our rows reproduces every COMPLETE row of the
    # answer key from the files it cites.
    with open(DATASET_DIR / "sample_output.csv", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["deletion_status"] == "COMPLETE"]
    assert len(rows) == 10
    for r in rows:
        steps = json.loads(r["steps_json"])
        issues, status, _ = complete_check(r["case_id"], steps, parse_ts(SAMPLES[r["case_id"]]["audit_at"]), WORLD)
        assert not issues and status is DeletionStatus.COMPLETE, r["case_id"]


# ---------------------------------------------------------------- downgrade

def test_complete_row_refuted_by_its_citation_is_downgraded_with_a_cited_finding():
    # case_30: evidence_92 shows search_9359 still PRESENT. A row claiming it
    # ABSENT on that evidence is re-derived into the finding it actually shows.
    req = SAMPLES["case_30"]
    bad = decision("case_30", DeletionStatus.COMPLETE,
                   [step("VERIFICATION", "VERIFY_RECORD_ABSENT", "ABSENT", "search_service", "search_9359",
                         "evidence_92")])
    g, trace = guard_decision(bad, {}, req, CTX)
    assert g.status is DeletionStatus.INCOMPLETE
    assert Flag.COMPLETE_DOWNGRADED in FlagSet(g.flags)
    assert trace["guards"]["action"] == "downgraded"
    ops = [(s.operation.value, s.evidence_id) for s in g.steps]
    assert ops == [("REPORT_REMAINING_RECORD", "evidence_92"), ("DELETE_RECORD", ""), ("VERIFY_RECORD_ABSENT", "")]
    audit_at = parse_ts(req["audit_at"])
    assert check_row("case_30", g.status.value, g.to_row()["steps_json"], audit_at, CTX.registry) == []
    assert citation_issues("case_30", as_dicts(g), audit_at, WORLD) == []


def test_downgrade_never_goes_the_other_way():
    # An INCOMPLETE engine row whose findings are all backed is never turned
    # COMPLETE, even when one resource in it is compliant.
    for cid, req in SAMPLES.items():
        d, t = decide_case(WORLD, req)
        if d.status is DeletionStatus.INCOMPLETE:
            assert guard_decision(d, t, req, CTX)[0].status is DeletionStatus.INCOMPLETE, cid


def test_unsupported_complete_step_is_regenerated_by_the_engine():
    # evidence_92 says nothing about account_4163 (its state comes from
    # evidence_51): not downgradable, so the engine regenerates the row.
    req = SAMPLES["case_30"]
    bad = decision("case_30", DeletionStatus.COMPLETE,
                   [step("VERIFICATION", "VERIFY_RECORD_ABSENT", "ABSENT", "account_service", "account_4163",
                         "evidence_92")])
    engine = decide_case(WORLD, req)
    g, trace = guard_decision(bad, {}, req, CTX, regenerate=lambda: engine)
    assert trace["guards"]["action"] == "regenerated"
    assert g.steps == engine[0].steps and g.status is engine[0].status
    assert Flag.CONTRACT_REPAIRED in FlagSet(g.flags)


def test_conservative_repair_keeps_supported_findings_and_drops_the_rest():
    req = SAMPLES["case_30"]
    # A finding on the right file, one on a file that never mentions the
    # resource, and no remediation at all: the contract fails twice (the
    # regenerated row is the same bad row).
    bad = decision("case_30", DeletionStatus.INCOMPLETE, [
        step("FINDING", "REPORT_REMAINING_RECORD", "PRESENT", "search_service", "search_9359", "evidence_92"),
        step("FINDING", "REPORT_REMAINING_RECORD", "PRESENT", "account_service", "account_4163", "evidence_92"),
    ])
    g, trace = guard_decision(bad, {}, req, CTX, regenerate=lambda: (bad, {}))
    assert trace["guards"]["action"] == "repaired"
    assert Flag.CONTRACT_REPAIRED in FlagSet(g.flags)
    assert g.status is DeletionStatus.INCOMPLETE
    findings = [(s.resource_id, s.evidence_id) for s in g.steps if s.phase.value == "FINDING"]
    assert findings == [("search_9359", "evidence_92")]
    assert check_row("case_30", g.status.value, g.to_row()["steps_json"], parse_ts(req["audit_at"]),
                     CTX.registry) == []


def test_repair_with_nothing_citable_falls_back_to_needs_review():
    req = SAMPLES["case_30"]
    bad = decision("case_30", DeletionStatus.INCOMPLETE, [
        step("FINDING", "REPORT_REMAINING_RECORD", "PRESENT", "account_service", "account_4163", "evidence_92"),
    ])
    g, _ = guard_decision(bad, {}, req, CTX, regenerate=lambda: (bad, {}))
    assert g.status is DeletionStatus.COMPLETE and g.steps == ()
    assert {Flag.CONTRACT_REPAIRED, Flag.NEEDS_REVIEW} <= {h.flag for h in g.flags}


def test_agreed_needs_review_fallback_is_kept_as_is():
    req = SAMPLES["case_04"]
    d, t = decide_case(WORLD, req)
    assert d.status is DeletionStatus.COMPLETE and d.steps == () and Flag.NEEDS_REVIEW in FlagSet(d.flags)
    g, trace = guard_decision(d, t, req, CTX, regenerate=lambda: (_ for _ in ()).throw(AssertionError()))
    assert g == d and trace["guards"].get("accepted_fallback")


# ---------------------------------------------------------------- citation guard

def test_citation_must_observe_the_resource():
    audit_at = parse_ts(SAMPLES["case_30"]["audit_at"])
    ok = [{"step_no": 1, "phase": "FINDING", "operation": "REPORT_REMAINING_RECORD", "state": "PRESENT",
           "system_id": "search_service", "resource_id": "search_9359", "evidence_id": "evidence_92"}]
    assert citation_issues("c", ok, audit_at, WORLD) == []
    wrong = [dict(ok[0], resource_id="search_0000")]
    assert [i.code for i in citation_issues("c", wrong, audit_at, WORLD)] == ["citation_does_not_observe"]
    not_file = [dict(ok[0], evidence_id="act_000001")]
    assert [i.code for i in citation_issues("c", not_file, audit_at, WORLD)] == ["citation_not_evidence_file"]


def test_snapshot_omission_supports_only_an_absent_claim():
    cov = next(c for c in WORLD.table.coverage if c.resource_ids)
    missing = ("x_" + cov.system_id, "never_listed_1")
    res = (cov.system_id, missing[1])
    absent = [{"step_no": 1, "phase": "VERIFICATION", "operation": "VERIFY_RECORD_ABSENT", "state": "ABSENT",
               "system_id": res[0], "resource_id": res[1], "evidence_id": cov.file_id}]
    late = datetime(2100, 1, 1, tzinfo=UTC)
    assert citation_issues("c", absent, late, WORLD) == []
    present = [dict(absent[0], phase="FINDING", operation="REPORT_REMAINING_RECORD", state="PRESENT")]
    assert [i.code for i in citation_issues("c", present, late, WORLD)] == ["citation_does_not_observe"]


# ---------------------------------------------------------------- policy contract

def _row(op, phase, state, system, rid):
    return {"step_no": 1, "phase": phase, "operation": op, "state": state, "system_id": system,
            "resource_id": rid, "evidence_id": ""}


def test_policy_contract_on_synthetic_rows():
    delete = lambda system, rid: [_row("DELETE_RECORD", "REMEDIATION", "ABSENT", system, rid)]  # noqa: E731
    for system, rid in (("security_service", "audit_record_1"), ("legal_service", "legal_hold_1"),
                        ("backup_service", "backup_1"), ("identity_service", "tombstone_1")):
        assert [i.code for i in policy_issues("c", delete(system, rid), None)] == ["delete_on_kept_record"]
    cats = {"support_platform/support_1": ("SUPPORT_MESSAGE", ""),
            "analytics_platform/analytics_1": ("ANALYTICS_PROFILE", ""),
            "support_platform/support_2": ("SUPPORT_ATTACHMENT", ""),
            "analytics_platform/analytics_2": ("MARKETING_PROFILE", ""),
            "security_service/audit_record_1": ("SECURITY_AUDIT_LOG", RETENTION_EXPIRED)}
    assert [i.code for i in policy_issues("c", delete("support_platform", "support_1"), cats)] == \
        ["delete_on_kept_category"]
    assert [i.code for i in policy_issues("c", delete("analytics_platform", "analytics_1"), cats)] == \
        ["delete_on_kept_category"]
    assert policy_issues("c", delete("support_platform", "support_2"), cats) == []
    assert policy_issues("c", delete("analytics_platform", "analytics_2"), cats) == []  # sample case_12
    assert policy_issues("c", delete("security_service", "audit_record_1"), cats) == []  # retention passed
    for system, rid in (("cache_service", "cache_1"), ("email_delivery_service", "email_delivery_1")):
        verify = [_row("VERIFY_RECORD_ABSENT", "VERIFICATION", "ABSENT", system, rid)]
        assert [i.code for i in policy_issues("c", verify, None)] == ["activity_only_resource"]


def test_policy_contract_holds_on_all_518_engine_rows():
    issues = []
    for cid, req in EVAL.items():
        d, t = decide_case(WORLD, req)
        issues += policy_issues(cid, as_dicts(d), categories_from_trace(t))
    assert issues == []


def _latest_traces(output: Path):
    want = str(output.resolve())
    best = None
    for m in (CODE_ROOT / "runs").glob("tune/*/run_manifest.json"):
        man = json.loads(m.read_text(encoding="utf-8"))
        if str(Path(man.get("output") or "").resolve()) == want and (best is None or m.parent.name > best.name):
            best = m.parent
    return best / "traces" if best else None


def test_policy_contract_holds_on_the_real_tune_output():
    output = CODE_ROOT / "runs" / "tune" / "output.csv"
    traces = _latest_traces(output) if output.is_file() else None
    if traces is None:
        pytest.skip("no tune output with traces (run code/main.py --mode tune)")
    with open(output, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 518
    issues = []
    for r in rows:
        trace = json.loads((traces / f"{r['case_id']}.json").read_text(encoding="utf-8"))
        issues += policy_issues(r["case_id"], json.loads(r["steps_json"]), categories_from_trace(trace))
    assert issues == []


# ---------------------------------------------------------------- conditional retention

def test_retention_dates_are_read_from_fields_and_text():
    assert retain_until_in({"retention_until": "2033-12-31T23:59:59Z"}) == datetime(2033, 12, 31, 23, 59, 59,
                                                                                     tzinfo=UTC)
    assert retain_until_in({"retained_until": "2031-01-01"}) == datetime(2031, 1, 1, tzinfo=UTC)
    assert retain_until_in({"note": "Retain until 2027-05-01 for tax"}) == datetime(2027, 5, 1, tzinfo=UTC)
    assert retain_until_in({"note": "kept"}) is None


def _obs(retain=None, category=DataCategory.SECURITY_AUDIT_LOG, ids=(), at=datetime(2026, 1, 1, tzinfo=UTC)):
    return Observation(file_id="evidence_x", captured_at=at, observed_at=at, system_id="security_service",
                       resource_id="audit_record_1", data_category=category, state=ObservedState.PRESENT,
                       identifiers=tuple(ids), source_detail="t", retain_until=retain)


def test_retain_holds_until_the_date_then_the_record_is_remaining():
    audit_at = datetime(2026, 6, 1, tzinfo=UTC)
    assert not retention_expired([_obs()], audit_at)
    assert not retention_expired([_obs(datetime(2030, 1, 1, tzinfo=UTC))], audit_at)
    assert retention_expired([_obs(datetime(2026, 5, 1, tzinfo=UTC))], audit_at)
    # the latest statement wins: an extended hold supersedes an expired date
    assert not retention_expired([_obs(datetime(2026, 5, 1, tzinfo=UTC)),
                                  _obs(datetime(2030, 1, 1, tzinfo=UTC), at=datetime(2026, 2, 1, tzinfo=UTC))],
                                 audit_at)
    rules = load_rules()
    st = ResourceState(system_id="security_service", resource_id="audit_record_1",
                       data_category=DataCategory.SECURITY_AUDIT_LOG, state=ObservedState.PRESENT,
                       evidence_id="evidence_x", as_of=audit_at)
    kept = treat("security_service", "audit_record_1", DataCategory.SECURITY_AUDIT_LOG, st, rules)
    assert (kept.finding, kept.verification) == (None, Operation.VERIFY_RECORD_PRESENT)
    gone = treat("security_service", "audit_record_1", DataCategory.SECURITY_AUDIT_LOG, st, rules, True)
    assert (gone.finding, gone.remediation, gone.verification) == (
        Operation.REPORT_REMAINING_RECORD, Operation.DELETE_RECORD, Operation.VERIFY_RECORD_ABSENT)
    assert gone.note == RETENTION_EXPIRED


def _copy_dataset(tmp_path: Path) -> Path:
    ds = tmp_path / "dataset"
    shutil.copytree(DATASET_DIR, ds)
    return ds


def test_passed_retention_date_turns_a_verified_audit_record_into_a_finding(tmp_path):
    # Synthetic: evidence_92 states audit_record_3737 (case_21) is retained
    # until a date before case_21's audit_at. Every real date is in the future.
    req = SAMPLES["case_21"]
    before, _ = decide_case(WORLD, req)
    assert ("VERIFY_RECORD_PRESENT", "audit_record_3737") in {(s.operation.value, s.resource_id)
                                                               for s in before.steps}
    ds = _copy_dataset(tmp_path)
    path = ds / "evidence/records/service_resource_export_20260915_114455.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    hits = 0
    for entries in data["service_exports"]["security_service"].values():
        for e in entries:
            if e.get("resource_id") == "audit_record_3737":
                e["retention_until"] = "2026-01-01T00:00:00Z"
                hits += 1
    assert hits == 1
    path.write_text(json.dumps(data), encoding="utf-8")
    world = World.load(ds)
    after, trace = decide_case(world, req)
    assert after.status is DeletionStatus.INCOMPLETE
    got = {(s.operation.value, s.evidence_id) for s in after.steps if s.resource_id == "audit_record_3737"}
    assert got == {("REPORT_REMAINING_RECORD", "evidence_92"), ("DELETE_RECORD", ""), ("VERIFY_RECORD_ABSENT", "")}
    # the policy contract allows that DELETE only because the trace says why
    steps = as_dicts(after)
    assert policy_issues("case_21", steps, categories_from_trace(trace)) == []
    assert [i.code for i in policy_issues("case_21", steps, None)] == ["delete_on_kept_record"]
    # and the guards re-derive the same row from its citations
    ctx = GuardContext(world, CitationRegistry.load(ds))
    assert guard_decision(after, trace, req, ctx)[0] == after


# ---------------------------------------------------------------- shared attachments

def test_attachment_subjects():
    owner = {"a@x.io": "user_1", "b@x.io": "user_2"}.get
    att = lambda *ids: _obs(category=DataCategory.SUPPORT_ATTACHMENT, ids=ids)  # noqa: E731
    assert not shared_attachment([att("a@x.io")], owner)
    assert shared_attachment([att("a@x.io", "b@x.io")], owner)
    assert shared_attachment([att("a@x.io", "stranger@y.io")], owner)
    assert not shared_attachment([att("a@x.io", "+44 20 7946 0000")], owner)  # an unknown phone is not a person
    msg = _obs(category=DataCategory.SUPPORT_MESSAGE, ids=("a@x.io", "b@x.io"))
    assert not shared_attachment([msg], owner)  # only attachments are judged this way


def test_policy_diff_leaves_shared_attachments_alone():
    rules = load_rules()
    st = ResourceState(system_id="support_platform", resource_id="support_1",
                       data_category=DataCategory.SUPPORT_ATTACHMENT, state=ObservedState.PRESENT,
                       evidence_id="evidence_x", as_of=datetime(2026, 1, 1, tzinfo=UTC))
    own = Scoped("support_platform", "support_1", DataCategory.SUPPORT_ATTACHMENT, st)
    treatments, unmapped = policy_diff([own], rules)
    assert [t.remediation for t in treatments] == [Operation.DELETE_RECORD]
    treatments, unmapped = policy_diff([own._replace(shared=True)], rules)
    assert treatments == [] and unmapped == []


def test_shared_attachment_in_a_real_case_gets_no_delete_and_is_flagged(tmp_path):
    # Synthetic: the ticket attachment support_4665 (case_479) also shows a
    # second known customer's email.
    req = EVAL["case_479"]
    before, _ = decide_case(WORLD, req)
    assert ("DELETE_RECORD", "support_4665") in {(s.operation.value, s.resource_id) for s in before.steps}
    ds = _copy_dataset(tmp_path)
    with open(ds / "user_identities.csv", newline="", encoding="utf-8") as f:
        other = next(r["email"] for r in csv.DictReader(f) if r["user_id"] == "user_501")
    path = ds / "evidence/records/customer_data_snapshot_bundle_20260223.json"  # evidence_45
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    hits = 0
    for snap in data["snapshots"]:
        for rec in snap.get("records") or []:
            if rec.get("resource_id") == "support_4665":
                key = next(k for k, v in rec.items() if isinstance(v, str) and re.search(r"@", v))
                rec["message_excerpt"] = f"forwarded by {other}; original owner {rec[key]}"
                hits += 1
    assert hits >= 1
    path.write_text(json.dumps(data), encoding="utf-8")
    after, trace = decide_case(World.load(ds), req)
    assert all(s.resource_id != "support_4665" for s in after.steps)
    assert "SHARED_ATTACHMENT" in trace["flags_fired"]


def test_no_shared_attachment_in_the_real_data():
    for cid, req in EVAL.items():
        _, t = decide_case(WORLD, req)
        assert "SHARED_ATTACHMENT" not in t["flags_fired"], cid


# ---------------------------------------------------------------- cross-row

def _decisions(statuses):
    return [Decision(case_id=f"case_{i}", status=DeletionStatus(s), steps=()) for i, s in enumerate(statuses, 1)]


def test_cross_row_counts_and_drift():
    ref = sample_mix(DATASET_DIR)
    assert ref == {"COMPLETE": 10 / 32, "INCOMPLETE": 22 / 32, "NOT_APPLICABLE": 0.0}
    statuses = ["COMPLETE"] * 154 + ["INCOMPLETE"] * 354 + ["NOT_APPLICABLE"] * 10
    ds = _decisions(statuses)
    reqs = [{"case_id": d.case_id} for d in ds]
    ok = cross_row_check(ds, reqs, Path("deletion_requests.csv"), ref)
    assert ok["errors"] == [] and ok["warnings"] == [] and ok["drift"] < 0.05
    short = cross_row_check(ds[:-1], reqs, Path("deletion_requests.csv"), ref)
    assert len(short["errors"]) == 2
    skewed = _decisions(["NOT_APPLICABLE"] * 200 + ["INCOMPLETE"] * 318)
    warn = cross_row_check(skewed, [{"case_id": d.case_id} for d in skewed], Path("deletion_requests.csv"), ref)
    assert warn["errors"] == [] and len(warn["warnings"]) == 1


def test_rederive_is_identity_on_engine_rows():
    for cid, req in SAMPLES.items():
        d, _ = decide_case(WORLD, req)
        status, steps = rederive(as_dicts(d), parse_ts(req["audit_at"]), WORLD)
        if d.steps:
            assert (status, tuple(steps)) == (d.status, d.steps), cid
