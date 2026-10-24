"""The scorer must score 1.0 on the key against itself and drop below 1.0 on
perturbed predictions, otherwise a perfect self-score proves nothing."""

import csv
import json

from evaluation.main import main
from ghostaudit import CODE_ROOT, DATASET_DIR


def _report():
    return json.loads((CODE_ROOT / "runs" / "tune" / "samples_report.json").read_text())


def test_contract_suite_on_samples_is_clean():
    assert main(["--suite", "contract", "--output", str(DATASET_DIR / "sample_output.csv"),
                 "--requests", str(DATASET_DIR / "sample_deletion_requests.csv")]) == 0


def test_samples_self_score_is_perfect():
    assert main(["--suite", "samples", "--pred", str(DATASET_DIR / "sample_output.csv")]) == 0


def test_samples_detects_perturbation(tmp_path):
    with open(DATASET_DIR / "sample_output.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    steps = json.loads(rows[4]["steps_json"])  # case_05: INCOMPLETE
    steps[0]["evidence_id"] = "evidence_01"
    rows[4]["steps_json"] = json.dumps(steps)
    rows[0]["deletion_status"] = "INCOMPLETE"
    rows.pop()  # drop case_32
    pred = tmp_path / "pred.csv"
    with open(pred, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "deletion_status", "steps_json"])
        w.writeheader()
        w.writerows(rows)
    assert main(["--suite", "samples", "--pred", str(pred)]) == 1
    m = _report()["metrics"]
    assert m["status_accuracy"] == 30 / 32
    assert m["field_accuracy"]["evidence_id"] < 1.0
    assert m["field_accuracy"]["operation"] < 1.0  # case_32's steps are missed entirely
    assert m["step_recall"] < 1.0


def test_complete_check_suite_passes_on_the_answer_key():
    assert main(["--suite", "complete-check", "--output", str(DATASET_DIR / "sample_output.csv"),
                 "--requests", str(DATASET_DIR / "sample_deletion_requests.csv")]) == 0


def test_complete_check_suite_catches_a_refuted_complete_row(tmp_path):
    with open(DATASET_DIR / "sample_output.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    case_30 = next(r for r in rows if r["case_id"] == "case_30")
    case_30["deletion_status"] = "COMPLETE"
    case_30["steps_json"] = json.dumps([{"step_no": 1, "phase": "VERIFICATION", "system_id": "search_service",
                                         "resource_id": "search_9359", "operation": "VERIFY_RECORD_ABSENT",
                                         "state": "ABSENT", "evidence_id": "evidence_92"}])
    pred = tmp_path / "pred.csv"
    with open(pred, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "deletion_status", "steps_json"])
        w.writeheader()
        w.writerows(rows)
    assert main(["--suite", "complete-check", "--output", str(pred),
                 "--requests", str(DATASET_DIR / "sample_deletion_requests.csv")]) == 1
    report = json.loads((CODE_ROOT / "runs" / "tune" / "complete_check_report.json").read_text())
    assert [i["case_id"] for i in report["refuted"]] == ["case_30"]


def test_custom_suite_refuses_to_run_offline():
    assert main(["--suite", "custom"]) == 2
