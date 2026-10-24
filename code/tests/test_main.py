import csv
import importlib.util
import json

from ghostaudit import CODE_ROOT, DATASET_DIR
from ghostaudit.engine.decide import World
from ghostaudit.schemas import LinkLayer

# Loaded by path: evaluation/main.py shares the module name "main".
_spec = importlib.util.spec_from_file_location("ghostaudit_runner", CODE_ROOT / "main.py")
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)


def test_a_failing_case_still_gets_a_flagged_row(tmp_path, monkeypatch):
    with open(DATASET_DIR / "sample_deletion_requests.csv", newline="", encoding="utf-8") as f:
        requests = list(csv.DictReader(f))[:3]
    real = runner.decide_case

    def flaky(world, req, *a, **kw):
        if req["case_id"] == "case_02":
            raise ValueError("boom")
        return real(world, req, *a, **kw)

    monkeypatch.setattr(runner, "decide_case", flaky)
    decisions = runner.decide_all(requests, World.load(), tmp_path, LinkLayer.SIBLING)
    assert [d.case_id for d in decisions] == ["case_01", "case_02", "case_03"]
    failed = decisions[1]
    assert failed.status.value == "COMPLETE" and failed.steps == ()
    assert "NEEDS_REVIEW" in failed.flags and "ENGINE_ERROR:ValueError" in failed.flags
    assert json.loads((tmp_path / "case_02.json").read_text())["error"] == "ValueError('boom')"
