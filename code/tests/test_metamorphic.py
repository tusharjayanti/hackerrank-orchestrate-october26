"""Metamorphic checks on the 32 samples (offline: no model is called)."""

import json
import shutil
import sys
from pathlib import Path

from ghostaudit import CODE_ROOT, DATASET_DIR

sys.path.insert(0, str(CODE_ROOT / "evaluation"))
import metamorphic as mm  # noqa: E402

REQUESTS = mm.requests_of(DATASET_DIR)


def test_all_metamorphic_checks_pass():
    checks = {c.name: c for c in mm.run_checks(DATASET_DIR, REQUESTS)}
    assert set(checks) == {"rerun_identical", "manifest_order", "email_case", "decoy_user", "late_file_vanishes"}
    for c in checks.values():
        assert c.ok, f"{c.name}: {c.detail} {c.changed}"


def test_checks_are_not_vacuous(tmp_path):
    # Dropping a cited file from the manifest must change some rows, or the
    # equality checks above would prove nothing.
    base = mm.decide_rows(DATASET_DIR, REQUESTS)
    fid = mm.pick_cited_text_file(DATASET_DIR, base, with_email=False)
    ds = tmp_path / "dataset"
    shutil.copytree(DATASET_DIR, ds)
    header, rows = mm._read_csv(ds / "evidence_files.csv")
    mm._write_csv(ds / "evidence_files.csv", header, [r for r in rows if r["file_id"] != fid])
    changed = mm.diff(base, mm.decide_rows(ds, REQUESTS))
    assert changed
    assert all(fid not in {s["evidence_id"] for s in json.loads(sj)}
               for _, sj in mm.decide_rows(ds, REQUESTS).values())


def test_decoy_is_a_near_miss_that_no_evidence_mentions(tmp_path):
    ds = tmp_path / "dataset"
    shutil.copytree(DATASET_DIR, ds)
    subject = REQUESTS[0]["email"]
    mm.inject_decoy(ds, subject)
    _, rows = mm._read_csv(ds / "user_identities.csv")
    decoy = rows[-1]["email"]
    local, domain = subject.split("@")
    assert decoy.startswith(local) and decoy.endswith("@" + domain) and decoy != subject
    assert not any(decoy.lower() in p.read_text(encoding="utf-8", errors="ignore").lower()
                   for p in Path(ds / "evidence").rglob("*") if p.suffix in mm.TEXT_FORMATS)
