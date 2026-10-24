"""Output guardrails (L3). Contract checks operate on raw serialised rows so the
same code validates our output, the sample answers, and hand-built bad rows.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .ingest.manifest import CitationRegistry, parse_ts
from .schemas import (
    OUTPUT_HEADER,
    PHASE_ORDER,
    STEP_FIELDS,
    TRIPLE_SET,
    DeletionStatus,
    Phase,
    case_sort_key,
    step_sort_key,
)


@dataclass(frozen=True)
class Issue:
    code: str
    case_id: str
    detail: str


Bad = Callable[[str, str], None]


def check_row(case_id: str, status: str, steps_json: str, audit_at: datetime | None,
              registry: CitationRegistry | None) -> list[Issue]:
    issues: list[Issue] = []

    def bad(code: str, detail: str) -> None:
        issues.append(Issue(code, case_id, detail))

    if status not in {s.value for s in DeletionStatus}:
        bad("status_invalid", repr(status))
    steps = _parse_steps(steps_json, bad)
    if steps is None:
        return issues
    by_phase = _check_order(steps, bad)
    if status == DeletionStatus.NOT_APPLICABLE:
        if steps:
            bad("na_with_steps", f"{len(steps)} steps")
        return issues
    if status == DeletionStatus.COMPLETE:
        _check_complete(steps, by_phase, bad)
    if status == DeletionStatus.INCOMPLETE:
        _check_incomplete(by_phase, bad)
    if registry is not None:
        _check_citations(steps, audit_at, registry, bad)
    return issues


def _parse_steps(steps_json: str, bad: Bad) -> list[dict] | None:
    """JSON array of exactly-7-field steps with allowed triples, or None."""
    try:
        steps = json.loads(steps_json)
    except json.JSONDecodeError as e:
        bad("json_invalid", str(e))
        return None
    if not isinstance(steps, list):
        bad("json_not_array", type(steps).__name__)
        return None
    ok = True
    for i, s in enumerate(steps):
        if not isinstance(s, dict) or set(s) != set(STEP_FIELDS):
            bad("step_fields", f"step[{i}] keys={sorted(s) if isinstance(s, dict) else type(s).__name__}")
        elif not isinstance(s["step_no"], int) or isinstance(s["step_no"], bool):
            bad("step_no_type", f"step[{i}] step_no={s['step_no']!r}")
        elif not all(isinstance(s[k], str) for k in STEP_FIELDS if k != "step_no"):
            bad("step_field_type", f"step[{i}]")
        elif (s["phase"], s["operation"], s["state"]) not in TRIPLE_SET:
            bad("triple", f"step[{i}] {s['phase']}/{s['operation']}/{s['state']}")
        else:
            continue
        ok = False
    return steps if ok else None


def _check_order(steps: list[dict], bad: Bad) -> dict[str, list[dict]]:
    """Numbering, phase order, canonical sort, one step per resource per phase."""
    if [s["step_no"] for s in steps] != list(range(1, len(steps) + 1)):
        bad("numbering", f"step_no={[s['step_no'] for s in steps]}")
    phases = [PHASE_ORDER[s["phase"]] for s in steps]
    if phases != sorted(phases):
        bad("phase_order", f"phases={[s['phase'] for s in steps]}")
    keys = [step_sort_key(s["phase"], s["operation"], s["system_id"], s["resource_id"]) for s in steps]
    if keys != sorted(keys):
        bad("step_order", "steps not sorted by (phase, operation, system_id, resource_id)")
    by_phase: dict[str, list[dict]] = {p.value: [] for p in Phase}
    for s in steps:
        by_phase[s["phase"]].append(s)
    for phase, items in by_phase.items():
        dup = [r for r, n in Counter((s["system_id"], s["resource_id"]) for s in items).items() if n > 1]
        if dup:
            bad("resource_repeated", f"{phase} {dup}")
    return by_phase


def _check_complete(steps: list[dict], by_phase: dict[str, list[dict]], bad: Bad) -> None:
    if not steps:
        bad("complete_empty", "COMPLETE needs evidence-backed verification")
    if by_phase["FINDING"] or by_phase["REMEDIATION"]:
        bad("complete_non_verification", "COMPLETE has FINDING/REMEDIATION steps")
    for s in by_phase["VERIFICATION"]:
        if not s["evidence_id"]:
            bad("complete_missing_evidence", f"step {s['step_no']}")


def _check_incomplete(by_phase: dict[str, list[dict]], bad: Bad) -> None:
    """Findings cite evidence; future steps do not; every finding has a
    remediation and a verification ending in the remediated state."""
    findings = by_phase["FINDING"]
    if not findings:
        bad("incomplete_no_finding", "INCOMPLETE needs at least one finding")
    for s in findings:
        if not s["evidence_id"]:
            bad("finding_missing_evidence", f"step {s['step_no']}")
    for s in by_phase["REMEDIATION"] + by_phase["VERIFICATION"]:
        if s["evidence_id"]:
            bad("future_step_has_evidence", f"step {s['step_no']} cites {s['evidence_id']}")
    rem = {(s["system_id"], s["resource_id"]): s for s in by_phase["REMEDIATION"]}
    ver = {(s["system_id"], s["resource_id"]): s for s in by_phase["VERIFICATION"]}
    found = {(s["system_id"], s["resource_id"]) for s in findings}
    for res in sorted(found):
        if res not in rem:
            bad("finding_without_remediation", str(res))
        if res not in ver:
            bad("finding_without_verification", str(res))
    for res, s in sorted(rem.items()):
        if res not in found:
            bad("remediation_without_finding", str(res))
        elif res in ver and ver[res]["state"] != s["state"]:
            bad("verification_state_mismatch", f"{res} remediation={s['state']} verification={ver[res]['state']}")


def _check_citations(steps: list[dict], audit_at: datetime | None, registry: CitationRegistry, bad: Bad) -> None:
    for s in steps:
        eid = s["evidence_id"]
        if not eid:
            continue
        if not registry.known(eid):
            bad("citation_unknown", f"step {s['step_no']} cites {eid}")
        elif audit_at is not None and not registry.citable_at(eid, audit_at):
            bad("citation_after_audit", f"step {s['step_no']} cites {eid} "
                f"({registry.times[eid].isoformat()} > {audit_at.isoformat()})")


def load_requests(path: Path) -> dict[str, dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return {r["case_id"]: r for r in csv.DictReader(f)}


def check_output(output: Path, requests: dict[str, dict], registry: CitationRegistry) -> list[Issue]:
    issues: list[Issue] = []
    with open(output, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        rows = list(reader)
    if header is None or tuple(header) != OUTPUT_HEADER:
        issues.append(Issue("header", "-", f"header={header}"))
        return issues

    ids = []
    for n, row in enumerate(rows, start=2):
        if len(row) != len(OUTPUT_HEADER):
            issues.append(Issue("row_width", f"line{n}", f"{len(row)} columns"))
            continue
        case_id, status, steps_json = row
        ids.append(case_id)
        req = requests.get(case_id)
        audit_at = parse_ts(req["audit_at"]) if req else None
        issues.extend(check_row(case_id, status, steps_json, audit_at, registry))

    if len(rows) != len(requests):
        issues.append(Issue("row_count", "-", f"{len(rows)} rows, expected {len(requests)}"))
    dup = sorted(c for c, n in Counter(ids).items() if n > 1)
    if dup:
        issues.append(Issue("case_duplicate", "-", ",".join(dup)))
    missing = sorted(set(requests) - set(ids), key=case_sort_key)
    extra = sorted(set(ids) - set(requests), key=case_sort_key)
    if missing:
        issues.append(Issue("case_missing", "-", ",".join(missing)))
    if extra:
        issues.append(Issue("case_unexpected", "-", ",".join(extra)))
    if ids != sorted(ids, key=case_sort_key):
        issues.append(Issue("case_order", "-", "rows not sorted by case_id"))
    return issues
