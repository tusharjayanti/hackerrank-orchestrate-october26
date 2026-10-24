"""Scoring predictions against the sample answer key: status, step multiset,
field accuracy on aligned steps, exact sequence, and the per-case dump."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from ghostaudit.schemas import OPERATION_ORDER, DeletionStatus, case_sort_key

STEP_KEY_FIELDS = ("phase", "operation", "state", "system_id", "resource_id", "evidence_id")
SCORED_FIELDS = ("operation", "state", "system_id", "resource_id", "evidence_id")
STATUSES = [s.value for s in DeletionStatus]

Rows = dict[str, tuple[str, list[dict]]]


def read_output(path: Path) -> Rows:
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                steps = json.loads(r["steps_json"])
            except json.JSONDecodeError:
                steps = []
            rows[r["case_id"]] = (r["deletion_status"], steps if isinstance(steps, list) else [])
    return rows


def step_key(step: dict) -> tuple:
    return tuple(step.get(k) for k in STEP_KEY_FIELDS)


def align(expected: list[dict], actual: list[dict]) -> tuple[list[tuple[dict, dict]], list[dict], list[dict]]:
    """Pair expected with actual steps: exact match, then same
    (phase, system_id, resource_id), then same (phase, resource_id), then
    remaining in order within the phase. Returns (pairs, missed, extra)."""
    exp_left = list(expected)
    act_left = list(actual)
    pairs: list[tuple[dict, dict]] = []
    for keyfn in (
        step_key,
        lambda s: (s.get("phase"), s.get("system_id"), s.get("resource_id")),
        lambda s: (s.get("phase"), s.get("resource_id")),
        lambda s: (s.get("phase"),),
    ):
        still = []
        for e in exp_left:
            k = keyfn(e)
            match = next((a for a in act_left if keyfn(a) == k), None)
            if match is None:
                still.append(e)
            else:
                pairs.append((e, match))
                act_left.remove(match)
        exp_left = still
    return pairs, exp_left, act_left


@dataclass
class SampleScore:
    metrics: dict
    confusion: dict
    per_operation: dict
    per_case: list[dict] = field(default_factory=list)

    @property
    def perfect(self) -> bool:
        return all(v == 1.0 for v in self.metrics.values() if isinstance(v, float)) and \
            all(v == 1.0 for v in self.metrics["field_accuracy"].values())


def score(expected: Rows, actual: Rows) -> SampleScore:
    case_ids = sorted(expected, key=case_sort_key)
    confusion = {e: {a: 0 for a in STATUSES + ["MISSING"]} for e in STATUSES}
    totals = Counter()
    field_ok = Counter()
    per_op = defaultdict(lambda: {"expected": 0, "matched": 0, "predicted": 0})
    per_case = []
    for cid in case_ids:
        e_status, e_steps = expected[cid]
        a_status, a_steps = actual.get(cid, ("MISSING", []))
        confusion[e_status][a_status if a_status in confusion[e_status] else "MISSING"] += 1
        e_ms = Counter(step_key(s) for s in e_steps)
        a_ms = Counter(step_key(s) for s in a_steps)
        matched = e_ms & a_ms
        row = {"case_id": cid, "status_ok": e_status == a_status, "set_exact": e_ms == a_ms,
               "sequence_exact": e_steps == a_steps, "expected_steps": len(e_steps),
               "predicted_steps": len(a_steps), "matched_steps": sum(matched.values())}
        per_case.append(row)
        totals.update(status_ok=row["status_ok"], set_exact=row["set_exact"], seq_exact=row["sequence_exact"],
                      tp=row["matched_steps"], n_exp=len(e_steps), n_act=len(a_steps))
        for s in e_steps:
            per_op[s["operation"]]["expected"] += 1
        for s in a_steps:
            per_op[str(s.get("operation"))]["predicted"] += 1
        for k in matched.elements():
            per_op[k[1]]["matched"] += 1
        for e, a in align(e_steps, a_steps)[0]:
            for f in SCORED_FIELDS:
                field_ok[f] += e.get(f) == a.get(f)
    n, n_exp, n_act, tp = len(case_ids), totals["n_exp"], totals["n_act"], totals["tp"]
    precision = tp / n_act if n_act else (1.0 if not n_exp else 0.0)
    recall = tp / n_exp if n_exp else 1.0
    metrics = {
        "cases": n,
        "status_accuracy": totals["status_ok"] / n,
        "step_precision": precision,
        "step_recall": recall,
        "step_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "step_set_exact_rate": totals["set_exact"] / n,
        "sequence_exact_rate": totals["seq_exact"] / n,
        "field_accuracy": {f: (field_ok[f] / n_exp if n_exp else 1.0) for f in SCORED_FIELDS},
    }
    ordered_ops = {op: per_op[op] for op in sorted(per_op, key=lambda o: OPERATION_ORDER.get(o, 99))}
    return SampleScore(metrics, confusion, ordered_ops, per_case)


def _fmt(s: dict | None) -> str:
    if s is None:
        return "—"
    return f"{s.get('phase', '?')[:1]} {s.get('operation')} {s.get('system_id')}/{s.get('resource_id')} " \
           f"{s.get('state')} [{s.get('evidence_id') or ''}]"


def _trace_observations(traces: Path, case_id: str) -> list[dict]:
    p = traces / f"{case_id}.json"
    if not p.is_file():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("observations", [])
    except (json.JSONDecodeError, AttributeError):
        return []


def render_dump(expected: Rows, actual: Rows, traces: Path, dump_all: bool, header: str) -> str:
    """Side-by-side expected vs predicted for every mismatching case, with the
    observations behind the predicted steps."""
    out = [header]
    for cid in sorted(expected, key=case_sort_key):
        e_status, e_steps = expected[cid]
        a_status, a_steps = actual.get(cid, ("MISSING", []))
        s_ok, seq_ok = e_status == a_status, e_steps == a_steps
        set_ok = Counter(map(step_key, e_steps)) == Counter(map(step_key, a_steps))
        if s_ok and seq_ok and not dump_all:
            continue
        out.append(f"\n## {cid}: expected {e_status}, predicted {a_status}"
                   f"{'' if s_ok else '  ← STATUS'}{'' if set_ok else '  ← STEPS'}"
                   f"{'' if seq_ok or not set_ok else '  ← ORDER'}\n")
        out.append("| # | expected | predicted |\n|---|---|---|")
        for i in range(max(len(e_steps), len(a_steps))):
            e = e_steps[i] if i < len(e_steps) else None
            a = a_steps[i] if i < len(a_steps) else None
            out.append(f"| {i + 1}{'' if e == a else ' ✗'} | {_fmt(e)} | {_fmt(a)} |")
        _, missed, extra = align(e_steps, a_steps)
        if missed:
            out.append("\nmissed: " + "; ".join(_fmt(s) for s in missed))
        if extra:
            out.append("\nextra: " + "; ".join(_fmt(s) for s in extra))
        obs = _trace_observations(traces, cid)
        if obs:
            out.append("\nobservations behind predicted steps:")
            wanted = {s.get("resource_id") for s in a_steps + e_steps}
            out.extend(f"- {json.dumps(o, sort_keys=True)}" for o in obs if o.get("resource_id") in wanted)
        elif a_steps:
            out.append(f"\n(no trace at {traces / (cid + '.json')})")
    return "\n".join(out) + "\n"
