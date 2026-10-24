"""Evaluation harness.

    python evaluation/main.py --suite contract [--output PATH --requests PATH]
    python evaluation/main.py --suite samples  [--pred PATH] [--mode tune|measure]

Reports are written to code/runs/<mode>/. Only --mode measure numbers go into
evaluation_report.md; tune is for iteration.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ghostaudit import CODE_ROOT, DATASET_DIR, REPO_ROOT  # noqa: E402
from ghostaudit.guards import check_output, load_requests  # noqa: E402
from ghostaudit.identity import IdentityIndex  # noqa: E402
from ghostaudit.ingest.manifest import CitationRegistry, Manifest, parse_ts  # noqa: E402
from ghostaudit.ingest.observations import build_table  # noqa: E402
from ghostaudit.schemas import OPERATION_ORDER, DeletionStatus, case_sort_key  # noqa: E402

STEP_KEY_FIELDS = ("phase", "operation", "state", "system_id", "resource_id", "evidence_id")
SCORED_FIELDS = ("operation", "state", "system_id", "resource_id", "evidence_id")
STATUSES = [s.value for s in DeletionStatus]


# ---------------------------------------------------------------- contract

def suite_contract(args: argparse.Namespace) -> int:
    output = Path(args.output)
    requests = load_requests(Path(args.requests))
    issues = check_output(output, requests, CitationRegistry.load(Path(args.dataset)))
    report = {
        "suite": "contract",
        "output": str(output),
        "requests": str(args.requests),
        "rows_expected": len(requests),
        "violations": len(issues),
        "by_code": dict(sorted(Counter(i.code for i in issues).items())),
        "issues": [i.__dict__ for i in issues],
    }
    _write_report(args, "contract", report)
    print(f"contract: {output} vs {len(requests)} requests -> {len(issues)} violations")
    for code, n in report["by_code"].items():
        print(f"  {code}: {n}")
    for i in issues[:20]:
        print(f"  [{i.code}] {i.case_id}: {i.detail}")
    return 0 if not issues else 1


# ---------------------------------------------------------------- samples

def _read_output(path: Path) -> dict[str, tuple[str, list[dict]]]:
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                steps = json.loads(r["steps_json"])
            except json.JSONDecodeError:
                steps = []
            rows[r["case_id"]] = (r["deletion_status"], steps if isinstance(steps, list) else [])
    return rows


def _key(step: dict) -> tuple:
    return tuple(step.get(k) for k in STEP_KEY_FIELDS)


def align(expected: list[dict], actual: list[dict]) -> tuple[list[tuple[dict, dict]], list[dict], list[dict]]:
    """Pair expected with actual steps: exact match, then same
    (phase, system_id, resource_id), then same (phase, resource_id), then
    remaining in order within the phase. Returns (pairs, missed, extra)."""
    exp_left = list(expected)
    act_left = list(actual)
    pairs: list[tuple[dict, dict]] = []
    for keyfn in (
        _key,
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


def _load_trace(traces: Path, case_id: str) -> list[dict]:
    p = traces / f"{case_id}.json"
    if not p.is_file():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("observations", [])
    except (json.JSONDecodeError, AttributeError):
        return []


def _fmt(s: dict | None) -> str:
    if s is None:
        return "—"
    return f"{s.get('phase','?')[:1]} {s.get('operation')} {s.get('system_id')}/{s.get('resource_id')} " \
           f"{s.get('state')} [{s.get('evidence_id') or ''}]"


def suite_samples(args: argparse.Namespace) -> int:
    expected = _read_output(Path(args.expected))
    actual = _read_output(Path(args.pred))
    traces = Path(args.traces)
    case_ids = sorted(expected, key=case_sort_key)

    confusion = {e: {a: 0 for a in STATUSES + ["MISSING"]} for e in STATUSES}
    status_ok = 0
    tp = n_exp = n_act = 0
    set_exact = seq_exact = 0
    field_ok = Counter()
    per_op = defaultdict(lambda: {"expected": 0, "matched": 0, "predicted": 0})
    per_case = []
    dump = [f"# Samples dump\n\nexpected: `{args.expected}`  \npredicted: `{args.pred}`\n"]

    for cid in case_ids:
        e_status, e_steps = expected[cid]
        a_status, a_steps = actual.get(cid, ("MISSING", []))
        confusion[e_status][a_status if a_status in confusion[e_status] else "MISSING"] += 1
        s_ok = e_status == a_status
        status_ok += s_ok

        e_ms = Counter(_key(s) for s in e_steps)
        a_ms = Counter(_key(s) for s in a_steps)
        case_tp = sum((e_ms & a_ms).values())
        tp += case_tp
        n_exp += len(e_steps)
        n_act += len(a_steps)
        set_ok = e_ms == a_ms
        seq_ok = e_steps == a_steps
        set_exact += set_ok
        seq_exact += seq_ok

        for s in e_steps:
            per_op[s["operation"]]["expected"] += 1
        for s in a_steps:
            per_op[str(s.get("operation"))]["predicted"] += 1
        for k in (e_ms & a_ms).elements():
            per_op[k[1]]["matched"] += 1

        pairs, missed, extra = align(e_steps, a_steps)
        for e, a in pairs:
            for f in SCORED_FIELDS:
                field_ok[f] += e.get(f) == a.get(f)

        per_case.append({"case_id": cid, "status_ok": s_ok, "set_exact": set_ok, "sequence_exact": seq_ok,
                         "expected_steps": len(e_steps), "predicted_steps": len(a_steps),
                         "matched_steps": case_tp})

        if not (s_ok and seq_ok) or args.dump_all:
            obs = _load_trace(traces, cid)
            dump.append(f"\n## {cid}: expected {e_status}, predicted {a_status}"
                        f"{'' if s_ok else '  ← STATUS'}{'' if set_ok else '  ← STEPS'}"
                        f"{'' if seq_ok or not set_ok else '  ← ORDER'}\n")
            dump.append("| # | expected | predicted |\n|---|---|---|")
            for i in range(max(len(e_steps), len(a_steps))):
                e = e_steps[i] if i < len(e_steps) else None
                a = a_steps[i] if i < len(a_steps) else None
                mark = "" if e == a else " ✗"
                dump.append(f"| {i + 1}{mark} | {_fmt(e)} | {_fmt(a)} |")
            if missed:
                dump.append("\nmissed: " + "; ".join(_fmt(s) for s in missed))
            if extra:
                dump.append("\nextra: " + "; ".join(_fmt(s) for s in extra))
            if obs:
                dump.append("\nobservations behind predicted steps:")
                wanted = {s.get("resource_id") for s in a_steps} | {s.get("resource_id") for s in e_steps}
                for o in obs:
                    if o.get("resource_id") in wanted:
                        dump.append(f"- {json.dumps(o, sort_keys=True)}")
            elif a_steps:
                dump.append(f"\n(no trace at {traces / (cid + '.json')})")

    n = len(case_ids)
    precision = tp / n_act if n_act else (1.0 if not n_exp else 0.0)
    recall = tp / n_exp if n_exp else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics = {
        "cases": n,
        "status_accuracy": status_ok / n,
        "step_precision": precision,
        "step_recall": recall,
        "step_f1": f1,
        "step_set_exact_rate": set_exact / n,
        "sequence_exact_rate": seq_exact / n,
        "field_accuracy": {f: (field_ok[f] / n_exp if n_exp else 1.0) for f in SCORED_FIELDS},
    }
    report = {
        "suite": "samples",
        "expected": str(args.expected),
        "pred": str(args.pred),
        "metrics": metrics,
        "confusion": confusion,
        "per_operation": {op: per_op[op] for op in sorted(per_op, key=lambda o: OPERATION_ORDER.get(o, 99))},
        "per_case": per_case,
    }
    _write_report(args, "samples", report)
    dump_path = _out_dir(args) / "samples_dump.md"
    dump_path.write_text("\n".join(dump) + "\n", encoding="utf-8")

    print(f"samples: {n} cases, pred={args.pred}")
    for k in ("status_accuracy", "step_precision", "step_recall", "step_f1",
              "step_set_exact_rate", "sequence_exact_rate"):
        print(f"  {k:22s} {metrics[k]:.3f}")
    for f, v in metrics["field_accuracy"].items():
        print(f"  field.{f:16s} {v:.3f}")
    print("  confusion (expected -> predicted):")
    for e in STATUSES:
        print(f"    {e:15s} " + " ".join(f"{a}={c}" for a, c in confusion[e].items() if c))
    print(f"  dump: {dump_path}")
    perfect = all(v == 1.0 for k, v in metrics.items() if isinstance(v, float)) and \
        all(v == 1.0 for v in metrics["field_accuracy"].values())
    return 0 if perfect else 1


# ---------------------------------------------------------------- shared world

def _formats(args: argparse.Namespace) -> set[str] | None:
    return set(args.formats.split(",")) if args.formats else None


def _world(args: argparse.Namespace):
    dataset = Path(args.dataset)
    identity = IdentityIndex.load(dataset)
    manifest = Manifest.load(dataset)
    table = build_table(manifest, identity, _formats(args))
    return identity, manifest, table


def _requests(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return sorted(csv.DictReader(f), key=lambda r: case_sort_key(r["case_id"]))


def _gold_resources(steps: list[dict]) -> set[tuple[str, str]]:
    return {(s["system_id"], s["resource_id"]) for s in steps}


def _obs_dict(o) -> dict:
    d = o.model_dump(mode="json")
    d["identifiers"] = list(o.identifiers)
    return d


# ---------------------------------------------------------------- coverage

def suite_coverage(args: argparse.Namespace) -> int:
    identity, manifest, table = _world(args)
    dataset = Path(args.dataset)

    by_format: dict[str, Counter] = defaultdict(Counter)
    for st in table.stats:
        c = by_format[st.format]
        c["files"] += 1
        c[st.status] += 1
        c["rows"] += st.rows
        c["observations"] += st.observations
        c["coverage"] += st.coverage
        c["skipped"] += st.skipped

    # Sample cases: does the file the answer cites yield anything for the user?
    gold = _read_output(Path(args.expected))
    status_of = {st.file_id: st.status for st in table.stats}
    sample_rows = []
    dump = ["# Coverage: sample cases whose cited file yields no observation for the user\n"]
    for r in _requests(dataset / "sample_deletion_requests.csv"):
        cid = r["case_id"]
        audit_at = parse_ts(r["audit_at"])
        uid = identity.resolve_email(r["email"]).user_id
        steps = gold.get(cid, ("", []))[1]
        cited = sorted({s["evidence_id"] for s in steps if s["evidence_id"]})
        linked = table.user_resources(uid, audit_at) if uid else set()
        case_obs = table.case_observations(uid, audit_at) if uid else []
        row = {"case_id": cid, "user_id": uid, "cited": cited, "files": {}}
        for fid in cited:
            direct = sum(1 for o in table.by_user.get(uid, ()) if o.file_id == fid)
            closure = sum(1 for o in case_obs if o.file_id == fid)
            row["files"][fid] = {"status": status_of.get(fid, "not_evidence_file"), "direct": direct,
                                 "closure": closure}
        row["direct_hit"] = any(v["direct"] for v in row["files"].values())
        row["closure_hit"] = any(v["closure"] for v in row["files"].values())
        row["cited_parsed"] = any(v["status"] == "parsed" for v in row["files"].values())
        g = _gold_resources(steps)
        row["gold_resources"] = len(g)
        row["gold_linked"] = len(g & linked)
        sample_rows.append(row)
        if row["cited_parsed"] and not row["closure_hit"]:
            dump.append(f"\n## {cid} ({uid}, audit_at {r['audit_at']}) cites {cited}\n")
            for res in sorted(g):
                seen = table.by_resource.get(res, [])
                where = "; ".join(f"{o.file_id}@{o.captured_at:%Y-%m-%d} {o.state} ids={list(o.identifiers)}"
                                  for o in seen) or "no observation anywhere"
                dump.append(f"- gold {res[0]}/{res[1]} {'LINKED' if res in linked else 'unlinked'}: {where}")

    # All evaluation cases.
    kinds = Counter()
    zero = []
    obs_hist = Counter()
    per_case = []
    for r in _requests(Path(args.requests)):
        audit_at = parse_ts(r["audit_at"])
        res = identity.resolve_email(r["email"])
        kinds[res.kind.value] += 1
        obs = table.case_observations(res.user_id, audit_at) if res.user_id else []
        if not obs:
            zero.append(r["case_id"])
        obs_hist[min(len(obs), 10)] += 1
        per_case.append({"case_id": r["case_id"], "identity": res.kind.value, "observations": len(obs),
                         "files": sorted({o.file_id for o in obs}),
                         "resources": sorted({f"{o.system_id}/{o.resource_id}" for o in obs})})

    conflicts = {}
    for res, obs in table.by_resource.items():
        owners = set().union(*(table.owners(o) for o in obs)) if obs else set()
        if len(owners) > 1:
            conflicts[f"{res[0]}/{res[1]}"] = sorted(owners)

    report = {
        "suite": "coverage", "formats_enabled": args.formats or "all",
        "by_format": {k: dict(v) for k, v in sorted(by_format.items())},
        "files": [st.__dict__ for st in table.stats],
        "samples": sample_rows,
        "identity": dict(kinds), "zero_observation_cases": zero,
        "observation_histogram": {str(k): v for k, v in sorted(obs_hist.items())},
        "owner_conflicts": conflicts, "per_case": per_case,
    }
    _write_report(args, "coverage", report)
    dump_path = _out_dir(args) / "coverage_dump.md"
    dump_path.write_text("\n".join(dump) + "\n", encoding="utf-8")

    print(f"coverage (formats={args.formats or 'all'})")
    print(f"  {'format':6s} {'files':>5s} {'parsed':>6s} {'failed':>6s} {'deferred':>8s} {'rows':>5s} "
          f"{'obs':>5s} {'cov':>4s} {'skip':>4s}")
    for fmt, c in sorted(by_format.items()):
        print(f"  {fmt:6s} {c['files']:5d} {c['parsed']:6d} {c['failed']:6d} {c['deferred']:8d} "
              f"{c['rows']:5d} {c['observations']:5d} {c['coverage']:4d} {c['skipped']:4d}")
    n = len(sample_rows)
    print(f"  samples: cited file parsed {sum(r['cited_parsed'] for r in sample_rows)}/{n}; "
          f"obs for user from cited file: direct-identifier {sum(r['direct_hit'] for r in sample_rows)}/{n}, "
          f"via linked resource {sum(r['closure_hit'] for r in sample_rows)}/{n}")
    gl = sum(r["gold_linked"] for r in sample_rows)
    gt = sum(r["gold_resources"] for r in sample_rows)
    print(f"  samples: gold resources linked to the user by identifier {gl}/{gt}")
    print(f"  eval cases: identity {dict(kinds)}; zero-observation cases {len(zero)}/{len(per_case)}")
    print(f"  resources claimed by more than one user: {len(conflicts)}")
    print(f"  dump: {dump_path}")
    return 0


# ---------------------------------------------------------------- samples, observations only

def suite_samples_observations(args: argparse.Namespace) -> int:
    identity, manifest, table = _world(args)
    dataset = Path(args.dataset)
    gold = _read_output(Path(args.expected))
    traces = Path(args.traces)
    traces.mkdir(parents=True, exist_ok=True)
    tp = n_gold = n_pred = 0
    exact = 0
    dump = [f"# Samples, observations only (formats={args.formats or 'all'})\n"]
    for r in _requests(dataset / "sample_deletion_requests.csv"):
        cid = r["case_id"]
        audit_at = parse_ts(r["audit_at"])
        uid = identity.resolve_email(r["email"]).user_id
        steps = gold[cid][1]
        g = _gold_resources(steps)
        obs = table.case_observations(uid, audit_at) if uid else []
        pred = {(o.system_id, o.resource_id) for o in obs}
        tp += len(g & pred)
        n_gold += len(g)
        n_pred += len(pred)
        exact += g == pred
        (traces / f"{cid}.json").write_text(json.dumps(
            {"case_id": cid, "user_id": uid, "audit_at": r["audit_at"],
             "observations": [_obs_dict(o) for o in obs]}, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        if g == pred and not args.dump_all:
            continue
        dump.append(f"\n## {cid} ({uid}, audit_at {r['audit_at']}): gold {len(g)}, found {len(pred)}, "
                    f"matched {len(g & pred)}\n")
        for res in sorted(g | pred):
            tag = "ok" if res in g and res in pred else ("MISSING" if res in g else "extra")
            mine = [o for o in obs if (o.system_id, o.resource_id) == res]
            if mine:
                where = "; ".join(f"{o.file_id}@{o.observed_at:%m-%dT%H:%M} {o.state}" for o in mine)
            else:
                anywhere = table.by_resource.get(res, [])
                where = "not linked; elsewhere: " + ("; ".join(
                    f"{o.file_id}@{o.captured_at:%m-%d}{'(late)' if o.captured_at > audit_at else ''} "
                    f"{o.state} ids={list(o.identifiers)}" for o in anywhere) or "nowhere in parsed evidence")
            dump.append(f"- [{tag}] {res[0]}/{res[1]}: {where}")
    n = len(gold)
    recall = tp / n_gold if n_gold else 1.0
    precision = tp / n_pred if n_pred else 0.0
    report = {"suite": "samples-observations", "formats_enabled": args.formats or "all",
              "resource_recall": recall, "resource_precision": precision, "resource_set_exact": exact / n}
    _write_report(args, "samples_observations", report)
    dump_path = _out_dir(args) / "samples_observations_dump.md"
    dump_path.write_text("\n".join(dump) + "\n", encoding="utf-8")
    print(f"samples (observations only, formats={args.formats or 'all'}): {n} cases")
    print(f"  gold resources found   {tp}/{n_gold} (recall {recall:.3f})")
    print(f"  found resources in gold {tp}/{n_pred} (precision {precision:.3f})")
    print(f"  resource set exact     {exact}/{n}")
    print(f"  dump: {dump_path}")
    return 0


# ---------------------------------------------------------------- stubs

def _stub(name: str):
    def run(args: argparse.Namespace) -> int:
        print(f"suite '{name}' is not implemented yet", file=sys.stderr)
        return 2
    return run


SUITES = {
    "contract": suite_contract,
    "samples": suite_samples,
    "coverage": suite_coverage,
    "custom": _stub("custom"),
    "metamorphic": _stub("metamorphic"),
    "complete-check": _stub("complete-check"),
}


def _out_dir(args: argparse.Namespace) -> Path:
    d = CODE_ROOT / "runs" / args.mode
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_report(args: argparse.Namespace, suite: str, report: dict) -> None:
    report["mode"] = args.mode
    (_out_dir(args) / f"{suite}_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", required=True, choices=list(SUITES))
    p.add_argument("--mode", choices=["tune", "measure"], default="tune")
    p.add_argument("--dataset", default=str(DATASET_DIR))
    p.add_argument("--output", default=str(REPO_ROOT / "output.csv"), help="contract: CSV to check")
    p.add_argument("--requests", default=str(DATASET_DIR / "deletion_requests.csv"),
                   help="contract: requests the output must cover")
    p.add_argument("--expected", default=str(DATASET_DIR / "sample_output.csv"), help="samples: answer key")
    p.add_argument("--pred", default=None, help="samples: predictions (default runs/<mode>/sample_output.csv)")
    p.add_argument("--traces", default=str(CODE_ROOT / "traces"))
    p.add_argument("--dump-all", action="store_true", help="samples: dump every case, not only mismatches")
    p.add_argument("--observations-only", action="store_true",
                   help="samples: compare resources found in evidence with the answer key (no policy)")
    p.add_argument("--formats", default=None, help="comma list of evidence formats to parse (default all)")
    args = p.parse_args(argv)
    if args.pred is None:
        args.pred = str(CODE_ROOT / "runs" / args.mode / "sample_output.csv")
    if args.suite == "samples" and args.observations_only:
        return suite_samples_observations(args)
    return SUITES[args.suite](args)


if __name__ == "__main__":
    raise SystemExit(main())
