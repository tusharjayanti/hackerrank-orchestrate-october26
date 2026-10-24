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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ghostaudit import CODE_ROOT, DATASET_DIR, REPO_ROOT  # noqa: E402
from ghostaudit.guards import check_output, load_requests  # noqa: E402
from ghostaudit.identity import IdentityIndex  # noqa: E402
from ghostaudit.ingest.manifest import CitationRegistry, Manifest, parse_ts  # noqa: E402
from ghostaudit.engine.decide import World, case_context, decide_case  # noqa: E402
from ghostaudit.intent import classify  # noqa: E402
from ghostaudit.engine.state import out_of_scope  # noqa: E402
from ghostaudit.ingest.activity import Activity  # noqa: E402
from ghostaudit.ingest.observations import build_table  # noqa: E402
from ghostaudit.llm import LLM, credentials_present  # noqa: E402
from ghostaudit.schemas import LinkLayer, case_sort_key  # noqa: E402
from scoring import STATUSES, read_output, render_dump, score  # noqa: E402


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

def suite_samples(args: argparse.Namespace) -> int:
    expected = read_output(Path(args.expected))
    actual = read_output(Path(args.pred))
    result = score(expected, actual)
    _write_report(args, "samples", {"suite": "samples", "expected": str(args.expected), "pred": str(args.pred),
                                    "metrics": result.metrics, "confusion": result.confusion,
                                    "per_operation": result.per_operation, "per_case": result.per_case})
    dump_path = _out_dir(args) / "samples_dump.md"
    dump_path.write_text(render_dump(expected, actual, Path(args.traces), args.dump_all,
                                     f"# Samples dump\n\nexpected: `{args.expected}`  \npredicted: `{args.pred}`\n"),
                         encoding="utf-8")
    _print_samples(result, args.pred, dump_path)
    return 0 if result.perfect else 1


def _print_samples(result, pred: str, dump_path: Path) -> None:
    m = result.metrics
    print(f"samples: {m['cases']} cases, pred={pred}")
    for k in ("status_accuracy", "step_precision", "step_recall", "step_f1",
              "step_set_exact_rate", "sequence_exact_rate"):
        print(f"  {k:22s} {m[k]:.3f}")
    for f, v in m["field_accuracy"].items():
        print(f"  field.{f:16s} {v:.3f}")
    print("  confusion (expected -> predicted):")
    for e in STATUSES:
        print(f"    {e:15s} " + " ".join(f"{a}={c}" for a, c in result.confusion[e].items() if c))
    print(f"  dump: {dump_path}")


# ---------------------------------------------------------------- shared world

def _formats(args: argparse.Namespace) -> set[str] | None:
    return set(args.formats.split(",")) if args.formats else None


def _world(args: argparse.Namespace):
    dataset = Path(args.dataset)
    identity = IdentityIndex.load(dataset)
    manifest = Manifest.load(dataset)
    llm = LLM() if credentials_present() else None
    table = build_table(manifest, identity, _formats(args), activity=Activity.load(dataset), llm=llm)
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
    by_format = _format_rollup(table)
    sample_rows, dump = _cited_file_coverage(identity, table, read_output(Path(args.expected)),
                                             _requests(dataset / "sample_deletion_requests.csv"))
    kinds, zero, obs_hist, per_case = _eval_case_coverage(identity, table, _requests(Path(args.requests)))
    conflicts = _owner_conflicts(table)
    _write_report(args, "coverage", {
        "suite": "coverage", "formats_enabled": args.formats or "all",
        "by_format": {k: dict(v) for k, v in sorted(by_format.items())},
        "files": [st.__dict__ for st in table.stats], "samples": sample_rows,
        "identity": dict(kinds), "zero_observation_cases": zero,
        "observation_histogram": {str(k): v for k, v in sorted(obs_hist.items())},
        "owner_conflicts": conflicts, "per_case": per_case,
    })
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
    print(f"  samples: gold resources linked to the user by identifier "
          f"{sum(r['gold_linked'] for r in sample_rows)}/{sum(r['gold_resources'] for r in sample_rows)}")
    print(f"  eval cases: identity {dict(kinds)}; zero-observation cases {len(zero)}/{len(per_case)}")
    print(f"  resources claimed by more than one user: {len(conflicts)}")
    print(f"  dump: {dump_path}")
    return 0


def _format_rollup(table) -> dict[str, Counter]:
    by_format: dict[str, Counter] = defaultdict(Counter)
    for st in table.stats:
        c = by_format[st.format]
        c["files"] += 1
        c[st.status] += 1
        c.update(rows=st.rows, observations=st.observations, coverage=st.coverage, skipped=st.skipped)
    return by_format


def _cited_file_coverage(identity, table, gold, requests: list[dict]) -> tuple[list[dict], list[str]]:
    """Per sample case: does the file the answer cites yield anything for the user?"""
    status_of = {st.file_id: st.status for st in table.stats}
    rows = []
    dump = ["# Coverage: sample cases whose cited file yields no observation for the user\n"]
    for r in requests:
        cid, audit_at = r["case_id"], parse_ts(r["audit_at"])
        uid = identity.resolve_email(r["email"]).user_id
        steps = gold.get(cid, ("", []))[1]
        cited = sorted({s["evidence_id"] for s in steps if s["evidence_id"]})
        linked = table.user_resources(uid, audit_at) if uid else set()
        case_obs = table.case_observations(uid, audit_at) if uid else []
        files = {fid: {"status": status_of.get(fid, "not_evidence_file"),
                       "direct": sum(1 for o in table.by_user.get(uid, ()) if o.file_id == fid),
                       "closure": sum(1 for o in case_obs if o.file_id == fid)} for fid in cited}
        g = _gold_resources(steps)
        row = {"case_id": cid, "user_id": uid, "cited": cited, "files": files,
               "direct_hit": any(v["direct"] for v in files.values()),
               "closure_hit": any(v["closure"] for v in files.values()),
               "cited_parsed": any(v["status"] == "parsed" for v in files.values()),
               "gold_resources": len(g), "gold_linked": len(g & linked)}
        rows.append(row)
        if row["cited_parsed"] and not row["closure_hit"]:
            dump.append(f"\n## {cid} ({uid}, audit_at {r['audit_at']}) cites {cited}\n")
            for res in sorted(g):
                where = "; ".join(f"{o.file_id}@{o.captured_at:%Y-%m-%d} {o.state} ids={list(o.identifiers)}"
                                  for o in table.by_resource.get(res, [])) or "no observation anywhere"
                dump.append(f"- gold {res[0]}/{res[1]} {'LINKED' if res in linked else 'unlinked'}: {where}")
    return rows, dump


def _eval_case_coverage(identity, table, requests: list[dict]):
    kinds, obs_hist = Counter(), Counter()
    zero, per_case = [], []
    for r in requests:
        res = identity.resolve_email(r["email"])
        kinds[res.kind.value] += 1
        obs = table.case_observations(res.user_id, parse_ts(r["audit_at"])) if res.user_id else []
        if not obs:
            zero.append(r["case_id"])
        obs_hist[min(len(obs), 10)] += 1
        per_case.append({"case_id": r["case_id"], "identity": res.kind.value, "observations": len(obs),
                         "files": sorted({o.file_id for o in obs}),
                         "resources": sorted({f"{o.system_id}/{o.resource_id}" for o in obs})})
    return kinds, zero, obs_hist, per_case


def _owner_conflicts(table) -> dict[str, list[str]]:
    conflicts = {}
    for res, obs in table.by_resource.items():
        owners = set().union(*(table.owners(o) for o in obs)) if obs else set()
        if len(owners) > 1:
            conflicts[f"{res[0]}/{res[1]}"] = sorted(owners)
    return conflicts


# ---------------------------------------------------------------- samples, observations only

def suite_samples_observations(args: argparse.Namespace) -> int:
    """Linking only (no policy): which gold resources does each layer of the
    ladder tie to the user, and how many of its links are wrong."""
    world = World.load(Path(args.dataset), _formats(args))
    dataset = Path(args.dataset)
    gold = read_output(Path(args.expected))
    layers = list(LinkLayer)
    per_layer = {lay: Counter() for lay in layers}
    n_gold = 0
    scoped_tp = scoped_pred = 0
    still = []
    dump = [f"# Samples, linking only (formats={args.formats or 'all'})\n"]
    for r in _requests(dataset / "sample_deletion_requests.csv"):
        cid = r["case_id"]
        audit_at, reported_at, requested_at = (parse_ts(r[k]) for k in ("audit_at", "reported_at", "requested_at"))
        uid = world.identity.resolve_email(r["email"]).user_id
        g = _gold_resources(gold[cid][1])
        n_gold += len(g)
        links = world.linker.links(case_context(world.identity, r))
        l5_cands = "-"
        for res, lk in links.items():
            per_layer[lk.layer]["links"] += 1
            per_layer[lk.layer]["correct"] += res in g
        scoped = {res for res, lk in links.items()
                  if not out_of_scope(res, world.table, requested_at, audit_at, lk.linked_at)}
        scoped_tp += len(scoped & g)
        scoped_pred += len(scoped)
        missing = sorted(g - set(links))
        extra = sorted(set(links) - g)
        if missing:
            still.append((cid, uid, missing))
        if missing or extra or args.dump_all:
            dump.append(f"\n## {cid} ({uid}) gold {len(g)}, linked {len(links)}, L5 candidates {l5_cands}\n")
            for res in sorted(g | set(links)):
                lk = links.get(res)
                tag = ("ok" if res in g else ("out-of-scope" if res not in scoped else "EXTRA")) if lk else "MISSING"
                how = f"L{int(lk.layer)} {lk.source_id} conf={lk.confidence}" if lk else ""
                dump.append(f"- [{tag}] {res[0]}/{res[1]} {how}")
    cum_tp = cum_links = 0
    rows = []
    print(f"samples, linking ladder: {len(gold)} cases, {n_gold} gold resources")
    print(f"  {'layer':22s} {'links':>5s} {'correct':>7s} {'precision':>9s} {'cum recall':>10s}")
    for lay in layers:
        c = per_layer[lay]
        cum_tp += c["correct"]
        cum_links += c["links"]
        prec = c["correct"] / c["links"] if c["links"] else float("nan")
        rows.append({"layer": int(lay), "name": lay.name, "links": c["links"], "correct": c["correct"],
                     "precision": prec, "cumulative_recall": cum_tp / n_gold,
                     "cumulative_precision": cum_tp / cum_links if cum_links else float("nan")})
        print(f"  L{int(lay)} {lay.name:19s} {c['links']:5d} {c['correct']:7d} {prec:9.3f} "
              f"{cum_tp:4d}/{n_gold} {cum_tp / n_gold:.3f}")
    print(f"  after scope rule: {scoped_tp}/{scoped_pred} linked resources in gold "
          f"(precision {scoped_tp / scoped_pred if scoped_pred else 0:.3f})")
    print(f"  still unlinked: {sum(len(m) for _, _, m in still)} resources in {len(still)} cases")
    for cid, uid, missing in still:
        print(f"    {cid} ({uid}): " + ", ".join(f"{s}/{r}" for s, r in missing))
    report = {"suite": "samples-linking", "layers": rows, "gold_resources": n_gold,
              "scoped_precision": scoped_tp / scoped_pred if scoped_pred else None,
              "still_unlinked": [{"case_id": c, "user_id": u, "resources": [f"{s}/{r}" for s, r in m]}
                                 for c, u, m in still]}
    _write_report(args, "samples_linking", report)
    dump_path = _out_dir(args) / "samples_linking_dump.md"
    dump_path.write_text("\n".join(dump) + "\n", encoding="utf-8")
    print(f"  dump: {dump_path}")
    return 0


# ---------------------------------------------------------------- custom

# Carrier for engine-identity checks: a real INCOMPLETE sample case whose
# request text is swapped for the custom text.
CARRIER_CASE = "case_24"


def suite_custom(args: argparse.Namespace) -> int:
    from intent_gold import INTENT_GOLD, STRIPPED

    world = World.load(Path(args.dataset))
    if world.llm is None:
        print("custom: intent_gold needs model credentials (or a warm cache); none found", file=sys.stderr)
        return 2
    texts = [t for _, _, t, _, _ in INTENT_GOLD] + list(STRIPPED.values())
    got = classify(texts, world.llm)
    rows, failures = [], []
    for n, lang, text, expected, inj_expected in INTENT_GOLD:
        c = got[text]
        ok_label = c.intent.label.value == expected
        ok_inj = c.intent.injection_suspected == inj_expected
        rows.append({"row": n, "lang_expected": lang, "lang": c.language, "expected": expected,
                     "raw_label": c.raw_label, "label": c.intent.label.value, "confidence": c.intent.confidence,
                     "injection_expected": inj_expected, "injection": c.intent.injection_suspected,
                     "flags": list(c.flags), "ok": ok_label and ok_inj})
        if not ok_label:
            failures.append(f"row {n}: expected {expected}, got {c.intent.label.value} (raw {c.raw_label})")
        if not ok_inj:
            failures.append(f"row {n}: injection expected {inj_expected}, got {c.intent.injection_suspected}")

    carrier = next(r for r in _requests(Path(args.dataset) / "sample_deletion_requests.csv")
                   if r["case_id"] == CARRIER_CASE)
    identity_checks = []
    for n, stripped in STRIPPED.items():
        text = next(t for m, _, t, _, _ in INTENT_GOLD if m == n)
        outs = []
        for variant in (text, stripped):
            req = dict(carrier, request_text=variant)
            decision, _ = decide_case(world, req, classified=got[variant])
            row = decision.to_row()
            outs.append((row["deletion_status"], row["steps_json"]))
        same = outs[0] == outs[1]
        identity_checks.append({"row": n, "carrier": CARRIER_CASE, "byte_identical": same,
                                "status": outs[0][0], "stripped_flagged": got[stripped].intent.injection_suspected})
        if not same:
            failures.append(f"row {n}: steps differ from the stripped request")
        if got[stripped].intent.injection_suspected:
            failures.append(f"row {n}: stripped request was flagged as injection")

    n_ok = sum(r["ok"] for r in rows)
    report = {"suite": "custom", "group": "intent_gold", "label_accuracy": f"{n_ok}/{len(rows)}",
              "rows": rows, "injection_identity": identity_checks, "failures": failures}
    _write_report(args, "custom", report)
    print(f"custom/intent_gold: {n_ok}/{len(rows)} rows correct (label and injection flag)")
    print(f"  {'#':>2} {'lang':7s} {'expected':10s} {'raw':10s} {'effective':10s} conf  inj")
    for r in rows:
        mark = "" if r["ok"] else "  <-"
        print(f"  {r['row']:2d} {r['lang_expected']}/{r['lang']:4s} {r['expected']:10s} {r['raw_label']:10s} "
              f"{r['label']:10s} {r['confidence']:5s} {'yes' if r['injection'] else 'no'}{mark}")
    for c in identity_checks:
        print(f"  row {c['row']} vs stripped on {c['carrier']}: byte-identical={c['byte_identical']} "
              f"status={c['status']} stripped flagged={c['stripped_flagged']}")
    for f in failures:
        print(f"  FAIL {f}")
    return 0 if not failures else 1


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
    "custom": suite_custom,
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
