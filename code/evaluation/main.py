"""Evaluation harness.

    python evaluation/main.py --suite contract [--output PATH --requests PATH]
    python evaluation/main.py --suite samples  [--pred PATH] [--mode tune|measure]
    python evaluation/main.py --suite coverage | custom | metamorphic
    python evaluation/main.py --suite complete-check [--output PATH --requests PATH]

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
from ghostaudit.guards import (  # noqa: E402
    categories_from_trace,
    check_output,
    citation_issues,
    complete_check,
    load_requests,
    policy_issues,
)
from ghostaudit.identity import IdentityIndex  # noqa: E402
from ghostaudit.ingest.manifest import CitationRegistry, Manifest, parse_ts  # noqa: E402
from ghostaudit.engine.decide import World, case_context, decide_case  # noqa: E402
from ghostaudit.intent import classify  # noqa: E402
from ghostaudit.engine.state import out_of_scope  # noqa: E402
from ghostaudit.ingest.activity import Activity  # noqa: E402
from ghostaudit.ingest.observations import build_table  # noqa: E402
from ghostaudit.llm import LLM, model_access  # noqa: E402
from ghostaudit.schemas import LinkLayer, case_sort_key  # noqa: E402
from scoring import STATUSES, read_output, render_dump, score  # noqa: E402


# ---------------------------------------------------------------- contract

def suite_contract(args: argparse.Namespace) -> int:
    """Row contract, plus the policy contract (DELETE_RECORD never on kept
    records; no activity-only resource). Category-based policy checks use
    the traces of the run that wrote the output, when one is found."""
    output = Path(args.output)
    requests = load_requests(Path(args.requests))
    issues = check_output(output, requests, CitationRegistry.load(Path(args.dataset)))
    traces = _traces_for(output)
    rows = read_output(output) if output.is_file() else {}
    for cid in sorted(rows, key=case_sort_key):
        trace = _load_trace(traces, cid) if traces else None
        issues.extend(policy_issues(cid, rows[cid][1], categories_from_trace(trace) if trace else None))
    report = {
        "suite": "contract",
        "output": str(output),
        "requests": str(args.requests),
        "traces": str(traces) if traces else None,
        "rows_expected": len(requests),
        "violations": len(issues),
        "by_code": dict(sorted(Counter(i.code for i in issues).items())),
        "issues": [i.__dict__ for i in issues],
    }
    _write_report(args, "contract", report)
    print(f"contract: {output} vs {len(requests)} requests -> {len(issues)} violations "
          f"(policy categories from {'traces ' + str(traces) if traces else 'resource prefixes only: no run traces'})")
    for code, n in report["by_code"].items():
        print(f"  {code}: {n}")
    for i in issues[:20]:
        print(f"  [{i.code}] {i.case_id}: {i.detail}")
    return 0 if not issues else 1


def _traces_for(output: Path) -> Path | None:
    """Traces dir of the newest run whose manifest wrote this output."""
    want = str(output.resolve())
    best = None
    for m in (CODE_ROOT / "runs").glob("*/*/run_manifest.json"):
        try:
            man = json.loads(m.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if str(Path(man.get("output", "")).resolve()) == want and (best is None or m.parent.name > best.name):
            best = m.parent
    return best / "traces" if best else None


def _load_trace(traces: Path, case_id: str) -> dict | None:
    p = traces / f"{case_id}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


# ---------------------------------------------------------------- complete-check

def suite_complete_check(args: argparse.Namespace) -> int:
    """Re-derive every COMPLETE row of an output from its cited evidence alone
    (the same code as the per-row guard) and check every citation of every
    row observes the resource it is cited for."""
    world = World.load(Path(args.dataset))
    output = Path(args.output)
    rows = read_output(output)
    requests = load_requests(Path(args.requests))
    refuted, unsupported, citations, empty = [], [], [], []
    n_complete = 0
    for cid in sorted(rows, key=case_sort_key):
        status, steps = rows[cid]
        req = requests.get(cid)
        if req is None:
            continue
        audit_at = parse_ts(req["audit_at"])
        if status != "NOT_APPLICABLE":
            citations.extend(citation_issues(cid, steps, audit_at, world))
        if status != "COMPLETE":
            continue
        n_complete += 1
        if not steps:
            empty.append(cid)
            continue
        issues, derived_status, _ = complete_check(cid, steps, audit_at, world)
        (refuted if derived_status.value == "INCOMPLETE" else unsupported).extend(issues)
    report = {"suite": "complete-check", "output": str(output), "complete_rows": n_complete,
              "refuted": [i.__dict__ for i in refuted], "unsupported": [i.__dict__ for i in unsupported],
              "citation_issues": [i.__dict__ for i in citations], "complete_without_steps": empty}
    _write_report(args, "complete_check", report)
    bad_rows = {i.case_id for i in refuted + unsupported}
    print(f"complete-check: {output}: {n_complete} COMPLETE rows, "
          f"{n_complete - len(bad_rows) - len(empty)} re-derived exactly from cited evidence")
    print(f"  refuted (cited evidence shows a violation): {len({i.case_id for i in refuted})} rows")
    print(f"  unsupported (cited evidence does not give the claimed state): {len({i.case_id for i in unsupported})}")
    print(f"  citations that do not observe their resource (all rows): {len(citations)}")
    print(f"  COMPLETE with no steps (needs_review fallback): {', '.join(empty) or 'none'}")
    for i in (refuted + unsupported + citations)[:20]:
        print(f"  [{i.code}] {i.case_id}: {i.detail}")
    return 0 if not (refuted or unsupported or citations) else 1


# ---------------------------------------------------------------- metamorphic

def _live_calls() -> int:
    from ghostaudit.llm import CALL_LOG
    if not CALL_LOG.is_file():
        return 0
    return sum(1 for line in CALL_LOG.read_text(encoding="utf-8").splitlines()
               if line.strip() and not json.loads(line).get("cached"))


def suite_metamorphic(args: argparse.Namespace) -> int:
    from metamorphic import run_checks

    live_before = _live_calls()
    checks = run_checks(Path(args.dataset))
    live = _live_calls() - live_before
    _write_report(args, "metamorphic", {"suite": "metamorphic", "cases": "samples (32)",
                                        "live_model_calls": live,
                                        "checks": [c.__dict__ for c in checks]})
    print(f"metamorphic (32 sample requests, live model calls during the suite: {live})")
    for c in checks:
        print(f"  {'PASS' if c.ok else 'FAIL'} {c.name:20s} {c.detail}"
              + (f"; rows changed: {', '.join(c.changed)}" if c.changed and c.name != "late_file_vanishes" else ""))
    return 0 if all(c.ok for c in checks) and live == 0 else 1


# ---------------------------------------------------------------- samples

def suite_samples(args: argparse.Namespace) -> int:
    expected = read_output(Path(args.expected))
    actual = read_output(Path(args.pred))
    result = score(expected, actual)
    _write_report(args, "samples", {"suite": "samples", "expected": str(args.expected), "pred": str(args.pred),
                                    "metrics": result.metrics, "confusion": result.confusion,
                                    "per_operation": result.per_operation, "per_case": result.per_case})
    dump_path = _out_dir(args) / "samples_dump.md"
    traces = Path(args.traces) if args.traces else (_traces_for(Path(args.pred)) or CODE_ROOT / "runs")
    dump_path.write_text(render_dump(expected, actual, traces, args.dump_all,
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
    llm = LLM() if model_access() != "none" else None
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
        "single_file_cases": [c["case_id"] for c in per_case if len(c["files"]) == 1],
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
    single = [c["case_id"] for c in per_case if len(c["files"]) == 1]
    several = [f"{c['case_id']}({len(c['files'])})" for c in per_case if len(c["files"]) > 1]
    print(f"  eval cases whose subject is observed in exactly one eligible evidence file: {len(single)} "
          f"(ids in coverage_report.json single_file_cases); in several: {len(several)}: {', '.join(several)}")
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

def _with_sibling(world):
    """The same world with layer 6 on, solved again over every request."""
    from dataclasses import replace

    from ghostaudit.linking import Linker

    linker = Linker(world.table, world.activity, enable_sibling=True)
    linker.prepare([c for c in (case_context(world.identity, r) for r in world.requests.values()) if c])
    return replace(world, linker=linker)


def suite_samples_observations(args: argparse.Namespace) -> int:
    """Linking only (no policy): which gold resources does each layer of the
    ladder tie to the user, and how many of its links are wrong."""
    world = World.load(Path(args.dataset), _formats(args))
    if args.enable_sibling:
        world = _with_sibling(world)
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
                     "flags": [f"{h.flag.value}:{h.detail}" for h in c.flags], "ok": ok_label and ok_inj})
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

    crosscheck = _intake_crosscheck(world, Path(args.dataset))
    if crosscheck["missing"] or crosscheck["extra"]:
        failures.append(f"intake cross-check: missing {crosscheck['missing']}, extra {crosscheck['extra']}")
    n_ok = sum(r["ok"] for r in rows)
    report = {"suite": "custom", "group": "intent_gold", "label_accuracy": f"{n_ok}/{len(rows)}",
              "rows": rows, "injection_identity": identity_checks, "intake_crosscheck": crosscheck,
              "failures": failures}
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
    print(f"custom/intake_crosscheck: {len(crosscheck['intake_cases'])} cases privacy intake classified "
          f"non-deletion; classifier NOT_APPLICABLE {len(crosscheck['classifier_cases'])}; "
          f"missing {crosscheck['missing']} extra {crosscheck['extra']}")
    for f in failures:
        print(f"  FAIL {f}")
    return 0 if not failures else 1


def _intake_crosscheck(world, dataset: Path) -> dict:
    """Cross-check only, never an input: the evaluation cases whose subject's
    account privacy intake classified as non-deletion (CLASSIFY_REQUEST rows
    in system_activity.csv, within the case's request..audit window) against
    the cases the intent classifier calls NOT_APPLICABLE."""
    from datetime import datetime, timezone

    requests = _requests(dataset / "deletion_requests.csv")
    labels = classify([r["request_text"] for r in requests], world.llm)
    classifier = sorted((r["case_id"] for r in requests
                         if not labels[r["request_text"]].intent.authorizes_deletion), key=case_sort_key)
    far = datetime(9999, 1, 1, tzinfo=timezone.utc)
    intake = set()
    for c in world.activity.classifications:
        owners = world.linker.owners(("account_service", c.resource_id), far)
        for r in requests:
            uid = world.identity.resolve_email(r["email"]).user_id
            if uid in owners and parse_ts(r["requested_at"]) <= c.event_time <= parse_ts(r["audit_at"]):
                intake.add(r["case_id"])
    intake_cases = sorted(intake, key=case_sort_key)
    return {"intake_cases": intake_cases, "classifier_cases": classifier,
            "missing": [c for c in intake_cases if c not in classifier],
            "extra": [c for c in classifier if c not in intake]}


SUITES = {
    "contract": suite_contract,
    "samples": suite_samples,
    "coverage": suite_coverage,
    "custom": suite_custom,
    "metamorphic": suite_metamorphic,
    "complete-check": suite_complete_check,
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
    p.add_argument("--traces", default=None,
                   help="samples: traces dir for the dump (default: the run that wrote --pred)")
    p.add_argument("--enable-sibling", action="store_true",
                   help="samples --observations-only: switch L6 on to measure it (off in every run)")
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
