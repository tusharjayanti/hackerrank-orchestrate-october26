"""Run the deletion audit.

    python code/main.py --mode measure     # all evaluation cases -> output.csv at the repo root
    python code/main.py --mode tune        # same cases -> code/runs/tune/output.csv
    python code/main.py --mode tune --requests dataset/sample_deletion_requests.csv \
        --out code/runs/tune/sample_output.csv

Every run gets a run id (UTC time + request file + git commit). A run writes
one trace per case and a run manifest with the commit, models, flags,
model-access mode, budget state and model usage. Model answers are replayed
from the on-disk cache when the same request was asked before. With no key
but a cache, only cached answers are used (replay); with neither, or with
GHOSTAUDIT_OFFLINE=1, screenshots are deferred, intent is not classified
(every request is audited as a deletion) and the agent does not run, each
with its flag. Every case always gets a row.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import traceback
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ghostaudit import CODE_ROOT, DATASET_DIR, REPO_ROOT  # noqa: E402
from ghostaudit.engine.decide import World, decide_case  # noqa: E402
from ghostaudit.guards import GuardContext, cross_row_check, guard_decision, sample_mix  # noqa: E402
from ghostaudit.ingest.manifest import CitationRegistry  # noqa: E402
from ghostaudit.intent import classify  # noqa: E402
from ghostaudit.llm import CALL_LOG, model_access, model_for  # noqa: E402
from ghostaudit.schemas import (  # noqa: E402
    OUTPUT_HEADER,
    Decision,
    DeletionStatus,
    Flag,
    FlagSet,
    LinkLayer,
    case_sort_key,
)
from ghostaudit.telemetry import get_logger  # noqa: E402

log = get_logger("main")


def git_commit() -> tuple[str, bool]:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
                                    capture_output=True, text=True, check=True).stdout.strip())
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return "nogit", False


def engine_default(case_id: str, reason: str) -> Decision:
    """The row written when a case cannot be decided: no finding is invented
    and the row is marked for review. A missing row is never acceptable."""
    flags = FlagSet()
    flags.add(Flag.ENGINE_ERROR, reason)
    flags.add(Flag.NEEDS_REVIEW, "engine error")
    return Decision(case_id=case_id, status=DeletionStatus.COMPLETE, steps=(), flags=flags.hits)


def decide_all(requests: list[dict], world: World, traces_dir: Path, max_layer: LinkLayer,
               guards: GuardContext | None = None) -> list[Decision]:
    """One guarded row per request. Guards (ghostaudit/guards.py) re-derive
    COMPLETE rows from their cited evidence, check every citation and the
    contract, and regenerate or repair a failing row."""
    try:
        intents = classify([r["request_text"] for r in requests], world.llm)
    except Exception as e:  # noqa: BLE001 - intent failure must not cost any case its row
        log.error("intent_failed", error=repr(e), traceback=traceback.format_exc())
        intents = classify([r["request_text"] for r in requests], None)
    guards = guards or GuardContext(world, CitationRegistry.load(DATASET_DIR))
    decisions = []
    for req in requests:
        intent = intents[req["request_text"]]
        try:
            decision, trace = decide_case(world, req, max_layer, intent)
            # The engine's own decision is the regeneration of a failing row.
            # Regeneration runs the engine alone: a row the guards rejected
            # never goes back through the investigator.
            agent_record = trace.get("agent")
            decision, trace = guard_decision(decision, trace, req, guards,
                                             regenerate=lambda req=req, intent=intent: decide_case(
                                                 world, req, max_layer, intent, use_agent=False))
            if agent_record and agent_record.get("ran") and trace.get("agent") != agent_record:
                # The guards replaced a row the agent had chosen; keep what the
                # agent did so the override is visible in the trace.
                trace["agent_overridden_by_guard"] = agent_record
        except Exception as e:  # noqa: BLE001 - one case must never cost the others their rows
            log.error("case_failed", case_id=req["case_id"], error=repr(e), traceback=traceback.format_exc())
            decision = engine_default(req["case_id"], type(e).__name__)
            trace = {"case_id": req["case_id"], "error": repr(e), **FlagSet(decision.flags).trace()}
        decisions.append(decision)
        (traces_dir / f"{req['case_id']}.json").write_text(
            json.dumps(trace, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return decisions


def write_output(decisions: list[Decision], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(OUTPUT_HEADER)
        for d in sorted(decisions, key=lambda d: case_sort_key(d.case_id)):
            row = d.to_row()
            w.writerow([row[k] for k in OUTPUT_HEADER])


def usage_for(run_id: str) -> dict:
    """Model usage of one run, from the call log (live calls and cache hits)."""
    rows = []
    if CALL_LOG.is_file():
        rows = [r for r in map(json.loads, CALL_LOG.read_text(encoding="utf-8").splitlines())
                if r.get("run_id") == run_id]
    by_model: dict[str, Counter] = {}
    for r in rows:
        c = by_model.setdefault(r["model"], Counter())
        c["cache_hits" if r["cached"] else "live_calls"] += 1
        c.update(input_tokens=r["input_tokens"], output_tokens=r["output_tokens"])
        c["usd_micro"] += round(r["usd"] * 1e6)
    return {m: {**{k: v for k, v in c.items() if k != "usd_micro"}, "usd": c["usd_micro"] / 1e6}
            for m, c in sorted(by_model.items())}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["tune", "measure"], default="tune")
    p.add_argument("--requests", default=str(DATASET_DIR / "deletion_requests.csv"))
    p.add_argument("--out", default=None, help="default: output.csv at the repo root (measure) or runs/tune/")
    p.add_argument("--max-layer", type=int, default=int(LinkLayer.SIBLING), help="highest link layer to use (1-6)")
    p.add_argument("--allow-live", action="store_true",
                   help="measure: allow live model calls (default: cached answers only; a miss is flagged)")
    args = p.parse_args(argv)
    allow_live = args.mode != "measure" or args.allow_live

    sha, dirty = git_commit()
    stem = Path(args.requests).stem.replace("_", "-")
    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{stem}-{sha}{'-dirty' if dirty else ''}"
    run_dir = CODE_ROOT / "runs" / args.mode / run_id
    traces_dir = run_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    default_out = REPO_ROOT / "output.csv" if args.mode == "measure" else CODE_ROOT / "runs" / "tune" / "output.csv"
    out_path = Path(args.out) if args.out else default_out

    with open(args.requests, newline="", encoding="utf-8") as f:
        requests = sorted(csv.DictReader(f), key=lambda r: case_sort_key(r["case_id"]))
    try:
        world = World.load(run_id=run_id, allow_live=allow_live)
    except Exception as e:  # noqa: BLE001 - even with no world, every case gets a row
        log.error("world_load_failed", error=repr(e), traceback=traceback.format_exc())
        decisions = [engine_default(r["case_id"], f"world load: {type(e).__name__}") for r in requests]
        write_output(decisions, out_path)
        print(f"run {run_id}: world failed to load ({e!r}); wrote {len(decisions)} NEEDS_REVIEW rows to {out_path}")
        return 1
    decisions = decide_all(requests, world, traces_dir, LinkLayer(args.max_layer))
    # Cross-row first: a file with the wrong rows is never written.
    cross_row = cross_row_check(decisions, requests, Path(args.requests), sample_mix(DATASET_DIR))
    if not cross_row["errors"]:
        write_output(decisions, out_path)

    flags = Counter(f for d in decisions for f in d.fired())
    manifest = {
        "run_id": run_id, "mode": args.mode, "commit": sha, "dirty": dirty, "requests": str(args.requests),
        "output": str(out_path) if not cross_row["errors"] else None, "rows": len(decisions),
        "live_calls_allowed": allow_live,
        "status": dict(Counter(d.status.value for d in decisions)),
        "cross_row": cross_row,
        "flags": dict(sorted(flags.items())),
        "models": {stage: model_for(stage) for stage in ("VISION", "INTENT", "AGENT")},
        # The effective mode: a key with live calls refused is a replay.
        "model_access": "replay" if model_access() == "live" and not allow_live else model_access(),
        "budget": {"max_calls": world.llm.budget.max_calls, "max_usd": world.llm.budget.max_usd,
                   "tripped": world.llm.budget.tripped} if world.llm else None,
        "usage": usage_for(run_id),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n",
                                               encoding="utf-8")
    if cross_row["errors"]:
        print(f"run {run_id}: cross-row check failed; {out_path} NOT written; traces and manifest in {run_dir}")
    else:
        print(f"run {run_id}: wrote {len(decisions)} rows to {out_path}; traces and manifest in {run_dir}")
    for w in cross_row["warnings"]:
        print(f"WARNING: {w}", file=sys.stderr)
    for e in cross_row["errors"]:
        print(f"ERROR: {e}", file=sys.stderr)
    return 1 if cross_row["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
