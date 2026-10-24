"""Write evaluation/usage_report.md for one run, from runs/llm_calls.jsonl.

    python code/evaluation/usage.py <run_dir> --command "<command used>"

A measure run normally replays every model answer from the cache, so its own
cost is zero. The report therefore shows two views: the run itself (cache
hits and any live calls), and the cold cost — the live calls that originally
produced every answer the run replayed, i.e. what a run with an empty cache
would spend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ghostaudit.budget import PRICES  # noqa: E402
from ghostaudit.llm import CALL_LOG  # noqa: E402

STAGES = {"vision": "vision (screenshots)", "intent": "intent", "agent": "investigator agent"}


def _stage(purpose: str) -> str:
    return purpose.split(":", 1)[0]


def originating_calls(rows: list[dict], run_rows: list[dict]) -> list[dict]:
    """Live calls that produced the answers a run replayed."""
    live = [r for r in rows if not r["cached"]]
    by_key = defaultdict(list)
    for r in live:
        if r.get("cache_key"):
            by_key[r["cache_key"]].append(r)
    out = []
    for hit in (r for r in run_rows if r["cached"]):
        if hit.get("cache_key") in by_key and _stage(hit["purpose"]) != "agent":
            out.extend(by_key[hit["cache_key"]])
            continue
        # Agent turns are logged per turn without the cache key: take the
        # latest run that made live calls for the same case and prompt.
        same = [r for r in live if r["purpose"] == hit["purpose"] and r["prompt_version"] == hit["prompt_version"]]
        if same:
            last_run = same[-1].get("run_id")
            out.extend(r for r in same if r.get("run_id") == last_run)
    return out


def table(rows: list[dict], key) -> list[tuple]:
    agg = defaultdict(lambda: [0, 0, 0, 0.0])
    for r in rows:
        a = agg[key(r)]
        a[0] += 1
        a[1] += r["input_tokens"]
        a[2] += r["output_tokens"]
        a[3] += r["usd"]
    return sorted((k, *v) for k, v in agg.items())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--command", required=True)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "usage_report.md"))
    args = p.parse_args(argv)
    manifest = json.loads((Path(args.run_dir) / "run_manifest.json").read_text(encoding="utf-8"))
    run_id = manifest["run_id"]
    rows = [json.loads(line) for line in CALL_LOG.read_text(encoding="utf-8").splitlines()]
    run_rows = [r for r in rows if r.get("run_id") == run_id]
    cold = originating_calls(rows, run_rows)
    output = Path(manifest["output"])
    sha = hashlib.sha256(output.read_bytes()).hexdigest()
    n = manifest["rows"]

    md = [f"# Usage report — measure run `{run_id}`", "",
          "| | |", "|---|---|",
          f"| Run id | `{run_id}` |",
          f"| Commit | `{manifest['commit']}`{' (dirty)' if manifest['dirty'] else ' (clean tree)'} |",
          f"| Command | `{args.command}` |",
          f"| Output | `output.csv`, {n} rows, sha256 `{sha}` |",
          f"| Model access | `{manifest['model_access']}` (replay: cached answers only; a miss is flagged, never called) |",
          f"| Models | vision `{manifest['models']['VISION']}`, intent `{manifest['models']['INTENT']}`, "
          f"agent and vision tiebreak `{manifest['models']['AGENT']}` |",
          f"| Budget breaker | max {manifest['budget']['max_calls']} calls / ${manifest['budget']['max_usd']:.2f}; "
          f"tripped: {'yes' if manifest['budget']['tripped'] else 'no'} |",
          f"| Status mix | {manifest['status']} |", ""]

    md += ["## 1. This run", "",
           "| Model | Cache hits | Live calls | Input tokens | Output tokens | USD |", "|---|---|---|---|---|---|"]
    for model, u in sorted(manifest["usage"].items()):
        md.append(f"| `{model}` | {u.get('cache_hits', 0)} | {u.get('live_calls', 0)} | {u['input_tokens']:,} | "
                  f"{u['output_tokens']:,} | {u['usd']:.4f} |")
    tot_live = sum(u.get("live_calls", 0) for u in manifest["usage"].values())
    md += [f"| **Total** | {sum(u.get('cache_hits', 0) for u in manifest['usage'].values())} | {tot_live} | "
           f"{sum(u['input_tokens'] for u in manifest['usage'].values()):,} | "
           f"{sum(u['output_tokens'] for u in manifest['usage'].values()):,} | "
           f"{sum(u['usd'] for u in manifest['usage'].values()):.4f} |", ""]

    md += ["## 2. Cold cost: the live calls that produced every replayed answer", "",
           "What the same run costs with an empty cache (each call priced at the list price in "
           "`ghostaudit/budget.py`).", "",
           "| Model | Stage | Calls | Input tokens | Output tokens | USD |", "|---|---|---|---|---|---|"]
    for (model, stage), calls, tin, tout, usd in table(cold, lambda r: (r["model"], _stage(r["purpose"]))):
        md.append(f"| `{model}` | {STAGES.get(stage, stage)} | {calls} | {tin:,} | {tout:,} | {usd:.4f} |")
    md += ["", "| Model | Calls | Input tokens | Output tokens | USD | $/MTok in / out |", "|---|---|---|---|---|---|"]
    for (model,), calls, tin, tout, usd in table(cold, lambda r: (r["model"],)):
        pin, pout = PRICES.get(model, (0, 0))
        md.append(f"| `{model}` | {calls} | {tin:,} | {tout:,} | {usd:.4f} | {pin:.2f} / {pout:.2f} |")
    calls = len(cold)
    tin = sum(r["input_tokens"] for r in cold)
    tout = sum(r["output_tokens"] for r in cold)
    usd = sum(r["usd"] for r in cold)
    md += [f"| **Overall** | {calls} | {tin:,} | {tout:,} | {usd:.4f} | |", "",
           "| Per request (over all {n} requests) | Value |".replace("{n}", str(n)), "|---|---|",
           f"| Model calls | {calls / n:.3f} |",
           f"| Total tokens | {(tin + tout) / n:,.1f} (input {tin / n:,.1f}, output {tout / n:,.1f}) |",
           f"| Cost | ${usd / n:.5f} |", "",
           f"Total tokens across the run: {tin + tout:,}. Estimated total cost of a cold run: ${usd:.4f}.", "",
           "Notes: vision calls read the 11 screenshots twice each (Sonnet); intent classifies the unique "
           "request texts 25 per call (Sonnet); the agent runs only on gated cases, up to six Opus turns "
           "each. PDFs and every text format are parsed without a model.", ""]
    Path(args.out).write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {args.out}: cold cost {calls} calls, {tin + tout:,} tokens, ${usd:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
