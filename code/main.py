"""Run the deletion audit.

    python main.py                                    # all evaluation cases -> ../output.csv
    python main.py --requests ../dataset/sample_deletion_requests.csv --out runs/tune/sample_output.csv

Intent classification is not wired in yet: every request is treated as a
deletion request and flagged INTENT_NOT_CLASSIFIED.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ghostaudit import CODE_ROOT, DATASET_DIR, REPO_ROOT  # noqa: E402
from ghostaudit.engine.decide import World, decide_case  # noqa: E402
from ghostaudit.schemas import OUTPUT_HEADER, LinkLayer, case_sort_key  # noqa: E402


def run(requests_path: Path, out_path: Path, traces_dir: Path, world: World,
        max_layer: LinkLayer = LinkLayer.PROXIMITY) -> list:
    with open(requests_path, newline="", encoding="utf-8") as f:
        requests = sorted(csv.DictReader(f), key=lambda r: case_sort_key(r["case_id"]))
    traces_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    decisions = []
    for req in requests:
        decision, trace = decide_case(world, req, max_layer)
        decision = decision.model_copy(update={"flags": decision.flags + ("INTENT_NOT_CLASSIFIED",)})
        trace["flags"] = list(decision.flags)
        decisions.append(decision)
        (traces_dir / f"{req['case_id']}.json").write_text(
            json.dumps(trace, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(OUTPUT_HEADER)
        for d in decisions:
            row = d.to_row()
            w.writerow([row[k] for k in OUTPUT_HEADER])
    return decisions


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--requests", default=str(DATASET_DIR / "deletion_requests.csv"))
    p.add_argument("--out", default=str(REPO_ROOT / "output.csv"))
    p.add_argument("--traces", default=str(CODE_ROOT / "traces"))
    p.add_argument("--max-layer", type=int, default=int(LinkLayer.PROXIMITY), help="highest link layer to use (1-5)")
    args = p.parse_args(argv)
    world = World.load()
    decisions = run(Path(args.requests), Path(args.out), Path(args.traces), world, LinkLayer(args.max_layer))
    print(f"wrote {len(decisions)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
