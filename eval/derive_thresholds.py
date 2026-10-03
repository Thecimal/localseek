"""Derive a thresholds file from a recorded baseline run, by a fixed and documented rule.

    python eval/run_eval.py --corpus C --queries Q --model M --json baseline.json
    python eval/derive_thresholds.py --baseline baseline.json --queries Q --out thresholds.json

The rule (so the numbers are auditable, not hand-picked):

* Overall, per mode: recall@1, recall@5 and mrr must stay above the baseline minus 0.02 (about two queries of 100).
* Per category, per mode: recall@5 and mrr must stay above the baseline minus 1/n, where n is the number of queries
  in that category (one query's worth). recall@1 is not gated per category because categories with several
  relevant documents cannot reach 1.0 at recall@1.
* Floors are rounded down to two decimals, and a floor of 0 is dropped because it checks nothing.

These are regression floors, not quality goals: they lock in what the baseline achieved, including weak spots.
The output records the fingerprints of the dataset and corpus it was derived from, and the evaluator refuses to
use it against a different version of the benchmark.

Exit status: 0 success, 2 invalid input.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

from retrieval_eval import dataset_fingerprint

MODES = ("hybrid", "vector", "keyword")
OVERALL_METRICS = ("recall@1", "recall@5", "mrr")
CATEGORY_METRICS = ("recall@5", "mrr")
OVERALL_SLACK = 0.02


def floor2(value: float) -> float:
    """Round down to two decimals, tolerating float noise (0.77 - 0.02 must give 0.75, not 0.74)."""
    return math.floor(round(value * 100, 6)) / 100


def derive(baseline: dict, category_sizes: dict[str, int], source: str) -> dict:
    env = baseline["environment"]
    thresholds: dict = {}
    for mode in MODES:
        thresholds[mode] = {
            metric: floor
            for metric in OVERALL_METRICS
            if (floor := floor2(baseline["results"][mode][metric] - OVERALL_SLACK)) > 0
        }
    by_category: dict = {}
    for mode in MODES:
        cells = {}
        for category, size in category_sizes.items():
            observed = baseline["by_category"][mode][category]
            floors = {
                metric: floor
                for metric in CATEGORY_METRICS
                if (floor := floor2(observed[metric] - 1 / size)) > 0
            }
            if floors:
                cells[category] = floors
        by_category[mode] = cells
    thresholds["by_category"] = by_category
    thresholds["meta"] = {
        "derived_from": source,
        "model": env["model"],
        "dataset_sha256": env["dataset"]["sha256"],
        "corpus_sha256": env["corpus"]["sha256"],
        "rule": "overall: baseline - 0.02 (recall@1, recall@5, mrr); per category: baseline - 1/n (recall@5, mrr); "
        "rounded down to 2 decimals",
    }
    return thresholds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--baseline", required=True, help="JSON written by run_eval.py --json")
    parser.add_argument("--queries", required=True, help="the query file the baseline was run on")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    try:
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))
        recorded = baseline["environment"]["dataset"]["sha256"]
        for mode in MODES:
            baseline["results"][mode]
            baseline["by_category"][mode]
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: cannot use baseline or queries: {exc!r}", file=sys.stderr)
        print("  (the baseline must come from run_eval.py --json)", file=sys.stderr)
        return 2
    actual = dataset_fingerprint(args.queries)
    if actual != recorded:
        print(f"error: baseline used queries sha256 {recorded[:12]}; {args.queries} is {actual[:12]}", file=sys.stderr)
        return 2
    sizes = Counter(q["category"] for q in queries if q.get("category"))
    if not sizes:
        print("error: the queries have no categories", file=sys.stderr)
        return 2

    result = derive(baseline, dict(sizes), Path(args.baseline).name)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    overall = sum(len(result[m]) for m in MODES)
    per_category = sum(len(c) for cats in result["by_category"].values() for c in cats.values())
    print(f"Wrote {args.out}: {overall} overall + {per_category} per-category checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
