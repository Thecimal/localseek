"""Compare saved evaluation runs against a baseline and apply the pre-registered dev rule mechanically.

    python eval/compare_runs.py baseline.json candidate1.json candidate2.json ...

Each file is the output of `run_eval.py --json`. The first file is the baseline. Runs are only compared if they used
the same model, query file, split, corpus and chunk settings; anything else exits with status 2.

The rule for hybrid search (see eval/benchmark/experiments.md): recall@1 must rise by at least two queries' worth
(0.08 on dev, 0.06 on test, chosen with --rule), MRR must rise, and recall@5 must not fall. Meeting it is
necessary, not sufficient: the candidate must also pass the v1 gates and, once, the test split.

Exit status: 0 success, 2 runs that cannot be compared or unreadable files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MIN_RECALL_GAIN = 0.08  # two queries of the 25 in the dev split
MIN_RECALL_GAIN_TEST = 0.06  # two queries of the 36 in the test split
_EPSILON = 1e-9


def comparable(base: dict, other: dict) -> list[str]:
    """Reasons two runs must not be compared; tuning is expected to differ and is not listed."""
    problems = []
    b, o = base["environment"], other["environment"]
    for label, left, right in (
        ("model", b["model"], o["model"]),
        ("query file", b["dataset"]["sha256"], o["dataset"]["sha256"]),
        ("split", b["dataset"].get("split"), o["dataset"].get("split")),
        ("corpus", b["corpus"]["sha256"], o["corpus"]["sha256"]),
        ("chunk size", (b["chunk_words"], b["overlap_words"]), (o["chunk_words"], o["overlap_words"])),
    ):
        if left != right:
            problems.append(f"{label} differs ({left} vs {right})")
    return problems


def _meets(base: dict, candidate: dict, gain: float) -> bool:
    b, c = base["results"]["hybrid"], candidate["results"]["hybrid"]
    return (
        c["recall@1"] >= b["recall@1"] + gain - _EPSILON
        and c["mrr"] > b["mrr"] + _EPSILON
        and c["recall@5"] >= b["recall@5"] - _EPSILON
    )


def meets_dev_rule(base: dict, candidate: dict) -> bool:
    return _meets(base, candidate, MIN_RECALL_GAIN)


def meets_test_rule(base: dict, candidate: dict) -> bool:
    return _meets(base, candidate, MIN_RECALL_GAIN_TEST)


def describe(run: dict) -> str:
    tuning = run["environment"].get("tuning") or {}
    return ", ".join(f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in tuning.items()) or "(defaults)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("baseline")
    parser.add_argument("candidates", nargs="+")
    parser.add_argument("--rule", choices=("dev", "test"), default="dev", help="which pre-registered rule to apply")
    args = parser.parse_args()

    runs = {}
    try:
        for name in [args.baseline, *args.candidates]:
            runs[name] = json.loads(Path(name).read_text(encoding="utf-8"))
            runs[name]["results"]["hybrid"]["recall@1"]  # noqa: B018 - fail early on files without metrics
            runs[name]["environment"]["dataset"]["sha256"]  # noqa: B018
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: cannot read a run record: {exc!r} (files must come from run_eval.py --json)", file=sys.stderr)
        return 2
    base = runs[args.baseline]
    for name in args.candidates:
        problems = comparable(base, runs[name])
        if problems:
            print(f"error: {name} cannot be compared with {args.baseline}:", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 2

    env = base["environment"]
    split = env["dataset"].get("split") or "all"
    print(f"model {env['model']}, split {split}, queries {env['dataset']['sha256'][:12]}, "
          f"corpus {env['corpus']['sha256'][:12]}\n")  # fmt: skip
    columns = [("run", 28), ("tuning", 40), ("recall@1", 10), ("recall@5", 10), ("MRR", 8), ("d recall@1", 12)]
    columns.append(("d MRR", 9))
    print("".join(f"{title:<{width}}" for title, width in columns) + f"{args.rule} rule")
    reference = base["results"]["hybrid"]
    for name in [args.baseline, *args.candidates]:
        run, h = runs[name], runs[name]["results"]["hybrid"]
        label = Path(name).stem
        if name == args.baseline:
            verdict, d1, dm = "baseline", "-", "-"
        else:
            passed = (meets_dev_rule if args.rule == "dev" else meets_test_rule)(base, run)
            verdict = "meets" if passed else "does not meet"
            d1, dm = f"{h['recall@1'] - reference['recall@1']:+.2f}", f"{h['mrr'] - reference['mrr']:+.2f}"
        print(f"{label:<28}{describe(run):<40}{h['recall@1']:<10.2f}{h['recall@5']:<10.2f}{h['mrr']:<8.2f}"
              f"{d1:<12}{dm:<9}{verdict}")  # fmt: skip
    v = base["results"]["vector"]
    print(f"\nreference, vector arm of the baseline run: recall@1 {v['recall@1']:.2f}, recall@5 {v['recall@5']:.2f}, "
          f"MRR {v['mrr']:.2f}")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
