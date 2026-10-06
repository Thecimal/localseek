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
import random
import sys
from math import comb
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


# -- paired statistics ---------------------------------------------------------------------------------------------

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0  # fixed, so the same two files always print the same interval


def sign_test_p(wins: int, losses: int) -> float:
    """Exact two-sided sign test: the chance of a split at least this lopsided if wins and losses were equally likely.

    Ties carry no information and are left out by the caller. With no wins and no losses the answer is 1.
    """
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(wins, losses) + 1))
    return min(1.0, 2 * tail / 2**n)


def bootstrap_ci(diffs: list[float], level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of paired differences."""
    if not diffs:
        raise ValueError("no differences to resample")
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(diffs)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(BOOTSTRAP_RESAMPLES))
    low = means[int((1 - level) / 2 * BOOTSTRAP_RESAMPLES)]
    high = means[int((1 + level) / 2 * BOOTSTRAP_RESAMPLES) - 1]
    return low, high


def paired_values(run_x: dict, mode_x: str, run_y: dict, mode_y: str, metric: str) -> list[tuple[float, float]]:
    """Each query's value for one run and mode against another, matched by question text."""
    try:
        x = {row["query"]: row[metric] for row in run_x["per_query"][mode_x]}
        y = {row["query"]: row[metric] for row in run_y["per_query"][mode_y]}
    except KeyError as exc:
        raise ValueError(f"run record has no per-query data ({exc}); record it again with run_eval.py --json") from exc
    if x.keys() != y.keys():
        raise ValueError("the two runs did not answer the same questions")
    return [(x[q], y[q]) for q in x]


def summarize_pairs(pairs: list[tuple[float, float]]) -> dict:
    """Wins, losses and ties of y over x, the exact sign-test p-value, and the mean difference with its interval."""
    diffs = [y - x for x, y in pairs]
    wins = sum(d > _EPSILON for d in diffs)
    losses = sum(d < -_EPSILON for d in diffs)
    low, high = bootstrap_ci(diffs)
    return {
        "n": len(diffs),
        "wins": wins,
        "losses": losses,
        "ties": len(diffs) - wins - losses,
        "p": sign_test_p(wins, losses),
        "mean": sum(diffs) / len(diffs),
        "low": low,
        "high": high,
    }


def _wlt(summary: dict) -> str:
    return f"{summary['wins']}/{summary['losses']}/{summary['ties']}"


def paired_rows(title: str, rows: list[tuple[str, dict]]) -> list[str]:
    """rows: (label, {"recall@1": summary, "recall@5": summary, "rr": summary})."""
    head = f"{'run':<28}{'recall@1 w/l/t':<17}{'p':<8}{'recall@5 w/l/t':<17}{'p':<8}MRR difference [95% CI]"
    lines = [title, head]
    for label, summaries in rows:
        r1, r5, rr = summaries["recall@1"], summaries["recall@5"], summaries["rr"]
        lines.append(
            f"{label:<28}{_wlt(r1):<17}{r1['p']:<8.3f}{_wlt(r5):<17}{r5['p']:<8.3f}"
            f"{rr['mean']:+.3f} [{rr['low']:+.3f}, {rr['high']:+.3f}]"
        )
    return lines


def has_per_query(run: dict) -> bool:
    return all(mode in run.get("per_query", {}) for mode in ("hybrid", "vector"))


def describe(run: dict) -> str:
    tuning = run["environment"].get("tuning") or {}
    return ", ".join(f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in tuning.items()) or "(defaults)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("baseline")
    parser.add_argument("candidates", nargs="+")
    parser.add_argument("--rule", choices=("dev", "test"), default="dev", help="which pre-registered rule to apply")
    parser.add_argument("--versus-arm", choices=("vector", "keyword"),
                        help="also compare hybrid with this arm inside each run (paired, per question)")
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
    return print_pairs(args, runs)


def print_pairs(args: argparse.Namespace, runs: dict) -> int:
    names = [args.baseline, *args.candidates]
    lacking = [Path(n).name for n in names if not has_per_query(runs[n])]
    if lacking:
        print("\nPaired statistics skipped: no per-query data in " + ", ".join(lacking)
              + " (record again with run_eval.py --json).")
        return 0
    base = runs[args.baseline]
    metrics = ("recall@1", "recall@5", "rr")
    try:
        if args.candidates:
            rows = []
            for n in args.candidates:
                summaries = {m: summarize_pairs(paired_values(base, "hybrid", runs[n], "hybrid", m)) for m in metrics}
                rows.append((Path(n).stem, summaries))
            size = rows[0][1]["recall@1"]["n"]
            print()
            print("\n".join(paired_rows(f"Paired comparison with the baseline: hybrid, {size} questions "
                                         "(w/l/t = candidate better / worse / same)", rows)))  # fmt: skip
        if args.versus_arm:
            rows = []
            for n in names:
                arm = args.versus_arm
                summaries = {m: summarize_pairs(paired_values(runs[n], "hybrid", runs[n], arm, m)) for m in metrics}
                rows.append((Path(n).stem, summaries))
            print()
            print("\n".join(paired_rows(f"The {args.versus_arm} arm against hybrid inside each run "
                                         f"(w/l/t = {args.versus_arm} better / worse / same)", rows)))  # fmt: skip
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
