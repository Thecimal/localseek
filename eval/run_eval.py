"""Measure retrieval quality: recall@k and MRR for vector, keyword, and hybrid search.

    python eval/run_eval.py --corpus examples/corpus --queries eval/queries.json --model BAAI/bge-small-en-v1.5

Each query lists the documents that should be found, as paths relative to the corpus folder:

    {"query": "bread fermentation", "relevant": ["recipes/sourdough.md", "recipes/fermentation.md"]}

Recall@k is the fraction of a query's relevant documents found in the top k; MRR uses the rank of the
first relevant document. Use `--model hash` for a fast, lexical-only baseline that needs no model download.

Every run starts by printing what it ran on: the model, chunk settings, library versions, and content hashes of
the corpus and the query file, so two runs can be compared. `--json FILE` writes the same record plus all metrics.

Reports: `--by-category` breaks results down by each query's optional `category` field. `--show-misses` lists, per
mode, every query whose first relevant document is not the top result.

Quality gate: pass `--thresholds FILE` to fail when a metric falls below its required minimum.

    {"hybrid": {"recall@1": 0.8, "mrr": 0.85}, "keyword": {"recall@5": 0.9}}

Metrics are recall@1, recall@5, recall@10, and mrr; modes are hybrid, vector, and keyword. Anything not listed
is reported but not checked. Without `--thresholds` the script only reports.

Exit status: 0 success (and every threshold met), 1 a threshold was not met, 2 invalid corpus, dataset, or
thresholds.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

from retrieval_eval import (
    KS,
    DatasetError,
    ThresholdError,
    check_indexed,
    check_thresholds,
    count_checks,
    environment_info,
    evaluate,
    evaluate_by_category,
    format_environment,
    format_miss,
    load_dataset,
    load_thresholds,
    misses,
    result_record,
    trace_queries,
)

from localseek.config import Settings
from localseek.embedder import get_embedder
from localseek.indexer import index_paths
from localseek.search import MODES, Searcher
from localseek.store import Store


def _fail(title: str, problems: list[str]) -> int:
    print(f"error: {title}", file=sys.stderr)
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)
    return 2


def _by_category(searcher, queries, root) -> dict:
    return {mode: evaluate_by_category(searcher, queries, root, mode, KS) for mode in MODES}


def _print_by_category(per_mode: dict, queries) -> None:
    names = list(per_mode[MODES[0]])
    width = max(len(n) for n in names) + 2
    print("\nBy category (recall@1 / recall@5 / MRR)")
    print(f"{'category':<{width}}{'n':<5}" + "".join(f"{mode:<19}" for mode in MODES))
    for name in names:
        count = sum(1 for q in queries if (q.category or "uncategorized") == name)
        cells = ""
        for mode in MODES:
            r = per_mode[mode][name]
            cells += f"{r.recall[1]:.2f} / {r.recall[5]:.2f} / {r.mrr:.2f}".ljust(19)
        print(f"{name:<{width}}{count:<5}{cells}")


def _print_misses(searcher, queries, root) -> dict:
    print("\nMisses: queries whose first relevant document is not the top result")
    recorded = {}
    for mode in MODES:
        missed = misses(trace_queries(searcher, queries, root, mode))
        print(f"\n[{mode}] {len(missed)} of {len(queries)}")
        for trace in missed:
            for line in format_miss(trace):
                print(line)
        recorded[mode] = [
            {
                "query": t.query.text,
                "category": t.query.category,
                "first_rank": t.first_rank,
                "top_hit": t.ranked[0] if t.ranked else None,
                "relevant": list(t.query.relevant),
            }
            for t in missed
        ]
    return recorded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--by-category", action="store_true", help="also report metrics per query category")
    parser.add_argument("--show-misses", action="store_true", help="list queries whose first relevant hit isn't rank 1")
    parser.add_argument("--json", metavar="FILE", help="also write the run record and all metrics to FILE")
    parser.add_argument("--thresholds", help="JSON file of minimum metrics per mode; enables the quality gate")
    args = parser.parse_args()

    root = Path(args.corpus).expanduser().resolve()
    if not root.is_dir():
        return _fail("invalid corpus", [f"{args.corpus} is not a directory"])
    try:
        queries = load_dataset(args.queries, root)
    except DatasetError as exc:
        return _fail(f"invalid evaluation dataset ({args.queries})", exc.problems)
    thresholds = None
    if args.thresholds:
        try:
            thresholds = load_thresholds(args.thresholds, MODES, KS)
        except ThresholdError as exc:
            return _fail(f"invalid thresholds ({args.thresholds})", exc.problems)

    embedder = get_embedder(args.model)
    settings = Settings(model=args.model)
    info = environment_info(args.model, settings, args.queries, args.corpus, root, queries)
    print("\n".join(format_environment(info)) + "\n")

    record: dict = {"environment": info}
    with tempfile.TemporaryDirectory() as tmp:
        store = Store(Path(tmp) / "eval.db")
        try:
            started = time.perf_counter()
            stats = index_paths(store, embedder, [root], settings)
            index_seconds = time.perf_counter() - started
            try:
                check_indexed(store.file_records(), root, queries)
            except DatasetError as exc:
                return _fail("invalid evaluation dataset", exc.problems)
            print(f"Indexed {stats.scanned} files ({stats.chunks} chunks) in {index_seconds:.1f}s\n")

            searcher = Searcher(store, embedder)
            print(f"{'mode':<9}" + "".join(f"recall@{k:<4}" for k in KS) + "MRR    latency")
            results = {}
            for mode in MODES:
                result = results[mode] = evaluate(searcher, queries, root, mode, KS)
                row = f"{mode:<9}" + "".join(f"{result.recall[k]:<11.2f}" for k in KS)
                print(f"{row}{result.mrr:<7.2f}{result.latency_ms:.1f} ms/query")
            record["indexed"] = {"files": stats.scanned, "chunks": stats.chunks}
            record["results"] = {mode: result_record(r) for mode, r in results.items()}
            if args.by_category or args.json:
                per_mode = _by_category(searcher, queries, root)
                record["by_category"] = {
                    mode: {name: result_record(r) for name, r in groups.items()} for mode, groups in per_mode.items()
                }
                if args.by_category:
                    _print_by_category(per_mode, queries)
            if args.show_misses or args.json:
                recorded = _print_misses(searcher, queries, root) if args.show_misses else None
                if args.json:
                    record["misses"] = recorded or {
                        mode: len(misses(trace_queries(searcher, queries, root, mode))) for mode in MODES
                    }
        finally:
            store.close()

    failures = check_thresholds(results, thresholds) if thresholds is not None else []
    if thresholds is not None:
        record["gate"] = {"checks": count_checks(thresholds), "failures": [str(f) for f in failures]}
    if args.json:
        Path(args.json).write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nWrote {args.json}")

    if thresholds is None:
        print("\nNo --thresholds given: report only, nothing was checked.")
        return 0
    if failures:
        print("\nEvaluation failed:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print(f"\nQuality gate passed ({count_checks(thresholds)} checks).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
