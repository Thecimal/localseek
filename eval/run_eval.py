"""Measure retrieval quality: recall@k and MRR for vector, keyword, and hybrid search.

    python eval/run_eval.py --corpus examples/corpus --queries eval/queries.json --model BAAI/bge-small-en-v1.5

Each query lists the file name that should be found. Use `--model hash` for a fast, lexical-only
baseline that needs no model download.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from localseek.config import Settings
from localseek.embedder import get_embedder
from localseek.indexer import index_paths
from localseek.search import MODES, Searcher
from localseek.store import Store

KS = (1, 5, 10)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    args = parser.parse_args()

    queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))
    embedder = get_embedder(args.model)
    settings = Settings(model=args.model)

    with tempfile.TemporaryDirectory() as tmp:
        store = Store(Path(tmp) / "eval.db")
        started = time.perf_counter()
        stats = index_paths(store, embedder, [args.corpus], settings)
        index_seconds = time.perf_counter() - started
        print(f"Indexed {stats.scanned} files ({stats.chunks} chunks) in {index_seconds:.1f}s\n")

        searcher = Searcher(store, embedder)
        header = f"{'mode':<9}" + "".join(f"recall@{k:<4}" for k in KS) + "MRR    latency"
        print(header)
        for mode in MODES:
            found = {k: 0 for k in KS}
            reciprocal = 0.0
            started = time.perf_counter()
            for item in queries:
                hits = searcher.search(item["query"], limit=max(KS), mode=mode)
                names = [Path(h.path).name for h in hits]
                rank = names.index(item["expected"]) + 1 if item["expected"] in names else None
                if rank:
                    reciprocal += 1 / rank
                    for k in KS:
                        found[k] += rank <= k
            latency = (time.perf_counter() - started) / len(queries) * 1000
            row = f"{mode:<9}" + "".join(f"{found[k] / len(queries):<11.2f}" for k in KS)
            print(f"{row}{reciprocal / len(queries):<7.2f}{latency:.1f} ms/query")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
