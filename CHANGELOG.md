# Changelog

## Unreleased

- **Breaking (eval dataset format):** `eval/queries.json` entries now use `"relevant": ["path/relative/to/corpus.md"]`
  (a list, one or more documents) instead of `"expected": "file-name.md"`. Matching by file name alone could count an
  unrelated file with the same name as a correct result. Old files are rejected with a message explaining the change;
  migrate by replacing `"expected": "x.md"` with `"relevant": ["<path of x.md relative to the corpus>"]`.
- Eval: recall@k is now the fraction of the relevant set found; the dataset is validated up front, and invalid input
  exits with status 2 and a readable error instead of a traceback.
- Eval: new `--thresholds FILE` quality gate. Minimum values per mode and metric are checked after the run; the
  script exits 1 and lists every failure when one is not met. Without the flag it only reports, as before. No
  threshold values ship with the repository yet.
- Eval: new benchmark in `eval/benchmark/` (62 documents in five formats, 97 queries in eleven categories, evidence
  for every relevance judgment) and a `--by-category` report. `tests/test_benchmark.py` keeps its composition and
  judgments honest without running any search. No-answer and multilingual queries are not covered yet.
- Eval: runs now print and record their configuration (model, chunk settings, library versions, SHA-256 fingerprints
  of the corpus and query file, stable across operating systems). New `--show-misses` lists queries whose first
  relevant result is not rank 1, and `--json FILE` saves the full run.
- Eval: thresholds can now set per-category floors (`by_category`) and record the benchmark version they were derived
  from (`meta`); a run against a different version is refused. `eval/derive_thresholds.py` derives a thresholds file
  from a recorded baseline by a documented rule. The benchmark ships `baselines/hash.json` and `thresholds-hash.json`.
- CI: the pull-request workflow now runs the hash-embedder retrieval gate on every operating system, and a manual
  "Model evaluation" workflow runs the real model against thresholds derived from its baseline.
- Eval: logic moved to `eval/retrieval_eval.py` and covered by `tests/test_eval.py`.

## 0.1.0

First release.

- Index folders of `.txt`, `.md`, `.pdf`, `.docx`, `.html`, `.epub`, and common source files
- Hybrid search: embeddings plus SQLite FTS5 keyword search, merged with Reciprocal Rank Fusion
- Incremental indexing, with cached embeddings for unchanged text
- Filters by file type, modified date, and path
- `watch` mode and a local search page (`serve`)
- Evaluation script for recall@k and MRR
