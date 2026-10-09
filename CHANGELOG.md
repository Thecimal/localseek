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
- Eval: held-out paraphrase set `eval/benchmark/queries-v2.json` (61 questions, one per document, no shared content
  words with their answers, fixed dev/test split by path hash) with a documented protocol for testing search changes,
  and a `--split` option. The optional `split` field is validated and shown in the run header.
- Search: `Searcher` accepts opt-in fusion options (`vector_weight`, `keyword_weight`, `keyword_limit`,
  `drop_stopwords`, in addition to the existing `rrf_k`) for evaluation experiments. The defaults are the shipped
  behaviour, which a test checks against the recorded baseline; nothing in the `localseek` command uses the options.
- Eval: `--tuning` for run_eval, `compare_runs.py`, `run_candidates.py`, and the pre-registered fusion experiment in
  `eval/benchmark/experiments.md` (candidates, decision rule and a single-use test split, fixed before any run).
- Eval: `eval/benchmark/queries-v3.json` (v2 unchanged plus a second question for every document: 119 questions, 47 dev
  and 72 test) with the same rules, an added check that no quote appears in an unlisted document, and
  `eval/make_review_sheet.py` for independent review of the relevance judgments.
- Eval: `--json` records now include every query's own outcome, and `compare_runs.py` prints paired comparisons
  (wins/losses/ties, exact sign test, bootstrap interval for the MRR difference) and `--versus-arm`. Records made
  before this change have no per-query data and must be recorded again to get paired statistics.
- Eval: `--snippets` and the `--json` record report whether the answer is in the top result's chunk and in the displayed
  snippet (`shows_answer@1`, `shows_answer@5`, `chunk_has_answer@1`) for query files with evidence quotes. Query files
  may now carry an optional, validated `evidence` field. A test checks that every benchmark quote fits inside one chunk.
- Eval: pre-registered snippet-selection experiment (`eval/benchmark/snippet-experiment.md`, `eval/snippet_experiment.py`):
  five candidate ways to choose the displayed window, a decision rule with a wide-window control, evaluated on
  (question, document) pairs given the correct chunk, with no model needed. The result is not yet recorded.
- Eval: logic moved to `eval/retrieval_eval.py` and covered by `tests/test_eval.py`.

## 0.1.0

First release.

- Index folders of `.txt`, `.md`, `.pdf`, `.docx`, `.html`, `.epub`, and common source files
- Hybrid search: embeddings plus SQLite FTS5 keyword search, merged with Reciprocal Rank Fusion
- Incremental indexing, with cached embeddings for unchanged text
- Filters by file type, modified date, and path
- `watch` mode and a local search page (`serve`)
- Evaluation script for recall@k and MRR
