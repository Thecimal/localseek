# LocalSeek retrieval benchmark (v1)

A fixed, synthetic corpus of personal documents plus queries with auditable relevance judgments. It exists to answer
one question: **does LocalSeek find what its README promises it will find?** It is independent of anyone's real
documents and contains no private data.

```bash
python eval/run_eval.py --corpus eval/benchmark/corpus --queries eval/benchmark/queries.json \
    --model BAAI/bge-small-en-v1.5 --by-category
```

## What it tests, and why

| README promise | Query categories |
|---|---|
| Finds the right file "even if it never uses those words" | `semantic_paraphrase`, `long_question` |
| Exact phrases and error codes work | `exact_keyword`, `technical_term`, `rare_term` |
| Works on a real, messy folder | `shared_filename`, `near_duplicate`, `similar_subject`, `redundant_answer`, `ambiguous` |
| Long documents are searchable by their best chunk | `late_answer` (answer is never in the first two chunks) |
| Information is spread across files | `multi_document` |
| PDF, DOCX, HTML, text, Markdown | the corpus itself (2 PDF, 2 DOCX, 2 HTML, 10 TXT, 46 MD) |

Composition: 62 documents (19 to 644 words, median 59; 80 chunks at the default chunk size; 7 documents have three or
more chunks), 97 queries, 129 relevance judgments. Five file names are shared across folders (`notes.md` x5,
`checklist.md` x4, `README.md` x3, `meeting-notes.md` x3, `summary.txt` x2). It includes an exact duplicate, a stale
copy of a newer document, templated year-over-year pairs, and revised addenda that differ in one figure.

| Category | Queries | Best possible recall@1 |
|---|---|---|
| exact_keyword | 10 | 1.00 |
| technical_term | 11 | 1.00 |
| rare_term | 8 | 1.00 |
| semantic_paraphrase | 12 | 1.00 |
| long_question | 8 | 0.94 |
| near_duplicate | 6 | 1.00 |
| similar_subject | 5 | 1.00 |
| shared_filename | 8 | 1.00 |
| late_answer | 9 | 1.00 |
| multi_document | 8 | 0.40 |
| redundant_answer | 6 | 0.47 |
| ambiguous | 6 | 0.39 |

Recall@k is the fraction of the relevant set found, so categories with several relevant documents cannot reach 1.0 at
recall@1. In `redundant_answer` any one document suffices, so MRR is the fairer number there.

## How relevance was judged

* A document is relevant if it contains a substantive answer to the question as asked. Incidental mentions do not count.
* **Every judgment carries evidence**: `queries.json` has, per query, `evidence` mapping each relevant path to an exact
  phrase from that document, and `relevant` is exactly the keys of that map. `tests/test_benchmark.py` checks that
  each phrase really occurs in the document as the indexer extracts it.
* For `exact_keyword`, `technical_term` and `rare_term`, the phrase must occur in **no other** document, which makes
  the relevant set provably complete.
* `late_answer` evidence must lie outside the first two chunks, measured with the real chunker.
* Outdated information is **not** relevant: `archive/tech/router-setup-old.md` has the same guest network name as the
  current one but a different password, and is a deliberate distractor.
* Ambiguous queries list every document that substantively answers under any reasonable reading.

`tests/test_benchmark.py` is the executable specification: minimum corpus size, formats, shared names, long documents,
near-duplicates (by text similarity), and a minimum number of queries per category. It runs no searches, so it cannot
be satisfied by tuning the data to an algorithm.

## Recording and comparing runs

Every run prints its configuration first: model, chunk settings, library versions, and a content fingerprint of the
corpus and of the query file. Fingerprints are SHA-256 over file contents with text line endings normalised, so the
same benchmark hashes the same on Linux, macOS and Windows, and any edit to a document or query changes them. Two runs
are comparable only if their fingerprints match.

Fingerprints of benchmark v1: corpus `39c2e84b7a80`, queries `8a99ed3d727d` (first 12 hex digits).

```bash
python eval/run_eval.py --corpus eval/benchmark/corpus --queries eval/benchmark/queries.json \
    --by-category --show-misses --json run.json
```

`--show-misses` lists, for each mode, the queries whose first relevant document is not the top result, with the rank
it reached and what ranked first. `--json` writes the same configuration plus every metric (overall, per category)
for archiving or diffing; it records the corpus path as given on the command line, not an absolute path.

## Quality gate and thresholds

`thresholds-hash.json` is the gate CI runs on every pull request (Linux, macOS and Windows). It uses the hash
embedder, so it protects the keyword search, the fusion and the indexing from regressions, but it says nothing about
semantic quality. The semantic check is the manual "Model evaluation" workflow, which needs a thresholds file derived
from a real-model baseline:

```bash
python eval/derive_thresholds.py --baseline eval/benchmark/baselines/bge-small-en-v1.5.json \
    --queries eval/benchmark/queries.json --out eval/benchmark/thresholds-bge-small-en-v1.5.json
```

Thresholds are **derived by a fixed rule from a recorded baseline**, not chosen by hand:

* Overall, per mode: `recall@1`, `recall@5` and `mrr` must stay above the baseline minus 0.02 (about two queries).
* Per category, per mode: `recall@5` and `mrr` must stay above the baseline minus 1/n, one query's worth for that
  category. `recall@1` is not gated per category, because categories with several relevant documents cannot reach 1.0.
* Floors are rounded down to two decimals; a floor of zero checks nothing and is dropped.

They are **regression floors, not quality goals**: they lock in what the baseline achieved, weak spots included, so
improving search never fails the gate but making it worse does. The file records the fingerprints of the benchmark it
was derived from, and a run against a different corpus or query file is refused until the thresholds are re-derived.
Changing the benchmark therefore means: record a new baseline with `--json`, derive the thresholds, commit both.

The hash gate was checked against a deliberately broken ranking (keyword results returned in reverse order): the clean
run passes all its checks and the broken one fails with every affected row listed, while the untouched vector mode is
correctly not reported.

## Rules for changing it

1. Write documents and queries **before** looking at any retrieval results.
2. Never delete or reword a query because the system fails it. Fix a judgment only when the evidence shows it is
   wrong, and say so in the commit message.
3. Add a new category to the spec when you add one. Removing a category is a product decision.
4. Changing the corpus or queries creates a new benchmark version; metrics are comparable only within a version.

## Not covered yet

* **No-answer queries.** `Searcher` returns a relative score (normalised to the best possible fusion score) and vector
  search ranks every chunk, so a query with no answer still returns a confident-looking list. Measuring this needs an
  absolute relevance signal or threshold in the search API first; that is a product change, not an evaluation change.
* **Multilingual queries.** The default model is English-only; multilingual retrieval is an optional alternative model.
  If it becomes a product claim, add a separate multilingual corpus and run it against that model, not this one.
* **Scale.** 62 documents is a floor, not a realistic folder. Recall@10 over 62 documents is more forgiving than over
  1,000+. Grow the corpus, with new judgments written under the rules above, before treating high-end scores as proof.
* **Authorship.** One author wrote the corpus and the judgments. Independent review of the judgments is worth doing.

## Baseline (hash embedder, informational only)

`--model hash` is a deterministic lexical embedder that needs no download. It is **not** the semantic model, so these
numbers say nothing about semantic quality and are **not thresholds**. They are a reference for regression comparison
of the lexical machinery, and show that the benchmark discriminates (the old toy benchmark scored 1.00 at recall@5).

| mode | recall@1 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|
| hybrid | 0.63 | 0.84 | 0.88 | 0.79 |
| vector | 0.38 | 0.66 | 0.75 | 0.55 |
| keyword | 0.70 | 0.93 | 0.97 | 0.86 |

Per category, recall@1 / MRR for hybrid:

| Category | hybrid | keyword | vector |
|---|---|---|---|
| exact_keyword | 1.00 / 1.00 | 1.00 / 1.00 | 0.60 / 0.73 |
| technical_term | 1.00 / 1.00 | 1.00 / 1.00 | 0.82 / 0.86 |
| rare_term | 0.88 / 0.94 | 1.00 / 1.00 | 0.50 / 0.66 |
| semantic_paraphrase | 0.17 / 0.26 | 0.17 / 0.32 | 0.08 / 0.13 |
| long_question | 0.62 / 0.67 | 0.69 / 0.84 | 0.25 / 0.33 |
| near_duplicate | 0.67 / 0.79 | 0.83 / 0.92 | 0.33 / 0.48 |
| similar_subject | 0.60 / 0.70 | 0.80 / 0.90 | 0.60 / 0.62 |
| shared_filename | 1.00 / 1.00 | 1.00 / 1.00 | 0.62 / 0.81 |
| late_answer | 0.44 / 0.64 | 0.78 / 0.84 | 0.22 / 0.40 |
| multi_document | 0.36 / 0.94 | 0.27 / 0.85 | 0.12 / 0.49 |
| redundant_answer | 0.25 / 0.66 | 0.42 / 0.88 | 0.08 / 0.38 |
| ambiguous | 0.39 / 1.00 | 0.39 / 1.00 | 0.31 / 0.88 |

What can and cannot be concluded:

* Paraphrase retrieval is **unmeasured**: all modes score near zero because the hash embedder has no semantics. This is
  the README's headline claim, and it needs a run with the real model before any threshold is set.
* Keyword beating hybrid here reflects a weak, noisy vector arm, not a fact about the real model.
* Independent of the embedder, three weaknesses showed up and are worth checking with the real model: long natural-
  language questions pull in long, generic documents; a query naming one year can rank the previous year's near-copy
  first; and a query for the "original" document can rank a later addendum above it.
* A review of the misses outside the semantic categories found no wrong judgments.
