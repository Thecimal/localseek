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

## Held-out paraphrase set (`queries-v2.json`)

Benchmark v1 has been studied in detail, so changes to search should not be tuned against it. `queries-v2.json` is a
second query set over the **same corpus** (so the v1 fingerprints and thresholds stay valid), built to measure the
README's headline claim, finding a file when the question shares no words with it, on queries nobody has tuned for.

* **One question for every document** (61 questions, 64 relevance judgments), so the hard ones cannot be skipped.
  A document that is a copy or a variant of another is listed as relevant to whichever questions it answers.
* **Paraphrase rule, enforced by a test:** a question may not share a content word (three or more letters, not a
  stopword; numbers are exempt, so a question can still name a year) with the quote that answers it.
* **Evidence for every judgment**, exactly as in v1: each relevant document has a quote that must appear in it.
* **Fixed dev/test split.** `split` is `dev` or `test` by the parity of the SHA-256 of the first relevant path, so
  nobody picks which questions are held out. It gives 25 dev and 36 test questions. Use `--split dev` while
  working and `--split test` **once**, at the end. Twenty-five questions is thin: treat dev results as a sanity
  check, not a precise measurement.
* **Not a gate.** There are no thresholds for it; it exists to compare a change against the baseline.

```bash
python eval/run_eval.py --corpus eval/benchmark/corpus --queries eval/benchmark/queries-v2.json \
    --split dev --show-misses --json dev-run.json
```

### Larger set for new work (`queries-v3.json`)

With only 25 dev questions, a change of one or two queries cannot be told from noise. `queries-v3.json` contains all
61 questions of v2, **unchanged and first** (a test checks this), followed by 58 new ones: a second question for every
document, about a **different fact** than its v2 question. It follows exactly the same rules as v2: no shared content
word between a question and its quote, a quote for every judgment, and the same hash-based split. It gives 47 dev and
72 test questions. Every v3 test check also runs on v2.

* **Which file for what.** `queries-v2.json` stays frozen because the recorded baseline and the fusion experiment
  point at its fingerprint (`e156b98f38fb`); use it only to reproduce them. New work should use `queries-v3.json`.
* **The test split is still unused.** Neither v2's test questions nor the 47 new ones have been run, and the
  single-use rule applies to the whole v3 test split. The 25 dev questions inherited from v2 were used to choose among
  candidates in the first experiment; the 22 added dev questions are fresh.
* **Evidence completeness.** For both sets a test checks that no quote appears in a document that is not listed as
  relevant, so a relevant set cannot silently miss a duplicate.
* **Independent review.** `make_review_sheet.py` writes a sheet listing each question with its judged documents and
  quotes and two checkboxes, so someone other than the author can check the judgments without running any search.
  Corrections that come out of a review change the question file, and therefore its fingerprint, so make them before
  recording new baselines.

```bash
python eval/make_review_sheet.py --queries eval/benchmark/queries-v3.json --out review-sheet.md
python eval/run_eval.py --corpus eval/benchmark/corpus --queries eval/benchmark/queries-v3.json \
    --split dev --show-misses --json eval/benchmark/baselines/bge-v3-dev.json
```

### What the user is shown (`--snippets`)

Finding the right file is not the same as showing the answer. A result is the best-ranked chunk of a file, and its
snippet is a 280-character window cut from that chunk, anchored on the first query word found in it. When the question
shares no word with the chunk (the paraphrase case) the window simply starts at the beginning of the chunk, so the
answer can be in the chunk and still not on screen. For query files with evidence quotes, `run_eval.py --snippets`
(and every `--json` record) reports, per mode:

* **right file @1**: the top result is a relevant document;
* **answer in chunk @1**: the top result is a relevant document and the chunk it came from contains that document's
  quote (the engine found the right passage, whatever it displays);
* **answer in snippet @1** and **in a top-5 snippet**: the displayed text of a relevant document contains its quote.

The display loses answers that retrieval found: with the real model on the v3 dev questions about a quarter of the
correct top results show text without the answer. `snippet-experiment.md` pre-registers a model-free experiment on how
the window is chosen (`snippet_experiment.py`); read its disclosure that no independent confirmation data exists.

A few cautions. The match is an exact, case- and whitespace-insensitive substring of the quote, so it measures "the
quoted fact is visible", not "a reader could find the answer". The quotes are short fragments chosen by the
benchmark's author, and a snippet can answer the question in other words and still count as a miss. A test checks that
every quote lies inside a single chunk, so no quote is unreachable. The metric is new and has no thresholds yet;
record a baseline first.

```bash
python eval/run_eval.py --corpus eval/benchmark/corpus --queries eval/benchmark/queries-v3.json \
    --split dev --snippets
```

### Tools for experiments

* `run_eval.py --tuning rrf_k=10,keyword_limit=20,...` runs hybrid search with experimental fusion settings
  (`rrf_k`, `vector_weight`, `keyword_weight`, `keyword_limit`, `drop_stopwords`). The shipped defaults are unchanged,
  a test checks that default search still reproduces the recorded hash baseline exactly, and the settings are printed
  in the run header and saved by `--json`.
* `compare_runs.py baseline.json candidate.json ...` tabulates runs and applies the numeric rule below. It refuses to
  compare runs made with a different model, query file, split, corpus or chunk size.
  When the records carry per-query outcomes (every `--json` run now does; older records need re-recording) it also
  prints a **paired comparison**: per question, how many the candidate answers better, worse or the same, an exact
  sign test, and a bootstrap 95% interval for the MRR difference (seeded, so the same files always print the same
  numbers). `--versus-arm vector` makes the same comparison between hybrid and one arm inside each run. Averages
  alone cannot say whether a difference is real: with no losses it takes at least six wins to reach p < 0.05, and a
  3/0 split gives p = 0.25.
* `run_candidates.py` runs the whole pre-registered experiment in `experiments.md`: every candidate on the dev split
  and on both v1 gates, then a single confirmation run on the test split for the one candidate it selects.

### Protocol for a change to search (fixed before the first experiment)

A change to retrieval or fusion is accepted only if all three hold:

1. It improves `recall@1` and `MRR` for hybrid on the **dev** split compared with the recorded baseline.
2. The v1 gates still pass: the hash thresholds in CI, and `thresholds-bge-small-en-v1.5.json` with the real model.
   This is what protects the keyword, identifier and near-duplicate queries from being traded away.
3. Run once on the **test** split, it shows the same direction of improvement. A change that helps dev but not test
   is overfit and is rejected; do not tweak and re-run the test split.

Candidates are listed before they are run, so that failing ones stay on the record. Record each variant's dev result
with `--json` and keep the files.

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
