# Fusion experiment (pre-registered)

**Status: not yet run.** Commit this file before running any candidate. Git history is the record that the candidates,
the rules and the order of operations were fixed first; anything changed afterwards is visible there.

## Question

Hybrid search trails vector-only search on paraphrased questions, and we want to know whether a small, principled
change to how the two arms are combined can close the gap without losing hybrid's strength on exact terms, error
codes, near-duplicates and shared file names.

## What is already known

* Benchmark v1 with `bge-small-en-v1.5`, `semantic_paraphrase` (12 queries): hybrid recall@1 0.50, vector 0.75.
* Held-out dev split (25 queries), recorded in `baselines/bge-v2-dev.json`: hybrid recall@1 0.40, MRR 0.54; vector
  recall@1 0.60, MRR 0.73.
* On the dev split the two arms miss largely the same questions (nine are missed by both), so even choosing the better
  arm per query would only reach about 0.64 recall@1. **The realistic best case for a change is therefore parity with
  vector search on paraphrases, roughly +5 dev queries.** The point is to get there without giving up the exact-match
  categories, which the v1 gates protect.
* The keyword arm does no stopword filtering and joins every query term with OR, so for a natural-language question it
  matches most of the index, and Reciprocal Rank Fusion (k = 60) rewards a document that appears anywhere in both lists
  over one that is first in only one. This is the working hypothesis behind the candidates; the experiment is what tests it.

## Candidates

The list is fixed. It is mirrored in `run_candidates.py`, and a test fails if the two disagree. Candidates may not be
added once any of them has been run; new ideas go in a new pre-registration, and need new held-out data because the
test split is single-use.

| Id | Setting | Reasoning |
|---|---|---|
| C0 | (defaults) | the baseline |
| C1 | `rrf_k=20` | a smaller constant makes each arm's top ranks count for more |
| C2 | `rrf_k=10` | the same, more strongly |
| C3 | `vector_weight=2` | trust the semantic arm more |
| C4 | `keyword_limit=20` | only the first 20 keyword hits take part in fusion, so a document ranked low by BM25 cannot ride along |
| C5 | `keyword_limit=10` | the same, more strongly |
| C6 | `drop_stopwords=true` | stop function words from matching most of the index |
| C7 | `drop_stopwords=true,keyword_limit=20` | the two keyword-arm fixes together |

## Decision rule

A candidate is **eligible** only if all of the following hold on the dev split:

1. Hybrid recall@1 is at least **0.08** above the baseline (two queries), hybrid MRR is higher, and recall@5 is not lower.
2. It passes **both v1 gates** while running with the candidate's setting: the hash thresholds
   (`thresholds-hash.json`) and, for the real model, `thresholds-bge-small-en-v1.5.json`.

**Selection:** among eligible candidates, the one with the highest dev hybrid MRR; ties go to the candidate with fewer
changed options, then to the earlier one in the table. At most one candidate is selected.

**Confirmation:** the selected candidate and the baseline are run **once** on the test split. It is accepted only if
hybrid recall@1 is at least **0.06** above the baseline (two of the 36 test queries), MRR is higher, and recall@5 is not
lower. A candidate that fails confirmation is rejected, and is not adjusted and re-run.

**If no candidate is eligible, or the selected one is rejected,** the result is recorded as negative and the shipped
defaults stay as they are.

An accepted candidate is only a proposal for a product change. It still needs review, for example the stopword list is
English-only and the alternative multilingual model would need its own handling.

## How to run

```bash
python eval/run_candidates.py dev --model BAAI/bge-small-en-v1.5 --out eval/benchmark/experiments
python eval/run_candidates.py confirm <selected id> --model BAAI/bge-small-en-v1.5 --out eval/benchmark/experiments
```

`dev` writes `dev-<id>.json` and `summary.json` for every candidate and prints which one, if any, is selected.
`confirm` refuses any candidate the dev step did not select, and refuses a second test-split run unless
`--allow-rerun` is passed (which should only be used to recover from a crash, and noted in the PR).
Commit the contents of `experiments/` with the result.

## Results

Not yet recorded.
