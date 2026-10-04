# Fusion experiment (pre-registered)

**Status: dev step run; no candidate qualified (see Results). The test split has not been run.** This file was committed before any candidate was run. Git history is the record that the candidates,
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

Dev step, `BAAI/bge-small-en-v1.5` (fastembed 0.8.1), dev split of `queries-v2.json` (25 queries, sha256 `e156b98f38fb`),
corpus `39c2e84b7a80`. The baseline row reproduces the earlier recorded dev run exactly. Records are in `experiments/`.

| Id | Setting | recall@1 | recall@5 | MRR | v1 gates (real model, hash) | Dev rule |
|---|---|---|---|---|---|---|
| C0 | (defaults) | 0.40 | 0.80 | 0.54 | pass, pass | baseline |
| C1 | `rrf_k=20` | 0.44 | 0.80 | 0.57 | pass, pass | not met |
| C2 | `rrf_k=10` | 0.44 | 0.80 | 0.59 | pass, pass | not met |
| C3 | `vector_weight=2` | 0.44 | 0.80 | 0.58 | **FAIL, FAIL** | not met |
| C4 | `keyword_limit=20` | 0.40 | 0.76 | 0.53 | pass, pass | not met |
| C5 | `keyword_limit=10` | 0.40 | 0.64 | 0.50 | pass, pass | not met |
| C6 | `drop_stopwords=true` | 0.44 | 0.84 | 0.59 | pass, pass | not met |
| C7 | `drop_stopwords=true,keyword_limit=20` | 0.44 | 0.84 | 0.59 | pass, pass | not met |

For reference, the vector arm of the baseline run scored recall@1 0.60, recall@5 0.92, MRR 0.73.

**Outcome: negative.** No candidate reached the required +0.08 recall@1, so none was eligible and none was selected.
The confirmation step was not run, the test split has never been used, and the shipped defaults are unchanged.

### What the result does and does not show

* The five best candidates (C1, C2, C3, C6, C7) each gained one dev query (+0.04 recall@1) and 0.02 to 0.05 MRR,
  against a gap to the vector arm of 0.20 and 0.19. That is about a fifth of the gap, in a direction consistent across
  candidates, and far from the parity that was the realistic best case.
* With 25 queries a one-query change is within sampling noise. The result means "no clear improvement on this set",
  not "no improvement exists".
* C3 failed both v1 gates: weighting the vector arm more heavily gives up exact-match behaviour, which is what the
  gates protect.

### Observations made after the decision (post-hoc; they did not influence it)

* Capping the keyword arm hurt (C4, C5: recall@5 0.76 and 0.64), so its deeper hits carry useful signal. The
  "too many keyword hits ride along" part of the hypothesis was not supported.
* By default the keyword arm returns a median of 75 of the 80 chunks for a dev query (at least 21 for every query), so
  it is effectively a whole-corpus ranking. With stopwords dropped the median is 11 and only 3 of 25 queries exceed 20
  hits, which is why C7 equals C6. This was measured with the keyword arm alone, which does not depend on the model.
* Plausible mechanism (an inference from arithmetic, not a tested result): at `rrf_k=60` the fused scores of the vector
  arm's top ten ranks differ by only 0.0021 in total (1/61 - 1/70), while a document's position anywhere in a keyword
  list that spans the whole corpus changes its score by up to 0.009 (1/61 - 1/135). A broad keyword arm therefore
  dominates the ordering. Dropping stopwords and a smaller `rrf_k` both reduce that and both helped slightly, but
  keyword matches on ordinary content words still mislead, which rank-based fusion cannot tell apart from
  informative ones.

### What this does not decide

Ideas that use the strength of the evidence rather than only ranks, for example trusting the keyword arm only for rare
or identifier-like terms, or fusing normalised scores, are not candidates here and were not run. They need their own
pre-registration. The test split remains unused and can serve as the single confirmation for such an experiment; a
larger dev set would make small effects measurable.
