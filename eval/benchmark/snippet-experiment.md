# Snippet-selection experiment (pre-registered)

**Status: not yet run.** Commit this file before running the experiment. At the time of writing no candidate other
than the shipped function (S0) has been run on any data, by the author or anyone else; git history is the record.

## Question

A search result shows a 280-character window cut from the chunk it came from, starting 60 characters before the first
occurrence of any query word. Which way of choosing that window shows the answer most often, when the engine has
found the right chunk?

## What is already known (baseline only)

* End to end, with the real model, on the 47 v3 dev questions (`run_eval.py --snippets`): the right file is first for
  19 (hybrid) and 27 (vector) questions, and the answer is visible in the snippet for 14 and 21. About a quarter of the
  correct top results show text without the answer.
* Given the **correct** chunk (model-independent, all questions, shipped function): v3 shows the answer in 95 of 127
  pairs, v1 in 111 of 129. Chunks of 280 characters or fewer are shown whole and never fail (36 of 36 in v3).
* In v1, queries that share words with the answer do well: exact keywords 10/10, rare terms 8/8, technical terms 10/11.
  The `late_answer` category, whose answers sit deep in long documents, shows the answer in **1 of 9**.
* In v3's longer chunks the window anchored on a function word (such as "the") in 64 of 91 pairs and on a content word in
  25. The answer was missed in 38% and 32% of those, which is not a meaningful difference. v3's questions share no
  content word with their answer *by construction*, so on v3 any word-based anchor can only land near the answer by
  luck. Whether function-word anchoring hurts questions that do share words (as in `late_answer`) is the open question.

## Disclosure: no independent confirmation data

The candidates below were designed after a baseline diagnostic over **all** questions in `queries.json` and
`queries-v3.json`, including v3's test split. There is no untouched data for this experiment. The consequences are
built into the rules: estimates of improvement will be optimistic, so the thresholds are conservative; and any adopted
change must be re-validated on newly written questions with answers deep in long documents before it is gated or
described as an improvement.

## Candidates

The list is fixed and mirrored in `snippet_experiment.py`; a test fails if they disagree. Every candidate is a pure
function of the chunk text and the query, and may use at most 420 characters (plus two ellipses): that is the most
extra screen space considered acceptable (1.5 times the current window).

| Id | Definition | Max width |
|---|---|---|
| S0 | the shipped function: 280 characters from the first occurrence of any query word | 280 |
| S1 | S0, but anchored on the first occurrence of a content word (common function words ignored) | 280 |
| S2 | 280 characters around the densest cluster of distinct content words | 280 |
| S3 | S2 with a 420-character window | 420 |
| S4 | control: S0 with a 420-character window | 420 |

S4 exists to separate "a smarter anchor" from "just show more text".

## Evaluation set

Every (question, relevant document) pair in `queries.json` and `queries-v3.json` whose correct chunk is longer than 280
characters, taking the first chunk that contains the document's evidence quote: 192 pairs at the time of writing (101
from v1, 91 from v3). Pairs on shorter chunks are excluded because they are always shown whole. `queries-v2.json` is
left out because v3 contains it.

## What this experiment cannot show

Every candidate chooses its window from words the question and the chunk share. v3's questions share no content word
with their answers by construction, so none of these candidates can systematically help on the 91 v3 pairs; they can
only improve pairs from v1, where the shipped function fails on a few dozen of the 101. The experiment is therefore
informative about questions that share words with their answer (the `late_answer` pattern) and says nothing about
paraphrases. Choosing the window by meaning, for example by embedding the sentences of a chunk, would address
paraphrases but needs an embedding model; it is deliberately not part of this experiment and would need its own
pre-registration, with candidate ids that do not reuse S0 to S4.

## Decision rule

A candidate is **eligible** if, against S0 and paired by pair:

1. it shows the answer on at least **8 more** pairs than it loses (net wins ≥ 8);
2. an exact two-sided sign test gives **p < 0.05**;
3. it loses at most **3** pairs.

**Control:** if S4 is eligible, every other candidate must also beat S4 by a net of at least **4** pairs, otherwise it is
dropped, because showing more text is the simpler change.

**Selection:** among the remaining eligible candidates, the one with the smallest maximum width; ties go to the one that
shows the answer on more pairs, then to the earlier one in the table. If none is eligible the result is negative and
the shipped snippet stays.

An accepted candidate is only a proposal. Adopting it means changing `make_snippet`, re-running
`run_eval.py --snippets` with the real model to confirm the end-to-end numbers move the same way, and checking the extra
width in the CLI and web output.

## How to run

```bash
python eval/snippet_experiment.py --out eval/benchmark/experiments/snippets.json
```

It needs no model and is deterministic. The JSON keeps every pair's outcome for each candidate so the result can be audited.

## Results

Not yet recorded.
