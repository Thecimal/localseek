# Snippet-selection experiment (pre-registered)

**Status: run; S4 was selected, as a proposal only (see Results).** Commit this file before running the experiment. At the time of writing no candidate other
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

Run with the command above (no model, deterministic) on 192 (question, document) pairs whose correct chunk is longer
than 280 characters: 101 from `queries.json` and 91 from `queries-v3.json`. The per-pair outcomes are in
`experiments/snippets.json`.

| Id | Width | All (192) | `queries.json` (101) | of which `late_answer` (9) | `queries-v3.json` (91) | Wins/losses vs S0 | p | Verdict |
|---|---|---|---|---|---|---|---|---|
| S0 | 280 | 142 | 83 | 1 | 59 | - | - | baseline |
| S1 | 280 | 140 | 87 | 3 | 53 | 7/9 | 0.804 | not eligible: net -2 |
| S2 | 280 | 145 | 90 | 5 | 55 | 13/10 | 0.678 | not eligible: net +3 |
| S3 | 420 | 178 | 98 | 9 | 80 | 40/4 | < 0.001 | not eligible: 4 losses, at most 3 allowed |
| S4 | 420 | 168 | 90 | 3 | 78 | 26/0 | < 0.001 | eligible |

**Outcome: S4 is selected, as a proposal only.** It is the only eligible candidate (net +26, p < 0.001, no losses), so
the control rule has nothing to compare it with. The shipped snippet is unchanged.

### Observations made after the decision (post-hoc; they did not influence it)

* The width did the work, not the anchor. S4 anchors exactly like S0 and gains 26 pairs without losing any, while S1 and
  S2, which change the anchor at 280 characters, are not clearly better than S0 (net -2 and +3, p 0.80 and 0.68).
* S4's gain appears in both sets: 83 to 90 of 101 on `queries.json` and 59 to 78 of 91 on `queries-v3.json`. The
  pre-registration says a different anchor cannot systematically help v3's paraphrases; a wider window can, because it
  covers more of the chunk. This is an inference, not a tested result.
* S3 (the S2 window at 420 characters) shows the answer on 178 pairs to S4's 168, and on all 9 `late_answer` pairs, but
  it loses 4 pairs to S0, one more than the rule allows. Under the rule it is not eligible and cannot be selected
  after the fact. Whether a smarter anchor adds anything on top of a wide window is a question for a new
  pre-registration with its own candidate ids.
* Even with S4, 24 of the 192 pairs still do not show the answer.

### What this does not decide

* The disclosure above stands: there is no untouched data, so the size of the improvement is likely overstated.
* Every pair is given its correct chunk. The effect on real searches, where the top result is sometimes the wrong
  chunk, is not measured here.
* S4 uses 1.5 times the screen space of the current window.
* Adopting S4 means changing `make_snippet`, re-running `run_eval.py --snippets` with the real model, validating on
  newly written questions with answers deep in long documents, and checking the CLI and web output.
