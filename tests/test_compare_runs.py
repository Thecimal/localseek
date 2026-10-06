import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))

from compare_runs import (  # noqa: E402
    MIN_RECALL_GAIN,
    bootstrap_ci,
    comparable,
    describe,
    meets_dev_rule,
    paired_values,
    sign_test_p,
    summarize_pairs,
)


def run(recall1=0.40, recall5=0.80, mrr=0.54, tuning=None, **overrides):
    record = {
        "environment": {
            "model": "m",
            "tuning": tuning or {},
            "chunk_words": 220,
            "overlap_words": 30,
            "dataset": {"sha256": "d" * 64, "split": "dev", "queries": 25},
            "corpus": {"sha256": "c" * 64},
        },
        "results": {
            "hybrid": {"recall@1": recall1, "recall@5": recall5, "mrr": mrr},
            "vector": {"recall@1": 0.6, "recall@5": 0.92, "mrr": 0.73},
        },
    }
    for key, value in overrides.items():
        record["environment"][key] = value
    return record


def with_queries(record, hybrid, vector=None, queries=None):
    """Attach per-query data: hybrid and vector are lists of (recall@1, recall@5, rr) per question."""
    names = queries or [f"question {i}" for i in range(len(hybrid))]

    def rows(values):
        return [{"query": n, "recall@1": a, "recall@5": b, "rr": c} for n, (a, b, c) in zip(names, values, strict=True)]

    record["per_query"] = {"hybrid": rows(hybrid), "vector": rows(vector if vector is not None else hybrid)}
    return record


class SignTestTests(unittest.TestCase):
    def test_no_information_gives_one(self):
        self.assertEqual(sign_test_p(0, 0), 1.0)

    def test_hand_computed_values(self):
        # n=5, k=1: 2*(1+5)/32.   n=16, k=4: 2*2517/65536.   n=12, k=1: 2*13/4096.   n=9, k=1: 2*10/512.
        self.assertAlmostEqual(sign_test_p(4, 1), 12 / 32)
        self.assertAlmostEqual(sign_test_p(12, 4), 5034 / 65536)
        self.assertAlmostEqual(sign_test_p(11, 1), 26 / 4096)
        self.assertAlmostEqual(sign_test_p(8, 1), 20 / 512)

    def test_all_wins_need_six_to_reach_five_percent(self):
        self.assertAlmostEqual(sign_test_p(5, 0), 2 / 32)  # 0.0625
        self.assertAlmostEqual(sign_test_p(6, 0), 2 / 64)  # 0.03125
        self.assertGreater(sign_test_p(5, 0), 0.05)
        self.assertLess(sign_test_p(6, 0), 0.05)

    def test_symmetric_and_capped_at_one(self):
        self.assertEqual(sign_test_p(3, 9), sign_test_p(9, 3))
        self.assertEqual(sign_test_p(5, 5), 1.0)

    def test_more_lopsided_is_never_less_significant(self):
        values = [sign_test_p(w, 20 - w) for w in range(10, 21)]
        self.assertEqual(values, sorted(values, reverse=True))


class BootstrapTests(unittest.TestCase):
    def test_constant_differences_give_a_point_interval(self):
        self.assertEqual(bootstrap_ci([0.25] * 12), (0.25, 0.25))
        self.assertEqual(bootstrap_ci([0.0] * 7), (0.0, 0.0))

    def test_same_input_gives_the_same_interval(self):
        diffs = [1.0, 0.0, -1.0, 0.5, 0.0, 1.0, 0.0, 0.0]
        self.assertEqual(bootstrap_ci(diffs), bootstrap_ci(diffs))

    def test_interval_brackets_the_mean_and_widens_with_noise(self):
        steady = [0.1, 0.12, 0.08, 0.1, 0.11, 0.09, 0.1, 0.1]
        noisy = [1.0, -0.8, 0.9, -0.7, 1.0, -0.9, 0.8, -0.2]
        for diffs in (steady, noisy):
            low, high = bootstrap_ci(diffs)
            self.assertLessEqual(low, sum(diffs) / len(diffs))
            self.assertGreaterEqual(high, sum(diffs) / len(diffs))
        width = lambda diffs: bootstrap_ci(diffs)[1] - bootstrap_ci(diffs)[0]  # noqa: E731
        self.assertLess(width(steady), width(noisy))

    def test_all_positive_differences_exclude_zero(self):
        low, _ = bootstrap_ci([0.2, 0.3, 0.1, 0.25, 0.15, 0.2, 0.3, 0.1])
        self.assertGreater(low, 0)

    def test_nothing_to_resample_is_an_error(self):
        with self.assertRaises(ValueError):
            bootstrap_ci([])


class PairedValueTests(unittest.TestCase):
    def test_pairs_are_matched_by_question_text_not_by_position(self):
        a = with_queries(run(), [(1, 1, 1.0), (0, 1, 0.5)], queries=["x", "y"])
        b = with_queries(run(), [(0, 1, 0.5), (1, 1, 1.0)], queries=["y", "x"])
        self.assertEqual(paired_values(a, "hybrid", b, "hybrid", "recall@1"), [(1, 1), (0, 0)])

    def test_different_questions_are_rejected(self):
        a = with_queries(run(), [(1, 1, 1.0)], queries=["x"])
        b = with_queries(run(), [(1, 1, 1.0)], queries=["z"])
        with self.assertRaisesRegex(ValueError, "same questions"):
            paired_values(a, "hybrid", b, "hybrid", "rr")

    def test_records_without_per_query_data_say_how_to_fix_it(self):
        with self.assertRaisesRegex(ValueError, "record it again"):
            paired_values(run(), "hybrid", run(), "hybrid", "rr")

    def test_summary_counts_wins_losses_and_ties(self):
        pairs = [(0, 1), (0, 1), (1, 0), (1, 1), (0.5, 0.5 + 1e-12), (0, 1)]
        summary = summarize_pairs(pairs)
        self.assertEqual((summary["wins"], summary["losses"], summary["ties"], summary["n"]), (3, 1, 2, 6))
        self.assertAlmostEqual(summary["p"], sign_test_p(3, 1))
        self.assertAlmostEqual(summary["mean"], 2 / 6, places=6)
        self.assertLessEqual(summary["low"], summary["mean"])
        self.assertLessEqual(summary["mean"], summary["high"])


class ComparableTests(unittest.TestCase):
    def test_identical_setups_are_comparable_even_when_tuning_differs(self):
        self.assertEqual(comparable(run(), run(tuning={"rrf_k": 10})), [])

    def test_each_difference_is_reported(self):
        cases = {
            "model": run(model="other"),
            "query file": run(dataset={"sha256": "x" * 64, "split": "dev"}),
            "split": run(dataset={"sha256": "d" * 64, "split": "test"}),
            "corpus": run(corpus={"sha256": "y" * 64}),
            "chunk size": run(chunk_words=100),
        }
        for label, other in cases.items():
            with self.subTest(label=label):
                found = comparable(run(), other)
                self.assertEqual(len(found), 1)
                self.assertIn(label, found[0])


class DevRuleTests(unittest.TestCase):
    def test_exactly_two_queries_better_passes_despite_float_noise(self):
        self.assertTrue(meets_dev_rule(run(0.40), run(0.40 + MIN_RECALL_GAIN, mrr=0.60)))
        self.assertTrue(meets_dev_rule(run(0.1 + 0.2), run(0.1 + 0.2 + 0.08, mrr=0.60)))

    def test_one_query_better_is_not_enough(self):
        self.assertFalse(meets_dev_rule(run(0.40), run(0.44, mrr=0.60)))

    def test_mrr_must_rise(self):
        self.assertFalse(meets_dev_rule(run(0.40, mrr=0.54), run(0.52, mrr=0.54)))
        self.assertFalse(meets_dev_rule(run(0.40, mrr=0.54), run(0.52, mrr=0.50)))

    def test_recall_at_5_must_not_fall(self):
        self.assertFalse(meets_dev_rule(run(0.40, 0.80), run(0.52, 0.76, mrr=0.60)))
        self.assertTrue(meets_dev_rule(run(0.40, 0.80), run(0.52, 0.80, mrr=0.60)))

    def test_describe(self):
        self.assertEqual(describe(run()), "(defaults)")
        self.assertEqual(describe(run(tuning={"rrf_k": 10, "drop_stopwords": True})), "rrf_k=10, drop_stopwords=true")


class CliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def save(self, name, record):
        path = self.dir / name
        path.write_text(json.dumps(record), encoding="utf-8")
        return str(path)

    def compare(self, *paths):
        script = REPO / "eval" / "compare_runs.py"
        return subprocess.run([sys.executable, str(script), *paths], capture_output=True, text=True, cwd=REPO)

    def test_table_applies_the_rule_to_each_candidate(self):
        base = self.save("base.json", run())
        good = self.save("good.json", run(0.52, 0.84, 0.62, tuning={"rrf_k": 10}))
        bad = self.save("bad.json", run(0.44, 0.80, 0.56, tuning={"rrf_k": 20}))
        result = self.compare(base, good, bad)
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [line for line in result.stdout.splitlines() if line.startswith(("base", "good", "bad"))]
        lines = {line.split()[0]: line for line in rows}
        self.assertTrue(lines["base"].rstrip().endswith("baseline"))
        self.assertTrue(lines["good"].rstrip().endswith("meets"))
        self.assertTrue(lines["bad"].rstrip().endswith("does not meet"))
        self.assertIn("reference, vector arm", result.stdout)

    def test_incomparable_runs_exit_2(self):
        base = self.save("base.json", run())
        other = self.save("other.json", run(model="different"))
        result = self.compare(base, other)
        self.assertEqual(result.returncode, 2)
        self.assertIn("model differs", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_unreadable_or_incomplete_files_exit_2(self):
        base = self.save("base.json", run())
        broken = self.save("broken.json", {"results": {}})
        for other in (broken, str(self.dir / "missing.json")):
            with self.subTest(other=other):
                result = self.compare(base, other)
                self.assertEqual(result.returncode, 2)
                self.assertNotIn("Traceback", result.stderr)

    def test_paired_section_is_printed_when_runs_have_per_query_data(self):
        wins = [(1, 1, 1.0)] * 4 + [(0, 1, 0.5)] * 6
        base = self.save("base.json", with_queries(run(), [(0, 1, 0.5)] * 10, [(1, 1, 1.0)] * 10))
        better = self.save("better.json", with_queries(run(tuning={"rrf_k": 10}), wins, [(1, 1, 1.0)] * 10))
        result = self.compare(base, better, "--versus-arm", "vector")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Paired comparison with the baseline: hybrid, 10 questions", result.stdout)
        self.assertIn("4/0/6", result.stdout)  # recall@1: four wins, no losses, six ties
        self.assertIn("0.125", result.stdout)  # sign test for 4 wins and 0 losses: 2/16
        self.assertIn("The vector arm against hybrid inside each run", result.stdout)
        self.assertIn("MRR difference [95% CI]", result.stdout)

    def test_older_records_skip_the_paired_section_with_a_note(self):
        base = self.save("base.json", run())
        other = self.save("other.json", run(0.5, mrr=0.6))
        result = self.compare(base, other)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Paired statistics skipped", result.stdout)
        self.assertIn("other.json", result.stdout)

    def test_mismatched_questions_in_per_query_data_exit_2(self):
        base = self.save("base.json", with_queries(run(), [(1, 1, 1.0)], queries=["x"]))
        other = self.save("other.json", with_queries(run(), [(1, 1, 1.0)], queries=["z"]))
        result = self.compare(base, other)
        self.assertEqual(result.returncode, 2)
        self.assertIn("same questions", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_records_are_not_modified(self):
        base = run()
        before = copy.deepcopy(base)
        meets_dev_rule(base, run(0.6, mrr=0.7))
        self.assertEqual(base, before)


if __name__ == "__main__":
    unittest.main()
