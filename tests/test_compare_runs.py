import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))

from compare_runs import MIN_RECALL_GAIN, comparable, describe, meets_dev_rule  # noqa: E402


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

    def test_records_are_not_modified(self):
        base = run()
        before = copy.deepcopy(base)
        meets_dev_rule(base, run(0.6, mrr=0.7))
        self.assertEqual(base, before)


if __name__ == "__main__":
    unittest.main()
