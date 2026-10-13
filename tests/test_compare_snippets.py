import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))

from compare_runs import ranking_changes, snippet_pairs  # noqa: E402


def run_record():
    return {
        "environment": {
            "model": "m",
            "tuning": {},
            "chunk_words": 220,
            "overlap_words": 30,
            "dataset": {"sha256": "d" * 64, "split": None, "queries": 10},
            "corpus": {"sha256": "c" * 64},
        },
        "results": {
            "hybrid": {"recall@1": 0.5, "recall@5": 0.8, "mrr": 0.6},
            "vector": {"recall@1": 0.6, "recall@5": 0.9, "mrr": 0.7},
        },
    }


def rows(shown1, shown5=None, chunk1=None, ranks=None, with_evidence=None):
    """Per-query rows. A question without evidence has no snippet fields, like in a real record."""
    n = len(shown1)
    shown5 = shown5 if shown5 is not None else shown1
    chunk1 = chunk1 if chunk1 is not None else shown1
    ranks = ranks if ranks is not None else [1] * n
    with_evidence = with_evidence if with_evidence is not None else [True] * n
    out = []
    for i in range(n):
        row = {"query": f"question {i}", "first_rank": ranks[i], "recall@1": 1, "recall@5": 1, "rr": 1.0}
        if with_evidence[i]:
            row.update({"shows_answer@1": shown1[i], "shows_answer@5": shown5[i], "chunk_has_answer@1": chunk1[i]})
        out.append(row)
    return out


def without_snippets(row_list):
    return [{k: v for k, v in r.items() if "answer" not in k} for r in row_list]


def record(hybrid, vector=None):
    rec = run_record()
    rec["per_query"] = {"hybrid": hybrid, "vector": vector if vector is not None else hybrid}
    return rec


class SnippetPairTests(unittest.TestCase):
    def test_only_questions_with_evidence_in_both_runs_are_paired(self):
        a = record(rows([False, False, True], with_evidence=[True, True, False]))
        b = record(rows([True, False, True], with_evidence=[True, False, True]))
        self.assertEqual(snippet_pairs(a, b, "hybrid", "shows_answer@1"), [(False, True)])

    def test_a_missing_mode_gives_nothing_rather_than_an_error(self):
        a = record(rows([True]))
        self.assertEqual(snippet_pairs(a, a, "keyword", "shows_answer@1"), [])

    def test_ranking_changes_counts_questions_whose_first_hit_moved(self):
        a = record(rows([True] * 4, ranks=[1, 2, None, 5]))
        b = record(rows([True] * 4, ranks=[1, 3, None, 4]))
        self.assertEqual(ranking_changes(a, b, "hybrid"), 2)
        self.assertEqual(ranking_changes(a, a, "hybrid"), 0)
        self.assertIsNone(ranking_changes(a, a, "keyword"))


class SnippetReportTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def save(self, name, rec):
        path = self.dir / name
        path.write_text(json.dumps(rec), encoding="utf-8")
        return str(path)

    def compare(self, *paths):
        return subprocess.run(
            [sys.executable, str(REPO / "eval" / "compare_runs.py"), *paths], capture_output=True, text=True
        )

    def test_gains_are_reported_per_mode_with_the_sign_test_and_unchanged_rankings(self):
        before = [False] * 6 + [True] * 4
        after = [True] * 3 + [False] * 3 + [True] * 4  # three gained, none lost
        base = self.save("base.json", record(rows(before), rows([False] * 10)))
        wide = self.save("wide.json", record(rows(after, chunk1=before), rows([False] * 10)))
        result = self.compare(base, wide)
        self.assertEqual(result.returncode, 0, result.stderr)
        out = result.stdout
        self.assertIn("Answer shown to the user: hybrid, 10 questions with evidence", out)
        self.assertIn("Answer shown to the user: vector, 10 questions with evidence", out)
        hybrid_block = out.split("Answer shown to the user: hybrid")[1].split("Answer shown to the user: vector")[0]
        row = next(line for line in hybrid_block.splitlines() if line.startswith("wide"))
        self.assertIn("3/0/7", row)  # answer in snippet @1: three wins, no losses, seven ties
        self.assertIn("0.250", row)  # exact sign test for 3 wins and 0 losses: 2/8
        self.assertIn("0/0/10", row)  # answer in the chunk did not change

    def test_a_long_run_name_does_not_run_into_the_next_column(self):
        base = self.save("base.json", record(rows([False, False])))
        long = self.save("snippet-adoption-after-a-rather-long-name.json", record(rows([True, False])))
        result = self.compare(base, long)
        self.assertEqual(result.returncode, 0, result.stderr)
        block = result.stdout.split("Answer shown to the user: hybrid")[1].split("Answer shown to the user: vector")[0]
        row = next(line for line in block.splitlines() if line.startswith("snippet-adoption"))
        self.assertRegex(row, r"a-rather-long-name\s{2,}\d")

    def test_a_run_without_snippet_data_is_skipped_with_a_note(self):
        base = self.save("base.json", record(rows([False, True])))
        plain = self.save("plain.json", record(without_snippets(rows([True, True]))))
        result = self.compare(base, plain)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Snippet comparison skipped", result.stdout)
        self.assertNotIn("Answer shown to the user", result.stdout)

    def test_runs_that_never_used_snippets_print_nothing_extra(self):
        plain = without_snippets(rows([True, True]))
        a, b = self.save("a.json", record(plain)), self.save("b.json", record(plain))
        result = self.compare(a, b)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("Snippet comparison", result.stdout)
        self.assertNotIn("Answer shown to the user", result.stdout)


if __name__ == "__main__":
    unittest.main()
