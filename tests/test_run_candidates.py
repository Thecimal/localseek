"""The experiment runner. These tests use a tiny temporary benchmark: they never run the real held-out test split."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from helpers import write

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))

import run_candidates as rc  # noqa: E402


def row(cid, tuning="", dev_rule=True, mrr=0.5, order=1, **gates):
    gates = gates or {"hash": True}
    return {"id": cid, "tuning": tuning, "dev_rule": dev_rule, "mrr": mrr, "order": order, "gates": gates}


class SelectTests(unittest.TestCase):
    def test_the_baseline_is_never_selected(self):
        self.assertIsNone(rc.select([row("C0", mrr=0.9)]))

    def test_candidates_must_meet_the_dev_rule_and_every_gate(self):
        rows = [row("C1", dev_rule=False, mrr=0.9), row("C2", mrr=0.8, hash=True, bge=False), row("C3", mrr=0.4)]
        self.assertEqual(rc.select(rows), "C3")
        self.assertIsNone(rc.select(rows[:2]))

    def test_highest_dev_mrr_wins(self):
        self.assertEqual(rc.select([row("C1", mrr=0.6), row("C2", mrr=0.7, order=2)]), "C2")

    def test_ties_go_to_fewer_changed_options_then_to_list_order(self):
        # The two-option candidate comes first in the list, so only the option count can make the other one win.
        two = row("C1", "drop_stopwords=true,keyword_limit=20", mrr=0.6, order=1)
        one = row("C2", "keyword_limit=20", mrr=0.6, order=2)
        self.assertEqual(rc.select([two, one]), "C2")
        earlier, later = row("C1", "rrf_k=20", mrr=0.6, order=1), row("C2", "rrf_k=10", mrr=0.6, order=2)
        self.assertEqual(rc.select([later, earlier]), "C1")


class HelperTests(unittest.TestCase):
    def test_thresholds_file_names(self):
        self.assertEqual(rc.thresholds_path("BAAI/bge-small-en-v1.5").name, "thresholds-bge-small-en-v1.5.json")
        self.assertEqual(rc.thresholds_path("hash").name, "thresholds-hash.json")

    def test_candidate_selection(self):
        self.assertEqual([i for i, _ in rc.chosen_candidates(None)], list(rc.SPECS))
        self.assertEqual([i for i, _ in rc.chosen_candidates("C3, C6")], ["C0", "C3", "C6"])
        self.assertIn("unknown candidate(s) C9", rc.chosen_candidates("C3,C9"))

    def test_candidate_options_are_all_valid_tunings(self):
        from retrieval_eval import parse_tuning

        for cid, spec in rc.CANDIDATES:
            with self.subTest(candidate=cid):
                parse_tuning(spec)

    def test_the_documented_list_matches_the_code(self):
        doc = (REPO / "eval" / "benchmark" / "experiments.md").read_text(encoding="utf-8")
        for cid, spec in rc.CANDIDATES:
            self.assertIn(cid, doc)
            if spec:
                self.assertIn(f"`{spec}`", doc, f"{cid} is not documented as `{spec}`")


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        corpus = self.dir / "corpus"
        write(corpus / "recipes" / "sourdough.md", "Feed the sourdough starter twice a day.")
        write(corpus / "recipes" / "notes.md", "Sourdough starter feeding schedule and flour ratios.")
        write(corpus / "work" / "notes.md", "Quarterly tax filing deadlines and invoice reminders.")
        write(corpus / "notes.md", "Bicycle chain lubrication and tire pressure.")
        rows = [
            ("tax invoice deadlines", "work/notes.md", "dev"),
            ("bicycle chain tire", "notes.md", "dev"),
            ("flour ratios", "recipes/notes.md", "test"),
            ("feed the starter twice", "recipes/sourdough.md", "test"),
        ]
        queries = [{"query": q, "relevant": [p], "category": "paraphrase", "split": s} for q, p, s in rows]
        (self.dir / "queries.json").write_text(json.dumps(queries), encoding="utf-8")
        (self.dir / "thresholds.json").write_text(json.dumps({"hybrid": {"recall@1": 0.0}}), encoding="utf-8")
        self.patches = {"CORPUS": corpus, "V1": self.dir / "queries.json", "V2": self.dir / "queries.json"}
        self.originals = {name: getattr(rc, name) for name in self.patches}
        for name, value in self.patches.items():
            setattr(rc, name, value)
        self.original_thresholds = rc.thresholds_path
        rc.thresholds_path = lambda model: self.dir / "thresholds.json"
        self.out = self.dir / "out"

    def tearDown(self):
        for name, value in self.originals.items():
            setattr(rc, name, value)
        rc.thresholds_path = self.original_thresholds
        self._tmp.cleanup()

    def call(self, function, **kwargs):
        buffer, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(errors):
            code = function(Namespace(model="hash", out=str(self.out), **kwargs))
        return code, buffer.getvalue(), errors.getvalue()

    def select_in_summary(self, candidate):
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "summary.json").write_text(json.dumps({"selected": candidate}), encoding="utf-8")

    def test_dev_writes_records_and_a_summary(self):
        code, stdout, _ = self.call(rc.dev, candidates="C6")
        self.assertEqual(code, 0)
        self.assertTrue((self.out / "dev-C0.json").exists() and (self.out / "dev-C6.json").exists())
        summary = json.loads((self.out / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual([r["id"] for r in summary["candidates"]], ["C0", "C6"])
        self.assertIn(summary["selected"], (None, "C6"))
        self.assertIn("v1 gates", stdout)
        record = json.loads((self.out / "dev-C6.json").read_text(encoding="utf-8"))
        self.assertEqual(record["environment"]["dataset"]["split"], "dev")
        self.assertEqual(record["environment"]["tuning"], {"drop_stopwords": True})

    def test_dev_rejects_unknown_candidates_and_missing_thresholds(self):
        code, _, errors = self.call(rc.dev, candidates="C9")
        self.assertEqual((code, self.out.exists()), (2, False))
        self.assertIn("unknown candidate", errors)
        rc.thresholds_path = lambda model: self.dir / "absent.json"
        code, _, errors = self.call(rc.dev, candidates="C6")
        self.assertEqual(code, 2)
        self.assertIn("derive it with eval/derive_thresholds.py", errors)

    def test_confirm_refuses_without_a_dev_step_or_for_a_candidate_that_was_not_selected(self):
        self.assertEqual(self.call(rc.confirm, candidate="C6", allow_rerun=False)[0], 2)
        self.select_in_summary("C3")
        code, _, errors = self.call(rc.confirm, candidate="C6", allow_rerun=False)
        self.assertEqual(code, 2)
        self.assertIn("selected 'C3'", errors)
        self.assertFalse(list(self.out.glob("test-*.json")))

    def test_confirm_refuses_the_baseline_and_unknown_ids(self):
        self.select_in_summary("C6")
        for candidate in ("C0", "C9"):
            self.assertEqual(self.call(rc.confirm, candidate=candidate, allow_rerun=False)[0], 2)

    def test_the_test_split_can_only_be_used_once(self):
        self.select_in_summary("C6")
        code, stdout, _ = self.call(rc.confirm, candidate="C6", allow_rerun=False)
        self.assertIn(code, (0, 1))
        self.assertIn("on the test split", stdout)
        marker = (self.out / rc.MARKER).read_text(encoding="utf-8")
        self.assertIn("C6", marker)
        record = json.loads((self.out / "test-C6.json").read_text(encoding="utf-8"))
        self.assertEqual(record["environment"]["dataset"]["split"], "test")
        code, _, errors = self.call(rc.confirm, candidate="C6", allow_rerun=False)
        self.assertEqual(code, 2)
        self.assertIn("already used", errors)
        self.assertIn(self.call(rc.confirm, candidate="C6", allow_rerun=True)[0], (0, 1))


if __name__ == "__main__":
    unittest.main()
