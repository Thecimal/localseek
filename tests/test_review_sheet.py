import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))

from make_review_sheet import render  # noqa: E402

ITEMS = [
    {"query": "First question?", "evidence": {"a/one.md": "quote one"}, "split": "dev", "relevant": ["a/one.md"]},
    {"query": "Second question?", "evidence": {"b.md": "quote b", "c.md": "quote c"}, "split": "test"},
]


class ReviewSheetTests(unittest.TestCase):
    def test_every_question_and_quote_is_listed_with_two_checkboxes(self):
        sheet = render(ITEMS, "q.json")
        self.assertIn("## 1. First question?", sheet)
        self.assertIn("## 2. Second question?", sheet)
        for fragment in ('`a/one.md`: "quote one"', '`b.md`: "quote b"', '`c.md`: "quote c"'):
            self.assertIn(fragment, sheet)
        self.assertEqual(sheet.count("- [ ] The quoted text answers the question"), 2)
        self.assertEqual(sheet.count("- [ ] No other document answers it"), 2)

    def test_the_reviewer_is_not_shown_splits_or_search_results(self):
        sheet = render(ITEMS, "q.json")
        self.assertNotIn("dev", sheet.replace("developer", ""))
        self.assertNotIn("test split", sheet)

    def test_output_is_deterministic(self):
        self.assertEqual(render(ITEMS, "q.json"), render(ITEMS, "q.json"))

    def test_cli_on_the_real_v3_set_and_on_bad_input(self):
        script = REPO / "eval" / "make_review_sheet.py"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sheet.md"
            v3 = REPO / "eval" / "benchmark" / "queries-v3.json"
            ok = subprocess.run([sys.executable, str(script), "--queries", str(v3), "--out", str(out)],
                                capture_output=True, text=True)  # fmt: skip
            self.assertEqual(ok.returncode, 0, ok.stderr)
            expected = len(json.loads(v3.read_text(encoding="utf-8")))
            self.assertEqual(out.read_text(encoding="utf-8").count("\n## "), expected)
            bad = Path(tmp) / "bad.json"
            bad.write_text('[{"query": "no evidence"}]', encoding="utf-8")
            failed = subprocess.run([sys.executable, str(script), "--queries", str(bad), "--out", str(out)],
                                    capture_output=True, text=True)  # fmt: skip
            self.assertEqual(failed.returncode, 2)
            self.assertNotIn("Traceback", failed.stderr)


if __name__ == "__main__":
    unittest.main()
