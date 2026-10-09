import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from helpers import write
from localseek.search import make_snippet, query_terms

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))

import snippet_experiment as se  # noqa: E402

FILLER = " ".join(f"filler{i}" for i in range(60))  # about 400 characters


class ContentTermTests(unittest.TestCase):
    def test_function_words_are_removed(self):
        self.assertEqual(se.content_terms(query_terms("how do I roll back a failed deployment")),
                         ["roll", "back", "failed", "deployment"])  # fmt: skip

    def test_a_query_of_only_function_words_keeps_them(self):
        self.assertEqual(se.content_terms(["what", "is", "it"]), ["what", "is", "it"])


class DensestWindowTests(unittest.TestCase):
    def text(self):
        return f"alpha {FILLER} beta gamma alpha delta {FILLER}"

    def test_short_text_is_returned_whole(self):
        self.assertEqual(se.densest_window("a short chunk about alpha", ["alpha"]), "a short chunk about alpha")

    def test_the_window_goes_to_the_cluster_not_to_the_first_mention(self):
        text = self.text()
        first = make_snippet(text, ["alpha", "beta", "gamma", "delta"])
        dense = se.densest_window(text, ["alpha", "beta", "gamma", "delta"])
        self.assertNotIn("delta", first)  # the shipped function anchors on the lone early "alpha"
        for word in ("beta", "gamma", "delta"):
            self.assertIn(word, dense)

    def test_ties_go_to_the_earliest_window(self):
        text = f"alpha {FILLER} alpha {FILLER}"
        self.assertTrue(se.densest_window(text, ["alpha"]).startswith("alpha"))

    def test_no_term_found_means_the_start_without_a_leading_ellipsis(self):
        window = se.densest_window(self.text(), ["zzz"])
        self.assertTrue(window.startswith("alpha"))
        self.assertTrue(window.endswith("…"))

    def test_width_is_respected(self):
        for width in (280, 420):
            self.assertLessEqual(len(se.densest_window(self.text(), ["beta", "gamma"], width)), width + 2)

    def test_function_words_do_not_attract_the_window(self):
        # Three distinct function words sit together at the start; two content words sit together later.
        # Counting function words would favour the start (3 against 2) and miss the answer.
        text = f"how do the setup steps begin {FILLER} roll back by running kubectl rollout undo {FILLER}"
        window = se.densest_window(text, ["how", "do", "the", "roll", "back"])
        self.assertIn("kubectl", window)
        self.assertNotIn("setup", window)


class CandidateDefinitionTests(unittest.TestCase):
    def test_candidates_are_what_the_document_says(self):
        text = f"the {FILLER} roll back with kubectl {FILLER}"
        terms = query_terms("how do the roll back")
        _, _, s0 = se.STRATEGIES["S0"]
        _, _, s1 = se.STRATEGIES["S1"]
        _, _, s4 = se.STRATEGIES["S4"]
        self.assertEqual(s0(text, terms), make_snippet(text, terms))
        self.assertEqual(s1(text, terms), make_snippet(text, se.content_terms(terms)))
        self.assertEqual(s4(text, terms), make_snippet(text, terms, width=420))

    def test_widths_are_declared_honestly(self):
        self.assertEqual({sid: spec[1] for sid, spec in se.STRATEGIES.items()},
                         {"S0": 280, "S1": 280, "S2": 280, "S3": 420, "S4": 420})  # fmt: skip

    def test_a_candidate_over_its_limit_is_an_error(self):
        original = se.STRATEGIES["S1"]
        se.STRATEGIES["S1"] = (original[0], 280, lambda text, terms: "x" * 400)
        try:
            pair = {"chunk": "c", "terms": ["c"], "quote": "c"}
            with self.assertRaisesRegex(ValueError, "S1 produced 400 characters"):
                se.evaluate([pair])
        finally:
            se.STRATEGIES["S1"] = original

    def test_the_document_matches_the_code(self):
        doc = (REPO / "eval" / "benchmark" / "snippet-experiment.md").read_text(encoding="utf-8")
        for sid, (description, width, _) in se.STRATEGIES.items():
            self.assertIn(f"| {sid} |", doc)
            self.assertIn(description[:40], doc)
            self.assertIn(f"| {width} |", doc)
        for number in (se.MIN_NET_WINS, se.MAX_LOSSES, se.MIN_NET_OVER_CONTROL, se.MAX_P):
            self.assertIn(str(number), doc)


def shown(**wins):
    """A baseline of 40 misses, and for each candidate the given number of wins and losses against it."""
    result = {"S0": [False] * 40}
    for sid in ("S1", "S2", "S3", "S4"):
        w, ell = wins.get(sid, (0, 0))
        result[sid] = [True] * w + [False] * (40 - w - ell) + [False] * ell
    return result


class DecisionTests(unittest.TestCase):
    def verdict(self, **wins):
        data = shown(**wins)
        # a loss means S0 was right and the candidate is wrong: give S0 those pairs
        for sid in ("S1", "S2", "S3", "S4"):
            w, ell = wins.get(sid, (0, 0))
            for i in range(ell):
                data["S0"][39 - i] = True
        return se.decide(data)

    def test_eight_net_wins_with_p_below_five_percent_is_eligible(self):
        out = self.verdict(S1=(10, 0))
        self.assertTrue(out["verdicts"]["S1"]["eligible"])
        self.assertEqual(out["selected"], "S1")

    def test_seven_net_wins_is_not_enough(self):
        self.assertFalse(self.verdict(S1=(7, 0))["verdicts"]["S1"]["eligible"])

    def test_more_than_three_losses_disqualifies(self):
        self.assertFalse(self.verdict(S1=(14, 4))["verdicts"]["S1"]["eligible"])

    def test_eight_net_wins_can_still_fail_the_sign_test(self):
        v = self.verdict(S1=(11, 3))["verdicts"]["S1"]
        self.assertEqual(v["net"], 8)
        self.assertGreater(v["p"], 0.05)
        self.assertFalse(v["eligible"])

    def test_nothing_eligible_means_no_selection(self):
        self.assertIsNone(self.verdict()["selected"])

    def test_the_narrowest_eligible_candidate_wins_when_it_also_beats_the_control(self):
        out = self.verdict(S1=(20, 0), S4=(15, 0))  # S1 is 5 pairs ahead of the control, and narrower
        self.assertTrue(out["verdicts"]["S1"]["survives_control"])
        self.assertEqual(out["selected"], "S1")

    def test_a_narrower_candidate_that_does_not_beat_an_eligible_control_is_dropped(self):
        # By the pre-registered rule, showing more text is the simpler change and wins when it is clearly better.
        out = self.verdict(S1=(9, 0), S4=(15, 0))
        self.assertTrue(out["verdicts"]["S1"]["eligible"])
        self.assertFalse(out["verdicts"]["S1"]["survives_control"])
        self.assertEqual(out["selected"], "S4")

    def test_a_candidate_must_beat_the_wide_control_when_the_control_is_eligible(self):
        # S3 is eligible, but only 2 better than the control S4, so it is dropped and S4 is selected.
        out = self.verdict(S3=(14, 0), S4=(12, 0))
        self.assertTrue(out["verdicts"]["S3"]["eligible"])
        self.assertFalse(out["verdicts"]["S3"]["survives_control"])
        self.assertEqual(out["selected"], "S4")

    def test_a_candidate_that_beats_the_control_by_enough_is_kept(self):
        out = self.verdict(S3=(17, 0), S4=(12, 0))
        self.assertTrue(out["verdicts"]["S3"]["survives_control"])
        self.assertEqual(out["selected"], "S3")  # same width as S4, shows more

    def test_the_control_is_not_needed_when_it_is_not_eligible(self):
        out = self.verdict(S3=(12, 0), S4=(3, 0))
        self.assertEqual(out["selected"], "S3")


class PairTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        write(self.dir / "c" / "short.md", "A short note about the handshake.")
        words = [f"w{i}" for i in range(500)]
        words[300:305] = ["secret", "handshake", "is", "a", "knock"]
        write(self.dir / "c" / "long.md", " ".join(words))

    def tearDown(self):
        self._tmp.cleanup()

    def run_cli(self, *args):
        script = str(REPO / "eval" / "snippet_experiment.py")
        command = [sys.executable, script, "--corpus", str(self.dir / "c"), *args]
        return subprocess.run(command, capture_output=True, text=True)

    def queries(self, items):
        path = self.dir / "q.json"
        path.write_text(json.dumps(items), encoding="utf-8")
        return path

    def test_only_pairs_on_long_chunks_are_built_and_the_correct_chunk_is_used(self):
        q = self.queries([
            {"query": "what knock", "evidence": {"long.md": "secret handshake is a knock"}, "category": "x"},
            {"query": "a note", "evidence": {"short.md": "short note about the handshake"}},
        ])  # fmt: skip
        pairs = se.build_pairs(self.dir / "c", [q])
        self.assertEqual([p["path"] for p in pairs], ["long.md"])
        self.assertIn("secret handshake is a knock", " ".join(pairs[0]["chunk"].split()))
        self.assertEqual(pairs[0]["terms"], ["what", "knock"])

    def test_a_quote_in_no_chunk_is_an_error(self):
        q = self.queries([{"query": "q", "evidence": {"long.md": "absent words"}}])
        with self.assertRaisesRegex(ValueError, "is in no chunk"):
            se.build_pairs(self.dir / "c", [q])

    def test_cli_writes_a_complete_record(self):
        q = self.queries([{"query": "what knock", "evidence": {"long.md": "secret handshake is a knock"}}])
        out = self.dir / "out" / "snippets.json"
        run = self.run_cli("--queries", str(q), "--out", str(out))
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("1 pairs whose correct chunk is longer than 280 characters", run.stdout)
        record = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(set(record["shown"]), set(se.STRATEGIES))
        self.assertEqual(len(record["pairs"]), 1)
        self.assertNotIn("chunk", record["pairs"][0])

    def test_cli_rejects_unusable_input(self):
        short_only = self.queries([{"query": "q", "evidence": {"short.md": "short note about the handshake"}}])
        for queries in (short_only, self.dir / "missing.json"):
            run = self.run_cli("--queries", str(queries))
            self.assertEqual(run.returncode, 2, run.stderr)
            self.assertNotIn("Traceback", run.stderr)


if __name__ == "__main__":
    unittest.main()
