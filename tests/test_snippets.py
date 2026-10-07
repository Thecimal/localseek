"""Answer-in-snippet metrics: does what the user is shown contain the answer, not just the right file?"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from helpers import write
from localseek.chunker import chunk_sections
from localseek.config import Settings
from localseek.embedder import HashEmbedder
from localseek.extractors import extract
from localseek.indexer import index_paths
from localseek.search import Searcher
from localseek.store import Store

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))

from retrieval_eval import (  # noqa: E402
    DatasetError,
    Query,
    Trace,
    answer_outcomes,
    corpus_chunk_texts,
    load_dataset,
    per_query_records,
    snippet_summary,
)

QUOTE = "the secret handshake is a double knock"


def trace(snippets, ranked=("a.md", "b.md"), chunks=(0, 0), evidence=None, first=1):
    evidence = {"a.md": QUOTE} if evidence is None else evidence
    query = Query("q", tuple(evidence) or ("a.md",), evidence=evidence)
    return Trace(query, tuple(ranked), first, tuple(snippets), tuple(chunks))


class AnswerOutcomeTests(unittest.TestCase):
    def test_the_snippet_of_the_right_file_shows_the_answer(self):
        out = answer_outcomes(trace([f"intro … {QUOTE}. more", "x"]))
        self.assertEqual((out["shows_answer@1"], out["shows_answer@5"]), (True, True))

    def test_matching_ignores_case_and_whitespace(self):
        out = answer_outcomes(trace(["The  SECRET\nhandshake  is a double\tknock", "x"]))
        self.assertTrue(out["shows_answer@1"])

    def test_a_quote_cut_by_the_window_does_not_count(self):
        out = answer_outcomes(trace(["… the secret hands…", "x"]))
        self.assertFalse(out["shows_answer@1"])

    def test_text_in_an_irrelevant_file_does_not_count(self):
        out = answer_outcomes(trace([QUOTE, QUOTE], ranked=("other.md", "a.md"), first=2))
        self.assertFalse(out["shows_answer@1"])
        self.assertTrue(out["shows_answer@5"])  # a.md is second and its snippet shows the quote

    def test_top_five_looks_at_five_hits_and_no_more(self):
        ranked = ("x1.md", "x2.md", "x3.md", "x4.md", "x5.md", "a.md")
        snippets = ["", "", "", "", "", QUOTE]
        out = answer_outcomes(trace(snippets, ranked, chunks=(0,) * 6, first=6))
        self.assertEqual((out["shows_answer@1"], out["shows_answer@5"]), (False, False))
        ranked = ("x1.md", "x2.md", "x3.md", "x4.md", "a.md")
        out = answer_outcomes(trace(["", "", "", "", QUOTE], ranked, chunks=(0,) * 5, first=5))
        self.assertTrue(out["shows_answer@5"])

    def test_the_chunk_check_uses_the_hit_chunk_ordinal(self):
        chunks = {"a.md": ["unrelated", f"filler {QUOTE} filler"]}
        self.assertTrue(answer_outcomes(trace(["x", "y"], chunks=(1, 0)), chunks)["chunk_has_answer@1"])
        self.assertFalse(answer_outcomes(trace(["x", "y"], chunks=(0, 0)), chunks)["chunk_has_answer@1"])
        self.assertFalse(answer_outcomes(trace(["x", "y"], chunks=(7, 0)), chunks)["chunk_has_answer@1"])

    def test_the_right_chunk_can_be_found_while_the_snippet_misses_the_answer(self):
        chunks = {"a.md": [f"start {QUOTE}"]}
        out = answer_outcomes(trace(["start of the chunk only", "y"]), chunks)
        self.assertEqual((out["chunk_has_answer@1"], out["shows_answer@1"]), (True, False))

    def test_chunk_outcome_is_absent_without_chunk_texts(self):
        self.assertNotIn("chunk_has_answer@1", answer_outcomes(trace([QUOTE, "y"])))

    def test_queries_without_evidence_have_no_outcome(self):
        self.assertIsNone(answer_outcomes(trace([QUOTE, "y"], evidence={})))

    def test_no_results_does_not_crash(self):
        out = answer_outcomes(trace([], ranked=(), chunks=(), first=None), {"a.md": ["x"]})
        self.assertEqual(out, {"shows_answer@1": False, "shows_answer@5": False, "chunk_has_answer@1": False})


class SummaryTests(unittest.TestCase):
    def test_rates_are_over_queries_with_evidence_only(self):
        rows = [
            {"first_rank": 1, "shows_answer@1": True, "shows_answer@5": True, "chunk_has_answer@1": True},
            {"first_rank": 2, "shows_answer@1": False, "shows_answer@5": True, "chunk_has_answer@1": False},
            {"first_rank": None, "shows_answer@1": False, "shows_answer@5": False, "chunk_has_answer@1": False},
            {"first_rank": 1},  # a query without evidence is not counted
        ]
        self.assertEqual(
            snippet_summary(rows),
            {"queries": 3, "right_file@1": 1 / 3, "shows_answer@1": 1 / 3, "shows_answer@5": 2 / 3,
             "chunk_has_answer@1": 1 / 3},
        )  # fmt: skip

    def test_no_evidence_means_no_summary(self):
        self.assertIsNone(snippet_summary([{"first_rank": 1}]))


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.root = (self.dir / "corpus").resolve()
        filler = " ".join(f"word{i}" for i in range(70))
        write(self.root / "docs" / "long.md", f"zebrafish {filler} {QUOTE.capitalize()}. " + " ".join(["tail"] * 20))
        write(self.root / "docs" / "other.md", "Quarterly tax filing deadlines and invoice reminders.")

    def tearDown(self):
        self._tmp.cleanup()

    def dataset(self, items):
        path = self.dir / "queries.json"
        path.write_text(json.dumps(items), encoding="utf-8")
        return path


class DatasetEvidenceTests(Base):
    def test_valid_evidence_is_loaded(self):
        items = [{"query": "q", "relevant": ["docs/long.md"], "evidence": {"docs/long.md": QUOTE}}]
        queries = load_dataset(self.dataset(items), self.root)
        self.assertEqual(queries[0].evidence, {"docs/long.md": QUOTE})
        plain = load_dataset(self.dataset([{"query": "q", "relevant": ["docs/long.md"]}]), self.root)
        self.assertEqual(plain[0].evidence, {})

    def test_invalid_evidence_is_rejected(self):
        for bad in ({}, "text", {"docs/long.md": ""}, {"docs/long.md": 3}, {"docs/other.md": "not relevant"}):
            with self.subTest(evidence=bad), self.assertRaises(DatasetError) as ctx:
                load_dataset(self.dataset([{"query": "q", "relevant": ["docs/long.md"], "evidence": bad}]), self.root)
            self.assertIn("'evidence'", "\n".join(ctx.exception.problems))


class ChunkTextTests(Base):
    def test_chunks_match_the_indexer_and_unreadable_files_are_skipped(self):
        write(self.root / "docs" / "big.md", " ".join(f"w{i}" for i in range(600)))
        (self.root / "docs" / "broken.pdf").write_bytes(b"this is not a pdf")
        texts = corpus_chunk_texts(self.root, Settings())
        self.assertNotIn("docs/broken.pdf", texts)
        self.assertGreater(len(texts["docs/big.md"]), 2)
        settings = Settings()
        expected = chunk_sections(extract(self.root / "docs" / "big.md"), settings.chunk_words, settings.overlap_words)
        self.assertEqual(texts["docs/big.md"], [" ".join(c.text.lower().split()) for c in expected])


class EndToEndTests(Base):
    def setUp(self):
        super().setUp()
        self.store = Store(self.dir / "i.db")
        self.embedder = HashEmbedder()
        index_paths(self.store, self.embedder, [self.root], Settings(model="hash"))
        self.searcher = Searcher(self.store, self.embedder)

    def tearDown(self):
        self.store.close()
        super().tearDown()

    def records(self, text):
        query = Query(text, ("docs/long.md",), evidence={"docs/long.md": QUOTE})
        chunk_texts = corpus_chunk_texts(self.root, Settings())
        return per_query_records(self.searcher, [query], self.root, "keyword", chunk_texts=chunk_texts)[0]

    def test_a_query_word_near_the_answer_makes_the_snippet_show_it(self):
        record = self.records("handshake")
        observed = (record["first_rank"], record["chunk_has_answer@1"], record["shows_answer@1"])
        self.assertEqual(observed, (1, True, True))

    def test_the_right_file_and_chunk_can_still_show_the_wrong_part(self):
        # The only query word sits at the very start; the snippet is a 280-character window from there.
        record = self.records("zebrafish")
        observed = (record["first_rank"], record["chunk_has_answer@1"], record["shows_answer@1"])
        self.assertEqual(observed, (1, True, False))


class CliTests(Base):
    def run_cli(self, queries, *extra):
        command = [sys.executable, str(REPO / "eval" / "run_eval.py"), "--corpus", str(self.root),
                   "--queries", str(queries), "--model", "hash", *extra]  # fmt: skip
        return subprocess.run(command, capture_output=True, text=True, cwd=REPO)

    def test_snippets_table_and_json_agree(self):
        items = [
            {"query": "handshake", "relevant": ["docs/long.md"], "evidence": {"docs/long.md": QUOTE}},
            {"query": "zebrafish", "relevant": ["docs/long.md"], "evidence": {"docs/long.md": QUOTE}},
        ]
        out = self.dir / "run.json"
        run = self.run_cli(self.dataset(items), "--snippets", "--json", str(out))
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("Answer shown to the user (2 questions with evidence)", run.stdout)
        record = json.loads(out.read_text(encoding="utf-8"))
        keyword = record["snippets"]["keyword"]
        self.assertEqual((keyword["queries"], keyword["right_file@1"], keyword["chunk_has_answer@1"]), (2, 1.0, 1.0))
        self.assertEqual(keyword["shows_answer@1"], 0.5)
        rows = record["per_query"]["keyword"]
        self.assertEqual([r["shows_answer@1"] for r in rows], [True, False])
        self.assertEqual(keyword["shows_answer@1"], sum(r["shows_answer@1"] for r in rows) / len(rows))

    def test_without_evidence_the_report_says_so_and_the_record_has_no_snippet_data(self):
        items = [{"query": "handshake", "relevant": ["docs/long.md"]}]
        out = self.dir / "run.json"
        run = self.run_cli(self.dataset(items), "--snippets", "--json", str(out))
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("needs evidence quotes", run.stdout)
        record = json.loads(out.read_text(encoding="utf-8"))
        self.assertNotIn("snippets", record)
        self.assertNotIn("shows_answer@1", record["per_query"]["keyword"][0])

    def test_the_shipped_benchmark_reports_snippets(self):
        bench = REPO / "eval" / "benchmark"
        command = [
            sys.executable, str(REPO / "eval" / "run_eval.py"),
            "--corpus", str(bench / "corpus"), "--queries", str(bench / "queries-v3.json"),
            "--model", "hash", "--split", "dev", "--snippets",
        ]  # fmt: skip
        run = subprocess.run(command, capture_output=True, text=True, cwd=REPO)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("Answer shown to the user (47 questions with evidence)", run.stdout)


if __name__ == "__main__":
    unittest.main()
