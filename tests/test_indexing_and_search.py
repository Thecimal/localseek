import os
import socket
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from helpers import CountingEmbedder, make_docx, write
from localseek.config import Settings
from localseek.embedder import HashEmbedder
from localseek.indexer import index_paths
from localseek.scanner import scan
from localseek.search import Searcher, fts_query, make_snippet, rrf
from localseek.store import ModelMismatch, Store


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.docs = self.dir / "docs"
        self.docs.mkdir()
        self.store = Store(self.dir / "index.db")
        self.embedder = CountingEmbedder()
        self.settings = Settings(model="hash")

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def index(self):
        return index_paths(self.store, self.embedder, [self.docs], self.settings)

    def populate(self):
        write(self.docs / "lease.md", "The landlord agreed to repair the heating system and radiators.")
        write(self.docs / "bread.txt", "Feed the sourdough starter twice a day with flour and water.")
        write(self.docs / "notes" / "bike.md", "Lubricate the bicycle chain and check tire pressure.")
        make_docx(self.docs / "budget.docx", ["Save twenty percent of income every month."])


class IndexingTests(Base):
    def test_first_index_adds_files(self):
        self.populate()
        stats = self.index()
        self.assertEqual((stats.scanned, stats.added), (4, 4))
        self.assertEqual(self.store.stats()["files"], 4)

    def test_second_run_changes_nothing_and_embeds_nothing(self):
        self.populate()
        self.index()
        before = self.embedder.passages_embedded
        stats = self.index()
        self.assertEqual((stats.added, stats.updated, stats.unchanged), (0, 0, 4))
        self.assertEqual(self.embedder.passages_embedded, before)

    def test_modified_file_is_reindexed(self):
        self.populate()
        self.index()
        path = self.docs / "lease.md"
        path.write_text("Completely different content about gardening tools.", encoding="utf-8")
        future = time.time() + 5
        os.utime(path, (future, future))
        stats = self.index()
        self.assertEqual(stats.updated, 1)
        hits = Searcher(self.store, self.embedder).search("gardening")
        self.assertEqual(Path(hits[0].path).name, "lease.md")
        self.assertEqual(Searcher(self.store, self.embedder).search("heating", mode="keyword"), [])

    def test_touched_but_identical_file_is_not_reembedded(self):
        self.populate()
        self.index()
        before = self.embedder.passages_embedded
        path = self.docs / "lease.md"
        future = time.time() + 5
        os.utime(path, (future, future))
        stats = self.index()
        self.assertEqual(stats.unchanged, 4)
        self.assertEqual(self.embedder.passages_embedded, before)

    def test_identical_chunk_text_reuses_embedding(self):
        write(self.docs / "a.txt", "shared paragraph text about turtles")
        self.index()
        before = self.embedder.passages_embedded
        write(self.docs / "b.txt", "shared paragraph text about turtles")
        self.index()
        self.assertEqual(self.embedder.passages_embedded, before)

    def test_deleted_file_is_removed(self):
        self.populate()
        self.index()
        (self.docs / "bread.txt").unlink()
        stats = self.index()
        self.assertEqual(stats.removed, 1)
        self.assertEqual(self.store.stats()["files"], 3)
        self.assertEqual(Searcher(self.store, self.embedder).search("sourdough", mode="keyword"), [])

    def test_model_change_requires_rebuild(self):
        self.populate()
        self.index()

        class Other(HashEmbedder):
            name = "other-model"

        with self.assertRaises(ModelMismatch):
            index_paths(self.store, Other(), [self.docs], self.settings)
        self.store.clear_index()
        stats = index_paths(self.store, Other(), [self.docs], self.settings)
        self.assertEqual(stats.added, 4)

    def test_unreadable_file_is_reported_not_fatal(self):
        write(self.docs / "ok.txt", "fine document")
        (self.docs / "broken.docx").write_text("not a zip", encoding="utf-8")
        stats = self.index()
        self.assertEqual(stats.added, 1)
        self.assertEqual(len(stats.errors), 1)

    def test_ignore_patterns(self):
        write(self.docs / "keep.txt", "keep me")
        write(self.docs / "node_modules" / "lib.txt", "ignore me")
        write(self.docs / "drafts" / "wip.txt", "ignore me too")
        write(self.docs / ".localseekignore", "drafts\n# comment\n")
        stats = self.index()
        self.assertEqual(stats.scanned, 1)

    def test_size_limit_skips_large_files(self):
        write(self.docs / "big.txt", "x" * 2000)
        found = list(scan([self.docs], [], 1000, {".txt"}, on_skip=lambda p, r: None))
        self.assertEqual(found, [])


class SearchTests(Base):
    def setUp(self):
        super().setUp()
        self.populate()
        self.index()
        self.searcher = Searcher(self.store, self.embedder)

    def top(self, query, **kwargs):
        hits = self.searcher.search(query, **kwargs)
        return Path(hits[0].path).name if hits else None

    def test_finds_expected_documents(self):
        self.assertEqual(self.top("landlord heating repair"), "lease.md")
        self.assertEqual(self.top("sourdough starter feeding"), "bread.txt")
        self.assertEqual(self.top("bicycle chain tire"), "bike.md")
        self.assertEqual(self.top("income savings percent"), "budget.docx")

    def test_all_modes_work(self):
        for mode in ("hybrid", "vector", "keyword"):
            self.assertEqual(self.top("sourdough starter", mode=mode), "bread.txt")

    def test_one_hit_per_file_and_limit(self):
        hits = self.searcher.search("the", limit=2)
        self.assertLessEqual(len(hits), 2)
        self.assertEqual(len({h.path for h in hits}), len(hits))

    def test_type_filter(self):
        hits = self.searcher.search("the income month sourdough heating", ftype="docx")
        self.assertEqual([Path(h.path).name for h in hits], ["budget.docx"])
        self.assertEqual(self.searcher.search("sourdough", ftype="pdf"), [])

    def test_path_and_since_filters(self):
        hits = self.searcher.search("bicycle chain", path_contains="notes")
        self.assertEqual(Path(hits[0].path).name, "bike.md")
        self.assertEqual(self.searcher.search("bicycle", since=time.time() + 3600), [])

    def test_empty_and_symbol_queries_are_safe(self):
        self.assertEqual(self.searcher.search("   "), [])
        self.searcher.search('AND OR NOT ( ) " * : ^ NEAR/2', mode="keyword")

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            self.searcher.search("x", mode="magic")

    def test_scores_are_relative_and_bounded(self):
        hits = self.searcher.search("sourdough starter")
        self.assertTrue(all(0 < h.score <= 1.0 for h in hits))
        self.assertEqual(hits[0].score, max(h.score for h in hits))


class HelperTests(unittest.TestCase):
    def test_rrf_prefers_items_ranked_high_in_both(self):
        scores = rrf([[1, 2, 3], [3, 1, 4]])
        self.assertGreater(scores[1], scores[2])
        self.assertGreater(scores[1], scores[4])

    def test_fts_query_quotes_terms(self):
        self.assertEqual(fts_query('Hello, "World" AND'), '"hello" OR "world" OR "and"')

    def test_snippet_centres_on_match(self):
        text = "filler " * 100 + "needle " + "filler " * 100
        snippet = make_snippet(text, ["needle"], width=120)
        self.assertIn("needle", snippet)
        self.assertTrue(snippet.startswith("…") and snippet.endswith("…"))


class NoNetworkTests(Base):
    def test_indexing_and_search_make_no_network_calls(self):
        self.populate()

        def refuse(*args, **kwargs):
            raise AssertionError("network access attempted")

        with mock.patch.object(socket.socket, "connect", refuse), mock.patch.object(
            socket, "create_connection", refuse
        ):
            self.index()
            hits = Searcher(self.store, self.embedder).search("heating repair")
        self.assertEqual(Path(hits[0].path).name, "lease.md")


if __name__ == "__main__":
    unittest.main()
