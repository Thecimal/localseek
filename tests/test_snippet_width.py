import inspect
import random
import tempfile
import unittest
from pathlib import Path

from helpers import write
from localseek.config import Settings
from localseek.embedder import HashEmbedder
from localseek.indexer import index_paths
from localseek.search import SNIPPET_WIDTH, Searcher, make_snippet, query_terms
from localseek.store import Store

MARKER = "the secret handshake is a double knock"


class SnippetWidthTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name) / "docs"
        filler = " ".join(f"w{i:02d}" for i in range(75))
        # One chunk. The marker starts 310 characters after the only query word: past a 280-character window, inside
        # a 420-character one.
        self.text = f"zebrafish {filler} {MARKER.capitalize()}. " + " ".join(["tail"] * 40)
        write(root / "long.md", self.text)
        self.store = Store(Path(self._tmp.name) / "i.db")
        index_paths(self.store, HashEmbedder(), [root], Settings(model="hash"))
        self.searcher = Searcher(self.store, HashEmbedder())

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def test_search_results_show_the_wider_window(self):
        snippet = self.searcher.search("zebrafish", mode="keyword")[0].snippet
        self.assertIn(MARKER, snippet.lower())
        self.assertLessEqual(len(snippet), SNIPPET_WIDTH + 2)  # the window plus two ellipses

    def test_the_fixture_really_needs_the_wider_window(self):
        old = make_snippet(" ".join(self.text.split()), query_terms("zebrafish"))
        self.assertNotIn(MARKER, old.lower())

    def test_make_snippet_keeps_its_default_so_the_experiment_stays_reproducible(self):
        # eval/snippet_experiment.py defines S0 as make_snippet(text, terms) and S4 as the same with width=420. Changing
        # this default would make the two identical and the recorded result impossible to reproduce.
        self.assertEqual(inspect.signature(make_snippet).parameters["width"].default, 280)
        self.assertEqual(SNIPPET_WIDTH, 420)

    def test_the_wider_window_never_hides_what_the_narrower_one_showed(self):
        # This is why no question can lose its answer when the width grows. Both windows are cut from the same start
        # and the wider one only continues further (or shows the whole text), so whatever the narrow snippet contains
        # is inside the wide one. A change to how the window is anchored would break this on purpose: it would need
        # its own experiment, because it could make a result lose its answer.
        rng = random.Random(20261012)
        vocabulary = [f"w{i}" for i in range(40)]
        for _ in range(500):
            words = [rng.choice(vocabulary) for _ in range(rng.randint(1, 160))]
            text = "  ".join(words)  # irregular spacing exercises the whitespace normalisation
            terms = rng.sample(vocabulary, rng.randint(0, 3)) + ["absent"]
            narrow = make_snippet(text, terms).strip("…")
            wide = make_snippet(text, terms, width=SNIPPET_WIDTH).strip("…")
            self.assertIn(narrow, wide, (text, terms))


if __name__ == "__main__":
    unittest.main()
