"""Opt-in fusion options on Searcher, and the evaluator's --tuning flag.

The defaults must be the shipped behaviour exactly; the options exist only for evaluation experiments.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from helpers import write
from localseek.config import Settings
from localseek.embedder import HashEmbedder
from localseek.indexer import index_paths
from localseek.search import STOPWORDS, Searcher, fts_query, query_terms, rrf
from localseek.store import Store

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))

from retrieval_eval import (  # noqa: E402
    TuningError,
    evaluate,
    format_tuning,
    load_dataset,
    parse_tuning,
)

BENCH = REPO / "eval" / "benchmark"
MODES = ("hybrid", "vector", "keyword")


class RrfTests(unittest.TestCase):
    def test_default_weights_are_the_classic_formula(self):
        scores = rrf([[1, 2], [2, 1]])
        self.assertAlmostEqual(scores[1], 1 / 61 + 1 / 62)
        self.assertEqual(rrf([[1, 2], [2, 1]]), rrf([[1, 2], [2, 1]], 60, [1.0, 1.0]))

    def test_weights_scale_each_arm(self):
        scores = rrf([[1, 2], [2, 1]], 60, [2.0, 1.0])
        self.assertAlmostEqual(scores[1], 2 / 61 + 1 / 62)
        self.assertAlmostEqual(scores[2], 2 / 62 + 1 / 61)

    def test_smaller_k_rewards_top_ranks_more(self):
        self.assertGreater(rrf([[1]], 1)[1] / rrf([[2, 1]], 1)[1], rrf([[1]], 60)[1] / rrf([[2, 1]], 60)[1])


class StopwordTests(unittest.TestCase):
    def test_default_query_is_unchanged(self):
        expected = '"how" OR "do" OR "i" OR "quiet" OR "a" OR "noisy" OR "chain"'
        self.assertEqual(fts_query("how do I quiet a noisy chain"), expected)

    def test_function_words_are_dropped_on_request(self):
        filtered = fts_query("how do I quiet a noisy chain", drop_stopwords=True)
        self.assertEqual(filtered, '"quiet" OR "noisy" OR "chain"')

    def test_a_query_of_only_function_words_is_not_emptied(self):
        self.assertEqual(fts_query("what is it", drop_stopwords=True), '"what" OR "is" OR "it"')

    def test_snippet_terms_are_unaffected(self):
        self.assertEqual(query_terms("how do I quiet"), ["how", "do", "i", "quiet"])

    def test_the_list_is_lowercase_words(self):
        self.assertTrue(all(w == w.lower() and w.isalpha() for w in STOPWORDS))


class SearcherFusionTests(unittest.TestCase):
    """The arms are stubbed so the fused order depends only on the fusion settings."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        for name in "abcdef":
            write(root / "docs" / f"{name}.md", f"document {name} text")
        self.store = Store(root / "i.db")
        self.embedder = HashEmbedder()
        index_paths(self.store, self.embedder, [root / "docs"], Settings(model="hash"))
        ids = sorted(int(i) for i in self.store.load_matrix()[0])
        self.a, self.b, self.c, self.d, self.e, self.f = ids
        self.names = {chunk_id: Path(row["path"]).stem for chunk_id, row in self.store.fetch_chunks(ids).items()}

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def order(self, vector, keyword, **options):
        class Stub(Searcher):
            def _vector_rank(self_, *args):
                return vector

            def _keyword_rank(self_, *args):
                return keyword

        hits = Stub(self.store, self.embedder, **options).search("anything", limit=10)
        return [Path(h.path).stem for h in hits]

    def test_defaults_break_a_perfect_tie_by_chunk_id(self):
        # a is first for the vector arm and second for the keyword arm, b the reverse: equal fused scores.
        self.assertEqual(self.order([self.a, self.b], [self.b, self.a]), [self.names[self.a], self.names[self.b]])

    def test_vector_weight_favours_the_vector_arm(self):
        order = self.order([self.b, self.a], [self.a, self.b], vector_weight=2)
        self.assertEqual(order, [self.names[self.b], self.names[self.a]])

    def test_keyword_weight_favours_the_keyword_arm(self):
        order = self.order([self.b, self.a], [self.a, self.b], keyword_weight=2)
        self.assertEqual(order, [self.names[self.a], self.names[self.b]])

    def test_keyword_limit_keeps_only_the_first_keyword_hits(self):
        vector, keyword = [self.a, self.b], [self.b, self.a]
        self.assertEqual(self.order(vector, keyword), [self.names[self.a], self.names[self.b]])  # tie, id order
        self.assertEqual(self.order(vector, keyword, keyword_limit=1), [self.names[self.b], self.names[self.a]])
        self.assertEqual(self.order(vector, keyword, keyword_limit=50), self.order(vector, keyword))

    def test_small_k_lets_single_arm_leaders_beat_documents_that_are_merely_everywhere(self):
        # f leads the vector arm only. a is third for the vector arm and fourth for the keyword arm; c leads the
        # keyword arm only. At k=60, a scores 1/63 + 1/64 (about 0.031) against 1/61 (0.016) for the leaders.
        # At k=1, a scores 1/4 + 1/5 = 0.45 against 0.5 for each leader, and c wins the tie on chunk id.
        vector = [self.f, self.b, self.a]
        keyword = [self.c, self.d, self.e, self.a]
        self.assertEqual(self.order(vector, keyword, rrf_k=60)[0], self.names[self.a])
        self.assertEqual(self.order(vector, keyword, rrf_k=1)[0], self.names[self.c])

    def test_top_score_is_normalised_to_one_for_any_weights(self):
        class Stub(Searcher):
            def _vector_rank(self_, *args):
                return [self.a, self.b]

            def _keyword_rank(self_, *args):
                return [self.a, self.b]

        for options in ({}, {"vector_weight": 3.0}, {"keyword_weight": 0.5, "rrf_k": 10}):
            hits = Stub(self.store, self.embedder, **options).search("x")
            self.assertEqual(hits[0].score, 1.0, options)

    def test_invalid_options_are_rejected(self):
        for options in ({"rrf_k": 0}, {"vector_weight": 0}, {"keyword_weight": -1}, {"keyword_limit": 0},
                        {"vector_weight": float("inf")}):  # fmt: skip
            with self.subTest(options=options), self.assertRaises(ValueError):
                Searcher(self.store, self.embedder, **options)


class StopwordSearchTests(unittest.TestCase):
    def test_searcher_applies_the_option_to_the_real_keyword_arm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root / "docs" / "x.md", "the cat sat on the mat")
            write(root / "docs" / "y.md", "a quiet chain")
            write(root / "docs" / "z.md", "the and of to")
            store = Store(root / "i.db")
            embedder = HashEmbedder()
            index_paths(store, embedder, [root / "docs"], Settings(model="hash"))
            query = "the quiet chain"
            everything = Searcher(store, embedder).search(query, mode="keyword")
            filtered = Searcher(store, embedder, drop_stopwords=True).search(query, mode="keyword")
            store.close()
        self.assertEqual(sorted(Path(h.path).stem for h in everything), ["x", "y", "z"])  # "the" matches everything
        self.assertEqual([Path(h.path).stem for h in filtered], ["y"])


class DefaultsAreUnchangedTests(unittest.TestCase):
    def test_default_search_reproduces_the_committed_hash_baseline(self):
        corpus = (BENCH / "corpus").resolve()
        queries = load_dataset(BENCH / "queries.json", corpus)
        baseline = json.loads((BENCH / "baselines" / "hash.json").read_text(encoding="utf-8"))["results"]
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "b.db")
            embedder = HashEmbedder()
            index_paths(store, embedder, [corpus], Settings(model="hash"))
            explicit = Searcher(store, embedder, 60, 1.0, 1.0, None, False)
            for searcher in (Searcher(store, embedder), explicit):
                for mode in MODES:
                    got = evaluate(searcher, queries, corpus, mode)
                    for k in (1, 5, 10):
                        self.assertAlmostEqual(got.recall[k], baseline[mode][f"recall@{k}"], places=12, msg=mode)
                    self.assertAlmostEqual(got.mrr, baseline[mode]["mrr"], places=12, msg=mode)
            store.close()


class ParseTuningTests(unittest.TestCase):
    def test_empty_means_defaults(self):
        for spec in (None, "", "  "):
            self.assertEqual(parse_tuning(spec), {})

    def test_valid_specifications(self):
        self.assertEqual(
            parse_tuning("rrf_k=10, vector_weight=2, keyword_weight=0.5, keyword_limit=20, drop_stopwords=TRUE"),
            {"rrf_k": 10, "vector_weight": 2.0, "keyword_weight": 0.5, "keyword_limit": 20, "drop_stopwords": True},
        )
        self.assertEqual(parse_tuning("drop_stopwords=false"), {"drop_stopwords": False})

    def test_every_problem_is_reported(self):
        with self.assertRaises(TuningError) as ctx:
            parse_tuning("rrf_k=0,bogus=1,keyword_limit=x,vector_weight,rrf_k=5,drop_stopwords=maybe,keyword_weight=inf")
        text = "\n".join(ctx.exception.problems)
        expected = [
            "rrf_k: '0' is not valid", "unknown option 'bogus'", "keyword_limit: 'x'", "expected name=value",
            "rrf_k is given more than once", "drop_stopwords: 'maybe'", "keyword_weight: 'inf'",
        ]  # fmt: skip
        for fragment in expected:
            self.assertIn(fragment, text)

    def test_formatting_round_trips(self):
        tuning = parse_tuning("rrf_k=10,drop_stopwords=true,vector_weight=2")
        self.assertEqual(format_tuning(tuning), "rrf_k=10, drop_stopwords=true, vector_weight=2.0")
        self.assertEqual(parse_tuning(format_tuning(tuning).replace(" ", "")), tuning)


class CliTuningTests(unittest.TestCase):
    def run_cli(self, *extra):
        command = [
            sys.executable, str(REPO / "eval" / "run_eval.py"),
            "--corpus", str(BENCH / "corpus"), "--queries", str(BENCH / "queries-v2.json"),
            "--model", "hash", "--split", "dev", *extra,
        ]  # fmt: skip
        return subprocess.run(command, capture_output=True, text=True, cwd=REPO)

    def test_tuning_is_in_the_header_and_the_json_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run.json"
            run = self.run_cli("--tuning", "rrf_k=10,drop_stopwords=true", "--json", str(out))
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn("tuning:   rrf_k=10, drop_stopwords=true", run.stdout)
            self.assertEqual(json.loads(out.read_text(encoding="utf-8"))["environment"]["tuning"],
                             {"rrf_k": 10, "drop_stopwords": True})  # fmt: skip

    def test_default_run_has_no_tuning_line_and_an_empty_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run.json"
            run = self.run_cli("--json", str(out))
            self.assertNotIn("tuning:", run.stdout)
            self.assertEqual(json.loads(out.read_text(encoding="utf-8"))["environment"]["tuning"], {})

    def test_keyword_settings_change_hybrid_and_keyword_but_never_vector(self):
        def results(*extra):
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "run.json"
                self.assertEqual(self.run_cli("--json", str(out), *extra).returncode, 0)
                return json.loads(out.read_text(encoding="utf-8"))["results"]

        default = results()
        tuned = results("--tuning", "keyword_limit=1,keyword_weight=3,drop_stopwords=true")
        strip = lambda r: {k: v for k, v in r.items() if k != "latency_ms"}  # noqa: E731
        self.assertEqual(strip(default["vector"]), strip(tuned["vector"]))
        self.assertNotEqual(strip(default["hybrid"]), strip(tuned["hybrid"]))
        self.assertNotEqual(strip(default["keyword"]), strip(tuned["keyword"]))

    def test_invalid_tuning_exits_2_before_indexing(self):
        run = self.run_cli("--tuning", "rrf_k=0,bogus=1")
        self.assertEqual(run.returncode, 2)
        self.assertIn("invalid --tuning", run.stderr)
        self.assertNotIn("Indexed", run.stdout)
        self.assertNotIn("Traceback", run.stderr)


if __name__ == "__main__":
    unittest.main()
