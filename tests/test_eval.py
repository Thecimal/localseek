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
from localseek.search import Searcher
from localseek.store import Store

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))

from retrieval_eval import (  # noqa: E402
    DatasetError,
    Failure,
    ModeResult,
    Query,
    ThresholdError,
    check_indexed,
    check_thresholds,
    corpus_fingerprint,
    count_checks,
    dataset_fingerprint,
    environment_info,
    evaluate,
    evaluate_by_category,
    format_miss,
    load_dataset,
    load_thresholds,
    misses,
    relative_path,
    score_query,
    trace_queries,
)

MODES = ("hybrid", "vector", "keyword")


class EvalBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.root = (self.dir / "corpus").resolve()
        write(self.root / "recipes" / "sourdough.md", "Feed the sourdough starter twice a day.")
        write(self.root / "recipes" / "notes.md", "Sourdough starter feeding schedule and flour ratios.")
        write(self.root / "work" / "notes.md", "Quarterly tax filing deadlines and invoice reminders.")
        write(self.root / "notes.md", "Bicycle chain lubrication and tire pressure.")

    def tearDown(self):
        self._tmp.cleanup()

    def dataset(self, content) -> Path:
        path = self.dir / "queries.json"
        path.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
        return path

    def problems(self, content) -> list[str]:
        with self.assertRaises(DatasetError) as ctx:
            load_dataset(self.dataset(content), self.root)
        return ctx.exception.problems


class DatasetValidationTests(EvalBase):
    def assertRejected(self, content, fragment):
        joined = "\n".join(self.problems(content))
        self.assertIn(fragment, joined)

    def test_valid_dataset_with_multiple_relevant(self):
        queries = load_dataset(
            self.dataset([{"query": " sourdough ", "relevant": ["recipes/sourdough.md", "recipes/notes.md"]}]),
            self.root,
        )
        self.assertEqual(queries, [Query("sourdough", ("recipes/sourdough.md", "recipes/notes.md"))])

    def test_malformed_json(self):
        self.assertRejected('[{"query": "x",', "not valid JSON")

    def test_missing_dataset_file(self):
        with self.assertRaises(DatasetError):
            load_dataset(self.dir / "nope.json", self.root)

    def test_top_level_must_be_non_empty_list(self):
        self.assertRejected({"query": "x"}, "JSON list")
        self.assertRejected([], "no queries")

    def test_item_must_be_object(self):
        self.assertRejected(["just a string"], "must be an object")

    def test_query_problems(self):
        good = ["notes.md"]
        self.assertRejected([{"relevant": good}], "missing 'query'")
        self.assertRejected([{"query": "", "relevant": good}], "non-empty string")
        self.assertRejected([{"query": "   ", "relevant": good}], "non-empty string")
        self.assertRejected([{"query": 7, "relevant": good}], "non-empty string")

    def test_relevant_problems(self):
        self.assertRejected([{"query": "q"}], "missing 'relevant'")
        self.assertRejected([{"query": "q", "relevant": []}], "must not be empty")
        self.assertRejected([{"query": "q", "relevant": "notes.md"}], "must be a list")
        self.assertRejected([{"query": "q", "relevant": [""]}], "non-empty string")
        self.assertRejected([{"query": "q", "relevant": [3]}], "non-empty string")

    def test_duplicate_relevant_paths(self):
        self.assertRejected([{"query": "q", "relevant": ["notes.md", "notes.md"]}], "more than once")

    def test_paths_outside_or_malformed(self):
        write(self.dir / "outside.md", "outside the corpus")
        cases = {
            "../outside.md": "no '..'",
            "recipes/../../outside.md": "no '..'",
            str(self.dir / "outside.md"): "not absolute",
            "/notes.md": "not absolute",
            "./notes.md": "normalized form",
            "recipes//notes.md": "normalized form",
            "recipes\\notes.md": "forward slashes",
        }
        for value, fragment in cases.items():
            with self.subTest(value=value):
                self.assertRejected([{"query": "q", "relevant": [value]}], fragment)

    def test_nonexistent_and_non_file_paths(self):
        self.assertRejected([{"query": "q", "relevant": ["recipes/missing.md"]}], "does not exist")
        self.assertRejected([{"query": "q", "relevant": ["recipes"]}], "not a file")

    def test_bare_filename_is_not_accepted_when_it_does_not_exist_at_that_path(self):
        # 'sourdough.md' exists only under recipes/, so the bare name is not a valid identity.
        self.assertRejected([{"query": "q", "relevant": ["sourdough.md"]}], "does not exist")

    def test_old_expected_format_gets_migration_hint(self):
        self.assertRejected([{"query": "q", "expected": "notes.md"}], "no longer supported")

    def test_all_problems_are_reported_together(self):
        found = self.problems([{"query": ""}, {"query": "ok", "relevant": []}, "bad"])
        self.assertGreaterEqual(len(found), 4)


class CategoryTests(EvalBase):
    def test_category_is_optional_and_loaded(self):
        queries = load_dataset(
            self.dataset([
                {"query": "a", "relevant": ["notes.md"], "category": "exact"},
                {"query": "b", "relevant": ["notes.md"]},
                {"query": "c", "relevant": ["notes.md"], "evidence": {"notes.md": "extra fields are ignored"}},
            ]),
            self.root,
        )  # fmt: skip
        self.assertEqual([q.category for q in queries], ["exact", None, None])

    def test_invalid_category_is_rejected(self):
        for bad in ("", "  ", 3, None):
            with self.subTest(category=bad):
                found = "\n".join(self.problems([{"query": "q", "relevant": ["notes.md"], "category": bad}]))
                self.assertIn("'category' must be a non-empty string", found)

    def test_results_are_grouped_per_category(self):
        store = Store(self.dir / "i.db")
        embedder = HashEmbedder()
        index_paths(store, embedder, [self.root], Settings(model="hash"))
        queries = [
            Query("tax invoice deadlines", ("work/notes.md",), "hit"),
            Query("bicycle chain tire", ("notes.md",), "hit"),
            Query("tax invoice deadlines", ("notes.md",), "miss"),
            Query("bicycle chain tire", ("work/notes.md",)),
        ]
        grouped = evaluate_by_category(Searcher(store, embedder), queries, self.root, "keyword")
        self.assertEqual(list(grouped), ["hit", "miss", "uncategorized"])
        self.assertEqual(grouped["hit"].recall[1], 1.0)
        self.assertEqual(grouped["miss"].recall[10], 0.0)
        self.assertEqual(grouped["uncategorized"].mrr, 0.0)
        store.close()


class FingerprintTests(EvalBase):
    def copy_corpus(self, name, transform=lambda path, data: data):
        target = self.dir / name
        for path in self.root.rglob("*"):
            if path.is_file():
                out = target / path.relative_to(self.root)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(transform(path, path.read_bytes()))
        return target.resolve()

    def test_text_line_endings_do_not_change_the_fingerprint(self):
        crlf = self.copy_corpus("crlf", lambda path, data: data.replace(b"\n", b"\r\n"))
        self.assertEqual(corpus_fingerprint(self.root), corpus_fingerprint(crlf))
        lf, windows = self.dir / "a.json", self.dir / "b.json"
        lf.write_bytes(b'[\n  {"query": "x"}\n]\n')
        windows.write_bytes(b'[\r\n  {"query": "x"}\r\n]\r\n')
        self.assertEqual(dataset_fingerprint(lf), dataset_fingerprint(windows))

    def test_content_name_and_new_files_change_the_fingerprint(self):
        base = corpus_fingerprint(self.root)
        edited = self.copy_corpus("edited", lambda p, d: d.replace(b"Feed", b"Skip") if p.name == "sourdough.md" else d)
        self.assertNotEqual(base, corpus_fingerprint(edited))
        renamed = self.copy_corpus("renamed")
        (renamed / "work" / "notes.md").rename(renamed / "work" / "memo.md")
        self.assertNotEqual(base, corpus_fingerprint(renamed))
        added = self.copy_corpus("added")
        write(added / "extra.md", "one more file")
        self.assertNotEqual(base, corpus_fingerprint(added))

    def test_binary_files_are_hashed_as_is(self):
        one, two = self.copy_corpus("one"), self.copy_corpus("two")
        (one / "doc.pdf").write_bytes(b"%PDF line\nbreak")
        (two / "doc.pdf").write_bytes(b"%PDF line\r\nbreak")
        self.assertNotEqual(corpus_fingerprint(one), corpus_fingerprint(two))

    def test_environment_record_is_complete_and_has_no_absolute_corpus_path(self):
        path = self.dataset([{"query": "q", "relevant": ["notes.md"]}])
        queries = load_dataset(path, self.root)
        info = environment_info("hash", Settings(model="hash"), path, "corpus", self.root, queries)
        self.assertEqual(info["model"], "hash")
        self.assertEqual((info["chunk_words"], info["overlap_words"]), (220, 30))
        self.assertEqual(info["corpus"], {"path": "corpus", "files": 4, "sha256": corpus_fingerprint(self.root)})
        self.assertEqual(info["dataset"]["queries"], 1)
        self.assertEqual({"python", "localseek", "fastembed", "numpy"}, set(info["versions"]))
        self.assertNotIn(str(self.dir), json.dumps(info["corpus"]))  # the corpus path is recorded as given


class TraceTests(EvalBase):
    def setUp(self):
        super().setUp()
        self.store = Store(self.dir / "i.db")
        self.embedder = HashEmbedder()
        index_paths(self.store, self.embedder, [self.root], Settings(model="hash"))
        self.searcher = Searcher(self.store, self.embedder)

    def tearDown(self):
        self.store.close()
        super().tearDown()

    def trace(self, text, relevant):
        return trace_queries(self.searcher, [Query(text, tuple(relevant), "demo")], self.root, "keyword")[0]

    def test_top_hit_is_not_a_miss(self):
        trace = self.trace("tax invoice deadlines", ["work/notes.md"])
        self.assertEqual(trace.first_rank, 1)
        self.assertEqual(misses([trace]), [])

    def test_lower_rank_is_reported_with_its_position(self):
        trace = self.trace("sourdough starter feeding schedule flour ratios", ["recipes/sourdough.md"])
        self.assertEqual(trace.first_rank, 2)
        self.assertEqual(trace.ranked[0], "recipes/notes.md")
        self.assertEqual(misses([trace]), [trace])  # found, but not first, still counts as a miss
        lines = "\n".join(format_miss(trace))
        self.assertIn("rank 2", lines)
        self.assertIn("top hit: recipes/notes.md", lines)
        self.assertIn("wanted: recipes/sourdough.md", lines)

    def test_absent_relevant_document_is_reported_as_not_in_top_k(self):
        trace = self.trace("tax invoice deadlines", ["recipes/sourdough.md"])
        self.assertIsNone(trace.first_rank)
        self.assertEqual(misses([trace]), [trace])
        self.assertIn("not in top 10", "\n".join(format_miss(trace)))

    def test_query_with_no_results_is_handled(self):
        trace = self.trace("zzzzqqqq", ["notes.md"])
        self.assertEqual(trace.ranked, ())
        self.assertIn("(no results)", "\n".join(format_miss(trace)))

    def test_long_queries_are_truncated_in_the_report(self):
        trace = self.trace("tax " * 80, ["work/notes.md"])
        self.assertLess(len(format_miss(trace)[0]), 140)


class IndexedCheckTests(EvalBase):
    def test_unindexed_relevant_document_is_rejected(self):
        write(self.root / "data.xyz", "unsupported suffix, never indexed")
        store = Store(self.dir / "i.db")
        index_paths(store, HashEmbedder(), [self.root], Settings(model="hash"))
        queries = [Query("q", ("data.xyz",))]
        with self.assertRaises(DatasetError) as ctx:
            check_indexed(store.file_records(), self.root, queries)
        self.assertIn("not indexed", ctx.exception.problems[0])
        check_indexed(store.file_records(), self.root, [Query("q", ("notes.md",))])
        store.close()


class ScoringTests(unittest.TestCase):
    def test_relative_path_normalizes_hits(self):
        root = Path("/data/corpus")
        self.assertEqual(relative_path(root / "recipes" / "sourdough.md", root), "recipes/sourdough.md")
        self.assertEqual(relative_path("/elsewhere/x.md", root), "/elsewhere/x.md")

    def test_recall_is_fraction_of_relevant_set(self):
        score = score_query(["a", "x", "b"], ["a", "b"], ks=(1, 2, 3))
        self.assertEqual(score.recall, {1: 0.5, 2: 0.5, 3: 1.0})
        self.assertEqual(score.reciprocal_rank, 1.0)

    def test_reciprocal_rank_uses_first_relevant(self):
        self.assertEqual(score_query(["x", "y", "b", "a"], ["a", "b"]).reciprocal_rank, 1 / 3)

    def test_no_relevant_result_scores_zero(self):
        score = score_query(["x", "y"], ["a"])
        self.assertEqual(score.reciprocal_rank, 0.0)
        self.assertTrue(all(v == 0 for v in score.recall.values()))
        self.assertEqual(score_query([], ["a"]).reciprocal_rank, 0.0)

    def test_empty_relevant_set_is_an_error_not_a_division(self):
        with self.assertRaises(ValueError):
            score_query(["a"], [])


class DuplicateFilenameTests(EvalBase):
    def setUp(self):
        super().setUp()
        self.store = Store(self.dir / "i.db")
        self.embedder = HashEmbedder()
        index_paths(self.store, self.embedder, [self.root], Settings(model="hash"))
        self.searcher = Searcher(self.store, self.embedder)

    def tearDown(self):
        self.store.close()
        super().tearDown()

    def run_mode(self, relevant, mode="keyword"):
        queries = [Query("sourdough starter feeding schedule", (relevant,))]
        return evaluate(self.searcher, queries, self.root, mode)

    def test_same_filename_in_another_folder_is_not_a_false_positive(self):
        right = self.run_mode("recipes/notes.md")
        self.assertEqual((right.recall[1], right.mrr), (1.0, 1.0))
        # work/notes.md shares the file name but is a different document that never matches.
        wrong = self.run_mode("work/notes.md")
        self.assertEqual((wrong.recall[10], wrong.mrr), (0.0, 0.0))
        wrong_root = self.run_mode("notes.md")
        self.assertEqual((wrong_root.recall[10], wrong_root.mrr), (0.0, 0.0))

    def test_multiple_relevant_documents_partial_credit(self):
        # Only work/notes.md matches these terms, so exactly half of the relevant set is found.
        queries = [Query("tax invoice deadlines", ("work/notes.md", "recipes/sourdough.md"))]
        result = evaluate(self.searcher, queries, self.root, "keyword")
        self.assertEqual(result.recall[1], 0.5)
        self.assertEqual(result.recall[10], 0.5)
        self.assertEqual(result.mrr, 1.0)

    def test_evaluate_rejects_empty_query_list(self):
        with self.assertRaises(ValueError):
            evaluate(self.searcher, [], self.root, "keyword")


def result(r1=1.0, r5=1.0, r10=1.0, mrr=1.0):
    return ModeResult({1: r1, 5: r5, 10: r10}, mrr, 0.0)


class ThresholdLoadingTests(EvalBase):
    def load(self, content):
        path = self.dir / "thresholds.json"
        path.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
        return load_thresholds(path, MODES)

    def rejected(self, content):
        with self.assertRaises(ThresholdError) as ctx:
            self.load(content)
        return "\n".join(ctx.exception.problems)

    def test_valid_thresholds(self):
        loaded = self.load({"hybrid": {"recall@1": 0.8, "mrr": 1}, "keyword": {"recall@10": 0}})
        self.assertEqual(loaded, {"hybrid": {"recall@1": 0.8, "mrr": 1.0}, "keyword": {"recall@10": 0.0}})
        self.assertEqual(count_checks(loaded), 3)

    def test_unusable_files(self):
        self.assertIn("not valid JSON", self.rejected("{nope"))
        self.assertIn("top level", self.rejected(["hybrid"]))
        self.assertIn("checks nothing", self.rejected({}))
        with self.assertRaises(ThresholdError):
            load_thresholds(self.dir / "missing.json", MODES)

    def test_unknown_mode_and_metric_are_rejected_not_ignored(self):
        # A typo must not silently disable a check.
        self.assertIn("unknown mode 'hybird'", self.rejected({"hybird": {"mrr": 0.5}}))
        self.assertIn("unknown metric 'recall@3'", self.rejected({"hybrid": {"recall@3": 0.5}}))
        self.assertIn("unknown metric 'MRR'", self.rejected({"hybrid": {"MRR": 0.5}}))

    def test_bad_values(self):
        for value in ("0.8", None, True, [0.8], 1.5, -0.1, float("nan"), float("inf")):
            with self.subTest(value=value):
                self.assertIn("threshold must be", self.rejected({"hybrid": {"mrr": value}}))

    def test_mode_must_map_to_metrics(self):
        self.assertIn("non-empty object", self.rejected({"hybrid": {}}))
        self.assertIn("non-empty object", self.rejected({"hybrid": 0.5}))

    def test_all_problems_reported_together(self):
        text = self.rejected({"bogus": {"mrr": 1}, "hybrid": {"recall@3": 1, "mrr": 2}})
        self.assertEqual(len(text.splitlines()), 3)


class ThresholdCheckTests(unittest.TestCase):
    def test_passes_when_all_met(self):
        results = {"hybrid": result(0.9, mrr=0.95)}
        self.assertEqual(check_thresholds(results, {"hybrid": {"recall@1": 0.85, "mrr": 0.9}}), [])

    def test_fails_below_threshold_and_names_mode_and_metric(self):
        failures = check_thresholds({"hybrid": result(0.71)}, {"hybrid": {"recall@1": 0.85}})
        self.assertEqual(failures, [Failure("hybrid", "recall@1", 0.71, 0.85)])
        self.assertEqual(str(failures[0]), "hybrid recall@1: 0.710 < required 0.850")

    def test_exactly_at_threshold_passes(self):
        self.assertEqual(check_thresholds({"hybrid": result(0.85)}, {"hybrid": {"recall@1": 0.85}}), [])

    def test_floating_point_noise_does_not_fail_a_met_threshold(self):
        noisy = 0.1 + 0.2  # 0.30000000000000004 vs required 0.3, and the reverse below
        self.assertEqual(check_thresholds({"hybrid": result(0.3)}, {"hybrid": {"recall@1": noisy}}), [])
        self.assertEqual(check_thresholds({"hybrid": result(noisy)}, {"hybrid": {"recall@1": 0.3}}), [])

    def test_just_below_threshold_fails(self):
        self.assertEqual(len(check_thresholds({"hybrid": result(0.849)}, {"hybrid": {"recall@1": 0.85}})), 1)

    def test_every_failure_is_reported_in_file_order(self):
        results = {"hybrid": result(0.5, mrr=0.5), "vector": result(1.0), "keyword": result(0.2)}
        thresholds = {"keyword": {"recall@1": 0.9}, "vector": {"mrr": 0.9}, "hybrid": {"mrr": 0.9, "recall@1": 0.9}}
        failures = check_thresholds(results, thresholds)
        self.assertEqual([(f.mode, f.metric) for f in failures],
                         [("keyword", "recall@1"), ("hybrid", "mrr"), ("hybrid", "recall@1")])  # fmt: skip

    def test_unlisted_modes_and_metrics_are_not_checked(self):
        results = {"hybrid": result(0.0, mrr=0.0), "vector": result(0.0)}
        self.assertEqual(check_thresholds(results, {"vector": {"recall@10": 0.5}}), [])
        self.assertEqual(len(check_thresholds(results, {"vector": {"recall@1": 0.5}})), 1)

    def test_threshold_for_unevaluated_mode_is_an_error(self):
        with self.assertRaises(ValueError):
            check_thresholds({"hybrid": result()}, {"keyword": {"mrr": 0.5}})


class GateAcceptanceTests(EvalBase):
    """A clean run passes its own scores as thresholds; a deliberately broken ranker must fail them."""

    class Reversed:
        def __init__(self, inner):
            self.inner = inner

        def search(self, *args, **kwargs):
            return list(reversed(self.inner.search(*args, **kwargs)))

    def test_ranking_regression_is_detected(self):
        store = Store(self.dir / "i.db")
        embedder = HashEmbedder()
        index_paths(store, embedder, [self.root], Settings(model="hash"))
        healthy = Searcher(store, embedder)
        queries = [
            Query("flour ratios", ("recipes/notes.md",)),
            Query("feed twice day", ("recipes/sourdough.md",)),
            Query("tax invoice deadlines", ("work/notes.md",)),
            Query("bicycle chain tire", ("notes.md",)),
        ]
        good = {mode: evaluate(healthy, queries, self.root, mode) for mode in MODES}
        self.assertEqual(good["vector"].recall[1], 1.0)  # precondition: the baseline is actually good
        required = {mode: {"recall@1": good[mode].recall[1], "mrr": good[mode].mrr} for mode in MODES}
        self.assertEqual(check_thresholds(good, required), [])

        broken = {mode: evaluate(self.Reversed(healthy), queries, self.root, mode) for mode in MODES}
        failed = {(f.mode, f.metric) for f in check_thresholds(broken, required)}
        self.assertIn(("vector", "recall@1"), failed)
        self.assertIn(("hybrid", "mrr"), failed)
        store.close()


class CliTests(EvalBase):
    def run_cli(self, queries_path, corpus=None, thresholds=None, extra=()):
        command = [sys.executable, str(REPO / "eval" / "run_eval.py"), "--corpus", str(corpus or self.root),
                   "--queries", str(queries_path), "--model", "hash"]  # fmt: skip
        command += list(extra)
        if thresholds is not None:
            path = self.dir / "thresholds.json"
            path.write_text(thresholds if isinstance(thresholds, str) else json.dumps(thresholds), encoding="utf-8")
            command += ["--thresholds", str(path)]
        return subprocess.run(command, capture_output=True, text=True, cwd=REPO)

    def shipped(self, thresholds):
        return self.run_cli(REPO / "eval" / "queries.json", REPO / "examples" / "corpus", thresholds)

    def test_invalid_dataset_exits_2_with_readable_error(self):
        for content in ([], [{"query": "q"}], "{broken", [{"query": "q", "expected": "notes.md"}]):
            with self.subTest(content=content):
                result = self.run_cli(self.dataset(content))
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("invalid evaluation dataset", result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_corpus_must_be_a_directory(self):
        result = self.run_cli(self.dataset([{"query": "q", "relevant": ["notes.md"]}]), corpus=self.root / "notes.md")
        self.assertEqual(result.returncode, 2)
        self.assertNotIn("Traceback", result.stderr)

    def test_valid_run_reports_all_modes(self):
        data = [{"query": "tax invoice deadlines", "relevant": ["work/notes.md"]}]
        result = self.run_cli(self.dataset(data))
        self.assertEqual(result.returncode, 0, result.stderr)
        for mode in ("hybrid", "vector", "keyword"):
            self.assertIn(mode, result.stdout)

    def test_benchmark_runs_end_to_end_with_category_report(self):
        bench = REPO / "eval" / "benchmark"
        result = self.run_cli(bench / "queries.json", bench / "corpus", extra=["--by-category"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Indexed 62 files", result.stdout)
        for category in ("semantic_paraphrase", "late_answer", "near_duplicate"):
            self.assertIn(category, result.stdout)

    def test_run_record_is_printed_first(self):
        data = [{"query": "tax invoice deadlines", "relevant": ["work/notes.md"]}]
        result = self.run_cli(self.dataset(data))
        self.assertEqual(result.returncode, 0, result.stderr)
        head = result.stdout.split("Indexed")[0]
        for label in ("model:    hash", "chunking: 220 words, 30 overlap", "dataset:", "corpus:", "versions:"):
            self.assertIn(label, head)
        self.assertRegex(head, r"sha256 [0-9a-f]{12}")

    def test_show_misses_lists_failures_per_mode(self):
        data = [
            {"query": "tax invoice deadlines", "relevant": ["work/notes.md"]},
            {"query": "tax invoice deadlines", "relevant": ["recipes/sourdough.md"], "category": "wrong"},
        ]
        result = self.run_cli(self.dataset(data), extra=["--show-misses"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("[keyword] 1 of 2", result.stdout)
        self.assertIn("not in top 10", result.stdout)
        self.assertIn("wanted: recipes/sourdough.md", result.stdout)

    def test_json_record_matches_the_printed_run(self):
        data = [{"query": "tax invoice deadlines", "relevant": ["work/notes.md"], "category": "exact"}]
        out = self.dir / "run.json"
        result = self.run_cli(self.dataset(data), thresholds={"keyword": {"recall@1": 1.0}}, extra=["--json", str(out)])
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(record["environment"]["model"], "hash")
        self.assertEqual(record["results"]["keyword"]["recall@1"], 1.0)
        self.assertEqual(set(record["by_category"]["keyword"]), {"exact"})
        self.assertEqual(record["gate"], {"checks": 1, "failures": []})
        self.assertEqual(set(record["misses"]), {"hybrid", "vector", "keyword"})
        self.assertIn(record["environment"]["corpus"]["sha256"][:12], result.stdout)

    def test_json_records_gate_failures_and_still_exits_1(self):
        data = [{"query": "tax invoice deadlines", "relevant": ["recipes/sourdough.md"]}]
        out = self.dir / "run.json"
        result = self.run_cli(self.dataset(data), thresholds={"keyword": {"mrr": 0.5}}, extra=["--json", str(out)])
        self.assertEqual(result.returncode, 1)
        record = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(len(record["gate"]["failures"]), 1)

    def test_shipped_dataset_is_valid(self):
        result = self.run_cli(REPO / "eval" / "queries.json", corpus=REPO / "examples" / "corpus")
        self.assertEqual(result.returncode, 0, result.stderr)

    # On the shipped corpus the hash embedder is deterministic: keyword recall@1 is 1.0, vector recall@1 is ~0.43.
    def test_without_thresholds_it_reports_only_and_says_so(self):
        result = self.shipped(None)
        self.assertEqual(result.returncode, 0)
        self.assertIn("nothing was checked", result.stdout)

    def test_met_thresholds_exit_0(self):
        result = self.shipped({"keyword": {"recall@1": 1.0}, "vector": {"recall@10": 0.9}})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Quality gate passed (2 checks)", result.stdout)

    def test_unmet_thresholds_exit_1_with_each_failure_listed(self):
        result = self.shipped({"vector": {"recall@1": 0.85, "mrr": 0.9}, "keyword": {"recall@1": 1.0}})
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("Evaluation failed:", result.stderr)
        self.assertIn("vector recall@1: 0.429 < required 0.850", result.stderr)
        self.assertIn("vector mrr: 0.671 < required 0.900", result.stderr)
        self.assertNotIn("keyword", result.stderr)
        self.assertIn("vector", result.stdout)  # metrics table is still printed

    def test_invalid_thresholds_exit_2_before_indexing(self):
        for content in ("{nope", {}, {"hybrid": {"recall@3": 0.5}}, {"hybrid": {"mrr": 2}}):
            with self.subTest(content=content):
                result = self.shipped(content)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("invalid thresholds", result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                self.assertNotIn("Indexed", result.stdout)


if __name__ == "__main__":
    unittest.main()
