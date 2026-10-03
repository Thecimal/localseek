"""The benchmark under eval/benchmark must stay what it claims to be.

These tests never run a search, so they cannot be satisfied by tuning the benchmark to an algorithm. They check
composition (size, formats, shared names, near-duplicates, long documents, query categories) and that every
relevance judgment is backed by an exact phrase from the document it names.
"""

import hashlib
import json
import re
import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from localseek.chunker import chunk_sections
from localseek.config import Settings
from localseek.extractors import extract, supported_suffixes

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))

from retrieval_eval import load_dataset  # noqa: E402

BENCH = REPO / "eval" / "benchmark"
CORPUS = BENCH / "corpus"
QUERIES = BENCH / "queries.json"

# Minimum queries per category. Raising a number is fine; removing a category is a product decision.
REQUIRED_CATEGORIES = {
    "exact_keyword": 8,
    "technical_term": 8,
    "rare_term": 6,
    "semantic_paraphrase": 10,
    "multi_document": 6,
    "ambiguous": 6,
    "long_question": 6,
    "redundant_answer": 6,
    "near_duplicate": 6,
    "shared_filename": 6,
    "similar_subject": 4,
    "late_answer": 8,
}
# Deliberately absent for now (see eval/benchmark/README.md). Listed so nobody mistakes the gap for coverage.
DEFERRED_CATEGORIES = {"no_answer", "multilingual"}
# Categories whose evidence phrase identifies the answer uniquely, so it must not occur in any other document.
UNIQUE_EVIDENCE = {"exact_keyword", "technical_term", "rare_term"}


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def shingles(text: str, size: int = 5) -> set[tuple[str, ...]]:
    words = norm(text).split()
    return {tuple(words[i : i + size]) for i in range(max(1, len(words) - size + 1))}


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = json.loads(QUERIES.read_text(encoding="utf-8"))
        cls.files = sorted(p for p in CORPUS.rglob("*") if p.is_file())
        cls.rel = {p: p.relative_to(CORPUS).as_posix() for p in cls.files}
        settings = Settings()
        cls.text: dict[str, str] = {}
        cls.chunks: dict[str, list[str]] = {}
        for path in cls.files:
            sections = extract(path)
            rel = cls.rel[path]
            cls.text[rel] = norm(" ".join(s.text for s in sections))
            chunks = chunk_sections(sections, settings.chunk_words, settings.overlap_words)
            cls.chunks[rel] = [norm(c.text) for c in chunks]

    # -- the corpus ------------------------------------------------------------------------

    def test_every_file_is_indexable(self):
        self.assertTrue(self.files)
        for path in self.files:
            self.assertIn(path.suffix.lower(), supported_suffixes(), path)
            self.assertTrue(self.text[self.rel[path]], f"{path} extracted no text")

    def test_corpus_scale_and_variety(self):
        self.assertGreaterEqual(len(self.files), 50)
        suffixes = {p.suffix.lower() for p in self.files}
        self.assertTrue({".md", ".txt", ".html", ".docx", ".pdf"} <= suffixes, suffixes)
        words = [len(t.split()) for t in self.text.values()]
        self.assertLessEqual(min(words), 60, "needs very short documents")
        self.assertGreaterEqual(max(words), 550, "needs long documents")

    def test_multi_chunk_documents(self):
        deep = [rel for rel, chunks in self.chunks.items() if len(chunks) >= 3]
        self.assertGreaterEqual(len(deep), 6, deep)

    def test_several_documents_share_file_names(self):
        by_name = defaultdict(list)
        for path in self.files:
            by_name[path.name].append(self.rel[path])
        shared = {name: paths for name, paths in by_name.items() if len(paths) > 1}
        self.assertGreaterEqual(len(shared), 4, shared)
        self.assertGreaterEqual(max(len(p) for p in shared.values()), 4)

    def test_near_duplicates_exist(self):
        rels = sorted(self.text)
        grams = {rel: shingles(self.text[rel]) for rel in rels}
        pairs = [
            (a, b)
            for i, a in enumerate(rels)
            for b in rels[i + 1 :]
            if len(grams[a] | grams[b]) and len(grams[a] & grams[b]) / len(grams[a] | grams[b]) >= 0.5
        ]
        self.assertGreaterEqual(len(pairs), 4, pairs)
        self.assertTrue(any(self.text[a] == self.text[b] for a, b in pairs), "needs one exact duplicate")

    # -- the queries -----------------------------------------------------------------------

    def test_dataset_loads_with_the_evaluator_validation(self):
        queries = load_dataset(QUERIES, CORPUS.resolve())
        self.assertEqual(len(queries), len(self.raw))
        self.assertTrue(all(q.category for q in queries))

    def test_category_coverage(self):
        counts = Counter(item["category"] for item in self.raw)
        for category, minimum in REQUIRED_CATEGORIES.items():
            self.assertGreaterEqual(counts[category], minimum, category)
        self.assertEqual(set(counts) - set(REQUIRED_CATEGORIES), set(), "unknown category: add it to this spec")
        self.assertEqual(DEFERRED_CATEGORIES & set(counts), set(), "category is no longer deferred: update this spec")

    def test_queries_are_unique(self):
        texts = [norm(item["query"]) for item in self.raw]
        self.assertEqual(len(texts), len(set(texts)))

    def test_relevance_is_exactly_the_evidence(self):
        for item in self.raw:
            self.assertEqual(set(item["relevant"]), set(item["evidence"]), item["query"])
            self.assertEqual(len(item["relevant"]), len(set(item["relevant"])), item["query"])

    def test_every_judgment_is_supported_by_its_document(self):
        for item in self.raw:
            for rel, phrase in item["evidence"].items():
                self.assertIn(norm(phrase), self.text[rel], f"{item['query']!r}: {rel}")

    def test_unique_term_queries_are_complete(self):
        # If an identifying phrase also appears in a document not listed as relevant, the judgment is incomplete.
        for item in self.raw:
            if item["category"] not in UNIQUE_EVIDENCE:
                continue
            for phrase in item["evidence"].values():
                holders = {r for r, t in self.text.items() if norm(phrase) in t}
                self.assertLessEqual(holders, set(item["relevant"]), f"{item['query']!r}: {phrase!r} also in {holders}")

    def test_late_answers_are_not_in_the_first_two_chunks(self):
        for item in self.raw:
            if item["category"] != "late_answer":
                continue
            for rel, phrase in item["evidence"].items():
                holders = [i for i, chunk in enumerate(self.chunks[rel]) if norm(phrase) in chunk]
                self.assertTrue(holders, f"{item['query']!r}: phrase spans a chunk boundary in {rel}")
                self.assertGreaterEqual(min(holders), 2, f"{item['query']!r}: answer in chunk {min(holders)} of {rel}")

    def test_multi_and_redundant_queries_have_several_relevant_documents(self):
        for item in self.raw:
            if item["category"] in {"multi_document", "redundant_answer"}:
                self.assertGreaterEqual(len(item["relevant"]), 2, item["query"])
            if item["category"] in {"near_duplicate", "exact_keyword", "technical_term", "rare_term", "late_answer"}:
                self.assertGreaterEqual(len(item["relevant"]), 1, item["query"])

    def test_similar_subject_queries_have_one_right_answer(self):
        for item in self.raw:
            if item["category"] == "similar_subject":
                self.assertEqual(len(item["relevant"]), 1, item["query"])

    def test_shared_filename_queries_target_shared_names(self):
        counts = Counter(p.name for p in self.files)
        for item in self.raw:
            if item["category"] == "shared_filename":
                for rel in item["relevant"]:
                    self.assertGreater(counts[Path(rel).name], 1, f"{rel} has a unique file name")

    def test_near_duplicate_queries_have_a_similar_document_that_is_not_relevant(self):
        grams = {rel: shingles(text) for rel, text in self.text.items()}
        for item in self.raw:
            if item["category"] != "near_duplicate":
                continue
            self.assertEqual(len(item["relevant"]), 1, item["query"])
            target = item["relevant"][0]
            twins = [
                rel
                for rel in grams
                if rel != target and len(grams[rel] & grams[target]) / len(grams[rel] | grams[target]) >= 0.4
            ]
            self.assertTrue(twins, f"{item['query']!r}: {target} has no similar distractor")


if __name__ == "__main__":
    unittest.main()


# -- the held-out paraphrase set (queries-v2.json) -----------------------------------------------------------------

QUERIES_V2 = BENCH / "queries-v2.json"
STOPWORDS = {
    "the", "and", "for", "are", "was", "were", "with", "that", "this", "from", "not", "but", "has", "have", "had",
    "you", "your", "our", "can", "how", "what", "when", "where", "which", "who", "why", "does", "did", "any", "into",
    "than", "then", "its", "there", "about", "would", "should", "will", "one", "all", "also", "may", "per", "too",
    "very", "shouldn",
}  # fmt: skip


def content_words(text: str) -> set[str]:
    """Alphabetic words of three letters or more that are not stopwords; numbers are deliberately ignored."""
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 2 and w not in STOPWORDS}


def split_for(path: str) -> str:
    """Deterministic dev/test assignment from the first relevant path, so nobody chooses which queries are held out."""
    return "dev" if int(hashlib.sha256(path.encode("utf-8")).hexdigest()[0], 16) % 2 == 0 else "test"


class HeldOutSetTests(unittest.TestCase):
    """Rules fixed before any search was run, so the set cannot be shaped by how an algorithm performs on it."""

    @classmethod
    def setUpClass(cls):
        cls.raw = json.loads(QUERIES_V2.read_text(encoding="utf-8"))
        cls.v1 = {norm(item["query"]) for item in json.loads(QUERIES.read_text(encoding="utf-8"))}
        cls.text = {}
        for path in sorted(p for p in CORPUS.rglob("*") if p.is_file()):
            cls.text[path.relative_to(CORPUS).as_posix()] = norm(" ".join(s.text for s in extract(path)))

    def test_loads_with_the_evaluator_validation(self):
        queries = load_dataset(QUERIES_V2, CORPUS.resolve())
        self.assertEqual(len(queries), len(self.raw))
        self.assertEqual({q.category for q in queries}, {"paraphrase"})
        self.assertTrue(all(q.split in {"dev", "test"} for q in queries))

    def test_every_document_is_the_answer_to_at_least_one_question(self):
        covered = {path for item in self.raw for path in item["relevant"]}
        missing = sorted(set(self.text) - covered)
        self.assertEqual(missing, [], "documents with no question: add one, do not skip hard ones")

    def test_relevance_is_exactly_the_evidence_and_the_evidence_is_in_the_document(self):
        for item in self.raw:
            self.assertEqual(set(item["relevant"]), set(item["evidence"]), item["query"])
            for path, phrase in item["evidence"].items():
                self.assertIn(norm(phrase), self.text[path], f"{item['query']!r}: {path}")

    def test_questions_share_no_content_word_with_their_evidence(self):
        for item in self.raw:
            for path, phrase in item["evidence"].items():
                shared = content_words(item["query"]) & content_words(phrase)
                self.assertEqual(shared, set(), f"{item['query']!r} repeats words from its answer in {path}")

    def test_split_follows_the_hash_rule(self):
        for item in self.raw:
            self.assertEqual(item["split"], split_for(item["relevant"][0]), item["query"])

    def test_both_splits_are_populated(self):
        counts = Counter(item["split"] for item in self.raw)
        self.assertGreaterEqual(counts["dev"], 20)
        self.assertGreaterEqual(counts["test"], 20)

    def test_questions_are_unique_and_new(self):
        texts = [norm(item["query"]) for item in self.raw]
        self.assertEqual(len(texts), len(set(texts)))
        self.assertEqual(set(texts) & self.v1, set(), "a question repeats one from benchmark v1")

    def test_the_rule_helpers_behave(self):
        self.assertEqual(content_words("How do I fix the 2025 tax errors?"), {"fix", "tax", "errors"})
        self.assertEqual(content_words("It is 42"), set())
        self.assertTrue({split_for(name) for name in ("a.md", "b.md", "c.md", "d.md")} <= {"dev", "test"})
        self.assertEqual(split_for("housing/notes.md"), split_for("housing/notes.md"))
