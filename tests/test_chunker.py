import unittest

from localseek.chunker import chunk_sections
from localseek.extractors import Section


def words(n: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


class ChunkerTests(unittest.TestCase):
    def test_empty_input_gives_no_chunks(self):
        self.assertEqual(chunk_sections([]), [])
        self.assertEqual(chunk_sections([Section("   \n\n  ")]), [])

    def test_short_text_is_one_chunk(self):
        chunks = chunk_sections([Section("Just a short note.")])
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "Just a short note.")

    def test_long_text_respects_max_words(self):
        chunks = chunk_sections([Section(words(1000))], max_words=100, overlap_words=10)
        self.assertGreater(len(chunks), 5)
        self.assertTrue(all(len(c.text.split()) <= 100 for c in chunks))
        self.assertEqual([c.ordinal for c in chunks], list(range(len(chunks))))

    def test_neighbouring_chunks_overlap(self):
        chunks = chunk_sections([Section(words(300))], max_words=100, overlap_words=20)
        first_tail = chunks[0].text.split()[-20:]
        second_head = chunks[1].text.split()[:20]
        self.assertEqual(first_tail, second_head)

    def test_every_word_is_kept(self):
        text = words(500)
        chunks = chunk_sections([Section(text)], max_words=90, overlap_words=15)
        seen = {w for c in chunks for w in c.text.split()}
        self.assertEqual(seen, set(text.split()))

    def test_no_overlap_only_chunk_at_the_end(self):
        chunks = chunk_sections([Section(words(100))], max_words=100, overlap_words=20)
        self.assertEqual(len(chunks), 1)

    def test_headings_and_pages_are_recorded(self):
        text = "# Intro\n\n" + words(30, "a") + "\n\n## Details\n\n" + words(30, "b")
        chunks = chunk_sections([Section(text, page=4)], max_words=40, overlap_words=5)
        self.assertTrue(all(c.page == 4 for c in chunks))
        self.assertEqual(chunks[-1].heading, "Details")

    def test_paragraph_boundary_preferred(self):
        text = words(70, "a") + "\n\n" + words(70, "b")
        chunks = chunk_sections([Section(text)], max_words=100, overlap_words=0)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[0].text.startswith("a0"))
        self.assertTrue(chunks[1].text.startswith("b0"))

    def test_rejects_tiny_max_words(self):
        with self.assertRaises(ValueError):
            chunk_sections([Section("x")], max_words=5)


if __name__ == "__main__":
    unittest.main()
