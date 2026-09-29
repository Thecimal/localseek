import tempfile
import unittest
from pathlib import Path

from helpers import make_docx, make_epub, make_pdf, write
from localseek.extractors import ExtractionError, extract, supported_suffixes


class ExtractorTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_text_and_markdown(self):
        path = write(self.dir / "a.md", "# Title\n\nHello world")
        sections = extract(path)
        self.assertEqual(len(sections), 1)
        self.assertIn("Hello world", sections[0].text)

    def test_empty_text_file_gives_no_sections(self):
        self.assertEqual(extract(write(self.dir / "empty.txt", "  \n")), [])

    def test_binary_rejected(self):
        path = self.dir / "blob.txt"
        path.write_bytes(b"abc\x00\x01\x02")
        with self.assertRaises(ExtractionError):
            extract(path)

    def test_html_strips_scripts_and_tags(self):
        html = (
            "<html><head><style>p{color:red}</style></head><body><h1>Heading</h1>"
            "<script>alert('x')</script><p>First <b>bold</b> para.</p><p>Second.</p></body></html>"
        )
        text = extract(write(self.dir / "page.html", html))[0].text
        self.assertIn("Heading", text)
        self.assertIn("First bold para.", text)
        self.assertNotIn("alert", text)
        self.assertNotIn("color:red", text)

    def test_docx(self):
        path = make_docx(self.dir / "letter.docx", ["Dear tenant", "The boiler is fixed."])
        text = extract(path)[0].text
        self.assertIn("Dear tenant", text)
        self.assertIn("boiler is fixed", text)

    def test_broken_docx_raises(self):
        path = write(self.dir / "bad.docx", "not a zip")
        with self.assertRaises(ExtractionError):
            extract(path)

    def test_epub_reads_chapters_in_order(self):
        path = make_epub(
            self.dir / "book.epub",
            {"ch2.xhtml": "<p>Second chapter</p>", "ch1.xhtml": "<p>First chapter</p>"},
        )
        texts = [s.text for s in extract(path)]
        self.assertEqual(texts, ["First chapter", "Second chapter"])

    def test_pdf_pages(self):
        try:
            import pypdf  # noqa: F401
        except ImportError:
            self.skipTest("pypdf not installed")
        path = make_pdf(self.dir / "doc.pdf", ["Alpha page text", "Beta page text"])
        sections = extract(path)
        self.assertEqual([s.page for s in sections], [1, 2])
        self.assertIn("Alpha", sections[0].text)

    def test_unsupported_type(self):
        with self.assertRaises(ExtractionError):
            extract(write(self.dir / "photo.jpg", "x"))

    def test_registry_contains_core_formats(self):
        for suffix in (".txt", ".md", ".pdf", ".docx", ".html", ".epub", ".py"):
            self.assertIn(suffix, supported_suffixes())


if __name__ == "__main__":
    unittest.main()
