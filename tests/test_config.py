import tempfile
import unittest
from pathlib import Path

from localseek.config import DEFAULT_IGNORE, load_settings


class ConfigTests(unittest.TestCase):
    def write(self, text):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "config.toml"
        path.write_text(text, encoding="utf-8")
        return path

    def test_values_and_extended_ignore(self):
        s = load_settings(self.write('chunk_words = 100\nignore = ["*.tmp"]\n'))
        self.assertEqual(s.chunk_words, 100)
        self.assertEqual(s.ignore, DEFAULT_IGNORE + ["*.tmp"])

    def test_unknown_key_rejected(self):
        with self.assertRaises(ValueError):
            load_settings(self.write("bogus = 1\n"))

    def test_bad_overlap_rejected(self):
        with self.assertRaises(ValueError):
            load_settings(self.write("chunk_words = 100\noverlap_words = 80\n"))

    def test_missing_explicit_file_errors(self):
        with self.assertRaises(FileNotFoundError):
            load_settings("/nonexistent/config.toml")


if __name__ == "__main__":
    unittest.main()
