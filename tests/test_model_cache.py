import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from localseek.embedder import FastEmbedEmbedder, HashEmbedder, get_embedder, model_cache_dir


class FakeTextEmbedding:
    """Stands in for fastembed.TextEmbedding and records how it was constructed."""

    calls: list[dict] = []

    def __init__(self, model_name, cache_dir=None, **kwargs):
        FakeTextEmbedding.calls.append({"model_name": model_name, "cache_dir": cache_dir})


def fake_fastembed():
    return {"fastembed": types.SimpleNamespace(TextEmbedding=FakeTextEmbedding)}


class ModelCacheTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.data = Path(self._tmp.name)
        FakeTextEmbedding.calls = []
        env = {"LOCALSEEK_DATA_DIR": str(self.data)}
        patcher = mock.patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("FASTEMBED_CACHE_PATH", None)

    def tearDown(self):
        self._tmp.cleanup()

    def test_models_live_in_the_data_folder_not_the_temp_directory(self):
        self.assertEqual(model_cache_dir(), str(self.data / "models"))

    def test_fastembed_cache_path_still_wins(self):
        with mock.patch.dict(os.environ, {"FASTEMBED_CACHE_PATH": str(self.data / "elsewhere")}):
            self.assertIsNone(model_cache_dir())

    def test_the_real_model_is_loaded_from_that_folder(self):
        with mock.patch.dict(sys.modules, fake_fastembed()):
            get_embedder("BAAI/bge-small-en-v1.5")
        self.assertEqual(
            FakeTextEmbedding.calls,
            [{"model_name": "BAAI/bge-small-en-v1.5", "cache_dir": str(self.data / "models")}],
        )

    def test_the_override_reaches_fastembed_untouched(self):
        with mock.patch.dict(os.environ, {"FASTEMBED_CACHE_PATH": str(self.data / "elsewhere")}):
            with mock.patch.dict(sys.modules, fake_fastembed()):
                get_embedder("BAAI/bge-small-en-v1.5")
        self.assertIsNone(FakeTextEmbedding.calls[0]["cache_dir"])

    def test_an_explicit_cache_dir_is_passed_through(self):
        with mock.patch.dict(sys.modules, fake_fastembed()):
            FastEmbedEmbedder("some/model", cache_dir="/somewhere")
        self.assertEqual(FakeTextEmbedding.calls[0]["cache_dir"], "/somewhere")

    def test_the_hash_embedder_needs_neither_fastembed_nor_a_folder(self):
        with mock.patch.dict(sys.modules, {"fastembed": None}):
            self.assertIsInstance(get_embedder("hash"), HashEmbedder)
        self.assertFalse((self.data / "models").exists())


if __name__ == "__main__":
    unittest.main()
