"""Embedding backends.

`FastEmbedEmbedder` runs small ONNX models on the CPU. `HashEmbedder` is a dependency-free,
purely lexical stand-in used by the tests and for quick offline experiments.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np


class EmbedderUnavailable(RuntimeError):
    """The requested embedding backend cannot be loaded."""


class Embedder(Protocol):
    name: str

    def embed_passages(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class HashEmbedder:
    """Signed feature hashing of word tokens. Lexical only, not semantic."""

    name = "hash-256"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in re.findall(r"\w+", text.lower()):
            digest = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")
            vec[digest % self.dim] += 1.0 if (digest >> 63) & 1 else -1.0
        return vec

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        return _normalize(np.vstack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(self._vector(text).reshape(1, -1))[0]


class FastEmbedEmbedder:
    """Wraps the `fastembed` library. The model is downloaded once, then cached on disk."""

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise EmbedderUnavailable(
                "fastembed is not installed. Run: pip install fastembed"
            ) from exc
        self.name = model_name
        self._batch_size = batch_size
        try:
            self._model = TextEmbedding(model_name=model_name)
        except Exception as exc:
            raise EmbedderUnavailable(
                f"Could not load model '{model_name}': {exc}. "
                "The first run needs internet access to download the model once."
            ) from exc
        self._e5 = "e5" in model_name.lower()

    def _run(self, texts: list[str]) -> np.ndarray:
        vectors = list(self._model.embed(texts, batch_size=self._batch_size))
        return _normalize(np.asarray(vectors, dtype=np.float32))

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        return self._run([f"passage: {t}" for t in texts] if self._e5 else texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._run([f"query: {text}" if self._e5 else text])[0]


def get_embedder(model: str) -> Embedder:
    if model.lower().startswith("hash"):
        return HashEmbedder()
    return FastEmbedEmbedder(model)
