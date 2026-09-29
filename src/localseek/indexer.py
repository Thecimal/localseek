"""Incremental indexing: scan, extract, chunk, embed, store."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .chunker import chunk_sections
from .config import Settings
from .embedder import Embedder
from .extractors import ExtractionError, extract, supported_suffixes
from .scanner import scan
from .store import Store

_EMBED_BATCH = 128


@dataclass
class IndexStats:
    scanned: int = 0
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    chunks: int = 0
    skipped: list[tuple[str, str]] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)


Progress = Callable[[IndexStats, str], None]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def index_paths(
    store: Store,
    embedder: Embedder,
    roots: Iterable[str | Path],
    settings: Settings,
    progress: Progress | None = None,
) -> IndexStats:
    """Bring the index up to date for `roots`. Raises ModelMismatch if the model changed."""
    store.ensure_model(embedder.name)
    resolved_roots = [Path(r).expanduser().resolve() for r in roots]
    known = store.file_records()
    stats = IndexStats()
    seen: set[str] = set()

    def on_skip(path: str, reason: str) -> None:
        stats.skipped.append((path, reason))

    for info in scan(
        resolved_roots, settings.ignore, settings.max_file_bytes, supported_suffixes(), on_skip
    ):
        key = str(info.path)
        seen.add(key)
        stats.scanned += 1
        if progress:
            progress(stats, key)

        record = known.get(key)
        if record and record["size"] == info.size and abs(record["mtime"] - info.mtime) < 1e-6:
            stats.unchanged += 1
            continue

        try:
            digest = file_hash(info.path)
        except OSError as exc:
            stats.errors.append((key, str(exc)))
            continue
        if record and record["hash"] == digest:
            store.touch_file(key, info.mtime, info.size)
            stats.unchanged += 1
            continue

        try:
            sections = extract(info.path)
        except ExtractionError as exc:
            stats.errors.append((key, str(exc)))
            continue

        chunks = chunk_sections(sections, settings.chunk_words, settings.overlap_words)
        hashes = [text_hash(c.text) for c in chunks]
        vectors = store.cached_vectors(set(hashes))
        missing = [i for i, h in enumerate(hashes) if h not in vectors]
        for start in range(0, len(missing), _EMBED_BATCH):
            batch = missing[start : start + _EMBED_BATCH]
            embedded = embedder.embed_passages([chunks[i].text for i in batch])
            for offset, index in enumerate(batch):
                vectors[hashes[index]] = np.asarray(embedded[offset], dtype=np.float32).tobytes()

        rows = [(chunk, h, vectors[h]) for chunk, h in zip(chunks, hashes, strict=True)]
        store.upsert_file(
            key, digest, info.mtime, info.size, info.path.suffix.lower().lstrip("."), rows
        )
        stats.chunks += len(chunks)
        if record:
            stats.updated += 1
        else:
            stats.added += 1

    for path in known:
        if path not in seen and any(Path(path).is_relative_to(root) for root in resolved_roots):
            store.delete_file(path)
            stats.removed += 1
    return stats
