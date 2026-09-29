"""SQLite storage: file records, chunks, vectors (as BLOBs), and an FTS5 keyword index."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import numpy as np

from .chunker import Chunk

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS roots(path TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS files(
  id INTEGER PRIMARY KEY,
  path TEXT UNIQUE NOT NULL,
  hash TEXT NOT NULL,
  mtime REAL NOT NULL,
  size INTEGER NOT NULL,
  type TEXT NOT NULL,
  indexed_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks(
  id INTEGER PRIMARY KEY,
  file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  ordinal INTEGER NOT NULL,
  text TEXT NOT NULL,
  text_hash TEXT NOT NULL,
  page INTEGER,
  heading TEXT
);
CREATE INDEX IF NOT EXISTS chunks_file ON chunks(file_id);
CREATE INDEX IF NOT EXISTS chunks_hash ON chunks(text_hash);
CREATE TABLE IF NOT EXISTS vectors(
  chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
  embedding BLOB NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  text, tokenize = 'unicode61 remove_diacritics 2'
);
"""

_BATCH = 500


class ModelMismatch(Exception):
    def __init__(self, indexed: str, requested: str) -> None:
        super().__init__(
            f"The index was built with model '{indexed}', but '{requested}' was requested."
        )
        self.indexed = indexed
        self.requested = requested


class Store:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        try:
            self.conn.execute("PRAGMA journal_mode = WAL")
        except sqlite3.DatabaseError:
            pass
        try:
            self.conn.executescript(SCHEMA)
        except sqlite3.OperationalError as exc:
            raise RuntimeError(f"Could not initialise the index (is FTS5 available?): {exc}") from exc
        self._matrix: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def close(self) -> None:
        self.conn.close()

    # -- metadata -------------------------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def ensure_model(self, name: str) -> None:
        current = self.get_meta("model")
        if current is None:
            self.set_meta("model", name)
        elif current != name:
            raise ModelMismatch(current, name)

    def clear_index(self) -> None:
        """Remove all indexed content but keep the registered folders."""
        with self.conn:
            self.conn.execute("DELETE FROM chunks_fts")
            self.conn.execute("DELETE FROM vectors")
            self.conn.execute("DELETE FROM chunks")
            self.conn.execute("DELETE FROM files")
            self.conn.execute("DELETE FROM meta WHERE key = 'model'")
        self._matrix = None

    # -- roots ----------------------------------------------------------------------------

    def add_root(self, path: str) -> None:
        with self.conn:
            self.conn.execute("INSERT OR IGNORE INTO roots(path) VALUES(?)", (path,))

    def remove_root(self, path: str) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM roots WHERE path = ?", (path,))

    def list_roots(self) -> list[str]:
        return [r["path"] for r in self.conn.execute("SELECT path FROM roots ORDER BY path")]

    # -- files ----------------------------------------------------------------------------

    def file_records(self) -> dict[str, sqlite3.Row]:
        rows = self.conn.execute("SELECT id, path, hash, mtime, size FROM files")
        return {r["path"]: r for r in rows}

    def upsert_file(
        self,
        path: str,
        digest: str,
        mtime: float,
        size: int,
        ftype: str,
        rows: list[tuple[Chunk, str, bytes]],
    ) -> None:
        """Replace everything stored for `path`. Each row is (chunk, text_hash, embedding)."""
        now = time.time()
        with self.conn:
            existing = self.conn.execute("SELECT id FROM files WHERE path = ?", (path,)).fetchone()
            if existing:
                file_id = existing["id"]
                self.conn.execute(
                    "DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE file_id = ?)",
                    (file_id,),
                )
                self.conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
                self.conn.execute(
                    "UPDATE files SET hash = ?, mtime = ?, size = ?, type = ?, indexed_at = ? "
                    "WHERE id = ?",
                    (digest, mtime, size, ftype, now, file_id),
                )
            else:
                file_id = self.conn.execute(
                    "INSERT INTO files(path, hash, mtime, size, type, indexed_at) "
                    "VALUES(?, ?, ?, ?, ?, ?)",
                    (path, digest, mtime, size, ftype, now),
                ).lastrowid
            for chunk, text_hash, embedding in rows:
                chunk_id = self.conn.execute(
                    "INSERT INTO chunks(file_id, ordinal, text, text_hash, page, heading) "
                    "VALUES(?, ?, ?, ?, ?, ?)",
                    (file_id, chunk.ordinal, chunk.text, text_hash, chunk.page, chunk.heading),
                ).lastrowid
                self.conn.execute(
                    "INSERT INTO vectors(chunk_id, embedding) VALUES(?, ?)", (chunk_id, embedding)
                )
                self.conn.execute(
                    "INSERT INTO chunks_fts(rowid, text) VALUES(?, ?)", (chunk_id, chunk.text)
                )
        self._matrix = None

    def touch_file(self, path: str, mtime: float, size: int) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE files SET mtime = ?, size = ? WHERE path = ?", (mtime, size, path)
            )

    def delete_file(self, path: str) -> None:
        with self.conn:
            self.conn.execute(
                "DELETE FROM chunks_fts WHERE rowid IN "
                "(SELECT c.id FROM chunks c JOIN files f ON f.id = c.file_id WHERE f.path = ?)",
                (path,),
            )
            self.conn.execute("DELETE FROM files WHERE path = ?", (path,))
        self._matrix = None

    def cached_vectors(self, hashes: set[str]) -> dict[str, bytes]:
        """Embeddings already stored for identical chunk text, so unchanged text is never re-embedded."""
        found: dict[str, bytes] = {}
        pending = list(hashes)
        for start in range(0, len(pending), _BATCH):
            batch = pending[start : start + _BATCH]
            marks = ",".join("?" * len(batch))
            rows = self.conn.execute(
                "SELECT c.text_hash AS h, v.embedding AS e FROM chunks c "
                f"JOIN vectors v ON v.chunk_id = c.id WHERE c.text_hash IN ({marks})",
                batch,
            )
            for row in rows:
                found[row["h"]] = row["e"]
        return found

    # -- search support -------------------------------------------------------------------

    def allowed_file_ids(
        self,
        ftype: str | None = None,
        since: float | None = None,
        path_contains: str | None = None,
    ) -> set[int] | None:
        """File ids matching the filters, or None when no filter is set."""
        clauses: list[str] = []
        params: list[object] = []
        if ftype:
            clauses.append("type = ?")
            params.append(ftype.lower().lstrip("."))
        if since is not None:
            clauses.append("mtime >= ?")
            params.append(since)
        if path_contains:
            clauses.append("instr(lower(path), lower(?)) > 0")
            params.append(path_contains)
        if not clauses:
            return None
        rows = self.conn.execute("SELECT id FROM files WHERE " + " AND ".join(clauses), params)
        return {r["id"] for r in rows}

    def load_matrix(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(chunk_ids, file_ids, embeddings) for every chunk, cached until the next write."""
        if self._matrix is None:
            rows = self.conn.execute(
                "SELECT v.chunk_id AS cid, c.file_id AS fid, v.embedding AS emb "
                "FROM vectors v JOIN chunks c ON c.id = v.chunk_id ORDER BY v.chunk_id"
            ).fetchall()
            if not rows:
                self._matrix = (
                    np.empty(0, dtype=np.int64),
                    np.empty(0, dtype=np.int64),
                    np.empty((0, 0), dtype=np.float32),
                )
            else:
                self._matrix = (
                    np.fromiter((r["cid"] for r in rows), dtype=np.int64, count=len(rows)),
                    np.fromiter((r["fid"] for r in rows), dtype=np.int64, count=len(rows)),
                    np.vstack([np.frombuffer(r["emb"], dtype=np.float32) for r in rows]),
                )
        return self._matrix

    def fts_search(self, match: str, limit: int) -> list[tuple[int, int]]:
        """(chunk_id, file_id) pairs ordered by BM25 relevance."""
        try:
            rows = self.conn.execute(
                "SELECT chunks_fts.rowid AS cid, c.file_id AS fid FROM chunks_fts "
                "JOIN chunks c ON c.id = chunks_fts.rowid "
                "WHERE chunks_fts MATCH ? ORDER BY chunks_fts.rank LIMIT ?",
                (match, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(r["cid"], r["fid"]) for r in rows]

    def fetch_chunks(self, chunk_ids: list[int]) -> dict[int, sqlite3.Row]:
        found: dict[int, sqlite3.Row] = {}
        for start in range(0, len(chunk_ids), _BATCH):
            batch = chunk_ids[start : start + _BATCH]
            marks = ",".join("?" * len(batch))
            rows = self.conn.execute(
                "SELECT c.id, c.text, c.page, c.heading, c.ordinal, f.path, f.mtime, f.type "
                f"FROM chunks c JOIN files f ON f.id = c.file_id WHERE c.id IN ({marks})",
                batch,
            )
            for row in rows:
                found[row["id"]] = row
        return found

    def stats(self) -> dict[str, object]:
        files = self.conn.execute("SELECT COUNT(*) AS n FROM files").fetchone()["n"]
        chunks = self.conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        size = self.path.stat().st_size if self.path.exists() else 0
        return {
            "files": files,
            "chunks": chunks,
            "model": self.get_meta("model"),
            "db_bytes": size,
            "roots": self.list_roots(),
        }
