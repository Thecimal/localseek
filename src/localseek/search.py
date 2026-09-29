"""Hybrid search: vector similarity plus BM25 keywords, merged with Reciprocal Rank Fusion."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import numpy as np

from .embedder import Embedder
from .store import Store

MODES = ("hybrid", "vector", "keyword")
_TOKEN = re.compile(r"\w+", re.UNICODE)


@dataclass
class Hit:
    path: str
    score: float  # relative, 1.0 means top-ranked by every active method
    snippet: str
    page: int | None
    heading: str | None
    chunk: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def query_terms(query: str) -> list[str]:
    seen: dict[str, None] = {}
    for token in _TOKEN.findall(query.lower()):
        seen.setdefault(token, None)
    return list(seen)[:32]


def fts_query(query: str) -> str:
    """Build a safe FTS5 query: every term quoted, joined with OR."""
    return " OR ".join(f'"{t}"' for t in query_terms(query))


def rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for position, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + position)
    return scores


def make_snippet(text: str, terms: list[str], width: int = 280) -> str:
    text = " ".join(text.split())
    if len(text) <= width:
        return text
    lowered = text.lower()
    positions = [p for p in (lowered.find(t) for t in terms) if p >= 0]
    start = max(0, min(positions) - 60) if positions else 0
    if start > 0:
        space = text.find(" ", start)
        start = space + 1 if space != -1 else start
    end = min(len(text), start + width)
    return ("…" if start > 0 else "") + text[start:end].strip() + ("…" if end < len(text) else "")


class Searcher:
    def __init__(self, store: Store, embedder: Embedder, rrf_k: int = 60) -> None:
        self.store = store
        self.embedder = embedder
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        limit: int = 10,
        mode: str = "hybrid",
        ftype: str | None = None,
        since: float | None = None,
        path_contains: str | None = None,
        pool: int = 100,
    ) -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        query = query.strip()
        if not query or limit < 1:
            return []
        allowed = self.store.allowed_file_ids(ftype, since, path_contains)
        if allowed is not None and not allowed:
            return []

        rankings: list[list[int]] = []
        active = 0
        if mode in ("hybrid", "vector"):
            active += 1
            rankings.append(self._vector_rank(query, allowed, pool))
        if mode in ("hybrid", "keyword"):
            active += 1
            rankings.append(self._keyword_rank(query, allowed, pool))
        rankings = [r for r in rankings if r]
        if not rankings:
            return []

        fused = rrf(rankings, self.rrf_k)
        best_possible = active / (self.rrf_k + 1)
        ordered = sorted(fused, key=lambda c: (-fused[c], c))[:pool]
        rows = self.store.fetch_chunks(ordered)

        terms = query_terms(query)
        hits: list[Hit] = []
        seen: set[str] = set()
        for chunk_id in ordered:
            row = rows.get(chunk_id)
            if row is None or row["path"] in seen:
                continue
            seen.add(row["path"])
            hits.append(
                Hit(
                    path=row["path"],
                    score=round(fused[chunk_id] / best_possible, 4),
                    snippet=make_snippet(row["text"], terms),
                    page=row["page"],
                    heading=row["heading"],
                    chunk=row["ordinal"],
                )
            )
            if len(hits) >= limit:
                break
        return hits

    def _vector_rank(self, query: str, allowed: set[int] | None, pool: int) -> list[int]:
        ids, file_ids, matrix = self.store.load_matrix()
        if len(ids) == 0:
            return []
        vector = np.asarray(self.embedder.embed_query(query), dtype=np.float32).reshape(-1)
        if matrix.shape[1] != vector.shape[0]:
            raise ValueError("Embedding size mismatch. Rebuild the index with `--rebuild`.")
        sims = matrix @ vector
        if allowed is not None:
            mask = np.isin(file_ids, np.fromiter(allowed, dtype=np.int64, count=len(allowed)))
            sims = np.where(mask, sims, -np.inf)
        count = min(pool, len(sims))
        if count < len(sims):
            top = np.argpartition(-sims, count - 1)[:count]
        else:
            top = np.arange(len(sims))
        top = top[np.argsort(-sims[top], kind="stable")]
        return [int(ids[i]) for i in top if np.isfinite(sims[i])]

    def _keyword_rank(self, query: str, allowed: set[int] | None, pool: int) -> list[int]:
        match = fts_query(query)
        if not match:
            return []
        fetch = pool * 5 if allowed is not None else pool
        pairs = self.store.fts_search(match, fetch)
        return [c for c, f in pairs if allowed is None or f in allowed][:pool]
