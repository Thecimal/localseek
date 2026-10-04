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


# Common English function words. Used only when the keyword arm is asked to ignore them (see `drop_stopwords`).
STOPWORDS = frozenset(
    """
    a about above after again all also am an and any are as at be because been before being below between
    both but by can could did do does doing down during each few for from further had has have having he her
    here hers him his how i if in into is it its just me more most my no nor not of off on once only or other
    our out over own same she should so some such than that the their them then there these they this those
    through to too under until up very was we were what when where which while who whom why will with would
    you your
    """.split()
)


def fts_query(query: str, drop_stopwords: bool = False) -> str:
    """Build a safe FTS5 query: every term quoted, joined with OR.

    With `drop_stopwords`, function words are left out so they cannot match most of the index; a query made only of
    function words keeps all of its terms rather than becoming empty.
    """
    terms = query_terms(query)
    if drop_stopwords:
        terms = [t for t in terms if t not in STOPWORDS] or terms
    return " OR ".join(f'"{t}"' for t in terms)


def rrf(rankings: list[list[int]], k: int = 60, weights: list[float] | None = None) -> dict[int, float]:
    scores: dict[int, float] = {}
    for index, ranking in enumerate(rankings):
        weight = 1.0 if weights is None else weights[index]
        for position, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + weight / (k + position)
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
    """Hybrid searcher. The defaults are the shipped behaviour; the other options exist for evaluation experiments.

    rrf_k: Reciprocal Rank Fusion constant. Smaller values make the top ranks of each arm count for more.
    vector_weight, keyword_weight: multiply each arm's contribution to the fused score.
    keyword_limit: only the first N keyword results take part in fusion (None means all, up to the pool).
    drop_stopwords: the keyword arm ignores common English function words.
    """

    def __init__(
        self,
        store: Store,
        embedder: Embedder,
        rrf_k: int = 60,
        vector_weight: float = 1.0,
        keyword_weight: float = 1.0,
        keyword_limit: int | None = None,
        drop_stopwords: bool = False,
    ) -> None:
        if rrf_k < 1:
            raise ValueError("rrf_k must be at least 1")
        if not (0 < vector_weight < float("inf") and 0 < keyword_weight < float("inf")):
            raise ValueError("weights must be positive and finite")
        if keyword_limit is not None and keyword_limit < 1:
            raise ValueError("keyword_limit must be at least 1")
        self.store = store
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.keyword_limit = keyword_limit
        self.drop_stopwords = drop_stopwords

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

        arms: list[tuple[list[int], float]] = []
        total_weight = 0.0
        if mode in ("hybrid", "vector"):
            total_weight += self.vector_weight
            arms.append((self._vector_rank(query, allowed, pool), self.vector_weight))
        if mode in ("hybrid", "keyword"):
            total_weight += self.keyword_weight
            keyword = self._keyword_rank(query, allowed, pool)
            if self.keyword_limit is not None:
                keyword = keyword[: self.keyword_limit]
            arms.append((keyword, self.keyword_weight))
        arms = [(ranking, weight) for ranking, weight in arms if ranking]
        if not arms:
            return []

        fused = rrf([ranking for ranking, _ in arms], self.rrf_k, [weight for _, weight in arms])
        best_possible = total_weight / (self.rrf_k + 1)
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
        match = fts_query(query, self.drop_stopwords)
        if not match:
            return []
        fetch = pool * 5 if allowed is not None else pool
        pairs = self.store.fts_search(match, fetch)
        return [c for c, f in pairs if allowed is None or f in allowed][:pool]
