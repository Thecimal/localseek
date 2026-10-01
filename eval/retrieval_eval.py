"""Dataset loading, validation, and scoring for `eval/run_eval.py`.

Documents are identified by their path relative to the evaluation corpus (POSIX style, e.g.
`recipes/sourdough.md`), never by file name alone, so two files called `notes.md` stay distinct.

Dataset format: a non-empty JSON list of objects

    {"query": "bread fermentation", "relevant": ["recipes/sourdough.md", "recipes/fermentation.md"]}
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from importlib import metadata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

KS = (1, 5, 10)


class DatasetError(Exception):
    """The evaluation dataset is unusable. `problems` lists every issue found."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("\n".join(problems))
        self.problems = problems


@dataclass(frozen=True)
class Query:
    text: str
    relevant: tuple[str, ...]
    category: str | None = None


@dataclass(frozen=True)
class QueryScore:
    recall: dict[int, float]  # fraction of the relevant set found in the top k
    reciprocal_rank: float  # 1 / rank of the first relevant hit, 0 when none is returned


@dataclass(frozen=True)
class ModeResult:
    recall: dict[int, float]
    mrr: float
    latency_ms: float


# -- identity ------------------------------------------------------------------------------


def relative_path(path: str | Path, root: Path) -> str:
    """A hit's path relative to the corpus root. Paths outside the root are returned unchanged."""
    try:
        return Path(path).relative_to(root).as_posix()
    except ValueError:
        return Path(path).as_posix()


def _path_problem(value: object, root: Path) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return "must be a non-empty string"
    if "\\" in value:
        return "use forward slashes"
    pure = PurePosixPath(value)
    if pure.is_absolute():
        return "must be relative to the corpus, not absolute"
    if ".." in pure.parts:
        return "must stay inside the corpus (no '..')"
    if pure.as_posix() != value:
        return f"is not in normalized form (use '{pure.as_posix()}')"
    target = root / value
    if not target.exists():
        return "does not exist in the corpus"
    if not target.is_file():
        return "is not a file"
    return None


# -- loading and validation ----------------------------------------------------------------


def load_dataset(path: str | Path, root: Path) -> list[Query]:
    """Read and validate a dataset. Raises DatasetError listing every problem found."""
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise DatasetError([f"cannot read {path}: {exc}"]) from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DatasetError([f"not valid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})"]) from exc
    if not isinstance(data, list):
        raise DatasetError(["the top level must be a JSON list of query objects"])
    if not data:
        raise DatasetError(["the dataset contains no queries"])

    problems: list[str] = []
    queries: list[Query] = []
    for number, item in enumerate(data, start=1):
        label = f"query #{number}"
        if not isinstance(item, dict):
            problems.append(f"{label}: must be an object with 'query' and 'relevant'")
            continue
        before = len(problems)

        text = item.get("query")
        if "query" not in item:
            problems.append(f"{label}: missing 'query'")
        elif not isinstance(text, str) or not text.strip():
            problems.append(f"{label}: 'query' must be a non-empty string")
        else:
            label = f"query #{number} ({text.strip()!r})"

        relevant = item.get("relevant")
        if "expected" in item and "relevant" not in item:
            problems.append(
                f"{label}: 'expected' is no longer supported; use "
                "\"relevant\": [\"<path relative to the corpus>\"]"
            )
        elif "relevant" not in item:
            problems.append(f"{label}: missing 'relevant'")
        elif not isinstance(relevant, list):
            problems.append(f"{label}: 'relevant' must be a list of corpus-relative paths")
        elif not relevant:
            problems.append(f"{label}: 'relevant' must not be empty")
        else:
            seen: set[str] = set()
            for entry in relevant:
                issue = _path_problem(entry, root)
                if issue:
                    problems.append(f"{label}: relevant path {entry!r} {issue}")
                elif entry in seen:
                    problems.append(f"{label}: relevant path {entry!r} is listed more than once")
                else:
                    seen.add(entry)

        category = item.get("category")
        if "category" in item and (not isinstance(category, str) or not category.strip()):
            problems.append(f"{label}: 'category' must be a non-empty string when present")

        if len(problems) == before:
            queries.append(Query(text.strip(), tuple(relevant), category))
    if problems:
        raise DatasetError(problems)
    return queries


def check_indexed(indexed_paths: Iterable[str | Path], root: Path, queries: Sequence[Query]) -> None:
    """Every relevant document must actually be in the index, or recall could never reach 1.0.

    Catches files that exist but were ignored, too large, unsupported, or unreadable, and
    case mismatches on case-insensitive filesystems.
    """
    indexed = {relative_path(p, root) for p in indexed_paths}
    problems = [
        f"query #{n} ({q.text!r}): relevant document {rel!r} exists but was not indexed "
        "(unsupported type, ignored, too large, or unreadable)"
        for n, q in enumerate(queries, start=1)
        for rel in q.relevant
        if rel not in indexed
    ]
    if problems:
        raise DatasetError(problems)


# -- scoring -------------------------------------------------------------------------------


def score_query(ranked: Sequence[str], relevant: Iterable[str], ks: Sequence[int] = KS) -> QueryScore:
    """Score one ranked list of corpus-relative paths against a relevance set."""
    wanted = set(relevant)
    if not wanted:
        raise ValueError("relevant set must not be empty")
    recall = {k: len(wanted.intersection(ranked[:k])) / len(wanted) for k in ks}
    reciprocal = next((1 / rank for rank, path in enumerate(ranked, start=1) if path in wanted), 0.0)
    return QueryScore(recall, reciprocal)


def evaluate(searcher, queries: Sequence[Query], root: Path, mode: str, ks: Sequence[int] = KS) -> ModeResult:
    if not queries:
        raise ValueError("no queries to evaluate")
    scores: list[QueryScore] = []
    started = time.perf_counter()
    for query in queries:
        hits = searcher.search(query.text, limit=max(ks), mode=mode)
        ranked = [relative_path(h.path, root) for h in hits]
        scores.append(score_query(ranked, query.relevant, ks))
    latency = (time.perf_counter() - started) / len(queries) * 1000
    return ModeResult(
        recall={k: sum(s.recall[k] for s in scores) / len(scores) for k in ks},
        mrr=sum(s.reciprocal_rank for s in scores) / len(scores),
        latency_ms=latency,
    )


def evaluate_by_category(
    searcher, queries: Sequence[Query], root: Path, mode: str, ks: Sequence[int] = KS
) -> dict[str, ModeResult]:
    """Metrics per query category, in first-seen order. Queries without a category are grouped as 'uncategorized'."""
    groups: dict[str, list[Query]] = {}
    for query in queries:
        groups.setdefault(query.category or "uncategorized", []).append(query)
    return {name: evaluate(searcher, group, root, mode, ks) for name, group in groups.items()}


# -- quality gate --------------------------------------------------------------------------

# Slack for floating-point noise only (0.8 computed as 4/5 must satisfy a 0.8 threshold).
_EPSILON = 1e-9

# mode -> metric -> minimum required value
Thresholds = dict[str, dict[str, float]]


class ThresholdError(DatasetError):
    """The thresholds file is unusable."""


@dataclass(frozen=True)
class Failure:
    mode: str
    metric: str
    actual: float
    required: float

    def __str__(self) -> str:
        return f"{self.mode} {self.metric}: {self.actual:.3f} < required {self.required:.3f}"


def metric_names(ks: Sequence[int] = KS) -> list[str]:
    return [f"recall@{k}" for k in ks] + ["mrr"]


def metric_values(result: ModeResult) -> dict[str, float]:
    values = {f"recall@{k}": v for k, v in result.recall.items()}
    values["mrr"] = result.mrr
    return values


def load_thresholds(path: str | Path, modes: Sequence[str], ks: Sequence[int] = KS) -> Thresholds:
    """Read minimum required metrics, as {mode: {metric: minimum}}.

    Example: {"hybrid": {"recall@1": 0.8, "mrr": 0.85}, "keyword": {"recall@5": 0.9}}
    Modes and metrics that are not listed are not checked. Raises ThresholdError listing every problem.
    """
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ThresholdError([f"cannot read {path}: {exc}"]) from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ThresholdError([f"not valid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})"]) from exc
    if not isinstance(data, dict):
        raise ThresholdError(['the top level must be an object like {"hybrid": {"recall@1": 0.8}}'])
    if not data:
        raise ThresholdError(["no thresholds defined; a gate that checks nothing is not allowed"])

    valid_metrics = metric_names(ks)
    problems: list[str] = []
    result: Thresholds = {}
    for mode, wanted in data.items():
        if mode not in modes:
            problems.append(f"unknown mode {mode!r} (expected one of: {', '.join(modes)})")
            continue
        if not isinstance(wanted, dict) or not wanted:
            problems.append(f"{mode}: must be a non-empty object mapping metric names to minimum values")
            continue
        result[mode] = {}
        for metric, value in wanted.items():
            if metric not in valid_metrics:
                problems.append(f"{mode}: unknown metric {metric!r} (expected one of: {', '.join(valid_metrics)})")
            elif isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
                problems.append(f"{mode} {metric}: threshold must be a number, got {value!r}")
            elif not 0 <= value <= 1:
                problems.append(f"{mode} {metric}: threshold must be between 0 and 1, got {value!r}")
            else:
                result[mode][metric] = float(value)
    if problems:
        raise ThresholdError(problems)
    return result


def check_thresholds(results: dict[str, ModeResult], thresholds: Thresholds) -> list[Failure]:
    """Every threshold the results fail to meet, ordered by mode then metric as listed in the file."""
    failures: list[Failure] = []
    for mode, wanted in thresholds.items():
        if mode not in results:
            raise ValueError(f"thresholds reference mode {mode!r} that was not evaluated")
        actual = metric_values(results[mode])
        for metric, required in wanted.items():
            if actual[metric] < required - _EPSILON:
                failures.append(Failure(mode, metric, actual[metric], required))
    return failures


def count_checks(thresholds: Thresholds) -> int:
    return sum(len(wanted) for wanted in thresholds.values())


# -- reporting: misses and run record --------------------------------------------------------


@dataclass(frozen=True)
class Trace:
    """What one query returned: the ranked corpus-relative paths and where the first relevant one landed."""

    query: Query
    ranked: tuple[str, ...]
    first_rank: int | None  # 1-based; None when no relevant document was returned


def trace_queries(searcher, queries: Sequence[Query], root: Path, mode: str, limit: int = max(KS)) -> list[Trace]:
    traces = []
    for query in queries:
        ranked = tuple(relative_path(h.path, root) for h in searcher.search(query.text, limit=limit, mode=mode))
        wanted = set(query.relevant)
        first = next((rank for rank, path in enumerate(ranked, start=1) if path in wanted), None)
        traces.append(Trace(query, ranked, first))
    return traces


def misses(traces: Iterable[Trace]) -> list[Trace]:
    """Queries whose first relevant document is not the top result."""
    return [t for t in traces if t.first_rank != 1]


def format_miss(trace: Trace, limit: int = max(KS), width: int = 72) -> list[str]:
    text = trace.query.text if len(trace.query.text) <= width else trace.query.text[: width - 1] + "…"
    where = f"rank {trace.first_rank}" if trace.first_rank else f"not in top {limit}"
    category = trace.query.category or "uncategorized"
    top = trace.ranked[0] if trace.ranked else "(no results)"
    return [
        f"  {category:<20} {where:<14} {text!r}",
        f"      top hit: {top}   wanted: {', '.join(trace.query.relevant)}",
    ]


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stable_bytes(path: Path) -> bytes:
    """File bytes, with line endings normalised for text so the same content hashes the same on every OS."""
    data = path.read_bytes()
    if path.suffix.lower() in {".pdf", ".docx", ".epub"}:
        return data
    return data.replace(b"\r\n", b"\n")


def dataset_fingerprint(path: str | Path) -> str:
    return _digest(_stable_bytes(Path(path)))


def corpus_fingerprint(root: Path) -> str:
    """Hash of every file's relative path and content, independent of platform line endings and file order."""
    lines = sorted(
        f"{p.relative_to(root).as_posix()}\0{_digest(_stable_bytes(p))}" for p in root.rglob("*") if p.is_file()
    )
    return _digest("\n".join(lines).encode("utf-8"))


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not installed"


def environment_info(
    model: str, settings, queries_path: str | Path, corpus_path: str | Path, root: Path, queries: Sequence[Query]
) -> dict:
    """Everything needed to tell whether two runs are comparable."""
    return {
        "model": model,
        "chunk_words": settings.chunk_words,
        "overlap_words": settings.overlap_words,
        "dataset": {"path": str(queries_path), "queries": len(queries), "sha256": dataset_fingerprint(queries_path)},
        "corpus": {
            "path": str(corpus_path),
            "files": sum(1 for p in root.rglob("*") if p.is_file()),
            "sha256": corpus_fingerprint(root),
        },
        "versions": {
            "python": platform.python_version(),
            "localseek": _version("localseek"),
            "fastembed": _version("fastembed"),
            "numpy": _version("numpy"),
        },
        "platform": platform.platform(),
    }


def format_environment(info: dict) -> list[str]:
    v, d, c = info["versions"], info["dataset"], info["corpus"]
    return [
        f"model:    {info['model']}",
        f"chunking: {info['chunk_words']} words, {info['overlap_words']} overlap",
        f"dataset:  {d['path']}  ({d['queries']} queries, sha256 {d['sha256'][:12]})",
        f"corpus:   {c['path']}  ({c['files']} files, sha256 {c['sha256'][:12]})",
        f"versions: python {v['python']}, localseek {v['localseek']}, fastembed {v['fastembed']}, numpy {v['numpy']}",
        f"platform: {info['platform']}",
    ]


def result_record(result: ModeResult) -> dict:
    values = metric_values(result)
    values["latency_ms"] = result.latency_ms
    return values
