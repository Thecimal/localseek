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
import re
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
    split: str | None = None


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
    pure = PurePosixPath(value)
    if pure.is_absolute() or re.match(r"[A-Za-z]:[\\/]", value) or value.startswith("\\\\"):
        return "must be relative to the corpus, not absolute"  # POSIX, Windows drive, or UNC path
    if "\\" in value:
        return "use forward slashes"
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
        split = item.get("split")
        if "split" in item and (not isinstance(split, str) or not split.strip()):
            problems.append(f"{label}: 'split' must be a non-empty string when present")

        if len(problems) == before:
            queries.append(Query(text.strip(), tuple(relevant), category, split))
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

class ThresholdError(DatasetError):
    """The thresholds file is unusable."""


@dataclass(frozen=True)
class Thresholds:
    """Minimum metrics: `overall` is {mode: {metric: min}}; `by_category` is {mode: {category: {metric: min}}}."""

    overall: dict[str, dict[str, float]]
    by_category: dict[str, dict[str, dict[str, float]]]
    meta: dict

    @property
    def checks(self) -> int:
        categories = sum(len(m) for cats in self.by_category.values() for m in cats.values())
        return sum(len(m) for m in self.overall.values()) + categories


@dataclass(frozen=True)
class Failure:
    mode: str
    metric: str
    actual: float
    required: float
    category: str | None = None

    def __str__(self) -> str:
        scope = f"{self.mode} {self.category}" if self.category else self.mode
        return f"{scope} {self.metric}: {self.actual:.3f} < required {self.required:.3f}"


def metric_names(ks: Sequence[int] = KS) -> list[str]:
    return [f"recall@{k}" for k in ks] + ["mrr"]


def metric_values(result: ModeResult) -> dict[str, float]:
    values = {f"recall@{k}": v for k, v in result.recall.items()}
    values["mrr"] = result.mrr
    return values


def _parse_metrics(label: str, wanted: object, valid_metrics: Sequence[str], problems: list[str]) -> dict[str, float]:
    parsed: dict[str, float] = {}
    if not isinstance(wanted, dict) or not wanted:
        problems.append(f"{label}: must be a non-empty object mapping metric names to minimum values")
        return parsed
    for metric, value in wanted.items():
        if metric not in valid_metrics:
            problems.append(f"{label}: unknown metric {metric!r} (expected one of: {', '.join(valid_metrics)})")
        elif isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
            problems.append(f"{label} {metric}: threshold must be a number, got {value!r}")
        elif not 0 <= value <= 1:
            problems.append(f"{label} {metric}: threshold must be between 0 and 1, got {value!r}")
        else:
            parsed[metric] = float(value)
    return parsed


def load_thresholds(
    path: str | Path, modes: Sequence[str], ks: Sequence[int] = KS, categories: Iterable[str] | None = None
) -> Thresholds:
    """Read minimum required metrics.

        {"hybrid": {"recall@1": 0.8, "mrr": 0.85},
         "by_category": {"hybrid": {"late_answer": {"recall@5": 0.9}}},
         "meta": {"dataset_sha256": "..."}}

    Top-level modes give overall minimums; "by_category" gives per-category minimums for categories that exist in
    the dataset; "meta" is free-form provenance (its fingerprints are checked against the run). Anything not listed
    is not checked. Raises ThresholdError listing every problem.
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

    known_categories = set(categories or ())
    valid_metrics = metric_names(ks)
    problems: list[str] = []
    overall: dict[str, dict[str, float]] = {}
    by_category: dict[str, dict[str, dict[str, float]]] = {}
    meta: dict = {}
    for key, wanted in data.items():
        if key == "meta":
            if isinstance(wanted, dict):
                meta = wanted
            else:
                problems.append("meta: must be an object")
        elif key == "by_category":
            if not isinstance(wanted, dict) or not wanted:
                problems.append("by_category: must be a non-empty object mapping modes to categories")
                continue
            for mode, cats in wanted.items():
                if mode not in modes:
                    problems.append(f"by_category: unknown mode {mode!r} (expected one of: {', '.join(modes)})")
                elif not isinstance(cats, dict) or not cats:
                    problems.append(f"by_category {mode}: must be a non-empty object mapping categories to metrics")
                else:
                    by_category[mode] = {}
                    for category, metrics in cats.items():
                        if category not in known_categories:
                            known = ", ".join(sorted(known_categories)) or "none: the dataset has no categories"
                            problems.append(f"by_category {mode}: unknown category {category!r} (dataset has: {known})")
                            continue
                        by_category[mode][category] = _parse_metrics(
                            f"{mode} {category}", metrics, valid_metrics, problems
                        )
        elif key not in modes:
            problems.append(f"unknown mode {key!r} (expected one of: {', '.join(modes)}, by_category, meta)")
        else:
            overall[key] = _parse_metrics(key, wanted, valid_metrics, problems)
    if not problems and not overall and not by_category:
        problems.append("no thresholds defined; a gate that checks nothing is not allowed")
    if problems:
        raise ThresholdError(problems)
    return Thresholds(overall, by_category, meta)


def check_thresholds(results: dict[str, ModeResult], thresholds: dict[str, dict[str, float]]) -> list[Failure]:
    """Every overall threshold the results fail to meet, ordered by mode then metric as listed in the file."""
    failures: list[Failure] = []
    for mode, wanted in thresholds.items():
        if mode not in results:
            raise ValueError(f"thresholds reference mode {mode!r} that was not evaluated")
        actual = metric_values(results[mode])
        for metric, required in wanted.items():
            if actual[metric] < required - _EPSILON:
                failures.append(Failure(mode, metric, actual[metric], required))
    return failures


def check_category_thresholds(
    results: dict[str, dict[str, ModeResult]], thresholds: dict[str, dict[str, dict[str, float]]]
) -> list[Failure]:
    """Every per-category threshold the results fail to meet."""
    failures: list[Failure] = []
    for mode, categories in thresholds.items():
        for category, wanted in categories.items():
            if category not in results.get(mode, {}):
                raise ValueError(f"thresholds reference {mode}/{category}, which was not evaluated")
            actual = metric_values(results[mode][category])
            for metric, required in wanted.items():
                if actual[metric] < required - _EPSILON:
                    failures.append(Failure(mode, metric, actual[metric], required, category))
    return failures


def check_gate(
    results: dict[str, ModeResult],
    category_results: dict[str, dict[str, ModeResult]] | None,
    thresholds: Thresholds,
) -> list[Failure]:
    failures = check_thresholds(results, thresholds.overall)
    if thresholds.by_category:
        if category_results is None:
            raise ValueError("category thresholds need per-category results")
        failures += check_category_thresholds(category_results, thresholds.by_category)
    return failures


def count_checks(thresholds: Thresholds) -> int:
    return thresholds.checks


def provenance_problems(meta: dict, info: dict) -> list[str]:
    """Thresholds derived for one version of the benchmark must not gate another."""
    problems = []
    for key, actual, label in (
        ("dataset_sha256", info["dataset"]["sha256"], "query file"),
        ("corpus_sha256", info["corpus"]["sha256"], "corpus"),
    ):
        expected = meta.get(key)
        if expected is not None and expected != actual:
            problems.append(
                f"thresholds were derived for a {label} with sha256 {str(expected)[:12]}, but this run uses "
                f"{actual[:12]}; re-derive them with eval/derive_thresholds.py"
            )
    return problems


# -- search tuning (evaluation experiments) ------------------------------------------------------------------------


class TuningError(DatasetError):
    """The --tuning specification is unusable."""


_TUNING_OPTIONS = {
    "rrf_k": ("int, at least 1", lambda v: int(v), lambda v: v >= 1),
    "vector_weight": ("number above 0", lambda v: float(v), lambda v: 0 < v < math.inf),
    "keyword_weight": ("number above 0", lambda v: float(v), lambda v: 0 < v < math.inf),
    "keyword_limit": ("int, at least 1", lambda v: int(v), lambda v: v >= 1),
    "drop_stopwords": ("true or false", lambda v: {"true": True, "false": False}[v.lower()], lambda v: True),
}


def parse_tuning(spec: str | None) -> dict:
    """Parse "rrf_k=10,vector_weight=2" into Searcher keyword arguments; None or "" means the shipped defaults."""
    if not spec or not spec.strip():
        return {}
    tuning: dict = {}
    seen: set[str] = set()
    problems: list[str] = []
    for part in spec.split(","):
        name, separator, raw = (piece.strip() for piece in part.partition("="))
        if not separator or not raw:
            problems.append(f"{part.strip()!r}: expected name=value")
        elif name not in _TUNING_OPTIONS:
            problems.append(f"unknown option {name!r} (expected one of: {', '.join(_TUNING_OPTIONS)})")
        elif name in seen:
            problems.append(f"{name} is given more than once")
        else:
            seen.add(name)
            description, convert, valid = _TUNING_OPTIONS[name]
            try:
                value = convert(raw)
            except (ValueError, KeyError):
                problems.append(f"{name}: {raw!r} is not valid ({description})")
                continue
            if valid(value):
                tuning[name] = value
            else:
                problems.append(f"{name}: {raw!r} is not valid ({description})")
    if problems:
        raise TuningError(problems)
    return tuning


def format_tuning(tuning: dict) -> str:
    def show(value) -> str:
        return str(value).lower() if isinstance(value, bool) else str(value)

    return ", ".join(f"{name}={show(value)}" for name, value in tuning.items())


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


def per_query_records(searcher, queries: Sequence[Query], root: Path, mode: str, ks: Sequence[int] = KS) -> list[dict]:
    """One record per query: where the first relevant document landed and each query's own metrics.

    These are what paired comparisons are built from; averages alone cannot say whether two runs differ by chance.
    """
    records = []
    for trace in trace_queries(searcher, queries, root, mode, limit=max(ks)):
        score = score_query(list(trace.ranked), trace.query.relevant, ks)
        records.append(
            {
                "query": trace.query.text,
                "category": trace.query.category,
                "first_rank": trace.first_rank,
                **{f"recall@{k}": value for k, value in score.recall.items()},
                "rr": score.reciprocal_rank,
            }
        )
    return records


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
    model: str,
    settings,
    queries_path: str | Path,
    corpus_path: str | Path,
    root: Path,
    queries: Sequence[Query],
    split: str | None = None,
    tuning: dict | None = None,
) -> dict:
    """Everything needed to tell whether two runs are comparable."""
    return {
        "model": model,
        "tuning": dict(tuning or {}),
        "chunk_words": settings.chunk_words,
        "overlap_words": settings.overlap_words,
        "dataset": {
            "path": str(queries_path),
            "queries": len(queries),
            "split": split,
            "sha256": dataset_fingerprint(queries_path),
        },
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


def _split_note(dataset: dict) -> str:
    return ", split " + dataset["split"] if dataset.get("split") else ""


def format_environment(info: dict) -> list[str]:
    v, d, c = info["versions"], info["dataset"], info["corpus"]
    return [
        f"model:    {info['model']}",
        f"chunking: {info['chunk_words']} words, {info['overlap_words']} overlap",
        *([f"tuning:   {format_tuning(info['tuning'])}"] if info.get("tuning") else []),
        f"dataset:  {d['path']}  ({d['queries']} queries{_split_note(d)}, sha256 {d['sha256'][:12]})",
        f"corpus:   {c['path']}  ({c['files']} files, sha256 {c['sha256'][:12]})",
        f"versions: python {v['python']}, localseek {v['localseek']}, fastembed {v['fastembed']}, numpy {v['numpy']}",
        f"platform: {info['platform']}",
    ]


def result_record(result: ModeResult) -> dict:
    values = metric_values(result)
    values["latency_ms"] = result.latency_ms
    return values
