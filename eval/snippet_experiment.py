"""Pre-registered experiment on snippet selection (eval/benchmark/snippet-experiment.md).

    python eval/snippet_experiment.py --out eval/benchmark/experiments/snippets.json

A search result shows a window cut from the chunk it came from. This asks: given the *correct* chunk, which way of
choosing the window shows the answer most often? Every candidate is a pure function of the chunk text and the query,
so the experiment is deterministic and needs no embedding model. The evaluation set is every (question, relevant
document) pair in queries.json and queries-v3.json whose correct chunk is longer than the window (shorter chunks
are always shown whole, so display cannot fail on them).

Exit status: 0 on completion, 2 for unusable input.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from compare_runs import sign_test_p

from localseek.chunker import chunk_sections
from localseek.config import Settings
from localseek.extractors import extract, supported_suffixes
from localseek.search import STOPWORDS, make_snippet, query_terms

HERE = Path(__file__).resolve().parent
BENCH = HERE / "benchmark"
BASE_WIDTH = 280
WIDE = 420  # 1.5 times the current window: the most extra screen space any candidate may use
MIN_NET_WINS = 8
MAX_P = 0.05
MAX_LOSSES = 3
MIN_NET_OVER_CONTROL = 4
LEADING = 60  # characters of context before the anchor, as in the shipped function


def content_terms(terms: list[str]) -> list[str]:
    """Terms that are not common function words; if every term is one, keep them all rather than none."""
    return [t for t in terms if t not in STOPWORDS] or terms


def _cut(text: str, start: int, width: int) -> str:
    if start > 0:
        space = text.find(" ", start)
        start = space + 1 if space != -1 else start
    end = min(len(text), start + width)
    return ("…" if start > 0 else "") + text[start:end].strip() + ("…" if end < len(text) else "")


def _occurrences(lowered: str, term: str):
    position = lowered.find(term)
    while position >= 0:
        yield position
        position = lowered.find(term, position + 1)


def densest_window(text: str, terms: list[str], width: int = BASE_WIDTH) -> str:
    """The window holding the most distinct content terms; ties go to the earliest; no term found means the start."""
    text = " ".join(text.split())
    if len(text) <= width:
        return text
    lowered = text.lower()
    found = {t: list(_occurrences(lowered, t)) for t in content_terms(terms)}
    starts = sorted({max(0, p - LEADING) for positions in found.values() for p in positions})
    best_start, best_count = 0, 0
    for start in starts:
        count = sum(
            1 for t, positions in found.items() if any(start <= p and p + len(t) <= start + width for p in positions)
        )
        if count > best_count:
            best_start, best_count = start, count
    return _cut(text, best_start, width)


# id: (description, maximum snippet width, function(chunk text, query terms) -> snippet)
STRATEGIES = {
    "S0": ("the shipped function: 280 characters from the first occurrence of any query word",
           BASE_WIDTH, lambda text, terms: make_snippet(text, terms)),
    "S1": ("S0, but anchored on the first occurrence of a content word (function words ignored)",
           BASE_WIDTH, lambda text, terms: make_snippet(text, content_terms(terms))),
    "S2": ("280 characters around the densest cluster of distinct content words",
           BASE_WIDTH, lambda text, terms: densest_window(text, terms, BASE_WIDTH)),
    "S3": ("S2 with a 420-character window",
           WIDE, lambda text, terms: densest_window(text, terms, WIDE)),
    "S4": ("control: S0 with a 420-character window",
           WIDE, lambda text, terms: make_snippet(text, terms, width=WIDE)),
}


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def build_pairs(corpus: Path, query_files: list[Path], settings: Settings | None = None) -> list[dict]:
    """(question, relevant document) pairs whose correct chunk is longer than the window."""
    settings = settings or Settings()
    chunks = {}
    for path in sorted(p for p in corpus.rglob("*") if p.is_file() and p.suffix.lower() in supported_suffixes()):
        sections = extract(path)
        chunks[path.relative_to(corpus).as_posix()] = [
            c.text for c in chunk_sections(sections, settings.chunk_words, settings.overlap_words)
        ]
    pairs = []
    for file in query_files:
        for item in json.loads(file.read_text(encoding="utf-8")):
            for path, quote in item["evidence"].items():
                wanted = _norm(quote)
                chunk = next((c for c in chunks[path] if wanted in _norm(c)), None)
                if chunk is None:
                    raise ValueError(f"{file.name}: the quote for {item['query']!r} is in no chunk of {path}")
                if len(" ".join(chunk.split())) > BASE_WIDTH:
                    pairs.append(
                        {
                            "set": file.stem,
                            "category": item.get("category") or "-",
                            "query": item["query"],
                            "path": path,
                            "terms": query_terms(item["query"]),
                            "chunk": chunk,
                            "quote": wanted,
                        }
                    )
    return pairs


def evaluate(pairs: list[dict]) -> dict[str, list[bool]]:
    shown = {}
    for sid, (_, width, function) in STRATEGIES.items():
        outcomes = []
        for pair in pairs:
            snippet = function(pair["chunk"], pair["terms"])
            if len(snippet) > width + 2:  # the window plus its two ellipses
                raise ValueError(f"{sid} produced {len(snippet)} characters, more than its {width}-character limit")
            outcomes.append(pair["quote"] in _norm(snippet))
        shown[sid] = outcomes
    return shown


def compare(a: list[bool], b: list[bool]) -> dict:
    """b against a, per pair."""
    wins = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    losses = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    return {"wins": wins, "losses": losses, "net": wins - losses, "p": sign_test_p(wins, losses)}


def decide(shown: dict[str, list[bool]]) -> dict:
    """The pre-registered decision rule. Returns per-candidate verdicts and the selected id (or None)."""
    verdicts = {}
    for sid in STRATEGIES:
        if sid == "S0":
            continue
        v = compare(shown["S0"], shown[sid])
        v["eligible"] = v["net"] >= MIN_NET_WINS and v["p"] < MAX_P and v["losses"] <= MAX_LOSSES
        v["vs_control"] = compare(shown["S4"], shown[sid]) if sid != "S4" else None
        verdicts[sid] = v
    contenders = []
    for sid, v in verdicts.items():
        if not v["eligible"]:
            continue
        beats_control = sid == "S4" or not verdicts["S4"]["eligible"] or v["vs_control"]["net"] >= MIN_NET_OVER_CONTROL
        v["survives_control"] = beats_control
        if beats_control:
            contenders.append(sid)
    order = list(STRATEGIES)
    selected = min(
        contenders, key=lambda s: (STRATEGIES[s][1], -sum(shown[s]), order.index(s)), default=None
    )
    return {"verdicts": verdicts, "selected": selected}


def subsets(pairs: list[dict]) -> dict[str, list[int]]:
    groups = defaultdict(list)
    for index, pair in enumerate(pairs):
        groups["all"].append(index)
        groups[pair["set"]].append(index)
        if pair["category"] == "late_answer":
            groups["late_answer (v1)"].append(index)
        if pair["set"] == "queries-v3":
            groups["paraphrase (v3)"].append(index)
    return dict(groups)


def render(pairs: list[dict], shown: dict[str, list[bool]], outcome: dict) -> list[str]:
    groups = subsets(pairs)
    names = list(groups)
    lines = [f"{len(pairs)} pairs whose correct chunk is longer than {BASE_WIDTH} characters", ""]
    lines.append(f"{'id':<5}{'width':<7}" + "".join(f"{n:<20}" for n in names) + "vs S0: wins/losses  p      verdict")
    for sid, (_, width, _) in STRATEGIES.items():
        cells = "".join(f"{sum(shown[sid][i] for i in groups[n])}/{len(groups[n])}".ljust(20) for n in names)
        if sid == "S0":
            lines.append(f"{sid:<5}{width:<7}{cells}-")
            continue
        v = outcome["verdicts"][sid]
        verdict = "eligible" if v["eligible"] else "not eligible"
        if v["eligible"] and not v["survives_control"]:
            verdict += ", does not beat the wide control"
        lines.append(f"{sid:<5}{width:<7}{cells}{v['wins']}/{v['losses']:<16}{v['p']:<7.3f}{verdict}")
    lines.append("")
    chosen = outcome["selected"]
    lines.append(f"Selected: {chosen}" if chosen else "No candidate qualifies: keep the shipped snippet.")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--corpus", default=str(BENCH / "corpus"))
    parser.add_argument("--queries", action="append", help="query file with evidence (repeatable)")
    parser.add_argument("--out", help="write the full record, including every pair's outcome, to this JSON file")
    args = parser.parse_args()
    files = [Path(q) for q in args.queries] if args.queries else [BENCH / "queries.json", BENCH / "queries-v3.json"]
    try:
        pairs = build_pairs(Path(args.corpus), files)
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not pairs:
        print("error: no (question, document) pairs with a chunk longer than the window", file=sys.stderr)
        return 2
    shown = evaluate(pairs)
    outcome = decide(shown)
    print("\n".join(render(pairs, shown, outcome)))
    if args.out:
        record = {
            "pairs": [{k: v for k, v in p.items() if k not in ("chunk", "terms")} for p in pairs],
            "shown": shown,
            "verdicts": outcome["verdicts"],
            "selected": outcome["selected"],
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
