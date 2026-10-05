"""Write a review sheet for a query file, so someone other than the author can check the relevance judgments.

    python eval/make_review_sheet.py --queries eval/benchmark/queries-v3.json --out review-sheet.md

For every question the sheet lists the documents judged to answer it, with the exact quote that supports each, and
two checkboxes: does the quote answer the question, and does any *other* document answer it as well. The reviewer
needs only the corpus folder (eval/benchmark/corpus) and a text search.

Exit status: 0 success, 2 unreadable or malformed query file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HEADER = """# Review sheet

Source: `{source}` ({count} questions). Documents are in `eval/benchmark/corpus/`.

You are checking someone else's work, so be sceptical. For each question:

1. Read the question as a person would ask it, without looking at the listed quote first.
2. Open each listed document and check that the quoted text really answers the question as asked.
3. Search the corpus (`grep -ril "<word>" eval/benchmark/corpus` is enough) for **other** documents that also answer it.
   A document counts only if it contains a substantive answer; an incidental mention does not. Out-of-date
   information (for example an old password for a retired router) does not count.
4. Mark the boxes and note anything unclear. A question you cannot answer without outside knowledge is worth flagging.

Do not run any search tool on these questions while reviewing; the point is a judgment independent of the software.

"""


def render(items: list[dict], source: str) -> str:
    parts = [HEADER.format(source=source, count=len(items))]
    for number, item in enumerate(items, start=1):
        parts.append(f"## {number}. {item['query']}\n")
        parts.append("Judged to answer it:\n")
        for path, phrase in item["evidence"].items():
            parts.append(f'- `{path}`: "{phrase}"')
        parts.append("")
        parts.append("- [ ] The quoted text answers the question")
        parts.append("- [ ] No other document answers it (if one does, write its path here: ______)")
        parts.append("- Notes:\n")
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        items = json.loads(Path(args.queries).read_text(encoding="utf-8"))
        for item in items:
            item["query"], item["evidence"]  # noqa: B018 - fail early on files without evidence
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"error: cannot use {args.queries}: {exc!r}", file=sys.stderr)
        print("  (every item needs 'query' and 'evidence')", file=sys.stderr)
        return 2
    Path(args.out).write_text(render(items, Path(args.queries).name), encoding="utf-8")
    print(f"Wrote {args.out}: {len(items)} questions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
