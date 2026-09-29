"""Split extracted text into overlapping, roughly word-count-sized chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .extractors import Section

_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_MIN_FILL = 0.6  # flush at a paragraph boundary once a chunk is at least this full


@dataclass
class Chunk:
    text: str
    ordinal: int = 0
    page: int | None = None
    heading: str | None = None


def chunk_sections(
    sections: list[Section], max_words: int = 220, overlap_words: int = 30
) -> list[Chunk]:
    if max_words < 20:
        raise ValueError("max_words must be at least 20")
    overlap = max(0, min(overlap_words, max_words // 2))
    chunks: list[Chunk] = []
    for section in sections:
        chunks.extend(_chunk_text(section.text, section.page, max_words, overlap))
    for index, chunk in enumerate(chunks):
        chunk.ordinal = index
    return chunks


def _chunk_text(text: str, page: int | None, max_words: int, overlap: int) -> list[Chunk]:
    out: list[Chunk] = []
    buf: list[str] = []
    fresh = 0  # words added since the last flush (excludes carried-over overlap)
    heading: str | None = None

    def flush() -> None:
        nonlocal fresh
        if fresh > 0:
            out.append(Chunk(" ".join(buf), page=page, heading=heading))
        buf[:] = buf[-overlap:] if overlap else []
        fresh = 0

    for paragraph in _PARAGRAPH_BREAK.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        match = _HEADING.match(paragraph.splitlines()[0])
        if match and match.group(1):
            heading = match.group(1)
        words = paragraph.split()

        if fresh >= max_words * _MIN_FILL and len(buf) + len(words) > max_words:
            flush()

        position = 0
        while position < len(words):
            take = words[position : position + (max_words - len(buf))]
            buf.extend(take)
            fresh += len(take)
            position += len(take)
            if len(buf) >= max_words:
                flush()

    flush()
    return out
