"""File-type extractors. Add a new format by writing a function and registering its suffix."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .base import ExtractionError, Section
from .docx import extract_docx
from .html import extract_epub, extract_html
from .pdf import extract_pdf
from .text import TEXT_SUFFIXES, extract_text

Extractor = Callable[[Path], list[Section]]

_REGISTRY: dict[str, Extractor] = {suffix: extract_text for suffix in TEXT_SUFFIXES}
_REGISTRY.update(
    {
        ".pdf": extract_pdf,
        ".docx": extract_docx,
        ".html": extract_html,
        ".htm": extract_html,
        ".epub": extract_epub,
    }
)


def supported_suffixes() -> set[str]:
    return set(_REGISTRY)


def extract(path: Path) -> list[Section]:
    handler = _REGISTRY.get(path.suffix.lower())
    if handler is None:
        raise ExtractionError(f"unsupported file type: {path.suffix or '(none)'}")
    return handler(path)


__all__ = ["ExtractionError", "Section", "extract", "supported_suffixes"]
