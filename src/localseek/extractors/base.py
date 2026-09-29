from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Section:
    """A piece of extracted text, optionally tied to a page number."""

    text: str
    page: int | None = None


class ExtractionError(Exception):
    """Raised when a file cannot be turned into text."""
