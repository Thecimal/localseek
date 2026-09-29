from __future__ import annotations

from pathlib import Path

from .base import ExtractionError, Section


def extract_pdf(path: Path) -> list[Section]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise ExtractionError("pypdf is not installed") from exc

    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ExtractionError("encrypted PDF")
        sections: list[Section] = []
        for number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:  # a single bad page should not sink the file
                text = ""
            if text.strip():
                sections.append(Section(text, page=number))
    except ExtractionError:
        raise
    except Exception as exc:
        raise ExtractionError(f"could not read PDF: {exc}") from exc

    if not sections:
        raise ExtractionError("no extractable text (scanned PDF? OCR is not supported yet)")
    return sections
