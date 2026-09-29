from __future__ import annotations

from pathlib import Path

from .base import ExtractionError, Section

TEXT_SUFFIXES = {
    ".txt", ".md", ".markdown", ".rst", ".org", ".log", ".csv", ".tsv", ".json",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".tex",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".c", ".h", ".cpp",
    ".hpp", ".cs", ".rb", ".php", ".sh", ".sql", ".kt", ".swift",
}  # fmt: skip


def extract_text(path: Path) -> list[Section]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ExtractionError(str(exc)) from exc
    if b"\x00" in data[:8192]:
        raise ExtractionError("looks like a binary file")
    text = data.decode("utf-8-sig", errors="replace")
    return [Section(text)] if text.strip() else []
