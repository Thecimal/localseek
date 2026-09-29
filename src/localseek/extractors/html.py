from __future__ import annotations

import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path

from .base import ExtractionError, Section

_SKIP = {"script", "style", "noscript", "template"}
_BLOCK = {
    "p", "div", "br", "li", "tr", "section", "article", "blockquote", "pre",
    "ul", "ol", "table", "h1", "h2", "h3", "h4", "h5", "h6", "title",
}  # fmt: skip


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in _SKIP:
            self._skip_depth += 1
        elif tag in _BLOCK:
            self.parts.append("\n\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in _BLOCK:
            self.parts.append("\n\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)


def html_to_text(markup: str) -> str:
    parser = _TextParser()
    parser.feed(markup)
    parser.close()
    lines = [" ".join(line.split()) for line in "".join(parser.parts).splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def extract_html(path: Path) -> list[Section]:
    try:
        markup = path.read_bytes().decode("utf-8", errors="replace")
    except OSError as exc:
        raise ExtractionError(str(exc)) from exc
    text = html_to_text(markup)
    return [Section(text)] if text else []


def extract_epub(path: Path) -> list[Section]:
    """Reads chapters in file-name order, which matches reading order for most books."""
    try:
        with zipfile.ZipFile(path) as archive:
            names = sorted(n for n in archive.namelist() if n.lower().endswith((".xhtml", ".html", ".htm")))
            sections: list[Section] = []
            for name in names:
                text = html_to_text(archive.read(name).decode("utf-8", errors="replace"))
                if text:
                    sections.append(Section(text))
    except (zipfile.BadZipFile, OSError) as exc:
        raise ExtractionError(f"could not read epub: {exc}") from exc
    return sections
