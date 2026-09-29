from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from .base import ExtractionError, Section

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def extract_docx(path: Path) -> list[Section]:
    try:
        with zipfile.ZipFile(path) as archive:
            xml = archive.read("word/document.xml")
        root = ET.fromstring(xml)
    except (zipfile.BadZipFile, KeyError, OSError, ET.ParseError) as exc:
        raise ExtractionError(f"could not read docx: {exc}") from exc

    paragraphs: list[str] = []
    for para in root.iter(f"{_W}p"):
        parts: list[str] = []
        for node in para.iter():
            if node.tag == f"{_W}t":
                parts.append(node.text or "")
            elif node.tag in (f"{_W}tab", f"{_W}br"):
                parts.append(" ")
        line = "".join(parts).strip()
        if line:
            paragraphs.append(line)

    text = "\n\n".join(paragraphs)
    return [Section(text)] if text.strip() else []
