from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np

from localseek.embedder import HashEmbedder


class CountingEmbedder(HashEmbedder):
    """HashEmbedder that records how many passages it was asked to embed."""

    def __init__(self) -> None:
        super().__init__()
        self.passages_embedded = 0

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        self.passages_embedded += len(texts)
        return super().embed_passages(texts)


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def make_docx(path: Path, paragraphs: list[str]) -> Path:
    body = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs)
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}</w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", xml)
    return path


def make_epub(path: Path, chapters: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, html in chapters.items():
            archive.writestr(name, html)
    return path


def make_pdf(path: Path, pages: list[str]) -> Path:
    """Write a minimal text PDF, one line of text per page."""
    objects: list[bytes] = []
    page_count = len(pages)
    kids = " ".join(f"{3 + i * 2} 0 R" for i in range(page_count))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>".encode())
    font_id = 3 + page_count * 2
    for i, text in enumerate(pages):
        content_id = 4 + i * 2
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode()
        )
        stream = f"BT /F1 12 Tf 72 700 Td ({text}) Tj ET".encode()
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(f"{number} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets:
        out.write(f"{offset:010d} 00000 n \n".encode())
    out.write(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    path.write_bytes(out.getvalue())
    return path
