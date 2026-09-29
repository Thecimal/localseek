"""Walk folders and yield files worth indexing."""

from __future__ import annotations

import fnmatch
import os
import stat
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

IGNORE_FILE = ".localseekignore"


@dataclass(frozen=True)
class FileInfo:
    path: Path
    size: int
    mtime: float


def load_ignore_file(root: Path) -> list[str]:
    """Read simple glob patterns, one per line. Lines starting with # are comments."""
    try:
        lines = (root / IGNORE_FILE).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    return [ln.strip().rstrip("/") for ln in lines if ln.strip() and not ln.startswith("#")]


def _ignored(relative: Path, name: str, patterns: list[str]) -> bool:
    posix = relative.as_posix()
    return any(fnmatch.fnmatch(name, p) or fnmatch.fnmatch(posix, p) for p in patterns)


def scan(
    roots: Iterable[str | os.PathLike[str]],
    ignore: list[str],
    max_bytes: int,
    suffixes: set[str],
    on_skip: Callable[[str, str], None] | None = None,
) -> Iterator[FileInfo]:
    def skip(path: str, reason: str) -> None:
        if on_skip:
            on_skip(path, reason)

    for raw_root in roots:
        root = Path(raw_root).expanduser().resolve()
        if root.is_file():
            if root.suffix.lower() in suffixes:
                st = root.stat()
                yield FileInfo(root, st.st_size, st.st_mtime)
            continue
        if not root.is_dir():
            skip(str(root), "not found")
            continue

        patterns = list(ignore) + load_ignore_file(root)
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            relative_dir = Path(dirpath).relative_to(root)
            dirnames[:] = sorted(
                d for d in dirnames if not _ignored(relative_dir / d, d, patterns)
            )
            for name in sorted(filenames):
                if _ignored(relative_dir / name, name, patterns):
                    continue
                path = Path(dirpath) / name
                if path.suffix.lower() not in suffixes:
                    continue
                try:
                    st = path.stat()
                except OSError:
                    skip(str(path), "cannot stat")
                    continue
                if not stat.S_ISREG(st.st_mode):
                    continue
                if st.st_size > max_bytes:
                    skip(str(path), "too large")
                    continue
                yield FileInfo(path, st.st_size, st.st_mtime)
