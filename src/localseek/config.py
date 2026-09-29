"""Settings and default file locations."""

from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

DEFAULT_IGNORE = [
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".idea",
    ".cache",
    ".DS_Store",
    "*.pyc",
]


def config_dir() -> Path:
    override = os.environ.get("LOCALSEEK_CONFIG_DIR")
    if override:
        return Path(override)
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "localseek"


def data_dir() -> Path:
    override = os.environ.get("LOCALSEEK_DATA_DIR")
    if override:
        return Path(override)
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "localseek"


@dataclass
class Settings:
    model: str = DEFAULT_MODEL
    chunk_words: int = 220
    overlap_words: int = 30
    max_file_mb: float = 50.0
    ignore: list[str] = field(default_factory=lambda: list(DEFAULT_IGNORE))
    db_path: str | None = None

    @property
    def max_file_bytes(self) -> int:
        return int(self.max_file_mb * 1024 * 1024)

    def resolved_db_path(self) -> Path:
        if self.db_path:
            return Path(self.db_path).expanduser()
        return data_dir() / "index.db"


def load_settings(path: str | os.PathLike[str] | None = None) -> Settings:
    """Load settings from a TOML file. A missing default file means defaults."""
    explicit = path or os.environ.get("LOCALSEEK_CONFIG")
    config_path = Path(explicit).expanduser() if explicit else config_dir() / "config.toml"
    settings = Settings()
    if not config_path.exists():
        if explicit:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        return settings

    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    known = {f.name for f in fields(Settings)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"Unknown setting(s) in {config_path}: {', '.join(unknown)}")

    for key, value in data.items():
        if key == "ignore":
            settings.ignore = list(DEFAULT_IGNORE) + [str(v) for v in value]
        else:
            setattr(settings, key, value)

    if settings.chunk_words < 20:
        raise ValueError("chunk_words must be at least 20")
    if not 0 <= settings.overlap_words <= settings.chunk_words // 2:
        raise ValueError("overlap_words must be between 0 and chunk_words / 2")
    return settings
