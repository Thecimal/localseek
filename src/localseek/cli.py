"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from . import __version__
from .config import Settings, load_settings
from .embedder import Embedder, EmbedderUnavailable, get_embedder
from .indexer import IndexStats, index_paths
from .search import MODES, Searcher
from .store import ModelMismatch, Store


class Context:
    def __init__(self, args: argparse.Namespace) -> None:
        self.settings: Settings = load_settings(args.config)
        if args.model:
            self.settings.model = args.model
        db = Path(args.db).expanduser() if args.db else self.settings.resolved_db_path()
        self.store = Store(db)

    def indexing_embedder(self) -> Embedder:
        return get_embedder(self.settings.model)

    def search_embedder(self) -> Embedder:
        # Queries must be embedded by the same model that built the index.
        return get_embedder(self.store.get_meta("model") or self.settings.model)


def _short(path: str) -> str:
    home = str(Path.home())
    return "~" + path[len(home) :] if path.startswith(home) else path


def _print_summary(stats: IndexStats) -> None:
    print(
        f"Scanned {stats.scanned} files: {stats.added} added, {stats.updated} updated, "
        f"{stats.unchanged} unchanged, {stats.removed} removed ({stats.chunks} new chunks)."
    )
    for path, reason in (stats.skipped + stats.errors)[:10]:
        print(f"  skipped {_short(path)}: {reason}", file=sys.stderr)
    extra = len(stats.skipped) + len(stats.errors) - 10
    if extra > 0:
        print(f"  ...and {extra} more", file=sys.stderr)


def _progress(stats: IndexStats, path: str) -> None:
    if sys.stderr.isatty():
        line = f"\r\033[K{stats.scanned} files  {_short(path)[-70:]}"
        print(line, end="", file=sys.stderr, flush=True)


def _run_index(ctx: Context, roots: list[str], rebuild: bool) -> IndexStats:
    if rebuild:
        ctx.store.clear_index()
    stats = index_paths(ctx.store, ctx.indexing_embedder(), roots, ctx.settings, _progress)
    if sys.stderr.isatty():
        print("\r\033[K", end="", file=sys.stderr)
    return stats


def cmd_index(ctx: Context, args: argparse.Namespace) -> int:
    roots = [str(Path(p).expanduser().resolve()) for p in args.paths]
    for root in roots:
        ctx.store.add_root(root)
    _print_summary(_run_index(ctx, roots, args.rebuild))
    return 0


def cmd_update(ctx: Context, args: argparse.Namespace) -> int:
    roots = ctx.store.list_roots()
    if not roots:
        print("No folders registered yet. Run: localseek index <folder>", file=sys.stderr)
        return 1
    _print_summary(_run_index(ctx, roots, args.rebuild))
    return 0


def _parse_since(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").timestamp()
    except ValueError as exc:
        raise SystemExit(f"--since expects a date like 2026-01-31, got '{value}'") from exc


def cmd_search(ctx: Context, args: argparse.Namespace) -> int:
    if ctx.store.stats()["chunks"] == 0:
        print("The index is empty. Run: localseek index <folder>", file=sys.stderr)
        return 1
    searcher = Searcher(ctx.store, ctx.search_embedder())
    hits = searcher.search(
        " ".join(args.query),
        limit=args.n,
        mode=args.mode,
        ftype=args.type,
        since=_parse_since(args.since),
        path_contains=args.path,
    )
    if args.json:
        print(json.dumps([h.to_dict() for h in hits], indent=2, ensure_ascii=False))
        return 0
    if not hits:
        print("No results.")
        return 0
    for rank, hit in enumerate(hits, start=1):
        where = f"  p.{hit.page}" if hit.page else ""
        print(f"{rank}. {_short(hit.path)}{where}  ({hit.score:.2f})")
        print(f"   {hit.snippet}")
    return 0


def cmd_status(ctx: Context, args: argparse.Namespace) -> int:
    stats = ctx.store.stats()
    print(f"Index:   {ctx.store.path}")
    print(f"Model:   {stats['model'] or '(none yet)'}")
    print(f"Files:   {stats['files']}    Chunks: {stats['chunks']}")
    print(f"Size:    {int(stats['db_bytes']) / 1_048_576:.1f} MB")
    roots = stats["roots"]
    print("Folders: " + (", ".join(_short(r) for r in roots) if roots else "(none)"))  # type: ignore[union-attr]
    return 0


def cmd_forget(ctx: Context, args: argparse.Namespace) -> int:
    root = Path(args.path).expanduser().resolve()
    ctx.store.remove_root(str(root))
    removed = 0
    for path in list(ctx.store.file_records()):
        if Path(path).is_relative_to(root):
            ctx.store.delete_file(path)
            removed += 1
    print(f"Removed {removed} files under {_short(str(root))} from the index.")
    return 0


def cmd_watch(ctx: Context, args: argparse.Namespace) -> int:
    roots = ctx.store.list_roots()
    if not roots:
        print("No folders registered yet. Run: localseek index <folder>", file=sys.stderr)
        return 1
    embedder = ctx.indexing_embedder()
    print(f"Watching {len(roots)} folder(s) every {args.interval}s. Press Ctrl+C to stop.")
    try:
        while True:
            stats = index_paths(ctx.store, embedder, roots, ctx.settings)
            if stats.added or stats.updated or stats.removed:
                stamp = time.strftime("%H:%M:%S")
                print(
                    f"[{stamp}] {stats.added} added, {stats.updated} updated, "
                    f"{stats.removed} removed"
                )
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print()
        return 0


def cmd_serve(ctx: Context, args: argparse.Namespace) -> int:
    from .web import serve

    if ctx.store.stats()["chunks"] == 0:
        print("The index is empty. Run: localseek index <folder>", file=sys.stderr)
        return 1
    serve(Searcher(ctx.store, ctx.search_embedder()), port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="localseek", description="Private, on-device semantic search for your documents."
    )
    parser.add_argument("--version", action="version", version=f"localseek {__version__}")
    parser.add_argument("--db", help="path to the index database")
    parser.add_argument("--config", help="path to a config.toml")
    parser.add_argument("--model", help="embedding model to use when building the index")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("index", help="index one or more folders and remember them")
    p.add_argument("paths", nargs="+")
    p.add_argument("--rebuild", action="store_true", help="discard the index and start over")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("update", help="re-scan every remembered folder")
    p.add_argument("--rebuild", action="store_true", help="discard the index and start over")
    p.set_defaults(func=cmd_update)

    p = sub.add_parser("search", help="search the index")
    p.add_argument("query", nargs="+")
    p.add_argument("-n", type=int, default=10, help="number of results (default 10)")
    p.add_argument("--mode", choices=MODES, default="hybrid")
    p.add_argument("--type", help="only this file type, e.g. pdf")
    p.add_argument("--since", help="only files modified on or after YYYY-MM-DD")
    p.add_argument("--path", help="only paths containing this text")
    p.add_argument("--json", action="store_true", help="print results as JSON")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("status", help="show index size and registered folders")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("forget", help="stop tracking a folder and remove its files from the index")
    p.add_argument("path")
    p.set_defaults(func=cmd_forget)

    p = sub.add_parser("watch", help="keep the index up to date by re-scanning periodically")
    p.add_argument("--interval", type=int, default=30, help="seconds between scans")
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser("serve", help="start the local search page on 127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        ctx = Context(args)
        return args.func(ctx, args)
    except ModelMismatch as exc:
        print(f"{exc}\nRun again with --rebuild to re-index with the new model.", file=sys.stderr)
        return 2
    except EmbedderUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
