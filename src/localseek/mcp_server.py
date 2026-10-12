"""A read-only MCP server, so an AI assistant can search the local index.

`IndexTools` holds the tool logic and does not need the MCP SDK. `build_server` and `run_stdio` connect it to the
SDK, which is an optional dependency (`pip install "localseek[mcp]"`) and is imported only when the server starts.

The server speaks MCP over stdio, so nothing here may print to stdout. It exposes two tools, `search` and `status`,
and nothing that indexes, changes or deletes anything.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import datetime

from .embedder import Embedder, EmbedderUnavailable
from .search import MODES, Searcher
from .store import Store

MAX_LIMIT = 50


class McpUnavailable(RuntimeError):
    """The optional `mcp` package is missing or cannot be imported."""


def _parse_since(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").timestamp()
    except ValueError as exc:
        raise ValueError(f"since expects a date like 2026-01-31, got '{value}'") from exc


class IndexTools:
    """The tools an assistant can call. Safe to call from several threads.

    The embedding model is loaded on the first search, not at start-up, so a host that starts the server does not
    wait for it. The server stays up for a long time, so it also notices when another process (for example
    `localseek watch`) has changed the index and drops its cached vectors, and it loads the model again if the
    index was rebuilt with a different one.
    """

    def __init__(self, store: Store, embedder_factory: Callable[[], Embedder]) -> None:
        self.store = store
        self._embedder_factory = embedder_factory
        self._lock = threading.Lock()
        self._searcher: Searcher | None = None
        self._model: str | None = None

    def _current_searcher(self) -> Searcher:
        self.store.refresh_if_changed()  # `watch` or `index` may have written in another process
        model = self.store.get_meta("model")
        if self._searcher is None or model != self._model:
            self._searcher = Searcher(self.store, self._embedder_factory())
            self._model = model
        return self._searcher

    def search(
        self,
        query: str,
        limit: int = 10,
        mode: str = "hybrid",
        file_type: str | None = None,
        since: str | None = None,
        path_contains: str | None = None,
    ) -> list[dict[str, object]]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if mode not in MODES:
            raise ValueError(f"mode must be one of: {', '.join(MODES)}")
        since_timestamp = _parse_since(since)
        limit = max(1, min(MAX_LIMIT, limit))
        with self._lock:
            if self.store.stats()["chunks"] == 0:
                raise ValueError("The index is empty. Run: localseek index <folder>")
            hits = self._current_searcher().search(
                query,
                limit=limit,
                mode=mode,
                ftype=file_type,
                since=since_timestamp,
                path_contains=path_contains,
            )
        return [hit.to_dict() for hit in hits]

    def status(self) -> dict[str, object]:
        with self._lock:
            stats = self.store.stats()
        return {
            "index": str(self.store.path),
            "model": stats["model"],
            "files": stats["files"],
            "chunks": stats["chunks"],
            "size_mb": round(int(stats["db_bytes"]) / 1_048_576, 1),
            "folders": stats["roots"],
        }


def build_server(tools: IndexTools):
    """Create the MCP server object. Needs the optional `mcp` package."""
    try:
        from mcp.server import MCPServer
        from mcp.server.mcpserver.exceptions import ToolError
        from mcp.types import ToolAnnotations
    except ImportError as exc:
        raise McpUnavailable(
            f'The MCP server needs the optional "mcp" package ({exc}). Run: pip install "localseek[mcp]"'
        ) from exc

    server = MCPServer(
        "localseek",
        instructions="Searches the user's own documents on this computer. Read-only: it cannot change any file.",
    )
    read_only = ToolAnnotations(
        read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False
    )

    @server.tool(annotations=read_only)
    def search(
        query: str,
        limit: int = 10,
        mode: str = "hybrid",
        file_type: str | None = None,
        since: str | None = None,
        path_contains: str | None = None,
    ) -> dict[str, object]:
        """Search the user's local documents by meaning and by keywords.

        Returns the best matching passage of each file: its path, page (PDFs), heading, a text snippet and a relative
        score (1.0 means ranked first by every method). Snippets are short excerpts, not whole files.
        mode is hybrid (default), vector (by meaning only) or keyword (exact words only). limit is capped at 50.
        Optional filters: file_type such as "pdf", since as YYYY-MM-DD (files modified on or after that day), and
        path_contains, a piece of the file path.
        """
        try:
            return {"results": tools.search(query, limit, mode, file_type, since, path_contains)}
        except (ValueError, EmbedderUnavailable) as exc:
            raise ToolError(str(exc)) from exc

    @server.tool(annotations=read_only)
    def status() -> dict[str, object]:
        """Show what is indexed: the index location, embedding model, number of files and chunks, and folders."""
        return tools.status()

    return server


def run_stdio(tools: IndexTools) -> None:
    """Serve over stdin and stdout until the host closes the connection."""
    build_server(tools).run()
