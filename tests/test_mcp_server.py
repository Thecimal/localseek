import asyncio
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from helpers import write
from localseek.cli import main
from localseek.config import Settings
from localseek.embedder import HashEmbedder
from localseek.indexer import index_paths
from localseek.mcp_server import MAX_LIMIT, IndexTools, McpUnavailable, build_server
from localseek.store import Store

HAS_MCP = importlib.util.find_spec("mcp") is not None


class CountingFactory:
    """An embedder factory that records how often it is called."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> HashEmbedder:
        self.calls += 1
        return HashEmbedder()


class Fixture(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.docs = self.dir / "docs"
        write(self.docs / "a.md", "Sourdough starter feeding schedule and flour ratios.")
        write(self.docs / "b.md", "Router firmware update and guest wifi password.")
        self.db = self.dir / "i.db"
        self.stores = []

    def tearDown(self):
        for store in self.stores:
            store.close()
        self._tmp.cleanup()

    def open_store(self) -> Store:
        store = Store(self.db)
        self.stores.append(store)
        return store

    def indexed_store(self) -> Store:
        store = self.open_store()
        index_paths(store, HashEmbedder(), [self.docs], Settings(model="hash"))
        return store


class IndexToolsTests(Fixture):
    def test_search_returns_hits_as_plain_dicts(self):
        tools = IndexTools(self.indexed_store(), CountingFactory())
        results = tools.search("guest wifi")
        self.assertTrue(results[0]["path"].endswith("b.md"))
        self.assertEqual(set(results[0]), {"path", "score", "snippet", "page", "heading", "chunk"})
        json.dumps(results)  # must be serialisable as it is

    def test_filters_are_applied(self):
        tools = IndexTools(self.indexed_store(), CountingFactory())
        self.assertEqual(tools.search("sourdough", path_contains="nothing-like-this"), [])
        self.assertEqual(tools.search("sourdough", file_type="pdf"), [])
        self.assertTrue(tools.search("sourdough", file_type="md", since="2000-01-01"))

    def test_bad_input_is_a_value_error(self):
        tools = IndexTools(self.indexed_store(), CountingFactory())
        with self.assertRaisesRegex(ValueError, "mode must be one of"):
            tools.search("x", mode="fuzzy")
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            tools.search("   ")
        with self.assertRaisesRegex(ValueError, "since expects a date"):
            tools.search("x", since="yesterday")

    def test_limit_is_clamped(self):
        tools = IndexTools(self.indexed_store(), CountingFactory())
        self.assertEqual(len(tools.search("the and of a", limit=0, mode="keyword")), 1)
        self.assertLessEqual(len(tools.search("a", limit=10_000)), MAX_LIMIT)

    def test_empty_index_explains_next_step_without_loading_the_model(self):
        factory = CountingFactory()
        tools = IndexTools(self.open_store(), factory)
        with self.assertRaisesRegex(ValueError, "localseek index"):
            tools.search("anything")
        self.assertEqual(factory.calls, 0)

    def test_model_loads_on_first_search_and_only_once(self):
        factory = CountingFactory()
        tools = IndexTools(self.indexed_store(), factory)
        self.assertEqual(factory.calls, 0)
        tools.search("wifi")
        tools.search("sourdough")
        self.assertEqual(factory.calls, 1)

    def test_sees_files_another_process_indexed_after_the_server_started(self):
        tools = IndexTools(self.indexed_store(), CountingFactory())
        tools.search("wifi", mode="vector")  # caches the vectors
        write(self.docs / "c.md", "Bicycle chain lubrication and brake pads.")
        other = self.open_store()  # a second connection, as `localseek watch` would be
        index_paths(other, HashEmbedder(), [self.docs], Settings(model="hash"))
        results = tools.search("bicycle chain lubrication", mode="vector")
        self.assertTrue(results[0]["path"].endswith("c.md"))

    def test_status(self):
        status = IndexTools(self.indexed_store(), CountingFactory()).status()
        self.assertEqual(status["files"], 2)
        self.assertEqual(status["model"], "hash-256")
        self.assertEqual(status["index"], str(self.db))
        self.assertIn("folders", status)

    def test_concurrent_searches_do_not_fail(self):
        tools = IndexTools(self.indexed_store(), CountingFactory())
        errors = []

        def work():
            try:
                for _ in range(10):
                    tools.search("guest wifi")
            except Exception as exc:  # noqa: BLE001 - collected and asserted below
                errors.append(exc)

        threads = [threading.Thread(target=work) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])


def _text(result):
    return result.content[0].text


@unittest.skipUnless(HAS_MCP, "the optional mcp package is not installed")
class ServerWiringTests(Fixture):
    def call(self, tools, name, arguments=None):
        from mcp import Client

        async def go():
            async with Client(build_server(tools)) as client:
                return await client.call_tool(name, arguments or {})

        return asyncio.run(go())

    def test_exposes_exactly_the_two_read_only_tools(self):
        from mcp import Client

        async def go():
            async with Client(build_server(IndexTools(self.indexed_store(), CountingFactory()))) as client:
                return (await client.list_tools()).tools

        tools = asyncio.run(go())
        self.assertEqual({t.name for t in tools}, {"search", "status"})
        for tool in tools:
            self.assertTrue(tool.annotations.read_only_hint, tool.name)

    def test_search_over_the_protocol(self):
        tools = IndexTools(self.indexed_store(), CountingFactory())
        result = self.call(tools, "search", {"query": "guest wifi", "limit": 1})
        self.assertFalse(result.is_error)
        data = json.loads(_text(result))
        self.assertEqual(len(data["results"]), 1)
        self.assertTrue(data["results"][0]["path"].endswith("b.md"))

    def test_bad_input_reaches_the_model_as_a_readable_error(self):
        tools = IndexTools(self.indexed_store(), CountingFactory())
        result = self.call(tools, "search", {"query": "x", "mode": "fuzzy"})
        self.assertTrue(result.is_error)
        self.assertIn("mode must be one of", _text(result))

    def test_empty_index_reaches_the_model_as_a_readable_error(self):
        result = self.call(IndexTools(self.open_store(), CountingFactory()), "search", {"query": "x"})
        self.assertTrue(result.is_error)
        self.assertIn("localseek index", _text(result))

    def test_status_over_the_protocol(self):
        result = self.call(IndexTools(self.indexed_store(), CountingFactory()), "status")
        self.assertFalse(result.is_error)
        self.assertEqual(json.loads(_text(result))["files"], 2)

    def test_the_command_works_end_to_end_over_stdio(self):
        from mcp import Client, StdioServerParameters

        self.indexed_store()
        self.stores.pop().close()  # release the database before the child process opens it
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "localseek", "--db", str(self.db), "--model", "hash", "mcp"],
        )

        async def go():
            async with Client(params) as client:
                return await client.call_tool("search", {"query": "guest wifi"})

        result = asyncio.run(go())
        self.assertFalse(result.is_error)
        self.assertTrue(json.loads(_text(result))["results"][0]["path"].endswith("b.md"))


class MissingSdkTests(Fixture):
    def test_build_server_reports_how_to_install(self):
        tools = IndexTools(self.open_store(), CountingFactory())
        with mock.patch.dict(sys.modules, {"mcp": None, "mcp.server": None}):
            with self.assertRaisesRegex(McpUnavailable, r"localseek\[mcp\]"):
                build_server(tools)

    def test_command_explains_instead_of_crashing(self):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.dict(sys.modules, {"mcp": None, "mcp.server": None}):
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = main(["--db", str(self.db), "--model", "hash", "mcp"])
        self.assertEqual(code, 2)
        self.assertEqual(out.getvalue(), "")
        self.assertIn("localseek[mcp]", err.getvalue())


if __name__ == "__main__":
    unittest.main()
