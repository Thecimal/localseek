import http.client
import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

from helpers import write
from localseek.config import Settings
from localseek.embedder import HashEmbedder
from localseek.indexer import index_paths
from localseek.search import Searcher
from localseek.store import Store
from localseek.web import make_handler


class Fixture(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.docs = self.dir / "docs"
        write(self.docs / "a.md", "Sourdough starter feeding schedule and flour ratios.")
        write(self.docs / "b.md", "Router firmware update and guest wifi password.")
        self.db = self.dir / "i.db"
        self.stores = []
        self.store = self.open_store()
        index_paths(self.store, HashEmbedder(), [self.docs], Settings(model="hash"))

    def tearDown(self):
        for store in self.stores:
            store.close()
        self._tmp.cleanup()

    def open_store(self) -> Store:
        store = Store(self.db)
        self.stores.append(store)
        return store

    def add_document_from_another_connection(self):
        write(self.docs / "c.md", "Bicycle chain lubrication and brake pads.")
        index_paths(self.open_store(), HashEmbedder(), [self.docs], Settings(model="hash"))


class RefreshTests(Fixture):
    def test_an_untouched_index_reports_no_change(self):
        self.assertFalse(self.store.refresh_if_changed())
        self.store.load_matrix()
        self.assertFalse(self.store.refresh_if_changed())

    def test_another_connections_write_is_noticed_and_the_vectors_reload(self):
        before = len(self.store.load_matrix()[0])
        self.add_document_from_another_connection()
        self.assertTrue(self.store.refresh_if_changed())
        self.assertGreater(len(self.store.load_matrix()[0]), before)
        self.assertFalse(self.store.refresh_if_changed())

    def test_the_stores_own_writes_are_not_reported_but_still_reload_the_vectors(self):
        before = len(self.store.load_matrix()[0])
        write(self.docs / "c.md", "Bicycle chain lubrication and brake pads.")
        index_paths(self.store, HashEmbedder(), [self.docs], Settings(model="hash"))
        self.assertFalse(self.store.refresh_if_changed())
        self.assertGreater(len(self.store.load_matrix()[0]), before)


class ServeSeesLaterIndexingTests(Fixture):
    def setUp(self):
        super().setUp()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), lambda *a: None)
        self.port = self.server.server_address[1]
        self.server.RequestHandlerClass = make_handler(
            Searcher(self.store, HashEmbedder()), "testnonce", {f"127.0.0.1:{self.port}"}
        )
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        super().tearDown()

    def search(self, query):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.putrequest("GET", "/api/search?q=" + query.replace(" ", "+"), skip_host=True)
        conn.putheader("Host", f"127.0.0.1:{self.port}")
        conn.endheaders()
        body = conn.getresponse().read()
        conn.close()
        return json.loads(body)

    def test_a_file_indexed_while_the_page_server_runs_is_found_by_meaning_too(self):
        self.search("guest wifi")  # loads and caches the vectors
        self.add_document_from_another_connection()
        top = self.search("bicycle chain lubrication")[0]
        self.assertTrue(top["path"].endswith("c.md"))
        # 1.0 means both the keyword and the vector arm ranked it first. A stale vector cache cannot see the new
        # file, which leaves only the keyword arm and a score of 0.5.
        self.assertEqual(top["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
