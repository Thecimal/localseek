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


class WebTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        docs = Path(self._tmp.name) / "docs"
        write(docs / "a.md", "The landlord agreed to repair the heating.")
        write(docs / "b.md", "Sourdough starter needs feeding.")
        self.store = Store(Path(self._tmp.name) / "i.db")
        embedder = HashEmbedder()
        index_paths(self.store, embedder, [docs], Settings(model="hash"))
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), lambda *a: None)
        self.port = self.server.server_address[1]
        self.server.RequestHandlerClass = make_handler(
            Searcher(self.store, embedder), "testnonce", {f"127.0.0.1:{self.port}"}
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.store.close()
        self._tmp.cleanup()

    def get(self, path, host=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.putrequest("GET", path, skip_host=True)
        conn.putheader("Host", host or f"127.0.0.1:{self.port}")
        conn.endheaders()
        response = conn.getresponse()
        body = response.read()
        headers = dict(response.getheaders())
        conn.close()
        return response.status, headers, body

    def test_page_is_served_with_csp_nonce(self):
        status, headers, body = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn("nonce-testnonce", headers["Content-Security-Policy"])
        self.assertIn(b'nonce="testnonce"', body)
        self.assertNotIn(b"__NONCE__", body)

    def test_search_endpoint(self):
        status, _, body = self.get("/api/search?q=heating+repair")
        self.assertEqual(status, 200)
        hits = json.loads(body)
        self.assertTrue(hits[0]["path"].endswith("a.md"))

    def test_foreign_host_header_is_rejected(self):
        status, _, _ = self.get("/api/search?q=heating", host="evil.example")
        self.assertEqual(status, 403)

    def test_bad_limit_and_unknown_route(self):
        self.assertEqual(self.get("/api/search?q=heating&n=abc")[0], 200)
        self.assertEqual(self.get("/nope")[0], 404)


if __name__ == "__main__":
    unittest.main()
