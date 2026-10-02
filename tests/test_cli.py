import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from helpers import write
from localseek.cli import main


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(list(argv))
    return code, out.getvalue(), err.getvalue()


class CliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.db = str(self.dir / "i.db")
        self.docs = self.dir / "docs"
        write(self.docs / "a.md", "Sourdough starter feeding schedule and flour ratios.")
        write(self.docs / "b.md", "Router firmware update and guest wifi password.")

    def tearDown(self):
        self._tmp.cleanup()

    def base(self):
        return ["--db", self.db, "--model", "hash"]

    def test_full_flow(self):
        code, out, _ = run(*self.base(), "index", str(self.docs))
        self.assertEqual(code, 0)
        self.assertIn("2 added", out)

        code, out, _ = run(*self.base(), "search", "guest", "wifi")
        self.assertEqual(code, 0)
        self.assertIn("b.md", out.splitlines()[0])

        code, out, _ = run(*self.base(), "search", "--json", "-n", "1", "sourdough")
        data = json.loads(out)
        self.assertEqual(len(data), 1)
        self.assertTrue(data[0]["path"].endswith("a.md"))

        (self.docs / "c.md").write_text("Bicycle chain lubrication.", encoding="utf-8")
        code, out, _ = run(*self.base(), "update")
        self.assertIn("1 added", out)

        code, out, _ = run(*self.base(), "status")
        self.assertIn("Files:   3", out)

        code, out, _ = run(*self.base(), "forget", str(self.docs))
        self.assertIn("Removed 3 files", out)

    def test_search_on_empty_index_explains_next_step(self):
        code, _, err = run(*self.base(), "search", "anything")
        self.assertEqual(code, 1)
        self.assertIn("localseek index", err)

    def test_store_is_closed_after_every_command(self):
        # Windows cannot delete a database file that still has an open connection.
        import localseek.cli as cli

        opened = []

        class Tracking(cli.Store):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.closed = False
                opened.append(self)

            def close(self):
                self.closed = True
                super().close()

        with mock.patch.object(cli, "Store", Tracking):
            run(*self.base(), "index", str(self.docs))
            run(*self.base(), "search", "guest")
            run(*self.base(), "status")
            code, _, _ = run("--db", str(self.dir / "empty.db"), "--model", "hash", "search", "x")
            self.assertEqual(code, 1)  # error path: nothing indexed
            with self.assertRaises(SystemExit):  # error path: invalid --since
                run(*self.base(), "search", "--since", "yesterday", "x")
        self.assertEqual(len(opened), 5)
        self.assertTrue(all(store.closed for store in opened))

    def test_bad_since_value(self):
        run(*self.base(), "index", str(self.docs))
        with self.assertRaises(SystemExit):
            run(*self.base(), "search", "--since", "yesterday", "x")


if __name__ == "__main__":
    unittest.main()
