"""A tiny local search page. Binds to 127.0.0.1 only and uses the standard library."""

from __future__ import annotations

import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .search import Searcher

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>localseek</title>
<style nonce="__NONCE__">
  :root {
    --bg: #f3f5f7; --panel: #ffffff; --text: #1b2127; --muted: #5c6772;
    --line: #d8dee4; --accent: #0a6c8f; --focus: #0a6c8f;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #12161a; --panel: #1a2026; --text: #e6eaee; --muted: #93a0ac;
      --line: #2b343c; --accent: #5bc0de; --focus: #5bc0de;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font: 16px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  main { max-width: 46rem; margin: 0 auto; padding: 2.5rem 1.25rem 4rem; }
  h1 { font-size: 1.1rem; font-weight: 600; margin: 0 0 1rem; color: var(--muted); }
  form { display: flex; gap: .5rem; flex-wrap: wrap; }
  input, select, button { font: inherit; color: inherit; }
  input[type=search] {
    flex: 1 1 18rem; padding: .7rem .9rem; border: 1px solid var(--line);
    border-radius: 6px; background: var(--panel);
  }
  input.type { width: 7rem; padding: .7rem .8rem; border: 1px solid var(--line);
    border-radius: 6px; background: var(--panel); }
  :focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }
  #status { margin: 1rem 0; color: var(--muted); min-height: 1.5rem; }
  ol { list-style: none; margin: 0; padding: 0; }
  li { padding: 1rem 0; border-top: 1px solid var(--line); }
  .name { font-weight: 600; word-break: break-word; }
  .where { color: var(--muted); font-size: .875rem; word-break: break-all; }
  .snippet { margin: .4rem 0 0; }
  button.copy {
    margin-top: .4rem; padding: .2rem .6rem; border: 1px solid var(--line);
    border-radius: 6px; background: transparent; cursor: pointer; font-size: .85rem;
  }
  button.copy:hover { border-color: var(--accent); color: var(--accent); }
</style>
</head>
<body>
<main>
  <h1>localseek: searching this computer only</h1>
  <form id="f">
    <input id="q" type="search" placeholder="Describe what you are looking for" autofocus
           aria-label="Search query">
    <input id="t" class="type" type="text" placeholder="type: pdf" aria-label="File type filter">
  </form>
  <div id="status" role="status">Type a question or a few words.</div>
  <ol id="results"></ol>
</main>
<script nonce="__NONCE__">
  const q = document.getElementById("q"), t = document.getElementById("t");
  const statusEl = document.getElementById("status"), list = document.getElementById("results");
  let timer = null, latest = 0;

  function render(hits) {
    list.textContent = "";
    for (const hit of hits) {
      const li = document.createElement("li");
      const parts = hit.path.split(/[\\\\/]/);
      const name = document.createElement("div"); name.className = "name";
      name.textContent = parts[parts.length - 1] + (hit.page ? "  (page " + hit.page + ")" : "");
      const where = document.createElement("div"); where.className = "where";
      where.textContent = hit.path;
      const snip = document.createElement("p"); snip.className = "snippet";
      snip.textContent = hit.snippet;
      const copy = document.createElement("button"); copy.className = "copy"; copy.type = "button";
      copy.textContent = "Copy path";
      copy.addEventListener("click", async () => {
        try { await navigator.clipboard.writeText(hit.path); copy.textContent = "Copied"; }
        catch { copy.textContent = "Copy failed"; }
        setTimeout(() => (copy.textContent = "Copy path"), 1500);
      });
      li.append(name, where, snip, copy);
      list.append(li);
    }
  }

  async function run() {
    const text = q.value.trim();
    if (!text) { list.textContent = ""; statusEl.textContent = "Type a question or a few words."; return; }
    const mine = ++latest;
    statusEl.textContent = "Searching...";
    const params = new URLSearchParams({ q: text, n: "15" });
    if (t.value.trim()) params.set("type", t.value.trim());
    try {
      const res = await fetch("/api/search?" + params);
      const data = await res.json();
      if (mine !== latest) return;
      if (!res.ok) throw new Error(data.error || res.statusText);
      render(data);
      statusEl.textContent = data.length ? data.length + " files" : "No matches. Try different words.";
    } catch (err) {
      if (mine === latest) statusEl.textContent = "Search failed: " + err.message;
    }
  }
  const schedule = () => { clearTimeout(timer); timer = setTimeout(run, 250); };
  q.addEventListener("input", schedule); t.addEventListener("input", schedule);
  document.getElementById("f").addEventListener("submit", (e) => { e.preventDefault(); run(); });
</script>
</body>
</html>
"""


def make_handler(searcher: Searcher, nonce: str, allowed_hosts: set[str]):
    lock = threading.Lock()
    page = PAGE.replace("__NONCE__", nonce).encode("utf-8")
    csp = (
        f"default-src 'none'; script-src 'nonce-{nonce}'; style-src 'nonce-{nonce}'; "
        "connect-src 'self'; base-uri 'none'; form-action 'none'"
    )

    class Handler(BaseHTTPRequestHandler):
        server_version = "localseek"

        def log_message(self, format: str, *args) -> None:  # noqa: A002 - silence access log
            pass

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", csp)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: object) -> None:
            self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

        def do_GET(self) -> None:  # noqa: N802
            # Reject unexpected Host headers so a hostile web page cannot reach this server
            # through DNS rebinding.
            if self.headers.get("Host", "") not in allowed_hosts:
                self._json(403, {"error": "forbidden host"})
                return
            url = urlparse(self.path)
            if url.path == "/":
                self._send(200, page, "text/html; charset=utf-8")
            elif url.path == "/api/search":
                params = parse_qs(url.query)
                query = params.get("q", [""])[0]
                try:
                    limit = max(1, min(50, int(params.get("n", ["10"])[0])))
                except ValueError:
                    limit = 10
                ftype = params.get("type", [None])[0]
                try:
                    with lock:
                        hits = searcher.search(query, limit=limit, ftype=ftype)
                except Exception as exc:  # report the problem to the page, keep serving
                    self._json(500, {"error": str(exc)})
                    return
                self._json(200, [h.to_dict() for h in hits])
            else:
                self._json(404, {"error": "not found"})

    return Handler


def serve(searcher: Searcher, port: int = 8765) -> None:
    nonce = secrets.token_urlsafe(16)
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(searcher, nonce, hosts))
    print(f"Serving on http://127.0.0.1:{port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
