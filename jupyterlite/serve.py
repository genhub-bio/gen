"""Serve the built site with Wasm MIME types and a disposable HTTP error demo."""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".wasm": "application/wasm",
        ".whl": "application/zip",
    }

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path != "/api/repos/demo/rejected/remote-capability":
            self.send_error(404)
            return
        body = b'{"error":"Gen browser transport demo: disposable remote rejected"}'
        self.send_response(422)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=4502)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent / "_output"
    if not (directory / "index.html").exists():
        raise SystemExit("Run build-site.py before serving the site.")
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port), partial(Handler, directory=str(directory))
    )
    print(f"Open http://127.0.0.1:{server.server_port}/lab/index.html", flush=True)
    server.serve_forever()
