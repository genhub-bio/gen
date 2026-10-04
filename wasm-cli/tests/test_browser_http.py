"""Build the C bridge and exercise static/dynamic modules in Chromium workers.

Requires emcc on PATH and Python Playwright with Chromium installed.
Run from any directory: python wasm-cli/tests/test_browser_http.py
"""

import functools
import http.server
import json
from pathlib import Path
import subprocess
import tempfile
import threading

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[2]
HTTP = ROOT / "src/commands/remote/http"
TESTS = ROOT / "wasm-cli/tests"
PAYLOAD = bytes([0, 128, 255, 13])
LARGE_LENGTH = 20 * 1024 * 1024


class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        if self.path != "/blocked":
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header(
                "Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS"
            )
            self.send_header("Access-Control-Allow-Headers", "X-Gen-Test, Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        if self.path != "/echo":
            return super().do_GET()
        self.send_response(200)
        self.send_header("Content-Length", str(len(PAYLOAD)))
        self.end_headers()
        self.wfile.write(PAYLOAD)

    def do_HEAD(self):
        if self.path != "/echo":
            return super().do_HEAD()
        self.send_response(200)
        self.send_header("Content-Length", str(len(PAYLOAD)))
        self.end_headers()

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        assert body == PAYLOAD
        assert self.headers["X-Gen-Test"] == "binary"
        status = 422 if self.path == "/error" else 200
        response = (
            bytes(range(256)) * (LARGE_LENGTH // 256) if self.path == "/large" else body
        )
        if self.path == "/empty":
            status, response = 204, b""
        self.send_response(status)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    do_PUT = do_POST
    do_PATCH = do_POST
    do_DELETE = do_POST


def build(directory, shared):
    common = [
        "emcc",
        "-I",
        str(HTTP),
        "-O1",
        "-sALLOW_MEMORY_GROWTH=1",
        "-sINITIAL_MEMORY=16777216",
        "-sMAXIMUM_MEMORY=67108864",
    ]
    if shared:
        common += ["-pthread"]
    sources = [
        str(HTTP / "browser_http.c"),
        str(TESTS / "browser_http.c"),
        "-Wl,--wrap=malloc",
    ]
    runtime = [
        "--no-entry",
        "-sMODULARIZE=1",
        "-sEXPORT_NAME=createHttp",
        "-sEXPORTED_FUNCTIONS=_run_test",
        "-sEXPORTED_RUNTIME_METHODS=ccall",
    ]
    subprocess.run(
        common + sources + runtime + ["-o", str(directory / "static.js")], check=True
    )
    subprocess.run(
        common + sources + ["-sSIDE_MODULE=1", "-o", str(directory / "http-side.wasm")],
        check=True,
    )
    subprocess.run(
        common
        + [str(TESTS / "browser_http_dynamic.c"), "-sMAIN_MODULE=1"]
        + runtime
        + ["-o", str(directory / "dynamic.js")],
        check=True,
    )


def main():
    with tempfile.TemporaryDirectory(prefix="gen-http-") as temporary:
        directory = Path(temporary)
        (directory / "index.html").write_text(
            "<!doctype html><title>Browser HTTP test</title>"
        )
        handler = functools.partial(Handler, directory=str(directory))
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        remote = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        for instance in (server, remote):
            threading.Thread(target=instance.serve_forever, daemon=True).start()
        origin = f"http://127.0.0.1:{server.server_port}"
        remote_origin = f"http://127.0.0.1:{remote.server_port}"
        cases = [
            ("/echo", method, 200, 4, 0)
            for method in ("POST", "PUT", "PATCH", "DELETE")
        ]
        cases += [("/echo", "GET", 200, 4, 0), ("/echo", "HEAD", 200, 0, 0)]
        cases += [
            ("/error", "POST", 422, 4, 0),
            ("/empty", "POST", 204, 0, 0),
            ("/large", "POST", 200, LARGE_LENGTH, 0),
            ("/echo", "POST", 0, 0, 2),
            ("/blocked", "POST", 0, 0, 1),
        ]
        arguments = [
            [remote_origin + path, method, status, length, result]
            for path, method, status, length, result in cases
        ]
        arguments += [["http://127.0.0.1:1/unreachable", "POST", 0, 0, 1]]
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(origin)
                for shared in (False, True):
                    build(directory, shared)
                    for module in ("static", "dynamic"):
                        worker = f"""
importScripts('{origin}/{module}.js');
createHttp({{locateFile: path => '{origin}/' + path}}).then(module => {{
  const cases = {json.dumps(arguments)};
  postMessage(cases.map(args => {{
    const result = module.ccall('run_test', 'number',
      ['string', 'string', 'number', 'number', 'number'], args);
    return result && module.__genHttpResponses.size === 0 ? 1 : 0;
  }}));
}}).catch(error => postMessage({{error: String(error)}}));
"""
                        results = page.evaluate(
                            """source => new Promise((resolve, reject) => {
  const worker = new Worker(URL.createObjectURL(new Blob([source], {type: 'text/javascript'})));
  const timeout = setTimeout(() => { worker.terminate(); reject('worker timed out'); }, 30000);
  worker.onmessage = event => { clearTimeout(timeout); worker.terminate(); resolve(event.data); };
  worker.onerror = event => { clearTimeout(timeout); worker.terminate(); reject(event.message); };
})""",
                            worker,
                        )
                        assert results == [1] * len(arguments), (
                            shared,
                            module,
                            results,
                        )
                        print(
                            f"PASS {module}, shared memory={shared}: {len(arguments)} cases",
                            flush=True,
                        )
                browser.close()
        finally:
            server.shutdown()
            remote.shutdown()


if __name__ == "__main__":
    main()
