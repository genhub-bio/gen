"""Exercise the real wheel and C bridge in standard Pyodide's Chromium worker.

Pass --runtime pointing to a downloaded full Pyodide 0.29.4 distribution.
Requires emcc 4.0.9, Python Playwright and installed Chromium.
"""

import argparse
from functools import partial
from http.client import HTTPConnection
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
from urllib.parse import urlsplit

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
PAYLOAD = bytes([0, 128, 255, 13])


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def end_headers(self):
        if self.path != "/blocked":
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header(
                "Access-Control-Allow-Headers",
                "X-Gen-Test, Content-Type, Authorization",
            )
            self.send_header(
                "Access-Control-Allow-Methods",
                "GET, HEAD, POST, PUT, PATCH, DELETE, OPTIONS",
            )
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        if self.path == "/echo":
            self.send_response(200)
            self.send_header("Content-Length", str(len(PAYLOAD)))
            self.end_headers()
            self.wfile.write(PAYLOAD)
        elif self.path.startswith("/dolt/"):
            self.server.dolt_requests.append(self.path)
            if self.path.startswith("/dolt/fixture.db"):
                self.proxy_dolt()
            else:
                self.send_error(503, "Disposable Dolt remote unavailable")
        else:
            super().do_GET()

    def do_HEAD(self):
        if self.path == "/echo":
            self.send_response(200)
            self.send_header("Content-Length", str(len(PAYLOAD)))
            self.end_headers()
        else:
            super().do_HEAD()

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path == "/api/repos/demo/success/remote-capability":
            self.server.capability_requests.append(body)
            status, response = (
                200,
                json.dumps(
                    {
                        "remote_url": f"http://127.0.0.1:{self.server.server_port}/dolt/fixture.db",
                        "expires_at": "2030-01-01T00:00:00Z",
                        "default_branch": "main",
                        "transfer_id": "123e4567-e89b-12d3-a456-426614174000",
                    }
                ).encode(),
            )
        elif self.path == "/api/repos/demo/success/asset-transfers":
            status, response = 200, b'{"assets":[]}'
        elif self.path.endswith("/remote-capability"):
            self.server.capability_requests.append(body)
            status, response = 422, b'{"error":"disposable capability rejected"}'
        elif self.path.startswith("/dolt/"):
            self.server.dolt_requests.append(self.path)
            if self.path.startswith("/dolt/fixture.db"):
                return self.proxy_dolt(body)
            status, response = 503, b"disposable Dolt remote unavailable"
        else:
            assert body == PAYLOAD
            assert self.headers["X-Gen-Test"] == "binary"
            status, response = (422 if self.path == "/error" else 200), body
            if self.path == "/empty":
                status, response = 204, b""
            if self.path == "/large":
                response = bytes(range(256)) * (20 * 1024 * 1024 // 256)
        self.send_response(status)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def proxy_dolt(self, body=None):
        # Add CORS at the test boundary; the underlying native server and all
        # database data belong to the disposable fixture process.
        upstream = HTTPConnection(self.server.native_origin, timeout=30)
        try:
            headers = {
                name: value
                for name, value in self.headers.items()
                if name.lower() not in {"host", "connection"}
            }
            upstream.request(
                self.command, self.path.removeprefix("/dolt"), body, headers
            )
            response = upstream.getresponse()
            data = response.read()
            self.send_response(response.status)
            for name, value in response.getheaders():
                if name.lower() not in {
                    "connection",
                    "transfer-encoding",
                    "content-length",
                }:
                    self.send_header(name, value)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        finally:
            upstream.close()

    do_PUT = do_POST
    do_PATCH = do_POST
    do_DELETE = do_POST


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--wheel", type=Path)
    args = parser.parse_args()
    wheel = args.wheel or next((ROOT / "jupyterlite/wheels").glob("gen-*.whl"))
    with tempfile.TemporaryDirectory(prefix="gen-pyodide-worker-") as temporary:
        directory = Path(temporary)
        (directory / "runtime").symlink_to(
            args.runtime.resolve(), target_is_directory=True
        )
        shutil.copy(wheel, directory / wheel.name)
        (directory / "index.html").write_text(
            "<html><body>Gen Pyodide worker regression</body></html>"
        )
        subprocess.run(
            [
                "emcc",
                "-O1",
                "-fwasm-exceptions",
                "-sSIDE_MODULE=2",
                "-sEXPORTED_FUNCTIONS=['_run_test']",
                "-Wl,--wrap=malloc",
                "-I",
                str(ROOT / "src/commands/remote/http"),
                str(ROOT / "src/commands/remote/http/browser_http.c"),
                str(ROOT / "wasm-cli/tests/browser_http.c"),
                "-o",
                str(directory / "http-side.wasm"),
            ],
            check=True,
        )
        servers = []
        fixture = subprocess.Popen(
            [str(ROOT / "target/debug/examples/pyodide_remote_fixture")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        native_url = fixture.stdout.readline().strip()
        assert native_url.startswith("http://"), (
            "native fixture should announce its URL"
        )
        for _ in range(2):
            server = ThreadingHTTPServer(
                ("127.0.0.1", 0), partial(Handler, directory=str(directory))
            )
            server.capability_requests = []
            server.dolt_requests = []
            server.native_origin = urlsplit(native_url).netloc
            threading.Thread(target=server.serve_forever, daemon=True).start()
            servers.append(server)
        origin = f"http://127.0.0.1:{servers[0].server_port}"
        remote = f"http://127.0.0.1:{servers[1].server_port}"
        python = f"""
import ctypes
import gen
from pathlib import Path
root = Path('/tmp/gen-smoke')
root.mkdir()
repository = gen.Repository(str(root))
fasta = root / 'small.fa'
fasta.write_text('>tiny\\nACGTACGT\\n')
repository.import_fasta(str(fasta), sample='initial')
repository.export_fasta(str(root / 'before.fa'), sample='initial')
assert 'ACGTACGT' in (root / 'before.fa').read_text()
repository.update_with_sequence('ACGTTTGT', 'initial', 'edited', 'tiny:0-8')
repository.export_fasta(str(root / 'after.fa'), sample='edited')
assert 'ACGTTTGT' in (root / 'after.fa').read_text()
assert len(repository.get_operations()) >= 3
cloned = gen.clone('{remote}/api/repos/demo/success', str(root / 'successful-clone'))
assert cloned.query('SELECT value FROM browser_fixture') == [['cloned through DoltLite']]
try:
    gen.clone('{remote}/api/repos/demo/rejected', str(root / 'clone'))
except RuntimeError as error:
    assert 'disposable capability rejected' in str(error), str(error)
else:
    raise AssertionError('clone should reject HTTP 422')
library = ctypes.CDLL('/tmp/http-side.wasm')
test = library.run_test
test.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
test.restype = ctypes.c_int
cases = [('/echo', 'GET', 200, 4, 0), ('/echo', 'HEAD', 200, 0, 0),
         ('/error', 'POST', 422, 4, 0), ('/empty', 'POST', 204, 0, 0),
         ('/large', 'POST', 200, 20 * 1024 * 1024, 0),
         ('/echo', 'POST', 0, 0, 2), ('/blocked', 'POST', 0, 0, 1)]
cases += [('/echo', method, 200, 4, 0) for method in ['POST', 'PUT', 'PATCH', 'DELETE']]
for path, method, status, length, result in cases:
    assert test(('{remote}' + path).encode(), method.encode(), status, length, result), (path, method)
assert test(b'http://127.0.0.1:1/offline', b'GET', 0, 0, 1)
repository.query("SELECT dolt_remote('add', 'probe', '{remote}/dolt/probe')")
try:
    repository.query("SELECT dolt_fetch('probe')")
except RuntimeError:
    pass
else:
    raise AssertionError('disposable Dolt remote should fail')
print('PASS: local API, successful remote clone, Rust HTTP error body, 12 bridge cases, Dolt HTTP error path')
"""
        worker = f"""
importScripts('/runtime/pyodide.js');
(async () => {{
try {{
const pyodide = await loadPyodide({{indexURL: '{origin}/runtime/'}});
await pyodide.loadPackage('micropip');
await pyodide.runPythonAsync("import micropip; await micropip.install('{origin}/{wheel.name}')");
pyodide.FS.writeFile('/tmp/http-side.wasm', new Uint8Array(await (await fetch('/http-side.wasm')).arrayBuffer()));
await pyodide.runPythonAsync({json.dumps(python)});
postMessage({{ok: true}});
}} catch(error) {{ postMessage({{ok: false, error: String(error)}}); }}
}})();
"""
        (directory / "worker.js").write_text(worker)
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch()
                page = browser.new_page()
                page.on("console", lambda message: print(message.text, flush=True))
                page.goto(origin)
                result = page.evaluate("""() => new Promise(resolve => {
                    const worker = new Worker('/worker.js');
                    const timeout = setTimeout(() => {worker.terminate(); resolve({ok:false, error:'worker timeout'});}, 120000);
                    worker.onmessage = event => {clearTimeout(timeout); resolve(event.data);};
                    worker.onerror = event => {clearTimeout(timeout); resolve({ok:false, error:event.message});};
                })""")
                browser.close()
                assert result["ok"], result
                assert servers[1].capability_requests, (
                    "Rust capability bridge should reach server"
                )
                assert servers[1].dolt_requests, "DoltLite bridge should reach server"
        finally:
            for server in servers:
                server.shutdown()
            fixture.stdin.close()
            fixture.wait(timeout=10)


if __name__ == "__main__":
    main()
