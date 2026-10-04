# Gen in JupyterLite

This local site runs the existing Gen Python API in a standard Pyodide worker.
Publishing is outside this setup. The wheel comes from this checkout and is
bundled in the site's relative piplite index; it needs no external wheel host.
The standard Pyodide runtime and its kernel packages load from the pinned CDN.

## Build and launch

Use Python 3.13 on the host. Install the repository's Rust toolchain with sources
and activate Emscripten **4.0.9** (the ABI requires the exact patch version):

```sh
python3.13 -m venv .venv-pyodide
source .venv-pyodide/bin/activate
pip install -r jupyterlite/requirements.txt
rustup toolchain install nightly-2026-06-26 --component rust-src
source "$HOME/emsdk/emsdk_env.sh"  # emsdk install/activate 4.0.9 first
bash jupyterlite/build-wheel.sh
python jupyterlite/build-site.py
python jupyterlite/serve.py
```

Open http://127.0.0.1:4502/lab/index.html and run all cells in `Gen.ipynb`.
`serve.py` provides Wasm/JavaScript MIME types and a disposable HTTP 422 endpoint
for the notebook's browser transport example. It binds only to localhost.

Pinned versions:

| Component | Version |
| --- | --- |
| Pyodide | 0.29.4 |
| Runtime Python | 3.13.2 |
| Emscripten | 4.0.9 |
| Rust | nightly-2026-06-26 |
| pyodide-build | 0.39.1 |
| Maturin | 1.13.0 |
| JupyterLite core | 0.7.3 |
| Pyodide kernel | 0.7.1 |
| JupyterLab | 4.5.6 |

The [kernel compatibility matrix](https://jupyterlite-pyodide-kernel.readthedocs.io/en/latest/index.html#compatibility)
supports kernel 0.7 with Pyodide 0.29, Python 3.13 and Emscripten 4.0.9.
The [Pyodide ABI specification](https://pyodide.org/en/0.29.4/development/abi/313.html)
requires Wasm exception handling. This Rust nightly enables it by default and
has removed the older `-Z emscripten-wasm-eh` option. We rebuild std with
`-Z build-std=std,panic_abort`; the CLI's installed sysroot must not be reused
for the side module. The actual wheel uses `cp311-abi3-pyemscripten_2025_0_wasm32`.
Both Cargo workspaces explicitly pin the same DoltLite fork revision `e5ae83c`.

The shared C/XHR transport loads through the stock Pyodide dynamic linker.
It requires a worker and dynamic evaluation allowed by CSP (`unsafe-eval`).
Servers must allow browser CORS and the relevant preflight headers. Gen does
not route HTTP through Python or enable browser database threading.

## Files and persistence

Keep live Gen repositories in `/tmp`, which is the kernel's MEMFS. The notebook
creates, imports, reads, edits and automatically commits a small sequence there,
then inspects Dolt history. MEMFS and installed packages disappear when the
kernel restarts. The notebook closes the repository, archives all its files,
and writes `gen-demo.zip` to `/drive` for download from the file browser.

JupyterLite's Contents API drive stores user files in IndexedDB for the site
origin. Those survive reloads, but clearing browser storage removes them, and
changing origin uses a different store. Do not treat that browser storage as a
backup. Download archives to retain them independently. `/drive` buffers files
and does not provide the SQLite sync behavior installed specifically in the
standalone Cockle integration, so use it for closed archives rather than live
Python repository databases.

## Validation

Native checks and standalone CLI validation remain the root repository's
commands. For Pyodide worker regression, download the standard 0.29.4 full
distribution from the [Pyodide release](https://github.com/pyodide/pyodide/releases/tag/0.29.4)
and pass its directory containing `pyodide.js`, Wasm, stdlib, lockfile and wheels:

```sh
pip install playwright
playwright install chromium
cargo build --example pyodide_remote_fixture --locked
python jupyterlite/tests/test_worker.py --runtime /path/to/pyodide
python wasm-cli/tests/test_drive_import.py
# Stop serve.py before this test, which owns port 4502.
python jupyterlite/tests/test_site.py
```

The worker regression validates the real extension's import, local SQLite,
FASTA import/export, sequence edit/history, Rust HTTP error bodies, and a C bridge
side module loaded by the actual Pyodide runtime. Its 12 transport cases cover
binary methods, HTTP errors, empty responses, 20 MiB memory growth, allocation
failure, blocked CORS and network failure. It also clones a disposable native
Dolt database through Rust capability negotiation and DoltLite's HTTP bridge,
verifies replicated data, and checks a failing Dolt endpoint. The native fixture
process owns temporary database files; the test proxy supplies browser CORS.
`test_site.py` separately runs the bundled notebook in the actual JupyterLite UI
and verifies its outputs and exported ZIP. No production writes or credentials
are used.
