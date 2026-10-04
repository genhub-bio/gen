# Gen in JupyterLite

This local site runs the existing Gen Python API in a standard Pyodide worker.
Publishing is outside this setup. The wheel comes from this checkout and is
bundled in the site's relative piplite index; it needs no external wheel host.
The standard Pyodide runtime and its kernel packages load from the pinned CDN.
A local loader installs a DriveFS sync adapter before the kernel mounts `/drive`.

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

Keep live Gen repositories in `/drive`, the kernel's JupyterLite Contents API
filesystem. `pyodide-drive.js` wraps the standard Pyodide loader and installs
synchronous `fsync` on DriveFS before mounting it. SQLite sync therefore publishes
buffered writes while the connection is open. Open handles within one kernel
share file buffers, so plotting connections also see current writes. This allows
Dolt's branch-head checks to reopen the current database bytes. Unlike the standalone Cockle integration,
this kernel mounts DriveFS directly and needs no PROXYFS forwarding. The adapter
also normalizes directory creation requests for JupyterLite 0.7, so private
directories (such as Python's `mkdtemp` folders) persist as directories.

The notebook creates a uniquely named repository on `/drive`, imports and edits a
small sequence, inspects Dolt history, and plots the original and edited sequence
graphs with Gen's interactive canvas viewer. The site includes the anywidget and
ipywidgets extensions; the notebook installs matching widget package versions
with `gen[jupyter]` for plotting. Its path is saved in
`/drive/gen-demo-path.txt`. After a restart, reinstall the bundled wheel and reopen
that repository:

```python
import piplite
await piplite.install(['anywidget==0.9.21', 'ipywidgets==8.1.8'])
await piplite.install('gen[jupyter]==0.3.1')
import gen
from pathlib import Path
root = Path(Path('/drive/gen-demo-path.txt').read_text())
repository = gen.Repository(str(root))
```

JupyterLite's drive stores user files in IndexedDB for the site origin. Repositories
survive kernel restarts and page reloads, but clearing browser storage removes
them, and changing origin uses a different store. Installed packages and `/tmp`
(MEMFS) disappear when the kernel restarts. Use one kernel at a time for a given
repository; this adapter does not add cross-worker SQLite locking. Close Gen
objects before archiving. The notebook writes `gen-demo.zip` to the file browser;
download it to retain a backup independently of browser storage.

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
node --test jupyterlite/tests/test_drive_sync.cjs
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
and verifies its outputs and exported ZIP, then reopens and edits the same
`/drive` repository after a page reload and verifies it again after a kernel
restart. The loader tests check sync publication, shared buffers, directory
creation and error propagation. No production writes or credentials are used.
