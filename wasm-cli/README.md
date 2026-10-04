# gen-wasm-cli

A self-contained, browser-based terminal that runs the `gen` CLI compiled to
`wasm32-unknown-emscripten`, alongside a few coreutils (`cd`, `ls`, `grep`,
`less`, `sed`, and friends) provided by
[`@jupyterlite/cockle`](https://github.com/jupyterlite/cockle) and rendered
with [xterm.js](https://xtermjs.org/). No sibling checkout of the cockle repo
is needed; cockle is consumed as a published npm package.

## Building

```sh
make wasm       # builds gen for wasm32-unknown-emscripten and bundles wasm-cli/dist/
make wasm-test  # builds, then serves wasm-cli/dist/ at http://localhost:4501
```

Both targets live in the root `Makefile`. One-time toolchain setup (see the
comment above the `wasm` target for the up-to-date version pins):

1. **emsdk** (`emcc`/`em++`/`emar`):
   ```sh
   git clone https://github.com/emscripten-core/emsdk.git ~/emsdk
   ~/emsdk/emsdk install 4.0.9 && ~/emsdk/emsdk activate 4.0.9
   ```
2. **micromamba**, used by cockle's own wasm-package fetch step
   (`postbuild:prepare-wasm` below) to pull in `coreutils`/`grep`/`less`/`sed`/
   `cockle_fs`:
   ```sh
   brew install micromamba
   ```
   The `wasm` Makefile target expects `micromamba` at `$(MICROMAMBA_DIR)`
   (defaults to `/opt/homebrew/Caskroom/miniforge/base`, i.e. a miniforge
   install already has it) and prepends that to `PATH` itself. If you only
   have it via miniforge (not also `brew install`ed standalone), `npm run
   build` run directly -- rather than through `make wasm`/`make wasm-test` --
   won't find it on `PATH` and `postbuild:prepare-wasm` will fail with
   "Unable to find micromamba" even though it's installed; either prefix
   `PATH` the same way the Makefile does
   (`PATH="$(MICROMAMBA_DIR):$PATH" npm run build`) or build through `make
   wasm`/`make wasm-test` instead.

If you only changed TypeScript/CSS/HTML under `wasm-cli/` (not the Rust
`gen` binary itself) and `wasm-cli/gen-wasm/` is already populated from a
prior `make wasm`, `cd wasm-cli && npm run build` is enough to refresh
`dist/` -- you don't need to rebuild the wasm `gen` binary itself. This does
still re-run `postbuild:prepare-wasm` every time (npm always chains a
`postbuild` script after `build`; there's no flag to skip it), so the
`micromamba`-on-`PATH` requirement above still applies. `npm run typecheck`
type-checks without bundling or needing `micromamba` at all.

`npm run serve` (also what `make wasm-test` calls) does not auto-reload —
restart it after every rebuild before testing, and hard-reload or
unregister the page's service worker in the browser, since it aggressively
caches `dist/` assets across reloads.

## Browser HTTP transport

Gen's Rust HTTP interface uses `src/commands/remote/http/browser_http.c` on
`target_os = "emscripten"`. The root crate's build script compiles this shim
for both the standalone CLI and consumers such as the PyO3 extension. It
uses `EM_JS` and synchronous XMLHttpRequest, with no Fetch library or Python
networking dependency. Native builds continue to use reqwest. DoltLite's
own upstream XHR transport remains separate and unchanged.

Run Gen in a **browser worker**: synchronous binary XHR is unavailable on
the main window thread. Cockle already runs the CLI in a worker; a future
Pyodide integration must do the same. This change does not enable Rust or
DoltLite threading. The shared-memory test configuration exercises buffer
compatibility, rather than enabling application threads.

Browsers enforce CORS, including preflight for authorization, custom headers
and binary uploads. Remote servers must allow the site's origin, request
methods and headers. TLS, mixed-content and CORS failures produce a sanitized
network error; browser exception text is discarded to avoid exposing tokens
or request URLs. HTTP responses, including 4xx/5xx, retain their status and
binary body for the caller to interpret. No cross-origin cookie credentials
are enabled automatically.

The C boundary borrows input buffers only during the call. Request bytes are
copied into an ordinary Uint8Array before XHR, because XHR cannot accept
shared Wasm memory. Responses use ArrayBuffer, with no text conversion.
C malloc owns the response buffer; Rust copies it into a fallibly allocated
Vec and releases the C buffer on every return path. The heap view is resolved
again after malloc, which may grow memory. Response lengths above wasm32's
signed addressable range are rejected. Keep `ABORTING_MALLOC=0` in consumers
so allocation failures can return errors (Rust's Emscripten linker already
sets this). Requests and responses remain fully buffered.

The shim references only Emscripten's heap and Module globals, with no
separately exported string or allocation helpers. Emscripten's dynamic linker
loads side-module EM_JS bodies using dynamic evaluation; deployments must
permit this under their CSP. See the
[Emscripten dynamic linking documentation](https://emscripten.org/docs/compiling/Dynamic-Linking.html).
Passing the harness below establishes compatibility with the standalone
SDK's dynamic loader; it does **not** establish compatibility with a selected
Pyodide runtime. That remains part of the wheel implementation.

### Transport regression tests

With Emscripten 4.0.9 activated and Python Playwright/Chromium installed:

```sh
source "$HOME/emsdk/emsdk_env.sh"
python wasm-cli/tests/test_browser_http.py
```

The test starts disposable local servers and exercises Chromium workers
using static and dynamically loaded side modules, with ordinary and shared
Wasm memory. Cases cover binary POST/PUT/PATCH/DELETE, GET/HEAD without a request
body or headers, custom headers and CORS
preflight, empty responses, HTTP error bodies, a 20 MiB response that forces
memory growth, injected C allocation failure, blocked CORS and network failure.
It does not contact a production remote or require credentials.

## Scripting the terminal from the embedding page

This page has no copy or links of its own for running specific commands (that context lives with
whoever embeds it, e.g. GenHub's `/terminal` page), so it exposes a small `postMessage` contract
instead, handled in `src/index.ts`'s `setupCommandRunner`/`setupThemeListener`:

- **`{ type: 'gen-wasm-cli:ready' }`** — posted by this page to `window.parent` (target origin
  `'*'`, since this page doesn't know the embedder's origin in advance) once the shell has started
  and is ready to accept input. Wait for this before sending commands that should run
  automatically as soon as the terminal loads; it's not needed before a click-triggered command,
  since the reader can't click before the page has rendered anyway.
- **`{ type: 'gen-wasm-cli:run-commands', commands: string[] }`** — sent by the embedding page to
  this iframe's `contentWindow` to type and submit each command in turn, pausing briefly after
  each is typed so the reader can read it before its output appears. Only accepted from
  `window.parent` (this page has no notion of the embedder's origin to allowlist instead).
- **`{ type: 'gen-wasm-cli:set-theme', mode: 'light' | 'dark' }`** — sent by the embedding page to
  this iframe's `contentWindow` to switch the terminal's rendered theme, page background, and the
  shell's `GEN_THEME` environment variable together, in step with the embedder's own theme. This
  page has no theme control of its own — it starts from a `?theme=light|dark` query parameter on
  its `src` URL (falling back to `prefers-color-scheme` if omitted, e.g. when opened standalone via
  `make wasm-test`) and only changes afterward in response to this message. Only accepted from
  `window.parent`.

## Testing with Playwright

Drive the terminal with headless Playwright, **in Python** (not
`.mjs`/`.js`), and **not** the Chrome MCP extension — those are this
project's standing conventions for browser-driven testing, not specific to
wasm-cli.

Two details are easy to get wrong and will make the shell appear to hang
forever with no console output and an empty terminal buffer:

- Navigate to the **served root** (`http://localhost:4501`), not
  `http://localhost:4501/index.html` explicitly — the explicit path can end
  up on the wrong side of the service worker's registration scope and the
  shell never finishes booting.
- Wait for `.xterm` to appear (`page.wait_for_selector(".xterm", timeout=30000)`)
  rather than a fixed `sleep()`, then give it a couple more seconds and click
  into the terminal to focus it before typing — xterm.js only accepts
  keyboard input once focused.

```python
import time
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto("http://localhost:4501")
    page.wait_for_selector(".xterm", timeout=30000)
    time.sleep(2)
    page.click(".xterm")
    page.keyboard.type("gen clone http://localhost:5800/api/repos/admin/small-repo")
    page.keyboard.press("Enter")
    time.sleep(8)  # wait for the async operation to actually finish, watch a screenshot first
    page.keyboard.type("gen list-samples")  # one command at a time -- this shell does not support `&&`
    page.keyboard.press("Enter")
    time.sleep(4)
    page.screenshot(path="<scratchpad>/whatever.png")
```

Read the result back with a screenshot (and the `Read` tool on the resulting
image), not `page.inner_text(".xterm")` — that has returned empty even when
the terminal clearly had content rendered.

For anything exercising `gen clone`/`push`/`pull`/remote login against a real
GenHub backend, you additionally need the genhub backend running
(`docker compose ps` in the genhub checkout for `postgres`/`gcs-server`, plus
its API server on `:5800` via `make wasm-backend`) — plain coreutils and
local-only `gen` commands (`ls`, `cd`, `gen init`, etc.) don't need any of
that.

To test the `postMessage` contract from "Scripting the terminal from the
embedding page" above (rather than typing directly into the terminal), serve
`dist/` (`npm run serve`) and drive a small local harness page that embeds it
in an iframe and posts `run-commands`/listens for `ready` the same way a real
embedder would, instead of navigating Playwright to `:4501` directly.

### DriveFS database regression

Cockle's shared DriveFS buffers writes until close. `init_drive_fs.ts` implements
its synchronous `fsync` and forwards PROXYFS sync calls to that stream, so Dolt's
branch-head confirmation can reopen the bytes it just wrote. Keep both layers:
adding only DriveFS sync leaves command modules' stock PROXYFS dropping it.
The existing pinned DoltLite fork remains in use; no additional trust-local
patch is needed for fresh import.

After `make wasm`, run:

```sh
python wasm-cli/tests/test_drive_import.py
```

This checks init, fresh FASTA import, sample/history reads, and sample persistence
after reloading the page. It saves screenshots and a terminal transcript.
