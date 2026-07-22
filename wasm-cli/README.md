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
   install already has it) and prepends that to `PATH` itself — always build
   through `make wasm`/`make wasm-test` rather than running `npm run build`
   directly, or `postbuild:prepare-wasm` will fail with "Unable to find
   micromamba" even when it's installed, since it won't be on `PATH`.

If you only changed TypeScript/CSS/HTML under `wasm-cli/` (not the Rust
`gen` binary itself), `cd wasm-cli && npm run build` is enough to refresh
`dist/`; you don't need to rebuild the wasm `gen` binary or re-run
`postbuild:prepare-wasm` for those changes to take effect. `npm run typecheck`
type-checks without bundling.

`npm run serve` (also what `make wasm-test` calls) does not auto-reload —
restart it after every rebuild before testing, and hard-reload or
unregister the page's service worker in the browser, since it aggressively
caches `dist/` assets across reloads.

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
