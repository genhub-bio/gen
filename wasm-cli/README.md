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

## Scripting the terminal from the embedding page

This page has no copy or links of its own for running specific commands (that context lives with
whoever embeds it, e.g. GenHub's `/terminal` page), so it exposes a small `postMessage` contract
instead, handled in `src/index.ts`'s `setupCommandRunner`:

- **`{ type: 'gen-wasm-cli:ready' }`** — posted by this page to `window.parent` (target origin
  `'*'`, since this page doesn't know the embedder's origin in advance) once the shell has started
  and is ready to accept input. Wait for this before sending commands that should run
  automatically as soon as the terminal loads; it's not needed before a click-triggered command,
  since the reader can't click before the page has rendered anyway.
- **`{ type: 'gen-wasm-cli:run-commands', commands: string[] }`** — sent by the embedding page to
  this iframe's `contentWindow` to type and submit each command in turn, pausing briefly after
  each is typed so the reader can read it before its output appears. Only accepted from
  `window.parent` (this page has no notion of the embedder's origin to allowlist instead).

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
