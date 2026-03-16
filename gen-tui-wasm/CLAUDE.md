# gen-tui-wasm

Web-based GenGraph viewer built with [Ratzilla](https://github.com/ratatui/ratzilla) (Ratatui over WASM). The goal is to replicate the GenGraph visualization from `src/views/gen_graph_widget.rs` in a browser, without any dependency on SQLite/Rusqlite.

## Architecture

```
Browser (WASM)                      Server (native Rust)
─────────────────────────────────   ──────────────────────────────
gen-tui-wasm (this crate)           server crate (TBD)
 ├── Ratzilla (WebGl2Backend)         ├── gen-models / DB access
 ├── gen-tui (no crossterm)           └── serializes graph + sequences
 └── renders GenGraph                     → sends to browser
```

- **No DB access in WASM.** `gen-models` and `rusqlite` must never be added as dependencies here.
- Genomic sequence data and graph structure are provided by a server-side Rust application, serialized (e.g. JSON/bincode) and sent to the browser.
- The WASM crate deserializes the data and hands it to `gen-tui` types for rendering.

## Key gen-tui types

See `../gen-tui/src/` for full API. The main integration points:

- `GraphController<G, S>` — owns viewport state, cursor, camera, zoom, layout
- `plot_viewport_graph(...)` — renders the visible subgraph into a `WorldBuffer`
- `NodeRenderer<G>` / `NodeSizer<G>` — domain-specific traits to implement for GenGraph data
- `WorldBuffer` — wraps a ratatui `Buffer` with world-coordinate helpers
- `VisualDetail` — `Minimal` / `Truncated` / `Full` zoom levels

In the WASM context the graph is leaked to `'static` so it can live inside `Rc<RefCell<App>>`. See `main.rs` for the current pattern.

## Ratzilla patterns

```rust
// App state held in Rc<RefCell<_>>
let app = Rc::new(RefCell::new(App::new()));

// Key events
terminal.on_key_event(move |key_event| { ... });

// Render loop (browser RAF)
terminal.draw_web(move |frame| { ... });
```

Use `WebGl2Backend` (requires a bitmap font atlas). `DomBackend` (DOM/CSS rendering) is an alternative but we use WebGL2 here.

## Build & serve

```sh
# Prerequisites (once)
rustup target add wasm32-unknown-unknown
cargo install --locked trunk

# Dev server
trunk serve                  # http://localhost:8080

# Release bundle → dist/
trunk build --release
```

## WASM constraints

- `gen-tui` is included with `default-features = false` (disables `crossterm`).
- `tachyonfx` must use `features = ["wasm"]` (transitive dep via gen-tui).
- `getrandom` requires `features = ["wasm_js"]`.
- Always call `console_error_panic_hook::set_once()` at startup.
- No threads, no blocking I/O, no filesystem access.

## Reference

- Ratzilla docs: `https://docs.rs/ratzilla`
- Ratzilla repo (local): `/Users/bvh/git/ratzilla/`
- exabind web example (local): `/Users/bvh/git/exabind/web/`
- GenGraph widget (desktop): `../src/views/gen_graph_widget.rs`
- gen-tui library: `../gen-tui/`
