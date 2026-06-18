# Gen Ratatui Browser Demo

This demo runs the real Gen Ratatui full-screen TUI inside `xterm.js`, with `gen.wasm` executing client-side through `@wasmer/sdk`.

Build the WASI binary from the repository root:

```bash
./examples/browser-ratatui/build-gen-wasm.sh
```

Then run the browser demo:

```bash
npm install
npm run dev
```

The Vite dev server sets COOP/COEP headers. The page mounts `/workspace` with `plasmid.fa`, `plasmid_mut.fa`, and `feature.gff3`, then starts `gen browser-demo-tui`.

`wasm32-wasip1` requires a WASI-capable C toolchain for bundled SQLite. If `libsqlite3-sys` fails with C header errors such as `stdarg.h` or `bits/libc-header-start.h` not being found, install WASI libc headers and rerun the helper script.

On Debian/Ubuntu, these packages are often enough:

```bash
sudo apt-get install clang wasi-libc
```

The helper supports Debian's split layout (`/usr/include/wasm32-wasi` and `/usr/lib/wasm32-wasi`) and wasi-sdk's `share/wasi-sysroot` layout.

For wasi-sdk installs, set:

```bash
export WASI_SDK_PATH=/opt/wasi-sdk
```

or:

```bash
export WASI_SYSROOT=/opt/wasi-sdk/share/wasi-sysroot
```
