#!/usr/bin/env bash
set -euo pipefail
repository_root=$(cd "$(dirname "$0")/.." && pwd)
cd "$repository_root"
# Activate Emscripten 4.0.9 before running this script.
command -v emcc >/dev/null
emcc --version | head -1 | grep -F '4.0.9' >/dev/null
export RUSTUP_TOOLCHAIN=nightly-2026-06-26
export RUST_TOOLCHAIN="$RUSTUP_TOOLCHAIN"
# This nightly uses Wasm EH by default; rebuild std with the same side-module ABI.
export CARGO_TARGET_WASM32_UNKNOWN_EMSCRIPTEN_RUSTFLAGS='-C link-arg=-sSIDE_MODULE=2 -Z link-native-libraries=yes'
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$repository_root/target/pyodide}"
export PYODIDE_XBUILDENV_PATH="${PYODIDE_XBUILDENV_PATH:-$repository_root/jupyterlite/.xbuildenv}"
pyodide xbuildenv install 0.29.4 --path "$PYODIDE_XBUILDENV_PATH"
pyodide build --no-isolation -C build-args='-Z build-std=std,panic_abort --locked' --outdir jupyterlite/wheels
