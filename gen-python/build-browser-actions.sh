#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
output_dir="${1:?Usage: build-browser-actions.sh OUTPUT_DIR}"
target_dir="${CARGO_TARGET_DIR:-${repo_root}/target/browser-actions}"

if [[ -z "${PYODIDE_XBUILDENV_PATH:-}" ]]; then
    printf 'Set PYODIDE_XBUILDENV_PATH to the Pyodide 314.0.6 cross-build environment.\n' >&2
    exit 2
fi

xbuildenv_release="${PYODIDE_XBUILDENV_PATH}/314.0.6"
if [[ ! -f "${xbuildenv_release}/.emscripten-version" ]] ||
    ! grep --quiet --fixed-strings --line-regexp '5.0.3' "${xbuildenv_release}/.emscripten-version" ||
    ! grep --quiet --fixed-strings --line-regexp '3.14' "${xbuildenv_release}/.build-python-version"; then
    printf 'PYODIDE_XBUILDENV_PATH must point to the Pyodide 314.0.6 xbuildenv (Python 3.14, Emscripten 5.0.3).\n' >&2
    exit 2
fi
if ! command -v cargo >/dev/null 2>&1 || ! command -v rustup >/dev/null 2>&1; then
    printf 'Could not find cargo and rustup.\n' >&2
    exit 2
fi
if ! rustup target list --toolchain 1.95.0 --installed |
    grep --quiet --fixed-strings --line-regexp 'wasm32-unknown-emscripten'; then
    printf 'Install wasm32-unknown-emscripten for Rust 1.95.0 first.\n' >&2
    exit 2
fi

source "${xbuildenv_release}/emsdk/emsdk_env.sh" >/dev/null
if ! command -v emcc >/dev/null 2>&1 || ! emcc --version | head -n 1 | grep --quiet --fixed-strings '5.0.3'; then
    printf 'The Pyodide cross-build environment did not provide Emscripten 5.0.3.\n' >&2
    exit 2
fi

export CARGO_TARGET_DIR="${target_dir}"
export RUST_TOOLCHAIN=1.95.0
export RUSTUP_TOOLCHAIN=1.95.0

declare -a rustflags=(
    -C 'link-arg=-sMODULARIZE=1'
    -C 'link-arg=-sEXPORT_ES6=1'
    -C 'link-arg=-sEXPORT_NAME=createGenActionsWasm'
    -C 'link-arg=-sEXIT_RUNTIME=0'
    -C 'link-arg=-sALLOW_MEMORY_GROWTH=1'
    -C 'link-arg=-sENVIRONMENT=web,worker'
    -C 'link-arg=-sEXPORTED_FUNCTIONS=["_main","_genhub_repository_command","_genhub_free_string","_malloc","_free"]'
    -C 'link-arg=-sEXPORTED_RUNTIME_METHODS=["UTF8ToString","stringToNewUTF8"]'
)
if [[ -n "${CARGO_ENCODED_RUSTFLAGS:-}" ]]; then
    IFS=$'\x1f' read -r -a inherited_rustflags <<< "${CARGO_ENCODED_RUSTFLAGS}"
    rustflags=("${inherited_rustflags[@]}" "${rustflags[@]}")
fi
CARGO_ENCODED_RUSTFLAGS="$(IFS=$'\x1f'; printf '%s' "${rustflags[*]}")"
export CARGO_ENCODED_RUSTFLAGS

cargo build \
    --manifest-path "${repo_root}/gen-wasm/Cargo.toml" \
    --target wasm32-unknown-emscripten \
    --bin gen_browser_actions \
    --release

wasm_target_dir="${target_dir}/wasm32-unknown-emscripten/release"
mkdir -p "${output_dir}"
python3 - "${wasm_target_dir}/gen_browser_actions.js" "${output_dir}/gen_actions.js" <<'PY'
from pathlib import Path
import sys

runtime_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
runtime = runtime_path.read_text(encoding="utf-8").rstrip("\n")
export = "export default createGenActionsWasm;"
if not runtime.endswith(export):
    raise SystemExit("Emscripten runtime has an unexpected default export")
output_path.write_text(runtime[: -len(export)], encoding="utf-8")
PY
cat "${repo_root}/gen-wasm/facade.mjs" >> "${output_dir}/gen_actions.js"
cp "${wasm_target_dir}/gen_browser_actions.wasm" "${output_dir}/gen_actions.wasm"
test -s "${output_dir}/gen_actions.js"
test -s "${output_dir}/gen_actions.wasm"
