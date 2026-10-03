#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
pyodide_cli="${PYODIDE_CLI:-pyodide}"
maturin_cli="${MATURIN_CLI:-maturin}"
gen_target_dir="${CARGO_TARGET_DIR:-${repo_root}/target/pyodide}"
wheel_dir="${1:-${gen_target_dir}/wheels}"

if [[ -z "${PYODIDE_XBUILDENV_PATH:-}" ]]; then
    printf 'Set PYODIDE_XBUILDENV_PATH to the Pyodide 314.0.6 cross-build environment.\n' >&2
    exit 2
fi

if ! command -v "${pyodide_cli}" >/dev/null 2>&1; then
    printf 'Could not find the Pyodide CLI: %s\n' "${pyodide_cli}" >&2
    exit 2
fi

pyodide_cli_path="$(command -v "${pyodide_cli}")"
pyodide_python="${PYODIDE_PYTHON:-$(dirname -- "${pyodide_cli_path}")/python}"
if [[ ! -x "${pyodide_python}" ]]; then
    printf 'Could not find the Python interpreter for the Pyodide CLI: %s\n' "${pyodide_python}" >&2
    exit 2
fi

if ! command -v cargo >/dev/null 2>&1; then
    printf 'Could not find cargo; install the pinned Rust toolchain first.\n' >&2
    exit 2
fi

if ! command -v "${maturin_cli}" >/dev/null 2>&1; then
    printf 'Could not find the pinned Maturin CLI: %s\n' "${maturin_cli}" >&2
    exit 2
fi

if ! command -v rustup >/dev/null 2>&1; then
    printf 'Could not find rustup; install Rust 1.95.0 and its Emscripten target.\n' >&2
    exit 2
fi

pyodide_version="$("${pyodide_cli}" --version)"
if [[ "${pyodide_version}" != *"pyodide CLI version: 0.5.0"* || \
    "${pyodide_version}" != *"pyodide-build version: 0.39.1"* ]]; then
    printf 'Expected pyodide-cli 0.5.0 and pyodide-build 0.39.1.\n' >&2
    exit 2
fi

maturin_version="$("${maturin_cli}" --version)"
if [[ "${maturin_version}" != "maturin 1.15.0" ]]; then
    printf 'Expected Maturin 1.15.0.\n' >&2
    exit 2
fi

if ! maturin_backend_version="$("${pyodide_python}" -c 'from importlib.metadata import version; import maturin; print(version("maturin"))' 2>/dev/null)"; then
    printf 'The Pyodide CLI Python must be able to import the Maturin build backend.\n' >&2
    exit 2
fi
if [[ "${maturin_backend_version}" != "1.15.0" ]]; then
    printf 'Expected Maturin 1.15.0 in the Pyodide CLI Python environment.\n' >&2
    exit 2
fi

xbuildenv_release="${PYODIDE_XBUILDENV_PATH}/314.0.6"
if [[ ! -f "${xbuildenv_release}/.emscripten-version" ]] ||
    ! grep --quiet --fixed-strings --line-regexp '5.0.3' "${xbuildenv_release}/.emscripten-version" ||
    ! grep --quiet --fixed-strings --line-regexp '3.14' "${xbuildenv_release}/.build-python-version"; then
    printf 'PYODIDE_XBUILDENV_PATH must point to the Pyodide 314.0.6 xbuildenv (Python 3.14, Emscripten 5.0.3).\n' >&2
    exit 2
fi

if ! rustup target list --toolchain 1.95.0 --installed |
    grep --quiet --fixed-strings --line-regexp 'wasm32-unknown-emscripten'; then
    printf 'Install wasm32-unknown-emscripten for Rust 1.95.0 first.\n' >&2
    exit 2
fi

export CARGO_TARGET_DIR="${gen_target_dir}"
export RUST_TOOLCHAIN=1.95.0
export RUSTUP_TOOLCHAIN=1.95.0
export UV_CACHE_DIR="${UV_CACHE_DIR:-${gen_target_dir}/uv-cache}"
mkdir -p "${wheel_dir}" "${UV_CACHE_DIR}"

"${pyodide_cli}" build "${repo_root}" --outdir "${wheel_dir}" \
    --no-isolation --skip-dependency-check \
    --config-setting 'build-args=--no-default-features --features extension-module'
