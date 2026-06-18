#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

find_wasi_sysroot() {
  if [[ -n "${WASI_SYSROOT:-}" && -f "${WASI_SYSROOT}/include/stdio.h" ]]; then
    printf 'sdk:%s\n' "$WASI_SYSROOT"
    return 0
  fi

  if [[ -n "${WASI_SDK_PATH:-}" && -f "${WASI_SDK_PATH}/share/wasi-sysroot/include/stdio.h" ]]; then
    printf 'sdk:%s\n' "${WASI_SDK_PATH}/share/wasi-sysroot"
    return 0
  fi

  for candidate in \
    /opt/wasi-sdk/share/wasi-sysroot \
    /usr/share/wasi-sysroot \
    /usr/local/share/wasi-sysroot
  do
    if [[ -f "${candidate}/include/stdio.h" ]]; then
      printf 'sdk:%s\n' "$candidate"
      return 0
    fi
  done

  if [[ -f /usr/include/wasm32-wasi/stdio.h && -f /usr/lib/wasm32-wasi/libc.a ]]; then
    printf 'debian:/usr\n'
    return 0
  fi

  for candidate in /usr/local /opt/wasi
  do
    if [[ -f "${candidate}/include/wasm32-wasi/stdio.h" && -f "${candidate}/lib/wasm32-wasi/libc.a" ]]; then
      printf 'debian:%s\n' "$candidate"
      return 0
    fi
  done

  return 1
}

if ! wasi_sysroot="$(find_wasi_sysroot)"; then
  cat >&2 <<'EOF'
Missing WASI C sysroot.

libsqlite3-sys builds bundled SQLite from C and runs bindgen over sqlite3.h.
For wasm32-wasip1 that requires WASI libc headers and libraries.

Install wasi-sdk or wasi-libc, then set one of:
  export WASI_SDK_PATH=/opt/wasi-sdk
  export WASI_SYSROOT=/opt/wasi-sdk/share/wasi-sysroot

On Debian/Ubuntu, these packages are often enough:
  sudo apt-get install clang wasi-libc
EOF
  exit 1
fi

sysroot_kind="${wasi_sysroot%%:*}"
sysroot_path="${wasi_sysroot#*:}"

case "$sysroot_kind" in
  sdk)
    export WASI_SYSROOT="$sysroot_path"
    export CFLAGS_wasm32_wasip1="--target=wasm32-wasi --sysroot=${sysroot_path} ${CFLAGS_wasm32_wasip1:-}"
    export BINDGEN_EXTRA_CLANG_ARGS_wasm32_wasip1="--target=wasm32-wasi --sysroot=${sysroot_path} ${BINDGEN_EXTRA_CLANG_ARGS_wasm32_wasip1:-}"
    ;;
  debian)
    export WASI_SYSROOT="$sysroot_path"
    export CFLAGS_wasm32_wasip1="--target=wasm32-wasi --sysroot=${sysroot_path} -isystem ${sysroot_path}/include/wasm32-wasi -L${sysroot_path}/lib/wasm32-wasi ${CFLAGS_wasm32_wasip1:-}"
    export BINDGEN_EXTRA_CLANG_ARGS_wasm32_wasip1="--target=wasm32-wasi --sysroot=${sysroot_path} -isystem ${sysroot_path}/include/wasm32-wasi ${BINDGEN_EXTRA_CLANG_ARGS_wasm32_wasip1:-}"
    ;;
  *)
    printf 'Unknown WASI sysroot kind: %s\n' "$sysroot_kind" >&2
    exit 1
    ;;
esac

sqlite_wasi_flags=(
  -USQLITE_THREADSAFE
  -DSQLITE_THREADSAFE=0
  -DLONGDOUBLE_TYPE=double
  -D_WASI_EMULATED_MMAN
  -D_WASI_EMULATED_GETPID
  -D_WASI_EMULATED_SIGNAL
  -D_WASI_EMULATED_PROCESS_CLOCKS
)
export LIBSQLITE3_FLAGS="${sqlite_wasi_flags[*]} ${LIBSQLITE3_FLAGS:-}"

if [[ -n "${WASI_SDK_PATH:-}" && -x "${WASI_SDK_PATH}/bin/clang" ]]; then
  export CC_wasm32_wasip1="${CC_wasm32_wasip1:-${WASI_SDK_PATH}/bin/clang}"
elif command -v clang >/dev/null 2>&1; then
  export CC_wasm32_wasip1="${CC_wasm32_wasip1:-clang}"
fi

rustup target add wasm32-wasip1
cargo build \
  -p gen \
  --bin gen \
  --no-default-features \
  --features browser-wasi \
  --target wasm32-wasip1 \
  --release

cp target/wasm32-wasip1/release/gen.wasm examples/browser-ratatui/gen.wasm
