#!/usr/bin/env bash
# Build the repositories bin/benchmark_lazy_view.py times: "mhc" (many small GFA segments) and
# "hub" (chr22 plus a 100k-variant VCF, one node with ~390k edges). Viewing is read-only, so
# these are built once and reused; delete target/lazy-view-bench to start over.
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GEN_BIN="${GEN_BIN:-${BASE_DIR}/target/release/gen}"
OUT_DIR="${OUT_DIR:-${BASE_DIR}/target/lazy-view-bench}"
MHC_GFA="${MHC_GFA:-${BASE_DIR}/../gen-large-files/MHC-57.gfa}"
FIXTURES="${BASE_DIR}/fixtures"

setup() {
  local name="$1"
  shift
  if [ -e "${OUT_DIR}/${name}/.benchmark-ready" ]; then
    echo "${name}: already built"
    return
  fi
  echo "${name}: building"
  rm -rf "${OUT_DIR:?}/${name}"
  mkdir -p "${OUT_DIR}/${name}"
  (cd "${OUT_DIR}/${name}" && "${GEN_BIN}" init && "$@" </dev/null)
  touch "${OUT_DIR}/${name}/.benchmark-ready"
}

setup_mhc() {
  "${GEN_BIN}" import gfa "${MHC_GFA}" --sample mhc
}

setup_hub() {
  "${GEN_BIN}" import fasta -n test --reference ref "${FIXTURES}/chr22.fa.gz"
  "${GEN_BIN}" update vcf -n test -g "0|1" -s test --parent-samples ref \
    "${FIXTURES}/chr22_100k_no_samples.vcf.gz"
}

setup mhc setup_mhc
setup hub setup_hub
echo "done: ${OUT_DIR}"
