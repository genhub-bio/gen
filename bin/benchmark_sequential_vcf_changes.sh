#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "${SCRIPT_DIR}")"
GEN_BIN="${GEN_BIN:-${BASE_DIR}/target/release/gen}"
FASTA_PATH="${BASE_DIR}/fixtures/simple.fa"
FIXTURE_DIR="${BASE_DIR}/fixtures/benchmarks/100_sequential_changes"
GENERATOR_SCRIPT="${BASE_DIR}/bin/generate_sequential_change_fixtures.sh"
KEEP_WORKDIR="${KEEP_WORKDIR:-0}"
BENCHMARK_WORKSPACE=""

now_ms() {
  perl -MTime::HiRes=time -e 'printf "%.0f\n", time() * 1000'
}

read_fasta_sequence() {
  awk 'BEGIN { sequence = "" } /^>/ { next } { sequence = sequence $0 } END { print sequence }' "${FASTA_PATH}"
}

run_timed() {
  local start_time
  start_time="$(now_ms)"
  local command_output
  if ! command_output="$("$@" 2>&1)"; then
    printf '%s\n' "${command_output}" >&2
    return 1
  fi
  if [[ "${command_output}" == *"No changes made."* ]]; then
    printf '%s\n' "${command_output}" >&2
    return 1
  fi
  local end_time
  end_time="$(now_ms)"
  printf '%s\n' "$((end_time - start_time))"
}

ensure_benchmark_binary() {
  if [[ -x "${GEN_BIN}" ]]; then
    return
  fi

  cargo build --release
}

ensure_fixtures() {
  "${GENERATOR_SCRIPT}" > /dev/null
}

record_row() {
  local task="$1"
  local duration_ms="$2"
  local cumulative_ms="$3"
  printf '%-24s %-14s %-14s\n' "${task}" "${duration_ms}" "${cumulative_ms}"
}

cleanup() {
  if [[ -z "${BENCHMARK_WORKSPACE}" ]]; then
    return
  fi

  if [[ "${KEEP_WORKDIR}" == "1" ]]; then
    printf 'Benchmark workspace kept at %s\n' "${BENCHMARK_WORKSPACE}"
    return
  fi

  rm -rf "${BENCHMARK_WORKSPACE}"
}

main() {
  ensure_benchmark_binary
  ensure_fixtures

  local sequence
  sequence="$(read_fasta_sequence)"
  local sequence_length=${#sequence}
  BENCHMARK_WORKSPACE="$(mktemp -d /tmp/gen-sequential-benchmark.XXXXXX)"
  trap cleanup EXIT

  printf 'Sequential VCF benchmark\n'
  printf 'Workspace: %s\n' "${BENCHMARK_WORKSPACE}"
  printf 'FASTA: %s\n' "${FASTA_PATH}"
  printf 'Fixtures: %s\n' "${FIXTURE_DIR}"
  printf 'Total changes: %s\n\n' "$((sequence_length * 3))"

  (
    cd "${BENCHMARK_WORKSPACE}"
    "${GEN_BIN}" init > /dev/null 2>&1
  )

  local cumulative_ms=0
  local import_ms
  import_ms=$(
    (
      cd "${BENCHMARK_WORKSPACE}"
      run_timed "${GEN_BIN}" import fasta "${FASTA_PATH}" --reference reference
    )
  )
  cumulative_ms=$((cumulative_ms + import_ms))

  printf '%-24s %-14s %-14s\n' 'Task' 'Time (ms)' 'Cumulative (ms)'
  printf '%-24s %-14s %-14s\n' '------------------------' '--------------' '---------------'
  record_row 'reference_import' "${import_ms}" "${cumulative_ms}"

  local nucleotides=(A C T G)
  local position
  for ((position = 1; position <= sequence_length; position += 1)); do
    local reference_base="${sequence:position-1:1}"

    local alternate_base
    for alternate_base in "${nucleotides[@]}"; do
      if [[ "${alternate_base}" == "${reference_base}" ]]; then
        continue
      fi

      local sample_name="change_${position}_${alternate_base}"
      local vcf_path="${FIXTURE_DIR}/${sample_name}.vcf"
      local duration_ms
      duration_ms=$(
        (
          cd "${BENCHMARK_WORKSPACE}"
          run_timed "${GEN_BIN}" update vcf "${vcf_path}" --parent-samples reference
        )
      )
      cumulative_ms=$((cumulative_ms + duration_ms))
      record_row "${sample_name}" "${duration_ms}" "${cumulative_ms}"
    done
  done
}

main "$@"
