#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "${SCRIPT_DIR}")"
GEN_BIN="${GEN_BIN:-${BASE_DIR}/target/release/gen}"
REFERENCE_PATH="${1:-${SCRIPT_DIR}/incremental_vcf_update_fixtures/reference.gb}"
ALIGNED_FASTA_PATH="${2:-${SCRIPT_DIR}/incremental_vcf_update_fixtures/aligned-1000.fa}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-100}"
KEEP_WORKDIR="${KEEP_WORKDIR:-0}"
PROFILE_SAMPLES="${PROFILE_SAMPLES:-}"
PROFILE_MODE="${PROFILE_MODE:-sample}"
PROFILE_DIR="${PROFILE_DIR:-${PWD}/incremental-vcf-profiles}"
BENCHMARK_WORKSPACE=""

usage() {
  printf 'Usage: %s [reference.gb] [aligned.fa[.zst]]\n' "$0"
  printf '\nEnvironment:\n'
  printf '  GEN_BIN         gen executable (default: target/release/gen)\n'
  printf '  SAMPLE_LIMIT    records to process (default: 100; 0 means all)\n'
  printf '  KEEP_WORKDIR    set to 1 to retain the benchmark repository\n'
  printf '  PROFILE_SAMPLES comma-separated sample numbers to profile\n'
  printf '  PROFILE_MODE    sample or exact (default: sample)\n'
  printf '  PROFILE_DIR     profile output directory\n'
}

now_ms() {
  perl -MTime::HiRes=time -e 'printf "%.0f\n", time() * 1000'
}

elapsed_ms() {
  local start_time=$1
  local end_time
  end_time="$(now_ms)"
  printf '%s\n' "$((end_time - start_time))"
}

database_size_bytes() {
  local database_path="${BENCHMARK_WORKSPACE}/.gen/gen.db"
  if [[ ! -f "${database_path}" ]]; then
    printf '0\n'
    return
  fi
  wc -c < "${database_path}" | tr -d '[:space:]'
}

cleanup() {
  if [[ -z "${BENCHMARK_WORKSPACE}" ]]; then
    return
  fi
  if [[ "${KEEP_WORKDIR}" == "1" ]]; then
    printf 'Benchmark workspace kept at %s\n' "${BENCHMARK_WORKSPACE}" >&2
    return
  fi
  rm -rf -- "${BENCHMARK_WORKSPACE}"
}

check_inputs() {
  if [[ ! -x "${GEN_BIN}" ]]; then
    printf 'gen executable is missing: %s\n' "${GEN_BIN}" >&2
    printf 'Build it with: cargo build --release\n' >&2
    exit 1
  fi
  if [[ ! -f "${REFERENCE_PATH}" ]]; then
    printf 'Reference is missing: %s\n' "${REFERENCE_PATH}" >&2
    exit 1
  fi
  if [[ ! -f "${ALIGNED_FASTA_PATH}" ]]; then
    printf 'Aligned FASTA is missing: %s\n' "${ALIGNED_FASTA_PATH}" >&2
    exit 1
  fi
  command -v minimap2 > /dev/null
  command -v paftools.js > /dev/null
  command -v perl > /dev/null
  if [[ ! "${SAMPLE_LIMIT}" =~ ^[0-9]+$ ]]; then
    printf 'SAMPLE_LIMIT must be a non-negative integer: %s\n' "${SAMPLE_LIMIT}" >&2
    exit 1
  fi
  if [[ "${PROFILE_MODE}" != "sample" && "${PROFILE_MODE}" != "exact" ]]; then
    printf 'PROFILE_MODE must be sample or exact: %s\n' "${PROFILE_MODE}" >&2
    exit 1
  fi
}

should_profile() {
  local sample_number=$1
  [[ ",${PROFILE_SAMPLES}," == *",${sample_number},"* ]]
}

stream_aligned_fasta() {
  if [[ "${ALIGNED_FASTA_PATH}" == *.zst ]]; then
    zstd --decompress --stdout -- "${ALIGNED_FASTA_PATH}"
  else
    command cat -- "${ALIGNED_FASTA_PATH}"
  fi
}

process_record() {
  local record_path=$1
  local sample_name=$2
  local sample_number=$3
  local alignment_start
  local alignment_ms
  local update_start
  local update_ms

  alignment_start="$(now_ms)"
  minimap2 -cx asm5 --cs=long reference.fa "${record_path}" > alignment.paf 2> /dev/null
  paftools.js call -f reference.fa -L0 -l0 -s "${sample_name}" alignment.paf > variants.vcf
  alignment_ms="$(elapsed_ms "${alignment_start}")"

  update_start="$(now_ms)"
  if should_profile "${sample_number}"; then
    local profile_path="${PROFILE_DIR}/sample-${sample_number}.txt"
    if [[ "${PROFILE_MODE}" == "sample" ]]; then
      GEN_PROFILE=1 GEN_PROFILE_SAMPLE=1 \
        "${GEN_BIN}" update vcf --reference reference variants.vcf > "${profile_path}"
    else
      GEN_PROFILE=1 \
        "${GEN_BIN}" update vcf --reference reference variants.vcf > "${profile_path}"
    fi
  else
    "${GEN_BIN}" update vcf --reference reference variants.vcf > /dev/null
  fi
  update_ms="$(elapsed_ms "${update_start}")"

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${sample_number}" "${sample_name}" "${alignment_ms}" "${update_ms}" \
    "$((alignment_ms + update_ms))" "$(database_size_bytes)"
}

run_records() {
  local record_path="${BENCHMARK_WORKSPACE}/record.fa"
  local have_record=0
  local sample_name=""
  local sample_number=0
  local line

  while IFS= read -r line || [[ -n "${line}" ]]; do
    if [[ "${line}" == '>'* ]]; then
      if ((have_record == 1)); then
        sample_number=$((sample_number + 1))
        process_record "${record_path}" "${sample_name}" "${sample_number}"
        if ((SAMPLE_LIMIT > 0 && sample_number >= SAMPLE_LIMIT)); then
          return
        fi
      fi
      printf '%s\n' "${line}" > "${record_path}"
      sample_name="${line:1}"
      have_record=1
    elif ((have_record == 1)); then
      printf '%s\n' "${line}" >> "${record_path}"
    elif [[ -n "${line}" ]]; then
      printf 'Invalid FASTA: content found before the first header\n' >&2
      return 1
    fi
  done

  if ((have_record == 1)) && ((SAMPLE_LIMIT == 0 || sample_number < SAMPLE_LIMIT)); then
    sample_number=$((sample_number + 1))
    process_record "${record_path}" "${sample_name}" "${sample_number}"
  fi
}

main() {
  if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    return
  fi
  check_inputs
  if [[ -n "${PROFILE_SAMPLES}" ]]; then
    mkdir -p "${PROFILE_DIR}"
  fi
  BENCHMARK_WORKSPACE="$(mktemp -d "${TMPDIR:-/tmp}/gen-incremental-vcf-benchmark.XXXXXXXX")"
  trap cleanup EXIT HUP INT TERM

  cd "${BENCHMARK_WORKSPACE}"
  "${GEN_BIN}" init > /dev/null
  "${GEN_BIN}" import genbank --reference reference "${REFERENCE_PATH}" > /dev/null
  "${GEN_BIN}" export fasta --sample reference reference.fa > /dev/null

  printf 'sample_number\tsample_name\talignment_ms\tupdate_ms\ttotal_ms\tgen_db_bytes\n'
  run_records < <(stream_aligned_fasta)
}

main "$@"
