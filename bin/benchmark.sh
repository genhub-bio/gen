#!/usr/bin/env bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_DIR="$(dirname "${SCRIPT_DIR}")"
GEN_DIR="${BASE_DIR}/.gen"
GEN_BIN="${BASE_DIR}/target/release/gen"
PROFILE_BIN="${BASE_DIR}/target/debug/gen"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
RESULTS=""
PROFILE_DIR="${BASE_DIR}/.benchmark-profiles"
PROFILE_RESULTS=""
PROFILE_BENCHMARKS="${PROFILE_BENCHMARKS:-1}"

init_test() {
  cd ${BASE_DIR}
  rm -rf ${GEN_DIR}
  ${GEN_BIN} init 2> /dev/null > /dev/null
}

get_size () {
  GEN_DIR="${GEN_DIR}" GEN_DB_PATH="${GEN_DIR}/gen.db" DEFAULT_DB_PATH="${GEN_DIR}/default.db" ${PYTHON_BIN} -c "import os
from pathlib import Path

def size_mb(path):
    p = Path(path)
    if not p.exists():
        return '0.0000'
    return f'{p.stat().st_size / (1024 * 1024):.4f}'

gen_dir = os.environ['GEN_DIR']
gen_db_path = os.environ['GEN_DB_PATH']
default_db_path = os.environ['DEFAULT_DB_PATH']

total = 0
for root, _, files in os.walk(gen_dir):
    for name in files:
        total += os.path.getsize(os.path.join(root, name))

print(f'{total / (1024 * 1024):.4f} {size_mb(gen_db_path)} {size_mb(default_db_path)}')"
}

time_taken() {
  local start_time=$(${PYTHON_BIN} -c "import time;print(round(time.time()*1000))")
  "$@" > /dev/null 2> /dev/null
  local end_time=$(${PYTHON_BIN} -c "import time;print(round(time.time()*1000))")
  local duration=$((end_time - start_time))
  echo $duration
}

record_result() {
  local task="$1"
  local duration="$2"
  local total_size="$3"
  local gen_db_size="$4"
  local default_db_size="$5"
  RESULTS+=$(printf "%-35s %-10s %-10s %-10s %-10s\n" "${task}" "${duration}" "${total_size}" "${gen_db_size}" "${default_db_size}")
  RESULTS+=$'\n'
}

record_profile_result() {
  local task="$1"
  local profile_file="$2"
  PROFILE_RESULTS+=$(printf "%-35s %s\n" "${task}" "${profile_file}")
  PROFILE_RESULTS+=$'\n'
}

profile_taken() {
  local output_file="$1"
  shift
  local start_time=$(${PYTHON_BIN} -c "import time;print(round(time.time()*1000))")
  "$@" > "${output_file}" 2> /dev/null
  local end_time=$(${PYTHON_BIN} -c "import time;print(round(time.time()*1000))")
  local duration=$((end_time - start_time))
  echo $duration
}

run_benchmark() {
  local task="$1"
  shift
  local record_profile=1

  if [[ "${1:-}" == "--no-profile-record" ]]; then
    record_profile=0
    shift
  fi

  if [[ "${PROFILE_BENCHMARKS}" == "1" ]]; then
    local profile_file
    profile_file="$(profile_path "${task}")"
    local duration
    duration=$(profile_taken "${profile_file}" "$@")
    if [[ "${record_profile}" == "1" ]]; then
      record_profile_result "${task}" "${profile_file}"
    fi
    printf '%s' "${duration}"
  else
    time_taken "$@"
  fi
}

latest_operation_hash() {
  ${GEN_BIN} operations | awk 'NR > 1 { if ($1 == ">") hash=$2; else hash=$1 } END { print hash }'
}

first_operation_hash() {
  ${GEN_BIN} operations | awk 'NR > 1 { if ($1 == ">") print $2; else print $1; exit }'
}

setup_simple_fasta_repo() {
  init_test
  ${GEN_BIN} import fasta ${BASE_DIR}/fixtures/simple.fa --reference reference > /dev/null 2> /dev/null
}

setup_hg_branch_apply_repo() {
  init_test
  ${GEN_BIN} import fasta ${BASE_DIR}/fixtures/chr22.fa.gz --reference reference --shallow > /dev/null 2> /dev/null
  ${GEN_BIN} checkout -b branch-a > /dev/null 2> /dev/null
  ${GEN_BIN} update vcf ${BASE_DIR}/fixtures/HG00096.vcf.gz --parent-samples reference > /dev/null 2> /dev/null
  BRANCH_A_HASH=$(latest_operation_hash)
  ${GEN_BIN} checkout main > /dev/null 2> /dev/null
}

setup_hg_branch_switch_repo() {
  setup_hg_branch_apply_repo
  ${GEN_BIN} checkout -b branch-b > /dev/null 2> /dev/null
  ${GEN_BIN} update vcf ${BASE_DIR}/fixtures/HG00097.vcf.gz --parent-samples reference > /dev/null 2> /dev/null
}

setup_hg_reset_repo() {
  init_test
  ${GEN_BIN} import fasta ${BASE_DIR}/fixtures/chr22.fa.gz --reference reference --shallow > /dev/null 2> /dev/null
  RESET_HASH=$(first_operation_hash)
  ${GEN_BIN} update vcf ${BASE_DIR}/fixtures/HG00096.vcf.gz --parent-samples reference > /dev/null 2> /dev/null
}

setup_library_update_repo() {
  setup_simple_fasta_repo
}

setup_gfa_update_repo() {
  init_test
  ${GEN_BIN} import gfa ${BASE_DIR}/fixtures/simple.gfa --reference reference > /dev/null 2> /dev/null
}

print_results() {
  echo "Benchmark results"
  printf "%-35s %-10s %-10s %-10s %-10s\n" "Task" "Time (ms)" "Total (MB)" "gen.db (MB)" "default.db (MB)"
  printf "%-35s %-10s %-10s %-10s %-10s\n" "---------------" "--------" "----------" "-----------" "--------------"
  printf "%s" "${RESULTS}"
}


rm -rf "${PROFILE_DIR}"
mkdir -p "${PROFILE_DIR}"

cargo build --release
cargo build --features profiling

profile_path() {
  local task="$1"
  local slug
  slug=$(printf '%s' "${task}" | tr '[:upper:]' '[:lower:]' | tr -cs '[:alnum:]' '_')
  echo "${PROFILE_DIR}/${slug}.txt"
}

echo "full import benchmark"
init_test
FULL_IMPORT=$(run_benchmark "Full import" ${PROFILE_BIN} profile import fasta ${BASE_DIR}/fixtures/chr22.fa.gz --reference reference)
read -r FULL_SIZE FULL_GEN_DB_SIZE FULL_DEFAULT_DB_SIZE <<< "$(get_size)"
record_result "Full import" "${FULL_IMPORT}" "${FULL_SIZE}" "${FULL_GEN_DB_SIZE}" "${FULL_DEFAULT_DB_SIZE}"

echo "shallow import benchmark"
init_test
SHALLOW_IMPORT=$(run_benchmark "Shallow import" ${PROFILE_BIN} profile import fasta ${BASE_DIR}/fixtures/chr22.fa.gz --reference reference --shallow)
read -r SHALLOW_SIZE SHALLOW_GEN_DB_SIZE SHALLOW_DEFAULT_DB_SIZE <<< "$(get_size)"
record_result "Shallow import" "${SHALLOW_IMPORT}" "${SHALLOW_SIZE}" "${SHALLOW_GEN_DB_SIZE}" "${SHALLOW_DEFAULT_DB_SIZE}"

echo "simple fasta import benchmark"
init_test
SIMPLE_FASTA_IMPORT=$(run_benchmark "Simple FASTA import" ${PROFILE_BIN} profile import fasta ${BASE_DIR}/fixtures/simple.fa --reference reference)
read -r SIMPLE_FASTA_SIZE SIMPLE_FASTA_GEN_DB_SIZE SIMPLE_FASTA_DEFAULT_DB_SIZE <<< "$(get_size)"
record_result "Simple FASTA import" "${SIMPLE_FASTA_IMPORT}" "${SIMPLE_FASTA_SIZE}" "${SIMPLE_FASTA_GEN_DB_SIZE}" "${SIMPLE_FASTA_DEFAULT_DB_SIZE}"

echo "multi fasta import benchmark"
init_test
MULTI_FASTA_IMPORT=$(run_benchmark "Multi FASTA import" ${PROFILE_BIN} profile import fasta ${BASE_DIR}/fixtures/fastas/multiple.fa --reference reference)
read -r MULTI_FASTA_SIZE MULTI_FASTA_GEN_DB_SIZE MULTI_FASTA_DEFAULT_DB_SIZE <<< "$(get_size)"
record_result "Multi FASTA import" "${MULTI_FASTA_IMPORT}" "${MULTI_FASTA_SIZE}" "${MULTI_FASTA_GEN_DB_SIZE}" "${MULTI_FASTA_DEFAULT_DB_SIZE}"

echo "gfa import benchmark"
init_test
GFA_IMPORT=$(run_benchmark "GFA import" ${PROFILE_BIN} profile import gfa ${BASE_DIR}/fixtures/chr22_het.gfa --reference reference)
read -r GFA_IMPORT_SIZE GFA_IMPORT_GEN_DB_SIZE GFA_IMPORT_DEFAULT_DB_SIZE <<< "$(get_size)"
record_result "GFA import" "${GFA_IMPORT}" "${GFA_IMPORT_SIZE}" "${GFA_IMPORT_GEN_DB_SIZE}" "${GFA_IMPORT_DEFAULT_DB_SIZE}"

echo "simple gfa import benchmark"
init_test
SIMPLE_GFA_IMPORT=$(run_benchmark "Simple GFA import" ${PROFILE_BIN} profile import gfa ${BASE_DIR}/fixtures/simple.gfa --reference reference)
read -r SIMPLE_GFA_IMPORT_SIZE SIMPLE_GFA_IMPORT_GEN_DB_SIZE SIMPLE_GFA_IMPORT_DEFAULT_DB_SIZE <<< "$(get_size)"
record_result "Simple GFA import" "${SIMPLE_GFA_IMPORT}" "${SIMPLE_GFA_IMPORT_SIZE}" "${SIMPLE_GFA_IMPORT_GEN_DB_SIZE}" "${SIMPLE_GFA_IMPORT_DEFAULT_DB_SIZE}"

echo "gff translation benchmark"
setup_simple_fasta_repo
GFF_TRANSLATION=$(run_benchmark "GFF translation" ${PROFILE_BIN} profile translate --gff ${BASE_DIR}/fixtures/simple.gff --sample reference)
read -r GFF_TRANSLATION_SIZE GFF_TRANSLATION_GEN_DB_SIZE GFF_TRANSLATION_DEFAULT_DB_SIZE <<< "$(get_size)"
record_result "GFF translation" "${GFF_TRANSLATION}" "${GFF_TRANSLATION_SIZE}" "${GFF_TRANSLATION_GEN_DB_SIZE}" "${GFF_TRANSLATION_DEFAULT_DB_SIZE}"

echo "library update benchmark"
setup_library_update_repo
LIBRARY_UPDATE=$(run_benchmark "Library update" ${PROFILE_BIN} profile update library --new-sample library-sample --region-name m123:7-20 --library ${BASE_DIR}/fixtures/combinatorial_design.csv --parts ${BASE_DIR}/fixtures/parts.fa)
read -r LIBRARY_UPDATE_SIZE LIBRARY_UPDATE_GEN_DB_SIZE LIBRARY_UPDATE_DEFAULT_DB_SIZE <<< "$(get_size)"
record_result "Library update" "${LIBRARY_UPDATE}" "${LIBRARY_UPDATE_SIZE}" "${LIBRARY_UPDATE_GEN_DB_SIZE}" "${LIBRARY_UPDATE_DEFAULT_DB_SIZE}"

echo "gfa update benchmark"
setup_gfa_update_repo
GFA_UPDATE=$(run_benchmark "GFA update" ${PROFILE_BIN} profile update gfa ${BASE_DIR}/fixtures/path-diff.gfa --sample reference --new-sample gfa-update-sample)
read -r GFA_UPDATE_SIZE GFA_UPDATE_GEN_DB_SIZE GFA_UPDATE_DEFAULT_DB_SIZE <<< "$(get_size)"
record_result "GFA update" "${GFA_UPDATE}" "${GFA_UPDATE_SIZE}" "${GFA_UPDATE_GEN_DB_SIZE}" "${GFA_UPDATE_DEFAULT_DB_SIZE}"

echo "Update with HG00096 benchmark"
init_test
${GEN_BIN} import fasta ${BASE_DIR}/fixtures/chr22.fa.gz --reference reference --shallow 2> /dev/null > /dev/null
HG96_IMPORT=$(run_benchmark "HG00096 Update" ${PROFILE_BIN} profile update vcf ${BASE_DIR}/fixtures/HG00096.vcf.gz --parent-samples reference)
read -r HG96_SIZE HG96_GEN_DB_SIZE HG96_DEFAULT_DB_SIZE <<< "$(get_size)"
record_result "HG00096 Update" "${HG96_IMPORT}" "${HG96_SIZE}" "${HG96_GEN_DB_SIZE}" "${HG96_DEFAULT_DB_SIZE}"

echo "Update with HG00097 benchmark"
init_test
${GEN_BIN} import fasta ${BASE_DIR}/fixtures/chr22.fa.gz --reference reference --shallow 2> /dev/null > /dev/null
HG97_IMPORT=$(run_benchmark "HG00097 Update" ${PROFILE_BIN} profile update vcf ${BASE_DIR}/fixtures/HG00097.vcf.gz --parent-samples reference)
read -r HG97_SIZE HG97_GEN_DB_SIZE HG97_DEFAULT_DB_SIZE <<< "$(get_size)"
record_result "HG00097 Update" "${HG97_IMPORT}" "${HG97_SIZE}" "${HG97_GEN_DB_SIZE}" "${HG97_DEFAULT_DB_SIZE}"

echo "Update with  HG00096 + HG00097 benchmark"
init_test
${GEN_BIN} import fasta ${BASE_DIR}/fixtures/chr22.fa.gz --reference reference --shallow 2> /dev/null > /dev/null
HG96_IMPORT=$(run_benchmark "HG00096 Update" --no-profile-record ${PROFILE_BIN} profile update vcf ${BASE_DIR}/fixtures/HG00096.vcf.gz --parent-samples reference)
HG97_IMPORT=$(run_benchmark "HG00097 Update" --no-profile-record ${PROFILE_BIN} profile update vcf ${BASE_DIR}/fixtures/HG00097.vcf.gz --parent-samples reference)
SUM=$(echo "${HG96_IMPORT} + ${HG97_IMPORT}" | bc)
read -r BOTH_SIZE BOTH_GEN_DB_SIZE BOTH_DEFAULT_DB_SIZE <<< "$(get_size)"
record_result "HG00096 + HG00097 Update" "${SUM}" "${BOTH_SIZE}" "${BOTH_GEN_DB_SIZE}" "${BOTH_DEFAULT_DB_SIZE}"

echo "switch branch benchmark"
setup_hg_branch_switch_repo
SWITCH_BRANCH=$(run_benchmark "Switch HG branch" ${PROFILE_BIN} profile checkout branch-b)
read -r SWITCH_BRANCH_SIZE SWITCH_BRANCH_GEN_DB_SIZE SWITCH_BRANCH_DEFAULT_DB_SIZE <<< "$(get_size)"
record_result "Switch HG branch" "${SWITCH_BRANCH}" "${SWITCH_BRANCH_SIZE}" "${SWITCH_BRANCH_GEN_DB_SIZE}" "${SWITCH_BRANCH_DEFAULT_DB_SIZE}"

echo "merge benchmark"
setup_hg_branch_apply_repo
MERGE_BRANCH=$(run_benchmark "Merge HG branch" ${PROFILE_BIN} profile merge branch-a)
read -r MERGE_BRANCH_SIZE MERGE_BRANCH_GEN_DB_SIZE MERGE_BRANCH_DEFAULT_DB_SIZE <<< "$(get_size)"
record_result "Merge HG branch" "${MERGE_BRANCH}" "${MERGE_BRANCH_SIZE}" "${MERGE_BRANCH_GEN_DB_SIZE}" "${MERGE_BRANCH_DEFAULT_DB_SIZE}"

echo "apply benchmark"
setup_hg_branch_apply_repo
APPLY_OPERATION=$(run_benchmark "Apply HG operation" ${PROFILE_BIN} profile apply ${BRANCH_A_HASH})
read -r APPLY_OPERATION_SIZE APPLY_OPERATION_GEN_DB_SIZE APPLY_OPERATION_DEFAULT_DB_SIZE <<< "$(get_size)"
record_result "Apply HG operation" "${APPLY_OPERATION}" "${APPLY_OPERATION_SIZE}" "${APPLY_OPERATION_GEN_DB_SIZE}" "${APPLY_OPERATION_DEFAULT_DB_SIZE}"

echo "reset benchmark"
setup_hg_reset_repo
RESET_OPERATION=$(run_benchmark "Reset HG operation" ${PROFILE_BIN} profile reset ${RESET_HASH})
read -r RESET_OPERATION_SIZE RESET_OPERATION_GEN_DB_SIZE RESET_OPERATION_DEFAULT_DB_SIZE <<< "$(get_size)"
record_result "Reset HG operation" "${RESET_OPERATION}" "${RESET_OPERATION_SIZE}" "${RESET_OPERATION_GEN_DB_SIZE}" "${RESET_OPERATION_DEFAULT_DB_SIZE}"

print_results
