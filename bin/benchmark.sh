#!/usr/bin/env bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_DIR="$(dirname "${SCRIPT_DIR}")"
GEN_DIR="${BASE_DIR}/.gen"
GEN_BIN="${BASE_DIR}/target/release/gen"
PROFILE_BIN="${BASE_DIR}/target/debug/gen"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
GEN_DB="${GEN_DIR}/gen.db"
RESULTS=""
PROFILE_DIR="${BASE_DIR}/.benchmark-profiles"
PROFILE_RESULTS=""

init_test() {
  cd ${BASE_DIR}
  rm -rf ${GEN_DIR}
  ${GEN_BIN} init 2> /dev/null > /dev/null
}

get_size () {
  local filesize_mb=$(${PYTHON_BIN} -c "import os;size=os.path.getsize('${GEN_DB}') / (1024 * 1024);print(f'{size:.4f}')")
  echo $filesize_mb
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
  local size="$3"
  RESULTS+=$(printf "%-35s %-10s %-10s\n" "${task}" "${duration}" "${size}")
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
  printf "%-35s %-10s %-10s\n" "Task" "Time (ms)" "Storage (mb)"
  printf "%-35s %-10s %-10s\n" "---------------" "--------" "----------"
  printf "%s" "${RESULTS}"
  echo "Profile reports"
  printf "%-35s %s\n" "Task" "Report"
  printf "%-35s %s\n" "---------------" "------"
  printf "%s" "${PROFILE_RESULTS}"
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
FULL_IMPORT=$(time_taken ${GEN_BIN} import fasta ${BASE_DIR}/fixtures/chr22.fa.gz --reference reference)
FULL_SIZE=$(get_size)
record_result "Full import" "${FULL_IMPORT}" "${FULL_SIZE}"
init_test
profile_taken "$(profile_path "Full import")" ${PROFILE_BIN} profile import fasta ${BASE_DIR}/fixtures/chr22.fa.gz --reference reference
record_profile_result "Full import" "$(profile_path "Full import")"

echo "shallow import benchmark"
init_test
SHALLOW_IMPORT=$(time_taken ${GEN_BIN} import fasta ${BASE_DIR}/fixtures/chr22.fa.gz --reference reference --shallow)
SHALLOW_SIZE=$(get_size)
record_result "Shallow import" "${SHALLOW_IMPORT}" "${SHALLOW_SIZE}"
init_test
profile_taken "$(profile_path "Shallow import")" ${PROFILE_BIN} profile import fasta ${BASE_DIR}/fixtures/chr22.fa.gz --reference reference --shallow
record_profile_result "Shallow import" "$(profile_path "Shallow import")"

echo "simple fasta import benchmark"
init_test
SIMPLE_FASTA_IMPORT=$(time_taken ${GEN_BIN} import fasta ${BASE_DIR}/fixtures/simple.fa --reference reference)
SIMPLE_FASTA_SIZE=$(get_size)
record_result "Simple FASTA import" "${SIMPLE_FASTA_IMPORT}" "${SIMPLE_FASTA_SIZE}"
setup_simple_fasta_repo
profile_taken "$(profile_path "Simple FASTA import")" ${PROFILE_BIN} profile import fasta ${BASE_DIR}/fixtures/simple.fa --reference reference
record_profile_result "Simple FASTA import" "$(profile_path "Simple FASTA import")"

echo "multi fasta import benchmark"
init_test
MULTI_FASTA_IMPORT=$(time_taken ${GEN_BIN} import fasta ${BASE_DIR}/fixtures/fastas/multiple.fa --reference reference)
MULTI_FASTA_SIZE=$(get_size)
record_result "Multi FASTA import" "${MULTI_FASTA_IMPORT}" "${MULTI_FASTA_SIZE}"
init_test
profile_taken "$(profile_path "Multi FASTA import")" ${PROFILE_BIN} profile import fasta ${BASE_DIR}/fixtures/fastas/multiple.fa --reference reference
record_profile_result "Multi FASTA import" "$(profile_path "Multi FASTA import")"

echo "gfa import benchmark"
init_test
GFA_IMPORT=$(time_taken ${GEN_BIN} import gfa ${BASE_DIR}/fixtures/chr22_het.gfa --reference reference)
GFA_IMPORT_SIZE=$(get_size)
record_result "GFA import" "${GFA_IMPORT}" "${GFA_IMPORT_SIZE}"
init_test
profile_taken "$(profile_path "GFA import")" ${PROFILE_BIN} profile import gfa ${BASE_DIR}/fixtures/chr22_het.gfa --reference reference
record_profile_result "GFA import" "$(profile_path "GFA import")"

echo "simple gfa import benchmark"
init_test
SIMPLE_GFA_IMPORT=$(time_taken ${GEN_BIN} import gfa ${BASE_DIR}/fixtures/simple.gfa --reference reference)
SIMPLE_GFA_IMPORT_SIZE=$(get_size)
record_result "Simple GFA import" "${SIMPLE_GFA_IMPORT}" "${SIMPLE_GFA_IMPORT_SIZE}"
init_test
profile_taken "$(profile_path "Simple GFA import")" ${PROFILE_BIN} profile import gfa ${BASE_DIR}/fixtures/simple.gfa --reference reference
record_profile_result "Simple GFA import" "$(profile_path "Simple GFA import")"

echo "gff translation benchmark"
setup_simple_fasta_repo
GFF_TRANSLATION=$(time_taken ${GEN_BIN} translate --gff ${BASE_DIR}/fixtures/simple.gff --sample reference)
GFF_TRANSLATION_SIZE=$(get_size)
record_result "GFF translation" "${GFF_TRANSLATION}" "${GFF_TRANSLATION_SIZE}"
setup_simple_fasta_repo
profile_taken "$(profile_path "GFF translation")" ${PROFILE_BIN} profile translate --gff ${BASE_DIR}/fixtures/simple.gff --sample reference
record_profile_result "GFF translation" "$(profile_path "GFF translation")"

echo "library update benchmark"
setup_library_update_repo
LIBRARY_UPDATE=$(time_taken ${GEN_BIN} update library --new-sample library-sample --region-name m123:7-20 --library ${BASE_DIR}/fixtures/combinatorial_design.csv --parts ${BASE_DIR}/fixtures/parts.fa)
LIBRARY_UPDATE_SIZE=$(get_size)
record_result "Library update" "${LIBRARY_UPDATE}" "${LIBRARY_UPDATE_SIZE}"
setup_library_update_repo
profile_taken "$(profile_path "Library update")" ${PROFILE_BIN} profile update library --new-sample library-sample --region-name m123:7-20 --library ${BASE_DIR}/fixtures/combinatorial_design.csv --parts ${BASE_DIR}/fixtures/parts.fa
record_profile_result "Library update" "$(profile_path "Library update")"

echo "gfa update benchmark"
setup_gfa_update_repo
GFA_UPDATE=$(time_taken ${GEN_BIN} update gfa ${BASE_DIR}/fixtures/path-diff.gfa --sample reference --new-sample gfa-update-sample)
GFA_UPDATE_SIZE=$(get_size)
record_result "GFA update" "${GFA_UPDATE}" "${GFA_UPDATE_SIZE}"
setup_gfa_update_repo
profile_taken "$(profile_path "GFA update")" ${PROFILE_BIN} profile update gfa ${BASE_DIR}/fixtures/path-diff.gfa --sample reference --new-sample gfa-update-sample
record_profile_result "GFA update" "$(profile_path "GFA update")"

echo "Update with HG00096 benchmark"
init_test
${GEN_BIN} import fasta ${BASE_DIR}/fixtures/chr22.fa.gz --reference reference --shallow 2> /dev/null > /dev/null
HG96_IMPORT=$(time_taken ${GEN_BIN} update vcf ${BASE_DIR}/fixtures/HG00096.vcf.gz --parent-samples reference)
HG96_SIZE=$(get_size)
record_result "HG00096 Update" "${HG96_IMPORT}" "${HG96_SIZE}"
init_test
${GEN_BIN} import fasta ${BASE_DIR}/fixtures/chr22.fa.gz --reference reference --shallow 2> /dev/null > /dev/null
profile_taken "$(profile_path "HG00096 Update")" ${PROFILE_BIN} profile update vcf ${BASE_DIR}/fixtures/HG00096.vcf.gz --parent-samples reference
record_profile_result "HG00096 Update" "$(profile_path "HG00096 Update")"

echo "Update with HG00097 benchmark"
init_test
${GEN_BIN} import fasta ${BASE_DIR}/fixtures/chr22.fa.gz --reference reference --shallow 2> /dev/null > /dev/null
HG97_IMPORT=$(time_taken ${GEN_BIN} update vcf ${BASE_DIR}/fixtures/HG00097.vcf.gz --parent-samples reference)
HG97_SIZE=$(get_size)
record_result "HG00097 Update" "${HG97_IMPORT}" "${HG97_SIZE}"
init_test
${GEN_BIN} import fasta ${BASE_DIR}/fixtures/chr22.fa.gz --reference reference --shallow 2> /dev/null > /dev/null
profile_taken "$(profile_path "HG00097 Update")" ${PROFILE_BIN} profile update vcf ${BASE_DIR}/fixtures/HG00097.vcf.gz --parent-samples reference
record_profile_result "HG00097 Update" "$(profile_path "HG00097 Update")"

echo "Update with  HG00096 + HG00097 benchmark"
init_test
${GEN_BIN} import fasta ${BASE_DIR}/fixtures/chr22.fa.gz --reference reference --shallow 2> /dev/null > /dev/null
HG96_IMPORT=$(time_taken ${GEN_BIN} update vcf ${BASE_DIR}/fixtures/HG00096.vcf.gz --parent-samples reference)
HG97_IMPORT=$(time_taken ${GEN_BIN} update vcf ${BASE_DIR}/fixtures/HG00097.vcf.gz --parent-samples reference)
SUM=$(echo "${HG96_IMPORT} + ${HG97_IMPORT}" | bc)
BOTH_SIZE=$(get_size)
record_result "HG00096 + HG00097 Update" "${SUM}" "${BOTH_SIZE}"
record_profile_result "HG00096 + HG00097 Update" "$(profile_path "HG00096 Update"), $(profile_path "HG00097 Update")"

echo "checkout benchmark"
setup_hg_branch_switch_repo
CHECKOUT_BRANCH=$(time_taken ${GEN_BIN} checkout branch-a)
CHECKOUT_BRANCH_SIZE=$(get_size)
record_result "Checkout HG branch" "${CHECKOUT_BRANCH}" "${CHECKOUT_BRANCH_SIZE}"
setup_hg_branch_switch_repo
profile_taken "$(profile_path "Checkout HG branch")" ${PROFILE_BIN} profile checkout branch-a
record_profile_result "Checkout HG branch" "$(profile_path "Checkout HG branch")"

echo "switch branch benchmark"
SWITCH_BRANCH=$(time_taken ${GEN_BIN} checkout branch-b)
SWITCH_BRANCH_SIZE=$(get_size)
record_result "Switch HG branch" "${SWITCH_BRANCH}" "${SWITCH_BRANCH_SIZE}"
setup_hg_branch_apply_repo
${GEN_BIN} checkout -b branch-b > /dev/null 2> /dev/null
${GEN_BIN} update vcf ${BASE_DIR}/fixtures/HG00097.vcf.gz --parent-samples reference > /dev/null 2> /dev/null
profile_taken "$(profile_path "Switch HG branch")" ${PROFILE_BIN} profile checkout branch-b
record_profile_result "Switch HG branch" "$(profile_path "Switch HG branch")"

echo "merge benchmark"
setup_hg_branch_apply_repo
MERGE_BRANCH=$(time_taken ${GEN_BIN} merge branch-a)
MERGE_BRANCH_SIZE=$(get_size)
record_result "Merge HG branch" "${MERGE_BRANCH}" "${MERGE_BRANCH_SIZE}"
setup_hg_branch_apply_repo
profile_taken "$(profile_path "Merge HG branch")" ${PROFILE_BIN} profile merge branch-a
record_profile_result "Merge HG branch" "$(profile_path "Merge HG branch")"

echo "apply benchmark"
setup_hg_branch_apply_repo
APPLY_OPERATION=$(time_taken ${GEN_BIN} apply ${BRANCH_A_HASH})
APPLY_OPERATION_SIZE=$(get_size)
record_result "Apply HG operation" "${APPLY_OPERATION}" "${APPLY_OPERATION_SIZE}"
setup_hg_branch_apply_repo
profile_taken "$(profile_path "Apply HG operation")" ${PROFILE_BIN} profile apply ${BRANCH_A_HASH}
record_profile_result "Apply HG operation" "$(profile_path "Apply HG operation")"

echo "reset benchmark"
setup_hg_reset_repo
RESET_OPERATION=$(time_taken ${GEN_BIN} reset ${RESET_HASH})
RESET_OPERATION_SIZE=$(get_size)
record_result "Reset HG operation" "${RESET_OPERATION}" "${RESET_OPERATION_SIZE}"
setup_hg_reset_repo
profile_taken "$(profile_path "Reset HG operation")" ${PROFILE_BIN} profile reset ${RESET_HASH}
record_profile_result "Reset HG operation" "$(profile_path "Reset HG operation")"

print_results
