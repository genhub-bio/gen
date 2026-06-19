#!/usr/bin/env bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_DIR="$(dirname "${SCRIPT_DIR}")"
GEN_DIR="${BASE_DIR}/.gen"
GEN_BIN="${BASE_DIR}/target/release/gen"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
GEN_DB="${GEN_DIR}/gen.db"

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


cargo build --release
echo "full import benchmark"
init_test
FULL_IMPORT=$(time_taken ${GEN_BIN} import fasta ${BASE_DIR}/fixtures/chr22.fa.gz --reference reference)
FULL_SIZE=$(get_size)
echo "shallow import benchmark"
init_test
SHALLOW_IMPORT=$(time_taken ${GEN_BIN} import fasta ${BASE_DIR}/fixtures/chr22.fa.gz --reference reference --shallow)
SHALLOW_SIZE=$(get_size)
echo "Update with HG00096 benchmark"
init_test
${GEN_BIN} import fasta ${BASE_DIR}/fixtures/chr22.fa.gz --reference reference --shallow 2> /dev/null > /dev/null
HG96_IMPORT=$(time_taken ${GEN_BIN} update vcf ${BASE_DIR}/fixtures/HG00096.vcf.gz --parent-samples reference)
HG96_SIZE=$(get_size)
echo "Update with HG00097 benchmark"
init_test
${GEN_BIN} import fasta ${BASE_DIR}/fixtures/chr22.fa.gz --reference reference --shallow 2> /dev/null > /dev/null
HG97_IMPORT=$(time_taken ${GEN_BIN} update vcf ${BASE_DIR}/fixtures/HG00097.vcf.gz --parent-samples reference)
HG97_SIZE=$(get_size)
echo "Update with  HG00096 + HG00097 benchmark"
init_test
${GEN_BIN} import fasta ${BASE_DIR}/fixtures/chr22.fa.gz --reference reference --shallow 2> /dev/null > /dev/null
HG96_IMPORT=$(time_taken ${GEN_BIN} update vcf ${BASE_DIR}/fixtures/HG00096.vcf.gz --parent-samples reference)
HG97_IMPORT=$(time_taken ${GEN_BIN} update vcf ${BASE_DIR}/fixtures/HG00097.vcf.gz --parent-samples reference)
SUM=$(echo "${HG96_IMPORT} + ${HG97_IMPORT}" | bc)
BOTH_SIZE=$(get_size)

echo "Benchmark results"
printf "%-35s %-10s %-10s\n" "Task" "Time (ms)" "Storage (mb)"
printf "%-35s %-10s %-10s\n" "---------------" "--------" "----------"
printf "%-35s %-10s %-10s\n" "Full import" ${FULL_IMPORT} ${FULL_SIZE}
printf "%-35s %-10s %-10s\n" "Shallow import" ${SHALLOW_IMPORT} ${SHALLOW_SIZE}
printf "%-35s %-10s %-10s\n" "HG00096 Update" ${HG96_IMPORT} ${HG96_SIZE}
printf "%-35s %-10s %-10s\n" "HG00097 Update" ${HG97_IMPORT} ${HG97_SIZE}
printf "%-35s %-10s %-10s\n" "HG00096 + HG00097 Update" ${SUM} ${BOTH_SIZE}
