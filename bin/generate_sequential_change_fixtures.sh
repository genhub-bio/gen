#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "${SCRIPT_DIR}")"
FASTA_PATH="${BASE_DIR}/fixtures/simple.fa"
OUTPUT_DIR="${BASE_DIR}/fixtures/benchmarks/100_sequential_changes"
CONTIG_NAME="m123"

read_fasta_sequence() {
  awk 'BEGIN { sequence = "" } /^>/ { next } { sequence = sequence $0 } END { print sequence }' "${FASTA_PATH}"
}

main() {
  local sequence
  sequence="$(read_fasta_sequence)"
  local sequence_length=${#sequence}

  mkdir -p "${OUTPUT_DIR}"
  find "${OUTPUT_DIR}" -maxdepth 1 -type f -name 'change_*.vcf' -delete

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
      local output_path="${OUTPUT_DIR}/${sample_name}.vcf"

      printf '##fileformat=VCFv4.1\n' > "${output_path}"
      printf '##reference=simple.fa\n' >> "${output_path}"
      printf '##contig=<ID=%s,length=%s>\n' "${CONTIG_NAME}" "${sequence_length}" >> "${output_path}"
      printf '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n' >> "${output_path}"
      printf '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t%s\n' "${sample_name}" >> "${output_path}"
      printf '%s\t%s\t.\t%s\t%s\t.\tPASS\t.\tGT\t1/1\n' \
        "${CONTIG_NAME}" \
        "${position}" \
        "${reference_base}" \
        "${alternate_base}" >> "${output_path}"
    done
  done

  printf 'Generated %s fixtures in %s\n' "$((sequence_length * 3))" "${OUTPUT_DIR}"
}

main "$@"
