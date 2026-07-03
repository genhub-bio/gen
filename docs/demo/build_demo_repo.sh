#!/usr/bin/env bash
# Build the repository that demo.tape records.
#
# Produces /tmp/gen-demo in a PRE-MERGE state: an annotated pUC19 whose MCS
# region has been replaced by a combinatorial library on the "design" sample.
# main advances with a QC annotation, and a sequencing_results branch (cut from
# that point) records an observed sequence, so the histories diverge.
#
# The repo is left parked on a "baseline" branch pinned at the library
# operation, which is an ancestor of main's tip. That makes the tape's first
# command forward-only:
#   gen branch --checkout main  ->  applies main's QC annotation (no revert)
#   gen merge sequencing_results ->  applies the observed sequence
#   gen view
# Because nothing is reverted, the two operation hashes shown never collide.
#
# Usage:  ./build_demo_repo.sh [REPO_ROOT]
#   REPO_ROOT defaults to the current git checkout root.
set -euo pipefail

REPO="${1:-$(git rev-parse --show-toplevel)}"
GEN="$REPO/target/release/gen"
if [ ! -x "$GEN" ]; then
  echo "gen binary not found at $GEN"
  echo "Build it first:  (cd \"$REPO\" && cargo build --release --bin gen)"
  exit 1
fi

WORK=/tmp/gen-demo
rm -rf "$WORK"
mkdir -p "$WORK"
cd "$WORK"

cp "$REPO/gen-python/examples/puc19.gbk" .
cp "$REPO/examples/combinatorial_plasmid_design/parts.fa" .
cp "$REPO/examples/combinatorial_plasmid_design/design.csv" .

# Record the binary path so demo.tape can alias gen without hardcoding.
printf '%s\n' "$GEN" > gen_path

"$GEN" init
"$GEN" import genbank puc19.gbk --sample reference
"$GEN" update library --sample reference --new-sample design \
    --region-name MCS --library design.csv --parts parts.fa

# Pin a baseline branch at the library operation, before main advances.
"$GEN" branch --create baseline

# main advances with a QC annotation (this is what the tape's checkout applies).
"$GEN" add-annotation --name lacZ_promoter_verified --sample design pUC19:100-300

# sequencing_results branches from main and records the observed sequence.
"$GEN" branch --create sequencing_results
"$GEN" branch --checkout sequencing_results
"$GEN" update sequence CTAGCTAGCTAGCTAGCT --sample design --new-sample design \
    --region-name pUC19:465-483

# Park on baseline (an ancestor of main) so the tape's checkout is forward-only.
"$GEN" branch --checkout baseline

echo
echo "Ready in $WORK (parked on baseline, pre-merge)."
echo "Record with:  cd $(dirname "$0") && vhs demo.tape"
