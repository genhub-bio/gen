#!/usr/bin/env bash
# Build the repository that demo.tape records.
#
# Produces /tmp/gen-demo, checked out on main: an annotated pUC19 whose MCS
# region has been replaced by a combinatorial library on the "design" sample,
# plus a QC annotation. The tape itself performs the interesting version-
# control beats on top of this starting point:
#   gen checkout -b sequencing_results
#   gen update vcf                    (an observed sequencing result)
#   gen view
#   gen checkout main
#   gen merge sequencing_results
#
# A fake VCF is generated here rather than reused from gen-python/examples so
# this demo doesn't reach into another subproject's fixtures. Its single SNP
# sits at pUC19 position 370, just upstream of the MCS (395-452): that's on
# the "design" sample's own linear coordinate frame (shared by every
# combinatorial branch and the untouched original alike, since it's before
# the library splice point), so a plain position-based VCF call lands
# cleanly and shows up right next to the library branch when scrolled to.
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

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK=/tmp/gen-demo
rm -rf "$WORK/.gen"
rm -rf "$WORK"
mkdir -p "$WORK"
cd "$WORK"

cp "$REPO/gen-python/examples/puc19.gbk" .
cp "$SCRIPT_DIR/parts.fa" .
cp "$SCRIPT_DIR/design.csv" .

# Record the binary path so demo.tape can alias gen without hardcoding.
printf '%s\n' "$GEN" > gen_path

# A fake sequencing result: one SNP just upstream of the MCS library region,
# so `gen update vcf` applies cleanly on top of the design sample.
cat > sequencing_run.vcf <<'EOF'
##fileformat=VCFv4.2
##source=gen-demo-fixture
##reference=pUC19
##contig=<ID=pUC19,length=2686>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	sequenced
pUC19	370	.	C	T	48	PASS	.	GT	1/1
EOF

"$GEN" init
"$GEN" import genbank puc19.gbk --sample reference
"$GEN" update library --sample reference --new-sample design \
    --region-name MCS --library design.csv --parts parts.fa

# main advances with a QC annotation, so it has diverged by the time the
# tape branches sequencing_results off of it.
"$GEN" add-annotation --name lacZ_promoter_verified --sample design pUC19:100-300

echo
echo "Ready in $WORK (on main, before the branch/vcf/merge beats)."
echo "Record with:  cd $(dirname "$0") && vhs demo.tape"
