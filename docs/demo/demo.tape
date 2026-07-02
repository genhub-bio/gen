Output demo.gif

Set Width 1200
Set Height 420
Set FontSize 14
Set Theme "Catppuccin Mocha"
Set Shell "bash"
Set Padding 20

# Prerequisites:
#   1. Build the CLI from the repo root:  cargo build --release --bin gen
#   2. Record from this directory:        cd docs/demo && vhs demo.tape

# Everything up to the first visible command is hidden setup. It builds a
# repository whose "design" sample carries a combinatorial library dropped
# into the MCS of an annotated pUC19, then branches off a point mutation.
Hide
Type "REPO=$(cd ../.. && pwd)"
Enter
Type "cd /tmp && rm -rf gen-demo && mkdir gen-demo && cd gen-demo"
Enter
Type 'cp "$REPO"/gen-python/examples/puc19.gbk .'
Enter
Type 'cp "$REPO"/examples/combinatorial_plasmid_design/parts.fa .'
Enter
Type 'cp "$REPO"/examples/combinatorial_plasmid_design/design.csv .'
Enter
Type 'alias gen="$REPO/target/release/gen"'
Enter

Type "gen init"
Enter
Type "gen import genbank puc19.gbk --sample reference"
Enter
Type "gen update library --sample reference --new-sample design --region-name MCS --library design.csv --parts parts.fa"
Enter

Type "gen branch --create variant && gen branch --checkout variant"
Enter
Type "gen update sequence CTAGCTAGCTAGCTAGCT --sample design --new-sample design --region-name pUC19:465-483"
Enter
Type "clear"
Enter
Sleep 500ms
Show

# --- Switch back to main ---
Sleep 800ms
Type "gen branch --checkout main"
Sleep 1.2s
Enter
Sleep 1.5s

# --- Merge the mutation branch: version control for sequences ---
Type "gen merge variant"
Sleep 1.2s
Enter
Sleep 2s

# --- View the result: an annotated combinatorial library with a variant on top ---
Type "gen view pUC19 --sample design"
Sleep 1.2s
Enter
Sleep 3s

Right
Sleep 700ms
Right
Sleep 700ms
Right
Sleep 1s
Right
Sleep 700ms
Right
Sleep 1s

Type "+"
Sleep 3.5s

Type "q"
Sleep 500ms
