Output demo.gif

Set Width 850
Set Height 400
Set FontSize 14
Set Theme "Catppuccin Mocha"
Set Shell "bash"
Set Padding 20

# Prerequisites:
#   1. Build the CLI:   cargo build --release --bin gen   (from the repo root)
#   2. Build the repo:  ./build_demo_repo.sh              (creates /tmp/gen-demo)
#   3. Record:          cd docs/demo && vhs demo.tape

Hide
Type "cd /tmp/gen-demo"
Enter
Type 'alias gen="$(cat gen_path)"'
Enter
Type "clear"
Enter
Sleep 300ms
Show

# --- Branch off to record a sequencing result ---
Sleep 800ms
Type "gen checkout -b sequencing_results"
Sleep 1.2s
Enter
Sleep 1.5s

# --- A VCF comes back from the sequencer: version control for sequences ---
Type "gen update vcf sequencing_run.vcf --parent-samples design"
Sleep 1.2s
Enter
Sleep 2s

# --- View the result: the combinatorial library, with the variant on top ---
Type "gen view pUC19 --sample sequenced"
Sleep 1.2s
Enter
Sleep 3s

Right
Sleep 500ms
Right
Sleep 500ms
Up
Sleep 500ms

Type "+"
Sleep 1.2s
Type "+"
Sleep 2.5s
Type "-"
Sleep 1.2s
Type "-"
Sleep 1.2s
Type "-"
Sleep 700ms

Type "q"
Sleep 500ms

# --- Switch back to main and bring the sequencing result in ---
# Checkout's own output is hidden from the recording: it reverts the vcf
# operation to leave sequencing_results, which prints the same hash the
# next command re-applies via merge (redundant to show twice).
Sleep 800ms
Type "gen checkout main"
Sleep 1.2s
Hide
Enter
Sleep 1.5s
Show

Type "gen merge sequencing_results"
Sleep 1.2s
Enter
Sleep 3s
