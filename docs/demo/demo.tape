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
#
# The heavy setup lives in build_demo_repo.sh, not here, so this tape records no
# blank lead-in while imports and library expansion run. The repo it produces is
# checked out on sequencing_results (main holds an independent QC annotation),
# so the visible checkout -> merge -> view below is a genuine branch merge.
Hide
Type "cd /tmp/gen-demo"
Enter
Type 'alias gen="$(cat gen_path)"'
Enter
Type "clear"
Enter
Sleep 300ms
Show

# --- Switch back to main ---
Sleep 800ms
Type "gen branch --checkout main"
Sleep 1.2s
Enter
Sleep 1.5s

# --- Merge the sequencing results: version control for sequences ---
Type "gen merge sequencing_results"
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
