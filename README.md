# Gen

Gen brings version control to genetic sequences. You can clone repositories, create branches, make edits, and push changes to a shared remote using the same workflow developers know from Git. It works across FASTA files, VCFs, GenBank records, and other common bioinformatics formats. Under the hood, Gen stores data as a sequence graph, allowing a single repository to represent a reference genome, known variants, and engineered modifications without repeatedly storing the same sequence.

[![PyPI](https://img.shields.io/pypi/v/gen.svg)](https://pypi.org/project/gen/) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

<!-- TODO: replace with screenshot of Jupyter widget or TUI showing a real genome graph -->
![Gen graph viewer](docs/figures/placeholder.png)

## Install

**CLI binary** — prebuilt binaries for macOS and Linux are on the [releases page](https://github.com/genhub-bio/gen/releases): [macOS (.pkg)](https://github.com/genhub-bio/gen/releases/download/nightly/gen.macos.pkg), [Linux x86_64 (.zip)](https://github.com/genhub-bio/gen/releases/download/nightly/gen.linux-x86_64.zip), [Linux arm64 (.zip)](https://github.com/genhub-bio/gen/releases/download/nightly/gen.linux-arm64.zip). Gen is built for Unix-like systems; on Windows, install [WSL](https://learn.microsoft.com/en-us/windows/wsl/) to get a Linux environment, then use the Linux binary above from inside it.

**Python**

```sh
pip install gen
```

**R** — see the [installation guide](https://www.genhub.bio/docs/installation) for platform-specific instructions. On macOS (Apple silicon):

```r
install.packages("remotes")
remotes::install_url(
  "https://github.com/genhub-bio/gen/releases/download/v0.1.31/genr_0.1.31-macos-arm64.tgz"
)
```

Windows builds are published as `genr-windows-<version>.zip` on the same [releases page](https://github.com/genhub-bio/gen/releases).

## Quick start

```sh
# Set up a new repository and import a reference genome
gen init
gen import fasta reference.fa --reference hg38

# Branch before making changes
gen branch --create experiment/na12878
gen checkout --branch experiment/na12878

# Apply variants from a VCF — Gen adds new edges to the graph without touching existing nodes
gen update vcf variants.vcf --reference hg38 --sample NA12878

# Review the operation log
gen operations

# Browse the graph in the terminal
gen view

# Push to a remote GenHub repository
gen push
```

Python and R libraries expose the same operations:

```python
import gen

repo = gen.Repository(".")
repo.import_reference_fasta("reference.fa", "hg38")
repo.update_with_vcf("variants.vcf", reference="hg38", sample="NA12878")
sgs = repo.get_sequence_graphs()
sgs[0].plot()
```

## Features

- Every import, update, and merge is a recorded operation. You can roll back to any prior state with `gen checkout`, compare two branches with `gen view-diff`, or share a set of changes as a patch file.
- Gen can import from FASTA, GenBank, GFA, VCF, GAF, and combinatorial part libraries, and export to FASTA, GenBank, or GFA for downstream tools like `vg` or Bandage.
- Sequence search works across all paths in a graph, including IUPAC ambiguity codes, via `gen search` or `repo.search()` in Python and R.
- GFF3 annotation tracks are visible in both the terminal viewer and the interactive widget.
- For combinatorial library design, you define a parts list and a slot table; Gen builds the graph of all combinations without enumerating the sequences explicitly.
- `gen clone`, `gen push`, and `gen pull` work against GenHub. Any public repository is clonable with a single URL.
- The R package includes direct import from Bioconductor `DNAStringSet` and `GRanges` objects.

## Screenshots

<!-- TODO: replace with real screenshots before publishing -->

**Jupyter widget**

![Jupyter widget — interactive canvas with zoom and pan](docs/figures/placeholder_jupyter.png)

**RStudio viewer**

![RStudio viewer — GenPlot htmlwidget](docs/figures/placeholder_rstudio.png)

**Terminal viewer**

![Terminal viewer — ratatui graph navigator](docs/figures/placeholder_tui.png)

## Example workflows

- [Variation-aware alignment against hg38](examples/human_variation_aware_alignment/Analysis.ipynb) — import a reference, encode GIAB variants into the graph, export to GFA for `vg map`.
- [Genome editing in yeast](examples/yeast_editing/edit-yeast-and-export-fasta.ipynb) — import S288C, apply a cassette edit at a specific locus, export the modified sequence as FASTA.
- [Combinatorial plasmid design](examples/combinatorial_plasmid_design/combinatorial_design.md) — insert a promoter/RBS library into pUC19, then identify colony genotypes from long-read sequencing.
- [Modeling a yeast cross](examples/yeast_crosses/Analysis.md) — compare two beer yeast strains starting from VCF variant calls or whole-genome assemblies.

## Data model

Gen represents sequences as a sequence graph. Nodes hold sequence fragments, edges connect them, and any linear sequence is reconstructed by walking a defined path. New variants extend the graph without splitting existing nodes, so node IDs remain stable across updates.

![Figure 1](docs/figures/figure_1.svg)

**_Figure 1_**: _Sequence graph representation of a variant where two nucleotides AT are replaced by TG; the modified sequence (shown in bold) is stored as a path over a list of edges that address specific coordinates._

This differs from the segment graph model used by tools like vg and Bandage, where the reference sequence is split into pieces to accommodate each variant. Gen converts between the two formats on GFA export.

![Figure 2](docs/figures/figure_2.svg)

**_Figure 2_**: _Segment graph model corresponding to the variant in Figure 1. The original sequence is split into 3 parts; the modified path is defined by a list of nodes that refer to these segments._

For a longer explanation see [docs/coordinates.md](docs/coordinates.md).

## Documentation

Full command reference, Python and R API docs, and tutorials are at [genhub.bio/docs](https://www.genhub.bio/docs).

## Building from source

Requires a Rust toolchain ([rustup](https://rustup.rs/)).

```sh
git clone https://github.com/genhub-bio/gen.git
cd gen
cargo build --release
# binary at ./target/release/gen
```

For Python and R bindings, see [gen-python/README.md](gen-python/README.md) and [gen-r/README.md](gen-r/README.md).
