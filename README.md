# Gen
Gen is a version control system for genetic sequences. It efficiently stores genome-length sequences and sequence
variations, with native support for polyploid genomes and pooled genotypes. Each project is organized into a repository,
where collections of sequences and associated data are stored and tracked over time. Within a repository, branches can
be created to explore different modifications or variations without affecting the main project. These branches can later
be merged to integrate results from different experiments or collaborators.

The gen client can import standard sequence file formats from sources like NCBI and genetic design tools. The sequence
model takes the form of a graph structure as shown in the figure below. Each molecule is made up out of a network of
_nodes_ that represent sequence fragments, and _edges_ that define how sequence fragments are connected. Multiple
molecules are organized into _collections_ that could represent the different chromosomes in a reference genome,
proteins in a proteome, or pieces of DNA in a cloning reaction mixture. Molecules generally start out as a node that
holds the reference sequence, and new edges and nodes are added for every sequence variant that is designed or observed.
To reconstitute a linear sequence, the client walks from node to node along a defined _path_. This data model allows the
representation of a wide range of biological complexity -- from a single molecule to complex cultivars and cell lines,
including any naturally occurring variation in addition to intended engineering.

![Figure 1](docs/figures/figure_1.svg)

**_Figure 1_**: _Block graph model representation of a sequence variant where two nucleotides AT are replaced by TG;
the modified sequence (shown in bold) is stored as a path over a list of edges that address specific coordinates._

The block graph model is designed to be additive: new sequence variants only add to the graph, existing nodes are stable
and do not have to be split to accomodate the new topology. This is different from the more commonly used segment graph 
model shown in Figure 2, and gen automatically converts between formats as needed. 

![Figure 2](docs/figures/figure_2.svg)

**_Figure 2_**: _Segment graph model corresponding to the sequence variant in Figure 1. The original sequence was
split into 3 parts; the modified sequence path is defined by a list of nodes that refer to these segments. Nodes are
labeled by their parent ID and starting coordinate, new edges internal to the reference are shown as dashed arrows._ 

Individual chromosomes or contigs are stored as _block groups_ that refer to a specific grouping of edges from the main
graph. Block groups are adressed on three facets: name, sample, and collection. The name is an identifier like
"chromosome II", and the sample could refer to a real individual or a virtual outcome of an experiment. The meaning of a
collection is flexible, and for example could refer to the type of data (genomic, protein, ...) or experiment.

## Prebuilt binaries

For a download link, visit our [Installation page](https://www.genhub.bio/docs/installation).

Prebuilt binaries exist for Mac OS X and Linux on the [Release
page](https://github.com/genhub-bio/gen/releases). The latest version can be found
[here](https://github.com/genhub-bio/gen/releases/tag/nightly).  We do not currently support a Windows native
binary.

## Python library

Gen is available as a package on [pypi.org](https://pypi.org/), so you can install it with

```
pip install gen
```

## Installing from Source
Make sure you have a Rust compiler installed on your system. You can install the Rust toolset using the [rustup
installer](https://rustup.rs/).


1. Clone the [source](https://github.com/genhub-bio/gen) with `git`:

   ```sh
   git clone https://github.com/genhub-bio/gen.git
   cd rust
   ```

2. Compile the gen package and its dependencies:

    ```
    cargo build --release
    ```

3. You can find the gen executable in ./target/release/ or execute it via cargo:

    ```
    cargo run -- <arguments>
    ```

To cross-compile gen to run on a different architecture, you need to first add a target to the Rust toolchain and
install a linker. For macOS to Linux this can be done as follows:

    ```
    rustup target add x86_64-unknown-linux-gnu
    brew install SergioBenitez/osxct/x86_64-unknown-linux-gnu
    cargo build --release --target=x86_64-unknown-linux-gnu
    ```

The executable will be placed in ./target/x86_64-unknown-linux-gnu/release/

## Usage
Gen is a command line tool with multiple subcommands that each have their own flags and arguments. The currently
available commands are listed below and described in more detail [here](docs/commands.md) or built-in help pages (`gen
[COMMAND] --help`).

```
Usage: gen [OPTIONS] [COMMAND]

Commands:
  import      Import a new sequence collection
  update      Update a sequence collection with new data
  init        Initialize a gen repository
  branch      Manage and create branches
  checkout    Migrate a database to a given operation
  reset       
  operations  View operations carried out against a database
  apply       
  export      Export a sequence collection
  defaults    Set a default database and collection to use
  help        Print this message or the help of the given subcommand(s)

Options:
  -d, --db <DB>  The path to the database you wish to utilize
  -h, --help     Print help
  -V, --version  Print version
```
Typical workflows are illustrated in these example workflows:

- [Mapping reads to the human genome, taking into account all known variants](examples/human_variation_aware_alignment/Analysis.ipynb)
- [Design and analysis of a combinatorial plasmid library constructed via one-pot cloning](examples/combinatorial_plasmid_design/combinatorial_design.md)
- [Modeling a cross between two yeast strains starting from variant calls or genome assemblies](examples/yeast_crosses/Analysis.md)
