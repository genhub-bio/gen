Gen is a command line tool with multiple subcommands that each have their own flags and arguments. The currently
available commands are listed below, with links to more information.

# Init

Run `gen init` in a new directory to set up a gen repo.  This creates a database with repo information and a default collection.

# Import

[Import a file](import.md) to set up a reference graph in a repo.  Currently gen supports FASTA, Genbank, GFA, and combinatorial library formats.

# Update

# Export

# View

# Operations

Gen implements commands similar to git for managing operational data like changesets and branches.  [More information is here](operations.md).

# Command-line help

Descriptions of the commands are also available with the built-in help pages,
viewable with `gen [COMMAND] --help`.  Current output of `gen --help`:

```
A sequence graph and version control system.

Usage: gen [OPTIONS] [COMMAND]

Commands:
  init
  import                 Commands for importing
  update                 Commands for updating
  export                 Commands for exporting
  transform              Commands for transforming file types for input to Gen
  translate              Translate coordinates of standard bioinformatic file formats
  view                   Show a visual representation of a graph in the terminal
  patch-create           Export a set of operations to a patch file
  patch-apply            Apply changes from a patch file
  patch-view             View a patch in dot format
  branch                 Manage and create branches
  merge                  Merge branches
  checkout               Migrate a database to a given operation
  reset                  Reset a branch to a previous operation
  operations             View operations carried out against a database
  apply                  Apply an operation to a branch
  defaults               Configure default options
  set-remote             Set the remote URL for this repo
  push                   Push the local repo to the remote
  pull
  propagate-annotations  Convert annotation coordinates between two sequence graphs
  list-samples           List all samples in the current collection
  list-graphs            List all regions/contigs in the current collection and given sequence graph
  get-sequence           Extract a sequence from a graph
  diff                   Output a file representing the "diff" between two samples
  derive-subgraph        Replace a sequence graph with a subgraph in the range of the specified coordinates
  derive-chunks          Replace a sequence graph with subgraphs in the ranges of the specified coordinates
  make-stitch            Combine multiple sequence graphs into one. Example:
                             gen make-stitch --sample parent_sample --new-sample my_child_sample --regions chr1.2,chr1.3 --new-region spliced_chr1
  help                   Print this message or the help of the given subcommand(s)

Options:
  -d, --db <DB>  The path to the database you wish to utilize
  -h, --help     Print help
  -V, --version  Print version
```
