use crate::commands::cli_context::CliContext;
use crate::config::get_gen_dir;
use clap::{Parser, Subcommand};
use rusqlite::Connection;
use std::path::PathBuf;

pub mod cli_context;
pub mod export;
pub mod import;
pub mod update;

#[derive(Subcommand)]
#[allow(clippy::large_enum_variant)]
pub enum Commands {
    Init {},
    /// Commands for importing
    Import(import::Command),
    /// Commands for updating
    Update(update::Command),
    /// Commands for exporting
    Export(export::Command),
    /// Commands for transforming file types for input to Gen.
    #[command(arg_required_else_help(true))]
    Transform {
        /// For update-gaf, this transforms the csv to a fasta for use in alignments
        #[arg(long)]
        format_csv_for_gaf: Option<String>,
    },
    /// Translate coordinates of standard bioinformatic file formats.
    #[command(arg_required_else_help(true))]
    Translate {
        /// Transform coordinates of a BED to graph nodes
        #[arg(long)]
        bed: Option<String>,
        /// Transform coordinates of a GFF to graph nodes
        #[arg(long)]
        gff: Option<String>,
        /// The name of the collection to map sequences against
        #[arg(short, long)]
        collection: Option<String>,
        /// The sample name whose graph coordinates are mapped against
        #[arg(short, long)]
        sample: Option<String>,
    },
    /// Show a visual representation of a graph in the terminal
    #[command()]
    View {
        /// The name of the graph to view
        #[clap(index = 1)]
        graph: Option<String>,
        /// View the graph for a specific sample
        #[arg(short, long)]
        sample: Option<String>,
        /// Look for the sample in a specific collection
        #[arg(short, long)]
        collection: Option<String>,
        /// Position as "node id:coordinate" to center the graph on
        #[arg(short, long)]
        position: Option<String>,
    },
    /// Export a set of operations to a patch file
    #[command(name = "patch-create", arg_required_else_help(true))]
    PatchCreate {
        /// To create a patch against a non-checked out branch.
        #[arg(short, long)]
        branch: Option<String>,
        /// The patch name
        #[arg(short, long)]
        name: String,
        /// The operation(s) to create a patch from. For a range, use start..end and for multiple
        /// or discontinuous ranges, use commas. HEAD and HEAD~<number> syntax is supported.
        #[clap(index = 1)]
        operation: String,
    },
    /// Apply changes from a patch file
    #[command(name = "patch-apply", arg_required_else_help(true))]
    PatchApply {
        /// The patch file
        #[clap(index = 1)]
        patch: String,
    },
    /// View a patch in dot format
    #[command(name = "patch-view", arg_required_else_help(true))]
    PatchView {
        /// The prefix to use in the output filenames. One dot file is created for each operation and graph,
        /// following the pattern {prefix}_{operation}_{graph_id}.dot. Defaults to patch filename.
        #[arg(long, short)]
        prefix: Option<String>,
        /// The patch file
        #[clap(index = 1)]
        patch: String,
    },
    /// Manage and create branches
    #[command(arg_required_else_help(true))]
    Branch {
        /// Create a branch with the given name
        #[arg(long, action)]
        create: bool,
        /// Delete a given branch
        #[arg(short, long, action)]
        delete: bool,
        /// Checkout a given branch
        #[arg(long, action)]
        checkout: bool,
        /// List all branches
        #[arg(short, long, action)]
        list: bool,
        #[arg(short, long, action)]
        merge: bool,
        /// The branch name
        #[clap(index = 1)]
        branch_name: Option<String>,
    },
    /// Merge branches
    #[command(arg_required_else_help(true))]
    Merge {
        /// The branch name to merge
        #[clap(index = 1)]
        branch_name: Option<String>,
    },
    /// Migrate a database to a given operation
    #[command(arg_required_else_help(true))]
    Checkout {
        /// Create and checkout a new branch.
        #[arg(short, long)]
        branch: Option<String>,
        /// The operation hash to move to
        #[clap(index = 1)]
        hash: Option<String>,
    },
    /// Reset a branch to a previous operation
    #[command(arg_required_else_help(true))]
    Reset {
        /// The operation hash to reset to
        #[clap(index = 1)]
        hash: String,
    },
    /// View operations carried out against a database
    #[command()]
    Operations {
        /// Edit operation messages
        #[arg(short, long)]
        interactive: bool,
        /// The branch to list operations for
        #[arg(short, long)]
        branch: Option<String>,
    },
    /// Apply an operation to a branch
    #[command(arg_required_else_help(true))]
    Apply {
        /// The operation hash to apply
        #[clap(index = 1)]
        hash: String,
    },
    /// Configure default options
    #[command(arg_required_else_help(true))]
    Defaults {
        /// The default database to use
        #[arg(short, long)]
        database: Option<String>,
        /// The default collection to use
        #[arg(short, long)]
        collection: Option<String>,
    },
    /// Set the remote URL for this repo
    #[command(arg_required_else_help(true))]
    SetRemote {
        /// The remote URL to set
        #[arg(short, long)]
        remote: String,
    },
    /// Push the local repo to the remote
    #[command()]
    Push {},
    #[command()]
    Pull {},
    /// Convert annotation coordinates between two samples
    #[command(arg_required_else_help(true))]
    PropagateAnnotations {
        /// The name of the collection to annotate
        #[arg(short, long)]
        name: Option<String>,
        /// The name of the sample the annotations are referenced to (if not provided, the default)
        #[arg(short, long)]
        from_sample: Option<String>,
        /// The name of the sample to annotate
        #[arg(short, long)]
        to_sample: String,
        /// The name of the annotation file to propagate
        #[arg(short, long)]
        gff: String,
        /// The name of the output file
        #[arg(short, long)]
        output_gff: String,
    },
    /// List all samples in the current collection
    ListSamples {},
    #[command()]
    /// List all regions/contigs in the current collection and given sample
    ListGraphs {
        /// The name of the collection to list graphs for
        #[arg(short, long)]
        name: Option<String>,
        /// The name of the sample to list graphs for
        #[arg(short, long)]
        sample: Option<String>,
    },
    /// Extract a sequence from a graph
    #[command(arg_required_else_help(true))]
    GetSequence {
        /// The name of the collection containing the sequence
        #[arg(short, long)]
        name: Option<String>,
        /// The name of the sample containing the sequence
        #[arg(short, long)]
        sample: Option<String>,
        /// The name of the graph to get the sequence for
        #[arg(short, long)]
        graph: Option<String>,
        /// The start coordinate of the sequence
        #[arg(long)]
        start: Option<i64>,
        /// The end coordinate of the sequence
        #[arg(long)]
        end: Option<i64>,
        /// The region (name:start-end format) of the sequence
        #[arg(long)]
        region: Option<String>,
    },
    /// Output a file representing the "diff" between two samples
    Diff {
        /// The name of the collection to diff
        #[arg(short, long)]
        name: Option<String>,
        /// The name of the first sample to diff
        #[arg(long)]
        sample1: Option<String>,
        /// The name of the second sample to diff
        #[arg(long)]
        sample2: Option<String>,
        /// The name of the output GFA file
        #[arg(long)]
        gfa: String,
    },
    /// Replace a sequence graph with a subgraph in the range of the specified coordinates
    DeriveSubgraph {
        /// The name of the collection to derive the subgraph from
        #[arg(short, long)]
        name: Option<String>,
        /// The name of the parent sample
        #[arg(short, long)]
        sample: Option<String>,
        /// The name of the new sample
        #[arg(long)]
        new_sample: String,
        /// The name of the region to derive the subgraph from
        #[arg(short, long)]
        region: String,
        /// Name of alternate path (not current) to use
        #[arg(long)]
        backbone: Option<String>,
    },
    /// Replace a sequence graph with subgraphs in the ranges of the specified coordinates
    DeriveChunks {
        /// The name of the collection to derive the subgraph from
        #[arg(short, long)]
        name: Option<String>,
        /// The name of the parent sample
        #[arg(short, long)]
        sample: Option<String>,
        /// The name of the new sample
        #[arg(long)]
        new_sample: String,
        /// The name of the region to derive the subgraph from
        #[arg(short, long)]
        region: String,
        /// Name of alternate path (not current) to use
        #[arg(long)]
        backbone: Option<String>,
        /// Breakpoints to derive chunks from
        #[arg(long)]
        breakpoints: Option<String>,
        /// The size of the chunks to derive
        #[arg(long)]
        chunk_size: Option<i64>,
    },
    #[command(
        verbatim_doc_comment,
        long_about = "Combine multiple sequence graphs into one. Example:
    gen make-stitch --sample parent_sample --new-sample my_child_sample --regions chr1.2,chr1.3 --new-region spliced_chr1"
    )]
    MakeStitch {
        /// The name of the collection to derive the subgraph from
        #[arg(short, long)]
        name: Option<String>,
        /// The name of the parent sample
        #[arg(short, long)]
        sample: Option<String>,
        /// The name of the new sample
        #[arg(long)]
        new_sample: String,
        /// The names of the regions to combine
        #[arg(long)]
        regions: String,
        /// The name of the new region
        #[arg(long)]
        new_region: String,
    },
}

#[derive(Parser)]
#[command(version, about, long_about = None, arg_required_else_help(true))]
pub struct Cli {
    /// The path to the database you wish to utilize
    #[arg(short, long)]
    pub db: Option<String>,
    #[command(subcommand)]
    pub command: Option<Commands>,
}

pub fn get_db_for_command(cli_context: &CliContext, operation_conn: &Connection) -> String {
    let binding = cli_context.db.clone().unwrap_or_else(|| {
        let mut stmt = operation_conn
            .prepare("select db_name from defaults where id = 1;")
            .unwrap();
        let row: Option<String> = stmt.query_row((), |row| row.get(0)).unwrap();
        row.unwrap_or_else(|| match get_gen_dir() {
            Some(dir) => PathBuf::from(dir)
                .join("default.db")
                .to_str()
                .unwrap()
                .to_string(),
            None => {
                panic!("No .gen directory found. Please run 'gen init' first.")
            }
        })
    });
    binding
}
