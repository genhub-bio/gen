use clap::{Parser, Subcommand};
use gen_models::db::OperationsConnection;

pub mod cli_context;
pub mod clone;
pub mod derive;
pub mod export;
pub mod graph_operations;
pub mod import;
#[cfg(all(debug_assertions, feature = "profiling"))]
pub mod profile;
pub mod remote;
pub mod update;

#[derive(Subcommand)]
#[allow(clippy::large_enum_variant)]
pub enum Commands {
    Init {},
    /// Clone a GenHub repository
    ///
    /// Example: gen clone https://www.genhub.bio/api/repos/david-genhub-bio/addgene-plasmid-122028-genbank-diff
    #[command(arg_required_else_help(true))]
    Clone {
        /// The GenHub repository URL to clone
        #[clap(index = 1)]
        url: String,
    },
    #[cfg(all(debug_assertions, feature = "profiling"))]
    /// Profile a command in a dev build and print cumulative per-function timings.
    #[command(arg_required_else_help(true))]
    Profile(profile::Command),
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
        sample: String,
    },
    /// Show a visual representation of a graph in the terminal
    #[command()]
    View {
        /// The name of the graph to view
        #[clap(index = 1)]
        graph: Option<String>,
        /// Optional sample to open directly. If omitted, choose it in the UI.
        #[arg(short, long)]
        sample: Option<String>,
        /// Look for the sample in a specific collection
        #[arg(short, long)]
        collection: Option<String>,
        /// Position as "node id:coordinate" to center the graph on
        #[arg(short, long)]
        position: Option<String>,
        /// Show the full TUI explorer instead of the inline preview. Includes sidebar explorer and additional interactive features.
        #[arg(short, long)]
        full: bool,
        /// Number of terminal rows for the inline graph view
        #[arg(long, default_value_t = 10)]
        height: u16,
    },
    /// Show a diff of operations and render the consolidated graph
    #[command(name = "view-diff", arg_required_else_help(true))]
    ViewDiff {
        /// The base ref (operation hash/prefix, branch name, or HEAD shorthand) to diff from
        #[clap(index = 1)]
        from: String,
        /// The target ref to diff to (defaults to the currently checked out operation)
        #[clap(index = 2)]
        to: Option<String>,
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
        /// Set the remote for the current branch
        #[arg(long)]
        set_remote: Option<String>,
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
    /// Manage remote repositories
    #[command(subcommand)]
    Remote(remote::RemoteCommand),

    /// Push the local repo to the remote
    #[command()]
    Push {
        /// The remote to push to
        #[arg(short, long)]
        remote: Option<String>,
    },
    #[command()]
    Pull {
        /// The remote to pull from
        #[arg(short, long)]
        remote: Option<String>,
    },
    /// Convert annotation coordinates between two samples
    #[command(arg_required_else_help(true))]
    PropagateAnnotations {
        /// The name of the collection to annotate
        #[arg(short, long)]
        name: Option<String>,
        /// The name of the sample the annotations are referenced to (if not provided, the default)
        #[arg(short, long)]
        from_sample: String,
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
    /// Add an annotation and accession for a region
    #[command(name = "add-annotation", arg_required_else_help(true))]
    AddAnnotation {
        /// The annotation name
        #[arg(short, long)]
        name: String,
        /// The annotation group name
        #[arg(short, long)]
        group: Option<String>,
        /// The sample name to annotate (defaults to the collection's default sample)
        #[arg(short, long)]
        sample: String,
        /// The region to annotate (region:start-end)
        #[clap(index = 1)]
        region: String,
    },
    /// Add an annotation file without importing its intervals
    #[command(name = "add-annotation-file", arg_required_else_help(true))]
    AddAnnotationFile {
        /// The annotation file path
        #[clap(index = 1)]
        path: String,
        /// The annotation file format (gff3, bed, genbank). If omitted, infer from the file extension.
        #[arg(short, long)]
        format: Option<String>,
        /// Optional tabix index file path for the annotation file
        #[arg(long)]
        index: Option<String>,
        /// Optional annotation file name
        #[arg(short, long)]
        name: Option<String>,
        /// Optional operation summary message
        #[arg(short, long)]
        message: Option<String>,
    },
    /// Add one or more files as an operation
    #[command(name = "add-file", arg_required_else_help(true))]
    AddFile {
        /// Files to add to the operation
        #[clap(index = 1, required = true, num_args = 1..)]
        files: Vec<String>,
        /// Optional operation summary message
        #[arg(short, long)]
        message: Option<String>,
    },
    /// Build a k-mer index for fast sequence search
    #[command(name = "build-index")]
    BuildIndex {
        /// The collection to index (defaults to the configured default)
        #[arg(short, long)]
        collection: Option<String>,
        /// Restrict indexing to a specific sample
        #[arg(short, long)]
        sample: Option<String>,
        /// K-mer size used when building the index
        #[arg(short, long, default_value = "16")]
        kmer_size: usize,
    },
    /// Clear the search index cache
    #[command(name = "clear-index")]
    ClearIndex {
        /// The collection to clear indices for (defaults to the configured default)
        #[arg(short, long)]
        collection: Option<String>,
        /// Restrict clearing to a specific sample
        #[arg(short, long)]
        sample: Option<String>,
    },
    /// Search for an exact sequence across all block groups
    ///
    /// Each match is reported with a blocks column formatted as [hash:start-end, ...],
    /// where hash is a 12-character node hash prefix from the original node the block
    /// was carved from at coordinates start/end. The offset column gives the position
    /// within the first block where the match begins.
    #[command(arg_required_else_help(true))]
    Search {
        /// The sequence to search for
        #[clap(index = 1)]
        query: String,
        /// Restrict the search to a specific sample
        #[arg(short, long)]
        sample: Option<String>,
        /// The collection to search in
        #[arg(short, long)]
        collection: Option<String>,
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
        sample: String,
    },
    /// Extract a sequence from a graph
    #[command(arg_required_else_help(true))]
    GetSequence {
        /// The name of the collection containing the sequence
        #[arg(short, long)]
        name: Option<String>,
        /// The name of the sample containing the sequence
        #[arg(short, long)]
        sample: String,
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
        sample1: String,
        /// The name of the second sample to diff
        #[arg(long)]
        sample2: String,
        /// The name of the output GFA file
        #[arg(long)]
        gfa: String,
    },
    /// Commands for deriving new sequence graphs
    #[command(arg_required_else_help(true))]
    Derive(derive::Command),
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
        sample: String,
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
    AddReferenceAliases {
        /// The name of the reference to set up aliases for
        #[arg(long)]
        reference_name: String,
        /// The refseq accession ID
        #[arg(long)]
        refseq_accession_id: Option<String>,
        /// The genbank ID
        #[arg(long)]
        genbank_id: Option<String>,
        /// The ensembl ID
        #[arg(long)]
        ensembl_id: Option<String>,
        /// The UCSC ID
        #[arg(long)]
        ucsc_id: Option<String>,
        /// A custom ID, can be anything
        #[arg(long)]
        custom_id: Option<String>,
        /// The chromosome number (for cases like Roman numberals, eg 11 if the ensembl ID is XI)
        #[arg(long)]
        chromosome: Option<i64>,
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

pub fn get_default_collection(conn: &OperationsConnection) -> String {
    let mut stmt = conn
        .prepare("select collection_name from defaults where id = 1")
        .unwrap();
    stmt.query_row((), |row| row.get(0))
        .unwrap_or("default".to_string())
}
