#![allow(warnings)]
use clap::{Parser, Subcommand};
use core::ops::Range;
use gen::config;
use gen::config::{get_gen_dir, get_operation_connection};
use gen::Commands;
use gen::NewCli;
use rusqlite::params;

use gen::annotations::gff::propagate_gff;
use gen::commands::cli_context::CliContext;
use gen::diffs::gfa::gfa_sample_diff;
use gen::exports::fasta::export_fasta;
use gen::exports::genbank::export_genbank;
use gen::exports::gfa::export_gfa;
use gen::fasta::FastaError;
use gen::genbank::GenBankError;
use gen::get_connection;
use gen::graph_operators::{derive_chunks, get_path, make_stitch};
use gen::imports::fasta::import_fasta;
use gen::imports::genbank::import_genbank;
use gen::imports::gfa::{import_gfa, GFAImportError};
use gen::imports::library::import_library;
use gen::models::block_group::BlockGroup;
use gen::models::file_types::FileTypes;
use gen::models::metadata;
use gen::models::operations::{
    setup_db, Branch, Operation, OperationFile, OperationInfo, OperationState,
};
use gen::models::sample::Sample;
use gen::models::traits::Query;
use gen::operation_management;
use gen::operation_management::{parse_patch_operations, push, OperationError};
use gen::patch;
use gen::translate;
use gen::updates::fasta::update_with_fasta;
use gen::updates::gaf::{transform_csv_to_fasta, update_with_gaf};
use gen::updates::genbank::update_with_genbank;
use gen::updates::gfa::update_with_gfa;
use gen::updates::library::update_with_library;
use gen::updates::vcf::{update_with_vcf, VcfError};
use gen::views::block_group::view_block_group;
use gen::views::operations::view_operations;
use gen::views::patch::view_patches;

use itertools::Itertools;
use noodles::core::Region;
use rusqlite::{types::Value, Connection};
use std::fmt::Debug;
use std::fs::File;
use std::io::{BufReader, Write};
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::{io, str};

#[derive(Parser)]
#[command(version, about, long_about = None, arg_required_else_help(true))]
pub struct Cli {
    /// The path to the database you wish to utilize
    #[arg(short, long)]
    pub db: Option<String>,
    #[command(subcommand)]
    pub command: Option<OldCommands>,
}

fn get_default_collection(conn: &Connection) -> String {
    let mut stmt = conn
        .prepare("select collection_name from defaults where id = 1")
        .unwrap();
    stmt.query_row((), |row| row.get(0))
        .unwrap_or("default".to_string())
}

#[derive(Subcommand)]
#[allow(clippy::large_enum_variant)]
enum OldCommands {
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
    /// Update a sequence collection with new data
    #[command(arg_required_else_help(true))]
    Update {
        /// The name of the collection to update
        #[arg(short, long)]
        name: Option<String>,
        /// A fasta file to insert
        #[arg(short, long)]
        fasta: Option<String>,
        /// A VCF file to incorporate
        #[arg(short, long)]
        vcf: Option<String>,
        /// A GenBank file to update from
        #[arg(long)]
        gb: Option<String>,
        /// If no genotype is provided, enter the genotype to assign variants
        #[arg(short, long)]
        genotype: Option<String>,
        /// If no sample is provided, enter the sample to associate variants to
        #[arg(short, long)]
        sample: Option<String>,
        /// New sample name if we are updating with intentional edits
        #[arg(long)]
        new_sample: Option<String>,
        /// Use the given sample as the parent sample for changes.
        #[arg(long, alias = "cf")]
        coordinate_frame: Option<String>,
        /// A CSV with combinatorial library information
        #[arg(short, long)]
        library: Option<String>,
        /// A fasta with the combinatorial library parts
        #[arg(long)]
        parts: Option<String>,
        /// The name of the path to add the library to
        #[arg(short, long)]
        path_name: Option<String>,
        /// The name of the region to update (eg "chr1")
        #[arg(long)]
        region_name: Option<String>,
        /// The start coordinate for the region to add the library to
        #[arg(long)]
        start: Option<i64>,
        /// The end coordinate for the region to add the library to
        #[arg(short, long)]
        end: Option<i64>,
        /// For fasta updates, do not update the sample's reference path if there is a single fasta entry
        #[arg(long, action)]
        no_reference_path_update: bool,
        /// If a new entity is found, create it as a normal import
        #[arg(long, action, alias = "cm")]
        create_missing: bool,
        /// A GFA file to update from
        #[arg(long)]
        gfa: Option<String>,
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
    /// Update a sequence collecting using GAF results.
    #[command(name = "update-gaf", arg_required_else_help(true))]
    UpdateGaf {
        /// The name of the collection to update
        #[arg(short, long)]
        name: Option<String>,
        /// The GAF input
        #[arg(short, long)]
        gaf: String,
        /// The csv describing changes to make
        #[arg(short, long)]
        csv: String,
        /// The sample to update or create
        #[arg(short, long)]
        sample: String,
        /// If specified, the newly created sample will inherit this sample's existing graph
        #[arg(short, long)]
        parent_sample: Option<String>,
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
    /// Initialize a gen repository
    Init {},
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
    /// Export sequence data
    #[command(arg_required_else_help(true))]
    Export {
        /// The name of the collection to export
        #[arg(short, long)]
        name: Option<String>,
        /// The name of the GFA file to export to
        #[arg(short, long)]
        gfa: Option<String>,
        /// An optional sample name
        #[arg(short, long)]
        sample: Option<String>,
        /// The name of the fasta file to export to
        #[arg(short, long)]
        fasta: Option<String>,
        /// The name of the GenBank file to export to
        #[arg(long)]
        gb: Option<String>,
        /// The max node size for gfa export
        #[arg(long)]
        node_max: Option<i64>,
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

fn main() {
    // Start logger (gets log level from RUST_LOG environment variable, sends output to stderr)
    env_logger::init();

    let cli = NewCli::parse();
    let cli_context = CliContext::from(&cli);

    match cli.command {
        Some(Commands::Import(cmd)) => gen::commands::import::execute(&cli_context, cmd),
        None => println!("No command provided."),
    }
}

fn main2() {
    // Start logger (gets log level from RUST_LOG environment variable, sends output to stderr)
    env_logger::init();

    let cli = Cli::parse();

    // commands not requiring a db connection are handled here
    if let Some(OldCommands::Init {}) = &cli.command {
        config::get_or_create_gen_dir();
        println!("Gen repository initialized.");
        return;
    }

    let operation_conn = get_operation_connection(None);
    if let Some(OldCommands::Defaults {
        database,
        collection,
    }) = &cli.command
    {
        if let Some(name) = database {
            operation_conn
                .execute("update defaults set db_name=?1 where id = 1", (name,))
                .unwrap();
            println!("Default database set to {name}");
        }
        if let Some(name) = collection {
            operation_conn
                .execute(
                    "update defaults set collection_name=?1 where id = 1",
                    (name,),
                )
                .unwrap();
            println!("Default collection set to {name}");
        }
        return;
    }
    if let Some(OldCommands::SetRemote { remote }) = &cli.command {
        operation_conn
            .execute("update defaults set remote_url=?1 where id = 1", (remote,))
            .unwrap();
        println!("Remote URL set to {remote}");
        return;
    }

    if let Some(OldCommands::Transform { format_csv_for_gaf }) = &cli.command {
        let csv = format_csv_for_gaf
            .clone()
            .expect("csv for transformation not provided.");
        let stdout = io::stdout();
        let mut handle = stdout.lock();
        let mut csv_file = File::open(csv).unwrap();
        transform_csv_to_fasta(&mut csv_file, &mut handle);
        return;
    }

    let binding = cli.db.unwrap_or_else(|| {
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
    let db = binding.as_str();
    let conn = get_connection(db);
    let db_uuid = metadata::get_db_uuid(&conn);

    // initialize the selected database if needed.
    setup_db(&operation_conn, &db_uuid);

    match &cli.command {
        Some(OldCommands::View {
            graph,
            sample,
            collection,
            position,
        }) => {
            let collection_name = &collection
                .clone()
                .unwrap_or_else(|| get_default_collection(&operation_conn));

            // view_block_group is a long-running operation that manages its own transactions
            view_block_group(
                &conn,
                graph.clone(),
                sample.clone(),
                collection_name,
                position.clone(),
            );
        }
        Some(OldCommands::Update {
            name,
            fasta,
            vcf,
            gb,
            library,
            parts,
            genotype,
            sample,
            new_sample,
            path_name,
            region_name,
            start,
            end,
            no_reference_path_update,
            coordinate_frame,
            create_missing,
            gfa,
        }) => {
            conn.execute("BEGIN TRANSACTION", []).unwrap();
            operation_conn.execute("BEGIN TRANSACTION", []).unwrap();
            let name = &name
                .clone()
                .unwrap_or_else(|| get_default_collection(&operation_conn));
            if let Some(library_path) = library {
                update_with_library(
                    &conn,
                    &operation_conn,
                    name,
                    sample.clone().as_deref(),
                    &new_sample.clone().unwrap(),
                    &path_name.clone().unwrap(),
                    start.unwrap(),
                    end.unwrap(),
                    &parts.clone().unwrap(),
                    library_path,
                )
                .unwrap();
            } else if let Some(fasta_path) = fasta {
                // NOTE: This has to go after library because the library update also uses a fasta
                // file
                update_with_fasta(
                    &conn,
                    &operation_conn,
                    name,
                    sample.clone().as_deref(),
                    &new_sample
                        .clone()
                        .expect("new-sample flag must be provided."),
                    &region_name.clone().expect("region-name must be provided."),
                    start.expect("start flag must be provided."),
                    end.expect("end flag must be provided."),
                    fasta_path,
                    *no_reference_path_update,
                )
                .unwrap();
            } else if let Some(vcf_path) = vcf {
                match update_with_vcf(
                    vcf_path,
                    name,
                    genotype.clone().unwrap_or("".to_string()),
                    sample.clone().unwrap_or("".to_string()),
                    &conn,
                    &operation_conn,
                    coordinate_frame.as_deref(),
                ) {
                    Ok(_) => {},
                    Err(VcfError::OperationError(OperationError::NoChanges)) => println!("No changes made. If the VCF lacks a sample or genotype, they need to be provided via --sample and --genotype."),
                    Err(e) => panic!("Error updating with vcf: {e}"),
                }
            } else if let Some(gb_path) = gb {
                let f = File::open(gb_path).unwrap();
                match update_with_genbank(
                    &conn,
                    &operation_conn,
                    &f,
                    name.deref(),
                    *create_missing,
                    &OperationInfo {
                        files: vec![OperationFile {
                            file_path: gb_path.clone(),
                            file_type: FileTypes::GenBank,
                        }],
                        description: "Update from GenBank".to_string(),
                    },
                ) {
                    Ok(_) => {}
                    Err(e) => panic!("Failed to update. Error is: {e}"),
                }
            } else if let Some(gfa_path) = gfa {
                match update_with_gfa(
                    &conn,
                    &operation_conn,
                    name,
                    sample.clone().as_deref(),
                    &new_sample.clone().unwrap(),
                    gfa_path,
                ) {
                    Ok(_) => {}
                    Err(e) => panic!("Failed to update. Error is: {e}"),
                }
            } else {
                panic!("Unknown file type provided for update.");
            }

            conn.execute("END TRANSACTION", []).unwrap();
            operation_conn.execute("END TRANSACTION", []).unwrap();
        }
        Some(OldCommands::UpdateGaf {
            name,
            gaf,
            csv,
            sample,
            parent_sample,
        }) => {
            conn.execute("BEGIN TRANSACTION", []).unwrap();
            operation_conn.execute("BEGIN TRANSACTION", []).unwrap();
            let name = &name
                .clone()
                .unwrap_or_else(|| get_default_collection(&operation_conn));
            update_with_gaf(
                &conn,
                &operation_conn,
                gaf,
                csv,
                name,
                Some(sample.as_ref()),
                parent_sample.as_deref(),
            );
            conn.execute("END TRANSACTION", []).unwrap();
            operation_conn.execute("END TRANSACTION", []).unwrap();
        }
        Some(OldCommands::Translate {
            bed,
            gff,
            collection,
            sample,
        }) => {
            let collection = &collection
                .clone()
                .unwrap_or_else(|| get_default_collection(&operation_conn));
            if let Some(bed) = bed {
                let stdout = io::stdout();
                let mut handle = stdout.lock();
                let mut bed_file = File::open(bed).unwrap();
                match translate::bed::translate_bed(
                    &conn,
                    collection,
                    sample.as_deref(),
                    &mut bed_file,
                    &mut handle,
                ) {
                    Ok(_) => {}
                    Err(err) => {
                        panic!("Error Translating Bed. {err}");
                    }
                }
            } else if let Some(gff) = gff {
                let stdout = io::stdout();
                let mut handle = stdout.lock();
                let mut gff_file = BufReader::new(File::open(gff).unwrap());
                match translate::gff::translate_gff(
                    &conn,
                    collection,
                    sample.as_deref(),
                    &mut gff_file,
                    &mut handle,
                ) {
                    Ok(_) => {}
                    Err(err) => {
                        panic!("Error Translating GFF. {err}");
                    }
                }
            }
        }
        Some(OldCommands::Operations {
            interactive,
            branch,
        }) => {
            let current_op = OperationState::get_operation(&operation_conn, &db_uuid);
            if let Some(current_op) = current_op {
                let branch_name = branch.clone().unwrap_or_else(|| {
                    let current_branch_id =
                        OperationState::get_current_branch(&operation_conn, &db_uuid)
                            .expect("No current branch is set.");
                    Branch::get_by_id(&operation_conn, current_branch_id)
                        .unwrap_or_else(|| panic!("No branch with id {current_branch_id}"))
                        .name
                });
                let operations = Branch::get_operations(
                    &operation_conn,
                    Branch::get_by_name(&operation_conn, &db_uuid, &branch_name)
                        .unwrap_or_else(|| panic!("No branch named {branch_name}."))
                        .id,
                );
                if *interactive {
                    view_operations(&conn, &operation_conn, &operations);
                } else {
                    let mut indicator = "";
                    println!(
                        "{indicator:<3}{col1:>64}   {col2:<70}",
                        col1 = "Id",
                        col2 = "Summary"
                    );
                    for op in operations.iter() {
                        if op.hash == current_op {
                            indicator = ">";
                        } else {
                            indicator = "";
                        }
                        println!(
                            "{indicator:<3}{col1:>64}   {col2:<70}",
                            col1 = op.hash,
                            col2 = op.change_type
                        );
                    }
                }
            } else {
                println!("No operations found.");
            }
        }
        Some(OldCommands::Branch {
            create,
            delete,
            checkout,
            list,
            merge,
            branch_name,
        }) => {
            if *create {
                Branch::create(
                    &operation_conn,
                    &db_uuid,
                    &branch_name
                        .clone()
                        .expect("Must provide a branch name to create."),
                );
            } else if *delete {
                Branch::delete(
                    &operation_conn,
                    &db_uuid,
                    &branch_name
                        .clone()
                        .expect("Must provide a branch name to delete."),
                );
            } else if *checkout {
                operation_management::checkout(
                    &conn,
                    &operation_conn,
                    &db_uuid,
                    &Some(
                        branch_name
                            .clone()
                            .expect("Must provide a branch name to checkout.")
                            .to_string(),
                    ),
                    None,
                );
            } else if *list {
                let current_branch = OperationState::get_current_branch(&operation_conn, &db_uuid);
                let mut indicator = "";
                println!(
                    "{indicator:<3}{col1:<30}   {col2:<20}",
                    col1 = "Name",
                    col2 = "Operation",
                );
                for branch in Branch::query(
                    &operation_conn,
                    "select * from branch where db_uuid = ?1",
                    params![Value::from(db_uuid.to_string())],
                )
                .iter()
                {
                    if let Some(current_branch_id) = current_branch {
                        if current_branch_id == branch.id {
                            indicator = ">";
                        } else {
                            indicator = "";
                        }
                    }
                    println!(
                        "{indicator:<3}{col1:<30}   {col2:<20}",
                        col1 = branch.name,
                        col2 = branch
                            .current_operation_hash
                            .clone()
                            .unwrap_or(String::new())
                    );
                }
            } else if *merge {
                let branch_name = branch_name.clone().expect("Branch name must be provided.");
                let other_branch = Branch::get_by_name(&operation_conn, &db_uuid, &branch_name)
                    .unwrap_or_else(|| panic!("Unable to find branch {branch_name}."));
                let current_branch = OperationState::get_current_branch(&operation_conn, &db_uuid)
                    .expect("Unable to find current branch.");
                conn.execute("BEGIN TRANSACTION", []).unwrap();
                operation_conn.execute("BEGIN TRANSACTION", []).unwrap();
                match operation_management::merge(
                    &conn,
                    &operation_conn,
                    &db_uuid,
                    current_branch,
                    other_branch.id,
                    None,
                ) {
                    Ok(_) => println!("Merge successful"),
                    Err(_) => {
                        conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
                        operation_conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
                        panic!("Merge failed.");
                    }
                }
                conn.execute("END TRANSACTION", []).unwrap();
                operation_conn.execute("END TRANSACTION", []).unwrap();
            } else {
                println!("No options selected.");
            }
        }
        Some(OldCommands::Merge { branch_name }) => {
            let branch_name = branch_name.clone().expect("Branch name must be provided.");
            let other_branch = Branch::get_by_name(&operation_conn, &db_uuid, &branch_name)
                .unwrap_or_else(|| panic!("Unable to find branch {branch_name}."));
            let current_branch = OperationState::get_current_branch(&operation_conn, &db_uuid)
                .expect("Unable to find current branch.");
            conn.execute("BEGIN TRANSACTION", []).unwrap();
            operation_conn.execute("BEGIN TRANSACTION", []).unwrap();
            match operation_management::merge(
                &conn,
                &operation_conn,
                &db_uuid,
                current_branch,
                other_branch.id,
                None,
            ) {
                Ok(_) => println!("Merge successful"),
                Err(details) => {
                    conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
                    operation_conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
                    panic!("Merge failed: {details}");
                }
            }
            conn.execute("END TRANSACTION", []).unwrap();
            operation_conn.execute("END TRANSACTION", []).unwrap();
        }
        Some(OldCommands::Apply { hash }) => {
            conn.execute("BEGIN TRANSACTION", []).unwrap();
            operation_conn.execute("BEGIN TRANSACTION", []).unwrap();
            match operation_management::apply(&conn, &operation_conn, hash, None) {
                Ok(_) => println!("Operation applied"),
                Err(_) => {
                    conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
                    operation_conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
                    panic!("Apply failed.");
                }
            }
            conn.execute("END TRANSACTION", []).unwrap();
            operation_conn.execute("END TRANSACTION", []).unwrap();
        }
        Some(OldCommands::Checkout { branch, hash }) => {
            if let Some(name) = branch.clone() {
                if Branch::get_by_name(&operation_conn, &db_uuid, &name).is_none() {
                    Branch::create(&operation_conn, &db_uuid, &name);
                    println!("Created branch {name}");
                }
                println!("Checking out branch {name}");
                operation_management::checkout(&conn, &operation_conn, &db_uuid, &Some(name), None);
            } else if let Some(hash_name) = hash.clone() {
                // if the hash is a branch, check it out
                if Branch::get_by_name(&operation_conn, &db_uuid, &hash_name).is_some() {
                    println!("Checking out branch {hash_name}");
                    operation_management::checkout(
                        &conn,
                        &operation_conn,
                        &db_uuid,
                        &Some(hash_name),
                        None,
                    );
                } else {
                    println!("Checking out operation {hash_name}");
                    operation_management::checkout(
                        &conn,
                        &operation_conn,
                        &db_uuid,
                        &None,
                        Some(hash_name),
                    );
                }
            } else {
                println!("No branch or hash to checkout provided.");
            }
        }
        Some(OldCommands::Reset { hash }) => {
            operation_management::reset(&conn, &operation_conn, &db_uuid, hash);
        }
        Some(OldCommands::Export {
            name,
            gb,
            gfa,
            sample,
            fasta,
            node_max,
        }) => {
            let name = &name
                .clone()
                .unwrap_or_else(|| get_default_collection(&operation_conn));
            conn.execute("BEGIN TRANSACTION", []).unwrap();
            operation_conn.execute("BEGIN TRANSACTION", []).unwrap();
            if let Some(gfa_path) = gfa {
                export_gfa(
                    &conn,
                    name,
                    &PathBuf::from(gfa_path),
                    sample.clone(),
                    *node_max,
                );
            } else if let Some(fasta_path) = fasta {
                export_fasta(
                    &conn,
                    name,
                    sample.clone().as_deref(),
                    &PathBuf::from(fasta_path),
                );
            } else if let Some(gb_path) = gb {
                export_genbank(
                    &conn,
                    name,
                    sample.clone().as_deref(),
                    &PathBuf::from(gb_path),
                );
            } else {
                println!("No file type specified for export.");
            }
            conn.execute("END TRANSACTION", []).unwrap();
            operation_conn.execute("END TRANSACTION", []).unwrap();
        }
        Some(OldCommands::PatchCreate {
            name,
            operation,
            branch,
        }) => {
            let branch = if let Some(branch_name) = branch {
                Branch::get_by_name(&operation_conn, &db_uuid, branch_name)
                    .unwrap_or_else(|| panic!("No branch with name {branch_name} found."))
            } else {
                let current_branch_id =
                    OperationState::get_current_branch(&operation_conn, &db_uuid)
                        .expect("No current branch is checked out.");
                Branch::get_by_id(&operation_conn, current_branch_id).unwrap()
            };
            let branch_ops = Branch::get_operations(&operation_conn, branch.id);
            let operations = parse_patch_operations(
                &branch_ops,
                &branch.current_operation_hash.unwrap(),
                operation,
            );
            let mut f = File::create(format!("{name}.gz")).unwrap();
            patch::create_patch(&operation_conn, &operations, &mut f);
        }
        Some(OldCommands::PatchApply { patch }) => {
            let mut f = File::open(patch).unwrap();
            let patches = patch::load_patches(&mut f);
            patch::apply_patches(&conn, &operation_conn, &patches);
        }
        Some(OldCommands::PatchView { prefix, patch }) => {
            let patch_path = Path::new(patch);
            let mut f = File::open(patch_path).unwrap();
            let patches = patch::load_patches(&mut f);
            let diagrams = view_patches(&patches);
            for (patch_hash, patch_diagrams) in diagrams.iter() {
                for (bg_id, dot) in patch_diagrams.iter() {
                    let path = if let Some(p) = prefix {
                        format!("{p}_{patch_hash:.7}_{bg_id}.dot")
                    } else {
                        format!(
                            "{patch_base}_{patch_hash:.7}_{bg_id}.dot",
                            patch_base = patch_path
                                .with_extension("")
                                .file_name()
                                .unwrap()
                                .to_str()
                                .unwrap()
                        )
                    };
                    let mut f = File::create(path).unwrap();
                    f.write_all(dot.as_bytes())
                        .expect("Failed to write diagram");
                }
            }
        }
        None => {}
        // these will never be handled by this method as we search for them earlier.
        Some(OldCommands::Init {}) => {
            config::get_or_create_gen_dir();
            println!("Gen repository initialized.");
        }
        Some(OldCommands::Defaults {
            database,
            collection,
        }) => {}
        Some(OldCommands::SetRemote { remote }) => {}
        Some(OldCommands::Transform { format_csv_for_gaf }) => {}
        Some(OldCommands::PropagateAnnotations {
            name,
            from_sample,
            to_sample,
            gff,
            output_gff,
        }) => {
            let name = &name
                .clone()
                .unwrap_or_else(|| get_default_collection(&operation_conn));
            let from_sample_name = from_sample.clone();

            conn.execute("BEGIN TRANSACTION", []).unwrap();
            operation_conn.execute("BEGIN TRANSACTION", []).unwrap();

            propagate_gff(
                &conn,
                name,
                from_sample_name.as_deref(),
                to_sample,
                gff,
                output_gff,
            );

            conn.execute("END TRANSACTION", []).unwrap();
            operation_conn.execute("END TRANSACTION", []).unwrap();
        }
        Some(OldCommands::ListSamples {}) => {
            let sample_names = Sample::get_all_names(&conn);
            // Null sample
            println!();
            for sample_name in sample_names {
                println!("{}", sample_name);
            }
        }
        Some(OldCommands::ListGraphs { name, sample }) => {
            let name = &name
                .clone()
                .unwrap_or_else(|| get_default_collection(&operation_conn));
            let block_groups = Sample::get_block_groups(&conn, name, sample.as_deref());
            for block_group in block_groups {
                println!("{}", block_group.name);
            }
        }
        Some(OldCommands::GetSequence {
            name,
            sample,
            graph,
            start,
            end,
            region,
        }) => {
            let name = &name
                .clone()
                .unwrap_or_else(|| get_default_collection(&operation_conn));
            let parsed_graph_name = if region.is_some() {
                let parsed_region = region.as_ref().unwrap().parse::<Region>().unwrap();
                parsed_region.name().to_string()
            } else {
                graph.clone().unwrap()
            };
            let block_groups = Sample::get_block_groups(&conn, name, sample.as_deref());
            let formatted_sample_name = if sample.is_some() {
                format!("sample {}", sample.clone().unwrap())
            } else {
                "default sample".to_string()
            };
            let block_group = block_groups
                .iter()
                .find(|bg| bg.name == parsed_graph_name)
                .unwrap_or_else(|| {
                    panic!("Graph {parsed_graph_name} not found for {formatted_sample_name}")
                });
            let path = BlockGroup::get_current_path(&conn, block_group.id);
            let sequence = path.sequence(&conn);
            let start_coordinate;
            let mut end_coordinate;
            if region.is_some() {
                let parsed_region = region.as_ref().unwrap().parse::<Region>().unwrap();
                let interval = parsed_region.interval();
                start_coordinate = interval.start().unwrap().get() as i64;
                end_coordinate = interval.end().unwrap().get() as i64;
            } else {
                start_coordinate = start.unwrap_or(0);
                end_coordinate = end.unwrap_or(sequence.len() as i64);
            }
            println!(
                "{}",
                &sequence[start_coordinate as usize..end_coordinate as usize]
            );
        }
        Some(OldCommands::Diff {
            name,
            sample1,
            sample2,
            gfa,
        }) => {
            let name = &name
                .clone()
                .unwrap_or_else(|| get_default_collection(&operation_conn));
            gfa_sample_diff(
                &conn,
                name,
                &PathBuf::from(gfa),
                sample1.as_deref(),
                sample2.as_deref(),
            );
        }
        Some(OldCommands::DeriveSubgraph {
            name,
            sample,
            new_sample,
            region,
            backbone,
        }) => {
            conn.execute("BEGIN TRANSACTION", []).unwrap();
            operation_conn.execute("BEGIN TRANSACTION", []).unwrap();
            let name = &name
                .clone()
                .unwrap_or_else(|| get_default_collection(&operation_conn));
            let sample_name = sample.clone();
            let new_sample_name = new_sample.clone();
            let parsed_region = region.parse::<Region>().unwrap();
            let interval = parsed_region.interval();
            let start_coordinate = interval.start().unwrap().get() as i64;
            let end_coordinate = interval.end().unwrap().get() as i64;
            match derive_chunks(
                &conn,
                &operation_conn,
                name,
                sample_name.as_deref(),
                &new_sample_name,
                &parsed_region.name().to_string(),
                backbone.as_deref(),
                vec![Range {
                    start: start_coordinate,
                    end: end_coordinate,
                }],
            ) {
                Ok(_) => {}
                Err(e) => panic!("Error deriving subgraph: {e}"),
            }
            conn.execute("END TRANSACTION", []).unwrap();
            operation_conn.execute("END TRANSACTION", []).unwrap();
        }
        Some(OldCommands::DeriveChunks {
            name,
            sample,
            new_sample,
            region,
            backbone,
            breakpoints,
            chunk_size,
        }) => {
            conn.execute("BEGIN TRANSACTION", []).unwrap();
            operation_conn.execute("BEGIN TRANSACTION", []).unwrap();
            let name = &name
                .clone()
                .unwrap_or_else(|| get_default_collection(&operation_conn));
            let sample_name = sample.clone();
            let new_sample_name = new_sample.clone();
            let parsed_region = region.parse::<Region>().unwrap();
            let interval = parsed_region.interval();

            let path_length = match get_path(
                &conn,
                name,
                sample_name.as_deref(),
                &parsed_region.name().to_string(),
                backbone.as_deref(),
            ) {
                Ok(path) => path.length(&conn),
                Err(e) => panic!("Error deriving subgraph(s): {e}"),
            };

            let chunk_points;
            if let Some(breakpoints) = breakpoints {
                chunk_points = breakpoints
                    .split(",")
                    .map(|x| x.parse::<i64>().unwrap())
                    .sorted()
                    .collect::<Vec<i64>>();
            } else if let Some(chunk_size) = chunk_size {
                let chunk_count = path_length / chunk_size;
                chunk_points = (0..chunk_count)
                    .map(|i| i * chunk_size)
                    .collect::<Vec<i64>>();
            } else {
                panic!("No chunking method specified.");
            }

            if chunk_points.is_empty() {
                panic!("No chunk coordinates provided.");
            }
            if chunk_points[chunk_points.len() - 1] > path_length {
                panic!("At least one chunk coordinate exceeds path length.");
            }

            let mut range_start = 0;
            let mut chunk_ranges = vec![];
            for chunk_point in chunk_points {
                chunk_ranges.push(Range {
                    start: range_start,
                    end: chunk_point,
                });
                range_start = chunk_point;
            }
            chunk_ranges.push(Range {
                start: range_start,
                end: path_length,
            });

            match derive_chunks(
                &conn,
                &operation_conn,
                name,
                sample_name.as_deref(),
                &new_sample_name,
                &parsed_region.name().to_string(),
                backbone.as_deref(),
                chunk_ranges,
            ) {
                Ok(_) => {}
                Err(e) => panic!("Error deriving subgraph(s): {e}"),
            }
            conn.execute("END TRANSACTION", []).unwrap();
            operation_conn.execute("END TRANSACTION", []).unwrap();
        }
        Some(OldCommands::MakeStitch {
            name,
            sample,
            new_sample,
            regions,
            new_region,
        }) => {
            conn.execute("BEGIN TRANSACTION", []).unwrap();
            operation_conn.execute("BEGIN TRANSACTION", []).unwrap();
            let name = &name
                .clone()
                .unwrap_or_else(|| get_default_collection(&operation_conn));
            let sample_name = sample.clone();
            let new_sample_name = new_sample.clone();

            let region_names = regions.split(",").collect::<Vec<&str>>();

            match make_stitch(
                &conn,
                &operation_conn,
                name,
                sample_name.as_deref(),
                &new_sample_name,
                &region_names,
                new_region,
            ) {
                Ok(_) => {}
                Err(e) => panic!("Error stitching subgraphs: {e}"),
            }
            conn.execute("END TRANSACTION", []).unwrap();
            operation_conn.execute("END TRANSACTION", []).unwrap();
        }
        Some(OldCommands::Push {}) => match push(&operation_conn, &db_uuid) {
            Ok(_) => {
                println!("Push succeeded.");
            }
            Err(e) => {
                println!("Push failed: {e}");
            }
        },
        Some(OldCommands::Pull {}) => {}
    }
}
