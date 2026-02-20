#![allow(warnings)]
use std::{
    fmt::Debug,
    fs::File,
    io,
    io::{BufReader, Write},
    ops::Deref,
    panic,
    path::{Path, PathBuf},
    str,
};

use anyhow::anyhow;
use clap::{Parser, Subcommand};
use r#gen::{
    annotations::gff::propagate_gff,
    commands::{
        Cli, Commands,
        cli_context::CliContext,
        graph_operations::{
            derive_chunks::derive_chunks_operation, derive_subgraph::derive_subgraph_operation,
            make_stitch::make_stitch_operation,
        },
        remote::handle_remote_command,
    },
    config,
    diffs::gfa::gfa_sample_diff,
    get_connection, get_operation_connection,
    operation_management,
    operation_management::{parse_patch_operations, pull, push},
    patch, track_database,
    updates::gaf::transform_csv_to_fasta,
    views::{
        block_group::view_block_group, block_group_inline::show_inline_gen_graph_widget,
        diff::view_diff, operations::view_operations, patch::view_patches,
    },
};
use gen_annotations::translate;
use gen_core::config::Workspace;
use gen_diff::operations::collect_operation_diff;
use gen_models::{
    annotations::{add_annotation, add_annotation_file},
    block_group::BlockGroup,
    db::{DbContext, OperationsConnection},
    errors::RemoteError,
    file_types::FileTypes,
    metadata,
    operations::{
        Branch, Defaults, Operation, OperationFile, OperationInfo, OperationState, parse_hash,
    },
    sample::Sample,
    traits::Query,
};
use noodles::core::Region;
use rusqlite::{Connection, params, types::Value};
use sha2::digest::typenum::Gr;

fn get_default_collection(conn: &OperationsConnection) -> Result<String, rusqlite::Error> {
    let mut stmt = conn.prepare("select collection_name from defaults where id = 1")?;
    Ok(stmt
        .query_row((), |row| row.get(0))
        .unwrap_or("default".to_string()))
}

fn call_cli() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let workspace = Workspace::from_current_dir();

    // commands not requiring a db connection are handled here
    if let Some(Commands::Init {}) = &cli.command {
        workspace.ensure_gen_dir();
        get_operation_connection(Some(workspace.gen_db_path()?))?;
        println!("Gen repository initialized.");
        return Ok(());
    }

    let operation_conn = get_operation_connection(Some(workspace.gen_db_path()?))?;
    if let Some(Commands::Defaults {
        database,
        collection,
    }) = &cli.command
    {
        if let Some(name) = database {
            operation_conn.execute("update defaults set db_name=?1 where id = 1", (name,))?;
            println!("Default database set to {name}");
        }
        if let Some(name) = collection {
            operation_conn.execute(
                "update defaults set collection_name=?1 where id = 1",
                (name,),
            )?;
            println!("Default collection set to {name}");
        }
        return Ok(());
    }

    if let Some(Commands::Transform { format_csv_for_gaf }) = &cli.command {
        let csv = format_csv_for_gaf
            .clone()
            .expect("csv for transformation not provided.");
        let stdout = io::stdout();
        let mut handle = stdout.lock();
        let mut csv_file = File::open(csv)?;
        return match transform_csv_to_fasta(&mut csv_file, &mut handle) {
            Ok(_) => Ok(()),
            Err(err) => {
                eprintln!("Failed to transform CSV for GAF usage: {err}");
                Err(err.into())
            }
        };
    }
    let binding = match cli.db {
        Some(db) => db,
        None => {
            let mut stmt = operation_conn.prepare("select db_name from defaults where id = 1;")?;
            let row: Option<String> = stmt.query_row((), |row| row.get(0)).ok();

            match row {
                Some(db_name) => db_name,
                None => match workspace.find_gen_dir() {
                    Some(dir) => dir
                        .join("default.db")
                        .to_str()
                        .ok_or("Invalid path encoding")?
                        .to_string(),
                    None => {
                        return Err("No .gen directory found. Please run 'gen init' first.".into());
                    }
                },
            }
        }
    };
    let db = binding.as_str();
    let graph_connection = get_connection(db)?;
    let db_context = DbContext::new(workspace.clone(), graph_connection, operation_conn);
    let operation_conn = db_context.operations().conn();
    let graph_conn = db_context.graph().conn();
    let db_uuid = metadata::get_db_uuid(graph_conn);
    let cli_context = CliContext {
        context: &db_context,
    };

    match track_database(graph_conn, operation_conn) {
        Ok(_) => {}
        Err(err) => {
            panic!("Error tracking database: {err}");
        }
    };

    match cli.command {
        Some(Commands::Init {}) => {
            workspace.ensure_gen_dir();
            println!("Gen repository initialized.");
            Ok(())
        }
        Some(Commands::Import(cmd)) => Ok(r#gen::commands::import::execute(&cli_context, cmd)?),
        Some(Commands::Update(cmd)) => Ok(r#gen::commands::update::execute(&cli_context, cmd)?),
        Some(Commands::Export(cmd)) => Ok(r#gen::commands::export::execute(&cli_context, cmd)?),
        Some(Commands::Remote(cmd)) => Ok(handle_remote_command(operation_conn, &cmd)?),
        Some(Commands::View {
            graph,
            sample,
            collection,
            position,
            full,
        }) => {
            let collection_name = &(match collection {
                Some(collection) => collection,
                None => get_default_collection(operation_conn)?,
            });

            if !full && let Some(name) = graph.as_ref() {
                // Use the inline widget by default if a graph is specified
                let block_group = if let Some(ref sample_name) = sample {
                    BlockGroup::get(
                        graph_conn,
                        "select * from block_groups where collection_name = ?1 AND sample_name = ?2 AND name = ?3",
                        params![collection_name, sample_name, name],
                    )
                } else {
                    BlockGroup::get(
                        graph_conn,
                        "select * from block_groups where collection_name = ?1 AND sample_name is null AND name = ?2",
                        params![collection_name, name],
                    )
                };

                match block_group {
                    Ok(bg) => {
                        let block_graph = BlockGroup::get_graph(graph_conn, &bg.id);
                        let current_path = BlockGroup::get_current_path(graph_conn, &bg.id);
                        // Use a default height of 10 for now
                        match show_inline_gen_graph_widget(
                            graph_conn,
                            &block_graph,
                            vec![current_path],
                            10,
                        ) {
                            Ok(true) => {
                                // User requested upgrade to full TUI
                                view_block_group(
                                    graph_conn,
                                    operation_conn,
                                    &workspace,
                                    graph,
                                    sample,
                                    collection_name,
                                    position,
                                )?;
                            }
                            Ok(false) => {}
                            Err(e) => {
                                eprintln!("Error showing inline widget: {}", e);
                            }
                        }
                    }
                    Err(_) => {
                        eprintln!(
                            "No block group found with name {:?} and sample {:?} in collection {}",
                            name,
                            sample.clone().unwrap_or_else(|| "null".to_string()),
                            collection_name
                        );
                    }
                }
            } else {
                // Use the full-screen viewer if --full is specified or no graph is provided
                view_block_group(
                    graph_conn,
                    operation_conn,
                    &workspace,
                    graph,
                    sample,
                    collection_name,
                    position,
                )?;
            }
            Ok(())
        }
        Some(Commands::ViewDiff { from, to }) => {
            let to_ref = to.clone().unwrap_or_else(|| {
                OperationState::get_operation(operation_conn)
                    .map(|h| format!("{h}"))
                    .unwrap_or_else(|| "HEAD".to_string())
            });
            let from_range = parse_hash(operation_conn, &from)?;
            let from_hash = from_range
                .from
                .or(from_range.to)
                .ok_or_else(|| anyhow!("No operation resolved for {from}"))?;
            let to_range = parse_hash(operation_conn, &to_ref)?;
            let to_hash = to_range
                .to
                .or(to_range.from)
                .ok_or_else(|| anyhow!("No operation resolved for {to_ref}"))?;
            let diffs =
                collect_operation_diff(&workspace, operation_conn, Some(from_hash), to_hash, None)?;
            if diffs.is_empty() {
                println!("No differences found between {from} and {to_ref}.");
            } else {
                view_diff(graph_conn, &diffs)?;
            }
            Ok(())
        }
        Some(Commands::Translate {
            bed,
            gff,
            collection,
            sample,
        }) => {
            let collection_name = &(match collection {
                Some(collection) => collection,
                None => get_default_collection(operation_conn)?,
            });

            if let Some(bed) = bed {
                let stdout = io::stdout();
                let mut handle = stdout.lock();
                let mut bed_file = File::open(bed)?;
                Ok(translate::bed::translate_bed(
                    graph_conn,
                    collection_name,
                    sample.as_deref(),
                    &mut bed_file,
                    &mut handle,
                )?)
            } else if let Some(gff) = gff {
                let stdout = io::stdout();
                let mut handle = stdout.lock();
                let mut gff_file = BufReader::new(File::open(gff)?);
                Ok(translate::gff::translate_gff(
                    graph_conn,
                    collection_name,
                    sample.as_deref(),
                    &mut gff_file,
                    &mut handle,
                )?)
            } else {
                Err("No input file specified.".into())
            }
        }
        Some(Commands::Operations {
            interactive,
            branch,
        }) => {
            let current_op = OperationState::get_operation(operation_conn);
            if let Some(current_op) = current_op {
                let branch_name = if let Some(name) = branch.clone() {
                    name
                } else {
                    let current_branch_id = OperationState::get_current_branch(operation_conn)
                        .ok_or("No current branch is set.")?;
                    Branch::get_by_id(operation_conn, current_branch_id)
                        .ok_or_else(|| format!("No branch with id {current_branch_id}"))?
                        .name
                };

                let operations = Branch::get_operations(
                    operation_conn,
                    Branch::get_by_name(operation_conn, &branch_name)
                        .ok_or_else(|| format!("No branch named {branch_name}."))?
                        .id,
                );
                if interactive {
                    return Ok(view_operations(&db_context, &operations)?);
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
            Ok(())
        }
        Some(Commands::Branch {
            create,
            delete,
            checkout,
            list,
            merge,
            set_remote,
            branch_name,
        }) => {
            if create {
                Branch::get_or_create(
                    operation_conn,
                    branch_name
                        .as_ref()
                        .ok_or("Must provide a branch name to create.")?,
                );
            } else if delete {
                let branch = Branch::get_by_name(
                    operation_conn,
                    branch_name
                        .as_ref()
                        .ok_or("Must provide a branch name to delete.")?,
                )
                .ok_or_else(|| format!("Unable to find branch {branch_name:?}."))?;
                Branch::delete(operation_conn, branch.id);
            } else if checkout {
                operation_management::checkout(
                    &db_context,
                    &Some(
                        branch_name
                            .clone()
                            .ok_or("Must provide a branch name to checkout.")?
                            .to_string(),
                    ),
                    None,
                )?;
            } else if list {
                let current_branch = OperationState::get_current_branch(operation_conn);
                let mut indicator = "";
                println!(
                    "{indicator:<3}{col1:<30}   {col2:<20}   {col3:<15}",
                    col1 = "Name",
                    col2 = "Operation",
                    col3 = "Remote",
                );
                for branch in
                    Branch::query(operation_conn, "select * from branch", params![]).iter()
                {
                    if let Some(current_branch_id) = current_branch {
                        if current_branch_id == branch.id {
                            indicator = ">";
                        } else {
                            indicator = "";
                        }
                    }
                    let remote_display = branch
                        .remote_name
                        .clone()
                        .unwrap_or_else(|| "none".to_string());
                    println!(
                        "{indicator:<3}{col1:<30}   {col2:<20}   {col3:<15}",
                        col1 = branch.name,
                        col2 = branch
                            .current_operation_hash
                            .map(|h| format!("{h}"))
                            .unwrap_or_default(),
                        col3 = remote_display
                    );
                }
            } else if merge {
                let branch_name = branch_name.clone().ok_or("Branch name must be provided.")?;
                let other_branch = Branch::get_by_name(operation_conn, &branch_name)
                    .ok_or_else(|| format!("Unable to find branch {branch_name}."))?;
                let current_branch = OperationState::get_current_branch(operation_conn)
                    .ok_or("Unable to find current branch.")?;
                operation_management::merge(&db_context, current_branch, other_branch.id, None)?;
                println!("Merge successful");
            } else if let Some(remote_name) = set_remote {
                let current_branch_id = OperationState::get_current_branch(operation_conn)
                    .ok_or("No current branch is checked out.")?;

                let remote_to_set = if remote_name.is_empty() || remote_name == "null" {
                    None
                } else {
                    Some(remote_name.as_str())
                };

                Branch::set_remote_validated(operation_conn, current_branch_id, remote_to_set)?;

                if remote_to_set.is_some() {
                    println!("Remote '{remote_name}' associated with current branch");
                } else {
                    println!("Remote association cleared for current branch");
                }
            } else {
                println!("No options selected.");
            }
            Ok(())
        }
        Some(Commands::Merge { branch_name }) => {
            let branch_name = branch_name.clone().ok_or("Branch name must be provided.")?;
            let other_branch = Branch::get_by_name(operation_conn, &branch_name)
                .ok_or_else(|| format!("Unable to find branch {branch_name}."))?;
            let current_branch = OperationState::get_current_branch(operation_conn)
                .ok_or("Unable to find current branch.")?;
            match operation_management::merge(&db_context, current_branch, other_branch.id, None) {
                Ok(_) => {
                    println!("Merge successful");
                    Ok(())
                }
                Err(e) => Err(format!("Merge failed: {}", e).into()),
            }
        }
        Some(Commands::Apply { hash }) => {
            let operation = Operation::search_hash(operation_conn, &hash)?;
            match operation_management::apply(&db_context, &operation.hash, None, true) {
                Ok(_) => {
                    println!("Operation applied");
                    Ok(())
                }
                Err(e) => Err(format!("Operation application failed: {}", e).into()),
            }
        }
        Some(Commands::Checkout { branch, hash }) => {
            if let Some(name) = branch.clone() {
                if Branch::get_by_name(operation_conn, &name).is_none() {
                    Branch::get_or_create(operation_conn, &name);
                    println!("Created branch {name}");
                }
                println!("Checking out branch {name}");
                operation_management::checkout(&db_context, &Some(name), None)?;
            } else if let Some(hash_name) = hash.clone() {
                if Branch::get_by_name(operation_conn, &hash_name).is_some() {
                    println!("Checking out branch {hash_name}");
                    operation_management::checkout(&db_context, &Some(hash_name), None)?;
                } else {
                    let operation = Operation::search_hash(operation_conn, &hash_name)?;
                    println!("Checking out operation {hash_name}");
                    operation_management::checkout(&db_context, &None, Some(operation.hash))?;
                }
            } else {
                println!("No branch or hash to checkout provided.");
            }
            Ok(())
        }
        Some(Commands::Reset { hash }) => {
            let operation = Operation::search_hash(operation_conn, &hash)?;
            match operation_management::reset(&db_context, &operation.hash) {
                Ok(_) => {
                    println!("Operation reset");
                    Ok(())
                }
                Err(e) => Err(format!("Operation reset failed: {}", e).into()),
            }
        }
        Some(Commands::PatchCreate {
            name,
            operation,
            branch,
        }) => {
            let branch = if let Some(branch_name) = branch {
                Branch::get_by_name(operation_conn, &branch_name)
                    .ok_or_else(|| format!("No branch with name {branch_name} found."))?
            } else {
                let current_branch_id = OperationState::get_current_branch(operation_conn)
                    .ok_or("No current branch is checked out.")?;
                Branch::get_by_id(operation_conn, current_branch_id)
                    .ok_or("Unable to get current branch")?
            };
            let branch_ops = Branch::get_operations(operation_conn, branch.id);
            let operations = parse_patch_operations(operation_conn, &branch_ops, &operation)?;
            let mut f = File::create(format!("{name}.gz"))?;
            patch::create_patch(&db_context, &operations, &mut f)?;
            Ok(())
        }
        Some(Commands::PatchApply { patch }) => {
            let mut f = File::open(patch)?;
            let patches = patch::load_patches(&mut f);
            match patch::apply_patches(&db_context, &patches) {
                Ok(_) => {
                    println!("Patch applied");
                    Ok(())
                }
                Err(e) => Err(format!("Patch application failed: {}", e).into()),
            }
        }
        Some(Commands::PatchView { prefix, patch }) => {
            let patch_path = Path::new(&patch);
            let mut f = File::open(patch_path)?;
            let patches = patch::load_patches(&mut f);
            let diagrams = view_patches(&workspace, &patches);
            for (patch_hash, patch_diagrams) in diagrams.iter() {
                for (bg_id, dot) in patch_diagrams.iter() {
                    let path = if let Some(ref p) = prefix {
                        format!("{p}_{patch_hash:.7}_{bg_id}.dot")
                    } else {
                        format!(
                            "{patch_base}_{patch_hash:.7}_{bg_id}.dot",
                            patch_base = patch_path
                                .with_extension("")
                                .file_name()
                                .ok_or("Invalid patch path")?
                                .to_str()
                                .ok_or("Invalid UTF-8 in path")?
                        )
                    };
                    let mut f = File::create(path)?;
                    f.write_all(dot.as_bytes())?;
                }
            }
            Ok(())
        }
        None => Ok(()),
        Some(Commands::Defaults {
            database,
            collection,
        }) => Ok(()),
        Some(Commands::Transform { format_csv_for_gaf }) => Ok(()),
        Some(Commands::PropagateAnnotations {
            name,
            from_sample,
            to_sample,
            gff,
            output_gff,
        }) => {
            let collection_name = &(match name {
                Some(collection) => collection,
                None => get_default_collection(operation_conn)?,
            });
            let from_sample_name = from_sample.clone();

            graph_conn.execute("BEGIN TRANSACTION", [])?;
            operation_conn.execute("BEGIN TRANSACTION", [])?;

            propagate_gff(
                graph_conn,
                collection_name,
                from_sample_name.as_deref(),
                &to_sample,
                &gff,
                &output_gff,
            )?;

            graph_conn.execute("END TRANSACTION", [])?;
            operation_conn.execute("END TRANSACTION", [])?;
            Ok(())
        }
        Some(Commands::AddAnnotation {
            name,
            group,
            sample,
            region,
        }) => {
            let collection_name = get_default_collection(operation_conn)?;
            let operation = add_annotation(
                &db_context,
                &collection_name,
                &name,
                group.as_deref(),
                sample.as_deref(),
                &region,
            )?;
            println!("Annotation {name} added in operation {}", operation.hash);
            Ok(())
        }
        Some(Commands::AddAnnotationFile {
            path,
            format,
            index,
            name,
            message,
        }) => {
            let operation = add_annotation_file(
                &db_context,
                &path,
                format.as_deref(),
                index.as_deref(),
                name.as_deref(),
                message.as_deref(),
            )?;
            println!("Annotation file added in operation {}", operation.hash);
            Ok(())
        }
        Some(Commands::ListSamples {}) => {
            let sample_names = Sample::get_all_names(graph_conn);
            println!();
            for sample_name in sample_names {
                println!("{sample_name}");
            }
            Ok(())
        }
        Some(Commands::ListGraphs { name, sample }) => {
            let collection_name = &(match name {
                Some(collection) => collection,
                None => get_default_collection(operation_conn)?,
            });
            let block_groups =
                Sample::get_block_groups(graph_conn, collection_name, sample.as_deref());
            for block_group in block_groups {
                println!("{}", block_group.name);
            }
            Ok(())
        }
        Some(Commands::GetSequence {
            name,
            sample,
            graph,
            start,
            end,
            region,
        }) => {
            let collection_name = &(match name {
                Some(collection) => collection,
                None => get_default_collection(operation_conn)?,
            });
            let block_groups =
                Sample::get_block_groups(graph_conn, collection_name, sample.as_deref());

            let formatted_sample_name = match sample {
                Some(s) => format!("sample {s}"),
                None => "default sample".to_string(),
            };

            let (parsed_graph_name, start_coordinate, mut end_coordinate) =
                if let Some(region) = region {
                    let parsed_region = region.parse::<Region>()?;
                    let interval = parsed_region.interval();
                    (
                        parsed_region.name().to_string(),
                        interval.start().ok_or("Region missing start")?.get() as i64,
                        interval.end().ok_or("Region missing end")?.get() as i64,
                    )
                } else {
                    (
                        graph.clone().ok_or("Graph name required")?,
                        start.unwrap_or_default(),
                        end.unwrap_or(-1),
                    )
                };

            let block_group = block_groups
                .iter()
                .find(|bg| bg.name == parsed_graph_name)
                .ok_or_else(|| {
                    format!("Graph {parsed_graph_name} not found for {formatted_sample_name}")
                })?;
            let path = BlockGroup::get_current_path(graph_conn, &block_group.id);
            let sequence = path.sequence(graph_conn);
            if end_coordinate == -1 {
                end_coordinate = sequence.len() as i64;
            }
            println!(
                "{}",
                &sequence[start_coordinate as usize..end_coordinate as usize]
            );
            Ok(())
        }
        Some(Commands::Diff {
            name,
            sample1,
            sample2,
            gfa,
        }) => {
            let collection_name = &(match name {
                Some(collection) => collection,
                None => get_default_collection(operation_conn)?,
            });
            gfa_sample_diff(
                graph_conn,
                collection_name,
                &PathBuf::from(gfa),
                sample1.as_deref(),
                sample2.as_deref(),
            )?;
            Ok(())
        }
        Some(Commands::DeriveSubgraph {
            name,
            sample,
            new_sample,
            region,
            backbone,
        }) => {
            match derive_subgraph_operation(&db_context, name, sample, new_sample, region, backbone)
            {
                Ok(_) => Ok(()),
                Err(e) => Err(format!("Error deriving subgraph: {e}").into()),
            }
        }
        Some(Commands::DeriveChunks {
            name,
            sample,
            new_sample,
            region,
            backbone,
            breakpoints,
            chunk_size,
        }) => {
            match derive_chunks_operation(
                &db_context,
                name,
                sample,
                new_sample,
                region,
                backbone,
                breakpoints,
                chunk_size,
            ) {
                Ok(_) => Ok(()),
                Err(e) => Err(format!("Error deriving chunks: {e}").into()),
            }
        }
        Some(Commands::MakeStitch {
            name,
            sample,
            new_sample,
            regions,
            new_region,
        }) => {
            match make_stitch_operation(&db_context, name, sample, new_sample, regions, new_region)
            {
                Ok(_) => Ok(()),
                Err(e) => Err(format!("Error making a stitch: {e}").into()),
            }
        }
        Some(Commands::Push { remote }) => {
            push(&db_context, remote.as_deref())?;
            println!("Push succeeded.");
            Ok(())
        }
        Some(Commands::Pull { remote }) => {
            pull(&db_context, remote.as_deref())?;
            Ok(())
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    panic::set_hook(Box::new(|info| {
        eprintln!("❗ The application has encountered an unexpected error and must exit.");
        eprintln!("Message: {}", info);
        eprintln!();
        eprintln!("👉 Please file an issue at: https://github.com/genhub-bio/gen/issues");
        eprintln!("   Include the full output above, what you were doing, and system info.");
    }));

    // Start logger (gets log level from RUST_LOG environment variable, sends output to stderr)
    env_logger::init();
    match call_cli() {
        Ok(_) => Ok(()),
        Err(e) => Err(anyhow!("{e}").into()),
    }
}
