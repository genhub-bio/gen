#![allow(warnings)]
use core::ops::Range;
use std::{
    fmt::Debug,
    fs::File,
    io,
    io::{BufReader, Write},
    ops::Deref,
    path::{Path, PathBuf},
    str,
};

use clap::{Parser, Subcommand};
use gen::{
    annotations::gff::propagate_gff,
    commands::{cli_context::CliContext, remote::handle_remote_command, Cli, Commands},
    config,
    diffs::gfa::gfa_sample_diff,
    get_connection, get_operation_connection,
    graph_operators::{derive_chunks, get_path, make_stitch},
    operation_management,
    operation_management::{parse_patch_operations, push},
    patch, track_database, translate,
    updates::gaf::transform_csv_to_fasta,
    views::{block_group::view_block_group, operations::view_operations, patch::view_patches},
};
use gen_core::config::{get_gen_dir, get_or_create_gen_dir};
use gen_models::{
    block_group::BlockGroup,
    errors::{OperationError, RemoteError},
    file_types::FileTypes,
    metadata,
    operations::{
        setup_db, Branch, Defaults, Operation, OperationFile, OperationInfo, OperationState,
    },
    sample::Sample,
    traits::Query,
};
use itertools::Itertools;
use noodles::core::Region;
use r#gen::graph_operators::GraphOperationError;
use rusqlite::{params, types::Value, Connection};
use sha2::digest::typenum::Gr;

fn get_default_collection(conn: &Connection) -> String {
    let mut stmt = conn
        .prepare("select collection_name from defaults where id = 1")
        .unwrap();
    stmt.query_row((), |row| row.get(0))
        .unwrap_or("default".to_string())
}

fn main() {
    // Start logger (gets log level from RUST_LOG environment variable, sends output to stderr)
    env_logger::init();

    let cli = Cli::parse();
    let cli_context = CliContext::from(&cli);

    // commands not requiring a db connection are handled here
    if let Some(Commands::Init {}) = &cli.command {
        get_or_create_gen_dir();
        println!("Gen repository initialized.");
        return;
    }

    let operation_conn = get_operation_connection(None).unwrap();
    if let Some(Commands::Defaults {
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

    if let Some(Commands::Transform { format_csv_for_gaf }) = &cli.command {
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
    let conn = get_connection(db).unwrap();
    let db_uuid = metadata::get_db_uuid(&conn);

    match track_database(&conn, &operation_conn) {
        Ok(_) => {}
        Err(err) => {
            panic!("Error tracking database: {err}");
        }
    };
    setup_db(&operation_conn);

    match cli.command {
        Some(Commands::Init {}) => {
            get_or_create_gen_dir();
            println!("Gen repository initialized.");
        }
        Some(Commands::Import(cmd)) => {
            gen::commands::import::execute(&cli_context, cmd);
        }
        Some(Commands::Update(cmd)) => {
            gen::commands::update::execute(&cli_context, cmd);
        }
        Some(Commands::Export(cmd)) => {
            gen::commands::export::execute(&cli_context, cmd);
        }
        Some(Commands::Remote(cmd)) => match handle_remote_command(&operation_conn, &cmd) {
            Ok(_) => {}
            Err(err) => {
                eprintln!("Remote command failed: {err}");
                std::process::exit(1);
            }
        },

        Some(Commands::View {
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
        Some(Commands::Translate {
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
        Some(Commands::Operations {
            interactive,
            branch,
        }) => {
            let current_op = OperationState::get_operation(&operation_conn);
            if let Some(current_op) = current_op {
                let branch_name = branch.clone().unwrap_or_else(|| {
                    let current_branch_id = OperationState::get_current_branch(&operation_conn)
                        .expect("No current branch is set.");
                    Branch::get_by_id(&operation_conn, current_branch_id)
                        .unwrap_or_else(|| panic!("No branch with id {current_branch_id}"))
                        .name
                });
                let operations = Branch::get_operations(
                    &operation_conn,
                    Branch::get_by_name(&operation_conn, &branch_name)
                        .unwrap_or_else(|| panic!("No branch named {branch_name}."))
                        .id,
                );
                if interactive {
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
                    &operation_conn,
                    &branch_name
                        .clone()
                        .expect("Must provide a branch name to create."),
                );
            } else if delete {
                let branch = Branch::get_by_name(
                    &operation_conn,
                    &branch_name
                        .clone()
                        .expect("Must provide a branch name to delete."),
                )
                .unwrap_or_else(|| panic!("Unable to find branch {branch_name:?}."));
                Branch::delete(&operation_conn, branch.id);
            } else if checkout {
                operation_management::checkout(
                    None,
                    &operation_conn,
                    &Some(
                        branch_name
                            .clone()
                            .expect("Must provide a branch name to checkout.")
                            .to_string(),
                    ),
                    None,
                );
            } else if list {
                let current_branch = OperationState::get_current_branch(&operation_conn);
                let mut indicator = "";
                println!(
                    "{indicator:<3}{col1:<30}   {col2:<20}   {col3:<15}",
                    col1 = "Name",
                    col2 = "Operation",
                    col3 = "Remote",
                );
                for branch in
                    Branch::query(&operation_conn, "select * from branch", params![]).iter()
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
                let branch_name = branch_name.clone().expect("Branch name must be provided.");
                let other_branch = Branch::get_by_name(&operation_conn, &branch_name)
                    .unwrap_or_else(|| panic!("Unable to find branch {branch_name}."));
                let current_branch = OperationState::get_current_branch(&operation_conn)
                    .expect("Unable to find current branch.");
                match operation_management::merge(
                    None,
                    &operation_conn,
                    current_branch,
                    other_branch.id,
                    None,
                ) {
                    Ok(_) => println!("Merge successful"),
                    Err(_) => {
                        panic!("Merge failed.");
                    }
                }
            } else if let Some(remote_name) = set_remote {
                // Handle setting remote for current branch
                let current_branch_id = OperationState::get_current_branch(&operation_conn)
                    .expect("No current branch is checked out.");

                let remote_to_set = if remote_name.is_empty() || remote_name == "null" {
                    None
                } else {
                    Some(remote_name.as_str())
                };

                // Use the validated method for setting remote
                match Branch::set_remote_validated(
                    &operation_conn,
                    current_branch_id,
                    remote_to_set,
                ) {
                    Ok(_) => {
                        if remote_to_set.is_some() {
                            println!("Remote '{remote_name}' associated with current branch");
                        } else {
                            println!("Remote association cleared for current branch");
                        }
                    }
                    Err(err) => {
                        eprintln!("Error: {err}");
                        std::process::exit(1);
                    }
                }
            } else {
                println!("No options selected.");
            }
        }
        Some(Commands::Merge { branch_name }) => {
            let branch_name = branch_name.clone().expect("Branch name must be provided.");
            let other_branch = Branch::get_by_name(&operation_conn, &branch_name)
                .unwrap_or_else(|| panic!("Unable to find branch {branch_name}."));
            let current_branch = OperationState::get_current_branch(&operation_conn)
                .expect("Unable to find current branch.");
            match operation_management::merge(
                None,
                &operation_conn,
                current_branch,
                other_branch.id,
                None,
            ) {
                Ok(_) => println!("Merge successful"),
                Err(details) => {
                    panic!("Merge failed: {details}");
                }
            }
        }
        Some(Commands::Apply { hash }) => {
            let operation = match Operation::search_hash(&operation_conn, &hash) {
                Ok(op) => op,
                Err(e) => {
                    panic!("Unable to find operation by hash {hash}");
                }
            };
            match operation_management::apply(None, &operation_conn, &operation.hash, None) {
                Ok(_) => println!("Operation applied"),
                Err(_) => {
                    panic!("Apply failed.");
                }
            }
        }
        Some(Commands::Checkout { branch, hash }) => {
            if let Some(name) = branch.clone() {
                if Branch::get_by_name(&operation_conn, &name).is_none() {
                    Branch::get_or_create(&operation_conn, &name);
                    println!("Created branch {name}");
                }
                println!("Checking out branch {name}");
                operation_management::checkout(None, &operation_conn, &Some(name), None);
            } else if let Some(hash_name) = hash.clone() {
                // if the hash is a branch, check it out
                if Branch::get_by_name(&operation_conn, &hash_name).is_some() {
                    println!("Checking out branch {hash_name}");
                    operation_management::checkout(None, &operation_conn, &Some(hash_name), None);
                } else {
                    let operation = match Operation::search_hash(&operation_conn, &hash_name) {
                        Ok(op) => op,
                        Err(err) => {
                            panic!("Unable to find hash {hash_name}.")
                        }
                    };
                    println!("Checking out operation {hash_name}");
                    operation_management::checkout(
                        None,
                        &operation_conn,
                        &None,
                        Some(operation.hash),
                    );
                }
            } else {
                println!("No branch or hash to checkout provided.");
            }
        }
        Some(Commands::Reset { hash }) => {
            let operation = match Operation::search_hash(&operation_conn, &hash) {
                Ok(op) => op,
                Err(err) => {
                    panic!("Unable to find hash {hash}.")
                }
            };
            operation_management::reset(None, &operation_conn, &operation.hash);
        }
        Some(Commands::PatchCreate {
            name,
            operation,
            branch,
        }) => {
            let branch = if let Some(branch_name) = branch {
                Branch::get_by_name(&operation_conn, &branch_name)
                    .unwrap_or_else(|| panic!("No branch with name {branch_name} found."))
            } else {
                let current_branch_id = OperationState::get_current_branch(&operation_conn)
                    .expect("No current branch is checked out.");
                Branch::get_by_id(&operation_conn, current_branch_id).unwrap()
            };
            let branch_ops = Branch::get_operations(&operation_conn, branch.id);
            let operations = parse_patch_operations(
                &branch_ops,
                &branch.current_operation_hash.unwrap(),
                &operation,
            );
            let mut f = File::create(format!("{name}.gz")).unwrap();
            patch::create_patch(&operation_conn, &operations, &mut f);
        }
        Some(Commands::PatchApply { patch }) => {
            let mut f = File::open(patch).unwrap();
            let patches = patch::load_patches(&mut f);
            patch::apply_patches(None, &operation_conn, &patches)
                .unwrap_or_else(|op| panic!("Failed to apply patch: {op:?}"));
        }
        Some(Commands::PatchView { prefix, patch }) => {
            let patch_path = Path::new(&patch);
            let mut f = File::open(patch_path).unwrap();
            let patches = patch::load_patches(&mut f);
            let diagrams = view_patches(&patches);
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
        Some(Commands::Init {}) => {
            get_or_create_gen_dir();
            println!("Gen repository initialized.");
        }
        Some(Commands::Defaults {
            database,
            collection,
        }) => {}

        Some(Commands::Transform { format_csv_for_gaf }) => {}
        Some(Commands::PropagateAnnotations {
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
                &to_sample,
                &gff,
                &output_gff,
            );

            conn.execute("END TRANSACTION", []).unwrap();
            operation_conn.execute("END TRANSACTION", []).unwrap();
        }
        Some(Commands::ListSamples {}) => {
            let sample_names = Sample::get_all_names(&conn);
            // Null sample
            println!();
            for sample_name in sample_names {
                println!("{sample_name}");
            }
        }
        Some(Commands::ListGraphs { name, sample }) => {
            let name = &name
                .clone()
                .unwrap_or_else(|| get_default_collection(&operation_conn));
            let block_groups = Sample::get_block_groups(&conn, name, sample.as_deref());
            for block_group in block_groups {
                println!("{}", block_group.name);
            }
        }
        Some(Commands::GetSequence {
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
            let path = BlockGroup::get_current_path(&conn, &block_group.id);
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
        Some(Commands::Diff {
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
        Some(Commands::DeriveSubgraph {
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
        Some(Commands::DeriveChunks {
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
        Some(Commands::MakeStitch {
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
                &new_region,
            ) {
                Ok(_) => {}
                Err(GraphOperationError::OperationError(OperationError::NoChanges)) => {}
                Err(e) => panic!("Error stitching subgraphs: {e}"),
            }
            conn.execute("END TRANSACTION", []).unwrap();
            operation_conn.execute("END TRANSACTION", []).unwrap();
        }
        Some(Commands::Push { remote }) => match push(&operation_conn, remote.as_deref()) {
            Ok(_) => {
                println!("Push succeeded.");
            }
            Err(e) => {
                println!("Push failed: {e}");
            }
        },
        Some(Commands::Pull { remote }) => {}
    }
}
