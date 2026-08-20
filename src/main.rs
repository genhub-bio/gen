#![allow(warnings)]
use std::{fmt::Debug, fs::File, io, io::BufReader, ops::Deref, path::PathBuf, str};

use anyhow::anyhow;
use clap::{Parser, Subcommand};
use crossterm::terminal;
#[cfg(feature = "profiling")]
use r#gen::profiling::{Profiler, SamplingProfiler};
use r#gen::{
    annotations::gff::propagate_gff,
    commands::{
        Cli, Commands, cache, cli_context::CliContext, commit_operation,
        graph_operations::make_stitch::make_stitch_operation, parse_diff_revisions,
        remote::handle_remote_command,
    },
    diffs::gfa::gfa_sample_diff,
    get_config_connection, get_connection_for_branch, get_raw_connection,
    graphs::graph_search::{GenGraphMatcher, SeedIndex},
    history::{ensure_clean_working_set, history_action_error, operations_history_entries},
    patch,
    theme::init_theme,
    updates::gaf::transform_csv_to_fasta,
    views::{
        block_group::view_block_group, block_group_inline::show_inline_block_group_widget,
        diff::view_diff, operations::view_operations, patch::view_patch,
        tui_runtime::install_global_panic_hook,
    },
};
use gen_annotations::translate;
use gen_core::{BranchName, CommitRef, config::Workspace, range::Range, region::Region};
use gen_diff::operations::collect_operation_diff;
use gen_models::{
    annotations::{AnnotationFileChecksumOverrides, add_annotation, add_annotation_file},
    block_group::BlockGroup,
    collection::Collection,
    db::{ConfigConnection, DbContext, GraphConnection},
    errors::{OperationError, RemoteError},
    history::{
        HistoryStore,
        dolt::{DoltHistoryStore, branch_rows},
    },
    operations::{Defaults, OperationFile, RemoteBranch, add_files_operation},
    reference_alias::ReferenceAlias,
    sample::Sample,
    traits::Query,
};
use rusqlite::{params, types::Value};
use sha2::digest::typenum::Gr;

fn get_default_collection(conn: &ConfigConnection) -> Result<String, rusqlite::Error> {
    let mut stmt = conn.prepare("select collection_name from defaults where id = 1")?;
    Ok(stmt
        .query_row((), |row| row.get(0))
        .unwrap_or("default".to_string()))
}

fn resolve_initial_collection(
    graph_conn: &GraphConnection,
    config_conn: &ConfigConnection,
    requested_collection: Option<String>,
    history_ref: Option<&str>,
) -> Result<String, rusqlite::Error> {
    if let Some(collection) = requested_collection {
        return Ok(collection);
    }

    let collections = Collection::all(graph_conn, history_ref);
    if let [collection] = collections.as_slice() {
        return Ok(collection.name.clone());
    }

    get_default_collection(config_conn)
}

/// Clamp a requested inline view height to a usable range: at least enough rows for the
/// top/bottom border, the graph, and the footer, and no taller than the terminal itself.
fn clamp_inline_view_height(requested_height: u16) -> u16 {
    const MINIMUM_HEIGHT: u16 = 5;
    let clamped_height = requested_height.max(MINIMUM_HEIGHT);
    match terminal::size() {
        Ok((_, terminal_rows)) => clamped_height.min(terminal_rows),
        Err(_) => clamped_height,
    }
}

fn call_cli() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let workspace = Workspace::from_current_dir();

    // commands not requiring a db connection are handled here
    if let Some(Commands::Init {}) = &cli.command {
        workspace.ensure_gen_dir();
        get_config_connection(Some(workspace.gen_db_path()?))?;
        println!("Gen repository initialized.");
        return Ok(());
    }

    if let Some(Commands::Clone { url }) = &cli.command {
        return r#gen::commands::clone::execute(url, &workspace);
    }
    if let Some(Commands::CacheClear {}) = &cli.command {
        if cache::clear(&workspace)? {
            println!("Cache cleared.");
        } else {
            println!("No cache found.");
        }
        return Ok(());
    }
    #[cfg(feature = "profiling")]
    if let Some(Commands::Profile(cmd)) = &cli.command {
        return r#gen::commands::profile::execute(cmd.clone());
    }
    if let Some(Commands::Operations {
        interactive: false,
        branch,
    }) = &cli.command
    {
        let config_conn = get_config_connection(Some(workspace.gen_db_path()?))?;
        let intended_branch = Defaults::get_current_branch(&config_conn);
        let graph_connection =
            get_connection_for_branch(workspace.graph_db_path()?, intended_branch.as_deref())?;
        let context = DbContext::new(workspace.clone(), graph_connection, config_conn)?;
        let history_store = DoltHistoryStore::new(context.graph().conn());
        let (branch_name, history_entries) =
            operations_history_entries(&history_store, branch.as_deref())?;
        if history_entries.is_empty() {
            println!("No operations found.");
            return Ok(());
        }

        println!(
            "Listing Dolt history for {branch_name}. Commit messages are the operation summaries."
        );
        println!(
            "{head:<3}{hash:>40}   {summary:<70}",
            head = "",
            hash = "Id",
            summary = "Summary"
        );
        for entry in history_entries {
            let head_marker = if entry.is_head { ">" } else { "" };
            println!(
                "{head_marker:<3}{hash:>40}   {summary:<70}",
                hash = entry.commit_hash,
                summary = entry.message
            );
        }
        return Ok(());
    }

    let config_conn = get_config_connection(Some(workspace.gen_db_path()?))?;
    if let Some(Commands::Checkout { branch, hash }) = &cli.command {
        let graph_connection = get_raw_connection(workspace.graph_db_path()?)
            .map_err(|error| format!("Failed to open graph for checkout: {error}"))?;
        return r#gen::commands::checkout::execute(
            &graph_connection,
            &config_conn,
            &workspace,
            branch.as_deref(),
            hash.as_deref(),
        );
    }
    if let Some(Commands::Remote(cmd)) = &cli.command {
        return handle_remote_command(&config_conn, cmd);
    }
    if let Some(Commands::Push {
        remote,
        branch,
        force,
    }) = &cli.command
    {
        r#gen::commands::remote::operations::execute_push(
            &workspace,
            remote.as_deref(),
            branch.as_deref(),
            *force,
        )?;
        return Ok(());
    }
    if let Some(Commands::Pull { remote, branch }) = &cli.command {
        return r#gen::commands::remote::operations::execute_pull(
            &workspace,
            remote.as_deref(),
            branch.as_deref(),
        );
    }
    if let Some(Commands::Defaults {
        collection,
        committer_name,
        committer_email,
    }) = &cli.command
    {
        if let Some(name) = collection {
            config_conn.execute(
                "update defaults set collection_name=?1 where id = 1",
                (name,),
            )?;
            println!("Default collection set to {name}");
        }
        if let Some(name) = committer_name {
            Defaults::set_default_committer_name(&config_conn, name)?;
            println!("Default committer name set to {name}");
        }
        if let Some(email) = committer_email {
            Defaults::set_default_committer_email(&config_conn, email)?;
            println!("Default committer email set to {email}");
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
    let graph_db_path = workspace.graph_db_path()?;
    let binding = graph_db_path
        .to_str()
        .ok_or("Invalid path encoding")?
        .to_string();
    let db = binding.as_str();
    let intended_branch = Defaults::get_current_branch(&config_conn);
    let graph_connection = get_connection_for_branch(db, intended_branch.as_deref())?;
    let mut db_context = DbContext::new(workspace.clone(), graph_connection, config_conn)?;
    let config_conn = db_context.config().conn();
    let graph_conn = db_context.graph().conn();
    let cli_context = CliContext {
        context: &db_context,
        history_ref: None,
    };

    match cli.command {
        Some(Commands::Init {}) => {
            workspace.ensure_gen_dir();
            println!("Gen repository initialized.");
            Ok(())
        }
        Some(Commands::Clone { .. }) => {
            unreachable!("clone commands are handled before opening the workspace databases")
        }
        Some(Commands::CacheClear {}) => {
            unreachable!("cache-clear is handled before opening the workspace databases")
        }
        #[cfg(feature = "profiling")]
        Some(Commands::Profile(cmd)) => r#gen::commands::profile::execute(cmd.clone()),
        Some(Commands::Import(cmd)) => Ok(r#gen::commands::import::execute(&cli_context, cmd)?),
        Some(Commands::Update(cmd)) => Ok(r#gen::commands::update::execute(&cli_context, cmd)?),
        Some(Commands::Export(cmd)) => {
            let export_history_ref = cmd.history_ref.clone();
            let export_cli_context = CliContext {
                context: &db_context,
                history_ref: export_history_ref.as_deref(),
            };
            Ok(r#gen::commands::export::execute(&export_cli_context, cmd)?)
        }
        Some(Commands::Derive(cmd)) => Ok(r#gen::commands::derive::execute(&cli_context, cmd)?),
        Some(Commands::Remote(cmd)) => Ok(handle_remote_command(config_conn, &cmd)?),
        Some(Commands::View {
            graph,
            history_ref,
            sample,
            collection,
            position,
            full,
            height,
        }) => {
            let collection_name = &resolve_initial_collection(
                graph_conn,
                config_conn,
                collection,
                history_ref.as_deref(),
            )?;

            if !full && let (Some(name), Some(sample_name)) = (graph.as_ref(), sample.as_ref()) {
                // Use the inline widget by default if a graph is specified
                let block_group = BlockGroup::get_by_name(
                    graph_conn,
                    collection_name,
                    sample_name,
                    name,
                    history_ref.as_deref(),
                );

                match block_group {
                    Ok(bg) => {
                        let current_path = BlockGroup::get_current_path(
                            graph_conn,
                            &bg.id,
                            history_ref.as_deref(),
                        )?;
                        match show_inline_block_group_widget(
                            graph_conn,
                            bg.id,
                            vec![current_path],
                            clamp_inline_view_height(height),
                            history_ref.as_deref(),
                        ) {
                            Ok(true) => {
                                // User requested upgrade to full TUI
                                view_block_group(
                                    graph_conn,
                                    config_conn,
                                    &workspace,
                                    graph,
                                    sample,
                                    collection_name,
                                    position,
                                    history_ref.as_deref(),
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
                            name, sample_name, collection_name
                        );
                    }
                }
            } else {
                // Use the full-screen viewer if --full is specified or no graph is provided
                view_block_group(
                    graph_conn,
                    config_conn,
                    &workspace,
                    graph,
                    sample,
                    collection_name,
                    position,
                    history_ref.as_deref(),
                )?;
            }
            Ok(())
        }
        Some(Commands::ViewDiff { source, target }) => {
            let (source_ref, target_ref, range) =
                parse_diff_revisions(&source, target.as_deref()).map_err(|error| anyhow!(error))?;
            let history_store = DoltHistoryStore::new(graph_conn);
            let source_hash =
                history_store.resolve_operation_hash(&CommitRef(source_ref.clone()))?;
            let target_hash =
                history_store.resolve_operation_hash(&CommitRef(target_ref.clone()))?;
            let diff = collect_operation_diff(graph_conn, Some(source_hash), target_hash, range)?;
            if diff.operations.is_empty() {
                println!("No differences found between {source_ref} and {target_ref}.");
            } else {
                view_diff(graph_conn, &diff)?;
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
                None => get_default_collection(config_conn)?,
            });

            if let Some(bed) = bed {
                let stdout = io::stdout();
                let mut handle = stdout.lock();
                let mut bed_file = File::open(bed)?;
                Ok(translate::bed::translate_bed(
                    graph_conn,
                    collection_name,
                    sample.as_str(),
                    None,
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
                    sample.as_str(),
                    None,
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
            let history_store = DoltHistoryStore::new(graph_conn);
            let (branch_name, history_entries) =
                operations_history_entries(&history_store, branch.as_deref())?;
            if history_entries.is_empty() {
                println!("No operations found.");
                return Ok(());
            }

            if interactive {
                return view_operations(&db_context, &history_entries).map_err(Into::into);
            }

            println!(
                "Listing Dolt history for {branch_name}. Commit messages are the operation summaries."
            );
            println!(
                "{head:<3}{hash:>40}   {summary:<70}",
                head = "",
                hash = "Id",
                summary = "Summary"
            );
            for entry in history_entries {
                let head_marker = if entry.is_head { ">" } else { "" };
                println!(
                    "{head_marker:<3}{hash:>40}   {summary:<70}",
                    hash = entry.commit_hash,
                    summary = entry.message
                );
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
            let history_store = DoltHistoryStore::new(graph_conn);
            if create {
                let branch_name = branch_name
                    .clone()
                    .ok_or("Must provide a branch name to create.")?;
                history_store.create_branch(&BranchName(branch_name.clone()), None)?;
            } else if delete {
                history_store.delete_branch(&BranchName(
                    branch_name
                        .clone()
                        .ok_or("Must provide a branch name to delete.")?,
                ))?;
            } else if checkout {
                ensure_clean_working_set(&history_store, "checkout")?;
                let branch_name = branch_name
                    .clone()
                    .ok_or("Must provide a branch name to checkout.")?;
                history_store.checkout_branch(&BranchName(branch_name.clone()))?;
                Defaults::set_current_branch(config_conn, Some(&branch_name))?;
            } else if list {
                let current_branch = history_store.current_branch()?;
                println!(
                    "{indicator:<3}{col1:<30}   {col2:<20}   {col3:<15}",
                    indicator = "",
                    col1 = "Name",
                    col2 = "Operation",
                    col3 = "Remote",
                );
                for branch in branch_rows(graph_conn)?.iter() {
                    let indicator =
                        if current_branch.as_ref() == Some(&BranchName(branch.name.clone())) {
                            ">"
                        } else {
                            ""
                        };
                    let remote_display = if branch.remote.is_empty() {
                        RemoteBranch::get_remote(config_conn, &branch.name)
                            .unwrap_or_else(|| "none".to_string())
                    } else {
                        branch.remote.clone()
                    };
                    println!(
                        "{indicator:<3}{col1:<30}   {col2:<20}   {col3:<15}",
                        col1 = branch.name,
                        col2 = branch.hash,
                        col3 = remote_display
                    );
                }
            } else if merge {
                let branch_name = branch_name.clone().ok_or("Branch name must be provided.")?;
                history_store.merge(&CommitRef(branch_name))?;
                println!("Merge successful");
            } else if let Some(remote_name) = set_remote {
                let current_branch_name = history_store
                    .current_branch()?
                    .ok_or("No current branch is checked out.")?;
                let remote_to_set = if remote_name.is_empty() || remote_name == "null" {
                    None
                } else {
                    Some(remote_name.as_str())
                };

                RemoteBranch::set_remote_validated(
                    config_conn,
                    &current_branch_name.0,
                    remote_to_set,
                )?;

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
            let history_store = DoltHistoryStore::new(graph_conn);
            ensure_clean_working_set(&history_store, "merge")?;
            match history_store.merge(&CommitRef(branch_name)) {
                Ok(_) => {
                    println!("Merge successful");
                    Ok(())
                }
                Err(e) => Err(history_action_error("Merge", &e)),
            }
        }
        Some(Commands::Apply { hash }) => {
            let history_store = DoltHistoryStore::new(graph_conn);
            ensure_clean_working_set(&history_store, "apply")?;
            let commit_hash = history_store.resolve_operation_hash(&CommitRef(hash))?;
            match history_store.cherry_pick(&commit_hash) {
                Ok(_) => {
                    println!("Operation applied");
                    Ok(())
                }
                Err(e) => Err(history_action_error("Apply", &e)),
            }
        }
        Some(Commands::Checkout { .. }) => {
            unreachable!("checkout commands are handled before running graph migrations")
        }
        Some(Commands::Reset { hash }) => {
            let history_store = DoltHistoryStore::new(graph_conn);
            ensure_clean_working_set(&history_store, "reset")?;
            match history_store.reset_hard(&CommitRef(hash)) {
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
            let history_store = DoltHistoryStore::new(graph_conn);
            let operations = match branch.as_deref() {
                Some(branch_name) => {
                    patch::patch_operations_for_branch(&history_store, branch_name, &operation)?
                }
                None => {
                    let branch_name = history_store
                        .current_branch()?
                        .ok_or("No current branch is checked out.")?;
                    patch::parse_patch_commit_selection(&history_store, &branch_name.0, &operation)?
                }
            };
            let mut f = File::create(format!("{name}.gz"))?;
            patch::create_patch(&db_context, &operations, &mut f)?;
            Ok(())
        }
        Some(Commands::PatchApply { patch }) => {
            let mut f = File::open(patch)?;
            match patch::apply_patch_archive(&mut db_context, &mut f) {
                Ok(_) => {
                    println!("Patch applied");
                    Ok(())
                }
                Err(e) => Err(format!("Patch application failed: {}", e).into()),
            }
        }
        Some(Commands::PatchView { patch }) => {
            let mut file = File::open(patch)?;
            view_patch(&db_context, &mut file)?;
            Ok(())
        }
        None => Ok(()),
        Some(Commands::Defaults { .. }) => Ok(()),
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
                None => get_default_collection(config_conn)?,
            });
            graph_conn.execute("BEGIN TRANSACTION", [])?;

            propagate_gff(
                graph_conn,
                collection_name,
                from_sample.as_str(),
                &to_sample,
                &gff,
                &output_gff,
            )?;

            graph_conn.execute("END TRANSACTION", [])?;
            Ok(())
        }
        Some(Commands::AddAnnotation {
            name,
            group,
            sample,
            region,
        }) => {
            let collection_name = get_default_collection(config_conn)?;
            graph_conn.execute("BEGIN TRANSACTION", [])?;
            let operation_summary = match add_annotation(
                &db_context,
                &collection_name,
                &name,
                group.as_deref(),
                sample.as_str(),
                &region,
            ) {
                Ok(operation_summary) => operation_summary,
                Err(err) => {
                    graph_conn.execute("ROLLBACK TRANSACTION;", [])?;
                    return Err(err);
                }
            };
            graph_conn.execute("END TRANSACTION", [])?;
            match commit_operation(&db_context, &operation_summary) {
                Ok(commit_hash) => {
                    println!("Annotation {name} added in operation {commit_hash}");
                }
                Err(OperationError::NoChanges) => {
                    println!("Annotation {name} already exists.");
                }
                Err(err) => return Err(err.into()),
            }
            Ok(())
        }
        Some(Commands::AddAnnotationFile {
            path,
            format,
            index,
            name,
            message,
        }) => {
            let commit_hash = add_annotation_file(
                &db_context,
                &path,
                format.as_deref(),
                index.as_deref(),
                name.as_deref(),
                message.as_deref(),
                AnnotationFileChecksumOverrides::default(),
            )?;
            println!("Annotation file added in operation {commit_hash}");
            Ok(())
        }
        Some(Commands::AddFile { files, message }) => {
            let operation_files = files
                .into_iter()
                .map(OperationFile::new)
                .collect::<Vec<_>>();
            let commit_hash =
                add_files_operation(&db_context, &operation_files, message.as_deref())?;
            println!("Files added in operation {commit_hash}");
            Ok(())
        }
        Some(Commands::BuildIndex {
            collection,
            sample,
            kmer_size,
        }) => {
            let collection_name = match collection {
                Some(c) => c,
                None => get_default_collection(config_conn)?,
            };
            let block_groups = match sample {
                Some(ref s) => Sample::get_block_groups(graph_conn, &collection_name, s, None),
                None => Collection::get_block_groups(graph_conn, &collection_name, None),
            };
            let index_dir = workspace
                .ensure_search_index()
                .map_err(|_| "No .gen directory found. Run 'gen init' first.")?;
            for bg in block_groups {
                let graph = BlockGroup::get_graph(graph_conn, &bg.id, None)?;
                let matcher = GenGraphMatcher::new(graph_conn, graph);
                let index = SeedIndex::build(&matcher, kmer_size, true);
                let path = index_dir.join(format!("{}.bin", bg.id));
                index.save_to_path(&path).map_err(|e| anyhow!("{e}"))?;
                println!("indexed {}/{}", bg.sample_name, bg.name);
            }
            Ok(())
        }
        Some(Commands::ClearIndex { collection, sample }) => {
            let collection_name = match collection {
                Some(c) => c,
                None => get_default_collection(config_conn)?,
            };
            let block_groups = match sample {
                Some(ref s) => Sample::get_block_groups(graph_conn, &collection_name, s, None),
                None => Collection::get_block_groups(graph_conn, &collection_name, None),
            };
            let index_dir = workspace
                .find_search_index()
                .ok_or("No .gen directory found. Run 'gen init' first.")?;
            if !index_dir.exists() {
                println!("No search index cache found.");
                return Ok(());
            }
            for bg in block_groups {
                let path = index_dir.join(format!("{}.bin", bg.id));
                if path.exists() {
                    std::fs::remove_file(&path)?;
                    println!("cleared {}/{}", bg.sample_name, bg.name);
                }
            }
            Ok(())
        }
        Some(Commands::Search {
            query,
            sample,
            collection,
        }) => {
            let block_groups = match (collection, sample.as_deref()) {
                (Some(c), Some(s)) => Sample::get_block_groups(graph_conn, &c, s, None),
                (Some(c), None) => Collection::get_block_groups(graph_conn, &c, None),
                (None, Some(s)) => BlockGroup::all(graph_conn)
                    .into_iter()
                    .filter(|bg| bg.sample_name == s)
                    .collect(),
                (None, None) => BlockGroup::all(graph_conn),
            };
            let index_dir = workspace.find_search_index();
            let query_bytes = query.as_bytes();
            println!("sample\tgraph\tblocks\toffset");
            for bg in block_groups {
                let graph = BlockGroup::get_graph(graph_conn, &bg.id, None)?;
                let matcher = GenGraphMatcher::new(graph_conn, graph);
                let matches = index_dir
                    .as_ref()
                    .and_then(|dir| {
                        let p = dir.join(format!("{}.bin", bg.id));
                        SeedIndex::load_from_path(p).ok()
                    })
                    .map(|idx| matcher.find_all_with_seed_index(&idx, query_bytes))
                    .unwrap_or_else(|| Ok(matcher.find_all(query_bytes)))?;
                for m in matches {
                    let block_strings: Vec<String> = m
                        .slices
                        .iter()
                        .map(|s| {
                            let hash = format!("{}", s.block.node_id);
                            format!(
                                "{}:{}-{}",
                                &hash[..12],
                                s.block.sequence_start,
                                s.block.sequence_end
                            )
                        })
                        .collect();
                    println!(
                        "{}\t{}\t[{}]\t{}",
                        bg.sample_name,
                        bg.name,
                        block_strings.join(", "),
                        m.slices.first().map(|s| s.start).unwrap_or(0)
                    );
                }
            }
            Ok(())
        }
        Some(Commands::ListSamples { history_ref }) => {
            let sample_names = Sample::get_all_names(graph_conn, history_ref.as_deref());
            println!();
            for sample_name in sample_names {
                println!("{sample_name}");
            }
            Ok(())
        }
        Some(Commands::ListGraphs {
            history_ref,
            name,
            sample,
        }) => {
            let collection_name = &(match name {
                Some(collection) => collection,
                None => get_default_collection(config_conn)?,
            });
            let block_groups = Sample::get_block_groups(
                graph_conn,
                collection_name,
                &sample,
                history_ref.as_deref(),
            );
            for block_group in block_groups {
                println!("{}", block_group.name);
            }
            Ok(())
        }
        Some(Commands::GetSequence {
            history_ref,
            name,
            sample,
            graph,
            start,
            end,
            region,
        }) => {
            let collection_name = &(match name {
                Some(collection) => collection,
                None => get_default_collection(config_conn)?,
            });
            let block_groups = Sample::get_block_groups(
                graph_conn,
                collection_name,
                &sample,
                history_ref.as_deref(),
            );

            let formatted_sample_name = format!("sample {sample}");

            let (parsed_graph_name, start_coordinate, mut end_coordinate) =
                if let Some(region) = region {
                    let parsed_region = Region::parse(&region);
                    match parsed_region {
                        Ok(parsed_region) => {
                            let (start, end) = parsed_region.require_coordinates()?;
                            (parsed_region.name, start, end)
                        }
                        Err(parse_error) => {
                            return Err(Box::new(parse_error));
                        }
                    }
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
            let path =
                BlockGroup::get_current_path(graph_conn, &block_group.id, history_ref.as_deref())?;
            let sequence = path.sequence(graph_conn, history_ref.as_deref())?;
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
                None => get_default_collection(config_conn)?,
            });
            gfa_sample_diff(
                graph_conn,
                collection_name,
                &PathBuf::from(gfa),
                sample1.as_str(),
                sample2.as_str(),
            )?;
            Ok(())
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
        Some(Commands::Push { .. }) | Some(Commands::Pull { .. }) => unreachable!(),
        Some(Commands::AddReferenceAliases {
            reference_name,
            refseq_accession_id,
            genbank_id,
            ensembl_id,
            ucsc_id,
            custom_id,
            chromosome,
        }) => {
            match ReferenceAlias::create(
                graph_conn,
                &reference_name,
                refseq_accession_id,
                genbank_id,
                ucsc_id,
                ensembl_id,
                custom_id,
                chromosome,
            ) {
                Ok(_) => Ok(()),
                Err(e) => Err(format!("Error creating reference aliases: {e}").into()),
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    install_global_panic_hook();
    init_theme();

    // Start logger (gets log level from RUST_LOG environment variable, sends output to stderr)
    env_logger::init();
    let result = if std::env::var_os("GEN_PROFILE").is_some() {
        #[cfg(feature = "profiling")]
        {
            if std::env::var_os("GEN_PROFILE_SAMPLE").is_some() {
                SamplingProfiler.run(call_cli)
            } else {
                Profiler::default().run(call_cli)
            }
        }
        #[cfg(not(feature = "profiling"))]
        {
            call_cli()
        }
    } else {
        call_cli()
    };
    match result {
        Ok(_) => Ok(()),
        Err(e) => Err(anyhow!("{e}").into()),
    }
}

#[cfg(test)]
mod tests {
    use r#gen::test_helpers::setup_gen;
    use gen_models::collection::Collection;

    use super::resolve_initial_collection;

    #[test]
    fn test_view_opens_only_collection_by_default() {
        let context = setup_gen();
        Collection::create(context.graph().conn(), "only-collection")
            .expect("should create only collection");

        let collection =
            resolve_initial_collection(context.graph().conn(), context.config().conn(), None, None)
                .expect("should resolve initial collection");

        assert_eq!(collection, "only-collection");
    }
}
