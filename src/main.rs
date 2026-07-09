#![allow(warnings)]
use std::{
    fmt::Debug,
    fs::File,
    io,
    io::{BufReader, Write},
    ops::Deref,
    path::{Path, PathBuf},
    str,
};

use anyhow::anyhow;
use clap::{Parser, Subcommand};
use crossterm::terminal;
#[cfg(feature = "profiling")]
use r#gen::profiling::{Profiler, SamplingProfiler};
use r#gen::{
    annotations::gff::propagate_gff,
    commands::{
        Cli, Commands,
        cli_context::CliContext,
        graph_operations::make_stitch::make_stitch_operation,
        remote::{discover_dolt_remote_url, handle_remote_command, validate_dolt_remote_url},
    },
    diffs::gfa::gfa_sample_diff,
    get_config_connection, get_connection, get_history_connection,
    graphs::graph_search::{GenGraphMatcher, SeedIndex},
    patch,
    theme::init_theme,
    updates::gaf::transform_csv_to_fasta,
    views::{
        block_group::view_block_group, block_group_inline::show_inline_block_group_widget,
        diff::view_diff, operations::view_operations, patch::view_patches,
        tui_runtime::install_global_panic_hook,
    },
};
use gen_annotations::translate;
use gen_core::{BranchName, CommitRef, config::Workspace, range::Range, region::Region};
use gen_diff::operations::collect_operation_diff;
use gen_models::{
    annotations::{add_annotation, add_annotation_file},
    block_group::BlockGroup,
    collection::Collection,
    db::{ConfigConnection, DbContext, GraphConnection},
    errors::RemoteError,
    history::{
        HistoryStore,
        dolt::{DoltHistoryStore, add_remote as add_dolt_remote, branch_rows, remote_rows},
    },
    operations::{Defaults, RemoteBranch, add_files_operation},
    reference_alias::ReferenceAlias,
    sample::Sample,
    traits::Query,
};
use rusqlite::{Connection, params, types::Value};
use sha2::digest::typenum::Gr;

fn get_default_collection(conn: &ConfigConnection) -> Result<String, rusqlite::Error> {
    let mut stmt = conn.prepare("select collection_name from defaults where id = 1")?;
    Ok(stmt
        .query_row((), |row| row.get(0))
        .unwrap_or("default".to_string()))
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

fn resolve_commit_ref(
    history_store: &impl HistoryStore,
    reference: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    history_store
        .resolve_ref(&CommitRef(reference.to_string()))?
        .map(|commit_hash| commit_hash.0)
        .ok_or_else(|| format!("No commit resolved for {reference}.").into())
}

fn resolve_operation_hash_from_history(
    history_store: &impl HistoryStore,
    reference: &str,
) -> Result<gen_core::HashId, Box<dyn std::error::Error>> {
    let commit_hash = history_store
        .resolve_ref(&CommitRef(reference.to_string()))?
        .ok_or_else(|| format!("No commit resolved for {reference}."))?;
    Ok(gen_core::HashId::pad_str(&commit_hash.0))
}

fn parse_patch_commit_selection(
    history_store: &impl HistoryStore,
    branch_operations: &[gen_core::HashId],
    operations: &str,
) -> Result<Vec<gen_core::HashId>, Box<dyn std::error::Error>> {
    let mut selected_operations = Vec::new();

    for operation in operations.split(',') {
        let operation = operation.trim();
        if operation.is_empty() {
            return Err("Patch selection cannot be empty.".into());
        }

        if let Some((start_ref, end_ref)) = operation.split_once("..") {
            let start_hash =
                resolve_patch_operation_hash(history_store, branch_operations, start_ref)?;
            let end_hash = resolve_patch_operation_hash(history_store, branch_operations, end_ref)?;
            let start_position = branch_operations
                .iter()
                .position(|candidate| *candidate == start_hash)
                .ok_or_else(|| format!("Resolved start ref '{start_ref}' is not present in the selected branch history."))?;
            let end_position = branch_operations
                .iter()
                .position(|candidate| *candidate == end_hash)
                .ok_or_else(|| format!("Resolved end ref '{end_ref}' is not present in the selected branch history."))?;
            selected_operations.extend(
                branch_operations[start_position..=end_position]
                    .iter()
                    .copied(),
            );
            continue;
        }

        selected_operations.push(resolve_patch_operation_hash(
            history_store,
            branch_operations,
            operation,
        )?);
    }

    Ok(selected_operations)
}

fn current_branch_operations(
    history_store: &impl HistoryStore,
) -> Result<Vec<gen_core::HashId>, Box<dyn std::error::Error>> {
    Ok(history_store
        .log(None)?
        .into_iter()
        .rev()
        .map(|entry| gen_core::HashId::pad_str(&entry.commit_hash.0))
        .collect())
}

fn patch_operations_for_branch(
    history_store: &impl HistoryStore,
    branch_name: &str,
    operation: &str,
) -> Result<Vec<gen_core::HashId>, Box<dyn std::error::Error>> {
    let current_branch_name = history_store
        .current_branch()?
        .ok_or("No current branch is checked out.")?;
    let current_operations = current_branch_operations(history_store)?;
    if branch_name == current_branch_name.0 {
        return parse_patch_commit_selection(history_store, &current_operations, operation);
    }

    history_store.checkout_branch(&BranchName(branch_name.to_string()))?;
    let selected_operations = (|| {
        let branch_operations = current_branch_operations(history_store)?;
        if operation == "HEAD" {
            let shared_prefix_length = current_operations
                .iter()
                .zip(branch_operations.iter())
                .take_while(|(current_operation, branch_operation)| {
                    current_operation == branch_operation
                })
                .count();
            let selected_operations = branch_operations[shared_prefix_length..].to_vec();
            if selected_operations.is_empty() {
                return Err(format!(
                    "Branch '{branch_name}' does not diverge from the current head."
                )
                .into());
            }
            return Ok(selected_operations);
        }

        parse_patch_commit_selection(history_store, &branch_operations, operation)
    })();
    let restore_result = history_store.checkout_branch(&current_branch_name);
    restore_result?;
    selected_operations
}

fn operations_history_entries(
    history_store: &impl HistoryStore,
    branch_name: Option<&str>,
) -> Result<(String, Vec<gen_models::history::HistoryEntry>), Box<dyn std::error::Error>> {
    let selected_branch_name = match branch_name {
        Some(branch_name) => branch_name.to_string(),
        None => history_store
            .current_branch()?
            .map(|branch| branch.0)
            .unwrap_or_else(|| "HEAD".to_string()),
    };

    if let Some(branch_name) = branch_name {
        history_store.checkout_branch(&BranchName(branch_name.to_string()))?;
    }
    let history_entries = history_store.log(None)?;

    Ok((selected_branch_name, history_entries))
}

fn resolve_patch_operation_hash(
    history_store: &impl HistoryStore,
    branch_operations: &[gen_core::HashId],
    reference: &str,
) -> Result<gen_core::HashId, Box<dyn std::error::Error>> {
    if reference == "HEAD" {
        return branch_operations
            .last()
            .copied()
            .ok_or_else(|| "No operations present in the selected branch.".into());
    }

    if let Some(offset) = reference.strip_prefix("HEAD~") {
        let offset: usize = offset
            .parse()
            .map_err(|_| format!("Invalid HEAD reference '{reference}'."))?;
        let head_index = branch_operations
            .len()
            .checked_sub(1)
            .ok_or("No operations present in the selected branch.")?;
        let target_index = head_index.checked_sub(offset).ok_or_else(|| {
            format!("HEAD offset {offset} is out of range for the selected branch.")
        })?;
        return Ok(branch_operations[target_index]);
    }

    let commit_hash = history_store
        .resolve_ref(&CommitRef(reference.to_string()))?
        .ok_or_else(|| format!("No commit resolved for {reference}."))?;
    let operation_hash = gen_core::HashId::pad_str(&commit_hash.0);

    branch_operations
        .iter()
        .find(|candidate| **candidate == operation_hash)
        .copied()
        .ok_or_else(|| {
            format!("Resolved commit '{reference}' is not present in the selected branch history.")
                .into()
        })
}

fn ensure_graph_remote(
    graph_conn: &gen_models::db::GraphConnection,
    operation_conn: &ConfigConnection,
    remote_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let graph_remotes = remote_rows(graph_conn)?;
    if graph_remotes
        .iter()
        .any(|remote| remote.name == remote_name)
    {
        return Ok(());
    }

    let dolt_remote_url = normalized_graph_remote_url(operation_conn, remote_name)?;
    add_dolt_remote(graph_conn, remote_name, &dolt_remote_url)?;
    Ok(())
}

fn ensure_clean_working_set(
    history_store: &impl HistoryStore,
    action: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let status_rows = history_store.status()?;
    if status_rows.is_empty() {
        return Ok(());
    }

    Err(format!(
        "Cannot {action}: the working set has uncommitted changes. Commit or reset them first."
    )
    .into())
}

fn history_action_error(action: &str, error: &dyn std::error::Error) -> Box<dyn std::error::Error> {
    let message = error.to_string();
    if message.to_ascii_lowercase().contains("conflict") {
        return format!(
            "{action} failed with Dolt conflicts. Gen does not yet provide conflict resolution commands. Original error: {message}"
        )
        .into();
    }

    format!("{action} failed: {message}").into()
}

fn normalized_graph_remote_url(
    operation_conn: &ConfigConnection,
    remote_name: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let remote = gen_models::operations::Remote::get_by_name(operation_conn, remote_name)?;
    let dolt_remote_url = match url::Url::parse(&remote.url) {
        Ok(parsed_url) if parsed_url.scheme() == "file" => match parsed_url.to_file_path() {
            Ok(remote_path)
                if remote_path.extension().and_then(|value| value.to_str()) != Some("db") =>
            {
                let graph_db_path = remote_path.join(".gen").join("default.db");
                if !graph_db_path.exists() && remote_path.join(".gen").is_dir() {
                    let _ = open_unmigrated_graph_connection(&graph_db_path)?;
                }
                url::Url::from_file_path(graph_db_path)
                    .map(|url| url.to_string())
                    .unwrap_or_else(|()| remote.url.clone())
            }
            _ => remote.url.clone(),
        },
        _ => discover_dolt_remote_url(&remote.url)?.unwrap_or_else(|| remote.url.clone()),
    };
    validate_dolt_remote_url(&dolt_remote_url)?;
    Ok(dolt_remote_url)
}

fn file_remote_graph_path(remote_url: &str) -> Option<PathBuf> {
    let parsed_url = url::Url::parse(remote_url).ok()?;
    if parsed_url.scheme() != "file" {
        return None;
    }
    let remote_path = parsed_url.to_file_path().ok()?;
    if remote_path.extension().and_then(|value| value.to_str()) == Some("db") {
        Some(remote_path)
    } else {
        Some(remote_path.join(".gen").join("default.db"))
    }
}

fn is_virgin_history(
    history_store: &impl HistoryStore,
) -> Result<bool, Box<dyn std::error::Error>> {
    let history = history_store.log(Some(2))?;
    Ok(history.len() == 1 && history[0].message == "Initialize data repository")
}

fn open_unmigrated_graph_connection(
    db_path: &Path,
) -> Result<GraphConnection, Box<dyn std::error::Error>> {
    let connection = Connection::open(db_path)?;
    rusqlite::vtab::array::load_module(&connection)?;
    Ok(GraphConnection(connection))
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
    #[cfg(feature = "profiling")]
    if let Some(Commands::Profile(cmd)) = &cli.command {
        return r#gen::commands::profile::execute(cmd.clone());
    }
    if let Some(Commands::Operations {
        interactive: false,
        branch,
    }) = &cli.command
    {
        let operation_conn = get_config_connection(Some(workspace.gen_db_path()?))?;
        let graph_connection = get_connection(workspace.graph_db_path()?)?;
        let context = DbContext::new(workspace.clone(), graph_connection, operation_conn)?;
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
            "{head:<3}{hash:>64}   {summary:<70}",
            head = "",
            hash = "Id",
            summary = "Summary"
        );
        for entry in history_entries {
            let head_marker = if entry.is_head { ">" } else { "" };
            println!(
                "{head_marker:<3}{hash:>64}   {summary:<70}",
                hash = entry.commit_hash.0,
                summary = entry.message
            );
        }
        return Ok(());
    }

    let operation_conn = get_config_connection(Some(workspace.gen_db_path()?))?;
    if let Some(Commands::Remote(cmd)) = &cli.command {
        return handle_remote_command(&operation_conn, cmd);
    }
    if let Some(Commands::Pull { remote }) = &cli.command {
        let remote_name = remote
            .clone()
            .or_else(|| Defaults::get_default_remote(&operation_conn))
            .ok_or(
                "No remote configured. Set one with `gen remote add` or `gen defaults --remote`.",
            )?;
        let graph_db_path = workspace.graph_db_path()?;
        let normalized_remote_url = normalized_graph_remote_url(&operation_conn, &remote_name)?;
        if file_remote_graph_path(&normalized_remote_url).is_some() && !graph_db_path.exists() {
            let graph_conn = open_unmigrated_graph_connection(&graph_db_path)?;
            gen_models::history::dolt::clone_remote(&graph_conn, &normalized_remote_url)?;
            return Ok(());
        }
    }
    if let Some(Commands::Defaults {
        database,
        collection,
        committer_name,
        committer_email,
    }) = &cli.command
    {
        if let Some(name) = database {
            return Err(format!(
                "`gen defaults --database {name}` is no longer supported; Gen always uses `.gen/default.db`."
            )
            .into());
        }
        if let Some(name) = collection {
            operation_conn.execute(
                "update defaults set collection_name=?1 where id = 1",
                (name,),
            )?;
            println!("Default collection set to {name}");
        }
        let graph_conn = if committer_name.is_some() || committer_email.is_some() {
            Some(get_connection(workspace.graph_db_path()?)?)
        } else {
            None
        };
        if let Some(name) = committer_name {
            gen_models::history::dolt::set_commit_author_name(
                graph_conn
                    .as_ref()
                    .expect("should open graph db for committer update"),
                name,
            )?;
            println!("Default committer name set to {name}");
        }
        if let Some(email) = committer_email {
            gen_models::history::dolt::set_commit_author_email(
                graph_conn
                    .as_ref()
                    .expect("should open graph db for committer update"),
                email,
            )?;
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
    let binding = match cli.db {
        Some(db) => {
            return Err(format!(
                "`--db {db}` is no longer supported; Gen always uses `.gen/default.db`."
            )
            .into());
        }
        None => graph_db_path
            .to_str()
            .ok_or("Invalid path encoding")?
            .to_string(),
    };
    let db = binding.as_str();
    let graph_connection = get_connection(db)?;
    let mut db_context = DbContext::new(workspace.clone(), graph_connection, operation_conn)?;
    let operation_conn = db_context.config().conn();
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
        Some(Commands::Clone { .. }) => Ok(()),
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
        Some(Commands::Remote(cmd)) => Ok(handle_remote_command(operation_conn, &cmd)?),
        Some(Commands::View {
            graph,
            history_ref,
            sample,
            collection,
            position,
            full,
            height,
        }) => {
            let collection_name = &(match collection {
                Some(collection) => collection,
                None => get_default_collection(operation_conn)?,
            });

            if !full && let (Some(name), Some(sample_name)) = (graph.as_ref(), sample.as_ref()) {
                // Use the inline widget by default if a graph is specified
                let block_group = BlockGroup::get(
                    graph_conn,
                    "select * from block_groups where collection_name = ?1 AND sample_name = ?2 AND name = ?3",
                    params![collection_name, sample_name, name],
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
                                    operation_conn,
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
                    operation_conn,
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
        Some(Commands::ViewDiff { from, to }) => {
            let to_ref = to.clone().unwrap_or_else(|| "HEAD".to_string());
            let history_store = DoltHistoryStore::new(graph_conn);
            let from_hash = resolve_operation_hash_from_history(&history_store, &from)?;
            let to_hash = resolve_operation_hash_from_history(&history_store, &to_ref)?;
            let diff = collect_operation_diff(graph_conn, Some(from_hash), to_hash)?;
            if diff.operations.is_empty() {
                println!("No differences found between {from} and {to_ref}.");
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
                None => get_default_collection(operation_conn)?,
            });

            if let Some(bed) = bed {
                let stdout = io::stdout();
                let mut handle = stdout.lock();
                let mut bed_file = File::open(bed)?;
                Ok(translate::bed::translate_bed(
                    graph_conn,
                    collection_name,
                    sample.as_str(),
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
                "{head:<3}{hash:>64}   {summary:<70}",
                head = "",
                hash = "Id",
                summary = "Summary"
            );
            for entry in history_entries {
                let head_marker = if entry.is_head { ">" } else { "" };
                println!(
                    "{head_marker:<3}{hash:>64}   {summary:<70}",
                    hash = entry.commit_hash.0,
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
                Defaults::set_current_branch(operation_conn, Some(&branch_name))?;
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
                        RemoteBranch::get_remote(operation_conn, &branch.name)
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
                    operation_conn,
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
            let commit_hash = resolve_commit_ref(&history_store, &hash)?;
            match history_store.cherry_pick(&gen_core::CommitHash(commit_hash)) {
                Ok(_) => {
                    println!("Operation applied");
                    Ok(())
                }
                Err(e) => Err(history_action_error("Apply", &e)),
            }
        }
        Some(Commands::Checkout { branch, hash }) => {
            let history_store = DoltHistoryStore::new(graph_conn);
            ensure_clean_working_set(&history_store, "checkout")?;
            if let Some(name) = branch.clone() {
                let branch_exists = branch_rows(graph_conn)?
                    .iter()
                    .any(|branch| branch.name == name);
                if !branch_exists {
                    history_store.create_branch(&BranchName(name.clone()), None)?;
                    println!("Created branch {name}");
                }
                println!("Checking out branch {name}");
                history_store.checkout_branch(&BranchName(name))?;
                Defaults::set_current_branch(operation_conn, branch.as_deref())?;
            } else if let Some(hash_name) = hash.clone() {
                if branch_rows(graph_conn)?
                    .iter()
                    .any(|branch| branch.name == hash_name)
                {
                    println!("Checking out branch {hash_name}");
                    history_store.checkout_branch(&BranchName(hash_name.clone()))?;
                    Defaults::set_current_branch(operation_conn, Some(&hash_name))?;
                } else {
                    let commit_hash = resolve_commit_ref(&history_store, &hash_name)?;
                    return Err(format!(
                        "Detached HEAD checkouts are not supported for ref '{hash_name}' (resolved to {commit_hash}). Use --ref with read-only commands such as export, view, list-samples, list-graphs, or get-sequence."
                    )
                    .into());
                }
            } else {
                println!("No branch or hash to checkout provided.");
            }
            Ok(())
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
                    patch_operations_for_branch(&history_store, branch_name, &operation)?
                }
                None => {
                    let branch_ops = current_branch_operations(&history_store)?;
                    if branch_ops.is_empty() {
                        return Err("No operations present in the selected branch.".into());
                    }
                    parse_patch_commit_selection(&history_store, &branch_ops, &operation)?
                }
            };
            let mut f = File::create(format!("{name}.gz"))?;
            patch::create_patch(&db_context, &operations, branch.as_deref(), &mut f)?;
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
        Some(Commands::PatchView { prefix, patch }) => {
            let patch_path = Path::new(&patch);
            let mut f = File::open(patch_path)?;
            let patches = patch::load_patches(&mut f);
            let diagrams = view_patches(&workspace, &patches)?;
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
            committer_name,
            committer_email,
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
            graph_conn.execute("BEGIN TRANSACTION", [])?;
            operation_conn.execute("BEGIN TRANSACTION", [])?;

            propagate_gff(
                graph_conn,
                collection_name,
                from_sample.as_str(),
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
            let commit_hash = add_annotation(
                &db_context,
                &collection_name,
                &name,
                group.as_deref(),
                sample.as_str(),
                &region,
            )?;
            println!("Annotation {name} added in operation {}", commit_hash.0);
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
            )?;
            println!("Annotation file added in operation {}", commit_hash.0);
            Ok(())
        }
        Some(Commands::AddFile { files, message }) => {
            let commit_hash = add_files_operation(&db_context, &files, message.as_deref())?;
            println!("Files added in operation {}", commit_hash.0);
            Ok(())
        }
        Some(Commands::BuildIndex {
            collection,
            sample,
            kmer_size,
        }) => {
            let collection_name = match collection {
                Some(c) => c,
                None => get_default_collection(operation_conn)?,
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
                None => get_default_collection(operation_conn)?,
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
                None => get_default_collection(operation_conn)?,
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
                None => get_default_collection(operation_conn)?,
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
                None => get_default_collection(operation_conn)?,
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
        Some(Commands::Push { remote }) => {
            let history_store = DoltHistoryStore::new(graph_conn);
            let remote_name = remote
                .or_else(|| Defaults::get_default_remote(operation_conn))
                .ok_or("No remote configured. Set one with `gen remote add` or `gen defaults --remote`.")?;
            let branch_name = history_store
                .current_branch()?
                .ok_or("No current branch is checked out.")?;
            ensure_graph_remote(graph_conn, operation_conn, &remote_name)?;
            let normalized_remote_url = normalized_graph_remote_url(operation_conn, &remote_name)?;
            let should_force_push =
                if let Some(remote_graph_path) = file_remote_graph_path(&normalized_remote_url) {
                    if remote_graph_path.exists() {
                        let remote_graph_conn = get_connection(&remote_graph_path)?;
                        let remote_history_store = DoltHistoryStore::new(&remote_graph_conn);
                        is_virgin_history(&remote_history_store)?
                    } else {
                        false
                    }
                } else {
                    false
                };
            if should_force_push {
                gen_models::history::dolt::push_force(graph_conn, &remote_name, &branch_name.0)?;
            } else {
                history_store.push(&remote_name, &branch_name)?;
            }
            println!("Push succeeded.");
            Ok(())
        }
        Some(Commands::Pull { remote }) => {
            let history_store = DoltHistoryStore::new(graph_conn);
            let remote_name = remote
                .or_else(|| Defaults::get_default_remote(operation_conn))
                .ok_or("No remote configured. Set one with `gen remote add` or `gen defaults --remote`.")?;
            let branch_name = history_store
                .current_branch()?
                .ok_or("No current branch is checked out.")?;
            ensure_graph_remote(graph_conn, operation_conn, &remote_name)?;
            let normalized_remote_url = normalized_graph_remote_url(operation_conn, &remote_name)?;
            if file_remote_graph_path(&normalized_remote_url).is_some()
                && is_virgin_history(&history_store)?
            {
                gen_models::history::dolt::clone_remote(graph_conn, &normalized_remote_url)?;
            } else {
                history_store.pull(&remote_name, &branch_name)?;
            }
            Ok(())
        }
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
