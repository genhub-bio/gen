use std::{collections::HashMap, error::Error};

use gen_core::{BranchName, CommitRef, DoltHashId, HashId, config::Workspace};
use gen_models::{
    assets::{AssetRef, materialization_destination_path},
    db::{ConfigConnection, GraphConnection},
    history::{
        HistoryStore,
        dolt::{DoltHistoryStore, branch_exists, checkout, connect_branch, hash_of},
    },
    operations::Defaults,
};

use crate::{
    commands::remote::operations::{
        DownloadAssetOutcome, materialize_versioned_asset, warn_asset_conflict,
    },
    get_raw_connection,
    history::ensure_clean_working_set,
};

/// Restores assets present at the requested commit from `.gen/assets` into the workspace
///
/// Previously stored versioned files are safely replaced by requested versions.
/// Unknown local contents are preserved and receive the requested version marked as a conflict.
fn materialize_checked_out_assets(
    graph: &GraphConnection,
    workspace: &Workspace,
    commit_hash: &DoltHashId,
    previous_assets: &HashMap<HashId, AssetRef>,
) -> Result<(), Box<dyn Error>> {
    for asset in AssetRef::get_materialized_assets_at(graph, None, Some(commit_hash))? {
        let destination_logical_path = asset.logical_path.as_deref();
        let versioned_path =
            materialization_destination_path(workspace, &asset.uri, asset.checksum.as_ref(), None)?;
        if let DownloadAssetOutcome::Conflict(conflict_path) = materialize_versioned_asset(
            workspace,
            &asset,
            previous_assets,
            destination_logical_path,
            &versioned_path,
            false,
        )? {
            warn_asset_conflict(workspace, &asset, destination_logical_path, &conflict_path)?;
        }
    }
    Ok(())
}

fn materialize_branch_head_assets(
    workspace: &Workspace,
    branch: &str,
    previous_assets: &HashMap<HashId, AssetRef>,
) -> Result<(), Box<dyn Error>> {
    // The published DoltLite history table starts its walk at the connection's branch even when
    // its rows are joined to an explicit hash. Attach a fresh connection to the requested branch,
    // then keep asset selection bounded by that branch's exact head.
    let graph = get_raw_connection(workspace.graph_db_path()?)?;
    connect_branch(&graph, branch)?;
    let commit_hash = hash_of(&graph, branch)?;
    materialize_checked_out_assets(&graph, workspace, &commit_hash, previous_assets)
}

pub fn execute(
    graph: &GraphConnection,
    config: &ConfigConnection,
    workspace: &Workspace,
    branch: Option<&str>,
    hash: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    if let Some(current_branch) = Defaults::get_current_branch(config)
        && branch_exists(graph, &current_branch)?
    {
        connect_branch(graph, &current_branch)?;
    }
    let history_store = DoltHistoryStore::new(graph);
    ensure_clean_working_set(&history_store, "checkout")?;
    // Keep every version reachable from the current checkout so the materializer can distinguish
    // a tracked version it may replace from unknown local content it must preserve.
    let previous_assets = AssetRef::get_cumulative_assets_at(graph, None, None)?
        .into_iter()
        .map(|asset| (asset.id, asset))
        .collect();
    if let Some(name) = branch {
        if !branch_exists(graph, name)? {
            history_store.create_branch(&BranchName(name.to_string()), None)?;
            println!("Created branch {name}");
        }
        println!("Checking out branch {name}");
        checkout(graph, name)
            .map_err(|error| format!("Failed to check out branch '{name}': {error}"))?;
        Defaults::set_current_branch(config, Some(name))
            .map_err(|error| format!("Failed to save current branch '{name}': {error}"))?;
        materialize_branch_head_assets(workspace, name, &previous_assets)?;
    } else if let Some(hash_name) = hash {
        if branch_exists(graph, hash_name)? {
            println!("Checking out branch {hash_name}");
            checkout(graph, hash_name)
                .map_err(|error| format!("Failed to check out branch '{hash_name}': {error}"))?;
            Defaults::set_current_branch(config, Some(hash_name))
                .map_err(|error| format!("Failed to save current branch '{hash_name}': {error}"))?;
            materialize_branch_head_assets(workspace, hash_name, &previous_assets)?;
        } else {
            let commit_hash =
                history_store.resolve_operation_hash(&CommitRef(hash_name.to_string()))?;
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
