//! Clone, push, and pull orchestration for Gen workspaces.
//!
//! This module is the bridge between Gen's CLI commands, the config database, the graph
//! database's native Dolt remote operations, and the GenHub client. `gen push` and
//! `gen pull` call [`execute_push`] and [`execute_pull`] from the CLI dispatcher. The
//! clone command creates the destination workspace and its canonical `origin` config
//! entry, then calls [`clone_into_workspace`]. Remote and branch selection follows the
//! explicit command arguments, the remote tracked by the branch, and finally the
//! workspace defaults.
//!
//! Graph history is transferred by Dolt rather than reconstructed by Gen. Clone asks
//! Dolt to clone the remote database; push sends the selected local branch, optionally
//! with force; and pull integrates the selected remote branch into its local branch.
//! After a successful push, Gen fetches the branch to refresh Dolt's remote-tracking ref.
//! After a successful push or pull, the config database records that the local branch
//! tracks the selected remote.
//!
//! A repository remote URL and an asset URI have separate meanings here. A repository
//! `file://` remote points directly to another Dolt database (or a workspace containing
//! `.gen/default.db`), so graph operations use that path without contacting GenHub and
//! no separate asset-byte transfer is attempted. An HTTP(S) repository remote stores its
//! canonical GenHub URL in config. Before each graph operation, Gen requests a scoped,
//! short-lived transfer capability, installs the returned URL as the graph database's
//! Dolt remote, performs the operation, and restores the canonical URL. An authorization
//! failure is retried once with a fresh capability. Failure to restore the canonical URL
//! is reported as a warning because the graph transfer may already have succeeded.
//!
//! Assets referenced by the transferred branch are handled after the graph operation.
//! Only asset records whose URI uses the `file://` scheme represent file bytes managed by
//! this transfer protocol. Asset URIs with other schemes, such as HTTP or S3, remain
//! external references in the graph database and are not uploaded to or downloaded from
//! GenHub. For an HTTP(S) repository remote, GenHub can return presigned URLs for the complete
//! branch asset history. Gen uses the previous and destination commits to transfer only newly
//! required versions. Push verifies each selected local file against its recorded checksum before
//! uploading it with an HTTP PUT. Clone and pull download each selected file to a temporary path
//! and verify its checksum. Only the asset version selected by the destination commit's
//! materialized view is moved to its logical workspace path. Superseded versions are retained
//! under `.gen/assets` using checksum-derived names. Stored `file://` paths are also resolved as
//! safe workspace-relative paths, including `.gen/outside_root` paths used to represent inputs
//! that originally came from outside the workspace.
//!
//! Pull records the branch commit from before the Dolt operation so downloads can
//! distinguish a clean old version from a local modification. If the destination still
//! matches the previous commit and the remote asset changed, the download replaces it as
//! an intended update. If the destination is untracked, locally modified, or otherwise
//! does not match the previous version, Gen preserves it and writes the downloaded bytes
//! beside it as `filename.conflict`, then `filename.conflict.N` as needed. An existing
//! conflict copy with the expected checksum is reused. The command warns the user and
//! leaves choosing the correct file to them.

use std::{
    collections::{HashMap, HashSet},
    error::Error,
    fs,
    io::{self, Write as _},
    path::{Path, PathBuf},
};

use gen_core::{
    DoltHashId, HashId, Sha256Hash,
    config::{DEFAULT_GRAPH_DB_NAME, Workspace},
};
use gen_models::{
    assets::{AssetRef, LocalAssetUri, materialization_destination_path},
    db::{ConfigConnection, GraphConnection},
    history::dolt::{
        active_branch, add_remote, branch_hash, checkout, clone_remote, fetch, hash_of, pull, push,
        push_force, remote_rows, set_remote_url,
    },
    operations::{Defaults, Remote, RemoteBranch, calculate_file_checksum},
};
use reqwest::blocking::{Body, Client};
use rusqlite::Error as SqlError;
use url::Url;

use crate::{
    commands::remote::{
        client::{
            AssetTransferRequest, CapabilityRequest, RemoteOperation, RepositoryRemote,
            acquire_asset_transfers, acquire_capability,
        },
        login_origin,
    },
    get_config_connection, get_connection_for_branch, get_raw_connection,
};

fn file_graph_url(remote_url: &str) -> Result<String, Box<dyn Error>> {
    let parsed = Url::parse(remote_url)?;
    let mut path = parsed
        .to_file_path()
        .map_err(|_| format!("Invalid file remote URL: {remote_url}"))?;
    if path.extension().and_then(|extension| extension.to_str()) != Some("db") {
        path = path.join(".gen").join(DEFAULT_GRAPH_DB_NAME);
    }
    Url::from_file_path(&path)
        .map(String::from)
        .map_err(|_| format!("Invalid file remote path: {}", path.display()).into())
}

fn transfer_url(
    remote: &Remote,
    operation: RemoteOperation,
    branch: Option<&str>,
    force: bool,
) -> Result<String, Box<dyn Error>> {
    if remote.url.starts_with("file://") {
        return file_graph_url(&remote.url);
    }
    let repository = RepositoryRemote::parse(&remote.url)?;
    let request = CapabilityRequest {
        operation,
        branch,
        force,
    };
    let capability = acquire_capability(&repository, &request, login_origin)?;
    Ok(capability.remote_url)
}

fn ensure_graph_remote(
    graph: &GraphConnection,
    remote_name: &str,
    remote_url: &str,
) -> Result<(), SqlError> {
    if remote_rows(graph)?
        .iter()
        .any(|remote| remote.name == remote_name)
    {
        set_remote_url(graph, remote_name, remote_url)
    } else {
        add_remote(graph, remote_name, remote_url)
    }
}

fn restore_canonical_url(graph: &GraphConnection, remote: &Remote) {
    if let Err(error) = set_remote_url(graph, &remote.name, &remote.url) {
        eprintln!(
            "Warning: failed to restore the canonical URL for graph remote '{}': {error}",
            remote.name
        );
    }
}

fn is_authorization_error(error: &SqlError) -> bool {
    matches!(
        error,
        SqlError::SqliteFailure(code, _) if code.extended_code == rusqlite::ffi::SQLITE_AUTH
    )
}

fn resolve_remote(
    config: &ConfigConnection,
    explicit_remote: Option<&str>,
    branch: &str,
) -> Result<Remote, Box<dyn Error>> {
    let remote_name = explicit_remote
        .map(str::to_string)
        .or_else(|| RemoteBranch::get_remote(config, branch))
        .or_else(|| Defaults::get_default_remote(config))
        .ok_or("No remote specified, tracked for this branch, or configured as default")?;
    Ok(Remote::get_by_name(config, &remote_name)?)
}

fn connect_persisted_branch(
    graph: &GraphConnection,
    config: &ConfigConnection,
) -> Result<Option<String>, SqlError> {
    let persisted_branch = Defaults::get_current_branch(config);
    if let Some(branch) = persisted_branch.as_deref()
        && active_branch(graph)? != branch
    {
        checkout(graph, branch)?;
    }
    Ok(persisted_branch)
}

fn run_graph_transfer(
    graph: &GraphConnection,
    remote: &Remote,
    operation: RemoteOperation,
    branch: &str,
    force: bool,
    mut transfer: impl FnMut() -> Result<(), SqlError>,
) -> Result<(), Box<dyn Error>> {
    if remote.url.starts_with("file://") {
        let remote_url = transfer_url(remote, operation, Some(branch), force)?;
        ensure_graph_remote(graph, &remote.name, &remote_url)?;
        let result = transfer();
        restore_canonical_url(graph, remote);
        return Ok(result?);
    }

    let mut last_error = None;
    for attempt in 0..2 {
        let remote_url = transfer_url(remote, operation, Some(branch), force)?;
        ensure_graph_remote(graph, &remote.name, &remote_url)?;
        match transfer() {
            Ok(()) => {
                restore_canonical_url(graph, remote);
                return Ok(());
            }
            Err(error) if attempt == 0 && is_authorization_error(&error) => {
                last_error = Some(error);
            }
            Err(error) => {
                restore_canonical_url(graph, remote);
                return Err(error.into());
            }
        }
    }
    restore_canonical_url(graph, remote);
    Err(last_error
        .expect("should retain authorization error")
        .into())
}

fn asset_checksum(asset: &AssetRef) -> Result<Sha256Hash, Box<dyn Error>> {
    asset
        .checksum
        .ok_or_else(|| format!("Local asset {} has no checksum", asset.id).into())
}

fn upload_asset(
    client: &Client,
    workspace: &Workspace,
    asset: &AssetRef,
    url: &str,
) -> Result<(), Box<dyn Error>> {
    let relative_path = LocalAssetUri::path_from_uri(&asset.uri)
        .ok_or_else(|| format!("Invalid local asset URI: {}", asset.uri))?;
    let expected_checksum = asset_checksum(asset)?;
    let uri_path = LocalAssetUri::repo_relative_destination_path(workspace, &relative_path)?;
    let source_path = if uri_path.is_file() {
        uri_path
    } else {
        materialization_destination_path(
            workspace,
            &asset.uri,
            Some(&expected_checksum),
            asset.logical_path.as_deref(),
        )?
    };
    let actual_checksum = calculate_file_checksum(&source_path).map_err(|error| {
        format!(
            "Unable to read asset {} at {}: {error}",
            asset.id,
            source_path.display()
        )
    })?;
    if actual_checksum != expected_checksum {
        return Err(format!(
            "Asset {} at {} does not match its recorded checksum",
            asset.id,
            source_path.display()
        )
        .into());
    }
    let file = fs::File::open(&source_path)?;
    let length = file.metadata()?.len();
    let response = client
        .put(url)
        .header("content-type", "application/octet-stream")
        .body(Body::sized(file, length))
        .send()
        .map_err(|error| error.without_url())?;
    if !response.status().is_success() {
        return Err(format!(
            "Asset {} upload failed with HTTP {}",
            asset.id,
            response.status()
        )
        .into());
    }
    Ok(())
}

#[derive(Debug, Eq, PartialEq)]
enum DownloadAssetOutcome {
    Unchanged,
    Downloaded,
    Conflict(PathBuf),
}

/// This is effectively a dirty file check. On clones/pulls we want to update
/// files if they match expected checksums. The previous asset set is cumulative
/// so a workspace that still contains any committed version
/// is safe to advance. Unknown contents remain a conflict and are never overwritten.
fn destination_matches_previous_asset(
    workspace: &Workspace,
    destination_path: &Path,
    existing_checksum: &Sha256Hash,
    previous_assets: &HashMap<HashId, AssetRef>,
) -> Result<bool, Box<dyn Error>> {
    for asset in previous_assets.values() {
        let checksum = asset_checksum(asset)?;
        if checksum != *existing_checksum {
            continue;
        }
        let asset_path = materialization_destination_path(
            workspace,
            &asset.uri,
            Some(&checksum),
            asset.logical_path.as_deref(),
        )?;
        if asset_path == destination_path {
            return Ok(true);
        }
    }
    Ok(false)
}

/// If a conflict exists for a file we are pulling/cloning, rename it as .conflict for user resolution
fn conflict_destination_path(
    destination_path: &Path,
    expected_checksum: &Sha256Hash,
) -> Result<(PathBuf, bool), Box<dyn Error>> {
    let file_name = destination_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("asset");
    for index in 0_usize.. {
        let suffix = if index == 0 {
            ".conflict".to_string()
        } else {
            format!(".conflict.{index}")
        };
        let candidate = destination_path.with_file_name(format!("{file_name}{suffix}"));
        if !candidate.exists() {
            return Ok((candidate, false));
        }
        if calculate_file_checksum(&candidate).is_ok_and(|checksum| checksum == *expected_checksum)
        {
            return Ok((candidate, true));
        }
    }
    unreachable!("conflict suffix space should not be exhausted")
}

/// On downloading an asset, if it is a file we compare the hash on disk to the hashes of the
/// asset from previous updates. If it matches, we replace it assuming it's a natural evolution
/// of the file. If the hash is unknown, we download the file and mark it as a conflict via a
/// .conflict extension. Historical assets that are not materialized at the destination commit are
/// stored in `.gen/assets` so they cannot replace the current workspace file.
fn download_asset(
    client: &Client,
    workspace: &Workspace,
    asset: &AssetRef,
    previous_assets: &HashMap<HashId, AssetRef>,
    // `None` stores a historical version under `.gen/assets` instead of its recorded logical path.
    destination_logical_path: Option<&str>,
    url: &str,
) -> Result<DownloadAssetOutcome, Box<dyn Error>> {
    let expected_checksum = asset_checksum(asset)?;
    let destination_path = materialization_destination_path(
        workspace,
        &asset.uri,
        Some(&expected_checksum),
        destination_logical_path,
    )?;
    if destination_path.exists()
        && calculate_file_checksum(&destination_path)
            .is_ok_and(|checksum| checksum == expected_checksum)
    {
        return Ok(DownloadAssetOutcome::Unchanged);
    }
    let existing_checksum = destination_path
        .exists()
        .then(|| calculate_file_checksum(&destination_path))
        .transpose()?;
    let intended_change = existing_checksum
        .as_ref()
        .map(|checksum| {
            destination_matches_previous_asset(
                workspace,
                &destination_path,
                checksum,
                previous_assets,
            )
        })
        .transpose()?
        .unwrap_or(false);
    let has_conflict = existing_checksum.is_some() && !intended_change;
    if let Some(parent) = destination_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file_name = destination_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("asset");
    let temporary_path = destination_path.with_file_name(format!(".{file_name}.{}.part", asset.id));
    let result = (|| {
        let mut response = client
            .get(url)
            .send()
            .map_err(|error| error.without_url())?;
        if !response.status().is_success() {
            return Err(format!(
                "Asset {} download failed with HTTP {}",
                asset.id,
                response.status()
            )
            .into());
        }
        let mut file = fs::File::create(&temporary_path)?;
        io::copy(&mut response, &mut file)?;
        file.flush()?;
        drop(file);
        if calculate_file_checksum(&temporary_path)? != expected_checksum {
            return Err(format!("Downloaded asset {} failed checksum validation", asset.id).into());
        }

        if has_conflict {
            let (conflict_path, already_downloaded) =
                conflict_destination_path(&destination_path, &expected_checksum)?;
            if already_downloaded {
                fs::remove_file(&temporary_path)?;
            } else {
                fs::rename(&temporary_path, &conflict_path)?;
            }
            return Ok(DownloadAssetOutcome::Conflict(conflict_path));
        }

        if destination_path.exists() {
            fs::remove_file(&destination_path)?;
        }
        fs::rename(&temporary_path, &destination_path)?;
        Ok::<DownloadAssetOutcome, Box<dyn Error>>(DownloadAssetOutcome::Downloaded)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary_path);
    }
    result
}

/// Transfers the asset versions needed to move from `previous_hash` to the selected branch state.
///
/// GenHub may advertise the branch's complete asset history, so this function filters those URLs
/// to versions absent from the previous state. The cumulative view supplies that transfer delta
/// and the checksums used for conflict detection, while the materialized view decides which of the
/// selected versions belongs at its logical workspace path instead of under `.gen/assets`.
fn transfer_assets(
    graph: &GraphConnection,
    workspace: &Workspace,
    remote: &Remote,
    operation: RemoteOperation,
    branch: &str,
    previous_hash: Option<&DoltHashId>,
) -> Result<(), Box<dyn Error>> {
    if remote.url.starts_with("file://") {
        return Ok(());
    }
    let repository = RepositoryRemote::parse(&remote.url)?;
    let response = acquire_asset_transfers(
        &repository,
        &AssetTransferRequest { operation, branch },
        login_origin,
    )?;

    let commit_hash = hash_of(graph, branch)?;
    let current_assets: HashMap<_, _> =
        AssetRef::get_cumulative_assets_at(graph, previous_hash, Some(&commit_hash))?
            .into_iter()
            .map(|asset| (asset.id, asset))
            .collect();
    let materialized_asset_ids: HashSet<_> =
        AssetRef::get_materialized_assets_at(graph, None, Some(&commit_hash))?
            .into_iter()
            .map(|asset| asset.id)
            .collect();
    let previous_assets = if let Some(previous_hash) = previous_hash {
        AssetRef::get_cumulative_assets_at(graph, None, Some(previous_hash))?
            .into_iter()
            .map(|asset| (asset.id, asset))
            .collect()
    } else {
        HashMap::new()
    };
    // These are assets we expect to be in the current batch of transfers
    let mut assets: HashMap<_, _> = current_assets
        .iter()
        .filter(|(id, _)| !previous_assets.contains_key(id))
        .map(|(id, asset)| (*id, asset.clone()))
        .collect();
    let client = Client::new();
    for transfer in response.assets {
        let Some(asset) = assets.remove(&transfer.id) else {
            if current_assets.contains_key(&transfer.id)
                || previous_assets.contains_key(&transfer.id)
            {
                continue;
            }
            return Err(format!(
                "GenHub returned an asset transfer not present on branch '{branch}': {}",
                transfer.id
            )
            .into());
        };
        match operation {
            RemoteOperation::Push => upload_asset(&client, workspace, &asset, &transfer.url)?,
            RemoteOperation::Clone | RemoteOperation::Pull => {
                let destination_logical_path = if materialized_asset_ids.contains(&asset.id) {
                    asset.logical_path.as_deref()
                } else {
                    None
                };
                if let DownloadAssetOutcome::Conflict(conflict_path) = download_asset(
                    &client,
                    workspace,
                    &asset,
                    &previous_assets,
                    destination_logical_path,
                    &transfer.url,
                )? {
                    let destination_path = materialization_destination_path(
                        workspace,
                        &asset.uri,
                        asset.checksum.as_ref(),
                        destination_logical_path,
                    )?;
                    eprintln!(
                        "Warning: remote asset conflicts with the local file at {}. The local file was preserved and the remote version was written to {}. Choose the correct version before continuing.",
                        destination_path.display(),
                        conflict_path.display()
                    );
                }
            }
        }
    }
    if !assets.is_empty() {
        return Err(format!(
            "GenHub omitted {} local asset transfer(s) for branch '{branch}'",
            assets.len()
        )
        .into());
    }
    Ok(())
}

pub fn clone_into_workspace(
    remote: &Remote,
    workspace: &Workspace,
) -> Result<String, Box<dyn Error>> {
    workspace.ensure_gen_dir();
    let graph_path = workspace.graph_db_path()?;
    let graph = get_raw_connection(graph_path)?;
    let attempt_count = if remote.url.starts_with("file://") {
        1
    } else {
        2
    };
    let mut clone_result = None;
    for attempt in 0..attempt_count {
        let remote_url = transfer_url(remote, RemoteOperation::Clone, None, false)?;
        match clone_remote(&graph, &remote_url) {
            Ok(()) => {
                clone_result = Some(Ok(()));
                break;
            }
            Err(error) => {
                let lost_authorization_code = matches!(
                    &error,
                    SqlError::SqliteFailure(code, Some(message))
                        if code.extended_code == rusqlite::ffi::SQLITE_ERROR
                            && message == "clone failed"
                );
                let should_retry = attempt == 0
                    && attempt_count > 1
                    && (is_authorization_error(&error) || lost_authorization_code);
                clone_result = Some(Err(error));
                if should_retry {
                    continue;
                }
                break;
            }
        }
    }
    restore_canonical_url(&graph, remote);
    clone_result.expect("should attempt clone at least once")?;
    let branch = active_branch(&graph)?;
    drop(graph);
    let graph = get_raw_connection(workspace.graph_db_path()?)?;
    transfer_assets(
        &graph,
        workspace,
        remote,
        RemoteOperation::Clone,
        &branch,
        None,
    )?;
    Ok(branch)
}

pub fn execute_push(
    workspace: &Workspace,
    explicit_remote: Option<&str>,
    explicit_branch: Option<&str>,
    force: bool,
) -> Result<(), Box<dyn Error>> {
    let config = get_config_connection(Some(workspace.gen_db_path()?))?;
    let intended_branch = Defaults::get_current_branch(&config);
    let graph = get_connection_for_branch(workspace.graph_db_path()?, intended_branch.as_deref())?;
    let persisted_branch = connect_persisted_branch(&graph, &config)?;
    let branch = explicit_branch
        .map(str::to_string)
        .or(persisted_branch)
        .unwrap_or(active_branch(&graph)?);
    let remote = resolve_remote(&config, explicit_remote, &branch)?;
    // A missing or stale tracking ref only makes the transfer conservatively include more assets.
    // Force pushes cannot use the tracking ref as a lower bound because they may replace history.
    let tracking_ref = format!("{}/{branch}", remote.name);
    let previous_hash = (!force)
        .then(|| hash_of(&graph, &tracking_ref).ok())
        .flatten();
    run_graph_transfer(
        &graph,
        &remote,
        RemoteOperation::Push,
        &branch,
        force,
        || {
            if force {
                push_force(&graph, &remote.name, &branch)?;
            } else {
                push(&graph, &remote.name, &branch)?;
            }
            Ok(())
        },
    )?;
    transfer_assets(
        &graph,
        workspace,
        &remote,
        RemoteOperation::Push,
        &branch,
        previous_hash.as_ref(),
    )?;
    if let Err(error) = run_graph_transfer(
        &graph,
        &remote,
        RemoteOperation::Pull,
        &branch,
        false,
        || fetch(&graph, &remote.name, Some(&branch)),
    ) {
        eprintln!(
            "Warning: push completed, but failed to refresh remote-tracking branch '{}/{}': {error}",
            remote.name, branch,
        );
    }
    RemoteBranch::set_remote_validated(&config, &branch, Some(&remote.name))?;
    println!("Pushed branch '{branch}' to '{}'.", remote.name);
    Ok(())
}

pub fn execute_pull(
    workspace: &Workspace,
    explicit_remote: Option<&str>,
    explicit_branch: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    let config = get_config_connection(Some(workspace.gen_db_path()?))?;
    let graph_path = workspace.graph_db_path()?;
    if !graph_path.exists() {
        return Err("Cannot pull without an existing graph database; use `gen clone` to initialize a workspace from a remote repository".into());
    }
    let intended_branch = Defaults::get_current_branch(&config);
    let graph = get_connection_for_branch(&graph_path, intended_branch.as_deref())?;
    let persisted_branch = connect_persisted_branch(&graph, &config)?;
    let branch = explicit_branch
        .map(str::to_string)
        .or(persisted_branch)
        .unwrap_or(active_branch(&graph)?);
    let remote = resolve_remote(&config, explicit_remote, &branch)?;
    let previous_hash = branch_hash(&graph, &branch)?;
    run_graph_transfer(
        &graph,
        &remote,
        RemoteOperation::Pull,
        &branch,
        false,
        || pull(&graph, &remote.name, &branch),
    )?;
    transfer_assets(
        &graph,
        workspace,
        &remote,
        RemoteOperation::Pull,
        &branch,
        previous_hash.as_ref(),
    )?;
    RemoteBranch::set_remote_validated(&config, &branch, Some(&remote.name))?;
    println!("Pulled branch '{branch}' from '{}'.", remote.name);
    Ok(())
}

pub fn clone_destination_name(remote_url: &str) -> Result<String, Box<dyn Error>> {
    if remote_url.starts_with("http://") || remote_url.starts_with("https://") {
        return Ok(RepositoryRemote::parse(remote_url)?.slug().to_string());
    }
    let parsed = Url::parse(remote_url)?;
    let path = parsed
        .to_file_path()
        .map_err(|_| format!("Invalid file remote URL: {remote_url}"))?;
    path.file_name()
        .and_then(|name| name.to_str())
        .map(str::to_string)
        .ok_or_else(|| "Remote URL has no destination name".into())
}

pub fn canonical_remote_url(remote_url: &str) -> Result<String, Box<dyn Error>> {
    if remote_url.starts_with("http://") || remote_url.starts_with("https://") {
        Ok(RepositoryRemote::parse(remote_url)?
            .canonical_url()
            .to_string())
    } else {
        Ok(remote_url.to_string())
    }
}

pub fn clone_destination_path(
    parent: &Workspace,
    remote_url: &str,
) -> Result<PathBuf, Box<dyn Error>> {
    Ok(parent.base_dir().join(clone_destination_name(remote_url)?))
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        env,
        ffi::OsString,
        fs,
        io::{Cursor, Read as _, Write as _},
        net::TcpListener,
        sync::Mutex,
        thread,
    };

    use gen_core::config::Workspace;
    use gen_models::{
        assets::{AssetRef, AssetRole, LocalAssetUri},
        collection::Collection,
        db::GraphConnection,
        history::dolt::{commit_all, hash_of, remote_rows, remove_remote},
        operations::{Defaults, Remote, calculate_reader_checksum},
    };
    use reqwest::blocking::Client;
    use rusqlite::{Connection, Error as SqlError};
    use serde_json::json;
    use tempfile::tempdir;

    use super::{
        DownloadAssetOutcome, RemoteOperation, canonical_remote_url, clone_destination_name,
        download_asset, execute_push, file_graph_url, resolve_remote, run_graph_transfer,
        transfer_assets,
    };
    use crate::{get_config_connection, get_connection};

    static ENVIRONMENT_LOCK: Mutex<()> = Mutex::new(());

    mod clone {
        use std::{
            io::{self, Read as _, Write as _},
            net::TcpListener,
            sync::{
                Arc,
                atomic::{AtomicBool, Ordering},
            },
            thread,
            time::Duration,
        };

        use gen_core::config::Workspace;
        use gen_models::{collection::Collection, history::dolt::commit_all, operations::Remote};
        use tempfile::{TempDir, tempdir};
        use url::Url;

        use super::super::clone_into_workspace;
        use crate::get_connection;

        struct CloneFixture {
            _temp: TempDir,
            expired_server: thread::JoinHandle<()>,
            capability_stop: Arc<AtomicBool>,
            capability_server: thread::JoinHandle<Vec<String>>,
            remote: Remote,
            workspace: Workspace,
        }

        impl CloneFixture {
            fn new() -> Self {
                let temp = tempdir().expect("should create clone fixture directory");
                let source_path = temp.path().join("source.db");
                let source =
                    get_connection(&source_path).expect("should create source graph database");
                Collection::create(&source, "clone-fixture")
                    .expect("should create source graph state");
                commit_all(&source, "seed clone fixture")
                    .expect("should commit source graph state");
                drop(source);
                let valid_remote_url = Url::from_file_path(&source_path)
                    .expect("should convert source graph path to a file URL")
                    .to_string();

                let (expired_remote_url, expired_server) = serve_expired_remote();
                let capability_stop = Arc::new(AtomicBool::new(false));
                let (remote, capability_server) = serve_capabilities(
                    &expired_remote_url,
                    &valid_remote_url,
                    Arc::clone(&capability_stop),
                );
                let workspace = Workspace::new(temp.path().join("clone"));

                Self {
                    _temp: temp,
                    expired_server,
                    capability_stop,
                    capability_server,
                    remote,
                    workspace,
                }
            }
        }

        fn serve_expired_remote() -> (String, thread::JoinHandle<()>) {
            let listener =
                TcpListener::bind("127.0.0.1:0").expect("should bind expired capability server");
            let address = listener
                .local_addr()
                .expect("should read expired capability server address");
            let handle = thread::spawn(move || {
                let (mut stream, _) = listener
                    .accept()
                    .expect("should accept expired capability request");
                let mut request = [0_u8; 4096];
                let _ = stream
                    .read(&mut request)
                    .expect("should read expired capability request");
                stream
                    .write_all(
                        b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    )
                    .expect("should write expired capability response");
            });
            (format!("http://{address}/origin.db"), handle)
        }

        fn serve_capabilities(
            expired_remote_url: &str,
            valid_remote_url: &str,
            stop: Arc<AtomicBool>,
        ) -> (Remote, thread::JoinHandle<Vec<String>>) {
            let listener = TcpListener::bind("127.0.0.1:0").expect("should bind capability server");
            listener
                .set_nonblocking(true)
                .expect("should make capability server nonblocking");
            let address = listener
                .local_addr()
                .expect("should read capability server address");
            let expired_remote_url = expired_remote_url.to_string();
            let valid_remote_url = valid_remote_url.to_string();
            let handle = thread::spawn(move || {
                let mut requests = Vec::new();
                let mut capability_count = 0;
                while requests.len() < 3 && !stop.load(Ordering::Acquire) {
                    let (mut stream, _) = match listener.accept() {
                        Ok(connection) => connection,
                        Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                            thread::sleep(Duration::from_millis(10));
                            continue;
                        }
                        Err(error) => panic!("should accept capability request: {error}"),
                    };
                    let mut request = [0_u8; 8192];
                    let read = stream
                        .read(&mut request)
                        .expect("should read capability request");
                    let request = String::from_utf8_lossy(&request[..read]).into_owned();
                    let body = if request
                        .starts_with("POST /api/repos/alice/example/remote-capability ")
                    {
                        capability_count += 1;
                        let remote_url = if capability_count == 1 {
                            &expired_remote_url
                        } else {
                            &valid_remote_url
                        };
                        format!(
                            "{{\"remote_url\":\"{remote_url}\",\
                             \"expires_at\":\"2030-01-01T00:00:00Z\",\
                             \"default_branch\":\"main\"}}"
                        )
                    } else if request.starts_with("POST /api/repos/alice/example/asset-transfers ")
                    {
                        "{\"assets\":[]}".to_string()
                    } else {
                        panic!("unexpected clone fixture request: {request}");
                    };
                    requests.push(request);
                    write!(
                        stream,
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    )
                    .expect("should write capability response");
                }
                requests
            });
            (
                Remote {
                    name: "origin".to_string(),
                    url: format!("http://{address}/api/repos/alice/example"),
                },
                handle,
            )
        }

        #[test]
        fn test_clone_rejected_capability_retries_with_a_fresh_capability() {
            let CloneFixture {
                _temp,
                expired_server,
                capability_stop,
                capability_server,
                remote,
                workspace,
            } = CloneFixture::new();

            let result = clone_into_workspace(&remote, &workspace);
            capability_stop.store(true, Ordering::Release);
            expired_server
                .join()
                .expect("expired capability server should finish");
            let requests = capability_server
                .join()
                .expect("capability server should finish");
            let capability_requests = requests
                .iter()
                .filter(|request| request.contains("/remote-capability "))
                .count();

            assert_eq!(
                capability_requests, 2,
                "clone should request a fresh capability after the first capability is rejected; result={result:?}"
            );
            assert_eq!(result.expect("clone retry should succeed"), "main");
        }
    }

    fn test_asset(contents: &[u8], logical_path: &str, created_on: i64) -> AssetRef {
        let checksum = calculate_reader_checksum(Cursor::new(contents)).expect("should checksum");
        let uri = LocalAssetUri::asset_uri(logical_path);
        let role = AssetRole::Input;
        AssetRef {
            id: AssetRef::id_hash(
                &uri,
                "text",
                Some(&checksum),
                &role,
                Some(logical_path),
                None,
            ),
            uri,
            file_type: "text".to_string(),
            checksum: Some(checksum),
            size: Some(i64::try_from(contents.len()).expect("asset should fit in i64")),
            role,
            logical_path: Some(logical_path.to_string()),
            name: None,
            created_on,
        }
    }

    fn serve_asset(contents: &[u8]) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("should bind asset server");
        let address = listener
            .local_addr()
            .expect("should read asset server address");
        let contents = contents.to_vec();
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("should accept asset request");
            let mut request = [0_u8; 4096];
            let _ = stream
                .read(&mut request)
                .expect("should read asset request");
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                contents.len()
            )
            .expect("should write asset response headers");
            stream
                .write_all(&contents)
                .expect("should write asset response body");
        });
        (format!("http://{address}/asset"), handle)
    }

    struct EnvironmentGuard {
        name: &'static str,
        previous: Option<OsString>,
    }

    impl EnvironmentGuard {
        fn set(name: &'static str, value: &str) -> Self {
            let previous = env::var_os(name);
            unsafe { env::set_var(name, value) };
            Self { name, previous }
        }
    }

    impl Drop for EnvironmentGuard {
        fn drop(&mut self) {
            if let Some(previous) = &self.previous {
                unsafe { env::set_var(self.name, previous) };
            } else {
                unsafe { env::remove_var(self.name) };
            }
        }
    }

    #[test]
    fn test_download_asset_preserves_an_untracked_workspace_file() {
        let temp = tempdir().expect("should create workspace");
        let workspace = Workspace::new(temp.path());
        workspace.ensure_gen_dir();
        let destination = temp.path().join("reference.fa");
        fs::write(&destination, b"local untracked\n").expect("should write local file");
        let remote_contents = b"remote version\n";
        let remote_asset = test_asset(remote_contents, "reference.fa", 2);
        let (url, server) = serve_asset(remote_contents);

        let outcome = download_asset(
            &Client::new(),
            &workspace,
            &remote_asset,
            &HashMap::new(),
            remote_asset.logical_path.as_deref(),
            &url,
        )
        .expect("should download conflicting asset");
        server.join().expect("asset server should finish");

        let conflict = temp.path().join("reference.fa.conflict");
        assert_eq!(outcome, DownloadAssetOutcome::Conflict(conflict.clone()));
        assert_eq!(fs::read(&destination).unwrap(), b"local untracked\n");
        assert_eq!(fs::read(conflict).unwrap(), remote_contents);
    }

    #[test]
    fn test_download_asset_replaces_an_unchanged_tracked_file() {
        let temp = tempdir().expect("should create workspace");
        let workspace = Workspace::new(temp.path());
        workspace.ensure_gen_dir();
        let destination = temp.path().join("reference.fa");
        let previous_contents = b"previous version\n";
        fs::write(&destination, previous_contents).expect("should write previous file");
        let previous_asset = test_asset(previous_contents, "reference.fa", 1);
        let previous_assets = HashMap::from([(previous_asset.id, previous_asset)]);
        let remote_contents = b"remote version\n";
        let remote_asset = test_asset(remote_contents, "reference.fa", 2);
        let (url, server) = serve_asset(remote_contents);

        let outcome = download_asset(
            &Client::new(),
            &workspace,
            &remote_asset,
            &previous_assets,
            remote_asset.logical_path.as_deref(),
            &url,
        )
        .expect("should replace unchanged tracked file");
        server.join().expect("asset server should finish");

        assert_eq!(outcome, DownloadAssetOutcome::Downloaded);
        assert_eq!(fs::read(&destination).unwrap(), remote_contents);
        assert!(!temp.path().join("reference.fa.conflict").exists());
    }

    #[test]
    fn test_download_asset_replaces_any_known_previous_version() {
        let temp = tempdir().expect("should create workspace");
        let workspace = Workspace::new(temp.path());
        workspace.ensure_gen_dir();
        let destination = temp.path().join("reference.fa");
        let older_contents = b"older version\n";
        fs::write(&destination, older_contents).expect("should write older managed file");
        let older_asset = test_asset(older_contents, "reference.fa", 1);
        let newer_asset = test_asset(b"newer version\n", "reference.fa", 2);
        let previous_assets =
            HashMap::from([(older_asset.id, older_asset), (newer_asset.id, newer_asset)]);
        let remote_contents = b"remote version\n";
        let remote_asset = test_asset(remote_contents, "reference.fa", 3);
        let (url, server) = serve_asset(remote_contents);

        let outcome = download_asset(
            &Client::new(),
            &workspace,
            &remote_asset,
            &previous_assets,
            remote_asset.logical_path.as_deref(),
            &url,
        )
        .expect("should replace any previously managed version");
        server.join().expect("asset server should finish");

        assert_eq!(outcome, DownloadAssetOutcome::Downloaded);
        assert_eq!(fs::read(&destination).unwrap(), remote_contents);
        assert!(!temp.path().join("reference.fa.conflict").exists());
    }

    #[test]
    fn test_download_asset_preserves_a_dirty_tracked_file() {
        let temp = tempdir().expect("should create workspace");
        let workspace = Workspace::new(temp.path());
        workspace.ensure_gen_dir();
        let destination = temp.path().join("reference.fa");
        let previous_contents = b"previous version\n";
        let previous_asset = test_asset(previous_contents, "reference.fa", 1);
        let previous_assets = HashMap::from([(previous_asset.id, previous_asset)]);
        fs::write(&destination, b"local edits\n").expect("should write dirty local file");
        let remote_contents = b"remote version\n";
        let remote_asset = test_asset(remote_contents, "reference.fa", 2);
        let (url, server) = serve_asset(remote_contents);

        let outcome = download_asset(
            &Client::new(),
            &workspace,
            &remote_asset,
            &previous_assets,
            remote_asset.logical_path.as_deref(),
            &url,
        )
        .expect("should download conflicting asset");
        server.join().expect("asset server should finish");

        let conflict = temp.path().join("reference.fa.conflict");
        assert_eq!(outcome, DownloadAssetOutcome::Conflict(conflict.clone()));
        assert_eq!(fs::read(&destination).unwrap(), b"local edits\n");
        assert_eq!(fs::read(conflict).unwrap(), remote_contents);
    }

    #[test]
    fn test_pull_transfers_only_assets_after_previous_hash() {
        let temp = tempdir().expect("should create transfer workspace");
        let workspace = Workspace::new(temp.path());
        workspace.ensure_gen_dir();
        let graph =
            get_connection(workspace.graph_db_path().unwrap()).expect("should open graph database");
        let previous_contents = b"previous version\n";
        let previous_asset = test_asset(previous_contents, "reference.fa", 1);
        AssetRef::create(&graph, &previous_asset).expect("should insert previous asset");
        let previous_hash =
            commit_all(&graph, "add previous asset").expect("should commit previous asset");
        fs::write(temp.path().join("reference.fa"), previous_contents)
            .expect("should materialize previous asset");

        let current_contents = b"current version\n";
        let current_asset = test_asset(current_contents, "reference.fa", 2);
        AssetRef::create(&graph, &current_asset).expect("should insert current asset");
        commit_all(&graph, "add current asset").expect("should commit current asset");

        let (current_url, asset_server) = serve_asset(current_contents);
        let transfer_listener =
            TcpListener::bind("127.0.0.1:0").expect("should bind transfer server");
        let transfer_address = transfer_listener
            .local_addr()
            .expect("should read transfer server address");
        let response_body = json!({
            "assets": [
                {
                    "id": previous_asset.id,
                    "url": "http://127.0.0.1:1/should-not-transfer"
                },
                { "id": current_asset.id, "url": current_url }
            ]
        })
        .to_string();
        let transfer_server = thread::spawn(move || {
            let (mut stream, _) = transfer_listener
                .accept()
                .expect("should accept transfer request");
            let mut request = [0_u8; 8192];
            let read = stream
                .read(&mut request)
                .expect("should read transfer request");
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response_body}",
                response_body.len()
            )
            .expect("should write transfer response");
            String::from_utf8_lossy(&request[..read]).into_owned()
        });
        let remote = Remote {
            name: "origin".to_string(),
            url: format!("http://{transfer_address}/api/repos/alice/example"),
        };

        transfer_assets(
            &graph,
            &workspace,
            &remote,
            RemoteOperation::Pull,
            "main",
            Some(&previous_hash),
        )
        .expect("should transfer only the asset delta");
        let transfer_request = transfer_server
            .join()
            .expect("transfer server should finish");
        asset_server.join().expect("asset server should finish");

        assert!(transfer_request.contains("\"operation\":\"pull\""));
        assert_eq!(
            fs::read(temp.path().join("reference.fa")).unwrap(),
            current_contents
        );
    }

    #[test]
    fn test_file_remote_resolves_graph_database() {
        assert_eq!(
            file_graph_url("file:///tmp/example").unwrap(),
            "file:///tmp/example/.gen/default.db"
        );
    }

    #[test]
    fn test_clone_destination_names() {
        assert_eq!(
            clone_destination_name("https://genhub.bio/api/repos/alice/example").unwrap(),
            "example"
        );
        assert_eq!(
            clone_destination_name("file:///tmp/example").unwrap(),
            "example"
        );
    }

    #[test]
    fn test_http_remote_is_stored_canonically() {
        assert_eq!(
            canonical_remote_url("https://genhub.bio/repos/alice/example").unwrap(),
            "https://genhub.bio/api/repos/alice/example"
        );
    }

    #[test]
    fn test_remote_resolution_prefers_explicit_then_branch_then_default() {
        let temp = tempdir().expect("should create remote resolution directory");
        let workspace = Workspace::new(temp.path());
        workspace.ensure_gen_dir();
        let config = get_config_connection(Some(workspace.gen_db_path().unwrap()))
            .expect("should open config database");
        for name in ["default", "tracked", "explicit"] {
            Remote::create(&config, name, &format!("file:///tmp/{name}"))
                .expect("should create remote");
        }
        Defaults::set_default_remote(&config, Some("default")).expect("should set default remote");
        gen_models::operations::RemoteBranch::set_remote_validated(
            &config,
            "feature",
            Some("tracked"),
        )
        .expect("should set tracked remote");

        assert_eq!(
            resolve_remote(&config, Some("explicit"), "feature")
                .unwrap()
                .name,
            "explicit"
        );
        assert_eq!(
            resolve_remote(&config, None, "feature").unwrap().name,
            "tracked"
        );
        assert_eq!(
            resolve_remote(&config, None, "main").unwrap().name,
            "default"
        );
    }

    #[test]
    fn test_authorization_failure_retries_with_a_fresh_capability() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("should bind capability server");
        let address = listener
            .local_addr()
            .expect("should read capability server address");
        let server = thread::spawn(move || {
            for attempt in 0..2 {
                let (mut stream, _) = listener.accept().expect("should accept capability request");
                let mut request = [0_u8; 8192];
                let _ = stream
                    .read(&mut request)
                    .expect("should read capability request");
                let body = format!(
                    "{{\"remote_url\":\"http://127.0.0.1:1/transfer-{attempt}\",\
                     \"expires_at\":\"2030-01-01T00:00:00Z\",\
                     \"default_branch\":\"main\"}}"
                );
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                    body.len()
                )
                .expect("should write capability response");
            }
        });

        let connection = Connection::open_in_memory().expect("should open graph database");
        let graph = GraphConnection(connection);
        let remote = Remote {
            name: "origin".to_string(),
            url: format!("http://{address}/api/repos/alice/example"),
        };
        let mut attempts = 0;
        run_graph_transfer(
            &graph,
            &remote,
            RemoteOperation::Pull,
            "main",
            false,
            || {
                attempts += 1;
                if attempts == 1 {
                    Err(SqlError::SqliteFailure(
                        rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_AUTH),
                        Some("expired capability".to_string()),
                    ))
                } else {
                    Ok(())
                }
            },
        )
        .expect("should retry authorization failure");
        server.join().expect("capability server should finish");

        assert_eq!(attempts, 2);
        let remotes = remote_rows(&graph).expect("should read restored canonical URL");
        assert!(
            remotes
                .iter()
                .any(|graph_remote| graph_remote.name == "origin" && graph_remote.url == remote.url)
        );
    }

    #[test]
    fn test_push_uploads_assets_before_tracking_fetch() {
        let _environment_lock = ENVIRONMENT_LOCK
            .lock()
            .expect("should lock process environment");
        let _api_key = EnvironmentGuard::set("GENHUB_API_KEY", "push-test-key");
        let listener = TcpListener::bind("127.0.0.1:0").expect("should bind capability server");
        let address = listener
            .local_addr()
            .expect("should read capability server address");
        let temp = tempdir().expect("should create push test directory");
        let remote_graph = temp.path().join("remote.db");
        let transfer_url = format!("file://{}", remote_graph.display());
        let server = thread::spawn(move || {
            let mut requests = Vec::new();
            for request_index in 0..3 {
                let (mut stream, _) = listener.accept().expect("should accept capability request");
                let mut request = [0_u8; 8192];
                let read = stream
                    .read(&mut request)
                    .expect("should read capability request");
                requests.push(String::from_utf8_lossy(&request[..read]).into_owned());
                let body = if request_index == 1 {
                    "{\"assets\":[]}".to_string()
                } else {
                    format!(
                        "{{\"remote_url\":\"{transfer_url}\",\
                         \"expires_at\":\"2030-01-01T00:00:00Z\",\
                         \"default_branch\":\"main\"}}"
                    )
                };
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                    body.len()
                )
                .expect("should write capability response");
            }
            requests
        });

        let workspace = Workspace::new(temp.path().join("local"));
        workspace.ensure_gen_dir();
        let config = get_config_connection(Some(workspace.gen_db_path().unwrap()))
            .expect("should open push config");
        let graph = get_connection(workspace.graph_db_path().unwrap()).expect("should open graph");
        Collection::create(&graph, "push-fixture").expect("should create push fixture");
        commit_all(&graph, "push fixture").expect("should commit push fixture");
        Remote::create(
            &config,
            "origin",
            &format!("http://{address}/api/repos/alice/example"),
        )
        .expect("should configure origin");
        Defaults::set_default_remote(&config, Some("origin")).expect("should set default remote");
        drop(graph);
        drop(config);

        execute_push(&workspace, None, None, false).expect("push should succeed");
        let requests = server.join().expect("capability server should finish");

        assert_eq!(requests.len(), 3);
        assert!(requests[0].contains("\"operation\":\"push\""));
        assert!(requests[1].starts_with("POST /api/repos/alice/example/asset-transfers "));
        assert!(requests[1].contains("\"operation\":\"push\""));
        assert!(requests[2].contains("\"operation\":\"pull\""));
        let graph =
            get_connection(workspace.graph_db_path().unwrap()).expect("should reopen graph");
        let local_hash = hash_of(&graph, "main").expect("should query local branch");
        let tracking_hash = hash_of(&graph, "origin/main").expect("should query tracking branch");
        assert_eq!(tracking_hash, local_hash);
    }

    #[test]
    fn test_push_succeeds_when_tracking_fetch_fails() {
        let _environment_lock = ENVIRONMENT_LOCK
            .lock()
            .expect("should lock process environment");
        let _api_key = EnvironmentGuard::set("GENHUB_API_KEY", "push-test-key");
        let listener = TcpListener::bind("127.0.0.1:0").expect("should bind capability server");
        let address = listener
            .local_addr()
            .expect("should read capability server address");
        let temp = tempdir().expect("should create push test directory");
        let remote_graph = temp.path().join("remote.db");
        let transfer_url = format!("file://{}", remote_graph.display());
        let server = thread::spawn(move || {
            let mut requests = Vec::new();
            for request_index in 0..3 {
                let (mut stream, _) = listener.accept().expect("should accept capability request");
                let mut request = [0_u8; 8192];
                let read = stream
                    .read(&mut request)
                    .expect("should read capability request");
                requests.push(String::from_utf8_lossy(&request[..read]).into_owned());
                if request_index < 2 {
                    let body = if request_index == 0 {
                        format!(
                            "{{\"remote_url\":\"{transfer_url}\",\
                             \"expires_at\":\"2030-01-01T00:00:00Z\",\
                             \"default_branch\":\"main\"}}"
                        )
                    } else {
                        "{\"assets\":[]}".to_string()
                    };
                    write!(
                        stream,
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                        body.len()
                    )
                    .expect("should write successful response");
                } else {
                    let body = "tracking fetch unavailable";
                    write!(
                        stream,
                        "HTTP/1.1 500 Internal Server Error\r\nContent-Type: text/plain\r\nContent-Length: {}\r\n\r\n{body}",
                        body.len()
                    )
                    .expect("should write failed response");
                }
            }
            requests
        });

        let workspace = Workspace::new(temp.path().join("local"));
        workspace.ensure_gen_dir();
        let config = get_config_connection(Some(workspace.gen_db_path().unwrap()))
            .expect("should open push config");
        let graph = get_connection(workspace.graph_db_path().unwrap()).expect("should open graph");
        Collection::create(&graph, "push-fixture").expect("should create push fixture");
        commit_all(&graph, "push fixture").expect("should commit push fixture");
        Remote::create(
            &config,
            "origin",
            &format!("http://{address}/api/repos/alice/example"),
        )
        .expect("should configure origin");
        Defaults::set_default_remote(&config, Some("origin")).expect("should set default remote");
        drop(graph);
        drop(config);

        execute_push(&workspace, None, None, false)
            .expect("tracking fetch failure should not fail push");
        let requests = server.join().expect("capability server should finish");

        assert_eq!(requests.len(), 3);
        assert!(requests[0].contains("\"operation\":\"push\""));
        assert!(requests[1].starts_with("POST /api/repos/alice/example/asset-transfers "));
        assert!(requests[1].contains("\"operation\":\"push\""));
        assert!(requests[2].contains("\"operation\":\"pull\""));
    }

    #[test]
    fn test_restoration_failure_is_nonfatal_and_the_next_transfer_heals_it() {
        let graph = GraphConnection(Connection::open_in_memory().expect("should open graph"));
        let remote = Remote {
            name: "origin".to_string(),
            url: "file:///tmp/canonical-remote.db".to_string(),
        };

        run_graph_transfer(
            &graph,
            &remote,
            RemoteOperation::Pull,
            "main",
            false,
            || {
                remove_remote(&graph, "origin")?;
                Ok(())
            },
        )
        .expect("successful transfer should survive restoration failure");
        assert!(
            remote_rows(&graph)
                .expect("should query remotes")
                .is_empty()
        );

        run_graph_transfer(
            &graph,
            &remote,
            RemoteOperation::Pull,
            "main",
            false,
            || Ok(()),
        )
        .expect("next transfer should recreate the missing remote");
        let remotes = remote_rows(&graph).expect("should query remotes");
        assert!(
            remotes
                .iter()
                .any(|graph_remote| graph_remote.name == "origin" && graph_remote.url == remote.url)
        );
    }

    #[test]
    fn test_restore_canonical_url_does_not_recreate_a_missing_graph_remote() {
        let graph = GraphConnection(Connection::open_in_memory().expect("should open graph"));
        let remote = Remote {
            name: "origin".to_string(),
            url: "https://genhub.bio/api/repos/alice/example".to_string(),
        };

        super::restore_canonical_url(&graph, &remote);

        assert!(
            remote_rows(&graph)
                .expect("should query remotes")
                .is_empty()
        );
    }
}
