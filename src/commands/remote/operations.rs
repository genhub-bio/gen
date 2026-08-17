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
//! `.gen/default.db`), so graph operations use that path directly and transfers happen
//! directly between the workspaces. For an HTTP(S) repository remote, Gen requests a scoped,
//! short-lived transfer capability, installs the returned URL as the graph database's Dolt
//! remote, performs the operation, and restores the canonical URL. An authorization failure
//! is retried once with a fresh capability. Failure to restore the canonical URL is reported
//! as a warning because the graph transfer may already have succeeded and will be replaced on
//! the next attempt.
//!
//! Assets referenced by the transferred branch are handled after the graph operation.
//! Only asset records whose URI uses the `file://` scheme represent file bytes managed by
//! this transfer protocol. Asset URIs with other schemes, such as HTTP or S3, remain
//! external references in the graph database and are not uploaded to or downloaded from
//! GenHub. For an HTTP(S) repository remote, GenHub can return presigned URLs for the complete
//! branch asset history. Gen uses the previous and destination commits to transfer only newly
//! required versions. Push verifies each selected local file against its recorded checksum before
//! uploading it with an HTTP PUT. Clone and pull download and checksum-verify each selected file
//! into `.gen/assets`. Only the asset version selected by the destination commit's materialized
//! view is copied from that versioned store to its logical workspace path. Superseded versions
//! remain only under `.gen/assets` using checksum-derived names. Stored `file://` paths are also
//! resolved as safe workspace-relative paths, including `.gen/outside_root` paths used to
//! represent inputs that originally came from outside the workspace.
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
    fs::{self, OpenOptions},
    io::{self, BufReader, Read as _, Write as _},
    path::{Path, PathBuf},
};

use base64::{Engine as _, engine::general_purpose};
use crc32c::crc32c_append;
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
use md5::Md5;
use reqwest::{
    StatusCode,
    blocking::{Body, Client, Response},
    header::{CONTENT_RANGE, RANGE},
};
use rusqlite::Error as SqlError;
use sha2::{Digest as _, Sha256};
use url::Url;

use crate::{
    commands::remote::{
        client::{
            AssetTransferCompletionRequest, AssetTransferRequest, AssetUploadReceipt,
            CapabilityRequest, RemoteOperation, RepositoryRemote, acquire_asset_transfers,
            acquire_capability, complete_asset_transfers,
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

/// Resolves a `file://` repository remote to the Gen workspace that owns its asset store.
///
/// Graph transfer uses [`file_graph_url`] to address the Dolt database itself. Asset transfer uses
/// the surrounding workspace so [`copy_to_versioned_store`] can reach `.gen/assets` on both sides.
fn file_remote_workspace(remote_url: &str) -> Result<Workspace, Box<dyn Error>> {
    let parsed = Url::parse(remote_url)?;
    let path = parsed
        .to_file_path()
        .map_err(|_| format!("Invalid file remote URL: {remote_url}"))?;
    let repo_root = if path.extension().and_then(|extension| extension.to_str()) == Some("db") {
        let gen_dir = path.parent().ok_or_else(|| {
            format!(
                "File remote database has no parent directory: {}",
                path.display()
            )
        })?;
        if gen_dir.file_name().and_then(|name| name.to_str()) != Some(".gen") {
            return Err(format!(
                "File remote database is not inside a Gen workspace: {}",
                path.display()
            )
            .into());
        }
        gen_dir.parent().ok_or_else(|| {
            format!(
                "File remote .gen directory has no workspace root: {}",
                gen_dir.display()
            )
        })?
    } else {
        &path
    };
    let workspace = Workspace::new(repo_root);
    let resolved_root = workspace.repo_root()?;
    if resolved_root != repo_root {
        return Err(format!(
            "File remote is not a Gen workspace root: {}",
            repo_root.display()
        )
        .into());
    }
    Ok(workspace)
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

fn calculate_upload_checksums(path: &Path) -> Result<(Sha256Hash, String, String), std::io::Error> {
    let file = fs::File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut sha256 = Sha256::new();
    let mut md5 = Md5::new();
    let mut crc32c = 0;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let length = reader.read(&mut buffer)?;
        if length == 0 {
            break;
        }
        sha256.update(&buffer[..length]);
        md5.update(&buffer[..length]);
        crc32c = crc32c_append(crc32c, &buffer[..length]);
    }
    Ok((
        Sha256Hash(sha256.finalize().into()),
        general_purpose::STANDARD.encode(md5.finalize()),
        general_purpose::STANDARD.encode(crc32c.to_be_bytes()),
    ))
}

fn upload_asset(
    client: &Client,
    workspace: &Workspace,
    asset: &AssetRef,
    url: &str,
) -> Result<AssetUploadReceipt, Box<dyn Error>> {
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
    let (actual_checksum, md5, crc32c) =
        calculate_upload_checksums(&source_path).map_err(|error| {
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
        // For GCS, content-md5 will be used as a server side integrity verification. It is ignored for
        // composite objects (those > 5GB)
        .header("content-md5", &md5)
        .header("x-goog-if-generation-match", "0")
        .body(Body::sized(file, length))
        .send()
        .map_err(|error| error.without_url())?;
    if !response.status().is_success()
        && response.status() != reqwest::StatusCode::PRECONDITION_FAILED
    {
        return Err(format!(
            "Asset {} upload failed with HTTP {}",
            asset.id,
            response.status()
        )
        .into());
    }
    Ok(AssetUploadReceipt {
        id: asset.id,
        crc32c,
    })
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) enum DownloadAssetOutcome {
    Unchanged,
    Downloaded,
    Conflict(PathBuf),
}

/// This is effectively a dirty file check. On clones/pulls we want to update
/// files if they match previously known checksums. The previous asset set is cumulative
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

/// Returns the stable sibling path used while an asset is being written.
///
/// HTTP downloads intentionally reuse this path across invocations so interrupted transfers can
/// resume. Copies from the versioned store also stage here so a failed copy does not truncate the
/// logical destination.
fn temporary_path(destination_path: &Path) -> Result<PathBuf, Box<dyn Error>> {
    let file_name = destination_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            format!(
                "Asset destination has no file name: {}",
                destination_path.display()
            )
        })?;
    Ok(destination_path.with_file_name(format!("{file_name}.tmp")))
}

/// Confirms that a partial response begins exactly after the bytes already staged on disk.
///
/// A response is never appended unless this check passes; otherwise the downloader restarts from
/// byte zero so a server that ignores or mishandles ranges cannot corrupt the staged asset.
fn content_range_starts_at(response: &Response, expected_start: u64) -> bool {
    response
        .headers()
        .get(CONTENT_RANGE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("bytes "))
        .and_then(|value| value.split_once('-'))
        .and_then(|(start, _)| start.parse::<u64>().ok())
        == Some(expected_start)
}

/// Streams a response directly to the durable staged file without buffering the asset in memory.
///
/// Partial bytes are deliberately left in place when the stream fails so the next invocation can
/// resume them. `append` is true only after validating the response's `Content-Range`.
fn stream_asset_response_to_staged_file(
    response: &mut Response,
    staged_path: &Path,
    append: bool,
) -> Result<(), Box<dyn Error>> {
    let mut staged_file = OpenOptions::new()
        .create(true)
        .write(true)
        .append(append)
        .truncate(!append)
        .open(staged_path)?;
    io::copy(response, &mut staged_file)?;
    staged_file.flush()?;
    staged_file.sync_all()?;
    Ok(())
}

/// Completes or resumes one HTTP asset download in its durable staged file.
///
/// [`download_to_versioned_store`] owns the final rename into `.gen/assets`; this function owns
/// only the HTTP range protocol. It appends a valid partial response, overwrites the staged file
/// when a server ignores `Range` and returns a complete response, and retries once from byte zero
/// after an invalid partial response or a resumed checksum mismatch.
fn download_to_staged_path(
    client: &Client,
    asset: &AssetRef,
    url: &str,
    staged_path: &Path,
    expected_checksum: &Sha256Hash,
) -> Result<(), Box<dyn Error>> {
    let mut resume_offset = staged_path
        .metadata()
        .map(|metadata| metadata.len())
        .unwrap_or(0);
    // Size is only a cheap hint. It avoids hashing a known-partial multi-gigabyte file before
    // resuming, while the checksum remains authoritative whenever the staged size could be final.
    let staged_size_could_be_complete = asset
        .size
        .and_then(|size| u64::try_from(size).ok())
        .is_none_or(|expected_size| expected_size == resume_offset);
    if resume_offset > 0
        && staged_size_could_be_complete
        && calculate_file_checksum(staged_path).is_ok_and(|checksum| checksum == *expected_checksum)
    {
        return Ok(());
    }

    loop {
        let mut request = client.get(url);
        if resume_offset > 0 {
            request = request.header(RANGE, format!("bytes={resume_offset}-"));
        }
        let mut response = request.send().map_err(|error| error.without_url())?;
        let append = resume_offset > 0
            && response.status() == StatusCode::PARTIAL_CONTENT
            && content_range_starts_at(&response, resume_offset);
        let invalid_partial_response = resume_offset > 0
            && (response.status() == StatusCode::RANGE_NOT_SATISFIABLE
                || (response.status() == StatusCode::PARTIAL_CONTENT && !append));
        if invalid_partial_response {
            fs::remove_file(staged_path)?;
            resume_offset = 0;
            continue;
        }
        if !response.status().is_success() {
            return Err(format!(
                "Asset {} download failed with HTTP {}",
                asset.id,
                response.status()
            )
            .into());
        }

        stream_asset_response_to_staged_file(&mut response, staged_path, append)?;
        if calculate_file_checksum(staged_path)? == *expected_checksum {
            return Ok(());
        }
        if append {
            fs::remove_file(staged_path)?;
            resume_offset = 0;
            continue;
        }

        fs::remove_file(staged_path)?;
        return Err(format!("Downloaded asset {} failed checksum validation", asset.id).into());
    }
}

/// Ensures that an HTTP asset exists under its checksum-derived `.gen/assets` path.
///
/// [`download_asset`] calls this storage phase before considering the logical workspace path. The
/// returned boolean reports whether this call published new bytes; the returned file is always
/// checksum-verified. No logical file is read or written here.
fn download_to_versioned_store(
    client: &Client,
    workspace: &Workspace,
    asset: &AssetRef,
    url: &str,
) -> Result<(PathBuf, bool), Box<dyn Error>> {
    let expected_checksum = asset_checksum(asset)?;
    let versioned_path =
        materialization_destination_path(workspace, &asset.uri, Some(&expected_checksum), None)?;
    if versioned_path.exists()
        && calculate_file_checksum(&versioned_path)
            .is_ok_and(|checksum| checksum == expected_checksum)
    {
        return Ok((versioned_path, false));
    }

    let asset_dir = versioned_path.parent().ok_or_else(|| {
        format!(
            "Versioned asset path has no parent: {}",
            versioned_path.display()
        )
    })?;
    fs::create_dir_all(asset_dir)?;
    let staged_path = temporary_path(&versioned_path)?;
    download_to_staged_path(client, asset, url, &staged_path, &expected_checksum)?;
    if versioned_path.exists() {
        fs::remove_file(&versioned_path)?;
    }
    fs::rename(&staged_path, &versioned_path)?;
    Ok((versioned_path, true))
}

/// Copies one checksum-addressed version between two `file://` remote workspaces.
///
/// [`transfer_file_remote_assets`] calls this in either direction after graph transfer. It verifies
/// the source before copying, reuses an already-valid destination, and verifies newly copied bytes
/// before allowing materialization.
fn copy_to_versioned_store(
    source_workspace: &Workspace,
    destination_workspace: &Workspace,
    asset: &AssetRef,
) -> Result<(PathBuf, bool), Box<dyn Error>> {
    let expected_checksum = asset_checksum(asset)?;
    let source_path = materialization_destination_path(
        source_workspace,
        &asset.uri,
        Some(&expected_checksum),
        None,
    )?;
    let source_checksum = calculate_file_checksum(&source_path).map_err(|error| {
        format!(
            "Unable to read versioned asset {} at {}: {error}",
            asset.id,
            source_path.display()
        )
    })?;
    if source_checksum != expected_checksum {
        return Err(format!(
            "Versioned asset {} at {} does not match its recorded checksum",
            asset.id,
            source_path.display()
        )
        .into());
    }

    let destination_path = materialization_destination_path(
        destination_workspace,
        &asset.uri,
        Some(&expected_checksum),
        None,
    )?;
    if destination_path.exists()
        && calculate_file_checksum(&destination_path)
            .is_ok_and(|checksum| checksum == expected_checksum)
    {
        return Ok((destination_path, false));
    }

    copy_versioned_asset(&source_path, &destination_path)?;
    if calculate_file_checksum(&destination_path)? != expected_checksum {
        fs::remove_file(&destination_path)?;
        return Err(format!("Copied asset {} failed checksum validation", asset.id).into());
    }
    Ok((destination_path, true))
}

/// Copies a verified versioned asset to another path through a synced staged file.
///
/// [`materialize_versioned_asset`] uses this for logical and conflict files. The `file://` remote
/// transport also uses it when moving checksum-addressed versions between workspace stores.
fn copy_versioned_asset(
    versioned_path: &Path,
    destination_path: &Path,
) -> Result<(), Box<dyn Error>> {
    let parent = destination_path.parent().ok_or_else(|| {
        format!(
            "Asset destination has no parent: {}",
            destination_path.display()
        )
    })?;
    fs::create_dir_all(parent)?;
    let staged_path = temporary_path(destination_path)?;
    let result = (|| {
        let mut staged_file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&staged_path)?;
        let mut versioned_file = fs::File::open(versioned_path)?;
        io::copy(&mut versioned_file, &mut staged_file)?;
        staged_file.flush()?;
        staged_file.sync_all()?;
        drop(staged_file);
        fs::rename(&staged_path, destination_path)?;
        Ok::<(), Box<dyn Error>>(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&staged_path);
    }
    result
}

/// Copies a versioned asset to the workspace without overwriting unknown local content.
///
/// Versioned assets replace a known prior version or are copied beside unknown local
/// content as a conflict, preventing unintentional overwrites of data.
pub(crate) fn materialize_versioned_asset(
    workspace: &Workspace,
    asset: &AssetRef,
    previous_assets: &HashMap<HashId, AssetRef>,
    destination_logical_path: Option<&str>,
    versioned_path: &Path,
    versioned_file_created: bool,
) -> Result<DownloadAssetOutcome, Box<dyn Error>> {
    let expected_checksum = asset_checksum(asset)?;
    if !versioned_path.is_file() {
        return Err(format!(
            "Cannot materialize asset {} because its versioned file is missing: {}",
            asset.id,
            versioned_path.display()
        )
        .into());
    }
    let Some(destination_logical_path) = destination_logical_path else {
        return Ok(if versioned_file_created {
            DownloadAssetOutcome::Downloaded
        } else {
            DownloadAssetOutcome::Unchanged
        });
    };
    let destination_path = materialization_destination_path(
        workspace,
        &asset.uri,
        Some(&expected_checksum),
        Some(destination_logical_path),
    )?;
    if destination_path == versioned_path {
        return Ok(if versioned_file_created {
            DownloadAssetOutcome::Downloaded
        } else {
            DownloadAssetOutcome::Unchanged
        });
    }
    if destination_path.exists()
        && calculate_file_checksum(&destination_path)
            .is_ok_and(|checksum| checksum == expected_checksum)
    {
        return Ok(DownloadAssetOutcome::Unchanged);
    }
    let existing_checksum = if destination_path.exists() {
        Some(calculate_file_checksum(&destination_path)?)
    } else {
        None
    };
    let intended_change = if let Some(checksum) = existing_checksum.as_ref() {
        destination_matches_previous_asset(workspace, &destination_path, checksum, previous_assets)?
    } else {
        false
    };
    let has_conflict = existing_checksum.is_some() && !intended_change;
    if has_conflict {
        let (conflict_path, already_downloaded) =
            conflict_destination_path(&destination_path, &expected_checksum)?;
        if !already_downloaded {
            copy_versioned_asset(versioned_path, &conflict_path)?;
        }
        return Ok(DownloadAssetOutcome::Conflict(conflict_path));
    }

    copy_versioned_asset(versioned_path, &destination_path)?;
    Ok(DownloadAssetOutcome::Downloaded)
}

/// Runs the HTTP asset pipeline: versioned storage first, optional materialization second.
///
/// [`transfer_assets`] calls this for each clone or pull URL returned by GenHub. Keeping the phases
/// ordered here prevents an already-current logical file from bypassing `.gen/assets` population.
fn download_asset(
    client: &Client,
    workspace: &Workspace,
    asset: &AssetRef,
    previous_assets: &HashMap<HashId, AssetRef>,
    // `None` stores a historical version under `.gen/assets` instead of its recorded logical path.
    destination_logical_path: Option<&str>,
    url: &str,
) -> Result<DownloadAssetOutcome, Box<dyn Error>> {
    let (versioned_path, versioned_file_created) =
        download_to_versioned_store(client, workspace, asset, url)?;
    materialize_versioned_asset(
        workspace,
        asset,
        previous_assets,
        destination_logical_path,
        &versioned_path,
        versioned_file_created,
    )
}

/// Reports a conflict returned by [`materialize_versioned_asset`] using workspace-relative paths.
pub(crate) fn warn_asset_conflict(
    workspace: &Workspace,
    asset: &AssetRef,
    destination_logical_path: Option<&str>,
    conflict_path: &Path,
) -> Result<(), Box<dyn Error>> {
    let destination_path = materialization_destination_path(
        workspace,
        &asset.uri,
        asset.checksum.as_ref(),
        destination_logical_path,
    )?;
    eprintln!(
        "Warning: the requested asset version conflicts with the local file at {}. The local file was preserved and the requested version was written to {}. Choose the correct version before continuing.",
        destination_path.display(),
        conflict_path.display()
    );
    Ok(())
}

/// Executes the asset portion of a `file://` clone, pull, or push.
///
/// [`transfer_assets`] supplies the graph-derived version delta. Push copies those versions into
/// the remote store; clone and pull copy them locally and then call [`materialize_versioned_asset`]
/// only for the versions selected at the destination branch head.
fn transfer_file_remote_assets(
    workspace: &Workspace,
    remote: &Remote,
    operation: RemoteOperation,
    assets: &HashMap<HashId, AssetRef>,
    materialized_asset_ids: &HashSet<HashId>,
    previous_assets: &HashMap<HashId, AssetRef>,
) -> Result<(), Box<dyn Error>> {
    if assets.is_empty() {
        return Ok(());
    }
    let remote_workspace = file_remote_workspace(&remote.url)?;
    for asset in assets.values() {
        match operation {
            RemoteOperation::Push => {
                copy_to_versioned_store(workspace, &remote_workspace, asset)?;
            }
            RemoteOperation::Clone | RemoteOperation::Pull => {
                let destination_logical_path = if materialized_asset_ids.contains(&asset.id) {
                    asset.logical_path.as_deref()
                } else {
                    None
                };
                let (versioned_path, versioned_file_created) =
                    copy_to_versioned_store(&remote_workspace, workspace, asset)?;
                if let DownloadAssetOutcome::Conflict(conflict_path) = materialize_versioned_asset(
                    workspace,
                    asset,
                    previous_assets,
                    destination_logical_path,
                    &versioned_path,
                    versioned_file_created,
                )? {
                    warn_asset_conflict(
                        workspace,
                        asset,
                        destination_logical_path,
                        &conflict_path,
                    )?;
                }
            }
        }
    }
    Ok(())
}

/// Transfers the asset versions needed after a graph clone, pull, or push.
///
/// The CLI orchestration calls this after Dolt transfers graph history. It derives the asset delta
/// between `previous_hash` and the destination branch, then dispatches `file://` transfers to
/// [`transfer_file_remote_assets`] or asks GenHub for HTTP transfer URLs. Push uploads local
/// versions; HTTP clone and pull send downloads through [`download_asset`]. The cumulative asset
/// view supplies history and conflict checksums, while the materialized view selects the one version
/// per logical path that is additionally copied out of `.gen/assets`.
fn transfer_assets(
    graph: &GraphConnection,
    workspace: &Workspace,
    remote: &Remote,
    operation: RemoteOperation,
    branch: &str,
    previous_hash: Option<&DoltHashId>,
) -> Result<Vec<AssetUploadReceipt>, Box<dyn Error>> {
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
    if remote.url.starts_with("file://") {
        transfer_file_remote_assets(
            workspace,
            remote,
            operation,
            &assets,
            &materialized_asset_ids,
            &previous_assets,
        )?;
        return Ok(Vec::new());
    }

    let repository = RepositoryRemote::parse(&remote.url)?;
    let response = acquire_asset_transfers(
        &repository,
        &AssetTransferRequest { operation, branch },
        login_origin,
    )?;
    let client = Client::new();
    let mut upload_receipts = Vec::new();
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
            RemoteOperation::Push => {
                upload_receipts.push(upload_asset(&client, workspace, &asset, &transfer.url)?);
            }
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
                    warn_asset_conflict(
                        workspace,
                        &asset,
                        destination_logical_path,
                        &conflict_path,
                    )?;
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
    Ok(upload_receipts)
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
    let upload_receipts = transfer_assets(
        &graph,
        workspace,
        &remote,
        RemoteOperation::Push,
        &branch,
        previous_hash.as_ref(),
    )?;
    if !remote.url.starts_with("file://") {
        let repository = RepositoryRemote::parse(&remote.url)?;
        complete_asset_transfers(
            &repository,
            &AssetTransferCompletionRequest {
                branch: &branch,
                assets: &upload_receipts,
            },
            login_origin,
        )?;
    }
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
        path::PathBuf,
        sync::Mutex,
        thread,
    };

    use gen_core::config::Workspace;
    use gen_models::{
        assets::{AssetRef, AssetRole, LocalAssetUri, materialization_destination_path},
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
        copy_versioned_asset, download_asset, download_to_versioned_store, execute_push,
        file_graph_url, resolve_remote, run_graph_transfer, temporary_path, transfer_assets,
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

    fn serve_resumable_asset(
        contents: &[u8],
        resume_offset: usize,
    ) -> (String, thread::JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("should bind asset server");
        let address = listener
            .local_addr()
            .expect("should read asset server address");
        let contents = contents.to_vec();
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("should accept asset request");
            let mut request = [0_u8; 4096];
            let read = stream
                .read(&mut request)
                .expect("should read asset request");
            let body = &contents[resume_offset..];
            write!(
                stream,
                "HTTP/1.1 206 Partial Content\r\nContent-Type: application/octet-stream\r\nContent-Range: bytes {resume_offset}-{}/{}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                contents.len() - 1,
                contents.len(),
                body.len()
            )
            .expect("should write partial asset response headers");
            stream
                .write_all(body)
                .expect("should write partial asset response body");
            String::from_utf8_lossy(&request[..read]).into_owned()
        });
        (format!("http://{address}/asset"), handle)
    }

    fn versioned_asset_path(workspace: &Workspace, asset: &AssetRef) -> PathBuf {
        materialization_destination_path(workspace, &asset.uri, asset.checksum.as_ref(), None)
            .expect("should resolve versioned asset path")
    }

    fn serve_transfer_response(response_body: String) -> (Remote, thread::JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("should bind transfer server");
        let address = listener
            .local_addr()
            .expect("should read transfer server address");
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("should accept transfer request");
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
        (
            Remote {
                name: "origin".to_string(),
                url: format!("http://{address}/api/repos/alice/example"),
            },
            server,
        )
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

    // This ensures we don't overwrite files the user has in their workspace that are unknown. This prevents
    // destructive actions against unstaged files.
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
        assert_eq!(
            fs::read(versioned_asset_path(&workspace, &remote_asset))
                .expect("should read versioned remote asset"),
            remote_contents,
            "remote asset should be retained before creating a conflict copy"
        );
    }

    // If a file has been updated on the remote, and the local file is one that belongs to an older revision,
    // assert that we replace the file with the newer one.
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
        assert_eq!(
            fs::read(versioned_asset_path(&workspace, &remote_asset))
                .expect("should read versioned remote asset"),
            remote_contents,
            "updated asset should be retained before logical materialization"
        );
        assert!(!temp.path().join("reference.fa.conflict").exists());
    }

    // Ensure that if a file exists in the logical path, we still download to versioned storage if it is
    // missing there.
    #[test]
    fn test_download_asset_populates_versioned_store_when_logical_path_exists() {
        let temp = tempdir().expect("should create workspace");
        let workspace = Workspace::new(temp.path());
        workspace.ensure_gen_dir();
        let remote_contents = b"current version\n";
        let remote_asset = test_asset(remote_contents, "reference.fa", 1);
        fs::write(temp.path().join("reference.fa"), remote_contents)
            .expect("should write current logical file");
        let (url, server) = serve_asset(remote_contents);

        let outcome = download_asset(
            &Client::new(),
            &workspace,
            &remote_asset,
            &HashMap::new(),
            remote_asset.logical_path.as_deref(),
            &url,
        )
        .expect("should retain matching asset before returning unchanged");
        server.join().expect("asset server should finish");

        assert_eq!(
            outcome,
            DownloadAssetOutcome::Unchanged,
            "matching logical asset should remain unchanged"
        );
        assert_eq!(
            fs::read(versioned_asset_path(&workspace, &remote_asset))
                .expect("should read versioned remote asset"),
            remote_contents,
            "matching logical asset should still be populated in versioned storage"
        );
    }

    #[test]
    fn test_download_to_versioned_store_resumes_partial_asset_file() {
        let temp = tempdir().expect("should create workspace");
        let workspace = Workspace::new(temp.path());
        workspace.ensure_gen_dir();
        let remote_contents = b"resumable remote asset\n";
        let remote_asset = test_asset(remote_contents, "reference.fa", 1);
        let versioned_path = versioned_asset_path(&workspace, &remote_asset);
        fs::create_dir_all(
            versioned_path
                .parent()
                .expect("should resolve versioned asset directory"),
        )
        .expect("should create versioned asset directory");
        let staged_path =
            temporary_path(&versioned_path).expect("should resolve staged asset path");
        let resume_offset = 10;
        fs::write(&staged_path, &remote_contents[..resume_offset])
            .expect("should seed partial asset download");
        let (url, server) = serve_resumable_asset(remote_contents, resume_offset);

        let (downloaded_path, downloaded) =
            download_to_versioned_store(&Client::new(), &workspace, &remote_asset, &url)
                .expect("should resume partial versioned asset download");
        let request = server.join().expect("asset server should finish");

        assert!(downloaded, "resumed asset should be reported as downloaded");
        assert_eq!(
            downloaded_path, versioned_path,
            "download should resolve to versioned asset path"
        );
        assert_eq!(
            fs::read(&versioned_path).expect("should read resumed versioned asset"),
            remote_contents,
            "resumed versioned asset should contain the complete verified content"
        );
        assert!(
            request
                .to_ascii_lowercase()
                .contains(&format!("\r\nrange: bytes={resume_offset}-\r\n")),
            "resume request should begin after the staged bytes"
        );
        assert!(
            !staged_path.exists(),
            "successful atomic rename should remove the staged path"
        );
    }

    // Conflict detection accepts any known version before the pull, so an older clean
    // checkout can still advance without a false conflict.
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

    // If a file has been edited locally and the user pulls remote changes, ensure
    // that the newer version ends up in versioned storage while presenting a .conflict file to the user
    // at the logical path
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
        assert_eq!(
            fs::read(versioned_asset_path(&workspace, &remote_asset))
                .expect("should read versioned remote asset"),
            remote_contents,
            "conflicting remote asset should be retained in versioned storage"
        );
    }

    // Ensure that if the staged download is corrupt, it does not end up in the versioned storage or logical path
    #[test]
    fn test_download_asset_rejects_invalid_checksum_before_saving() {
        let temp = tempdir().expect("should create workspace");
        let workspace = Workspace::new(temp.path());
        workspace.ensure_gen_dir();
        let remote_asset = test_asset(b"expected version\n", "reference.fa", 1);
        let (url, server) = serve_asset(b"tampered version\n");

        let error = download_asset(
            &Client::new(),
            &workspace,
            &remote_asset,
            &HashMap::new(),
            remote_asset.logical_path.as_deref(),
            &url,
        )
        .expect_err("should reject an asset with the wrong checksum");
        server.join().expect("asset server should finish");

        assert!(
            error.to_string().contains("failed checksum validation"),
            "checksum mismatch should be reported"
        );
        assert!(
            !versioned_asset_path(&workspace, &remote_asset).exists(),
            "invalid download should not populate versioned storage"
        );
        assert!(
            !temporary_path(&versioned_asset_path(&workspace, &remote_asset))
                .expect("should resolve staged asset path")
                .exists(),
            "invalid complete download should remove its staged asset"
        );
        assert!(
            !temp.path().join("reference.fa").exists(),
            "invalid download should not create a logical file"
        );
    }

    // Ensure that if there is an issue with the versioned file (such as it being deleted), if it is attempted
    // to be copied to a logical path, it fails
    #[test]
    fn test_copy_versioned_asset_preserves_destination_when_copy_fails() {
        let temp = tempdir().expect("should create workspace");
        let missing_versioned_path = temp.path().join("missing-versioned.fa");
        let destination_path = temp.path().join("reference.fa");
        fs::write(&destination_path, b"previous logical bytes\n")
            .expect("should write previous logical asset");

        copy_versioned_asset(&missing_versioned_path, &destination_path)
            .expect_err("should reject a missing versioned asset");

        assert_eq!(
            fs::read(destination_path).expect("should read preserved logical asset"),
            b"previous logical bytes\n",
            "failed copy should preserve the previous logical asset"
        );
    }

    // This ensures that we only request and pull assets that we don't already have by requesting versions
    // after the commit hash prior to the pull.
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
        let (remote, transfer_server) = serve_transfer_response(response_body);

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
        assert_eq!(
            fs::read(versioned_asset_path(&workspace, &current_asset))
                .expect("should read versioned current asset"),
            current_contents,
            "pull should retain the current asset before materialization"
        );
    }

    // Ensure that clone retains every historical version in `.gen/assets` while materializing only the
    // version selected at the cloned branch head.
    #[test]
    fn test_clone_retains_history_and_materializes_only_current_asset() {
        let temp = tempdir().expect("should create transfer workspace");
        let workspace = Workspace::new(temp.path());
        workspace.ensure_gen_dir();
        let graph = get_connection(
            workspace
                .graph_db_path()
                .expect("should resolve graph database path"),
        )
        .expect("should open graph database");
        let historical_contents = b"historical version\n";
        let historical_asset = test_asset(historical_contents, "reference.fa", 1);
        AssetRef::create(&graph, &historical_asset).expect("should insert historical asset");
        commit_all(&graph, "add historical asset").expect("should commit historical asset");
        let current_contents = b"current version\n";
        let current_asset = test_asset(current_contents, "reference.fa", 2);
        AssetRef::create(&graph, &current_asset).expect("should insert current asset");
        commit_all(&graph, "add current asset").expect("should commit current asset");

        let (historical_url, historical_server) = serve_asset(historical_contents);
        let (current_url, current_server) = serve_asset(current_contents);
        let response_body = json!({
            "assets": [
                { "id": historical_asset.id, "url": historical_url },
                { "id": current_asset.id, "url": current_url }
            ]
        })
        .to_string();
        let (remote, transfer_server) = serve_transfer_response(response_body);

        transfer_assets(
            &graph,
            &workspace,
            &remote,
            RemoteOperation::Clone,
            "main",
            None,
        )
        .expect("should transfer clone assets");
        let transfer_request = transfer_server
            .join()
            .expect("transfer server should finish");
        historical_server
            .join()
            .expect("historical asset server should finish");
        current_server
            .join()
            .expect("current asset server should finish");

        assert!(
            transfer_request.contains("\"operation\":\"clone\""),
            "asset transfer request should identify the clone operation"
        );
        assert_eq!(
            fs::read(versioned_asset_path(&workspace, &historical_asset))
                .expect("should read versioned historical asset"),
            historical_contents,
            "clone should retain the historical asset in versioned storage"
        );
        assert_eq!(
            fs::read(versioned_asset_path(&workspace, &current_asset))
                .expect("should read versioned current asset"),
            current_contents,
            "clone should retain the selected current asset in versioned storage"
        );
        assert_eq!(
            fs::read(temp.path().join("reference.fa"))
                .expect("should read materialized current asset"),
            current_contents,
            "clone should materialize only the selected current version"
        );
        assert!(
            !temp.path().join("reference.fa.conflict").exists(),
            "clean clone should not create a conflict file"
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
            for request_index in 0..4 {
                let (mut stream, _) = listener.accept().expect("should accept capability request");
                let mut request = [0_u8; 8192];
                let read = stream
                    .read(&mut request)
                    .expect("should read capability request");
                requests.push(String::from_utf8_lossy(&request[..read]).into_owned());
                if request_index == 2 {
                    write!(
                        stream,
                        "HTTP/1.1 204 No Content\r\nContent-Length: 0\r\n\r\n"
                    )
                    .expect("should write completion response");
                    continue;
                }
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

        assert_eq!(requests.len(), 4);
        assert!(requests[0].contains("\"operation\":\"push\""));
        assert!(requests[1].starts_with("POST /api/repos/alice/example/asset-transfers "));
        assert!(requests[1].contains("\"operation\":\"push\""));
        assert!(requests[2].starts_with("POST /api/repos/alice/example/asset-transfers/complete "));
        assert!(requests[2].contains("\"branch\":\"main\""));
        assert!(requests[2].contains("\"assets\":[]"));
        assert!(requests[3].contains("\"operation\":\"pull\""));
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
            for request_index in 0..4 {
                let (mut stream, _) = listener.accept().expect("should accept capability request");
                let mut request = [0_u8; 8192];
                let read = stream
                    .read(&mut request)
                    .expect("should read capability request");
                requests.push(String::from_utf8_lossy(&request[..read]).into_owned());
                if request_index < 3 {
                    if request_index == 2 {
                        write!(
                            stream,
                            "HTTP/1.1 204 No Content\r\nContent-Length: 0\r\n\r\n"
                        )
                        .expect("should write completion response");
                        continue;
                    }
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

        assert_eq!(requests.len(), 4);
        assert!(requests[0].contains("\"operation\":\"push\""));
        assert!(requests[1].starts_with("POST /api/repos/alice/example/asset-transfers "));
        assert!(requests[1].contains("\"operation\":\"push\""));
        assert!(requests[2].starts_with("POST /api/repos/alice/example/asset-transfers/complete "));
        assert!(requests[2].contains("\"branch\":\"main\""));
        assert!(requests[3].contains("\"operation\":\"pull\""));
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
