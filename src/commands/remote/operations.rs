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
//! Clones, pushes, and pulls are distributed operations where the graph database is sync'd and
//! then the assets are transfered. Thus, an asset transfer can fail and need to be resumed. We
//! track the state of asset transfers and checkpoint after each commit is successfully transfered.
//! When the entire commit range has been transfered, we mark the operation as complete. For pushes,
//! a server provides a transfer ID and expiry that serve as a lease. Gen persists both so an
//! interrupted asset phase can reuse an active lease, while an expired lease triggers fresh graph
//! authorization. The lease restricts uploads to a single client and is released after assets are
//! uploaded.
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
use chrono::Utc;
use crc32c::crc32c_append;
use flate2::read::MultiGzDecoder;
use gen_core::{
    DoltHashId, HashId, Sha256Hash,
    config::{DEFAULT_GRAPH_DB_NAME, Workspace},
    errors::{ConfigError, ConnectionError},
};
use gen_models::{
    assets::{
        AssetRef, AssetView, ChecksummedWriter, CompressionType, LocalAssetUri,
        materialization_destination_path,
    },
    db::{ConfigConnection, GraphConnection},
    errors::{QueryError, RemoteError as ModelRemoteError},
    history::dolt::{
        active_branch, add_remote, branch_hash, checkout, clone_remote, fetch, hash_of, pull,
        push_force_with_idempotency_token, push_with_idempotency_token, remote_rows,
        set_remote_url,
    },
    operations::{
        Defaults, Remote, RemoteBranch, RemoteOperationKind as StoredRemoteOperationKind,
        RemoteOperationRecord, calculate_file_checksum,
    },
};
use indexmap::IndexMap;
use md5::Md5;
use noodles::bgzf;
use reqwest::{
    StatusCode,
    blocking::{Body, Client, Response},
    header::{CONTENT_RANGE, RANGE},
};
use rusqlite::Error as SqlError;
use sha2::{Digest as _, Sha256};
use url::Url;
use uuid::Uuid;

use crate::{
    commands::remote::{
        client::{
            AssetTransferCompletionRequest, AssetTransferRequest, AssetUploadReceipt,
            CapabilityRequest, RemoteClientError, RemoteOperation, RepositoryRemote,
            acquire_asset_transfers, acquire_asset_transfers_with_idempotency_token,
            acquire_capability, acquire_capability_with_idempotency_token,
            complete_asset_transfers, complete_asset_transfers_with_idempotency_token,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PushTransferLease {
    transfer_id: Uuid,
    expires_at: i64,
}

struct GraphTransferAuthorization {
    remote_url: String,
    push_lease: Option<PushTransferLease>,
}

fn transfer_authorization(
    remote: &Remote,
    operation: RemoteOperation,
    branch: Option<&str>,
    force: bool,
    idempotency_token: Option<&Uuid>,
) -> Result<GraphTransferAuthorization, Box<dyn Error>> {
    if remote.url.starts_with("file://") {
        return Ok(GraphTransferAuthorization {
            remote_url: file_graph_url(&remote.url)?,
            push_lease: None,
        });
    }
    let repository = RepositoryRemote::parse(&remote.url)?;
    let request = CapabilityRequest {
        operation,
        branch,
        force,
    };
    let capability = match idempotency_token {
        Some(idempotency_token) => acquire_capability_with_idempotency_token(
            &repository,
            &request,
            idempotency_token,
            login_origin,
        )?,
        None => acquire_capability(&repository, &request, login_origin)?,
    };
    let push_lease = (operation == RemoteOperation::Push).then_some(PushTransferLease {
        transfer_id: capability.transfer_id,
        expires_at: capability.expires_at.timestamp(),
    });
    Ok(GraphTransferAuthorization {
        remote_url: capability.remote_url,
        push_lease,
    })
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
        .or_else(|| Defaults::get_default_remote(config));
    let remote_name = if let Some(remote_name) = remote_name {
        remote_name
    } else {
        let mut remotes = Remote::list_all(config);
        if remotes.len() == 1 {
            remotes.remove(0).name
        } else {
            return Err(
                "No remote specified, tracked for this branch, or configured as default".into(),
            );
        }
    };
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
    idempotency_token: Option<&Uuid>,
    mut transfer: impl FnMut() -> Result<(), SqlError>,
) -> Result<Option<PushTransferLease>, Box<dyn Error>> {
    if remote.url.starts_with("file://") {
        let authorization = transfer_authorization(remote, operation, Some(branch), force, None)?;
        ensure_graph_remote(graph, &remote.name, &authorization.remote_url)?;
        let result = transfer();
        restore_canonical_url(graph, remote);
        result?;
        return Ok(None);
    }

    let mut last_error = None;
    for attempt in 0..2 {
        let authorization =
            transfer_authorization(remote, operation, Some(branch), force, idempotency_token)?;
        ensure_graph_remote(graph, &remote.name, &authorization.remote_url)?;
        match transfer() {
            Ok(()) => {
                restore_canonical_url(graph, remote);
                return Ok(authorization.push_lease);
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

fn push_graph_branch(
    graph: &GraphConnection,
    remote_name: &str,
    branch: &str,
    force: bool,
    idempotency_token: Option<&Uuid>,
) -> Result<(), SqlError> {
    match (force, idempotency_token) {
        (true, Some(token)) => push_force_with_idempotency_token(graph, remote_name, branch, token),
        (false, Some(token)) => push_with_idempotency_token(graph, remote_name, branch, token),
        (true, None) => gen_models::history::dolt::push_force(graph, remote_name, branch),
        (false, None) => gen_models::history::dolt::push(graph, remote_name, branch),
    }
}

fn asset_checksum(asset: &AssetRef) -> Result<Sha256Hash, Box<dyn Error>> {
    asset
        .checksum
        .ok_or_else(|| format!("Local asset {} has no checksum", asset.id).into())
}

fn materialized_asset_checksum(asset: &AssetRef) -> Result<Sha256Hash, Box<dyn Error>> {
    match asset.materialized_checksum {
        Some(checksum) => Ok(checksum),
        None => asset_checksum(asset),
    }
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
    } else if asset.materialized_checksum.is_some() {
        materialization_destination_path(workspace, &asset.uri, Some(&expected_checksum), None)?
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

#[derive(Clone, Copy)]
struct AssetTransferRange<'commit> {
    from_commit: Option<&'commit DoltHashId>,
    previous_hash: Option<&'commit DoltHashId>,
    materialize: bool,
}

struct AssetTransferTarget<'transfer> {
    branch: &'transfer str,
    history_ref: &'transfer str,
    range: AssetTransferRange<'transfer>,
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
    let mut decoded_destination_checksum = None;
    let mut destination_decode_failed = false;
    for asset in previous_assets.values() {
        let archive_checksum = asset_checksum(asset)?;
        let logical_path = materialization_destination_path(
            workspace,
            &asset.uri,
            Some(&archive_checksum),
            asset.logical_path.as_deref(),
        )?;
        // Cumulative history includes unrelated logical paths, so ignore entries targeting elsewhere.
        if logical_path != destination_path {
            continue;
        }
        // This exact checksum identifies a workspace version already known to history.
        if materialized_asset_checksum(asset)? == *existing_checksum {
            return Ok(true);
        }
        // Skip plain representations because the checksum check above is sufficient; compressed
        // encodings can differ while preserving known payloads, so compare their decoded contents.
        if !matches!(
            asset.materialized_compression_type(),
            CompressionType::Gzip | CompressionType::Bgzf
        ) {
            continue;
        }
        let archived_path =
            materialization_destination_path(workspace, &asset.uri, Some(&archive_checksum), None)?;
        // This is the checksum-addressed archive under .gen/assets; without local bytes, this
        // version cannot be decoded for comparison, so try others and conflict if none match.
        if !archived_path.is_file() {
            continue;
        }
        // Hash mismatch means cached bytes may be partial, corrupt, or locally modified.
        if !calculate_file_checksum(&archived_path)
            .is_ok_and(|checksum| checksum == archive_checksum)
        {
            continue;
        }
        // This means we can't hash the destination asset, so just skip everything else trying to use this checksum.
        if destination_decode_failed {
            continue;
        }
        // Reuse this decoded checksum across the cumulative versions of this destination.
        let existing_decoded_checksum = if let Some(checksum) = decoded_destination_checksum {
            checksum
        } else {
            match decoded_compressed_checksum(destination_path) {
                Ok(checksum) => {
                    decoded_destination_checksum = Some(checksum);
                    checksum
                }
                Err(_) => {
                    // A local edit or damaged compressed stream may make the destination undecodable.
                    destination_decode_failed = true;
                    continue;
                }
            }
        };
        // Raw hash verification does not guarantee the archive stream can be decoded.
        let Ok(archived_decoded_checksum) = decoded_compressed_checksum(&archived_path) else {
            continue;
        };
        // Matching decoded bytes means different gzip/BGZF encodings represent a known workspace version.
        if existing_decoded_checksum == archived_decoded_checksum {
            return Ok(true);
        }
    }
    Ok(false)
}

fn decoded_compressed_checksum(path: &Path) -> Result<Sha256Hash, io::Error> {
    let mut reader = MultiGzDecoder::new(fs::File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let length = reader.read(&mut buffer)?;
        if length == 0 {
            break;
        }
        hasher.update(&buffer[..length]);
    }
    Ok(Sha256Hash(hasher.finalize().into()))
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

    copy_versioned_asset(&source_path, &destination_path, None, &expected_checksum)?;
    Ok((destination_path, true))
}

/// Copies a versioned archive or restores it to a workspace path without premature replacement.
///
/// Repository transfers preserve stored bytes, while workspace restoration decodes BGZF when a
/// materialized checksum is recorded. The staged output is synced and verified before rename so
/// incomplete or invalid data cannot replace the existing destination.
fn copy_versioned_asset(
    versioned_path: &Path,
    destination_path: &Path,
    materialized_checksum: Option<&Sha256Hash>,
    expected_output_checksum: &Sha256Hash,
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
        let versioned_file = fs::File::open(versioned_path)?;
        let staged_checksum = {
            let mut checksummed_writer = ChecksummedWriter::new(&mut staged_file);
            match materialized_checksum {
                Some(_) => {
                    let mut decoded_file = bgzf::io::Reader::new(versioned_file);
                    io::copy(&mut decoded_file, &mut checksummed_writer)?;
                }
                None => {
                    let mut versioned_file = versioned_file;
                    io::copy(&mut versioned_file, &mut checksummed_writer)?;
                }
            }
            checksummed_writer.flush()?;
            checksummed_writer.checksum()
        };
        staged_file.sync_all()?;
        drop(staged_file);
        if staged_checksum != *expected_output_checksum {
            return Err(format!(
                "Copied asset at {} failed checksum validation",
                destination_path.display()
            )
            .into());
        }
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
    let expected_materialized_checksum = materialized_asset_checksum(asset)?;
    if !versioned_path.is_file() {
        return Err(format!(
            "Cannot materialize asset {} because its versioned file is missing: {}",
            asset.id,
            versioned_path.display()
        )
        .into());
    }
    if calculate_file_checksum(versioned_path)? != expected_checksum {
        return Err(format!(
            "Versioned asset {} at {} does not match its recorded checksum",
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
            .is_ok_and(|checksum| checksum == expected_materialized_checksum)
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
            conflict_destination_path(&destination_path, &expected_materialized_checksum)?;
        if !already_downloaded {
            copy_versioned_asset(
                versioned_path,
                &conflict_path,
                asset.materialized_checksum.as_ref(),
                &expected_materialized_checksum,
            )?;
        }
        return Ok(DownloadAssetOutcome::Conflict(conflict_path));
    }

    copy_versioned_asset(
        versioned_path,
        &destination_path,
        asset.materialized_checksum.as_ref(),
        &expected_materialized_checksum,
    )?;
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

/// Get the assets between two commits and excludes the already checkpointed commit.
fn get_remaining_assets_to_transfer(
    graph: &GraphConnection,
    assets_transfer_checkpoint: Option<&DoltHashId>,
    to_hash: &DoltHashId,
) -> Result<IndexMap<DoltHashId, Vec<AssetRef>>, QueryError> {
    let mut assets_by_commit = AssetRef::get_assets_by_commit(
        graph,
        assets_transfer_checkpoint,
        Some(to_hash),
        AssetView::Cumulative,
    )?;
    if let Some(assets_transfer_checkpoint) = assets_transfer_checkpoint
        && assets_by_commit
            .shift_remove(assets_transfer_checkpoint)
            .is_none()
    {
        return Err(QueryError::ResultsNotFound(format!(
            "Asset transfer checkpoint {assets_transfer_checkpoint} is not in the first-parent history of {to_hash}"
        )));
    }
    Ok(assets_by_commit)
}

/// Transfers the asset versions needed after a graph clone, pull, or push.
///
/// The CLI orchestration calls this after Dolt transfers graph history. It derives the asset delta
/// for the requested commit range, then dispatches `file://` transfers to
/// [`transfer_file_remote_assets`] or asks GenHub for HTTP transfer URLs. Clone and pull process
/// commits in order so each completed asset batch can advance the durable checkpoint. The
/// materialized view selects the one version per logical path copied out of `.gen/assets`.
fn transfer_assets(
    graph: &GraphConnection,
    workspace: &Workspace,
    remote: &Remote,
    operation: RemoteOperation,
    idempotency_token: Option<&Uuid>,
    target: AssetTransferTarget<'_>,
    mut complete_commit: impl FnMut(&DoltHashId) -> Result<(), Box<dyn Error>>,
) -> Result<Vec<AssetUploadReceipt>, Box<dyn Error>> {
    let commit_hash = hash_of(graph, target.history_ref)?;
    let range_assets: HashMap<_, _> =
        AssetRef::get_cumulative_assets_at(graph, target.range.from_commit, Some(&commit_hash))?
            .into_iter()
            .map(|asset| (asset.id, asset))
            .collect();
    let materialized_asset_ids: HashSet<_> = if target.range.materialize {
        AssetRef::get_materialized_assets_at(graph, None, Some(&commit_hash))?
            .into_iter()
            .map(|asset| asset.id)
            .collect()
    } else {
        HashSet::new()
    };
    let excluded_assets = if let Some(from_commit) = target.range.from_commit {
        AssetRef::get_cumulative_assets_at(graph, None, Some(from_commit))?
            .into_iter()
            .map(|asset| (asset.id, asset))
            .collect()
    } else {
        HashMap::new()
    };
    let mut previous_assets = if let Some(previous_hash) = target.range.previous_hash {
        AssetRef::get_cumulative_assets_at(graph, None, Some(previous_hash))?
            .into_iter()
            .map(|asset| (asset.id, asset))
            .collect()
    } else {
        HashMap::new()
    };
    previous_assets.extend(
        excluded_assets
            .iter()
            .map(|(asset_id, asset)| (*asset_id, asset.clone())),
    );
    let assets: HashMap<_, _> = range_assets
        .iter()
        .filter(|(id, _)| !excluded_assets.contains_key(id))
        .map(|(id, asset)| (*id, asset.clone()))
        .collect();
    if remote.url.starts_with("file://") {
        match operation {
            RemoteOperation::Push => transfer_file_remote_assets(
                workspace,
                remote,
                operation,
                &assets,
                &materialized_asset_ids,
                &previous_assets,
            )?,
            RemoteOperation::Clone | RemoteOperation::Pull => {
                let assets_by_commit = get_remaining_assets_to_transfer(
                    graph,
                    target.range.from_commit,
                    &commit_hash,
                )?;
                for (commit_hash, assets) in assets_by_commit {
                    let assets = assets
                        .into_iter()
                        .map(|asset| (asset.id, asset))
                        .collect::<HashMap<_, _>>();
                    transfer_file_remote_assets(
                        workspace,
                        remote,
                        operation,
                        &assets,
                        &materialized_asset_ids,
                        &previous_assets,
                    )?;
                    complete_commit(&commit_hash)?;
                }
            }
        }
        return Ok(Vec::new());
    }

    let repository = RepositoryRemote::parse(&remote.url)?;
    let request = AssetTransferRequest {
        operation,
        branch: target.branch,
        from_commit: target.range.from_commit,
        to_commit: Some(&commit_hash),
    };
    let idempotency_token = if operation == RemoteOperation::Push {
        idempotency_token
    } else {
        None
    };
    let response = match idempotency_token {
        Some(idempotency_token) => acquire_asset_transfers_with_idempotency_token(
            &repository,
            &request,
            idempotency_token,
            login_origin,
        )?,
        None => acquire_asset_transfers(&repository, &request, login_origin)?,
    };
    let client = Client::new();
    for transfer in &response.assets {
        if !range_assets.contains_key(&transfer.id) && !excluded_assets.contains_key(&transfer.id) {
            return Err(format!(
                "GenHub returned an asset transfer not present on branch '{}': {}",
                target.branch, transfer.id
            )
            .into());
        }
    }
    if operation == RemoteOperation::Push {
        let mut assets = assets;
        let mut upload_receipts = Vec::new();
        for transfer in response.assets {
            let Some(asset) = assets.remove(&transfer.id) else {
                continue;
            };
            upload_receipts.push(upload_asset(&client, workspace, &asset, &transfer.url)?);
        }
        if !assets.is_empty() {
            return Err(format!(
                "GenHub omitted {} local asset transfer(s) for branch '{}'",
                assets.len(),
                target.branch
            )
            .into());
        }
        return Ok(upload_receipts);
    }

    let mut transfer_urls = response
        .assets
        .into_iter()
        .map(|transfer| (transfer.id, transfer.url))
        .collect::<HashMap<_, _>>();
    let assets_by_commit =
        get_remaining_assets_to_transfer(graph, target.range.from_commit, &commit_hash)?;
    for (commit_hash, assets) in assets_by_commit {
        for asset in assets {
            let Some(url) = transfer_urls.remove(&asset.id) else {
                return Err(format!(
                    "GenHub omitted asset {} for branch '{}'",
                    asset.id, target.branch
                )
                .into());
            };
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
                &url,
            )? {
                warn_asset_conflict(workspace, &asset, destination_logical_path, &conflict_path)?;
            }
        }
        complete_commit(&commit_hash)?;
    }
    Ok(Vec::new())
}

pub fn clone_into_workspace(
    config: &ConfigConnection,
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
        let authorization =
            transfer_authorization(remote, RemoteOperation::Clone, None, false, None)?;
        match clone_remote(&graph, &authorization.remote_url) {
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
    let mut operation = RemoteOperationRecord::begin_or_resume(
        config,
        &remote.name,
        &branch,
        StoredRemoteOperationKind::Clone,
        None,
    )?;
    let destination_hash = hash_of(&graph, &branch)?;
    operation.set_destination(config, &destination_hash)?;
    let assets_transfer_checkpoint = operation.assets_transfer_checkpoint;
    let previous_hash = operation.from_commit;
    transfer_assets(
        &graph,
        workspace,
        remote,
        RemoteOperation::Clone,
        None,
        AssetTransferTarget {
            branch: &branch,
            history_ref: &branch,
            range: AssetTransferRange {
                from_commit: assets_transfer_checkpoint.as_ref(),
                previous_hash: previous_hash.as_ref(),
                materialize: true,
            },
        },
        |commit_hash| {
            operation.advance_assets_transfer_checkpoint(config, commit_hash)?;
            Ok(())
        },
    )?;
    operation.complete(config)?;
    Ok(branch)
}

/// Errors that can occur while pushing a branch to a remote repository.
#[derive(Debug, thiserror::Error)]
pub enum RemotePushError {
    /// The workspace paths could not be resolved.
    #[error(transparent)]
    Config(#[from] ConfigError),
    /// A config or graph database connection could not be opened.
    #[error(transparent)]
    Connection(#[from] ConnectionError),
    /// A local database operation failed.
    #[error(transparent)]
    Database(#[from] SqlError),
    /// The configured remote is invalid or could not be updated.
    #[error(transparent)]
    Remote(#[from] ModelRemoteError),
    /// A GenHub request failed.
    #[error(transparent)]
    Client(#[from] RemoteClientError),
    /// The remote could not be selected from the command and repository configuration.
    #[error("Unable to resolve push remote: {0}")]
    RemoteResolution(#[source] Box<dyn Error>),
    /// Dolt could not transfer the graph branch.
    #[error("Graph transfer failed: {0}")]
    GraphTransfer(#[source] Box<dyn Error>),
    /// The branch's assets could not be transferred.
    #[error("Asset transfer failed: {0}")]
    AssetTransfer(#[source] Box<dyn Error>),
    /// A successful GenHub graph push did not return its push lease.
    #[error("GenHub push did not return a transfer ID")]
    MissingTransferId,
    /// A pending push has only part of its persisted transfer lease metadata.
    #[error("Pending push metadata for branch '{branch}' has an incomplete transfer lease")]
    IncompleteTransferLease { branch: String },
}

pub fn execute_push(
    workspace: &Workspace,
    explicit_remote: Option<&str>,
    explicit_branch: Option<&str>,
    force: bool,
) -> Result<(), RemotePushError> {
    let config = get_config_connection(Some(workspace.gen_db_path()?))?;
    let intended_branch = Defaults::get_current_branch(&config);
    let graph = get_connection_for_branch(workspace.graph_db_path()?, intended_branch.as_deref())?;
    let persisted_branch = connect_persisted_branch(&graph, &config)?;
    let branch = if let Some(explicit_branch) = explicit_branch {
        explicit_branch.to_string()
    } else if let Some(persisted_branch) = persisted_branch {
        persisted_branch
    } else {
        active_branch(&graph)?
    };
    let remote = resolve_remote(&config, explicit_remote, &branch)
        .map_err(RemotePushError::RemoteResolution)?;
    let push_idempotency_token = if remote.url.starts_with("file://") {
        None
    } else {
        Some(Uuid::now_v7())
    };
    // A missing or stale tracking ref only makes the transfer conservatively include more assets.
    // Force pushes cannot use the tracking ref as a lower bound because they may replace history.
    let tracking_ref = format!("{}/{branch}", remote.name);
    let previous_hash = (!force)
        .then(|| hash_of(&graph, &tracking_ref).ok())
        .flatten();
    let mut push_context = if remote.url.starts_with("file://") {
        run_graph_transfer(
            &graph,
            &remote,
            RemoteOperation::Push,
            &branch,
            force,
            None,
            || push_graph_branch(&graph, &remote.name, &branch, force, None),
        )
        .map_err(RemotePushError::GraphTransfer)?;
        None
    } else {
        let mut operation = RemoteOperationRecord::begin_or_resume(
            &config,
            &remote.name,
            &branch,
            StoredRemoteOperationKind::Push,
            previous_hash.as_ref(),
        )?;
        let destination_hash = hash_of(&graph, &branch)?;
        let transfer_lease = match (
            operation.to_commit.as_ref(),
            operation.transfer_id,
            operation.transfer_expires_at,
        ) {
            (Some(recorded_destination), Some(transfer_id), Some(expires_at))
                if recorded_destination == &destination_hash
                    && expires_at > Utc::now().timestamp() =>
            {
                PushTransferLease {
                    transfer_id,
                    expires_at,
                }
            }
            (Some(_), Some(_), Some(_)) | (None, None, None) => {
                let graph_transfer = run_graph_transfer(
                    &graph,
                    &remote,
                    RemoteOperation::Push,
                    &branch,
                    force,
                    push_idempotency_token.as_ref(),
                    || {
                        push_graph_branch(
                            &graph,
                            &remote.name,
                            &branch,
                            force,
                            push_idempotency_token.as_ref(),
                        )
                    },
                );
                let transfer_lease = match graph_transfer {
                    Ok(Some(transfer_lease)) => transfer_lease,
                    Ok(None) => {
                        return Err(RemotePushError::MissingTransferId);
                    }
                    Err(error) => {
                        if operation.to_commit.is_none()
                            && let Err(metadata_error) = operation.fail(&config)
                        {
                            eprintln!(
                                "Warning: failed to record unsuccessful push operation for branch '{branch}': {metadata_error}"
                            );
                        }
                        return Err(RemotePushError::GraphTransfer(error));
                    }
                };
                operation.set_push_destination(
                    &config,
                    &destination_hash,
                    transfer_lease.transfer_id,
                    transfer_lease.expires_at,
                )?;
                transfer_lease
            }
            _ => {
                return Err(RemotePushError::IncompleteTransferLease { branch });
            }
        };
        Some((operation, transfer_lease, destination_hash))
    };
    let assets_transfer_checkpoint = if let Some((operation, _, _)) = push_context.as_ref() {
        operation.assets_transfer_checkpoint.as_ref()
    } else {
        previous_hash.as_ref()
    };
    let upload_receipts = transfer_assets(
        &graph,
        workspace,
        &remote,
        RemoteOperation::Push,
        push_idempotency_token.as_ref(),
        AssetTransferTarget {
            branch: &branch,
            history_ref: &branch,
            range: AssetTransferRange {
                from_commit: assets_transfer_checkpoint,
                previous_hash: previous_hash.as_ref(),
                materialize: true,
            },
        },
        |_| Ok(()),
    )
    .map_err(RemotePushError::AssetTransfer)?;
    if let Some((operation, transfer_lease, destination_hash)) = push_context.as_mut() {
        let repository = RepositoryRemote::parse(&remote.url)?;
        let completion = AssetTransferCompletionRequest {
            transfer_id: transfer_lease.transfer_id,
            branch: &branch,
            assets: &upload_receipts,
        };
        match push_idempotency_token.as_ref() {
            Some(idempotency_token) => complete_asset_transfers_with_idempotency_token(
                &repository,
                &completion,
                idempotency_token,
                login_origin,
            )?,
            None => complete_asset_transfers(&repository, &completion, login_origin)?,
        }
        operation.advance_assets_transfer_checkpoint(&config, destination_hash)?;
        operation.complete(&config)?;
    }
    if let Err(error) = run_graph_transfer(
        &graph,
        &remote,
        RemoteOperation::Pull,
        &branch,
        false,
        None,
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
    let mut operation = RemoteOperationRecord::begin_or_resume(
        &config,
        &remote.name,
        &branch,
        StoredRemoteOperationKind::Pull,
        previous_hash.as_ref(),
    )?;
    if let Err(error) = run_graph_transfer(
        &graph,
        &remote,
        RemoteOperation::Pull,
        &branch,
        false,
        None,
        || pull(&graph, &remote.name, &branch),
    ) {
        if let Err(metadata_error) = operation.fail(&config) {
            eprintln!(
                "Warning: failed to record unsuccessful pull operation for branch '{branch}': {metadata_error}"
            );
        }
        return Err(error);
    }
    let destination_hash = hash_of(&graph, &branch)?;
    operation.set_destination(&config, &destination_hash)?;
    let assets_transfer_checkpoint = operation.assets_transfer_checkpoint;
    let previous_hash = operation.from_commit;
    transfer_assets(
        &graph,
        workspace,
        &remote,
        RemoteOperation::Pull,
        None,
        AssetTransferTarget {
            branch: &branch,
            history_ref: &branch,
            range: AssetTransferRange {
                from_commit: assets_transfer_checkpoint.as_ref(),
                previous_hash: previous_hash.as_ref(),
                materialize: true,
            },
        },
        |commit_hash| {
            operation.advance_assets_transfer_checkpoint(&config, commit_hash)?;
            Ok(())
        },
    )?;
    operation.complete(&config)?;
    RemoteBranch::set_remote_validated(&config, &branch, Some(&remote.name))?;
    println!("Pulled branch '{branch}' from '{}'.", remote.name);
    Ok(())
}

/// Fetches a remote branch into its remote-tracking ref and hydrates immutable asset history.
///
/// Fetch deliberately leaves the local branch, checkout, and logical workspace files unchanged.
pub fn execute_fetch(
    workspace: &Workspace,
    explicit_remote: Option<&str>,
    explicit_branch: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    let config = get_config_connection(Some(workspace.gen_db_path()?))?;
    let graph = get_connection_for_branch(workspace.graph_db_path()?, None)?;
    let branch = explicit_branch
        .map(str::to_string)
        .or_else(|| Defaults::get_current_branch(&config))
        .unwrap_or(active_branch(&graph)?);
    let remote = resolve_remote(&config, explicit_remote, &branch)?;
    run_graph_transfer(
        &graph,
        &remote,
        RemoteOperation::Pull,
        &branch,
        false,
        None,
        || fetch(&graph, &remote.name, Some(&branch)),
    )?;

    let tracking_ref = format!("{}/{}", remote.name, branch);
    transfer_assets(
        &graph,
        workspace,
        &remote,
        RemoteOperation::Pull,
        None,
        AssetTransferTarget {
            branch: &branch,
            history_ref: &tracking_ref,
            range: AssetTransferRange {
                from_commit: None,
                previous_hash: None,
                materialize: false,
            },
        },
        |_| Ok(()),
    )?;
    println!("Fetched branch '{branch}' from '{}'.", remote.name);
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
        net::{TcpListener, TcpStream},
        path::PathBuf,
        sync::Mutex,
        thread,
        time::{Duration, Instant},
    };

    use chrono::Utc;
    use flate2::{Compression, write::GzEncoder};
    use gen_core::{DoltHashId, HashId, Sha256Hash, config::Workspace};
    use gen_models::{
        assets::{AssetRef, AssetRole, LocalAssetUri, materialization_destination_path},
        collection::Collection,
        db::GraphConnection,
        file_types::FileTypes,
        history::dolt::{
            add_remote, clone_remote, commit_all, hash_of, pull, remote_rows, remove_remote,
        },
        operations::{
            Defaults, FileAddition, OperationFile, Remote,
            RemoteOperationKind as StoredRemoteOperationKind, RemoteOperationRecord,
            calculate_reader_checksum,
        },
    };
    use noodles::bgzf;
    use reqwest::blocking::Client;
    use rusqlite::{Connection, Error as SqlError};
    use serde_json::json;
    use tempfile::tempdir;
    use uuid::Uuid;

    use super::{
        AssetTransferRange, AssetTransferTarget, DownloadAssetOutcome, PushTransferLease,
        RemoteOperation, canonical_remote_url, clone_destination_name, copy_versioned_asset,
        download_asset, download_to_versioned_store, execute_pull, execute_push, file_graph_url,
        get_remaining_assets_to_transfer, push_graph_branch, resolve_remote, run_graph_transfer,
        temporary_path, transfer_assets,
    };
    use crate::{get_config_connection, get_connection, get_raw_connection};

    static ENVIRONMENT_LOCK: Mutex<()> = Mutex::new(());
    const TEST_TRANSFER_ID: Uuid = Uuid::from_u128(1);
    const RETRIED_TRANSFER_ID: Uuid = Uuid::from_u128(2);

    fn request_idempotency_token(request: &str) -> Option<Uuid> {
        request
            .lines()
            .take_while(|line| !line.is_empty())
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("idempotency-token")
                    .then(|| Uuid::parse_str(value.trim()).ok())
                    .flatten()
            })
    }

    fn read_native_protocol_request(stream: &mut TcpStream) -> String {
        let mut request = Vec::new();
        let mut expected_length = None;
        let mut buffer = [0_u8; 4096];
        loop {
            let read = stream
                .read(&mut buffer)
                .expect("should read native Dolt HTTP request");
            if read == 0 {
                break;
            }
            request.extend_from_slice(&buffer[..read]);
            if expected_length.is_none()
                && let Some(header_end) =
                    request.windows(4).position(|window| window == b"\r\n\r\n")
            {
                let headers = String::from_utf8_lossy(&request[..header_end]);
                let content_length = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length").then(|| {
                            value
                                .trim()
                                .parse::<usize>()
                                .expect("should parse request content length")
                        })
                    })
                    .unwrap_or(0);
                expected_length = Some(header_end + 4 + content_length);
            }
            if expected_length.is_some_and(|length| request.len() >= length) {
                break;
            }
        }
        String::from_utf8_lossy(&request).into_owned()
    }

    fn serve_unauthorized_dolt_remote() -> (String, thread::JoinHandle<String>) {
        let listener =
            TcpListener::bind("127.0.0.1:0").expect("should bind native Dolt HTTP endpoint");
        listener
            .set_nonblocking(true)
            .expect("should make native Dolt listener nonblocking");
        let address = listener
            .local_addr()
            .expect("should read native Dolt HTTP address");
        let server = thread::spawn(move || {
            let started_at = Instant::now();
            let (mut stream, _) = loop {
                match listener.accept() {
                    Ok(connection) => break connection,
                    Err(error)
                        if error.kind() == std::io::ErrorKind::WouldBlock
                            && started_at.elapsed() < Duration::from_secs(10) =>
                    {
                        thread::sleep(Duration::from_millis(10));
                    }
                    Err(error) => panic!("should accept native Dolt HTTP request: {error}"),
                }
            };
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .expect("should set request read timeout");
            let request = read_native_protocol_request(&mut stream);
            stream
                .write_all(
                    b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                )
                .expect("should reject native Dolt HTTP request");
            request
        });
        (format!("http://{address}/dolt/remote.db"), server)
    }

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
        use crate::{get_config_connection, get_connection};

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
                             \"default_branch\":\"main\",\
                             \"transfer_id\":\"{}\"}}",
                            super::TEST_TRANSFER_ID
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

            workspace.ensure_gen_dir();
            let config = get_config_connection(Some(
                workspace
                    .gen_db_path()
                    .expect("should resolve clone config path"),
            ))
            .expect("should create clone config database");
            Remote::create(&config, &remote.name, &remote.url)
                .expect("should configure clone remote");
            let result = clone_into_workspace(&config, &remote, &workspace);
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
            assert!(
                requests
                    .iter()
                    .all(|request| super::request_idempotency_token(request).is_none()),
                "clone should not send an idempotency token"
            );
            assert_eq!(result.expect("clone retry should succeed"), "main");
        }
    }

    fn test_asset(contents: &[u8], logical_path: &str, created_on: i64) -> AssetRef {
        let checksum = calculate_reader_checksum(Cursor::new(contents)).expect("should checksum");
        let uri = LocalAssetUri::asset_uri(logical_path);
        let role = AssetRole::Input;
        let file_addition = FileAddition {
            id: HashId::convert_str("remote-test-asset"),
            asset_uri: uri.clone(),
            file_type: FileTypes::None,
            checksum: Some(checksum),
            materialized_checksum: None,
        };
        AssetRef {
            id: AssetRef::id_hash(&file_addition, &role, Some(logical_path), None, None),
            uri,
            file_type: FileTypes::None.as_str().to_string(),
            checksum: Some(checksum),
            materialized_checksum: None,
            size: Some(i64::try_from(contents.len()).expect("asset should fit in i64")),
            role,
            logical_path: Some(logical_path.to_string()),
            name: None,
            created_on,
            upstream_asset_ref_id: None,
        }
    }

    fn test_bgzf_asset(
        contents: &[u8],
        logical_path: &str,
        created_on: i64,
    ) -> (AssetRef, Vec<u8>) {
        let archived_contents = test_bgzf_contents(contents);

        let mut asset = test_asset(&archived_contents, logical_path, created_on);
        asset.materialized_checksum = Some(
            calculate_reader_checksum(Cursor::new(contents)).expect("should checksum plain bytes"),
        );
        set_test_asset_identity(&mut asset, FileTypes::None);
        (asset, archived_contents)
    }

    fn test_bgzf_contents(contents: &[u8]) -> Vec<u8> {
        let mut archived_contents = Vec::new();
        let mut writer = bgzf::io::Writer::new(&mut archived_contents);
        writer
            .write_all(contents)
            .expect("should write test BGZF contents");
        writer.finish().expect("should finish test BGZF stream");
        archived_contents
    }

    fn test_bam_contents(comment: &str) -> Vec<u8> {
        let header = format!("@CO\t{comment}\n");
        let mut contents = b"BAM\x01".to_vec();
        contents.extend_from_slice(
            &i32::try_from(header.len())
                .expect("should fit BAM header length in i32")
                .to_le_bytes(),
        );
        contents.extend_from_slice(header.as_bytes());
        contents.extend_from_slice(&0_i32.to_le_bytes());
        contents
    }

    fn set_test_asset_identity(asset: &mut AssetRef, file_type: FileTypes) {
        asset.file_type = file_type.as_str().to_string();
        let file_addition = FileAddition {
            id: HashId::convert_str("remote-test-asset-identity"),
            asset_uri: asset.uri.clone(),
            file_type,
            checksum: asset.checksum,
            materialized_checksum: asset.materialized_checksum,
        };
        asset.id = AssetRef::id_hash(
            &file_addition,
            &asset.role,
            asset.logical_path.as_deref(),
            asset.name.as_deref(),
            asset.upstream_asset_ref_id.as_ref(),
        );
    }

    fn test_gzip_contents(contents: &[u8]) -> Vec<u8> {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder
            .write_all(contents)
            .expect("should write gzip content");
        encoder.finish().expect("should finish gzip stream")
    }

    fn write_test_versioned_asset(
        workspace: &Workspace,
        asset: &AssetRef,
        archived_contents: &[u8],
    ) -> PathBuf {
        let path = versioned_asset_path(workspace, asset);
        fs::create_dir_all(path.parent().expect("should have versioned asset parent"))
            .expect("should create versioned asset directory");
        fs::write(&path, archived_contents).expect("should write versioned archive");
        path
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

    fn serve_pull_api(
        graph_url: &str,
        asset_id: HashId,
        asset_url: &str,
        pull_count: usize,
        fail_first_asset: bool,
    ) -> (String, thread::JoinHandle<Vec<String>>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("should bind pull API server");
        let address = listener
            .local_addr()
            .expect("should read pull API server address");
        let graph_url = graph_url.to_string();
        let asset_url = asset_url.to_string();
        let handle = thread::spawn(move || {
            let mut requests = Vec::new();
            let mut asset_request_count = 0;
            for _ in 0..(pull_count * 2) {
                let (mut stream, _) = listener.accept().expect("should accept pull API request");
                let mut request = [0_u8; 8192];
                let read = stream
                    .read(&mut request)
                    .expect("should read pull API request");
                let request = String::from_utf8_lossy(&request[..read]).into_owned();
                let response_body = if request.contains("/remote-capability ") {
                    json!({
                        "remote_url": graph_url,
                        "expires_at": "2030-01-01T00:00:00Z",
                        "default_branch": "main",
                        "transfer_id": TEST_TRANSFER_ID
                    })
                    .to_string()
                } else if request.contains("/asset-transfers ") {
                    let download_url = if fail_first_asset && asset_request_count == 0 {
                        "http://127.0.0.1:1/unavailable"
                    } else {
                        &asset_url
                    };
                    asset_request_count += 1;
                    json!({
                        "assets": [{ "id": asset_id, "url": download_url }]
                    })
                    .to_string()
                } else {
                    panic!("unexpected pull API request: {request}");
                };
                requests.push(request);
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response_body}",
                    response_body.len()
                )
                .expect("should write pull API response");
            }
            requests
        });
        (format!("http://{address}/api/repos/alice/example"), handle)
    }

    fn serve_checkpoint_pull_api(
        graph_url: &str,
        first_asset: (HashId, &str),
        second_asset: (HashId, &str),
    ) -> (String, thread::JoinHandle<Vec<String>>) {
        let listener =
            TcpListener::bind("127.0.0.1:0").expect("should bind checkpoint pull API server");
        let address = listener
            .local_addr()
            .expect("should read checkpoint pull API server address");
        let graph_url = graph_url.to_string();
        let first_asset_url = first_asset.1.to_string();
        let second_asset_url = second_asset.1.to_string();
        let handle = thread::spawn(move || {
            let mut requests = Vec::new();
            let mut asset_request_count = 0;
            for _ in 0..4 {
                let (mut stream, _) = listener.accept().expect("should accept pull API request");
                let mut request = [0_u8; 8192];
                let read = stream
                    .read(&mut request)
                    .expect("should read pull API request");
                let request = String::from_utf8_lossy(&request[..read]).into_owned();
                let response_body = if request.contains("/remote-capability ") {
                    json!({
                        "remote_url": graph_url,
                        "expires_at": "2030-01-01T00:00:00Z",
                        "default_branch": "main",
                        "transfer_id": TEST_TRANSFER_ID
                    })
                    .to_string()
                } else if request.contains("/asset-transfers ") {
                    let assets = if asset_request_count == 0 {
                        json!([
                            {
                                "id": second_asset.0,
                                "url": "http://127.0.0.1:1/unavailable"
                            },
                            { "id": first_asset.0, "url": first_asset_url }
                        ])
                    } else {
                        json!([{ "id": second_asset.0, "url": second_asset_url }])
                    };
                    asset_request_count += 1;
                    json!({ "assets": assets }).to_string()
                } else {
                    panic!("unexpected pull API request: {request}");
                };
                requests.push(request);
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response_body}",
                    response_body.len()
                )
                .expect("should write pull API response");
            }
            requests
        });
        (format!("http://{address}/api/repos/alice/example"), handle)
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
    fn test_download_asset_restores_plain_fasta_and_vcf_from_bgzf() {
        let inputs = [
            (
                "reference.fa",
                b">chr1\nACGTACGT\n".as_slice(),
            ),
            (
                "variants.vcf",
                b"##fileformat=VCFv4.3\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\nchr1\t2\t.\tC\tT\t.\tPASS\t.\n"
                    .as_slice(),
            ),
            ("ordinary.fa.gz", b">chr2\nTTGGCC\n".as_slice()),
        ];

        for (logical_path, plain_contents) in inputs {
            let temp = tempdir().expect("should create workspace");
            let workspace = Workspace::new(temp.path());
            workspace.ensure_gen_dir();
            let (asset, archived_contents) = test_bgzf_asset(plain_contents, logical_path, 1);
            let (url, server) = serve_asset(&archived_contents);

            let outcome = download_asset(
                &Client::new(),
                &workspace,
                &asset,
                &HashMap::new(),
                asset.logical_path.as_deref(),
                &url,
            )
            .expect("should download and decode archived input");
            server.join().expect("asset server should finish");

            assert_eq!(outcome, DownloadAssetOutcome::Downloaded);
            assert_eq!(
                fs::read(temp.path().join(logical_path)).expect("should read plain input"),
                plain_contents,
                "the workspace copy should restore the original {logical_path} bytes"
            );
            assert_eq!(
                fs::read(versioned_asset_path(&workspace, &asset))
                    .expect("should read archived input"),
                archived_contents,
                "the checksum-addressed store should retain BGZF bytes"
            );
        }
    }

    #[test]
    fn test_download_asset_preserves_bgzf_for_normalized_gzip_source() {
        let source_contents = b">chr1\nACGTACGT\n";
        let source_temp = tempdir().expect("should create source workspace");
        let source_workspace = Workspace::new(source_temp.path());
        source_workspace.ensure_gen_dir();
        let source_path = source_temp.path().join("ordinary.fa.gz");
        let source_gzip_contents = test_gzip_contents(source_contents);
        fs::write(&source_path, &source_gzip_contents).expect("should write ordinary gzip source");
        let asset = OperationFile::new(source_path.to_string_lossy())
            .set_file_type(FileTypes::Fasta)
            .prepare_asset_ref(&source_workspace, 1)
            .expect("should retain and describe the gzip source");
        let archived_path = versioned_asset_path(&source_workspace, &asset);
        let archived_contents =
            fs::read(&archived_path).expect("should read normalized BGZF archive");

        assert_eq!(fs::read(&source_path).unwrap(), source_gzip_contents);
        assert_ne!(archived_contents, source_gzip_contents);
        assert_eq!(asset.materialized_checksum, None);
        let mut decoded = bgzf::io::Reader::new(Cursor::new(&archived_contents));
        let mut decoded_contents = Vec::new();
        decoded
            .read_to_end(&mut decoded_contents)
            .expect("should decode normalized BGZF archive");
        assert_eq!(decoded_contents, source_contents);

        let destination_temp = tempdir().expect("should create destination workspace");
        let workspace = Workspace::new(destination_temp.path());
        workspace.ensure_gen_dir();
        let (url, server) = serve_asset(&archived_contents);

        let outcome = download_asset(
            &Client::new(),
            &workspace,
            &asset,
            &HashMap::new(),
            asset.logical_path.as_deref(),
            &url,
        )
        .expect("should download normalized gzip archive");
        server.join().expect("should finish asset server");

        assert_eq!(outcome, DownloadAssetOutcome::Downloaded);
        assert_eq!(
            fs::read(destination_temp.path().join("ordinary.fa.gz"))
                .expect("should read preserved compressed source"),
            archived_contents,
            "a normalized gzip source should retain BGZF bytes at its workspace path"
        );
    }

    #[test]
    fn test_download_asset_replaces_a_known_plain_bgzf_version() {
        let temp = tempdir().expect("should create workspace");
        let workspace = Workspace::new(temp.path());
        workspace.ensure_gen_dir();
        let destination = temp.path().join("reference.fa");
        let previous_contents = b">chr1\nAACCGG\n";
        fs::write(&destination, previous_contents).expect("should write previous plain file");
        let (previous_asset, _) = test_bgzf_asset(previous_contents, "reference.fa", 1);
        let previous_assets = HashMap::from([(previous_asset.id, previous_asset)]);
        let current_contents = b">chr1\nTTGGCC\n";
        let (current_asset, archived_contents) =
            test_bgzf_asset(current_contents, "reference.fa", 2);
        let (url, server) = serve_asset(&archived_contents);

        let outcome = download_asset(
            &Client::new(),
            &workspace,
            &current_asset,
            &previous_assets,
            current_asset.logical_path.as_deref(),
            &url,
        )
        .expect("should replace a known previous plain file");
        server.join().expect("asset server should finish");

        assert_eq!(outcome, DownloadAssetOutcome::Downloaded);
        assert_eq!(
            fs::read(&destination).expect("should read replaced plain file"),
            current_contents
        );
        assert_eq!(
            fs::read(versioned_asset_path(&workspace, &current_asset))
                .expect("should read retained BGZF archive"),
            archived_contents
        );
        assert!(!temp.path().join("reference.fa.conflict").exists());
    }

    #[test]
    fn test_download_asset_deduplicates_materialized_file_and_conflict_by_plain_hash() {
        let plain_contents = b">chr1\nACGTACGT\n";
        let (asset, archived_contents) = test_bgzf_asset(plain_contents, "reference.fa", 1);

        let matching_temp = tempdir().expect("should create matching workspace");
        let matching_workspace = Workspace::new(matching_temp.path());
        matching_workspace.ensure_gen_dir();
        fs::write(matching_temp.path().join("reference.fa"), plain_contents)
            .expect("should write matching plain source");
        let (url, server) = serve_asset(&archived_contents);
        let outcome = download_asset(
            &Client::new(),
            &matching_workspace,
            &asset,
            &HashMap::new(),
            asset.logical_path.as_deref(),
            &url,
        )
        .expect("should recognize matching materialized bytes");
        server.join().expect("asset server should finish");
        assert_eq!(outcome, DownloadAssetOutcome::Unchanged);
        assert_eq!(
            fs::read(versioned_asset_path(&matching_workspace, &asset))
                .expect("should read stored archive"),
            archived_contents
        );

        let conflict_temp = tempdir().expect("should create conflict workspace");
        let conflict_workspace = Workspace::new(conflict_temp.path());
        conflict_workspace.ensure_gen_dir();
        fs::write(conflict_temp.path().join("reference.fa"), b"local edits\n")
            .expect("should write locally modified source");
        let conflict_path = conflict_temp.path().join("reference.fa.conflict");
        fs::write(&conflict_path, plain_contents).expect("should seed decoded conflict copy");
        let (url, server) = serve_asset(&archived_contents);
        let outcome = download_asset(
            &Client::new(),
            &conflict_workspace,
            &asset,
            &HashMap::new(),
            asset.logical_path.as_deref(),
            &url,
        )
        .expect("should reuse matching decoded conflict copy");
        server.join().expect("asset server should finish");

        assert_eq!(
            outcome,
            DownloadAssetOutcome::Conflict(conflict_path.clone())
        );
        assert_eq!(
            fs::read(conflict_path).expect("should read reused conflict"),
            plain_contents
        );
        assert!(
            !conflict_temp
                .path()
                .join("reference.fa.conflict.1")
                .exists(),
            "matching decoded conflicts should be reused by their plain-byte checksum"
        );
        assert_eq!(
            fs::read(versioned_asset_path(&conflict_workspace, &asset))
                .expect("should read stored archive"),
            archived_contents
        );
    }

    #[test]
    fn test_download_asset_fetches_bgzf_without_materializing_workspace_file() {
        let temp = tempdir().expect("should create workspace");
        let workspace = Workspace::new(temp.path());
        workspace.ensure_gen_dir();
        let plain_contents = b">chr1\nACGT\n";
        let (asset, archived_contents) = test_bgzf_asset(plain_contents, "reference.fa", 1);
        let (url, server) = serve_asset(&archived_contents);

        let outcome = download_asset(
            &Client::new(),
            &workspace,
            &asset,
            &HashMap::new(),
            None,
            &url,
        )
        .expect("should retain an archived-only historical asset");
        server.join().expect("asset server should finish");

        assert_eq!(outcome, DownloadAssetOutcome::Downloaded);
        assert!(!temp.path().join("reference.fa").exists());
        assert_eq!(
            fs::read(versioned_asset_path(&workspace, &asset))
                .expect("should read archived-only BGZF asset"),
            archived_contents
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
        let older_contents = b">chr1\nAAAACCCC\n";
        fs::write(&destination, older_contents).expect("should write older managed file");
        let (older_asset, _) = test_bgzf_asset(older_contents, "reference.fa", 1);
        let (newer_asset, _) = test_bgzf_asset(b">chr1\nGGGGTTTT\n", "reference.fa", 2);
        let previous_assets =
            HashMap::from([(older_asset.id, older_asset), (newer_asset.id, newer_asset)]);
        let remote_contents = b">chr1\nACACACAC\n";
        let (remote_asset, archived_contents) = test_bgzf_asset(remote_contents, "reference.fa", 3);
        let (url, server) = serve_asset(&archived_contents);

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
        assert_eq!(
            fs::read(versioned_asset_path(&workspace, &remote_asset))
                .expect("should read archived latest version"),
            archived_contents
        );
        assert!(!temp.path().join("reference.fa.conflict").exists());
    }

    #[test]
    fn test_download_asset_compares_compressed_source_by_decoded_content() {
        for logical_path in [
            "variants.vcf.gz",
            "variants.vcf.bgz",
            "variants.vcf.bgzf",
            "reads.bam",
        ] {
            let (previous_contents, changed_contents, current_contents) =
                if logical_path.ends_with(".bam") {
                    (
                        test_bam_contents("previous"),
                        test_bam_contents("local edit"),
                        test_bam_contents("current"),
                    )
                } else {
                    (
                        b"##fileformat=VCFv4.3\nchr1\t2\t.\tC\tT\t.\tPASS\t.\n".to_vec(),
                        b"##fileformat=VCFv4.3\nchr1\t2\t.\tC\tG\t.\tPASS\t.\n".to_vec(),
                        b"##fileformat=VCFv4.3\nchr1\t3\t.\tG\tA\t.\tPASS\t.\n".to_vec(),
                    )
                };
            let previous_archive = test_bgzf_contents(&previous_contents);
            let mut previous_asset = test_asset(&previous_archive, logical_path, 1);
            if logical_path.ends_with(".gz") {
                set_test_asset_identity(&mut previous_asset, FileTypes::VCF);
            }
            let previous_assets = HashMap::from([(previous_asset.id, previous_asset.clone())]);

            let current_archive = test_bgzf_contents(&current_contents);
            let mut current_asset = test_asset(&current_archive, logical_path, 2);
            if logical_path.ends_with(".gz") {
                set_test_asset_identity(&mut current_asset, FileTypes::VCF);
            }

            for (workspace_contents, should_conflict) in [
                (previous_contents.as_slice(), false),
                (changed_contents.as_slice(), true),
            ] {
                let temp = tempdir().expect("should create workspace");
                let workspace = Workspace::new(temp.path());
                workspace.ensure_gen_dir();
                write_test_versioned_asset(&workspace, &previous_asset, &previous_archive);
                let destination = temp.path().join(logical_path);
                let compressed_source = if logical_path.ends_with(".gz") {
                    test_gzip_contents(workspace_contents)
                } else {
                    let mut contents = test_bgzf_contents(workspace_contents);
                    // A valid BGZF timestamp change keeps the payload but changes its raw checksum.
                    contents[4..8].copy_from_slice(&1_u32.to_le_bytes());
                    contents
                };
                fs::write(&destination, &compressed_source)
                    .expect("should write compressed workspace source");
                let (url, server) = serve_asset(&current_archive);

                let outcome = download_asset(
                    &Client::new(),
                    &workspace,
                    &current_asset,
                    &previous_assets,
                    current_asset.logical_path.as_deref(),
                    &url,
                )
                .expect("should compare compressed asset content");
                server.join().expect("should finish asset server");

                let conflict_path = temp.path().join(format!("{logical_path}.conflict"));
                if should_conflict {
                    assert_eq!(
                        outcome,
                        DownloadAssetOutcome::Conflict(conflict_path.clone())
                    );
                    assert_eq!(
                        fs::read(&destination).expect("should read preserved compressed source"),
                        compressed_source,
                        "changed compressed source should remain untouched"
                    );
                    assert_eq!(
                        fs::read(conflict_path).expect("should read current conflict asset"),
                        current_archive
                    );
                } else {
                    assert_eq!(outcome, DownloadAssetOutcome::Downloaded);
                    assert_eq!(
                        fs::read(&destination).expect("should read updated compressed path"),
                        current_archive,
                        "the updated compressed path should retain BGZF bytes"
                    );
                    assert!(
                        !conflict_path.exists(),
                        "matching decoded source should not create a conflict"
                    );
                }
            }
        }
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
        let remote_contents = b">chr1\nGATTACA\n";
        let (remote_asset, archived_contents) = test_bgzf_asset(remote_contents, "reference.fa", 2);
        let (url, server) = serve_asset(&archived_contents);

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
            archived_contents,
            "conflicting remote asset should be retained in versioned storage"
        );
    }

    #[test]
    fn test_materialize_versioned_bgzf_asset_failures_preserve_destination() {
        let previous_contents = b"previous plain version\n";
        let previous_asset = test_asset(previous_contents, "reference.fa", 1);
        let previous_assets = HashMap::from([(previous_asset.id, previous_asset)]);
        let cases = [
            (
                "checksum mismatch",
                test_bgzf_asset(b">chr1\nACGT\n", "reference.fa", 2).1,
                calculate_reader_checksum(Cursor::new(b"wrong decoded bytes\n"))
                    .expect("should checksum wrong plain bytes"),
            ),
            (
                "invalid BGZF",
                b"not a BGZF stream".to_vec(),
                calculate_reader_checksum(Cursor::new(b"expected plain bytes\n"))
                    .expect("should checksum expected plain bytes"),
            ),
        ];

        for (failure, archived_contents, materialized_checksum) in cases {
            let temp = tempdir().expect("should create workspace");
            let workspace = Workspace::new(temp.path());
            workspace.ensure_gen_dir();
            let destination = temp.path().join("reference.fa");
            fs::write(&destination, previous_contents).expect("should write previous file");
            let mut asset = test_asset(&archived_contents, "reference.fa", 2);
            asset.materialized_checksum = Some(materialized_checksum);
            let versioned_path = versioned_asset_path(&workspace, &asset);
            fs::create_dir_all(
                versioned_path
                    .parent()
                    .expect("should resolve versioned asset directory"),
            )
            .expect("should create versioned asset directory");
            fs::write(&versioned_path, &archived_contents).expect("should write versioned archive");

            let error = super::materialize_versioned_asset(
                &workspace,
                &asset,
                &previous_assets,
                asset.logical_path.as_deref(),
                &versioned_path,
                false,
            )
            .expect_err("should reject invalid materialized input");

            assert!(
                !error.to_string().is_empty(),
                "{failure} should return a useful error"
            );
            assert_eq!(
                fs::read(&destination).expect("should read preserved destination"),
                previous_contents,
                "{failure} must not truncate or replace the existing logical file"
            );
            assert!(
                !temporary_path(&destination)
                    .expect("should resolve materialization staging path")
                    .exists(),
                "{failure} should remove its staged workspace file"
            );
            assert_eq!(
                fs::read(&versioned_path).expect("should read retained archive"),
                archived_contents,
                "{failure} should leave archived bytes intact"
            );
        }
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
        let expected_checksum = Sha256Hash::convert_str("missing-versioned-asset");

        copy_versioned_asset(
            &missing_versioned_path,
            &destination_path,
            None,
            &expected_checksum,
        )
        .expect_err("should reject a missing versioned asset");

        assert_eq!(
            fs::read(destination_path).expect("should read preserved logical asset"),
            b"previous logical bytes\n",
            "failed copy should preserve the previous logical asset"
        );
    }

    #[test]
    fn test_remaining_assets_to_transfer_rejects_unrelated_checkpoint() {
        let temp = tempdir().expect("should create transfer workspace");
        let graph = get_connection(temp.path().join("graph.db"))
            .expect("should create transfer graph database");
        Collection::create(&graph, "base").expect("should create base collection");
        let destination =
            commit_all(&graph, "create base collection").expect("should commit base collection");
        let unrelated_checkpoint = DoltHashId([9_u8; 20]);

        let error =
            get_remaining_assets_to_transfer(&graph, Some(&unrelated_checkpoint), &destination)
                .expect_err("should reject a checkpoint outside first-parent history");

        assert!(
            error
                .to_string()
                .contains("is not in the first-parent history")
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
            None,
            AssetTransferTarget {
                branch: "main",
                history_ref: "main",
                range: AssetTransferRange {
                    from_commit: Some(&previous_hash),
                    previous_hash: Some(&previous_hash),
                    materialize: true,
                },
            },
            |_| Ok(()),
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
            None,
            AssetTransferTarget {
                branch: "main",
                history_ref: "main",
                range: AssetTransferRange {
                    from_commit: None,
                    previous_hash: None,
                    materialize: true,
                },
            },
            |_| Ok(()),
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
    fn test_execute_pull_transfers_assets_after_graph_db_is_synced() {
        // Test that if we have transferred a graph.db, on a subsequent pull we will transfer assets. This mimics
        // things like resumes.
        let temp = tempdir().expect("should create unhydrated pull workspace");
        let remote_graph_path = temp.path().join("remote.db");
        let remote_graph =
            get_connection(&remote_graph_path).expect("should create remote graph database");
        Collection::create(&remote_graph, "base").expect("should create remote base state");
        let contents = b"branch-only asset\n";
        let asset = test_asset(contents, "feature.gfa", 1);
        AssetRef::create(&remote_graph, &asset).expect("should insert remote asset");
        let current_hash =
            commit_all(&remote_graph, "add branch asset").expect("should commit remote asset");
        drop(remote_graph);

        let workspace = Workspace::new(temp.path().join("local"));
        workspace.ensure_gen_dir();
        let local_graph = get_raw_connection(
            workspace
                .graph_db_path()
                .expect("should resolve local graph database path"),
        )
        .expect("should open local graph database");
        let graph_url = format!("file://{}", remote_graph_path.display());
        // This does a direct clone of the graph db and avoids transfering assets
        clone_remote(&local_graph, &graph_url).expect("should clone remote graph state");
        drop(local_graph);

        let (asset_url, asset_server) = serve_asset(contents);
        let (remote_url, api_server) = serve_pull_api(&graph_url, asset.id, &asset_url, 1, false);
        let config = get_config_connection(Some(
            workspace
                .gen_db_path()
                .expect("should resolve local config database path"),
        ))
        .expect("should open local config database");
        let remote =
            Remote::create(&config, "origin", &remote_url).expect("should configure origin");
        Defaults::set_default_remote(&config, Some(&remote.name))
            .expect("should set default remote");

        execute_pull(&workspace, None, None).expect("pull should hydrate the missing asset");

        let requests = api_server.join().expect("pull API server should finish");
        let asset_request = requests
            .iter()
            .find(|request| request.contains("/asset-transfers "))
            .expect("should request asset transfers");
        assert!(asset_request.contains("\"from_commit\":null"));
        assert!(asset_request.contains(&format!("\"to_commit\":\"{current_hash}\"")));
        assert_eq!(
            fs::read(temp.path().join("local/feature.gfa"))
                .expect("should read hydrated branch asset"),
            contents
        );
        let completed_operations = config
            .query_row(
                "SELECT COUNT(*) FROM remote_operations \
                 WHERE operation = 'pull' AND completed_at IS NOT NULL",
                [],
                |row| row.get::<_, i64>(0),
            )
            .expect("should count completed pull operations");
        assert_eq!(completed_operations, 1);
        asset_server.join().expect("asset server should finish");
    }

    #[test]
    fn test_execute_pull_resumes_an_incomplete_asset_operation() {
        let temp = tempdir().expect("should create pull retry workspace");
        let remote_graph_path = temp.path().join("remote.db");
        let remote_graph =
            get_connection(&remote_graph_path).expect("should create remote graph database");
        Collection::create(&remote_graph, "base").expect("should create remote base state");
        let previous_hash =
            commit_all(&remote_graph, "base").expect("should commit remote base state");
        drop(remote_graph);

        let workspace = Workspace::new(temp.path().join("local"));
        workspace.ensure_gen_dir();
        let local_graph = get_raw_connection(
            workspace
                .graph_db_path()
                .expect("should resolve local graph database path"),
        )
        .expect("should open local graph database");
        let graph_url = format!("file://{}", remote_graph_path.display());
        clone_remote(&local_graph, &graph_url).expect("should clone remote base state");
        drop(local_graph);

        let remote_graph =
            get_connection(&remote_graph_path).expect("should reopen remote graph database");
        let contents = b"retry asset\n";
        let asset = test_asset(contents, "retry.gfa", 1);
        AssetRef::create(&remote_graph, &asset).expect("should insert remote asset");
        let current_hash =
            commit_all(&remote_graph, "add retry asset").expect("should commit remote asset");
        drop(remote_graph);

        let (asset_url, asset_server) = serve_asset(contents);
        let (remote_url, api_server) = serve_pull_api(&graph_url, asset.id, &asset_url, 2, true);
        let config = get_config_connection(Some(
            workspace
                .gen_db_path()
                .expect("should resolve local config database path"),
        ))
        .expect("should open local config database");
        let remote =
            Remote::create(&config, "origin", &remote_url).expect("should configure origin");
        Defaults::set_default_remote(&config, Some(&remote.name))
            .expect("should set default remote");
        let mut baseline = RemoteOperationRecord::begin_or_resume(
            &config,
            &remote.name,
            "main",
            StoredRemoteOperationKind::Clone,
            None,
        )
        .expect("should begin baseline clone operation");
        baseline
            .set_destination(&config, &previous_hash)
            .expect("should record baseline clone destination");
        baseline
            .advance_assets_transfer_checkpoint(&config, &previous_hash)
            .expect("should record baseline asset checkpoint");
        baseline
            .complete(&config)
            .expect("should complete baseline clone operation");

        execute_pull(&workspace, None, None)
            .expect_err("first pull should fail during its asset phase");
        assert!(!temp.path().join("local/retry.gfa").exists());
        let pending_commits: (DoltHashId, DoltHashId) = config
            .query_row(
                "SELECT from_commit, assets_transfer_checkpoint FROM remote_operations \
                 WHERE completed_at IS NULL AND failed_at IS NULL",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .expect("should retain the incomplete operation bounds");
        assert_eq!(pending_commits, (previous_hash, previous_hash));
        execute_pull(&workspace, None, None).expect("pull retry should succeed");

        let requests = api_server.join().expect("pull API server should finish");
        let asset_requests = requests
            .iter()
            .filter(|request| request.contains("/asset-transfers "))
            .collect::<Vec<_>>();
        let from_commit_json = format!("\"from_commit\":\"{previous_hash}\"");
        let to_commit_json = format!("\"to_commit\":\"{current_hash}\"");
        assert_eq!(asset_requests.len(), 2);
        for request in asset_requests {
            assert!(request.contains(&from_commit_json));
            assert!(request.contains(&to_commit_json));
        }
        assert_eq!(
            fs::read(temp.path().join("local/retry.gfa")).expect("should read retried asset"),
            contents
        );
        let pending_operations = config
            .query_row(
                "SELECT COUNT(*) FROM remote_operations \
                 WHERE completed_at IS NULL AND failed_at IS NULL",
                [],
                |row| row.get::<_, i64>(0),
            )
            .expect("should count pending remote operations");
        assert_eq!(pending_operations, 0);
        asset_server.join().expect("asset server should finish");
    }

    #[test]
    fn test_execute_pull_records_asset_checkpoint_and_resumes_remaining_transfer() {
        let temp = tempdir().expect("should create checkpoint pull workspace");
        let remote_graph_path = temp.path().join("remote.db");
        let remote_graph =
            get_connection(&remote_graph_path).expect("should create remote graph database");
        Collection::create(&remote_graph, "base").expect("should create remote base state");
        let baseline_hash =
            commit_all(&remote_graph, "base").expect("should commit remote base state");
        drop(remote_graph);

        let workspace = Workspace::new(temp.path().join("local"));
        workspace.ensure_gen_dir();
        let local_graph = get_raw_connection(
            workspace
                .graph_db_path()
                .expect("should resolve local graph database path"),
        )
        .expect("should open local graph database");
        let graph_url = format!("file://{}", remote_graph_path.display());
        clone_remote(&local_graph, &graph_url).expect("should clone remote base state");
        drop(local_graph);

        let remote_graph =
            get_connection(&remote_graph_path).expect("should reopen remote graph database");
        let first_contents = b"first checkpoint asset\n";
        let first_asset = test_asset(first_contents, "first.gfa", 1);
        AssetRef::create(&remote_graph, &first_asset).expect("should insert first remote asset");
        let first_asset_commit =
            commit_all(&remote_graph, "add first asset").expect("should commit first remote asset");
        let second_contents = b"second checkpoint asset\n";
        let second_asset = test_asset(second_contents, "second.gfa", 2);
        AssetRef::create(&remote_graph, &second_asset).expect("should insert second remote asset");
        let destination_hash = commit_all(&remote_graph, "add second asset")
            .expect("should commit second remote asset");
        drop(remote_graph);

        let (first_asset_url, first_asset_server) = serve_asset(first_contents);
        let (second_asset_url, second_asset_server) = serve_asset(second_contents);
        let (remote_url, api_server) = serve_checkpoint_pull_api(
            &graph_url,
            (first_asset.id, &first_asset_url),
            (second_asset.id, &second_asset_url),
        );
        let config = get_config_connection(Some(
            workspace
                .gen_db_path()
                .expect("should resolve local config database path"),
        ))
        .expect("should open local config database");
        let remote =
            Remote::create(&config, "origin", &remote_url).expect("should configure origin");
        Defaults::set_default_remote(&config, Some(&remote.name))
            .expect("should set default remote");
        let mut baseline = RemoteOperationRecord::begin_or_resume(
            &config,
            &remote.name,
            "main",
            StoredRemoteOperationKind::Clone,
            None,
        )
        .expect("should begin baseline clone operation");
        baseline
            .set_destination(&config, &baseline_hash)
            .expect("should record baseline clone destination");
        baseline
            .advance_assets_transfer_checkpoint(&config, &baseline_hash)
            .expect("should record baseline asset checkpoint");
        baseline
            .complete(&config)
            .expect("should complete baseline clone operation");

        execute_pull(&workspace, None, None)
            .expect_err("first pull should fail in the second commit batch");
        assert_eq!(
            fs::read(temp.path().join("local/first.gfa"))
                .expect("should retain the completed first commit asset"),
            first_contents
        );
        assert!(!temp.path().join("local/second.gfa").exists());
        let (assets_transfer_checkpoint, to_commit): (DoltHashId, DoltHashId) = config
            .query_row(
                "SELECT assets_transfer_checkpoint, to_commit FROM remote_operations \
                 WHERE completed_at IS NULL AND failed_at IS NULL",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .expect("should retain commit-level asset progress");
        assert_eq!(
            assets_transfer_checkpoint, first_asset_commit,
            "checkpoint should advance through the last fully transferred commit"
        );
        assert_eq!(
            to_commit, destination_hash,
            "interrupted operation should retain its graph destination"
        );
        first_asset_server
            .join()
            .expect("first asset transfer should finish before the interruption");

        execute_pull(&workspace, None, None).expect("pull retry should resume after checkpoint");

        let requests = api_server.join().expect("pull API server should finish");
        let asset_requests = requests
            .iter()
            .filter(|request| request.contains("/asset-transfers "))
            .collect::<Vec<_>>();
        assert_eq!(asset_requests.len(), 2);
        assert!(asset_requests[0].contains(&format!("\"from_commit\":\"{baseline_hash}\"")));
        assert!(asset_requests[1].contains(&format!("\"from_commit\":\"{first_asset_commit}\"")));
        for request in asset_requests {
            assert!(request.contains(&format!("\"to_commit\":\"{destination_hash}\"")));
        }
        assert_eq!(
            fs::read(temp.path().join("local/second.gfa"))
                .expect("should materialize the resumed second commit asset"),
            second_contents
        );
        let completed_checkpoint: DoltHashId = config
            .query_row(
                "SELECT assets_transfer_checkpoint FROM remote_operations \
                 WHERE completed_at IS NOT NULL ORDER BY id DESC LIMIT 1",
                [],
                |row| row.get(0),
            )
            .expect("should store the destination as the completed checkpoint");
        assert_eq!(completed_checkpoint, destination_hash);
        second_asset_server
            .join()
            .expect("second asset server should finish");
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
    fn test_remote_resolution_uses_the_only_configured_remote() {
        let temp = tempdir().expect("should create remote resolution directory");
        let workspace = Workspace::new(temp.path());
        workspace.ensure_gen_dir();
        let config = get_config_connection(Some(workspace.gen_db_path().unwrap()))
            .expect("should open config database");
        Remote::create(&config, "sole", "file:///tmp/sole").expect("should create sole remote");

        assert_eq!(
            resolve_remote(&config, None, "new-branch")
                .expect("should resolve the only configured remote")
                .name,
            "sole"
        );
    }

    #[test]
    fn test_authorization_failure_retries_with_a_fresh_capability() {
        let _environment_lock = ENVIRONMENT_LOCK
            .lock()
            .expect("should lock process environment");
        let _api_key = EnvironmentGuard::set("GENHUB_API_KEY", "push-test-key");
        let listener = TcpListener::bind("127.0.0.1:0").expect("should bind capability server");
        let address = listener
            .local_addr()
            .expect("should read capability server address");
        let server = thread::spawn(move || {
            let mut requests = Vec::new();
            for attempt in 0..2 {
                let (mut stream, _) = listener.accept().expect("should accept capability request");
                let mut request = [0_u8; 8192];
                let read = stream
                    .read(&mut request)
                    .expect("should read capability request");
                requests.push(String::from_utf8_lossy(&request[..read]).into_owned());
                let body = format!(
                    "{{\"remote_url\":\"http://127.0.0.1:1/transfer-{attempt}\",\
                     \"expires_at\":\"2030-01-01T00:00:00Z\",\
                     \"default_branch\":\"main\",\
                     \"transfer_id\":\"{}\"}}",
                    if attempt == 0 {
                        TEST_TRANSFER_ID
                    } else {
                        RETRIED_TRANSFER_ID
                    }
                );
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                    body.len()
                )
                .expect("should write capability response");
            }
            requests
        });

        let connection = Connection::open_in_memory().expect("should open graph database");
        let graph = GraphConnection(connection);
        let remote = Remote {
            name: "origin".to_string(),
            url: format!("http://{address}/api/repos/alice/example"),
        };
        let mut attempts = 0;
        let idempotency_token = Uuid::now_v7();
        let transfer_lease = run_graph_transfer(
            &graph,
            &remote,
            RemoteOperation::Push,
            "main",
            false,
            Some(&idempotency_token),
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
        let requests = server.join().expect("capability server should finish");

        assert_eq!(attempts, 2);
        assert_eq!(requests.len(), 2);
        assert_eq!(
            request_idempotency_token(&requests[0]),
            Some(idempotency_token)
        );
        assert_eq!(
            request_idempotency_token(&requests[1]),
            Some(idempotency_token)
        );
        assert_eq!(
            transfer_lease,
            Some(PushTransferLease {
                transfer_id: RETRIED_TRANSFER_ID,
                expires_at: 1_893_456_000,
            })
        );
        let remotes = remote_rows(&graph).expect("should read restored canonical URL");
        assert!(
            remotes
                .iter()
                .any(|graph_remote| graph_remote.name == "origin" && graph_remote.url == remote.url)
        );
    }

    #[test]
    fn test_push_keeps_one_idempotency_token_across_native_auth_refreshes() {
        let _environment_lock = ENVIRONMENT_LOCK
            .lock()
            .expect("should lock process environment");
        let _api_key = EnvironmentGuard::set("GENHUB_API_KEY", "push-test-key");
        let native_remotes = (0..4)
            .map(|_| serve_unauthorized_dolt_remote())
            .collect::<Vec<_>>();
        let remote_urls = native_remotes
            .iter()
            .map(|(url, _)| url.clone())
            .collect::<Vec<_>>();
        let capability_listener =
            TcpListener::bind("127.0.0.1:0").expect("should bind capability server");
        let capability_address = capability_listener
            .local_addr()
            .expect("should read capability server address");
        let capability_server = thread::spawn(move || {
            let mut requests = Vec::new();
            for (attempt, remote_url) in remote_urls.into_iter().enumerate() {
                let (mut stream, _) = capability_listener
                    .accept()
                    .expect("should accept capability request");
                let request = read_native_protocol_request(&mut stream);
                requests.push(request);
                let body = json!({
                    "remote_url": remote_url,
                    "expires_at": "2030-01-01T00:00:00Z",
                    "default_branch": "main",
                    "transfer_id": Uuid::from_u128(attempt as u128 + 1),
                })
                .to_string();
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )
                .expect("should write capability response");
            }
            requests
        });

        let temp = tempdir().expect("should create native push fixture");
        let graph = get_connection(temp.path().join("source.db"))
            .expect("should create source graph database");
        Collection::create(&graph, "native-push-fixture")
            .expect("should create source graph state");
        commit_all(&graph, "native push fixture").expect("should commit source graph state");
        let remote = Remote {
            name: "origin".to_string(),
            url: format!("http://{capability_address}/api/repos/alice/example"),
        };
        let idempotency_tokens = [Uuid::now_v7(), Uuid::now_v7()];

        for idempotency_token in idempotency_tokens {
            let result = run_graph_transfer(
                &graph,
                &remote,
                RemoteOperation::Push,
                "main",
                false,
                Some(&idempotency_token),
                || {
                    push_graph_branch(
                        &graph,
                        &remote.name,
                        "main",
                        false,
                        Some(&idempotency_token),
                    )
                },
            );
            assert!(result.is_err(), "the mock remote should reject the push");
        }

        let capability_requests = capability_server
            .join()
            .expect("capability server should finish");
        let native_requests = native_remotes
            .into_iter()
            .map(|(_, server)| server.join().expect("native remote should finish"))
            .collect::<Vec<_>>();
        let expected_tokens = [
            Some(idempotency_tokens[0]),
            Some(idempotency_tokens[0]),
            Some(idempotency_tokens[1]),
            Some(idempotency_tokens[1]),
        ];

        assert_eq!(capability_requests.len(), 4);
        assert_eq!(native_requests.len(), 4);
        for ((capability_request, native_request), expected_token) in capability_requests
            .iter()
            .zip(&native_requests)
            .zip(expected_tokens)
        {
            assert_eq!(
                request_idempotency_token(capability_request),
                expected_token,
                "capability refreshes within a push should reuse its token"
            );
            assert!(
                native_request.starts_with("GET /dolt/remote.db/refs "),
                "push should negotiate refs through the native Dolt HTTP protocol: {native_request}"
            );
            assert_eq!(
                request_idempotency_token(native_request),
                expected_token,
                "the native Dolt request should carry the same push token"
            );
        }
        assert_ne!(idempotency_tokens[0], idempotency_tokens[1]);
    }

    #[test]
    fn test_native_clone_and_pull_requests_do_not_send_idempotency_tokens() {
        let temp = tempdir().expect("should create non-push remote fixture");
        let (clone_url, clone_server) = serve_unauthorized_dolt_remote();
        let clone_graph = GraphConnection(
            Connection::open_in_memory().expect("should open clone graph database"),
        );
        assert!(
            clone_remote(&clone_graph, &clone_url).is_err(),
            "mock remote should reject clone"
        );
        let clone_request = clone_server
            .join()
            .expect("clone remote server should finish");
        assert_eq!(request_idempotency_token(&clone_request), None);

        let (pull_url, pull_server) = serve_unauthorized_dolt_remote();
        let pull_graph =
            get_connection(temp.path().join("pull.db")).expect("should create pull graph database");
        Collection::create(&pull_graph, "pull-fixture").expect("should create pull graph state");
        commit_all(&pull_graph, "pull fixture").expect("should commit pull graph state");
        add_remote(&pull_graph, "origin", &pull_url).expect("should configure pull remote");
        assert!(
            pull(&pull_graph, "origin", "main").is_err(),
            "mock remote should reject pull"
        );
        let pull_request = pull_server
            .join()
            .expect("pull remote server should finish");
        assert_eq!(request_idempotency_token(&pull_request), None);
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
                         \"default_branch\":\"main\",\
                         \"transfer_id\":\"{TEST_TRANSFER_ID}\"}}"
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
        assert!(requests[2].contains(&format!("\"transfer_id\":\"{TEST_TRANSFER_ID}\"")));
        assert!(requests[2].contains("\"branch\":\"main\""));
        assert!(requests[2].contains("\"assets\":[]"));
        assert!(requests[3].contains("\"operation\":\"pull\""));
        let push_token = request_idempotency_token(&requests[0])
            .expect("push capability should include an idempotency token");
        assert_eq!(request_idempotency_token(&requests[1]), Some(push_token));
        assert_eq!(request_idempotency_token(&requests[2]), Some(push_token));
        assert_eq!(request_idempotency_token(&requests[3]), None);
        let graph =
            get_connection(workspace.graph_db_path().unwrap()).expect("should reopen graph");
        let local_hash = hash_of(&graph, "main").expect("should query local branch");
        let tracking_hash = hash_of(&graph, "origin/main").expect("should query tracking branch");
        assert_eq!(tracking_hash, local_hash);
    }

    #[test]
    fn test_push_retry_reuses_persisted_transfer_lease() {
        let _environment_lock = ENVIRONMENT_LOCK
            .lock()
            .expect("should lock process environment");
        let _api_key = EnvironmentGuard::set("GENHUB_API_KEY", "push-test-key");
        let listener = TcpListener::bind("127.0.0.1:0").expect("should bind capability server");
        let address = listener
            .local_addr()
            .expect("should read capability server address");
        let temp = tempdir().expect("should create push retry directory");
        let remote_graph = temp.path().join("remote.db");
        let transfer_url = format!("file://{}", remote_graph.display());
        let server = thread::spawn(move || {
            let mut requests = Vec::new();
            for request_index in 0..6 {
                let (mut stream, _) = listener.accept().expect("should accept GenHub request");
                let mut request = [0_u8; 8192];
                let read = stream
                    .read(&mut request)
                    .expect("should read GenHub request");
                requests.push(String::from_utf8_lossy(&request[..read]).into_owned());
                match request_index {
                    0 | 5 => {
                        let body = format!(
                            "{{\"remote_url\":\"{transfer_url}\",\
                             \"expires_at\":\"2030-01-01T00:00:00Z\",\
                             \"default_branch\":\"main\",\
                             \"transfer_id\":\"{TEST_TRANSFER_ID}\"}}"
                        );
                        write!(
                            stream,
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                            body.len()
                        )
                        .expect("should write capability response");
                    }
                    1 | 3 => {
                        let body = "{\"assets\":[]}";
                        write!(
                            stream,
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                            body.len()
                        )
                        .expect("should write asset response");
                    }
                    2 => {
                        let body = "asset verification unavailable";
                        write!(
                            stream,
                            "HTTP/1.1 500 Internal Server Error\r\nContent-Type: text/plain\r\nContent-Length: {}\r\n\r\n{body}",
                            body.len()
                        )
                        .expect("should write failed completion response");
                    }
                    4 => {
                        stream
                            .write_all(b"HTTP/1.1 204 No Content\r\nContent-Length: 0\r\n\r\n")
                            .expect("should write completion response");
                    }
                    _ => unreachable!("request index should be covered"),
                }
            }
            requests
        });

        let workspace = Workspace::new(temp.path().join("local"));
        workspace.ensure_gen_dir();
        let config = get_config_connection(Some(workspace.gen_db_path().unwrap()))
            .expect("should open push config");
        let graph = get_connection(workspace.graph_db_path().unwrap()).expect("should open graph");
        Collection::create(&graph, "push-retry-fixture").expect("should create push retry fixture");
        let destination_hash =
            commit_all(&graph, "push retry fixture").expect("should commit push retry fixture");
        Remote::create(
            &config,
            "origin",
            &format!("http://{address}/api/repos/alice/example"),
        )
        .expect("should configure origin");
        Defaults::set_default_remote(&config, Some("origin")).expect("should set default remote");
        drop(graph);

        execute_push(&workspace, None, None, false)
            .expect_err("first push should retain its lease after completion fails");
        let pending_transfer: (DoltHashId, Uuid) = config
            .query_row(
                "SELECT to_commit, transfer_id FROM remote_operations \
                 WHERE operation = 'push' AND completed_at IS NULL AND failed_at IS NULL",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .expect("should retain pending push lease");
        assert_eq!(pending_transfer, (destination_hash, TEST_TRANSFER_ID));

        execute_push(&workspace, None, None, false).expect("push retry should complete");
        let requests = server.join().expect("GenHub server should finish");

        assert_eq!(requests.len(), 6);
        assert!(requests[0].contains("\"operation\":\"push\""));
        assert!(requests[1].starts_with("POST /api/repos/alice/example/asset-transfers "));
        assert!(requests[2].starts_with("POST /api/repos/alice/example/asset-transfers/complete "));
        assert!(requests[3].starts_with("POST /api/repos/alice/example/asset-transfers "));
        assert!(requests[4].starts_with("POST /api/repos/alice/example/asset-transfers/complete "));
        for request in [&requests[2], &requests[4]] {
            assert!(request.contains(&format!("\"transfer_id\":\"{TEST_TRANSFER_ID}\"")));
            assert!(request.contains("\"branch\":\"main\""));
            assert!(request.contains("\"assets\":[]"));
        }
        assert!(requests[5].contains("\"operation\":\"pull\""));
        let first_push_token = request_idempotency_token(&requests[0])
            .expect("initial push capability should include an idempotency token");
        assert_eq!(
            request_idempotency_token(&requests[1]),
            Some(first_push_token)
        );
        assert_eq!(
            request_idempotency_token(&requests[2]),
            Some(first_push_token)
        );
        let resumed_push_token = request_idempotency_token(&requests[3])
            .expect("resumed push should create a fresh idempotency token");
        assert_ne!(resumed_push_token, first_push_token);
        assert_eq!(
            request_idempotency_token(&requests[4]),
            Some(resumed_push_token)
        );
        assert_eq!(request_idempotency_token(&requests[5]), None);
        let pending_operations = config
            .query_row(
                "SELECT COUNT(*) FROM remote_operations \
                 WHERE completed_at IS NULL AND failed_at IS NULL",
                [],
                |row| row.get::<_, i64>(0),
            )
            .expect("should count pending operations");
        assert_eq!(pending_operations, 0);
    }

    #[test]
    fn test_push_retry_pushes_advanced_head_with_new_transfer_lease() {
        // This tests that if we have a resumed push, but have commited work since the last failed push, the to_commit recorded
        // as the end state of the push is advanced to the current head commit.
        let _environment_lock = ENVIRONMENT_LOCK
            .lock()
            .expect("should lock process environment");
        let _api_key = EnvironmentGuard::set("GENHUB_API_KEY", "push-test-key");
        let listener = TcpListener::bind("127.0.0.1:0").expect("should bind capability server");
        let address = listener
            .local_addr()
            .expect("should read capability server address");
        let temp = tempdir().expect("should create advanced push retry directory");
        let remote_graph = temp.path().join("remote.db");
        let transfer_url = format!("file://{}", remote_graph.display());
        let server = thread::spawn(move || {
            let mut requests = Vec::new();
            for request_index in 0..7 {
                let (mut stream, _) = listener.accept().expect("should accept GenHub request");
                let mut request = [0_u8; 8192];
                let read = stream
                    .read(&mut request)
                    .expect("should read GenHub request");
                requests.push(String::from_utf8_lossy(&request[..read]).into_owned());
                match request_index {
                    0 | 3 | 6 => {
                        let transfer_id = if request_index == 0 {
                            TEST_TRANSFER_ID
                        } else {
                            RETRIED_TRANSFER_ID
                        };
                        let body = format!(
                            "{{\"remote_url\":\"{transfer_url}\",\
                             \"expires_at\":\"2030-01-01T00:00:00Z\",\
                             \"default_branch\":\"main\",\
                             \"transfer_id\":\"{transfer_id}\"}}"
                        );
                        write!(
                            stream,
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                            body.len()
                        )
                        .expect("should write capability response");
                    }
                    1 | 4 => {
                        let body = "{\"assets\":[]}";
                        write!(
                            stream,
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                            body.len()
                        )
                        .expect("should write asset response");
                    }
                    2 => {
                        let body = "asset verification unavailable";
                        write!(
                            stream,
                            "HTTP/1.1 500 Internal Server Error\r\nContent-Type: text/plain\r\nContent-Length: {}\r\n\r\n{body}",
                            body.len()
                        )
                        .expect("should write failed completion response");
                    }
                    5 => {
                        stream
                            .write_all(b"HTTP/1.1 204 No Content\r\nContent-Length: 0\r\n\r\n")
                            .expect("should write completion response");
                    }
                    _ => unreachable!("request index should be covered"),
                }
            }
            requests
        });

        let workspace = Workspace::new(temp.path().join("local"));
        workspace.ensure_gen_dir();
        let config = get_config_connection(Some(workspace.gen_db_path().unwrap()))
            .expect("should open push config");
        let graph = get_connection(workspace.graph_db_path().unwrap()).expect("should open graph");
        Collection::create(&graph, "push-retry-fixture").expect("should create push fixture");
        let original_destination =
            commit_all(&graph, "push retry fixture").expect("should commit push retry fixture");
        Remote::create(
            &config,
            "origin",
            &format!("http://{address}/api/repos/alice/example"),
        )
        .expect("should configure origin");
        Defaults::set_default_remote(&config, Some("origin")).expect("should set default remote");
        drop(graph);

        execute_push(&workspace, None, None, false)
            .expect_err("first push should retain its lease after completion fails");
        let pending_transfer: (DoltHashId, Uuid) = config
            .query_row(
                "SELECT to_commit, transfer_id FROM remote_operations \
                 WHERE operation = 'push' AND completed_at IS NULL AND failed_at IS NULL",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .expect("should retain pending push lease");
        assert_eq!(pending_transfer, (original_destination, TEST_TRANSFER_ID));

        let graph =
            get_connection(workspace.graph_db_path().unwrap()).expect("should reopen graph");
        Collection::create(&graph, "advanced-head").expect("should advance local graph");
        let advanced_destination =
            commit_all(&graph, "advance push retry").expect("should commit advanced local head");
        drop(graph);

        execute_push(&workspace, None, None, false).expect("advanced push retry should complete");
        let requests = server.join().expect("GenHub server should finish");

        assert_eq!(requests.len(), 7);
        assert!(requests[0].contains("\"operation\":\"push\""));
        assert!(requests[2].contains(&format!("\"transfer_id\":\"{TEST_TRANSFER_ID}\"")));
        assert!(requests[3].contains("\"operation\":\"push\""));
        assert!(requests[4].starts_with("POST /api/repos/alice/example/asset-transfers "));
        assert!(requests[5].starts_with("POST /api/repos/alice/example/asset-transfers/complete "));
        assert!(requests[5].contains(&format!("\"transfer_id\":\"{RETRIED_TRANSFER_ID}\"")));
        assert!(requests[6].contains("\"operation\":\"pull\""));
        let completed_transfer: (DoltHashId, DoltHashId, Uuid) = config
            .query_row(
                "SELECT to_commit, assets_transfer_checkpoint, transfer_id \
                 FROM remote_operations WHERE operation = 'push' AND completed_at IS NOT NULL",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .expect("should complete advanced push operation");
        assert_eq!(
            completed_transfer,
            (
                advanced_destination,
                advanced_destination,
                RETRIED_TRANSFER_ID
            )
        );
        let graph =
            get_connection(workspace.graph_db_path().unwrap()).expect("should reopen graph");
        let tracking_hash = hash_of(&graph, "origin/main").expect("should query tracking branch");
        assert_eq!(tracking_hash, advanced_destination);
    }

    #[test]
    fn test_push_retry_refreshes_expired_transfer_lease() {
        let _environment_lock = ENVIRONMENT_LOCK
            .lock()
            .expect("should lock process environment");
        let _api_key = EnvironmentGuard::set("GENHUB_API_KEY", "push-test-key");
        let listener =
            TcpListener::bind("127.0.0.1:0").expect("should bind expired lease retry server");
        let address = listener
            .local_addr()
            .expect("should read expired lease retry server address");
        let temp = tempdir().expect("should create expired lease retry directory");
        let remote_graph = temp.path().join("remote.db");
        let transfer_url = format!("file://{}", remote_graph.display());
        let server = thread::spawn(move || {
            let mut requests = Vec::new();
            for request_index in 0..7 {
                let (mut stream, _) = listener.accept().expect("should accept GenHub request");
                let mut request = [0_u8; 8192];
                let read = stream
                    .read(&mut request)
                    .expect("should read GenHub request");
                requests.push(String::from_utf8_lossy(&request[..read]).into_owned());
                match request_index {
                    0 | 3 | 6 => {
                        let transfer_id = if request_index == 0 {
                            TEST_TRANSFER_ID
                        } else {
                            RETRIED_TRANSFER_ID
                        };
                        let body = format!(
                            "{{\"remote_url\":\"{transfer_url}\",\
                             \"expires_at\":\"2030-01-01T00:00:00Z\",\
                             \"default_branch\":\"main\",\
                             \"transfer_id\":\"{transfer_id}\"}}"
                        );
                        write!(
                            stream,
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                            body.len()
                        )
                        .expect("should write capability response");
                    }
                    1 | 4 => {
                        let body = "{\"assets\":[]}";
                        write!(
                            stream,
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                            body.len()
                        )
                        .expect("should write asset response");
                    }
                    2 => {
                        let body = "asset verification unavailable";
                        write!(
                            stream,
                            "HTTP/1.1 500 Internal Server Error\r\nContent-Type: text/plain\r\nContent-Length: {}\r\n\r\n{body}",
                            body.len()
                        )
                        .expect("should write failed completion response");
                    }
                    5 => {
                        stream
                            .write_all(b"HTTP/1.1 204 No Content\r\nContent-Length: 0\r\n\r\n")
                            .expect("should write completion response");
                    }
                    _ => unreachable!("request index should be covered"),
                }
            }
            requests
        });

        let workspace = Workspace::new(temp.path().join("local"));
        workspace.ensure_gen_dir();
        let config = get_config_connection(Some(workspace.gen_db_path().unwrap()))
            .expect("should open push config");
        let graph = get_connection(workspace.graph_db_path().unwrap()).expect("should open graph");
        Collection::create(&graph, "expired-lease-retry-fixture")
            .expect("should create expired lease retry fixture");
        let destination_hash = commit_all(&graph, "expired lease retry fixture")
            .expect("should commit expired lease retry fixture");
        Remote::create(
            &config,
            "origin",
            &format!("http://{address}/api/repos/alice/example"),
        )
        .expect("should configure origin");
        Defaults::set_default_remote(&config, Some("origin")).expect("should set default remote");
        drop(graph);

        execute_push(&workspace, None, None, false)
            .expect_err("first push should retain its lease after completion fails");
        config
            .execute(
                "UPDATE remote_operations SET transfer_expires_at = 0 \
                 WHERE operation = 'push' AND completed_at IS NULL AND failed_at IS NULL",
                [],
            )
            .expect("should expire the persisted transfer lease");

        execute_push(&workspace, None, None, false)
            .expect("expired lease retry should obtain a new transfer");
        let requests = server.join().expect("GenHub server should finish");

        assert_eq!(requests.len(), 7);
        assert!(requests[0].contains("\"operation\":\"push\""));
        assert!(requests[2].contains(&format!("\"transfer_id\":\"{TEST_TRANSFER_ID}\"")));
        assert!(requests[3].contains("\"operation\":\"push\""));
        assert!(requests[5].contains(&format!("\"transfer_id\":\"{RETRIED_TRANSFER_ID}\"")));
        assert!(requests[6].contains("\"operation\":\"pull\""));
        let completed_transfer: (DoltHashId, Uuid, i64) = config
            .query_row(
                "SELECT to_commit, transfer_id, transfer_expires_at \
                 FROM remote_operations WHERE operation = 'push' AND completed_at IS NOT NULL",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .expect("should complete refreshed transfer lease");
        assert_eq!(completed_transfer.0, destination_hash);
        assert_eq!(completed_transfer.1, RETRIED_TRANSFER_ID);
        assert!(completed_transfer.2 > Utc::now().timestamp());
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
                             \"default_branch\":\"main\",\
                             \"transfer_id\":\"{TEST_TRANSFER_ID}\"}}"
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
        assert!(requests[2].contains(&format!("\"transfer_id\":\"{TEST_TRANSFER_ID}\"")));
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
            None,
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
            None,
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
