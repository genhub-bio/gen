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
//! directly between the workspaces. For a GenHub remote, clone, pull, and fetch use a scoped,
//! short-lived GCS capability through BlockCacheVFS. Push opens a direct GCS
//! database URI behind an in-process loopback `RemoteServer`, stages the Dolt push, and asks
//! GenHub to publish the accepted manifest. An idempotency token persisted with the push operation
//! lets an interrupted push resume its staged session. Failure to restore a temporary clone or
//! pull URL is reported as a warning because the graph transfer may already have succeeded and the
//! canonical URL will be restored on the next attempt.
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
//! the client idempotency token identifies a durable direct-GCS staging session. Gen persists that
//! token and the transfer lease so an interrupted graph push can reopen the same session and resume
//! from its accepted manifest checkpoints. An expired lease is renewed against the same session
//! before graph or asset requests continue. Changing the push destination starts a distinct
//! session. The lease restricts asset uploads to a single client and is released after assets are
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
    fmt,
    fs::{self, OpenOptions},
    io::{self, BufReader, Read as _, Write as _},
    path::{Path, PathBuf},
    time::Instant,
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
        active_branch, add_remote, branch_hash, checkout, clone_remote, fetch, hash_of, pull, push,
        push_force, remote_rows, set_remote_url,
    },
    operations::{
        Defaults, Remote, RemoteBranch, RemoteOperationKind as StoredRemoteOperationKind,
        RemoteOperationRecord, calculate_file_checksum, calculate_reader_checksum,
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
use rusqlite::{
    Error as SqlError, OpenFlags, RemoteFailureKind, RemoteServer, RemoteServerOptions,
    blockcachevfs::{CloudConnectionIdentity, CloudConnectionUri},
};
use sha2::{Digest as _, Sha256};
use url::Url;
use uuid::Uuid;

mod capability;
mod direct_gcs;
mod push_lock;

use capability::{
    PushCapabilityRenewal, PushTransferLease, acquire_push_session_renewal, transfer_authorization,
};
use direct_gcs::{
    PushGraphTransferError, PushGraphTransferRequest, is_non_fast_forward_push_error,
    run_push_graph_transfer,
};

use crate::{
    commands::remote::{
        client::{
            AssetTransferCompletionRequest, AssetTransferRequest, AssetUploadReceipt,
            RemoteClientError, RemoteOperation, RepositoryRemote, acquire_asset_transfers,
            complete_asset_transfers,
        },
        login_origin,
        progress::{
            AssetDownloadProgressReporter, AssetUploadProgressReporter,
            GraphUploadProgressReporter, TransferProgress, UploadBodyReader, format_elapsed,
        },
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
    let graph_remotes = match remote_rows(graph) {
        Ok(graph_remotes) => graph_remotes,
        Err(error) => {
            eprintln!(
                "Warning: failed to inspect graph remotes while restoring '{}': {error}",
                remote.name
            );
            return;
        }
    };
    if !graph_remotes
        .iter()
        .any(|graph_remote| graph_remote.name == remote.name)
    {
        return;
    }
    if let Err(error) = set_remote_url(graph, &remote.name, &remote.url) {
        eprintln!(
            "Warning: failed to restore the canonical URL for graph remote '{}': {error}",
            remote.name
        );
    }
}

fn is_authorization_error(error: &SqlError) -> bool {
    error.remote_failure_kind() == Some(RemoteFailureKind::Authorization)
}

struct CapabilityGraphRemote {
    url: String,
    _cloud_server: RemoteServer,
}

#[derive(Default)]
struct CapabilityTransportIdentity {
    cloud: Option<CloudConnectionIdentity>,
}

/// Serves a read CAB through a temporary loopback remote backed by BlockCacheVFS.
///
/// The server owns the read-only cloud connection during clone, fetch, or pull. Its local URL
/// contains no credentials.
fn capability_graph_remote(
    remote_url: &str,
    identity: &mut CapabilityTransportIdentity,
) -> Result<CapabilityGraphRemote, SqlError> {
    let cloud_uri = validate_capability_transport_identity(remote_url, identity)?;
    let read_flags = OpenFlags::SQLITE_OPEN_READ_ONLY
        | OpenFlags::SQLITE_OPEN_URI
        | OpenFlags::SQLITE_OPEN_NO_MUTEX;
    let cloud_server = RemoteServer::start_with_options(
        Path::new(cloud_uri.as_str()),
        &RemoteServerOptions::new().database_open_flags(read_flags),
    )
    .map_err(normalize_cloud_open_authorization_error)?;
    Ok(CapabilityGraphRemote {
        url: cloud_server.database_url(DEFAULT_GRAPH_DB_NAME),
        _cloud_server: cloud_server,
    })
}

/// Raw GCS HTTP authorization statuses become typed failures so callers can request a fresh CAB.
fn normalize_cloud_open_authorization_error(error: SqlError) -> SqlError {
    let is_http_authorization_error = error
        .sqlite_error()
        .is_some_and(|sqlite_error| matches!(sqlite_error.extended_code, 401 | 403));
    if is_http_authorization_error {
        SqlError::SqliteFailure(
            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_IOERR_AUTH),
            Some("cloud graph capability was rejected".to_string()),
        )
    } else {
        error
    }
}

/// A renewed read CAB may rotate credentials but cannot change its cloud object scope.
fn validate_capability_transport_identity(
    remote_url: &str,
    identity: &mut CapabilityTransportIdentity,
) -> Result<CloudConnectionUri, SqlError> {
    let cloud_uri = CloudConnectionUri::parse(remote_url)?;
    let cloud_identity = cloud_uri.identity();
    if identity
        .cloud
        .as_ref()
        .is_some_and(|expected| expected != &cloud_identity)
    {
        return Err(changed_capability_scope_error());
    }
    identity.cloud = Some(cloud_identity);
    Ok(cloud_uri)
}

fn changed_capability_scope_error() -> SqlError {
    SqlError::SqliteFailure(
        rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_MISUSE),
        Some("GenHub changed the graph capability scope during renewal".to_string()),
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

fn persist_refreshed_push_lease(
    operation: &mut RemoteOperationRecord,
    config: &ConfigConnection,
    destination_hash: &DoltHashId,
    renewal: &PushCapabilityRenewal,
) -> Result<(), Box<dyn Error>> {
    let lease = renewal.ensure_fresh()?;
    if operation
        .transfer_id
        .is_some_and(|transfer_id| transfer_id != lease.transfer_id)
    {
        return Err(Box::new(PushGraphTransferError::Protocol(
            "GenHub changed the active push transfer ID".to_string(),
        )));
    }
    if operation.transfer_id != Some(lease.transfer_id)
        || operation.transfer_expires_at != Some(lease.expires_at)
    {
        operation.set_push_destination(
            config,
            destination_hash,
            lease.transfer_id,
            lease.expires_at,
        )?;
    }
    Ok(())
}

fn run_graph_transfer(
    graph: &GraphConnection,
    remote: &Remote,
    operation: RemoteOperation,
    branch: &str,
    force: bool,
    mut transfer: impl FnMut() -> Result<(), SqlError>,
) -> Result<Option<PushTransferLease>, Box<dyn Error>> {
    if remote.url.starts_with("file://") {
        let authorization =
            transfer_authorization(remote, operation, Some(branch), force, None, login_origin)?;
        let remote_url = authorization
            .remote_url
            .as_deref()
            .ok_or("file remote did not resolve to a graph database URL")?;
        ensure_graph_remote(graph, &remote.name, remote_url)?;
        let result = transfer();
        restore_canonical_url(graph, remote);
        result?;
        return Ok(None);
    }

    let mut last_error = None;
    let mut capability_identity = CapabilityTransportIdentity::default();
    for attempt in 0..2 {
        let authorization = match transfer_authorization(
            remote,
            operation,
            Some(branch),
            force,
            None,
            login_origin,
        ) {
            Ok(authorization) => authorization,
            Err(error) => {
                restore_canonical_url(graph, remote);
                return Err(error);
            }
        };
        let Some(remote_url) = authorization.remote_url.as_deref() else {
            restore_canonical_url(graph, remote);
            return Err(format!(
                "GenHub {:?} capability did not include a graph remote URL",
                operation
            )
            .into());
        };
        let transfer_result = match capability_graph_remote(remote_url, &mut capability_identity) {
            Ok(graph_remote) => match ensure_graph_remote(graph, &remote.name, &graph_remote.url) {
                Ok(()) => transfer(),
                Err(error) => Err(error),
            },
            Err(error) => Err(error),
        };
        match transfer_result {
            Ok(()) => {
                restore_canonical_url(graph, remote);
                return Ok(authorization.push_lease);
            }
            Err(error) if attempt == 0 && is_authorization_error(&error) => {
                restore_canonical_url(graph, remote);
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
) -> Result<(), SqlError> {
    if force {
        push_force(graph, remote_name, branch)
    } else {
        push(graph, remote_name, branch)
    }
}

fn push_graph_branch_with_progress(
    graph: &GraphConnection,
    remote_name: &str,
    branch: &str,
    force: bool,
    progress: &GraphUploadProgressReporter,
) -> Result<(), SqlError> {
    let progress_callback = progress.clone();
    let push_result = {
        let _progress_callback = graph.dolt_push_progress_callback(move |event| {
            progress_callback.report_dolt_progress(event);
        })?;
        push_graph_branch(graph, remote_name, branch, force)
    };
    if !matches!(&push_result, Err(error) if is_non_fast_forward_push_error(error)) {
        progress.finish_dolt_progress(push_result.is_ok());
    }
    push_result
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

fn calculate_upload_checksums(
    path: &Path,
    progress: &AssetUploadProgressReporter,
) -> Result<(Sha256Hash, String, String), std::io::Error> {
    let file = fs::File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut sha256 = Sha256::new();
    let mut md5 = Md5::new();
    let mut crc32c = 0;
    let mut bytes_read = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let length = reader.read(&mut buffer)?;
        if length == 0 {
            break;
        }
        bytes_read += length as u64;
        progress.checksum_bytes(bytes_read);
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

fn progress_asset_name(asset: &AssetRef, source_path: &Path) -> String {
    let name = asset
        .logical_path
        .as_deref()
        .or_else(|| source_path.file_name().and_then(|name| name.to_str()))
        .unwrap_or("asset");
    sanitize_progress_label(name)
}

fn sanitize_progress_label(name: &str) -> String {
    name.chars()
        .map(|character| {
            if character.is_control() {
                '?'
            } else {
                character
            }
        })
        .collect()
}

fn upload_asset(
    client: &Client,
    workspace: &Workspace,
    asset: &AssetRef,
    url: &str,
    index: usize,
    total: usize,
    progress: &TransferProgress,
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
    let length = fs::metadata(&source_path)?.len();
    let progress = progress.asset_reporter(
        index,
        total,
        progress_asset_name(asset, &source_path),
        length,
    );
    let result = upload_asset_with_progress(
        client,
        asset,
        url,
        source_path,
        expected_checksum,
        length,
        &progress,
    );
    if result.is_err() {
        progress.failed();
    }
    result
}

fn upload_asset_with_progress(
    client: &Client,
    asset: &AssetRef,
    url: &str,
    source_path: PathBuf,
    expected_checksum: Sha256Hash,
    length: u64,
    progress: &AssetUploadProgressReporter,
) -> Result<AssetUploadReceipt, Box<dyn Error>> {
    progress.checksum_started();
    let (actual_checksum, md5, crc32c) = {
        let _heartbeat = progress.heartbeat("scanning and checksumming the local asset");
        calculate_upload_checksums(&source_path, progress)
    }
    .map_err(|error| {
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
    if file.metadata()?.len() != length {
        return Err(format!(
            "Asset {} at {} changed size while its checksum was being verified",
            asset.id,
            source_path.display()
        )
        .into());
    }
    progress.checksum_verified();
    progress.upload_started();
    let request_body = UploadBodyReader::new(file, progress.clone());
    let response = {
        let _heartbeat =
            progress.heartbeat("sending the asset request and waiting for storage acceptance");
        client
            .put(url)
            .header("content-type", "application/octet-stream")
            // For GCS, content-md5 will be used as a server side integrity verification. It is ignored for
            // composite objects (those > 5GB)
            .header("content-md5", &md5)
            .header("x-goog-if-generation-match", "0")
            .body(Body::sized(request_body, length))
            .send()
            .map_err(|error| error.without_url())?
    };
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
    if response.status() == reqwest::StatusCode::PRECONDITION_FAILED {
        progress.upload_already_present();
    } else {
        progress.upload_accepted();
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

struct AssetTransferRun<'transfer> {
    target: AssetTransferTarget<'transfer>,
    progress: &'transfer TransferProgress,
}

struct AssetDownloadBatch<'progress> {
    progress: &'progress TransferProgress,
    total: usize,
    next_index: usize,
}

impl<'progress> AssetDownloadBatch<'progress> {
    fn from_commits(
        progress: &'progress TransferProgress,
        assets_by_commit: &IndexMap<DoltHashId, Vec<AssetRef>>,
    ) -> Self {
        let total = assets_by_commit.values().map(Vec::len).sum();
        let mut known_bytes = 0_u64;
        let mut unknown_size_assets = 0;
        for asset in assets_by_commit.values().flatten() {
            if let Some(size) = asset.size.and_then(|size| u64::try_from(size).ok()) {
                known_bytes = known_bytes.saturating_add(size);
            } else {
                unknown_size_assets += 1;
            }
        }
        progress.report_asset_download_plan(total, known_bytes, unknown_size_assets);
        Self {
            progress,
            total,
            next_index: 0,
        }
    }

    fn start_asset(&mut self, asset: &AssetRef) -> AssetDownloadProgressReporter {
        self.next_index += 1;
        let name = asset
            .name
            .clone()
            .or_else(|| {
                asset
                    .logical_path
                    .as_deref()
                    .and_then(|path| path.rsplit('/').next())
                    .map(str::to_string)
            })
            .unwrap_or_else(|| asset.id.to_string());
        let name = sanitize_progress_label(&name);
        self.progress.asset_download_reporter(
            self.next_index,
            self.total,
            name,
            asset.size.and_then(|size| u64::try_from(size).ok()),
        )
    }
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
    progress: Option<&AssetDownloadProgressReporter>,
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
        let archived_checksum = if let Some(progress) = progress {
            calculate_file_checksum_with_progress(
                &archived_path,
                progress,
                "checking a previous asset archive",
            )
        } else {
            calculate_file_checksum(&archived_path)
        };
        if !archived_checksum.is_ok_and(|checksum| checksum == archive_checksum) {
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
            let checksum = if let Some(progress) = progress {
                decoded_compressed_checksum_with_progress(
                    destination_path,
                    progress,
                    "checking the existing workspace asset",
                )
            } else {
                decoded_compressed_checksum(destination_path)
            };
            match checksum {
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
        let archived_decoded_checksum = if let Some(progress) = progress {
            decoded_compressed_checksum_with_progress(
                &archived_path,
                progress,
                "checking a previous asset archive",
            )
        } else {
            decoded_compressed_checksum(&archived_path)
        };
        let Ok(archived_decoded_checksum) = archived_decoded_checksum else {
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

fn decoded_compressed_checksum_with_progress(
    path: &Path,
    reporter: &AssetDownloadProgressReporter,
    phase: &str,
) -> Result<Sha256Hash, io::Error> {
    reporter.checksum_started(phase, None);
    let decoder = MultiGzDecoder::new(fs::File::open(path)?);
    let mut reader = CountingReader::new(decoder, |bytes| reporter.checksum_bytes(bytes));
    calculate_reader_checksum(&mut reader)
}

struct CountingReader<R, F> {
    reader: R,
    bytes_read: u64,
    report_bytes: F,
}

impl<R, F> CountingReader<R, F> {
    fn new(reader: R, report_bytes: F) -> Self {
        Self {
            reader,
            bytes_read: 0,
            report_bytes,
        }
    }
}

impl<R, F> io::Read for CountingReader<R, F>
where
    R: io::Read,
    F: FnMut(u64),
{
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let read = self.reader.read(buffer)?;
        if read > 0 {
            self.bytes_read = self.bytes_read.saturating_add(read as u64);
            (self.report_bytes)(self.bytes_read);
        }
        Ok(read)
    }
}

fn calculate_file_checksum_with_progress(
    file_path: &Path,
    reporter: &AssetDownloadProgressReporter,
    phase: &str,
) -> Result<Sha256Hash, io::Error> {
    let expected_bytes = fs::metadata(file_path).ok().map(|metadata| metadata.len());
    reporter.checksum_started(phase, expected_bytes);
    let file = fs::File::open(file_path)?;
    let mut reader = CountingReader::new(BufReader::new(file), |bytes| {
        reporter.checksum_bytes(bytes);
    });
    calculate_reader_checksum(&mut reader)
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

fn content_range_total(response: &Response) -> Option<u64> {
    response
        .headers()
        .get(CONTENT_RANGE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.rsplit_once('/'))
        .and_then(|(_, total)| total.parse::<u64>().ok())
}

/// Keeps streamed response errors free of messages that may contain a signed URL.
#[derive(Debug, thiserror::Error)]
#[error("asset response body read failed ({kind:?})")]
struct AssetResponseReadError {
    kind: io::ErrorKind,
}

/// Streams a response directly to the durable staged file without buffering the asset in memory.
///
/// Partial bytes are deliberately left in place when the stream fails so the next invocation can
/// resume them. `append` is true only after validating the response's `Content-Range`.
fn stream_asset_response_to_staged_file(
    response: &mut Response,
    staged_path: &Path,
    append: bool,
    resumed_bytes: u64,
    progress: Option<&AssetDownloadProgressReporter>,
) -> Result<(), Box<dyn Error>> {
    let mut staged_file = OpenOptions::new()
        .create(true)
        .write(true)
        .append(append)
        .truncate(!append)
        .open(staged_path)?;
    let mut buffer = [0_u8; 64 * 1024];
    let mut bytes_read = 0_u64;
    loop {
        let read = response.read(&mut buffer).map_err(|error| {
            Box::new(AssetResponseReadError { kind: error.kind() }) as Box<dyn Error>
        })?;
        if read == 0 {
            break;
        }
        staged_file.write_all(&buffer[..read])?;
        bytes_read = bytes_read.saturating_add(read as u64);
        if let Some(progress) = progress {
            progress.download_bytes(resumed_bytes.saturating_add(bytes_read));
        }
    }
    staged_file.flush()?;
    staged_file.sync_all()?;
    Ok(())
}

/// Completes or resumes one HTTP asset download in its durable staged file.
///
/// [`download_to_versioned_store_with_progress`] owns the final rename into `.gen/assets`; this function owns
/// only the HTTP range protocol. It appends a valid partial response, overwrites the staged file
/// when a server ignores `Range` and returns a complete response, and retries once from byte zero
/// after an invalid partial response or a resumed checksum mismatch.
fn download_to_staged_path_with_progress(
    client: &Client,
    asset: &AssetRef,
    url: &str,
    staged_path: &Path,
    expected_checksum: &Sha256Hash,
    progress: Option<&AssetDownloadProgressReporter>,
) -> Result<(), Box<dyn Error>> {
    let mut resume_offset = staged_path
        .metadata()
        .map(|metadata| metadata.len())
        .unwrap_or(0);
    if let Some(progress) = progress {
        progress.download_started(resume_offset);
    }
    // Size is only a cheap hint. It avoids hashing a known-partial multi-gigabyte file before
    // resuming, while the checksum remains authoritative whenever the staged size could be final.
    let staged_size_could_be_complete = asset
        .size
        .and_then(|size| u64::try_from(size).ok())
        .is_none_or(|expected_size| expected_size == resume_offset);
    if resume_offset > 0 && staged_size_could_be_complete {
        let checksum = if let Some(progress) = progress {
            calculate_file_checksum_with_progress(
                staged_path,
                progress,
                "verifying a resumed asset download",
            )
        } else {
            calculate_file_checksum(staged_path)
        };
        if checksum.is_ok_and(|checksum| checksum == *expected_checksum) {
            return Ok(());
        }
    }

    loop {
        let mut request = client.get(url);
        if resume_offset > 0 {
            request = request.header(RANGE, format!("bytes={resume_offset}-"));
        }
        let send_result = if let Some(progress) = progress {
            let _heartbeat = progress.heartbeat("waiting for the asset download response");
            request.send().map_err(|error| error.without_url())
        } else {
            request.send().map_err(|error| error.without_url())
        };
        let mut response = send_result?;
        let append = resume_offset > 0
            && response.status() == StatusCode::PARTIAL_CONTENT
            && content_range_starts_at(&response, resume_offset);
        let invalid_partial_response = resume_offset > 0
            && (response.status() == StatusCode::RANGE_NOT_SATISFIABLE
                || (response.status() == StatusCode::PARTIAL_CONTENT && !append));
        if invalid_partial_response {
            fs::remove_file(staged_path)?;
            resume_offset = 0;
            if let Some(progress) = progress {
                progress.download_started(0);
            }
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

        let effective_resume_offset = if append { resume_offset } else { 0 };
        if let Some(progress) = progress {
            let response_total = if append {
                content_range_total(&response).or_else(|| {
                    response
                        .content_length()
                        .map(|length| effective_resume_offset.saturating_add(length))
                })
            } else {
                response.content_length()
            };
            if let Some(response_total) = response_total {
                progress.set_expected_bytes(response_total);
            }
            progress.download_started(effective_resume_offset);
        }

        let body_heartbeat = progress.map(|reporter| reporter.heartbeat("downloading asset bytes"));
        let stream_result = stream_asset_response_to_staged_file(
            &mut response,
            staged_path,
            append,
            effective_resume_offset,
            progress,
        );
        drop(body_heartbeat);
        stream_result?;
        let checksum = if let Some(progress) = progress {
            calculate_file_checksum_with_progress(
                staged_path,
                progress,
                "verifying the downloaded asset checksum",
            )?
        } else {
            calculate_file_checksum(staged_path)?
        };
        if checksum == *expected_checksum {
            return Ok(());
        }
        if append {
            fs::remove_file(staged_path)?;
            resume_offset = 0;
            if let Some(progress) = progress {
                progress.download_started(0);
            }
            continue;
        }

        fs::remove_file(staged_path)?;
        return Err(format!("Downloaded asset {} failed checksum validation", asset.id).into());
    }
}

/// Ensures an HTTP asset is available under its checksum-derived `.gen/assets` path.
///
/// [`download_asset_with_progress`] uses this storage phase before considering the logical
/// workspace path. The returned boolean reports whether this call published new bytes; the
/// returned file is always checksum-verified. No logical file is read or written here.
fn download_to_versioned_store_with_progress(
    client: &Client,
    workspace: &Workspace,
    asset: &AssetRef,
    url: &str,
    progress: Option<&AssetDownloadProgressReporter>,
) -> Result<(PathBuf, bool), Box<dyn Error>> {
    let expected_checksum = asset_checksum(asset)?;
    let versioned_path =
        materialization_destination_path(workspace, &asset.uri, Some(&expected_checksum), None)?;
    if versioned_path.exists() {
        let cached_checksum = if let Some(progress) = progress {
            calculate_file_checksum_with_progress(
                &versioned_path,
                progress,
                "checking the local asset cache",
            )
        } else {
            calculate_file_checksum(&versioned_path)
        };
        if cached_checksum.is_ok_and(|checksum| checksum == expected_checksum) {
            if let Some(progress) = progress {
                progress.reused_cached_asset();
            }
            return Ok((versioned_path, false));
        }
    }

    let asset_dir = versioned_path.parent().ok_or_else(|| {
        format!(
            "Versioned asset path has no parent: {}",
            versioned_path.display()
        )
    })?;
    fs::create_dir_all(asset_dir)?;
    let staged_path = temporary_path(&versioned_path)?;
    download_to_staged_path_with_progress(
        client,
        asset,
        url,
        &staged_path,
        &expected_checksum,
        progress,
    )?;
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
    copy_to_versioned_store_with_progress(source_workspace, destination_workspace, asset, None)
}

fn copy_to_versioned_store_with_progress(
    source_workspace: &Workspace,
    destination_workspace: &Workspace,
    asset: &AssetRef,
    progress: Option<&AssetDownloadProgressReporter>,
) -> Result<(PathBuf, bool), Box<dyn Error>> {
    let expected_checksum = asset_checksum(asset)?;
    let source_path = materialization_destination_path(
        source_workspace,
        &asset.uri,
        Some(&expected_checksum),
        None,
    )?;
    let source_checksum = (if let Some(progress) = progress {
        calculate_file_checksum_with_progress(
            &source_path,
            progress,
            "verifying the source asset checksum",
        )
    } else {
        calculate_file_checksum(&source_path)
    })
    .map_err(|error| {
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
    if destination_path.exists() {
        let destination_checksum = if let Some(progress) = progress {
            calculate_file_checksum_with_progress(
                &destination_path,
                progress,
                "checking the local asset cache",
            )
        } else {
            calculate_file_checksum(&destination_path)
        };
        if destination_checksum.is_ok_and(|checksum| checksum == expected_checksum) {
            if let Some(progress) = progress {
                progress.reused_cached_asset();
            }
            return Ok((destination_path, false));
        }
    }

    copy_versioned_asset_with_progress(
        &source_path,
        &destination_path,
        None,
        &expected_checksum,
        progress,
        AssetCopyProgress::Download,
    )?;
    Ok((destination_path, true))
}

#[derive(Clone, Copy)]
enum AssetCopyProgress {
    Download,
    Materialize,
}

/// Copies a stored archive or restores it to a workspace path without premature replacement.
///
/// Asset transfers use this stage to retain immutable bytes or decode BGZF for the workspace. The
/// staged output is synced and verified before rename so incomplete data cannot replace the target.
fn copy_versioned_asset_with_progress(
    versioned_path: &Path,
    destination_path: &Path,
    materialized_checksum: Option<&Sha256Hash>,
    expected_output_checksum: &Sha256Hash,
    progress: Option<&AssetDownloadProgressReporter>,
    copy_progress: AssetCopyProgress,
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
        if let Some(progress) = progress {
            match copy_progress {
                AssetCopyProgress::Download => progress.download_started(0),
                AssetCopyProgress::Materialize => progress.materialization_started(),
            }
        }
        let staged_checksum = {
            let mut checksummed_writer = ChecksummedWriter::new(&mut staged_file);
            match materialized_checksum {
                Some(_) => {
                    let mut decoded_file = bgzf::io::Reader::new(versioned_file);
                    if let Some(progress) = progress {
                        let mut decoded_file = CountingReader::new(decoded_file, |bytes| {
                            progress.materialized_bytes(bytes);
                        });
                        io::copy(&mut decoded_file, &mut checksummed_writer)?;
                    } else {
                        io::copy(&mut decoded_file, &mut checksummed_writer)?;
                    }
                }
                None => {
                    let mut versioned_file = versioned_file;
                    if let Some(progress) = progress {
                        let mut versioned_file =
                            CountingReader::new(versioned_file, |bytes| match copy_progress {
                                AssetCopyProgress::Download => progress.download_bytes(bytes),
                                AssetCopyProgress::Materialize => {
                                    progress.materialized_bytes(bytes)
                                }
                            });
                        io::copy(&mut versioned_file, &mut checksummed_writer)?;
                    } else {
                        io::copy(&mut versioned_file, &mut checksummed_writer)?;
                    }
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
    materialize_versioned_asset_with_progress(
        workspace,
        asset,
        previous_assets,
        destination_logical_path,
        versioned_path,
        versioned_file_created,
        None,
    )
}

fn materialize_versioned_asset_with_progress(
    workspace: &Workspace,
    asset: &AssetRef,
    previous_assets: &HashMap<HashId, AssetRef>,
    destination_logical_path: Option<&str>,
    versioned_path: &Path,
    versioned_file_created: bool,
    progress: Option<&AssetDownloadProgressReporter>,
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
    let versioned_checksum = if let Some(progress) = progress {
        calculate_file_checksum_with_progress(
            versioned_path,
            progress,
            "verifying the stored asset checksum",
        )?
    } else {
        calculate_file_checksum(versioned_path)?
    };
    if versioned_checksum != expected_checksum {
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
    if destination_path.exists() {
        let destination_checksum = if let Some(progress) = progress {
            calculate_file_checksum_with_progress(
                &destination_path,
                progress,
                "checking the existing workspace asset",
            )
        } else {
            calculate_file_checksum(&destination_path)
        };
        if destination_checksum.is_ok_and(|checksum| checksum == expected_materialized_checksum) {
            return Ok(DownloadAssetOutcome::Unchanged);
        }
    }
    let existing_checksum = if destination_path.exists() {
        Some(if let Some(progress) = progress {
            calculate_file_checksum_with_progress(
                &destination_path,
                progress,
                "checking the existing workspace asset",
            )?
        } else {
            calculate_file_checksum(&destination_path)?
        })
    } else {
        None
    };
    let intended_change = if let Some(checksum) = existing_checksum.as_ref() {
        destination_matches_previous_asset(
            workspace,
            &destination_path,
            checksum,
            previous_assets,
            progress,
        )?
    } else {
        false
    };
    let has_conflict = existing_checksum.is_some() && !intended_change;
    if has_conflict {
        let (conflict_path, already_downloaded) =
            conflict_destination_path(&destination_path, &expected_materialized_checksum)?;
        if !already_downloaded {
            copy_versioned_asset_with_progress(
                versioned_path,
                &conflict_path,
                asset.materialized_checksum.as_ref(),
                &expected_materialized_checksum,
                progress,
                AssetCopyProgress::Materialize,
            )?;
        }
        return Ok(DownloadAssetOutcome::Conflict(conflict_path));
    }

    copy_versioned_asset_with_progress(
        versioned_path,
        &destination_path,
        asset.materialized_checksum.as_ref(),
        &expected_materialized_checksum,
        progress,
        AssetCopyProgress::Materialize,
    )?;
    Ok(DownloadAssetOutcome::Downloaded)
}

/// Runs the HTTP asset pipeline: versioned storage first, optional materialization second.
///
/// [`transfer_assets`] calls this for each clone or pull URL returned by GenHub. Keeping the phases
/// ordered here prevents an already-current logical file from bypassing `.gen/assets` population.
fn download_asset_with_progress(
    client: &Client,
    workspace: &Workspace,
    asset: &AssetRef,
    previous_assets: &HashMap<HashId, AssetRef>,
    // `None` stores a historical version under `.gen/assets` instead of its recorded logical path.
    destination_logical_path: Option<&str>,
    url: &str,
    progress: Option<&AssetDownloadProgressReporter>,
) -> Result<DownloadAssetOutcome, Box<dyn Error>> {
    let (versioned_path, versioned_file_created) =
        download_to_versioned_store_with_progress(client, workspace, asset, url, progress)?;
    let result = materialize_versioned_asset_with_progress(
        workspace,
        asset,
        previous_assets,
        destination_logical_path,
        &versioned_path,
        versioned_file_created,
        progress,
    )?;
    if let Some(progress) = progress {
        progress.completed();
    }
    Ok(result)
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
    download_batch: Option<&mut AssetDownloadBatch<'_>>,
) -> Result<(), Box<dyn Error>> {
    if assets.is_empty() {
        return Ok(());
    }
    let remote_workspace = file_remote_workspace(&remote.url)?;
    let mut download_batch = download_batch;
    for asset in assets.values() {
        match operation {
            RemoteOperation::Push => {
                copy_to_versioned_store(workspace, &remote_workspace, asset)?;
            }
            RemoteOperation::Clone | RemoteOperation::Pull => {
                let reporter = download_batch
                    .as_deref_mut()
                    .map(|batch| batch.start_asset(asset));
                let destination_logical_path = if materialized_asset_ids.contains(&asset.id) {
                    asset.logical_path.as_deref()
                } else {
                    None
                };
                let result = (|| {
                    let (versioned_path, versioned_file_created) =
                        copy_to_versioned_store_with_progress(
                            &remote_workspace,
                            workspace,
                            asset,
                            reporter.as_ref(),
                        )?;
                    if let DownloadAssetOutcome::Conflict(conflict_path) =
                        materialize_versioned_asset_with_progress(
                            workspace,
                            asset,
                            previous_assets,
                            destination_logical_path,
                            &versioned_path,
                            versioned_file_created,
                            reporter.as_ref(),
                        )?
                    {
                        warn_asset_conflict(
                            workspace,
                            asset,
                            destination_logical_path,
                            &conflict_path,
                        )?;
                    }
                    Ok::<(), Box<dyn Error>>(())
                })();
                if let Err(error) = result {
                    if let Some(reporter) = reporter {
                        let context = reporter.failed();
                        return Err(
                            format!("{context}: {}", safe_transfer_error(error.as_ref())).into(),
                        );
                    }
                    return Err(error);
                }
                if let Some(reporter) = reporter {
                    reporter.completed();
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
    request: AssetTransferRun<'_>,
    mut complete_commit: impl FnMut(&DoltHashId) -> Result<(), Box<dyn Error>>,
    mut before_transfer: impl FnMut() -> Result<(), Box<dyn Error>>,
) -> Result<Vec<AssetUploadReceipt>, Box<dyn Error>> {
    let AssetTransferRun { target, progress } = request;
    let asset_history_heartbeat = progress.heartbeat("planning branch asset history");
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
    drop(asset_history_heartbeat);
    if remote.url.starts_with("file://") {
        match operation {
            RemoteOperation::Push => transfer_file_remote_assets(
                workspace,
                remote,
                operation,
                &assets,
                &materialized_asset_ids,
                &previous_assets,
                None,
            )?,
            RemoteOperation::Clone | RemoteOperation::Pull => {
                let assets_by_commit = {
                    let _heartbeat = progress.heartbeat("planning asset download checkpoints");
                    get_remaining_assets_to_transfer(graph, target.range.from_commit, &commit_hash)?
                };
                let mut download_batch =
                    AssetDownloadBatch::from_commits(progress, &assets_by_commit);
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
                        Some(&mut download_batch),
                    )?;
                    complete_commit(&commit_hash)?;
                }
            }
        }
        return Ok(Vec::new());
    }

    let repository = RepositoryRemote::parse(&remote.url)?;
    if operation == RemoteOperation::Push {
        let local_asset_count = assets.len();
        let asset_label = if local_asset_count == 1 {
            "asset"
        } else {
            "assets"
        };
        progress.println(&format!(
            "Checking {local_asset_count} local {asset_label} for transfer..."
        ));
        {
            let _heartbeat = progress.heartbeat("refreshing the push lease before asset transfer");
            before_transfer()?;
        }
    }
    let response = {
        let _heartbeat = progress.heartbeat("requesting GenHub asset transfer URLs");
        acquire_asset_transfers(
            &repository,
            &AssetTransferRequest {
                operation,
                branch: target.branch,
                from_commit: target.range.from_commit,
                to_commit: Some(&commit_hash),
            },
            login_origin,
        )?
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
        let upload_total = response
            .assets
            .iter()
            .filter(|transfer| assets.contains_key(&transfer.id))
            .count();
        if upload_total == 0 {
            progress.println("No new assets need uploading.");
        } else {
            let asset_label = if upload_total == 1 { "asset" } else { "assets" };
            progress.println(&format!(
                "Preparing {upload_total} {asset_label} for upload..."
            ));
        }
        let mut assets = assets;
        let mut upload_receipts = Vec::new();
        let mut upload_index = 0;
        for transfer in response.assets {
            let Some(asset) = assets.remove(&transfer.id) else {
                continue;
            };
            upload_index += 1;
            {
                let _heartbeat =
                    progress.heartbeat("refreshing the push lease before asset upload");
                before_transfer()?;
            }
            upload_receipts.push(upload_asset(
                &client,
                workspace,
                &asset,
                &transfer.url,
                upload_index,
                upload_total,
                progress,
            )?);
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
    let assets_by_commit = {
        let _heartbeat = progress.heartbeat("planning asset download checkpoints");
        get_remaining_assets_to_transfer(graph, target.range.from_commit, &commit_hash)?
    };
    let mut download_batch = AssetDownloadBatch::from_commits(progress, &assets_by_commit);
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
            let reporter = download_batch.start_asset(&asset);
            let download_result = download_asset_with_progress(
                &client,
                workspace,
                &asset,
                &previous_assets,
                destination_logical_path,
                &url,
                Some(&reporter),
            );
            let outcome = match download_result {
                Ok(outcome) => outcome,
                Err(error) => {
                    let progress_context = reporter.failed();
                    return Err(format!(
                        "{progress_context}: {}",
                        safe_transfer_error(error.as_ref())
                    )
                    .into());
                }
            };
            if let DownloadAssetOutcome::Conflict(conflict_path) = outcome {
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
    let started = Instant::now();
    let progress = TransferProgress::stderr();
    progress.println("Preparing graph clone...");
    workspace.ensure_gen_dir();
    let graph_path = workspace.graph_db_path().map_err(|error| {
        clone_transfer_failure("resolving the local graph database path", started, &error)
    })?;
    let graph = {
        let _heartbeat = progress.heartbeat("opening the local graph database");
        get_raw_connection(graph_path).map_err(|error| {
            clone_transfer_failure("opening the local graph database", started, &error)
        })?
    };
    let attempt_count = if remote.url.starts_with("file://") {
        1
    } else {
        2
    };
    let mut clone_result = None;
    let mut capability_identity = CapabilityTransportIdentity::default();
    for attempt in 0..attempt_count {
        let attempt_number = attempt + 1;
        if attempt > 0 {
            progress.println("Requesting fresh graph access for the clone retry...");
        } else if remote.url.starts_with("file://") {
            progress.println("Opening the local graph remote...");
        } else {
            progress.println(&format!(
                "Requesting graph access (attempt {attempt_number}/{attempt_count})..."
            ));
        }
        let authorization_phase = if remote.url.starts_with("file://") {
            "opening the local graph remote"
        } else {
            "requesting GenHub graph clone capability"
        };
        let authorization_result = if remote.url.starts_with("file://") {
            transfer_authorization(
                remote,
                RemoteOperation::Clone,
                None,
                false,
                None,
                login_origin,
            )
        } else {
            let _heartbeat = progress.heartbeat(authorization_phase);
            transfer_authorization(
                remote,
                RemoteOperation::Clone,
                None,
                false,
                None,
                login_origin,
            )
        };
        let authorization = match authorization_result {
            Ok(authorization) => authorization,
            Err(error) => {
                restore_canonical_url(&graph, remote);
                return Err(clone_transfer_failure(
                    authorization_phase,
                    started,
                    error.as_ref(),
                ));
            }
        };
        let Some(remote_url) = authorization.remote_url.as_deref() else {
            restore_canonical_url(&graph, remote);
            return Err(clone_transfer_failure(
                "reading the GenHub graph clone capability",
                started,
                &std::io::Error::other(
                    "GenHub clone capability did not include a graph remote URL",
                ),
            ));
        };
        progress.println(&format!(
            "Cloning graph history (attempt {attempt_number}/{attempt_count})..."
        ));
        let clone_result_for_attempt = {
            let _heartbeat = progress.heartbeat("cloning graph history from the remote");
            if remote.url.starts_with("file://") {
                clone_remote(&graph, remote_url)
            } else {
                match capability_graph_remote(remote_url, &mut capability_identity) {
                    Ok(graph_remote) => clone_remote(&graph, &graph_remote.url),
                    Err(error) => Err(error),
                }
            }
        };
        match clone_result_for_attempt {
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
                    progress.println(
                        "The graph session could not be used; retrying with freshly requested graph access...",
                    );
                    continue;
                }
                break;
            }
        }
    }
    restore_canonical_url(&graph, remote);
    clone_result
        .expect("should attempt clone at least once")
        .map_err(|error| {
            clone_transfer_failure("cloning graph history from the remote", started, &error)
        })?;
    progress.println("Graph history cloned; preparing branch assets...");
    let branch = active_branch(&graph)
        .map_err(|error| clone_transfer_failure("reading the cloned branch", started, &error))?;
    drop(graph);
    let graph_path = workspace.graph_db_path().map_err(|error| {
        clone_transfer_failure("resolving the cloned graph database path", started, &error)
    })?;
    let graph = {
        let _heartbeat = progress.heartbeat("reopening the cloned graph database");
        get_raw_connection(graph_path).map_err(|error| {
            clone_transfer_failure("reopening the cloned graph database", started, &error)
        })?
    };
    let mut operation = RemoteOperationRecord::begin_or_resume(
        config,
        &remote.name,
        &branch,
        StoredRemoteOperationKind::Clone,
        None,
    )
    .map_err(|error| {
        clone_transfer_failure("initializing the clone asset checkpoint", started, &error)
    })?;
    let destination_hash = {
        let _heartbeat = progress.heartbeat("reading the cloned branch head");
        hash_of(&graph, &branch).map_err(|error| {
            clone_transfer_failure("reading the cloned branch head", started, &error)
        })?
    };
    operation
        .set_destination(config, &destination_hash)
        .map_err(|error| {
            clone_transfer_failure("recording the clone destination", started, &error)
        })?;
    let assets_transfer_checkpoint = operation.assets_transfer_checkpoint;
    let previous_hash = operation.from_commit;
    transfer_assets(
        &graph,
        workspace,
        remote,
        RemoteOperation::Clone,
        AssetTransferRun {
            target: AssetTransferTarget {
                branch: &branch,
                history_ref: &branch,
                range: AssetTransferRange {
                    from_commit: assets_transfer_checkpoint.as_ref(),
                    previous_hash: previous_hash.as_ref(),
                    materialize: true,
                },
            },
            progress: &progress,
        },
        |commit_hash| {
            operation.advance_assets_transfer_checkpoint(config, commit_hash)?;
            Ok(())
        },
        || Ok(()),
    )
    .map_err(|error| {
        clone_transfer_failure("downloading branch assets", started, error.as_ref())
    })?;
    operation.complete(config).map_err(|error| {
        clone_transfer_failure("completing the clone checkpoint", started, &error)
    })?;
    progress.println(&format!("Clone of branch '{branch}' is complete."));
    Ok(branch)
}

#[derive(Debug)]
struct CloneTransferFailure(String);

impl fmt::Display for CloneTransferFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for CloneTransferFailure {}

fn clone_transfer_failure(
    phase: &str,
    started: Instant,
    error: &(dyn Error + 'static),
) -> Box<dyn Error> {
    let message = format!(
        "Clone failed after {} during {phase}: {}",
        format_elapsed(started.elapsed()),
        safe_transfer_error(error),
    );
    Box::new(CloneTransferFailure(message))
}

fn safe_transfer_error(error: &(dyn Error + 'static)) -> String {
    if let Some(client_error) = error.downcast_ref::<RemoteClientError>() {
        return match client_error {
            RemoteClientError::InvalidRepositoryUrl(_) => {
                "the configured GenHub repository URL is invalid".to_string()
            }
            RemoteClientError::AuthenticationRequired => {
                "authentication is required; run `gen remote login` and confirm repository read access".to_string()
            }
            RemoteClientError::Http { status, .. } => {
                let guidance = match status.as_u16() {
                    401 | 403 => "Check your GenHub login and repository read access.",
                    404 => "Check that the repository URL is correct and your account can access it.",
                    429 => "GenHub is rate limiting requests; retry after a short delay.",
                    500..=599 => "GenHub reported a server error; retry after the service recovers.",
                    _ => "Check the repository permissions and retry.",
                };
                format!("GenHub returned HTTP {status}. {guidance}")
            }
            RemoteClientError::StaleGraphSession => {
                "GenHub rejected the graph session; retry the clone for a fresh session".to_string()
            }
            RemoteClientError::ResponseDecode { status, .. } => {
                format!("GenHub returned an invalid response (HTTP {status})")
            }
            RemoteClientError::Request(_) => {
                "the GenHub request failed; check network connectivity and retry".to_string()
            }
            RemoteClientError::TokenStorage(_) => {
                "Gen could not read saved GenHub credentials; check credential file access".to_string()
            }
        };
    }
    if let Some(sql_error) = error.downcast_ref::<SqlError>() {
        return match sql_error {
            SqlError::SqliteFailure(code, _) => format!(
                "Dolt returned SQLite error {:?} (extended code {})",
                code.code, code.extended_code
            ),
            _ => "the local Dolt graph database operation failed".to_string(),
        };
    }
    if error.downcast_ref::<reqwest::Error>().is_some() {
        return "the asset HTTP request failed; check network connectivity and retry".to_string();
    }
    if let Some(source) = error.source()
        && let Some(message) = safe_transfer_error_chain(source)
    {
        return message;
    }
    error.to_string()
}

fn safe_transfer_error_chain(error: &(dyn Error + 'static)) -> Option<String> {
    if error.downcast_ref::<RemoteClientError>().is_some()
        || error.downcast_ref::<SqlError>().is_some()
        || error.downcast_ref::<reqwest::Error>().is_some()
    {
        return Some(safe_transfer_error(error));
    }
    error.source().and_then(safe_transfer_error_chain)
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
    /// Another process owns the pending push session for this remote and branch.
    #[error(
        "Another push to remote '{remote}' branch '{branch}' is already running in this workspace"
    )]
    PushLockBusy { remote: String, branch: String },
    /// The workspace push lock could not be created or acquired.
    #[error(
        "Could not acquire push lock for remote '{remote}' branch '{branch}' at '{}': {source}",
        .path.display()
    )]
    PushLockIo {
        remote: String,
        branch: String,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// A GenHub request failed.
    #[error(transparent)]
    Client(#[from] RemoteClientError),
    /// The remote could not be selected from the command and repository configuration.
    #[error("Unable to resolve push remote: {0}")]
    RemoteResolution(#[source] Box<dyn Error>),
    /// Dolt could not transfer the graph branch.
    #[error("Graph transfer failed: {0}")]
    GraphTransfer(#[source] Box<dyn Error>),
    /// The remote branch has commits that are not in the local branch.
    #[error(
        "The branch is not a fast-forward of the remote. Use --force to overwrite the remote.{cleanup_warning}"
    )]
    NonFastForward { cleanup_warning: String },
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

fn remote_push_graph_error(error: Box<dyn Error>) -> RemotePushError {
    if error
        .downcast_ref::<SqlError>()
        .is_some_and(is_non_fast_forward_push_error)
    {
        return RemotePushError::NonFastForward {
            cleanup_warning: String::new(),
        };
    }
    let Some(graph_error) = error.downcast_ref::<PushGraphTransferError>() else {
        return RemotePushError::GraphTransfer(error);
    };
    if graph_error.is_non_fast_forward() {
        let warning = graph_error.non_fast_forward_cleanup_warning();
        return RemotePushError::NonFastForward {
            cleanup_warning: if warning.is_empty() {
                String::new()
            } else {
                format!(" Cleanup warning: {warning}")
            },
        };
    }
    RemotePushError::GraphTransfer(error)
}

fn remote_push_asset_error(
    operation: Option<&mut RemoteOperationRecord>,
    config: &ConfigConnection,
    branch: &str,
    error: Box<dyn Error>,
) -> RemotePushError {
    if let Some(graph_error) = error.downcast_ref::<PushGraphTransferError>() {
        if graph_error.is_terminal()
            && let Some(operation) = operation
            && let Err(metadata_error) = operation.fail(config)
        {
            eprintln!(
                "Warning: failed to record unsuccessful push operation for branch '{branch}': {metadata_error}"
            );
        }
        remote_push_graph_error(error)
    } else {
        RemotePushError::AssetTransfer(error)
    }
}

fn same_push_session_scope(
    operation: &RemoteOperationRecord,
    destination_hash: &DoltHashId,
    force: bool,
) -> bool {
    operation.push_session_target_commit.as_ref() == Some(destination_hash)
        && operation.push_session_force == Some(force)
}

/// An incomplete push session can hold the repository claim needed by a later graph head.
///
/// When a pending operation moves to a new destination or force scope, retire its previous
/// intended session before persisting the replacement. This covers interrupted staging as well as
/// a published manifest whose asset transfer did not finish.
fn retire_superseded_push_session(
    operation: &RemoteOperationRecord,
    remote: &Remote,
    branch: &str,
    progress: &TransferProgress,
) -> Result<(), PushGraphTransferError> {
    let (Some(session_id), Some(_old_destination), Some(old_force)) = (
        operation.push_session_id,
        operation.push_session_target_commit.as_ref(),
        operation.push_session_force,
    ) else {
        return Ok(());
    };

    progress.println("Retiring the previous direct GCS push session...");
    let _heartbeat = progress.heartbeat("retiring the previous direct GCS push session");
    match acquire_push_session_renewal(remote, branch, old_force, session_id, None)
        .and_then(|renewal| renewal.abort())
    {
        Ok(()) | Err(PushGraphTransferError::CapabilityLeaseInactive) => Ok(()),
        Err(error) => Err(error),
    }
}

pub fn execute_push(
    workspace: &Workspace,
    explicit_remote: Option<&str>,
    explicit_branch: Option<&str>,
    force: bool,
) -> Result<(), RemotePushError> {
    let progress = TransferProgress::stderr();
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
    let _push_lock = push_lock::acquire(workspace, &remote.name, &branch)?;
    progress.println(&format!(
        "Preparing push of branch '{branch}' to '{}'...",
        remote.name,
    ));
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
            || push_graph_branch(&graph, &remote.name, &branch, force),
        )
        .map_err(remote_push_graph_error)?;
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
        let same_session_scope = same_push_session_scope(&operation, &destination_hash, force);
        if !same_session_scope
            && let Err(error) =
                retire_superseded_push_session(&operation, &remote, &branch, &progress)
        {
            if error.is_terminal()
                && let Err(metadata_error) = operation.fail(&config)
            {
                eprintln!(
                    "Warning: failed to record unsuccessful push operation for branch '{branch}': {metadata_error}"
                );
            }
            return Err(RemotePushError::GraphTransfer(Box::new(error)));
        }
        operation.prepare_push_session(&config, &destination_hash, force)?;
        let published_session_matches =
            same_session_scope && operation.to_commit.as_ref() == Some(&destination_hash);
        let existing_transfer_lease = if published_session_matches {
            match (operation.transfer_id, operation.transfer_expires_at) {
                (Some(transfer_id), Some(expires_at)) if expires_at > Utc::now().timestamp() => {
                    Some(PushTransferLease {
                        transfer_id,
                        expires_at,
                    })
                }
                (Some(_), Some(_)) => None,
                (None, None) => None,
                _ => return Err(RemotePushError::IncompleteTransferLease { branch }),
            }
        } else {
            None
        };
        let renewal = if let Some(transfer_lease) = existing_transfer_lease {
            progress.println("Graph manifest already published; resuming asset transfer...");
            let push_session_id = operation
                .push_session_id
                .expect("push session UUID should be persisted before transfer");
            let renewal_result = {
                let _heartbeat = progress.heartbeat("resuming the existing direct-push lease");
                acquire_push_session_renewal(
                    &remote,
                    &branch,
                    force,
                    push_session_id,
                    Some(transfer_lease.transfer_id),
                )
            };
            let renewal = match renewal_result {
                Ok(renewal) => renewal,
                Err(error) => {
                    if error.is_terminal()
                        && let Err(metadata_error) = operation.fail(&config)
                    {
                        eprintln!(
                            "Warning: failed to record unsuccessful push operation for branch '{branch}': {metadata_error}"
                        );
                    }
                    return Err(RemotePushError::GraphTransfer(Box::new(error)));
                }
            };
            let refreshed_lease_result = {
                let _heartbeat = progress.heartbeat("refreshing the existing push lease");
                renewal.ensure_fresh()
            };
            let refreshed_lease = match refreshed_lease_result {
                Ok(lease) => lease,
                Err(error) => {
                    if error.is_terminal()
                        && let Err(metadata_error) = operation.fail(&config)
                    {
                        eprintln!(
                            "Warning: failed to record unsuccessful push operation for branch '{branch}': {metadata_error}"
                        );
                    }
                    return Err(RemotePushError::GraphTransfer(Box::new(error)));
                }
            };
            operation.set_push_destination(
                &config,
                &destination_hash,
                refreshed_lease.transfer_id,
                refreshed_lease.expires_at,
            )?;
            renewal
        } else {
            let push_session_id = operation
                .push_session_id
                .expect("push session UUID should be persisted before transfer");
            let expected_transfer_id = published_session_matches
                .then_some(operation.transfer_id)
                .flatten();
            let mut graph_transfer = run_push_graph_transfer(PushGraphTransferRequest {
                graph: &graph,
                remote: &remote,
                branch: &branch,
                force,
                idempotency_token: push_session_id,
                expected_transfer_id,
                destination_hash: &destination_hash,
                progress: &progress,
            });
            let confirmed_stale_session = graph_transfer.as_ref().is_err_and(|error| {
                error
                    .downcast_ref::<PushGraphTransferError>()
                    .is_some_and(|error| {
                        matches!(error, PushGraphTransferError::StaleSessionConfirmed)
                    })
            });
            if confirmed_stale_session {
                progress.println(
                    "GenHub confirmed the accepted graph session is stale; opening a fresh session...",
                );
                let replacement_session_id = Uuid::new_v4();
                operation.reset_stale_push_session(&config, replacement_session_id)?;
                graph_transfer = run_push_graph_transfer(PushGraphTransferRequest {
                    graph: &graph,
                    remote: &remote,
                    branch: &branch,
                    force,
                    idempotency_token: replacement_session_id,
                    expected_transfer_id: None,
                    destination_hash: &destination_hash,
                    progress: &progress,
                });
            }
            let renewal = match graph_transfer {
                Ok(renewal) => renewal,
                Err(error) => {
                    let terminal_direct_failure = error
                        .downcast_ref::<PushGraphTransferError>()
                        .is_some_and(PushGraphTransferError::is_terminal);
                    let should_fail_operation = terminal_direct_failure;
                    if should_fail_operation && let Err(metadata_error) = operation.fail(&config) {
                        eprintln!(
                            "Warning: failed to record unsuccessful push operation for branch '{branch}': {metadata_error}"
                        );
                    }
                    return Err(remote_push_graph_error(error));
                }
            };
            let transfer_lease = renewal
                .push_lease()
                .map_err(|error| RemotePushError::GraphTransfer(Box::new(error)))?;
            operation.set_push_destination(
                &config,
                &destination_hash,
                transfer_lease.transfer_id,
                transfer_lease.expires_at,
            )?;
            renewal
        };
        Some((operation, renewal, destination_hash))
    };
    let assets_transfer_checkpoint = if let Some((operation, _, _)) = push_context.as_ref() {
        operation.assets_transfer_checkpoint
    } else {
        previous_hash
    };
    let upload_receipts_result = {
        let mut refresh_push_lease = || {
            if let Some((operation, renewal, destination_hash)) = push_context.as_mut() {
                persist_refreshed_push_lease(operation, &config, destination_hash, renewal)?;
            }
            Ok(())
        };
        transfer_assets(
            &graph,
            workspace,
            &remote,
            RemoteOperation::Push,
            AssetTransferRun {
                target: AssetTransferTarget {
                    branch: &branch,
                    history_ref: &branch,
                    range: AssetTransferRange {
                        from_commit: assets_transfer_checkpoint.as_ref(),
                        previous_hash: previous_hash.as_ref(),
                        materialize: true,
                    },
                },
                progress: &progress,
            },
            |_| Ok(()),
            &mut refresh_push_lease,
        )
    };
    let upload_receipts = match upload_receipts_result {
        Ok(upload_receipts) => upload_receipts,
        Err(error) => {
            let operation = push_context.as_mut().map(|(operation, _, _)| operation);
            return Err(remote_push_asset_error(operation, &config, &branch, error));
        }
    };
    if let Some((operation, renewal, destination_hash)) = push_context.as_mut() {
        if let Err(error) =
            persist_refreshed_push_lease(operation, &config, destination_hash, renewal)
        {
            return Err(remote_push_asset_error(
                Some(operation),
                &config,
                &branch,
                error,
            ));
        }
        let transfer_lease = renewal
            .push_lease()
            .map_err(|error| RemotePushError::GraphTransfer(Box::new(error)))?;
        let repository = RepositoryRemote::parse(&remote.url)?;
        progress.println("Verifying asset uploads and finishing push...");
        let verification_started = Instant::now();
        let verification_result = {
            let _heartbeat = progress
                .heartbeat("waiting for GenHub to verify asset uploads and finish the push");
            complete_asset_transfers(
                &repository,
                &AssetTransferCompletionRequest {
                    transfer_id: transfer_lease.transfer_id,
                    branch: &branch,
                    assets: &upload_receipts,
                },
                login_origin,
            )
        };
        if let Err(error) = verification_result {
            progress.println(&format!(
                "GenHub did not confirm asset verification after {}; per-asset request progress above reflects bytes consumed.",
                format_elapsed(verification_started.elapsed()),
            ));
            return Err(RemotePushError::Client(error));
        }
        progress.println("GenHub verified and accepted the asset uploads.");
        operation.advance_assets_transfer_checkpoint(&config, destination_hash)?;
        operation.complete(&config)?;
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
    let progress = TransferProgress::stderr();
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
        AssetTransferRun {
            target: AssetTransferTarget {
                branch: &branch,
                history_ref: &branch,
                range: AssetTransferRange {
                    from_commit: assets_transfer_checkpoint.as_ref(),
                    previous_hash: previous_hash.as_ref(),
                    materialize: true,
                },
            },
            progress: &progress,
        },
        |commit_hash| {
            operation.advance_assets_transfer_checkpoint(&config, commit_hash)?;
            Ok(())
        },
        || Ok(()),
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
    let progress = TransferProgress::stderr();
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
        || fetch(&graph, &remote.name, Some(&branch)),
    )?;

    let tracking_ref = format!("{}/{}", remote.name, branch);
    transfer_assets(
        &graph,
        workspace,
        &remote,
        RemoteOperation::Pull,
        AssetTransferRun {
            target: AssetTransferTarget {
                branch: &branch,
                history_ref: &tracking_ref,
                range: AssetTransferRange {
                    from_commit: None,
                    previous_hash: None,
                    materialize: false,
                },
            },
            progress: &progress,
        },
        |_| Ok(()),
        || Ok(()),
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
    mod fake_gcs {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/support/fake_gcs.rs"
        ));
    }

    #[cfg(unix)]
    use std::os::unix::fs::symlink as create_file_symlink;
    #[cfg(windows)]
    use std::os::windows::fs::symlink_file as create_file_symlink;
    use std::{
        collections::HashMap,
        env,
        ffi::OsString,
        fs,
        io::{BufRead as _, BufReader, Cursor, Read as _, Write as _},
        net::TcpListener,
        path::{Path, PathBuf},
        sync::Mutex,
        thread,
        time::{Duration, Instant},
    };

    use chrono::{DateTime, Duration as ChronoDuration, Utc};
    use fake_gcs::FakeGcsServer;
    use flate2::{Compression, write::GzEncoder};
    use gen_core::{DoltHashId, HashId, Sha256Hash, config::Workspace};
    use gen_models::{
        assets::{AssetRef, AssetRole, LocalAssetUri, materialization_destination_path},
        collection::Collection,
        db::{ConfigConnection, GraphConnection},
        file_types::FileTypes,
        history::dolt::{clone_remote, commit_all, remote_rows, remove_remote},
        operations::{
            Defaults, FileAddition, OperationFile, Remote,
            RemoteOperationKind as StoredRemoteOperationKind, RemoteOperationRecord,
            calculate_reader_checksum,
        },
    };
    use noodles::bgzf;
    use reqwest::blocking::Client;
    use rusqlite::{Connection, Error as SqlError, blockcachevfs::AuthRefreshReason};
    use serde_json::json;
    use tempfile::tempdir;
    use url::Url;
    use uuid::Uuid;

    use super::{
        AssetDownloadBatch, AssetTransferRange, AssetTransferRun, AssetTransferTarget,
        DownloadAssetOutcome, RemoteOperation, canonical_remote_url, clone_destination_name,
        clone_into_workspace, download_to_staged_path_with_progress, ensure_graph_remote,
        execute_pull, execute_push, file_graph_url, get_remaining_assets_to_transfer,
        resolve_remote, run_graph_transfer, temporary_path, transfer_assets, upload_asset,
    };
    use crate::{
        commands::remote::{
            client::{
                CapabilityResponse, DirectPushCapability, SessionScope as ClientSessionScope,
            },
            progress::tests::captured_transfer_progress,
        },
        get_config_connection, get_connection, get_raw_connection,
    };

    static ENVIRONMENT_LOCK: Mutex<()> = Mutex::new(());
    const TEST_TRANSFER_ID: Uuid = Uuid::from_u128(1);
    const RETRIED_TRANSFER_ID: Uuid = Uuid::from_u128(2);

    fn file_path_uri(path: &Path) -> String {
        Url::from_file_path(path)
            .map(String::from)
            .expect("should convert fixture file path to a file URI")
    }

    fn file_directory_uri(path: &Path) -> String {
        Url::from_directory_path(path)
            .map(String::from)
            .expect("should convert fixture directory path to a file URI")
    }

    fn download_to_versioned_store(
        client: &Client,
        workspace: &Workspace,
        asset: &AssetRef,
        url: &str,
    ) -> Result<(PathBuf, bool), Box<dyn std::error::Error>> {
        super::download_to_versioned_store_with_progress(client, workspace, asset, url, None)
    }

    fn copy_versioned_asset(
        versioned_path: &std::path::Path,
        destination_path: &std::path::Path,
        materialized_checksum: Option<&Sha256Hash>,
        expected_output_checksum: &Sha256Hash,
    ) -> Result<(), Box<dyn std::error::Error>> {
        super::copy_versioned_asset_with_progress(
            versioned_path,
            destination_path,
            materialized_checksum,
            expected_output_checksum,
            None,
            super::AssetCopyProgress::Materialize,
        )
    }

    fn download_asset(
        client: &Client,
        workspace: &Workspace,
        asset: &AssetRef,
        previous_assets: &HashMap<HashId, AssetRef>,
        destination_logical_path: Option<&str>,
        url: &str,
    ) -> Result<DownloadAssetOutcome, Box<dyn std::error::Error>> {
        super::download_asset_with_progress(
            client,
            workspace,
            asset,
            previous_assets,
            destination_logical_path,
            url,
            None,
        )
    }

    fn push_capability_response(
        token: &str,
        token_expires_at: DateTime<Utc>,
        expires_at: DateTime<Utc>,
        database_uri: &str,
        publish_url: &str,
        abort_url: &str,
        session_scope: ClientSessionScope,
    ) -> CapabilityResponse {
        CapabilityResponse {
            remote_url: None,
            expires_at,
            default_branch: "main".to_string(),
            transfer_id: TEST_TRANSFER_ID,
            direct_push: Some(DirectPushCapability {
                database_uri: database_uri.replace("TOKEN", token),
                token_expires_at,
                session_id: TEST_TRANSFER_ID,
                session_scope,
                publish_url: publish_url.to_string(),
                abort_url: abort_url.to_string(),
            }),
        }
    }

    fn push_session_scope() -> ClientSessionScope {
        ClientSessionScope {
            principal: "repository:repo-id".to_string(),
            target_database: "default.db".to_string(),
            operations: r#"["write","push","main",false]"#.to_string(),
        }
    }

    fn push_database_uri() -> &'static str {
        "gcs://bucket/.gen/graph_db/?database=default.db&project=test-project&vfs=blockcachevfs&endpoint=https%3A%2F%2Fstorage.googleapis.com&access_token=TOKEN"
    }

    fn request_session_id(request: &str) -> Uuid {
        request
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("idempotency-token")
                    .then(|| value.trim().parse::<Uuid>().ok())
                    .flatten()
            })
            .expect("capability request should include its persisted session UUID")
    }

    fn direct_push_capability_body(request: &str, origin: &str, transfer_id: Uuid) -> String {
        let scope = push_session_scope();
        json!({
            "remote_url": null,
            "expires_at": (Utc::now() + ChronoDuration::minutes(30)).to_rfc3339(),
            "default_branch": "main",
            "transfer_id": transfer_id,
            "direct_push": {
                "database_uri": push_database_uri(),
                "token_expires_at": (Utc::now() + ChronoDuration::hours(1)).to_rfc3339(),
                "session_id": request_session_id(request),
                "session_scope": {
                    "principal": scope.principal,
                    "target_database": scope.target_database,
                    "operations": scope.operations,
                },
                "publish_url": format!("{origin}/api/publish?capability=mock-signed-value"),
                "abort_url": format!("{origin}/api/abort?capability=mock-signed-value")
            }
        })
        .to_string()
    }

    #[test]
    fn test_push_capability_renewal_rotates_token_and_publication_capability() {
        let _environment_lock = ENVIRONMENT_LOCK
            .lock()
            .expect("should lock process environment");
        let _api_key = EnvironmentGuard::set("GENHUB_API_KEY", "renewal-test-key");
        let now = Utc::now();
        let scope = push_session_scope();
        let listener = TcpListener::bind("127.0.0.1:0").expect("should bind renewal server");
        let address = listener
            .local_addr()
            .expect("should read renewal server address");
        let publish_origin = format!("http://{address}");
        let refreshed_expiry = now + ChronoDuration::minutes(30);
        let refreshed_token_expiry = now + ChronoDuration::hours(2);
        let initial = push_capability_response(
            "initial-token",
            now + ChronoDuration::hours(1),
            now + ChronoDuration::minutes(15),
            push_database_uri(),
            &format!("{publish_origin}/api/publish?capability=initial-signed-value"),
            &format!("{publish_origin}/api/abort?capability=initial-signed-value"),
            scope.clone(),
        );
        let body = json!({
            "remote_url": null,
            "expires_at": refreshed_expiry.to_rfc3339(),
            "default_branch": "main",
            "transfer_id": TEST_TRANSFER_ID,
            "direct_push": {
                "database_uri": "gcs://bucket/.gen/graph_db/?access_token=rotated-token&endpoint=https%3A%2F%2Fstorage.googleapis.com&vfs=blockcachevfs&project=test-project&database=default.db",
                "token_expires_at": refreshed_token_expiry.to_rfc3339(),
                "session_id": TEST_TRANSFER_ID,
                "session_scope": {
                    "principal": scope.principal,
                    "target_database": scope.target_database,
                    "operations": scope.operations,
                },
                "publish_url": format!("{publish_origin}/api/publish?capability=refreshed-signed-value"),
                "abort_url": format!("{publish_origin}/api/abort?capability=refreshed-signed-value")
            }
        })
        .to_string();
        let server = thread::spawn(move || {
            let mut requests = Vec::new();
            for request_number in 0..2 {
                let (mut stream, _) = listener
                    .accept()
                    .expect("should accept capability or publication request");
                let mut request = [0_u8; 8192];
                let length = stream
                    .read(&mut request)
                    .expect("should read capability or publication request");
                requests.push(String::from_utf8_lossy(&request[..length]).into_owned());
                if request_number == 0 {
                    write!(
                        stream,
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    )
                    .expect("should send renewed capability");
                } else {
                    stream
                        .write_all(b"HTTP/1.1 204 No Content\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                        .expect("should accept manifest publication");
                }
            }
            requests
        });

        let remote = Remote {
            name: "origin".to_string(),
            url: format!("http://{address}/api/repos/alice/example"),
        };
        let renewal = super::capability::push_capability_renewal(
            &remote,
            "main",
            false,
            TEST_TRANSFER_ID,
            TEST_TRANSFER_ID,
            initial,
        )
        .expect("should create the direct-push renewal coordinator");

        let token = renewal
            .auth_token(AuthRefreshReason::Unauthorized)
            .expect("should renew after an authorization rejection");
        renewal
            .publish()
            .expect("should publish through the refreshed capability");
        let requests = server
            .join()
            .expect("should finish capability and publication server");
        let capability_request = &requests[0];
        let publication_request = &requests[1];
        let latest = renewal
            .capability_response()
            .expect("should read the renewed capability response");
        let latest_lease = renewal
            .push_lease()
            .expect("should read the renewed push lease");

        assert_eq!(token, "rotated-token");
        assert!(
            capability_request
                .to_ascii_lowercase()
                .contains(&format!("idempotency-token: {TEST_TRANSFER_ID}"))
        );
        assert!(
            publication_request.starts_with("POST /api/publish?capability=refreshed-signed-value ")
        );
        assert!(
            publication_request
                .lines()
                .any(|line| line.eq_ignore_ascii_case("content-length: 0"))
        );
        assert!(publication_request.ends_with("\r\n\r\n"));
        assert_eq!(latest.transfer_id, TEST_TRANSFER_ID);
        assert_eq!(latest_lease.transfer_id, TEST_TRANSFER_ID);
        assert_eq!(latest_lease.expires_at, refreshed_expiry.timestamp());
        assert_eq!(
            latest
                .direct_push
                .as_ref()
                .expect("should keep a direct-push capability")
                .publish_url,
            format!("{publish_origin}/api/publish?capability=refreshed-signed-value")
        );
    }

    #[test]
    fn test_session_auth_io_error_is_retryable_authorization_failure() {
        let error = SqlError::SqliteFailure(
            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_IOERR_AUTH),
            Some("credential refresh failed".to_string()),
        );

        assert!(super::is_authorization_error(&error));
    }

    fn seed_push_lease(
        config: &ConfigConnection,
        branch: &str,
        destination_hash: &DoltHashId,
        transfer_id: Uuid,
        expires_at: i64,
    ) -> Uuid {
        let mut operation = seed_push_session(config, branch, destination_hash, false);
        let session_id = operation
            .push_session_id
            .expect("should allocate a push session UUID");
        operation
            .set_push_destination(config, destination_hash, transfer_id, expires_at)
            .expect("should seed graph transfer lease");
        session_id
    }

    fn seed_push_session(
        config: &ConfigConnection,
        branch: &str,
        destination_hash: &DoltHashId,
        force: bool,
    ) -> RemoteOperationRecord {
        let mut operation = RemoteOperationRecord::begin_or_resume(
            config,
            "origin",
            branch,
            StoredRemoteOperationKind::Push,
            None,
        )
        .expect("should create pending push operation");
        operation
            .prepare_push_session(config, destination_hash, force)
            .expect("should prepare the seeded push scope");
        operation
    }

    fn reload_push_operation(
        config: &ConfigConnection,
        operation_id: i64,
    ) -> RemoteOperationRecord {
        RemoteOperationRecord::select(config)
            .get_by_id(operation_id)
            .expect("should query the persisted push operation")
            .expect("should find the persisted push operation")
    }

    enum MockGenHubReply {
        Drop,
        Http(u16, String),
        DirectPushCapability(Uuid),
        EmptySuccess,
    }

    impl From<Option<(u16, String)>> for MockGenHubReply {
        fn from(reply: Option<(u16, String)>) -> Self {
            match reply {
                Some((status, body)) => Self::Http(status, body),
                None => Self::Drop,
            }
        }
    }

    fn read_mock_http_request(stream: &mut std::net::TcpStream) -> String {
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .expect("should bound mock HTTP request reads");
        let mut request = Vec::new();
        loop {
            let mut buffer = [0_u8; 4096];
            let read = stream
                .read(&mut buffer)
                .expect("should read mock HTTP request");
            if read == 0 {
                break;
            }
            request.extend_from_slice(&buffer[..read]);
            let Some(header_end) = request.windows(4).position(|window| window == b"\r\n\r\n")
            else {
                continue;
            };
            let content_length = String::from_utf8_lossy(&request[..header_end])
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())
                        .flatten()
                })
                .unwrap_or(0);
            if request.len() >= header_end + 4 + content_length {
                break;
            }
        }
        String::from_utf8_lossy(&request).into_owned()
    }

    fn serve_capability_replies<R>(replies: Vec<R>) -> (String, thread::JoinHandle<Vec<String>>)
    where
        R: Into<MockGenHubReply>,
    {
        let replies = replies.into_iter().map(Into::into).collect::<Vec<_>>();
        let listener = TcpListener::bind("127.0.0.1:0").expect("should bind capability server");
        listener
            .set_nonblocking(true)
            .expect("should bound capability server waits");
        let address = listener
            .local_addr()
            .expect("should read capability server address");
        let server = thread::spawn(move || {
            let mut requests = Vec::with_capacity(replies.len());
            for reply in replies {
                let deadline = Instant::now() + Duration::from_secs(10);
                let (mut stream, _) = loop {
                    match listener.accept() {
                        Ok(accepted) => break accepted,
                        Err(error)
                            if error.kind() == std::io::ErrorKind::WouldBlock
                                && Instant::now() < deadline =>
                        {
                            thread::sleep(Duration::from_millis(10));
                        }
                        Err(error) => {
                            panic!("should accept capability request before timeout: {error}")
                        }
                    }
                };
                let request = read_mock_http_request(&mut stream);
                requests.push(request.clone());
                match reply {
                    MockGenHubReply::Drop => {}
                    MockGenHubReply::DirectPushCapability(transfer_id) => {
                        let body = direct_push_capability_body(
                            &request,
                            &format!("http://{address}"),
                            transfer_id,
                        );
                        write!(
                            stream,
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                            body.len()
                        )
                        .expect("should write direct-push capability");
                    }
                    MockGenHubReply::EmptySuccess => stream
                        .write_all(
                            b"HTTP/1.1 204 No Content\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                        )
                        .expect("should write empty success response"),
                    MockGenHubReply::Http(status, body) => {
                        let reason = match status {
                            200 => "OK",
                            409 => "Conflict",
                            500 => "Internal Server Error",
                            503 => "Service Unavailable",
                            _ => panic!("test capability server does not support HTTP {status}"),
                        };
                        write!(
                            stream,
                            "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                            body.len()
                        )
                        .expect("should write capability response");
                    }
                }
            }
            requests
        });
        (format!("http://{address}/api/repos/alice/example"), server)
    }

    fn observe_push_requests() -> (String, thread::JoinHandle<Vec<String>>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("should bind push observer");
        listener
            .set_nonblocking(true)
            .expect("should bound push observer waits");
        let address = listener
            .local_addr()
            .expect("should read push observer address");
        let observer = thread::spawn(move || {
            let mut requests = Vec::new();
            let deadline = Instant::now() + Duration::from_millis(300);
            while Instant::now() < deadline {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        requests.push(read_mock_http_request(&mut stream));
                        stream
                            .write_all(
                                b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                            )
                            .expect("should reject an unexpected push request");
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(10));
                    }
                    Err(error) => panic!("should accept unexpected push requests: {error}"),
                }
            }
            requests
        });
        (
            format!("http://{address}/api/repos/alice/example"),
            observer,
        )
    }

    fn serve_published_session_retirement() -> (String, thread::JoinHandle<Vec<String>>) {
        let unavailable = r#"{"code":503,"message":"capability service unavailable"}"#;
        serve_capability_replies(vec![
            MockGenHubReply::DirectPushCapability(TEST_TRANSFER_ID),
            MockGenHubReply::EmptySuccess,
            MockGenHubReply::Http(503, unavailable.to_string()),
            MockGenHubReply::Http(503, unavailable.to_string()),
        ])
    }

    fn create_http_push_fixture(
        root: &Path,
        remote_url: &str,
    ) -> (Workspace, ConfigConnection, DoltHashId) {
        let workspace = Workspace::new(root.join("local"));
        workspace.ensure_gen_dir();
        let config = get_config_connection(
            workspace
                .gen_db_path()
                .expect("should resolve push config path"),
        )
        .expect("should open push config");
        let graph = get_connection(
            workspace
                .graph_db_path()
                .expect("should resolve push graph path"),
        )
        .expect("should open push graph");
        Collection::create(&graph, "push-session-fixture")
            .expect("should create push session fixture");
        let destination =
            commit_all(&graph, "push session fixture").expect("should commit push session fixture");
        let remote = Remote::create(&config, "origin", remote_url)
            .expect("should configure push session remote");
        Defaults::set_default_remote(&config, Some(&remote.name))
            .expect("should set push session remote as default");
        drop(graph);
        (workspace, config, destination)
    }

    fn push_operation_sessions(config: &ConfigConnection) -> Vec<(Option<Uuid>, bool)> {
        let mut statement = config
            .prepare(
                "SELECT push_session_id, failed_at IS NOT NULL FROM remote_operations \
                 WHERE operation = 'push' ORDER BY id",
            )
            .expect("should query push operation session state");
        statement
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .expect("should read push operation session state")
            .collect::<Result<Vec<_>, _>>()
            .expect("should collect push operation session state")
    }

    mod clone {
        use gen_core::config::Workspace;
        use gen_models::{history::dolt::remote_rows, operations::Remote};
        use serde_json::json;
        use tempfile::tempdir;

        use super::{
            super::{clone_into_workspace, ensure_graph_remote},
            ENVIRONMENT_LOCK, EnvironmentGuard, TEST_TRANSFER_ID, serve_capability_replies,
        };
        use crate::{get_config_connection, get_raw_connection};

        #[test]
        fn test_clone_rejects_unsupported_network_capabilities() {
            let _environment_lock = ENVIRONMENT_LOCK
                .lock()
                .expect("should lock process environment");
            let _api_key = EnvironmentGuard::set("GENHUB_API_KEY", "read-test-key");

            for graph_url in [
                "https://genhub.bio/api/repos/alice/example/dolt/capability/default.db",
                "file:///tmp/graph/default.db",
            ] {
                let body = json!({
                    "remote_url": graph_url,
                    "expires_at": "2035-01-01T00:00:00Z",
                    "default_branch": "main",
                    "transfer_id": TEST_TRANSFER_ID,
                })
                .to_string();
                let (repository_url, capability_server) =
                    serve_capability_replies(vec![Some((200, body))]);
                let temp = tempdir().expect("should create clone fixture");
                let workspace = Workspace::new(temp.path().join("clone"));
                workspace.ensure_gen_dir();
                let config = get_config_connection(Some(
                    workspace
                        .gen_db_path()
                        .expect("should resolve clone config path"),
                ))
                .expect("should open clone config");
                let remote = Remote::create(&config, "origin", &repository_url)
                    .expect("should configure clone remote");
                let graph = get_raw_connection(
                    workspace
                        .graph_db_path()
                        .expect("should resolve clone graph path"),
                )
                .expect("should open clone graph");
                ensure_graph_remote(&graph, &remote.name, &remote.url)
                    .expect("should seed the canonical graph remote");
                drop(graph);

                let error = clone_into_workspace(&config, &remote, &workspace)
                    .expect_err("should reject an unsupported cloud capability");
                let requests = capability_server
                    .join()
                    .expect("should finish capability server");
                let graph = get_raw_connection(
                    workspace
                        .graph_db_path()
                        .expect("should resolve cloned graph path"),
                )
                .expect("should reopen cloned graph");
                let graph_remotes = remote_rows(&graph)
                    .expect("should read graph remotes after rejected capability");

                assert_eq!(
                    requests.len(),
                    1,
                    "clone should not renew an unsupported grant"
                );
                assert!(!error.to_string().contains(graph_url));
                assert!(graph_remotes.iter().any(|graph_remote| {
                    graph_remote.name == remote.name && graph_remote.url == remote.url
                }));
            }
        }
    }

    #[test]
    fn test_asset_download_progress_sanitizes_terminal_control_characters() {
        let (progress, lines) = captured_transfer_progress(std::time::Duration::ZERO);
        let mut batch = AssetDownloadBatch {
            progress: &progress,
            total: 1,
            next_index: 0,
        };
        let mut asset = test_asset(b"contents", "reference.fa", 1);
        asset.name = Some("ref\u{1b}[2J.fa\n".to_string());

        batch.start_asset(&asset).download_started(0);

        let lines = lines
            .lock()
            .expect("should read captured download progress");
        assert!(
            lines.iter().any(|line| line.contains("ref?[2J.fa?")),
            "control characters in remote asset names should be replaced before display"
        );
        assert!(
            lines.iter().all(|line| !line.chars().any(char::is_control)),
            "progress output should not contain terminal control characters"
        );
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

    fn serve_asset_ignoring_range(contents: &[u8]) -> (String, thread::JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("should bind asset server");
        let address = listener
            .local_addr()
            .expect("should read asset server address");
        let contents = contents.to_vec();
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("should accept asset request");
            let mut request = Vec::new();
            let mut buffer = [0_u8; 1024];
            while !request.windows(4).any(|window| window == b"\r\n\r\n") {
                let read = stream.read(&mut buffer).expect("should read asset request");
                if read == 0 {
                    break;
                }
                request.extend_from_slice(&buffer[..read]);
            }
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                contents.len()
            )
            .expect("should write full asset response headers");
            stream
                .write_all(&contents)
                .expect("should write full asset response body");
            String::from_utf8_lossy(&request).into_owned()
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

    #[test]
    fn test_ignored_range_resets_progress_to_the_full_response_length() {
        let temp = tempdir().expect("should create workspace");
        let workspace = Workspace::new(temp.path());
        workspace.ensure_gen_dir();
        let remote_contents = b"the complete body returned when Range is ignored";
        let mut remote_asset = test_asset(remote_contents, "reference.fa", 1);
        remote_asset.size = Some(1);
        let versioned_path = versioned_asset_path(&workspace, &remote_asset);
        fs::create_dir_all(
            versioned_path
                .parent()
                .expect("should resolve versioned asset directory"),
        )
        .expect("should create versioned asset directory");
        let staged_path =
            temporary_path(&versioned_path).expect("should resolve staged asset path");
        let stale_prefix = b"a stale prefix that is larger than the metadata size";
        fs::write(&staged_path, stale_prefix).expect("should seed stale staged bytes");
        let (url, server) = serve_asset_ignoring_range(remote_contents);
        let (progress, lines) = captured_transfer_progress(std::time::Duration::ZERO);
        let reporter = progress.asset_download_reporter(1, 1, "reference.fa".to_string(), Some(1));

        download_to_staged_path_with_progress(
            &Client::new(),
            &remote_asset,
            &url,
            &staged_path,
            remote_asset
                .checksum
                .as_ref()
                .expect("test asset should have a checksum"),
            Some(&reporter),
        )
        .expect("ignored Range should restart from the full response");
        let request = server.join().expect("asset server should finish");

        assert!(
            request
                .to_ascii_lowercase()
                .contains(&format!("\r\nrange: bytes={}-\r\n", stale_prefix.len())),
            "the downloader should preserve its range request and accept a full 200 response"
        );
        assert_eq!(
            fs::read(&staged_path).expect("should read restarted staged asset"),
            remote_contents
        );
        let lines = lines
            .lock()
            .expect("should read captured download progress");
        assert!(
            lines.iter().any(|line| {
                line.contains(&format!(
                    "{} B read this attempt; {} B of {} B staged (100%)",
                    remote_contents.len(),
                    remote_contents.len(),
                    remote_contents.len(),
                ))
            }),
            "the 200 response should reset the resumed prefix and replace metadata size with Content-Length: {lines:?}"
        );
        assert!(
            lines
                .iter()
                .any(|line| line.contains("size pending verification")),
            "an oversized staged prefix should not appear complete against stale metadata"
        );
        assert!(
            !lines
                .iter()
                .any(|line| line.contains(&format!("{} B of 1 B staged", remote_contents.len(),))),
            "the progress denominator should come from the response length"
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
            AssetTransferRun {
                target: AssetTransferTarget {
                    branch: "main",
                    history_ref: "main",
                    range: AssetTransferRange {
                        from_commit: Some(&previous_hash),
                        previous_hash: Some(&previous_hash),
                        materialize: true,
                    },
                },
                progress: &crate::commands::remote::progress::TransferProgress::stderr(),
            },
            |_| Ok(()),
            || Ok(()),
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
            AssetTransferRun {
                target: AssetTransferTarget {
                    branch: "main",
                    history_ref: "main",
                    range: AssetTransferRange {
                        from_commit: None,
                        previous_hash: None,
                        materialize: true,
                    },
                },
                progress: &crate::commands::remote::progress::TransferProgress::stderr(),
            },
            |_| Ok(()),
            || Ok(()),
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
    fn test_clone_materializes_source_symlinked_asset_as_regular_files() {
        let temp = tempdir().expect("should create clone transfer workspace");
        let source_root = temp.path().join("source");
        let source_workspace = Workspace::new(&source_root);
        source_workspace.ensure_gen_dir();
        let source_graph = get_connection(
            source_workspace
                .graph_db_path()
                .expect("should resolve source graph path"),
        )
        .expect("should open source graph");
        Collection::create(&source_graph, "base").expect("should create source collection");
        let contents = b"source symlink asset\n";
        let asset = test_asset(contents, "linked.gfa", 1);
        AssetRef::create(&source_graph, &asset).expect("should insert source asset");
        commit_all(&source_graph, "add source asset").expect("should commit source asset");
        drop(source_graph);

        let external_payload = temp.path().join("source-payload.gfa");
        fs::write(&external_payload, contents).expect("should write source payload");
        let source_archive = versioned_asset_path(&source_workspace, &asset);
        fs::create_dir_all(
            source_archive
                .parent()
                .expect("should have source archive parent"),
        )
        .expect("should create source archive directory");
        create_file_symlink(&external_payload, &source_archive)
            .expect("should link retained asset to source payload");
        assert!(
            fs::symlink_metadata(&source_archive)
                .expect("should inspect source archive link")
                .file_type()
                .is_symlink(),
            "fixture should contain a source-owned retained-asset symlink"
        );

        let destination_root = temp.path().join("clone");
        let destination_workspace = Workspace::new(&destination_root);
        destination_workspace.ensure_gen_dir();
        let config = get_config_connection(Some(
            destination_workspace
                .gen_db_path()
                .expect("should resolve destination config path"),
        ))
        .expect("should open destination config");
        let remote = Remote::create(&config, "origin", &file_directory_uri(&source_root))
            .expect("should configure source workspace remote");

        clone_into_workspace(&config, &remote, &destination_workspace)
            .expect("should clone source asset bytes");

        let received_archive = versioned_asset_path(&destination_workspace, &asset);
        let archive_metadata = fs::symlink_metadata(&received_archive)
            .expect("should inspect received versioned asset");
        assert!(archive_metadata.file_type().is_file());
        assert!(!archive_metadata.file_type().is_symlink());
        assert_eq!(
            fs::read(&received_archive).expect("should read received versioned asset"),
            contents
        );
        let materialized_asset = destination_root.join("linked.gfa");
        let materialized_metadata = fs::symlink_metadata(&materialized_asset)
            .expect("should inspect received materialized asset");
        assert!(materialized_metadata.file_type().is_file());
        assert!(!materialized_metadata.file_type().is_symlink());
        assert_eq!(
            fs::read(materialized_asset).expect("should read received materialized asset"),
            contents
        );
        assert_eq!(
            fs::read(&external_payload).expect("should reread source payload"),
            contents,
            "clone should read through the source symlink without writing its target"
        );
    }

    #[test]
    fn test_pull_materializes_source_symlinked_asset_as_regular_files() {
        let temp = tempdir().expect("should create pull transfer workspace");
        let source_root = temp.path().join("source");
        let source_workspace = Workspace::new(&source_root);
        source_workspace.ensure_gen_dir();
        let source_graph = get_connection(
            source_workspace
                .graph_db_path()
                .expect("should resolve source graph path"),
        )
        .expect("should open source graph");
        Collection::create(&source_graph, "base").expect("should create source collection");
        commit_all(&source_graph, "create base state").expect("should commit source base");
        drop(source_graph);

        let destination_root = temp.path().join("pull");
        let destination_workspace = Workspace::new(&destination_root);
        destination_workspace.ensure_gen_dir();
        let destination_graph = get_raw_connection(
            destination_workspace
                .graph_db_path()
                .expect("should resolve destination graph path"),
        )
        .expect("should open destination graph");
        clone_remote(
            &destination_graph,
            &file_path_uri(
                &source_workspace
                    .graph_db_path()
                    .expect("should resolve source graph path"),
            ),
        )
        .expect("should clone source base state");
        drop(destination_graph);

        let config = get_config_connection(Some(
            destination_workspace
                .gen_db_path()
                .expect("should resolve destination config path"),
        ))
        .expect("should open destination config");
        let remote = Remote::create(&config, "origin", &file_directory_uri(&source_root))
            .expect("should configure source workspace remote");
        Defaults::set_default_remote(&config, Some(&remote.name))
            .expect("should set source workspace as default remote");

        let contents = b"source symlink pull asset\n";
        let asset = test_asset(contents, "linked.gfa", 1);
        let source_graph = get_connection(
            source_workspace
                .graph_db_path()
                .expect("should resolve source graph path"),
        )
        .expect("should reopen source graph");
        AssetRef::create(&source_graph, &asset).expect("should insert source asset");
        commit_all(&source_graph, "add source asset").expect("should commit source asset");
        drop(source_graph);

        let external_payload = temp.path().join("source-payload.gfa");
        fs::write(&external_payload, contents).expect("should write source payload");
        let source_archive = versioned_asset_path(&source_workspace, &asset);
        fs::create_dir_all(
            source_archive
                .parent()
                .expect("should have source archive parent"),
        )
        .expect("should create source archive directory");
        create_file_symlink(&external_payload, &source_archive)
            .expect("should link retained asset to source payload");
        assert!(
            fs::symlink_metadata(&source_archive)
                .expect("should inspect source archive link")
                .file_type()
                .is_symlink(),
            "fixture should contain a source-owned retained-asset symlink"
        );

        execute_pull(&destination_workspace, None, None).expect("should pull source asset");

        let received_archive = versioned_asset_path(&destination_workspace, &asset);
        let archive_metadata = fs::symlink_metadata(&received_archive)
            .expect("should inspect received versioned asset");
        assert!(archive_metadata.file_type().is_file());
        assert!(!archive_metadata.file_type().is_symlink());
        assert_eq!(
            fs::read(&received_archive).expect("should read received versioned asset"),
            contents
        );
        let materialized_asset = destination_root.join("linked.gfa");
        let materialized_metadata = fs::symlink_metadata(&materialized_asset)
            .expect("should inspect received materialized asset");
        assert!(materialized_metadata.file_type().is_file());
        assert!(!materialized_metadata.file_type().is_symlink());
        assert_eq!(
            fs::read(materialized_asset).expect("should read received materialized asset"),
            contents
        );
        assert_eq!(
            fs::read(&external_payload).expect("should reread source payload"),
            contents,
            "pull should read through the source symlink without writing its target"
        );
    }

    #[test]
    fn test_http_roundtrip_transfers_source_symlinked_asset_as_regular_files() {
        let temp = tempdir().expect("should create HTTP transfer workspace");
        let source_workspace = Workspace::new(temp.path().join("source"));
        source_workspace.ensure_gen_dir();
        let destination_root = temp.path().join("receiver");
        let destination_workspace = Workspace::new(&destination_root);
        destination_workspace.ensure_gen_dir();

        let contents = b"source symlink HTTP asset\n";
        let (asset, archived_contents) = test_bgzf_asset(contents, "linked.gfa", 1);

        let external_payload = temp.path().join("source-payload.gfa");
        fs::write(&external_payload, &archived_contents).expect("should write source payload");
        let source_archive = versioned_asset_path(&source_workspace, &asset);
        fs::create_dir_all(
            source_archive
                .parent()
                .expect("should have source archive parent"),
        )
        .expect("should create source archive directory");
        create_file_symlink(&external_payload, &source_archive)
            .expect("should link retained asset to source payload");
        assert!(
            fs::symlink_metadata(&source_archive)
                .expect("should inspect source archive link")
                .file_type()
                .is_symlink(),
            "fixture should contain a source-owned retained-asset symlink"
        );

        let listener = TcpListener::bind("127.0.0.1:0").expect("should bind asset server");
        let address = listener
            .local_addr()
            .expect("should read asset server address");
        let server = thread::spawn(move || {
            let (upload_stream, _) = listener.accept().expect("should accept asset upload");
            let mut upload_reader = BufReader::new(upload_stream);
            let mut upload_request = String::new();
            upload_reader
                .read_line(&mut upload_request)
                .expect("should read upload request line");
            let mut content_length = None;
            loop {
                let mut header = String::new();
                upload_reader
                    .read_line(&mut header)
                    .expect("should read upload headers");
                if header == "\r\n" {
                    break;
                }
                if let Some((name, value)) = header.split_once(':')
                    && name.eq_ignore_ascii_case("content-length")
                {
                    content_length = value.trim().parse::<usize>().ok();
                }
            }
            let mut uploaded = vec![0; content_length.expect("should receive content length")];
            upload_reader
                .read_exact(&mut uploaded)
                .expect("should read uploaded asset bytes");
            write!(
                upload_reader.get_mut(),
                "HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            )
            .expect("should finish upload response");
            drop(upload_reader);

            let (download_stream, _) = listener.accept().expect("should accept asset download");
            let mut download_reader = BufReader::new(download_stream);
            let mut download_request = String::new();
            download_reader
                .read_line(&mut download_request)
                .expect("should read download request line");
            loop {
                let mut header = String::new();
                download_reader
                    .read_line(&mut header)
                    .expect("should read download headers");
                if header == "\r\n" {
                    break;
                }
            }
            let mut response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                uploaded.len()
            )
            .into_bytes();
            response.extend_from_slice(&uploaded);
            download_reader
                .get_mut()
                .write_all(&response)
                .expect("should return uploaded asset bytes");
            (upload_request, download_request, uploaded)
        });

        let client = Client::new();
        upload_asset(
            &client,
            &source_workspace,
            &asset,
            &format!("http://{address}/upload"),
            1,
            1,
            &crate::commands::remote::progress::TransferProgress::stderr(),
        )
        .expect("should upload source symlink asset bytes");
        let download_result = download_asset(
            &client,
            &destination_workspace,
            &asset,
            &HashMap::new(),
            asset.logical_path.as_deref(),
            &format!("http://{address}/download"),
        );
        let (upload_request, download_request, uploaded) =
            server.join().expect("should finish asset server");

        assert!(upload_request.starts_with("PUT /upload "));
        assert!(download_request.starts_with("GET /download "));
        assert_eq!(uploaded, archived_contents);
        assert_eq!(
            download_result.expect("should download and materialize uploaded bytes"),
            DownloadAssetOutcome::Downloaded
        );
        let received_archive = versioned_asset_path(&destination_workspace, &asset);
        let archive_metadata = fs::symlink_metadata(&received_archive)
            .expect("should inspect received versioned asset");
        assert!(archive_metadata.file_type().is_file());
        assert!(!archive_metadata.file_type().is_symlink());
        assert_eq!(
            fs::read(&received_archive).expect("should read received versioned asset"),
            archived_contents
        );
        let materialized_asset = destination_root.join("linked.gfa");
        let materialized_metadata = fs::symlink_metadata(&materialized_asset)
            .expect("should inspect received materialized asset");
        assert!(materialized_metadata.file_type().is_file());
        assert!(!materialized_metadata.file_type().is_symlink());
        assert_eq!(
            fs::read(materialized_asset).expect("should read received materialized asset"),
            contents
        );
        assert_eq!(
            fs::read(&external_payload).expect("should reread source payload"),
            archived_contents,
            "HTTP transfer should read through the source symlink without writing its target"
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
        let fake_gcs = FakeGcsServer::start();
        let gcs_database = fake_gcs.create_database(&remote_graph_path);
        let cloud_graph_url = gcs_database.uri();
        let (remote_url, api_server) =
            serve_pull_api(&cloud_graph_url, asset.id, &asset_url, 1, false);
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
        let fake_gcs = FakeGcsServer::start();
        let gcs_database = fake_gcs.create_database(&remote_graph_path);
        let cloud_graph_url = gcs_database.uri();
        let (remote_url, api_server) =
            serve_pull_api(&cloud_graph_url, asset.id, &asset_url, 2, true);
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
        let fake_gcs = FakeGcsServer::start();
        let gcs_database = fake_gcs.create_database(&remote_graph_path);
        let cloud_graph_url = gcs_database.uri();
        let (remote_url, api_server) = serve_checkpoint_pull_api(
            &cloud_graph_url,
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
    fn test_pull_rejects_non_cloud_network_capability() {
        let _environment_lock = ENVIRONMENT_LOCK
            .lock()
            .expect("should lock process environment");
        let _api_key = EnvironmentGuard::set("GENHUB_API_KEY", "push-test-key");
        let listener = TcpListener::bind("127.0.0.1:0").expect("should bind capability server");
        let address = listener
            .local_addr()
            .expect("should read capability server address");
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("should accept capability request");
            let mut request = [0_u8; 8192];
            let _ = stream
                .read(&mut request)
                .expect("should read capability request");
            let body = format!(
                "{{\"remote_url\":\"http://127.0.0.1:1/legacy-transfer\",\
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
        });

        let connection = Connection::open_in_memory().expect("should open graph database");
        let graph = GraphConnection(connection);
        let remote = Remote {
            name: "origin".to_string(),
            url: format!("http://{address}/api/repos/alice/example"),
        };
        ensure_graph_remote(&graph, &remote.name, &remote.url)
            .expect("should configure canonical graph remote");
        let error = run_graph_transfer(
            &graph,
            &remote,
            RemoteOperation::Pull,
            "main",
            false,
            || panic!("Dolt should not run after rejecting a non-cloud capability"),
        )
        .expect_err("should reject the signed HTTP network capability");
        server.join().expect("capability server should finish");

        assert!(!error.to_string().contains("legacy-transfer"));
        let remotes = remote_rows(&graph).expect("should read restored canonical URL");
        assert!(
            remotes
                .iter()
                .any(|graph_remote| graph_remote.name == "origin" && graph_remote.url == remote.url)
        );
    }

    #[test]
    fn test_push_rejects_legacy_http_capability_without_uploading_assets() {
        let _environment_lock = ENVIRONMENT_LOCK
            .lock()
            .expect("should lock process environment");
        let _api_key = EnvironmentGuard::set("GENHUB_API_KEY", "push-test-key");
        let listener = TcpListener::bind("127.0.0.1:0").expect("should bind capability server");
        let address = listener
            .local_addr()
            .expect("should read capability server address");
        let temp = tempdir().expect("should create push test directory");
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("should accept capability request");
            let mut request = [0_u8; 8192];
            let read = stream
                .read(&mut request)
                .expect("should read capability request");
            let request = String::from_utf8_lossy(&request[..read]).into_owned();
            let body = format!(
                "{{\"remote_url\":\"http://127.0.0.1:1/legacy-transfer\",\
                 \"expires_at\":\"2030-01-01T00:00:00Z\",\
                 \"default_branch\":\"main\",\
                 \"transfer_id\":\"{TEST_TRANSFER_ID}\"}}"
            );
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            )
            .expect("should write legacy capability response");
            request
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

        let error = execute_push(&workspace, None, None, false)
            .expect_err("push should reject a legacy HTTP-only capability");
        let request = server.join().expect("capability server should finish");

        assert!(error.to_string().contains("direct GCS session"));
        assert!(request.contains("\"operation\":\"push\""));
        assert!(request.to_ascii_lowercase().contains("idempotency-token:"));
        assert!(!request.contains("/asset-transfers "));
        let failed_operations = config
            .query_row(
                "SELECT COUNT(*) FROM remote_operations \
                 WHERE operation = 'push' AND failed_at IS NOT NULL",
                [],
                |row| row.get::<_, i64>(0),
            )
            .expect("should count failed push operations");
        assert_eq!(failed_operations, 1);
    }

    #[test]
    fn test_push_transfers_source_symlinked_asset_as_regular_archive() {
        let temp = tempdir().expect("should create push transfer workspace");
        let source_root = temp.path().join("source");
        let source_workspace = Workspace::new(&source_root);
        source_workspace.ensure_gen_dir();
        let source_graph = get_connection(
            source_workspace
                .graph_db_path()
                .expect("should resolve source graph path"),
        )
        .expect("should open source graph");
        Collection::create(&source_graph, "base").expect("should create source collection");
        commit_all(&source_graph, "create base state").expect("should commit source base");
        drop(source_graph);

        let destination_root = temp.path().join("remote");
        let destination_workspace = Workspace::new(&destination_root);
        destination_workspace.ensure_gen_dir();
        let destination_graph = get_raw_connection(
            destination_workspace
                .graph_db_path()
                .expect("should resolve destination graph path"),
        )
        .expect("should open destination graph");
        clone_remote(
            &destination_graph,
            &file_path_uri(
                &source_workspace
                    .graph_db_path()
                    .expect("should resolve source graph path"),
            ),
        )
        .expect("should clone source base state to destination");
        drop(destination_graph);

        let contents = b"source symlink push asset\n";
        let asset = test_asset(contents, "linked.gfa", 1);
        let source_graph = get_connection(
            source_workspace
                .graph_db_path()
                .expect("should resolve source graph path"),
        )
        .expect("should reopen source graph");
        AssetRef::create(&source_graph, &asset).expect("should insert source asset");
        commit_all(&source_graph, "add source asset").expect("should commit source asset");
        drop(source_graph);

        let external_payload = temp.path().join("source-payload.gfa");
        fs::write(&external_payload, contents).expect("should write source payload");
        let source_archive = versioned_asset_path(&source_workspace, &asset);
        fs::create_dir_all(
            source_archive
                .parent()
                .expect("should have source archive parent"),
        )
        .expect("should create source archive directory");
        create_file_symlink(&external_payload, &source_archive)
            .expect("should link retained asset to source payload");
        assert!(
            fs::symlink_metadata(&source_archive)
                .expect("should inspect source archive link")
                .file_type()
                .is_symlink(),
            "fixture should contain a source-owned retained-asset symlink"
        );

        let config = get_config_connection(Some(
            source_workspace
                .gen_db_path()
                .expect("should resolve source config path"),
        ))
        .expect("should open source config");
        let remote = Remote::create(&config, "origin", &file_directory_uri(&destination_root))
            .expect("should configure destination workspace remote");
        Defaults::set_default_remote(&config, Some(&remote.name))
            .expect("should set destination workspace as default remote");

        execute_push(&source_workspace, None, None, false).expect("should push source asset");

        let received_archive = versioned_asset_path(&destination_workspace, &asset);
        let archive_metadata = fs::symlink_metadata(&received_archive)
            .expect("should inspect received versioned asset");
        assert!(archive_metadata.file_type().is_file());
        assert!(!archive_metadata.file_type().is_symlink());
        assert_eq!(
            fs::read(&received_archive).expect("should read received versioned asset"),
            contents
        );
        assert_eq!(
            fs::read(&external_payload).expect("should reread source payload"),
            contents,
            "push should read through the source symlink without writing its target"
        );
    }

    #[test]
    fn test_permanent_initial_push_capability_conflict_retires_operation() {
        let environment_lock = ENVIRONMENT_LOCK
            .lock()
            .expect("should lock process environment");
        let api_key = EnvironmentGuard::set("GENHUB_API_KEY", "push-test-key");
        let (remote_url, server) = serve_capability_replies(vec![Some((
            409,
            json!({"code": 409, "message": "Remote transfer ID cannot be reused"}).to_string(),
        ))]);
        let temp = tempdir().expect("should create rejected push directory");
        let (workspace, config, _) = create_http_push_fixture(temp.path(), &remote_url);

        let result = execute_push(&workspace, None, None, false);
        let requests = server.join().expect("capability server should finish");
        let sessions = push_operation_sessions(&config);
        drop(api_key);
        drop(environment_lock);

        assert!(
            result.is_err(),
            "permanent capability conflict should fail push"
        );
        assert_eq!(
            requests.len(),
            1,
            "push should stop at the rejected capability"
        );
        assert_eq!(
            sessions.len(),
            1,
            "push should record one logical operation"
        );
        assert!(
            sessions[0].1,
            "permanent capability conflict should retire the rejected operation"
        );
    }

    #[test]
    fn test_retry_after_permanent_push_capability_conflict_uses_fresh_session() {
        let environment_lock = ENVIRONMENT_LOCK
            .lock()
            .expect("should lock process environment");
        let api_key = EnvironmentGuard::set("GENHUB_API_KEY", "push-test-key");
        let conflict = || {
            Some((
                409,
                json!({"code": 409, "message": "Remote transfer ID cannot be reused"}).to_string(),
            ))
        };
        let (remote_url, server) = serve_capability_replies(vec![conflict(), conflict()]);
        let temp = tempdir().expect("should create rejected push retry directory");
        let (workspace, config, _) = create_http_push_fixture(temp.path(), &remote_url);

        let first_result = execute_push(&workspace, None, None, false);
        let second_result = execute_push(&workspace, None, None, false);
        let requests = server.join().expect("capability server should finish");
        let sessions = push_operation_sessions(&config);
        drop(api_key);
        drop(environment_lock);

        assert!(
            first_result.is_err(),
            "first permanent conflict should fail push"
        );
        assert!(
            second_result.is_err(),
            "repeated permanent conflict should fail push"
        );
        assert_eq!(requests.len(), 2, "both push attempts should reach GenHub");
        assert_ne!(
            request_session_id(&requests[0]),
            request_session_id(&requests[1]),
            "a rejected permanent session UUID must not be reused on retry"
        );
        assert_eq!(sessions.len(), 2, "retry should start a new operation row");
        assert!(
            sessions.iter().all(|(_, failed)| *failed),
            "each permanent capability rejection should retire its operation"
        );
    }

    #[test]
    fn test_force_change_after_retryable_capability_failure_uses_fresh_session() {
        let environment_lock = ENVIRONMENT_LOCK
            .lock()
            .expect("should lock process environment");
        let api_key = EnvironmentGuard::set("GENHUB_API_KEY", "push-test-key");
        let unavailable = || {
            MockGenHubReply::Http(
                503,
                json!({"code": 503, "message": "capability service unavailable"}).to_string(),
            )
        };
        let (remote_url, server) = serve_capability_replies(vec![
            unavailable(),
            MockGenHubReply::DirectPushCapability(TEST_TRANSFER_ID),
            MockGenHubReply::EmptySuccess,
            unavailable(),
        ]);
        let temp = tempdir().expect("should create force retry directory");
        let (workspace, config, _) = create_http_push_fixture(temp.path(), &remote_url);

        let first_result = execute_push(&workspace, None, None, false);
        let second_result = execute_push(&workspace, None, None, true);
        let requests = server.join().expect("capability server should finish");
        let sessions = push_operation_sessions(&config);
        drop(api_key);
        drop(environment_lock);

        assert!(
            first_result.is_err(),
            "first retryable capability failure should fail push"
        );
        assert!(
            second_result.is_err(),
            "second retryable capability failure should fail push"
        );
        assert_eq!(
            requests.len(),
            4,
            "the old session should be reacquired and aborted before the new force scope is requested"
        );
        let first_body = requests[0]
            .split_once("\r\n\r\n")
            .map(|(_, body)| body)
            .expect("first capability request should have a body");
        let retired_body = requests[1]
            .split_once("\r\n\r\n")
            .map(|(_, body)| body)
            .expect("old session capability request should have a body");
        let second_body = requests[3]
            .split_once("\r\n\r\n")
            .map(|(_, body)| body)
            .expect("new force-scoped capability request should have a body");
        assert!(first_body.contains("\"force\":false"));
        assert!(retired_body.contains("\"force\":false"));
        assert!(second_body.contains("\"force\":true"));
        assert_ne!(
            request_session_id(&requests[0]),
            request_session_id(&requests[3]),
            "changing force changes the GenHub authorization scope and needs a fresh session UUID"
        );
        assert_eq!(
            request_session_id(&requests[0]),
            request_session_id(&requests[1])
        );
        assert!(requests[2].starts_with("POST /api/abort?capability=mock-signed-value "));
        assert!(
            sessions.iter().any(|(session_id, failed)| {
                !failed && *session_id == Some(request_session_id(&requests[3]))
            }),
            "the retry should leave a pending operation for its fresh force-scoped session"
        );
    }

    #[test]
    fn test_active_push_lock_preserves_session_when_intent_changes() {
        let (remote_url, server) = observe_push_requests();
        let temp = tempdir().expect("should create active push lock directory");
        let (workspace, config, original_destination) =
            create_http_push_fixture(temp.path(), &remote_url);
        let mut operation = seed_push_session(&config, "main", &original_destination, false);
        let session_id = operation
            .push_session_id
            .expect("should seed the active session UUID");
        operation
            .set_push_destination(
                &config,
                &original_destination,
                TEST_TRANSFER_ID,
                Utc::now().timestamp() + 3600,
            )
            .expect("should seed the published transfer lease");
        operation
            .advance_assets_transfer_checkpoint(&config, &original_destination)
            .expect("should seed the completed asset checkpoint");
        let operation_id = operation.id;
        let before = reload_push_operation(&config, operation_id);

        let graph = get_connection(
            workspace
                .graph_db_path()
                .expect("should resolve changed push graph path"),
        )
        .expect("should reopen changed push graph");
        Collection::create(&graph, "active-push-lock-advanced")
            .expect("should advance branch while lock holder is active");
        let changed_destination = commit_all(&graph, "advance active push destination")
            .expect("should commit changed active push destination");
        drop(graph);

        let holder =
            super::push_lock::tests::LockHolder::spawn(workspace.base_dir(), "origin", "main");
        let result = execute_push(&workspace, None, None, true);
        holder.release();
        let requests = server.join().expect("should finish push request observer");
        let after = reload_push_operation(&config, operation_id);

        assert_ne!(original_destination, changed_destination);
        assert_eq!(before.push_session_id, Some(session_id));
        assert!(
            matches!(result, Err(super::RemotePushError::PushLockBusy { .. })),
            "the contender should report active ownership before retiring the session"
        );
        assert_eq!(
            after, before,
            "a contender changing destination and force must leave the full operation row intact"
        );
        assert!(
            requests.is_empty(),
            "a lock contender must not send capability, lease, or abort requests"
        );
    }

    #[test]
    fn test_active_push_lock_prevents_creation_of_fresh_operation() {
        let (remote_url, server) = observe_push_requests();
        let temp = tempdir().expect("should create fresh push lock directory");
        let (workspace, config, _) = create_http_push_fixture(temp.path(), &remote_url);
        let holder =
            super::push_lock::tests::LockHolder::spawn(workspace.base_dir(), "origin", "main");

        let result = execute_push(&workspace, None, None, false);
        holder.release();
        let requests = server.join().expect("should finish push request observer");

        assert!(
            matches!(result, Err(super::RemotePushError::PushLockBusy { .. })),
            "a contender should be rejected before creating its operation row"
        );
        assert!(
            push_operation_sessions(&config).is_empty(),
            "a rejected push must not create a fresh resumable session"
        );
        assert!(
            requests.is_empty(),
            "a rejected fresh push must not contact the remote"
        );
    }

    #[test]
    fn test_network_capability_failure_then_503_preserves_push_session() {
        let environment_lock = ENVIRONMENT_LOCK
            .lock()
            .expect("should lock process environment");
        let api_key = EnvironmentGuard::set("GENHUB_API_KEY", "push-test-key");
        let unavailable = Some((
            503,
            json!({"code": 503, "message": "capability service unavailable"}).to_string(),
        ));
        let (remote_url, server) = serve_capability_replies(vec![None, unavailable]);
        let temp = tempdir().expect("should create network retry directory");
        let (workspace, config, _) = create_http_push_fixture(temp.path(), &remote_url);

        let first_result = execute_push(&workspace, None, None, false);
        let second_result = execute_push(&workspace, None, None, false);
        let requests = server.join().expect("capability server should finish");
        let sessions = push_operation_sessions(&config);
        drop(api_key);
        drop(environment_lock);

        assert!(
            first_result.is_err(),
            "closed capability connection should fail push"
        );
        assert!(
            second_result.is_err(),
            "503 capability response should fail push"
        );
        assert_eq!(
            requests.len(),
            2,
            "network retry should make a second request"
        );
        assert_eq!(
            request_session_id(&requests[0]),
            request_session_id(&requests[1]),
            "a network or server failure should preserve the resumable session UUID"
        );
        assert_eq!(
            sessions.len(),
            1,
            "retryable failures should share one operation"
        );
        assert!(
            !sessions[0].1,
            "retryable capability failures should remain pending"
        );
    }

    #[test]
    fn test_temporary_initial_push_conflict_preserves_push_session() {
        let environment_lock = ENVIRONMENT_LOCK
            .lock()
            .expect("should lock process environment");
        let api_key = EnvironmentGuard::set("GENHUB_API_KEY", "push-test-key");
        let contention = || {
            Some((
                409,
                json!({"code": 409, "message": "Another push is in progress"}).to_string(),
            ))
        };
        let (remote_url, server) = serve_capability_replies(vec![contention(), contention()]);
        let temp = tempdir().expect("should create contention retry directory");
        let (workspace, config, _) = create_http_push_fixture(temp.path(), &remote_url);

        let first_result = execute_push(&workspace, None, None, false);
        let second_result = execute_push(&workspace, None, None, false);
        let requests = server.join().expect("capability server should finish");
        let sessions = push_operation_sessions(&config);
        drop(api_key);
        drop(environment_lock);

        assert!(
            first_result.is_err(),
            "first temporary conflict should report contention"
        );
        assert!(
            second_result.is_err(),
            "second temporary conflict should report contention"
        );
        assert_eq!(
            requests.len(),
            2,
            "both temporary conflicts should reach GenHub"
        );
        assert_eq!(
            request_session_id(&requests[0]),
            request_session_id(&requests[1]),
            "temporary push contention should keep the same resumable session UUID"
        );
        assert_eq!(
            sessions.len(),
            1,
            "temporary contention should keep one operation"
        );
        assert!(
            !sessions[0].1,
            "temporary contention should not retire the operation"
        );
    }

    #[test]
    fn test_temporary_existing_lease_renewal_conflict_preserves_push_operation() {
        let environment_lock = ENVIRONMENT_LOCK
            .lock()
            .expect("should lock process environment");
        let api_key = EnvironmentGuard::set("GENHUB_API_KEY", "push-test-key");
        let (remote_url, server) = serve_capability_replies(vec![Some((
            409,
            json!({"code": 409, "message": "Another push is in progress"}).to_string(),
        ))]);
        let temp = tempdir().expect("should create lease renewal contention directory");
        let (workspace, config, destination) = create_http_push_fixture(temp.path(), &remote_url);
        let session_id = seed_push_lease(
            &config,
            "main",
            &destination,
            TEST_TRANSFER_ID,
            Utc::now().timestamp() + 3600,
        );

        let result = execute_push(&workspace, None, None, false);
        let requests = server.join().expect("capability server should finish");
        let sessions = push_operation_sessions(&config);
        drop(api_key);
        drop(environment_lock);

        assert!(
            result.is_err(),
            "temporary lease contention should report failure"
        );
        assert_eq!(
            requests.len(),
            1,
            "lease contention should stop before staging"
        );
        assert_eq!(request_session_id(&requests[0]), session_id);
        assert_eq!(sessions.len(), 1, "lease retry should retain its operation");
        assert_eq!(sessions[0].0, Some(session_id));
        assert!(
            !sessions[0].1,
            "temporary 409 during typed lease renewal should not retire the push operation"
        );
    }

    #[test]
    fn test_advanced_push_retry_keeps_its_new_session_until_publication() {
        let environment_lock = ENVIRONMENT_LOCK
            .lock()
            .expect("should lock process environment");
        let api_key = EnvironmentGuard::set("GENHUB_API_KEY", "push-test-key");
        let (remote_url, server) = serve_published_session_retirement();
        let temp = tempdir().expect("should create advanced branch retry directory");
        let (workspace, config, published_destination) =
            create_http_push_fixture(temp.path(), &remote_url);
        // A pending operation with a graph lease models manifest A published before its asset
        // phase finished; advancing to B must keep A's accepted destination until B publishes.
        let published_session = seed_push_lease(
            &config,
            "main",
            &published_destination,
            TEST_TRANSFER_ID,
            Utc::now().timestamp() + 3600,
        );
        let operation_id = RemoteOperationRecord::begin_or_resume(
            &config,
            "origin",
            "main",
            StoredRemoteOperationKind::Push,
            None,
        )
        .expect("should resume the published push operation")
        .id;
        let graph = get_connection(
            workspace
                .graph_db_path()
                .expect("should resolve advanced push graph path"),
        )
        .expect("should reopen advanced push graph");
        Collection::create(&graph, "advanced-push-state")
            .expect("should advance branch after published manifest A");
        let advanced_destination = commit_all(&graph, "advance to unpublished manifest B")
            .expect("should commit unpublished manifest B");
        drop(graph);

        let first_result = execute_push(&workspace, None, None, false);
        let first_state = reload_push_operation(&config, operation_id);
        let second_result = execute_push(&workspace, None, None, false);
        let requests = server.join().expect("capability server should finish");
        let final_state = reload_push_operation(&config, operation_id);
        drop(api_key);
        drop(environment_lock);

        assert!(
            first_result.is_err(),
            "first B attempt should stop before publication"
        );
        assert!(
            second_result.is_err(),
            "second B attempt should stop before publication"
        );
        assert_eq!(
            requests.len(),
            4,
            "retiring published A should be followed by both attempts to publish B; first={first_result:?}, second={second_result:?}, requests={requests:#?}"
        );
        assert_eq!(request_session_id(&requests[0]), published_session);
        assert!(
            requests[1].starts_with("POST /api/abort?capability=mock-signed-value "),
            "the previously published session should be retired through its signed abort endpoint: {}",
            requests[1]
        );
        let (abort_headers, abort_body) = requests[1]
            .split_once("\r\n\r\n")
            .expect("should find previous-session abort request headers");
        assert!(
            abort_headers.lines().any(|line| {
                line.split_once(':').is_some_and(|(name, value)| {
                    name.eq_ignore_ascii_case("content-length") && value.trim() == "0"
                })
            }),
            "old-session abort should explicitly send Content-Length: 0: {abort_headers}"
        );
        assert!(
            abort_body.is_empty(),
            "old-session abort body should be empty"
        );
        let first_b_session = request_session_id(&requests[2]);
        let retry_b_session = request_session_id(&requests[3]);
        assert_ne!(advanced_destination, published_destination);
        assert_ne!(
            first_b_session, published_session,
            "advancing from published A to B should begin a new graph session"
        );
        assert_eq!(
            retry_b_session, first_b_session,
            "retries for the same unpublished B destination should reuse B's session UUID"
        );
        assert_eq!(first_state.to_commit, Some(published_destination));
        assert_eq!(
            first_state.transfer_id, None,
            "retired A lease must not carry into B"
        );
        assert_eq!(first_state.push_session_id, Some(first_b_session));
        assert!(
            first_state.failed_at.is_none(),
            "retryable first B failure should remain pending"
        );
        assert_eq!(final_state.to_commit, Some(published_destination));
        assert_eq!(
            final_state.transfer_id, None,
            "B has not published a replacement lease"
        );
        assert_eq!(final_state.push_session_id, Some(first_b_session));
        assert!(
            final_state.failed_at.is_none(),
            "retryable B retries should remain pending"
        );
    }

    #[test]
    fn test_unpublished_push_session_is_retired_before_destination_changes() {
        let environment_lock = ENVIRONMENT_LOCK
            .lock()
            .expect("should lock process environment");
        let api_key = EnvironmentGuard::set("GENHUB_API_KEY", "push-test-key");
        let unavailable = MockGenHubReply::Http(
            503,
            json!({"code": 503, "message": "capability service unavailable"}).to_string(),
        );
        let (remote_url, server) = serve_capability_replies(vec![
            MockGenHubReply::DirectPushCapability(RETRIED_TRANSFER_ID),
            MockGenHubReply::EmptySuccess,
            unavailable,
        ]);
        let temp = tempdir().expect("should create unpublished session retry directory");
        let (workspace, config, old_destination) =
            create_http_push_fixture(temp.path(), &remote_url);
        let operation = seed_push_session(&config, "main", &old_destination, false);
        let operation_id = operation.id;
        let old_session_id = operation
            .push_session_id
            .expect("should allocate the original push session UUID");
        let graph = get_connection(
            workspace
                .graph_db_path()
                .expect("should resolve unpublished session graph path"),
        )
        .expect("should open unpublished session graph");
        Collection::create(&graph, "unpublished-session-advance")
            .expect("should advance beyond the interrupted graph session");
        let new_destination =
            commit_all(&graph, "advance after interrupted push").expect("should commit new head");
        drop(graph);

        let result = execute_push(&workspace, None, None, false);
        let requests = server.join().expect("GenHub server should finish");
        let state = reload_push_operation(&config, operation_id);
        drop(api_key);
        drop(environment_lock);

        assert!(
            result.is_err(),
            "new session capability failure should surface"
        );
        assert_eq!(requests.len(), 3);
        assert_eq!(request_session_id(&requests[0]), old_session_id);
        assert!(requests[1].starts_with("POST /api/abort?capability=mock-signed-value "));
        let new_session_id = request_session_id(&requests[2]);
        assert_ne!(new_session_id, old_session_id);
        assert_ne!(old_destination, new_destination);
        assert_eq!(state.to_commit, None, "no manifest has been published yet");
        assert_eq!(
            state.transfer_id, None,
            "unpublished sessions have no published lease"
        );
        assert_eq!(state.push_session_target_commit, Some(new_destination));
        assert_eq!(state.push_session_force, Some(false));
        assert_eq!(state.push_session_id, Some(new_session_id));
        assert!(
            state.failed_at.is_none(),
            "retryable new-session failure should stay pending"
        );
    }

    #[test]
    fn test_unpublished_session_cleanup_failure_preserves_old_intent_for_retry() {
        let environment_lock = ENVIRONMENT_LOCK
            .lock()
            .expect("should lock process environment");
        let api_key = EnvironmentGuard::set("GENHUB_API_KEY", "push-test-key");
        let unavailable = || {
            MockGenHubReply::Http(
                503,
                json!({"code": 503, "message": "cleanup temporarily unavailable"}).to_string(),
            )
        };
        let (remote_url, server) = serve_capability_replies(vec![
            MockGenHubReply::DirectPushCapability(RETRIED_TRANSFER_ID),
            unavailable(),
            MockGenHubReply::DirectPushCapability(RETRIED_TRANSFER_ID),
            MockGenHubReply::EmptySuccess,
            unavailable(),
        ]);
        let temp = tempdir().expect("should create cleanup retry directory");
        let (workspace, config, old_destination) =
            create_http_push_fixture(temp.path(), &remote_url);
        let operation = seed_push_session(&config, "main", &old_destination, false);
        let operation_id = operation.id;
        let old_session_id = operation
            .push_session_id
            .expect("should allocate the original push session UUID");
        let graph = get_connection(
            workspace
                .graph_db_path()
                .expect("should resolve cleanup retry graph path"),
        )
        .expect("should open cleanup retry graph");
        Collection::create(&graph, "cleanup-retry-advance")
            .expect("should advance beyond the interrupted graph session");
        let new_destination = commit_all(&graph, "advance after cleanup interruption")
            .expect("should commit new head");
        drop(graph);

        let first_result = execute_push(&workspace, None, None, false);
        let first_state = reload_push_operation(&config, operation_id);
        let second_result = execute_push(&workspace, None, None, false);
        let requests = server.join().expect("GenHub server should finish");
        let final_state = reload_push_operation(&config, operation_id);
        drop(api_key);
        drop(environment_lock);

        assert!(
            first_result.is_err(),
            "first cleanup failure should surface"
        );
        assert!(
            second_result.is_err(),
            "new capability failure should surface"
        );
        assert_eq!(requests.len(), 5);
        assert_eq!(request_session_id(&requests[0]), old_session_id);
        assert_eq!(request_session_id(&requests[2]), old_session_id);
        assert!(requests[1].starts_with("POST /api/abort?capability=mock-signed-value "));
        assert!(requests[3].starts_with("POST /api/abort?capability=mock-signed-value "));
        assert_ne!(request_session_id(&requests[4]), old_session_id);
        assert_eq!(
            first_state.push_session_target_commit,
            Some(old_destination)
        );
        assert_eq!(first_state.push_session_force, Some(false));
        assert_eq!(first_state.push_session_id, Some(old_session_id));
        assert_eq!(first_state.to_commit, None);
        assert_eq!(first_state.transfer_id, None);
        assert!(
            first_state.failed_at.is_none(),
            "transient cleanup failure must stay pending"
        );
        assert_eq!(
            final_state.push_session_target_commit,
            Some(new_destination)
        );
        assert_eq!(final_state.push_session_force, Some(false));
        assert_eq!(
            final_state.push_session_id,
            Some(request_session_id(&requests[4]))
        );
        assert_eq!(final_state.to_commit, None);
        assert_eq!(final_state.transfer_id, None);
        assert!(
            final_state.failed_at.is_none(),
            "retryable capability failure must stay pending"
        );
    }

    #[test]
    fn test_unpublished_force_session_cleanup_uses_its_id_not_older_published_lease() {
        let environment_lock = ENVIRONMENT_LOCK
            .lock()
            .expect("should lock process environment");
        let api_key = EnvironmentGuard::set("GENHUB_API_KEY", "push-test-key");
        let unavailable = MockGenHubReply::Http(
            503,
            json!({"code": 503, "message": "capability service unavailable"}).to_string(),
        );
        let (remote_url, server) = serve_capability_replies(vec![
            MockGenHubReply::DirectPushCapability(RETRIED_TRANSFER_ID),
            MockGenHubReply::EmptySuccess,
            unavailable,
        ]);
        let temp = tempdir().expect("should create force-scoped supersession directory");
        let (workspace, config, published_destination) =
            create_http_push_fixture(temp.path(), &remote_url);
        let published_session_id = seed_push_lease(
            &config,
            "main",
            &published_destination,
            TEST_TRANSFER_ID,
            Utc::now().timestamp() + 3600,
        );
        let mut operation = RemoteOperationRecord::begin_or_resume(
            &config,
            "origin",
            "main",
            StoredRemoteOperationKind::Push,
            None,
        )
        .expect("should resume the pending published push");
        let operation_id = operation.id;
        operation
            .prepare_push_session(&config, &published_destination, true)
            .expect("should persist a force-scoped session for the same commit");
        let unpublished_force_session_id = operation
            .push_session_id
            .expect("should allocate a force-scoped session UUID");
        assert_ne!(unpublished_force_session_id, published_session_id);
        drop(operation);

        let graph = get_connection(
            workspace
                .graph_db_path()
                .expect("should resolve force-scoped graph path"),
        )
        .expect("should open force-scoped graph");
        Collection::create(&graph, "force-scoped-session-advance")
            .expect("should advance beyond the unpublished force-scoped session");
        let next_destination = commit_all(&graph, "advance after force-scoped session")
            .expect("should commit new head");
        drop(graph);

        let result = execute_push(&workspace, None, None, false);
        let requests = server.join().expect("GenHub server should finish");
        let state = reload_push_operation(&config, operation_id);
        drop(api_key);
        drop(environment_lock);

        assert!(
            result.is_err(),
            "new session capability failure should surface"
        );
        assert_eq!(requests.len(), 3);
        assert_eq!(
            request_session_id(&requests[0]),
            unpublished_force_session_id
        );
        let old_scope_body = requests[0]
            .split_once("\r\n\r\n")
            .map(|(_, body)| body)
            .expect("old capability request should have a body");
        assert!(old_scope_body.contains("\"force\":true"));
        assert_ne!(RETRIED_TRANSFER_ID, TEST_TRANSFER_ID);
        assert!(requests[1].starts_with("POST /api/abort?capability=mock-signed-value "));
        let new_session_id = request_session_id(&requests[2]);
        assert_ne!(new_session_id, unpublished_force_session_id);
        assert_eq!(state.to_commit, Some(published_destination));
        assert_eq!(
            state.transfer_id, None,
            "the retired published lease must be cleared"
        );
        assert_eq!(state.push_session_target_commit, Some(next_destination));
        assert_eq!(state.push_session_force, Some(false));
        assert_eq!(state.push_session_id, Some(new_session_id));
        assert!(
            state.failed_at.is_none(),
            "retryable new-session failure should stay pending"
        );
    }

    #[test]
    fn test_force_same_head_retry_requests_current_session_after_lease_retirement() {
        let environment_lock = ENVIRONMENT_LOCK
            .lock()
            .expect("should lock process environment");
        let api_key = EnvironmentGuard::set("GENHUB_API_KEY", "push-test-key");
        let unavailable = json!({
            "code": 503,
            "message": "capability service unavailable",
        })
        .to_string();
        let (remote_url, server) =
            serve_capability_replies(vec![MockGenHubReply::Http(503, unavailable)]);
        let temp = tempdir().expect("should create same-head force retry directory");
        let (workspace, config, published_destination) =
            create_http_push_fixture(temp.path(), &remote_url);
        let published_session_id = seed_push_lease(
            &config,
            "main",
            &published_destination,
            TEST_TRANSFER_ID,
            Utc::now().timestamp() + 3600,
        );
        let mut operation = RemoteOperationRecord::begin_or_resume(
            &config,
            "origin",
            "main",
            StoredRemoteOperationKind::Push,
            None,
        )
        .expect("should resume the published push");
        let operation_id = operation.id;
        operation
            .prepare_push_session(&config, &published_destination, true)
            .expect("should retire the old lease when force scope changes");
        let force_session_id = operation
            .push_session_id
            .expect("should allocate the force-scoped session");
        assert_ne!(force_session_id, published_session_id);
        assert_eq!(operation.to_commit, Some(published_destination));
        assert_eq!(operation.transfer_id, None);
        assert_eq!(operation.transfer_expires_at, None);
        operation
            .prepare_push_session(&config, &published_destination, true)
            .expect("should retain the current session on same-head retry");
        assert_eq!(operation.push_session_id, Some(force_session_id));
        drop(operation);

        let result = execute_push(&workspace, None, None, true);
        let requests = server.join().expect("GenHub server should finish");
        let state = reload_push_operation(&config, operation_id);
        drop(api_key);
        drop(environment_lock);

        assert!(
            result.is_err(),
            "temporary capability failure should surface"
        );
        assert_eq!(requests.len(), 1);
        assert_eq!(request_session_id(&requests[0]), force_session_id);
        let request_body = requests[0]
            .split_once("\r\n\r\n")
            .map(|(_, body)| body)
            .expect("push capability request should have a body");
        assert!(request_body.contains("\"force\":true"));
        assert_eq!(state.to_commit, Some(published_destination));
        assert_eq!(state.transfer_id, None);
        assert_eq!(state.transfer_expires_at, None);
        assert_eq!(
            state.push_session_target_commit,
            Some(published_destination)
        );
        assert_eq!(state.push_session_force, Some(true));
        assert_eq!(state.push_session_id, Some(force_session_id));
        assert!(
            state.completed_at.is_none(),
            "retryable capability failure should keep the operation incomplete"
        );
        assert!(
            state.failed_at.is_none(),
            "retryable capability failure should keep the operation pending"
        );
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
        let server = thread::spawn(move || {
            let mut requests = Vec::new();
            for request_index in 0..7 {
                let (mut stream, _) = listener.accept().expect("should accept GenHub request");
                let mut request = [0_u8; 8192];
                let read = stream
                    .read(&mut request)
                    .expect("should read GenHub request");
                let request = String::from_utf8_lossy(&request[..read]).into_owned();
                requests.push(request.clone());
                match request_index {
                    0 | 3 => {
                        let body = direct_push_capability_body(
                            &request,
                            &format!("http://{address}"),
                            TEST_TRANSFER_ID,
                        );
                        write!(
                            stream,
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                            body.len()
                        )
                        .expect("should write refreshed direct-push capability");
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
                    6 => {
                        let body = "tracking fetch unavailable";
                        write!(
                            stream,
                            "HTTP/1.1 500 Internal Server Error\r\nContent-Type: text/plain\r\nContent-Length: {}\r\n\r\n{body}",
                            body.len()
                        )
                        .expect("should write failed tracking-fetch response");
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
        let session_id = seed_push_lease(
            &config,
            "main",
            &destination_hash,
            TEST_TRANSFER_ID,
            Utc::now().timestamp() + 3600,
        );
        drop(graph);

        execute_push(&workspace, None, None, false)
            .expect_err("first push should retain its lease after completion fails");
        let pending_transfer: (DoltHashId, Uuid, Uuid) = config
            .query_row(
                "SELECT to_commit, transfer_id, push_session_id FROM remote_operations \
                 WHERE operation = 'push' AND completed_at IS NULL AND failed_at IS NULL",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .expect("should retain pending push lease");
        assert_eq!(
            pending_transfer,
            (destination_hash, TEST_TRANSFER_ID, session_id)
        );

        execute_push(&workspace, None, None, false).expect("push retry should complete");
        let requests = server.join().expect("GenHub server should finish");

        assert_eq!(requests.len(), 7);
        assert!(requests[0].starts_with("POST /api/repos/alice/example/remote-capability "));
        assert!(requests[0].contains("\"operation\":\"push\""));
        assert!(requests[1].starts_with("POST /api/repos/alice/example/asset-transfers "));
        assert!(requests[2].starts_with("POST /api/repos/alice/example/asset-transfers/complete "));
        assert!(requests[3].starts_with("POST /api/repos/alice/example/remote-capability "));
        assert!(requests[4].starts_with("POST /api/repos/alice/example/asset-transfers "));
        assert!(requests[5].starts_with("POST /api/repos/alice/example/asset-transfers/complete "));
        for request in [&requests[2], &requests[5]] {
            assert!(request.contains(&format!("\"transfer_id\":\"{TEST_TRANSFER_ID}\"")));
            assert!(request.contains("\"branch\":\"main\""));
            assert!(request.contains("\"assets\":[]"));
        }
        assert_eq!(request_session_id(&requests[0]), session_id);
        assert_eq!(request_session_id(&requests[3]), session_id);
        assert!(requests[6].contains("\"operation\":\"pull\""));
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
    fn test_advanced_push_retry_requires_a_direct_gcs_capability() {
        let _environment_lock = ENVIRONMENT_LOCK
            .lock()
            .expect("should lock process environment");
        let _api_key = EnvironmentGuard::set("GENHUB_API_KEY", "push-test-key");
        let temp = tempdir().expect("should create advanced push retry directory");
        let legacy_capability = format!(
            "{{\"remote_url\":\"http://127.0.0.1:1/legacy-transfer\",\
             \"expires_at\":\"2030-01-01T00:00:00Z\",\
             \"default_branch\":\"main\",\
             \"transfer_id\":\"{RETRIED_TRANSFER_ID}\"}}"
        );
        let (remote_url, server) = serve_capability_replies(vec![
            MockGenHubReply::DirectPushCapability(TEST_TRANSFER_ID),
            MockGenHubReply::Http(200, "{\"assets\":[]}".to_string()),
            MockGenHubReply::Http(500, "asset verification unavailable".to_string()),
            MockGenHubReply::DirectPushCapability(TEST_TRANSFER_ID),
            MockGenHubReply::EmptySuccess,
            MockGenHubReply::Http(200, legacy_capability),
        ]);
        let workspace = Workspace::new(temp.path().join("local"));
        workspace.ensure_gen_dir();
        let config = get_config_connection(Some(workspace.gen_db_path().unwrap()))
            .expect("should open push config");
        let graph = get_connection(workspace.graph_db_path().unwrap()).expect("should open graph");
        Collection::create(&graph, "push-retry-fixture").expect("should create push fixture");
        let original_destination =
            commit_all(&graph, "push retry fixture").expect("should commit push retry fixture");
        Remote::create(&config, "origin", &remote_url).expect("should configure origin");
        Defaults::set_default_remote(&config, Some("origin")).expect("should set default remote");
        let original_session_id = seed_push_lease(
            &config,
            "main",
            &original_destination,
            TEST_TRANSFER_ID,
            Utc::now().timestamp() + 3600,
        );
        drop(graph);

        execute_push(&workspace, None, None, false)
            .expect_err("first push should retain its lease after completion fails");
        let pending_transfer: (DoltHashId, Uuid, Uuid) = config
            .query_row(
                "SELECT to_commit, transfer_id, push_session_id FROM remote_operations \
                 WHERE operation = 'push' AND completed_at IS NULL AND failed_at IS NULL",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .expect("should retain pending push lease");
        assert_eq!(
            pending_transfer,
            (original_destination, TEST_TRANSFER_ID, original_session_id)
        );

        let graph =
            get_connection(workspace.graph_db_path().unwrap()).expect("should reopen graph");
        Collection::create(&graph, "advanced-head").expect("should advance local graph");
        let advanced_destination =
            commit_all(&graph, "advance push retry").expect("should commit advanced local head");
        drop(graph);
        assert_ne!(advanced_destination, original_destination);

        let error = execute_push(&workspace, None, None, false)
            .expect_err("advanced push retry should reject a legacy HTTP capability");
        let requests = server.join().expect("GenHub server should finish");

        assert!(error.to_string().contains("direct GCS session"));
        assert_eq!(requests.len(), 6);
        assert!(requests[0].starts_with("POST /api/repos/alice/example/remote-capability "));
        assert!(requests[1].starts_with("POST /api/repos/alice/example/asset-transfers "));
        assert!(requests[2].starts_with("POST /api/repos/alice/example/asset-transfers/complete "));
        assert!(requests[3].starts_with("POST /api/repos/alice/example/remote-capability "));
        assert_eq!(request_session_id(&requests[3]), original_session_id);
        assert!(requests[4].starts_with("POST /api/abort?capability=mock-signed-value "));
        assert!(requests[5].contains("\"operation\":\"push\""));
        assert!(!requests[5].contains(&original_session_id.to_string()));
        assert!(
            requests[5]
                .to_ascii_lowercase()
                .contains("idempotency-token:")
        );
        let failed_transfer: (DoltHashId, Uuid, bool) = config
            .query_row(
                "SELECT to_commit, push_session_id, failed_at IS NOT NULL \
                 FROM remote_operations WHERE operation = 'push'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .expect("should mark advanced push as failed");
        assert_eq!(failed_transfer.0, original_destination);
        assert_ne!(failed_transfer.1, original_session_id);
        assert_eq!(failed_transfer.1, request_session_id(&requests[5]));
        assert!(failed_transfer.2);
    }

    #[test]
    fn test_expired_push_lease_requires_a_direct_gcs_capability() {
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
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("should accept GenHub request");
            let mut request = [0_u8; 8192];
            let read = stream
                .read(&mut request)
                .expect("should read GenHub request");
            let request = String::from_utf8_lossy(&request[..read]).into_owned();
            let body = format!(
                "{{\"remote_url\":\"http://127.0.0.1:1/legacy-transfer\",\
                 \"expires_at\":\"2030-01-01T00:00:00Z\",\
                 \"default_branch\":\"main\",\
                 \"transfer_id\":\"{RETRIED_TRANSFER_ID}\"}}"
            );
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            )
            .expect("should write legacy capability response");
            request
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
        let original_session_id =
            seed_push_lease(&config, "main", &destination_hash, TEST_TRANSFER_ID, 0);
        drop(graph);

        let error = execute_push(&workspace, None, None, false)
            .expect_err("expired push lease should require a direct GCS capability");
        let request = server.join().expect("GenHub server should finish");

        assert!(error.to_string().contains("direct GCS session"));
        assert!(request.contains("\"operation\":\"push\""));
        assert!(request.to_ascii_lowercase().contains("idempotency-token:"));
        assert!(request.contains(&original_session_id.to_string()));
        assert!(!request.contains("/asset-transfers "));
        let failed_transfer: (DoltHashId, Uuid, bool) = config
            .query_row(
                "SELECT to_commit, push_session_id, failed_at IS NOT NULL \
                 FROM remote_operations WHERE operation = 'push'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .expect("should mark expired push as failed");
        assert_eq!(failed_transfer.0, destination_hash);
        assert_eq!(failed_transfer.1, original_session_id);
        assert!(failed_transfer.2);
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
        let server = thread::spawn(move || {
            let mut requests = Vec::new();
            for request_index in 0..4 {
                let (mut stream, _) = listener.accept().expect("should accept capability request");
                let mut request = [0_u8; 8192];
                let read = stream
                    .read(&mut request)
                    .expect("should read capability request");
                let request = String::from_utf8_lossy(&request[..read]).into_owned();
                requests.push(request.clone());
                match request_index {
                    0 => {
                        let body = direct_push_capability_body(
                            &request,
                            &format!("http://{address}"),
                            TEST_TRANSFER_ID,
                        );
                        write!(
                            stream,
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                            body.len()
                        )
                        .expect("should write direct-push capability");
                    }
                    1 => {
                        let body = "{\"assets\":[]}";
                        write!(
                            stream,
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                            body.len()
                        )
                        .expect("should write asset response");
                    }
                    2 => {
                        stream
                            .write_all(b"HTTP/1.1 204 No Content\r\nContent-Length: 0\r\n\r\n")
                            .expect("should write completion response");
                    }
                    3 => {
                        let body = "tracking fetch unavailable";
                        write!(
                            stream,
                            "HTTP/1.1 500 Internal Server Error\r\nContent-Type: text/plain\r\nContent-Length: {}\r\n\r\n{body}",
                            body.len()
                        )
                        .expect("should write failed response");
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
        Collection::create(&graph, "push-fixture").expect("should create push fixture");
        let destination_hash =
            commit_all(&graph, "push fixture").expect("should commit push fixture");
        Remote::create(
            &config,
            "origin",
            &format!("http://{address}/api/repos/alice/example"),
        )
        .expect("should configure origin");
        Defaults::set_default_remote(&config, Some("origin")).expect("should set default remote");
        seed_push_lease(
            &config,
            "main",
            &destination_hash,
            TEST_TRANSFER_ID,
            Utc::now().timestamp() + 3600,
        );
        drop(graph);
        drop(config);

        execute_push(&workspace, None, None, false)
            .expect("tracking fetch failure should not fail push");
        let requests = server.join().expect("capability server should finish");

        assert_eq!(requests.len(), 4);
        assert!(requests[0].starts_with("POST /api/repos/alice/example/remote-capability "));
        assert!(requests[0].contains("\"operation\":\"push\""));
        assert!(requests[1].starts_with("POST /api/repos/alice/example/asset-transfers "));
        assert!(requests[2].starts_with("POST /api/repos/alice/example/asset-transfers/complete "));
        assert!(requests[2].contains(&format!("\"transfer_id\":\"{TEST_TRANSFER_ID}\"")));
        assert!(requests[2].contains("\"branch\":\"main\""));
        assert!(requests[3].contains("\"operation\":\"pull\""));
        let config = get_config_connection(workspace.gen_db_path().unwrap())
            .expect("should reopen push config");
        let completed_operations: i64 = config
            .query_row(
                "SELECT COUNT(*) FROM remote_operations \
                 WHERE operation = 'push' AND completed_at IS NOT NULL",
                [],
                |row| row.get(0),
            )
            .expect("should count completed push operations");
        assert_eq!(completed_operations, 1);
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

    #[test]
    fn test_non_fast_forward_classifier_requires_exact_sqlite_constraint_error() {
        let exact = SqlError::SqliteFailure(
            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_CONSTRAINT),
            Some("not a fast-forward of the remote branch (use force to overwrite)".to_string()),
        );
        let other_constraint = SqlError::SqliteFailure(
            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_CONSTRAINT),
            Some("constraint failed".to_string()),
        );
        let other_code = SqlError::SqliteFailure(
            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_BUSY),
            Some("not a fast-forward of the remote branch (use force to overwrite)".to_string()),
        );
        let unique_constraint = SqlError::SqliteFailure(
            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_CONSTRAINT_UNIQUE),
            Some("not a fast-forward of the remote branch (use force to overwrite)".to_string()),
        );
        assert!(!super::is_non_fast_forward_push_error(&other_constraint));
        let wrapped_conflict =
            super::remote_push_graph_error(Box::new(super::PushGraphTransferError::NonFastForward));
        let wrapped_unrelated_constraint = super::remote_push_graph_error(Box::new(
            super::PushGraphTransferError::database("running Dolt push", other_constraint),
        ));

        assert!(super::is_non_fast_forward_push_error(&exact));
        assert!(!super::is_non_fast_forward_push_error(&other_code));
        assert!(!super::is_non_fast_forward_push_error(&unique_constraint));
        assert!(matches!(
            wrapped_conflict,
            super::RemotePushError::NonFastForward { .. }
        ));
        assert!(matches!(
            wrapped_unrelated_constraint,
            super::RemotePushError::GraphTransfer(_)
        ));
        assert_eq!(
            super::remote_push_graph_error(Box::new(exact)).to_string(),
            "The branch is not a fast-forward of the remote. Use --force to overwrite the remote."
        );
    }

    #[test]
    fn test_terminal_asset_phase_capability_error_retires_push_operation() {
        let temp = tempdir().expect("should create asset-phase lease failure directory");
        let (_, config, destination_hash) =
            create_http_push_fixture(temp.path(), "https://genhub.bio/api/repos/alice/example");
        let mut operation = RemoteOperationRecord::begin_or_resume(
            &config,
            "origin",
            "main",
            StoredRemoteOperationKind::Push,
            None,
        )
        .expect("should create pending push operation");
        operation
            .prepare_push_session(&config, &destination_hash, false)
            .expect("should persist intended push session before asset phase");

        let error = super::remote_push_asset_error(
            Some(&mut operation),
            &config,
            "main",
            Box::new(super::PushGraphTransferError::CapabilityLeaseInactive),
        );
        let failed_at: bool = config
            .query_row(
                "SELECT failed_at IS NOT NULL FROM remote_operations WHERE operation = 'push'",
                [],
                |row| row.get(0),
            )
            .expect("should inspect terminal push state");

        assert!(matches!(error, super::RemotePushError::GraphTransfer(_)));
        assert!(
            failed_at,
            "terminal asset-phase lease errors must retire the UUID"
        );
    }

    #[cfg(feature = "remote")]
    mod direct_gcs_reads {
        use std::{
            io::Write as _,
            net::TcpListener,
            sync::{
                Arc,
                atomic::{AtomicBool, Ordering},
            },
            thread,
            time::{Duration, Instant},
        };

        use gen_core::config::Workspace;
        use gen_models::{db::GraphConnection, history::dolt::remote_rows, operations::Remote};
        use rusqlite::{
            Connection, Error as SqlError, RemoteFailureKind, blockcachevfs::CloudConnectionUri,
        };
        use serde_json::json;
        use tempfile::tempdir;

        use super::{
            super::{
                CapabilityTransportIdentity, RemoteOperation, clone_into_workspace,
                ensure_graph_remote, fetch, run_graph_transfer,
                validate_capability_transport_identity,
            },
            ENVIRONMENT_LOCK, EnvironmentGuard, TEST_TRANSFER_ID, read_mock_http_request,
            serve_capability_replies,
        };
        use crate::get_config_connection;

        fn read_capability_body(remote_url: &str) -> String {
            json!({
                "remote_url": remote_url,
                "expires_at": "2035-01-01T00:00:00Z",
                "default_branch": "main",
                "transfer_id": TEST_TRANSFER_ID,
            })
            .to_string()
        }

        fn read_capability_uri(bucket: &str, token: &str, endpoint: &str) -> String {
            CloudConnectionUri::gcs(
                bucket,
                "repos/alice/example/.gen/graph_db/",
                "test-project",
                token,
                "default.db",
                Some(endpoint),
            )
            .expect("should build read CAB URI")
            .as_str()
            .to_string()
        }

        fn serve_read_capability_replies(
            replies: Vec<String>,
        ) -> (String, thread::JoinHandle<Vec<String>>) {
            let listener =
                TcpListener::bind("127.0.0.1:0").expect("should bind read capability server");
            listener
                .set_nonblocking(true)
                .expect("should bound read capability server waits");
            let address = listener
                .local_addr()
                .expect("should read read capability server address");
            let server = thread::spawn(move || {
                let mut requests = Vec::with_capacity(replies.len());
                for body in replies {
                    let deadline = Instant::now() + Duration::from_secs(60);
                    let (mut stream, _) = loop {
                        match listener.accept() {
                            Ok(accepted) => break accepted,
                            Err(error)
                                if error.kind() == std::io::ErrorKind::WouldBlock
                                    && Instant::now() < deadline =>
                            {
                                thread::sleep(Duration::from_millis(10));
                            }
                            Err(error) => {
                                panic!(
                                    "should accept read capability request before timeout: {error}"
                                )
                            }
                        }
                    };
                    requests.push(read_mock_http_request(&mut stream));
                    write!(
                        stream,
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    )
                    .expect("should write read capability response");
                }
                requests
            });
            (format!("http://{address}/api/repos/alice/example"), server)
        }

        fn serve_read_gcs_authorization_failures(
            first_token_status: u16,
            renewed_token_status: u16,
        ) -> (String, Arc<AtomicBool>, thread::JoinHandle<Vec<String>>) {
            let listener = TcpListener::bind("127.0.0.1:0").expect("should bind fake GCS server");
            listener
                .set_nonblocking(true)
                .expect("should make fake GCS listener nonblocking");
            let address = listener
                .local_addr()
                .expect("should read fake GCS server address");
            let finished = Arc::new(AtomicBool::new(false));
            let server_finished = Arc::clone(&finished);
            let server = thread::spawn(move || {
                let mut requests = Vec::new();
                let mut last_request = Instant::now();
                let deadline = Instant::now() + Duration::from_secs(60);
                while Instant::now() < deadline {
                    match listener.accept() {
                        Ok((mut stream, _)) => {
                            let request = read_mock_http_request(&mut stream);
                            let request_lowercase = request.to_ascii_lowercase();
                            let status = if request_lowercase.contains("first-") {
                                first_token_status
                            } else {
                                renewed_token_status
                            };
                            let reason = match status {
                                401 => "Unauthorized",
                                403 => "Forbidden",
                                _ => panic!("unexpected fake GCS status {status}"),
                            };
                            let body =
                                format!(r#"{{"error":{{"code":{status},"message":"{reason}"}}}}"#);
                            write!(
                                stream,
                                "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                                body.len()
                            )
                            .expect("should reject fake GCS request");
                            requests.push(request);
                            last_request = Instant::now();
                        }
                        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                            if server_finished.load(Ordering::Acquire)
                                && last_request.elapsed() > Duration::from_millis(100)
                            {
                                break;
                            }
                            thread::sleep(Duration::from_millis(5));
                        }
                        Err(error) => panic!("should accept fake GCS request: {error}"),
                    }
                }
                requests
            });
            (format!("http://{address}/"), finished, server)
        }

        fn assert_gcs_requests_are_read_only(requests: &[String]) {
            assert!(!requests.is_empty(), "should issue a direct GCS request");
            assert!(requests.iter().all(|request| {
                !request.starts_with("POST ")
                    && !request.starts_with("PUT ")
                    && !request.starts_with("DELETE ")
            }));
        }

        #[test]
        fn test_read_capability_scope_allows_token_rotation_but_rejects_scope_changes() {
            let first_uri = CloudConnectionUri::gcs(
                "read-bucket",
                "repos/alice/example/.gen/graph_db/",
                "test-project",
                "first-private-token",
                "default.db",
                None,
            )
            .expect("should build initial read CAB");
            let renewed_uri = CloudConnectionUri::gcs(
                "read-bucket",
                "repos/alice/example/.gen/graph_db/",
                "test-project",
                "renewed-private-token",
                "default.db",
                None,
            )
            .expect("should build renewed read CAB");
            let changed_scope_uri = CloudConnectionUri::gcs(
                "other-bucket",
                "repos/alice/example/.gen/graph_db/",
                "test-project",
                "changed-private-token",
                "default.db",
                None,
            )
            .expect("should build changed-scope CAB");
            let mut identity = CapabilityTransportIdentity::default();

            validate_capability_transport_identity(first_uri.as_str(), &mut identity)
                .expect("should accept the initial read scope");
            validate_capability_transport_identity(renewed_uri.as_str(), &mut identity)
                .expect("should accept a rotated token for the same read scope");
            let changed_scopes = [
                changed_scope_uri,
                CloudConnectionUri::gcs(
                    "read-bucket",
                    "repos/alice/other/.gen/graph_db/",
                    "test-project",
                    "changed-prefix-token",
                    "default.db",
                    None,
                )
                .expect("should build changed-prefix CAB"),
                CloudConnectionUri::gcs(
                    "read-bucket",
                    "repos/alice/example/.gen/graph_db/",
                    "other-project",
                    "changed-project-token",
                    "default.db",
                    None,
                )
                .expect("should build changed-project CAB"),
                CloudConnectionUri::gcs(
                    "read-bucket",
                    "repos/alice/example/.gen/graph_db/",
                    "test-project",
                    "changed-database-token",
                    "other.db",
                    None,
                )
                .expect("should build changed-database CAB"),
                CloudConnectionUri::gcs(
                    "read-bucket",
                    "repos/alice/example/.gen/graph_db/",
                    "test-project",
                    "changed-endpoint-token",
                    "default.db",
                    Some("http://127.0.0.1:4444/"),
                )
                .expect("should build changed-endpoint CAB"),
            ];
            for changed_uri in changed_scopes {
                let error =
                    validate_capability_transport_identity(changed_uri.as_str(), &mut identity)
                        .expect_err("should reject a renewed CAB with changed cloud scope");

                assert!(
                    error
                        .to_string()
                        .contains("changed the graph capability scope")
                );
                assert!(!error.to_string().contains("token"));
            }

            for invalid_url in [
                "https://genhub.bio/api/repos/alice/example/dolt/capability/default.db",
                "file:///tmp/graph/default.db",
            ] {
                let error = validate_capability_transport_identity(invalid_url, &mut identity)
                    .expect_err("should reject a non-cloud capability URI");
                assert!(!error.to_string().contains(invalid_url));
            }
        }

        #[test]
        fn test_cloud_pull_transfer_retries_read_cab_and_restores_canonical_remote() {
            let _environment_lock = ENVIRONMENT_LOCK
                .lock()
                .expect("should lock process environment");
            let _api_key = EnvironmentGuard::set("GENHUB_API_KEY", "read-test-key");
            let (endpoint, stop, gcs_server) = serve_read_gcs_authorization_failures(401, 403);
            let first_uri = read_capability_uri("read-bucket", "first-read-token", &endpoint);
            let renewed_uri = read_capability_uri("read-bucket", "renewed-read-token", &endpoint);
            let (repository_url, capability_server) = serve_read_capability_replies(vec![
                read_capability_body(&first_uri),
                read_capability_body(&renewed_uri),
            ]);
            let graph = GraphConnection(
                Connection::open_in_memory().expect("should open local graph fixture"),
            );
            let remote = Remote {
                name: "origin".to_string(),
                url: repository_url,
            };
            ensure_graph_remote(&graph, &remote.name, &remote.url)
                .expect("should configure canonical graph remote");
            let transfer_result = run_graph_transfer(
                &graph,
                &remote,
                RemoteOperation::Pull,
                "main",
                false,
                || fetch(&graph, &remote.name, Some("main")),
            );
            stop.store(true, Ordering::Release);
            let capability_requests = capability_server
                .join()
                .expect("should finish capability server");
            let gcs_requests = gcs_server.join().expect("should finish fake GCS server");
            let error =
                transfer_result.expect_err("should preserve the read CAB authorization error");
            let gcs_error = error
                .downcast_ref::<SqlError>()
                .expect("should return the typed SQLite transfer error");

            assert_eq!(
                capability_requests.len(),
                2,
                "should refresh the read CAB once"
            );
            assert_gcs_requests_are_read_only(&gcs_requests);
            assert!(gcs_requests.iter().any(|request| {
                request
                    .to_ascii_lowercase()
                    .contains("authorization: bearer first-read-token")
            }));
            assert!(gcs_requests.iter().any(|request| {
                request
                    .to_ascii_lowercase()
                    .contains("authorization: bearer renewed-read-token")
            }));
            assert_eq!(
                gcs_error.remote_failure_kind(),
                Some(RemoteFailureKind::Authorization)
            );
            assert!(!error.to_string().contains("first-read-token"));
            assert!(!error.to_string().contains("renewed-read-token"));
            let graph_remotes = remote_rows(&graph).expect("should read restored graph remotes");
            assert!(graph_remotes.iter().any(|graph_remote| {
                graph_remote.name == remote.name && graph_remote.url == remote.url
            }));
        }

        #[test]
        fn test_clone_retries_direct_gcs_read_cab_without_graph_writes() {
            let _environment_lock = ENVIRONMENT_LOCK
                .lock()
                .expect("should lock process environment");
            let _api_key = EnvironmentGuard::set("GENHUB_API_KEY", "read-test-key");
            let (endpoint, stop, gcs_server) = serve_read_gcs_authorization_failures(401, 403);
            let first_uri = read_capability_uri("read-bucket", "first-clone-token", &endpoint);
            let renewed_uri = read_capability_uri("read-bucket", "renewed-clone-token", &endpoint);
            let (repository_url, capability_server) = serve_read_capability_replies(vec![
                read_capability_body(&first_uri),
                read_capability_body(&renewed_uri),
            ]);
            let temp = tempdir().expect("should create clone fixture");
            let workspace = Workspace::new(temp.path().join("clone"));
            workspace.ensure_gen_dir();
            let config = get_config_connection(Some(workspace.gen_db_path().unwrap()))
                .expect("should open clone config");
            let remote = Remote {
                name: "origin".to_string(),
                url: repository_url,
            };

            let error = clone_into_workspace(&config, &remote, &workspace)
                .expect_err("should fail after retrying unauthorized direct GCS reads");
            stop.store(true, Ordering::Release);
            let capability_requests = capability_server
                .join()
                .expect("should finish capability server");
            let gcs_requests = gcs_server.join().expect("should finish fake GCS server");

            assert_eq!(
                capability_requests.len(),
                2,
                "clone should refresh its read CAB once"
            );
            assert_gcs_requests_are_read_only(&gcs_requests);
            assert!(gcs_requests.iter().any(|request| {
                request
                    .to_ascii_lowercase()
                    .contains("authorization: bearer first-clone-token")
            }));
            assert!(gcs_requests.iter().any(|request| {
                request
                    .to_ascii_lowercase()
                    .contains("authorization: bearer renewed-clone-token")
            }));
            assert!(!error.to_string().contains("first-clone-token"));
            assert!(!error.to_string().contains("renewed-clone-token"));
        }

        #[test]
        fn test_clone_capability_refresh_failure_restores_canonical_remote_url() {
            let _environment_lock = ENVIRONMENT_LOCK
                .lock()
                .expect("should lock process environment");
            let _api_key = EnvironmentGuard::set("GENHUB_API_KEY", "read-test-key");
            let (endpoint, stop, gcs_server) = serve_read_gcs_authorization_failures(403, 403);
            let temporary_uri = read_capability_uri("read-bucket", "first-clone-token", &endpoint);
            let (repository_url, capability_server) = serve_capability_replies(vec![
                Some((200, read_capability_body(&temporary_uri))),
                Some((503, r#"{"code":503,"message":"unavailable"}"#.to_string())),
            ]);
            let temp = tempdir().expect("should create clone fixture");
            let workspace = Workspace::new(temp.path().join("clone"));
            workspace.ensure_gen_dir();
            let config = get_config_connection(Some(workspace.gen_db_path().unwrap()))
                .expect("should open clone config");
            let remote = Remote {
                name: "origin".to_string(),
                url: repository_url,
            };
            let graph = GraphConnection(
                Connection::open(
                    workspace
                        .graph_db_path()
                        .expect("should resolve failed clone graph path"),
                )
                .expect("should open failed clone graph fixture"),
            );
            ensure_graph_remote(&graph, &remote.name, "file:///temporary/clone.db")
                .expect("should seed a temporary graph remote");
            drop(graph);

            let error = clone_into_workspace(&config, &remote, &workspace)
                .expect_err("should return a failed read capability refresh");
            stop.store(true, Ordering::Release);
            let capability_requests = capability_server
                .join()
                .expect("should finish capability server");
            let gcs_requests = gcs_server.join().expect("should finish fake GCS server");
            let graph = GraphConnection(
                Connection::open(
                    workspace
                        .graph_db_path()
                        .expect("should resolve failed clone graph path"),
                )
                .expect("should reopen failed clone graph"),
            );
            let graph_remotes = remote_rows(&graph).expect("should read graph remotes");

            assert_eq!(
                capability_requests.len(),
                2,
                "clone should request a refreshed capability after an authorization failure"
            );
            assert_gcs_requests_are_read_only(&gcs_requests);
            assert!(graph_remotes.iter().any(|graph_remote| {
                graph_remote.name == remote.name && graph_remote.url == remote.url
            }));
            assert!(
                !graph_remotes
                    .iter()
                    .any(|graph_remote| graph_remote.url.starts_with("gcs://"))
            );
            assert!(!error.to_string().contains("first-clone-token"));
        }

        #[test]
        fn test_read_capability_failure_restores_canonical_url_after_refresh_error() {
            let _environment_lock = ENVIRONMENT_LOCK
                .lock()
                .expect("should lock process environment");
            let _api_key = EnvironmentGuard::set("GENHUB_API_KEY", "read-test-key");
            let (endpoint, stop, gcs_server) = serve_read_gcs_authorization_failures(403, 403);
            let first_uri = read_capability_uri("read-bucket", "first-read-token", &endpoint);
            let (repository_url, capability_server) = serve_capability_replies(vec![
                Some((200, read_capability_body(&first_uri))),
                Some((503, r#"{"code":503,"message":"unavailable"}"#.to_string())),
            ]);
            let graph = GraphConnection(
                Connection::open_in_memory().expect("should open local graph fixture"),
            );
            let remote = Remote {
                name: "origin".to_string(),
                url: repository_url,
            };
            ensure_graph_remote(&graph, &remote.name, &remote.url)
                .expect("should seed the canonical graph remote");
            let error = run_graph_transfer(
                &graph,
                &remote,
                RemoteOperation::Pull,
                "main",
                false,
                || fetch(&graph, &remote.name, Some("main")),
            )
            .expect_err("should return the read capability refresh failure");
            stop.store(true, Ordering::Release);
            let capability_requests = capability_server
                .join()
                .expect("should finish capability server");
            let gcs_requests = gcs_server.join().expect("should finish fake GCS server");
            let graph_remotes = remote_rows(&graph).expect("should read graph remotes");

            assert_eq!(
                capability_requests.len(),
                2,
                "should attempt one capability refresh"
            );
            assert_gcs_requests_are_read_only(&gcs_requests);
            assert!(gcs_requests.iter().any(|request| {
                request
                    .to_ascii_lowercase()
                    .contains("authorization: bearer first-read-token")
            }));
            assert!(
                graph_remotes
                    .iter()
                    .any(|graph_remote| graph_remote.name == "origin"
                        && graph_remote.url == remote.url),
                "failed refresh should leave the canonical repository URL configured"
            );
            assert!(!error.to_string().contains("first-read-token"));
        }
    }
}
