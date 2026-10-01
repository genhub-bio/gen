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
//! directly between the workspaces. For a GenHub remote, clone and pull use a scoped,
//! short-lived HTTP capability as the graph database's Dolt remote. Push instead opens a
//! direct GCS database URI behind an in-process loopback `RemoteServer`, stages the Dolt push,
//! and asks GenHub to publish the accepted manifest. An idempotency token persisted with the
//! push operation lets an interrupted push resume its staged session. Failure to restore a
//! temporary clone or pull URL is reported as a warning because the graph transfer may already
//! have succeeded and the canonical URL will be restored on the next attempt.
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
    fs::{self, OpenOptions},
    io::{self, BufReader, Read as _, Write as _},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Instant,
};

use base64::{Engine as _, engine::general_purpose};
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use crc32c::crc32c_append;
use gen_core::{
    DoltHashId, HashId, Sha256Hash,
    config::{DEFAULT_GRAPH_DB_NAME, Workspace},
    errors::{ConfigError, ConnectionError},
};
use gen_models::{
    assets::{AssetRef, AssetView, LocalAssetUri, materialization_destination_path},
    db::{ConfigConnection, GraphConnection},
    errors::{QueryError, RemoteError as ModelRemoteError},
    history::dolt::{
        active_branch, add_remote, branch_hash, checkout, clone_remote, fetch, hash_of, pull, push,
        push_force, remote_rows, set_remote_url,
    },
    operations::{
        Defaults, Remote, RemoteBranch, RemoteOperationKind as StoredRemoteOperationKind,
        RemoteOperationRecord, calculate_file_checksum,
    },
};
use indexmap::IndexMap;
use md5::Md5;
use reqwest::{
    StatusCode,
    blocking::{Body, Client, Response},
    header::{CONTENT_RANGE, RANGE},
};
use rusqlite::{
    BlockCacheSessionOptions, Error as SqlError, RemoteServer, RemoteServerOptions,
    SessionOperationId, SessionOperationStatus, SessionScope,
    blockcachevfs::{AuthError, AuthRefreshReason},
};
use sha2::{Digest as _, Sha256};
use url::Url;
use uuid::Uuid;

use crate::{
    commands::remote::{
        client::{
            AssetTransferCompletionRequest, AssetTransferRequest, AssetUploadReceipt,
            CapabilityRequest, CapabilityResponse, DirectPushCapability, RemoteClientError,
            RemoteOperation, RepositoryRemote, SessionScope as ClientSessionScope,
            acquire_asset_transfers, acquire_capability, acquire_push_capability,
            complete_asset_transfers, confirm_direct_push_session_stale, publish_direct_push,
        },
        login_origin,
        progress::{
            AssetUploadProgressReporter, GraphUploadProgressReporter, ProgressHeartbeat,
            UploadBodyReader, format_elapsed, write_progress_line,
        },
        server::AuthTokens,
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
    remote_url: Option<String>,
    push_lease: Option<PushTransferLease>,
    direct_push: Option<DirectPushCapability>,
    capability_response: Option<CapabilityResponse>,
}

const PUSH_AUTHORIZATION_REFRESH_MARGIN_SECONDS: i64 = 60;

type PushCapabilityFetcher =
    dyn Fn() -> Result<CapabilityResponse, RemoteClientError> + Send + Sync + 'static;

struct PushCapabilityState {
    response: CapabilityResponse,
    issued_at: DateTime<Utc>,
}

enum CapabilityRefreshFailure {
    Fetch(RemoteClientError),
    Expired,
    InvalidScope,
}

#[derive(Clone, Eq, PartialEq)]
struct GcsUriIdentity {
    bucket: String,
    path: String,
    port: Option<u16>,
    query: Vec<(String, String)>,
    scheme: String,
}

#[derive(Clone)]
struct PushCapabilityRenewal {
    expected_publish_origin: String,
    expected_scope: ClientSessionScope,
    expected_session_id: Uuid,
    expected_transfer_id: Uuid,
    expected_uri_identity: GcsUriIdentity,
    fetch: Arc<PushCapabilityFetcher>,
    state: Arc<Mutex<PushCapabilityState>>,
}

impl PushCapabilityRenewal {
    fn new(
        response: CapabilityResponse,
        expected_transfer_id: Uuid,
        expected_session_id: Uuid,
        publish_origin: &str,
        fetch: Arc<PushCapabilityFetcher>,
    ) -> Result<Self, PushGraphTransferError> {
        if response.transfer_id != expected_transfer_id || response.remote_url.is_some() {
            return Err(PushGraphTransferError::Protocol(
                "GenHub returned a direct-push capability for a different transfer".to_string(),
            ));
        }
        let direct_push = response.direct_push.as_ref().ok_or_else(|| {
            PushGraphTransferError::Protocol(
                "GenHub push capability did not include a direct GCS session".to_string(),
            )
        })?;
        if direct_push.session_id != expected_session_id {
            return Err(PushGraphTransferError::Protocol(
                "GenHub returned a direct-push session that does not match the idempotency token"
                    .to_string(),
            ));
        }
        let (expected_uri_identity, _) = gcs_uri_identity_and_token(&direct_push.database_uri)
            .map_err(|_| {
                PushGraphTransferError::Protocol(
                    "GenHub returned an invalid direct GCS database URI".to_string(),
                )
            })?;
        if !same_http_origin(publish_origin, &direct_push.publish_url) {
            return Err(PushGraphTransferError::Protocol(
                "GenHub returned a direct-push publication URL outside the repository origin"
                    .to_string(),
            ));
        }
        let expected_scope = direct_push.session_scope.clone();
        let renewal = Self {
            expected_publish_origin: publish_origin.to_string(),
            expected_scope,
            expected_session_id,
            expected_transfer_id,
            expected_uri_identity,
            fetch,
            state: Arc::new(Mutex::new(PushCapabilityState {
                response,
                issued_at: Utc::now(),
            })),
        };
        renewal
            .auth_token_internal(AuthRefreshReason::Request)
            .map_err(PushCapabilityRenewal::transfer_error)?;
        Ok(renewal)
    }

    fn auth_token(&self, reason: AuthRefreshReason) -> Result<String, AuthError> {
        self.auth_token_internal(reason)
            .map_err(|_| AuthError("GenHub rejected direct-push credential renewal".to_string()))
    }

    fn auth_token_internal(
        &self,
        reason: AuthRefreshReason,
    ) -> Result<String, CapabilityRefreshFailure> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| CapabilityRefreshFailure::InvalidScope)?;
        let should_refresh = matches!(reason, AuthRefreshReason::Unauthorized)
            || capability_needs_refresh(&state.response, state.issued_at, Utc::now());
        if should_refresh {
            let response = (self.fetch)().map_err(CapabilityRefreshFailure::Fetch)?;
            self.validate_renewal(&response)?;
            state.response = response;
            state.issued_at = Utc::now();
        }
        direct_push_access_token(&state.response)
            .map_err(|_| CapabilityRefreshFailure::InvalidScope)
    }

    fn validate_renewal(
        &self,
        response: &CapabilityResponse,
    ) -> Result<(), CapabilityRefreshFailure> {
        self.validate_session_identity(response)?;
        let direct_push = response
            .direct_push
            .as_ref()
            .ok_or(CapabilityRefreshFailure::InvalidScope)?;
        if response.expires_at <= Utc::now() || direct_push.token_expires_at <= Utc::now() {
            return Err(CapabilityRefreshFailure::Expired);
        }
        Ok(())
    }

    fn validate_session_identity(
        &self,
        response: &CapabilityResponse,
    ) -> Result<(), CapabilityRefreshFailure> {
        let direct_push = response
            .direct_push
            .as_ref()
            .ok_or(CapabilityRefreshFailure::InvalidScope)?;
        let uri_identity_matches = gcs_uri_identity_and_token(&direct_push.database_uri)
            .is_ok_and(|(identity, _)| identity == self.expected_uri_identity);
        if response.remote_url.is_some()
            || response.transfer_id != self.expected_transfer_id
            || direct_push.session_id != self.expected_session_id
            || direct_push.session_scope != self.expected_scope
            || !uri_identity_matches
            || !same_http_origin(&self.expected_publish_origin, &direct_push.publish_url)
        {
            return Err(CapabilityRefreshFailure::InvalidScope);
        }
        Ok(())
    }

    fn capability_response(&self) -> Result<CapabilityResponse, PushGraphTransferError> {
        let state = self.state.lock().map_err(|_| {
            PushGraphTransferError::Protocol(
                "direct-push credential state is unavailable".to_string(),
            )
        })?;
        Ok(state.response.clone())
    }

    fn push_lease(&self) -> Result<PushTransferLease, PushGraphTransferError> {
        let response = self.capability_response()?;
        Ok(PushTransferLease {
            transfer_id: response.transfer_id,
            expires_at: response.expires_at.timestamp(),
        })
    }

    fn ensure_fresh(&self) -> Result<PushTransferLease, PushGraphTransferError> {
        self.auth_token_internal(AuthRefreshReason::Request)
            .map_err(PushCapabilityRenewal::transfer_error)?;
        self.push_lease()
    }

    fn publish(&self) -> Result<(), PushGraphTransferError> {
        self.ensure_fresh()?;
        let response = self.capability_response()?;
        let direct_push = response.direct_push.as_ref().ok_or_else(|| {
            PushGraphTransferError::Protocol(
                "GenHub push capability did not include a direct GCS session".to_string(),
            )
        })?;
        publish_direct_push(direct_push).map_err(PushGraphTransferError::Client)
    }

    fn confirm_stale_session(&self) -> Result<bool, PushGraphTransferError> {
        let response = self.capability_response()?;
        let direct_push = response.direct_push.as_ref().ok_or_else(|| {
            PushGraphTransferError::Protocol(
                "GenHub push capability did not include a direct GCS session".to_string(),
            )
        })?;
        confirm_direct_push_session_stale(direct_push).map_err(PushGraphTransferError::Client)
    }
}

impl PushCapabilityRenewal {
    fn transfer_error(error: CapabilityRefreshFailure) -> PushGraphTransferError {
        match error {
            CapabilityRefreshFailure::Fetch(error) => {
                PushGraphTransferError::Client(sanitize_capability_fetch_error(error))
            }
            CapabilityRefreshFailure::Expired => {
                PushGraphTransferError::Client(RemoteClientError::AuthenticationRequired)
            }
            CapabilityRefreshFailure::InvalidScope => PushGraphTransferError::Protocol(
                "GenHub changed the scope of the direct-push capability".to_string(),
            ),
        }
    }
}

fn sanitize_capability_fetch_error(error: RemoteClientError) -> RemoteClientError {
    match error {
        RemoteClientError::InvalidRepositoryUrl(_) => {
            RemoteClientError::InvalidRepositoryUrl("configured GenHub repository".to_string())
        }
        RemoteClientError::AuthenticationRequired => RemoteClientError::AuthenticationRequired,
        RemoteClientError::StaleGraphSession => RemoteClientError::StaleGraphSession,
        RemoteClientError::Http { status, .. } => RemoteClientError::Http {
            status,
            message: "direct-push credential renewal failed".to_string(),
        },
        RemoteClientError::ResponseDecode {
            endpoint,
            status,
            declared_content_length,
            source,
        } => RemoteClientError::ResponseDecode {
            endpoint,
            status,
            declared_content_length,
            source: source.without_url(),
        },
        RemoteClientError::Request(source) => RemoteClientError::Request(source.without_url()),
        RemoteClientError::TokenStorage(_) => RemoteClientError::TokenStorage(io::Error::other(
            "direct-push credential storage failed",
        )),
    }
}

fn capability_needs_refresh(
    response: &CapabilityResponse,
    issued_at: DateTime<Utc>,
    now: DateTime<Utc>,
) -> bool {
    let Some(direct_push) = response.direct_push.as_ref() else {
        return true;
    };
    let expires_at = if response.expires_at <= direct_push.token_expires_at {
        response.expires_at
    } else {
        direct_push.token_expires_at
    };
    let lifetime_milliseconds = expires_at
        .signed_duration_since(issued_at)
        .num_milliseconds()
        .max(0);
    let margin_milliseconds =
        (lifetime_milliseconds / 10).min(PUSH_AUTHORIZATION_REFRESH_MARGIN_SECONDS * 1_000);
    now >= expires_at - ChronoDuration::milliseconds(margin_milliseconds)
}

fn direct_push_access_token(response: &CapabilityResponse) -> Result<String, AuthError> {
    let direct_push = response
        .direct_push
        .as_ref()
        .ok_or_else(|| AuthError("GenHub did not return a direct-push capability".to_string()))?;
    gcs_uri_identity_and_token(&direct_push.database_uri)
        .map(|(_, token)| token)
        .map_err(|_| AuthError("GenHub returned an invalid direct GCS database URI".to_string()))
}

fn gcs_uri_identity_and_token(uri: &str) -> Result<(GcsUriIdentity, String), ()> {
    let parsed = Url::parse(uri).map_err(|_| ())?;
    if parsed.scheme() != "gcs"
        || parsed.host_str().is_none()
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.fragment().is_some()
    {
        return Err(());
    }
    let mut access_token = None;
    let mut query = Vec::new();
    for (name, value) in parsed.query_pairs() {
        if name == "access_token" {
            if access_token.is_some() || value.is_empty() {
                return Err(());
            }
            access_token = Some(value.into_owned());
        } else {
            query.push((name.into_owned(), value.into_owned()));
        }
    }
    let access_token = access_token.ok_or(())?;
    query.sort_unstable();
    let identity = GcsUriIdentity {
        bucket: parsed.host_str().ok_or(())?.to_string(),
        path: parsed.path().to_string(),
        port: parsed.port(),
        query,
        scheme: parsed.scheme().to_string(),
    };
    Ok((identity, access_token))
}

fn same_http_origin(expected_origin: &str, candidate_url: &str) -> bool {
    let Ok(expected) = Url::parse(expected_origin) else {
        return false;
    };
    let Ok(candidate) = Url::parse(candidate_url) else {
        return false;
    };
    matches!(candidate.scheme(), "http" | "https")
        && candidate.username().is_empty()
        && candidate.password().is_none()
        && candidate.fragment().is_none()
        && expected.origin() == candidate.origin()
}

fn transfer_authorization(
    remote: &Remote,
    operation: RemoteOperation,
    branch: Option<&str>,
    force: bool,
    idempotency_token: Option<Uuid>,
    interactive_login: impl FnOnce(&str) -> Result<AuthTokens, Box<dyn Error>>,
) -> Result<GraphTransferAuthorization, Box<dyn Error>> {
    if remote.url.starts_with("file://") {
        return Ok(GraphTransferAuthorization {
            remote_url: Some(file_graph_url(&remote.url)?),
            push_lease: None,
            direct_push: None,
            capability_response: None,
        });
    }
    let repository = RepositoryRemote::parse(&remote.url)?;
    let request = CapabilityRequest {
        operation,
        branch,
        force,
    };
    let capability = if operation == RemoteOperation::Push {
        let idempotency_token = idempotency_token
            .ok_or("GenHub push capabilities require a persisted idempotency token")?;
        acquire_push_capability(&repository, &request, idempotency_token, interactive_login)?
    } else {
        acquire_capability(&repository, &request, interactive_login)?
    };
    let push_lease = (operation == RemoteOperation::Push).then_some(PushTransferLease {
        transfer_id: capability.transfer_id,
        expires_at: capability.expires_at.timestamp(),
    });
    let direct_push = capability.direct_push.clone();
    Ok(GraphTransferAuthorization {
        remote_url: capability.remote_url.clone(),
        push_lease,
        direct_push,
        capability_response: Some(capability),
    })
}

fn push_capability_renewal(
    remote: &Remote,
    branch: &str,
    force: bool,
    session_id: Uuid,
    expected_transfer_id: Uuid,
    response: CapabilityResponse,
) -> Result<PushCapabilityRenewal, PushGraphTransferError> {
    let repository =
        RepositoryRemote::parse(&remote.url).map_err(PushGraphTransferError::Client)?;
    let refresh_repository = repository.clone();
    let refresh_branch = branch.to_string();
    let fetch: Arc<PushCapabilityFetcher> = Arc::new(move || {
        let request = CapabilityRequest {
            operation: RemoteOperation::Push,
            branch: Some(&refresh_branch),
            force,
        };
        acquire_push_capability(&refresh_repository, &request, session_id, |_| {
            Err("interactive login is unavailable during a push".into())
        })
    });
    PushCapabilityRenewal::new(
        response,
        expected_transfer_id,
        session_id,
        repository.origin(),
        fetch,
    )
}

fn acquire_existing_push_renewal(
    remote: &Remote,
    branch: &str,
    force: bool,
    session_id: Uuid,
    expected_transfer_id: Uuid,
) -> Result<PushCapabilityRenewal, PushGraphTransferError> {
    let repository =
        RepositoryRemote::parse(&remote.url).map_err(PushGraphTransferError::Client)?;
    let request = CapabilityRequest {
        operation: RemoteOperation::Push,
        branch: Some(branch),
        force,
    };
    let response = acquire_push_capability(&repository, &request, session_id, login_origin)
        .map_err(PushGraphTransferError::Client)?;
    push_capability_renewal(
        remote,
        branch,
        force,
        session_id,
        expected_transfer_id,
        response,
    )
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
        SqlError::SqliteFailure(code, _)
            if code.extended_code == rusqlite::ffi::SQLITE_AUTH
                || code.extended_code == rusqlite::ffi::SQLITE_IOERR_AUTH
    )
}

fn is_stale_accepted_session_error(error: &SqlError) -> bool {
    matches!(
        error,
        SqlError::SqliteFailure(code, Some(message))
            if code.extended_code == rusqlite::ffi::SQLITE_BUSY
                && message == "published manifest changed since the accepted session head"
    )
}

fn is_non_fast_forward_push_error(error: &SqlError) -> bool {
    matches!(
        error,
        SqlError::SqliteFailure(code, Some(message))
            if code.extended_code == rusqlite::ffi::SQLITE_CONSTRAINT
                && message == "not a fast-forward of the remote branch (use force to overwrite)"
    )
}

#[derive(Debug, thiserror::Error)]
enum PushGraphTransferError {
    #[error("{phase} failed: {source:?}")]
    Database {
        phase: &'static str,
        #[source]
        source: SqlError,
    },
    #[error(transparent)]
    Client(#[from] RemoteClientError),
    #[error("GenHub confirmed the direct GCS graph session is stale")]
    StaleSessionConfirmed,
    #[error("The branch is not a fast-forward of the remote. Use --force to overwrite the remote.")]
    NonFastForward,
    #[error("{primary}; GenHub could not verify the stale session: {confirmation}")]
    StaleSessionCheckFailed {
        primary: Box<PushGraphTransferError>,
        confirmation: Box<PushGraphTransferError>,
    },
    #[error("{0}")]
    Protocol(String),
}

impl PushGraphTransferError {
    fn database(phase: &'static str, source: SqlError) -> Self {
        Self::Database { phase, source }
    }

    fn is_authorization_error(&self) -> bool {
        match self {
            Self::Database { source, .. } => is_authorization_error(source),
            Self::Client(RemoteClientError::Http { status, .. }) => {
                matches!(*status, StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN)
            }
            Self::Client(_)
            | Self::StaleSessionConfirmed
            | Self::NonFastForward
            | Self::StaleSessionCheckFailed { .. }
            | Self::Protocol(_) => false,
        }
    }

    fn is_terminal(&self) -> bool {
        match self {
            Self::Client(RemoteClientError::Http { status, .. }) => *status == StatusCode::CONFLICT,
            Self::Client(RemoteClientError::StaleGraphSession)
            | Self::StaleSessionConfirmed
            | Self::StaleSessionCheckFailed { .. } => false,
            Self::Protocol(_) => true,
            Self::Database { .. } | Self::Client(_) | Self::NonFastForward => false,
        }
    }

    fn is_stale_session(&self) -> bool {
        match self {
            Self::Database { source, .. } => is_stale_accepted_session_error(source),
            Self::Client(RemoteClientError::StaleGraphSession) => true,
            Self::Client(_)
            | Self::StaleSessionConfirmed
            | Self::NonFastForward
            | Self::StaleSessionCheckFailed { .. }
            | Self::Protocol(_) => false,
        }
    }

    fn is_non_fast_forward(&self) -> bool {
        matches!(self, Self::NonFastForward)
    }
}

fn close_direct_push_server(
    mut server: RemoteServer,
    progress: &GraphUploadProgressReporter,
    preserve_phase: bool,
) -> Result<(), PushGraphTransferError> {
    let initial_storage_failure = server.first_storage_error();
    if preserve_phase && let Some(failure) = initial_storage_failure {
        write_progress_line(&format!("Direct GCS storage diagnostic: {failure}"));
    }
    let _heartbeat = if preserve_phase {
        progress.heartbeat_preserving_phase("closing the direct GCS graph session")
    } else {
        progress.heartbeat("closing the direct GCS graph session")
    };
    let quiesce_result = server
        .quiesce()
        .map_err(|error| PushGraphTransferError::database("quiescing direct GCS server", error));
    let final_storage_failure = server.first_storage_error();
    let close_result = server
        .close()
        .map_err(|error| PushGraphTransferError::database("closing direct GCS server", error));
    let cleanup_failed = quiesce_result.is_err() || close_result.is_err();
    if let Some(failure) = final_storage_failure
        && ((preserve_phase && initial_storage_failure != Some(failure))
            || (!preserve_phase && cleanup_failed))
    {
        write_progress_line(&format!("Direct GCS storage diagnostic: {failure}"));
    }
    match quiesce_result {
        Err(quiesce_error) => {
            if let Err(close_error) = close_result {
                write_progress_line(&format!(
                    "Direct GCS server close also failed during {}; preserving the quiesce error.",
                    safe_graph_error_context(&close_error),
                ));
            }
            Err(quiesce_error)
        }
        Ok(()) => close_result,
    }
}

fn preserve_graph_error_after_close(
    primary_error: PushGraphTransferError,
    close_result: Result<(), PushGraphTransferError>,
) -> PushGraphTransferError {
    if let Err(close_error) = close_result {
        write_progress_line(&format!(
            "Direct GCS server cleanup also failed during {}; returning the original graph-transfer error.",
            safe_graph_error_context(&close_error),
        ));
    }
    primary_error
}

fn safe_graph_error_context(error: &PushGraphTransferError) -> String {
    match error {
        PushGraphTransferError::Database {
            phase,
            source: SqlError::SqliteFailure(code, _),
        } => format!("{phase} (SQLite code {})", code.extended_code),
        PushGraphTransferError::Database { phase, .. } => {
            format!("{phase} (database error)")
        }
        PushGraphTransferError::Client(RemoteClientError::Http { status, .. }) => {
            format!("HTTP {status}")
        }
        PushGraphTransferError::Client(_) => "remote client request".to_string(),
        PushGraphTransferError::StaleSessionConfirmed
        | PushGraphTransferError::StaleSessionCheckFailed { .. } => {
            "stale direct GCS session recovery".to_string()
        }
        PushGraphTransferError::NonFastForward => {
            "Dolt rejected a non-fast-forward branch update".to_string()
        }
        PushGraphTransferError::Protocol(_) => "remote protocol response".to_string(),
    }
}

fn close_server_after_graph_failure(
    server: RemoteServer,
    progress: &GraphUploadProgressReporter,
    primary_error: PushGraphTransferError,
) -> PushGraphTransferError {
    progress.waiting_for_local_server_cleanup();
    preserve_graph_error_after_close(
        primary_error,
        close_direct_push_server(server, progress, true),
    )
}

struct DirectGraphPushRequest<'a> {
    graph: &'a GraphConnection,
    remote: &'a Remote,
    branch: &'a str,
    force: bool,
    destination_hash: &'a DoltHashId,
    renewal: &'a PushCapabilityRenewal,
    attempt: usize,
}

fn push_graph_through_direct_session(
    graph: &GraphConnection,
    remote: &Remote,
    branch: &str,
    force: bool,
    destination_hash: &DoltHashId,
    renewal: &PushCapabilityRenewal,
    attempt: usize,
) -> Result<(), PushGraphTransferError> {
    let graph_progress = GraphUploadProgressReporter::new(attempt, 2);
    let request = DirectGraphPushRequest {
        graph,
        remote,
        branch,
        force,
        destination_hash,
        renewal,
        attempt,
    };
    let result = push_graph_through_direct_session_inner(request, &graph_progress);
    if let Err(error) = &result {
        if error.is_non_fast_forward() {
            graph_progress.failed_silently();
        } else {
            graph_progress.failed();
        }
    }
    result
}

fn push_graph_through_direct_session_inner(
    request: DirectGraphPushRequest<'_>,
    graph_progress: &GraphUploadProgressReporter,
) -> Result<(), PushGraphTransferError> {
    let DirectGraphPushRequest {
        graph,
        remote,
        branch,
        force,
        destination_hash,
        renewal,
        attempt,
    } = request;
    let response = {
        let _heartbeat = graph_progress.heartbeat("fetching direct-push session capability");
        renewal.capability_response()?
    };
    let capability = response.direct_push.as_ref().ok_or_else(|| {
        PushGraphTransferError::Protocol(
            "GenHub push capability did not include a direct GCS session".to_string(),
        )
    })?;
    if capability.session_id != renewal.expected_session_id {
        return Err(PushGraphTransferError::Protocol(
            "GenHub returned a direct-push session that does not match the idempotency token"
                .to_string(),
        ));
    }
    let scope = SessionScope::new(
        &capability.session_scope.principal,
        &capability.session_scope.target_database,
        &capability.session_scope.operations,
    )
    .map_err(|error| PushGraphTransferError::database("creating session scope", error))?;
    let operation_id = SessionOperationId::from_request("POST", "/default.db/commit", b"")
        .map_err(|error| {
            PushGraphTransferError::database("creating session operation ID", error)
        })?;
    write_progress_line(&format!(
        "Opening direct GCS graph session (attempt {attempt}/2)..."
    ));
    let renewal_callback = renewal.clone();
    let graph_progress_callback = graph_progress.clone();
    let session =
        BlockCacheSessionOptions::for_uri(capability.session_id.to_string(), scope, operation_id)
            .map_err(|error| PushGraphTransferError::database("attaching GCS session", error))?
            .auth_callback(move |_storage, _account, _container, reason| {
                renewal_callback.auth_token(reason)
            })
            .upload_progress_callback(move |progress| graph_progress_callback.report(progress));
    let options = RemoteServerOptions::new().blockcache_session(session);
    let mut local_server = {
        let _heartbeat = graph_progress.heartbeat("opening the direct GCS graph database");
        RemoteServer::start_with_options(Path::new(&capability.database_uri), &options)
            .map_err(|error| PushGraphTransferError::database("opening direct GCS server", error))?
    };
    let operation_status_result = {
        let _heartbeat = graph_progress.heartbeat("checking direct-push session status");
        local_server.operation_status()
    };
    let operation_status = match operation_status_result {
        Ok(operation_status) => operation_status,
        Err(error) => {
            return Err(close_server_after_graph_failure(
                local_server,
                graph_progress,
                PushGraphTransferError::database("reading direct-push session status", error),
            ));
        }
    };
    match operation_status {
        SessionOperationStatus::Accepted | SessionOperationStatus::Committed => {
            write_progress_line(
                "Reusing accepted graph checkpoint; verifying the staged branch...",
            );
            let validation = {
                let _heartbeat = graph_progress.heartbeat("validating the staged graph branch");
                validate_direct_session_branch(&local_server, branch, destination_hash)
            };
            if let Err(error) = validation {
                graph_progress.finish();
                return Err(close_server_after_graph_failure(
                    local_server,
                    graph_progress,
                    error,
                ));
            }
            graph_progress.finish();
            close_direct_push_server(local_server, graph_progress, false)?;
            write_progress_line("Accepted graph checkpoint is ready for publication.");
        }
        SessionOperationStatus::New => {
            write_progress_line(&format!(
                "Staging graph database to GCS (attempt {attempt}/2)..."
            ));
            let local_remote_url =
                local_server.database_url(&capability.session_scope.target_database);
            let configure_remote = {
                let _heartbeat = graph_progress.heartbeat("configuring the loopback Dolt remote");
                ensure_graph_remote(graph, &remote.name, &local_remote_url)
            };
            if let Err(error) = configure_remote {
                graph_progress.finish();
                return Err(close_server_after_graph_failure(
                    local_server,
                    graph_progress,
                    PushGraphTransferError::database("configuring loopback Dolt remote", error),
                ));
            }
            write_progress_line(
                "Planning destination-missing Dolt chunks before the first logical chunk upload...",
            );
            let push_result = {
                let _heartbeat =
                    graph_progress.heartbeat("running Dolt push through loopback RemoteServer");
                push_graph_branch_with_progress(graph, &remote.name, branch, force, graph_progress)
            };
            if let Err(error) = push_result {
                graph_progress.finish();
                let push_error = if is_non_fast_forward_push_error(&error) {
                    PushGraphTransferError::NonFastForward
                } else {
                    PushGraphTransferError::database(
                        "running Dolt push through loopback RemoteServer",
                        error,
                    )
                };
                return Err(close_server_after_graph_failure(
                    local_server,
                    graph_progress,
                    push_error,
                ));
            }
            let validation = {
                let _heartbeat = graph_progress.heartbeat("validating the staged graph branch");
                validate_direct_session_branch(&local_server, branch, destination_hash)
            };
            if let Err(error) = validation {
                graph_progress.finish();
                return Err(close_server_after_graph_failure(
                    local_server,
                    graph_progress,
                    error,
                ));
            }
            let stage_result = {
                let _heartbeat =
                    graph_progress.heartbeat("staging graph blocks and checkpointing the session");
                local_server.stage_request().map_err(|error| {
                    PushGraphTransferError::database("staging direct-push operation", error)
                })
            };
            if let Err(primary_error) = stage_result {
                return Err(close_server_after_graph_failure(
                    local_server,
                    graph_progress,
                    primary_error,
                ));
            }
            let close_result = close_direct_push_server(local_server, graph_progress, false);
            graph_progress.finish();
            close_result?;
            write_progress_line("Graph staging complete.");
        }
        SessionOperationStatus::Failed => {
            graph_progress.finish();
            return Err(close_server_after_graph_failure(
                local_server,
                graph_progress,
                PushGraphTransferError::Protocol(
                    "GenHub direct-push session has already failed".to_string(),
                ),
            ));
        }
        SessionOperationStatus::Conflict => {
            graph_progress.finish();
            return Err(close_server_after_graph_failure(
                local_server,
                graph_progress,
                PushGraphTransferError::Protocol(
                    "GenHub direct-push session conflicts with another operation".to_string(),
                ),
            ));
        }
    }
    write_progress_line("Publishing graph manifest...");
    {
        let _heartbeat = graph_progress.heartbeat("publishing the graph manifest to GenHub");
        renewal.publish()?;
    }
    write_progress_line("Graph manifest published.");
    Ok(())
}

fn validate_direct_session_branch(
    local_server: &RemoteServer,
    branch: &str,
    destination_hash: &DoltHashId,
) -> Result<(), PushGraphTransferError> {
    let Some(connection) = local_server.database_connection() else {
        return Err(PushGraphTransferError::Protocol(
            "GenHub direct-push server has no inspectable database connection".to_string(),
        ));
    };
    let staged_hash: DoltHashId = connection
        .query_row("SELECT dolt_hashof(?1)", [branch], |row| row.get(0))
        .map_err(|error| {
            PushGraphTransferError::Protocol(format!(
                "GenHub direct-push session has no valid target branch: {error:?}"
            ))
        })?;
    if staged_hash != *destination_hash {
        return Err(PushGraphTransferError::Protocol(format!(
            "GenHub direct-push session branch '{branch}' does not match the local destination"
        )));
    }
    Ok(())
}

fn run_push_graph_transfer(
    graph: &GraphConnection,
    remote: &Remote,
    branch: &str,
    force: bool,
    idempotency_token: Uuid,
    expected_transfer_id: Option<Uuid>,
    destination_hash: &DoltHashId,
) -> Result<PushCapabilityRenewal, Box<dyn Error>> {
    let mut last_error = None;
    let mut expected_session_identity: Option<PushCapabilityRenewal> = None;
    for attempt in 0..2 {
        if attempt > 0 {
            write_progress_line("Retrying direct GCS graph transfer (attempt 2/2)...");
        }
        let authorization = {
            let _heartbeat = ProgressHeartbeat::waiting("requesting GenHub direct-push capability");
            if attempt == 0 {
                transfer_authorization(
                    remote,
                    RemoteOperation::Push,
                    Some(branch),
                    force,
                    Some(idempotency_token),
                    login_origin,
                )
            } else {
                transfer_authorization(
                    remote,
                    RemoteOperation::Push,
                    Some(branch),
                    force,
                    Some(idempotency_token),
                    |_| Err("interactive login is unavailable during a push retry".into()),
                )
            }
        }?;
        let Some(capability) = authorization.direct_push.as_ref() else {
            restore_canonical_url(graph, remote);
            return Err(Box::new(PushGraphTransferError::Protocol(
                "GenHub push capability did not include a direct GCS session".to_string(),
            )));
        };
        let response = authorization
            .capability_response
            .clone()
            .ok_or_else(|| "GenHub push capability response was missing".to_string())?;
        if let Some(expected_session_identity) = expected_session_identity.as_ref() {
            expected_session_identity
                .validate_session_identity(&response)
                .map_err(PushCapabilityRenewal::transfer_error)?;
        }
        let expected_transfer_id = expected_transfer_id.unwrap_or(
            authorization
                .push_lease
                .ok_or_else(|| "GenHub push capability did not include a lease".to_string())?
                .transfer_id,
        );
        let renewal = push_capability_renewal(
            remote,
            branch,
            force,
            idempotency_token,
            expected_transfer_id,
            response,
        )?;
        if expected_session_identity.is_none() {
            expected_session_identity = Some(renewal.clone());
        }
        if capability.session_id != idempotency_token {
            restore_canonical_url(graph, remote);
            return Err(Box::new(PushGraphTransferError::Protocol(
                "GenHub returned a direct-push session that does not match the idempotency token"
                    .to_string(),
            )));
        }
        let transfer_result = push_graph_through_direct_session(
            graph,
            remote,
            branch,
            force,
            destination_hash,
            &renewal,
            attempt + 1,
        );
        match transfer_result {
            Ok(()) => {
                restore_canonical_url(graph, remote);
                return Ok(renewal);
            }
            Err(error) if attempt == 0 && error.is_authorization_error() => {
                restore_canonical_url(graph, remote);
                last_error = Some(error);
            }
            Err(error) if error.is_stale_session() => {
                restore_canonical_url(graph, remote);
                if matches!(
                    &error,
                    PushGraphTransferError::Client(RemoteClientError::StaleGraphSession)
                ) {
                    return Err(Box::new(PushGraphTransferError::StaleSessionConfirmed));
                }
                let confirmation = {
                    let _heartbeat = ProgressHeartbeat::waiting(
                        "confirming the stale graph session with GenHub",
                    );
                    renewal.confirm_stale_session()
                };
                return match confirmation {
                    Ok(true) => Err(Box::new(PushGraphTransferError::StaleSessionConfirmed)),
                    Ok(false) => Err(Box::new(error)),
                    Err(confirmation) => {
                        Err(Box::new(PushGraphTransferError::StaleSessionCheckFailed {
                            primary: Box::new(error),
                            confirmation: Box::new(confirmation),
                        }))
                    }
                };
            }
            Err(error) => {
                restore_canonical_url(graph, remote);
                return Err(Box::new(error));
            }
        }
    }
    restore_canonical_url(graph, remote);
    Err(Box::new(
        last_error.expect("should retain authorization error"),
    ))
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
    for attempt in 0..2 {
        let authorization =
            transfer_authorization(remote, operation, Some(branch), force, None, login_origin)?;
        let Some(remote_url) = authorization.remote_url.as_deref() else {
            restore_canonical_url(graph, remote);
            return Err(format!(
                "GenHub {:?} capability did not include a graph remote URL",
                operation
            )
            .into());
        };
        ensure_graph_remote(graph, &remote.name, remote_url)?;
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
    let length = fs::metadata(&source_path)?.len();
    let progress = AssetUploadProgressReporter::new(
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
    target: AssetTransferTarget<'_>,
    mut complete_commit: impl FnMut(&DoltHashId) -> Result<(), Box<dyn Error>>,
    mut before_transfer: impl FnMut() -> Result<(), Box<dyn Error>>,
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
    if operation == RemoteOperation::Push {
        let local_asset_count = assets.len();
        let asset_label = if local_asset_count == 1 {
            "asset"
        } else {
            "assets"
        };
        write_progress_line(&format!(
            "Checking {local_asset_count} local {asset_label} for transfer..."
        ));
        {
            let _heartbeat =
                ProgressHeartbeat::waiting("refreshing the push lease before asset transfer");
            before_transfer()?;
        }
    }
    let response = {
        let _heartbeat = ProgressHeartbeat::waiting("requesting GenHub asset transfer URLs");
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
            write_progress_line("No new assets need uploading.");
        } else {
            let asset_label = if upload_total == 1 { "asset" } else { "assets" };
            write_progress_line(&format!(
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
                    ProgressHeartbeat::waiting("refreshing the push lease before asset upload");
                before_transfer()?;
            }
            upload_receipts.push(upload_asset(
                &client,
                workspace,
                &asset,
                &transfer.url,
                upload_index,
                upload_total,
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
        let authorization = transfer_authorization(
            remote,
            RemoteOperation::Clone,
            None,
            false,
            None,
            login_origin,
        )?;
        let remote_url = authorization
            .remote_url
            .as_deref()
            .ok_or("GenHub clone capability did not include a graph remote URL")?;
        match clone_remote(&graph, remote_url) {
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
        || Ok(()),
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
    /// The remote branch has commits that are not in the local branch.
    #[error("The branch is not a fast-forward of the remote. Use --force to overwrite the remote.")]
    NonFastForward,
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
        || error
            .downcast_ref::<PushGraphTransferError>()
            .is_some_and(PushGraphTransferError::is_non_fast_forward)
    {
        RemotePushError::NonFastForward
    } else {
        RemotePushError::GraphTransfer(error)
    }
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
    write_progress_line(&format!(
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
        if operation.push_session_id.is_none() {
            operation.set_push_session_id(&config, Uuid::new_v4())?;
        }
        let destination_hash = hash_of(&graph, &branch)?;
        let same_destination = operation
            .to_commit
            .as_ref()
            .is_some_and(|recorded_destination| recorded_destination == &destination_hash);
        let destination_changed = operation
            .to_commit
            .as_ref()
            .is_some_and(|recorded_destination| recorded_destination != &destination_hash);
        let existing_transfer_lease = match (
            operation.to_commit.as_ref(),
            operation.transfer_id,
            operation.transfer_expires_at,
        ) {
            (Some(recorded_destination), Some(transfer_id), Some(expires_at))
                if recorded_destination == &destination_hash
                    && expires_at > Utc::now().timestamp() =>
            {
                Some(PushTransferLease {
                    transfer_id,
                    expires_at,
                })
            }
            (Some(_), Some(_), Some(_)) | (None, None, None) => None,
            _ => {
                return Err(RemotePushError::IncompleteTransferLease { branch });
            }
        };
        if destination_changed {
            operation.set_push_session_id(&config, Uuid::new_v4())?;
        }
        let renewal = if let Some(transfer_lease) = existing_transfer_lease {
            write_progress_line("Graph manifest already published; resuming asset transfer...");
            let push_session_id = operation
                .push_session_id
                .expect("push session UUID should be persisted before transfer");
            let renewal_result = {
                let _heartbeat =
                    ProgressHeartbeat::waiting("resuming the existing direct-push lease");
                acquire_existing_push_renewal(
                    &remote,
                    &branch,
                    force,
                    push_session_id,
                    transfer_lease.transfer_id,
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
            let refreshed_lease = {
                let _heartbeat = ProgressHeartbeat::waiting("refreshing the existing push lease");
                renewal
                    .ensure_fresh()
                    .map_err(|error| RemotePushError::GraphTransfer(Box::new(error)))?
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
            let expected_transfer_id = same_destination.then_some(operation.transfer_id).flatten();
            let mut graph_transfer = run_push_graph_transfer(
                &graph,
                &remote,
                &branch,
                force,
                push_session_id,
                expected_transfer_id,
                &destination_hash,
            );
            let confirmed_stale_session = graph_transfer.as_ref().is_err_and(|error| {
                error
                    .downcast_ref::<PushGraphTransferError>()
                    .is_some_and(|error| {
                        matches!(error, PushGraphTransferError::StaleSessionConfirmed)
                    })
            });
            if confirmed_stale_session {
                write_progress_line(
                    "GenHub confirmed the accepted graph session is stale; opening a fresh session...",
                );
                let replacement_session_id = Uuid::new_v4();
                operation.reset_stale_push_session(&config, replacement_session_id)?;
                graph_transfer = run_push_graph_transfer(
                    &graph,
                    &remote,
                    &branch,
                    force,
                    replacement_session_id,
                    None,
                    &destination_hash,
                );
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
    let upload_receipts = {
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
            AssetTransferTarget {
                branch: &branch,
                history_ref: &branch,
                range: AssetTransferRange {
                    from_commit: assets_transfer_checkpoint.as_ref(),
                    previous_hash: previous_hash.as_ref(),
                    materialize: true,
                },
            },
            |_| Ok(()),
            &mut refresh_push_lease,
        )
    }
    .map_err(RemotePushError::AssetTransfer)?;
    if let Some((operation, renewal, destination_hash)) = push_context.as_mut() {
        persist_refreshed_push_lease(operation, &config, destination_hash, renewal)
            .map_err(RemotePushError::AssetTransfer)?;
        let transfer_lease = renewal
            .push_lease()
            .map_err(|error| RemotePushError::GraphTransfer(Box::new(error)))?;
        let repository = RepositoryRemote::parse(&remote.url)?;
        write_progress_line("Verifying asset uploads and finishing push...");
        let verification_started = Instant::now();
        let verification_result = {
            let _heartbeat = ProgressHeartbeat::waiting(
                "waiting for GenHub to verify asset uploads and finish the push",
            );
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
            write_progress_line(&format!(
                "GenHub did not confirm asset verification after {}; per-asset request progress above reflects bytes consumed.",
                format_elapsed(verification_started.elapsed()),
            ));
            return Err(RemotePushError::Client(error));
        }
        write_progress_line("GenHub verified and accepted the asset uploads.");
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
    use std::{
        collections::HashMap,
        env,
        ffi::OsString,
        fs,
        io::{Cursor, Read as _, Write as _},
        net::TcpListener,
        path::PathBuf,
        sync::{Arc, Mutex},
        thread,
    };

    use chrono::{DateTime, Duration as ChronoDuration, Utc};
    use gen_core::{DoltHashId, HashId, config::Workspace};
    use gen_models::{
        assets::{AssetRef, AssetRole, LocalAssetUri, materialization_destination_path},
        collection::Collection,
        db::{ConfigConnection, GraphConnection},
        history::dolt::{clone_remote, commit_all, remote_rows, remove_remote},
        operations::{
            Defaults, Remote, RemoteOperationKind as StoredRemoteOperationKind,
            RemoteOperationRecord, calculate_reader_checksum,
        },
    };
    use reqwest::blocking::Client;
    use rusqlite::{Connection, Error as SqlError, blockcachevfs::AuthRefreshReason};
    use serde_json::json;
    use tempfile::tempdir;
    use uuid::Uuid;

    use super::{
        AssetTransferRange, AssetTransferTarget, DownloadAssetOutcome, PushCapabilityFetcher,
        PushCapabilityRenewal, RemoteOperation, canonical_remote_url, clone_destination_name,
        copy_versioned_asset, download_asset, download_to_versioned_store, execute_pull,
        execute_push, file_graph_url, get_remaining_assets_to_transfer,
        preserve_graph_error_after_close, push_capability_renewal, resolve_remote,
        run_graph_transfer, safe_graph_error_context, temporary_path, transfer_assets,
    };
    use crate::{
        commands::remote::client::{
            CapabilityResponse, DirectPushCapability, SessionScope as ClientSessionScope,
        },
        get_config_connection, get_connection, get_raw_connection,
    };

    static ENVIRONMENT_LOCK: Mutex<()> = Mutex::new(());
    const TEST_TRANSFER_ID: Uuid = Uuid::from_u128(1);
    const RETRIED_TRANSFER_ID: Uuid = Uuid::from_u128(2);

    fn push_capability_response(
        token: &str,
        token_expires_at: DateTime<Utc>,
        expires_at: DateTime<Utc>,
        database_uri: &str,
        publish_url: &str,
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

    fn push_publish_url(token: &str) -> String {
        format!("https://genhub.bio/api/publish?capability={token}")
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

    fn direct_push_capability_body(request: &str, origin: &str) -> String {
        let scope = push_session_scope();
        json!({
            "remote_url": null,
            "expires_at": (Utc::now() + ChronoDuration::minutes(30)).to_rfc3339(),
            "default_branch": "main",
            "transfer_id": TEST_TRANSFER_ID,
            "direct_push": {
                "database_uri": push_database_uri(),
                "token_expires_at": (Utc::now() + ChronoDuration::hours(1)).to_rfc3339(),
                "session_id": request_session_id(request),
                "session_scope": {
                    "principal": scope.principal,
                    "target_database": scope.target_database,
                    "operations": scope.operations,
                },
                "publish_url": format!("{origin}/api/publish?capability=mock-signed-value")
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
                "publish_url": format!("{publish_origin}/api/publish?capability=refreshed-signed-value")
            }
        })
        .to_string();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("should accept renewal request");
            let mut request = [0_u8; 8192];
            let length = stream
                .read(&mut request)
                .expect("should read renewal request");
            let request = String::from_utf8_lossy(&request[..length]).into_owned();
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            )
            .expect("should send renewed capability");
            request
        });

        let remote = Remote {
            name: "origin".to_string(),
            url: format!("http://{address}/api/repos/alice/example"),
        };
        let renewal = push_capability_renewal(
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
        let request = server.join().expect("renewal server should finish");
        let latest = renewal
            .capability_response()
            .expect("should read the renewed capability response");
        let latest_lease = renewal
            .push_lease()
            .expect("should read the renewed push lease");

        assert_eq!(token, "rotated-token");
        assert!(
            request
                .to_ascii_lowercase()
                .contains(&format!("idempotency-token: {TEST_TRANSFER_ID}"))
        );
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
    fn test_expired_initial_push_capability_renews_before_open() {
        let now = Utc::now();
        let initial = push_capability_response(
            "expired-token",
            now - ChronoDuration::seconds(1),
            now + ChronoDuration::minutes(15),
            push_database_uri(),
            &push_publish_url("initial-signed-value"),
            push_session_scope(),
        );
        let refreshed = push_capability_response(
            "rotated-token",
            now + ChronoDuration::hours(1),
            now + ChronoDuration::minutes(30),
            push_database_uri(),
            &push_publish_url("refreshed-signed-value"),
            push_session_scope(),
        );
        let refresh_count = Arc::new(Mutex::new(0));
        let fetch_count = Arc::clone(&refresh_count);
        let fetch_response = refreshed.clone();
        let fetch: Arc<PushCapabilityFetcher> = Arc::new(move || {
            *fetch_count
                .lock()
                .expect("should lock capability renewal counter") += 1;
            Ok(fetch_response.clone())
        });

        let renewal = PushCapabilityRenewal::new(
            initial,
            TEST_TRANSFER_ID,
            TEST_TRANSFER_ID,
            "https://genhub.bio",
            fetch,
        )
        .expect("should renew an expired capability before opening the database");

        assert_eq!(
            renewal
                .auth_token(AuthRefreshReason::Request)
                .expect("should use the renewed access token"),
            "rotated-token"
        );
        assert_eq!(
            *refresh_count
                .lock()
                .expect("should lock capability renewal counter"),
            1,
            "a valid renewed capability should be reused"
        );
        assert_eq!(
            renewal
                .capability_response()
                .expect("should read the renewed capability"),
            refreshed
        );
    }

    #[test]
    fn test_push_capability_request_uses_fractional_margin_for_short_lifetimes() {
        let now = Utc::now();
        let response = push_capability_response(
            "short-lived-token",
            now + ChronoDuration::seconds(10),
            now + ChronoDuration::minutes(15),
            push_database_uri(),
            &push_publish_url("short-lived-value"),
            push_session_scope(),
        );

        assert!(
            !super::capability_needs_refresh(
                &response,
                now,
                now + ChronoDuration::milliseconds(500)
            ),
            "a fresh short-lived token should not trigger an immediate renewal"
        );
        assert!(super::capability_needs_refresh(
            &response,
            now,
            now + ChronoDuration::seconds(9)
        ));
    }

    #[test]
    fn test_session_auth_io_error_is_retryable_authorization_failure() {
        let error = SqlError::SqliteFailure(
            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_IOERR_AUTH),
            Some("credential refresh failed".to_string()),
        );

        assert!(super::is_authorization_error(&error));
    }

    #[test]
    fn test_push_capability_renewal_rejects_changed_uri_or_scope() {
        let now = Utc::now();
        let cases = [
            push_capability_response(
                "rotated-token",
                now + ChronoDuration::hours(2),
                now + ChronoDuration::minutes(30),
                "gcs://other-bucket/.gen/graph_db/?database=default.db&project=test-project&vfs=blockcachevfs&endpoint=https%3A%2F%2Fstorage.googleapis.com&access_token=TOKEN",
                &push_publish_url("refreshed-signed-value"),
                push_session_scope(),
            ),
            push_capability_response(
                "rotated-token",
                now + ChronoDuration::hours(2),
                now + ChronoDuration::minutes(30),
                push_database_uri(),
                &push_publish_url("refreshed-signed-value"),
                ClientSessionScope {
                    operations: r#"["write","push","feature",false]"#.to_string(),
                    ..push_session_scope()
                },
            ),
        ];

        for refreshed in cases {
            let initial = push_capability_response(
                "initial-token",
                now + ChronoDuration::hours(1),
                now + ChronoDuration::minutes(15),
                push_database_uri(),
                &push_publish_url("initial-signed-value"),
                push_session_scope(),
            );
            let response = refreshed.clone();
            let fetch: Arc<PushCapabilityFetcher> = Arc::new(move || Ok(response.clone()));
            let renewal = PushCapabilityRenewal::new(
                initial.clone(),
                TEST_TRANSFER_ID,
                TEST_TRANSFER_ID,
                "https://genhub.bio",
                fetch,
            )
            .expect("should create a valid initial renewal coordinator");

            let error = renewal
                .auth_token(AuthRefreshReason::Unauthorized)
                .expect_err("renewal must reject changes outside access_token");

            assert!(!format!("{error:?}").contains("initial-token"));
            assert_eq!(
                renewal
                    .capability_response()
                    .expect("should retain the previous capability"),
                initial
            );
        }
    }

    fn seed_push_lease(
        config: &ConfigConnection,
        branch: &str,
        destination_hash: &DoltHashId,
        transfer_id: Uuid,
        expires_at: i64,
    ) -> Uuid {
        let mut operation = RemoteOperationRecord::begin_or_resume(
            config,
            "origin",
            branch,
            StoredRemoteOperationKind::Push,
            None,
        )
        .expect("should create pending push operation");
        let session_id = operation
            .push_session_id
            .expect("should allocate a push session UUID");
        operation
            .set_push_destination(config, destination_hash, transfer_id, expires_at)
            .expect("should seed graph transfer lease");
        session_id
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
            upstream_asset_ref_id: None,
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
    fn test_pull_authorization_failure_retries_with_a_fresh_capability() {
        let _environment_lock = ENVIRONMENT_LOCK
            .lock()
            .expect("should lock process environment");
        let _api_key = EnvironmentGuard::set("GENHUB_API_KEY", "push-test-key");
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
        });

        let connection = Connection::open_in_memory().expect("should open graph database");
        let graph = GraphConnection(connection);
        let remote = Remote {
            name: "origin".to_string(),
            url: format!("http://{address}/api/repos/alice/example"),
        };
        let mut attempts = 0;
        let transfer_lease = run_graph_transfer(
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
        assert_eq!(transfer_lease, None);
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
                        let body =
                            direct_push_capability_body(&request, &format!("http://{address}"));
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
        let listener = TcpListener::bind("127.0.0.1:0").expect("should bind capability server");
        let address = listener
            .local_addr()
            .expect("should read capability server address");
        let temp = tempdir().expect("should create advanced push retry directory");
        let server = thread::spawn(move || {
            let mut requests = Vec::new();
            for request_index in 0..4 {
                let (mut stream, _) = listener.accept().expect("should accept GenHub request");
                let mut request = [0_u8; 8192];
                let read = stream
                    .read(&mut request)
                    .expect("should read GenHub request");
                let request = String::from_utf8_lossy(&request[..read]).into_owned();
                requests.push(request.clone());
                match request_index {
                    0 => {
                        let body =
                            direct_push_capability_body(&request, &format!("http://{address}"));
                        write!(
                            stream,
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                            body.len()
                        )
                        .expect("should write existing direct-push capability");
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
                        let body = "asset verification unavailable";
                        write!(
                            stream,
                            "HTTP/1.1 500 Internal Server Error\r\nContent-Type: text/plain\r\nContent-Length: {}\r\n\r\n{body}",
                            body.len()
                        )
                        .expect("should write failed completion response");
                    }
                    3 => {
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
                        .expect("should write capability response");
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
        assert_eq!(requests.len(), 4);
        assert!(requests[0].starts_with("POST /api/repos/alice/example/remote-capability "));
        assert!(requests[1].starts_with("POST /api/repos/alice/example/asset-transfers "));
        assert!(requests[2].starts_with("POST /api/repos/alice/example/asset-transfers/complete "));
        assert!(requests[3].contains("\"operation\":\"push\""));
        assert!(!requests[3].contains(&original_session_id.to_string()));
        assert!(
            requests[3]
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
                        let body =
                            direct_push_capability_body(&request, &format!("http://{address}"));
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
    fn test_direct_push_cleanup_error_does_not_replace_primary_sql_error() {
        let primary = super::PushGraphTransferError::database(
            "running Dolt push through loopback RemoteServer",
            SqlError::InvalidQuery,
        );
        let close = Err(super::PushGraphTransferError::database(
            "closing direct GCS server",
            SqlError::SqliteFailure(
                rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_IOERR),
                Some("signed-url=secret access_token=secret".to_string()),
            ),
        ));

        let cleanup_context = safe_graph_error_context(close.as_ref().expect_err("should fail"));
        let error = preserve_graph_error_after_close(primary, close);

        assert!(matches!(
            error,
            super::PushGraphTransferError::Database {
                phase: "running Dolt push through loopback RemoteServer",
                source: SqlError::InvalidQuery,
            }
        ));
        assert!(cleanup_context.contains("SQLite code"));
        assert!(!cleanup_context.contains("secret"));
        assert!(!error.to_string().contains("access_token"));
        assert!(!error.to_string().contains("signed"));
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
            super::RemotePushError::NonFastForward
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
    fn test_stale_session_classifier_requires_exact_sqlite_busy_error() {
        let exact = SqlError::SqliteFailure(
            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_BUSY),
            Some("published manifest changed since the accepted session head".to_string()),
        );
        let other_busy = SqlError::SqliteFailure(
            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_BUSY),
            Some("database is locked".to_string()),
        );
        let other_code = SqlError::SqliteFailure(
            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_IOERR),
            Some("published manifest changed since the accepted session head".to_string()),
        );

        assert!(super::is_stale_accepted_session_error(&exact));
        assert!(!super::is_stale_accepted_session_error(&other_busy));
        assert!(!super::is_stale_accepted_session_error(&other_code));
    }
}
