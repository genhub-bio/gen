//! Opens a GenHub read capability as a live graph database for browser sessions.
//!
//! Unlike clone, this path keeps the CAB-backed SQLite VFS attached to the workspace instead of
//! copying graph history into a second database. The connection is local-write-only so editing and
//! Dolt commits stay in its private cache until the existing push workflow publishes them.

use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use gen_models::{
    db::GraphConnection,
    history::dolt::{active_branch, connect_branch},
};
use rusqlite::{
    Connection, OpenFlags,
    blockcachevfs::{AuthError, AuthRefreshReason},
};
use url::Url;

use super::{
    CapabilityTransportIdentity, capability::ReadCapabilityTokenCache, ensure_graph_remote,
    normalize_cloud_open_authorization_error, safe_transfer_error,
    validate_capability_transport_identity,
};
use crate::commands::remote::{
    client::{CapabilityRequest, RemoteOperation, RepositoryRemote, acquire_capability},
    login_origin,
};

/// The graph connection and VFS owner returned when a browser opens a repository directly.
///
/// `graph_path` contains only a credential-free SQLite URI. The original CAB URI remains inside
/// `anchor`, where its authorization refresh callback and registered VFS stay alive.
pub struct BrowserGraphOpen {
    /// Secondary handle used by repository queries and writes.
    pub graph: GraphConnection,
    /// Credentialed connection that registers and owns the cloud VFS.
    pub anchor: Connection,
    /// Credential-free file URI selecting the anchor's registered VFS.
    pub graph_path: PathBuf,
    /// Branch selected on the returned graph connection.
    pub branch: String,
    /// Canonical GenHub repository URL stored as the repository remote.
    pub canonical_remote_url: String,
}

/// Opens a read capability as a local-write-only graph connection without running Dolt clone.
///
/// The read capability is acquired and refreshed here so its bearer URL is never returned to
/// Python, stored in workspace configuration, or embedded in the graph's Dolt remote. The private
/// cache accepts local SQL and Dolt changes while preventing those writes from reaching GCS.
///
/// # Errors
///
/// Returns an error if GenHub rejects or changes the read capability, the CAB database cannot be
/// opened, the registered VFS cannot be reopened, or the requested branch is unavailable.
pub fn open_browser_graph(
    repository_url: &str,
    branch: Option<&str>,
) -> Result<BrowserGraphOpen, String> {
    let repository =
        RepositoryRemote::parse(repository_url).map_err(|error| safe_transfer_error(&error))?;
    let request = CapabilityRequest {
        operation: RemoteOperation::Clone,
        branch: None,
        force: false,
    };
    let capability = acquire_capability(&repository, &request, login_origin)
        .map_err(|error| safe_transfer_error(&error))?;
    let read_url = capability
        .remote_url
        .as_deref()
        .ok_or_else(|| "GenHub did not provide a graph read capability".to_string())?;

    let mut capability_identity = CapabilityTransportIdentity::default();
    let cloud_uri = validate_capability_transport_identity(read_url, &mut capability_identity)
        .map_err(|_| "GenHub returned an invalid graph read capability".to_string())?;
    let initial_token = cloud_uri
        .gcs_access_token()
        .map(str::to_string)
        .ok_or_else(|| "GenHub returned an invalid graph read capability".to_string())?;
    let initial_expiry: DateTime<Utc> = capability.expires_at;
    let token_cache = ReadCapabilityTokenCache::new(initial_token, initial_expiry, Utc::now())
        .map_err(|_| "GenHub returned an invalid graph read capability".to_string())?;
    let identity = cloud_uri.identity();
    let mut local_uri = Url::parse(cloud_uri.as_str())
        .map_err(|_| "GenHub returned an invalid graph read capability".to_string())?;
    if local_uri
        .query_pairs()
        .any(|(name, _)| name == "local_writes_only")
    {
        return Err("GenHub returned an invalid graph read capability".to_string());
    }
    local_uri
        .query_pairs_mut()
        .append_pair("local_writes_only", "true");

    let refresh_repository = repository.clone();
    let refresh_callback = move |_storage: &str,
                                 _project: &str,
                                 _container: &str,
                                 reason: AuthRefreshReason| {
        token_cache.auth_token(reason, || {
            let request = CapabilityRequest {
                operation: RemoteOperation::Clone,
                branch: None,
                force: false,
            };
            let capability = acquire_capability(&refresh_repository, &request, login_origin)
                .map_err(|_| AuthError("GenHub rejected graph capability renewal".to_string()))?;
            let read_url = capability
                .remote_url
                .as_deref()
                .ok_or_else(|| AuthError("GenHub did not renew graph access".to_string()))?;
            let refreshed_uri = rusqlite::blockcachevfs::CloudConnectionUri::parse(read_url)
                .map_err(|_| {
                    AuthError("GenHub returned an invalid graph capability".to_string())
                })?;
            if refreshed_uri.identity() != identity {
                return Err(AuthError(
                    "GenHub changed the graph capability scope during renewal".to_string(),
                ));
            }
            let token = refreshed_uri
                .gcs_access_token()
                .map(str::to_string)
                .ok_or_else(|| {
                    AuthError("GenHub returned an invalid graph capability".to_string())
                })?;
            Ok((token, capability.expires_at))
        })
    };

    // This owner registers the named VFS and retains its scoped auth refresh callback. Dolt
    // queries use a second credential-free file URI so this handle can remain a separate lifetime
    // guard across graph refreshes.
    let anchor = Connection::open_with_flags_and_cloud_auth_refresh(
        local_uri.as_str(),
        OpenFlags::SQLITE_OPEN_READ_WRITE
            | OpenFlags::SQLITE_OPEN_URI
            | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        refresh_callback,
    )
    .map_err(|error| {
        let error = normalize_cloud_open_authorization_error(error);
        let code = error
            .sqlite_error()
            .map_or(0, |sqlite_error| sqlite_error.extended_code);
        format!("opening the graph read capability failed (SQLite code {code})")
    })?;
    let database_path = anchor
        .blockcachevfs_path()
        .ok_or_else(|| "graph read capability did not register a cloud VFS".to_string())?;
    let vfs_name = anchor
        .blockcachevfs_name()
        .ok_or_else(|| "graph read capability did not register a cloud VFS".to_string())?;
    let mut graph_url = Url::from_file_path(Path::new(&database_path))
        .map_err(|_| "graph read capability returned an invalid database path".to_string())?;
    graph_url.query_pairs_mut().append_pair("vfs", &vfs_name);

    let graph_connection = Connection::open_with_flags(
        graph_url.as_str(),
        OpenFlags::SQLITE_OPEN_READ_WRITE
            | OpenFlags::SQLITE_OPEN_URI
            | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|_| "opening the private graph view failed".to_string())?;
    rusqlite::vtab::array::load_module(&graph_connection)
        .map_err(|_| "loading graph query support failed".to_string())?;
    let graph = GraphConnection(graph_connection);

    let active =
        active_branch(&graph).map_err(|_| "reading the graph branch failed".to_string())?;
    let branch = branch.unwrap_or(&capability.default_branch).to_string();
    if branch != active {
        connect_branch(&graph, &branch)
            .map_err(|_| format!("graph branch '{branch}' is unavailable"))?;
    }
    ensure_graph_remote(&graph, "origin", repository.canonical_url())
        .map_err(|_| "configuring the graph remote failed".to_string())?;

    Ok(BrowserGraphOpen {
        graph,
        anchor,
        graph_path: PathBuf::from(graph_url.as_str()),
        branch,
        canonical_remote_url: repository.canonical_url().to_string(),
    })
}
