//! Models and persistence helpers for Gen operations.
//!
//! An operation is a user-visible change committed to the Dolt-backed graph database. The Dolt
//! commit is the authoritative history entry and records every changed graph row. Gen also writes
//! a `gen_operation_log` row into that database to describe the semantic operation: its kind, the
//! command or summary that produced it, and when it was created. Because this row is committed
//! with the graph changes, it follows branches and can be queried at historical refs.
//!
//! Files used by an operation are tracked as metadata rather than stored in Dolt. A
//! `gen_asset_refs` row identifies each asset by its URI, content checksum, file type, logical
//! path, display name, and role. A `gen_operation_assets` row links an operation-log ID to an
//! asset-reference ID and records the asset's role in that operation, allowing one operation to
//! use multiple assets and one content-addressed asset reference to be reused by multiple
//! operations. These rows are included in the same Dolt commit as the operation, while the file
//! contents live in the workspace asset store or remote object storage addressed by the URI.

use std::{
    io::{self, BufReader},
    path::Path,
};

use gen_core::{DoltHashId, HashId, Sha256Hash, Workspace, calculate_hash};
use itertools::Itertools;
use rusqlite::{
    OptionalExtension, Result as SQLResult, Row, params,
    types::{FromSql, FromSqlResult, ValueRef},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::{
    assets::{
        AssetRef, AssetRole, AssetUri, LocalAssetUri, OperationAsset, OperationKind, OperationLog,
    },
    db::{ConfigConnection, DbContext},
    errors::{
        AddFilesOperationError, FileAdditionError, FileStoreError, OperationError, RemoteError,
    },
    file_types::FileTypes,
    history::{
        HistoryStore,
        dolt::{DoltHistoryStore, is_current_branch_dirty},
    },
    traits::*,
};

pub const GEN_DEFAULT_COMMITTER_NAME: &str = "gen";
pub const DEFAULT_COMMITTER_EMAIL: &str = "gen@genhub.bio";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperationFile {
    pub filename: String,
    pub file_path: String,
    pub file_type: FileTypes,
    pub checksum_override: Option<Sha256Hash>,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct OperationFileInfo {
    pub id: HashId,
    pub filename: String,
    pub file_path: String,
    pub asset_uri: String,
    pub file_type: FileTypes,
    pub checksum: Option<Sha256Hash>,
}

impl OperationFile {
    pub fn new(file_path: impl Into<String>) -> Self {
        let file_path = file_path.into();
        let file_type = FileTypes::infer_from_path(&file_path);
        let filename = Path::new(&file_path)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(&file_path)
            .to_string();
        Self {
            filename,
            file_path,
            file_type,
            checksum_override: None,
        }
    }

    pub fn set_file_type(mut self, file_type: FileTypes) -> Self {
        self.file_type = file_type;
        self
    }

    pub fn set_checksum_override(mut self, checksum: Sha256Hash) -> Self {
        self.checksum_override = Some(checksum);
        self
    }

    /// Resolves the path stored in operation metadata without materializing remote assets.
    ///
    /// Local inputs use their content-addressed repository path and therefore require a checksum;
    /// remote inputs keep their URI so they can be accessed lazily.
    pub fn storage_file_path(
        workspace: &Workspace,
        path_or_uri: &str,
        checksum: Option<&Sha256Hash>,
    ) -> Result<String, FileAdditionError> {
        if LocalAssetUri::is_local_path_or_file_uri(path_or_uri) {
            let checksum = checksum.ok_or_else(|| {
                FileAdditionError::ChecksumError(format!(
                    "local operation file has no checksum: {path_or_uri}"
                ))
            })?;
            LocalAssetUri::operation_file_path(workspace, path_or_uri, checksum)
        } else {
            Ok(path_or_uri.to_string())
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperationInfo {
    pub files: Vec<OperationFile>,
    pub description: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperationSummary {
    pub operation_info: OperationInfo,
    pub summary: String,
}

impl OperationSummary {
    pub fn new(operation_info: OperationInfo, summary: impl Into<String>) -> Self {
        Self {
            operation_info,
            summary: summary.into(),
        }
    }
}

#[cfg_attr(
    feature = "profiling",
    tracing::instrument(skip(context, operation_summary))
)]
pub fn commit_operation_summary(
    context: &DbContext,
    operation_summary: &OperationSummary,
) -> Result<DoltHashId, OperationError> {
    let workspace = context.workspace();
    let graph_conn = context.graph().conn();
    let history_store = DoltHistoryStore::new_with_config(graph_conn, context.config().conn());
    let operation_info = &operation_summary.operation_info;

    if !is_current_branch_dirty(graph_conn)? {
        return Err(OperationError::NoChanges);
    }

    let prepared_files = operation_info
        .files
        .iter()
        .map(|operation_file| {
            let file_addition = FileAddition::prepare(
                workspace,
                &operation_file.file_path,
                operation_file.file_type,
                operation_file.checksum_override,
            )?;
            let logical_path = OperationFile::storage_file_path(
                workspace,
                &operation_file.file_path,
                file_addition.checksum.as_ref(),
            )?;
            Ok((file_addition, logical_path))
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| match err {
            FileAdditionError::ConfigError(config_error) => {
                OperationError::ConfigError(config_error)
            }
            other => OperationError::SQLError(other.to_string()),
        })?;
    let tracked_assets = prepared_files
        .iter()
        .zip(operation_info.files.iter())
        .map(
            |((file_addition, logical_path), operation_file)| OperationAssetRecord {
                file_addition,
                role: AssetRole::Input,
                logical_path: Some(logical_path.as_str()),
                name: Some(operation_file.filename.as_str()),
                upstream_asset_ref_id: None,
            },
        )
        .collect::<Vec<_>>();
    let operation_kind = OperationKind::Other(operation_info.description.clone());
    track_operation_assets(
        graph_conn,
        None,
        &operation_kind,
        &operation_summary.summary,
        &tracked_assets,
    )?;

    let commit_result = history_store.commit_all(&operation_summary.summary);
    commit_result.map_err(OperationError::from)
}

pub(crate) struct OperationAssetRecord<'a> {
    pub file_addition: &'a FileAddition,
    pub role: AssetRole,
    pub logical_path: Option<&'a str>,
    pub name: Option<&'a str>,
    pub upstream_asset_ref_id: Option<&'a HashId>,
}

#[cfg_attr(
    feature = "profiling",
    tracing::instrument(skip(graph_conn, log_id, assets))
)]
pub(crate) fn track_operation_assets(
    graph_conn: &crate::db::GraphConnection,
    log_id: Option<&HashId>,
    operation_kind: &OperationKind,
    command: &str,
    assets: &[OperationAssetRecord<'_>],
) -> Result<(), rusqlite::Error> {
    let created_on = chrono::Utc::now()
        .timestamp_nanos_opt()
        .expect("should create operation asset timestamp");
    let log_id = log_id
        .copied()
        .unwrap_or_else(|| OperationLog::id_hash(operation_kind, command, created_on));
    let operation_log = OperationLog {
        id: log_id,
        operation_kind: operation_kind.clone(),
        command: command.to_string(),
        created_on,
    };
    OperationLog::create(graph_conn, &operation_log)?;

    for asset in assets {
        let file_type = asset.file_addition.file_type.as_str();
        let asset_ref_id = AssetRef::id_hash(
            &asset.file_addition.asset_uri,
            file_type,
            asset.file_addition.checksum.as_ref(),
            &asset.role,
            asset.logical_path,
            asset.name,
            asset.upstream_asset_ref_id,
        );
        let asset_ref = AssetRef {
            id: asset_ref_id,
            uri: asset.file_addition.asset_uri.clone(),
            file_type: file_type.to_string(),
            checksum: asset.file_addition.checksum,
            size: None,
            role: asset.role.clone(),
            logical_path: asset.logical_path.map(str::to_string),
            name: asset.name.map(str::to_string),
            created_on,
            upstream_asset_ref_id: asset.upstream_asset_ref_id.copied(),
        };
        AssetRef::create(graph_conn, &asset_ref)?;
        OperationAsset::create(
            graph_conn,
            &OperationAsset {
                log_id,
                asset_ref_id,
                role: asset.role.clone(),
            },
        )?;
    }

    Ok(())
}

#[cfg_attr(
    all(debug_assertions, feature = "profiling"),
    tracing::instrument(skip(context, files, message))
)]
/// Records files used by an operation, retaining local content and preserving remote URIs.
///
/// Checksum overrides are propagated from callers that already streamed remote content, avoiding
/// a credentials-dependent read whose only purpose would be hashing.
pub fn add_files_operation(
    context: &DbContext,
    files: &[OperationFile],
    message: Option<&str>,
) -> Result<DoltHashId, AddFilesOperationError> {
    let workspace = context.workspace();
    let graph_conn = context.graph().conn();

    let file_additions = files
        .iter()
        .map(|operation_file| {
            let file_addition = FileAddition::prepare(
                workspace,
                &operation_file.file_path,
                operation_file.file_type,
                operation_file.checksum_override,
            )?;
            let operation_file_path = OperationFile::storage_file_path(
                workspace,
                &operation_file.file_path,
                file_addition.checksum.as_ref(),
            )?;
            Ok::<(FileAddition, String, String), FileAdditionError>((
                file_addition,
                operation_file.filename.clone(),
                operation_file_path,
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let unique_file_additions = file_additions
        .into_iter()
        .unique_by(|(file_addition, filename, file_path)| {
            (file_addition.id, filename.clone(), file_path.clone())
        })
        .collect::<Vec<_>>();

    let log_id = HashId(calculate_hash(
        &unique_file_additions
            .iter()
            .map(|(file_addition, filename, file_path)| {
                format!("{}:{filename}:{file_path}", file_addition.id)
            })
            .sorted()
            .join(":"),
    ));

    let summary = message.map(str::to_string).unwrap_or_else(|| {
        if files.len() == 1 {
            format!("Add file {}", files[0].file_path)
        } else {
            format!("Add {} files", files.len())
        }
    });
    let tracked_assets = unique_file_additions
        .iter()
        .map(
            |(file_addition, filename, file_path)| OperationAssetRecord {
                file_addition,
                role: AssetRole::Input,
                logical_path: Some(file_path.as_str()),
                name: Some(filename.as_str()),
                upstream_asset_ref_id: None,
            },
        )
        .collect::<Vec<_>>();
    track_operation_assets(
        graph_conn,
        Some(&log_id),
        &OperationKind::AddFile,
        &summary,
        &tracked_assets,
    )?;
    let history_store = DoltHistoryStore::new_with_config(graph_conn, context.config().conn());
    if history_store.status()?.is_empty() {
        return Err(AddFilesOperationError::OperationError(
            OperationError::NoChanges,
        ));
    }

    history_store
        .commit_all(&summary)
        .map_err(AddFilesOperationError::from)
}

pub fn calculate_file_checksum<P: AsRef<Path>>(file_path: P) -> Result<Sha256Hash, std::io::Error> {
    let file = std::fs::File::open(file_path)?;
    let reader = BufReader::new(file);
    calculate_reader_checksum(reader)
}

pub fn calculate_reader_checksum<R: std::io::Read>(
    reader: R,
) -> Result<Sha256Hash, std::io::Error> {
    let hash_bytes = calculate_stream_hash(reader)?;
    Ok(Sha256Hash(hash_bytes))
}

fn calculate_stream_hash<R: std::io::Read>(mut reader: R) -> Result<[u8; 32], std::io::Error> {
    let mut hasher = Sha256::new();
    io::copy(&mut reader, &mut hasher)?;
    let result = hasher.finalize();
    let mut hash_array = [0u8; 32];
    hash_array.copy_from_slice(&result);
    Ok(hash_array)
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, Deserialize, Serialize)]
pub struct FileAddition {
    pub id: HashId,
    pub asset_uri: String,
    pub file_type: FileTypes,
    pub checksum: Option<Sha256Hash>,
}

impl Query for FileAddition {
    type Model = FileAddition;

    const TABLE_NAME: &'static str = "file_additions";

    fn process_row(row: &Row) -> Self::Model {
        Self::Model {
            id: row.get(0).unwrap(),
            asset_uri: row.get(1).unwrap(),
            file_type: row.get(2).unwrap(),
            checksum: row.get(3).unwrap(),
        }
    }
}

impl FileAddition {
    pub fn file_path(&self) -> &str {
        self.asset_uri
            .strip_prefix(LocalAssetUri::SCHEME)
            .unwrap_or(&self.asset_uri)
    }

    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip(workspace, checksum_override))
    )]
    /// Prepares an asset reference for operation storage without eagerly reading remote content.
    ///
    /// Local content is copied and hashed as part of retention. Remote content uses an optional
    /// checksum supplied by a caller that performed useful streaming work.
    pub fn prepare(
        workspace: &Workspace,
        file_path: &str,
        file_type: FileTypes,
        checksum_override: Option<Sha256Hash>,
    ) -> Result<FileAddition, FileAdditionError> {
        let asset_uri = <dyn AssetUri>::new(workspace, file_path);
        let checksum = asset_uri.prepare_asset(workspace, checksum_override)?;
        let stored_asset_uri = match checksum.as_ref() {
            Some(checksum) => asset_uri.stored_asset_uri(workspace, checksum)?,
            None => asset_uri.uri().to_string(),
        };
        Ok(FileAddition {
            id: LocalAssetUri::generate_file_addition_id(checksum.as_ref(), &stored_asset_uri),
            asset_uri: stored_asset_uri,
            file_type,
            checksum,
        })
    }

    #[cfg_attr(
        all(debug_assertions, feature = "profiling"),
        tracing::instrument(skip(self, workspace))
    )]
    pub fn store_file(&self, workspace: &Workspace) -> Result<(), FileStoreError> {
        let asset_uri = <dyn AssetUri>::new(workspace, &self.asset_uri);
        asset_uri.store_file(self, workspace)
    }

    /// Returns the content-addressed filename when this asset has a verified checksum.
    ///
    /// Checksumless remote references do not live under `.gen/assets`, so they have no hashed
    /// storage filename.
    pub fn hashed_filename(&self) -> Option<String> {
        self.checksum
            .as_ref()
            .map(|checksum| <dyn AssetUri>::from_uri(&self.asset_uri).hashed_filename(checksum))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Remote {
    pub name: String,
    pub url: String,
}

impl Query for Remote {
    type Model = Remote;

    const TABLE_NAME: &'static str = "remotes";

    fn process_row(row: &Row) -> Self::Model {
        Remote {
            name: row.get(0).unwrap(),
            url: row.get(1).unwrap(),
        }
    }
}

impl Remote {
    /// Validate remote name - no spaces or special characters except hyphens and underscores
    pub fn validate_name(name: &str) -> Result<(), RemoteError> {
        if name.is_empty() {
            return Err(RemoteError::EmptyName);
        }

        if name
            .chars()
            .any(|c| !c.is_alphanumeric() && c != '-' && c != '_')
        {
            return Err(RemoteError::InvalidNameCharacters);
        }

        Ok(())
    }

    /// Validate URL format
    pub fn validate_url(url: &str) -> Result<(), RemoteError> {
        if url.is_empty() {
            return Err(RemoteError::EmptyUrl);
        }

        // Check if it looks like a URL with a scheme
        if url.contains("://") {
            match url::Url::parse(url) {
                Ok(parsed_url) => {
                    // Only allow http, https, and ssh schemes
                    match parsed_url.scheme() {
                        "http" | "https" | "ssh" | "file" => Ok(()),
                        _ => Err(RemoteError::UnsupportedUrlScheme),
                    }
                }
                Err(_) => Err(RemoteError::InvalidUrl("Invalid URL format".to_string())),
            }
        } else if url.starts_with('/') || url.contains(':') {
            // Assume it's a file path or SSH-style path (like user@host:path)
            Ok(())
        } else {
            Err(RemoteError::UnsupportedUrlScheme)
        }
    }

    /// Create a new remote with the given name and URL
    /// Validates input and handles constraint violations gracefully
    pub fn create(conn: &ConfigConnection, name: &str, url: &str) -> Result<Remote, RemoteError> {
        // Validate input
        Self::validate_name(name)?;
        Self::validate_url(url)?;

        let query = "INSERT INTO remotes (name, url) VALUES (?1, ?2)";
        let mut stmt = conn.prepare(query)?;

        match stmt.execute(params![name, url]) {
            Ok(_) => Ok(Remote {
                name: name.to_string(),
                url: url.to_string(),
            }),
            Err(rusqlite::Error::SqliteFailure(err, _))
                if err.code == rusqlite::ErrorCode::ConstraintViolation =>
            {
                Err(RemoteError::RemoteAlreadyExists(name.to_string()))
            }
            Err(e) => Err(RemoteError::DatabaseError(e)),
        }
    }

    /// Get a remote by name
    pub fn get_by_name(conn: &ConfigConnection, name: &str) -> Result<Remote, RemoteError> {
        let query = "SELECT name, url FROM remotes WHERE name = ?1";
        match Remote::get(conn, query, params![name]) {
            Ok(remote) => Ok(remote),
            Err(rusqlite::Error::QueryReturnedNoRows) => {
                Err(RemoteError::RemoteNotFound(name.to_string()))
            }
            Err(e) => Err(RemoteError::DatabaseError(e)),
        }
    }

    /// Get a remote by name, returning None if not found (for backward compatibility)
    pub fn get_by_name_optional(conn: &ConfigConnection, name: &str) -> Option<Remote> {
        Self::get_by_name(conn, name).ok()
    }

    /// List all remotes
    pub fn list_all(conn: &ConfigConnection) -> Vec<Remote> {
        Remote::query(
            conn,
            "SELECT name, url FROM remotes ORDER BY name",
            params![],
        )
    }

    /// Delete a remote by name
    pub fn delete(conn: &ConfigConnection, name: &str) -> Result<(), RemoteError> {
        // Check if remote exists first
        Self::get_by_name(conn, name)?;
        RemoteBranch::clear_by_remote(conn, name)?;

        let query = "DELETE FROM remotes WHERE name = ?1";
        let mut stmt = conn.prepare(query)?;
        stmt.execute(params![name])?;
        Ok(())
    }

    /// Check if a remote exists
    pub fn exists(conn: &ConfigConnection, name: &str) -> bool {
        Self::get_by_name_optional(conn, name).is_some()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RemoteBranch {
    pub remote_name: Option<String>,
    pub name: String,
}

impl RemoteBranch {
    pub fn set_remote_validated(
        conn: &ConfigConnection,
        branch_name: &str,
        remote_name: Option<&str>,
    ) -> Result<(), RemoteError> {
        if let Some(name) = remote_name {
            Remote::get_by_name(conn, name)?;
            conn.execute(
                "DELETE FROM remote_branch WHERE name = ?1",
                params![branch_name],
            )?;
            conn.execute(
                "INSERT INTO remote_branch (name, remote_name) VALUES (?1, ?2)",
                params![branch_name, name],
            )?;
        } else {
            conn.execute(
                "DELETE FROM remote_branch WHERE name = ?1",
                params![branch_name],
            )?;
        }
        Ok(())
    }

    pub fn get_remote(conn: &ConfigConnection, branch_name: &str) -> Option<String> {
        conn.query_row(
            "SELECT remote_name FROM remote_branch WHERE name = ?1",
            params![branch_name],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .ok()
        .flatten()
    }

    pub fn clear_by_remote(conn: &ConfigConnection, remote_name: &str) -> Result<(), RemoteError> {
        conn.execute(
            "DELETE FROM remote_branch WHERE remote_name = ?1",
            params![remote_name],
        )?;
        Ok(())
    }
}

/// A remote operation whose graph and asset phases must complete together.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RemoteOperationKind {
    /// Initializes a workspace from a remote repository.
    Clone,
    /// Advances an existing local branch from a remote repository.
    Pull,
    /// Advances a remote branch from the local repository.
    Push,
}

impl RemoteOperationKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Clone => "clone",
            Self::Pull => "pull",
            Self::Push => "push",
        }
    }
}

impl FromSql for RemoteOperationKind {
    fn column_result(value: ValueRef<'_>) -> FromSqlResult<Self> {
        value.as_str().map(|operation| match operation {
            "clone" => Self::Clone,
            "pull" => Self::Pull,
            "push" => Self::Push,
            _ => unreachable!("remote operation should satisfy its database constraint"),
        })
    }
}

/// Tracks a remote operation until its graph and asset phases are complete.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RemoteOperationRecord {
    /// Config database row identifier.
    pub id: i64,
    /// Configured remote participating in the operation.
    pub remote_name: String,
    /// Branch transferred by the operation.
    pub branch_name: String,
    /// Direction of the remote graph transfer.
    pub operation: RemoteOperationKind,
    /// Local branch commit from before the graph transfer.
    pub from_commit: Option<DoltHashId>,
    /// Last commit whose complete asset batch was verified locally.
    pub assets_transfer_checkpoint: Option<DoltHashId>,
    /// Destination graph commit whose assets must be transferred.
    pub to_commit: Option<DoltHashId>,
    /// GenHub push lease retained by a pending operation.
    pub transfer_id: Option<Uuid>,
    /// Unix timestamp after which GenHub will reject the retained push lease.
    pub transfer_expires_at: Option<i64>,
    /// Time at which the operation began.
    pub started_at: String,
    /// Time at which both graph and asset transfer completed.
    pub completed_at: Option<String>,
    /// Time at which the operation became unrecoverable.
    pub failed_at: Option<String>,
}

impl Query for RemoteOperationRecord {
    type Model = RemoteOperationRecord;

    const TABLE_NAME: &'static str = "remote_operations";

    fn process_row(row: &Row) -> Self::Model {
        Self {
            id: row.get("id").unwrap(),
            remote_name: row.get("remote_name").unwrap(),
            branch_name: row.get("branch_name").unwrap(),
            operation: row.get("operation").unwrap(),
            from_commit: row.get("from_commit").unwrap(),
            assets_transfer_checkpoint: row.get("assets_transfer_checkpoint").unwrap(),
            to_commit: row.get("to_commit").unwrap(),
            transfer_id: row.get("transfer_id").unwrap(),
            transfer_expires_at: row.get("transfer_expires_at").unwrap(),
            started_at: row.get("started_at").unwrap(),
            completed_at: row.get("completed_at").unwrap(),
            failed_at: row.get("failed_at").unwrap(),
        }
    }
}

impl RemoteOperationRecord {
    /// Resumes an incomplete operation or starts one from the supplied local commit.
    pub fn begin_or_resume(
        conn: &ConfigConnection,
        remote_name: &str,
        branch_name: &str,
        operation: RemoteOperationKind,
        from_commit: Option<&DoltHashId>,
    ) -> SQLResult<Self> {
        let pending = conn
            .query_row(
                "SELECT * FROM remote_operations \
                 WHERE remote_name = ?1 AND branch_name = ?2 \
                   AND completed_at IS NULL AND failed_at IS NULL \
                   -- This is confusing, but this checks whether a clone operation is followed by a pull.
                   -- If a clone fails to download assets, a pull can finish the operation even though the
                   -- persisted operation is clone and the requested operation is pull.
                   AND (operation = ?3 OR operation = 'clone' AND ?3 = 'pull') \
                 ORDER BY id LIMIT 1",
                params![remote_name, branch_name, operation.as_str()],
                |row| Ok(Self::process_row(row)),
            )
            .optional()?;
        if let Some(pending) = pending {
            return Ok(pending);
        }

        let assets_transfer_checkpoint = conn
            .query_row(
                "SELECT assets_transfer_checkpoint FROM remote_operations \
                 WHERE remote_name = ?1 AND branch_name = ?2 AND completed_at IS NOT NULL \
                 ORDER BY id DESC LIMIT 1",
                params![remote_name, branch_name],
                |row| row.get::<_, DoltHashId>(0),
            )
            .optional()?;
        conn.execute(
            "INSERT INTO remote_operations \
             (remote_name, branch_name, operation, from_commit, assets_transfer_checkpoint) \
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                remote_name,
                branch_name,
                operation.as_str(),
                from_commit,
                assets_transfer_checkpoint
            ],
        )?;
        Self::get_by_id(conn, &conn.last_insert_rowid(), None)
            .ok_or(rusqlite::Error::QueryReturnedNoRows)
    }

    /// Records the desired end commit an operation is tracking.
    pub fn set_destination(
        &mut self,
        conn: &ConfigConnection,
        to_commit: &DoltHashId,
    ) -> SQLResult<()> {
        conn.execute(
            "UPDATE remote_operations SET to_commit = ?1 WHERE id = ?2",
            params![to_commit, self.id],
        )?;
        self.to_commit = Some(*to_commit);
        Ok(())
    }

    /// After transferring assets for a commit, mark it in the database so subsequent transfers for
    /// resumes can restart from there.
    pub fn advance_assets_transfer_checkpoint(
        &mut self,
        conn: &ConfigConnection,
        commit: &DoltHashId,
    ) -> SQLResult<()> {
        conn.execute(
            "UPDATE remote_operations SET assets_transfer_checkpoint = ?1 WHERE id = ?2",
            params![commit, self.id],
        )?;
        self.assets_transfer_checkpoint = Some(*commit);
        Ok(())
    }

    /// Records the successful push graph transfer and its capability-scoped lease.
    pub fn set_push_destination(
        &mut self,
        conn: &ConfigConnection,
        to_commit: &DoltHashId,
        transfer_id: Uuid,
        transfer_expires_at: i64,
    ) -> SQLResult<()> {
        conn.execute(
            "UPDATE remote_operations \
             SET to_commit = ?1, transfer_id = ?2, transfer_expires_at = ?3 WHERE id = ?4",
            params![to_commit, transfer_id, transfer_expires_at, self.id],
        )?;
        self.to_commit = Some(*to_commit);
        self.transfer_id = Some(transfer_id);
        self.transfer_expires_at = Some(transfer_expires_at);
        Ok(())
    }

    /// Marks the operation complete only after its graph and asset phases succeed.
    pub fn complete(&self, conn: &ConfigConnection) -> SQLResult<()> {
        if self.to_commit.is_none() || self.to_commit != self.assets_transfer_checkpoint {
            self.fail(conn)?;
            return Err(rusqlite::Error::InvalidQuery);
        }
        conn.execute(
            "UPDATE remote_operations SET completed_at = CURRENT_TIMESTAMP WHERE id = ?1",
            [self.id],
        )?;
        Ok(())
    }

    pub fn fail(&self, conn: &ConfigConnection) -> SQLResult<()> {
        conn.execute(
            "UPDATE remote_operations SET failed_at = CURRENT_TIMESTAMP WHERE id = ?1",
            [self.id],
        )?;
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Defaults {
    pub id: i64,
    pub collection_name: Option<String>,
    pub remote_name: Option<String>,
    pub current_branch_name: Option<String>,
    pub default_committer_name: String,
    pub default_committer_email: String,
}

impl Query for Defaults {
    type Model = Defaults;

    const TABLE_NAME: &'static str = "defaults";

    fn process_row(row: &Row) -> Self::Model {
        Defaults {
            id: row.get(0).unwrap(),
            collection_name: row.get(1).unwrap(),
            remote_name: row.get(2).unwrap(),
            current_branch_name: row.get(3).unwrap(),
            default_committer_name: row.get(4).unwrap(),
            default_committer_email: row.get(5).unwrap(),
        }
    }
}

impl Defaults {
    /// Set the default remote by name
    pub fn set_default_remote(
        conn: &ConfigConnection,
        remote_name: Option<&str>,
    ) -> Result<(), RemoteError> {
        // If setting a remote name, validate that it exists
        if let Some(name) = remote_name {
            Remote::get_by_name(conn, name)?;
        }

        let query = "UPDATE defaults SET remote_name = ?1 WHERE id = 1";
        let mut stmt = conn.prepare(query)?;
        stmt.execute(params![remote_name])?;
        Ok(())
    }

    pub fn set_default_remote_compat(
        conn: &ConfigConnection,
        remote_name: Option<&str>,
    ) -> SQLResult<()> {
        let query = "UPDATE defaults SET remote_name = ?1 WHERE id = 1";
        let mut stmt = conn.prepare(query)?;
        stmt.execute(params![remote_name])?;
        Ok(())
    }

    /// Get the default remote name
    pub fn get_default_remote(conn: &ConfigConnection) -> Option<String> {
        let query = "SELECT remote_name FROM defaults WHERE id = 1";
        let mut stmt = conn.prepare(query).ok()?;
        let mut rows = stmt
            .query_map(params![], |row| row.get::<_, Option<String>>(0))
            .ok()?;

        if let Some(Ok(remote_name)) = rows.next() {
            remote_name
        } else {
            None
        }
    }

    /// Helper method to get the default remote URL by resolving the remote name
    pub fn get_default_remote_url(conn: &ConfigConnection) -> Option<String> {
        if let Some(remote_name) = Self::get_default_remote(conn) {
            if let Some(remote) = Remote::get_by_name_optional(conn, &remote_name) {
                Some(remote.url)
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn set_current_branch(conn: &ConfigConnection, branch_name: Option<&str>) -> SQLResult<()> {
        let query = "UPDATE defaults SET current_branch_name = ?1 WHERE id = 1";
        let mut stmt = conn.prepare(query)?;
        stmt.execute(params![branch_name])?;
        Ok(())
    }

    pub fn get_current_branch(conn: &ConfigConnection) -> Option<String> {
        let query = "SELECT current_branch_name FROM defaults WHERE id = 1";
        let mut stmt = conn.prepare(query).ok()?;
        let mut rows = stmt
            .query_map(params![], |row| row.get::<_, Option<String>>(0))
            .ok()?;

        if let Some(Ok(branch_name)) = rows.next() {
            branch_name
        } else {
            None
        }
    }

    pub fn set_default_committer_name(conn: &ConfigConnection, name: &str) -> SQLResult<()> {
        let query = "UPDATE defaults SET default_committer_name = ?1 WHERE id = 1";
        let mut stmt = conn.prepare(query)?;
        stmt.execute(params![name])?;
        Ok(())
    }

    pub fn get_default_committer_name(conn: &ConfigConnection) -> String {
        conn.query_row(
            "SELECT default_committer_name FROM defaults WHERE id = 1",
            [],
            |row| row.get::<_, String>(0),
        )
        .unwrap_or_else(|_| GEN_DEFAULT_COMMITTER_NAME.to_string())
    }

    pub fn set_default_committer_email(conn: &ConfigConnection, email: &str) -> SQLResult<()> {
        let query = "UPDATE defaults SET default_committer_email = ?1 WHERE id = 1";
        let mut stmt = conn.prepare(query)?;
        stmt.execute(params![email])?;
        Ok(())
    }

    pub fn get_default_committer_email(conn: &ConfigConnection) -> String {
        conn.query_row(
            "SELECT default_committer_email FROM defaults WHERE id = 1",
            [],
            |row| row.get::<_, String>(0),
        )
        .unwrap_or_else(|_| DEFAULT_COMMITTER_EMAIL.to_string())
    }

    /// Get the defaults record
    pub fn get(conn: &ConfigConnection) -> Option<Defaults> {
        let query = "SELECT id, collection_name, remote_name, current_branch_name, default_committer_name, default_committer_email FROM defaults WHERE id = 1";
        Self::get_single(conn, query, params![]).ok()
    }

    /// Helper method to get a single defaults record using the Query trait
    fn get_single(
        conn: &ConfigConnection,
        query: &str,
        params: &[&dyn rusqlite::ToSql],
    ) -> SQLResult<Defaults> {
        let mut stmt = conn.prepare(query)?;
        let mut rows = stmt.query_map(params, |row| Ok(Self::process_row(row)))?;

        if let Some(row) = rows.next() {
            row
        } else {
            Err(rusqlite::Error::QueryReturnedNoRows)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        io::{Cursor, Write},
        path::PathBuf,
        slice::from_ref,
    };

    use tempfile::NamedTempFile;

    use super::*;
    use crate::{
        assets::{AssetRef, AssetRole, OperationAsset, OperationLog},
        history::{HistoryStore, dolt::DoltHistoryStore},
        test_helpers::setup_gen,
        traits::Query,
    };

    #[cfg(test)]
    mod defaults {
        use super::*;

        #[test]
        fn test_default_remote_functionality() {
            let context = setup_gen();
            let config_conn = context.config().conn();

            // Create test remotes
            Remote::create(config_conn, "origin", "https://example.com/repo.gen").unwrap();
            Remote::create(config_conn, "upstream", "https://upstream.com/repo.gen").unwrap();

            // Test getting default remote when none is set
            assert_eq!(Defaults::get_default_remote(config_conn), None);
            assert_eq!(Defaults::get_default_remote_url(config_conn), None);

            // Test setting default remote
            Defaults::set_default_remote(config_conn, Some("origin")).unwrap();
            assert_eq!(
                Defaults::get_default_remote(config_conn),
                Some("origin".to_string())
            );
            assert_eq!(
                Defaults::get_default_remote_url(config_conn),
                Some("https://example.com/repo.gen".to_string())
            );

            // Test changing default remote
            Defaults::set_default_remote(config_conn, Some("upstream")).unwrap();
            assert_eq!(
                Defaults::get_default_remote(config_conn),
                Some("upstream".to_string())
            );
            assert_eq!(
                Defaults::get_default_remote_url(config_conn),
                Some("https://upstream.com/repo.gen".to_string())
            );

            // Test clearing default remote
            Defaults::set_default_remote(config_conn, None).unwrap();
            assert_eq!(Defaults::get_default_remote(config_conn), None);
            assert_eq!(Defaults::get_default_remote_url(config_conn), None);

            // Test getting URL for non-existent remote (using the compat method to bypass validation)
            Defaults::set_default_remote_compat(config_conn, Some("nonexistent")).unwrap();
            assert_eq!(
                Defaults::get_default_remote(config_conn),
                Some("nonexistent".to_string())
            );
            assert_eq!(Defaults::get_default_remote_url(config_conn), None);
        }

        #[test]
        fn test_defaults_get() {
            let context = setup_gen();
            let config_conn = context.config().conn();

            // Test getting defaults record
            let defaults = Defaults::get(config_conn).unwrap();
            assert_eq!(defaults.id, 1);
            assert_eq!(defaults.collection_name, None);
            assert_eq!(defaults.remote_name, None);
            assert_eq!(defaults.current_branch_name, None);
            assert_eq!(defaults.default_committer_name, GEN_DEFAULT_COMMITTER_NAME);
            assert_eq!(defaults.default_committer_email, DEFAULT_COMMITTER_EMAIL);

            // Set a default remote and test again (using compat method to bypass validation)
            Defaults::set_default_remote_compat(config_conn, Some("test-remote")).unwrap();
            let defaults = Defaults::get(config_conn).unwrap();
            assert_eq!(defaults.remote_name, Some("test-remote".to_string()));
        }

        #[test]
        fn test_current_branch_round_trip() {
            let context = setup_gen();
            let config_conn = context.config().conn();

            assert_eq!(Defaults::get_current_branch(config_conn), None);

            Defaults::set_current_branch(config_conn, Some("feature")).unwrap();
            assert_eq!(
                Defaults::get_current_branch(config_conn),
                Some("feature".to_string())
            );

            Defaults::set_current_branch(config_conn, None).unwrap();
            assert_eq!(Defaults::get_current_branch(config_conn), None);
        }

        #[test]
        fn test_default_committer_round_trip() {
            let context = setup_gen();
            let config_conn = context.config().conn();

            assert_eq!(
                Defaults::get_default_committer_name(config_conn),
                GEN_DEFAULT_COMMITTER_NAME
            );
            assert_eq!(
                Defaults::get_default_committer_email(config_conn),
                DEFAULT_COMMITTER_EMAIL
            );

            Defaults::set_default_committer_name(config_conn, "Test User").unwrap();
            Defaults::set_default_committer_email(config_conn, "test@example.com").unwrap();

            assert_eq!(
                Defaults::get_default_committer_name(config_conn),
                "Test User".to_string()
            );
            assert_eq!(
                Defaults::get_default_committer_email(config_conn),
                "test@example.com".to_string()
            );
            let defaults = Defaults::get(config_conn).expect("should load defaults");
            assert_eq!(defaults.default_committer_name, "Test User");
            assert_eq!(defaults.default_committer_email, "test@example.com");
        }
    }

    #[cfg(test)]
    mod remote_operations {
        use gen_core::DoltHashId;
        use uuid::Uuid;

        use crate::{
            operations::{Remote, RemoteOperationKind, RemoteOperationRecord},
            test_helpers::setup_gen,
            traits::Query,
        };

        #[test]
        fn test_remote_operation_resumes_until_completed() {
            let context = setup_gen();
            let config = context.config().conn();
            Remote::create(config, "origin", "https://example.com/repo")
                .expect("should create remote");
            let original_commit = DoltHashId([1_u8; 20]);
            let advanced_commit = DoltHashId([2_u8; 20]);

            let mut baseline = RemoteOperationRecord::begin_or_resume(
                config,
                "origin",
                "main",
                RemoteOperationKind::Clone,
                Some(&original_commit),
            )
            .expect("should begin clone operation");
            assert_eq!(baseline.from_commit, Some(original_commit));
            assert_eq!(baseline.assets_transfer_checkpoint, None);
            baseline
                .set_destination(config, &original_commit)
                .expect("should record clone destination");
            baseline
                .advance_assets_transfer_checkpoint(config, &original_commit)
                .expect("should record clone asset checkpoint");
            baseline
                .complete(config)
                .expect("should complete clone operation");

            let operation = RemoteOperationRecord::begin_or_resume(
                config,
                "origin",
                "main",
                RemoteOperationKind::Pull,
                Some(&original_commit),
            )
            .expect("should begin pull operation");
            let mut resumed = RemoteOperationRecord::begin_or_resume(
                config,
                "origin",
                "main",
                RemoteOperationKind::Pull,
                Some(&advanced_commit),
            )
            .expect("should resume pull operation");

            assert_eq!(resumed, operation);
            assert_eq!(resumed.from_commit, Some(original_commit));
            assert_eq!(resumed.assets_transfer_checkpoint, Some(original_commit));

            resumed
                .set_destination(config, &advanced_commit)
                .expect("should record pull destination");
            resumed
                .advance_assets_transfer_checkpoint(config, &advanced_commit)
                .expect("should record pull asset checkpoint");
            resumed
                .complete(config)
                .expect("should complete pull operation");
            let completed = RemoteOperationRecord::get_by_id(config, &resumed.id, None)
                .expect("should refetch completed pull operation");
            assert!(
                completed.completed_at.is_some(),
                "completed_at should be set"
            );
            assert_eq!(completed.failed_at, None, "failed_at should remain unset");

            let next = RemoteOperationRecord::begin_or_resume(
                config,
                "origin",
                "main",
                RemoteOperationKind::Pull,
                Some(&advanced_commit),
            )
            .expect("should begin the next pull operation");
            assert_ne!(next.id, operation.id);
            assert_eq!(next.from_commit, Some(advanced_commit));
            assert_eq!(next.assets_transfer_checkpoint, Some(advanced_commit));
        }

        #[test]
        fn test_failed_graph_operation_does_not_resume() {
            let context = setup_gen();
            let config = context.config().conn();
            Remote::create(config, "origin", "https://example.com/repo")
                .expect("should create remote");
            let original_commit = DoltHashId([1_u8; 20]);

            let failed = RemoteOperationRecord::begin_or_resume(
                config,
                "origin",
                "main",
                RemoteOperationKind::Pull,
                Some(&original_commit),
            )
            .expect("should begin pull operation");
            failed.fail(config).expect("should fail pull operation");
            let next = RemoteOperationRecord::begin_or_resume(
                config,
                "origin",
                "main",
                RemoteOperationKind::Pull,
                Some(&original_commit),
            )
            .expect("should begin replacement pull operation");

            assert_ne!(next.id, failed.id);
        }

        #[test]
        fn test_remote_operation_cannot_complete_before_asset_checkpoint() {
            let context = setup_gen();
            let config = context.config().conn();
            Remote::create(config, "origin", "https://example.com/repo")
                .expect("should create remote");
            let destination_commit = DoltHashId([1_u8; 20]);
            let mut operation = RemoteOperationRecord::begin_or_resume(
                config,
                "origin",
                "main",
                RemoteOperationKind::Clone,
                None,
            )
            .expect("should begin clone operation");
            operation
                .set_destination(config, &destination_commit)
                .expect("should record graph destination");

            operation
                .complete(config)
                .expect_err("should require assets to reach graph destination");
            let failed = RemoteOperationRecord::get_by_id(config, &operation.id, None)
                .expect("should refetch failed clone operation");
            assert!(failed.failed_at.is_some(), "failed_at should be set");
        }

        #[test]
        fn test_push_operation_resumes_transfer_lease() {
            let context = setup_gen();
            let config = context.config().conn();
            Remote::create(config, "origin", "https://example.com/repo")
                .expect("should create remote");
            let destination_commit = DoltHashId([3_u8; 20]);
            let transfer_id = Uuid::from_u128(7);
            let transfer_expires_at = 1_900_000_000;
            let mut operation = RemoteOperationRecord::begin_or_resume(
                config,
                "origin",
                "main",
                RemoteOperationKind::Push,
                None,
            )
            .expect("should begin push operation");
            operation
                .set_push_destination(
                    config,
                    &destination_commit,
                    transfer_id,
                    transfer_expires_at,
                )
                .expect("should record push destination and lease");

            let mut resumed = RemoteOperationRecord::begin_or_resume(
                config,
                "origin",
                "main",
                RemoteOperationKind::Push,
                None,
            )
            .expect("should resume push operation");

            assert_eq!(resumed.to_commit.as_ref(), Some(&destination_commit));
            assert_eq!(resumed.transfer_id, Some(transfer_id));
            assert_eq!(resumed.transfer_expires_at, Some(transfer_expires_at));
            resumed
                .advance_assets_transfer_checkpoint(config, &destination_commit)
                .expect("should record completed push assets");
            resumed
                .complete(config)
                .expect("should complete push operation");
        }
    }

    #[cfg(test)]
    mod remote {
        use super::*;

        #[test]
        fn test_validate_remote_name() {
            // Valid names
            assert!(Remote::validate_name("origin").is_ok());
            assert!(Remote::validate_name("my-remote").is_ok());
            assert!(Remote::validate_name("remote_1").is_ok());
            assert!(Remote::validate_name("test123").is_ok());

            // Invalid names
            assert!(Remote::validate_name("").is_err());
            assert!(Remote::validate_name("remote with spaces").is_err());
            assert!(Remote::validate_name("remote@special").is_err());
            assert!(Remote::validate_name("remote.dot").is_err());
        }

        #[test]
        fn test_validate_url() {
            // Valid URLs
            assert!(Remote::validate_url("https://genhub.bio/user/repo.gen").is_ok());
            assert!(Remote::validate_url("http://example.com/repo").is_ok());
            assert!(Remote::validate_url("ssh://git@genhub.bio/user/repo.gen").is_ok());
            assert!(Remote::validate_url("/path/to/local/repo").is_ok());
            assert!(Remote::validate_url("user@host:path/to/repo").is_ok());

            // Invalid URLs
            assert!(Remote::validate_url("").is_err());
            assert!(Remote::validate_url("not-a-url").is_err());

            assert!(Remote::validate_url("ftp://invalid-protocol.com").is_err());
        }
    }

    mod remote_branch {
        use super::*;

        #[test]
        fn test_remote_branch_set_remote_valid() {
            let context = setup_gen();
            let config_conn = context.config().conn();

            Remote::create(config_conn, "origin", "https://genhub.bio/user/repo.gen").unwrap();
            let result =
                RemoteBranch::set_remote_validated(config_conn, "test_branch", Some("origin"));
            assert!(result.is_ok());
            assert_eq!(
                RemoteBranch::get_remote(config_conn, "test_branch"),
                Some("origin".to_string())
            );
        }

        #[test]
        fn test_remote_branch_set_remote_nonexistent() {
            let context = setup_gen();
            let config_conn = context.config().conn();

            let result =
                RemoteBranch::set_remote_validated(config_conn, "test_branch", Some("nonexistent"));
            assert_eq!(
                result,
                Err(RemoteError::RemoteNotFound("nonexistent".to_string()))
            );
            assert_eq!(RemoteBranch::get_remote(config_conn, "test_branch"), None);
        }

        #[test]
        fn test_remote_branch_clear_remote() {
            let context = setup_gen();
            let config_conn = context.config().conn();

            Remote::create(config_conn, "origin", "https://genhub.bio/user/repo.gen").unwrap();
            RemoteBranch::set_remote_validated(config_conn, "test_branch", Some("origin")).unwrap();
            assert_eq!(
                RemoteBranch::get_remote(config_conn, "test_branch"),
                Some("origin".to_string())
            );
            RemoteBranch::set_remote_validated(config_conn, "test_branch", None).unwrap();
            assert_eq!(RemoteBranch::get_remote(config_conn, "test_branch"), None);
        }

        #[test]
        fn test_remote_branch_clears_on_remote_delete() {
            let context = setup_gen();
            let config_conn = context.config().conn();

            Remote::create(config_conn, "origin", "https://genhub.bio/user/repo.gen").unwrap();
            RemoteBranch::set_remote_validated(config_conn, "test_branch_cascade", Some("origin"))
                .unwrap();
            assert_eq!(
                RemoteBranch::get_remote(config_conn, "test_branch_cascade"),
                Some("origin".to_string())
            );

            Remote::delete(config_conn, "origin").unwrap();
            assert_eq!(
                RemoteBranch::get_remote(config_conn, "test_branch_cascade"),
                None
            );
        }
    }

    #[test]
    fn test_remote_create() {
        let context = setup_gen();
        let config_conn = context.config().conn();

        // Test successful remote creation
        let remote = Remote::create(config_conn, "origin", "https://example.com/repo.gen").unwrap();
        assert_eq!(remote.name, "origin");
        assert_eq!(remote.url, "https://example.com/repo.gen");

        // Test duplicate name constraint violation
        let result = Remote::create(config_conn, "origin", "https://different.com/repo.gen");
        assert!(result.is_err());
    }

    #[test]
    fn test_remote_get_by_name() {
        let context = setup_gen();
        let config_conn = context.config().conn();

        // Test getting non-existent remote
        let result = Remote::get_by_name_optional(config_conn, "nonexistent");
        assert!(result.is_none());

        // Create a remote and test retrieval
        Remote::create(config_conn, "upstream", "https://upstream.com/repo.gen").unwrap();
        let result = Remote::get_by_name_optional(config_conn, "upstream");
        assert!(result.is_some());
        let remote = result.unwrap();
        assert_eq!(remote.name, "upstream");
        assert_eq!(remote.url, "https://upstream.com/repo.gen");
    }

    #[test]
    fn test_remote_list_all() {
        let context = setup_gen();
        let config_conn = context.config().conn();

        // Test empty list
        let remotes = Remote::list_all(config_conn);
        assert!(remotes.is_empty());

        // Create multiple remotes
        Remote::create(config_conn, "origin", "https://origin.com/repo.gen").unwrap();
        Remote::create(config_conn, "upstream", "https://upstream.com/repo.gen").unwrap();
        Remote::create(config_conn, "fork", "https://fork.com/repo.gen").unwrap();

        // Test list returns all remotes in alphabetical order
        let remotes = Remote::list_all(config_conn);
        assert_eq!(remotes.len(), 3);
        assert_eq!(remotes[0].name, "fork");
        assert_eq!(remotes[1].name, "origin");
        assert_eq!(remotes[2].name, "upstream");
    }

    #[test]
    fn test_remote_delete() {
        let context = setup_gen();
        let config_conn = context.config().conn();

        // Create a remote
        Remote::create(config_conn, "temp", "https://temp.com/repo.gen").unwrap();

        // Verify it exists
        let remote = Remote::get_by_name_optional(config_conn, "temp");
        assert!(remote.is_some());

        // Delete the remote
        let result = Remote::delete(config_conn, "temp");
        assert!(result.is_ok());

        // Verify it's gone
        let remote = Remote::get_by_name_optional(config_conn, "temp");
        assert!(remote.is_none());

        // Test deleting non-existent remote (should return error)
        let result = Remote::delete(config_conn, "nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_remote_delete_with_remote_branch_associations() {
        let context = setup_gen();
        let config_conn = context.config().conn();

        Remote::create(config_conn, "test_remote", "https://test.com/repo.gen").unwrap();
        RemoteBranch::set_remote_validated(config_conn, "test_branch", Some("test_remote"))
            .unwrap();
        assert_eq!(
            RemoteBranch::get_remote(config_conn, "test_branch"),
            Some("test_remote".to_string())
        );

        let result = Remote::delete(config_conn, "test_remote");
        assert!(result.is_ok());
        assert_eq!(RemoteBranch::get_remote(config_conn, "test_branch"), None);
        let remote = Remote::get_by_name_optional(config_conn, "test_remote");
        assert!(remote.is_none());
    }

    #[test]
    fn test_calculate_stream_hash() {
        let content = b"Hello, World!";
        let cursor = Cursor::new(content);
        let hash = calculate_stream_hash(cursor).unwrap();

        assert_eq!(hash.len(), 32);

        // Test consistency - same content should produce same hash
        let cursor2 = Cursor::new(content);
        let hash2 = calculate_stream_hash(cursor2).unwrap();
        assert_eq!(hash, hash2);

        // Test different content produces different hash
        let different_content = b"Hello, World!!";
        let cursor3 = Cursor::new(different_content);
        let hash3 = calculate_stream_hash(cursor3).unwrap();
        assert_ne!(hash, hash3);
    }

    #[test]
    fn test_calculate_file_checksum() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let content = b"Test file content for checksum calculation";
        temp_file.write_all(content).unwrap();
        temp_file.flush().unwrap();

        let checksum = calculate_file_checksum(temp_file.path()).unwrap();

        assert_eq!(checksum.0.len(), 32);

        // Test consistency - same file should produce same checksum
        let checksum2 = calculate_file_checksum(temp_file.path()).unwrap();
        assert_eq!(checksum, checksum2);

        // Test with different file content
        let mut temp_file2 = NamedTempFile::new().unwrap();
        let different_content = b"Different test file content";
        temp_file2.write_all(different_content).unwrap();
        temp_file2.flush().unwrap();

        let checksum3 = calculate_file_checksum(temp_file2.path()).unwrap();
        assert_ne!(checksum, checksum3);
    }

    #[test]
    fn test_calculate_file_checksum_nonexistent_file() {
        let result = calculate_file_checksum("/nonexistent/file/path");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err().kind(),
            std::io::ErrorKind::NotFound
        ));
    }

    #[test]
    fn test_operation_file_new_infers_file_type_from_path() {
        let operation_file = OperationFile::new("fixtures/sample.gb");

        assert_eq!(operation_file.filename, "sample.gb");
        assert_eq!(operation_file.file_path, "fixtures/sample.gb");
        assert_eq!(operation_file.file_type, FileTypes::GenBank);
        assert_eq!(operation_file.checksum_override, None);
    }

    #[test]
    fn test_operation_file_storage_path_normalizes_absolute_repo_path() {
        let context = setup_gen();
        let repo_root = context.workspace().repo_root().unwrap();
        let absolute_path = repo_root.join("nested").join("sample.fa");

        fs::create_dir_all(absolute_path.parent().unwrap()).unwrap();
        fs::write(&absolute_path, b"sample").unwrap();

        let checksum = calculate_file_checksum(&absolute_path).unwrap();
        let storage_path = OperationFile::storage_file_path(
            context.workspace(),
            &absolute_path.to_string_lossy(),
            Some(&checksum),
        )
        .unwrap();

        assert_eq!(storage_path, "nested/sample.fa");
    }

    #[test]
    fn operation_add_file_keeps_compression_suffix_for_asset_path() {
        let context = setup_gen();
        let fixture_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures/simple.fa.bgz");
        let outside_dir = tempfile::tempdir().unwrap();
        let outside_path = outside_dir.path().join("simple.fa.bgz");
        fs::copy(&fixture_path, &outside_path).unwrap();
        let outside_path_string = outside_path.to_string_lossy().to_string();

        let file_addition = FileAddition::prepare(
            context.workspace(),
            &outside_path_string,
            FileTypes::Fasta,
            None,
        )
        .unwrap();
        let storage_path = OperationFile::storage_file_path(
            context.workspace(),
            &outside_path_string,
            file_addition.checksum.as_ref(),
        )
        .unwrap();
        let checksum = file_addition
            .checksum
            .expect("local file addition should have a checksum");

        assert_eq!(storage_path, format!(".gen/assets/{checksum}.fa.bgz"));
        assert_eq!(
            file_addition.asset_uri,
            LocalAssetUri::asset_uri(".gen/outside_root/simple.fa.bgz"),
        );
    }

    #[test]
    fn test_file_addition_prepare() {
        let context = setup_gen();
        let repo_root = context.workspace().repo_root().unwrap();

        let file1_path = repo_root.join("test_file.txt");
        fs::write(&file1_path, b"Test file content").unwrap();
        let file1_path_str = file1_path.to_string_lossy().to_string();
        let fa1 =
            FileAddition::prepare(context.workspace(), &file1_path_str, FileTypes::Fasta, None)
                .expect("should prepare file addition");

        let checksum = calculate_file_checksum(&file1_path_str).unwrap();
        let expected_asset_path = format!(".gen/assets/{checksum}.txt");
        assert_eq!(
            fa1.asset_uri,
            LocalAssetUri::asset_uri(&expected_asset_path)
        );
        assert_eq!(fa1.file_path(), expected_asset_path);

        let relative1_id = LocalAssetUri::generate_file_addition_id(
            Some(&checksum),
            &LocalAssetUri::asset_uri(&expected_asset_path),
        );

        assert_eq!(fa1.id, relative1_id);
        assert!(
            context
                .workspace()
                .asset_dir()
                .unwrap()
                .join(
                    fa1.hashed_filename()
                        .expect("local file addition should have a checksum"),
                )
                .exists()
        );

        // Second call with same file should return the same FileAddition
        let fa2 =
            FileAddition::prepare(context.workspace(), &file1_path_str, FileTypes::Fasta, None)
                .expect("should prepare same file addition");

        assert_eq!(fa1, fa2);

        let file2_path = repo_root.join("nested").join("file2.txt");
        fs::create_dir_all(file2_path.parent().unwrap()).unwrap();
        fs::write(&file2_path, b"Test file content").unwrap();
        let file2_path_str = file2_path.to_string_lossy().to_string();

        let fa3 =
            FileAddition::prepare(context.workspace(), &file2_path_str, FileTypes::Fasta, None)
                .expect("should prepare matching file addition");

        assert_eq!(fa1.id, fa3.id);

        fs::write(&file1_path, b"new content").unwrap();
        let fa1_new =
            FileAddition::prepare(context.workspace(), &file1_path_str, FileTypes::Fasta, None)
                .expect("should prepare updated file addition");

        assert_ne!(fa1.id, fa1_new.id);

        let outside_dir = tempfile::tempdir().unwrap();
        let outside_file = outside_dir.path().join("simple.fa");
        fs::write(&outside_file, b"Outside repo file").unwrap();
        let outside_path = outside_file.to_string_lossy().to_string();

        let outside =
            FileAddition::prepare(context.workspace(), &outside_path, FileTypes::Fasta, None)
                .expect("should prepare external file addition");

        assert_eq!(
            outside.asset_uri,
            LocalAssetUri::asset_uri(".gen/outside_root/simple.fa")
        );
        assert_eq!(outside.file_path(), ".gen/outside_root/simple.fa");
        assert!(
            context
                .workspace()
                .asset_dir()
                .unwrap()
                .join(
                    outside
                        .hashed_filename()
                        .expect("local file addition should have a checksum"),
                )
                .exists()
        );
    }

    #[test]
    fn test_file_addition_prepare_http_uri() {
        let context = setup_gen();
        let asset_uri = "https://example.com/assets/reference.fa";

        let addition =
            FileAddition::prepare(context.workspace(), asset_uri, FileTypes::Fasta, None)
                .expect("should prepare HTTP file addition");

        assert_eq!(addition.asset_uri, asset_uri);
        assert_eq!(addition.file_path(), asset_uri);
        assert_eq!(addition.checksum, None);
        assert!(
            fs::read_dir(context.workspace().asset_dir().unwrap())
                .unwrap()
                .next()
                .is_none()
        );
    }

    #[test]
    fn test_file_addition_prepare_remote_uri() {
        let context = setup_gen();
        let asset_uri = "s3://bucket/reference.fa";
        let checksum = Sha256Hash::convert_str("remote S3 contents");

        let addition = FileAddition::prepare(
            context.workspace(),
            asset_uri,
            FileTypes::Fasta,
            Some(checksum),
        )
        .expect("should prepare remote file addition");

        assert_eq!(addition.asset_uri, asset_uri);
        assert_eq!(addition.file_path(), asset_uri);
        assert_eq!(addition.checksum, Some(checksum));
    }

    #[test]
    fn test_file_addition_rejects_incorrect_local_checksum_override() {
        let context = setup_gen();
        let path = context.workspace().repo_root().unwrap().join("asset.fa");
        fs::write(&path, "actual contents").expect("should write local asset");

        let error = FileAddition::prepare(
            context.workspace(),
            path.to_str().expect("should encode local asset path"),
            FileTypes::Fasta,
            Some(Sha256Hash::convert_str("different contents")),
        )
        .expect_err("should reject incorrect checksum override");

        assert!(
            matches!(error, FileAdditionError::ChecksumError(_)),
            "should report a checksum error: {error}"
        );
        assert_eq!(
            fs::read_dir(context.workspace().asset_dir().unwrap())
                .expect("should read asset directory")
                .count(),
            0
        );
    }

    #[test]
    fn test_add_files_operation_does_not_read_remote_asset() {
        let context = setup_gen();
        let asset_uri = "http://127.0.0.1:1/asset.fa".to_string();
        let operation_file = OperationFile::new(&asset_uri);

        add_files_operation(
            &context,
            from_ref(&operation_file),
            Some("track remote reference"),
        )
        .expect("should track remote asset");

        let asset_refs = AssetRef::all(context.graph().conn());
        assert_eq!(asset_refs.len(), 1);
        assert_eq!(asset_refs[0].uri, asset_uri);
        assert_eq!(asset_refs[0].checksum, None);
    }

    #[test]
    fn test_add_files_operation_passes_remote_checksum_override() {
        let context = setup_gen();
        let asset_uri = "http://127.0.0.1:1/asset.fa";
        let checksum = Sha256Hash::convert_str("known remote contents");
        let operation_file = OperationFile::new(asset_uri).set_checksum_override(checksum);

        add_files_operation(
            &context,
            from_ref(&operation_file),
            Some("track checksummed remote reference"),
        )
        .expect("should track remote asset without reading it");

        let asset_refs = AssetRef::all(context.graph().conn());
        assert_eq!(asset_refs.len(), 1);
        assert_eq!(asset_refs[0].uri, asset_uri);
        assert_eq!(asset_refs[0].checksum, Some(checksum));
    }

    #[test]
    fn test_add_files_operation_stores_shared_unmodified_asset_once_across_operations() {
        let context = setup_gen();
        let graph_conn = context.graph().conn();
        let workspace = context.workspace();

        let repo_root = workspace.repo_root().unwrap();
        fs::write(repo_root.join("shared.txt"), "shared contents").unwrap();
        fs::write(repo_root.join("unique.txt"), "unique contents").unwrap();

        let _operation_1_hash =
            add_files_operation(&context, &[OperationFile::new("shared.txt")], Some("first"))
                .unwrap();
        let _operation_2_hash = add_files_operation(
            &context,
            &[
                OperationFile::new("shared.txt"),
                OperationFile::new("unique.txt"),
            ],
            Some("second"),
        )
        .unwrap();

        let mut operation_logs = OperationLog::all(graph_conn);
        operation_logs.sort_by_key(|operation_log| operation_log.created_on);
        let mut first_assets = OperationAsset::by_log_id(graph_conn, &operation_logs[0].id);
        first_assets.sort_by_key(|asset| asset.asset_ref_id);
        let mut second_assets = OperationAsset::by_log_id(graph_conn, &operation_logs[1].id);
        second_assets.sort_by_key(|asset| asset.asset_ref_id);

        assert_eq!(operation_logs[0].command, "first");
        assert_eq!(operation_logs[1].command, "second");
        assert_eq!(first_assets.len(), 1);
        assert_eq!(second_assets.len(), 2);
        assert!(
            second_assets
                .iter()
                .any(|asset| asset.asset_ref_id == first_assets[0].asset_ref_id)
        );

        let shared_asset = AssetRef::get_by_id(graph_conn, &first_assets[0].asset_ref_id, None)
            .expect("should load shared asset ref");
        let unique_asset = second_assets
            .iter()
            .find(|asset| asset.asset_ref_id != first_assets[0].asset_ref_id)
            .and_then(|asset| AssetRef::get_by_id(graph_conn, &asset.asset_ref_id, None))
            .expect("should load unique asset ref");

        let assets_dir = workspace.asset_dir().unwrap();
        let asset_names = fs::read_dir(&assets_dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect::<Vec<_>>();

        assert_eq!(asset_names.len(), 2);
        let shared_asset_name = <dyn AssetUri>::from_uri(&shared_asset.uri)
            .hashed_filename(&shared_asset.checksum.unwrap());
        let unique_asset_name = <dyn AssetUri>::from_uri(&unique_asset.uri)
            .hashed_filename(&unique_asset.checksum.unwrap());
        assert!(asset_names.contains(&shared_asset_name));
        assert!(asset_names.contains(&unique_asset_name));
    }

    #[test]
    fn test_add_files_operation_tracks_distinct_filenames_for_same_asset()
    -> Result<(), Box<dyn std::error::Error>> {
        let context = setup_gen();
        let graph_conn = context.graph().conn();

        let outside_dir = tempfile::tempdir()?;
        let alpha_path = outside_dir.path().join("alpha.fa");
        let beta_path = outside_dir.path().join("beta.fa");
        fs::write(&alpha_path, "shared contents")?;
        fs::write(&beta_path, "shared contents")?;

        add_files_operation(
            &context,
            &[
                OperationFile::new(alpha_path.to_string_lossy()),
                OperationFile::new(beta_path.to_string_lossy()),
            ],
            Some("same asset, different filenames"),
        )?;

        let mut asset_refs = AssetRef::all(graph_conn);
        asset_refs.sort_by(|left, right| left.name.cmp(&right.name));

        assert_eq!(
            asset_refs
                .iter()
                .map(|asset| asset.name.clone().unwrap_or_default())
                .collect::<Vec<_>>(),
            vec!["alpha.fa".to_string(), "beta.fa".to_string()],
        );
        assert_ne!(asset_refs[0].id, asset_refs[1].id);
        assert_eq!(asset_refs[0].checksum, asset_refs[1].checksum);
        Ok(())
    }

    #[test]
    fn test_add_files_operation_tracks_graph_asset_refs() {
        let context = setup_gen();
        let graph_conn = context.graph().conn();

        let repo_root = context.workspace().repo_root().unwrap();
        fs::write(repo_root.join("alpha.fa"), "AAAA").unwrap();
        fs::write(repo_root.join("beta.fa"), "BBBB").unwrap();

        let _operation_hash = add_files_operation(
            &context,
            &[
                OperationFile::new("alpha.fa"),
                OperationFile::new("beta.fa"),
            ],
            Some("import references"),
        )
        .unwrap();

        let mut asset_refs = AssetRef::all(graph_conn);
        asset_refs.sort_by(|left, right| left.logical_path.cmp(&right.logical_path));
        let mut operation_logs = OperationLog::all(graph_conn);
        operation_logs.sort_by_key(|operation_log| operation_log.created_on);

        assert_eq!(asset_refs.len(), 2);
        assert_eq!(asset_refs[0].logical_path.as_deref(), Some("alpha.fa"));
        assert_eq!(asset_refs[0].name.as_deref(), Some("alpha.fa"));
        assert_eq!(asset_refs[0].role, AssetRole::Input);
        assert_eq!(asset_refs[1].logical_path.as_deref(), Some("beta.fa"));
        assert_eq!(asset_refs[1].name.as_deref(), Some("beta.fa"));
        assert_eq!(asset_refs[1].role, AssetRole::Input);
        assert_eq!(operation_logs.len(), 1);
        assert_eq!(operation_logs[0].operation_kind, OperationKind::AddFile);
        assert_eq!(operation_logs[0].command, "import references");
        let mut operation_assets = OperationAsset::by_log_id(graph_conn, &operation_logs[0].id);
        operation_assets.sort_by(|left, right| {
            left.role
                .as_str()
                .cmp(right.role.as_str())
                .then_with(|| left.asset_ref_id.cmp(&right.asset_ref_id))
        });
        assert_eq!(operation_assets.len(), 2);
        assert!(
            operation_assets
                .iter()
                .all(|asset| asset.role == AssetRole::Input)
        );
    }

    #[test]
    fn test_add_files_operation_records_history_without_config_db_rows() {
        let context = setup_gen();
        let graph_conn = context.graph().conn();
        let config_conn = context.config().conn();

        let repo_root = context.workspace().repo_root().unwrap();
        fs::write(repo_root.join("alpha.fa"), "AAAA").unwrap();

        let operation_hash = add_files_operation(
            &context,
            &[OperationFile::new("alpha.fa")],
            Some("import reference"),
        )
        .unwrap();

        let history_store = DoltHistoryStore::new(graph_conn);
        assert_eq!(
            history_store.current_head().unwrap(),
            Some(operation_hash),
            "add-file should commit through Dolt history"
        );
        let operation_count: i64 = config_conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'operations'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(operation_count, 0);
    }
}
