use std::{
    collections::{BTreeMap, HashSet},
    fmt::Write as _,
    fs::{self, File},
    io::{Read, Seek, Write},
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

use gen_core::{
    BranchName, CommitHash, CommitRef, HashId,
    errors::{ConfigError, ConnectionError},
};
use gen_models::{
    db::{DbContext, GraphConnection},
    errors::{FileAdditionError, OperationError},
    file_types::FileTypes,
    history::{HistoryStore, dolt::DoltHistoryStore},
    operations::OperationFileInfo,
};
use rusqlite::{Error as SQLError, OptionalExtension, params, types::Value};
use serde::{Deserialize, Serialize};
use tempfile::tempdir;
use thiserror::Error;
use zip::{CompressionMethod, ZipArchive, ZipWriter, result::ZipError, write::SimpleFileOptions};

const MANIFEST_ENTRY: &str = "manifest.json";
const PATCH_FORMAT_VERSION: u32 = 2;

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub(crate) struct PatchFile {
    file: OperationFileInfo,
    archive_path: String,
}

impl PatchFile {
    fn asset_entry_name(checksum: HashId, file_type: FileTypes) -> String {
        format!("assets/{checksum}.{}", FileTypes::suffix(file_type))
    }

    fn from_file_addition(
        context: &DbContext,
        file: OperationFileInfo,
    ) -> Result<Self, CreatePatchError> {
        if file.asset_uri.is_empty() {
            return Ok(Self {
                archive_path: String::new(),
                file,
            });
        }

        let source_asset_path = context
            .workspace()
            .asset_dir()?
            .join(file.clone().hashed_filename());
        let archive_path = if source_asset_path.exists() {
            Self::asset_entry_name(file.checksum, file.file_type)
        } else {
            String::new()
        };

        Ok(Self { file, archive_path })
    }

    fn source_asset_path(
        &self,
        context: &DbContext,
    ) -> Result<std::path::PathBuf, ConnectionError> {
        Ok(context
            .workspace()
            .asset_dir()?
            .join(self.file.hashed_filename()))
    }

    fn restore_from_archive<R>(
        &self,
        context: &DbContext,
        archive: &mut ZipArchive<R>,
    ) -> Result<bool, PatchError>
    where
        R: Read + Seek,
    {
        if self.file.asset_uri.is_empty() || self.archive_path.is_empty() {
            return Ok(false);
        }

        let asset_dir = context
            .workspace()
            .asset_dir()
            .map_err(ConnectionError::from)?;
        fs::create_dir_all(&asset_dir).map_err(|err| PatchError::Io(err.to_string()))?;

        let asset_path = asset_dir.join(self.file.clone().hashed_filename());
        if asset_path.exists() {
            return Ok(false);
        }

        let mut zip_file = archive.by_name(&self.archive_path)?;
        let mut asset_file =
            File::create(asset_path).map_err(|err| PatchError::Io(err.to_string()))?;
        std::io::copy(&mut zip_file, &mut asset_file)
            .map_err(|err| PatchError::Io(err.to_string()))?;
        Ok(true)
    }
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct PatchCommit {
    pub hash: HashId,
    pub parent_hash: Option<HashId>,
    pub change_type: String,
    pub commit_hash: CommitHash,
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct OperationPatch {
    pub commit: PatchCommit,
    pub(crate) files: Vec<PatchFile>,
    pub(crate) table_changes: Vec<PatchTableChange>,
    pub(crate) commit_message: String,
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub(crate) struct PatchTableChange {
    table_name: String,
    diff_type: String,
    predicate_columns: Vec<String>,
    before: Option<BTreeMap<String, PatchValue>>,
    after: Option<BTreeMap<String, PatchValue>>,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct OperationPatches {
    table_schemas: Vec<TableSchema>,
    pub base_commit_hash: Option<CommitHash>,
    pub target_commit_hash: CommitHash,
    pub patches: Vec<OperationPatch>,
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub(crate) enum PatchValue {
    Null,
    Integer(i64),
    Real(f64),
    Text(String),
    Blob(Vec<u8>),
}

#[derive(Serialize, Deserialize, Debug)]
struct PatchArchiveHeader {
    patch_format_version: u32,
}

#[derive(Serialize, Deserialize, Debug)]
struct VersionedOperationPatches {
    header: PatchArchiveHeader,
    manifest: OperationPatches,
}

#[derive(Debug, Error, PartialEq)]
pub enum PatchError {
    #[error("Connection Error: {0}")]
    ConnectionError(#[from] ConnectionError),
    #[error("Config Error: {0}")]
    ConfigError(#[from] ConfigError),
    #[error("I/O error: {0}")]
    Io(String),
    #[error("SQL Error: {0}")]
    SQLError(String),
    #[error("SQLite Error: {0}")]
    SqliteError(#[from] SQLError),
    #[error("Deserialization Error: {0}")]
    DeserializationError(String),
    #[error("Operation Error: {0}")]
    OperationError(#[from] OperationError),
    #[error("Zip Error: {0}")]
    Zip(String),
    #[error("Schema mismatch: {0}")]
    SchemaMismatch(String),
}

impl From<ZipError> for PatchError {
    fn from(value: ZipError) -> Self {
        PatchError::Zip(value.to_string())
    }
}

fn is_nothing_to_commit_error(error: &SQLError) -> bool {
    error
        .to_string()
        .contains("nothing to commit, working tree clean")
}

#[derive(Debug, Error)]
pub enum CreatePatchError {
    #[error("Operation {0} does not exist.")]
    OperationNotFound(HashId),
    #[error("Commit {0:?} does not exist.")]
    CommitNotFound(CommitHash),
    #[error("SQL Error: {0}")]
    SqliteError(#[from] SQLError),
    #[error("I/O Error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Config error: {0}")]
    Config(#[from] ConfigError),
    #[error("Connection error: {0}")]
    Connection(#[from] ConnectionError),
    #[error("Cap'n Proto error: {0}")]
    Capnp(#[from] capnp::Error),
    #[error("Zip error: {0}")]
    Zip(#[from] ZipError),
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("File addition error: {0}")]
    FileAddition(#[from] FileAdditionError),
    #[error("Invalid operation hash for patch history: {0}")]
    InvalidOperationHash(String),
}

fn serialize_operation_patches(
    operation_patches: &OperationPatches,
) -> Result<Vec<u8>, CreatePatchError> {
    Ok(serde_json::to_vec_pretty(&VersionedOperationPatches {
        header: PatchArchiveHeader {
            patch_format_version: PATCH_FORMAT_VERSION,
        },
        manifest: OperationPatches {
            table_schemas: operation_patches.table_schemas.clone(),
            base_commit_hash: operation_patches.base_commit_hash.clone(),
            target_commit_hash: operation_patches.target_commit_hash.clone(),
            patches: operation_patches.patches.clone(),
        },
    })?)
}

fn read_operation_patches<R>(archive: &mut ZipArchive<R>) -> Result<OperationPatches, PatchError>
where
    R: Read + Seek,
{
    let mut operation_patches_file = archive.by_name(MANIFEST_ENTRY)?;
    let mut buffer = Vec::new();
    operation_patches_file
        .read_to_end(&mut buffer)
        .map_err(|err| PatchError::Io(err.to_string()))?;
    drop(operation_patches_file);

    serde_json::from_slice::<VersionedOperationPatches>(&buffer)
        .map(|versioned| versioned.manifest)
        .map_err(|err| PatchError::DeserializationError(err.to_string()))
}

fn padded_hash(commit_hash: &CommitHash) -> HashId {
    HashId::pad_str(&commit_hash.0)
}

fn commit_hash_from_hash(hash: HashId) -> Result<CommitHash, CreatePatchError> {
    let hash = hash.to_string();
    if hash.len() < 40 {
        return Err(CreatePatchError::InvalidOperationHash(hash));
    }
    let commit_hash = hash[hash.len() - 40..].to_string();
    if !commit_hash
        .chars()
        .all(|character| character.is_ascii_hexdigit())
    {
        return Err(CreatePatchError::InvalidOperationHash(hash));
    }
    Ok(CommitHash(commit_hash))
}

fn snapshot_branch_name(history_ref: &str) -> String {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("should have a valid system clock")
        .as_nanos();
    let sanitized_ref = history_ref
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    format!("codex_patch_snapshot_{sanitized_ref}_{timestamp}")
}

fn open_snapshot_at_ref(
    graph_db_path: &Path,
    history_ref: &str,
    current_branch: Option<&str>,
) -> Result<GraphConnection, CreatePatchError> {
    let snapshot_dir = tempdir()?.keep();
    let snapshot_path = snapshot_dir.join("default.db");
    std::fs::copy(graph_db_path, &snapshot_path)?;
    let snapshot = crate::get_connection(&snapshot_path)?;
    if let Some(current_branch) = current_branch {
        gen_models::history::dolt::connect_branch(&snapshot, current_branch)?;
    }
    let history_store = DoltHistoryStore::new(&snapshot);
    let snapshot_branch = snapshot_branch_name(history_ref);
    history_store.create_branch(
        &BranchName(snapshot_branch.clone()),
        Some(&CommitRef(history_ref.to_string())),
    )?;
    history_store.checkout_branch(&BranchName(snapshot_branch))?;
    Ok(snapshot)
}

fn graph_file_type(file_type: &str) -> Option<FileTypes> {
    match file_type {
        "gb" => Some(FileTypes::GenBank),
        "fasta" => Some(FileTypes::Fasta),
        "gfa" => Some(FileTypes::GFA),
        "gaf" => Some(FileTypes::GAF),
        "vcf" => Some(FileTypes::VCF),
        "changeset" => Some(FileTypes::Changeset),
        "csv" => Some(FileTypes::CSV),
        "gff3" => Some(FileTypes::Gff3),
        "bed" => Some(FileTypes::Bed),
        "tabix" => Some(FileTypes::Tabix),
        "none" => Some(FileTypes::None),
        _ => None,
    }
}

fn patch_file_path(asset_uri: &str, logical_path: Option<&str>) -> String {
    logical_path.unwrap_or(asset_uri).to_string()
}

fn patch_file_name(asset_uri: &str, logical_path: Option<&str>, name: Option<&str>) -> String {
    if let Some(name) = name {
        return name.to_string();
    }

    let path = logical_path.unwrap_or(asset_uri.strip_prefix("file://").unwrap_or(asset_uri));
    Path::new(path)
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or(path)
        .to_string()
}

fn query_patch_files_for_logs(
    conn: &GraphConnection,
    log_ids: &[HashId],
) -> Result<Vec<OperationFileInfo>, CreatePatchError> {
    if log_ids.is_empty() {
        return Ok(Vec::new());
    }

    let mut statement = conn.prepare(
        "SELECT asset_refs.id, asset_refs.uri, asset_refs.file_type, asset_refs.checksum, \
         asset_refs.logical_path, asset_refs.name \
         FROM gen_operation_assets operation_assets \
         JOIN gen_asset_refs asset_refs ON asset_refs.id = operation_assets.asset_ref_id \
         WHERE operation_assets.log_id IN rarray(?1) \
         ORDER BY asset_refs.created_on, asset_refs.name, operation_assets.role",
    )?;
    let log_values = log_ids
        .iter()
        .map(|log_id| Value::from(*log_id))
        .collect::<Vec<_>>();
    statement
        .query_map(
            params![rusqlite::vtab::array::Array::new(log_values)],
            |row| {
                let asset_uri = row.get::<_, String>(1)?;
                let file_type = row.get::<_, String>(2)?;
                let logical_path = row.get::<_, Option<String>>(4)?;
                let name = row.get::<_, Option<String>>(5)?;
                let file_type = graph_file_type(&file_type).ok_or_else(|| {
                    rusqlite::Error::InvalidColumnType(
                        2,
                        "file_type".to_string(),
                        rusqlite::types::Type::Text,
                    )
                })?;
                let file_path = patch_file_path(&asset_uri, logical_path.as_deref());
                let filename =
                    patch_file_name(&asset_uri, logical_path.as_deref(), name.as_deref());
                Ok(OperationFileInfo {
                    id: row.get(0)?,
                    filename,
                    file_path,
                    asset_uri,
                    file_type,
                    checksum: row.get::<_, Option<HashId>>(3)?.ok_or_else(|| {
                        rusqlite::Error::InvalidColumnType(
                            3,
                            "checksum".to_string(),
                            rusqlite::types::Type::Blob,
                        )
                    })?,
                })
            },
        )?
        .collect::<Result<Vec<_>, _>>()
        .map_err(CreatePatchError::SqliteError)
}

fn patch_log_ids_and_kind(
    context: &DbContext,
    parent_commit_hash: Option<&CommitHash>,
    commit_ref: &str,
    parent_ref: Option<&str>,
    source_branch: Option<&str>,
) -> Result<(Vec<HashId>, String), CreatePatchError> {
    let history_store = DoltHistoryStore::new(context.graph().conn());
    let current_branch = source_branch
        .map(str::to_string)
        .or(history_store.current_branch()?.map(|branch| branch.0));
    let log_rows = if let Some(parent_commit_hash) = parent_commit_hash {
        let mut statement = context.graph().conn().prepare(
            "SELECT to_id, to_operation_kind, diff_type FROM dolt_diff_gen_operation_log(?1, ?2)",
        )?;
        statement
            .query_map(
                params![
                    parent_ref.unwrap_or(parent_commit_hash.0.as_str()),
                    commit_ref
                ],
                |row| {
                    Ok((
                        row.get::<_, Option<HashId>>(0)?,
                        row.get::<_, Option<String>>(1)?,
                        row.get::<_, String>(2)?,
                    ))
                },
            )?
            .filter_map(|row| match row {
                Ok((Some(log_id), operation_kind, diff_type)) if diff_type == "added" => {
                    Some(Ok((
                        log_id,
                        operation_kind.unwrap_or_else(|| "history-commit".to_string()),
                    )))
                }
                Ok(_) => None,
                Err(err) => Some(Err(err)),
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        let snapshot = open_snapshot_at_ref(
            &context.workspace().graph_db_path()?,
            commit_ref,
            current_branch.as_deref(),
        )?;
        let mut statement = snapshot.prepare(
            "SELECT id, operation_kind \
             FROM gen_operation_log \
             ORDER BY created_on, id",
        )?;
        statement
            .query_map([], |row| {
                Ok((row.get::<_, HashId>(0)?, row.get::<_, String>(1)?))
            })?
            .collect::<Result<Vec<_>, _>>()?
    };

    let mut log_ids = Vec::with_capacity(log_rows.len());
    let mut change_type = None;
    for (log_id, operation_kind) in log_rows {
        if change_type.is_none() {
            change_type = Some(operation_kind);
        }
        log_ids.push(log_id);
    }
    Ok((
        log_ids,
        change_type.unwrap_or_else(|| "history-commit".to_string()),
    ))
}

fn load_history_patch_files(
    context: &DbContext,
    parent_commit_hash: Option<&CommitHash>,
    commit_ref: &str,
    parent_ref: Option<&str>,
    source_branch: Option<&str>,
) -> Result<(Vec<OperationFileInfo>, String), CreatePatchError> {
    let history_store = DoltHistoryStore::new(context.graph().conn());
    let current_branch = source_branch
        .map(str::to_string)
        .or(history_store.current_branch()?.map(|branch| branch.0));
    let (log_ids, change_type) = patch_log_ids_and_kind(
        context,
        parent_commit_hash,
        commit_ref,
        parent_ref,
        source_branch,
    )?;

    if log_ids.is_empty() {
        return Ok((Vec::new(), change_type));
    }

    let snapshot = open_snapshot_at_ref(
        &context.workspace().graph_db_path()?,
        commit_ref,
        current_branch.as_deref(),
    )?;
    Ok((
        query_patch_files_for_logs(&snapshot, &log_ids)?,
        change_type,
    ))
}

fn commit_message_for_hash(
    context: &DbContext,
    commit_hash: &CommitHash,
    commit_ref: &str,
    source_branch: Option<&str>,
) -> Result<String, CreatePatchError> {
    let history_store = DoltHistoryStore::new(context.graph().conn());
    let current_branch = source_branch
        .map(str::to_string)
        .or(history_store.current_branch()?.map(|branch| branch.0));
    let snapshot = open_snapshot_at_ref(
        &context.workspace().graph_db_path()?,
        commit_ref,
        current_branch.as_deref(),
    )?;
    DoltHistoryStore::new(&snapshot)
        .log(Some(1))?
        .into_iter()
        .find(|entry| entry.commit_hash == *commit_hash)
        .map(|entry| entry.message)
        .ok_or_else(|| CreatePatchError::CommitNotFound(commit_hash.clone()))
}

fn build_patch_commit(
    commit_hash: &CommitHash,
    parent_commit_hash: Option<&CommitHash>,
    change_type: String,
) -> PatchCommit {
    PatchCommit {
        hash: padded_hash(commit_hash),
        parent_hash: parent_commit_hash.map(padded_hash),
        change_type,
        commit_hash: commit_hash.clone(),
    }
}

fn build_operation_patch(
    context: &DbContext,
    commit_hash: &CommitHash,
    parent_commit_hash: Option<&CommitHash>,
    commit_ref: &str,
    parent_ref: Option<&str>,
    source_branch: Option<&str>,
) -> Result<OperationPatch, CreatePatchError> {
    let (files, change_type) = load_history_patch_files(
        context,
        parent_commit_hash,
        commit_ref,
        parent_ref,
        source_branch,
    )?;
    let table_changes =
        build_patch_table_changes(context, commit_hash, parent_commit_hash, source_branch)?;
    let files = files
        .into_iter()
        .map(|file| PatchFile::from_file_addition(context, file))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(OperationPatch {
        commit: build_patch_commit(commit_hash, parent_commit_hash, change_type),
        files,
        table_changes,
        commit_message: commit_message_for_hash(context, commit_hash, commit_ref, source_branch)?,
    })
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
struct TableSchema {
    name: String,
    columns: Vec<String>,
    predicate_columns: Vec<String>,
    not_null_columns: Vec<String>,
    default_values: BTreeMap<String, String>,
}

fn quote_identifier(identifier: &str) -> String {
    format!("\"{}\"", identifier.replace('"', "\"\""))
}

fn sql_literal(value: &Value) -> String {
    match value {
        Value::Null => "NULL".to_string(),
        Value::Integer(value) => value.to_string(),
        Value::Real(value) => value.to_string(),
        Value::Text(value) => format!("'{}'", value.replace('\'', "''")),
        Value::Blob(value) => {
            let mut hex = String::with_capacity(value.len() * 2 + 3);
            hex.push_str("X'");
            for byte in value {
                write!(&mut hex, "{byte:02x}").expect("should write blob hex");
            }
            hex.push('\'');
            hex
        }
    }
}

fn patch_value_from_sql_value(value: &Value) -> PatchValue {
    match value {
        Value::Null => PatchValue::Null,
        Value::Integer(value) => PatchValue::Integer(*value),
        Value::Real(value) => PatchValue::Real(*value),
        Value::Text(value) => PatchValue::Text(value.clone()),
        Value::Blob(value) => PatchValue::Blob(value.clone()),
    }
}

fn sql_value_from_patch_value(value: &PatchValue) -> Value {
    match value {
        PatchValue::Null => Value::Null,
        PatchValue::Integer(value) => Value::Integer(*value),
        PatchValue::Real(value) => Value::Real(*value),
        PatchValue::Text(value) => Value::Text(value.clone()),
        PatchValue::Blob(value) => Value::Blob(value.clone()),
    }
}

fn load_table_schemas(context: &DbContext) -> Result<Vec<TableSchema>, CreatePatchError> {
    let mut statement = context.graph().conn().prepare(
        "SELECT name FROM sqlite_master \
         WHERE type = 'table' AND name NOT LIKE 'sqlite_%' AND name NOT LIKE 'dolt_%' \
         ORDER BY rowid",
    )?;
    let table_names = statement
        .query_map([], |row| row.get::<_, String>(0))?
        .collect::<Result<Vec<_>, _>>()?;

    table_names
        .into_iter()
        .map(|table_name| {
            let pragma = format!("PRAGMA table_info({})", quote_identifier(&table_name));
            let mut info_statement = context.graph().conn().prepare(&pragma)?;
            let mut columns_by_primary_key = info_statement
                .query_map([], |row| {
                    Ok((
                        row.get::<_, String>(1)?,
                        row.get::<_, bool>(3)?,
                        row.get::<_, Option<String>>(4)?,
                        row.get::<_, i64>(5)?,
                    ))
                })?
                .collect::<Result<Vec<_>, _>>()?;
            let columns = columns_by_primary_key
                .iter()
                .map(|(column_name, _, _, _)| column_name.clone())
                .collect::<Vec<_>>();
            let not_null_columns = columns_by_primary_key
                .iter()
                .filter(|(_, not_null, _, _)| *not_null)
                .map(|(column_name, _, _, _)| column_name.clone())
                .collect::<Vec<_>>();
            let default_values = columns_by_primary_key
                .iter()
                .filter_map(|(column_name, _, default_value, _)| {
                    default_value
                        .as_ref()
                        .map(|default_value| (column_name.clone(), default_value.clone()))
                })
                .collect::<BTreeMap<_, _>>();
            columns_by_primary_key.sort_by_key(|(_, _, _, primary_key_index)| *primary_key_index);
            let predicate_columns = columns_by_primary_key
                .iter()
                .filter(|(_, _, _, primary_key_index)| *primary_key_index > 0)
                .map(|(column_name, _, _, _)| column_name.clone())
                .collect::<Vec<_>>();
            Ok(TableSchema {
                name: table_name,
                predicate_columns: if predicate_columns.is_empty() {
                    columns.clone()
                } else {
                    predicate_columns
                },
                columns,
                not_null_columns,
                default_values,
            })
        })
        .collect()
}

fn build_patch_table_changes(
    context: &DbContext,
    commit_hash: &CommitHash,
    parent_commit_hash: Option<&CommitHash>,
    source_branch: Option<&str>,
) -> Result<Vec<PatchTableChange>, CreatePatchError> {
    let Some(parent_commit_hash) = parent_commit_hash else {
        return Ok(Vec::new());
    };
    let table_schemas = load_table_schemas(context)?;
    let mut delete_changes = Vec::new();
    let mut apply_changes = Vec::new();
    let mut added_collections = HashSet::new();
    let mut removed_collections = HashSet::new();

    for schema in table_schemas {
        if schema.name == "collections" {
            continue;
        }
        if matches!(
            schema.name.as_str(),
            "gen_operation_assets" | "sample_lineage"
        ) {
            let mut snapshot_changes = build_snapshot_table_changes(
                context,
                &schema,
                commit_hash,
                parent_commit_hash,
                source_branch,
            )?;
            for change in snapshot_changes.drain(..) {
                match change.diff_type.as_str() {
                    "removed" => delete_changes.push(change),
                    "added" | "modified" => apply_changes.push(change),
                    _ => {}
                }
            }
            continue;
        }
        let query = format!("SELECT * FROM dolt_diff_{}(?1, ?2)", schema.name);
        let mut statement = context.graph().conn().prepare(&query)?;
        let diff_column_names = statement
            .column_names()
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        let diff_rows =
            statement.query_map(params![parent_commit_hash.0, commit_hash.0], |row| {
                let values = (0..diff_column_names.len())
                    .map(|index| row.get::<_, Value>(index))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(values)
            })?;

        for row in diff_rows {
            let values = row?;
            let diff_type = match diff_value_by_column(&diff_column_names, &values, "diff_type")? {
                Value::Text(value) => value.clone(),
                _ => String::new(),
            };
            let before = diff_row_map(&schema, &diff_column_names, &values, "from_");
            let after = diff_row_map(&schema, &diff_column_names, &values, "to_");

            if schema.name == "block_groups" {
                if let Some(collection_name) =
                    extract_text_column(after.as_ref(), "collection_name")
                {
                    added_collections.insert(collection_name);
                }
                if let Some(collection_name) =
                    extract_text_column(before.as_ref(), "collection_name")
                {
                    removed_collections.insert(collection_name);
                }
            }

            match diff_type.as_str() {
                "added" => {
                    if let Some(after) = after {
                        apply_changes.push(PatchTableChange {
                            table_name: schema.name.clone(),
                            diff_type,
                            predicate_columns: schema.predicate_columns.clone(),
                            before: None,
                            after: Some(after),
                        });
                    }
                }
                "removed" => {
                    if let Some(before) = before {
                        delete_changes.push(PatchTableChange {
                            table_name: schema.name.clone(),
                            diff_type,
                            predicate_columns: schema.predicate_columns.clone(),
                            before: Some(before),
                            after: None,
                        });
                    }
                }
                "modified" => {
                    if let (Some(before), Some(after)) = (before, after) {
                        apply_changes.push(PatchTableChange {
                            table_name: schema.name.clone(),
                            diff_type,
                            predicate_columns: schema.predicate_columns.clone(),
                            before: Some(before),
                            after: Some(after),
                        });
                    }
                }
                _ => {}
            }
        }
    }

    for collection_name in removed_collections {
        delete_changes.push(PatchTableChange {
            table_name: "collections".to_string(),
            diff_type: "removed".to_string(),
            predicate_columns: vec!["name".to_string()],
            before: Some(BTreeMap::from([(
                "name".to_string(),
                PatchValue::Text(collection_name),
            )])),
            after: None,
        });
    }
    for collection_name in added_collections {
        apply_changes.push(PatchTableChange {
            table_name: "collections".to_string(),
            diff_type: "added".to_string(),
            predicate_columns: vec!["name".to_string()],
            before: None,
            after: Some(BTreeMap::from([(
                "name".to_string(),
                PatchValue::Text(collection_name),
            )])),
        });
    }

    delete_changes.reverse();
    delete_changes.extend(apply_changes);
    Ok(delete_changes)
}

fn build_snapshot_table_changes(
    context: &DbContext,
    schema: &TableSchema,
    commit_hash: &CommitHash,
    parent_commit_hash: &CommitHash,
    source_branch: Option<&str>,
) -> Result<Vec<PatchTableChange>, CreatePatchError> {
    let history_store = DoltHistoryStore::new(context.graph().conn());
    let current_branch = source_branch
        .map(str::to_string)
        .or(history_store.current_branch()?.map(|branch| branch.0));
    let parent_snapshot = open_snapshot_at_ref(
        &context.workspace().graph_db_path()?,
        parent_commit_hash.0.as_str(),
        current_branch.as_deref(),
    )?;
    let current_snapshot = open_snapshot_at_ref(
        &context.workspace().graph_db_path()?,
        commit_hash.0.as_str(),
        current_branch.as_deref(),
    )?;
    let before_rows = load_table_row_maps(&parent_snapshot, schema)?;
    let after_rows = load_table_row_maps(&current_snapshot, schema)?;
    let mut changes = Vec::new();

    for (predicate, before) in &before_rows {
        if let Some(after) = after_rows.get(predicate) {
            if before != after {
                changes.push(PatchTableChange {
                    table_name: schema.name.clone(),
                    diff_type: "modified".to_string(),
                    predicate_columns: schema.predicate_columns.clone(),
                    before: Some(before.clone()),
                    after: Some(after.clone()),
                });
            }
        } else {
            changes.push(PatchTableChange {
                table_name: schema.name.clone(),
                diff_type: "removed".to_string(),
                predicate_columns: schema.predicate_columns.clone(),
                before: Some(before.clone()),
                after: None,
            });
        }
    }

    for (predicate, after) in &after_rows {
        if !before_rows.contains_key(predicate) {
            changes.push(PatchTableChange {
                table_name: schema.name.clone(),
                diff_type: "added".to_string(),
                predicate_columns: schema.predicate_columns.clone(),
                before: None,
                after: Some(after.clone()),
            });
        }
    }

    Ok(changes)
}

fn load_table_row_maps(
    conn: &GraphConnection,
    schema: &TableSchema,
) -> Result<BTreeMap<String, BTreeMap<String, PatchValue>>, CreatePatchError> {
    let columns = schema
        .columns
        .iter()
        .map(|column| quote_identifier(column))
        .collect::<Vec<_>>()
        .join(", ");
    let order_by = schema
        .predicate_columns
        .iter()
        .map(|column| quote_identifier(column))
        .collect::<Vec<_>>()
        .join(", ");
    let query = format!(
        "SELECT {columns} FROM {} ORDER BY {order_by}",
        quote_identifier(&schema.name)
    );
    let mut statement = conn.prepare(&query)?;
    let rows = statement.query_map([], |row| {
        schema
            .columns
            .iter()
            .enumerate()
            .map(|(index, column)| {
                row.get::<_, Value>(index)
                    .map(|value| (column.clone(), patch_value_from_sql_value(&value)))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()
    })?;
    let row_maps = rows.collect::<Result<Vec<_>, _>>()?;
    let mut rows_by_predicate = BTreeMap::new();

    for row_values in row_maps {
        let predicate = predicate_for_patch_row(&schema.predicate_columns, &row_values);
        rows_by_predicate.insert(predicate, row_values);
    }

    Ok(rows_by_predicate)
}

fn predicate_for_patch_row(
    predicate_columns: &[String],
    row_values: &BTreeMap<String, PatchValue>,
) -> String {
    predicate_columns
        .iter()
        .map(|column| {
            let value = row_values
                .get(column)
                .expect("should include predicate columns in snapshot row");
            match value {
                PatchValue::Null => format!("{column}=NULL"),
                PatchValue::Integer(value) => format!("{column}=int:{value}"),
                PatchValue::Real(value) => format!("{column}=real:{value}"),
                PatchValue::Text(value) => format!("{column}=text:{value}"),
                PatchValue::Blob(value) => format!("{column}=blob:{value:?}"),
            }
        })
        .collect::<Vec<_>>()
        .join("|")
}

fn diff_value_by_column<'a>(
    column_names: &[String],
    values: &'a [Value],
    target_column: &str,
) -> Result<&'a Value, SQLError> {
    column_names
        .iter()
        .position(|column| column == target_column)
        .and_then(|index| values.get(index))
        .ok_or_else(|| SQLError::InvalidColumnName(target_column.to_string()))
}

fn diff_row_map(
    schema: &TableSchema,
    diff_column_names: &[String],
    values: &[Value],
    prefix: &str,
) -> Option<BTreeMap<String, PatchValue>> {
    let row_values = schema
        .columns
        .iter()
        .filter_map(|column| {
            let diff_column = format!("{prefix}{column}");
            diff_value_by_column(diff_column_names, values, &diff_column)
                .ok()
                .map(|value| (column.clone(), patch_value_from_sql_value(value)))
        })
        .collect::<BTreeMap<_, _>>();
    if row_values
        .values()
        .all(|value| matches!(value, PatchValue::Null))
    {
        None
    } else {
        Some(row_values)
    }
}

fn extract_text_column(
    row_values: Option<&BTreeMap<String, PatchValue>>,
    column_name: &str,
) -> Option<String> {
    match row_values?.get(column_name)? {
        PatchValue::Text(value) => Some(value.clone()),
        _ => None,
    }
}

fn predicate_for_row(
    predicate_columns: &[String],
    row_values: &BTreeMap<String, PatchValue>,
) -> Result<String, PatchError> {
    predicate_columns
        .iter()
        .map(|column| {
            let value = row_values.get(column).ok_or_else(|| {
                PatchError::SchemaMismatch(format!(
                    "missing predicate column '{column}' while applying patch"
                ))
            })?;
            let value = sql_value_from_patch_value(value);
            if matches!(value, Value::Null) {
                Ok(format!("{} IS NULL", quote_identifier(column)))
            } else {
                Ok(format!(
                    "{} = {}",
                    quote_identifier(column),
                    sql_literal(&value)
                ))
            }
        })
        .collect::<Result<Vec<_>, _>>()
        .map(|parts| parts.join(" AND "))
}

fn aligned_row_values(
    source_schema: &TableSchema,
    current_schema: &TableSchema,
    row_values: &BTreeMap<String, PatchValue>,
) -> Result<Vec<(String, String)>, PatchError> {
    current_schema
        .columns
        .iter()
        .enumerate()
        .map(|(index, column)| {
            if let Some(value) = row_values.get(column) {
                return Ok((column.clone(), sql_literal(&sql_value_from_patch_value(value))));
            }
            if let Some(source_column) = source_schema.columns.get(index)
                && let Some(value) = row_values.get(source_column)
            {
                return Ok((column.clone(), sql_literal(&sql_value_from_patch_value(value))));
            }
            if let Some(default_value) = current_schema.default_values.get(column) {
                return Ok((column.clone(), default_value.clone()));
            }
            if !current_schema.not_null_columns.contains(column) {
                return Ok((column.clone(), "NULL".to_string()));
            }
            Err(PatchError::SchemaMismatch(format!(
                "column '{}' on table '{}' is required in the current schema but missing from the patch payload",
                column, current_schema.name
            )))
        })
        .collect()
}

fn aligned_patch_values(
    source_schema: &TableSchema,
    current_schema: &TableSchema,
    row_values: &BTreeMap<String, PatchValue>,
) -> Result<Vec<(String, PatchValue)>, PatchError> {
    current_schema
        .columns
        .iter()
        .enumerate()
        .map(|(index, column)| {
            if let Some(value) = row_values.get(column) {
                return Ok((column.clone(), value.clone()));
            }
            if let Some(source_column) = source_schema.columns.get(index)
                && let Some(value) = row_values.get(source_column)
            {
                return Ok((column.clone(), value.clone()));
            }
            if !current_schema.not_null_columns.contains(column) {
                return Ok((column.clone(), PatchValue::Null));
            }
            Err(PatchError::SchemaMismatch(format!(
                "column '{}' on table '{}' is required in the current schema but missing from the patch payload",
                column, current_schema.name
            )))
        })
        .collect()
}

fn patch_change_to_statement(
    source_schema: &TableSchema,
    current_schema: &TableSchema,
    patch_change: &PatchTableChange,
) -> Result<String, PatchError> {
    match patch_change.diff_type.as_str() {
        "added" => {
            let row_values = patch_change.after.as_ref().ok_or_else(|| {
                PatchError::SchemaMismatch(format!(
                    "patch add for table '{}' is missing row values",
                    patch_change.table_name
                ))
            })?;
            let aligned_values = aligned_row_values(source_schema, current_schema, row_values)?;
            let columns = aligned_values
                .iter()
                .map(|(column, _)| quote_identifier(column))
                .collect::<Vec<_>>()
                .join(", ");
            let values = aligned_values
                .iter()
                .map(|(_, value)| value.clone())
                .collect::<Vec<_>>()
                .join(", ");
            Ok(format!(
                "INSERT INTO {} ({columns}) VALUES ({values});",
                quote_identifier(&current_schema.name)
            ))
        }
        "removed" => {
            let row_values = patch_change.before.as_ref().ok_or_else(|| {
                PatchError::SchemaMismatch(format!(
                    "patch delete for table '{}' is missing prior row values",
                    patch_change.table_name
                ))
            })?;
            let predicate = predicate_for_row(&patch_change.predicate_columns, row_values)?;
            Ok(format!(
                "DELETE FROM {} WHERE {predicate};",
                quote_identifier(&current_schema.name)
            ))
        }
        "modified" => {
            let before_values = patch_change.before.as_ref().ok_or_else(|| {
                PatchError::SchemaMismatch(format!(
                    "patch update for table '{}' is missing prior row values",
                    patch_change.table_name
                ))
            })?;
            let after_values = patch_change.after.as_ref().ok_or_else(|| {
                PatchError::SchemaMismatch(format!(
                    "patch update for table '{}' is missing next row values",
                    patch_change.table_name
                ))
            })?;
            let predicate = predicate_for_row(&patch_change.predicate_columns, before_values)?;
            let assignments = aligned_row_values(source_schema, current_schema, after_values)?
                .into_iter()
                .map(|(column, value)| format!("{} = {}", quote_identifier(&column), value))
                .collect::<Vec<_>>()
                .join(", ");
            Ok(format!(
                "UPDATE {} SET {assignments} WHERE {predicate};",
                quote_identifier(&current_schema.name)
            ))
        }
        diff_type => Err(PatchError::SchemaMismatch(format!(
            "unsupported patch diff type '{diff_type}' for table '{}'",
            patch_change.table_name
        ))),
    }
}

fn apply_patch_table_changes(
    context: &DbContext,
    source_table_schemas: &[TableSchema],
    current_table_schemas: &[TableSchema],
    patch_table_changes: &[PatchTableChange],
) -> Result<(), PatchError> {
    if patch_table_changes.is_empty() {
        return Ok(());
    }

    let current_schema_by_table = current_table_schemas
        .iter()
        .map(|schema| (schema.name.as_str(), schema))
        .collect::<BTreeMap<_, _>>();
    let source_schema_by_table = source_table_schemas
        .iter()
        .map(|schema| (schema.name.as_str(), schema))
        .collect::<BTreeMap<_, _>>();
    context
        .graph()
        .conn()
        .execute_batch("PRAGMA foreign_keys = OFF; BEGIN IMMEDIATE;")
        .map_err(PatchError::SqliteError)?;
    for patch_change in patch_table_changes {
        let current_schema = current_schema_by_table
            .get(patch_change.table_name.as_str())
            .ok_or_else(|| {
                PatchError::SchemaMismatch(format!(
                    "table '{}' from the patch does not exist in the current schema",
                    patch_change.table_name
                ))
            })?;
        let source_schema = source_schema_by_table
            .get(patch_change.table_name.as_str())
            .ok_or_else(|| {
                PatchError::SchemaMismatch(format!(
                    "table '{}' is missing source schema metadata in the patch",
                    patch_change.table_name
                ))
            })?;
        let statement = match patch_change_to_statement(source_schema, current_schema, patch_change)
        {
            Ok(statement) => statement,
            Err(error) => {
                let _ = context
                    .graph()
                    .conn()
                    .execute_batch("ROLLBACK; PRAGMA foreign_keys = ON;");
                return Err(error);
            }
        };
        if let Err(error) = context.graph().conn().execute_batch(&statement) {
            if patch_change.diff_type == "added"
                && existing_row_matches_added_patch(
                    context.graph().conn(),
                    source_schema,
                    current_schema,
                    patch_change,
                )?
            {
                continue;
            }
            let _ = context
                .graph()
                .conn()
                .execute_batch("ROLLBACK; PRAGMA foreign_keys = ON;");
            return Err(PatchError::SqliteError(error));
        }
    }
    context
        .graph()
        .conn()
        .execute_batch("COMMIT; PRAGMA foreign_keys = ON;")
        .map_err(PatchError::SqliteError)
}

fn existing_row_matches_added_patch(
    conn: &GraphConnection,
    source_schema: &TableSchema,
    current_schema: &TableSchema,
    patch_change: &PatchTableChange,
) -> Result<bool, PatchError> {
    if source_schema.columns != current_schema.columns {
        return Ok(false);
    }
    let Some(row_values) = patch_change.after.as_ref() else {
        return Ok(false);
    };
    let expected_row = aligned_patch_values(source_schema, current_schema, row_values)?
        .into_iter()
        .collect::<BTreeMap<_, _>>();
    let predicate = predicate_for_row(&current_schema.predicate_columns, &expected_row)?;
    let select_columns = current_schema
        .columns
        .iter()
        .map(|column| quote_identifier(column))
        .collect::<Vec<_>>()
        .join(", ");
    let query = format!(
        "SELECT {select_columns} FROM {} WHERE {predicate} LIMIT 1",
        quote_identifier(&current_schema.name)
    );
    let mut statement = conn.prepare(&query).map_err(PatchError::SqliteError)?;
    let existing_row = statement
        .query_row([], |row| {
            current_schema
                .columns
                .iter()
                .enumerate()
                .map(|(index, column)| {
                    Ok((
                        column.clone(),
                        patch_value_from_sql_value(&row.get::<_, Value>(index)?),
                    ))
                })
                .collect::<Result<BTreeMap<_, _>, SQLError>>()
        })
        .optional()
        .map_err(PatchError::SqliteError)?;
    Ok(existing_row == Some(expected_row))
}

fn patch_source_history(
    context: &DbContext,
    source_branch: Option<&str>,
) -> Result<Vec<gen_models::history::HistoryEntry>, CreatePatchError> {
    if let Some(source_branch) = source_branch {
        let history_store = DoltHistoryStore::new(context.graph().conn());
        let current_branch = history_store.current_branch()?.map(|branch| branch.0);
        let snapshot = open_snapshot_at_ref(
            &context.workspace().graph_db_path()?,
            source_branch,
            current_branch.as_deref(),
        )?;
        return Ok(DoltHistoryStore::new(&snapshot)
            .log(None)?
            .into_iter()
            .rev()
            .collect());
    }

    Ok(DoltHistoryStore::new(context.graph().conn())
        .log(None)?
        .into_iter()
        .rev()
        .collect())
}

pub fn create_patch<W>(
    context: &DbContext,
    operations: &[HashId],
    source_branch: Option<&str>,
    write_stream: &mut W,
) -> Result<(), CreatePatchError>
where
    W: Write + Seek,
{
    let current_branch_history = patch_source_history(context, source_branch)?;
    let history_ref_name = source_branch
        .map(str::to_string)
        .or(DoltHistoryStore::new(context.graph().conn())
            .current_branch()?
            .map(|branch| branch.0))
        .unwrap_or_else(|| "HEAD".to_string());
    let mut patches = vec![];
    for hash in operations {
        let commit_hash = commit_hash_from_hash(*hash)?;
        let selected_index = current_branch_history
            .iter()
            .position(|entry| entry.commit_hash == commit_hash)
            .ok_or(CreatePatchError::OperationNotFound(*hash))?;
        let commit_ref = branch_history_ref(
            &history_ref_name,
            current_branch_history.len(),
            selected_index,
        );
        let parent_commit_hash = if selected_index == 0 {
            None
        } else {
            Some(&current_branch_history[selected_index - 1].commit_hash)
        };
        let parent_ref = selected_index.checked_sub(1).map(|parent_index| {
            branch_history_ref(
                &history_ref_name,
                current_branch_history.len(),
                parent_index,
            )
        });
        println!("Creating patch for Operation {id}", id = hash);
        patches.push(build_operation_patch(
            context,
            &commit_hash,
            parent_commit_hash,
            &commit_ref,
            parent_ref.as_deref(),
            source_branch,
        )?);
    }

    let target_commit_hash = patches
        .last()
        .map(|patch| patch.commit.commit_hash.clone())
        .ok_or_else(|| {
            CreatePatchError::InvalidOperationHash("empty patch selection".to_string())
        })?;
    let base_commit_hash = patches
        .first()
        .and_then(|patch| patch.commit.parent_hash)
        .map(commit_hash_from_hash)
        .transpose()?;

    let operation_patches = OperationPatches {
        table_schemas: load_table_schemas(context)?,
        base_commit_hash,
        target_commit_hash,
        patches,
    };
    let manifest_bytes = serialize_operation_patches(&operation_patches)?;

    let manifest_options =
        SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
    let asset_options = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);

    let mut archive = ZipWriter::new(write_stream);
    archive.start_file(MANIFEST_ENTRY, manifest_options)?;
    archive.write_all(&manifest_bytes)?;

    let mut written_asset_entries = HashSet::new();
    for patch in &operation_patches.patches {
        for file in &patch.files {
            if file.archive_path.is_empty()
                || !written_asset_entries.insert(file.archive_path.clone())
            {
                continue;
            }

            archive.start_file(&file.archive_path, asset_options)?;
            let source_path = file.source_asset_path(context)?;
            let mut source_file = File::open(source_path)?;
            std::io::copy(&mut source_file, &mut archive)?;
        }
    }

    archive.finish()?;
    Ok(())
}

fn branch_history_ref(branch_ref: &str, history_len: usize, index: usize) -> String {
    let distance_from_head = history_len
        .checked_sub(index + 1)
        .expect("should build a history ref within branch bounds");
    if distance_from_head == 0 {
        branch_ref.to_string()
    } else {
        format!("{branch_ref}~{distance_from_head}")
    }
}

pub fn load_patches<R>(reader: R) -> Vec<OperationPatch>
where
    R: Read + Seek,
{
    let mut archive = ZipArchive::new(reader).unwrap();
    read_operation_patches(&mut archive).unwrap().patches
}

pub fn apply_patch_archive<R>(context: &mut DbContext, reader: R) -> Result<(), PatchError>
where
    R: Read + Seek,
{
    let mut archive = ZipArchive::new(reader)?;
    let operation_patches = read_operation_patches(&mut archive)?;
    let history_store = DoltHistoryStore::new(context.graph().conn());
    let current_table_schemas =
        load_table_schemas(context).map_err(|error| PatchError::SQLError(error.to_string()))?;
    let mut applied_patch = false;
    for patch in &operation_patches.patches {
        let patch_has_changes = !patch.table_changes.is_empty() || !patch.files.is_empty();
        apply_patch_table_changes(
            context,
            &operation_patches.table_schemas,
            &current_table_schemas,
            &patch.table_changes,
        )?;
        let mut restored_file = false;
        for file in &patch.files {
            restored_file |= file.restore_from_archive(context, &mut archive)?;
        }
        if patch_has_changes {
            match history_store.commit_all(&patch.commit_message) {
                Ok(_) => {
                    applied_patch = true;
                }
                Err(error) if is_nothing_to_commit_error(&error) => {
                    applied_patch |= restored_file;
                }
                Err(error) => return Err(PatchError::SqliteError(error)),
            }
        }
    }

    if !applied_patch {
        return Err(PatchError::OperationError(OperationError::NoChanges));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{fs, io::Cursor, path::PathBuf};

    use gen_models::{
        annotations::add_annotation_file,
        collection::Collection,
        file_types::FileTypes,
        history::dolt::DoltHistoryStore,
        operations::{FileAddition, add_files_operation},
    };
    use tempfile::Builder;

    use super::*;
    use crate::{
        imports::fasta::import_fasta, test_helpers::setup_gen_on_disk,
        views::annotation_files::load_annotation_file_entries,
    };

    fn create_history_backed_patch_operations(context: &DbContext) -> (HashId, HashId) {
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let gff_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.gff");

        let add_file_commit_hash = add_files_operation(
            context,
            &[fasta_path.to_string_lossy().to_string()],
            Some("track fasta fixture"),
        )
        .expect("should commit add-file fixture");
        add_annotation_file(
            context,
            gff_path
                .to_str()
                .expect("should encode annotation fixture path"),
            None,
            None,
            Some("fixture-track"),
            Some("track annotation fixture"),
        )
        .expect("should commit annotation fixture");
        let annotation_operation = current_history_operation_hash(context);
        let add_file_operation = HashId::pad_str(&add_file_commit_hash.0);
        (add_file_operation, annotation_operation)
    }

    fn current_history_operation_hash(context: &DbContext) -> HashId {
        let history_store = DoltHistoryStore::new(context.graph().conn());
        let current_commit_hash = history_store
            .current_head()
            .expect("should resolve current head")
            .expect("should have a current head");
        HashId::pad_str(&current_commit_hash.0)
    }

    fn assert_legacy_patch_file_tables_are_empty(context: &DbContext) {
        let operation_conn = context.config().conn();
        let legacy_table_count = operation_conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name IN ('file_additions', 'operation_files')",
                [],
                |row| row.get::<_, i64>(0),
            )
            .expect("should inspect config schema");
        assert_eq!(legacy_table_count, 0);
    }

    #[test]
    fn test_creates_patch() {
        let context = setup_gen_on_disk();
        let (op_1, op_2) = create_history_backed_patch_operations(&context);
        let mut write_stream = Cursor::new(Vec::new());
        create_patch(&context, &[op_1, op_2], None, &mut write_stream).unwrap();
        write_stream.set_position(0);
        load_patches(&mut write_stream);
    }

    #[test]
    fn test_patch_create_current_head_keeps_head_changes() {
        let context = setup_gen_on_disk();
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        import_fasta(
            &context,
            &fasta_path.to_string_lossy().to_string(),
            "default",
            "foo",
            false,
        )
        .expect("should import fasta fixture");
        let head_operation = current_history_operation_hash(&context);
        let mut write_stream = Cursor::new(Vec::new());

        create_patch(&context, &[head_operation], None, &mut write_stream)
            .expect("should create a patch from the current head");
        write_stream.set_position(0);
        let patches = load_patches(&mut write_stream);

        assert_eq!(patches.len(), 1, "should create a single patch entry");
        assert_eq!(
            patches[0].commit.change_type, "fasta_addition",
            "current head patch should point at the imported commit"
        );
        assert!(
            !patches[0].table_changes.is_empty(),
            "current head patch should include graph table changes"
        );
    }

    #[test]
    fn test_cross_db_patches() {
        let source_context = setup_gen_on_disk();
        let source_graph = source_context.graph().conn();
        let (op_1, op_2) = create_history_backed_patch_operations(&source_context);
        let mut write_stream = Cursor::new(Vec::new());
        create_patch(&source_context, &[op_1, op_2], None, &mut write_stream).unwrap();
        write_stream.set_position(0);

        let mut target_context = setup_gen_on_disk();
        apply_patch_archive(&mut target_context, &mut write_stream).unwrap();
        let source_history = DoltHistoryStore::new(source_graph)
            .log(None)
            .expect("should read source history");
        let target_history = DoltHistoryStore::new(target_context.graph().conn())
            .log(None)
            .expect("should read target history");
        assert_eq!(target_history.len(), source_history.len());
        assert_eq!(target_history[0].message, "track annotation fixture");
        assert_eq!(target_history[1].message, "track fasta fixture");

        let annotation_entries = load_annotation_file_entries(target_context.graph().conn());
        assert_eq!(annotation_entries.len(), 1);
        assert_eq!(annotation_entries[0].display_name, "fixture-track");
        assert_legacy_patch_file_tables_are_empty(&target_context);
    }

    #[test]
    fn test_cross_branch_patches() {
        let mut context = setup_gen_on_disk();
        let history_store = DoltHistoryStore::new(context.graph().conn());

        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let _op_1 = add_files_operation(
            &context,
            &[fasta_path.to_string_lossy().to_string()],
            Some("track fasta fixture"),
        )
        .expect("should commit main branch fixture");
        let main_head = history_store
            .current_head()
            .unwrap()
            .expect("should have a main branch head");
        history_store
            .create_branch(
                &gen_core::BranchName("new-branch".to_string()),
                Some(&gen_core::CommitRef("main".to_string())),
            )
            .unwrap();
        history_store
            .checkout_branch(&gen_core::BranchName("new-branch".to_string()))
            .unwrap();
        let gff_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.gff");
        add_annotation_file(
            &context,
            gff_path
                .to_str()
                .expect("should encode annotation fixture path"),
            None,
            None,
            Some("branch-track"),
            Some("track branch annotation"),
        )
        .expect("should commit branch annotation");
        let op_2 = current_history_operation_hash(&context);
        let mut write_stream = Cursor::new(Vec::new());
        create_patch(&context, &[op_2], None, &mut write_stream).unwrap();

        history_store
            .checkout_branch(&gen_core::BranchName("main".to_string()))
            .unwrap();
        assert_eq!(history_store.current_head().unwrap(), Some(main_head));
        Collection::create(context.graph().conn(), "main-side-collection")
            .expect("should insert a divergent main branch collection");
        history_store
            .commit_all("main side commit")
            .expect("should commit the divergent main branch change");
        write_stream.set_position(0);

        apply_patch_archive(&mut context, &mut write_stream).unwrap();
        let history_messages = DoltHistoryStore::new(context.graph().conn())
            .log(None)
            .unwrap()
            .into_iter()
            .map(|entry| entry.message)
            .collect::<Vec<_>>();
        assert_eq!(history_messages[0], "track branch annotation");
        assert_eq!(history_messages[1], "main side commit");
        assert_eq!(history_messages[2], "track fasta fixture");
        let divergent_collection_count = context
            .graph()
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM collections WHERE name = 'main-side-collection'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .expect("should query the divergent main branch collection");
        assert_eq!(divergent_collection_count, 1);

        write_stream.set_position(0);
        let res = apply_patch_archive(&mut context, &mut write_stream);
        assert!(
            matches!(
                res,
                Err(PatchError::OperationError(OperationError::NoChanges))
            ),
            "reapplying a diff patch should report that it has no changes left to apply"
        );
        let history_messages = DoltHistoryStore::new(context.graph().conn())
            .log(None)
            .unwrap()
            .into_iter()
            .map(|entry| entry.message)
            .collect::<Vec<_>>();
        assert_eq!(history_messages[0], "track branch annotation");
        assert_eq!(history_messages[1], "main side commit");
        assert_eq!(history_messages[2], "track fasta fixture");
    }

    #[test]
    fn test_patch_manifest_serialization() {
        let context = setup_gen_on_disk();
        let (op_1, op_2) = create_history_backed_patch_operations(&context);

        let mut write_stream = Cursor::new(Vec::new());
        create_patch(&context, &[op_1, op_2], None, &mut write_stream).unwrap();
        write_stream.set_position(0);
        let loaded_patches = load_patches(&mut write_stream);

        assert_eq!(loaded_patches[0].commit.hash, op_1);
        assert_eq!(loaded_patches[1].commit.hash, op_2);
        assert!(!loaded_patches[0].files.is_empty());
        assert!(!loaded_patches[1].files.is_empty());
        assert!(!loaded_patches[0].table_changes.is_empty());
        assert!(!loaded_patches[1].table_changes.is_empty());
        assert!(!loaded_patches[0].commit_message.is_empty());
        assert!(!loaded_patches[1].commit_message.is_empty());
        assert!(
            loaded_patches[0]
                .files
                .iter()
                .any(|patch_file| patch_file.file.file_type == FileTypes::Fasta)
        );
        assert!(
            loaded_patches[1]
                .files
                .iter()
                .any(|patch_file| patch_file.file.file_type == FileTypes::Gff3)
        );
    }

    #[test]
    fn test_patch_manifest_includes_format_version_header() {
        let context = setup_gen_on_disk();
        let (op_1, op_2) = create_history_backed_patch_operations(&context);
        let mut write_stream = Cursor::new(Vec::new());

        create_patch(&context, &[op_1, op_2], None, &mut write_stream).unwrap();
        write_stream.set_position(0);
        let mut archive = ZipArchive::new(&mut write_stream).expect("should open patch archive");
        let mut manifest_file = archive
            .by_name(MANIFEST_ENTRY)
            .expect("should read the manifest entry");
        let mut manifest_json = String::new();
        manifest_file
            .read_to_string(&mut manifest_json)
            .expect("should read manifest JSON");
        let manifest_value: serde_json::Value =
            serde_json::from_str(&manifest_json).expect("should parse manifest JSON");
        assert_eq!(
            manifest_value["header"]["patch_format_version"],
            serde_json::Value::from(PATCH_FORMAT_VERSION)
        );
    }

    #[test]
    fn test_patch_change_to_statement_rewrites_rows_for_current_schema() {
        let source_schema = TableSchema {
            name: "collections".to_string(),
            columns: vec!["name".to_string()],
            predicate_columns: vec!["name".to_string()],
            not_null_columns: vec!["name".to_string()],
            default_values: BTreeMap::new(),
        };
        let current_schema = TableSchema {
            name: "collections".to_string(),
            columns: vec!["collection_name".to_string(), "created_by".to_string()],
            predicate_columns: vec!["collection_name".to_string()],
            not_null_columns: vec!["collection_name".to_string(), "created_by".to_string()],
            default_values: BTreeMap::from([("created_by".to_string(), "'system'".to_string())]),
        };
        let patch_change = PatchTableChange {
            table_name: "collections".to_string(),
            diff_type: "added".to_string(),
            predicate_columns: vec!["name".to_string()],
            before: None,
            after: Some(BTreeMap::from([(
                "name".to_string(),
                PatchValue::Text("feature-only".to_string()),
            )])),
        };

        let statement = patch_change_to_statement(&source_schema, &current_schema, &patch_change)
            .expect("should rewrite a patch row for the current schema");

        assert_eq!(
            statement,
            "INSERT INTO \"collections\" (\"collection_name\", \"created_by\") VALUES ('feature-only', 'system');"
        );
    }

    #[test]
    fn test_patch_change_to_statement_errors_for_new_required_column_without_default() {
        let source_schema = TableSchema {
            name: "collections".to_string(),
            columns: vec!["name".to_string()],
            predicate_columns: vec!["name".to_string()],
            not_null_columns: vec!["name".to_string()],
            default_values: BTreeMap::new(),
        };
        let current_schema = TableSchema {
            name: "collections".to_string(),
            columns: vec!["collection_name".to_string(), "created_by".to_string()],
            predicate_columns: vec!["collection_name".to_string()],
            not_null_columns: vec!["collection_name".to_string(), "created_by".to_string()],
            default_values: BTreeMap::new(),
        };
        let patch_change = PatchTableChange {
            table_name: "collections".to_string(),
            diff_type: "added".to_string(),
            predicate_columns: vec!["name".to_string()],
            before: None,
            after: Some(BTreeMap::from([(
                "name".to_string(),
                PatchValue::Text("feature-only".to_string()),
            )])),
        };

        let error = patch_change_to_statement(&source_schema, &current_schema, &patch_change)
            .expect_err("should reject a patch row when the current schema adds a required column");

        assert!(
            matches!(error, PatchError::SchemaMismatch(_)),
            "expected a schema mismatch error, got {error:?}"
        );
    }

    #[test]
    fn test_patch_change_to_statement_allows_compatible_type_shift() {
        let source_schema = TableSchema {
            name: "samples".to_string(),
            columns: vec!["name".to_string(), "is_reference".to_string()],
            predicate_columns: vec!["name".to_string()],
            not_null_columns: vec!["name".to_string(), "is_reference".to_string()],
            default_values: BTreeMap::new(),
        };
        let current_schema = TableSchema {
            name: "samples".to_string(),
            columns: vec!["name".to_string(), "is_reference".to_string()],
            predicate_columns: vec!["name".to_string()],
            not_null_columns: vec!["name".to_string(), "is_reference".to_string()],
            default_values: BTreeMap::new(),
        };
        let patch_change = PatchTableChange {
            table_name: "samples".to_string(),
            diff_type: "added".to_string(),
            predicate_columns: vec!["name".to_string()],
            before: None,
            after: Some(BTreeMap::from([
                ("name".to_string(), PatchValue::Text("edited".to_string())),
                ("is_reference".to_string(), PatchValue::Integer(1)),
            ])),
        };

        let statement = patch_change_to_statement(&source_schema, &current_schema, &patch_change)
            .expect("should keep generating a valid statement when a compatible type shift occurs");

        assert_eq!(
            statement,
            "INSERT INTO \"samples\" (\"name\", \"is_reference\") VALUES ('edited', 1);"
        );
    }

    #[test]
    fn test_patch_empty_db() {
        let context = setup_gen_on_disk();
        let (_op_1, op_2) = create_history_backed_patch_operations(&context);
        let mut write_stream = Cursor::new(Vec::new());
        create_patch(&context, &[op_2], None, &mut write_stream).unwrap();

        let mut fresh_context = setup_gen_on_disk();
        write_stream.set_position(0);
        apply_patch_archive(&mut fresh_context, &mut write_stream).unwrap();
        let annotation_entries = load_annotation_file_entries(fresh_context.graph().conn());
        assert_eq!(annotation_entries.len(), 1);
        assert_eq!(annotation_entries[0].display_name, "fixture-track");
        assert_legacy_patch_file_tables_are_empty(&fresh_context);
    }

    #[test]
    fn test_patch_round_trip_without_changeset_directory() {
        let source_context = setup_gen_on_disk();
        let (op_1, op_2) = create_history_backed_patch_operations(&source_context);

        let changeset_dir = source_context
            .workspace()
            .ensure_gen_dir()
            .join("changesets");
        if changeset_dir.exists() {
            fs::remove_dir_all(&changeset_dir).unwrap();
        }

        let mut write_stream = Cursor::new(Vec::new());
        create_patch(&source_context, &[op_1, op_2], None, &mut write_stream).unwrap();
        write_stream.set_position(0);

        let mut target_context = setup_gen_on_disk();
        apply_patch_archive(&mut target_context, &mut write_stream).unwrap();
        let annotation_entries = load_annotation_file_entries(target_context.graph().conn());
        assert_eq!(annotation_entries.len(), 1);
        assert_eq!(annotation_entries[0].display_name, "fixture-track");
        assert_legacy_patch_file_tables_are_empty(&target_context);
    }

    #[test]
    fn test_patch_restores_external_assets_into_target_workspace() {
        let source_context = setup_gen_on_disk();

        let fixture_fasta_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");

        let external_file = Builder::new().suffix(".fa").tempfile().unwrap();
        fs::copy(&fixture_fasta_path, external_file.path()).unwrap();

        let operation_hash = add_files_operation(
            &source_context,
            &[external_file.path().to_string_lossy().to_string()],
            Some("track external fasta"),
        )
        .expect("should commit external fasta");

        let mut write_stream = Cursor::new(Vec::new());
        create_patch(
            &source_context,
            &[HashId::pad_str(&operation_hash.0)],
            None,
            &mut write_stream,
        )
        .unwrap();
        write_stream.set_position(0);
        let patches = load_patches(&mut write_stream);

        let mut target_context = setup_gen_on_disk();

        write_stream.set_position(0);
        apply_patch_archive(&mut target_context, &mut write_stream).unwrap();

        let restored_file = &patches[0]
            .files
            .iter()
            .find(|patch_file| patch_file.file.file_type == FileTypes::Fasta)
            .unwrap()
            .file;

        let restored_asset_path = target_context
            .workspace()
            .asset_dir()
            .unwrap()
            .join(restored_file.clone().hashed_filename());
        assert!(restored_asset_path.exists());
        assert_eq!(
            fs::read(restored_asset_path).unwrap(),
            fs::read(external_file.path()).unwrap()
        );
        assert_legacy_patch_file_tables_are_empty(&target_context);
    }

    #[test]
    fn test_patch_file_skips_archive_for_uri_only_asset() {
        let context = setup_gen_on_disk();
        let file = FileAddition {
            id: HashId::convert_str("patch-uri"),
            asset_uri: "https://example.com/assets/reference.fa.gz".to_string(),
            file_type: FileTypes::Fasta,
            checksum: HashId::convert_str("checksum"),
        };
        let operation_file = OperationFileInfo {
            id: file.id,
            filename: "reference.fa.gz".to_string(),
            file_path: file.asset_uri.clone(),
            asset_uri: file.asset_uri.clone(),
            file_type: file.file_type,
            checksum: file.checksum,
        };

        let patch_file = PatchFile::from_file_addition(&context, operation_file.clone()).unwrap();

        assert_eq!(patch_file.file, operation_file);
        assert_eq!(patch_file.archive_path, "");
    }
}
