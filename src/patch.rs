use std::{
    collections::{HashMap, HashSet},
    fs::{self, File},
    io::{Read, Seek, Write},
    path::PathBuf,
};

use gen_core::{
    CommitRef, DoltHashId, HashId, Sha256Hash,
    errors::{ConfigError, ConnectionError},
};
use gen_models::{
    assets::{AssetUri, LocalAssetUri, OperationKind},
    db::DbContext,
    errors::{FileAdditionError, OperationError},
    file_types::FileTypes,
    history::{
        HistoryError, HistoryStatus, HistoryStore,
        dolt::{DoltHistoryStore, log_entries_for_hashes, log_entries_for_revision},
    },
    operations::OperationFileInfo,
    patch::{
        DoltPatchStatement, apply_dolt_patch, load_dolt_patch, operation_asset_files_for_logs,
        operation_logs_added_between, operation_logs_at_ref,
    },
};
use rusqlite::Error as SQLError;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use zip::{CompressionMethod, ZipArchive, ZipWriter, result::ZipError, write::SimpleFileOptions};

const MANIFEST_ENTRY: &str = "manifest.json";
const PATCH_FORMAT_VERSION: u32 = 2;
const BOOTSTRAP_TABLES: [&str; 18] = [
    "accession_nodes",
    "accessions",
    "annotation_group_samples",
    "annotation_groups",
    "annotations",
    "block_group_edges",
    "block_groups",
    "collections",
    "edges",
    "gen_asset_refs",
    "gen_operation_assets",
    "gen_operation_log",
    "nodes",
    "paths",
    "reference_aliases",
    "sample_lineage",
    "samples",
    "sequences",
];

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub(crate) struct PatchFile {
    file: OperationFileInfo,
    archive_path: String,
}

impl PatchFile {
    fn asset_entry_name(checksum: Sha256Hash, file_type: FileTypes) -> String {
        format!("assets/{checksum}.{}", FileTypes::suffix(file_type))
    }

    // Only local references can name bytes stored under `.gen/assets`. Remote URIs remain patch
    // metadata even when a checksum is known.
    fn local_asset_filename(file: &OperationFileInfo) -> Option<String> {
        if !LocalAssetUri::is_file_uri(&file.asset_uri) {
            return None;
        }
        file.checksum
            .as_ref()
            .map(|checksum| <dyn AssetUri>::from_uri(&file.asset_uri).hashed_filename(checksum))
    }

    fn from_file_addition(
        context: &DbContext,
        file: OperationFileInfo,
    ) -> Result<Self, CreatePatchError> {
        // A local reference without a checksum cannot identify bytes to archive and must fail
        // rather than producing an incomplete patch. A remote reference needs no archive entry.
        let Some(checksum) = file.checksum else {
            if LocalAssetUri::is_file_uri(&file.asset_uri) {
                return Err(CreatePatchError::MissingAssetChecksum(file.id));
            }
            return Ok(Self {
                archive_path: String::new(),
                file,
            });
        };
        let Some(asset_filename) = Self::local_asset_filename(&file) else {
            return Ok(Self {
                archive_path: String::new(),
                file,
            });
        };

        let source_asset_path = context.workspace().asset_dir()?.join(asset_filename);
        let archive_path = if source_asset_path.exists() {
            Self::asset_entry_name(checksum, file.file_type)
        } else {
            String::new()
        };

        Ok(Self { file, archive_path })
    }

    fn source_asset_path(&self, context: &DbContext) -> Result<PathBuf, CreatePatchError> {
        let asset_filename = Self::local_asset_filename(&self.file)
            .ok_or(CreatePatchError::MissingAssetChecksum(self.file.id))?;
        Ok(context.workspace().asset_dir()?.join(asset_filename))
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

        let asset_filename = Self::local_asset_filename(&self.file)
            .ok_or(PatchError::MissingAssetChecksum(self.file.id))?;
        let asset_path = asset_dir.join(asset_filename);
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
    pub hash: DoltHashId,
    pub parent_hash: Option<DoltHashId>,
    pub change_type: String,
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct OperationPatch {
    pub commit: PatchCommit,
    pub(crate) files: Vec<PatchFile>,
    pub(crate) statements: Vec<DoltPatchStatement>,
    pub(crate) commit_message: String,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct OperationPatches {
    pub base_commit_hash: Option<DoltHashId>,
    pub target_commit_hash: DoltHashId,
    pub patches: Vec<OperationPatch>,
}

#[derive(Serialize, Deserialize, Debug)]
struct PatchArchiveHeader {
    patch_format_version: u32,
}

#[derive(Serialize, Deserialize, Debug)]
struct VersionedOperationPatches {
    header: PatchArchiveHeader,
    manifest: serde_json::Value,
}

#[derive(Debug, Error, PartialEq)]
pub enum PatchError {
    #[error("Connection Error: {0}")]
    ConnectionError(#[from] ConnectionError),
    #[error("Config Error: {0}")]
    ConfigError(#[from] ConfigError),
    #[error("I/O error: {0}")]
    Io(String),
    #[error("SQLite Error: {0}")]
    SqliteError(#[from] SQLError),
    #[error("Deserialization Error: {0}")]
    DeserializationError(String),
    #[error("Operation Error: {0}")]
    OperationError(#[from] OperationError),
    #[error("Zip Error: {0}")]
    Zip(String),
    #[error("Unsupported patch format version {found}; current version is {current}")]
    UnsupportedFormatVersion { found: u32, current: u32 },
    #[error("Patch asset {0} has archived content but no checksum")]
    MissingAssetChecksum(HashId),
    #[error(
        "Cannot apply patch: the working set has uncommitted changes. Commit or reset them first."
    )]
    DirtyWorkingSet,
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

/// Recognizes the one dirty working-set shape that patch application treats as an empty target.
///
/// A freshly bootstrapped or legacy Gen database can have its complete schema materialized before
/// that schema has a Dolt commit. Dolt then reports every Gen table as an uncommitted `new table`,
/// even though the database has no application-level work that a patch could overwrite.
/// [`apply_patch_archive_inner`] uses this check as a narrow exception to its normal dirty working
/// set rejection. The exception requires exactly the known bootstrap tables, requires every table
/// to be unstaged and new, and verifies that the application tables contain no rows. Any other
/// working-set shape is treated as user work and remains protected from patch application.
fn is_schema_bootstrap_working_set(
    conn: &gen_models::db::GraphConnection,
    status_rows: &[HistoryStatus],
) -> Result<bool, SQLError> {
    if status_rows.len() != BOOTSTRAP_TABLES.len()
        || status_rows.iter().any(|row| {
            row.staged
                || row.status != "new table"
                || !BOOTSTRAP_TABLES.contains(&row.table_name.as_str())
        })
    {
        return Ok(false);
    }

    let application_row_count = conn.query_row(
        "SELECT \
            (SELECT COUNT(*) FROM accession_nodes) + \
            (SELECT COUNT(*) FROM accessions) + \
            (SELECT COUNT(*) FROM annotation_group_samples) + \
            (SELECT COUNT(*) FROM annotation_groups) + \
            (SELECT COUNT(*) FROM annotations) + \
            (SELECT COUNT(*) FROM block_group_edges) + \
            (SELECT COUNT(*) FROM block_groups) + \
            (SELECT COUNT(*) FROM collections) + \
            (SELECT COUNT(*) FROM edges) + \
            (SELECT COUNT(*) FROM gen_asset_refs) + \
            (SELECT COUNT(*) FROM gen_operation_assets) + \
            (SELECT COUNT(*) FROM gen_operation_log) + \
            (SELECT COUNT(*) FROM paths) + \
            (SELECT COUNT(*) FROM sample_lineage) + \
            (SELECT COUNT(*) FROM samples)",
        [],
        |row| row.get::<_, i64>(0),
    )?;
    Ok(application_row_count == 0)
}

#[derive(Debug, Error)]
pub enum CreatePatchError {
    #[error("Operation {0} does not exist.")]
    OperationNotFound(DoltHashId),
    #[error("Patch selection does not contain an operation.")]
    EmptySelection,
    #[error("Patch asset {0} has archived content but no checksum")]
    MissingAssetChecksum(HashId),
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
    #[error("History error: {0}")]
    History(#[from] HistoryError),
}

/// Parses a comma-separated selection of commits and Dolt commit ranges for a patch.
///
/// Two-dot and three-dot selections use Dolt's native `dolt_log(revision)` behavior. As in Git,
/// `start..end` excludes commits reachable from `start`; it is not an inclusive slice of a
/// materialized branch history.
///
/// # Arguments
///
/// * `history_store` - The history store used to resolve commit references.
/// * `source_branch` - The branch used to qualify `HEAD` references without changing checkout.
/// * `operations` - Commit references or Dolt `start..end`/`start...end` ranges to select.
///
/// # Errors
///
/// Returns an error for an empty selection, an invalid reference or range, or a resolved commit
/// that does not exist in the repository.
pub fn parse_patch_commit_selection(
    history_store: &impl HistoryStore,
    source_branch: &str,
    operations: &str,
) -> Result<Vec<DoltHashId>, Box<dyn std::error::Error>> {
    let source_relative_ref = |reference: &str| {
        if reference == "HEAD" {
            source_branch.to_string()
        } else if let Some(offset) = reference.strip_prefix("HEAD~") {
            format!("{source_branch}~{offset}")
        } else {
            reference.to_string()
        }
    };
    let mut selected_operations = Vec::new();

    for operation in operations.split(',') {
        let operation = operation.trim();
        if operation.is_empty() {
            return Err("Patch selection cannot be empty.".into());
        }

        let range = operation
            .split_once("...")
            .map(|(start, end)| (start, end, "..."))
            .or_else(|| {
                operation
                    .split_once("..")
                    .map(|(start, end)| (start, end, ".."))
            });
        if let Some((start, end, separator)) = range {
            let revision = format!(
                "{}{separator}{}",
                source_relative_ref(start),
                source_relative_ref(end)
            );
            let mut entries = log_entries_for_revision(history_store.graph(), &revision)?;
            if entries.is_empty() {
                return Err(format!("Patch range '{operation}' selects no commits.").into());
            }
            entries.reverse();
            selected_operations.extend(entries.into_iter().map(|entry| entry.commit_hash));
            continue;
        }

        let commit_hash =
            history_store.resolve_operation_hash(&CommitRef(source_relative_ref(operation)))?;
        if log_entries_for_hashes(history_store.graph(), &[commit_hash])?.is_empty() {
            return Err(format!("Resolved commit '{operation}' does not exist.").into());
        }
        selected_operations.push(commit_hash);
    }
    Ok(selected_operations)
}

/// Selects patch commits from a branch without changing the active checkout.
///
/// `HEAD` selects every commit after the branch diverged from the current branch. Other selections
/// are parsed by [`parse_patch_commit_selection`].
///
/// # Errors
///
/// Returns an error if no branch is checked out, history queries fail, or the requested selection
/// is invalid.
pub fn patch_operations_for_branch(
    history_store: &impl HistoryStore,
    branch_name: &str,
    operation: &str,
) -> Result<Vec<DoltHashId>, Box<dyn std::error::Error>> {
    let current_branch_name = history_store
        .current_branch()?
        .ok_or("No current branch is checked out.")?;
    let selection = if branch_name != current_branch_name.0 && operation == "HEAD" {
        format!("{}..HEAD", current_branch_name.0)
    } else {
        operation.to_string()
    };
    parse_patch_commit_selection(history_store, branch_name, &selection)
}

fn serialize_operation_patches(
    operation_patches: &OperationPatches,
) -> Result<Vec<u8>, CreatePatchError> {
    Ok(serde_json::to_vec_pretty(&VersionedOperationPatches {
        header: PatchArchiveHeader {
            patch_format_version: PATCH_FORMAT_VERSION,
        },
        manifest: serde_json::to_value(operation_patches)?,
    })?)
}

fn deserialize_manifest<T>(manifest: serde_json::Value) -> Result<T, PatchError>
where
    T: for<'de> Deserialize<'de>,
{
    serde_json::from_value(manifest)
        .map_err(|error| PatchError::DeserializationError(error.to_string()))
}

fn deserialize_operation_patches(
    patch_format_version: u32,
    manifest: serde_json::Value,
) -> Result<OperationPatches, PatchError> {
    match patch_format_version {
        PATCH_FORMAT_VERSION => deserialize_manifest(manifest),
        found => Err(PatchError::UnsupportedFormatVersion {
            found,
            current: PATCH_FORMAT_VERSION,
        }),
    }
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

    let versioned = serde_json::from_slice::<VersionedOperationPatches>(&buffer)
        .map_err(|error| PatchError::DeserializationError(error.to_string()))?;
    deserialize_operation_patches(versioned.header.patch_format_version, versioned.manifest)
}

pub(crate) fn load_operation_patches<R>(reader: R) -> Result<OperationPatches, PatchError>
where
    R: Read + Seek,
{
    let mut archive = ZipArchive::new(reader)?;
    read_operation_patches(&mut archive)
}

fn patch_log_ids_and_kind(
    context: &DbContext,
    parent_commit_hash: Option<&DoltHashId>,
    commit_ref: &str,
    parent_ref: Option<&str>,
) -> Result<(Vec<HashId>, String), CreatePatchError> {
    let log_rows = if let Some(parent_commit_hash) = parent_commit_hash {
        let parent_ref = parent_ref
            .map(str::to_string)
            .unwrap_or_else(|| parent_commit_hash.to_string());
        operation_logs_added_between(context.graph().conn(), &parent_ref, commit_ref)?
    } else {
        operation_logs_at_ref(context.graph().conn(), commit_ref)?
    };

    let mut log_ids = Vec::with_capacity(log_rows.len());
    let mut change_type = None;
    for (log_id, operation_kind) in log_rows {
        if change_type.is_none() {
            change_type = Some(operation_kind.to_string());
        }
        log_ids.push(log_id);
    }
    Ok((
        log_ids,
        change_type.unwrap_or_else(|| OperationKind::HistoryCommit.to_string()),
    ))
}

/// Gathers files associated with the patch
fn load_history_patch_files(
    context: &DbContext,
    parent_commit_hash: Option<&DoltHashId>,
    commit_ref: &str,
    parent_ref: Option<&str>,
) -> Result<(Vec<OperationFileInfo>, String), CreatePatchError> {
    let (log_ids, change_type) =
        patch_log_ids_and_kind(context, parent_commit_hash, commit_ref, parent_ref)?;

    if log_ids.is_empty() {
        return Ok((Vec::new(), change_type));
    }

    Ok((
        operation_asset_files_for_logs(context.graph().conn(), &log_ids, Some(commit_ref))?,
        change_type,
    ))
}

fn build_patch_commit(
    commit_hash: &DoltHashId,
    parent_commit_hash: Option<&DoltHashId>,
    change_type: String,
) -> PatchCommit {
    PatchCommit {
        hash: *commit_hash,
        parent_hash: parent_commit_hash.copied(),
        change_type,
    }
}

fn build_operation_patch(
    context: &DbContext,
    commit_hash: &DoltHashId,
    parent_commit_hash: Option<&DoltHashId>,
    commit_ref: &str,
    parent_ref: Option<&str>,
    commit_message: &str,
) -> Result<OperationPatch, CreatePatchError> {
    let (files, change_type) =
        load_history_patch_files(context, parent_commit_hash, commit_ref, parent_ref)?;
    let statements = if let Some(parent_ref) = parent_ref {
        load_dolt_patch(context.graph().conn(), parent_ref, commit_ref)?
    } else {
        Vec::new()
    };
    let files = files
        .into_iter()
        .filter(|file| LocalAssetUri::is_file_uri(&file.asset_uri))
        .map(|file| PatchFile::from_file_addition(context, file))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(OperationPatch {
        commit: build_patch_commit(commit_hash, parent_commit_hash, change_type),
        files,
        statements,
        commit_message: commit_message.to_string(),
    })
}

pub fn create_patch<W>(
    context: &DbContext,
    operations: &[DoltHashId],
    write_stream: &mut W,
) -> Result<(), CreatePatchError>
where
    W: Write + Seek,
{
    let commit_entries = log_entries_for_hashes(context.graph().conn(), operations)?
        .into_iter()
        .map(|entry| (entry.commit_hash, entry))
        .collect::<HashMap<_, _>>();
    let mut patches = vec![];
    for hash in operations {
        let commit_entry = commit_entries
            .get(hash)
            .ok_or(CreatePatchError::OperationNotFound(*hash))?;
        let commit_ref = hash.to_string();
        let parent_commit_hash = commit_entry.parent_hash;
        let parent_ref = parent_commit_hash.map(|parent_hash| parent_hash.to_string());
        println!("Creating patch for Operation {id}", id = hash);
        patches.push(build_operation_patch(
            context,
            hash,
            parent_commit_hash.as_ref(),
            &commit_ref,
            parent_ref.as_deref(),
            &commit_entry.message,
        )?);
    }

    let target_commit_hash = patches
        .last()
        .map(|patch| patch.commit.hash)
        .ok_or(CreatePatchError::EmptySelection)?;
    let base_commit_hash = patches.first().and_then(|patch| patch.commit.parent_hash);

    let operation_patches = OperationPatches {
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

pub fn load_patches<R>(reader: R) -> Vec<OperationPatch>
where
    R: Read + Seek,
{
    load_operation_patches(reader).unwrap().patches
}

fn apply_patch_archive_inner<R>(
    context: &mut DbContext,
    reader: R,
    reject_dirty_working_set: bool,
) -> Result<(), PatchError>
where
    R: Read + Seek,
{
    let history_store = DoltHistoryStore::new(context.graph().conn());
    let status_rows = history_store.status()?;
    if reject_dirty_working_set
        && !status_rows.is_empty()
        && !is_schema_bootstrap_working_set(context.graph().conn(), &status_rows)?
    {
        return Err(PatchError::DirtyWorkingSet);
    }

    let mut archive = ZipArchive::new(reader)?;
    let operation_patches = read_operation_patches(&mut archive)?;
    let mut applied_patch = false;
    for patch in &operation_patches.patches {
        let patch_has_changes = !patch.statements.is_empty() || !patch.files.is_empty();
        apply_dolt_patch(context.graph().conn(), &patch.statements)?;
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

pub fn apply_patch_archive<R>(context: &mut DbContext, reader: R) -> Result<(), PatchError>
where
    R: Read + Seek,
{
    apply_patch_archive_inner(context, reader, true)
}

pub(crate) fn apply_patch_archive_to_isolated_context<R>(
    context: &mut DbContext,
    reader: R,
) -> Result<(), PatchError>
where
    R: Read + Seek,
{
    apply_patch_archive_inner(context, reader, false)
}

#[cfg(test)]
mod tests {
    use std::{fs, io::Cursor, path::PathBuf};

    use gen_core::BranchName;
    use gen_models::{
        annotations::{AnnotationFileChecksumOverrides, add_annotation_file},
        assets::{AssetRef, AssetRole, OperationAsset, OperationKind, OperationLog},
        collection::Collection,
        history::dolt::DoltHistoryStore,
        operations::{OperationFile, add_files_operation, commit_operation_summary},
        traits::Query,
    };
    use tempfile::Builder;

    use super::*;
    use crate::{
        imports::fasta::import_fasta, test_helpers::setup_gen_on_disk,
        views::annotation_files::load_annotation_file_entries,
    };

    fn create_history_backed_patch_operations(context: &DbContext) -> (DoltHashId, DoltHashId) {
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let gff_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.gff");

        let add_file_commit_hash = add_files_operation(
            context,
            &[OperationFile::new(fasta_path.to_string_lossy())],
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
            AnnotationFileChecksumOverrides::default(),
        )
        .expect("should commit annotation fixture");
        let annotation_operation = current_history_operation_hash(context);
        (add_file_commit_hash, annotation_operation)
    }

    fn current_history_operation_hash(context: &DbContext) -> DoltHashId {
        let history_store = DoltHistoryStore::new(context.graph().conn());
        history_store
            .current_head()
            .expect("should resolve current head")
            .expect("should have a current head")
    }

    fn write_patch_manifest(
        patch_format_version: u32,
        manifest: serde_json::Value,
    ) -> Cursor<Vec<u8>> {
        let versioned = VersionedOperationPatches {
            header: PatchArchiveHeader {
                patch_format_version,
            },
            manifest,
        };
        let mut archive = Cursor::new(Vec::new());
        let mut writer = ZipWriter::new(&mut archive);
        writer
            .start_file(MANIFEST_ENTRY, SimpleFileOptions::default())
            .expect("should create test patch manifest");
        writer
            .write_all(
                &serde_json::to_vec_pretty(&versioned)
                    .expect("should serialize test patch manifest"),
            )
            .expect("should write test patch manifest");
        writer.finish().expect("should finish test patch archive");
        archive.set_position(0);
        archive
    }

    fn archive_with_patch_format_version(
        source: &mut Cursor<Vec<u8>>,
        patch_format_version: u32,
    ) -> Cursor<Vec<u8>> {
        source.set_position(0);
        let mut source_archive = ZipArchive::new(source).expect("should open source patch archive");
        let mut manifest_file = source_archive
            .by_name(MANIFEST_ENTRY)
            .expect("should read source patch manifest");
        let mut manifest: serde_json::Value = serde_json::from_reader(&mut manifest_file)
            .expect("should parse source patch manifest");
        manifest["header"]["patch_format_version"] = serde_json::Value::from(patch_format_version);
        drop(manifest_file);
        drop(source_archive);

        write_patch_manifest(patch_format_version, manifest["manifest"].take())
    }

    #[test]
    fn test_parse_patch_commit_selection_resolves_refs_from_the_source_branch() {
        let context = setup_gen_on_disk();
        let history_store = DoltHistoryStore::new(context.graph().conn());
        let original_branch = history_store
            .current_branch()
            .expect("should read current branch")
            .expect("should have a current branch");
        let source_branch = BranchName("patch-source-refs".to_string());
        history_store
            .create_branch(&source_branch, None)
            .expect("should create source branch");
        history_store
            .checkout_branch(&source_branch)
            .expect("should checkout source branch");
        let (first_operation, second_operation) = create_history_backed_patch_operations(&context);
        history_store
            .checkout_branch(&original_branch)
            .expect("should restore original branch");

        let selected = parse_patch_commit_selection(
            &history_store,
            &source_branch.0,
            &format!("HEAD~1,HEAD,{first_operation}"),
        )
        .expect("should resolve branch-relative refs and an explicit commit hash");

        assert_eq!(
            selected,
            vec![first_operation, second_operation, first_operation],
            "Dolt refs should resolve relative to the non-current source branch"
        );
        Collection::create(context.graph().conn(), "main-only-range")
            .expect("should create main-only collection");
        let main_commit = history_store
            .commit_all("main-only range commit")
            .expect("should commit main-only change");
        let cross_branch_selection = parse_patch_commit_selection(
            &history_store,
            &source_branch.0,
            "HEAD,patch-source-refs..main",
        )
        .expect("should combine valid commits and ranges from different branches");
        assert_eq!(
            cross_branch_selection,
            vec![second_operation, main_commit],
            "patch selection should not impose source-branch reachability on valid commits"
        );
        assert_eq!(
            history_store
                .current_branch()
                .expect("should read current branch after selection"),
            Some(original_branch),
            "source-branch ref resolution should not change the active checkout"
        );
    }

    #[test]
    fn test_parse_patch_commit_selection_accepts_ranges_and_lists() {
        let context = setup_gen_on_disk();
        let (first_operation, second_operation) = create_history_backed_patch_operations(&context);
        let history_store = DoltHistoryStore::new(context.graph().conn());
        let branch_name = history_store
            .current_branch()
            .expect("should read current branch")
            .expect("should have a current branch");

        let selected =
            parse_patch_commit_selection(&history_store, &branch_name.0, "HEAD~1..HEAD, HEAD~1")
                .expect("should parse an inclusive range and a list item");

        assert_eq!(
            selected,
            vec![second_operation, first_operation],
            "two-dot Dolt range should exclude HEAD~1 before adding it explicitly"
        );
        assert!(
            parse_patch_commit_selection(&history_store, &branch_name.0, "HEAD,",).is_err(),
            "an empty list item should be rejected"
        );
    }

    #[test]
    fn test_parse_patch_commit_selection_rejects_invalid_commits_and_ranges() {
        let context = setup_gen_on_disk();
        create_history_backed_patch_operations(&context);
        let history_store = DoltHistoryStore::new(context.graph().conn());
        let branch_name = history_store
            .current_branch()
            .expect("should read current branch")
            .expect("should have a current branch");

        let missing_commit = DoltHashId::default().to_string();
        let missing_commit_error =
            parse_patch_commit_selection(&history_store, &branch_name.0, &missing_commit)
                .expect_err("should reject a hash that does not identify a repository commit");
        assert!(
            missing_commit_error.to_string().contains("does not exist"),
            "missing commit error should distinguish absence from branch reachability: {missing_commit_error}"
        );

        parse_patch_commit_selection(&history_store, &branch_name.0, "missing-ref..HEAD")
            .expect_err("should pass an invalid range error through from Dolt");

        let empty_range_error =
            parse_patch_commit_selection(&history_store, &branch_name.0, "HEAD..HEAD")
                .expect_err("should reject a valid range that selects no commits");
        assert!(
            empty_range_error.to_string().contains("selects no commits"),
            "empty range error should report that the range is valid but empty: {empty_range_error}"
        );
    }

    #[test]
    fn test_patch_operations_for_branch_selects_divergence_without_changing_checkout() {
        let context = setup_gen_on_disk();
        create_history_backed_patch_operations(&context);
        let history_store = DoltHistoryStore::new(context.graph().conn());
        let original_branch = history_store
            .current_branch()
            .expect("should read current branch")
            .expect("should have a current branch");
        let feature_branch = BranchName("patch-feature".to_string());
        history_store
            .create_branch(&feature_branch, None)
            .expect("should create feature branch");
        history_store
            .checkout_branch(&feature_branch)
            .expect("should check out feature branch");
        Collection::create(context.graph().conn(), "feature-only")
            .expect("should create feature collection");
        let feature_commit = history_store
            .commit_all("feature-only")
            .expect("should commit feature collection");
        history_store
            .checkout_branch(&original_branch)
            .expect("should restore original branch before selection");
        Collection::create(context.graph().conn(), "dirty-current-branch")
            .expect("should dirty the current branch");

        let selected = patch_operations_for_branch(&history_store, &feature_branch.0, "HEAD")
            .expect("should select divergent feature commits");

        assert_eq!(selected, vec![feature_commit]);
        assert_eq!(
            history_store
                .current_branch()
                .expect("should read unchanged branch"),
            Some(original_branch.clone()),
            "branch selection should preserve the original checkout"
        );
        assert!(
            Collection::exists(context.graph().conn(), "dirty-current-branch"),
            "branch selection should preserve uncommitted changes"
        );

        patch_operations_for_branch(&history_store, &feature_branch.0, "missing-commit")
            .expect_err("should reject an unresolved feature commit");
        assert_eq!(
            history_store
                .current_branch()
                .expect("should read branch after failed selection"),
            Some(original_branch),
            "failed branch selection should also preserve the original checkout"
        );
    }

    #[test]
    fn test_creates_patch() {
        let context = setup_gen_on_disk();
        let (op_1, op_2) = create_history_backed_patch_operations(&context);
        let mut write_stream = Cursor::new(Vec::new());
        create_patch(&context, &[op_1, op_2], &mut write_stream).unwrap();
        write_stream.set_position(0);
        load_patches(&mut write_stream);
    }

    #[test]
    fn test_create_patch_reads_commits_from_different_branches_without_changing_dirty_checkout() {
        let context = setup_gen_on_disk();
        let history_store = DoltHistoryStore::new(context.graph().conn());
        let current_branch = history_store
            .current_branch()
            .expect("should read current branch")
            .expect("should have a current branch");
        let source_branch = BranchName("patch-source".to_string());
        history_store
            .create_branch(&source_branch, None)
            .expect("should create source branch");
        history_store
            .checkout_branch(&source_branch)
            .expect("should checkout source branch");
        Collection::create(context.graph().conn(), "source-collection")
            .expect("should insert source collection");
        let source_commit = history_store
            .commit_all("source commit")
            .expect("should commit source branch change");
        history_store
            .checkout_branch(&current_branch)
            .expect("should restore current branch");
        Collection::create(context.graph().conn(), "current-branch-collection")
            .expect("should insert current-branch collection");
        let current_commit = history_store
            .commit_all("current branch commit")
            .expect("should commit current-branch change");
        Collection::create(context.graph().conn(), "uncommitted-current-collection")
            .expect("should dirty the current branch");

        let mut patch_archive = Cursor::new(Vec::new());
        create_patch(
            &context,
            &[source_commit, current_commit],
            &mut patch_archive,
        )
        .expect("should create patches without checking out either commit's branch");

        assert_eq!(
            history_store
                .current_branch()
                .expect("should read current branch after patch creation"),
            Some(current_branch),
            "patch creation should preserve the active branch"
        );
        assert!(
            Collection::exists(context.graph().conn(), "uncommitted-current-collection"),
            "patch creation should preserve uncommitted current-branch changes"
        );
        patch_archive.set_position(0);
        let patches = load_patches(&mut patch_archive);
        assert_eq!(
            patches.len(),
            2,
            "should create one patch per selected commit"
        );
        assert_eq!(
            patches[0].commit.hash, source_commit,
            "patch should retain the selected source commit hash"
        );
        assert_eq!(
            patches[0].commit_message, "source commit",
            "patch should retain the selected source commit message"
        );
        assert!(
            !patches[0].statements.is_empty(),
            "patch should contain the source branch change"
        );
        assert_eq!(
            patches[1].commit.hash, current_commit,
            "patch should retain the selected current-branch commit hash"
        );
        assert_eq!(
            patches[1].commit_message, "current branch commit",
            "patch should retain the current-branch commit message"
        );
        assert!(
            !patches[1].statements.is_empty(),
            "patch should contain the current-branch change"
        );
    }

    #[test]
    fn test_patch_create_current_head_keeps_head_changes() {
        let context = setup_gen_on_disk();
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let operation_summary = import_fasta(
            &context,
            &fasta_path.to_string_lossy().to_string(),
            "default",
            "foo",
            false,
            &[],
        )
        .expect("should import fasta fixture");
        commit_operation_summary(&context, &operation_summary)
            .expect("should commit fasta fixture import");
        let head_operation = current_history_operation_hash(&context);
        let mut write_stream = Cursor::new(Vec::new());

        create_patch(&context, &[head_operation], &mut write_stream)
            .expect("should create a patch from the current head");
        write_stream.set_position(0);
        let patches = load_patches(&mut write_stream);

        assert_eq!(patches.len(), 1, "should create a single patch entry");
        assert_eq!(
            patches[0].commit.change_type, "fasta_addition",
            "current head patch should point at the imported commit"
        );
        assert!(
            !patches[0].statements.is_empty(),
            "current head patch should include graph table changes"
        );
    }

    #[test]
    fn test_cross_db_patches() {
        let source_context = setup_gen_on_disk();
        let source_graph = source_context.graph().conn();
        let (op_1, op_2) = create_history_backed_patch_operations(&source_context);
        let mut write_stream = Cursor::new(Vec::new());
        create_patch(&source_context, &[op_1, op_2], &mut write_stream).unwrap();
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

        let annotation_entries = load_annotation_file_entries(target_context.graph().conn(), None);
        assert_eq!(annotation_entries.len(), 1);
        assert_eq!(annotation_entries[0].display_name, "fixture-track");
    }

    #[test]
    fn test_cross_branch_patches() {
        let mut context = setup_gen_on_disk();
        let history_store = DoltHistoryStore::new(context.graph().conn());

        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let _op_1 = add_files_operation(
            &context,
            &[OperationFile::new(fasta_path.to_string_lossy())],
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
            AnnotationFileChecksumOverrides::default(),
        )
        .expect("should commit branch annotation");
        let op_2 = current_history_operation_hash(&context);
        let mut write_stream = Cursor::new(Vec::new());
        create_patch(&context, &[op_2], &mut write_stream).unwrap();

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
        assert!(Collection::exists(
            context.graph().conn(),
            "main-side-collection"
        ));

        write_stream.set_position(0);
        let res = apply_patch_archive(&mut context, &mut write_stream);
        assert!(
            matches!(res, Err(PatchError::SqliteError(_))),
            "reapplying native Dolt patch statements should reject duplicate data, got {res:?}"
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
        create_patch(&context, &[op_1, op_2], &mut write_stream).unwrap();
        write_stream.set_position(0);
        let loaded_patches = load_patches(&mut write_stream);

        assert_eq!(loaded_patches[0].commit.hash, op_1);
        assert_eq!(loaded_patches[1].commit.hash, op_2);
        assert!(!loaded_patches[0].files.is_empty());
        assert!(!loaded_patches[1].files.is_empty());
        assert!(!loaded_patches[0].statements.is_empty());
        assert!(!loaded_patches[1].statements.is_empty());
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

        create_patch(&context, &[op_1, op_2], &mut write_stream).unwrap();
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
        assert_eq!(
            manifest_value["manifest"]["target_commit_hash"]
                .as_str()
                .expect("should serialize target Dolt hash")
                .len(),
            40
        );
        assert_eq!(
            manifest_value["manifest"]["patches"][0]["commit"]["hash"]
                .as_str()
                .expect("should serialize commit Dolt hash")
                .len(),
            40
        );
        assert!(
            manifest_value["manifest"]["patches"][0]["commit"]
                .get("commit_hash")
                .is_none()
        );
    }

    #[test]
    fn test_patch_manifest_rejects_unsupported_versions() {
        let context = setup_gen_on_disk();
        let (op_1, op_2) = create_history_backed_patch_operations(&context);
        let mut current_archive = Cursor::new(Vec::new());
        create_patch(&context, &[op_1, op_2], &mut current_archive).unwrap();

        for unsupported_version in [PATCH_FORMAT_VERSION - 1, PATCH_FORMAT_VERSION + 1] {
            let unsupported_archive =
                archive_with_patch_format_version(&mut current_archive, unsupported_version);
            let error = load_operation_patches(unsupported_archive)
                .expect_err("should reject an unsupported patch version");
            assert_eq!(
                error,
                PatchError::UnsupportedFormatVersion {
                    found: unsupported_version,
                    current: PATCH_FORMAT_VERSION,
                }
            );
        }
    }

    #[test]
    fn test_patch_apply_rejects_dirty_working_set() {
        let source_context = setup_gen_on_disk();
        let (op_1, _op_2) = create_history_backed_patch_operations(&source_context);
        let mut patch_archive = Cursor::new(Vec::new());
        create_patch(&source_context, &[op_1], &mut patch_archive).unwrap();

        let mut target_context = setup_gen_on_disk();
        Collection::create(target_context.graph().conn(), "uncommitted")
            .expect("should create an uncommitted collection");
        let history_store = DoltHistoryStore::new(target_context.graph().conn());
        let history_before = history_store.log(None).expect("should read target history");
        patch_archive.set_position(0);

        let error = apply_patch_archive(&mut target_context, &mut patch_archive)
            .expect_err("should reject a dirty target repository");

        assert_eq!(error, PatchError::DirtyWorkingSet);
        assert!(Collection::exists(
            target_context.graph().conn(),
            "uncommitted"
        ));
        assert_eq!(
            DoltHistoryStore::new(target_context.graph().conn())
                .log(None)
                .expect("should read unchanged target history"),
            history_before
        );
    }

    #[test]
    fn test_patch_empty_db() {
        let context = setup_gen_on_disk();
        let (_op_1, op_2) = create_history_backed_patch_operations(&context);
        let mut write_stream = Cursor::new(Vec::new());
        create_patch(&context, &[op_2], &mut write_stream).unwrap();

        let mut fresh_context = setup_gen_on_disk();
        write_stream.set_position(0);
        apply_patch_archive(&mut fresh_context, &mut write_stream).unwrap();
        let annotation_entries = load_annotation_file_entries(fresh_context.graph().conn(), None);
        assert_eq!(annotation_entries.len(), 1);
        assert_eq!(annotation_entries[0].display_name, "fixture-track");
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
        create_patch(&source_context, &[op_1, op_2], &mut write_stream).unwrap();
        write_stream.set_position(0);

        let mut target_context = setup_gen_on_disk();
        apply_patch_archive(&mut target_context, &mut write_stream).unwrap();
        let annotation_entries = load_annotation_file_entries(target_context.graph().conn(), None);
        assert_eq!(annotation_entries.len(), 1);
        assert_eq!(annotation_entries[0].display_name, "fixture-track");
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
            &[OperationFile::new(external_file.path().to_string_lossy())],
            Some("track external fasta"),
        )
        .expect("should commit external fasta");

        let mut write_stream = Cursor::new(Vec::new());
        create_patch(&source_context, &[operation_hash], &mut write_stream).unwrap();
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

        let restored_asset_path = target_context.workspace().asset_dir().unwrap().join(
            PatchFile::local_asset_filename(restored_file)
                .expect("restored archive asset should have a checksum"),
        );
        assert!(restored_asset_path.exists());
        assert_eq!(
            fs::read(restored_asset_path).unwrap(),
            fs::read(external_file.path()).unwrap()
        );
    }

    #[test]
    fn test_patch_file_skips_archive_for_uri_only_asset() {
        let context = setup_gen_on_disk();
        let operation_file = OperationFileInfo {
            id: HashId::convert_str("patch-uri"),
            asset_uri: "https://example.com/assets/reference.fa.gz".to_string(),
            filename: "reference.fa.gz".to_string(),
            file_path: "https://example.com/assets/reference.fa.gz".to_string(),
            file_type: FileTypes::Fasta,
            checksum: None,
        };

        let patch_file = PatchFile::from_file_addition(&context, operation_file.clone()).unwrap();

        assert_eq!(patch_file.file, operation_file);
        assert_eq!(patch_file.archive_path, "");
    }

    #[test]
    fn test_create_patch_rejects_checksumless_local_asset() {
        let source_context = setup_gen_on_disk();
        let graph_conn = source_context.graph().conn();
        let operation_log = OperationLog {
            id: HashId::convert_str("checksumless-local-operation"),
            operation_kind: OperationKind::AddFile,
            command: "track checksumless local file".to_string(),
            created_on: 1,
        };
        let local_asset = AssetRef {
            id: HashId::convert_str("checksumless-local-patch-asset"),
            uri: "file://inputs/reference.fa".to_string(),
            file_type: FileTypes::Fasta.as_str().to_string(),
            checksum: None,
            size: None,
            role: AssetRole::Input,
            logical_path: Some("inputs/reference.fa".to_string()),
            name: Some("reference.fa".to_string()),
            created_on: 1,
            upstream_asset_ref_id: None,
        };
        OperationLog::create(graph_conn, &operation_log).expect("should create operation log");
        AssetRef::create(graph_conn, &local_asset).expect("should create local asset");
        OperationAsset::create(
            graph_conn,
            &OperationAsset {
                log_id: operation_log.id,
                asset_ref_id: local_asset.id,
                role: AssetRole::Input,
            },
        )
        .expect("should link local asset");
        let operation_hash = DoltHistoryStore::new(graph_conn)
            .commit_all("track checksumless local file")
            .expect("should commit local asset");

        let mut patch_stream = Cursor::new(Vec::new());
        let error = create_patch(&source_context, &[operation_hash], &mut patch_stream)
            .expect_err("checksumless local patch asset should be rejected");

        assert!(matches!(
            error,
            CreatePatchError::MissingAssetChecksum(id)
                if id == local_asset.id
        ));
    }

    #[test]
    fn test_patch_round_trip_with_only_checksumless_remote_uri() {
        let source_context = setup_gen_on_disk();
        let graph_conn = source_context.graph().conn();
        let operation_log = OperationLog {
            id: HashId::convert_str("remote-only-operation"),
            operation_kind: OperationKind::AddFile,
            command: "track remote URI".to_string(),
            created_on: 1,
        };
        let remote_asset = AssetRef {
            id: HashId::convert_str("checksumless-remote-asset"),
            uri: "s3://private-bucket/reference.fa".to_string(),
            file_type: FileTypes::Fasta.as_str().to_string(),
            checksum: None,
            size: None,
            role: AssetRole::Input,
            logical_path: Some("inputs/reference.fa".to_string()),
            name: Some("reference.fa".to_string()),
            created_on: 1,
            upstream_asset_ref_id: None,
        };
        OperationLog::create(graph_conn, &operation_log).expect("should create operation log");
        AssetRef::create(graph_conn, &remote_asset).expect("should create remote asset");
        OperationAsset::create(
            graph_conn,
            &OperationAsset {
                log_id: operation_log.id,
                asset_ref_id: remote_asset.id,
                role: AssetRole::Input,
            },
        )
        .expect("should link remote asset");
        let operation_hash = DoltHistoryStore::new(graph_conn)
            .commit_all("track remote URI")
            .expect("should commit remote asset");

        let mut patch_stream = Cursor::new(Vec::new());
        create_patch(&source_context, &[operation_hash], &mut patch_stream)
            .expect("should create remote-only patch");
        patch_stream.set_position(0);
        let patches = load_patches(&mut patch_stream);
        assert_eq!(patches.len(), 1);
        assert!(
            patches[0].files.is_empty(),
            "remote URIs should be carried by Dolt statements, not ZIP asset entries"
        );
        assert!(
            !patches[0].statements.is_empty(),
            "remote asset metadata should remain in the Dolt patch"
        );

        let mut target_context = setup_gen_on_disk();
        patch_stream.set_position(0);
        apply_patch_archive(&mut target_context, &mut patch_stream)
            .expect("should apply remote-only patch");

        assert_eq!(
            AssetRef::all(target_context.graph().conn()),
            vec![remote_asset]
        );
    }
}
