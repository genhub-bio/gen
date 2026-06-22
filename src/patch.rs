use std::{
    fs::{self, File},
    io::{Read, Seek, Write},
};

use gen_core::{
    HashId,
    errors::{ConfigError, ConnectionError},
    traits::Capnp,
};
use gen_models::{
    changesets::{DatabaseChangeset, apply_changeset},
    db::DbContext,
    errors::{ChangesetError, OperationError},
    file_types::FileTypes,
    operations::{FileAddition, Operation, OperationFile, OperationInfo, OperationSummary},
    session_operations::{DependencyModels, end_operation, start_operation},
    traits::Query,
};
use rusqlite::{Error as SQLError, params, types::Value};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use zip::{CompressionMethod, ZipArchive, ZipWriter, result::ZipError, write::SimpleFileOptions};

use crate::{
    gen_schema_capnp::{operation_patch, operation_patches, patch_file},
    get_connection,
};

const MANIFEST_ENTRY: &str = "manifest.capnp";

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct PatchFile {
    file: FileAddition,
    archive_path: String,
}

impl PatchFile {
    fn asset_entry_name(checksum: HashId, file_type: FileTypes) -> String {
        format!("assets/{checksum}.{}", FileTypes::suffix(file_type))
    }

    fn from_file_addition(
        context: &DbContext,
        file: FileAddition,
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
            .join(self.file.clone().hashed_filename()))
    }

    fn restore_from_archive<R>(
        &self,
        context: &DbContext,
        archive: &mut ZipArchive<R>,
    ) -> Result<(), PatchError>
    where
        R: Read + Seek,
    {
        if self.file.asset_uri.is_empty() || self.archive_path.is_empty() {
            return Ok(());
        }

        let asset_dir = context
            .workspace()
            .asset_dir()
            .map_err(ConnectionError::from)?;
        fs::create_dir_all(&asset_dir).map_err(|err| PatchError::Io(err.to_string()))?;

        let asset_path = asset_dir.join(self.file.clone().hashed_filename());
        if asset_path.exists() {
            return Ok(());
        }

        let mut zip_file = archive.by_name(&self.archive_path)?;
        let mut asset_file =
            File::create(asset_path).map_err(|err| PatchError::Io(err.to_string()))?;
        std::io::copy(&mut zip_file, &mut asset_file)
            .map_err(|err| PatchError::Io(err.to_string()))?;
        Ok(())
    }
}

impl<'a> Capnp<'a> for PatchFile {
    type Builder = patch_file::Builder<'a>;
    type Reader = patch_file::Reader<'a>;

    fn write_capnp(&self, builder: &mut Self::Builder) {
        self.file.write_capnp(&mut builder.reborrow().init_file());
        builder.set_archive_path(&self.archive_path);
    }

    fn read_capnp(reader: Self::Reader) -> Self {
        Self {
            file: FileAddition::read_capnp(reader.get_file().expect("should have file")),
            archive_path: reader
                .get_archive_path()
                .expect("should have archive path")
                .to_string()
                .unwrap(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct OperationPatch {
    pub operation: Operation,
    files: Vec<PatchFile>,
    summary: OperationSummary,
    dependencies: DependencyModels,
    changeset: DatabaseChangeset,
}

impl<'a> Capnp<'a> for OperationPatch {
    type Builder = operation_patch::Builder<'a>;
    type Reader = operation_patch::Reader<'a>;

    fn write_capnp(&self, builder: &mut Self::Builder) {
        self.operation
            .write_capnp(&mut builder.reborrow().init_operation());

        let mut files_builder = builder.reborrow().init_files(self.files.len() as u32);
        for (i, file) in self.files.iter().enumerate() {
            file.write_capnp(&mut files_builder.reborrow().get(i as u32));
        }

        self.summary
            .write_capnp(&mut builder.reborrow().init_summary());

        self.dependencies
            .write_capnp(&mut builder.reborrow().init_dependencies());

        self.changeset
            .write_capnp(&mut builder.reborrow().init_changeset());
    }

    fn read_capnp(reader: Self::Reader) -> Self {
        let operation =
            Operation::read_capnp(reader.get_operation().expect("should have operation"));

        let files_reader = reader.get_files().expect("should have files");
        let mut files = Vec::with_capacity(files_reader.len() as usize);
        for file_reader in files_reader.iter() {
            files.push(PatchFile::read_capnp(file_reader));
        }

        let summary =
            OperationSummary::read_capnp(reader.get_summary().expect("should have summary"));

        let dependencies = DependencyModels::read_capnp(
            reader.get_dependencies().expect("should have dependencies"),
        );

        let changeset =
            DatabaseChangeset::read_capnp(reader.get_changeset().expect("should have changeset"));

        Self {
            operation,
            files,
            summary,
            dependencies,
            changeset,
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct OperationPatches {
    pub patches: Vec<OperationPatch>,
}

impl<'a> Capnp<'a> for OperationPatches {
    type Builder = operation_patches::Builder<'a>;
    type Reader = operation_patches::Reader<'a>;

    fn write_capnp(&self, builder: &mut Self::Builder) {
        let mut patches_builder = builder.reborrow().init_patches(self.patches.len() as u32);
        for (i, patch) in self.patches.iter().enumerate() {
            patch.write_capnp(&mut patches_builder.reborrow().get(i as u32));
        }
    }

    fn read_capnp(reader: Self::Reader) -> Self {
        let patches_reader = reader.get_patches().expect("should have patches");
        let mut patches = Vec::with_capacity(patches_reader.len() as usize);
        for patch_reader in patches_reader.iter() {
            patches.push(OperationPatch::read_capnp(patch_reader));
        }

        Self { patches }
    }
}

#[derive(Debug, Error, PartialEq)]
pub enum PatchError {
    #[error("Changeset Error: {0}")]
    ChangesetError(#[from] ChangesetError),
    #[error("Connection Error: {0}")]
    ConnectionError(#[from] ConnectionError),
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
}

impl From<ZipError> for PatchError {
    fn from(value: ZipError) -> Self {
        PatchError::Zip(value.to_string())
    }
}

#[derive(Debug, Error)]
pub enum CreatePatchError {
    #[error("Operation {0} does not exist.")]
    OperationNotFound(HashId),
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
}

fn serialize_operation_patches(
    operation_patches: &OperationPatches,
) -> Result<Vec<u8>, CreatePatchError> {
    let mut message = ::capnp::message::Builder::new_default();
    let mut root = message.init_root::<operation_patches::Builder>();
    operation_patches.write_capnp(&mut root);

    let mut buffer = Vec::new();
    ::capnp::serialize_packed::write_message(&mut buffer, &message)?;
    Ok(buffer)
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

    let message = ::capnp::serialize_packed::read_message(
        &mut buffer.as_slice(),
        ::capnp::message::ReaderOptions::new(),
    )
    .map_err(|err| PatchError::DeserializationError(err.to_string()))?;

    let root = message
        .get_root::<operation_patches::Reader>()
        .map_err(|err| PatchError::DeserializationError(err.to_string()))?;

    Ok(OperationPatches::read_capnp(root))
}

fn apply_patch(context: &DbContext, patch: &OperationPatch) -> Result<(), PatchError> {
    let workspace = context.workspace();
    let changeset = &patch.changeset;
    let dependencies = &patch.dependencies;
    let mut change_context = context.clone();
    let repo_root = workspace.repo_root().map_err(ConnectionError::from)?;
    let data_db_path = repo_root.join(&changeset.db_path);
    let graph_conn = get_connection(&data_db_path)?;
    change_context.set_graph(graph_conn);

    let conn = change_context.graph().conn();
    let mut session = start_operation(conn);

    conn.execute("BEGIN TRANSACTION", [])?;
    match apply_changeset(conn, &changeset.changes, dependencies) {
        Ok(_) => {
            conn.execute("END TRANSACTION", [])?;
        }
        Err(e) => {
            conn.execute("ROLLBACK TRANSACTION;", [])?;
            return Err(PatchError::ChangesetError(e));
        }
    }

    end_operation(
        &change_context,
        &mut session,
        &OperationInfo {
            files: patch
                .files
                .iter()
                .map(|patch_file| OperationFile {
                    filename: OperationFile::new(patch_file.file.asset_uri.clone()).filename,
                    file_path: patch_file.file.asset_uri.clone(),
                    file_type: patch_file.file.file_type,
                    checksum_override: Some(patch_file.file.checksum),
                })
                .collect::<Vec<_>>(),
            description: "unknown".to_string(),
        },
        &patch.summary.summary,
        None,
    )?;

    Ok(())
}

pub fn create_patch<W>(
    context: &DbContext,
    operations: &[HashId],
    write_stream: &mut W,
) -> Result<(), CreatePatchError>
where
    W: Write + Seek,
{
    let op_conn = context.operations().conn();
    let workspace = context.workspace();
    let mut patches = vec![];
    for hash in operations {
        let operation = Operation::get_by_id(op_conn, hash)
            .ok_or_else(|| CreatePatchError::OperationNotFound(*hash))?;
        println!("Creating patch for Operation {id}", id = operation.hash);

        let files = FileAddition::get_files_for_operation(op_conn, &operation.hash)
            .into_iter()
            .map(|file| PatchFile::from_file_addition(context, file))
            .collect::<Result<Vec<_>, _>>()?;

        patches.push(OperationPatch {
            operation: operation.clone(),
            files,
            summary: OperationSummary::get(
                op_conn,
                "select * from operation_summaries where operation_hash = ?1",
                params![Value::from(operation.hash)],
            )?,
            dependencies: operation.get_changeset_dependencies(workspace),
            changeset: operation.get_changeset(workspace),
        });
    }

    let operation_patches = OperationPatches { patches };
    let manifest_bytes = serialize_operation_patches(&operation_patches)?;

    let manifest_options =
        SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
    let asset_options = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);

    let mut archive = ZipWriter::new(write_stream);
    archive.start_file(MANIFEST_ENTRY, manifest_options)?;
    archive.write_all(&manifest_bytes)?;

    for patch in &operation_patches.patches {
        for file in &patch.files {
            if file.archive_path.is_empty() {
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
    let mut archive = ZipArchive::new(reader).unwrap();
    read_operation_patches(&mut archive).unwrap().patches
}

pub fn apply_patch_archive<R>(context: &DbContext, reader: R) -> Result<(), PatchError>
where
    R: Read + Seek,
{
    let mut archive = ZipArchive::new(reader)?;
    let operation_patches = read_operation_patches(&mut archive)?;

    for patch in &operation_patches.patches {
        for file in &patch.files {
            file.restore_from_archive(context, &mut archive)?;
        }
        apply_patch(context, patch)?;
    }

    Ok(())
}

pub fn apply_patches(context: &DbContext, patches: &[OperationPatch]) -> Result<(), PatchError> {
    for patch in patches {
        apply_patch(context, patch)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{fs, io::Cursor, path::PathBuf};

    use gen_models::{
        block_group::BlockGroup,
        file_types::FileTypes,
        operations::{Branch, FileAddition, OperationState},
        sample::Sample,
    };
    use tempfile::Builder;

    use super::*;
    use crate::{
        imports::fasta::import_fasta,
        operation_management,
        test_helpers::{setup_gen, setup_gen_on_disk},
        track_database,
        updates::vcf::update_with_vcf,
    };

    #[test]
    fn test_creates_patch() {
        let context = setup_gen();
        let vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.vcf");
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let conn = context.graph().conn();
        let operation_conn = context.operations().conn();

        track_database(conn, operation_conn).unwrap();
        let collection = "test".to_string();
        let op_1 = import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
        )
        .unwrap();
        let (op_2, _) = update_with_vcf(
            &context,
            &vcf_path.to_str().unwrap().to_string(),
            &collection,
            "".to_string(),
            None,
            vec![Sample::DEFAULT_NAME.to_string()],
            false,
        )
        .unwrap();
        let mut write_stream = Cursor::new(Vec::new());
        create_patch(&context, &[op_1.hash, op_2.hash], &mut write_stream).unwrap();
        write_stream.set_position(0);
        load_patches(&mut write_stream);
    }

    #[test]
    fn test_cross_db_patches() {
        let source_context = setup_gen_on_disk();
        let vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.vcf");
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let conn = source_context.graph().conn();
        let operation_conn = source_context.operations().conn();

        track_database(conn, operation_conn).unwrap();
        let collection = "test".to_string();
        let op_1 = import_fasta(
            &source_context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
        )
        .unwrap();
        let (op_2, _) = update_with_vcf(
            &source_context,
            &vcf_path.to_str().unwrap().to_string(),
            &collection,
            "".to_string(),
            None,
            vec![Sample::DEFAULT_NAME.to_string()],
            false,
        )
        .unwrap();
        let mut write_stream = Cursor::new(Vec::new());
        create_patch(&source_context, &[op_1.hash, op_2.hash], &mut write_stream).unwrap();
        write_stream.set_position(0);

        let target_context = setup_gen_on_disk();
        let target_conn = target_context.graph().conn();
        let target_operation_conn = target_context.operations().conn();
        track_database(target_conn, target_operation_conn).unwrap();

        apply_patch_archive(&target_context, &mut write_stream).unwrap();
        for bg in BlockGroup::query(conn, "select * from block_groups;", params![]).iter() {
            let seqs = BlockGroup::get_all_sequences(conn, &bg.id, false).unwrap();
            assert!(!seqs.is_empty());
            assert_eq!(
                seqs,
                BlockGroup::get_all_sequences(target_conn, &bg.id, false).unwrap(),
            );
        }
    }

    #[test]
    fn test_cross_branch_patches() {
        let context = setup_gen_on_disk();
        let vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.vcf");
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let conn = context.graph().conn();
        let operation_conn = context.operations().conn();

        track_database(conn, operation_conn).unwrap();

        let collection = "test".to_string();
        let _op_1 = import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
        )
        .unwrap();
        let main_branch = Branch::get_by_name(operation_conn, "main").unwrap();
        let _branch = Branch::get_or_create(operation_conn, "new-branch");
        OperationState::set_branch(operation_conn, "new-branch");
        let (op_2, _) = update_with_vcf(
            &context,
            &vcf_path.to_str().unwrap().to_string(),
            &collection,
            "".to_string(),
            None,
            vec![Sample::DEFAULT_NAME.to_string()],
            false,
        )
        .unwrap();
        let mut write_stream = Cursor::new(Vec::new());
        create_patch(&context, &[op_2.hash], &mut write_stream).unwrap();

        operation_management::checkout(&context, &Some("main".to_string()), None).unwrap();
        write_stream.set_position(0);

        apply_patch_archive(&context, &mut write_stream).unwrap();
        let branch_ops = Branch::get_operations(operation_conn, main_branch.id);
        assert_eq!(branch_ops.len(), 2);

        write_stream.set_position(0);
        let res = apply_patch_archive(&context, &mut write_stream);
        assert_eq!(
            res,
            Err(PatchError::OperationError(OperationError::NoChanges))
        );
        let branch_ops = Branch::get_operations(operation_conn, main_branch.id);
        assert_eq!(branch_ops.len(), 2);
    }

    #[test]
    fn test_capnp_serialization() {
        let context = setup_gen();
        let vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.vcf");
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let conn = context.graph().conn();
        let operation_conn = context.operations().conn();

        track_database(conn, operation_conn).unwrap();
        let collection = "test".to_string();
        let op_1 = import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
        )
        .unwrap();
        let (op_2, _) = update_with_vcf(
            &context,
            &vcf_path.to_str().unwrap().to_string(),
            &collection,
            "".to_string(),
            None,
            vec![Sample::DEFAULT_NAME.to_string()],
            false,
        )
        .unwrap();

        let mut write_stream = Cursor::new(Vec::new());
        create_patch(&context, &[op_1.hash, op_2.hash], &mut write_stream).unwrap();
        write_stream.set_position(0);
        let loaded_patches = load_patches(&mut write_stream);

        assert_eq!(loaded_patches[0].operation, op_1);
        assert_eq!(loaded_patches[1].operation, op_2);
        assert!(!loaded_patches[0].files.is_empty());
        assert!(!loaded_patches[1].files.is_empty());
    }

    #[test]
    fn test_patch_empty_db() {
        let context = setup_gen_on_disk();
        let vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.vcf");
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let conn = context.graph().conn();
        let operation_conn = context.operations().conn();

        track_database(conn, operation_conn).unwrap();

        let collection = "test".to_string();
        let _op_1 = import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
        )
        .unwrap();
        let (op_2, _) = update_with_vcf(
            &context,
            &vcf_path.to_str().unwrap().to_string(),
            &collection,
            "".to_string(),
            None,
            vec![Sample::DEFAULT_NAME.to_string()],
            false,
        )
        .unwrap();
        let mut write_stream = Cursor::new(Vec::new());
        create_patch(&context, &[op_2.hash], &mut write_stream).unwrap();

        let fresh_context = setup_gen_on_disk();
        track_database(
            fresh_context.graph().conn(),
            fresh_context.operations().conn(),
        )
        .unwrap();
        write_stream.set_position(0);
        apply_patch_archive(&fresh_context, &mut write_stream).unwrap();
    }

    #[test]
    fn test_patch_restores_external_assets_into_target_workspace() {
        let source_context = setup_gen_on_disk();
        track_database(
            source_context.graph().conn(),
            source_context.operations().conn(),
        )
        .unwrap();

        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let fixture_vcf_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.vcf");
        let collection = "test".to_string();
        import_fasta(
            &source_context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
        )
        .unwrap();

        let external_file = Builder::new().suffix(".vcf").tempfile().unwrap();
        fs::copy(&fixture_vcf_path, external_file.path()).unwrap();

        let (operation, _) = update_with_vcf(
            &source_context,
            &external_file.path().to_string_lossy().to_string(),
            &collection,
            "".to_string(),
            None,
            vec![Sample::DEFAULT_NAME.to_string()],
            false,
        )
        .unwrap();

        let expected_file = FileAddition::get_files_for_operation(
            source_context.operations().conn(),
            &operation.hash,
        )
        .into_iter()
        .find(|file| file.file_type == FileTypes::VCF)
        .unwrap();

        let mut write_stream = Cursor::new(Vec::new());
        create_patch(&source_context, &[operation.hash], &mut write_stream).unwrap();
        write_stream.set_position(0);
        let patches = load_patches(&mut write_stream);

        let target_context = setup_gen_on_disk();
        track_database(
            target_context.graph().conn(),
            target_context.operations().conn(),
        )
        .unwrap();

        write_stream.set_position(0);
        apply_patch_archive(&target_context, &mut write_stream).unwrap();

        let restored_file = &patches[0]
            .files
            .iter()
            .find(|patch_file| patch_file.file.file_type == FileTypes::VCF)
            .unwrap()
            .file;
        assert_eq!(restored_file.checksum, expected_file.checksum);

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

        let patch_file = PatchFile::from_file_addition(&context, file.clone()).unwrap();

        assert_eq!(patch_file.file, file);
        assert_eq!(patch_file.archive_path, "");
    }
}
