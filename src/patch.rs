use std::io::{Read, Write};

use flate2::{Compression, read::GzDecoder, write::GzEncoder};
use gen_core::{HashId, errors::ConnectionError, traits::Capnp};
use gen_models::{
    changesets::{DatabaseChangeset, apply_changeset},
    errors::{ChangesetError, OperationError},
    operations::{FileAddition, Operation, OperationFile, OperationInfo, OperationSummary},
    session_operations::{DependencyModels, end_operation, start_operation},
    traits::Query,
};
use rusqlite::{Connection, Error as SQLError, params, types::Value};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    gen_schema_capnp::{operation_patch, operation_patches},
    get_connection,
};

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct OperationPatch {
    pub operation: Operation,
    files: Vec<FileAddition>,
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
            files.push(FileAddition::read_capnp(file_reader));
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
    #[error("SQL Error: {0}")]
    SQLError(String),
    #[error("SQLite Error: {0}")]
    SqliteError(#[from] SQLError),
    #[error("Deserialization Error: {0}")]
    DeserializationError(String),
    #[error("Operation Error: {0}")]
    OperationError(#[from] OperationError),
}

pub fn create_patch<W>(op_conn: &Connection, operations: &[HashId], write_stream: &mut W)
where
    W: Write,
{
    let mut patches = vec![];
    for operation in operations.iter() {
        let operation = Operation::get_by_id(op_conn, operation)
            .unwrap_or_else(|| panic!("Hash {operation} does not exist."));
        println!("Creating patch for Operation {id}", id = operation.hash);
        patches.push(OperationPatch {
            operation: operation.clone(),
            files: FileAddition::get_files_for_operation(op_conn, &operation.hash),
            summary: OperationSummary::get(
                op_conn,
                "select * from operation_summaries where operation_hash = ?1",
                params![Value::from(operation.hash)],
            )
            .unwrap(),
            dependencies: operation.get_changeset_dependencies(),
            changeset: operation.get_changeset(),
        })
    }

    let operation_patches = OperationPatches { patches };

    // Serialize using Cap'n Proto
    let mut message = ::capnp::message::Builder::new_default();
    let mut root = message.init_root::<operation_patches::Builder>();
    operation_patches.write_capnp(&mut root);

    // Write the packed message to a buffer
    let mut capnp_buffer = Vec::new();
    ::capnp::serialize_packed::write_message(&mut capnp_buffer, &message).unwrap();

    // Compress the Cap'n Proto data
    let mut e = GzEncoder::new(Vec::new(), Compression::default());
    e.write_all(&capnp_buffer).unwrap();
    let compressed = e.finish().unwrap();
    write_stream.write_all(&compressed).unwrap();
}

pub fn load_patches<R>(reader: R) -> Vec<OperationPatch>
where
    R: Read,
{
    let mut d = GzDecoder::new(reader);
    let mut capnp_buffer = Vec::new();
    d.read_to_end(&mut capnp_buffer).unwrap();

    // Deserialize from Cap'n Proto
    let message = ::capnp::serialize_packed::read_message(
        &mut capnp_buffer.as_slice(),
        ::capnp::message::ReaderOptions::new(),
    )
    .unwrap();

    let root = message.get_root::<operation_patches::Reader>().unwrap();
    let operation_patches = OperationPatches::read_capnp(root);

    operation_patches.patches
}

pub fn apply_patches(
    connection: Option<&Connection>,
    op_conn: &Connection,
    patches: &[OperationPatch],
) -> Result<(), PatchError> {
    for patch in patches.iter() {
        let changeset = &patch.changeset;
        let c_bind;
        let conn = if let Some(c) = connection {
            c
        } else {
            c_bind = get_connection(&changeset.db_path)?;
            &c_bind
        };
        let dependencies = &patch.dependencies;
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
            conn,
            op_conn,
            &mut session,
            &OperationInfo {
                files: patch
                    .files
                    .iter()
                    .map(|fa| OperationFile {
                        file_path: fa.file_path.clone(),
                        file_type: fa.file_type,
                    })
                    .collect::<Vec<_>>(),
                description: "unknown".to_string(),
            },
            &patch.summary.summary,
            None,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use gen_models::{
        block_group::BlockGroup,
        operations::{Branch, OperationState},
    };

    use super::*;
    use crate::{
        imports::fasta::import_fasta,
        operation_management,
        test_helpers::{get_connection, get_operation_connection, setup_gen_dir},
        track_database,
        updates::vcf::update_with_vcf,
    };

    #[test]
    fn test_creates_patch() {
        setup_gen_dir();
        let vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.vcf");
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let conn = &mut get_connection(None).unwrap();
        let operation_conn = &get_operation_connection(None).unwrap();

        track_database(conn, operation_conn).unwrap();
        let collection = "test".to_string();
        let op_1 = import_fasta(
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            None,
            false,
            conn,
            operation_conn,
        )
        .unwrap();
        let op_2 = update_with_vcf(
            &vcf_path.to_str().unwrap().to_string(),
            &collection,
            "".to_string(),
            "".to_string(),
            conn,
            operation_conn,
            None,
        )
        .unwrap();
        let mut write_stream: Vec<u8> = Vec::new();
        create_patch(operation_conn, &[op_1.hash, op_2.hash], &mut write_stream);
        load_patches(&write_stream[..]);
    }

    #[test]
    fn test_cross_db_patches() {
        setup_gen_dir();
        let vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.vcf");
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let conn = &mut get_connection(None).unwrap();
        let conn2 = &mut get_connection(None).unwrap();
        let operation_conn = &get_operation_connection(None).unwrap();

        track_database(conn, operation_conn).unwrap();
        track_database(conn2, operation_conn).unwrap();
        let collection = "test".to_string();
        let op_1 = import_fasta(
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            None,
            false,
            conn,
            operation_conn,
        )
        .unwrap();
        let op_2 = update_with_vcf(
            &vcf_path.to_str().unwrap().to_string(),
            &collection,
            "".to_string(),
            "".to_string(),
            conn,
            operation_conn,
            None,
        )
        .unwrap();
        let mut write_stream: Vec<u8> = Vec::new();
        create_patch(operation_conn, &[op_1.hash, op_2.hash], &mut write_stream);
        let patches = load_patches(&write_stream[..]);
        apply_patches(Some(conn2), operation_conn, &patches).unwrap();
        for bg in BlockGroup::query(conn, "select * from block_groups;", params![]).iter() {
            let seqs = BlockGroup::get_all_sequences(conn, &bg.id, false);
            assert!(!seqs.is_empty());
            assert_eq!(seqs, BlockGroup::get_all_sequences(conn2, &bg.id, false),)
        }
    }

    #[test]
    fn test_cross_branch_patches() {
        setup_gen_dir();
        let vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.vcf");
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let conn = &mut get_connection(None).unwrap();
        let operation_conn = &get_operation_connection(None).unwrap();

        track_database(conn, operation_conn).unwrap();

        let collection = "test".to_string();
        let _op_1 = import_fasta(
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            None,
            false,
            conn,
            operation_conn,
        )
        .unwrap();
        let main_branch = Branch::get_by_name(operation_conn, "main").unwrap();
        let _branch = Branch::get_or_create(operation_conn, "new-branch");
        OperationState::set_branch(operation_conn, "new-branch");
        let op_2 = update_with_vcf(
            &vcf_path.to_str().unwrap().to_string(),
            &collection,
            "".to_string(),
            "".to_string(),
            conn,
            operation_conn,
            None,
        )
        .unwrap();
        let mut write_stream: Vec<u8> = Vec::new();
        create_patch(operation_conn, &[op_2.hash], &mut write_stream);

        operation_management::checkout(Some(conn), operation_conn, &Some("main".to_string()), None)
            .unwrap();
        let patches = load_patches(&write_stream[..]);
        apply_patches(Some(conn), operation_conn, &patches).unwrap();
        let branch_ops = Branch::get_operations(operation_conn, main_branch.id);
        assert_eq!(branch_ops.len(), 2);
        // ensure if we apply the operation again it'll be a no-op
        let res = apply_patches(Some(conn), operation_conn, &patches);
        assert_eq!(
            res,
            Err(PatchError::OperationError(OperationError::NoChanges))
        );
        let branch_ops = Branch::get_operations(operation_conn, main_branch.id);
        assert_eq!(branch_ops.len(), 2);
    }

    #[test]
    fn test_capnp_serialization() {
        setup_gen_dir();
        let vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.vcf");
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let conn = &mut get_connection(None).unwrap();
        let operation_conn = &get_operation_connection(None).unwrap();

        track_database(conn, operation_conn).unwrap();
        let collection = "test".to_string();
        let op_1 = import_fasta(
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            None,
            false,
            conn,
            operation_conn,
        )
        .unwrap();
        let op_2 = update_with_vcf(
            &vcf_path.to_str().unwrap().to_string(),
            &collection,
            "".to_string(),
            "".to_string(),
            conn,
            operation_conn,
            None,
        )
        .unwrap();

        // Test Cap'n Proto serialization/deserialization
        let mut write_stream: Vec<u8> = Vec::new();
        create_patch(operation_conn, &[op_1.hash, op_2.hash], &mut write_stream);
        let loaded_patches = load_patches(&write_stream[..]);

        // Verify we got the same patches back
        assert_eq!(loaded_patches[0].operation, op_1);
        assert_eq!(loaded_patches[1].operation, op_2);
    }

    #[test]
    fn test_patch_empty_db() {
        setup_gen_dir();
        let vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.vcf");
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let conn = &get_connection(None).unwrap();
        let operation_conn = &get_operation_connection(None).unwrap();

        track_database(conn, operation_conn).unwrap();

        let collection = "test".to_string();
        let _op_1 = import_fasta(
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            None,
            false,
            conn,
            operation_conn,
        )
        .unwrap();
        let op_2 = update_with_vcf(
            &vcf_path.to_str().unwrap().to_string(),
            &collection,
            "".to_string(),
            "".to_string(),
            conn,
            operation_conn,
            None,
        )
        .unwrap();
        let mut write_stream: Vec<u8> = Vec::new();
        create_patch(operation_conn, &[op_2.hash], &mut write_stream);

        let patches = load_patches(&write_stream[..]);
        let fresh_db = &get_connection(None).unwrap();
        track_database(fresh_db, operation_conn).unwrap();
        apply_patches(Some(fresh_db), operation_conn, &patches).unwrap();
    }
}
