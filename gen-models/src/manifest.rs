use gen_core::{HashId, traits::Capnp};
use serde::{Deserialize, Serialize};

use crate::{
    annotations::AnnotationFile,
    db::OperationsConnection,
    gen_models_capnp::{manifest, manifest_diff, manifest_operation},
    operations::{FileAddition, Operation, OperationSummary},
    traits::Query,
};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ManifestOperation {
    pub operation: Operation,
    pub file_additions: Vec<FileAddition>,
    pub annotation_file_additions: Vec<FileAddition>,
    pub operation_summary: Option<OperationSummary>,
}

impl<'a> Capnp<'a> for ManifestOperation {
    type Builder = manifest_operation::Builder<'a>;
    type Reader = manifest_operation::Reader<'a>;

    fn write_capnp(&self, builder: &mut Self::Builder) {
        let mut operation_builder = builder.reborrow().init_operation();
        self.operation.write_capnp(&mut operation_builder);

        let mut file_additions_builder = builder
            .reborrow()
            .init_file_additions(self.file_additions.len() as u32);
        for (i, file_addition) in self.file_additions.iter().enumerate() {
            let mut file_addition_builder = file_additions_builder.reborrow().get(i as u32);
            file_addition.write_capnp(&mut file_addition_builder);
        }

        let mut annotation_file_additions_builder = builder
            .reborrow()
            .init_annotation_file_additions(self.annotation_file_additions.len() as u32);
        for (i, file_addition) in self.annotation_file_additions.iter().enumerate() {
            let mut file_addition_builder =
                annotation_file_additions_builder.reborrow().get(i as u32);
            file_addition.write_capnp(&mut file_addition_builder);
        }

        match &self.operation_summary {
            None => {
                builder.reborrow().get_operation_summary().set_none(());
            }
            Some(summary) => {
                let mut summary_builder = builder.reborrow().get_operation_summary().init_some();
                summary.write_capnp(&mut summary_builder);
            }
        }
    }

    fn read_capnp(reader: Self::Reader) -> Self {
        let operation = Operation::read_capnp(reader.get_operation().unwrap());
        let file_additions_reader = reader.get_file_additions().unwrap();
        let mut file_additions = Vec::new();
        for file_addition_reader in file_additions_reader.iter() {
            file_additions.push(FileAddition::read_capnp(file_addition_reader));
        }

        let annotation_file_additions_reader = reader.get_annotation_file_additions().unwrap();
        let mut annotation_file_additions = Vec::new();
        for file_addition_reader in annotation_file_additions_reader.iter() {
            annotation_file_additions.push(FileAddition::read_capnp(file_addition_reader));
        }

        let operation_summary = match reader.get_operation_summary().which().unwrap() {
            manifest_operation::operation_summary::None(()) => None,
            manifest_operation::operation_summary::Some(summary_reader) => {
                Some(OperationSummary::read_capnp(summary_reader.unwrap()))
            }
        };

        ManifestOperation {
            operation,
            file_additions,
            annotation_file_additions,
            operation_summary,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub manifest_version: String,
    pub branch_name: String,
    pub end_hash: Option<HashId>,
    pub operations: Vec<ManifestOperation>,
}

impl<'a> Capnp<'a> for Manifest {
    type Builder = manifest::Builder<'a>;
    type Reader = manifest::Reader<'a>;

    fn write_capnp(&self, builder: &mut Self::Builder) {
        builder.set_manifest_version(&self.manifest_version);
        builder.set_branch_name(&self.branch_name);
        let mut end_hash_builder = builder.reborrow().get_end_hash();
        match &self.end_hash {
            Some(hash) => end_hash_builder.set_some(&hash.0).unwrap(),
            None => end_hash_builder.set_none(()),
        }

        let mut operations_builder = builder
            .reborrow()
            .init_operations(self.operations.len() as u32);
        for (i, operation) in self.operations.iter().enumerate() {
            let mut operation_builder = operations_builder.reborrow().get(i as u32);
            operation.write_capnp(&mut operation_builder);
        }
    }

    fn read_capnp(reader: Self::Reader) -> Self {
        let manifest_version = reader.get_manifest_version().unwrap().to_string().unwrap();
        let branch_name = reader.get_branch_name().unwrap().to_string().unwrap();

        let operations_reader = reader.get_operations().unwrap();
        let mut operations = Vec::new();
        for operation_reader in operations_reader.iter() {
            operations.push(ManifestOperation::read_capnp(operation_reader));
        }

        let end_hash = match reader.get_end_hash().which().unwrap() {
            manifest::end_hash::None(()) => None,
            manifest::end_hash::Some(hash_reader) => {
                let hash_reader = hash_reader.unwrap();
                let slice = hash_reader.as_slice().unwrap();
                Some(slice.try_into().unwrap())
            }
        };

        Manifest {
            manifest_version,
            branch_name,
            end_hash,
            operations,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ManifestDiff {
    pub missing_in_manifest2: Vec<ManifestOperation>,
    pub missing_in_manifest1: Vec<ManifestOperation>,
}

impl<'a> Capnp<'a> for ManifestDiff {
    type Builder = manifest_diff::Builder<'a>;
    type Reader = manifest_diff::Reader<'a>;

    fn write_capnp(&self, builder: &mut Self::Builder) {
        let mut missing_in_manifest2_builder = builder
            .reborrow()
            .init_missing_in_manifest2(self.missing_in_manifest2.len() as u32);
        for (i, operation) in self.missing_in_manifest2.iter().enumerate() {
            let mut operation_builder = missing_in_manifest2_builder.reborrow().get(i as u32);
            operation.write_capnp(&mut operation_builder);
        }

        let mut missing_in_manifest1_builder = builder
            .reborrow()
            .init_missing_in_manifest1(self.missing_in_manifest1.len() as u32);
        for (i, operation) in self.missing_in_manifest1.iter().enumerate() {
            let mut operation_builder = missing_in_manifest1_builder.reborrow().get(i as u32);
            operation.write_capnp(&mut operation_builder);
        }
    }

    fn read_capnp(reader: Self::Reader) -> Self {
        let missing_in_manifest2_reader = reader.get_missing_in_manifest2().unwrap();
        let mut missing_in_manifest2 = Vec::new();
        for operation_reader in missing_in_manifest2_reader.iter() {
            missing_in_manifest2.push(ManifestOperation::read_capnp(operation_reader));
        }

        let missing_in_manifest1_reader = reader.get_missing_in_manifest1().unwrap();
        let mut missing_in_manifest1 = Vec::new();
        for operation_reader in missing_in_manifest1_reader.iter() {
            missing_in_manifest1.push(ManifestOperation::read_capnp(operation_reader));
        }

        ManifestDiff {
            missing_in_manifest2,
            missing_in_manifest1,
        }
    }
}

pub struct ManifestGenerator<'a> {
    conn: &'a OperationsConnection,
}

impl<'a> ManifestGenerator<'a> {
    pub fn new(conn: &'a OperationsConnection) -> Self {
        Self { conn }
    }

    pub fn generate_manifest(
        &self,
        branch_name: &str,
        end_hash: Option<&HashId>,
    ) -> Result<Manifest, ManifestError> {
        let mut manifest_operations = vec![];

        if let Some(target_hash) = end_hash {
            let hashes = Operation::get_upstream(self.conn, target_hash);
            let mut operations_map = std::collections::HashMap::new();
            for op in Operation::query_by_ids(self.conn, &hashes) {
                operations_map.insert(op.hash, op.clone());
            }

            for hash in hashes.iter() {
                if let Some(op) = operations_map.get(hash) {
                    let file_additions = FileAddition::get_files_for_operation(self.conn, &op.hash);
                    let annotation_file_additions =
                        AnnotationFile::get_files_for_operation(self.conn, &op.hash);
                    let operation_summary = OperationSummary::query(
                        self.conn,
                        "select * from operation_summaries where operation_hash = ?1",
                        rusqlite::params![op.hash],
                    )
                    .into_iter()
                    .next();

                    manifest_operations.push(ManifestOperation {
                        operation: op.clone(),
                        file_additions,
                        annotation_file_additions,
                        operation_summary,
                    });
                }
            }
        }

        Ok(Manifest {
            manifest_version: "1.0".to_string(),
            branch_name: branch_name.to_string(),
            end_hash: end_hash.copied(),
            operations: manifest_operations,
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ManifestError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] rusqlite::Error),
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    #[error("Operation not found")]
    OperationNotFound(String),
    #[error("Branch not found")]
    BranchNotFound,
}

pub struct ManifestComparer;

impl ManifestComparer {
    pub fn diff_manifests(
        manifest1: &Manifest,
        manifest2: &Manifest,
    ) -> Result<ManifestDiff, ManifestDiffError> {
        if manifest1.manifest_version != manifest2.manifest_version {
            return Err(ManifestDiffError::IncompatibleVersions);
        }

        let ops1_hashes: std::collections::HashSet<_> = manifest1
            .operations
            .iter()
            .map(|op| &op.operation.hash)
            .collect();
        let ops2_hashes: std::collections::HashSet<_> = manifest2
            .operations
            .iter()
            .map(|op| &op.operation.hash)
            .collect();

        let missing_in_manifest2: Vec<ManifestOperation> = manifest1
            .operations
            .iter()
            .filter(|op| !ops2_hashes.contains(&op.operation.hash))
            .cloned()
            .collect();

        let missing_in_manifest1: Vec<ManifestOperation> = manifest2
            .operations
            .iter()
            .filter(|op| !ops1_hashes.contains(&op.operation.hash))
            .cloned()
            .collect();

        Ok(ManifestDiff {
            missing_in_manifest2,
            missing_in_manifest1,
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ManifestDiffError {
    #[error("Incompatible manifest versions")]
    IncompatibleVersions,
}

#[cfg(test)]
mod tests {
    use std::fs;

    use capnp::message::TypedBuilder;

    use super::*;
    use crate::{
        annotations::AnnotationFile,
        file_types::FileTypes,
        operations::OperationInfo,
        session_operations::{end_operation, start_operation},
        test_helpers::setup_gen,
    };

    #[test]
    fn test_manifest_operation_capnp_serialization() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        let db_uuid = crate::metadata::get_db_uuid(conn);
        crate::files::GenDatabase::create(op_conn, &db_uuid, "test_db", "test_db_path").unwrap();

        let mut session = start_operation(conn);
        crate::sequence::Sequence::new()
            .sequence("ACGT")
            .sequence_type("DNA")
            .save(conn);
        let op_info = OperationInfo {
            files: vec![],
            description: "test op".to_string(),
        };
        let operation = end_operation(&context, &mut session, &op_info, "test", None).unwrap();

        let manifest_operation = ManifestOperation {
            operation: operation.clone(),
            file_additions: vec![FileAddition {
                id: HashId([1u8; 32]),
                file_path: "/path/to/file.fa".to_string(),
                file_type: FileTypes::Fasta,
                checksum: HashId([2u8; 32]),
            }],
            annotation_file_additions: vec![FileAddition {
                id: HashId([3u8; 32]),
                file_path: "/path/to/annotation.gff3".to_string(),
                file_type: FileTypes::Gff3,
                checksum: HashId([4u8; 32]),
            }],
            operation_summary: Some(OperationSummary {
                id: 1,
                operation_hash: operation.hash,
                summary: "Test operation summary".to_string(),
            }),
        };

        let mut message = TypedBuilder::<manifest_operation::Owned>::new_default();
        let mut root = message.init_root();
        manifest_operation.write_capnp(&mut root);

        let deserialized = ManifestOperation::read_capnp(root.into_reader());
        assert_eq!(manifest_operation, deserialized);
    }

    #[test]
    fn test_manifest_capnp_serialization() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        let db_uuid = crate::metadata::get_db_uuid(conn);
        crate::files::GenDatabase::create(op_conn, &db_uuid, "test_db", "test_db_path").unwrap();

        let mut session = start_operation(conn);
        crate::sequence::Sequence::new()
            .sequence("ACGT")
            .sequence_type("DNA")
            .save(conn);
        let op_info = OperationInfo {
            files: vec![],
            description: "test op".to_string(),
        };
        let operation = end_operation(&context, &mut session, &op_info, "test", None).unwrap();

        let manifest = Manifest {
            manifest_version: "1.0".to_string(),
            branch_name: "main".to_string(),
            end_hash: Some(operation.hash),
            operations: vec![ManifestOperation {
                operation,
                file_additions: vec![],
                annotation_file_additions: vec![],
                operation_summary: None,
            }],
        };

        let mut message = TypedBuilder::<manifest::Owned>::new_default();
        let mut root = message.init_root();
        manifest.write_capnp(&mut root);

        let deserialized = Manifest::read_capnp(root.into_reader());
        assert_eq!(manifest, deserialized);
    }

    #[test]
    fn test_manifest_diff_capnp_serialization() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        let db_uuid = crate::metadata::get_db_uuid(conn);
        crate::files::GenDatabase::create(op_conn, &db_uuid, "test_db", "test_db_path").unwrap();

        let mut session = start_operation(conn);
        crate::sequence::Sequence::new()
            .sequence("ACGT")
            .sequence_type("DNA")
            .save(conn);
        let op_info = OperationInfo {
            files: vec![],
            description: "test op".to_string(),
        };
        let operation = end_operation(&context, &mut session, &op_info, "test", None).unwrap();

        let manifest_operation = ManifestOperation {
            operation,
            file_additions: vec![],
            annotation_file_additions: vec![],
            operation_summary: None,
        };

        let manifest_diff = ManifestDiff {
            missing_in_manifest2: vec![manifest_operation.clone()],
            missing_in_manifest1: vec![manifest_operation],
        };

        let mut message = TypedBuilder::<manifest_diff::Owned>::new_default();
        let mut root = message.init_root();
        manifest_diff.write_capnp(&mut root);

        let deserialized = ManifestDiff::read_capnp(root.into_reader());
        assert_eq!(manifest_diff, deserialized);
    }

    #[test]
    fn test_manifest_generator() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        let db_uuid = crate::metadata::get_db_uuid(conn);
        crate::files::GenDatabase::create(op_conn, &db_uuid, "test_db", "test_db_path").unwrap();

        let mut session = start_operation(conn);
        crate::sequence::Sequence::new()
            .sequence("ACGT")
            .sequence_type("DNA")
            .save(conn);
        let op_info = OperationInfo {
            files: vec![],
            description: "first op".to_string(),
        };
        let op1 = end_operation(&context, &mut session, &op_info, "test", None).unwrap();

        let mut session = start_operation(conn);
        crate::sequence::Sequence::new()
            .sequence("TGCA")
            .sequence_type("DNA")
            .save(conn);
        let op_info = OperationInfo {
            files: vec![],
            description: "second op".to_string(),
        };
        let op2 = end_operation(&context, &mut session, &op_info, "test", None).unwrap();

        let generator = ManifestGenerator::new(op_conn);
        let manifest = generator
            .generate_manifest("main", Some(&op2.hash))
            .unwrap();

        assert_eq!(manifest.branch_name, "main");
        assert_eq!(manifest.operations.len(), 2);
        assert_eq!(manifest.operations[0].operation.hash, op1.hash);
        assert_eq!(manifest.operations[1].operation.hash, op2.hash);

        let manifest = generator
            .generate_manifest("main", Some(&op1.hash))
            .unwrap();
        assert_eq!(manifest.operations.len(), 1);
        assert_eq!(manifest.operations[0].operation.hash, op1.hash);
    }

    #[test]
    fn test_manifest_generator_includes_annotation_files() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        let db_uuid = crate::metadata::get_db_uuid(conn);
        crate::files::GenDatabase::create(op_conn, &db_uuid, "test_db", "test_db_path").unwrap();

        let mut session = start_operation(conn);
        crate::sequence::Sequence::new()
            .sequence("ACGT")
            .sequence_type("DNA")
            .save(conn);
        let op_info = OperationInfo {
            files: vec![],
            description: "annotation op".to_string(),
        };
        let operation = end_operation(&context, &mut session, &op_info, "test", None).unwrap();

        let repo_root = context.workspace().repo_root().unwrap();
        let annotation_path = repo_root.join("fixtures").join("manifest_annotation.gff3");
        fs::create_dir_all(annotation_path.parent().unwrap()).unwrap();
        fs::write(&annotation_path, "##gff-version 3\n").unwrap();

        let file_addition = AnnotationFile::add_to_operation(
            context.workspace(),
            op_conn,
            &operation.hash,
            "fixtures/manifest_annotation.gff3",
            FileTypes::Gff3,
            None,
        )
        .unwrap();

        let generator = ManifestGenerator::new(op_conn);
        let manifest = generator
            .generate_manifest("main", Some(&operation.hash))
            .unwrap();

        assert_eq!(manifest.operations.len(), 1);
        assert_eq!(
            manifest.operations[0].annotation_file_additions,
            vec![file_addition]
        );
    }

    #[test]
    fn test_manifest_comparer() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        let db_uuid = crate::metadata::get_db_uuid(conn);
        crate::files::GenDatabase::create(op_conn, &db_uuid, "test_db", "test_db_path").unwrap();

        let mut session = start_operation(conn);
        crate::sequence::Sequence::new()
            .sequence("ACGT")
            .sequence_type("DNA")
            .save(conn);
        let op_info = OperationInfo {
            files: vec![],
            description: "first op".to_string(),
        };
        let op1 = end_operation(&context, &mut session, &op_info, "test", None).unwrap();

        let mut session = start_operation(conn);
        crate::sequence::Sequence::new()
            .sequence("TTTT")
            .sequence_type("DNA")
            .save(conn);
        let op_info = OperationInfo {
            files: vec![],
            description: "second op".to_string(),
        };
        let op2 = end_operation(&context, &mut session, &op_info, "test 2", None).unwrap();

        let mut session = start_operation(conn);
        crate::sequence::Sequence::new()
            .sequence("AAAA")
            .sequence_type("DNA")
            .save(conn);
        let op_info = OperationInfo {
            files: vec![],
            description: "third op".to_string(),
        };
        let op3 = end_operation(&context, &mut session, &op_info, "test 3", None).unwrap();

        let manifest1 = Manifest {
            manifest_version: "1.0".to_string(),
            branch_name: "main".to_string(),
            end_hash: Some(op2.hash),
            operations: vec![
                ManifestOperation {
                    operation: op1.clone(),
                    file_additions: vec![],
                    annotation_file_additions: vec![],
                    operation_summary: None,
                },
                ManifestOperation {
                    operation: op2.clone(),
                    file_additions: vec![],
                    annotation_file_additions: vec![],
                    operation_summary: None,
                },
            ],
        };

        let manifest2 = Manifest {
            manifest_version: "1.0".to_string(),
            branch_name: "main".to_string(),
            end_hash: Some(op3.hash),
            operations: vec![
                ManifestOperation {
                    operation: op2.clone(),
                    file_additions: vec![],
                    annotation_file_additions: vec![],
                    operation_summary: None,
                },
                ManifestOperation {
                    operation: op3.clone(),
                    file_additions: vec![],
                    annotation_file_additions: vec![],
                    operation_summary: None,
                },
            ],
        };

        let diff = ManifestComparer::diff_manifests(&manifest1, &manifest2).unwrap();

        assert_eq!(diff.missing_in_manifest2.len(), 1);
        assert_eq!(diff.missing_in_manifest2[0].operation.hash, op1.hash);

        assert_eq!(diff.missing_in_manifest1.len(), 1);
        assert_eq!(diff.missing_in_manifest1[0].operation.hash, op3.hash);
    }

    #[test]
    fn test_manifest_generator_operation_not_found() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        let db_uuid = crate::metadata::get_db_uuid(conn);
        crate::files::GenDatabase::create(op_conn, &db_uuid, "test_db", "test_db_path").unwrap();

        let mut session = start_operation(conn);
        crate::sequence::Sequence::new()
            .sequence("ACGT")
            .sequence_type("DNA")
            .save(conn);
        let op_info = OperationInfo {
            files: vec![],
            description: "first op".to_string(),
        };
        end_operation(&context, &mut session, &op_info, "test", None).unwrap();

        let generator = ManifestGenerator::new(op_conn);
        let manifest = generator
            .generate_manifest("main", Some(&HashId::convert_str("non_existent_op")))
            .unwrap();
        assert!(manifest.operations.is_empty());
    }
}
