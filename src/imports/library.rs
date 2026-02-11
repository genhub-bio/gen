use std::str;

use anyhow::Result;
use gen_models::{
    block_group::BlockGroup,
    collection::Collection,
    db::DbContext,
    errors::OperationError,
    file_types::FileTypes,
    operations::{Operation, OperationFile, OperationInfo},
    sample::Sample,
    session_operations,
};
use thiserror::Error;

use crate::graphs::combinatorial_library::{
    CombinatorialLibraryCreationError, CombinatorialLibraryParseError, create_library,
    parse_library,
};

#[derive(Error, Debug)]
pub enum LibraryImportError {
    #[error("No changes were made to the library")]
    NoChanges,
    #[error("Failed to import library")]
    ImportFailed(String),
    #[error("Operation Error: {0}")]
    OperationError(#[from] OperationError),
    #[error("Failed to parse library files")]
    FileParse(CombinatorialLibraryParseError),
    #[error("Failed to create library")]
    LibraryCreation(CombinatorialLibraryCreationError),
}

impl From<CombinatorialLibraryParseError> for LibraryImportError {
    fn from(err: CombinatorialLibraryParseError) -> Self {
        LibraryImportError::FileParse(err)
    }
}

impl From<CombinatorialLibraryCreationError> for LibraryImportError {
    fn from(err: CombinatorialLibraryCreationError) -> Self {
        LibraryImportError::LibraryCreation(err)
    }
}

pub fn import_library<'a>(
    context: &DbContext,
    collection_name: &str,
    sample: impl Into<Option<&'a str>>,
    parts_file_path: &str,
    library_file_path: &str,
    library_name: &str,
) -> Result<Operation, LibraryImportError> {
    let conn = context.graph().conn();
    let mut session = session_operations::start_operation(conn);

    if !Collection::exists(conn, collection_name) {
        Collection::create(conn, collection_name);
    }

    let sample = sample.into();
    if let Some(sample_name) = sample {
        Sample::get_or_create(conn, sample_name);
    }
    let new_block_group = BlockGroup::create(conn, collection_name, sample, library_name);

    let parts_list = parse_library(parts_file_path, library_file_path)?;
    let path_changes_count = create_library(conn, &new_block_group.id, library_name, parts_list)?;

    let summary_str = format!("{library_name}: {path_changes_count} changes.\n");
    let op = session_operations::end_operation(
        context,
        &mut session,
        &OperationInfo {
            files: vec![
                OperationFile {
                    file_path: library_file_path.to_string(),
                    file_type: FileTypes::CSV,
                },
                OperationFile {
                    file_path: parts_file_path.to_string(),
                    file_type: FileTypes::Fasta,
                },
            ],
            description: "library_csv_import".to_string(),
        },
        &summary_str,
        None,
    )?;

    println!("Imported library file {library_file_path} and parts file {parts_file_path}");

    Ok(op)
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, path::PathBuf};

    use gen_models::block_group::BlockGroup;

    use super::*;
    use crate::{test_helpers::setup_gen, track_database};

    #[test]
    fn imports_a_library() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();
        let collection = "test";

        let parts_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/affix_parts.fa");
        let library_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/affix_layout.csv");

        let _ = import_library(
            &context,
            collection,
            None,
            parts_path.to_str().unwrap(),
            library_path.to_str().unwrap(),
            "library graph",
        );

        let block_groups = Sample::get_block_groups(conn, collection, None);
        let block_group = &block_groups[0];

        let mut expected_sequences = HashSet::new();
        for part1 in &[
            "TCTAGAGAAAGAGGGGACAAACTAG",
            "TCTAGAGAAAGACAGGACCCACTAG",
            "TCTAGAGAAAGATCCGATGTACTAG",
            "TCTAGAGAAAGATTAGACAAACTAG",
            "TCTAGAGAAAGAAGGGACAGACTAG",
            "TCTAGAGAAAGACATGACGTACTAG",
            "TCTAGAGAAAGATAGGAGACACTAG",
            "TCTAGAGAAAGAAGAGACTCACTAG",
        ] {
            for part2 in &["ATGCGTAAAGGAGAAGAACTTTAA", "ATGAGTAAGGGTGAAGAGCTGTAA"] {
                expected_sequences.insert(format!("{part1}{part2}"));
            }
        }

        let actual_sequences = BlockGroup::get_all_sequences(conn, &block_group.id, false);
        assert_eq!(actual_sequences, expected_sequences);

        let current_path = BlockGroup::get_current_path(conn, &block_group.id);
        assert_eq!(
            current_path.sequence(conn),
            "TCTAGAGAAAGAGGGGACAAACTAGATGCGTAAAGGAGAAGAACTTTAA"
        );
    }

    #[test]
    fn one_column_of_parts() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();
        let collection = "test";

        let parts_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/parts.fa");
        let library_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/single_column_design.csv");

        let _ = import_library(
            &context,
            collection,
            None,
            parts_path.to_str().unwrap(),
            library_path.to_str().unwrap(),
            "m123",
        );

        let block_groups = Sample::get_block_groups(conn, collection, None);
        let block_group = &block_groups[0];

        let all_sequences = BlockGroup::get_all_sequences(conn, &block_group.id, false);
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec![
                "AAAA".to_string(),
                "TAAT".to_string(),
                "CAAC".to_string(),
            ])
        );
    }

    #[test]
    fn two_columns_of_same_parts() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();
        let collection = "test";

        let parts_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/parts.fa");
        let library_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/design_reusing_parts.csv");

        let _ = import_library(
            &context,
            collection,
            None,
            parts_path.to_str().unwrap(),
            library_path.to_str().unwrap(),
            "m123",
        );

        let block_groups = Sample::get_block_groups(conn, collection, None);
        let block_group = &block_groups[0];

        let mut expected_sequences = vec![];
        for part1 in ["AAAA", "TAAT", "CAAC"].iter() {
            for part2 in ["AAAA", "TAAT", "CAAC"].iter() {
                expected_sequences.push(part1.to_string() + part2);
            }
        }
        let all_sequences = BlockGroup::get_all_sequences(conn, &block_group.id, false);
        assert_eq!(
            all_sequences,
            expected_sequences
                .into_iter()
                .map(|x| x.to_string())
                .collect()
        );
    }
}
