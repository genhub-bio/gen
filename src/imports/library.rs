use std::str;

use anyhow::Result;
use gen_models::{
    block_group::{BlockGroup, NewBlockGroup},
    collection::Collection,
    db::DbContext,
    errors::{BlockGroupError, CollectionError, OperationError},
    file_types::FileTypes,
    operations::{Operation, OperationFile, OperationInfo},
    sample::Sample,
    session_operations,
};
use thiserror::Error;

use crate::graphs::combinatorial_library::{
    CombinatorialLibraryCreationError, CombinatorialLibraryParseError, SequencePart, create_library,
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
    #[error("Block group creation error: {0}")]
    BlockGroupError(#[from] BlockGroupError),
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

pub fn import_library(
    context: &DbContext,
    collection_name: &str,
    sample: &str,
    library_name: &str,
    parts_list: Vec<Vec<SequencePart>>,
    parts_file_path: Option<&str>,
    library_file_path: Option<&str>,
) -> Result<Operation, LibraryImportError> {
    let conn = context.graph().conn();
    let mut session = session_operations::start_operation(conn);
    match Collection::create(conn, collection_name) {
        Ok(_) => {}
        Err(CollectionError::Duplicate(_)) => {}
        Err(e) => {
            return Err(LibraryImportError::ImportFailed(format!(
                "Failed to get or create collection: {e}"
            )));
        }
    }

    match Sample::get_or_create(
        conn,
        gen_models::sample::NewSample {
            name: sample,
            ..Default::default()
        },
    ) {
        Ok(_) => {}
        Err(e) => {
            return Err(LibraryImportError::ImportFailed(format!(
                "Failed to get or create sample: {e}"
            )));
        }
    }
    let new_block_group = BlockGroup::create(
        conn,
        NewBlockGroup {
            collection_name,
            sample_name: sample,
            name: library_name,
            ..Default::default()
        },
    )?;

    let _block_group_boundaries =
        create_library(conn, new_block_group.id, library_name, parts_list, true)?;

    let mut files = vec![];
    if let Some(library_file_path) = library_file_path {
        files.push(OperationFile::new(library_file_path.to_string()).set_file_type(FileTypes::CSV));
    }
    if let Some(parts_file_path) = parts_file_path {
        files.push(OperationFile::new(parts_file_path.to_string()).set_file_type(FileTypes::Fasta));
    }

    let summary_str = format!("{library_name} created.\n");
    let op = session_operations::end_operation(
        context,
        &mut session,
        &OperationInfo {
            files,
            description: "library_csv_import".to_string(),
        },
        &summary_str,
        None,
    )?;

    Ok(op)
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, path::PathBuf};

    use gen_models::block_group::BlockGroup;

    use super::*;
    use crate::{
        graphs::combinatorial_library::parse_library, test_helpers::setup_gen, track_database,
    };

    #[test]
    fn imports_a_library() -> Result<()> {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();
        let collection = "test";

        let binding = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/affix_parts.fa");
        let parts_path = binding.to_str().unwrap();

        let binding = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/affix_layout.csv");
        let library_path = binding.to_str().unwrap();

        let parts_list = parse_library(parts_path, library_path)?;

        let _ = import_library(
            &context,
            collection,
            Sample::DEFAULT_NAME,
            "library graph",
            parts_list,
            Some(parts_path),
            Some(library_path),
        );

        let block_groups = Sample::get_block_groups(conn, collection, Sample::DEFAULT_NAME);
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
            current_path.sequence(conn).unwrap(),
            "TCTAGAGAAAGAGGGGACAAACTAGATGCGTAAAGGAGAAGAACTTTAA"
        );

        Ok(())
    }

    #[test]
    fn one_column_of_parts() -> Result<()> {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();
        let collection = "test";

        let binding = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/parts.fa");
        let parts_path = binding.to_str().unwrap();

        let binding =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/single_column_design.csv");
        let library_path = binding.to_str().unwrap();

        let parts_list = parse_library(parts_path, library_path)?;

        let _ = import_library(
            &context,
            collection,
            Sample::DEFAULT_NAME,
            "m123",
            parts_list,
            Some(parts_path),
            Some(library_path),
        );

        let block_groups = Sample::get_block_groups(conn, collection, Sample::DEFAULT_NAME);
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

        Ok(())
    }

    #[test]
    fn two_columns_of_same_parts() -> Result<()> {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();
        let collection = "test";

        let binding = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/parts.fa");
        let parts_path = binding.to_str().unwrap();

        let binding =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/design_reusing_parts.csv");
        let library_path = binding.to_str().unwrap();

        let parts_list = parse_library(parts_path, library_path)?;

        let _ = import_library(
            &context,
            collection,
            Sample::DEFAULT_NAME,
            "m123",
            parts_list,
            Some(parts_path),
            Some(library_path),
        );

        let block_groups = Sample::get_block_groups(conn, collection, Sample::DEFAULT_NAME);
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

        Ok(())
    }
}
