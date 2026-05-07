use core::ops::Range;
use std::str;

use gen_models::{
    block_group::{BlockGroup, NewBlockGroup},
    db::DbContext,
    errors::{BlockGroupError, PathError},
    file_types::FileTypes,
    operations::{OperationFile, OperationInfo},
    sample::Sample,
};
use thiserror::Error;

use crate::graphs::{
    BlockGroupChunk,
    combinatorial_library::{
        CombinatorialLibraryCreationError, CombinatorialLibraryParseError, SequencePart,
        create_library,
    },
    operators::{GraphOperationError, derive_chunks, make_stitch_from_block_groups},
};

#[derive(Error, Debug)]
pub enum UpdateWithLibraryError {
    #[error("Failed to find block group")]
    BlockGroupLookupFailed(String),
    #[error("Failed to create block group")]
    BlockGroupCreationFailed(#[from] BlockGroupError),
    #[error("Failed to create output graph(s)")]
    GraphOperation(GraphOperationError),
    #[error("Failed to read path")]
    Path(#[from] PathError),
    #[error("Failed to parse library files")]
    FileParse(CombinatorialLibraryParseError),
    #[error("Failed to create library")]
    LibraryCreation(CombinatorialLibraryCreationError),
}

impl From<CombinatorialLibraryParseError> for UpdateWithLibraryError {
    fn from(err: CombinatorialLibraryParseError) -> Self {
        UpdateWithLibraryError::FileParse(err)
    }
}

impl From<GraphOperationError> for UpdateWithLibraryError {
    fn from(err: GraphOperationError) -> Self {
        UpdateWithLibraryError::GraphOperation(err)
    }
}

impl From<CombinatorialLibraryCreationError> for UpdateWithLibraryError {
    fn from(err: CombinatorialLibraryCreationError) -> Self {
        UpdateWithLibraryError::LibraryCreation(err)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn update_with_library(
    context: &DbContext,
    collection_name: &str,
    parent_sample_name: &str,
    new_sample_name: &str,
    region_name: &str,
    start_coordinate: i64,
    end_coordinate: i64,
    parts_list: Vec<Vec<SequencePart>>,
    library_file_path: Option<&str>,
    parts_file_path: Option<&str>,
) -> Result<(), UpdateWithLibraryError> {
    let conn = context.graph().conn();
    let mut session = gen_models::session_operations::start_operation(conn);

    let _new_sample = Sample::create(conn, new_sample_name);

    let block_groups = Sample::get_block_groups(conn, collection_name, parent_sample_name);
    let parent_path = BlockGroup::get_current_path(conn, &block_groups[0].id);
    let parent_path_length = parent_path.length(conn)?;

    let mut chunk_ranges = vec![];
    if start_coordinate > 0 {
        chunk_ranges.push(Range {
            start: 0,
            end: start_coordinate,
        });
    }
    chunk_ranges.push(Range {
        start: start_coordinate,
        end: end_coordinate,
    });
    if end_coordinate < parent_path_length {
        chunk_ranges.push(Range {
            start: end_coordinate,
            end: parent_path_length,
        });
    }

    let child_block_group = BlockGroup::create(
        conn,
        NewBlockGroup {
            collection_name,
            sample_name: new_sample_name,
            name: new_sample_name,
            ..Default::default()
        },
    )?;

    let derived_block_group_chunks = derive_chunks(
        context,
        collection_name,
        parent_sample_name,
        new_sample_name,
        region_name,
        None,
        chunk_ranges,
        Some(child_block_group.id),
        false,
    )?;

    let library_block_group_chunk = create_library(
        conn,
        child_block_group.id,
        new_sample_name,
        parts_list,
        false,
    )?;

    let mut block_group_chunks = vec![];
    let mut reference_block_group_chunks = vec![];
    let mut chunk_index = 0;

    if start_coordinate > 0 {
        let start_chunk = derived_block_group_chunks[0].clone();
        reference_block_group_chunks.push(start_chunk.clone());
        let pathless_start_chunk = BlockGroupChunk {
            entry_node_points: start_chunk.entry_node_points.clone(),
            exit_node_points: start_chunk.exit_node_points.clone(),
            path_edges: vec![],
            path_start_point: None,
            path_end_point: None,
        };
        block_group_chunks.push(pathless_start_chunk);

        chunk_index += 1;
    }

    reference_block_group_chunks.push(derived_block_group_chunks[chunk_index].clone());
    block_group_chunks.push(library_block_group_chunk);

    chunk_index += 1;

    if end_coordinate < parent_path_length {
        let end_chunk = derived_block_group_chunks[chunk_index].clone();
        reference_block_group_chunks.push(end_chunk.clone());
        let pathless_end_chunk = BlockGroupChunk {
            entry_node_points: end_chunk.entry_node_points.clone(),
            exit_node_points: end_chunk.exit_node_points.clone(),
            path_edges: vec![],
            path_start_point: None,
            path_end_point: None,
        };
        block_group_chunks.push(pathless_end_chunk);
    }

    let _new_sample = Sample::get_or_create(conn, new_sample_name);

    // Create (re-create) the reference sequence/path out of the derived chunks,
    // in the child block group
    make_stitch_from_block_groups(
        context,
        &reference_block_group_chunks,
        child_block_group.id,
        new_sample_name,
    )?;

    // Stitch the library in between the first and last reference chunks
    make_stitch_from_block_groups(
        context,
        &block_group_chunks,
        child_block_group.id,
        new_sample_name,
    )?;

    let mut files = vec![];
    if let Some(library_file_path) = library_file_path {
        files.push(OperationFile::new(library_file_path.to_string()).set_file_type(FileTypes::CSV));
    }
    if let Some(parts_file_path) = parts_file_path {
        files.push(OperationFile::new(parts_file_path.to_string()).set_file_type(FileTypes::Fasta));
    }

    let summary_str = format!("{region_name} created.\n");
    gen_models::session_operations::end_operation(
        context,
        &mut session,
        &OperationInfo {
            files,
            description: "library_csv_update".to_string(),
        },
        &summary_str,
        None,
    )
    .unwrap();

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, path::PathBuf};

    use anyhow::Result;
    use gen_models::block_group::BlockGroup;

    use super::*;
    use crate::{
        graphs::combinatorial_library::parse_library, imports::fasta::import_fasta,
        test_helpers::setup_gen, track_database,
    };

    #[test]
    fn makes_a_pool() -> Result<()> {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();
        track_database(conn, op_conn).unwrap();

        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
        )
        .unwrap();

        let binding = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/parts.fa");
        let parts_path = binding.to_str().unwrap();

        let binding =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/combinatorial_design.csv");
        let library_path = binding.to_str().unwrap();

        let parts_list = parse_library(parts_path, library_path)?;

        let _ = update_with_library(
            &context,
            "test",
            Sample::DEFAULT_NAME,
            "new sample",
            "m123",
            7,
            20,
            parts_list,
            Some(parts_path),
            Some(library_path),
        );

        let block_groups = Sample::get_block_groups(conn, "test", "new sample");
        let block_group = &block_groups[0];

        let all_sequences = BlockGroup::get_all_sequences(conn, &block_group.id, false);
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec![
                "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
                "ATCGATCAAAAATGATAAGGAACACACAGAGA".to_string(),
                "ATCGATCAAAAATGTTAAGGAACACACAGAGA".to_string(),
                "ATCGATCAAAAATGCTAAGGAACACACAGAGA".to_string(),
                "ATCGATCTAATATGATAAGGAACACACAGAGA".to_string(),
                "ATCGATCTAATATGTTAAGGAACACACAGAGA".to_string(),
                "ATCGATCTAATATGCTAAGGAACACACAGAGA".to_string(),
                "ATCGATCCAACATGATAAGGAACACACAGAGA".to_string(),
                "ATCGATCCAACATGTTAAGGAACACACAGAGA".to_string(),
                "ATCGATCCAACATGCTAAGGAACACACAGAGA".to_string(),
            ])
        );

        Ok(())
    }

    #[test]
    fn one_column_of_parts() -> Result<()> {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();
        track_database(conn, op_conn).unwrap();

        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
        )
        .unwrap();

        let binding = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/parts.fa");
        let parts_path = binding.to_str().unwrap();
        let binding =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/single_column_design.csv");
        let library_path = binding.to_str().unwrap();
        let parts_list = parse_library(parts_path, library_path)?;

        let _ = update_with_library(
            &context,
            "test",
            Sample::DEFAULT_NAME,
            "new sample",
            "m123",
            7,
            20,
            parts_list,
            Some(parts_path),
            Some(library_path),
        );

        let block_groups = Sample::get_block_groups(conn, "test", "new sample");
        let block_group = &block_groups[0];

        let all_sequences = BlockGroup::get_all_sequences(conn, &block_group.id, false);
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec![
                "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
                "ATCGATCAAAAGGAACACACAGAGA".to_string(),
                "ATCGATCTAATGGAACACACAGAGA".to_string(),
                "ATCGATCCAACGGAACACACAGAGA".to_string(),
            ])
        );

        let path = BlockGroup::get_current_path(conn, &block_group.id);
        assert_eq!(
            path.sequence(conn).unwrap(),
            "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string()
        );

        Ok(())
    }

    #[test]
    fn two_columns_of_same_parts() -> Result<()> {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();
        track_database(conn, op_conn).unwrap();

        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
        )
        .unwrap();

        let binding = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/parts.fa");
        let parts_path = binding.to_str().unwrap();
        let binding =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/design_reusing_parts.csv");
        let library_path = binding.to_str().unwrap();
        let parts_list = parse_library(parts_path, library_path)?;

        let _ = update_with_library(
            &context,
            "test",
            Sample::DEFAULT_NAME,
            "new sample",
            "m123",
            7,
            20,
            parts_list,
            Some(parts_path),
            Some(library_path),
        );

        let block_groups = Sample::get_block_groups(conn, "test", "new sample");
        let block_group = &block_groups[0];

        let mut expected_sequences = vec!["ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string()];
        for part1 in ["AAAA", "TAAT", "CAAC"].iter() {
            for part2 in ["AAAA", "TAAT", "CAAC"].iter() {
                let seq = "ATCGATC".to_owned() + part1 + part2 + "GGAACACACAGAGA";
                expected_sequences.push(seq);
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

    #[test]
    fn one_column_of_parts_full_replacement() -> Result<()> {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();
        track_database(conn, op_conn).unwrap();

        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
        )
        .unwrap();

        let binding = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/parts.fa");
        let parts_path = binding.to_str().unwrap();
        let binding =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/single_column_design.csv");
        let library_path = binding.to_str().unwrap();
        let parts_list = parse_library(parts_path, library_path)?;

        let _ = update_with_library(
            &context,
            "test",
            Sample::DEFAULT_NAME,
            "new sample",
            "m123",
            0,
            34, // Full sequence replacement
            parts_list,
            Some(parts_path),
            Some(library_path),
        );

        let block_groups = Sample::get_block_groups(conn, "test", "new sample");
        let block_group = &block_groups[0];

        let all_sequences = BlockGroup::get_all_sequences(conn, &block_group.id, false);
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec![
                "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
                "AAAA".to_string(),
                "TAAT".to_string(),
                "CAAC".to_string(),
            ])
        );

        Ok(())
    }

    #[test]
    fn two_columns_of_same_parts_full_replacement() -> Result<()> {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();
        track_database(conn, op_conn).unwrap();

        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
        )
        .unwrap();

        let binding = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/parts.fa");
        let parts_path = binding.to_str().unwrap();
        let binding =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/design_reusing_parts.csv");
        let library_path = binding.to_str().unwrap();
        let parts_list = parse_library(parts_path, library_path)?;

        let _ = update_with_library(
            &context,
            "test",
            Sample::DEFAULT_NAME,
            "new sample",
            "m123",
            0,
            34, // Full sequence replacement
            parts_list,
            Some(parts_path),
            Some(library_path),
        );

        let block_groups = Sample::get_block_groups(conn, "test", "new sample");
        let block_group = &block_groups[0];

        let mut expected_sequences = vec!["ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string()];
        for part1 in ["AAAA", "TAAT", "CAAC"].iter() {
            for part2 in ["AAAA", "TAAT", "CAAC"].iter() {
                let seq = part1.to_owned().to_owned() + part2;
                expected_sequences.push(seq);
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
