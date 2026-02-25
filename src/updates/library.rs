use core::ops::Range;
use std::str;

use gen_models::{
    block_group::BlockGroup,
    db::DbContext,
    file_types::FileTypes,
    operations::{OperationFile, OperationInfo},
    sample::Sample,
};
use thiserror::Error;

use crate::graphs::{
    BlockGroupChunk,
    combinatorial_library::{
        CombinatorialLibraryCreationError, CombinatorialLibraryParseError, create_library,
        parse_library,
    },
    operators::{GraphOperationError, derive_chunks, make_stitch_from_block_groups},
};

#[derive(Error, Debug)]
pub enum UpdateWithLibraryError {
    #[error("Failed to find block group")]
    BlockGroupLookupFailed(String),
    #[error("Failed to create output graph(s)")]
    GraphOperation(GraphOperationError),
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
    parent_sample_name: Option<&str>,
    new_sample_name: &str,
    region_name: &str,
    start_coordinate: i64,
    end_coordinate: i64,
    parts_file_path: &str,
    library_file_path: &str,
) -> Result<(), UpdateWithLibraryError> {
    let conn = context.graph().conn();
    let mut session = gen_models::session_operations::start_operation(conn);

    let _new_sample = Sample::create(conn, new_sample_name);

    let block_groups = Sample::get_block_groups(conn, collection_name, parent_sample_name);
    let parent_path = BlockGroup::get_current_path(conn, &block_groups[0].id);

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
    if end_coordinate < parent_path.length(conn) {
        chunk_ranges.push(Range {
            start: end_coordinate,
            end: parent_path.length(conn),
        });
    }

    let child_block_group = BlockGroup::create(
        conn,
        collection_name,
        Some(new_sample_name),
        new_sample_name,
    );

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

    let parts_list = parse_library(parts_file_path, library_file_path)?;

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

    if end_coordinate < parent_path.length(conn) {
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

    let summary_str = format!("{region_name} created.\n");
    gen_models::session_operations::end_operation(
        context,
        &mut session,
        &OperationInfo {
            files: vec![OperationFile {
                file_path: library_file_path.to_string(),
                file_type: FileTypes::CSV,
            }],
            description: "library_csv_update".to_string(),
        },
        &summary_str,
        None,
    )
    .unwrap();

    println!("Updated with library file: {library_file_path}");

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, path::PathBuf};

    use gen_models::block_group::BlockGroup;

    use super::*;
    use crate::{imports::fasta::import_fasta, test_helpers::setup_gen, track_database};

    #[test]
    fn makes_a_pool() {
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
            None,
            false,
        )
        .unwrap();

        let parts_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/parts.fa");
        let library_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/combinatorial_design.csv");

        let _ = update_with_library(
            &context,
            "test",
            None,
            "new sample",
            "m123",
            7,
            20,
            parts_path.to_str().unwrap(),
            library_path.to_str().unwrap(),
        );

        let block_groups = Sample::get_block_groups(conn, "test", Some("new sample"));
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
    }

    #[test]
    fn one_column_of_parts() {
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
            None,
            false,
        )
        .unwrap();

        let parts_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/parts.fa");
        let library_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/single_column_design.csv");

        let _ = update_with_library(
            &context,
            "test",
            None,
            "new sample",
            "m123",
            7,
            20,
            parts_path.to_str().unwrap(),
            library_path.to_str().unwrap(),
        );

        let block_groups = Sample::get_block_groups(conn, "test", Some("new sample"));
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
            path.sequence(conn),
            "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string()
        );
    }

    #[test]
    fn two_columns_of_same_parts() {
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
            None,
            false,
        )
        .unwrap();

        let parts_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/parts.fa");
        let library_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/design_reusing_parts.csv");

        let _ = update_with_library(
            &context,
            "test",
            None,
            "new sample",
            "m123",
            7,
            20,
            parts_path.to_str().unwrap(),
            library_path.to_str().unwrap(),
        );

        let block_groups = Sample::get_block_groups(conn, "test", Some("new sample"));
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
    }

    #[test]
    fn one_column_of_parts_full_replacement() {
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
            None,
            false,
        )
        .unwrap();

        let parts_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/parts.fa");
        let library_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/single_column_design.csv");

        let _ = update_with_library(
            &context,
            "test",
            None,
            "new sample",
            "m123",
            0,
            34, // Full sequence replacement
            parts_path.to_str().unwrap(),
            library_path.to_str().unwrap(),
        );

        let block_groups = Sample::get_block_groups(conn, "test", Some("new sample"));
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
    }

    #[test]
    fn two_columns_of_same_parts_full_replacement() {
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
            None,
            false,
        )
        .unwrap();

        let parts_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/parts.fa");
        let library_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/design_reusing_parts.csv");

        let _ = update_with_library(
            &context,
            "test",
            None,
            "new sample",
            "m123",
            0,
            34, // Full sequence replacement
            parts_path.to_str().unwrap(),
            library_path.to_str().unwrap(),
        );

        let block_groups = Sample::get_block_groups(conn, "test", Some("new sample"));
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
    }
}
