use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::BufReader,
    str,
};

use anyhow::Result;
use gen_core::{HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand};
use gen_models::{
    block_group::BlockGroup,
    block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
    collection::Collection,
    edge::{Edge, EdgeData},
    errors::OperationError,
    file_types::FileTypes,
    node::Node,
    operations::{Operation, OperationFile, OperationInfo},
    path::Path,
    sample::Sample,
    sequence::Sequence,
    session_operations,
    traits::Query,
};
use itertools::Itertools;
use noodles::fasta;
use rusqlite::Connection;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum LibraryImportError {
    #[error("No changes were made to the library")]
    NoChanges,
    #[error("Failed to import library")]
    ImportFailed(String),
    #[error("Operation Error: {0}")]
    OperationError(#[from] OperationError),
}

pub fn import_library<'a>(
    conn: &Connection,
    operation_conn: &Connection,
    collection_name: &str,
    sample: impl Into<Option<&'a str>>,
    parts_file_path: &str,
    library_file_path: &str,
    library_name: &str,
) -> Result<Operation, LibraryImportError> {
    let mut session = session_operations::start_operation(conn);

    if !Collection::exists(conn, collection_name) {
        Collection::create(conn, collection_name);
    }

    let sample = sample.into();
    if let Some(sample_name) = sample {
        Sample::get_or_create(conn, sample_name);
    }
    let new_block_group = BlockGroup::create(conn, collection_name, sample, library_name);

    let mut parts_reader = fasta::io::reader::Builder
        .build_from_path(parts_file_path)
        .map_err(|e| LibraryImportError::ImportFailed(e.to_string()))?;

    let mut sequence_hashes_by_name = HashMap::new();
    let mut sequence_lengths_by_hash = HashMap::new();
    for result in parts_reader.records() {
        let record = result.map_err(|e| LibraryImportError::ImportFailed(e.to_string()))?;
        let sequence = str::from_utf8(record.sequence().as_ref())
            .map_err(|e| LibraryImportError::ImportFailed(e.to_string()))?;
        let name = String::from_utf8(record.name().to_vec()).unwrap();
        let seq = Sequence::new()
            .sequence_type("DNA")
            .sequence(sequence)
            .save(conn);

        if sequence_hashes_by_name.contains_key(&name) {
            return Err(LibraryImportError::ImportFailed(format!(
                "Duplicate sequence name: {name}"
            )));
        }
        sequence_hashes_by_name.insert(name, seq.hash);
        sequence_lengths_by_hash.insert(seq.hash, seq.length);
    }

    let library_file = File::open(library_file_path).map_err(|e| {
        LibraryImportError::ImportFailed(format!(
            "Failed to open library file {library_file_path}: {}",
            e
        ))
    })?;
    let library_reader = BufReader::new(library_file);

    let mut parts_by_index = HashMap::new();
    let mut library_csv_reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(library_reader);
    let mut max_index = 0;
    let mut sequence_lengths_by_node_id = HashMap::new();
    for result in library_csv_reader.records() {
        let record = result.map_err(|e| LibraryImportError::ImportFailed(e.to_string()))?;
        for (index, part) in record.iter().enumerate() {
            if !part.is_empty() {
                let part_hash = sequence_hashes_by_name.get(part).ok_or_else(|| {
                    LibraryImportError::ImportFailed(format!("Part {part} missing."))
                })?;
                let seq_length = sequence_lengths_by_hash.get(part_hash).ok_or_else(|| {
                    LibraryImportError::ImportFailed(format!("Part hash {part_hash} missing."))
                })?;
                let part_node_id = Node::create(
                    conn,
                    part_hash,
                    &HashId::convert_str(&format!(
                        "{library_name}:{part}:{ref_start}-{ref_end}->{sequence_hash}-column-{index}",
                        ref_start = 0,
                        ref_end = seq_length,
                        sequence_hash = part_hash
                    )),
                );
                sequence_lengths_by_node_id.insert(part_node_id, *seq_length);

                parts_by_index
                    .entry(index)
                    .or_insert(vec![])
                    .push(part_node_id);
                if index >= max_index {
                    max_index = index + 1;
                }
            }
        }
    }

    let mut parts_list = vec![];
    for index in 0..max_index {
        parts_list.push(
            parts_by_index.get(&index).ok_or_else(|| {
                LibraryImportError::ImportFailed(format!("Missing index {index}."))
            })?,
        );
    }

    let mut new_edges = HashSet::new();
    let start_parts = parts_list
        .first()
        .ok_or_else(|| LibraryImportError::ImportFailed("No parts found.".to_string()))?;
    for start_part in *start_parts {
        let edge = EdgeData {
            source_node_id: PATH_START_NODE_ID,
            source_coordinate: 0,
            source_strand: Strand::Forward,
            target_node_id: *start_part,
            target_coordinate: 0,
            target_strand: Strand::Forward,
        };
        new_edges.insert(edge);
    }

    let end_parts = parts_list
        .last()
        .ok_or_else(|| LibraryImportError::ImportFailed("No parts found.".to_string()))?;
    for end_part in *end_parts {
        let end_part_source_coordinate = sequence_lengths_by_node_id
            .get(end_part)
            .ok_or_else(|| LibraryImportError::ImportFailed(format!("Part {end_part} missing.")))?;
        let edge = EdgeData {
            source_node_id: *end_part,
            source_coordinate: *end_part_source_coordinate,
            source_strand: Strand::Forward,
            target_node_id: PATH_END_NODE_ID,
            target_coordinate: 0,
            target_strand: Strand::Forward,
        };
        new_edges.insert(edge);
    }

    let mut path_changes_count = 1;
    for (parts1, parts2) in parts_list.iter().tuple_windows() {
        path_changes_count *= parts1.len();
        for part1 in *parts1 {
            for part2 in *parts2 {
                let part1_source_coordinate =
                    sequence_lengths_by_node_id.get(part1).ok_or_else(|| {
                        LibraryImportError::ImportFailed(format!("Part {part1} missing."))
                    })?;
                let edge = EdgeData {
                    source_node_id: *part1,
                    source_coordinate: *part1_source_coordinate,
                    source_strand: Strand::Forward,
                    target_node_id: *part2,
                    target_coordinate: 0,
                    target_strand: Strand::Forward,
                };
                new_edges.insert(edge);
            }
        }
    }

    path_changes_count *= end_parts.len();

    let new_edge_ids = Edge::bulk_create(conn, &new_edges.iter().cloned().collect::<Vec<_>>());

    let new_block_group_edges = new_edge_ids
        .iter()
        .map(|edge_id| BlockGroupEdgeData {
            block_group_id: new_block_group.id,
            edge_id: *edge_id,
            chromosome_index: edge_id.extract_digits(), // TODO: This is a hack, clean it up with phase layers
            phased: 0,
        })
        .collect::<Vec<_>>();
    BlockGroupEdge::bulk_create(conn, &new_block_group_edges);

    let mut path_node_ids = vec![];
    path_node_ids.push(PATH_START_NODE_ID);
    for parts in &parts_list {
        path_node_ids.push(parts[0]);
    }
    path_node_ids.push(PATH_END_NODE_ID);

    let new_edges = Edge::query_by_ids(conn, &new_edge_ids);
    let new_edge_ids_by_source_and_target_node = new_edges
        .iter()
        .map(|edge| ((edge.source_node_id, edge.target_node_id), edge.id))
        .collect::<HashMap<_, _>>();
    let path_edge_ids = path_node_ids
        .iter()
        .tuple_windows()
        .map(|(source_node_id, target_node_id)| {
            *new_edge_ids_by_source_and_target_node
                .get(&(*source_node_id, *target_node_id))
                .unwrap()
        })
        .collect::<Vec<_>>();
    Path::create(
        conn,
        format!("{library_name} default path").as_str(),
        &new_block_group.id,
        &path_edge_ids,
    );

    let summary_str = format!("{library_name}: {path_changes_count} changes.\n");
    let op = session_operations::end_operation(
        conn,
        operation_conn,
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
    use std::path::PathBuf;

    use gen_models::block_group::BlockGroup;

    use super::*;
    use crate::{
        test_helpers::{get_connection, get_operation_connection, setup_gen_dir},
        track_database,
    };

    #[test]
    fn imports_a_library() {
        setup_gen_dir();
        let conn = &get_connection(None).unwrap();
        let op_conn = &get_operation_connection(None).unwrap();

        track_database(conn, op_conn).unwrap();
        let collection = "test";

        let parts_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/affix_parts.fa");
        let library_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/affix_layout.csv");

        let _ = import_library(
            conn,
            op_conn,
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
        setup_gen_dir();
        let conn = &get_connection(None).unwrap();
        let op_conn = &get_operation_connection(None).unwrap();

        track_database(conn, op_conn).unwrap();
        let collection = "test";

        let parts_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/parts.fa");
        let library_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/single_column_design.csv");

        let _ = import_library(
            conn,
            op_conn,
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
        setup_gen_dir();
        let conn = &get_connection(None).unwrap();
        let op_conn = &get_operation_connection(None).unwrap();

        track_database(conn, op_conn).unwrap();
        let collection = "test";

        let parts_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/parts.fa");
        let library_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/design_reusing_parts.csv");

        let _ = import_library(
            conn,
            op_conn,
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
