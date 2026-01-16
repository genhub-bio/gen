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
    db::GraphConnection,
    edge::{Edge, EdgeData},
    node::Node,
    path::Path,
    sequence::Sequence,
    traits::Query,
};
use itertools::Itertools;
use noodles::fasta;
use thiserror::Error;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct SequencePart {
    pub name: String,
    pub sequence: String,
    pub sequence_length: i64,
}

#[derive(Error, Debug)]
pub enum CombinatorialLibraryError {
    #[error("Failed to parse fasta")]
    FastaParseFailed(String),
    #[error("Failed to parse library CSV")]
    CSVParseFailed(String),
    #[error("Failed to create library")]
    CreationFailed(String),
}

pub fn parse_library(
    parts_filename: &str,
    design_filename: &str,
) -> Result<Vec<Vec<SequencePart>>, CombinatorialLibraryError> {
    let mut sequence_parts_by_name = HashMap::new();

    let mut parts_reader = fasta::io::reader::Builder
        .build_from_path(parts_filename)
        .map_err(|e| CombinatorialLibraryError::FastaParseFailed(e.to_string()))?;
    for result in parts_reader.records() {
        let record =
            result.map_err(|e| CombinatorialLibraryError::FastaParseFailed(e.to_string()))?;
        let sequence = str::from_utf8(record.sequence().as_ref())
            .map_err(|e| CombinatorialLibraryError::FastaParseFailed(e.to_string()))?;
        let name = String::from_utf8(record.name().to_vec()).unwrap();
        sequence_parts_by_name.insert(
            name.clone(),
            SequencePart {
                name: name.to_string(),
                sequence: sequence.to_string(),
                sequence_length: sequence.len() as i64,
            },
        );
    }

    let library_file = File::open(design_filename).map_err(|e| {
        CombinatorialLibraryError::CSVParseFailed(format!(
            "Failed to open library file {design_filename}: {}",
            e
        ))
    })?;
    let library_reader = BufReader::new(library_file);

    let mut parts_by_index = HashMap::new();
    let mut library_csv_reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(library_reader);
    let mut max_index = 0;
    for result in library_csv_reader.records() {
        let record =
            result.map_err(|e| CombinatorialLibraryError::CSVParseFailed(e.to_string()))?;
        for (index, part_name) in record.iter().enumerate() {
            if !part_name.is_empty() {
                let part = sequence_parts_by_name.get(part_name).ok_or_else(|| {
                    CombinatorialLibraryError::CSVParseFailed(format!("Part {part_name} missing."))
                })?;
                parts_by_index
                    .entry(index)
                    .or_insert(vec![])
                    .push(part.clone());
                if index >= max_index {
                    max_index = index + 1;
                }
            }
        }
    }

    let mut parts_list = vec![];
    for index in 0..max_index {
        let parts = parts_by_index.get(&index).ok_or_else(|| {
            CombinatorialLibraryError::CSVParseFailed(format!("Missing index {index}."))
        })?;
        parts_list.push(parts.clone());
    }

    Ok(parts_list)
}

pub fn create_library(
    conn: &GraphConnection,
    block_group: &BlockGroup,
    library_name: &str,
    parts_list: Vec<Vec<SequencePart>>,
) -> Result<u64, CombinatorialLibraryError> {
    let mut parts_set = HashSet::new();
    for parts in &parts_list {
        parts_set.extend(parts);
    }

    let mut sequence_hashes_by_name = HashMap::new();
    for part in parts_set {
        let seq = Sequence::new()
            .sequence_type("DNA")
            .sequence(&part.sequence)
            .save(conn);

        sequence_hashes_by_name.insert(part.name.clone(), seq.hash);
    }

    let mut part_nodes_list = vec![];
    let mut sequence_lengths_by_node_id = HashMap::new();
    for (index, parts) in parts_list.iter().enumerate() {
        let mut part_nodes = vec![];
        for part in parts {
            let part_hash = sequence_hashes_by_name.get(&part.name).ok_or_else(|| {
                CombinatorialLibraryError::CreationFailed(format!("Part {} missing.", part.name))
            })?;
            let part_node_id = Node::create(
                conn,
                part_hash,
                &HashId::convert_str(&format!(
                    "{library_name}:{part_name}:{ref_start}-{ref_end}->{sequence_hash}-column-{index}",
                    part_name = part.name,
                    ref_start = 0,
                    ref_end = part.sequence_length,
                    sequence_hash = part_hash
                )),
            );
            part_nodes.push(part_node_id);
            sequence_lengths_by_node_id.insert(part_node_id, part.sequence_length);
        }
        part_nodes_list.push(part_nodes);
    }

    let mut new_edges = HashSet::new();
    let start_parts = part_nodes_list
        .first()
        .ok_or_else(|| CombinatorialLibraryError::CreationFailed("No parts found.".to_string()))?;
    for start_part in start_parts {
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

    let end_parts = part_nodes_list
        .last()
        .ok_or_else(|| CombinatorialLibraryError::CreationFailed("No parts found.".to_string()))?;
    for end_part in end_parts {
        let end_part_source_coordinate =
            sequence_lengths_by_node_id.get(end_part).ok_or_else(|| {
                CombinatorialLibraryError::CreationFailed(format!("Part {end_part} missing."))
            })?;
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
    for (parts1, parts2) in part_nodes_list.iter().tuple_windows() {
        path_changes_count *= parts1.len();
        for part1 in parts1 {
            for part2 in parts2 {
                let part1_source_coordinate =
                    sequence_lengths_by_node_id.get(part1).ok_or_else(|| {
                        CombinatorialLibraryError::CreationFailed(format!("Part {part1} missing."))
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
            block_group_id: block_group.id,
            edge_id: *edge_id,
            chromosome_index: edge_id.extract_digits(), // TODO: This is a hack, clean it up with phase layers
            phased: 0,
        })
        .collect::<Vec<_>>();
    BlockGroupEdge::bulk_create(conn, &new_block_group_edges);

    let mut path_node_ids = vec![];
    path_node_ids.push(PATH_START_NODE_ID);
    for parts in &part_nodes_list {
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
        &block_group.id,
        &path_edge_ids,
    );

    Ok(path_changes_count as u64)
}
