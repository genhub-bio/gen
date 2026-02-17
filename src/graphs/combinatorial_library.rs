use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::BufReader,
    str,
};

use anyhow::Result;
use gen_core::{HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand};
use gen_models::{
    db::GraphConnection, edge::Edge, node::Node, path::Path, sequence::Sequence, traits::Query,
};
use itertools::Itertools;
use noodles::fasta;
use thiserror::Error;

use crate::graphs::{BlockGroupChunk, NodePoint, stitch};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct SequencePart {
    pub name: String,
    pub sequence: String,
    pub sequence_length: i64,
}

#[derive(Error, Debug)]
pub enum CombinatorialLibraryParseError {
    #[error("Failed to parse fasta")]
    FastaParseFailed(String),
    #[error("Failed to parse library CSV")]
    CSVParseFailed(String),
}

#[derive(Error, Debug)]
pub enum CombinatorialLibraryCreationError {
    #[error("Failed to create library")]
    CreationFailed(String),
}

pub fn parse_library(
    parts_filename: &str,
    design_filename: &str,
) -> Result<Vec<Vec<SequencePart>>, CombinatorialLibraryParseError> {
    let mut sequence_parts_by_name = HashMap::new();

    let mut parts_reader = fasta::io::reader::Builder
        .build_from_path(parts_filename)
        .map_err(|e| CombinatorialLibraryParseError::FastaParseFailed(e.to_string()))?;
    for result in parts_reader.records() {
        let record =
            result.map_err(|e| CombinatorialLibraryParseError::FastaParseFailed(e.to_string()))?;
        let sequence = str::from_utf8(record.sequence().as_ref())
            .map_err(|e| CombinatorialLibraryParseError::FastaParseFailed(e.to_string()))?;
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
        CombinatorialLibraryParseError::CSVParseFailed(format!(
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
            result.map_err(|e| CombinatorialLibraryParseError::CSVParseFailed(e.to_string()))?;
        for (index, part_name) in record.iter().enumerate() {
            if !part_name.is_empty() {
                let part = sequence_parts_by_name.get(part_name).ok_or_else(|| {
                    CombinatorialLibraryParseError::CSVParseFailed(format!(
                        "Part {part_name} missing."
                    ))
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
            CombinatorialLibraryParseError::CSVParseFailed(format!("Missing index {index}."))
        })?;
        parts_list.push(parts.clone());
    }

    Ok(parts_list)
}

pub fn create_library(
    conn: &GraphConnection,
    block_group_id: HashId,
    library_name: &str,
    parts_list: Vec<Vec<SequencePart>>,
    create_terminal_edges: bool,
) -> Result<BlockGroupChunk, CombinatorialLibraryCreationError> {
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
    if create_terminal_edges {
        part_nodes_list.push(vec![PATH_START_NODE_ID]);
        sequence_lengths_by_node_id.insert(PATH_START_NODE_ID, 0);
    }

    for (index, parts) in parts_list.iter().enumerate() {
        let mut part_nodes = vec![];
        for part in parts {
            let part_hash = sequence_hashes_by_name.get(&part.name).ok_or_else(|| {
                CombinatorialLibraryCreationError::CreationFailed(format!(
                    "Part {} missing.",
                    part.name
                ))
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

    if create_terminal_edges {
        part_nodes_list.push(vec![PATH_END_NODE_ID]);
        sequence_lengths_by_node_id.insert(PATH_END_NODE_ID, 0);
    }

    let mut new_edge_ids = vec![];

    for (parts1, parts2) in part_nodes_list.iter().tuple_windows() {
        let mut source_node_points = vec![];
        for part1 in parts1 {
            let part1_length = sequence_lengths_by_node_id.get(part1).ok_or_else(|| {
                CombinatorialLibraryCreationError::CreationFailed(format!("Part {part1} missing."))
            })?;
            source_node_points.push(NodePoint {
                id: *part1,
                coordinate: *part1_length,
                strand: Strand::Forward,
            });
        }

        let target_node_points = parts2
            .iter()
            .map(|part| NodePoint {
                id: *part,
                coordinate: 0,
                strand: Strand::Forward,
            })
            .collect();

        new_edge_ids.extend(stitch(
            conn,
            &source_node_points,
            &target_node_points,
            block_group_id,
        ));
    }

    let path_node_ids: Vec<_> = part_nodes_list.iter().map(|parts| parts[0]).collect();

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
        &block_group_id,
        &path_edge_ids,
    );

    let mut entry_node_points = vec![];
    for part in &part_nodes_list[0] {
        let part_length = sequence_lengths_by_node_id.get(part).ok_or_else(|| {
            CombinatorialLibraryCreationError::CreationFailed(format!("Part {part} missing."))
        })?;
        entry_node_points.push(NodePoint {
            id: *part,
            coordinate: *part_length,
            strand: Strand::Forward,
        });
    }

    let exit_node_points = part_nodes_list[part_nodes_list.len() - 1]
        .iter()
        .map(|part| NodePoint {
            id: *part,
            coordinate: 0,
            strand: Strand::Forward,
        })
        .collect();

    let block_group_chunk = BlockGroupChunk {
        entry_node_points,
        exit_node_points,
    };

    Ok(block_group_chunk)
}
