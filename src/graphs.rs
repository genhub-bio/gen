use std::collections::HashMap;

use gen_core::{HashId, Strand};
use gen_models::{
    block_group::BlockGroup,
    block_group_edge::{AugmentedEdge, BlockGroupEdge, BlockGroupEdgeData},
    db::GraphConnection,
    edge::{Edge, EdgeData},
    path_edge::PathEdge,
};

pub mod combinatorial_library;
pub mod operators;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct NodePoint {
    id: HashId,
    coordinate: i64,
    strand: Strand,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct BlockGroupChunk {
    entry_node_points: Vec<NodePoint>,
    exit_node_points: Vec<NodePoint>,
    //    path_edges: Vec<AugmentedEdge>,
}

pub fn get_block_group_chunk(conn: &GraphConnection, block_group_id: HashId) -> BlockGroupChunk {
    let edges = BlockGroupEdge::edges_for_block_group(conn, &block_group_id);

    let start_edges = edges
        .iter()
        .filter(|edge| edge.edge.is_start_edge())
        .collect::<Vec<_>>();

    let entry_node_points = start_edges
        .iter()
        .map(|start_edge| NodePoint {
            id: start_edge.edge.target_node_id,
            coordinate: start_edge.edge.target_coordinate,
            strand: start_edge.edge.target_strand,
        })
        .collect();

    let end_edges = edges.iter().filter(|edge| edge.edge.is_end_edge());
    let exit_node_points = end_edges
        .map(|edge| NodePoint {
            id: edge.edge.source_node_id,
            coordinate: edge.edge.source_coordinate,
            strand: edge.edge.source_strand,
        })
        .collect();

    let path = BlockGroup::get_current_path(conn, &block_group_id);
    let path_edges = PathEdge::edges_for_path(conn, &path.id);
    let edges_by_id = edges
        .iter()
        .map(|edge| (edge.edge.id, edge.clone()))
        .collect::<HashMap<HashId, AugmentedEdge>>();
    let mut chunk_path_edges = vec![];

    for path_edge in &path_edges {
        let augmented_edge = edges_by_id.get(&path_edge.id).unwrap();
        chunk_path_edges.push(augmented_edge);
    }

    BlockGroupChunk {
        entry_node_points,
        exit_node_points,
    }
}

/// Creates edges between source nodes and target nodes, and also creates block
/// group edges for the given block group ID.  Returns a vector of the created edge IDs.
pub fn stitch(
    conn: &GraphConnection,
    source_node_points: &Vec<NodePoint>,
    target_node_points: &Vec<NodePoint>,
    block_group_id: HashId,
) -> Vec<HashId> {
    let mut edges = vec![];

    // Create edges between source nodes and target nodes
    for source_point in source_node_points {
        for target_point in target_node_points {
            edges.push(EdgeData {
                source_node_id: source_point.id,
                source_coordinate: source_point.coordinate,
                source_strand: source_point.strand,
                target_node_id: target_point.id,
                target_coordinate: target_point.coordinate,
                target_strand: target_point.strand,
            });
        }
    }

    let edge_ids: Vec<HashId> = Edge::bulk_create(conn, &edges);

    // TODO: Set chromosome index correctly
    let block_group_edges = edge_ids
        .iter()
        .map(|edge_id| BlockGroupEdgeData {
            block_group_id,
            edge_id: *edge_id,
            chromosome_index: edge_id.extract_digits(), // TODO: This is a hack, clean it up with phase layers
            phased: 0,
        })
        .collect::<Vec<BlockGroupEdgeData>>();

    BlockGroupEdge::bulk_create(conn, &block_group_edges);

    edge_ids
}
