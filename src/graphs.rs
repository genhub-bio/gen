use gen_core::{HashId, Strand};
use gen_models::{
    block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
    db::GraphConnection,
    edge::{Edge, EdgeData},
};

pub mod combinatorial_library;
pub mod operators;

/// Creates edges between source nodes and target nodes, and also creates block
/// group edges for the given block group ID.  Returns a vector of the created edge IDs.
pub fn stitch(
    conn: &GraphConnection,
    source_nodes: &[HashId],
    source_coordinates: Vec<i64>,
    target_nodes: &Vec<HashId>,
    block_group_id: HashId,
) -> Vec<HashId> {
    let mut edges = vec![];

    // Create edges between source nodes and target nodes
    for (i, source) in source_nodes.iter().enumerate() {
        for target in target_nodes {
            edges.push(EdgeData {
                source_node_id: *source,
                source_coordinate: source_coordinates[i],
                source_strand: Strand::Forward,
                target_node_id: *target,
                target_coordinate: 0,
                target_strand: Strand::Forward,
            });
        }
    }

    let edge_ids: Vec<HashId> = Edge::bulk_create(conn, &edges);

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
