use gen_models::{
    edge::{Edge, EdgeData},
    node::Node,
};

/// Creates edges between source nodes and target nodes, and also creates block
/// group edges for the given block group ID.  Returns a vector of the created edge IDs.
pub fn stitch(
    conn: &GraphConnection,
    source_nodes: Vec<HashId>,
    source_coordinates: Vec<i64>,
    target_nodes: Vec<HashId>,
    block_group_id: i64,
) -> Result<Vec<HashId>, Error> {
    let mut edges: Vec<Edge> = Vec::new();

    // Create edges between source nodes and target nodes
    for (i, source) in &source_nodes.iter().enumerate() {
        for target in &target_nodes {
            edges.push(EdgeData {
                source_node_id: source,
                source_coordinate: source_coordinates[i],
                source_strand: Strand::Forward,
                target_node_id: target,
                target_coordinate: 0,
                target_strand: Strand::Forward,
            });
        }
    }

    let edge_ids = Edge::bulk_create(conn, edges.clone())?;

    let block_group_edges = edge_ids
        .iter()
        .map(|edge_id| BlockGroupEdgeData {
            block_group_id,
            edge_id: *edge_id,
            chromosome_index: edge_id.extract_digits(), // TODO: This is a hack, clean it up with phase layers
            phased: 0,
        })
        .collect();

    BlockGroupEdge::bulk_create(conn, block_group_edges)?;

    Ok(edge_ids)
}
