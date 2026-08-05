use gen_core::{GenGraph, HashId, NodeIntervalBlock};
use gen_graph::{flatten_to_interval_tree, graph_loader};
use intervaltree::IntervalTree;

use crate::{
    block_group::BlockGroupError, block_group_edge::BlockGroupEdge, db::GraphConnection, edge::Edge,
};

pub fn prune_graph(graph: &mut GenGraph) {
    graph_loader::prune_graph(graph);
}

/// Loads persisted block-group records and delegates graph construction to `gen-graph`.
pub fn load_block_group_graph(
    conn: &GraphConnection,
    block_group_id: &HashId,
    history_ref: Option<&str>,
) -> Result<GenGraph, BlockGroupError> {
    let edges = BlockGroupEdge::edges_for_block_group(conn, block_group_id, history_ref);
    let blocks = Edge::blocks_from_edges(conn, block_group_id, &edges, history_ref)?;
    let (load_edges, load_blocks) = Edge::graph_load_data(&edges, &blocks);
    let (graph, _) = graph_loader::build_graph(&load_edges, &load_blocks);
    Ok(graph)
}

/// Loads a block group and projects its graph nodes into linear intervals.
pub fn load_block_group_intervaltree(
    conn: &GraphConnection,
    block_group_id: &HashId,
    remove_ambiguous_positions: bool,
) -> Result<IntervalTree<i64, NodeIntervalBlock>, BlockGroupError> {
    let mut graph = load_block_group_graph(conn, block_group_id, None)?;
    prune_graph(&mut graph);
    Ok(flatten_to_interval_tree(&graph, remove_ambiguous_positions))
}

/// Identify all edges leading to and from a provided node_id in a block_group and merge them into an existing GenGraph.
/// Returns true if the graph was expanded, false if no new edges were added.
pub fn expand(
    conn: &GraphConnection,
    graph: &mut GenGraph,
    block_group_id: &HashId,
    node_id: HashId,
) -> bool {
    let candidate_edges =
        match Edge::edges_for_block_group_node_neighborhood(conn, block_group_id, node_id, None) {
            Ok(edges) => edges,
            Err(_) => return false,
        };
    let unloaded_edge_ids =
        graph_loader::unloaded_edge_ids(graph, candidate_edges.iter().map(|edge| edge.edge.id));
    let unloaded_edges = candidate_edges
        .into_iter()
        .filter(|edge| unloaded_edge_ids.contains(&edge.edge.id))
        .collect::<Vec<_>>();
    if unloaded_edges.is_empty() {
        return false;
    }
    let blocks = match Edge::blocks_from_edges(conn, block_group_id, &unloaded_edges, None) {
        Ok(blocks) => blocks,
        Err(_) => return false,
    };
    let (load_edges, load_blocks) = Edge::graph_load_data(&unloaded_edges, &blocks);
    let (fragment, _) = graph_loader::build_graph(&load_edges, &load_blocks);
    graph_loader::merge_fragment(graph, &fragment);
    true
}
