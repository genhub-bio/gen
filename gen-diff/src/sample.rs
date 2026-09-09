//! Builds a graph diff between two samples.
//!
//! The base is the reference state and the query is compared against it. Graph
//! elements found only in the query are additions, while elements found only
//! in the base are removals.

use gen_models::{
    block_group::{BlockGroup, BlockGroupError},
    block_group_edge::BlockGroupEdge,
    db::GraphConnection,
};

use crate::graph::{DiffGenGraph, EdgeDiffInput, build_diff_graph_from_edges};

/// The two sample block groups and their unified, annotated graph.
#[derive(Clone, Debug)]
pub struct SampleDiff {
    /// Graph selected as the query for the comparison.
    pub query_block_group: BlockGroup,
    /// Graph selected as the base for the comparison.
    pub base_block_group: BlockGroup,
    /// Unified graph annotated relative to the base.
    pub graph: DiffGenGraph,
}

/// Builds one graph showing the difference between matching query and base
/// block groups.
///
/// The base is the comparison baseline. Query-only graph elements are added,
/// while base-only elements are removed. The shared graph-diff builder aligns
/// sequence boundaries before constructing either diff input, then unifies
/// their explicitly annotated nodes and edges.
pub fn build_sample_diff(
    conn: &GraphConnection,
    collection_name: &str,
    graph_name: &str,
    query_name: &str,
    base_name: &str,
    history_ref: Option<&str>,
) -> Result<SampleDiff, BlockGroupError> {
    let query_block_group =
        BlockGroup::get_by_name(conn, collection_name, query_name, graph_name, history_ref)?;
    let base_block_group =
        BlockGroup::get_by_name(conn, collection_name, base_name, graph_name, history_ref)?;
    let query_edges =
        BlockGroupEdge::edges_for_block_group(conn, &query_block_group.id, history_ref);
    let base_edges = BlockGroupEdge::edges_for_block_group(conn, &base_block_group.id, history_ref);
    let graph = build_diff_graph_from_edges(
        EdgeDiffInput::unattributed(&base_edges),
        EdgeDiffInput::unattributed(&query_edges),
        None,
    );

    Ok(SampleDiff {
        query_block_group,
        base_block_group,
        graph,
    })
}
