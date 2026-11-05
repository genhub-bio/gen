use thiserror::Error;

pub mod center_doglegs;
pub mod layout_graph_process;
pub mod route_channel;
pub mod route_graph;
pub mod route_layer;
#[cfg(test)]
pub mod simple_test;
pub mod temp_graph;

#[derive(Clone, Debug, Eq, Error, Hash, PartialEq)]
pub enum LayoutError {
    #[error("Invalid side (only 'T', 'B', or None are allowed): {0}")]
    InvalidSide(String),
    #[error("Node not found in temp graph: {0}")]
    NodeNotFound(u64),
    #[error("Missing original node ID for data node: {0}")]
    MissingOriginalNodeId(u64),
}

#[derive(Clone, Debug)]
pub struct NodeData {
    pub node_id: u64,
    pub position: (i64, i64),
    pub node_type: Option<String>,
    pub size: (i64, i64),
    pub original_node_id: Option<u64>,
    pub layer: Option<i32>,
}

/// Call the Rust edge router directly with a LayoutNode/LayoutEdge graph.
/// Kept as test-only infrastructure to exercise `make_rectilinear` end to end.
#[cfg(test)]
pub(crate) fn call_rust_router(
    mut graph: petgraph::stable_graph::StableGraph<
        crate::layout::LayoutNode,
        crate::layout::LayoutEdge,
        petgraph::Undirected,
        u32,
    >,
) -> Result<
    petgraph::stable_graph::StableGraph<
        crate::layout::LayoutNode,
        crate::layout::LayoutEdge,
        petgraph::Undirected,
        u32,
    >,
    LayoutError,
> {
    crate::edge_router::route_graph::make_rectilinear(&mut graph)?;

    Ok(graph)
}
