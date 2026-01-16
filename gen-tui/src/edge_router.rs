use thiserror::Error;

pub mod adapter;
pub mod layout_graph_process;
pub mod route_channel;
pub mod route_graph;
pub mod route_layer;
pub mod simple_test;
pub mod temp_graph;

#[derive(Clone, Debug, Eq, Error, Hash, PartialEq)]
pub enum LayoutError {
    #[error("Invalid side (only 'T', 'B', or None are allowed): {0}")]
    InvalidSide(String),
    #[error("Failed to route the edges")]
    FailureAfterRetries,
    #[error("Invalid number of coordinates: {0}")]
    InvalidCoordinateNumber(i64),
    #[error("Node not found in temp graph: {0}")]
    NodeNotFound(u64),
    #[error("Edge not found in temp graph: {0}")]
    EdgeNotFound(u64),
    #[error("Invalid position: x={0}, y={1}")]
    InvalidPosition(i32, i32),
    #[error("Invalid size: width={0}, height={1}")]
    InvalidSize(u64, u64),
    #[error("Empty edge bundle")]
    EmptyEdgeBundle,
    #[error("Invalid graph: {0}")]
    InvalidGraph(String),
    #[error("Missing original node ID for data node: {0}")]
    MissingOriginalNodeId(u64),
}

#[derive(Clone, Debug)]
pub struct NodeData {
    pub node_id: u64, // Internal routing algorithm node ID
    pub position: (i64, i64),
    pub node_type: Option<String>,
    pub ports: Option<(bool, bool, bool, bool)>,
    pub glyph_index: Option<i64>,
    pub size: (i64, i64),
    // Original domain data - None for routing/stitch nodes
    pub original_node_id: Option<u64>, // Original domain node ID for data nodes
    pub layer: Option<i32>,            // Layer information for data nodes
    pub partition_index: Option<u64>,  // Partition index for data nodes
}

/// Call the Rust edge router using adapter functions to convert between LayoutNode/LayoutEdge and NodeData/LayoutEdge
pub fn call_rust_router(
    mut graph: petgraph::stable_graph::StableGraph<
        crate::layout::LayoutNode,
        crate::layout::LayoutEdge,
        petgraph::Undirected,
        u32,
    >,
    vertex_spacing: f64,
) -> Result<
    petgraph::stable_graph::StableGraph<
        crate::layout::LayoutNode,
        crate::layout::LayoutEdge,
        petgraph::Undirected,
        u32,
    >,
    LayoutError,
> {
    // Call make_rectilinear directly with the LayoutNode graph - no conversion needed!
    crate::edge_router::route_graph::make_rectilinear(&mut graph, vertex_spacing)?;

    Ok(graph)
}
