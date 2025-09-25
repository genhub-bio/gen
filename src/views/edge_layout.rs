use thiserror::Error;

pub mod layout_channel;
pub mod layout_graph;
pub mod layout_layer;
pub mod process_graph;
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
}

#[derive(Clone, Debug)]
pub struct NodeData {
    pub node_id: u64,
    pub position: (i64, i64),
    pub node_type: Option<String>,
    pub ports: Option<(bool, bool, bool, bool)>,
    pub glyph_index: Option<i64>,
    pub size: (i64, i64),
}

#[derive(Clone, Debug)]
pub struct EdgeData {
    pub role: Option<String>,
}
