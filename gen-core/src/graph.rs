use core::fmt;

use petgraph::graphmap::DiGraphMap;
use serde::{Deserialize, Serialize};

use crate::{HashId, Strand};

pub type GenGraph = DiGraphMap<GraphNode, Vec<GraphEdge>>;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Deserialize, Serialize)]
pub struct GraphEdge {
    pub edge_id: HashId,
    pub source_strand: Strand,
    pub target_strand: Strand,
    pub chromosome_index: i64,
    pub phased: i64,
    pub created_on: i64,
}

/// A contiguous slice of a stored sequence represented as a node in graph space.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Deserialize, Serialize)]
pub struct GraphNode {
    pub node_id: HashId,
    pub sequence_start: i64,
    pub sequence_end: i64,
}

impl fmt::Display for GraphNode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}[{}-{}]",
            self.node_id, self.sequence_start, self.sequence_end
        )
    }
}

impl GraphNode {
    pub const fn length(&self) -> i64 {
        self.sequence_end - self.sequence_start
    }
}

/// A local sequence slice within a graph node, including its traversal orientation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GraphNodeSlice {
    pub block: GraphNode,
    /// Local start offset within the block's sequence slice (`0..block.length()`).
    pub start: usize,
    /// Local end offset, exclusive (`start..=block.length()`).
    pub end: usize,
    /// 5'→3' orientation of this slice.
    pub strand: Strand,
}

impl GraphNodeSlice {
    pub fn full(block: GraphNode, strand: Strand) -> Self {
        Self {
            block,
            start: 0,
            end: block.length() as usize,
            strand,
        }
    }
}

/// A cursor within a graph node.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Deserialize, Serialize)]
pub struct GraphNodePosition {
    pub graph_node: GraphNode,
    /// Distance from `sequence_start` of `graph_node`.
    pub offset: i64,
}

impl GraphNodePosition {
    pub const fn coordinate(&self) -> i64 {
        self.graph_node.sequence_start + self.offset
    }
}
