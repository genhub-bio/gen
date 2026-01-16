use std::fmt;

use petgraph::stable_graph::{NodeIndex, StableDiGraph};
use serde::{Deserialize, Serialize};

use crate::layout::{PartitionLayout, VisualDetail};

/// Graph layout calculations can be computationally expensive, this can be mitigated
/// by partitioning the graph into partitions that are rendered as needed.
/// Partitions include a subgraph in the form of a StableDiGraph with two types of node weights:
/// the original nodes from the main graph and dummy nodes used to stitch partitions together.
/// Both are referenced by their NodeIndex during the actual Layout phase.
#[derive(Clone, Debug)]
pub enum PartitionNode {
    Data(NodeIndex),
    Stitch(StitchSide),
}

#[derive(Clone, Debug, Serialize, Deserialize, Copy, PartialEq, Eq)]
pub enum StitchSide {
    Left,
    Right,
}

/// Represents an edge in a partition graph.
/// Contains the original source and target node indices from the domain graph.
/// None represents edges to/from boundary stitch nodes that don't correspond to original edges.
pub type PartitionEdge = Option<(NodeIndex, NodeIndex)>;

/// Represents a single partition with its directed layout graph
/// and computed layouts at different levels of detail.
#[derive(Clone)]
pub struct Partition {
    pub graph: StableDiGraph<PartitionNode, PartitionEdge, u32>,
    /// Each partition has at most one left and one right stitch node.
    /// They are added to the graph during the partitioning process,
    /// and are used to align the partitions with each other.
    /// Do not try to use these with Layout graphs, which index their
    /// node differently.
    pub left_stitch_idx: Option<NodeIndex<u32>>,
    pub right_stitch_idx: Option<NodeIndex<u32>>,
    pub layouts: [Option<PartitionLayout>; 3],
}

impl Default for Partition {
    fn default() -> Self {
        Self::new()
    }
}

impl Partition {
    pub fn new() -> Self {
        Partition {
            graph: StableDiGraph::new(),
            left_stitch_idx: None,
            right_stitch_idx: None,
            layouts: [None, None, None],
        }
    }

    /// Get coordinates for a specific layout type
    #[allow(clippy::type_complexity)]
    pub fn get_coordinates(
        &self,
        detail_level: VisualDetail,
    ) -> Option<Box<dyn Iterator<Item = (NodeIndex<u32>, (i64, i64))> + '_>> {
        let layout = self.layouts[detail_level.as_index()].as_ref()?;

        Some(Box::new(layout.graph.node_indices().map(|node_idx| {
            let pos = layout.get_node_position(node_idx).unwrap();
            (node_idx, (pos.x, pos.y))
        })))
    }

    /// Check if a specific layout type has coordinates
    pub fn has_layout(&self, detail_level: VisualDetail) -> bool {
        self.layouts[detail_level.as_index()].is_some()
    }

    /// Get the width of a specific layout
    pub fn get_width(&self, detail_level: VisualDetail) -> i64 {
        self.layouts[detail_level.as_index()]
            .as_ref()
            .map_or(0, |l| l.width)
    }
}

impl fmt::Debug for Partition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Partition")
            .field("graph", &self.graph)
            .field("left_stitch_idx", &self.left_stitch_idx)
            .field("right_stitch_idx", &self.right_stitch_idx)
            .field("layouts", &self.layouts.as_ref().iter().map(|_| "<Layout>"))
            .finish()
    }
}
