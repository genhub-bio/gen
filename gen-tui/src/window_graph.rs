use std::{collections::HashSet, fmt};

use petgraph::stable_graph::{NodeIndex, StableDiGraph};

use crate::layout::WindowStructure;

/// A window's layout graph carries the original domain nodes, the pin nodes used to
/// render a backward (cycle-closing) edge as a bypass, and the wormhole nodes standing in
/// for off-window neighbours. All three kinds are referenced by their `NodeIndex` during
/// the actual Layout phase.
#[derive(Clone, Debug)]
pub enum WindowNode {
    Data(NodeIndex),
    /// Synthetic node injected to render a backward edge as a full-width loop.
    /// Carries no domain data; sized and rendered like an ordinary routing node.
    Pin,
    /// A door standing in for a windowed node's off-window neighbours on one side
    /// (see `crawl::ExternalEdge`). Carries the chosen off-window navigation target;
    /// the full collapsed neighbour list is carried on the edge(s) to this node instead
    /// (see `build_window_graph`), matching how ordinary parallel domain edges bundle.
    Wormhole(NodeIndex),
}

/// Represents an edge in a window's layout graph.
/// Contains the original source and target node indices from the domain graph.
/// None represents edges that don't correspond to original domain edges.
pub type WindowEdge = Option<(NodeIndex, NodeIndex)>;

/// A single crawled window's directed layout graph and its cached LOD-invariant Sugiyama
/// structure. Geometry (coordinates, routing, compaction) is not resident here: the widget
/// assembles and sizes the window fresh per frame from this cached structure.
#[derive(Clone)]
pub struct WindowGraph {
    pub graph: StableDiGraph<WindowNode, WindowEdge, u32>,
    /// Level-of-detail-independent Sugiyama layering for this window (layers plus
    /// within-layer order). Computed once when the window is first laid out. `None` for
    /// empty or single-node windows, which have no layering.
    pub structure: Option<WindowStructure>,
    /// Edges that render a backward (cycle-closing) edge's `left_pin -> right_pin` main
    /// span - the one leg of the three the loop is rewired onto (see `build_window_graph`)
    /// that actually runs backward, the other two being short excursions to/from the real
    /// endpoints. Keyed by the `(from, to)` node pair exactly as passed to `graph.add_edge`.
    /// Set at window-build time; empty for a window with no backward edges.
    /// `WindowStructure` resolves this once per edge into `backward_span_edges` (keyed by
    /// post-dummy-insertion vertex pairs) when the window's structure is built, since dummy
    /// insertion can split one of these edges into a multi-hop chain.
    pub backward_span_edges: HashSet<(NodeIndex<u32>, NodeIndex<u32>)>,
}

impl Default for WindowGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl WindowGraph {
    pub fn new() -> Self {
        WindowGraph {
            graph: StableDiGraph::new(),
            structure: None,
            backward_span_edges: HashSet::new(),
        }
    }
}

impl fmt::Debug for WindowGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WindowGraph")
            .field("graph", &self.graph)
            .field("structure", &self.structure.as_ref().map(|_| "<Structure>"))
            .finish()
    }
}
