//! The crawled, coordinate-free form of a window.
//!
//! A [`WindowGraph`] is the first product of the window pipeline. `crawl::build_window_graph`
//! walks the domain graph around a batch of nodes and records its domain nodes, the pins that
//! carry bends and loops, and the wormholes standing in for off-window neighbors, together with
//! the size-independent Sugiyama layering of all of them. `assembly::assemble_window` turns it
//! into an `AssembledLayout`, `layout::WindowGeometry::new` adds sizes, coordinates and routes,
//! and `RoutedWindowGraph::from_window_geometry` keys the result by world position. Nothing in
//! a `WindowGraph` depends on zoom, so `LayoutEngine` builds and caches it once per batch.

use std::{collections::HashSet, fmt};

use petgraph::stable_graph::{NodeIndex, StableDiGraph};

use crate::{
    edge_ports::NodeSide,
    layout::{PinSide, WindowStructure},
};

/// A window's layout graph carries the original domain nodes, the pin nodes that give an edge a
/// bend or carry it around the nodes it attaches to, the hairpin nodes that turn a self edge around,
/// and the wormhole nodes standing in for off-window neighbors. All four kinds are referenced by
/// their `NodeIndex` during the actual Layout phase.
#[derive(Clone, Debug)]
pub enum WindowNode {
    Data(NodeIndex),
    /// Synthetic node that bends an edge. A left or right pin carries a loop around the nodes it
    /// attaches to (a backward edge, or a west-to-west, east-to-east or west-to-east edge); an
    /// up or down pin separates two otherwise overlapping edges. Carries no domain data; sized
    /// and rendered like an ordinary routing node.
    Pin(PinSide),
    /// The turnaround beside one side of the node whose edge leaves that side and returns to it.
    /// Drawn as a glyph of its own, unlike the other synthetic nodes.
    HairPin(NodeSide),
    /// A door standing in for a windowed node's off-window neighbors on one side
    /// (see `crawl::WormholeDoor`). Carries the chosen off-window navigation target;
    /// the full collapsed neighbor list is carried on the edge(s) to this node instead
    /// (see `build_window_graph`), matching how ordinary parallel domain edges bundle.
    Wormhole(NodeIndex),
}

/// Represents an edge in a window's layout graph.
/// Contains the original source and target node indices from the domain graph.
/// None represents edges that don't correspond to original domain edges.
pub type WindowEdge = Option<(NodeIndex, NodeIndex)>;

/// A single crawled window's directed layout graph and its cached LOD-invariant Sugiyama
/// structure. Geometry (coordinates, routing, compaction) is not resident here: `LayoutEngine`
/// assembles this once per batch and caches the result, and each `WindowScene` sizes and routes
/// that assembly again whenever zoom, node sizes or the layout mode change.
#[derive(Clone)]
pub struct WindowGraph {
    pub graph: StableDiGraph<WindowNode, WindowEdge, u32>,
    /// Level-of-detail-independent Sugiyama layering for this window (layers plus
    /// within-layer order). Computed once when the window is first laid out. `None` for
    /// empty or single-node windows, which have no layering.
    pub structure: Option<WindowStructure>,
    /// Edges that span from a `left_pin` to a `right_pin` - the middle leg of an edge that
    /// bends around the end of the layout, the other legs being short excursions to and from
    /// the real endpoints (see `build_window_graph`). That leg runs against the layering for a
    /// backward edge but follows it for a forward west-to-east edge or a self edge. Keyed by
    /// the `(from, to)` node pair exactly as passed to `graph.add_edge`. Set at window-build
    /// time; empty for a window without such bends.
    /// `WindowStructure` resolves this once per edge into `pin_span_edges` (keyed by
    /// post-dummy-insertion vertex pairs) when the window's structure is built, since dummy
    /// insertion can split one of these edges into a multi-hop chain.
    pub pin_span_edges: HashSet<(NodeIndex<u32>, NodeIndex<u32>)>,
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
            pin_span_edges: HashSet::new(),
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
