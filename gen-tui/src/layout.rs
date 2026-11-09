use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
};

use petgraph::{graph::NodeIndex, stable_graph::StableDiGraph, visit::EdgeRef};
use rust_sugiyama::{LayoutVertex, configure::Config, from_edges_with_dummies};
use serde::{Deserialize, Serialize};

use crate::{
    edge_ports::NodeSide,
    geometry::{LocalPos, WorldPos, WorldRect},
    layout_compaction::base_step,
    window_graph::{WindowEdge, WindowGraph, WindowNode},
};
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VisualDetail {
    Minimal,
    Full,
    Truncated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(crate) struct JunctionSymbol {
    pub index: u8,
}

impl JunctionSymbol {
    /// Dashed box-drawing characters for routing nodes
    /// Index: 4-bit value where bits represent [North, East, South, West]
    const DASHED_ROUTING_GLYPHS: [char; 16] = [
        '?', // 0000 ____
        '╴', // 0001 ___W
        '╷', // 0010 __S_
        '╮', // 0011 __SW
        '╶', // 0100 _E__
        '┄', // 0101 _E_W (light triple dash horizontal)
        '╭', // 0110 _ES_
        '┬', // 0111 _ESW
        '╵', // 1000 N___
        '╯', // 1001 N__W
        '┆', // 1010 N_S_ (light triple dash vertical)
        '┤', // 1011 N_SW
        '╰', // 1100 NE__
        '┴', // 1101 NE_W
        '├', // 1110 NES_
        '┼', // 1111 NESW
    ];
    /// Heavy box-drawing characters for highlighted routing nodes
    /// Index: 4-bit value where bits represent [North, East, South, West]
    const HEAVY_ROUTING_GLYPHS: [char; 16] = [
        '?', // 0000 ____
        '╸', // 0001 ___W
        '╻', // 0010 __S_
        '┓', // 0011 __SW
        '╺', // 0100 _E__
        '━', // 0105 _E_W
        '┏', // 0110 _ES_
        '┳', // 0111 _ESW
        '╹', // 1000 N___
        '┛', // 1001 N__W
        '┃', // 1010 N_S_
        '┫', // 1011 N_SW
        '┗', // 1100 NE__
        '┻', // 1101 NE_W
        '┣', // 1110 NES_
        '╋', // 1111 NESW
    ];
    /// Box-drawing characters for routing nodes based on connection directions
    /// Index: 4-bit value where bits represent [North, East, South, West]
    const ROUTING_GLYPHS: [char; 16] = [
        '?', // 0000 ____
        '╴', // 0001 ___W
        '╷', // 0010 __S_
        '╮', // 0011 __SW
        '╶', // 0100 _E__
        '─', // 0101 _E_W
        '╭', // 0110 _ES_
        '┬', // 0111 _ESW
        '╵', // 1000 N___
        '╯', // 1001 N__W
        '│', // 1010 N_S_
        '┤', // 1011 N_SW
        '╰', // 1100 NE__
        '┴', // 1101 NE_W
        '├', // 1110 NES_
        '┼', // 1111 NESW
    ];

    /// Get the routing glyph character for this index
    pub fn glyph(&self) -> char {
        Self::ROUTING_GLYPHS
            .get(self.index as usize)
            .copied()
            .unwrap_or('?')
    }

    /// Get the heavy routing glyph character for this index
    pub(crate) fn heavy_glyph(&self) -> char {
        Self::HEAVY_ROUTING_GLYPHS
            .get(self.index as usize)
            .copied()
            .unwrap_or('?')
    }

    /// Get the dashed routing glyph character for this index
    pub(crate) fn dashed_glyph(&self) -> char {
        Self::DASHED_ROUTING_GLYPHS
            .get(self.index as usize)
            .copied()
            .unwrap_or('?')
    }

    /// Create a new GlyphIndex from a u8 value
    pub fn new(value: u8) -> Self {
        Self { index: value }
    }
}

/// The cells of a hairpin node: wide enough for the turnaround and its two arrows, and two rows tall.
pub const HAIRPIN_NODE_SIZE: (u64, u64) = (3, 2);

/// NodeRole represents the distinction between nodes that were in the input graph (Data)  
/// and dummy nodes that were added to route the edges during layout (Routing).
/// The Data variant stores the original domain graph NodeIndex.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NodeRole {
    Data(NodeIndex),
    Routing, // No stored data - glyph computed on-the-fly from connectivity
    /// Synthetic endpoint of a detour that carries an edge around the nodes it attaches to; the
    /// side says which end of the layout the detour runs along.
    Pin(PinSide),
    /// The turnaround beside one side of a node whose edge leaves that side and returns to it.
    /// Unlike the other synthetic nodes it is drawn as a glyph of its own, of
    /// [`HAIRPIN_NODE_SIZE`], with its top row on the row its bus meets.
    HairPin(NodeSide),
    /// Layered boundary door whose payload is its off-window navigation target.
    Wormhole(NodeIndex),
}

/// Where a pin sits relative to the nodes it serves. A left pin is reached from the west side of
/// its nodes and a right pin from the east side; both carry a loop around the nodes. An up or
/// down pin sits between two nodes and gives one of two otherwise overlapping edges a bend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PinSide {
    Left,
    Right,
    Up,
    Down,
}

impl PinSide {
    /// Whether the pin turns an edge around the end of the layout, as opposed to bending it
    /// between two neighboring layers.
    pub(crate) const fn is_loopback(self) -> bool {
        matches!(self, Self::Left | Self::Right)
    }
}

/// The row used for vertical alignment and edge attachment. World Y increases upward.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerticalAnchor {
    /// Center the node around its connection row (the historical layout).
    #[default]
    Center,
    /// Place the top row on the connection row, with all remaining rows below it.
    Top,
}

impl VerticalAnchor {
    /// Offset from the connection row to the discrete rectangle center.
    pub(crate) fn center_offset(self, height: u64) -> i64 {
        match self {
            Self::Center => 0,
            Self::Top => -(height as i64 / 2),
        }
    }

    /// The occupied rectangle around an attachment point; X remains centered.
    pub fn rect(self, position: WorldPos, size: (u64, u64)) -> WorldRect {
        WorldRect::from_center_and_size(
            WorldPos::new(position.x, position.y + self.center_offset(size.1)),
            size,
        )
    }
}

/// Layout graphs contain two types of nodes: nodes that represent the input nodes,
/// and new nodes that were added to route the edges during layout. The role field
/// with NodeRole enum indicate which type the node is.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutNode {
    pub role: NodeRole,
    /// Edge attachment point: horizontal center and the selected vertical anchor.
    pub pos: LocalPos,
    /// The row positioned at `pos.y`; remaining rows follow this anchor.
    #[serde(default)]
    pub vertical_anchor: VerticalAnchor,
    pub size: (u64, u64),
    /// The Sugiyama layer column this node was assembled into. Every node assembled from a
    /// window's structure has one - data, routing, pin, hairpin node and wormhole nodes alike.
    /// Junctions reset it to `None`, nodes added later (hairpin node expansions) never have one, and
    /// compaction re-snaps the layer of Data nodes to the nearest anchor column.
    pub layer: Option<i32>,
    /// Fixed horizontal offset of a curved side port from its layer's center column.
    #[serde(default)]
    pub(crate) side_port_offset: i64,
    /// Data node whose row and column this side port follows during compaction.
    #[serde(default)]
    pub(crate) side_port_anchor: Option<NodeIndex<u32>>,
    /// Sides with curved self-edges, used to reserve room beyond the visible node border.
    #[serde(default)]
    pub(crate) hairpin_west: bool,
    #[serde(default)]
    pub(crate) hairpin_east: bool,
    /// A node this one stays a fixed number of rows from during vertical compaction, and that
    /// number of rows (negative above it).
    #[serde(default)]
    pub(crate) row_tether: Option<(NodeIndex<u32>, i64)>,
}

impl LayoutNode {
    /// The occupied rectangle, translated by a world offset.
    pub fn rect(&self, offset: WorldPos) -> WorldRect {
        self.vertical_anchor.rect(
            WorldPos::new(self.pos.x + offset.x, self.pos.y + offset.y),
            self.size,
        )
    }

    /// Minimum distance to the next node in increasing world Y, before any blank rows.
    pub(crate) fn vertical_separation(&self, next: &Self) -> i64 {
        base_step(self.size.1 as i64, next.size.1 as i64)
            + self.vertical_anchor.center_offset(self.size.1)
            - next.vertical_anchor.center_offset(next.size.1)
    }

    pub fn new(role: NodeRole, pos: LocalPos, size: (u64, u64), layer: Option<i32>) -> Self {
        Self {
            role,
            vertical_anchor: VerticalAnchor::Center,
            pos,
            size,
            layer,
            side_port_offset: 0,
            side_port_anchor: None,
            hairpin_west: false,
            hairpin_east: false,
            row_tether: None,
        }
    }

    pub fn data(
        domain_node_idx: NodeIndex,
        pos: LocalPos,
        size: (u64, u64),
        layer: Option<i32>,
    ) -> Self {
        Self::new(NodeRole::Data(domain_node_idx), pos, size, layer)
    }

    pub fn routing(pos: LocalPos, size: (u64, u64)) -> Self {
        Self::new(NodeRole::Routing, pos, size, None)
    }
}

/// LayoutEdge represents a bundle of edges as a vector of node index pairs.
/// During layout, multiple edges may be bundled together for simplified visualization
/// and routing purposes.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LayoutEdge {
    pub bundle: Vec<(NodeIndex, NodeIndex)>,
    /// Whether this edge spans from a left pin to a right pin, the leg of a bend that goes around
    /// the end of the layout. That is the reversed main span of a backward edge, and also the
    /// span of a forward west-to-east edge or a self edge.
    pub(crate) is_pin_span: bool,
}

impl LayoutEdge {
    pub fn new(source: NodeIndex, target: NodeIndex) -> Self {
        Self {
            bundle: vec![(source, target)],
            is_pin_span: false,
        }
    }

    pub fn empty() -> Self {
        Self {
            bundle: Vec::new(),
            is_pin_span: false,
        }
    }
}

impl Default for LayoutEdge {
    fn default() -> Self {
        Self::empty()
    }
}

/// Which window-graph node this vertex stands for. `None` marks a routing dummy: every dummy
/// vertex here comes straight from `rust_sugiyama::from_edges_with_dummies`'s own
/// crossing-minimized placement, not something gen-tui synthesizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct StructureVertex {
    pub(crate) input_node_idx: Option<NodeIndex<u32>>,
}

/// Which window-graph edge (if any) this hop belongs to, so a dummy chain traces back to the
/// domain edge it routes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StructureEdge {
    pub(crate) input_node_idx_pair: Option<(NodeIndex<u32>, NodeIndex<u32>)>,
}

/// Cached, size-independent Sugiyama layers and routing vertices for one window.
#[derive(Debug, Clone)]
pub struct WindowStructure {
    /// Post-layering vertex graph. Dummy routing vertices are present; each data vertex
    /// carries the window-graph node index it stands for in `StructureVertex::input_node_idx`.
    pub(crate) vertex_graph: StableDiGraph<StructureVertex, StructureEdge, u32>,
    /// Per-layer vertex ordering from crossing minimization. The outer index is the layer;
    /// the inner order is the within-layer ordering. Entries index `vertex_graph`.
    pub(crate) vertex_layers: Vec<Vec<NodeIndex<u32>>>,
    /// Which `vertex_graph` edges render the `left_pin -> right_pin` span of a bend around the
    /// end of the layout (see `WindowGraph::pin_span_edges`), keyed by each edge's `(source, target)`
    /// endpoints. Resolved once in `WindowStructureBuilder::build_structure` by walking
    /// `WindowGraph::pin_span_edges` through any dummy-vertex chain Sugiyama inserted for
    /// that edge, so `assemble_window` can carry the flag straight into `LayoutEdge` without
    /// re-deriving it later. Empty for a graph without such bends.
    pub(crate) pin_span_edges: HashSet<(NodeIndex<u32>, NodeIndex<u32>)>,
}

impl WindowStructure {
    /// Locate the window-graph node `window_node` in the layered structure, returning its
    /// `(layer index, within-layer order)`. Both are indices into `vertex_layers`. Returns
    /// `None` when no vertex carries that window-graph node (for example a dummy routing
    /// vertex, which has no `input_node_idx`).
    pub fn locate(&self, window_node: NodeIndex<u32>) -> Option<(usize, usize)> {
        self.vertex_layers
            .iter()
            .enumerate()
            .find_map(|(layer_index, layer)| {
                layer
                    .iter()
                    .position(|&vertex_idx| {
                        self.vertex_graph
                            .node_weight(vertex_idx)
                            .and_then(|vertex| vertex.input_node_idx)
                            == Some(window_node)
                    })
                    .map(|order| (layer_index, order))
            })
    }
}

/// A builder for the size-independent Sugiyama structure of a crawled window's layout graph.
///
/// The cached layer framework is reused across zoom levels so the relative orientation of nodes
/// remains stable while `WindowGeometry::new` computes window geometry at the current detail.
#[derive(Debug)]
pub(crate) struct WindowStructureBuilder<'a> {
    window_graph: &'a StableDiGraph<WindowNode, WindowEdge, u32>,
    /// Pairs of data nodes that Sugiyama must layer in order. See `with_ordering_edges`.
    ordering_edges: Vec<(NodeIndex<u32>, NodeIndex<u32>)>,
    vertex_graph: StableDiGraph<StructureVertex, StructureEdge, u32>,
    vertex_layers: Option<Vec<Vec<NodeIndex<u32>>>>,
    config: Config,
    /// Copied from `WindowGraph::pin_span_edges` at construction. See
    /// `WindowStructure::pin_span_edges`.
    pin_span_edges: HashSet<(NodeIndex<u32>, NodeIndex<u32>)>,
    /// Resolved by `build_structure` once dummy vertices are inserted. See
    /// `WindowStructure::pin_span_edges`.
    resolved_pin_span_edges: HashSet<(NodeIndex<u32>, NodeIndex<u32>)>,
}

/// One slot in a recovered layer: a real window-graph node, or a routing dummy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum LayerEntry {
    Real(NodeIndex<u32>),
    Dummy(u32),
}

/// Per-layer slot ordering: outer index is the layer, inner is `(layer entry, x)` sorted by x.
type LayerSlots = Vec<Vec<(LayerEntry, f64)>>;
/// Each multi-layer window-graph edge's routing-dummy chain, as returned by
/// `from_edges_with_dummies`, keyed by the edge's position among the window graph's edges so that
/// parallel edges between the same two nodes keep chains of their own.
type DummyChains = HashMap<usize, Vec<u32>>;
/// One dense edge fed to `from_edges_with_dummies`, paired with the window-graph endpoints it
/// came from so a returned `LayoutVertex::Dummy(edge_id)` can be traced back to them.
type DenseEdgeWithEndpoints = ((u32, u32), (NodeIndex<u32>, NodeIndex<u32>));
/// Raw per-vertex output from `from_edges_with_dummies`, grouped by layer before dummy ids are
/// minted (see `WindowStructureBuilder::build_layers`).
type RawLayerGroups = HashMap<u64, Vec<(LayoutVertex<usize, usize>, f64)>>;

impl<'a> WindowStructureBuilder<'a> {
    /// Create a new WindowStructureBuilder for the given window graph.
    pub fn new(window: &'a WindowGraph) -> Self {
        Self {
            window_graph: &window.graph,
            ordering_edges: Vec::new(),
            vertex_graph: StableDiGraph::new(),
            vertex_layers: None,
            config: Config::default(),
            pin_span_edges: window.pin_span_edges.clone(),
            resolved_pin_span_edges: HashSet::new(),
        }
    }

    /// Supply pairs of data nodes whose layers must stay in the given order while Sugiyama runs.
    /// Edges bent by pins use them to keep their endpoints' layers fixed; the pairs add no edge to the
    /// structure, and the dummies Sugiyama routes along them are dropped from the layers.
    pub(crate) fn with_ordering_edges(
        mut self,
        ordering_edges: Vec<(NodeIndex<u32>, NodeIndex<u32>)>,
    ) -> Self {
        self.ordering_edges = ordering_edges;
        self
    }

    /// Establish the detail-independent structure of the window: validate the graph, then layer
    /// and order every vertex - real and dummy alike - via
    /// `rust_sugiyama::from_edges_with_dummies` (see `StructureVertex`). A window with a backward-edge
    /// bypass loop is mirrored to a consistent orientation in `build_layers`; every other window
    /// keeps the crate's own placement untouched. Idempotent; a single-node window needs no
    /// layering.
    pub(crate) fn build_structure(&mut self) -> Result<(), String> {
        if self.window_graph.node_count() == 0 {
            return Err("Cannot compute layout for empty graph".to_string());
        }

        // Single node graph: bypasses Sugiyama and edge routing entirely, so there is no
        // structure to build.
        if self.window_graph.node_count() == 1 {
            return Ok(());
        }

        // Sugiyama layering requires a connected graph.
        let components = count_connected_components(self.window_graph);
        if components > 1 {
            return Err(format!(
                "Graph is disconnected ({} components). Layout requires a connected graph.",
                components
            ));
        }

        if self.vertex_layers.is_some() {
            return Ok(());
        }

        // Dense id map: window-graph `NodeIndex` -> contiguous u32 id fed to `from_edges`.
        // Dummy nodes get fresh ids past the end of this range (see below).
        let window_ids: Vec<NodeIndex<u32>> = self.window_graph.node_indices().collect();
        let dense_id: HashMap<NodeIndex<u32>, u32> = window_ids
            .iter()
            .enumerate()
            .map(|(dense, &window_idx)| (window_idx, dense as u32))
            .collect();

        let (layers_full, dummy_chains): (LayerSlots, DummyChains) =
            self.build_layers(&window_ids, &dense_id);

        // Build `vertex_graph`: one vertex per real/dummy layer entry, then wire each
        // window-graph edge through its dummy chain (if any).
        let mut vertex_graph = StableDiGraph::<StructureVertex, StructureEdge, u32>::new();
        let mut window_to_vertex: HashMap<NodeIndex<u32>, NodeIndex<u32>> = HashMap::new();
        let mut dummy_to_vertex: HashMap<u32, NodeIndex<u32>> = HashMap::new();
        let mut vertex_layers: Vec<Vec<NodeIndex<u32>>> = Vec::with_capacity(layers_full.len());
        for layer in &layers_full {
            let mut vertex_layer = Vec::with_capacity(layer.len());
            for &(entry, _) in layer {
                let vertex_idx = match entry {
                    LayerEntry::Real(window_idx) => {
                        let v = vertex_graph.add_node(StructureVertex {
                            input_node_idx: Some(window_idx),
                        });
                        window_to_vertex.insert(window_idx, v);
                        v
                    }
                    LayerEntry::Dummy(dummy_id) => {
                        let v = vertex_graph.add_node(StructureVertex {
                            input_node_idx: None,
                        });
                        dummy_to_vertex.insert(dummy_id, v);
                        v
                    }
                };
                vertex_layer.push(vertex_idx);
            }
            vertex_layers.push(vertex_layer);
        }

        for (edge_position, edge_idx) in self.window_graph.edge_indices().enumerate() {
            let Some((tail_w, head_w)) = self.window_graph.edge_endpoints(edge_idx) else {
                continue;
            };
            let window_edge: WindowEdge = *self
                .window_graph
                .edge_weight(edge_idx)
                .expect("should find an edge weight");

            let mut tail_vertex = window_to_vertex[&tail_w];
            if let Some(chain) = dummy_chains.get(&edge_position) {
                for &dummy_id in chain {
                    let dummy_vertex = dummy_to_vertex[&dummy_id];
                    vertex_graph.add_edge(
                        tail_vertex,
                        dummy_vertex,
                        StructureEdge {
                            input_node_idx_pair: window_edge,
                        },
                    );
                    tail_vertex = dummy_vertex;
                }
            }
            let head_vertex = window_to_vertex[&head_w];
            vertex_graph.add_edge(
                tail_vertex,
                head_vertex,
                StructureEdge {
                    input_node_idx_pair: window_edge,
                },
            );
        }

        self.vertex_graph = vertex_graph;
        self.vertex_layers = Some(vertex_layers);
        self.resolve_pin_span_edges(&window_to_vertex);

        Ok(())
    }

    /// Call `from_edges_with_dummies` once, which layers and orders every vertex - real and
    /// dummy alike - via full-graph crossing minimization. Every multi-layer edge's dummy chain,
    /// backward-edge bypass loops included, comes straight from that single call; nothing is
    /// reconstructed or moved. The ordering edges (see `with_ordering_edges`) take part in the
    /// layering and crossing minimization, then their dummies are dropped, so they leave no trace
    /// in the layers. The only other override is a whole-window mirror: crossing counts are
    /// invariant under reversing every layer's order together (it's a reflection, not a
    /// reordering relative to anything else), so if a window's bypass loop came out bulging the
    /// "wrong" way, every layer gets reversed once to make it consistent - see the mirror step at
    /// the end of this function.
    fn build_layers(
        &self,
        window_ids: &[NodeIndex<u32>],
        dense_id: &HashMap<NodeIndex<u32>, u32>,
    ) -> (LayerSlots, DummyChains) {
        // Build the dense edge list `from_edges_with_dummies` needs alongside a parallel list of
        // window-graph endpoints, in lockstep, so `edge_endpoints[edge_id]` always names the
        // input edge a returned `LayoutVertex::Dummy(edge_id)` subdivides.
        let dense_edges_with_endpoints: Vec<DenseEdgeWithEndpoints> = self
            .window_graph
            .edge_indices()
            .filter_map(|edge_idx| self.window_graph.edge_endpoints(edge_idx))
            .map(|(tail_w, head_w)| ((dense_id[&tail_w], dense_id[&head_w]), (tail_w, head_w)))
            .collect();
        let mut dense_edges: Vec<(u32, u32)> =
            dense_edges_with_endpoints.iter().map(|&(e, _)| e).collect();
        let real_edges: HashSet<_> = dense_edges.iter().copied().collect();
        let mut edge_endpoints: Vec<(NodeIndex<u32>, NodeIndex<u32>)> =
            dense_edges_with_endpoints.iter().map(|&(_, w)| w).collect();
        // A pair already joined by a real edge is ordered by that edge and needs no extra one.
        let first_ordering_edge = dense_edges.len();
        let mut added_ordering_edges = Vec::new();
        for &(earlier, later) in &self.ordering_edges {
            let pair = (dense_id[&earlier], dense_id[&later]);
            if !real_edges.contains(&pair) && !added_ordering_edges.contains(&(earlier, later)) {
                dense_edges.push(pair);
                edge_endpoints.push((earlier, later));
                added_ordering_edges.push((earlier, later));
            }
        }

        let mut layouts = from_edges_with_dummies(&dense_edges, &self.config);
        let (positions, _width, _height) = layouts
            .pop()
            .expect("a connected, non-empty graph produces exactly one subgraph layout");
        debug_assert!(
            layouts.is_empty(),
            "window graph connectivity is validated above"
        );

        // Group by layer (y), keeping each vertex's raw `LayoutVertex` for now - the crate's
        // output order has no relationship to layer order, so dummy ids can only be minted once
        // layers are sorted low to high below; minting them here would scramble each edge's
        // chain into an arbitrary order instead of tail-to-head.
        let mut groups: RawLayerGroups = HashMap::new();
        for (vertex, (x, y)) in positions {
            groups.entry(y.to_bits()).or_default().push((vertex, x));
        }
        let mut layer_ys: Vec<f64> = groups.keys().map(|&bits| f64::from_bits(bits)).collect();
        layer_ys.sort_by(f64::total_cmp);

        // Now walk layers low to high, sorting each by x, minting a fresh globally-unique id for
        // each dummy vertex and appending it to its edge's chain accumulator - both in true layer
        // order, since that's the order this loop visits them in.
        let mut layers_full: LayerSlots = Vec::with_capacity(layer_ys.len());
        let mut chains_by_edge: HashMap<usize, Vec<u32>> = HashMap::new();
        let mut next_id: u32 = 0;
        for y in &layer_ys {
            let mut layer = groups.remove(&y.to_bits()).unwrap();
            layer.sort_by(|a, b| a.1.total_cmp(&b.1));
            let layer: Vec<(LayerEntry, f64)> = layer
                .into_iter()
                .filter_map(|(vertex, x)| {
                    let entry = match vertex {
                        LayoutVertex::Node(dense) => LayerEntry::Real(window_ids[dense]),
                        LayoutVertex::Dummy(edge_id) if edge_id >= first_ordering_edge => {
                            return None;
                        }
                        LayoutVertex::Dummy(edge_id) => {
                            let id = next_id;
                            next_id += 1;
                            chains_by_edge.entry(edge_id).or_default().push(id);
                            LayerEntry::Dummy(id)
                        }
                    };
                    Some((entry, x))
                })
                .collect();
            if !layer.is_empty() {
                layers_full.push(layer);
            }
        }

        let dummy_chains: DummyChains = chains_by_edge.into_iter().collect();

        // Mirror the whole window if it has a bypass loop. `from_edges_with_dummies` reliably
        // places dummy vertices - added to the graph after every real node, so tie-breaking
        // consistently favors their higher index - at the far extreme of their layer, opposite
        // this codebase's below-the-graph convention. Reversing every layer's order the once is a
        // pure reflection: it changes no relative order and so no crossing count, and always
        // corrects that bias in a single pass rather than checking for it per window.
        //
        // Scoped to windows with a backward-edge bypass, not every window: an unconditional flip
        // was tried and produced 6 non-rectilinear edges in `subsetting_grid`'s partial-budget
        // middle-anchor case (multiple boundary doors collapsing to the same off-window target).
        // `order` isn't just bookkeeping - `assemble_window` writes it straight into
        // `LayoutNode.pos` as a real row coordinate - and `edge_router::route_layer`'s
        // `route_layer` is not a pure function of that row order: it detects "backtracking" via
        // a Y-range-vs-envelope heuristic and conditionally re-routes with reversed vertical
        // order, tuned against whatever arrangement `build_layers` handed it before. Flipping
        // every window changes which windows trigger that fallback and how, so a wider flip
        // needs `route_layer` made robust to it first, not just more flipping here.
        if !self.pin_span_edges.is_empty() {
            for layer in &mut layers_full {
                layer.reverse();
            }
        }
        (layers_full, dummy_chains)
    }

    /// Resolve `pin_span_edges` (window-graph node indices, set at window-build time)
    /// into `resolved_pin_span_edges` (`vertex_graph` edge endpoints), walking through
    /// any dummy-vertex chain Sugiyama inserted for that edge.
    fn resolve_pin_span_edges(
        &mut self,
        window_to_vertex: &HashMap<NodeIndex<u32>, NodeIndex<u32>>,
    ) {
        for &(from_window_idx, to_window_idx) in &self.pin_span_edges {
            let (Some(&from_vertex), Some(&to_vertex)) = (
                window_to_vertex.get(&from_window_idx),
                window_to_vertex.get(&to_window_idx),
            ) else {
                continue;
            };
            let Some(bundle) = self
                .window_graph
                .find_edge(from_window_idx, to_window_idx)
                .and_then(|edge_idx| *self.window_graph.edge_weight(edge_idx).unwrap())
            else {
                continue;
            };
            for hop in trace_dummy_chain(&self.vertex_graph, from_vertex, to_vertex, bundle) {
                self.resolved_pin_span_edges.insert(hop);
            }
        }
    }

    /// Clone out the size-independent structure established by `build_structure`, or `None`
    /// if no layering was built (an empty or single-node window). The caller caches this
    /// on the window so later detail switches can reuse the layering without re-running
    /// Sugiyama.
    pub fn structure(&self) -> Option<WindowStructure> {
        self.vertex_layers
            .as_ref()
            .map(|vertex_layers| WindowStructure {
                vertex_graph: self.vertex_graph.clone(),
                vertex_layers: vertex_layers.clone(),
                pin_span_edges: self.resolved_pin_span_edges.clone(),
            })
    }
}

/// Walk from `from` to `to` in `vertex_graph`, following edges whose `input_node_idx_pair`
/// matches `bundle`, to find the (possibly dummy-expanded) chain Sugiyama inserted for the
/// single window-graph edge `(from, to)` originally represented. Returns every hop's
/// `(source, target)` endpoints in walk order, or empty if no matching chain reaches `to`.
///
/// A direct edge `(from, to)` can coexist with unrelated dummy chains also leaving `from`
/// for other edges sharing the same domain bundle - e.g. `left_pin` has two outgoing edges
/// (to `right_pin`, and to the real target) sharing one bundle, so the walk cannot just
/// follow the first matching edge out of `from`; it must confirm the chain actually
/// terminates at `to`. Past the first hop it is unambiguous: a dummy vertex inserted for one
/// edge belongs to no other chain, so it has exactly one qualifying successor.
fn trace_dummy_chain(
    vertex_graph: &StableDiGraph<StructureVertex, StructureEdge, u32>,
    from: NodeIndex<u32>,
    to: NodeIndex<u32>,
    bundle: (NodeIndex, NodeIndex),
) -> Vec<(NodeIndex<u32>, NodeIndex<u32>)> {
    for first_hop in vertex_graph.edges(from) {
        if first_hop.weight().input_node_idx_pair != Some(bundle) {
            continue;
        }
        let mut chain = vec![(from, first_hop.target())];
        let mut current = first_hop.target();
        while current != to {
            let Some(next) = vertex_graph
                .edges(current)
                .find(|edge_ref| edge_ref.weight().input_node_idx_pair == Some(bundle))
            else {
                break;
            };
            chain.push((current, next.target()));
            current = next.target();
        }
        if current == to {
            return chain;
        }
    }
    Vec::new()
}

/// Helper to count connected components in an undirected sense for a StableGraph
fn count_connected_components<N, E>(graph: &StableDiGraph<N, E, u32>) -> usize {
    let mut visited = std::collections::HashSet::new();
    let mut components = 0;

    for node in graph.node_indices() {
        if !visited.contains(&node) {
            components += 1;
            // BFS traversal
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(node);
            visited.insert(node);

            while let Some(current) = queue.pop_front() {
                for neighbor in graph.neighbors_undirected(current) {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        queue.push_back(neighbor);
                    }
                }
            }
        }
    }
    components
}
