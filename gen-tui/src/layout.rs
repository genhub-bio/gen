use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
};

use itertools::Itertools;
use petgraph::{
    Undirected,
    graph::NodeIndex,
    stable_graph::{StableDiGraph, StableGraph},
    visit::EdgeRef,
};
use rust_sugiyama::{LayoutVertex, configure::Config, from_edges_with_dummies};
use serde::{Deserialize, Serialize};

use crate::{
    cross_coordinates::assign_cross_coordinates,
    distribute_nodes::{GapSizes, compact_layout},
    edge_router::{layout_graph_process::prune_pin_stubs, route_graph::make_rectilinear},
    geometry::LocalPos,
    window_graph::{WindowEdge, WindowGraph, WindowNode},
};
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VisualDetail {
    Minimal,
    Full,
    Truncated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct JunctionSymbol {
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
    pub fn heavy_glyph(&self) -> char {
        Self::HEAVY_ROUTING_GLYPHS
            .get(self.index as usize)
            .copied()
            .unwrap_or('?')
    }

    /// Get the dashed routing glyph character for this index
    pub fn dashed_glyph(&self) -> char {
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

/// NodeRole represents the distinction between nodes that were in the input graph (Data)  
/// and dummy nodes that were added to route the edges during layout (Routing).
/// The Data variant stores the original domain graph NodeIndex.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NodeRole {
    Data(NodeIndex),
    Routing, // No stored data - glyph computed on-the-fly from connectivity
    /// Synthetic endpoint used while routing a backward-edge loop.
    Pin,
    /// Ranked boundary door whose payload is its off-window navigation target.
    Wormhole(NodeIndex),
}

/// Layout graphs contain two types of nodes: nodes that represent the input nodes,
/// and new nodes that were added to route the edges during layout. The role field
/// with NodeRole enum indicate which type the node is.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutNode {
    pub role: NodeRole,
    pub pos: LocalPos,
    pub size: (u64, u64),
    /// Layer information from Sugiyama algorithm. Only valid for Data nodes.
    pub layer: Option<i32>,
}

impl LayoutNode {
    pub fn new(role: NodeRole, pos: LocalPos, size: (u64, u64), layer: Option<i32>) -> Self {
        Self {
            role,
            pos,
            size,
            layer,
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
    /// Whether this is the reversed main span of a backward-edge loop.
    pub is_backward_span: bool,
}

impl LayoutEdge {
    pub fn new(source: NodeIndex, target: NodeIndex) -> Self {
        Self {
            bundle: vec![(source, target)],
            is_backward_span: false,
        }
    }

    pub fn empty() -> Self {
        Self {
            bundle: Vec::new(),
            is_backward_span: false,
        }
    }
}

impl Default for LayoutEdge {
    fn default() -> Self {
        Self::empty()
    }
}

/// Final geometry for one assembled neighbourhood window at a specific zoom level.
///
/// The graph is undirected to allow bidirectional edge routing, with final coordinates mapped
/// to its nodes.
#[derive(Clone, Serialize)]
pub struct WindowGeometry {
    pub graph: StableGraph<LayoutNode, LayoutEdge, Undirected, u32>,
    pub width: i64,
    pub height: i64,
    /// Each invisible domain node and the routing node it became (see [`Self::new`]).
    pub(crate) junctions: Vec<(NodeIndex, NodeIndex<u32>)>,
}

impl WindowGeometry {
    /// Route and compact an assembled window whose nodes already have their rendered sizes.
    ///
    /// The Data nodes for `junctions` (domain nodes the renderer doesn't draw) are laid out and
    /// routed like any other node, so every edge still meets at them and keeps its domain-edge
    /// bundle, then become routing nodes before compaction: they are drawn as the junction of
    /// their edges and are never placed as selectable nodes.
    pub fn new(
        mut layout_graph: StableGraph<LayoutNode, LayoutEdge, Undirected, u32>,
        gaps: &GapSizes,
        junctions: &HashSet<NodeIndex>,
    ) -> Self {
        // Assembly places rows at their size-independent within-layer ordinals. The edge router
        // reacts only to y, so assign real-size-aware cross coordinates before routing. This pass
        // only needs a small structural gap, not the caller's actual zoom target: compaction below
        // is the sole place either target gap is enforced.
        assign_cross_coordinates(&mut layout_graph, 1);

        if let Err(error) = make_rectilinear(&mut layout_graph) {
            log::warn!("Edge routing failed: {:?}", error);
        }

        // Routing has already merged straight runs and labelled every segment, so a junction
        // swapped only now keeps both the edges into it and the edges out of it.
        let mut junction_nodes = Vec::new();
        for layout_index in layout_graph.node_indices().collect::<Vec<_>>() {
            let node = &mut layout_graph[layout_index];
            if let NodeRole::Data(domain_index) = node.role
                && junctions.contains(&domain_index)
            {
                node.role = NodeRole::Routing;
                node.layer = None;
                junction_nodes.push((domain_index, layout_index));
            }
        }

        compact_layout(&mut layout_graph, gaps);

        prune_pin_stubs(&mut layout_graph);

        let (min_x, max_x) = layout_graph
            .node_weights()
            .map(|node| node.pos.x)
            .minmax()
            .into_option()
            .unwrap_or((0, 0));
        let (min_y, max_y) = layout_graph
            .node_weights()
            .map(|node| node.pos.y)
            .minmax()
            .into_option()
            .unwrap_or((0, 0));

        Self {
            graph: layout_graph,
            width: max_x - min_x,
            height: max_y - min_y,
            junctions: junction_nodes,
        }
    }

    /// The routing node an invisible domain node became, if it is in this window. A
    /// `prune_pin_stubs` splice can remove one lying on a backward-edge loop's stub, in which
    /// case there is none.
    pub fn junction_node(&self, domain_index: NodeIndex) -> Option<&LayoutNode> {
        self.junctions
            .iter()
            .find(|(junction, _)| *junction == domain_index)
            .and_then(|(_, layout_index)| self.graph.node_weight(*layout_index))
            .filter(|node| matches!(node.role, NodeRole::Routing))
    }

    /// The invisible domain nodes still present in this window as routing nodes.
    pub fn junctions(&self) -> impl Iterator<Item = (NodeIndex, &LayoutNode)> + '_ {
        self.junctions.iter().filter_map(|&(domain_index, _)| {
            self.junction_node(domain_index)
                .map(|node| (domain_index, node))
        })
    }

    /// Check if the layout graph is empty (no nodes)
    pub fn is_empty(&self) -> bool {
        self.graph.node_count() == 0
    }

    /// Check if the layout contains a specific node
    pub fn contains_key(&self, node_idx: &NodeIndex<u32>) -> bool {
        self.graph.node_weight(*node_idx).is_some()
    }

    /// Get the position of a node in the layout's coordinate system
    pub fn get_node_position(&self, node_idx: NodeIndex<u32>) -> Option<LocalPos> {
        self.graph.node_weight(node_idx).map(|node| node.pos)
    }

    /// Get the size of a node
    pub fn get_node_size(&self, node_idx: NodeIndex<u32>) -> Option<(u64, u64)> {
        self.graph.node_weight(node_idx).map(|node| node.size)
    }
}

impl std::fmt::Debug for WindowGeometry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug_struct = f.debug_struct("WindowGeometry");

        debug_struct
            .field("width", &self.width)
            .field("height", &self.height)
            .field("node_count", &self.graph.node_count())
            .field("edge_count", &self.graph.edge_count());

        // Add positions for all nodes
        let mut positions = Vec::new();
        for node_idx in self.graph.node_indices() {
            if let Some(node) = self.graph.node_weight(node_idx) {
                positions.push((node_idx, node.pos, &node.role));
            }
        }
        debug_struct.field("node_positions", &positions);

        // Add edge bundle information
        let mut edge_bundles = Vec::new();
        for edge_idx in self.graph.edge_indices() {
            if let Some(edge) = self.graph.edge_weight(edge_idx)
                && let Some((source, target)) = self.graph.edge_endpoints(edge_idx)
            {
                edge_bundles.push((source, target, &edge.bundle));
            }
        }
        debug_struct.field("edge_bundles", &edge_bundles);

        debug_struct.finish()
    }
}

/// Which window-graph node this vertex stands for. `None` marks a routing dummy: every dummy
/// vertex here comes straight from `rust_sugiyama::from_edges_with_dummies`'s own
/// crossing-minimized placement, not something gen-tui synthesizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Vertex {
    pub input_node_idx: Option<NodeIndex<u32>>,
}

/// Which window-graph edge (if any) this hop belongs to, so a dummy chain traces back to the
/// domain edge it routes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Edge {
    pub input_node_idx_pair: Option<(NodeIndex<u32>, NodeIndex<u32>)>,
}

/// Cached, size-independent Sugiyama layers and routing vertices for one window.
#[derive(Debug, Clone)]
pub struct WindowStructure {
    /// Post-layering vertex graph. Dummy routing vertices are present; each data vertex
    /// carries the window-graph node index it stands for in `Vertex::input_node_idx`.
    pub vertex_graph: StableDiGraph<Vertex, Edge, u32>,
    /// Per-layer vertex ordering from crossing minimization. The outer index is the layer;
    /// the inner order is the within-layer ordering. Entries index `vertex_graph`.
    pub vertex_layers: Vec<Vec<NodeIndex<u32>>>,
    /// Which `vertex_graph` edges render a backward edge's `left_pin -> right_pin` main span
    /// (see `WindowGraph::backward_span_edges`), keyed by each edge's `(source, target)`
    /// endpoints. Resolved once in `WindowStructureBuilder::build_structure` by walking
    /// `WindowGraph::backward_span_edges` through any dummy-vertex chain Sugiyama inserted for
    /// that edge, so `assemble_window` can carry the flag straight into `LayoutEdge` without
    /// re-deriving it later. Empty for a graph with no backward edges.
    pub backward_span_edges: HashSet<(NodeIndex<u32>, NodeIndex<u32>)>,
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
/// remains stable while the widget computes window geometry at the current detail.
#[derive(Debug)]
pub struct WindowStructureBuilder<'a> {
    window_graph: &'a StableDiGraph<WindowNode, WindowEdge, u32>,
    vertex_graph: StableDiGraph<Vertex, Edge, u32>,
    vertex_layers: Option<Vec<Vec<NodeIndex<u32>>>>,
    config: Config,
    /// Copied from `WindowGraph::backward_span_edges` at construction. See
    /// `WindowStructure::backward_span_edges`.
    backward_span_edges: HashSet<(NodeIndex<u32>, NodeIndex<u32>)>,
    /// Resolved by `build_structure` once dummy vertices are inserted. See
    /// `WindowStructure::backward_span_edges`.
    resolved_backward_span_edges: HashSet<(NodeIndex<u32>, NodeIndex<u32>)>,
}

/// One slot in a recovered layer: a real window-graph node, or a routing dummy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum LayerEntry {
    Real(NodeIndex<u32>),
    Dummy(u32),
}

/// Per-rank slot ordering: outer index is the rank, inner is `(layer entry, x)` sorted by x.
type RankedLayers = Vec<Vec<(LayerEntry, f64)>>;
/// Each multi-rank window-graph edge's routing-dummy chain, as returned by
/// `from_edges_with_dummies`, keyed by endpoints.
type DummyChains = HashMap<(NodeIndex<u32>, NodeIndex<u32>), Vec<u32>>;
/// One dense edge fed to `from_edges_with_dummies`, paired with the window-graph endpoints it
/// came from so a returned `LayoutVertex::Dummy(edge_id)` can be traced back to them.
type DenseEdgeWithEndpoints = ((u32, u32), (NodeIndex<u32>, NodeIndex<u32>));
/// Raw per-vertex output from `from_edges_with_dummies`, grouped by rank before dummy ids are
/// minted (see `WindowStructureBuilder::build_ranks`).
type RawRankGroups = HashMap<u64, Vec<(LayoutVertex<usize, usize>, f64)>>;

impl<'a> WindowStructureBuilder<'a> {
    /// Create a new WindowStructureBuilder for the given window graph.
    pub fn new(window: &'a WindowGraph) -> Self {
        Self {
            window_graph: &window.graph,
            vertex_graph: StableDiGraph::new(),
            vertex_layers: None,
            config: Config::default(),
            backward_span_edges: window.backward_span_edges.clone(),
            resolved_backward_span_edges: HashSet::new(),
        }
    }

    /// Establish the detail-independent structure of the window: validate the graph, then rank
    /// and order every vertex - real and dummy alike - via
    /// `rust_sugiyama::from_edges_with_dummies` (see `Vertex`). A window with a backward-edge
    /// bypass loop is mirrored to a consistent orientation in `build_ranks`; every other window
    /// keeps the crate's own placement untouched. Idempotent; a single-node window needs no
    /// layering.
    pub fn build_structure(&mut self) -> Result<(), String> {
        if self.window_graph.node_count() == 0 {
            return Err("Cannot compute layout for empty graph".to_string());
        }

        // Single node graph: bypasses Sugiyama and edge routing entirely, so there is no
        // structure to build.
        if self.window_graph.node_count() == 1 {
            return Ok(());
        }

        // Sugiyama ranking requires a connected graph.
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

        let (ranks_full, dummy_chains): (RankedLayers, DummyChains) =
            self.build_ranks(&window_ids, &dense_id);

        // Build `vertex_graph`: one vertex per real/dummy layer entry, then wire each
        // window-graph edge through its dummy chain (if any).
        let mut vertex_graph = StableDiGraph::<Vertex, Edge, u32>::new();
        let mut window_to_vertex: HashMap<NodeIndex<u32>, NodeIndex<u32>> = HashMap::new();
        let mut dummy_to_vertex: HashMap<u32, NodeIndex<u32>> = HashMap::new();
        let mut vertex_layers: Vec<Vec<NodeIndex<u32>>> = Vec::with_capacity(ranks_full.len());
        for layer in &ranks_full {
            let mut vertex_layer = Vec::with_capacity(layer.len());
            for &(entry, _) in layer {
                let vertex_idx = match entry {
                    LayerEntry::Real(window_idx) => {
                        let v = vertex_graph.add_node(Vertex {
                            input_node_idx: Some(window_idx),
                        });
                        window_to_vertex.insert(window_idx, v);
                        v
                    }
                    LayerEntry::Dummy(dummy_id) => {
                        let v = vertex_graph.add_node(Vertex {
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

        for edge_idx in self.window_graph.edge_indices() {
            let Some((tail_w, head_w)) = self.window_graph.edge_endpoints(edge_idx) else {
                continue;
            };
            let window_edge: WindowEdge = *self
                .window_graph
                .edge_weight(edge_idx)
                .expect("should find an edge weight");

            let mut tail_vertex = window_to_vertex[&tail_w];
            if let Some(chain) = dummy_chains.get(&(tail_w, head_w)) {
                for &dummy_id in chain {
                    let dummy_vertex = dummy_to_vertex[&dummy_id];
                    vertex_graph.add_edge(
                        tail_vertex,
                        dummy_vertex,
                        Edge {
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
                Edge {
                    input_node_idx_pair: window_edge,
                },
            );
        }

        self.vertex_graph = vertex_graph;
        self.vertex_layers = Some(vertex_layers);
        self.resolve_edge_legs(&window_to_vertex);

        Ok(())
    }

    /// Call `from_edges_with_dummies` once, which ranks and orders every vertex - real and
    /// dummy alike - via full-graph crossing minimization. Every multi-rank edge's dummy chain,
    /// backward-edge bypass loops included, comes straight from that single call; nothing is
    /// reconstructed or moved. The only override is a whole-window mirror: crossing counts are
    /// invariant under reversing every rank's order together (it's a reflection, not a
    /// reordering relative to anything else), so if a window's bypass loop came out bulging the
    /// "wrong" way, every rank gets reversed once to make it consistent - see the mirror step at
    /// the end of this function.
    fn build_ranks(
        &self,
        window_ids: &[NodeIndex<u32>],
        dense_id: &HashMap<NodeIndex<u32>, u32>,
    ) -> (RankedLayers, DummyChains) {
        // Build the dense edge list `from_edges_with_dummies` needs alongside a parallel list of
        // window-graph endpoints, in lockstep, so `edge_endpoints[edge_id]` always names the
        // input edge a returned `LayoutVertex::Dummy(edge_id)` subdivides.
        let dense_edges_with_endpoints: Vec<DenseEdgeWithEndpoints> = self
            .window_graph
            .edge_indices()
            .filter_map(|edge_idx| self.window_graph.edge_endpoints(edge_idx))
            .map(|(tail_w, head_w)| ((dense_id[&tail_w], dense_id[&head_w]), (tail_w, head_w)))
            .collect();
        let dense_edges: Vec<(u32, u32)> =
            dense_edges_with_endpoints.iter().map(|&(e, _)| e).collect();
        let edge_endpoints: Vec<(NodeIndex<u32>, NodeIndex<u32>)> =
            dense_edges_with_endpoints.iter().map(|&(_, w)| w).collect();

        let mut layouts = from_edges_with_dummies(&dense_edges, &self.config);
        let (positions, _width, _height) = layouts
            .pop()
            .expect("a connected, non-empty graph produces exactly one subgraph layout");
        debug_assert!(
            layouts.is_empty(),
            "window graph connectivity is validated above"
        );

        // Group by rank (y), keeping each vertex's raw `LayoutVertex` for now - the crate's
        // output order has no relationship to rank order, so dummy ids can only be minted once
        // ranks are sorted low to high below; minting them here would scramble each edge's
        // chain into an arbitrary order instead of tail-to-head.
        let mut groups: RawRankGroups = HashMap::new();
        for (vertex, (x, y)) in positions {
            groups.entry(y.to_bits()).or_default().push((vertex, x));
        }
        let mut rank_ys: Vec<f64> = groups.keys().map(|&bits| f64::from_bits(bits)).collect();
        rank_ys.sort_by(f64::total_cmp);

        // Now walk ranks low to high, sorting each by x, minting a fresh globally-unique id for
        // each dummy vertex and appending it to its edge's chain accumulator - both in true rank
        // order, since that's the order this loop visits them in.
        let mut ranks_full: RankedLayers = Vec::with_capacity(rank_ys.len());
        let mut rank_of: HashMap<NodeIndex<u32>, usize> = HashMap::new();
        let mut chains_by_edge: HashMap<usize, Vec<u32>> = HashMap::new();
        let mut next_id: u32 = 0;
        for (rank_index, y) in rank_ys.iter().enumerate() {
            let mut layer = groups.remove(&y.to_bits()).unwrap();
            layer.sort_by(|a, b| a.1.total_cmp(&b.1));
            let layer: Vec<(LayerEntry, f64)> = layer
                .into_iter()
                .map(|(vertex, x)| {
                    let entry = match vertex {
                        LayoutVertex::Node(dense) => LayerEntry::Real(window_ids[dense]),
                        LayoutVertex::Dummy(edge_id) => {
                            let id = next_id;
                            next_id += 1;
                            chains_by_edge.entry(edge_id).or_default().push(id);
                            LayerEntry::Dummy(id)
                        }
                    };
                    (entry, x)
                })
                .collect();
            for &(entry, _) in &layer {
                if let LayerEntry::Real(node) = entry {
                    rank_of.insert(node, rank_index);
                }
            }
            ranks_full.push(layer);
        }

        let mut dummy_chains: DummyChains = HashMap::new();
        for (edge_id, chain) in chains_by_edge {
            dummy_chains.insert(edge_endpoints[edge_id], chain);
        }

        // Mirror the whole window if it has a bypass loop. `from_edges_with_dummies` reliably
        // places dummy vertices - added to the graph after every real node, so tie-breaking
        // consistently favors their higher index - at the far extreme of their rank, opposite
        // this codebase's below-the-graph convention. Reversing every rank's order the once is a
        // pure reflection: it changes no relative order and so no crossing count, and always
        // corrects that bias in a single pass rather than checking for it per window.
        //
        // Scoped to windows with a backward-edge bypass, not every window: an unconditional flip
        // was tried and produced 6 non-rectilinear edges in `subsetting_grid`'s partial-budget
        // middle-anchor case (multiple boundary doors collapsing to the same off-window target).
        // `order` isn't just bookkeeping - `assemble_window` writes it straight into
        // `LayoutNode.pos` as a real row coordinate - and `edge_router::route_layer`'s
        // `layout_layer` is not a pure function of that row order: it detects "backtracking" via
        // a Y-range-vs-envelope heuristic and conditionally re-routes with reversed vertical
        // order, tuned against whatever arrangement `build_ranks` handed it before. Flipping
        // every window changes which windows trigger that fallback and how, so a wider flip
        // needs `layout_layer` made robust to it first, not just more flipping here.
        if !self.backward_span_edges.is_empty() {
            for layer in &mut ranks_full {
                layer.reverse();
            }
        }

        (ranks_full, dummy_chains)
    }

    /// Resolve `backward_span_edges` (window-graph node indices, set at window-build time)
    /// into `resolved_backward_span_edges` (`vertex_graph` edge endpoints), walking through
    /// any dummy-vertex chain just inserted for that edge. Called once, immediately after
    /// dummy insertion, from `build_structure`.
    fn resolve_edge_legs(&mut self, window_to_vertex: &HashMap<NodeIndex<u32>, NodeIndex<u32>>) {
        for &(from_window_idx, to_window_idx) in &self.backward_span_edges {
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
            for hop in resolve_leg_chain(&self.vertex_graph, from_vertex, to_vertex, bundle) {
                self.resolved_backward_span_edges.insert(hop);
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
                backward_span_edges: self.resolved_backward_span_edges.clone(),
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
fn resolve_leg_chain(
    vertex_graph: &StableDiGraph<Vertex, Edge, u32>,
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
