use std::{
    collections::{BTreeMap, HashMap, HashSet},
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
    cross_coordinates::{align_layers, assign_cross_coordinates, orient_aligned_bubbles},
    curved::LayoutMode,
    edge_ports::{NodeSide, PortedEdge},
    edge_router::{layout_graph_process::prune_pin_stubs, route_graph::make_rectilinear},
    geometry::{LocalPos, WorldPos, WorldRect},
    layout_compaction::{
        GapPolicy, MINIMUM_ROUTING_GAP_Y, VerticalPlacement, base_step,
        compact_layout_with_vertical_placement,
    },
    loopback_pins::{
        HorizontalBendNodes, drop_orphaned_routing_nodes, insert_horizontal_bend_nodes,
        reattach_loopback_pins, tether_self_loops,
    },
    pinning::hairpin_sides,
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

/// Final geometry for one assembled neighborhood window at a specific zoom level.
///
/// The graph is undirected to allow bidirectional edge routing, with final coordinates mapped
/// to its nodes.
#[derive(Clone, Serialize)]
pub(crate) struct WindowGeometry {
    pub graph: StableGraph<LayoutNode, LayoutEdge, Undirected, u32>,
    pub width: i64,
    pub height: i64,
    /// Each invisible domain node and the routing node it became (see [`Self::new`]).
    pub(crate) junctions: Vec<(NodeIndex, NodeIndex<u32>)>,
}

/// Dense layers need compact rows even if the routed coordinates spread their alternatives out.
/// Count visible data before routing, so routing nodes and display buffer size do not affect it.
fn vertical_placement_for_window(
    graph: &StableGraph<LayoutNode, LayoutEdge, Undirected, u32>,
    junctions: &HashSet<NodeIndex>,
) -> VerticalPlacement {
    const MAX_BALANCED_LAYER_MEMBERS: usize = 8;
    let mut members_by_layer = HashMap::<i32, usize>::new();
    for node in graph.node_weights() {
        if let NodeRole::Data(domain_index) = node.role
            && !junctions.contains(&domain_index)
            && let Some(layer) = node.layer
        {
            let members = members_by_layer.entry(layer).or_default();
            *members += 1;
            if *members > MAX_BALANCED_LAYER_MEMBERS {
                return VerticalPlacement::Tight;
            }
        }
    }
    VerticalPlacement::Balanced
}

/// Give every data node and every dummy a straight departure on either side of its layer before a
/// curved connection may bend. A dummy is a bypass for a long edge; giving it side ports like a data node
/// keeps its slopes in the gutters instead of cutting across the layer's data nodes. All
/// departures in a layer use the widest node's clearance columns.
fn add_curved_side_ports(graph: &mut StableGraph<LayoutNode, LayoutEdge, Undirected, u32>) {
    let mut layer_widths = BTreeMap::<i64, (i64, i64)>::new();
    for node in graph.node_weights() {
        let extents = layer_widths.entry(node.pos.x).or_insert((0, 0));
        if matches!(node.role, NodeRole::Data(_) | NodeRole::Routing) {
            let halves = crate::layout_compaction::halves(node.size.0 as i64);
            extents.0 = extents.0.max(halves.lo + 1);
            extents.1 = extents.1.max(halves.hi + 2);
        }
    }
    if layer_widths.is_empty() {
        return;
    }

    // Distinct starting columns are essential: curved compaction moves every node that
    // starts at the same x as one rigid group.
    let mut columns = HashMap::new();
    let mut offsets_by_center = HashMap::new();
    let mut previous_right = None;
    for (&original_x, &(left, right)) in &layer_widths {
        let center = previous_right.map_or(0, |end| end + left + 2);
        columns.insert(original_x, center);
        offsets_by_center.insert(center, (left, right));
        previous_right = Some(center + right);
    }
    for node in graph.node_weights_mut() {
        node.pos.x = columns[&node.pos.x];
    }

    // Port creation runs before any port exists, so every routing node here is a dummy.
    let side_ported_nodes: Vec<_> = graph
        .node_indices()
        .filter(|&index| matches!(graph[index].role, NodeRole::Data(_) | NodeRole::Routing))
        .collect();
    let mut ports = HashMap::new();
    for index in side_ported_nodes {
        let node = &graph[index];
        let (x, y, layer) = (node.pos.x, node.pos.y, node.layer);
        let (left, right) = offsets_by_center[&x];
        let mut left_node = LayoutNode::new(
            NodeRole::Routing,
            LocalPos::new_xy(x - left, y),
            (1, 1),
            layer,
        );
        left_node.side_port_offset = -left;
        left_node.side_port_anchor = Some(index);
        let left_port = graph.add_node(left_node);
        let mut right_node = LayoutNode::new(
            NodeRole::Routing,
            LocalPos::new_xy(x + right, y),
            (1, 1),
            layer,
        );
        right_node.side_port_offset = right;
        right_node.side_port_anchor = Some(index);
        let right_port = graph.add_node(right_node);
        ports.insert(index, (left_port, right_port));
    }

    let edges: Vec<_> = graph
        .edge_indices()
        .filter_map(|edge| {
            let (source, target) = graph.edge_endpoints(edge)?;
            Some((edge, source, target, graph[edge].clone()))
        })
        .collect();
    for (edge, source, target, bundle) in edges {
        let loopback_pin = [source, target]
            .into_iter()
            .any(|index| matches!(graph[index].role, NodeRole::Pin(side) if side.is_loopback()));
        if loopback_pin {
            continue;
        }
        let source_port = ports.get(&source).and_then(|&(left, right)| {
            (graph[source].pos.x < graph[target].pos.x)
                .then_some(right)
                .or_else(|| (graph[source].pos.x > graph[target].pos.x).then_some(left))
        });
        let target_port = ports.get(&target).and_then(|&(left, right)| {
            (graph[target].pos.x < graph[source].pos.x)
                .then_some(right)
                .or_else(|| (graph[target].pos.x > graph[source].pos.x).then_some(left))
        });
        if source_port.is_none() && target_port.is_none() {
            continue;
        }
        graph.remove_edge(edge);
        graph.add_edge(
            source_port.unwrap_or(source),
            target_port.unwrap_or(target),
            bundle.clone(),
        );
        if let Some(port) = source_port {
            graph.add_edge(source, port, bundle.clone());
        }
        if let Some(port) = target_port {
            graph.add_edge(port, target, bundle);
        }
    }
}

/// Keep each curved hairpin as a self-edge on its data node. Its domain edge retains the side.
fn add_hairpin_edges(
    graph: &mut StableGraph<LayoutNode, LayoutEdge, Undirected, u32>,
    hairpins: &HashSet<(NodeIndex, NodeSide)>,
) {
    let anchors: Vec<_> = graph
        .node_indices()
        .filter_map(|anchor| {
            let NodeRole::Data(domain_index) = graph[anchor].role else {
                return None;
            };
            let west = hairpins.contains(&(domain_index, NodeSide::West));
            let east = hairpins.contains(&(domain_index, NodeSide::East));
            (west || east).then_some((anchor, domain_index, west, east))
        })
        .collect();
    for (anchor, domain_index, west, east) in anchors {
        graph[anchor].hairpin_west = west;
        graph[anchor].hairpin_east = east;
        graph.add_edge(anchor, anchor, LayoutEdge::new(domain_index, domain_index));
    }
}

/// Align each hairpin node to the side of the layer the layout placed it in, so the loop starts at the
/// column where the layer's nodes start and does not drift across the layer. A pin on the east side
/// of its node has its attachment column on the left edge of the widest data node in its layer, and
/// a pin on the west side has it on the right edge. A layer with no data node leaves its pin where
/// compaction put it.
fn align_hairpin_nodes_beside_layer(
    graph: &mut StableGraph<LayoutNode, LayoutEdge, Undirected, u32>,
) {
    let origin = WorldPos::new(0, 0);
    let pins: Vec<_> = graph
        .node_indices()
        .filter(|&index| matches!(graph[index].role, NodeRole::HairPin(_)))
        .collect();
    for pin in pins {
        let NodeRole::HairPin(side) = graph[pin].role else {
            continue;
        };
        let Some(layer) = graph[pin].layer else {
            continue;
        };
        let widest = graph
            .node_indices()
            .filter(|&index| {
                graph[index].layer == Some(layer) && matches!(graph[index].role, NodeRole::Data(_))
            })
            .max_by_key(|&index| graph[index].size.0);
        let Some(widest) = widest else {
            continue;
        };
        let (pin_rect, widest_rect) = (graph[pin].rect(origin), graph[widest].rect(origin));
        graph[pin].pos.x += match side {
            NodeSide::East => widest_rect.min.x - pin_rect.min.x,
            NodeSide::West => widest_rect.max.x - pin_rect.max.x,
        };
    }
}

/// Replace each hairpin node with routing nodes that make its loop, so the shape is drawn by the
/// ordinary rules for routes and no mode draws a node of its own. The loop is two rows tall: the
/// route leaves the neighbor on the top row, which is the row of the bus, runs across the pin and
/// down at the far end, and comes back along the bottom row. The bus keeps its single edge to an
/// entry on the pin's near side. In the terminal the return rises to that entry; in the curve and
/// line modes it is an S-curve back to that neighbor, and the far end is a bend that starts and
/// ends flat.
fn expand_hairpin_nodes(
    graph: &mut StableGraph<LayoutNode, LayoutEdge, Undirected, u32>,
    layout_mode: LayoutMode,
) {
    let pins: Vec<_> = graph
        .node_indices()
        .filter(|&index| matches!(graph[index].role, NodeRole::HairPin(_)))
        .collect();
    for pin in pins {
        let NodeRole::HairPin(side) = graph[pin].role else {
            continue;
        };
        let neighbor_edge = graph.edges(pin).next().map(|edge| {
            let neighbor = if edge.source() == pin {
                edge.target()
            } else {
                edge.source()
            };
            (neighbor, edge.weight().clone())
        });
        let Some((neighbor, weight)) = neighbor_edge else {
            continue;
        };
        let rect = graph[pin].rect(WorldPos::new(0, 0));
        let row = graph[pin].pos.y;
        let (near_x, far_x) = match side {
            NodeSide::East => (rect.min.x, rect.max.x),
            NodeSide::West => (rect.max.x, rect.min.x),
        };
        let mut routing_node = |x: i64, y: i64| {
            graph.add_node(LayoutNode::new(
                NodeRole::Routing,
                LocalPos::new_xy(x, y),
                (1, 1),
                None,
            ))
        };
        let entry = routing_node(near_x, row);
        let far_top = routing_node(far_x, row);
        let far_bottom = routing_node(far_x, row - 1);
        let near_bottom = routing_node(near_x, row - 1);
        // The curve modes return to the neighbor the bus reached, so the S-curve ends where the
        // hairpin node was joined to the layout. The terminal's router needs the return to end on the
        // bus at the entry.
        let return_end = if layout_mode.is_curved() {
            neighbor
        } else {
            entry
        };
        let edges = vec![
            (neighbor, entry),
            (entry, far_top),
            (far_top, far_bottom),
            (far_bottom, near_bottom),
            (near_bottom, return_end),
        ];
        for (first, second) in edges {
            graph.add_edge(first, second, weight.clone());
        }
        graph.remove_node(pin);
    }
}

/// Shorten only a closed self-link route with one left pin, one right pin, and one exclusive
/// dummy between them. The domain index identifies a complete graph vertex, not a sequence node
/// ID shared by distinct slices. Other routing nodes keep their compaction coordinates.
fn shorten_self_link_detours(graph: &mut StableGraph<LayoutNode, LayoutEdge, Undirected, u32>) {
    let anchors: Vec<_> = graph
        .node_indices()
        .filter(|&index| matches!(graph[index].role, NodeRole::Data(_)))
        .collect();
    for anchor in anchors {
        let NodeRole::Data(domain_index) = graph[anchor].role else {
            continue;
        };
        let self_edge = |edge: &LayoutEdge| {
            !edge.bundle.is_empty()
                && edge
                    .bundle
                    .iter()
                    .all(|pair| *pair == (domain_index, domain_index))
        };
        let self_neighbors: Vec<_> = graph
            .edges(anchor)
            .filter(|edge| self_edge(edge.weight()))
            .map(|edge| {
                if edge.source() == anchor {
                    edge.target()
                } else {
                    edge.source()
                }
            })
            .filter(|&neighbor| neighbor != anchor)
            .collect();
        let [first, second] = self_neighbors.as_slice() else {
            continue;
        };
        let (first, second) = (*first, *second);
        let (left, right) = match (&graph[first].role, &graph[second].role) {
            (NodeRole::Pin(PinSide::Left), NodeRole::Pin(PinSide::Right)) => (first, second),
            (NodeRole::Pin(PinSide::Right), NodeRole::Pin(PinSide::Left)) => (second, first),
            _ => continue,
        };
        let pin_other = |pin| {
            let neighbors: Vec<_> = graph.neighbors(pin).collect();
            if neighbors.len() != 2 || !neighbors.contains(&anchor) {
                return None;
            }
            graph
                .edges(pin)
                .all(|edge| self_edge(edge.weight()))
                .then(|| {
                    neighbors
                        .into_iter()
                        .find(|&neighbor| neighbor != anchor)
                        .expect("should have the pin's other neighbor")
                })
        };
        let (Some(left_other), Some(right_other)) = (pin_other(left), pin_other(right)) else {
            continue;
        };
        if left_other != right_other
            || graph[left_other].role != NodeRole::Routing
            || graph.neighbors(left_other).collect::<HashSet<_>>() != HashSet::from([left, right])
            || graph.edges(left_other).count() != 2
            || graph
                .edges(left_other)
                .any(|edge| !self_edge(edge.weight()))
        {
            continue;
        }
        let middle = left_other;
        let ports: Vec<_> = graph
            .node_indices()
            .filter(|&index| graph[index].side_port_anchor == Some(middle))
            .collect();
        if ports
            .iter()
            .any(|&index| graph.edges(index).next().is_some())
        {
            continue;
        }

        let anchor_position = graph[anchor].pos;
        let side = if graph[middle].pos.y < anchor_position.y {
            -1
        } else {
            1
        };
        let clearance = graph[anchor].size.1.div_ceil(2) as i64 + 2;
        let reach = graph[anchor].size.0.div_ceil(2) as i64 + 6;
        let near = |index: NodeIndex<u32>| {
            LocalPos::new_xy(
                anchor_position.x + (graph[index].pos.x - anchor_position.x).clamp(-reach, reach),
                anchor_position.y + side * clearance,
            )
        };
        let (left_position, right_position) = (near(left), near(right));
        let candidates = [(left, left_position), (right, right_position)];
        let collides = candidates.iter().any(|&(pin, position)| {
            let mut moved = graph[pin].clone();
            moved.pos = position;
            graph.node_indices().any(|other| {
                other != pin
                    && other != left
                    && other != right
                    && other != middle
                    && !ports.contains(&other)
                    && moved
                        .rect(WorldPos::ZERO)
                        .intersects(&graph[other].rect(WorldPos::ZERO))
            })
        });
        if collides {
            continue;
        }

        for port in ports {
            graph.remove_node(port);
        }
        graph.remove_node(middle);
        graph.add_edge(
            left,
            right,
            LayoutEdge {
                bundle: vec![(domain_index, domain_index)],
                is_pin_span: true,
            },
        );
        graph[left].pos = left_position;
        graph[right].pos = right_position;
    }
}

/// Side ports are created in pairs before it is known which edges will use them. Drop ports with no
/// incident edge after compaction so the final geometry and DOT export contain only real routes.
fn prune_unused_side_ports(graph: &mut StableGraph<LayoutNode, LayoutEdge, Undirected, u32>) {
    let unused: Vec<_> = graph
        .node_indices()
        .filter(|&index| {
            graph[index].side_port_anchor.is_some() && graph.edges(index).next().is_none()
        })
        .collect();
    for index in unused {
        graph.remove_node(index);
    }
}

impl WindowGeometry {
    /// Route and compact an assembled window whose nodes already have their rendered sizes.
    ///
    /// The Data nodes for `junctions` (domain nodes the renderer doesn't draw) are laid out and
    /// routed like any other node, so every edge still meets at them and keeps its domain-edge
    /// bundle, then become routing nodes before compaction: they are drawn as the junction of
    /// their edges and are never placed as selectable nodes.
    ///
    /// `aligned_nodes` names the domain nodes aligned before routing (see
    /// [`crate::cross_coordinates::align_layers`]). The first surviving aligned node is
    /// returned to y = 0 after compaction; other nodes may drift if routing breaks the line.
    ///
    /// `domain_edges` tell a curved layout which edge ends attach on the side their edge does not
    /// leave toward and need a loop beside their node.
    ///
    /// With a curved `layout_mode` the router is skipped: every layer-to-layer segment, dummy
    /// chains and loopback pins included, stays a straight edge, and compaction keeps layers in
    /// columns far enough apart for those edges to slope gently.
    pub fn new(
        mut layout_graph: StableGraph<LayoutNode, LayoutEdge, Undirected, u32>,
        gap_policy: &GapPolicy,
        junctions: &HashSet<NodeIndex>,
        aligned_nodes: &HashSet<NodeIndex>,
        domain_edges: &[PortedEdge<NodeIndex>],
        layout_mode: LayoutMode,
    ) -> Self {
        let vertical_placement = vertical_placement_for_window(&layout_graph, junctions);
        // Assembly places rows at their size-independent within-layer ordinals. The edge router
        // reacts only to y, so assign real-size-aware cross coordinates before routing. This pass
        // only needs a small structural gap, not the caller's actual zoom target: compaction below
        // is the sole place the six gaps of the `GapPolicy` are enforced.
        // A curved drawing has no loopback pins in its layered layout: the ends that need one get a
        // routing node beside their node instead, and the pin returns once compaction is done.
        let horizontal_bend_nodes = if layout_mode.is_curved() {
            drop_orphaned_routing_nodes(&mut layout_graph);
            insert_horizontal_bend_nodes(&mut layout_graph, domain_edges)
        } else {
            HorizontalBendNodes::default()
        };
        orient_aligned_bubbles(&mut layout_graph, aligned_nodes);
        assign_cross_coordinates(&mut layout_graph, 1);
        align_layers(&mut layout_graph, aligned_nodes);
        tether_self_loops(&mut layout_graph, &horizontal_bend_nodes);

        if layout_mode.is_curved() {
            add_curved_side_ports(&mut layout_graph);
        }
        // A curved layout turns a hairpin into an arc beside its node. The layered layout cannot
        // carry a loop on one node, so the loop joins the node's side port to itself once the
        // ports exist, which lets compaction, drawing and highlighting treat it as an edge.
        if layout_mode.is_curved() {
            add_hairpin_edges(&mut layout_graph, &hairpin_sides(domain_edges));
        }

        if !layout_mode.is_curved()
            && let Err(error) = make_rectilinear(&mut layout_graph)
        {
            log::warn!("Edge routing failed: {:?}", error);
        }

        // The router replaces layout-node indices, so identify the first aligned domain node
        // after routing. A final translation restores its absolute row without bending edges.
        let first_aligned_node = layout_graph
            .node_indices()
            .filter(|&node_index| {
                matches!(layout_graph[node_index].role, NodeRole::Data(domain_index) if aligned_nodes.contains(&domain_index))
            })
            .min_by_key(|&node_index| {
                (layout_graph[node_index].pos.x, node_index.index())
            });

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

        // Curved edges keep the minimal zoom level's routing gaps at every level.
        let curved_gap_policy = GapPolicy {
            routing_routing_y: MINIMUM_ROUTING_GAP_Y,
            data_routing_y: |_| 1,
            ..*gap_policy
        };
        let gap_policy = if layout_mode.is_curved() {
            &curved_gap_policy
        } else {
            gap_policy
        };
        compact_layout_with_vertical_placement(
            &mut layout_graph,
            gap_policy,
            vertical_placement,
            first_aligned_node,
            layout_mode,
        );
        reattach_loopback_pins(&mut layout_graph, horizontal_bend_nodes.pins);
        align_hairpin_nodes_beside_layer(&mut layout_graph);
        expand_hairpin_nodes(&mut layout_graph, layout_mode);
        if layout_mode.is_curved() {
            shorten_self_link_detours(&mut layout_graph);
            prune_unused_side_ports(&mut layout_graph);
        }
        if let Some(anchor) = first_aligned_node {
            let offset = layout_graph[anchor].pos.y;
            for node in layout_graph.node_weights_mut() {
                node.pos.y -= offset;
            }
        }
        // A curved pin is an ordinary vertex of its loop's polyline, not a router stub.
        if !layout_mode.is_curved() {
            prune_pin_stubs(&mut layout_graph);
        }

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
    pub(crate) fn junction_node(&self, domain_index: NodeIndex) -> Option<&LayoutNode> {
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

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use petgraph::{Undirected, graph::NodeIndex, stable_graph::StableGraph};

    use super::{
        HAIRPIN_NODE_SIZE, LayoutEdge, LayoutNode, NodeRole, PinSide, VerticalAnchor,
        VerticalPlacement, add_curved_side_ports, add_hairpin_edges,
        align_hairpin_nodes_beside_layer, expand_hairpin_nodes, prune_unused_side_ports,
        shorten_self_link_detours, vertical_placement_for_window,
    };
    use crate::{
        curved::LayoutMode,
        edge_ports::NodeSide,
        geometry::{LocalPos, WorldPos, WorldRect},
    };

    #[test]
    fn test_more_than_eight_visible_data_nodes_in_one_layer_select_tight() {
        let mut graph = StableGraph::<LayoutNode, LayoutEdge, Undirected, u32>::default();
        let junctions = HashSet::new();
        for index in 0..8 {
            graph.add_node(LayoutNode::data(
                NodeIndex::new(index),
                LocalPos::new_xy(0, index as i64),
                (1, 1),
                Some(0),
            ));
        }
        assert_eq!(
            vertical_placement_for_window(&graph, &junctions),
            VerticalPlacement::Balanced
        );

        let ninth = graph.add_node(LayoutNode::data(
            NodeIndex::new(8),
            LocalPos::new_xy(0, 8),
            (1, 1),
            Some(1),
        ));
        assert_eq!(
            vertical_placement_for_window(&graph, &junctions),
            VerticalPlacement::Balanced
        );
        graph[ninth].layer = Some(0);
        let mut junctions = HashSet::from([NodeIndex::new(8)]);
        assert_eq!(
            vertical_placement_for_window(&graph, &junctions),
            VerticalPlacement::Balanced
        );
        junctions.clear();
        assert_eq!(
            vertical_placement_for_window(&graph, &junctions),
            VerticalPlacement::Tight
        );
    }

    #[test]
    fn test_self_link_detour_stays_near_its_data_node() {
        let mut graph = StableGraph::<LayoutNode, LayoutEdge, Undirected, u32>::default();
        let domain_index = NodeIndex::new(1);
        let anchor = graph.add_node(LayoutNode::data(
            domain_index,
            LocalPos::new_xy(10, 0),
            (5, 1),
            Some(1),
        ));
        let left = graph.add_node(LayoutNode::new(
            NodeRole::Pin(PinSide::Left),
            LocalPos::new_xy(0, 22),
            (1, 1),
            Some(0),
        ));
        let middle = graph.add_node(LayoutNode::routing(LocalPos::new_xy(10, 24), (1, 1)));
        let right = graph.add_node(LayoutNode::new(
            NodeRole::Pin(PinSide::Right),
            LocalPos::new_xy(20, 21),
            (1, 1),
            Some(2),
        ));
        for (from, to) in [
            (anchor, left),
            (left, middle),
            (middle, right),
            (right, anchor),
        ] {
            graph.add_edge(from, to, LayoutEdge::new(domain_index, domain_index));
        }

        shorten_self_link_detours(&mut graph);

        assert_eq!(graph[anchor].pos.y, 0);
        for index in [left, right] {
            assert_eq!(
                graph[index].pos.y, 3,
                "the self-link should stay near its anchor"
            );
        }
        assert_eq!(graph[left].pos.x, 1);
        assert_eq!(graph[right].pos.x, 19);
        assert!(graph.node_weight(middle).is_none());
        assert!(graph.find_edge(left, right).is_some());
    }

    #[test]
    fn test_bridge_routing_node_keeps_its_compacted_position() {
        let mut graph = StableGraph::<LayoutNode, LayoutEdge, Undirected, u32>::default();
        let first_domain = NodeIndex::new(1);
        let second_domain = NodeIndex::new(2);
        let first = graph.add_node(LayoutNode::data(
            first_domain,
            LocalPos::new_xy(0, 0),
            (5, 1),
            Some(0),
        ));
        let second = graph.add_node(LayoutNode::data(
            second_domain,
            LocalPos::new_xy(20, 0),
            (5, 1),
            Some(2),
        ));
        let bridge = graph.add_node(LayoutNode::routing(LocalPos::new_xy(10, 22), (1, 1)));
        graph.add_edge(first, bridge, LayoutEdge::new(first_domain, second_domain));
        graph.add_edge(bridge, second, LayoutEdge::new(first_domain, second_domain));

        shorten_self_link_detours(&mut graph);

        assert_eq!(graph[bridge].pos, LocalPos::new_xy(10, 22));
        assert_eq!(graph.edge_count(), 2);
    }

    #[test]
    fn test_self_link_with_shared_bridge_keeps_the_bridge_and_pins() {
        let mut graph = StableGraph::<LayoutNode, LayoutEdge, Undirected, u32>::default();
        let domain_index = NodeIndex::new(1);
        let other_domain = NodeIndex::new(2);
        let anchor = graph.add_node(LayoutNode::data(
            domain_index,
            LocalPos::new_xy(10, 0),
            (5, 1),
            Some(1),
        ));
        let other = graph.add_node(LayoutNode::data(
            other_domain,
            LocalPos::new_xy(30, 0),
            (5, 1),
            Some(3),
        ));
        let left = graph.add_node(LayoutNode::new(
            NodeRole::Pin(PinSide::Left),
            LocalPos::new_xy(0, 22),
            (1, 1),
            Some(0),
        ));
        let middle = graph.add_node(LayoutNode::routing(LocalPos::new_xy(10, 24), (1, 1)));
        let right = graph.add_node(LayoutNode::new(
            NodeRole::Pin(PinSide::Right),
            LocalPos::new_xy(20, 21),
            (1, 1),
            Some(2),
        ));
        for (from, to) in [
            (anchor, left),
            (left, middle),
            (middle, right),
            (right, anchor),
        ] {
            graph.add_edge(from, to, LayoutEdge::new(domain_index, domain_index));
        }
        graph.add_edge(middle, other, LayoutEdge::new(domain_index, other_domain));

        shorten_self_link_detours(&mut graph);

        assert_eq!(graph[left].pos, LocalPos::new_xy(0, 22));
        assert_eq!(graph[middle].pos, LocalPos::new_xy(10, 24));
        assert_eq!(graph[right].pos, LocalPos::new_xy(20, 21));
        assert_eq!(graph.edge_count(), 5);
    }

    #[test]
    fn test_unused_side_ports_do_not_leave_disconnected_routing_nodes() {
        let mut graph = StableGraph::<LayoutNode, LayoutEdge, Undirected, u32>::default();
        let first_domain = NodeIndex::new(1);
        let second_domain = NodeIndex::new(2);
        let first = graph.add_node(LayoutNode::data(
            first_domain,
            LocalPos::new_xy(0, 0),
            (5, 1),
            Some(0),
        ));
        let second = graph.add_node(LayoutNode::data(
            second_domain,
            LocalPos::new_xy(2, 0),
            (5, 1),
            Some(2),
        ));
        let pin = graph.add_node(LayoutNode::new(
            NodeRole::Pin(PinSide::Left),
            LocalPos::new_xy(1, 3),
            (1, 1),
            Some(1),
        ));
        graph.add_edge(first, pin, LayoutEdge::new(first_domain, first_domain));
        graph.add_edge(first, second, LayoutEdge::new(first_domain, second_domain));

        add_curved_side_ports(&mut graph);
        let pin_position = graph[pin].pos;
        assert!(graph.find_edge(first, pin).is_some());
        prune_unused_side_ports(&mut graph);

        assert_eq!(graph[pin].pos, pin_position);
        assert!(graph.find_edge(first, pin).is_some());
        assert!(graph.node_indices().all(|index| {
            graph[index].side_port_anchor.is_none() || graph.edges(index).next().is_some()
        }));
    }

    #[test]
    fn test_curved_hairpins_are_self_edges_on_the_data_node() {
        let mut graph = StableGraph::<LayoutNode, LayoutEdge, Undirected, u32>::default();
        let domain_index = NodeIndex::new(0);
        let anchor = graph.add_node(LayoutNode::data(
            domain_index,
            LocalPos::new_xy(0, 0),
            (5, 1),
            Some(0),
        ));
        add_curved_side_ports(&mut graph);
        add_hairpin_edges(
            &mut graph,
            &HashSet::from([
                (domain_index, NodeSide::West),
                (domain_index, NodeSide::East),
            ]),
        );
        prune_unused_side_ports(&mut graph);

        assert!(graph.find_edge(anchor, anchor).is_some());
        assert!(graph[anchor].hairpin_west && graph[anchor].hairpin_east);
        assert!(graph.node_indices().all(|index| {
            graph[index].side_port_anchor.is_none() || graph.edges(index).next().is_some()
        }));
        assert_eq!(graph.node_count(), 1);
    }

    /// A node with a hairpin node on `side`, joined by one edge carrying the node's self-edge, sized
    /// and anchored as the geometry builder does.
    fn node_with_hairpin_node(
        side: NodeSide,
    ) -> (
        StableGraph<LayoutNode, LayoutEdge, Undirected, u32>,
        NodeIndex<u32>,
        WorldRect,
    ) {
        let mut graph: StableGraph<LayoutNode, LayoutEdge, Undirected, u32> =
            StableGraph::default();
        let node = graph.add_node(LayoutNode::new(
            NodeRole::Data(NodeIndex::new(0)),
            LocalPos::new_xy(0, 0),
            (5, 3),
            Some(0),
        ));
        let mut hairpin_node = LayoutNode::new(
            NodeRole::HairPin(side),
            LocalPos::new_xy(12, 3),
            HAIRPIN_NODE_SIZE,
            Some(1),
        );
        hairpin_node.vertical_anchor = VerticalAnchor::Top;
        let rect = hairpin_node.rect(WorldPos::new(0, 0));
        let pin = graph.add_node(hairpin_node);
        graph.add_edge(
            node,
            pin,
            LayoutEdge::new(NodeIndex::new(0), NodeIndex::new(0)),
        );
        (graph, node, rect)
    }

    fn routing_positions(
        graph: &StableGraph<LayoutNode, LayoutEdge, Undirected, u32>,
    ) -> Vec<(i64, i64)> {
        let mut positions: Vec<(i64, i64)> = graph
            .node_weights()
            .filter(|node| matches!(node.role, NodeRole::Routing))
            .map(|node| (node.pos.x, node.pos.y))
            .collect();
        positions.sort_unstable();
        positions
    }

    #[test]
    fn test_expand_hairpin_nodes_makes_a_two_row_box_in_the_terminal() {
        for side in [NodeSide::East, NodeSide::West] {
            let (mut graph, node, rect) = node_with_hairpin_node(side);

            expand_hairpin_nodes(&mut graph, LayoutMode::Rectilinear);

            let (near_x, far_x) = match side {
                NodeSide::East => (rect.min.x, rect.max.x),
                NodeSide::West => (rect.max.x, rect.min.x),
            };
            let mut expected = vec![
                (near_x, rect.max.y),
                (near_x, rect.min.y),
                (far_x, rect.max.y),
                (far_x, rect.min.y),
            ];
            expected.sort_unstable();
            assert_eq!(routing_positions(&graph), expected);
            assert_eq!(rect.max.y - rect.min.y, 1, "two rows");
            assert_eq!(graph.node_count(), 5, "the pin is gone");
            assert_eq!(graph.edge_count(), 5);
            assert_eq!(graph.edges(node).count(), 1, "the bus stays one edge");
            assert!(
                graph
                    .edge_weights()
                    .all(|edge| edge.bundle == [(NodeIndex::new(0), NodeIndex::new(0))])
            );
        }
    }

    #[test]
    fn test_expand_hairpin_nodes_returns_to_the_neighbor_in_the_curve_modes() {
        let style = LayoutMode::Curved;
        for side in [NodeSide::East, NodeSide::West] {
            let (mut graph, node, rect) = node_with_hairpin_node(side);

            expand_hairpin_nodes(&mut graph, style);

            let (near_x, far_x) = match side {
                NodeSide::East => (rect.min.x, rect.max.x),
                NodeSide::West => (rect.max.x, rect.min.x),
            };
            let mut expected = vec![
                (near_x, rect.max.y),
                (near_x, rect.min.y),
                (far_x, rect.max.y),
                (far_x, rect.min.y),
            ];
            expected.sort_unstable();
            assert_eq!(routing_positions(&graph), expected);
            assert_eq!(rect.max.y - rect.min.y, 1, "two rows");
            assert_eq!(graph.node_count(), 5, "the pin is gone");
            assert_eq!(graph.edge_count(), 5);
            assert_eq!(
                graph.edges(node).count(),
                2,
                "the bus leaves the neighbor and the return comes back to it"
            );
        }
    }

    /// A layer of two data nodes, one narrow and one wide, with a hairpin node of `side` in the same
    /// layer, off to the side of both.
    fn layer_with_hairpin_node_beside_data_nodes(
        side: NodeSide,
        pin_x: i64,
    ) -> (
        StableGraph<LayoutNode, LayoutEdge, Undirected, u32>,
        NodeIndex<u32>,
        NodeIndex<u32>,
    ) {
        let mut graph: StableGraph<LayoutNode, LayoutEdge, Undirected, u32> =
            StableGraph::default();
        graph.add_node(LayoutNode::new(
            NodeRole::Data(NodeIndex::new(0)),
            LocalPos::new_xy(20, 0),
            (5, 3),
            Some(1),
        ));
        let wide = graph.add_node(LayoutNode::new(
            NodeRole::Data(NodeIndex::new(1)),
            LocalPos::new_xy(22, 5),
            (9, 3),
            Some(1),
        ));
        graph.add_node(LayoutNode::new(
            NodeRole::Data(NodeIndex::new(2)),
            LocalPos::new_xy(2, 0),
            (15, 3),
            Some(0),
        ));
        let mut hairpin_node = LayoutNode::new(
            NodeRole::HairPin(side),
            LocalPos::new_xy(pin_x, 9),
            HAIRPIN_NODE_SIZE,
            Some(1),
        );
        hairpin_node.vertical_anchor = VerticalAnchor::Top;
        let pin = graph.add_node(hairpin_node);
        (graph, wide, pin)
    }

    #[test]
    fn test_align_hairpin_nodes_puts_an_east_hairpin_node_to_the_left_of_the_widest_node_in_its_layer()
     {
        for pin_x in [10, 22, 40] {
            let (mut graph, wide, pin) =
                layer_with_hairpin_node_beside_data_nodes(NodeSide::East, pin_x);

            align_hairpin_nodes_beside_layer(&mut graph);

            let origin = WorldPos::new(0, 0);
            assert_eq!(
                graph[pin].rect(origin).min.x,
                graph[wide].rect(origin).min.x,
                "pin started at {pin_x}"
            );
            assert_eq!(graph[pin].pos.y, 9, "only x moves");
        }
    }

    #[test]
    fn test_align_hairpin_nodes_puts_a_west_hairpin_node_to_the_right_of_the_widest_node_in_its_layer()
     {
        for pin_x in [10, 22, 40] {
            let (mut graph, wide, pin) =
                layer_with_hairpin_node_beside_data_nodes(NodeSide::West, pin_x);

            align_hairpin_nodes_beside_layer(&mut graph);

            let origin = WorldPos::new(0, 0);
            assert_eq!(
                graph[pin].rect(origin).max.x,
                graph[wide].rect(origin).max.x,
                "pin started at {pin_x}"
            );
        }
    }

    #[test]
    fn test_align_hairpin_nodes_leaves_a_hairpin_node_in_a_layer_without_data_nodes() {
        let (mut graph, _, pin) = layer_with_hairpin_node_beside_data_nodes(NodeSide::East, 40);
        graph[pin].layer = Some(7);

        align_hairpin_nodes_beside_layer(&mut graph);

        assert_eq!(graph[pin].pos.x, 40);
    }
}
