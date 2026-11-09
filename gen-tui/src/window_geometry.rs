//! Positioned, routed and compacted geometry for one assembled neighborhood window.
//!
//! `WindowGeometry` holds the result; `build_window_geometry_with_anchors` is the sizing
//! step between `assembly::assemble_window` and `WindowGeometry::new`. It gives every node of an
//! [`AssembledLayout`] the size the current renderer wants, then hands the graph to
//! `WindowGeometry::new` for coordinates, routing and compaction;
//! `RoutedWindowGraph::from_window_geometry` consumes the result. It also supplies the
//! single-glyph node marker. `WindowScene`, tests and exports call it to get positioned,
//! compacted nodes without going through a full controller.

use std::collections::{BTreeMap, HashMap, HashSet};

use itertools::Itertools;
use petgraph::{Undirected, graph::NodeIndex, stable_graph::StableGraph, visit::EdgeRef};
use serde::Serialize;

use crate::{
    assembly::AssembledLayout,
    cross_coordinates::{align_layers, assign_cross_coordinates, orient_aligned_bubbles},
    edge_ports::{NodeSide, PortedEdge},
    edge_router::{route_graph::make_rectilinear, route_simplification::prune_pin_stubs},
    geometry::{LocalPos, WorldPos},
    layout::{HAIRPIN_NODE_SIZE, LayoutEdge, LayoutNode, NodeRole, PinSide, VerticalAnchor},
    layout_compaction::{
        GapPolicy, MINIMUM_ROUTING_GAP_Y, VerticalPlacement, compact_layout_with_vertical_placement,
    },
    layout_mode::LayoutMode,
    loopback_pins::{
        LateralBendNodes, drop_orphaned_routing_nodes, insert_lateral_bend_nodes,
        reattach_loopback_pins, tether_self_loops,
    },
    pinning::hairpin_sides,
};

/// Single-cell glyph a node is drawn as at the lowest zoom level.
pub const NODE_GLYPH: char = '●';

/// Route and compact an assembled neighborhood window into renderable geometry.
///
/// This is the detail-dependent half of the level-of-detail split: `LayoutEngine` assembles
/// the window once into a `StableGraph<LayoutNode, LayoutEdge>` (see
/// [`crate::assembly::assemble_window`]), and this re-sizes every node at the current level of
/// detail, then assigns cross coordinates, routes, and compacts it. Sizing is applied here
/// before coordinate assignment, so the geometry reflects the detail the visual currently
/// holds; the assembled neutral sizes are discarded.
///
/// `size_of` supplies the rendered size for each node role, and `is_visible` says which domain
/// nodes are drawn; the rest become junctions of their edges (see [`WindowGeometry::new`]) and
/// should be given a routing node's size. `is_aligned` marks nodes that pin their layer to
/// y = 0 (see `NodeRenderer::is_aligned`).
#[cfg(test)]
pub(crate) fn build_window_geometry(
    assembled: AssembledLayout,
    gap_policy: &GapPolicy,
    size_of: impl FnMut(&NodeRole) -> (u64, u64),
    is_visible: impl Fn(NodeIndex) -> bool,
    is_aligned: impl Fn(NodeIndex) -> bool,
) -> WindowGeometry {
    build_window_geometry_with_anchors(
        assembled,
        gap_policy,
        LayoutMode::Rectilinear,
        size_of,
        is_visible,
        is_aligned,
        |_| VerticalAnchor::Center,
    )
}

/// Build geometry with a per-node vertical attachment row and a layout mode, shared by layout
/// and rendering.
pub(crate) fn build_window_geometry_with_anchors(
    assembled: AssembledLayout,
    gap_policy: &GapPolicy,
    layout_mode: LayoutMode,
    mut size_of: impl FnMut(&NodeRole) -> (u64, u64),
    is_visible: impl Fn(NodeIndex) -> bool,
    is_aligned: impl Fn(NodeIndex) -> bool,
    vertical_anchor: impl Fn(NodeIndex) -> VerticalAnchor,
) -> WindowGeometry {
    let mut graph = assembled.graph;
    let mut junctions = HashSet::new();
    let mut aligned_nodes = HashSet::new();
    for node in graph.node_weights_mut() {
        node.size = size_of(&node.role);
        match node.role {
            NodeRole::Data(domain_index) => {
                node.vertical_anchor = vertical_anchor(domain_index);
                if !is_visible(domain_index) {
                    junctions.insert(domain_index);
                }
                if is_aligned(domain_index) {
                    aligned_nodes.insert(domain_index);
                }
            }
            // A turnaround has a glyph of its own whose top row is the row its bus meets.
            NodeRole::HairPin(_) => {
                node.size = HAIRPIN_NODE_SIZE;
                node.vertical_anchor = VerticalAnchor::Top;
            }
            _ => {}
        }
    }
    WindowGeometry::new(
        graph,
        gap_policy,
        &junctions,
        &aligned_nodes,
        &assembled.domain_edges,
        layout_mode,
    )
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
        let lateral_bend_nodes = if layout_mode.is_curved() {
            drop_orphaned_routing_nodes(&mut layout_graph);
            insert_lateral_bend_nodes(&mut layout_graph, domain_edges)
        } else {
            LateralBendNodes::default()
        };
        orient_aligned_bubbles(&mut layout_graph, aligned_nodes);
        assign_cross_coordinates(&mut layout_graph, 1);
        align_layers(&mut layout_graph, aligned_nodes);
        tether_self_loops(&mut layout_graph, &lateral_bend_nodes);

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
        reattach_loopback_pins(&mut layout_graph, lateral_bend_nodes.pins);
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
        edge_ports::NodeSide,
        geometry::{LocalPos, WorldPos, WorldRect},
        layout_mode::LayoutMode,
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

    mod build_window_geometry {
        use std::collections::HashMap;

        use super::super::*;
        use crate::{
            assembly::assemble_window,
            crawl::{EagerSource, GraphCursor, build_window_graph, neighborhood},
            layout::NodeRole,
            testing::mocks::{MockDomainGraph, TestGraphs},
        };

        /// Crawl the whole graph from its first node and assemble it into one window graph.
        fn assembled_window(graph: &MockDomainGraph) -> AssembledLayout {
            let anchor = graph
                .node_indices()
                .next()
                .expect("should have graph nodes");
            let mut graph = graph.clone();
            let subgraph = neighborhood(
                anchor,
                graph.node_count(),
                &mut GraphCursor::new(&mut graph, &mut EagerSource),
                &|_| false,
                &HashMap::new(),
            )
            .expect("should find the anchor in the graph");
            let (window, _) =
                build_window_graph(&subgraph, &graph, None).expect("should build the window");
            assemble_window(&window).expect("should assemble the window")
        }

        /// Size every data node 5x3 and every routing/pin node 1x1, standing in for a domain
        /// visual at a fixed level of detail.
        fn size_of(role: &NodeRole) -> (u64, u64) {
            match role {
                NodeRole::Data(_) => (5, 3),
                _ => (1, 1),
            }
        }

        #[test]
        fn test_top_anchors_survive_routing_and_compaction() {
            let graph = TestGraphs::domain_extended_diamond();
            let geometry = build_window_geometry_with_anchors(
                assembled_window(&graph),
                &GapPolicy::default(),
                LayoutMode::Rectilinear,
                |role| match role {
                    NodeRole::Data(index) => (5, 2 + index.index() as u64 % 5),
                    _ => (1, 1),
                },
                |_| true,
                |_| false,
                |_| VerticalAnchor::Top,
            );
            for index in geometry.graph.node_indices() {
                let node = &geometry.graph[index];
                if matches!(node.role, NodeRole::Data(_)) {
                    let rect = node.rect(WorldPos::ZERO);
                    assert_eq!(node.vertical_anchor, VerticalAnchor::Top);
                    assert_eq!(rect.max.y, node.pos.y);
                    assert_eq!(rect.max.y - rect.min.y + 1, node.size.1 as i64);
                    for neighbor in geometry.graph.neighbors(index) {
                        assert_eq!(
                            geometry.graph[neighbor].pos.y, rect.max.y,
                            "edges must attach horizontally on the top row"
                        );
                    }
                }
            }
        }

        #[test]
        fn test_window_geometry_keeps_data_nodes() {
            let graph = TestGraphs::domain_extended_diamond();
            let assembled = assembled_window(&graph);
            let expected_data = assembled
                .graph
                .node_weights()
                .filter(|node| matches!(node.role, NodeRole::Data(_)))
                .count();
            assert!(expected_data > 0);

            let geometry = build_window_geometry(
                assembled,
                &GapPolicy::default(),
                size_of,
                |_| true,
                |_| false,
            );

            let data_nodes = geometry
                .graph
                .node_weights()
                .filter(|node| matches!(node.role, NodeRole::Data(_)))
                .count();
            assert_eq!(
                data_nodes, expected_data,
                "routing and compaction must preserve every data node"
            );
        }

        #[test]
        fn test_window_geometry_applies_current_detail_sizes() {
            let graph = TestGraphs::domain_extended_diamond();
            let assembled = assembled_window(&graph);

            let geometry = build_window_geometry(
                assembled,
                &GapPolicy::default(),
                size_of,
                |_| true,
                |_| false,
            );

            for node in geometry.graph.node_weights() {
                if matches!(node.role, NodeRole::Data(_)) {
                    assert_eq!(
                        node.size,
                        (5, 3),
                        "data nodes must be re-sized from the renderer's sizes, not the \
                         assembled provisional size"
                    );
                }
            }
        }
    }
}
