//! Routes edges around the ends of their nodes in the curved layout mode without left or right
//! pins in the layered layout.
//!
//! In a curved drawing a pin is never drawn, yet while it is a vertex of the layered layout it
//! claims a layer, rows and columns of empty space, and it pulls the chains of the edges it
//! serves far from their nodes. So those edges join the layered layout as ordinary edges. Once
//! Sugiyama has ordered the layers, an end that attaches on the side its edge does not leave
//! toward gets a routing node in its own layer, seated beside its node at the place that keeps
//! the edge from crossing any other edge between the two layers. Coordinates and compaction see
//! only those routing nodes. When compaction is done each is joined to its node through a pin
//! again, so the renderers find the loop the way they always did.

use std::collections::{BTreeMap, HashMap, HashSet};

use petgraph::{
    Undirected,
    graph::NodeIndex,
    stable_graph::{EdgeIndex, StableGraph},
    visit::EdgeRef,
};

use crate::{
    edge_ports::{NodeSide, PortedEdge},
    geometry::{LocalPos, WorldPos},
    layout::{LayoutEdge, LayoutNode, NodeRole, PinSide},
};

type LayoutGraph = StableGraph<LayoutNode, LayoutEdge, Undirected, u32>;

/// One edge of a pin that is yet to be restored.
struct PinLeg {
    neighbor: NodeIndex<u32>,
    weight: LayoutEdge,
    neighbor_is_source: bool,
}

/// A pin to restore once compaction is done, with the stand-in edge that joins its two
/// neighbors until then.
pub(crate) struct DetachedPin {
    node: LayoutNode,
    stand_in: EdgeIndex<u32>,
    legs: [PinLeg; 2],
}

/// Remove the routing nodes that no edge reaches. When parallel edges between two nodes share one
/// dummy chain, the chains Sugiyama made for the others stay in their layers with nothing attached,
/// and each would claim a row of its layer for nothing.
pub(crate) fn drop_orphaned_routing_nodes(graph: &mut LayoutGraph) {
    let orphans: Vec<NodeIndex<u32>> = graph
        .node_indices()
        .filter(|&index| {
            matches!(graph[index].role, NodeRole::Routing)
                && graph.neighbors(index).next().is_none()
        })
        .collect();
    for index in orphans {
        graph.remove_node(index);
    }
}

/// What [`insert_lateral_bend_nodes`] added to the layout.
#[derive(Default)]
pub(crate) struct LateralBendNodes {
    /// The pins that put each loop back after compaction.
    pub(crate) pins: Vec<DetachedPin>,
    /// Each self edge's routing node with the node it serves, which have to stay together.
    self_loops: Vec<(NodeIndex<u32>, NodeIndex<u32>)>,
}

/// Where a routing node sits in its layer: in the gap before the original node at `slot`, ordered
/// among the other nodes in that gap by the row of the neighbor it connects to.
struct Seat {
    routing: NodeIndex<u32>,
    slot: usize,
    neighbor_row: usize,
    /// The layer the seat's edge continues into; `None` for a self edge.
    neighbor_layer: Option<i64>,
}

/// The order of every layer as Sugiyama left it, before any routing node is seated.
struct LayerOrder {
    layers: BTreeMap<i64, Vec<NodeIndex<u32>>>,
    row: HashMap<NodeIndex<u32>, usize>,
}

impl LayerOrder {
    fn new(graph: &LayoutGraph) -> Self {
        let mut layers: BTreeMap<i64, Vec<NodeIndex<u32>>> = BTreeMap::new();
        for index in graph.node_indices() {
            layers.entry(graph[index].pos.x).or_default().push(index);
        }
        let mut row = HashMap::new();
        for layer in layers.values_mut() {
            layer.sort_by_key(|&index| (graph[index].pos.y, index));
            for (position, &index) in layer.iter().enumerate() {
                row.insert(index, position);
            }
        }
        Self { layers, row }
    }
}

/// Give every end of `domain_edges` that attaches on the wrong side a routing node beside its
/// node, and return the pins that put each loop back after compaction.
///
/// An end is on the wrong side when its edge continues to a later layer from a west side or to
/// an earlier layer from an east side. A self edge between a node's two sides has no chain, so
/// one routing node serves both of its ends.
pub(crate) fn insert_lateral_bend_nodes(
    graph: &mut LayoutGraph,
    domain_edges: &[PortedEdge<NodeIndex>],
) -> LateralBendNodes {
    let node_of: HashMap<NodeIndex, NodeIndex<u32>> = graph
        .node_indices()
        .filter_map(|index| match graph[index].role {
            NodeRole::Data(domain_node) => Some((domain_node, index)),
            _ => None,
        })
        .collect();
    let mut lateral_bend_edges = Vec::new();
    let mut self_loops = Vec::new();
    for edge in domain_edges {
        let (Some(&source), Some(&target)) = (node_of.get(&edge.source), node_of.get(&edge.target))
        else {
            continue;
        };
        if source == target {
            if edge.ports.source != edge.ports.target {
                self_loops.push((source, edge));
            }
            continue;
        }
        let (source_layer, target_layer) = (graph[source].pos.x, graph[target].pos.x);
        let source_loops = on_wrong_side(edge.ports.source, target_layer > source_layer);
        let target_loops = on_wrong_side(edge.ports.target, source_layer > target_layer);
        if source_layer != target_layer && (source_loops || target_loops) {
            lateral_bend_edges.push((edge, [source_loops, target_loops]));
        }
    }

    let order = LayerOrder::new(graph);
    let mut seats: HashMap<i64, Vec<Seat>> = HashMap::new();
    let mut detached = Vec::new();
    let mut self_loop_nodes = Vec::new();
    let mut claimed: HashMap<EdgeIndex<u32>, usize> = HashMap::new();
    for (edge, loops) in lateral_bend_edges {
        let pair = (edge.source, edge.target);
        let (source, target) = (node_of[&edge.source], node_of[&edge.target]);
        let Some((chain, first_hop)) = claim_chain(graph, &claimed, [source, target], pair) else {
            continue;
        };
        // A first hop that is rerouted gives up its entry of the pair; one that stays has to be
        // remembered, so no other domain edge with the same pair claims the same chain.
        if chain.len() > 2 && !loops[0] {
            *claimed.entry(first_hop).or_default() += 1;
        }
        let routing_nodes = [
            loops[0].then(|| add_routing_node(graph, source)),
            loops[1].then(|| add_routing_node(graph, target)),
        ];
        let last = chain.len() - 1;
        let hops = if chain.len() == 2 {
            vec![(0, last)]
        } else {
            vec![(0, 1), (last - 1, last)]
        };
        for (from, to) in hops {
            let replacement = |position: usize| match (position == 0, position == last) {
                (true, _) => routing_nodes[0].unwrap_or(chain[position]),
                (_, true) => routing_nodes[1].unwrap_or(chain[position]),
                _ => chain[position],
            };
            reroute_hop(
                graph,
                [chain[from], chain[to]],
                pair,
                [replacement(from), replacement(to)],
            );
        }
        for (end, routing) in routing_nodes.into_iter().enumerate() {
            let Some(routing) = routing else { continue };
            let (node, neighbor, side) = if end == 0 {
                (source, chain[1], edge.ports.source)
            } else {
                (target, chain[last - 1], edge.ports.target)
            };
            let layer_seats = seats.entry(graph[node].pos.x).or_default();
            let slot = best_slot(graph, &order, node, Some(neighbor), layer_seats);
            layer_seats.push(Seat {
                routing,
                slot,
                neighbor_row: order.row[&neighbor],
                neighbor_layer: Some(graph[neighbor].pos.x),
            });
            detached.push(loop_pin(graph, node, routing, side, end == 0, pair));
        }
    }
    for (node, edge) in self_loops {
        let routing = add_routing_node(graph, node);
        let layer_seats = seats.entry(graph[node].pos.x).or_default();
        let slot = best_slot(graph, &order, node, None, layer_seats);
        // A self edge's routing node sits nearest its node in whichever gap it took.
        let nearest = if slot <= order.row[&node] {
            usize::MAX
        } else {
            0
        };
        layer_seats.push(Seat {
            routing,
            slot,
            neighbor_row: nearest,
            neighbor_layer: None,
        });
        self_loop_nodes.push((node, routing));
        let pair = (edge.source, edge.target);
        for (side, is_source) in [(edge.ports.source, true), (edge.ports.target, false)] {
            detached.push(loop_pin(graph, node, routing, side, is_source, pair));
        }
    }
    seat_in_layers(graph, &order, &seats);
    LateralBendNodes {
        pins: detached,
        self_loops: self_loop_nodes,
    }
}

/// Pull each self edge's routing node up against its node, or against the routing node of
/// another self edge that already sits between them. Coordinate assignment places a routing node
/// with no edge to another layer anywhere in its gap of the layer order, which can leave rows
/// between it and its node for other edges to cross the loop through. Compaction keeps the
/// distance it is given only as a minimum, so each also records a tie that holds it there.
pub(crate) fn tether_self_loops(graph: &mut LayoutGraph, lateral_bend_nodes: &LateralBendNodes) {
    let mut by_node: BTreeMap<NodeIndex<u32>, Vec<NodeIndex<u32>>> = BTreeMap::new();
    for &(node, routing) in &lateral_bend_nodes.self_loops {
        by_node.entry(node).or_default().push(routing);
    }
    for (node, mut routing_nodes) in by_node {
        let node_row = graph[node].pos.y;
        routing_nodes.sort_by_key(|&routing| graph[routing].pos.y);
        let (above, below): (Vec<_>, Vec<_>) = routing_nodes
            .into_iter()
            .partition(|&routing| graph[routing].pos.y < node_row);
        let mut previous = node;
        for routing in above.into_iter().rev() {
            let row = graph[previous].pos.y - graph[routing].vertical_separation(&graph[previous]);
            graph[routing].pos.y = row;
            graph[routing].row_tether = Some((previous, row - graph[previous].pos.y));
            previous = routing;
        }
        let mut previous = node;
        for routing in below {
            let row = graph[previous].pos.y + graph[previous].vertical_separation(&graph[routing]);
            graph[routing].pos.y = row;
            graph[routing].row_tether = Some((previous, row - graph[previous].pos.y));
            previous = routing;
        }
    }
}

/// The end of `edge` that is not `node`. An undirected edge reference always reports the node it
/// was found from as its source, so the true endpoints have to come from the graph.
fn other_end(graph: &LayoutGraph, edge: EdgeIndex<u32>, node: NodeIndex<u32>) -> NodeIndex<u32> {
    let (source, target) = graph
        .edge_endpoints(edge)
        .expect("should have the edge it was listed from");
    if source == node { target } else { source }
}

/// Whether an end on `side` of its node is the wrong side for an edge that continues east or west.
fn on_wrong_side(side: NodeSide, continues_east: bool) -> bool {
    matches!(
        (side, continues_east),
        (NodeSide::West, true) | (NodeSide::East, false)
    )
}

/// Claim one of the chains of layout edges that join `ends[0]` to `ends[1]` for the domain edge
/// `pair`, as the nodes along it, ends included. A chain's hops are routing nodes with two
/// neighbors, so it is followed from the first hop on to wherever it stops. Parallel domain edges
/// have chains of their own, and each one claims a chain no other has.
fn claim_chain(
    graph: &LayoutGraph,
    claimed: &HashMap<EdgeIndex<u32>, usize>,
    ends: [NodeIndex<u32>; 2],
    pair: (NodeIndex, NodeIndex),
) -> Option<(Vec<NodeIndex<u32>>, EdgeIndex<u32>)> {
    let [from, to] = ends;
    let mut first_hops: Vec<_> = graph
        .edges(from)
        .filter(|edge| {
            let other = other_end(graph, edge.id(), from);
            graph[other].pos.x != graph[from].pos.x
                && edge
                    .weight()
                    .bundle
                    .iter()
                    .filter(|&&entry| entry == pair)
                    .count()
                    > claimed.get(&edge.id()).copied().unwrap_or(0)
        })
        .map(|edge| (edge.id(), other_end(graph, edge.id(), from)))
        .collect();
    first_hops.sort_by_key(|&(edge, _)| edge);
    for (edge, first) in first_hops {
        let mut chain = vec![from, first];
        while let Some(&last) = chain.last() {
            if !matches!(graph[last].role, NodeRole::Routing) {
                break;
            }
            let neighbors: Vec<_> = graph.neighbors(last).collect();
            let [one, other] = neighbors[..] else { break };
            let previous = chain[chain.len() - 2];
            let next = if one == previous { other } else { one };
            if one == other || chain.contains(&next) {
                break;
            }
            chain.push(next);
        }
        if chain.last() == Some(&to) {
            return Some((chain, edge));
        }
    }
    None
}

/// Replace the hop between `hop` (two adjacent nodes of a chain) by one between `replacement`,
/// keeping the hop's direction and giving up one entry of `pair` from the old edge.
fn reroute_hop(
    graph: &mut LayoutGraph,
    hop: [NodeIndex<u32>; 2],
    pair: (NodeIndex, NodeIndex),
    replacement: [NodeIndex<u32>; 2],
) {
    let [from, to] = hop;
    let Some(edge) = graph
        .edges(from)
        .find(|edge| {
            other_end(graph, edge.id(), from) == to && edge.weight().bundle.contains(&pair)
        })
        .map(|edge| edge.id())
    else {
        return;
    };
    let from_is_source = graph
        .edge_endpoints(edge)
        .is_some_and(|(source, _)| source == from);
    let bundle = &mut graph[edge].bundle;
    if let Some(entry) = bundle.iter().position(|&entry| entry == pair) {
        bundle.remove(entry);
    }
    if bundle.is_empty() {
        graph.remove_edge(edge);
    }
    let weight = LayoutEdge::new(pair.0, pair.1);
    if from_is_source {
        graph.add_edge(replacement[0], replacement[1], weight);
    } else {
        graph.add_edge(replacement[1], replacement[0], weight);
    }
}

fn add_routing_node(graph: &mut LayoutGraph, beside: NodeIndex<u32>) -> NodeIndex<u32> {
    let node = &graph[beside];
    let routing = LayoutNode::new(NodeRole::Routing, node.pos, (1, 1), node.layer);
    graph.add_node(routing)
}

/// The gap in `node`'s layer where a routing node for an edge to `neighbor` crosses the fewest
/// edges between the two layers, the one nearest `node` among equals. Each gap is numbered by the
/// original node it precedes. An edge from a node above the gap to a neighbor below `neighbor`
/// crosses it, and so does one from a node below the gap to a neighbor above `neighbor`. The edge
/// from `node` to `neighbor` itself is the one being replaced, so it does not count. Without a
/// neighbor nothing can cross, and the gaps beside `node` that hold fewer seats win.
fn best_slot(
    graph: &LayoutGraph,
    order: &LayerOrder,
    node: NodeIndex<u32>,
    neighbor: Option<NodeIndex<u32>>,
    seated: &[Seat],
) -> usize {
    let layer = &order.layers[&graph[node].pos.x];
    let node_row = order.row[&node];
    let (neighbor_layer, neighbor_row) = neighbor.map_or((None, 0), |neighbor| {
        (Some(graph[neighbor].pos.x), order.row[&neighbor])
    });
    let mut below = vec![0usize; layer.len()];
    let mut above = vec![0usize; layer.len()];
    for (row, &member) in layer.iter().enumerate() {
        for other in graph.neighbors(member) {
            let Some(&other_row) = order.row.get(&other) else {
                continue;
            };
            let replaced = member == node && Some(other) == neighbor;
            if replaced || Some(graph[other].pos.x) != neighbor_layer {
                continue;
            }
            if other_row > neighbor_row {
                below[row] += 1;
            } else if other_row < neighbor_row {
                above[row] += 1;
            }
        }
    }
    // Seats already placed bring their own edges into the same layer pair.
    // A self edge has no neighbor, but its loop wraps the node's own edges on the side it seats on:
    // each edge to a neighbor on that side of the node, judged by relative position in its layer,
    // crosses the loop's return run.
    let fan = |above_node: bool| -> usize {
        graph
            .neighbors(node)
            .filter(|&other| graph[other].pos.x != graph[node].pos.x)
            .filter_map(|other| {
                let other_layer = &order.layers[&graph[other].pos.x];
                let other_row = *order.row.get(&other)?;
                let is_above = other_row * layer.len() < node_row * other_layer.len();
                let is_below = other_row * layer.len() > node_row * other_layer.len();
                (if above_node { is_above } else { is_below }).then_some(())
            })
            .count()
    };
    let crossings = |slot: usize| -> usize {
        if neighbor.is_none() {
            return fan(slot <= node_row);
        }
        let earlier = seated
            .iter()
            .filter(|seat| seat.neighbor_layer == neighbor_layer && neighbor_layer.is_some())
            .filter(|seat| {
                (seat.slot < slot && seat.neighbor_row > neighbor_row)
                    || (seat.slot > slot && seat.neighbor_row < neighbor_row)
            })
            .count();
        earlier + below[..slot].iter().sum::<usize>() + above[slot..].iter().sum::<usize>()
    };
    let distance = |slot: usize| {
        if slot <= node_row {
            node_row - slot
        } else {
            slot - node_row - 1
        }
    };
    let taken = |slot: usize| seated.iter().filter(|seat| seat.slot == slot).count();
    (0..=layer.len())
        .min_by_key(|&slot| (crossings(slot), distance(slot), taken(slot), slot))
        .expect("should have a gap in the layer")
}

/// The pin that carries `node`'s loop around to `routing` once compaction is done, with the
/// stand-in edge that joins them until then.
fn loop_pin(
    graph: &mut LayoutGraph,
    node: NodeIndex<u32>,
    routing: NodeIndex<u32>,
    side: NodeSide,
    is_source: bool,
    pair: (NodeIndex, NodeIndex),
) -> DetachedPin {
    let pin_side = match side {
        NodeSide::West => PinSide::Left,
        NodeSide::East => PinSide::Right,
    };
    let leg = |neighbor, neighbor_is_source| PinLeg {
        neighbor,
        weight: LayoutEdge::new(pair.0, pair.1),
        neighbor_is_source,
    };
    // The legs follow the domain edge: a source end leaves its node for the pin, a target end
    // arrives at its node from the pin.
    let legs = [leg(node, is_source), leg(routing, !is_source)];
    let stand_in = join(graph, &legs[0], &legs[1]);
    let layer = graph[node].layer;
    DetachedPin {
        node: LayoutNode::new(NodeRole::Pin(pin_side), graph[node].pos, (1, 1), layer),
        stand_in,
        legs,
    }
}

/// The edge that stands in for a pin: both legs' bundle, running from the return node to the
/// anchor when the return node was the head of its leg, and from the anchor to it otherwise.
fn join(graph: &mut LayoutGraph, anchor: &PinLeg, returning: &PinLeg) -> EdgeIndex<u32> {
    let weight = LayoutEdge {
        bundle: merged_bundle(&anchor.weight, &returning.weight),
        is_pin_span: false,
    };
    if returning.neighbor_is_source {
        graph.add_edge(anchor.neighbor, returning.neighbor, weight)
    } else {
        graph.add_edge(returning.neighbor, anchor.neighbor, weight)
    }
}

fn merged_bundle(first: &LayoutEdge, second: &LayoutEdge) -> Vec<(NodeIndex, NodeIndex)> {
    let mut bundle = first.bundle.clone();
    for pair in &second.bundle {
        if !bundle.contains(pair) {
            bundle.push(*pair);
        }
    }
    bundle
}

/// Put every routing node in its gap of the layer order, then make each layer's ordinals dense
/// again. Nodes in one gap keep the order of the rows they connect to.
fn seat_in_layers(graph: &mut LayoutGraph, order: &LayerOrder, seats: &HashMap<i64, Vec<Seat>>) {
    for (layer_x, members) in &order.layers {
        let mut keyed: Vec<((usize, usize, usize), NodeIndex<u32>)> = members
            .iter()
            .enumerate()
            .map(|(row, &index)| ((2 * row + 1, 0, index.index()), index))
            .collect();
        for seat in seats.get(layer_x).into_iter().flatten() {
            keyed.push((
                (2 * seat.slot, seat.neighbor_row, seat.routing.index()),
                seat.routing,
            ));
        }
        keyed.sort_by_key(|&(key, _)| key);
        for (row, (_, index)) in keyed.into_iter().enumerate() {
            graph[index].pos.y = row as i64;
        }
    }
}

/// Put detached pins back beside the nodes they serve, replacing each stand-in edge with the
/// pin's own two edges.
///
/// A pin sits just outside the nearer end of its two neighbors, on the anchor's row, on a cell no
/// other node uses: the routed window graph keys nodes by position, so a shared cell would merge two
/// nodes into one.
pub(crate) fn reattach_loopback_pins(graph: &mut LayoutGraph, detached: Vec<DetachedPin>) {
    for pin in detached {
        let [anchor_leg, return_leg] = &pin.legs;
        graph.remove_edge(pin.stand_in);
        let NodeRole::Pin(side) = pin.node.role else {
            continue;
        };
        let position = free_pin_position(graph, side, [anchor_leg.neighbor, return_leg.neighbor]);
        let mut node = pin.node;
        node.pos = position;
        let index = graph.add_node(node);
        for leg in &pin.legs {
            if leg.neighbor_is_source {
                graph.add_edge(leg.neighbor, index, leg.weight.clone());
            } else {
                graph.add_edge(index, leg.neighbor, leg.weight.clone());
            }
        }
    }
}

/// The cell outside the two neighbors on `side`, level with the first one, or the nearest row to
/// it that no other node's attachment point occupies.
fn free_pin_position(graph: &LayoutGraph, side: PinSide, ends: [NodeIndex<u32>; 2]) -> LocalPos {
    let origin = WorldPos::new(0, 0);
    let rects = ends.map(|end| graph[end].rect(origin));
    let x = match side {
        PinSide::Right => rects[0].right().max(rects[1].right()) + 1,
        _ => rects[0].left().min(rects[1].left()) - 1,
    };
    let level_with = graph[ends[0]].pos.y;
    let occupied: HashSet<(i64, i64)> = graph
        .node_weights()
        .map(|node| (node.pos.x, node.pos.y))
        .collect();
    let y = (0..)
        .flat_map(|distance| [level_with + distance, level_with - distance])
        .find(|&y| !occupied.contains(&(x, y)))
        .expect("should find a free row");
    LocalPos::new_xy(x, y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edge_ports::EdgePorts;

    fn domain(index: usize) -> NodeIndex {
        NodeIndex::new(index)
    }

    fn data(index: usize, x: i64, y: i64) -> LayoutNode {
        LayoutNode::new(
            NodeRole::Data(domain(index)),
            LocalPos::new_xy(x, y),
            (1, 1),
            Some(x as i32),
        )
    }

    fn ported_edge(source: usize, target: usize, ports: EdgePorts) -> PortedEdge<NodeIndex> {
        PortedEdge {
            source: domain(source),
            target: domain(target),
            ports,
        }
    }

    const WEST_EAST: EdgePorts = EdgePorts::new(NodeSide::West, NodeSide::East);
    const EAST_EAST: EdgePorts = EdgePorts::new(NodeSide::East, NodeSide::East);

    /// Data nodes 0 and 1 in adjacent layers, joined by one edge from 0 to 1.
    fn two_nodes() -> (LayoutGraph, NodeIndex<u32>, NodeIndex<u32>) {
        let mut graph = LayoutGraph::default();
        let first = graph.add_node(data(0, 0, 0));
        let second = graph.add_node(data(1, 1, 0));
        graph.add_edge(first, second, LayoutEdge::new(domain(0), domain(1)));
        (graph, first, second)
    }

    fn routing_nodes(graph: &LayoutGraph) -> Vec<NodeIndex<u32>> {
        graph
            .node_indices()
            .filter(|&index| matches!(graph[index].role, NodeRole::Routing))
            .collect()
    }

    #[test]
    fn test_routing_nodes_without_edges_are_dropped() {
        let (mut graph, first, second) = two_nodes();
        let orphan = graph.add_node(LayoutNode::new(
            NodeRole::Routing,
            LocalPos::new_xy(0, 1),
            (1, 1),
            Some(0),
        ));
        drop_orphaned_routing_nodes(&mut graph);
        assert!(graph.node_weight(orphan).is_none());
        assert!(graph.node_weight(first).is_some() && graph.node_weight(second).is_some());
    }

    #[test]
    fn test_edge_leaving_the_side_it_continues_toward_gets_no_routing_node() {
        let (mut graph, _, _) = two_nodes();
        let edges = [ported_edge(
            0,
            1,
            EdgePorts::new(NodeSide::East, NodeSide::West),
        )];
        assert!(
            insert_lateral_bend_nodes(&mut graph, &edges)
                .pins
                .is_empty()
        );
        assert!(routing_nodes(&graph).is_empty());
    }

    #[test]
    fn test_east_to_east_edge_gets_a_routing_node_beside_each_end() {
        let (mut graph, first, second) = two_nodes();
        // From 0 east to 1 east: 1 continues west toward 0 from its east side, so it loops.
        let edges = [ported_edge(0, 1, EAST_EAST)];
        let detached = insert_lateral_bend_nodes(&mut graph, &edges).pins;
        assert_eq!(detached.len(), 1);
        let [routing] = routing_nodes(&graph)[..] else {
            panic!("should add one routing node");
        };
        assert_eq!(graph[routing].pos.x, graph[second].pos.x);
        assert!(graph.find_edge(routing, first).is_some());
        assert!(graph.find_edge(first, second).is_none());
    }

    #[test]
    fn test_routing_node_takes_the_gap_that_crosses_the_fewest_edges() {
        // Layer 0 holds three nodes, and the edge from `node` (layer 1, row 1) continues to the
        // middle one. `other` above `node` connects to the lowest, so the only gap that does not
        // cross it is the one above `other`, though that is not beside `node`.
        let mut graph = LayoutGraph::default();
        let other = graph.add_node(data(2, 1, 0));
        let node = graph.add_node(data(0, 1, 1));
        let _first = graph.add_node(data(4, 0, 0));
        let middle = graph.add_node(data(1, 0, 1));
        let last = graph.add_node(data(3, 0, 2));
        graph.add_edge(node, middle, LayoutEdge::new(domain(0), domain(1)));
        graph.add_edge(other, last, LayoutEdge::new(domain(2), domain(3)));
        insert_lateral_bend_nodes(&mut graph, &[ported_edge(0, 1, EAST_EAST)]);
        let [routing] = routing_nodes(&graph)[..] else {
            panic!("should add one routing node");
        };
        assert!(graph[routing].pos.y < graph[other].pos.y);
    }

    #[test]
    fn test_loop_is_found_when_the_layout_edge_runs_toward_the_claiming_node() {
        // The layout edge runs from node 1 (layer 1) to node 0 (layer 0), but the domain edge
        // starts at node 0 and attaches east, so its end at node 1 needs a loop and its chain is
        // found from the edge's target side.
        let mut graph = LayoutGraph::default();
        let first = graph.add_node(data(0, 0, 0));
        let second = graph.add_node(data(1, 1, 0));
        graph.add_edge(second, first, LayoutEdge::new(domain(0), domain(1)));
        let detached = insert_lateral_bend_nodes(&mut graph, &[ported_edge(0, 1, EAST_EAST)]).pins;
        assert_eq!(detached.len(), 1);
        assert_eq!(routing_nodes(&graph).len(), 1);
    }

    #[test]
    fn test_routing_node_prefers_the_gap_beside_its_node_when_none_crosses_less() {
        let (mut graph, _, second) = two_nodes();
        insert_lateral_bend_nodes(&mut graph, &[ported_edge(0, 1, EAST_EAST)]);
        let [routing] = routing_nodes(&graph)[..] else {
            panic!("should add one routing node");
        };
        assert_eq!((graph[routing].pos.y - graph[second].pos.y).abs(), 1);
    }

    #[test]
    fn test_self_edge_between_sides_gets_one_routing_node_and_two_pins() {
        let mut graph = LayoutGraph::default();
        let node = graph.add_node(data(0, 0, 0));
        let edges = [ported_edge(0, 0, WEST_EAST)];
        let detached = insert_lateral_bend_nodes(&mut graph, &edges).pins;
        assert_eq!(detached.len(), 2);
        assert_eq!(routing_nodes(&graph).len(), 1);
        reattach_loopback_pins(&mut graph, detached);
        let pins: Vec<_> = graph
            .node_indices()
            .filter(|&index| matches!(graph[index].role, NodeRole::Pin(_)))
            .collect();
        assert_eq!(pins.len(), 2);
        assert!(pins.iter().all(|&pin| graph.neighbors(pin).count() == 2));
        assert!(
            graph
                .neighbors(node)
                .all(|neighbor| pins.contains(&neighbor))
        );
    }

    #[test]
    fn test_reattached_pins_use_cells_no_other_node_uses() {
        let (mut graph, _, _) = two_nodes();
        let detached = insert_lateral_bend_nodes(&mut graph, &[ported_edge(0, 1, EAST_EAST)]).pins;
        reattach_loopback_pins(&mut graph, detached);
        let positions: Vec<_> = graph
            .node_weights()
            .map(|weight| (weight.pos.x, weight.pos.y))
            .collect();
        let unique: HashSet<_> = positions.iter().collect();
        assert_eq!(unique.len(), positions.len());
    }
}
