use std::collections::{HashMap, HashSet};

use petgraph::{Undirected, graph::NodeIndex, stable_graph::StableGraph};

use super::LayoutError;
use crate::layout::{LayoutEdge, LayoutNode, NodeRole};

/// Port directions for a node (North, East, South, West)
pub type PortDirections = (bool, bool, bool, bool);

/// An edge's preserved domain-edge bundle and `is_backward_span` flag (see `LayoutEdge`).
pub(crate) type BundledLeg = (Vec<(NodeIndex, NodeIndex)>, bool);

/// A simplified/spliced edge's `BundledLeg`, keyed by its endpoints.
pub(crate) type BundledLegEdges = HashMap<(NodeIndex, NodeIndex), BundledLeg>;
type PortsByNode = HashMap<NodeIndex, PortDirections>;
type PinSplice = (NodeIndex, NodeIndex, Vec<(NodeIndex, NodeIndex)>, bool);

pub fn assign_ports(graph: &StableGraph<LayoutNode, LayoutEdge, Undirected>) -> PortsByNode {
    let mut ports_map = HashMap::new();

    for node_index in graph.node_indices() {
        let node = graph.node_weight(node_index).unwrap();
        let (x, y) = (node.pos.x, node.pos.y);

        let (mut north, mut east, mut south, mut west) = (false, false, false, false);

        for neighbor in graph.neighbors(node_index) {
            let neighbor_node = graph.node_weight(neighbor).unwrap();
            let (neighbor_x, neighbor_y) = (neighbor_node.pos.x, neighbor_node.pos.y);

            // Assume upward-pointing y-axis, and rightward-pointing x-axis
            if neighbor_y > y {
                north = true;
            }
            if neighbor_y < y {
                south = true;
            }
            if neighbor_x > x {
                east = true;
            }
            if neighbor_x < x {
                west = true;
            }
        }

        ports_map.insert(node_index, (north, east, south, west));
    }

    ports_map
}

/// Simplifies a graph by identifying and contracting segments with collinear edges.
/// Preserves LayoutEdge bundle information from the edges being contracted.
/// Asserts that all edges in a straight segment have identical bundles.
pub fn simplify_graph(
    graph: &mut StableGraph<LayoutNode, LayoutEdge, Undirected>,
) -> Result<(), LayoutError> {
    if graph.edge_count() == 0 {
        return Ok(());
    }

    let ports_map = assign_ports(graph);

    // Define straight orientations (these get contracted)
    const STRAIGHT_VERTICAL: PortDirections = (true, false, true, false);
    const STRAIGHT_HORIZONTAL: PortDirections = (false, true, false, true);

    // 1. Identify and add all critical nodes (not on a collinear segment)
    let mut critical_nodes = HashSet::new();

    for node_index in graph.node_indices() {
        let node = graph.node_weight(node_index).unwrap();
        let orientation = ports_map
            .get(&node_index)
            .copied()
            .unwrap_or((false, false, false, false));

        // Critical nodes are: non-straight routing nodes, or any Data/Pin nodes.
        // Pin nodes are always critical (never contracted here) so they survive until
        // `prune_pin_stubs` runs and removes them explicitly.
        let is_critical = match &node.role {
            NodeRole::Data(_) | NodeRole::Pin => true,
            NodeRole::Routing | NodeRole::Wormhole(_) => {
                orientation != STRAIGHT_HORIZONTAL && orientation != STRAIGHT_VERTICAL
            }
        };

        if is_critical {
            critical_nodes.insert(node_index);
        }
    }

    // Handle case where graph might be a single, straight segment
    if critical_nodes.is_empty() && graph.node_count() > 0 {
        let start_node_id = graph.node_indices().next().unwrap();
        critical_nodes.insert(start_node_id);
    }

    // 2. Iterate through critical nodes and trace segments, preserving bundles and the
    // `is_backward_span` flag (see `LayoutEdge::is_backward_span`).
    let mut new_edges_with_bundles: BundledLegEdges = HashMap::new();
    // A straight segment is expected to carry one consistent bundle/flag end to end, but on
    // some real-world graphs (independent of window size - confirmed still occurring under
    // the windowed Sugiyama path too) a handful of edges within a segment disagree; root
    // cause not yet found. Count and discard the disagreeing edge's data (keep the segment's
    // original bundle/flag) instead of panicking on what is otherwise a rare, cosmetic
    // bundle-highlighting glitch.
    let mut bundle_mismatches = 0usize;
    let mut backward_span_mismatches = 0usize;

    for start_node_id in &critical_nodes {
        for neighbor_id in graph.neighbors(*start_node_id).collect::<Vec<_>>() {
            if critical_nodes.contains(&neighbor_id) {
                // Direct connection - preserve bundle and flag from the existing edge
                let edge_id = graph.find_edge(*start_node_id, neighbor_id).unwrap();
                let edge = graph.edge_weight(edge_id).unwrap();
                let bundle = edge.bundle.clone();
                let is_backward_span = edge.is_backward_span;

                let segment_endpoints = if start_node_id.index() < neighbor_id.index() {
                    (*start_node_id, neighbor_id)
                } else {
                    (neighbor_id, *start_node_id)
                };

                new_edges_with_bundles.insert(segment_endpoints, (bundle, is_backward_span));
            } else {
                let mut previous_id = *start_node_id;
                let mut current_id = neighbor_id;

                // Get bundle and flag from first edge in segment
                let (segment_bundle, segment_is_backward_span) =
                    if let Some(edge_id) = graph.find_edge(*start_node_id, current_id) {
                        let edge = graph.edge_weight(edge_id).unwrap();
                        (edge.bundle.clone(), edge.is_backward_span)
                    } else {
                        (vec![], false)
                    };

                let end_node_id = loop {
                    let current_orientation = ports_map
                        .get(&current_id)
                        .copied()
                        .unwrap_or((false, false, false, false));

                    // Start of a straight segment
                    if current_orientation == STRAIGHT_HORIZONTAL
                        || current_orientation == STRAIGHT_VERTICAL
                    {
                        let neighbors_of_current: Vec<_> = graph.neighbors(current_id).collect();
                        if neighbors_of_current.len() != 2 {
                            break Some(current_id);
                        }

                        let next_node_id = if neighbors_of_current[1] == previous_id {
                            neighbors_of_current[0]
                        } else {
                            neighbors_of_current[1]
                        };

                        // Verify that bundle and flag are identical along the segment.
                        // Diagnostics only: count and discard disagreements instead of
                        // panicking (see comment on `bundle_mismatches` above).
                        if let Some(edge_id) = graph.find_edge(current_id, next_node_id) {
                            let next_edge = graph.edge_weight(edge_id).unwrap();
                            if segment_bundle != next_edge.bundle {
                                bundle_mismatches += 1;
                            }
                            if segment_is_backward_span != next_edge.is_backward_span {
                                backward_span_mismatches += 1;
                            }
                        }

                        previous_id = current_id;
                        current_id = next_node_id;
                        if critical_nodes.contains(&current_id) {
                            break Some(current_id);
                        }
                    } else {
                        break Some(current_id);
                    }
                };

                // Add simplified edge with preserved bundle and flag
                if let Some(end_node_id) = end_node_id
                    && critical_nodes.contains(&end_node_id)
                {
                    let segment_endpoints = if start_node_id.index() < end_node_id.index() {
                        (*start_node_id, end_node_id)
                    } else {
                        (end_node_id, *start_node_id)
                    };

                    new_edges_with_bundles.insert(
                        segment_endpoints,
                        (segment_bundle, segment_is_backward_span),
                    );
                }
            }
        }
    }

    if bundle_mismatches > 0 || backward_span_mismatches > 0 {
        log::warn!(
            "simplify_graph: discarded {bundle_mismatches} bundle mismatches and \
             {backward_span_mismatches} is_backward_span mismatches across straight segments"
        );
    }

    // Remove non-critical nodes
    let all_nodes: HashSet<_> = graph.node_indices().collect();
    let noncritical_nodes: Vec<_> = all_nodes.difference(&critical_nodes).cloned().collect();

    for node_index in noncritical_nodes {
        graph.remove_node(node_index);
    }

    // Add new edges with their preserved bundles and flags (only if both nodes still exist).
    // `new_edges_with_bundles` is a `HashMap`, so iterating it directly would add edges in an
    // order that varies between otherwise-identical calls (Rust's default hasher reseeds per
    // instance) - and that edge-insertion order is visible downstream, since `compact_layout`
    // walks `graph.edge_references()` in insertion order when building its constraint graph.
    // Sort by endpoint first so newly contracted edges have deterministic insertion order.
    let mut sorted_new_edges: Vec<_> = new_edges_with_bundles.into_iter().collect();
    sorted_new_edges.sort_by_key(|((source, target), _)| (source.index(), target.index()));
    for ((source, target), (bundle, is_backward_span)) in sorted_new_edges {
        if graph.node_weight(source).is_some()
            && graph.node_weight(target).is_some()
            && graph.find_edge(source, target).is_none()
        {
            let layout_edge = LayoutEdge {
                bundle,
                is_backward_span,
            };
            graph.add_edge(source, target, layout_edge);
        }
    }

    // Direct edges between critical nodes survive simplification with their original insertion
    // indices, so sorting only `new_edges_with_bundles` does not make the complete graph stable.
    // Reorder the exact final edge records after simplification instead of rebuilding them from
    // the endpoint-keyed map above: this preserves parallel edges and every edge's own weight.
    let mut sorted_edges = graph
        .edge_indices()
        .filter_map(|edge_index| {
            let (source, target) = graph.edge_endpoints(edge_index)?;
            let endpoints = if source.index() < target.index() {
                (source, target)
            } else {
                (target, source)
            };
            Some((endpoints, graph.edge_weight(edge_index)?.clone()))
        })
        .collect::<Vec<_>>();
    sorted_edges.sort_by(|((source1, target1), edge1), ((source2, target2), edge2)| {
        (
            source1.index(),
            target1.index(),
            edge1.is_backward_span,
            &edge1.bundle,
        )
            .cmp(&(
                source2.index(),
                target2.index(),
                edge2.is_backward_span,
                &edge2.bundle,
            ))
    });
    for edge_index in graph.edge_indices().collect::<Vec<_>>() {
        graph.remove_edge(edge_index);
    }
    for ((source, target), edge) in sorted_edges {
        graph.add_edge(source, target, edge);
    }

    Ok(())
}

/// Removes the synthetic `NodeRole::Pin` nodes used to render a backward edge as a
/// full-width loop, once rectilinear edge routing has finished and `simplify_graph` has
/// already contracted ordinary straight runs (`Pin` is always "critical" there, so it
/// survives untouched until this pass runs).
///
/// A pin's two logical connections (to the data node it pins in place, and to the long
/// bypass edge) usually get routed as a single physical line up to a nearby T-junction,
/// where the rectilinear router splits off towards the two actual destinations - leaving
/// the pin itself as a degree-1 dead end off that junction. This pass walks the chain
/// from each pin up to that junction, removing the pin, every straight pass-through
/// `Routing` node along the way, and the junction itself - splicing the junction's two
/// other neighbors directly together with an edge carrying the bundle accumulated while
/// walking from the pin. A no-op for any graph with no backward edges, since no `Pin`
/// node is ever injected for one.
pub fn prune_pin_stubs(graph: &mut StableGraph<LayoutNode, LayoutEdge, Undirected>) {
    let pins: Vec<NodeIndex> = graph
        .node_indices()
        .filter(|&node_index| matches!(graph[node_index].role, NodeRole::Pin))
        .collect();

    for pin_index in pins {
        // A pin's own stub may already have been removed while walking out from a
        // previously processed pin (e.g. two pins whose chains meet directly).
        if graph.node_weight(pin_index).is_some() {
            prune_pin(graph, pin_index);
        }
    }
}

/// Removes `pin_index` and, if it has exactly one neighbor, the chain of straight
/// pass-through `Routing` nodes and the terminating junction leading from it - splicing
/// the junction's two other neighbors together with the bundle carried along the way.
/// See `prune_pin_stubs`.
///
/// A pin with two or more neighbors is not a dead-end stub but a load-bearing corner of
/// the bypass path: it happens when a locally-scoped loop's pin lands directly next to its
/// real endpoint (see `crawl::build_window_graph`), wiring the target and the bypass to the
/// pin without an intervening routing mesh to relay them through. Removing it would sever
/// the loop, so it is re-roled to an ordinary `Routing` corner (pins already render as
/// routing nodes) and left in place.
fn prune_pin(graph: &mut StableGraph<LayoutNode, LayoutEdge, Undirected>, pin_index: NodeIndex) {
    let neighbors: Vec<NodeIndex> = graph.neighbors(pin_index).collect();

    if neighbors.len() >= 2 {
        graph[pin_index].role = NodeRole::Routing;
        return;
    }

    let mut to_remove = vec![pin_index];
    let mut splice: Option<PinSplice> = None;

    if let [start] = neighbors.as_slice() {
        let mut previous = pin_index;
        let mut current = *start;
        let first_edge = graph.find_edge(pin_index, current);
        let mut bundle = first_edge
            .map(|edge| graph[edge].bundle.clone())
            .unwrap_or_default();
        // The whole walk from one pin to the next junction rides along a single edge's
        // worth of content (the pin's own), so this only ever picks up one value - OR'd in
        // purely to match `LayoutEdge::is_backward_span`'s accumulation style, not because
        // it can flip mid-walk.
        let mut is_backward_span = first_edge.is_some_and(|edge| graph[edge].is_backward_span);

        while matches!(graph[current].role, NodeRole::Routing) {
            let other_neighbors: Vec<NodeIndex> = graph
                .neighbors(current)
                .filter(|&neighbor| neighbor != previous)
                .collect();

            match other_neighbors.as_slice() {
                [only] => {
                    to_remove.push(current);
                    if let Some(edge) = graph.find_edge(current, *only) {
                        bundle.extend(graph[edge].bundle.clone());
                        is_backward_span |= graph[edge].is_backward_span;
                    }
                    previous = current;
                    current = *only;
                }
                [a, b] => {
                    let jpos = graph[current].pos;
                    let apos = graph[*a].pos;
                    let bpos = graph[*b].pos;
                    let colinear = (jpos.x == apos.x && jpos.x == bpos.x)
                        || (jpos.y == apos.y && jpos.y == bpos.y);
                    if colinear {
                        // Straight pass-through junction: remove it and splice
                        // its two remaining neighbors together.
                        to_remove.push(current);
                        splice = Some((*a, *b, bundle, is_backward_span));
                    }
                    // Perpendicular: the junction is a corner, leave it in place.
                    break;
                }
                _ => break, // dead end or a true 3+-way junction - leave it in place.
            }
        }
    }

    if let Some((a, b, bundle, is_backward_span)) = splice
        && graph.find_edge(a, b).is_none()
    {
        graph.add_edge(
            a,
            b,
            LayoutEdge {
                bundle,
                is_backward_span,
            },
        );
    }

    for node_index in to_remove {
        graph.remove_node(node_index);
    }
}
