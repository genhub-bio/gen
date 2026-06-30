use std::collections::{HashMap, HashSet};

use petgraph::{Undirected, graph::NodeIndex, stable_graph::StableGraph};

use super::LayoutError;
use crate::layout::{LayoutEdge, LayoutNode, NodeRole};

/// Port directions for a node (North, East, South, West)
pub type PortDirections = (bool, bool, bool, bool);

/// Assigns ports to nodes based on their connectivity directions.
/// Returns a HashMap mapping NodeIndex to (N, E, S, W) boolean tuple.
#[allow(clippy::type_complexity)]
pub fn assign_ports(
    graph: &StableGraph<LayoutNode, LayoutEdge, Undirected>,
) -> HashMap<NodeIndex, PortDirections> {
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

        // Critical nodes are: non-straight routing nodes, or any Data/Stitch/Pin nodes.
        // Pin nodes are always critical (never contracted here) so they survive until
        // `prune_pin_stubs` runs and removes them explicitly.
        let is_critical = match &node.role {
            NodeRole::Data(_) | NodeRole::Stitch(_) | NodeRole::Pin => true,
            NodeRole::Routing => {
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

    // 2. Iterate through critical nodes and trace segments, preserving bundles
    // Store bundle for each simplified edge
    let mut new_edges_with_bundles: HashMap<(NodeIndex, NodeIndex), Vec<(NodeIndex, NodeIndex)>> =
        HashMap::new();

    for start_node_id in &critical_nodes {
        for neighbor_id in graph.neighbors(*start_node_id).collect::<Vec<_>>() {
            if critical_nodes.contains(&neighbor_id) {
                // Direct connection - preserve bundle from the existing edge
                let edge_id = graph.find_edge(*start_node_id, neighbor_id).unwrap();
                let bundle = graph.edge_weight(edge_id).unwrap().bundle.clone();

                let segment_endpoints = if start_node_id.index() < neighbor_id.index() {
                    (*start_node_id, neighbor_id)
                } else {
                    (neighbor_id, *start_node_id)
                };

                new_edges_with_bundles.insert(segment_endpoints, bundle);
            } else {
                let mut previous_id = *start_node_id;
                let mut current_id = neighbor_id;

                // Get bundle from first edge in segment
                let segment_bundle =
                    if let Some(edge_id) = graph.find_edge(*start_node_id, current_id) {
                        graph.edge_weight(edge_id).unwrap().bundle.clone()
                    } else {
                        vec![]
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

                        // Verify that bundle is identical along the segment
                        if let Some(edge_id) = graph.find_edge(current_id, next_node_id) {
                            let next_bundle = &graph.edge_weight(edge_id).unwrap().bundle;
                            assert_eq!(
                                &segment_bundle, next_bundle,
                                "Bundle mismatch in straight segment: expected {:?}, got {:?}",
                                segment_bundle, next_bundle
                            );
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

                // Add simplified edge with preserved bundle
                if let Some(end_node_id) = end_node_id
                    && critical_nodes.contains(&end_node_id)
                {
                    let segment_endpoints = if start_node_id.index() < end_node_id.index() {
                        (*start_node_id, end_node_id)
                    } else {
                        (end_node_id, *start_node_id)
                    };

                    new_edges_with_bundles.insert(segment_endpoints, segment_bundle);
                }
            }
        }
    }

    // Remove non-critical nodes
    let all_nodes: HashSet<_> = graph.node_indices().collect();
    let noncritical_nodes: Vec<_> = all_nodes.difference(&critical_nodes).cloned().collect();

    for node_index in noncritical_nodes {
        graph.remove_node(node_index);
    }

    // Add new edges with their preserved bundles (only if both nodes still exist)
    for ((source, target), bundle) in new_edges_with_bundles {
        if graph.node_weight(source).is_some()
            && graph.node_weight(target).is_some()
            && graph.find_edge(source, target).is_none()
        {
            let layout_edge = LayoutEdge { bundle };
            graph.add_edge(source, target, layout_edge);
        }
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
fn prune_pin(graph: &mut StableGraph<LayoutNode, LayoutEdge, Undirected>, pin_index: NodeIndex) {
    let neighbors: Vec<NodeIndex> = graph.neighbors(pin_index).collect();
    let mut to_remove = vec![pin_index];
    #[allow(clippy::type_complexity)]
    let mut splice: Option<(NodeIndex, NodeIndex, Vec<(NodeIndex, NodeIndex)>)> = None;

    if let [start] = neighbors.as_slice() {
        let mut previous = pin_index;
        let mut current = *start;
        let mut bundle = graph
            .find_edge(pin_index, current)
            .map(|edge| graph[edge].bundle.clone())
            .unwrap_or_default();

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
                        splice = Some((*a, *b, bundle));
                    }
                    // Perpendicular: the junction is a corner, leave it in place.
                    break;
                }
                _ => break, // dead end or a true 3+-way junction - leave it in place.
            }
        }
    }

    if let Some((a, b, bundle)) = splice
        && graph.find_edge(a, b).is_none()
    {
        graph.add_edge(a, b, LayoutEdge { bundle });
    }

    for node_index in to_remove {
        graph.remove_node(node_index);
    }
}
