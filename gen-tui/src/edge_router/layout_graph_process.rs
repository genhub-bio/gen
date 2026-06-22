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

    let pin_idx_for_debug: Vec<NodeIndex> = graph
        .node_indices()
        .filter(|&idx| matches!(graph[idx].role, NodeRole::Pin))
        .collect();
    for &idx in &pin_idx_for_debug {
        eprintln!(
            "DEBUG simplify_graph ENTRY: Pin {:?} degree={}",
            idx,
            graph.neighbors(idx).count()
        );
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

    for &idx in &pin_idx_for_debug {
        if graph.node_weight(idx).is_some() {
            eprintln!(
                "DEBUG simplify_graph EXIT: Pin {:?} degree={}",
                idx,
                graph.neighbors(idx).count()
            );
        } else {
            eprintln!("DEBUG simplify_graph EXIT: Pin {:?} was REMOVED", idx);
        }
    }

    Ok(())
}

/// Removes the synthetic `NodeRole::Pin` nodes used to render a backward edge as a
/// full-width loop, once routing has finished and `simplify_graph` has already
/// contracted ordinary straight runs (`Pin` is always "critical" there, so it survives
/// untouched until this pass runs).
///
/// A pin always has exactly two neighbors - one toward the data node it pins in place,
/// one continuing the long bypass edge - but, having both on the same side in x, it's a
/// same-side bend that `simplify_graph` cannot contract (its contraction only collapses
/// straight, opposite-side runs). This pass removes each pin and splices its two sides
/// back into one continuous edge, walking outward past any further bend-only `Routing`
/// nodes on either side until it reaches real content (`Data`/`Stitch`) or a genuine
/// junction (any node without exactly one way to continue) - the visible result is the
/// pin's two edges merging into a single edge spanning the whole loop. A no-op for any
/// graph with no backward edges, since no `Pin` node is ever injected for one.
pub fn prune_pin_stubs(graph: &mut StableGraph<LayoutNode, LayoutEdge, Undirected>) {
    let pins: Vec<NodeIndex> = graph
        .node_indices()
        .filter(|&node_index| matches!(graph[node_index].role, NodeRole::Pin))
        .collect();

    for pin_index in pins {
        // A pin's own stub may already have been removed while walking out from a
        // previously processed pin (e.g. the two ends of a tiny loop meeting directly).
        if graph.node_weight(pin_index).is_some() {
            prune_pin(graph, pin_index);
        }
    }
}

/// One side of a chain walked outward from a pin: the real node it ends at (if the walk
/// didn't dead-end), the bend-only `Routing` nodes passed through (to be removed), and
/// the merged bundle data of all edges traversed.
struct ChainEnd {
    terminal: Option<NodeIndex>,
    visited: Vec<NodeIndex>,
    bundle: Vec<(NodeIndex, NodeIndex)>,
}

/// Walks outward from `start` (excluding the direction back towards `came_from`) through
/// a chain of degree-2 `Routing` nodes, stopping at the first node that isn't a plain
/// `Routing` node (real content, a junction, or another `Pin`).
fn walk_chain(
    graph: &StableGraph<LayoutNode, LayoutEdge, Undirected>,
    came_from: NodeIndex,
    start: NodeIndex,
) -> ChainEnd {
    let mut previous = came_from;
    let mut current = start;
    let mut visited = Vec::new();
    let mut bundle = graph
        .find_edge(previous, current)
        .map(|edge| graph[edge].bundle.clone())
        .unwrap_or_default();

    loop {
        if !matches!(graph[current].role, NodeRole::Routing) {
            return ChainEnd {
                terminal: Some(current),
                visited,
                bundle,
            };
        }

        let other_neighbors: Vec<NodeIndex> = graph
            .neighbors(current)
            .filter(|&neighbor| neighbor != previous)
            .collect();

        match other_neighbors.as_slice() {
            [only] => {
                visited.push(current);
                if let Some(edge) = graph.find_edge(current, *only) {
                    bundle.extend(graph[edge].bundle.clone());
                }
                previous = current;
                current = *only;
            }
            _ => {
                // Either a dead end (no further neighbors) or a real junction (2+ other
                // neighbors) - leave it in place rather than removing/resplicing it.
                return ChainEnd {
                    terminal: Some(current),
                    visited,
                    bundle,
                };
            }
        }
    }
}

/// Removes `pin_index` and splices its two sides into one continuous edge. See
/// `prune_pin_stubs`.
fn prune_pin(graph: &mut StableGraph<LayoutNode, LayoutEdge, Undirected>, pin_index: NodeIndex) {
    let neighbors: Vec<NodeIndex> = graph.neighbors(pin_index).collect();
    let [first, second] = neighbors.as_slice() else {
        // Not the expected two-neighbor pin shape - just remove it.
        graph.remove_node(pin_index);
        return;
    };

    let left = walk_chain(graph, pin_index, *first);
    let right = walk_chain(graph, pin_index, *second);

    let mut to_remove = vec![pin_index];
    to_remove.extend(left.visited);
    to_remove.extend(right.visited);

    if let (Some(left_terminal), Some(right_terminal)) = (left.terminal, right.terminal)
        && left_terminal != right_terminal
        && graph.find_edge(left_terminal, right_terminal).is_none()
    {
        let mut merged_bundle = left.bundle;
        merged_bundle.extend(right.bundle);
        graph.add_edge(
            left_terminal,
            right_terminal,
            LayoutEdge {
                bundle: merged_bundle,
            },
        );
    }

    for node_index in to_remove {
        graph.remove_node(node_index);
    }
}
