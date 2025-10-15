use std::collections::{HashMap, HashSet};

use itertools::Itertools;
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

        // Critical nodes are: non-straight routing nodes, or any Data/Stitch nodes
        let is_critical = match &node.role {
            NodeRole::Data(_) | NodeRole::Stitch(_) => true,
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

/// Compresses a graph along a specified axis (0 for x-axis/horizontal, 1 for y-axis/vertical)
/// by reducing gaps between consecutive layers.
/// Modifies the graph in place.
///
/// # Arguments
/// * `graph` - The graph to compress
/// * `axis` - 0 for x-axis/horizontal, 1 for y-axis/vertical
/// * `minimum_spacing` - Minimum spacing between layers
/// * `account_for_node_dimensions` - If true, adds spacing based on node sizes (legacy behavior).
///   If false, only uses minimum_spacing (avoids double-counting when layout already accounts for dimensions).
pub fn compress_graph(
    graph: &mut StableGraph<LayoutNode, LayoutEdge, Undirected>,
    axis: i64,
    minimum_spacing: i64,
    account_for_node_dimensions: bool,
) -> Result<(), LayoutError> {
    if graph.node_count() == 0 {
        return Ok(());
    }

    // Group nodes by their position as projected on the x or y axis
    let mut nodes_by_layer: HashMap<i64, Vec<NodeIndex>> = HashMap::new();
    for node_index in graph.node_indices() {
        let node = graph.node_weight(node_index).unwrap();
        let layer_pos = if axis == 0 { node.pos.x } else { node.pos.y };

        nodes_by_layer
            .entry(layer_pos)
            .or_default()
            .push(node_index);
    }

    // Get sorted layer positions
    let layer_positions: Vec<i64> = nodes_by_layer.keys().copied().sorted().collect();

    if layer_positions.len() <= 1 {
        return Ok(());
    }

    // Find the maximum node size in each layer
    let layer_max_sizes: Vec<i64> = layer_positions
        .iter()
        .map(|&layer_pos| {
            nodes_by_layer[&layer_pos]
                .iter()
                .map(|&idx| {
                    let node = graph.node_weight(idx).unwrap();
                    if axis == 0 {
                        node.size.0 as i64
                    } else {
                        node.size.1 as i64
                    }
                })
                .max()
                .unwrap_or(1)
        })
        .collect();

    // Compute new positions for each layer
    let mut new_positions: Vec<i64> = Vec::new();
    new_positions.push(layer_positions[0]);

    for i in 1..layer_positions.len() {
        let required_spacing = if account_for_node_dimensions {
            // Nodes are positioned asymmetrically by BigRect::from_center_and_size:
            // For even widths, they extend more to the right than left.
            // Right extent (side facing next layer) = width / 2 (floor division)
            // Left extent (side facing prev layer) = div_ceil(width, 2) - 1
            let prev_layer_half_size = layer_max_sizes[i - 1] / 2; // Right extent
            let curr_layer_half_size = (layer_max_sizes[i] + 1) / 2 - 1; // Left extent (div_ceil - 1)
            // Add 1 to ensure at least one cell gap between nodes for edge routing
            prev_layer_half_size + curr_layer_half_size + 1 + minimum_spacing
        } else {
            minimum_spacing
        };
        let new_pos = new_positions[i - 1] + required_spacing;
        new_positions.push(new_pos);
    }

    // Update all nodes in each layer to their new positions
    for (old_pos, new_pos) in layer_positions.iter().zip(new_positions.iter()) {
        for &node_index in &nodes_by_layer[old_pos] {
            if let Some(node) = graph.node_weight_mut(node_index) {
                if axis == 0 {
                    node.pos.x = *new_pos;
                } else {
                    node.pos.y = *new_pos;
                }
            }
        }
    }

    Ok(())
}
