use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
};

use itertools::Itertools;
use petgraph::stable_graph::StableGraph;

use super::{EdgeData, LayoutError, NodeData, temp_graph::TempGraph};

pub fn preprocess_graph(graph: &mut TempGraph) {
    // Normalizes the coordinates so that the minimum coordinate is 0.
    let min_x = graph.nodes().map(|node| node.position.0).min();
    let min_y = graph.nodes().map(|node| node.position.1).min();

    for node_index in graph.node_indices() {
        let mut node = graph.get_node(node_index).unwrap();
        let (x, y) = node.position;
        let new_x = if let Some(min_x) = min_x {
            x - min_x
        } else {
            x
        };
        let new_y = if let Some(min_y) = min_y {
            y - min_y
        } else {
            y
        };

        node.position = (new_x, new_y);
        graph.add_node(node_index, node);
    }
}

pub fn assign_ports(graph: &mut TempGraph) -> Result<(), LayoutError> {
    // Records from which sides a node is connected to its neighbors.
    // Assumes that the graph has a 'pos' attribute for each node.
    // Assigns a 'ports' attribute to each node with a (N, E, S, W) tuple (all booleans).
    // Modifies the graph in place.
    for node_index in graph.node_indices() {
        let node = graph.get_node(node_index).unwrap();
        let (x, y) = node.position;

        let (mut north, mut east, mut south, mut west) = (false, false, false, false);

        for neighbor in graph.neighbors(node_index) {
            let (neighbor_x, neighbor_y) = graph.get_node(neighbor).unwrap().position;
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

        let mut changed_node = graph.get_node(node_index).unwrap();
        changed_node.ports = Some((north, east, south, west));
        graph.update_node(node_index, changed_node)?;
    }

    Ok(())
}

fn ports_to_glyph_index(ports: (bool, bool, bool, bool)) -> i64 {
    // Converts a (N, E, S, W) port tuple to a glyph index (0-15).
    // N is MSB, W is LSB.
    let (n, e, s, w) = ports;
    (n as i64) << (3 + (e as i64)) << (2 + (s as i64)) << (1 + w as i64)
}

pub fn assign_glyph_idx(graph: &mut TempGraph) -> Result<(), LayoutError> {
    for node_index in graph.node_indices() {
        let mut node = graph.get_node(node_index).unwrap();
        // Remove the old ports attribute
        let ports = node.ports.unwrap_or((false, false, false, false));

        if node.node_type == Some("Routing".to_string()) {
            node.glyph_index = Some(ports_to_glyph_index(ports));
        }

        graph.update_node(node_index, node)?;
    }

    Ok(())
}

pub fn simplify_graph(graph: &mut TempGraph) -> Result<(), LayoutError> {
    // Simplifies a graph by identifying and contracting segments with collinear edges.
    if graph.edge_count() == 0 {
        return Ok(());
    }

    // Define straight orientations (these get contracted)
    const STRAIGHT_VERTICAL: (bool, bool, bool, bool) = (true, false, true, false);
    const STRAIGHT_HORIZONTAL: (bool, bool, bool, bool) = (false, true, false, true);

    // 1. Identify and add all critical nodes (not on a collinear segment)
    let mut critical_nodes = HashSet::new();

    for node_index in graph.node_indices() {
        let node = graph.get_node(node_index).unwrap();
        let orientation = node.ports.unwrap_or((false, false, false, false));

        let node_type = if let Some(node_type) = &node.node_type {
            node_type
        } else {
            &"Graph".to_string()
        };

        if (orientation != STRAIGHT_HORIZONTAL && orientation != STRAIGHT_VERTICAL)
            || (node_type == "Graph" || node_type == "Stitching")
        {
            critical_nodes.insert(node_index);
        }
    }

    // Handle case where graph might be a single, straight segment
    if critical_nodes.is_empty() && graph.node_count() > 0 {
        let start_node_id = graph.node_indices().min().unwrap();
        critical_nodes.insert(start_node_id);
    }

    let mut processed_segments = HashSet::new();

    // 2. Iterate through critical nodes and trace segments using the 'ports' attribute
    let mut new_edges = HashSet::new();
    for start_node_id in &critical_nodes {
        for neighbor_id in graph.neighbors(*start_node_id) {
            if critical_nodes.contains(&neighbor_id) {
                // Direct connection
                let segment_endpoints = if *start_node_id < neighbor_id {
                    (*start_node_id, neighbor_id)
                } else {
                    (neighbor_id, *start_node_id)
                };
                processed_segments.insert(segment_endpoints);
            } else {
                let mut previous_id = *start_node_id;
                let mut current_id = neighbor_id;

                let end_node_id;

                loop {
                    let current_node = graph.get_node(current_id).unwrap();
                    let current_orientation =
                        current_node.ports.unwrap_or((false, false, false, false));

                    // Start of a straight segment
                    if current_orientation == STRAIGHT_HORIZONTAL
                        || current_orientation == STRAIGHT_VERTICAL
                    {
                        let neighbors_of_current: Vec<_> =
                            graph.neighbors(current_id).into_iter().collect();
                        if neighbors_of_current.len() != 2 {
                            end_node_id = Some(current_id);
                            break;
                        }

                        let next_node_id = if neighbors_of_current[1] == previous_id {
                            neighbors_of_current[0]
                        } else {
                            neighbors_of_current[1]
                        };
                        previous_id = current_id;
                        current_id = next_node_id;
                        if critical_nodes.contains(&current_id) {
                            end_node_id = Some(current_id);
                            break;
                        }
                    } else {
                        end_node_id = Some(current_id);
                        break;
                    }
                }

                // Add simplified edge
                if let Some(end_node_id) = end_node_id
                    && critical_nodes.contains(&end_node_id)
                {
                    let segment_endpoints = if *start_node_id < end_node_id {
                        (*start_node_id, end_node_id)
                    } else {
                        (end_node_id, *start_node_id)
                    };

                    processed_segments.insert(segment_endpoints);
                    new_edges.insert(segment_endpoints);
                }
            }
        }
    }

    let all_nodes = graph.node_indices().collect::<HashSet<_>>();
    let noncritical_nodes = all_nodes
        .difference(&critical_nodes)
        .collect::<HashSet<_>>();
    for node_index in noncritical_nodes {
        graph.remove_node(*node_index);
    }

    let mut edge_id_counter = graph.edge_indices().max().unwrap() + 1;
    for (source, target) in new_edges {
        if graph.find_edge(source, target).is_none() {
            graph.add_edge(edge_id_counter, source, target, EdgeData { role: None })?;
            edge_id_counter += 1;
        }
    }

    Ok(())
}

pub fn compress_graph(
    graph: &mut TempGraph,
    axis: i64,
    minimum_spacing: i64,
    margins: Option<(i64, i64)>,
) -> Result<(), LayoutError> {
    // Compresses a graph along a specified axis (0 for x-axis/horizontal, 1 for y-axis/vertical)
    // by reducing gaps between consecutive coordinate groups to a minimum_spacing.
    // If margins are provided (tuple of ints), the first and last gap will be set to
    // the first and second elements of the tuple, respectively, regardless of the minimum spacing.
    // Modifies the graph in place.

    if graph.node_count() == 0 {
        return Ok(());
    }

    // Get all unique coordinates on the specified axis and sort them
    let unique_coordinates = graph
        .nodes()
        .map(|node| {
            if axis == 0 {
                node.position.0
            } else {
                node.position.1
            }
        })
        .collect::<HashSet<i64>>();
    let coordinates = unique_coordinates
        .iter()
        .copied()
        .sorted()
        .collect::<Vec<i64>>();

    if coordinates.is_empty() {
        return Ok(());
    }

    // Group nodes by their coordinate on the specified axis
    let mut nodes_by_coordinate = HashMap::new();
    for node_index in graph.node_indices() {
        let node = graph.get_node(node_index).unwrap();
        let coordinate = if axis == 0 {
            node.position.0
        } else {
            node.position.1
        };

        nodes_by_coordinate
            .entry(coordinate)
            .or_insert(vec![])
            .push(node_index);
    }

    let new_coordinates = if let Some(margins) = margins {
        match coordinates.len().cmp(&2) {
            Ordering::Less => {
                return Err(LayoutError::InvalidCoordinateNumber(
                    coordinates.len() as i64
                ));
            }
            Ordering::Greater => {
                let mut dynamic_spacing = (0..coordinates.len())
                    .map(|_| minimum_spacing + 1)
                    .collect::<Vec<i64>>();
                dynamic_spacing[0] = margins.0 + 1;
                let dynamic_spacing_len = dynamic_spacing.len();
                dynamic_spacing[dynamic_spacing_len - 1] = margins.1 + 1;
                (0..coordinates.len())
                    .map(|i| coordinates[0] + dynamic_spacing[0..i].iter().sum::<i64>())
                    .collect::<Vec<i64>>()
            }
            Ordering::Equal => vec![
                coordinates[0],
                coordinates[0] + margins.0 + margins.1 + minimum_spacing + 1,
            ],
        }
    } else {
        // The new coordinates are evenly spaced, starting from the first coordinate
        (0..coordinates.len())
            .map(|i| coordinates[0] + (i as i64 * (minimum_spacing + 1)))
            .collect::<Vec<i64>>()
    };

    // Update node positions in place
    for (old_coordinate, new_coordinate) in coordinates.iter().zip(new_coordinates.iter()) {
        for node_index in nodes_by_coordinate.entry(*old_coordinate).or_default() {
            let mut node = graph.get_node(*node_index).unwrap();
            // Create a new position tuple, updating only the specified axis
            node.position = if axis == 0 {
                (*new_coordinate, node.position.1)
            } else {
                (node.position.0, *new_coordinate)
            };

            graph.update_node(*node_index, node)?;
        }
    }

    Ok(())
}

pub fn pad_graph(graph: &mut StableGraph<NodeData, EdgeData>, left: i64, right: i64) {
    // Identifies the columns of nodes that are most to the left and right,
    // and inserts padding (space) between them and the rest of the graph.
    if graph.node_count() == 0 {
        return;
    }

    let min_x = graph
        .node_weights()
        .map(|node| node.position.0)
        .min()
        .unwrap();
    let max_x = graph
        .node_weights()
        .map(|node| node.position.0)
        .max()
        .unwrap();

    for node in graph.node_weights_mut() {
        let mut delta = 0;
        if node.position.0 > min_x {
            delta += left;
        }
        if node.position.0 == max_x {
            delta += right;
        }

        node.position = (node.position.0 + delta, node.position.1);
    }
}

pub fn ensure_padding(graph: &mut StableGraph<NodeData, EdgeData>, left: i64, right: i64) {
    // Identifies the columns of nodes that are most to the left and right,
    // and inserts padding (space) between them and the rest of the graph.

    if graph.node_count() == 0 {
        return;
    }

    let x_coordinates = graph
        .node_weights()
        .map(|node| node.position.0)
        .sorted()
        .collect::<Vec<i64>>();

    // Get the current spacing between the first and second column of nodes,
    // and the spacing between the last and second-to-last column of nodes.
    // If the spacing is less than the minimum spacing, we need to add padding.

    let delta_left = if x_coordinates[1] - x_coordinates[0] < left {
        left - (x_coordinates[1] - x_coordinates[0])
    } else {
        0
    };

    let x_coordinates_length = x_coordinates.len();
    let delta_right = if x_coordinates[x_coordinates_length - 1]
        - x_coordinates[x_coordinates_length - 2]
        < right
    {
        right - (x_coordinates[x_coordinates_length - 1] - x_coordinates[x_coordinates_length - 2])
    } else {
        0
    };

    for node in graph.node_weights_mut() {
        let mut delta = 0;
        if node.position.0 > x_coordinates[0] {
            delta += delta_left;
        }
        if node.position.0 == x_coordinates[x_coordinates_length - 1] {
            delta += delta_right;
        }

        node.position = (node.position.0 + delta, node.position.1);
    }
}
