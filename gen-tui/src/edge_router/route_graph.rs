use std::collections::HashMap;

use itertools::Itertools;
use log::{debug, info, trace};
use petgraph::{
    Undirected,
    graph::NodeIndex,
    stable_graph::StableGraph,
    visit::{EdgeRef, IntoEdgeReferences},
};

use super::{
    LayoutError, center_doglegs::center_doglegs, layout_graph_process::simplify_graph,
    route_layer::layout_layer,
};
use crate::{
    geometry::{LocalPos, PartitionIndex},
    layout::{LayoutEdge, LayoutNode, NodeRole},
};

pub(crate) fn make_rectilinear(
    graph: &mut StableGraph<LayoutNode, LayoutEdge, Undirected>,
) -> Result<(), LayoutError> {
    info!(
        "layout_graph: Starting with {} nodes, {} edges",
        graph.node_count(),
        graph.edge_count()
    );

    // Debug: Show detailed input graph structure
    log::debug!("=== EDGE ROUTER INPUT GRAPH ===");
    for node_idx in graph.node_indices() {
        if let Some(node) = graph.node_weight(node_idx) {
            log::debug!(
                "Input node {:?}: role={:?}, pos=({}, {}), size=({}, {}), layer={:?}",
                node_idx,
                node.role,
                node.pos.x,
                node.pos.y,
                node.size.0,
                node.size.1,
                node.layer
            );
        }
    }

    log::debug!("Input edges:");
    for edge_ref in graph.edge_references() {
        let source = edge_ref.source();
        let target = edge_ref.target();
        let edge_data = edge_ref.weight();
        log::debug!(
            "Input edge: {:?} -> {:?}, bundle={:?}",
            source,
            target,
            edge_data.bundle
        );
    }
    log::debug!("=== END INPUT GRAPH ===");

    // Extract partition index from first data node (assume they're all from the same partition)
    let partition_idx: PartitionIndex = graph
        .node_weights()
        .find_map(|layout_node| {
            matches!(layout_node.role, NodeRole::Data(_)).then_some(layout_node.pos.partition_idx)
        })
        .unwrap_or(0);

    // Normalize coordinates so first data node is at x=0
    let first_data_x = graph
        .node_weights()
        .filter_map(|node| matches!(node.role, NodeRole::Data(_)).then_some(node.pos.x))
        .min()
        .unwrap_or(0);

    for node in graph.node_weights_mut() {
        node.pos.x -= first_data_x;
    }

    // Sort by x-coordinate, then by y-coordinate and group nodes by x-coordinate
    let node_indices = graph
        .node_indices()
        .sorted_by_key(|&node_index| {
            let node = graph.node_weight(node_index).unwrap();
            (node.pos.x, node.pos.y)
        })
        .collect::<Vec<_>>();
    let node_indices_by_layer = node_indices
        .iter()
        .map(|node_index| {
            let node = graph.node_weight(*node_index).unwrap();
            (node.pos.x, *node_index)
        })
        .into_group_map();
    let mut combined_graph: StableGraph<LayoutNode, LayoutEdge, Undirected> =
        StableGraph::default();

    // Map from position to NodeIndex in combined_graph for deduplication
    let mut position_to_node_idx: HashMap<(i64, i64), NodeIndex> = HashMap::new();

    // Map from domain NodeIndex to combined_graph NodeIndex for Data node deduplication
    let mut domain_to_combined_idx: HashMap<NodeIndex, NodeIndex> = HashMap::new();

    // Map from layer_graph NodeIndex to combined_graph NodeIndex
    let mut layer_node_to_combined_idx: HashMap<NodeIndex, NodeIndex> = HashMap::new();

    let mut x_offset = 0;
    // Sort the keys to ensure deterministic iteration order
    let mut layer_keys: Vec<_> = node_indices_by_layer.keys().cloned().collect();
    layer_keys.sort_unstable();

    for (x_left, x_right) in layer_keys.iter().tuple_windows() {
        info!(
            "Processing layer pair: x_left={}, x_right={}",
            x_left, x_right
        );
        let left_node_indices = node_indices_by_layer.get(x_left).unwrap();
        let right_node_indices = node_indices_by_layer.get(x_right).unwrap();
        // Create mappings from NodeIndex to array index for layout_layer
        let left_idx_to_array_idx: HashMap<NodeIndex, usize> = left_node_indices
            .iter()
            .enumerate()
            .map(|(array_idx, &node_idx)| (node_idx, array_idx))
            .collect();
        let right_idx_to_array_idx: HashMap<NodeIndex, usize> = right_node_indices
            .iter()
            .enumerate()
            .map(|(array_idx, &node_idx)| (node_idx, array_idx))
            .collect();

        // Collect edges and their labels from the original graph
        #[allow(clippy::type_complexity)]
        let edges_with_bundles: Vec<((NodeIndex, NodeIndex), Vec<(NodeIndex, NodeIndex)>)> =
            left_node_indices
                .iter()
                .cartesian_product(right_node_indices.iter())
                .filter_map(|(node_index1, node_index2)| {
                    // Check for edge in either direction and get its bundle
                    let bundle = if let Some(edge_idx) = graph.find_edge(*node_index1, *node_index2)
                    {
                        graph.edge_weight(edge_idx).unwrap().bundle.clone()
                    } else if let Some(edge_idx) = graph.find_edge(*node_index2, *node_index1) {
                        graph.edge_weight(edge_idx).unwrap().bundle.clone()
                    } else {
                        return None; // No edge exists
                    };

                    // Convert from NodeIndex to array indices for layout_layer
                    let left_array_idx = *left_idx_to_array_idx.get(node_index1).unwrap();
                    let right_array_idx = *right_idx_to_array_idx.get(node_index2).unwrap();

                    Some((
                        (
                            NodeIndex::new(left_array_idx),
                            NodeIndex::new(right_array_idx),
                        ),
                        bundle,
                    ))
                })
                .collect();

        eprintln!(
            "DEBUG layer pair x_left={} x_right={} left_count={} right_count={} edges_found={}",
            x_left,
            x_right,
            left_node_indices.len(),
            right_node_indices.len(),
            edges_with_bundles.len()
        );

        // Separate edges and bundles for passing to layout_layer
        let edges: Vec<(NodeIndex, NodeIndex)> =
            edges_with_bundles.iter().map(|(e, _)| *e).collect();
        let edge_bundles: HashMap<(NodeIndex, NodeIndex), Vec<(NodeIndex, NodeIndex)>> =
            edges_with_bundles.into_iter().collect();
        // Grab the nodes to use in position calculations
        let left_nodes = left_node_indices
            .iter()
            .map(|index| graph.node_weight(*index).unwrap().clone())
            .collect::<Vec<_>>();
        let right_nodes = right_node_indices
            .iter()
            .map(|index| graph.node_weight(*index).unwrap().clone())
            .collect::<Vec<_>>();

        // Check if either side has any stitch node - if so, skip routing
        let has_stitch = left_nodes
            .iter()
            .chain(right_nodes.iter())
            .any(|node| matches!(node.role, NodeRole::Stitch(_)));

        debug!(
            "Layer pair analysis: left_layer={}, right_layer={}, left_nodes={}, right_nodes={}",
            x_left,
            x_right,
            left_nodes.len(),
            right_nodes.len()
        );
        debug!(
            "Left nodes: {:?}",
            left_nodes
                .iter()
                .map(|n| format!("{:?}", n.role))
                .collect::<Vec<_>>()
        );
        debug!(
            "Right nodes: {:?}",
            right_nodes
                .iter()
                .map(|n| format!("{:?}", n.role))
                .collect::<Vec<_>>()
        );

        let mut layer_graph = if has_stitch {
            // Skip rectilinear routing, simplification, compression, bundle creation, etc.
            // Create a simple layer graph equivalent
            let mut layer_graph = StableGraph::<LayoutNode, LayoutEdge, Undirected>::with_capacity(
                left_nodes.len() + right_nodes.len(),
                edges.len(),
            );

            // Add all nodes from left and right to the layer graph
            let mut node_mapping = HashMap::new();

            // Add left nodes
            for (i, node) in left_nodes.iter().enumerate() {
                let new_idx = layer_graph.add_node(node.clone());
                node_mapping.insert(left_node_indices[i], new_idx);
            }

            // Add right nodes
            for (i, node) in right_nodes.iter().enumerate() {
                let new_idx = layer_graph.add_node(node.clone());
                node_mapping.insert(right_node_indices[i], new_idx);
            }

            // Add edges directly without routing (maintaining direct stitch connections)
            for (left_idx, right_idx) in left_node_indices
                .iter()
                .cartesian_product(right_node_indices.iter())
            {
                if graph.find_edge(*left_idx, *right_idx).is_some()
                    || graph.find_edge(*right_idx, *left_idx).is_some()
                {
                    // Find the original edge to copy its bundle
                    let mut bundle = Vec::new();
                    if let Some(edge_idx) = graph.find_edge(*left_idx, *right_idx) {
                        bundle = graph.edge_weight(edge_idx).unwrap().bundle.clone();
                    } else if let Some(edge_idx) = graph.find_edge(*right_idx, *left_idx) {
                        bundle = graph.edge_weight(edge_idx).unwrap().bundle.clone();
                    }

                    let left_new_idx = node_mapping[left_idx];
                    let right_new_idx = node_mapping[right_idx];
                    debug!(
                        "make_rectilinear: adding direct edge for stitch connection (no routing nodes)"
                    );
                    layer_graph.add_edge(left_new_idx, right_new_idx, LayoutEdge { bundle });
                }
            }

            layer_graph
        } else {
            // Distance between the layers according to the layout algorithm
            // Computed as the space between the centers of the nodes,
            // not including the center points themselves.
            let putative_layer_distance = *x_right - *x_left - 1;

            trace!("putative_layer_distance: {}", putative_layer_distance);

            let node_width_left = left_nodes
                .iter()
                .map(|node| node.size.0)
                .max()
                .expect("By this point the sizes have been set");

            let node_width_right = right_nodes
                .iter()
                .map(|node| node.size.0)
                .max()
                .expect("By this point the sizes have been set");

            // right half of the node on the left of our current layer pair
            let left_label_extent = (node_width_left / 2) as i64;
            // left half of the node on the right of our current layer pair
            let right_label_extent = (node_width_right.div_ceil(2) - 1) as i64;
            // we use this combination of ceil/floor to make sure that when combined,
            // the exact node dimensions are reconstructed
            trace!("left_label_extent: {}", left_label_extent);
            trace!("right_label_extent: {}", right_label_extent);

            let slack = putative_layer_distance - left_label_extent - right_label_extent;

            debug!(
                "Coordinate calculations: node_width_left={}, node_width_right={}, left_label_extent={}, right_label_extent={}, slack={}",
                node_width_left, node_width_right, left_label_extent, right_label_extent, slack
            );

            eprintln!(
                "DEBUG before layout_layer: x_left={} x_right={} edges={:?} left_roles={:?} right_roles={:?}",
                x_left,
                x_right,
                edges,
                left_nodes.iter().map(|n| format!("{:?}", n.role)).collect::<Vec<_>>(),
                right_nodes.iter().map(|n| format!("{:?}", n.role)).collect::<Vec<_>>(),
            );
            // Rectilinear edge routing to replace the original edges
            let mut layer_graph = layout_layer(&left_nodes, &right_nodes, &edges, &edge_bundles)?;
            eprintln!(
                "DEBUG after layout_layer: node_count={} edge_count={}",
                layer_graph.node_count(),
                layer_graph.edge_count()
            );
            for nidx in layer_graph.node_indices() {
                if matches!(layer_graph[nidx].role, NodeRole::Pin) {
                    eprintln!(
                        "DEBUG after layout_layer: Pin {:?} degree={}",
                        nidx,
                        layer_graph.neighbors(nidx).count()
                    );
                }
            }
            simplify_graph(&mut layer_graph)?;
            for nidx in layer_graph.node_indices() {
                if matches!(layer_graph[nidx].role, NodeRole::Pin) {
                    eprintln!(
                        "DEBUG after simplify_graph (per-layer): Pin {:?} degree={}",
                        nidx,
                        layer_graph.neighbors(nidx).count()
                    );
                }
            }
            // Label the rectilinear edges with a reference to original edge(s) they represent
            make_bundles(&mut layer_graph, graph)?;
            center_doglegs(&mut layer_graph)?;

            // Measure the space required for the rectilinear edge routing
            let (min_x, max_x) = layer_graph
                .node_weights()
                .map(|node| node.pos.x)
                .minmax()
                .into_option()
                .unwrap_or((0, 0));

            // Compare allocated space vs needed space and adjust accordingly
            let interlayer_span = max_x - min_x - 1; // Space needed by routing nodes
            info!(
                "Routing space analysis: routing_graph_span=({} to {}), needed_space={}, available_slack={}",
                min_x, max_x, interlayer_span, slack
            );

            layer_graph
        };

        // Common processing for both branches - normalize and handle the layer_graph
        // Apply normalization to ensure consistent coordinate system
        let (min_x, _max_x) = layer_graph
            .node_weights()
            .map(|node| node.pos.x)
            .minmax()
            .into_option()
            .unwrap_or((0, 0));

        // Normalize the x-coordinates so the subgraph starts at x = 0
        for node in layer_graph.node_weights_mut() {
            node.pos.x -= min_x;
        }
        debug!(
            "Applied normalization - shifted {} units to start at x=0",
            min_x
        );
        layer_node_to_combined_idx.clear();
        let mut max_x = 0;
        for node_index in layer_graph.node_indices() {
            let layout_node = layer_graph.node_weight(node_index).unwrap();
            let (x, y) = (layout_node.pos.x, layout_node.pos.y);
            let new_x = x + x_offset;
            let adjusted_position = (new_x, y);

            // Check if a Data node already exists with this domain NodeIndex (role-based deduplication)
            let combined_idx = match &layout_node.role {
                NodeRole::Data(domain_idx) => {
                    if let Some(&existing_idx) = domain_to_combined_idx.get(domain_idx) {
                        // Data node with this domain NodeIndex already exists, reuse it
                        log::debug!(
                            "edge_router: reusing existing Data node for domain NodeIndex({}) -> combined NodeIndex({})",
                            domain_idx.index(),
                            existing_idx.index()
                        );
                        existing_idx
                    } else if let Some(&existing_idx) = position_to_node_idx.get(&adjusted_position)
                    {
                        // Node already exists at this position, reuse it
                        info!("Matched node ID: {}", existing_idx.index());
                        existing_idx
                    } else {
                        // Create new Data node in combined graph
                        let pos = LocalPos::new(partition_idx, adjusted_position.into());
                        let role = layout_node.role.clone();
                        let new_layout_node =
                            LayoutNode::new(role.clone(), pos, layout_node.size, layout_node.layer);
                        log::debug!(
                            "edge_router: adding new node with role={:?} at position ({}, {})",
                            role,
                            adjusted_position.0,
                            adjusted_position.1
                        );
                        let new_idx = combined_graph.add_node(new_layout_node);

                        // Store both position and domain mappings for Data nodes
                        position_to_node_idx.insert(adjusted_position, new_idx);
                        domain_to_combined_idx.insert(*domain_idx, new_idx);
                        max_x = max_x.max(new_x);
                        new_idx
                    }
                }
                _ => {
                    // For non-Data nodes (Stitch, Routing), use position-based deduplication only
                    if let Some(&existing_idx) = position_to_node_idx.get(&adjusted_position) {
                        // Node already exists at this position, reuse it
                        info!("Matched node ID: {}", existing_idx.index());
                        existing_idx
                    } else {
                        // Create new non-Data node in combined graph
                        let pos = LocalPos::new(partition_idx, adjusted_position.into());
                        let role = layout_node.role.clone();
                        let new_layout_node =
                            LayoutNode::new(role.clone(), pos, layout_node.size, layout_node.layer);
                        log::debug!(
                            "edge_router: adding new node with role={:?} at position ({}, {})",
                            role,
                            adjusted_position.0,
                            adjusted_position.1
                        );
                        let new_idx = combined_graph.add_node(new_layout_node);
                        position_to_node_idx.insert(adjusted_position, new_idx);
                        max_x = max_x.max(new_x);
                        new_idx
                    }
                }
            };

            // Store the mapping from layer node index to combined graph node index
            layer_node_to_combined_idx.insert(node_index, combined_idx);
        }

        // Add edges from the layer graph to the combined graph
        for edge_index in layer_graph.edge_indices() {
            if let Some((source_idx, target_idx)) = layer_graph.edge_endpoints(edge_index)
                && let Some(edge_data) = layer_graph.edge_weight(edge_index)
            {
                // Map the source and target to their combined graph node indices
                let combined_source_idx = *layer_node_to_combined_idx.get(&source_idx).unwrap();
                let combined_target_idx = *layer_node_to_combined_idx.get(&target_idx).unwrap();

                // Check if edge already exists (it shouldn't due to position deduplication, but let's be safe)
                if combined_graph
                    .find_edge(combined_source_idx, combined_target_idx)
                    .is_none()
                    && combined_graph
                        .find_edge(combined_target_idx, combined_source_idx)
                        .is_none()
                {
                    let layout_edge = LayoutEdge {
                        bundle: edge_data.bundle.clone(),
                    };
                    combined_graph.add_edge(combined_source_idx, combined_target_idx, layout_edge);
                }
            }
        }

        // After merging, find the new rightmost x-coordinate in combined graph
        let combined_max_x = combined_graph
            .node_weights()
            .map(|node| node.pos.x)
            .max()
            .unwrap_or(0);

        // Make sure the next iteration has their left nodes start on the same spot:
        x_offset = combined_max_x;
    }

    simplify_graph(&mut combined_graph)?;

    // Copy the combined graph back to the input graph
    *graph = combined_graph;
    Ok(())
}

/// Label routing paths within a layer using BFS to connect original nodes
/// This replaces the expensive late-stage BFS with early per-layer labeling
fn make_bundles(
    layer_graph: &mut StableGraph<LayoutNode, LayoutEdge, Undirected>,
    original_graph: &StableGraph<LayoutNode, LayoutEdge, Undirected>,
) -> Result<(), LayoutError> {
    // Find all original (non-routing) nodes in this layer
    let original_nodes: Vec<NodeIndex> = layer_graph
        .node_indices()
        .filter(|&idx| {
            if let Some(node) = layer_graph.node_weight(idx) {
                !matches!(node.role, NodeRole::Routing | NodeRole::Pin)
            } else {
                false
            }
        })
        .collect();

    // For each pair of original nodes, check if there's an original edge between them
    for &start_node in &original_nodes {
        for &end_node in &original_nodes {
            if start_node == end_node {
                continue;
            }

            // Get the original node indices to look up edges in the original graph
            let start_original_idx =
                if let Some(start_layout_node) = layer_graph.node_weight(start_node) {
                    match &start_layout_node.role {
                        NodeRole::Data(original_idx) => *original_idx,
                        _ => continue,
                    }
                } else {
                    continue;
                };

            let end_original_idx = if let Some(end_layout_node) = layer_graph.node_weight(end_node)
            {
                match &end_layout_node.role {
                    NodeRole::Data(original_idx) => *original_idx,
                    _ => continue,
                }
            } else {
                continue;
            };

            // Find the corresponding nodes in the original graph using LayoutNode roles
            let mut original_bundle = Vec::new();
            let mut found_edge = false;

            // Search for an edge between the original nodes in the original graph
            for edge_ref in original_graph.edge_references() {
                let source_layout = original_graph.node_weight(edge_ref.source()).unwrap();
                let target_layout = original_graph.node_weight(edge_ref.target()).unwrap();

                // Check if this edge connects our nodes (in either direction) by comparing Data roles
                let source_matches = matches!(&source_layout.role, NodeRole::Data(idx) if *idx == start_original_idx);
                let target_matches =
                    matches!(&target_layout.role, NodeRole::Data(idx) if *idx == end_original_idx);
                let source_matches_end =
                    matches!(&source_layout.role, NodeRole::Data(idx) if *idx == end_original_idx);
                let target_matches_start = matches!(&target_layout.role, NodeRole::Data(idx) if *idx == start_original_idx);

                if (source_matches && target_matches)
                    || (source_matches_end && target_matches_start)
                {
                    original_bundle = edge_ref.weight().bundle.clone();
                    found_edge = true;
                    break;
                }
            }

            if !found_edge {
                continue; // No original edge between these nodes
            };

            // Use BFS to find routing path between start_node and end_node
            let path = find_path_bfs(layer_graph, start_node, end_node);

            if let Some(node_path) = path {
                // Label all edges in the path with the original bundle
                for i in 0..node_path.len() - 1 {
                    let u = node_path[i];
                    let v = node_path[i + 1];

                    if let Some(edge_idx) = layer_graph.find_edge(u, v)
                        && let Some(edge_weight) = layer_graph.edge_weight_mut(edge_idx)
                    {
                        // Add the original bundle to this edge (avoiding duplicates)
                        for &label in &original_bundle {
                            if !edge_weight.bundle.contains(&label) {
                                edge_weight.bundle.push(label);
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

/// BFS to find path between two nodes in the layer graph
fn find_path_bfs(
    graph: &StableGraph<LayoutNode, LayoutEdge, Undirected>,
    start: NodeIndex,
    end: NodeIndex,
) -> Option<Vec<NodeIndex>> {
    use std::collections::{HashMap as StdHashMap, VecDeque};

    let mut queue = VecDeque::new();
    let mut visited = std::collections::HashSet::new();
    let mut predecessors = StdHashMap::new();

    queue.push_back(start);
    visited.insert(start);

    while let Some(current) = queue.pop_front() {
        if current == end {
            // Reconstruct path
            let mut path = Vec::new();
            let mut node = end;

            while node != start {
                path.push(node);
                node = predecessors[&node];
            }
            path.push(start);
            path.reverse();

            return Some(path);
        }

        for neighbor in graph.neighbors(current) {
            if !visited.contains(&neighbor) {
                visited.insert(neighbor);
                predecessors.insert(neighbor, current);
                queue.push_back(neighbor);
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use petgraph::{Undirected, graph::NodeIndex, stable_graph::StableGraph};

    use super::*;
    use crate::{
        geometry::LocalPos,
        layout::{LayoutEdge, LayoutNode, NodeRole},
    };

    #[test]
    fn test_linear_graph_increasing_widths() {
        // Initialize logging for the test
        let _ = env_logger::builder()
            .filter_level(log::LevelFilter::Info)
            .is_test(true)
            .try_init();

        println!("=== Starting test_linear_graph_increasing_widths ===");

        let mut graph = StableGraph::<LayoutNode, LayoutEdge, Undirected>::with_capacity(100, 99);
        let mut node_indices = Vec::new();

        // Create a linear graph with nodes of increasing widths (1 to 100)
        for i in 1..=100 {
            let width = i as u64;
            let height = 1u64; // Keep height constant for simplicity

            let node = LayoutNode::data(
                NodeIndex::new(i),
                LocalPos::new_xy(0, (i - 1) as i64, 0), // Position nodes linearly along x-axis
                (width, height),
                Some(0), // All nodes in same layer
            );

            let node_idx = graph.add_node(node);
            node_indices.push(node_idx);
        }

        // Connect nodes linearly (1-2, 2-3, 3-4, ..., 99-100)
        for i in 0..node_indices.len() - 1 {
            let source_domain_idx = NodeIndex::new(i + 1);
            let target_domain_idx = NodeIndex::new(i + 2);

            let edge = LayoutEdge::new(source_domain_idx, target_domain_idx);
            graph.add_edge(node_indices[i], node_indices[i + 1], edge);
        }

        println!(
            "Created linear graph with {} nodes and {} edges",
            graph.node_count(),
            graph.edge_count()
        );

        // Print first few and last few nodes to verify structure
        for (i, &node_idx) in node_indices.iter().take(5).enumerate() {
            let node = graph.node_weight(node_idx).unwrap();
            println!(
                "Node {}: width={}, pos=({}, {})",
                i + 1,
                node.size.0,
                node.pos.x,
                node.pos.y
            );
        }

        println!("...");

        for (i, &node_idx) in node_indices.iter().skip(95).enumerate() {
            let node = graph.node_weight(node_idx).unwrap();
            println!(
                "Node {}: width={}, pos=({}, {})",
                i + 96,
                node.size.0,
                node.pos.x,
                node.pos.y
            );
        }

        // Run the layout algorithm
        println!("=== About to call layout_graph ===");
        info!(
            "Calling layout_graph with {} nodes and {} edges",
            graph.node_count(),
            graph.edge_count()
        );
        let result = make_rectilinear(&mut graph);
        println!("=== layout_graph call completed ===");

        match result {
            Ok(()) => {
                println!("Layout algorithm completed successfully");
                println!(
                    "Final graph has {} nodes and {} edges",
                    graph.node_count(),
                    graph.edge_count()
                );

                // Print some statistics about the final layout
                let x_positions: Vec<i64> = graph.node_weights().map(|node| node.pos.x).collect();

                let min_x = x_positions.iter().min().unwrap_or(&0);
                let max_x = x_positions.iter().max().unwrap_or(&0);

                println!(
                    "Layout spans from x={} to x={} (width={})",
                    min_x,
                    max_x,
                    max_x - min_x
                );

                // Verify that original data nodes are still present
                let data_node_count = graph
                    .node_weights()
                    .filter(|node| matches!(node.role, NodeRole::Data(_)))
                    .count();

                assert_eq!(
                    data_node_count, 100,
                    "All 100 data nodes should be preserved"
                );

                // Check that routing nodes were added if needed
                let routing_node_count = graph
                    .node_weights()
                    .filter(|node| matches!(node.role, NodeRole::Routing))
                    .count();

                println!(
                    "Added {} routing nodes for edge routing",
                    routing_node_count
                );
            }
            Err(e) => {
                panic!("Layout algorithm failed: {:?}", e);
            }
        }
    }
}
