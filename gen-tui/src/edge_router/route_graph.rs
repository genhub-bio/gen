use std::collections::{HashMap, HashSet, VecDeque};

use itertools::Itertools;
use log::{debug, info, trace};
use petgraph::{
    Undirected,
    graph::NodeIndex,
    stable_graph::StableGraph,
    visit::{EdgeRef, IntoEdgeReferences},
};

use super::{
    LayoutError,
    center_doglegs::center_doglegs,
    layout_graph_process::{BundledLeg, BundledLegEdges, simplify_graph},
    route_layer::layout_layer,
};
use crate::{
    geometry::LocalPos,
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

    // A single column has no layer pair to route between, so the loop below never runs and
    // never populates `combined_graph`. Leave the graph intact.
    if layer_keys.len() < 2 {
        return Ok(());
    }

    for (x_left, x_right) in layer_keys.iter().tuple_windows() {
        info!(
            "Processing layer pair: x_left={}, x_right={}",
            x_left, x_right
        );
        let all_left_indices = node_indices_by_layer.get(x_left).unwrap();
        let all_right_indices = node_indices_by_layer.get(x_right).unwrap();

        // A boundary node with no edge inside this layer pair contributes nothing to the pair's
        // routing, yet `layout_layer` still emits a stray routing stub off it. When that node is
        // a pin, the stub gives it a second neighbour, so `prune_pin_stubs` reads the pin as a
        // load-bearing corner and leaves the stub in the render. Route only the nodes
        // that participate in an edge here; a skipped node still appears in its other adjacent
        // layer pair, where its edges live (a wormhole node's one edge to its boundary is a real
        // edge like any other, so it's naturally picked up by whichever adjacent pair it falls
        // into - no separate bookkeeping needed).
        let has_pair_edge = |node: NodeIndex, others: &[NodeIndex]| {
            others.iter().any(|&other| {
                graph.find_edge(node, other).is_some() || graph.find_edge(other, node).is_some()
            })
        };
        let left_node_indices: Vec<NodeIndex> = all_left_indices
            .iter()
            .copied()
            .filter(|&node| has_pair_edge(node, all_right_indices))
            .collect();
        let right_node_indices: Vec<NodeIndex> = all_right_indices
            .iter()
            .copied()
            .filter(|&node| has_pair_edge(node, all_left_indices))
            .collect();

        // With every edge-bearing node filtered out there is nothing to route between these two
        // columns; the surviving nodes are picked up by their other adjacent pair.
        if left_node_indices.is_empty() || right_node_indices.is_empty() {
            continue;
        }

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

        // Collect edges and their labels (bundle and is_backward_span flag) from the original graph
        let edges_with_bundles: Vec<((NodeIndex, NodeIndex), BundledLeg)> = left_node_indices
            .iter()
            .cartesian_product(right_node_indices.iter())
            .filter_map(|(node_index1, node_index2)| {
                // Check for edge in either direction and get its bundle and flag
                let (bundle, is_backward_span) =
                    if let Some(edge_idx) = graph.find_edge(*node_index1, *node_index2) {
                        let edge = graph.edge_weight(edge_idx).unwrap();
                        (edge.bundle.clone(), edge.is_backward_span)
                    } else if let Some(edge_idx) = graph.find_edge(*node_index2, *node_index1) {
                        let edge = graph.edge_weight(edge_idx).unwrap();
                        (edge.bundle.clone(), edge.is_backward_span)
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
                    (bundle, is_backward_span),
                ))
            })
            .collect();

        // Separate edges and bundles for passing to layout_layer
        let edges: Vec<(NodeIndex, NodeIndex)> =
            edges_with_bundles.iter().map(|(e, _)| *e).collect();
        let edge_bundles: BundledLegEdges = edges_with_bundles.into_iter().collect();
        // Grab the nodes to use in position calculations
        let left_nodes = left_node_indices
            .iter()
            .map(|index| graph.node_weight(*index).unwrap().clone())
            .collect::<Vec<_>>();
        let right_nodes = right_node_indices
            .iter()
            .map(|index| graph.node_weight(*index).unwrap().clone())
            .collect::<Vec<_>>();

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

        let mut layer_graph = {
            // Distance between the layers according to the layout algorithm
            // Computed as the space between the centers of the nodes,
            // not including the center points themselves.
            let putative_layer_distance = *x_right - *x_left - 1;

            trace!("putative_layer_distance: {}", putative_layer_distance);

            let node_width_left = left_nodes
                .iter()
                .map(|node| node.size.0)
                .max()
                .expect("should have left-node sizes");

            let node_width_right = right_nodes
                .iter()
                .map(|node| node.size.0)
                .max()
                .expect("should have right-node sizes");

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

            // Rectilinear edge routing to replace the original edges
            let mut layer_graph = layout_layer(&left_nodes, &right_nodes, &edges, &edge_bundles)?;
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
                        let pos = LocalPos::new(adjusted_position.into());
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
                    // For non-Data nodes (Routing, Pin), use position-based deduplication only
                    if let Some(&existing_idx) = position_to_node_idx.get(&adjusted_position) {
                        // Node already exists at this position, reuse it
                        info!("Matched node ID: {}", existing_idx.index());
                        existing_idx
                    } else {
                        // Create new non-Data node in combined graph
                        let pos = LocalPos::new(adjusted_position.into());
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

                // Two distinct nodes in this layer's local graph can collapse onto the
                // same combined-graph node when their adjusted positions coincide,
                // which would turn a real edge into a self-loop. Skip it rather than
                // add one.
                //
                // Check if edge already exists (it shouldn't due to position deduplication, but let's be safe)
                if combined_source_idx != combined_target_idx
                    && combined_graph
                        .find_edge(combined_source_idx, combined_target_idx)
                        .is_none()
                    && combined_graph
                        .find_edge(combined_target_idx, combined_source_idx)
                        .is_none()
                {
                    let layout_edge = LayoutEdge {
                        bundle: edge_data.bundle.clone(),
                        is_backward_span: edge_data.is_backward_span,
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

/// Copy original edge bundles onto their routed paths.
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

/// Find a path between two nodes in the layer graph.
fn find_path_bfs(
    graph: &StableGraph<LayoutNode, LayoutEdge, Undirected>,
    start: NodeIndex,
    end: NodeIndex,
) -> Option<Vec<NodeIndex>> {
    let mut queue = VecDeque::new();
    let mut visited = HashSet::new();
    let mut predecessors = HashMap::new();

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
        let mut graph = StableGraph::<LayoutNode, LayoutEdge, Undirected>::with_capacity(100, 99);
        let mut node_indices = Vec::new();

        for i in 1..=100 {
            let width = i as u64;
            let node = LayoutNode::data(
                NodeIndex::new(i),
                LocalPos::new_xy((i - 1) as i64, 0),
                (width, 1),
                Some(0),
            );
            node_indices.push(graph.add_node(node));
        }

        for i in 0..node_indices.len() - 1 {
            let source_domain_idx = NodeIndex::new(i + 1);
            let target_domain_idx = NodeIndex::new(i + 2);
            let edge = LayoutEdge::new(source_domain_idx, target_domain_idx);
            graph.add_edge(node_indices[i], node_indices[i + 1], edge);
        }

        make_rectilinear(&mut graph).expect("should route the graph");

        let data_node_count = graph
            .node_weights()
            .filter(|node| matches!(node.role, NodeRole::Data(_)))
            .count();
        assert_eq!(data_node_count, 100);
    }

    /// A wormhole door is a real node from crawl time onward (see
    /// `crawl::build_window_graph`), not synthesized here - `make_rectilinear` must treat it
    /// exactly like any other degree-1 node: route it and leave it as exactly one node, even
    /// when its boundary also participates in another adjacent layer pair.
    #[test]
    fn test_make_rectilinear_preserves_a_real_wormhole_node() {
        let mut graph = StableGraph::<LayoutNode, LayoutEdge, Undirected>::default();

        let d0 = LayoutNode::data(NodeIndex::new(100), LocalPos::new_xy(0, 0), (1, 1), Some(0));
        let d1 = LayoutNode::data(NodeIndex::new(101), LocalPos::new_xy(1, 0), (1, 1), Some(1));
        let d2 = LayoutNode::data(NodeIndex::new(102), LocalPos::new_xy(2, 0), (1, 1), Some(2));

        let n0 = graph.add_node(d0);
        let n1 = graph.add_node(d1);
        let n2 = graph.add_node(d2);

        graph.add_edge(
            n0,
            n1,
            LayoutEdge::new(NodeIndex::new(100), NodeIndex::new(101)),
        );
        graph.add_edge(
            n1,
            n2,
            LayoutEdge::new(NodeIndex::new(101), NodeIndex::new(102)),
        );

        let external_target = NodeIndex::new(999);
        // Adjacent to its boundary's own column (x=1), on a distinct row from `n2` (also at
        // x=2) - a real Sugiyama layering would never place a wormhole node's only edge more
        // than one rank away from its boundary without inserting dummy vertices to fill the
        // gap, so this mirrors what `make_rectilinear` actually receives in production.
        let wormhole = graph.add_node(LayoutNode::new(
            NodeRole::Wormhole(external_target),
            LocalPos::new_xy(2, 1),
            (1, 1),
            Some(2),
        ));
        graph.add_edge(
            n1,
            wormhole,
            LayoutEdge::new(NodeIndex::new(101), external_target),
        );

        make_rectilinear(&mut graph).expect("should route the graph");

        let wormhole_count = graph
            .node_weights()
            .filter(
                |node| matches!(node.role, NodeRole::Wormhole(target) if target == external_target),
            )
            .count();
        assert_eq!(
            wormhole_count, 1,
            "a real wormhole node must survive routing as exactly one node"
        );
    }

    /// A wormhole door collapsing several off-window neighbours (see
    /// `crawl::neighborhood`'s external-edge collapsing) carries every one of them on its
    /// edge's bundle from crawl time onward - `make_rectilinear` must not lose any of them.
    #[test]
    fn test_make_rectilinear_keeps_a_wormhole_edges_full_bundle() {
        let mut graph = StableGraph::<LayoutNode, LayoutEdge, Undirected>::default();

        let d0 = LayoutNode::data(NodeIndex::new(100), LocalPos::new_xy(0, 0), (1, 1), Some(0));
        let d1 = LayoutNode::data(NodeIndex::new(101), LocalPos::new_xy(1, 0), (1, 1), Some(1));
        let d2 = LayoutNode::data(NodeIndex::new(102), LocalPos::new_xy(2, 0), (1, 1), Some(2));
        let n0 = graph.add_node(d0);
        let n1 = graph.add_node(d1);
        let n2 = graph.add_node(d2);
        graph.add_edge(
            n0,
            n1,
            LayoutEdge::new(NodeIndex::new(100), NodeIndex::new(101)),
        );
        graph.add_edge(
            n1,
            n2,
            LayoutEdge::new(NodeIndex::new(101), NodeIndex::new(102)),
        );

        let boundary = NodeIndex::new(101);
        let chosen_target = NodeIndex::new(201);
        let all_collapsed = [
            NodeIndex::new(201),
            NodeIndex::new(202),
            NodeIndex::new(203),
        ];
        let bundle: Vec<(NodeIndex, NodeIndex)> = all_collapsed
            .iter()
            .map(|&target| (boundary, target))
            .collect();
        let wormhole = graph.add_node(LayoutNode::new(
            NodeRole::Wormhole(chosen_target),
            LocalPos::new_xy(2, 1),
            (1, 1),
            Some(2),
        ));
        graph.add_edge(
            n1,
            wormhole,
            LayoutEdge {
                bundle: bundle.clone(),
                is_backward_span: false,
            },
        );

        make_rectilinear(&mut graph).expect("should route the graph");

        let stub_index = graph
            .node_indices()
            .find(|&index| {
                matches!(graph[index].role, NodeRole::Wormhole(target) if target == chosen_target)
            })
            .expect("should retain the wormhole node");
        let edge_index = graph
            .edges(stub_index)
            .next()
            .expect("should give a wormhole exactly one neighbor")
            .id();
        let result_bundle = &graph[edge_index].bundle;

        for pair in &bundle {
            assert!(
                result_bundle.contains(pair),
                "bundle {result_bundle:?} should contain every collapsed domain edge, missing {pair:?}"
            );
        }
    }
}
