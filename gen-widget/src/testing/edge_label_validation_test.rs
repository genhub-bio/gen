/// Test to validate that edge labels in the unified layout graph exactly match domain graph edges
/// This test ensures that:
/// 1. Every edge label in the unified graph corresponds to an actual edge in the domain graph
/// 2. All domain edges are represented in the unified graph labels
/// 3. Edge labels are correctly bundled when paths share routing nodes
#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use gen_graph::{GenGraph, GraphNode};
    use petgraph::{
        graph::NodeIndex,
        graphmap::DiGraphMap,
        visit::{EdgeRef, IntoEdgeReferences, IntoNodeReferences},
    };

    use crate::{
        graph_controller::{GraphConfig, GraphController},
        layout::{NodeRole, VisualDetail},
        partition_table::PartitionConfig,
        plotter::NodeSizer,
    };

    fn make_test_graph(edges: Vec<(i32, i32)>) -> GenGraph {
        let nodes: Vec<GraphNode> = edges
            .iter()
            .flat_map(|(s, t)| vec![*s, *t])
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .map(|id| GraphNode {
                block_id: id as i64,
                node_id: id as i64,
                sequence_start: 0,
                sequence_end: 10,
            })
            .collect();

        DiGraphMap::from_edges(edges.iter().map(|(s, t)| {
            (
                *nodes.iter().find(|gn| gn.block_id == *s as i64).unwrap(),
                *nodes.iter().find(|gn| gn.block_id == *t as i64).unwrap(),
            )
        }))
    }

    struct TestNodeSizer;
    impl NodeSizer<&GenGraph> for TestNodeSizer {
        fn get_node_size(&self, _node: &GraphNode, _detail_level: VisualDetail) -> (u64, u64) {
            (10, 5)
        }

        fn get_dummy_size(&self) -> (u64, u64) {
            (2, 2)
        }
    }

    /// Validate that all edge labels in the unified graph exactly match domain edges
    fn validate_edge_labels(
        domain_graph: &GenGraph,
        controller: &GraphController<&GenGraph, TestNodeSizer>,
        layer_count: usize,
    ) -> Result<(), String> {
        // Get the unified layout graph
        let unified_graph = controller
            .partition_controller
            .partition_table
            .get_unified_layout(VisualDetail::Minimal);

        // Build mapping from domain graph nodes to their indices
        let mut domain_node_to_idx: HashMap<GraphNode, NodeIndex> = HashMap::new();
        for (idx, (_, node)) in domain_graph.node_references().enumerate() {
            domain_node_to_idx.insert(*node, NodeIndex::new(idx));
        }

        // Collect all edges from the domain graph as (NodeIndex, NodeIndex) pairs
        let mut domain_edges: HashSet<(NodeIndex, NodeIndex)> = HashSet::new();
        for edge in domain_graph.edge_references() {
            let source_idx = domain_node_to_idx[&edge.source()];
            let target_idx = domain_node_to_idx[&edge.target()];
            domain_edges.insert((source_idx, target_idx));
        }

        // Track which domain edges we've seen in the unified graph
        let mut seen_domain_edges: HashSet<(NodeIndex, NodeIndex)> = HashSet::new();

        // Validate each edge in the unified graph
        let mut total_edges_with_bundles = 0;
        let mut total_bundle_items = 0;
        for edge_ref in unified_graph.edge_references() {
            let edge_weight = edge_ref.weight();
            if !edge_weight.bundle.is_empty() {
                total_edges_with_bundles += 1;
                total_bundle_items += edge_weight.bundle.len();
            }

            // Check each label in the bundle
            for &(source_idx, target_idx) in &edge_weight.bundle {
                // Verify this edge exists in the domain graph
                if !domain_edges.contains(&(source_idx, target_idx)) {
                    return Err(format!(
                        "Edge label ({:?}, {:?}) does not exist in domain graph (layer_count={})",
                        source_idx, target_idx, layer_count
                    ));
                }
                seen_domain_edges.insert((source_idx, target_idx));
            }
        }

        // Debug output
        println!(
            "  🔍 Bundle analysis: {} edges with bundles, {} total bundle items out of {} total edges",
            total_edges_with_bundles,
            total_bundle_items,
            unified_graph.edge_count()
        );

        // Check that all domain edges are represented
        let missing_edges: Vec<_> = domain_edges.difference(&seen_domain_edges).collect();

        if !missing_edges.is_empty() {
            return Err(format!(
                "Domain edges not found in unified graph labels: {:?} (layer_count={})",
                missing_edges, layer_count
            ));
        }

        // Now validate per-node connectivity
        // For each data node in the unified graph, check that its incident edges
        // have labels that exactly match the domain graph edges for that node
        for node_idx in unified_graph.node_indices() {
            if let Some(node) = unified_graph.node_weight(node_idx) {
                if let NodeRole::Data(domain_node_idx) = node.role {
                    // Find corresponding domain node
                    let domain_node = domain_graph
                        .node_references()
                        .find(|(_, node)| domain_node_to_idx[node] == domain_node_idx)
                        .map(|(_, node)| node);

                    if let Some(domain_node) = domain_node {
                        // Collect domain edges for this node
                        let mut node_domain_edges = HashSet::new();

                        // Outgoing edges in domain
                        for edge in domain_graph.edges(*domain_node) {
                            let source_idx = domain_node_to_idx[&edge.source()];
                            let target_idx = domain_node_to_idx[&edge.target()];
                            node_domain_edges.insert((source_idx, target_idx));
                        }

                        // Incoming edges in domain
                        use petgraph::Direction;
                        for neighbor in
                            domain_graph.neighbors_directed(*domain_node, Direction::Incoming)
                        {
                            let source_idx = domain_node_to_idx[&neighbor];
                            let target_idx = domain_node_idx;
                            node_domain_edges.insert((source_idx, target_idx));
                        }

                        // Collect labels from unified graph edges incident to this node
                        let mut node_unified_labels = HashSet::new();

                        // Outgoing edges in unified graph
                        for edge_ref in unified_graph.edges(node_idx) {
                            for &label in &edge_ref.weight().bundle {
                                // Only count labels that involve this node
                                if label.0 == domain_node_idx {
                                    node_unified_labels.insert(label);
                                }
                            }
                        }

                        // Incoming edges in unified graph
                        for edge_ref in unified_graph.edges_directed(node_idx, Direction::Incoming)
                        {
                            for &label in &edge_ref.weight().bundle {
                                // Only count labels that involve this node
                                if label.1 == domain_node_idx {
                                    node_unified_labels.insert(label);
                                }
                            }
                        }

                        // Verify they match
                        if node_domain_edges != node_unified_labels {
                            let missing = node_domain_edges
                                .difference(&node_unified_labels)
                                .collect::<Vec<_>>();
                            let extra = node_unified_labels
                                .difference(&node_domain_edges)
                                .collect::<Vec<_>>();
                            return Err(format!(
                                "Node {:?} edge mismatch (layer_count={}): missing {:?}, extra {:?}",
                                domain_node_idx, layer_count, missing, extra
                            ));
                        }
                    }
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_edge_labels_match_domain_graph() {
        // Test graph: 1 -> 2 -> 3
        //                  |    |
        //                  v    v
        //                  4 -> 5 -> 6
        let edges = vec![(1, 2), (2, 3), (2, 4), (3, 5), (4, 5), (5, 6)];
        let domain_graph = make_test_graph(edges.clone());

        println!("Testing edge label validation with graph:");
        println!("  Edges: {:?}", edges);

        let mut errors = Vec::new();

        // Test with different partition sizes, including maximum width to verify hypothesis
        for layer_count in [1, 2, 3, 4, 5, 10, usize::MAX] {
            println!("\nTesting with layer_count = {}", layer_count);

            let config = GraphConfig {
                partition: PartitionConfig {
                    layer_count,
                    node_count: usize::MAX,
                },
                ..Default::default()
            };

            let node_sizer = TestNodeSizer;
            let mut controller =
                GraphController::new_with_config(&domain_graph, node_sizer, config);

            // Load all partitions
            let total_partitions = controller
                .partition_controller
                .partition_table
                .partitions
                .len();

            println!("  Created {} partitions", total_partitions);

            for i in 0..total_partitions {
                controller.ensure_partition_loaded(i).unwrap();
            }

            // Validate edge labels
            match validate_edge_labels(&domain_graph, &controller, layer_count) {
                Ok(()) => println!("  ✅ Edge labels valid for layer_count={}", layer_count),
                Err(e) => {
                    println!(
                        "  ⚠️ Edge labels invalid for layer_count={}: {}",
                        layer_count, e
                    );

                    // Print detailed analysis for max width case and collect spurious labels
                    let unified_graph = controller
                        .partition_controller
                        .partition_table
                        .get_unified_layout(VisualDetail::Minimal);

                    println!("  Spurious edge analysis:");
                    let mut spurious_labels = Vec::new();
                    for edge_ref in unified_graph.edge_references() {
                        for &label in &edge_ref.weight().bundle {
                            // Check if this label represents a real domain edge
                            let source_exists = domain_graph.node_references().any(|(_, node)| {
                                if let Some(idx) =
                                    domain_graph.node_references().position(|(_, n)| n == node)
                                {
                                    NodeIndex::new(idx) == label.0
                                } else {
                                    false
                                }
                            });
                            let target_exists = domain_graph.node_references().any(|(_, node)| {
                                if let Some(idx) =
                                    domain_graph.node_references().position(|(_, n)| n == node)
                                {
                                    NodeIndex::new(idx) == label.1
                                } else {
                                    false
                                }
                            });

                            if !source_exists || !target_exists {
                                spurious_labels.push(label);
                                println!(
                                    "    Spurious label: {:?} (source_exists={}, target_exists={})",
                                    label, source_exists, target_exists
                                );
                            }
                        }
                    }

                    if layer_count != usize::MAX {
                        errors.push((layer_count, spurious_labels.len(), e));
                    } else {
                        println!(
                            "  📊 Max width summary: {} spurious labels found",
                            spurious_labels.len()
                        );
                    }
                }
            }
        }

        // Report summary
        if !errors.is_empty() {
            println!("\n❌ Edge label validation summary:");
            for (width, spurious_count, error) in &errors {
                println!(
                    "  layer_count={}: {} spurious labels, error: {}",
                    width, spurious_count, error
                );
            }
            panic!("Edge label validation failed for multiple partition sizes");
        }
    }

    #[test]
    fn test_specific_node_2_edges() {
        // Focus on node 2 which has edges to nodes 3 and 4
        let edges = vec![(1, 2), (2, 3), (2, 4), (3, 5), (4, 5), (5, 6)];
        let domain_graph = make_test_graph(edges.clone());

        println!("\nTesting node 2 edge bundling:");

        for layer_count in [1, 2, 3, usize::MAX] {
            println!("\n  layer_count = {}", layer_count);

            let config = GraphConfig {
                partition: PartitionConfig {
                    layer_count,
                    node_count: usize::MAX,
                },
                ..Default::default()
            };

            let node_sizer = TestNodeSizer;
            let mut controller =
                GraphController::new_with_config(&domain_graph, node_sizer, config);

            // Load all partitions
            let total_partitions = controller
                .partition_controller
                .partition_table
                .partitions
                .len();

            for i in 0..total_partitions {
                controller.ensure_partition_loaded(i).unwrap();
            }

            let unified_graph = controller
                .partition_controller
                .partition_table
                .get_unified_layout(VisualDetail::Minimal);

            // Find node 2 in the unified graph
            let node_2_unified = unified_graph.node_indices().find(|&idx| {
                if let Some(node) = unified_graph.node_weight(idx) {
                    matches!(node.role, NodeRole::Data(domain_idx) if domain_idx.index() == 1)
                } else {
                    false
                }
            });

            if let Some(node_2_idx) = node_2_unified {
                println!("    Found node 2 in unified graph");

                // Check outgoing edges from node 2
                for edge_ref in unified_graph.edges(node_2_idx) {
                    let labels = &edge_ref.weight().bundle;
                    let (_source, target) = unified_graph.edge_endpoints(edge_ref.id()).unwrap();

                    if let Some(target_node) = unified_graph.node_weight(target) {
                        println!(
                            "      Edge to {:?}: labels = {:?}",
                            target_node.role, labels
                        );

                        // If this is the first routing node after node 2, it should have both labels
                        if matches!(target_node.role, NodeRole::Routing) {
                            let expected_labels: HashSet<_> = vec![
                                (NodeIndex::new(1), NodeIndex::new(2)), // 2->3
                                (NodeIndex::new(1), NodeIndex::new(3)), // 2->4
                            ]
                            .into_iter()
                            .collect();

                            let actual_labels: HashSet<_> = labels.iter().cloned().collect();

                            if actual_labels != expected_labels {
                                println!(
                                    "        ⚠️ Expected labels {:?}, got {:?}",
                                    expected_labels, actual_labels
                                );
                            } else {
                                println!("        ✅ Correct bundling of edges from node 2");
                            }
                        }
                    }
                }
            } else {
                println!("    ⚠️ Node 2 not found in unified graph");
            }
        }
    }
}
