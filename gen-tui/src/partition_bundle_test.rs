#[cfg(test)]
mod partition_bundle_tests {
    use petgraph::{Direction, visit::EdgeRef};

    use crate::{
        layout::VisualDetail,
        partition::PartitionNode,
        partition_table::PartitionTable,
        testing::mocks::{FixedNodeSizer, MockDomainGraph, TestGraphs},
    };

    /// Test that stitch node edges are properly labeled with bundles
    /// for edges that cross partition boundaries
    #[test]
    fn test_stitch_edge_bundles_for_inter_partition_edges() {
        // Create a graph that will be split into multiple partitions
        // Using extended diamond: 0 -> {1,2} -> 3 -> {4,5} -> 6 -> {7,8} -> 9
        let domain_graph = TestGraphs::domain_extended_diamond();

        // Create partition table with settings that force partitioning
        // min_width=2 means we split every 2 layers, max_nodes=3 ensures splits
        let partition_table: PartitionTable<&MockDomainGraph> =
            PartitionTable::new_with_config(&domain_graph, 2, 3);

        // We expect multiple partitions due to the graph structure
        assert!(
            partition_table.partitions.len() > 1,
            "Graph should be split into multiple partitions"
        );

        // Check each section partition (even indices)
        for (idx, partition) in partition_table.partitions.iter().enumerate() {
            if PartitionTable::<&MockDomainGraph>::is_section(idx) {
                // Find stitch nodes
                let left_stitch = partition.left_stitch_idx;
                let right_stitch = partition.right_stitch_idx;

                if let Some(right_stitch_idx) = right_stitch {
                    // Check edges going TO the right stitch node
                    for edge in partition
                        .graph
                        .edges_directed(right_stitch_idx, Direction::Incoming)
                    {
                        let source_idx = edge.source();

                        // Get the domain node this partition node represents
                        if let Some(PartitionNode::Data(domain_idx)) =
                            partition.graph.node_weight(source_idx)
                        {
                            // Check if this node has outgoing edges to other partitions
                            let has_inter_partition_edge = partition_table
                                .inter_partition_edges
                                .iter()
                                .any(|((src_part, _), edges)| {
                                    *src_part == idx
                                        && edges.iter().any(|(src, _, _)| src == domain_idx)
                                });

                            if has_inter_partition_edge {
                                // The edge to the stitch node SHOULD have a bundle
                                assert!(
                                    edge.weight().is_some(),
                                    "Edge from node {:?} to right stitch in partition {} should have bundle for inter-partition edge",
                                    domain_idx,
                                    idx
                                );
                            }
                        }
                    }
                }

                if let Some(left_stitch_idx) = left_stitch {
                    // Check edges coming FROM the left stitch node
                    for edge in partition
                        .graph
                        .edges_directed(left_stitch_idx, Direction::Outgoing)
                    {
                        let target_idx = edge.target();

                        // Get the domain node this partition node represents
                        if let Some(PartitionNode::Data(domain_idx)) =
                            partition.graph.node_weight(target_idx)
                        {
                            // Check if this node has incoming edges from other partitions
                            let has_inter_partition_edge = partition_table
                                .inter_partition_edges
                                .iter()
                                .any(|((_, tgt_part), edges)| {
                                    *tgt_part == idx
                                        && edges.iter().any(|(_, tgt, _)| tgt == domain_idx)
                                });

                            if has_inter_partition_edge {
                                // The edge from the stitch node SHOULD have a bundle
                                assert!(
                                    edge.weight().is_some(),
                                    "Edge from left stitch to node {:?} in partition {} should have bundle for inter-partition edge",
                                    domain_idx,
                                    idx
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// Test that bridge graphs can be created when bundles are properly set
    #[test]
    fn test_bridge_graph_creation_with_bundles() {
        // Create a simple chain that will be partitioned
        let mut domain_graph = MockDomainGraph::new();
        let nodes: Vec<_> = (0..10).map(|_| domain_graph.add_node(())).collect();
        for i in 0..9 {
            domain_graph.add_edge(nodes[i], nodes[i + 1], ());
        }

        // Force partitioning by setting small limits
        let mut partition_table: PartitionTable<&MockDomainGraph> =
            PartitionTable::new_with_config(&domain_graph, 2, 2);

        let node_sizer = FixedNodeSizer {
            width: 3,
            height: 1,
        };

        // Load section partitions (which creates layouts)
        for idx in (0..partition_table.partitions.len()).step_by(2) {
            let _ = partition_table.load_partition(idx, &node_sizer, &&domain_graph, 1.0);
        }

        // Now try to load bridge partitions
        // This will fail if bundles aren't properly set on stitch edges
        for idx in (1..partition_table.partitions.len()).step_by(2) {
            let result = partition_table.load_partition(idx, &node_sizer, &&domain_graph, 1.0);

            // Bridge loading should succeed if bundles are correct
            assert!(
                result.is_ok(),
                "Bridge partition {} loading failed: {:?}",
                idx,
                result.err()
            );

            // Verify the bridge graph was created
            if let Some(bridge_layout) = partition_table.get_layout(idx, VisualDetail::Full) {
                // Bridge graphs should have nodes if edges were matched
                if partition_table
                    .inter_partition_edges
                    .iter()
                    .any(|((src, tgt), _)| {
                        (*src == idx - 1 && *tgt == idx + 1) || (*src < idx && *tgt > idx)
                    })
                {
                    assert!(
                        bridge_layout.graph.node_count() > 0,
                        "Bridge {} should have nodes when inter-partition edges exist",
                        idx
                    );
                }
            }
        }
    }

    /// Test edge bundle propagation through intermediate partitions
    #[test]
    fn test_multi_partition_spanning_edge_bundles() {
        // Create a graph with long-distance edges that span multiple partitions
        let mut domain_graph = MockDomainGraph::new();
        let nodes: Vec<_> = (0..12).map(|_| domain_graph.add_node(())).collect();

        // Create a chain with some long-distance edges
        for i in 0..11 {
            domain_graph.add_edge(nodes[i], nodes[i + 1], ());
        }
        // Add long-distance edges that will span multiple partitions
        domain_graph.add_edge(nodes[0], nodes[8], ()); // Spans several partitions
        domain_graph.add_edge(nodes[2], nodes[10], ()); // Another long span

        // Create partitions with very small limits to force many partitions
        let partition_table: PartitionTable<&MockDomainGraph> =
            PartitionTable::new_with_config(&domain_graph, 2, 2);

        // Find edges that span multiple partitions
        for ((src_part, tgt_part), edges) in &partition_table.inter_partition_edges {
            if *tgt_part > *src_part + 2 {
                // This edge spans multiple partitions

                // Check intermediate partitions have transit edges
                for intermediate_idx in (*src_part + 2..*tgt_part).step_by(2) {
                    let partition = &partition_table.partitions[intermediate_idx];

                    if let (Some(left_stitch), Some(right_stitch)) =
                        (partition.left_stitch_idx, partition.right_stitch_idx)
                    {
                        // Should have an edge from left to right stitch with the bundle
                        let has_transit_edge = partition
                            .graph
                            .edges_connecting(left_stitch, right_stitch)
                            .any(|edge| {
                                if let Some((src, tgt)) = edge.weight() {
                                    edges
                                        .iter()
                                        .any(|(e_src, e_tgt, _)| e_src == src && e_tgt == tgt)
                                } else {
                                    false
                                }
                            });

                        assert!(
                            has_transit_edge,
                            "Intermediate partition {} should have transit edge for edges spanning from {} to {}",
                            intermediate_idx, src_part, tgt_part
                        );
                    }
                }
            }
        }
    }

    /// Regression test for the specific issue: bundles being None on stitch edges
    #[test]
    fn test_regression_stitch_edges_have_bundles() {
        // This test specifically checks the bug where stitch edges had None bundles
        let domain_graph = TestGraphs::domain_diamond();

        let partition_table: PartitionTable<&MockDomainGraph> =
            PartitionTable::new_with_config(&domain_graph, 2, 2);

        // Count how many stitch edges have bundles vs None
        let mut edges_with_bundles = 0;
        let mut edges_without_bundles = 0;

        for partition in &partition_table.partitions {
            // Check right stitch edges
            if let Some(right_stitch) = partition.right_stitch_idx {
                for edge in partition
                    .graph
                    .edges_directed(right_stitch, Direction::Incoming)
                {
                    if edge.weight().is_some() {
                        edges_with_bundles += 1;
                    } else {
                        edges_without_bundles += 1;
                    }
                }
            }

            // Check left stitch edges
            if let Some(left_stitch) = partition.left_stitch_idx {
                for edge in partition
                    .graph
                    .edges_directed(left_stitch, Direction::Outgoing)
                {
                    if edge.weight().is_some() {
                        edges_with_bundles += 1;
                    } else {
                        edges_without_bundles += 1;
                    }
                }
            }
        }

        // Log for debugging
        println!(
            "Stitch edges with bundles: {}, without bundles: {}",
            edges_with_bundles, edges_without_bundles
        );

        // In a properly partitioned graph with inter-partition edges,
        // we should have at least some edges with bundles
        if !partition_table.inter_partition_edges.is_empty() {
            assert!(
                edges_with_bundles > 0,
                "Expected at least some stitch edges to have bundles when inter-partition edges exist"
            );
        }
    }
}
