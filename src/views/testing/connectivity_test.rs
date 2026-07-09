/// Test to verify unified graph connectivity
/// This test ensures that the unified graph is properly connected after edge labeling
#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use gen_models::sample::Sample;
    use gen_tui::{
        graph_controller::{GraphConfig, GraphController},
        layout::{NodeRole, VisualDetail},
        partition_table::PartitionConfig,
    };
    // TODO: This should be available in this crate once gen_graph_widget is ported
    use petgraph::visit::Dfs;

    use crate::{
        imports::gfa::import_gfa, test_helpers::setup_gen,
        views::gen_graph_widget::GenGraphNodeSizer,
    };

    /// Test that data nodes can navigate from sources to sinks in the unified graph
    /// This verifies that partition stitching preserves the original graph's connectivity
    #[test]
    fn test_source_to_sink_connectivity() {
        // Setup environment and database
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.config().conn();

        // Load Anderson GFA file
        let gfa_path = PathBuf::from("fixtures/anderson_promoters.gfa");
        if !gfa_path.exists() {
            println!("⚠️  Anderson GFA file not found, skipping connectivity test");
            return;
        }

        let collection_name = "/";

        // Track the database before starting operations
        import_gfa(&context, &gfa_path, collection_name, Sample::DEFAULT_NAME)
            .expect("GFA import failed");

        let gen_graph =
            Sample::get_graph(conn, collection_name, Sample::DEFAULT_NAME, None).unwrap();

        // Test with small partitions to force inter-partition edges
        let config = GraphConfig {
            partition: PartitionConfig {
                layer_count: 5,
                node_count: usize::MAX,
            },
            ..Default::default()
        };

        let node_sizer = GenGraphNodeSizer;
        let mut controller = GraphController::new_with_config(gen_graph, node_sizer, config);

        // Load all partitions
        controller.set_anchor_partition(0).unwrap();
        let total_partitions = controller
            .partition_controller
            .partition_table
            .partitions
            .len();

        for i in 0..total_partitions {
            let _ = controller.ensure_partition_loaded(i);
        }

        let unified_graph = controller
            .partition_controller
            .partition_table
            .get_unified_layout(VisualDetail::Minimal);

        println!(
            "Testing source-to-sink connectivity in unified graph with {} nodes and {} edges across {} partitions",
            unified_graph.node_count(),
            unified_graph.edge_count(),
            total_partitions
        );

        // Find source and sink data nodes in the original graph
        use petgraph::{Direction, visit::NodeIndexable};

        let mut source_nodes = Vec::new(); // Data nodes with no incoming edges
        let mut sink_nodes = Vec::new(); // Data nodes with no outgoing edges
        let mut all_data_nodes = Vec::new();

        // First, find all data nodes in the unified graph and map them back to original nodes
        for node_idx in unified_graph.node_indices() {
            if let Some(node) = unified_graph.node_weight(node_idx) {
                if let NodeRole::Data(original_node_idx) = &node.role {
                    all_data_nodes.push((node_idx, original_node_idx));
                }
            }
        }

        // Identify sources and sinks in the original graph
        for (_unified_idx, original_idx) in &all_data_nodes {
            let original_node_id = gen_graph.from_index(original_idx.index());
            let incoming = gen_graph
                .neighbors_directed(original_node_id, Direction::Incoming)
                .count();
            let outgoing = gen_graph
                .neighbors_directed(original_node_id, Direction::Outgoing)
                .count();

            if incoming == 0 {
                source_nodes.push(*original_idx);
            }
            if outgoing == 0 {
                sink_nodes.push(*original_idx);
            }
        }

        println!("Original graph analysis:");
        println!("  Total data nodes: {}", all_data_nodes.len());
        println!("  Source nodes (no incoming): {}", source_nodes.len());
        println!("  Sink nodes (no outgoing): {}", sink_nodes.len());

        if source_nodes.is_empty() || sink_nodes.is_empty() {
            println!(
                "⚠️  No clear sources or sinks found - graph may be cyclic or all nodes are intermediate"
            );
            return;
        }

        // For each source, try to reach each sink through the unified graph
        let mut successful_paths = 0;
        let mut total_paths_tested = 0;

        for &source_original in &source_nodes {
            // Find the unified graph node corresponding to this source
            let source_unified = all_data_nodes
                .iter()
                .find(|(_, orig)| *orig == source_original)
                .map(|(unified, _)| *unified);

            if let Some(source_node) = source_unified {
                for &sink_original in &sink_nodes {
                    // Find the unified graph node corresponding to this sink
                    let sink_unified = all_data_nodes
                        .iter()
                        .find(|(_, orig)| *orig == sink_original)
                        .map(|(unified, _)| *unified);

                    if let Some(sink_node) = sink_unified {
                        total_paths_tested += 1;

                        // Use DFS to see if we can reach the sink from the source
                        let mut dfs = Dfs::new(unified_graph, source_node);
                        let mut found_sink = false;

                        while let Some(node) = dfs.next(unified_graph) {
                            if node == sink_node {
                                found_sink = true;
                                break;
                            }
                        }

                        if found_sink {
                            successful_paths += 1;
                        }
                    }
                }
            }
        }

        println!("Source-to-sink connectivity results:");
        println!("  Paths tested: {}", total_paths_tested);
        println!("  Successful paths: {}", successful_paths);

        if total_paths_tested > 0 {
            let success_rate = successful_paths as f64 / total_paths_tested as f64;
            println!("  Success rate: {:.1}%", success_rate * 100.0);

            // DEPRECATED: This 80% connectivity test is flawed due to edge labeling bugs
            // TODO: Remove this test once edge labeling is fixed and validated by edge_label_validation_test
            // The edge labeling system creates spurious edges that artificially inflate connectivity
            // See edge_label_validation_test.rs for proper validation of edge correctness
            assert!(
                success_rate >= 0.8,
                "Source-to-sink connectivity rate {:.1}% is too low - partition stitching may have broken connections. NOTE: This test is deprecated due to edge labeling bugs.",
                success_rate * 100.0
            );

            println!(
                "✅ Source-to-sink connectivity test passed - unified graph preserves original connectivity (DEPRECATED - see edge_label_validation_test)"
            );
        } else {
            println!("⚠️  No source-to-sink paths could be tested");
        }
    }
}
