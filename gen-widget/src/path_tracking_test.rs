#[cfg(test)]
mod tests {
    use ratatui::layout::Rect;

    use crate::{
        graph_controller::{GraphConfig, GraphController},
        partition_table::PartitionConfig,
        testing::mocks::{MockDomainGraph, TestNodeSizers},
    };

    #[test]
    fn test_path_tracking_in_adjacent_nodes() {
        // Create a graph where routing nodes will be inserted
        let mut domain_graph = MockDomainGraph::new();
        let n0 = domain_graph.add_node(());
        let n1 = domain_graph.add_node(());
        let n2 = domain_graph.add_node(());
        let n3 = domain_graph.add_node(());

        // Create a structure that will likely have routing nodes
        domain_graph.add_edge(n0, n1, ());
        domain_graph.add_edge(n0, n2, ());
        domain_graph.add_edge(n1, n3, ());
        domain_graph.add_edge(n2, n3, ());

        let node_sizer = TestNodeSizers::fixed_1x1();

        // Use config that encourages creation of routing nodes
        let config = GraphConfig {
            partition: PartitionConfig {
                layer_count: 2, // Small partitions may create more complex layouts
                node_count: usize::MAX,
            },
            ..Default::default()
        };

        let mut controller = GraphController::new_with_config(&domain_graph, node_sizer, config);

        // Set up viewport
        controller.viewport_state.viewport_bounds = Rect::new(0, 0, 200, 100);
        let _ = controller.ensure_camera_coverage();
        controller.initialize_cursor();

        // Get the viewport graph and test path tracking
        let viewport_graph = &controller.viewport_graph;

        // Find the first data node
        let first_data_node = viewport_graph
            .nodes
            .iter()
            .find(|(_, node)| matches!(node.role, crate::layout::NodeRole::Data(_)))
            .map(|(pos, node)| (*pos, node));

        if let Some((start_pos, start_node)) = first_data_node {
            println!("Starting from node at position: {:?}", start_pos);

            // Extract domain index for the starting node
            let crate::layout::NodeRole::Data(start_domain_idx) = start_node.role else {
                panic!("Expected Data node");
            };

            // Find all adjacent data nodes with paths
            let adjacent_with_paths = viewport_graph.find_adjacent_data_nodes(start_domain_idx);

            println!("Found {} adjacent data nodes", adjacent_with_paths.len());

            for (target_pos, _node, path) in &adjacent_with_paths {
                let node_layer = viewport_graph.find_node_layer(target_pos);
                println!(
                    "\nAdjacent node at {:?} (layer {:?})",
                    target_pos, node_layer
                );
                println!("  Path length: {} positions", path.len());
                println!("  Path: {:?}", path);

                // Verify path starts with source and ends with target
                assert!(!path.is_empty(), "Path should not be empty");
                assert_eq!(path[0], start_pos, "Path should start with source position");
                assert_eq!(
                    path[path.len() - 1],
                    *target_pos,
                    "Path should end with target position"
                );

                // If path has intermediate nodes, they should be routing nodes
                if path.len() > 2 {
                    for intermediate_pos in path.iter().take(path.len() - 1).skip(1) {
                        if let Some(intermediate_node) = viewport_graph.nodes.get(intermediate_pos)
                        {
                            // Intermediate nodes should typically be routing nodes
                            println!(
                                "    Intermediate node at {:?} has role: {:?}",
                                intermediate_pos, intermediate_node.role
                            );
                        }
                    }
                }
            }

            // Ensure we found at least some adjacent nodes
            assert!(
                !adjacent_with_paths.is_empty(),
                "Should find at least one adjacent data node"
            );
        } else {
            panic!("No data nodes found in viewport graph");
        }
    }

    #[test]
    fn test_layer_filtering_in_cursor() {
        // Create a simple linear graph
        let mut domain_graph = MockDomainGraph::new();
        let n0 = domain_graph.add_node(());
        let n1 = domain_graph.add_node(());
        let n2 = domain_graph.add_node(());

        domain_graph.add_edge(n0, n1, ());
        domain_graph.add_edge(n1, n2, ());

        let node_sizer = TestNodeSizers::fixed_1x1();
        let mut controller = GraphController::new(&domain_graph, node_sizer);

        controller.viewport_state.viewport_bounds = Rect::new(0, 0, 100, 50);
        let _ = controller.ensure_camera_coverage();
        controller.initialize_cursor();

        // Get current position and verify we can navigate
        let initial_world_pos = controller
            .viewport_state
            .cursor
            .node_world_pos
            .expect("Cursor should have a tracked node");

        // Check that the current node has layer information
        let initial_layer = {
            let layer = controller
                .viewport_graph
                .find_node_layer(&initial_world_pos)
                .expect("Node should have layer information");

            println!(
                "Current node at {:?} has layer: {:?}",
                initial_world_pos, layer
            );
            layer
        };

        // Move right (to next layer)
        controller.move_cursor_horizontal(1);

        let new_world_pos = controller.viewport_state.cursor.node_world_pos;
        if let Some(new_pos) = new_world_pos {
            if new_pos != initial_world_pos {
                // Verify we moved to a node in the next layer
                let new_layer = controller
                    .viewport_graph
                    .find_node_layer(&new_pos)
                    .expect("New node should have layer");

                println!("Moved from layer {} to layer {}", initial_layer, new_layer);
                assert_eq!(
                    new_layer,
                    initial_layer + 1,
                    "Should move to next layer when moving right"
                );
            }
        }
    }
}
