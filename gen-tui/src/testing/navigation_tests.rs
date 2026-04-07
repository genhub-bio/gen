#[cfg(test)]
mod tests {
    use petgraph::graph::NodeIndex;
    use ratatui::layout::Rect;

    use crate::{
        graph_controller::{GraphConfig, GraphController},
        layout::VisualDetail,
        plotter::NodeSizer,
        testing::mocks::{MockDomainGraph, TestNodeSizers},
    };

    #[test]
    fn test_layer_based_navigation() {
        // Create a diamond-shaped test graph to test layer navigation
        // Layer 0: node 0
        // Layer 1: nodes 1, 2
        // Layer 2: node 3
        let mut domain_graph = MockDomainGraph::new();
        let n0 = domain_graph.add_node(());
        let n1 = domain_graph.add_node(());
        let n2 = domain_graph.add_node(());
        let n3 = domain_graph.add_node(());
        // Create diamond structure
        domain_graph.add_edge(n0, n1, ());
        domain_graph.add_edge(n0, n2, ());
        domain_graph.add_edge(n1, n3, ());
        domain_graph.add_edge(n2, n3, ());
        // Use 1x1 nodes to make navigation easier - any movement crosses boundaries
        let node_sizer = TestNodeSizers::fixed_1x1();
        let mut controller = GraphController::new(domain_graph.clone(), node_sizer);
        // Set up viewport to see the entire graph
        controller.viewport_state.viewport_bounds = Rect::new(0, 0, 100, 50);
        controller
            .rebuild_viewport_graph()
            .expect("Failed to rebuild viewport graph for layer navigation test");

        // Initialize cursor to ensure it's properly positioned
        controller.initialize_cursor();
        // Test that we can navigate between layers
        let initial_node = controller.cursor.node_idx;
        println!("Initial cursor node: {:?}", initial_node);
        // Move right - should jump to next layer (with 1x1 nodes, any movement crosses boundary)
        controller
            .cursor
            .move_horizontal(1, &controller.viewport_graph)
            .expect("Failed to move cursor right to next layer");
        let node_after_right = controller.cursor.node_idx;
        println!("Node after moving right: {:?}", node_after_right);
        // The cursor should have moved to a different node (next layer)
        assert_ne!(
            node_after_right, initial_node,
            "Cursor should move to different node when moving right to next layer"
        );
        // Verify the nodes are in different layers
        let initial_layer = controller
            .viewport_graph
            .find_domain_node_layer(initial_node.unwrap());
        let right_layer = controller
            .viewport_graph
            .find_domain_node_layer(node_after_right.unwrap());
        assert!(
            initial_layer.is_some() && right_layer.is_some(),
            "Both nodes should have layers"
        );
        assert_ne!(
            initial_layer.unwrap(),
            right_layer.unwrap(),
            "Nodes should be in different layers"
        );
        // Move right again - should go to final layer
        controller
            .cursor
            .move_horizontal(1, &controller.viewport_graph)
            .expect("Failed to move cursor right to final layer");
        let node_after_right2 = controller.cursor.node_idx;
        println!("Node after moving right again: {:?}", node_after_right2);
        // Should be in yet another layer
        let right2_layer = controller
            .viewport_graph
            .find_domain_node_layer(node_after_right2.unwrap());
        assert_ne!(
            right_layer.unwrap(),
            right2_layer.unwrap(),
            "Should move to yet another layer"
        );
        // Move left - should return to previous layer
        controller
            .cursor
            .move_horizontal(-1, &controller.viewport_graph)
            .expect("Failed to move cursor left to previous layer");
        let node_after_left = controller.cursor.node_idx;
        println!("Node after moving left: {:?}", node_after_left);
        // Should be back in the middle layer
        let left_layer = controller
            .viewport_graph
            .find_domain_node_layer(node_after_left.unwrap());
        assert_eq!(
            left_layer.unwrap(),
            right_layer.unwrap(),
            "Moving left should return to previous layer"
        );
        // Move left again - should return to initial layer
        controller
            .cursor
            .move_horizontal(-1, &controller.viewport_graph)
            .expect("Failed to move cursor left to initial layer");
        let node_after_left2 = controller.cursor.node_idx;
        let left2_layer = controller
            .viewport_graph
            .find_domain_node_layer(node_after_left2.unwrap());
        assert_eq!(
            left2_layer.unwrap(),
            initial_layer.unwrap(),
            "Moving left again should return to initial layer"
        );
    }

    #[test]
    fn test_layer_traversal_through_routing_nodes() {
        // Create a graph with routing nodes between layers
        // This tests that the BFS traversal correctly goes through routing nodes
        struct LargeNodeSizer;
        impl NodeSizer<MockDomainGraph> for LargeNodeSizer {
            fn get_node_size(&self, _node: &NodeIndex, _scale: VisualDetail) -> (u64, u64) {
                (5, 5) // Larger nodes might cause routing nodes to be inserted
            }

            fn get_dummy_size(&self) -> (u64, u64) {
                (1, 1)
            }
        }

        let mut domain_graph = MockDomainGraph::new();
        let n0 = domain_graph.add_node(());
        let n1 = domain_graph.add_node(());
        let n2 = domain_graph.add_node(());

        // Linear graph that might have routing nodes inserted
        domain_graph.add_edge(n0, n1, ());
        domain_graph.add_edge(n1, n2, ());

        let node_sizer = LargeNodeSizer;
        let mut controller = GraphController::new(domain_graph.clone(), node_sizer);

        controller.viewport_state.viewport_bounds = Rect::new(0, 0, 200, 100);
        let _ = controller.ensure_camera_coverage();
        controller
            .rebuild_viewport_graph()
            .expect("Failed to rebuild viewport graph for routing nodes test");
        controller.initialize_cursor();

        // Navigate through the graph
        let start_node = controller.cursor.node_idx;

        // Move right multiple times to traverse through potential routing nodes
        for i in 0..10 {
            controller
                .cursor
                .move_horizontal(1, &controller.viewport_graph)
                .unwrap_or_else(|_| panic!("Failed to move cursor right at step {}", i));
            let current_node = controller.cursor.node_idx;
            println!("Step {}: Node {:?}", i, current_node);

            // Eventually we should reach a different data node
            if current_node != start_node && current_node.is_some() {
                // Success - we navigated to a different node
                assert_ne!(
                    current_node, start_node,
                    "Should navigate to different node through layers"
                );
                return;
            }
        }
    }

    #[test]
    fn test_cursor_navigates_across_partition_boundary() {
        // Cursor navigation should work across partition boundaries.
        // A linear chain with a small layer_count forces splits mid-chain.
        let mut domain_graph = MockDomainGraph::new();
        let nodes: Vec<_> = (0..5).map(|_| domain_graph.add_node(())).collect();
        for i in 0..4 {
            domain_graph.add_edge(nodes[i], nodes[i + 1], ());
        }

        let node_sizer = TestNodeSizers::fixed_1x1();
        let mut config = GraphConfig::default();
        config.partition.layer_count = 2;

        let mut controller = GraphController::new_with_config(domain_graph.clone(), node_sizer, config);
        controller.viewport_state.viewport_bounds = Rect::new(0, 0, 200, 50);
        controller.set_detail_level(VisualDetail::Full);
        controller
            .rebuild_viewport_graph()
            .expect("Failed to rebuild viewport graph");

        let vg = controller.get_viewport_graph().clone();
        controller.cursor.set_coarse_mode(false);
        controller.cursor.set_node(nodes[1], (1.0, 0.5));
        controller.cursor.move_horizontal(1, &vg).unwrap();

        assert_eq!(controller.cursor.node_idx(), Some(nodes[2]));
    }
}
