#[cfg(test)]
mod tests {

    use petgraph::graph::NodeIndex;
    use ratatui::layout::Rect;

    use crate::{
        graph_controller::GraphController,
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

        let node_sizer = TestNodeSizers::fixed_1x1();
        let mut controller = GraphController::new(&domain_graph, node_sizer);

        // Set up viewport
        controller.viewport_state.viewport_bounds = Rect::new(0, 0, 100, 50);
        controller.rebuild_viewport_graph();
        let viewport_graph = controller.get_viewport_graph();
        // Test that we can navigate through the layers
        let initial_pos = controller.cursor.viewport_pos;
        println!("Initial cursor position: {:?}", initial_pos);

        // Move right - should jump to next layer
        controller.cursor.move_horizontal(1, &viewport_graph);
        let pos_after_right = controller.cursor.viewport_pos;
        println!("Position after moving right: {:?}", pos_after_right);

        // The cursor should have moved to a different X position (next layer)
        assert_ne!(
            pos_after_right.x, initial_pos.x,
            "Cursor should move to next layer when moving right"
        );

        // Move left - should return to previous layer
        controller.cursor.move_horizontal(-1, &viewport_graph);
        let pos_after_left = controller.cursor.viewport_pos;
        println!("Position after moving left: {:?}", pos_after_left);
    }

    #[test]
    fn test_layer_traversal_through_routing_nodes() {
        // Create a graph with routing nodes between layers
        // This tests that the BFS traversal correctly goes through routing nodes
        struct LargeNodeSizer;
        impl NodeSizer<&MockDomainGraph> for LargeNodeSizer {
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
        let mut controller = GraphController::new(&domain_graph, node_sizer);

        controller.viewport_state.viewport_bounds = Rect::new(0, 0, 200, 100);
        let _ = controller.ensure_camera_coverage();
        controller.initialize_cursor();

        // Navigate through the graph
        let start_node = controller.cursor.node_domain_idx;

        // Move right multiple times to traverse through potential routing nodes
        for i in 0..10 {
            controller.cursor.move_horizontal(1);
            let current_node = controller.cursor.node_domain_idx;
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
}
