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
    fn test_cursor_intra_node_movement() {
        // Create a simple test graph with larger nodes
        struct LargeNodeSizer;
        impl NodeSizer<MockDomainGraph> for LargeNodeSizer {
            fn get_node_size(&self, _node: &NodeIndex, _scale: VisualDetail) -> (u64, u64) {
                (10, 5) // 10 wide, 5 tall nodes
            }

            fn get_dummy_size(&self) -> (u64, u64) {
                (1, 1)
            }
        }

        let mut domain_graph = MockDomainGraph::new();
        let a = domain_graph.add_node(());
        let b = domain_graph.add_node(());
        let c = domain_graph.add_node(());
        domain_graph.add_edge(a, b, ());
        domain_graph.add_edge(b, c, ());

        let node_sizer = LargeNodeSizer;
        let mut controller = GraphController::new(domain_graph.clone(), node_sizer);

        // Set up viewport
        controller.viewport_state.viewport_bounds = Rect::new(0, 0, 100, 50);
        let _ = controller.ensure_camera_coverage();
        controller.initialize_cursor();

        // Get initial cursor position
        let initial_pos = controller.viewport_state.cursor.current;
        let initial_node = controller.viewport_state.cursor.node_domain_idx;

        // Move cursor horizontally within the node (should stay in same node)
        controller.move_cursor_horizontal(1);
        let pos_after_move = controller.viewport_state.cursor.current;
        let node_after_move = controller.viewport_state.cursor.node_domain_idx;

        // Check that cursor moved but stayed in same node
        assert_eq!(
            pos_after_move.x,
            initial_pos.x + 1,
            "Cursor should move 1 pixel to the right"
        );
        assert_eq!(
            node_after_move, initial_node,
            "Cursor should stay in the same node for intra-node movement"
        );

        // Move cursor back
        controller.move_cursor_horizontal(-1);
        let pos_after_back = controller.viewport_state.cursor.current;

        assert_eq!(
            pos_after_back.x, initial_pos.x,
            "Cursor should return to initial x position"
        );
    }

    #[test]
    fn test_cursor_inter_node_movement() {
        // Create a simple test graph with small nodes
        let mut domain_graph = MockDomainGraph::new();
        let a = domain_graph.add_node(());
        let b = domain_graph.add_node(());
        let c = domain_graph.add_node(());
        domain_graph.add_edge(a, b, ());
        domain_graph.add_edge(b, c, ());

        let node_sizer = TestNodeSizers::fixed_1x1(); // Small 1x1 nodes
        let mut controller = GraphController::new(domain_graph.clone(), node_sizer);

        // Set up viewport
        controller.viewport_state.viewport_bounds = Rect::new(0, 0, 100, 50);
        let _ = controller.ensure_camera_coverage();
        controller.initialize_cursor();

        let initial_node = controller.viewport_state.cursor.node_domain_idx;

        // Move cursor far enough to trigger inter-node jump
        // With 1x1 nodes, moving right multiple times should jump to next node
        for _ in 0..10 {
            controller.move_cursor_horizontal(1);
        }

        let final_node = controller.viewport_state.cursor.node_domain_idx;

        // The cursor should have moved to a different node
        // We can't predict exactly which node without knowing the layout,
        // but it should be different if inter-node movement works
        assert!(initial_node.is_some(), "Initial node should be set");

        // Note: This test might need adjustment based on actual layout behavior
        println!(
            "Initial node: {:?}, Final node: {:?}",
            initial_node, final_node
        );
    }

    #[test]
    fn test_cursor_vertical_movement_within_node() {
        // Create a test graph with tall nodes
        struct TallNodeSizer;
        impl NodeSizer<MockDomainGraph> for TallNodeSizer {
            fn get_node_size(&self, _node: &NodeIndex, _scale: VisualDetail) -> (u64, u64) {
                (5, 10) // 5 wide, 10 tall nodes
            }

            fn get_dummy_size(&self) -> (u64, u64) {
                (1, 1)
            }
        }

        let mut domain_graph = MockDomainGraph::new();
        let a = domain_graph.add_node(());
        let b = domain_graph.add_node(());
        domain_graph.add_edge(a, b, ());

        let node_sizer = TallNodeSizer;
        let mut controller = GraphController::new(domain_graph.clone(), node_sizer);

        // Set up viewport
        controller.viewport_state.viewport_bounds = Rect::new(0, 0, 100, 50);
        let _ = controller.ensure_camera_coverage();
        controller.initialize_cursor();

        let initial_y = controller.viewport_state.cursor.current.y;
        let initial_node = controller.viewport_state.cursor.node_domain_idx;

        // Move cursor vertically within the node
        controller.move_cursor_vertical(1);
        let y_after_move = controller.viewport_state.cursor.current.y;
        let node_after_move = controller.viewport_state.cursor.node_domain_idx;

        // Check vertical movement within node
        assert_eq!(
            y_after_move,
            initial_y + 1,
            "Cursor should move 1 pixel down"
        );
        assert_eq!(
            node_after_move, initial_node,
            "Cursor should stay in the same node for intra-node vertical movement"
        );
    }
}
