#[cfg(test)]
mod cursor_restoration_tests {
    use petgraph::graph::NodeIndex;
    use ratatui::layout::Rect;

    use crate::{
        geometry::{BigRect, WorldPos},
        graph_controller::GraphController,
        layout::VisualDetail,
        testing::mocks::{TestGraphs, TestNodeSizers},
    };

    /// Test that cursor terminal position is exactly preserved after disperse operation
    /// This test follows the specification in docs/cursor_restauration.txt section 6.1
    #[test]
    fn test_cursor_terminal_position_preserved_after_disperse() {
        eprintln!("\n=== TEST: Cursor Terminal Position Preservation (Disperse) ===");

        // Create test graph and controller
        let graph = TestGraphs::domain_diamond();
        let sizer = TestNodeSizers::scale_aware();
        let mut controller = GraphController::new(&graph, Box::new(sizer));

        // Set viewport bounds to a specific size
        controller.viewport_state.viewport_bounds = Rect::new(0, 0, 100, 50);

        // Enable cursor
        controller.enable_cursor();

        // Load initial viewport
        let viewport = BigRect::from_corners(WorldPos::new(-50, -25), WorldPos::new(50, 25));
        controller
            .update_viewport_graph(viewport, VisualDetail::Full)
            .expect("Failed to initialize viewport graph");

        // Initialize cursor - this should place it on NodeIndex(0)
        controller.initialize_cursor();

        // Capture state BEFORE disperse
        let cursor_node_before = controller.viewport_state.cursor.node_domain_idx;
        let terminal_pos_before = controller
            .viewport_state
            .world_to_terminal(controller.viewport_state.cursor.current);

        assert!(
            cursor_node_before.is_some(),
            "Cursor should be tracking a node before disperse"
        );
        assert!(
            terminal_pos_before.is_some(),
            "Cursor should be visible before disperse"
        );

        let node_before = cursor_node_before.unwrap();
        let (term_x_before, term_y_before) = terminal_pos_before.unwrap();

        eprintln!("BEFORE DISPERSE:");
        eprintln!("  Node: {:?}", node_before);
        eprintln!(
            "  Terminal position: ({}, {})",
            term_x_before, term_y_before
        );
        eprintln!(
            "  World position: {:?}",
            controller.viewport_state.cursor.current
        );
        eprintln!(
            "  Camera position: {:?}",
            controller.viewport_state.camera_current
        );

        // Perform disperse operation
        controller.disperse();

        // Rebuild viewport graph (this triggers cursor restoration)
        controller
            .update_viewport_graph(viewport, VisualDetail::Full)
            .expect("Failed to update viewport graph after disperse");

        // Capture state AFTER disperse
        let cursor_node_after = controller.viewport_state.cursor.node_domain_idx;
        let terminal_pos_after = controller
            .viewport_state
            .world_to_terminal(controller.viewport_state.cursor.current);

        eprintln!("\nAFTER DISPERSE:");
        eprintln!("  Node: {:?}", cursor_node_after);
        if let Some((x, y)) = terminal_pos_after {
            eprintln!("  Terminal position: ({}, {})", x, y);
        } else {
            eprintln!("  Terminal position: None (NOT VISIBLE - BUG!)");
        }
        eprintln!(
            "  World position: {:?}",
            controller.viewport_state.cursor.current
        );
        eprintln!(
            "  Camera position: {:?}",
            controller.viewport_state.camera_current
        );

        // VERIFICATION: Cursor should track the same node
        assert_eq!(
            cursor_node_after,
            Some(node_before),
            "Cursor must track the same node after disperse"
        );

        // VERIFICATION: Terminal position must be EXACTLY preserved
        assert!(
            terminal_pos_after.is_some(),
            "Cursor must be visible after disperse operation"
        );

        let (term_x_after, term_y_after) = terminal_pos_after.unwrap();
        assert_eq!(
            (term_x_after, term_y_after),
            (term_x_before, term_y_before),
            "Terminal position must be EXACTLY preserved after disperse"
        );

        eprintln!("\n✓ TEST PASSED: Terminal position preserved exactly!");
    }

    /// Test that cursor terminal position is exactly preserved after contract operation
    #[test]
    fn test_cursor_terminal_position_preserved_after_contract() {
        eprintln!("\n=== TEST: Cursor Terminal Position Preservation (Contract) ===");

        // Create test graph and controller
        let graph = TestGraphs::domain_diamond();
        let sizer = TestNodeSizers::scale_aware();
        let mut controller = GraphController::new(&graph, Box::new(sizer));

        // Set viewport bounds
        controller.viewport_state.viewport_bounds = Rect::new(0, 0, 100, 50);
        controller.enable_cursor();

        // Load initial viewport
        let viewport = BigRect::from_corners(WorldPos::new(-50, -25), WorldPos::new(50, 25));
        controller
            .update_viewport_graph(viewport, VisualDetail::Full)
            .expect("Failed to initialize viewport graph");

        // First disperse to increase spacing (so we have room to contract)
        controller.disperse();
        controller
            .update_viewport_graph(viewport, VisualDetail::Full)
            .expect("Failed to update after initial disperse");

        // Initialize cursor after disperse
        controller.initialize_cursor();

        // Capture state BEFORE contract
        let cursor_node_before = controller.viewport_state.cursor.node_domain_idx;
        let terminal_pos_before = controller
            .viewport_state
            .world_to_terminal(controller.viewport_state.cursor.current);

        assert!(
            cursor_node_before.is_some(),
            "Cursor should be tracking a node before contract"
        );
        assert!(
            terminal_pos_before.is_some(),
            "Cursor should be visible before contract"
        );

        let node_before = cursor_node_before.unwrap();
        let (term_x_before, term_y_before) = terminal_pos_before.unwrap();

        eprintln!("BEFORE CONTRACT:");
        eprintln!("  Node: {:?}", node_before);
        eprintln!(
            "  Terminal position: ({}, {})",
            term_x_before, term_y_before
        );

        // Perform contract operation
        controller.contract();

        // Rebuild viewport graph (this triggers cursor restoration)
        controller
            .update_viewport_graph(viewport, VisualDetail::Full)
            .expect("Failed to update viewport graph after contract");

        // Capture state AFTER contract
        let cursor_node_after = controller.viewport_state.cursor.node_domain_idx;
        let terminal_pos_after = controller
            .viewport_state
            .world_to_terminal(controller.viewport_state.cursor.current);

        eprintln!("\nAFTER CONTRACT:");
        eprintln!("  Node: {:?}", cursor_node_after);
        if let Some((x, y)) = terminal_pos_after {
            eprintln!("  Terminal position: ({}, {})", x, y);
        } else {
            eprintln!("  Terminal position: None (NOT VISIBLE - BUG!)");
        }

        // VERIFICATION: Cursor should track the same node
        assert_eq!(
            cursor_node_after,
            Some(node_before),
            "Cursor must track the same node after contract"
        );

        // VERIFICATION: Terminal position must be EXACTLY preserved
        assert!(
            terminal_pos_after.is_some(),
            "Cursor must be visible after contract operation"
        );

        let (term_x_after, term_y_after) = terminal_pos_after.unwrap();
        assert_eq!(
            (term_x_after, term_y_after),
            (term_x_before, term_y_before),
            "Terminal position must be EXACTLY preserved after contract"
        );

        eprintln!("\n✓ TEST PASSED: Terminal position preserved exactly!");
    }

    /// Test cursor restoration with cursor on a specific node (not the first one)
    #[test]
    fn test_cursor_restoration_on_specific_node() {
        eprintln!("\n=== TEST: Cursor Restoration on Specific Node ===");

        let graph = TestGraphs::domain_diamond();
        let sizer = TestNodeSizers::scale_aware();
        let mut controller = GraphController::new(&graph, Box::new(sizer));

        controller.viewport_state.viewport_bounds = Rect::new(0, 0, 100, 50);
        controller.enable_cursor();

        let viewport = BigRect::from_corners(WorldPos::new(-50, -25), WorldPos::new(50, 25));
        controller
            .update_viewport_graph(viewport, VisualDetail::Full)
            .expect("Failed to initialize viewport graph");

        // Manually set cursor to track NodeIndex(1) instead of the default
        if let Some(node_world_pos) = controller.find_domain_node_world_position(NodeIndex::new(1))
        {
            controller.viewport_state.cursor.current = node_world_pos;
            controller.viewport_state.cursor.target = node_world_pos;
            controller.viewport_state.cursor.node_domain_idx = Some(NodeIndex::new(1));
            controller.viewport_state.cursor.node_world_pos = Some(node_world_pos);
        } else {
            panic!("Could not find NodeIndex(1) in viewport");
        }

        let terminal_pos_before = controller
            .viewport_state
            .world_to_terminal(controller.viewport_state.cursor.current)
            .expect("Cursor should be visible");

        eprintln!("BEFORE DISPERSE:");
        eprintln!("  Tracking: NodeIndex(1)");
        eprintln!("  Terminal: {:?}", terminal_pos_before);

        // Disperse
        controller.disperse();
        controller
            .update_viewport_graph(viewport, VisualDetail::Full)
            .expect("Failed to update after disperse");

        // Verify
        let cursor_node_after = controller.viewport_state.cursor.node_domain_idx;
        let terminal_pos_after = controller
            .viewport_state
            .world_to_terminal(controller.viewport_state.cursor.current)
            .expect("Cursor should be visible after disperse");

        eprintln!("\nAFTER DISPERSE:");
        eprintln!("  Tracking: {:?}", cursor_node_after);
        eprintln!("  Terminal: {:?}", terminal_pos_after);

        assert_eq!(
            cursor_node_after,
            Some(NodeIndex::new(1)),
            "Should still track NodeIndex(1)"
        );
        assert_eq!(
            terminal_pos_after, terminal_pos_before,
            "Terminal position should be preserved for NodeIndex(1)"
        );

        eprintln!("\n✓ TEST PASSED!");
    }
}
