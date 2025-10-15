#[cfg(test)]
mod cursor_positioning_tests {
    use ratatui::layout::Rect;

    use crate::{
        geometry::{BigRect, WorldPos},
        graph_controller::GraphController,
        layout::VisualDetail,
        testing::mocks::{TestGraphs, TestNodeSizers},
    };

    /// Test that cursor position is preserved after detail level changes
    #[test]
    fn test_cursor_preserved_after_detail_level_change() {
        eprintln!("\n=== TEST: test_cursor_preserved_after_detail_level_change ===");

        // Create a test graph and controller
        let graph = TestGraphs::domain_diamond();
        let sizer = TestNodeSizers::scale_aware();

        let mut controller = GraphController::new(&graph, Box::new(sizer));

        // Set viewport bounds
        controller.viewport_state.viewport_bounds = Rect::new(0, 0, 100, 50);

        // Initialize with a specific viewport
        let viewport = BigRect::from_corners(WorldPos::new(-50, -25), WorldPos::new(50, 25));

        // Set initial detail level
        controller.set_detail_level(VisualDetail::Full);

        // Update viewport graph to populate it
        controller
            .rebuild_viewport_graph()
            .expect("Failed to update viewport graph");

        // Initialize cursor and get its position
        controller.initialize_cursor();
        let initial_cursor_world = controller.viewport_state.cursor.current;
        let initial_node = controller.viewport_state.cursor.node_domain_idx;
        let initial_offset = controller.viewport_state.cursor.node_offset;

        // Get terminal position of cursor before detail change
        let terminal_pos = controller
            .viewport_state
            .world_to_terminal(initial_cursor_world);
        assert!(
            terminal_pos.is_some(),
            "Cursor should be visible in viewport"
        );
        let (term_x, term_y) = terminal_pos.unwrap();

        eprintln!("Initial cursor state:");
        eprintln!("  World position: {:?}", initial_cursor_world);
        eprintln!("  Terminal position: ({}, {})", term_x, term_y);
        eprintln!("  Node: {:?}", initial_node);
        eprintln!("  Offset: {:?}", initial_offset);

        // Change detail level - this should preserve cursor position
        controller.set_detail_level(VisualDetail::Truncated);

        // Rebuild viewport graph after detail change
        controller
            .rebuild_viewport_graph()
            .expect("Failed to rebuild viewport graph after detail change");

        // Check cursor state after detail change
        let final_cursor_world = controller.viewport_state.cursor.current;
        let final_node = controller.viewport_state.cursor.node_domain_idx;
        let final_terminal_pos = controller
            .viewport_state
            .world_to_terminal(final_cursor_world);

        eprintln!("Final cursor state:");
        eprintln!("  World position: {:?}", final_cursor_world);
        if let Some((fx, fy)) = final_terminal_pos {
            eprintln!("  Terminal position: ({}, {})", fx, fy);
        } else {
            eprintln!("  Terminal position: None (cursor not visible!)");
        }
        eprintln!("  Node: {:?}", final_node);

        // Verify that cursor is still on the same node
        assert_eq!(
            initial_node, final_node,
            "Cursor should remain on the same node after detail level change"
        );

        // Verify that terminal position is preserved (with small tolerance for rounding)
        if let Some((final_x, final_y)) = final_terminal_pos {
            assert!(
                (final_x as i32 - term_x as i32).abs() <= 1,
                "Terminal X position should be preserved (was {}, now {})",
                term_x,
                final_x
            );
            assert!(
                (final_y as i32 - term_y as i32).abs() <= 1,
                "Terminal Y position should be preserved (was {}, now {})",
                term_y,
                final_y
            );
        } else {
            panic!("Cursor should still be visible after detail level change");
        }

        // Additional check: verify cursor is within the node bounds
        if let Some(node_idx) = final_node {
            let node_world_pos = controller.find_domain_node_world_position(node_idx);
            assert!(
                node_world_pos.is_some(),
                "Node should have a world position"
            );

            // The cursor should be reasonably close to the node
            let node_pos = node_world_pos.unwrap();
            let distance = ((final_cursor_world.x - node_pos.x).abs()
                + (final_cursor_world.y - node_pos.y).abs()) as f64;
            assert!(
                distance < 100.0,
                "Cursor should be close to its tracked node (distance: {})",
                distance
            );
        }

        eprintln!("=== TEST PASSED ===\n");
    }

    /// Test cursor positioning after disperse when cursor is near zone boundary
    #[test]
    fn test_cursor_preserved_after_disperse_near_zone_boundary() {
        eprintln!("\n=== TEST: test_cursor_preserved_after_disperse_near_zone_boundary ===");

        let graph = TestGraphs::domain_diamond();
        let sizer = TestNodeSizers::scale_aware();

        let mut controller = GraphController::new(&graph, Box::new(sizer));

        // Set viewport bounds
        controller.viewport_state.viewport_bounds = Rect::new(0, 0, 100, 50);

        // Set dead zone to 20% and soft zone to 40% to create clear boundaries
        controller.viewport_state.set_dead_zone_fraction(0.2, 0.2);
        controller.viewport_state.set_soft_zone_fraction(0.4, 0.4);

        // Initialize viewport
        let viewport = BigRect::from_corners(WorldPos::new(-50, -25), WorldPos::new(50, 25));

        controller
            .rebuild_viewport_graph()
            .expect("Failed to update viewport graph");

        controller.enable_cursor();
        controller.initialize_cursor();

        // Move cursor to near the edge of the dead zone (should trigger multizone after disperse)
        // With 100x50 viewport and 20% dead zone, dead zone extends ±10x, ±5y from center
        // Place cursor at x=35 (15 units from center, outside the 10 unit dead zone)
        controller.viewport_state.cursor.current = WorldPos::new(15, 0);
        controller.viewport_state.cursor.target = WorldPos::new(15, 0);

        let initial_terminal_pos = controller
            .viewport_state
            .world_to_terminal(controller.viewport_state.cursor.current)
            .expect("Cursor should be visible");

        eprintln!("Before disperse:");
        eprintln!(
            "  Cursor world: {:?}",
            controller.viewport_state.cursor.current
        );
        eprintln!("  Terminal position: {:?}", initial_terminal_pos);
        eprintln!(
            "  Dead zone: ±{:.1}x, ±{:.1}y",
            controller.viewport_state.viewport_bounds.width as f32 * 0.1,
            controller.viewport_state.viewport_bounds.height as f32 * 0.1
        );

        // Perform disperse operation
        controller.disperse();

        // Update viewport graph after disperse - this should trigger our assertions!
        controller
            .rebuild_viewport_graph()
            .expect("Failed to update viewport graph after disperse");

        eprintln!("=== TEST PASSED (no assertion triggered) ===");
    }

    /// Test cursor positioning after disperse operation
    #[test]
    fn test_cursor_preserved_after_disperse() {
        eprintln!("\n=== TEST: test_cursor_preserved_after_disperse ===");

        let graph = TestGraphs::domain_diamond();
        let sizer = TestNodeSizers::scale_aware();

        let mut controller = GraphController::new(&graph, Box::new(sizer));

        // Set viewport bounds first
        controller.viewport_state.viewport_bounds = Rect::new(0, 0, 100, 50);

        // Set viewport and initialize
        let viewport = BigRect::from_corners(WorldPos::new(-50, -25), WorldPos::new(50, 25));

        controller
            .rebuild_viewport_graph()
            .expect("Failed to update viewport graph");

        controller.enable_cursor();
        controller.initialize_cursor();
        let initial_terminal_pos = controller
            .viewport_state
            .world_to_terminal(controller.viewport_state.cursor.current);
        let initial_node = controller.viewport_state.cursor.node_domain_idx;

        assert!(
            initial_terminal_pos.is_some(),
            "Cursor should be visible initially"
        );
        let (init_x, init_y) = initial_terminal_pos.unwrap();

        eprintln!("Before disperse:");
        eprintln!("  Terminal position: ({}, {})", init_x, init_y);
        eprintln!("  Node: {:?}", initial_node);

        // Perform disperse operation
        controller.disperse();

        // Update viewport graph after disperse
        controller
            .rebuild_viewport_graph()
            .expect("Failed to update viewport graph after disperse");

        // Check that cursor is restored
        let final_node = controller.viewport_state.cursor.node_domain_idx;
        let final_terminal_pos = controller
            .viewport_state
            .world_to_terminal(controller.viewport_state.cursor.current);

        eprintln!("After disperse:");
        if let Some((fx, fy)) = final_terminal_pos {
            eprintln!("  Terminal position: ({}, {})", fx, fy);
        } else {
            eprintln!("  Terminal position: None (not visible)");
        }
        eprintln!("  Node: {:?}", final_node);

        assert_eq!(
            initial_node, final_node,
            "Cursor should track the same node after disperse"
        );

        if let Some((final_x, final_y)) = final_terminal_pos {
            // Terminal position should be approximately preserved
            assert!(
                (final_x as i32 - init_x as i32).abs() <= 2,
                "Terminal X should be preserved after disperse (was {}, now {})",
                init_x,
                final_x
            );
            assert!(
                (final_y as i32 - init_y as i32).abs() <= 2,
                "Terminal Y should be preserved after disperse (was {}, now {})",
                init_y,
                final_y
            );
        } else {
            panic!("Cursor should be visible after disperse operation");
        }

        eprintln!("=== TEST PASSED ===\n");
    }

    /// Test cursor positioning after contract operation
    #[test]
    fn test_cursor_preserved_after_contract() {
        eprintln!("\n=== TEST: test_cursor_preserved_after_contract ===");

        let graph = TestGraphs::domain_diamond();
        let sizer = TestNodeSizers::scale_aware();

        let mut controller = GraphController::new(&graph, Box::new(sizer));

        // Set viewport bounds
        controller.viewport_state.viewport_bounds = Rect::new(0, 0, 100, 50);

        // Set viewport and initialize
        let viewport = BigRect::from_corners(WorldPos::new(-50, -25), WorldPos::new(50, 25));

        controller
            .rebuild_viewport_graph()
            .expect("Failed to update viewport graph");

        // First disperse to have something to contract
        controller.disperse();
        controller
            .rebuild_viewport_graph()
            .expect("Failed to update after disperse");

        controller.initialize_cursor();
        let initial_terminal_pos = controller
            .viewport_state
            .world_to_terminal(controller.viewport_state.cursor.current);
        let initial_node = controller.viewport_state.cursor.node_domain_idx;

        assert!(
            initial_terminal_pos.is_some(),
            "Cursor should be visible initially"
        );
        let (init_x, init_y) = initial_terminal_pos.unwrap();

        eprintln!("Before contract:");
        eprintln!("  Terminal position: ({}, {})", init_x, init_y);
        eprintln!("  Node: {:?}", initial_node);

        // Perform contract operation
        controller.contract();

        // Update viewport graph after contract
        controller
            .rebuild_viewport_graph()
            .expect("Failed to update viewport graph after contract");

        // Check that cursor is restored
        let final_node = controller.viewport_state.cursor.node_domain_idx;
        let final_terminal_pos = controller
            .viewport_state
            .world_to_terminal(controller.viewport_state.cursor.current);

        eprintln!("After contract:");
        if let Some((fx, fy)) = final_terminal_pos {
            eprintln!("  Terminal position: ({}, {})", fx, fy);
        } else {
            eprintln!("  Terminal position: None (not visible)");
        }
        eprintln!("  Node: {:?}", final_node);

        assert_eq!(
            initial_node, final_node,
            "Cursor should track the same node after contract"
        );

        if let Some((final_x, final_y)) = final_terminal_pos {
            // Terminal position should be approximately preserved
            assert!(
                (final_x as i32 - init_x as i32).abs() <= 2,
                "Terminal X should be preserved after contract (was {}, now {})",
                init_x,
                final_x
            );
            assert!(
                (final_y as i32 - init_y as i32).abs() <= 2,
                "Terminal Y should be preserved after contract (was {}, now {})",
                init_y,
                final_y
            );
        } else {
            panic!("Cursor should be visible after contract operation");
        }

        eprintln!("=== TEST PASSED ===\n");
    }

    /// Test zooming out from partition 2 (reproduces bug with partition loading)
    #[test]
    fn test_zoom_out_from_partition_2() {
        eprintln!("\n=== TEST: test_zoom_out_from_partition_2 ===");

        use petgraph::graph::NodeIndex;

        let graph = TestGraphs::domain_diamond();
        let sizer = TestNodeSizers::scale_aware();

        let mut controller = GraphController::new(&graph, Box::new(sizer));

        // Set viewport bounds
        controller.viewport_state.viewport_bounds = Rect::new(0, 0, 100, 50);

        // Set viewport and initialize
        let viewport = BigRect::from_corners(WorldPos::new(-50, -25), WorldPos::new(50, 25));

        controller
            .rebuild_viewport_graph()
            .expect("Failed to update viewport graph");

        controller.enable_cursor();

        // Manually set cursor to track NodeIndex(1) or NodeIndex(2) which should be in partition 2
        // First, find where NodeIndex(1) is located
        if let Some(node_world_pos) = controller.find_domain_node_world_position(NodeIndex::new(1))
        {
            eprintln!("Found NodeIndex(1) at world position: {:?}", node_world_pos);
            controller.viewport_state.cursor.current = node_world_pos;
            controller.viewport_state.cursor.target = node_world_pos;
            controller.viewport_state.cursor.node_domain_idx = Some(NodeIndex::new(1));
            controller.viewport_state.cursor.node_world_pos = Some(node_world_pos);
            controller.viewport_state.cursor.node_offset = crate::geometry::Point::new(0, 0);
        } else {
            panic!("Could not find NodeIndex(1) in viewport");
        }

        let initial_terminal_pos = controller
            .viewport_state
            .world_to_terminal(controller.viewport_state.cursor.current)
            .expect("Cursor should be visible");

        eprintln!("Initial state (cursor on NodeIndex(1)):");
        eprintln!("  Terminal position: {:?}", initial_terminal_pos);

        // Zoom in several times (disperse multiple times)
        eprintln!("\n--- Zoom in (disperse) #1 ---");
        controller.disperse();
        controller
            .rebuild_viewport_graph()
            .expect("Failed after disperse #1");

        eprintln!("\n--- Zoom in (disperse) #2 ---");
        controller.disperse();
        controller
            .rebuild_viewport_graph()
            .expect("Failed after disperse #2");

        // At this point, partitions 0 and 1 should NOT be visible/rendered
        eprintln!(
            "\nAfter zooming in, cursor terminal position: {:?}",
            controller
                .viewport_state
                .world_to_terminal(controller.viewport_state.cursor.current)
        );

        // Now zoom back out (contract)
        eprintln!("\n--- Zoom out (contract) #1 ---");
        controller.contract();
        controller
            .rebuild_viewport_graph()
            .expect("Failed after contract #1 - THIS IS WHERE THE BUG OCCURS");

        eprintln!("\n=== TEST PASSED ===\n");
    }
}
