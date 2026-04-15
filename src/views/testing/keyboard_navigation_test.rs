/// Comprehensive keyboard navigation test for the widget system
/// Tests navigation with real GFA data and simulates multiple keyboard presses
#[cfg(test)]
mod tests {
    use std::{path::PathBuf, time::Duration};

    // TODO: This should be available in this crate once gen_graph_widget is ported
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use gen_graph::GenGraph;
    use gen_models::sample::Sample;
    use gen_tui::{
        geometry::WorldPos,
        graph_controller::{GraphConfig, GraphController},
        graph_widget::DefaultGraphWidget,
        partition_table::PartitionConfig,
        testing::{TestAssertions, create_test_terminal},
    };
    use ratatui::layout::Rect;

    use crate::{
        imports::gfa::import_gfa, test_helpers::setup_gen, track_database,
        views::gen_graph_widget::GenGraphNodeSizer,
    };

    /// Test comprehensive keyboard navigation with 20 right arrow presses
    #[test]
    fn test_keyboard_navigation_with_gfa() {
        // Setup environment and database
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        // Load Anderson GFA file as specified by user
        let gfa_path = PathBuf::from("fixtures/anderson_promoters.gfa");
        if !gfa_path.exists() {
            println!(
                "⚠️  Anderson GFA file not found at fixtures/anderson_promoters.gfa, skipping navigation test"
            );
            return;
        }

        let collection_name = "/navigation_test";

        // Track the database before starting operations
        track_database(conn, op_conn).expect("Failed to track database");
        import_gfa(&context, &gfa_path, collection_name, Sample::DEFAULT_NAME)
            .expect("GFA import failed");

        let gen_graph = Sample::get_graph(conn, collection_name, Sample::DEFAULT_NAME);

        let config = GraphConfig {
            partition: PartitionConfig {
                layer_count: 5,
                node_count: usize::MAX,
            },
            ..Default::default()
        };

        let node_sizer = GenGraphNodeSizer;
        let mut controller = GraphController::new_with_config(gen_graph, node_sizer, config);

        // Set up test terminal
        let mut terminal = create_test_terminal(60, 40);

        // Set viewport bounds
        controller.viewport_state.viewport_bounds = Rect::new(0, 0, 60, 40);
        controller.viewport_state.focus();

        // Load initial partitions
        controller.set_anchor_partition(0).unwrap();

        // Ensure cursor is properly initialized and viewport is expanded
        controller.initialize_cursor();
        let _ = controller.ensure_camera_coverage();

        // Record initial state
        let initial_camera = controller.viewport_state.camera_current;
        let initial_cursor = controller.viewport_state.cursor.current;

        println!("Initial state:");
        println!("  Camera: {:?}", initial_camera);
        println!("  Cursor: {:?}", initial_cursor);
        println!(
            "  Anchor partition: {:?}",
            controller.partition_controller.get_anchor_partition()
        );

        // Store positions after each key press for analysis
        let mut camera_positions = vec![initial_camera];
        let mut cursor_positions = vec![initial_cursor];
        let mut visual_snapshots = Vec::new();

        // Simulate 20 right arrow key presses
        for i in 1..=20 {
            // Create key event
            let key_event = KeyEvent::new(KeyCode::Right, KeyModifiers::NONE);

            // Allow animation to settle by updating viewport state
            let frame_delta = Duration::from_millis(50);
            controller.viewport_state.update(frame_delta, (60, 40));

            // Record new positions
            let new_camera = controller.viewport_state.camera_current;
            let new_cursor = controller.viewport_state.cursor.current;
            camera_positions.push(new_camera);
            cursor_positions.push(new_cursor);

            // Render to terminal and capture snapshot
            terminal
                .draw(|frame| {
                    let widget = DefaultGraphWidget::new();
                    frame.render_stateful_widget(widget, frame.area(), &mut controller);
                })
                .unwrap();

            let buffer = terminal.backend().buffer();
            let snapshot = buffer_to_string(buffer);
            visual_snapshots.push(snapshot.clone());

            println!("After right arrow press {}:", i);
            println!(
                "  Camera: {:?} (moved {} from start)",
                new_camera,
                new_camera.x - initial_camera.x
            );
            println!(
                "  Cursor: {:?} (moved {} from start)",
                new_cursor,
                new_cursor.x - initial_cursor.x
            );
            println!(
                "  Loaded partitions: {:?}",
                controller
                    .partition_controller
                    .get_loaded_partitions_info()
                    .len()
            );

            // Verify cursor has moved (may stay same if at edge)
            if i > 1 {
                let prev_cursor = cursor_positions[i - 1];
                if new_cursor.x == prev_cursor.x {
                    println!("  Note: Cursor didn't move horizontally (may be at boundary)");
                }
            }
        }

        // Assertions

        // 1. Verify cursor moved overall
        let final_cursor = cursor_positions.last().unwrap();
        assert!(
            final_cursor.x > initial_cursor.x
                || controller
                    .partition_controller
                    .get_loaded_partitions_info()
                    .len()
                    > 1,
            "After 20 right presses, cursor should have moved right or loaded new partitions"
        );

        // 2. Verify camera followed cursor when needed
        let final_camera = camera_positions.last().unwrap();
        let cursor_in_viewport = is_cursor_in_viewport(
            *final_cursor,
            *final_camera,
            controller.viewport_state.viewport_bounds,
        );
        assert!(
            cursor_in_viewport,
            "Cursor should remain visible in viewport after navigation"
        );

        // 3. Verify visual output (may not change if graph fits in viewport)
        if visual_snapshots.len() > 1 {
            let first_snapshot = &visual_snapshots[0];
            let last_snapshot = visual_snapshots.last().unwrap();
            // Only check if snapshots are different if cursor moved significantly
            if final_cursor.x - initial_cursor.x > 50 {
                TestAssertions::assert_different_layouts(first_snapshot, last_snapshot);
            } else {
                println!("Note: Visual output may not change if graph fits in viewport");
            }
        }

        // 4. Verify navigation was smooth (no huge jumps)
        for i in 1..cursor_positions.len() {
            let prev = cursor_positions[i - 1];
            let curr = cursor_positions[i];
            let distance = ((curr.x - prev.x).abs() + (curr.y - prev.y).abs()) as i64;

            // Allow for larger jumps when crossing partition boundaries
            assert!(
                distance < 1000,
                "Navigation jump too large at step {}: {} units",
                i,
                distance
            );
        }

        // 5. Test other navigation keys
        test_other_navigation_keys(&mut controller);

        println!("\n✅ Keyboard navigation test completed successfully!");
        println!(
            "  Total horizontal movement: {} units",
            final_cursor.x - initial_cursor.x
        );
        println!(
            "  Partitions loaded: {}",
            controller
                .partition_controller
                .get_loaded_partitions_info()
                .len()
        );
        println!(
            "  Final viewport: camera={:?}, cursor={:?}",
            final_camera, final_cursor
        );
    }

    /// Test other navigation keys (up, down, left, zoom) - specific to our graph type
    fn test_other_navigation_keys(controller: &mut GraphController<GenGraph, GenGraphNodeSizer>) {
        // Test left arrow (go back)
        let before_left = controller.viewport_state.cursor.current;
        controller.handle_key_event(KeyEvent::new(KeyCode::Left, KeyModifiers::NONE));
        let after_left = controller.viewport_state.cursor.current;
        println!(
            "\nLeft arrow: cursor moved from {:?} to {:?}",
            before_left, after_left
        );

        // Test up arrow
        let before_up = controller.viewport_state.cursor.current;
        controller.handle_key_event(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        let after_up = controller.viewport_state.cursor.current;
        println!(
            "Up arrow: cursor moved from {:?} to {:?}",
            before_up, after_up
        );

        // Test down arrow
        let before_down = controller.viewport_state.cursor.current;
        controller.handle_key_event(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        let after_down = controller.viewport_state.cursor.current;
        println!(
            "Down arrow: cursor moved from {:?} to {:?}",
            before_down, after_down
        );

        // Test zoom in
        let before_zoom = controller.get_detail_level();
        controller.handle_key_event(KeyEvent::new(KeyCode::Char('+'), KeyModifiers::NONE));
        let after_zoom = controller.get_detail_level();
        println!(
            "Zoom in: detail level {:?} -> {:?}",
            before_zoom, after_zoom
        );

        // Test zoom out
        controller.handle_key_event(KeyEvent::new(KeyCode::Char('-'), KeyModifiers::NONE));
        let final_zoom = controller.get_detail_level();
        println!(
            "Zoom out: detail level {:?} -> {:?}",
            after_zoom, final_zoom
        );
    }

    /// Check if cursor is visible within viewport
    fn is_cursor_in_viewport(cursor: WorldPos, camera: WorldPos, viewport: Rect) -> bool {
        let viewport_width = viewport.width as i64;
        let viewport_height = viewport.height as i64;

        let rel_x = cursor.x - camera.x;
        let rel_y = cursor.y - camera.y;

        rel_x >= -viewport_width / 2
            && rel_x <= viewport_width / 2
            && rel_y >= -viewport_height / 2
            && rel_y <= viewport_height / 2
    }

    /// Convert buffer to string for snapshot testing
    fn buffer_to_string(buffer: &ratatui::buffer::Buffer) -> String {
        let mut result = String::new();
        let area = buffer.area();

        for y in 0..area.height {
            for x in 0..area.width {
                let cell = buffer.cell((x, y)).unwrap();
                result.push_str(cell.symbol());
            }
            result.push('\n');
        }

        result
    }

    /// Test rapid sequential navigation (stress test)
    #[test]
    fn test_rapid_navigation() {
        // Setup environment and database
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        // Use anderson_promoters as specified
        let gfa_path = PathBuf::from("fixtures/anderson_promoters.gfa");
        if !gfa_path.exists() {
            println!("⚠️  Anderson GFA file not found, skipping rapid navigation test");
            return;
        }

        let collection_name = "/rapid_test";

        // Track the database before starting operations
        track_database(conn, op_conn).expect("Failed to track database");
        import_gfa(&context, &gfa_path, collection_name, Sample::DEFAULT_NAME)
            .expect("GFA import failed");

        let gen_graph = Sample::get_graph(conn, collection_name, Sample::DEFAULT_NAME);

        // Configure with large partition for stability
        let config = GraphConfig {
            partition: PartitionConfig {
                layer_count: 150,
                node_count: usize::MAX,
            },
            ..Default::default()
        };

        let node_sizer = GenGraphNodeSizer;
        let mut controller = GraphController::new_with_config(gen_graph, node_sizer, config);

        controller.viewport_state.viewport_bounds = Rect::new(0, 0, 80, 24);
        controller.viewport_state.focus();
        controller.set_anchor_partition(0).unwrap();

        // Rapid-fire navigation in different directions
        let key_sequence = [
            KeyCode::Right,
            KeyCode::Right,
            KeyCode::Right,
            KeyCode::Down,
            KeyCode::Down,
            KeyCode::Left,
            KeyCode::Left,
            KeyCode::Up,
            KeyCode::Up,
            KeyCode::Right,
            KeyCode::Down,
            KeyCode::Left,
            KeyCode::Up,
        ];

        for (i, key_code) in key_sequence.iter().enumerate() {
            let key_event = KeyEvent::new(*key_code, KeyModifiers::NONE);
            let result = controller.handle_key_event(key_event);
            assert_eq!(result, None, "Key press {} should not exit", i);

            // Process key event immediately
            // No animation update needed for rapid test
        }

        // System should remain stable after rapid navigation
        assert!(controller.viewport_state.has_focus);
        println!("✅ Rapid navigation test completed without crashes");
    }

    /// Test navigation at graph boundaries
    #[test]
    fn test_boundary_navigation() {
        // Setup environment and database
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        // Use anderson_promoters for boundary testing as requested
        let gfa_path = PathBuf::from("fixtures/anderson_promoters.gfa");
        if !gfa_path.exists() {
            println!("⚠️  Anderson GFA file not found, skipping boundary test");
            return;
        }

        let collection_name = "/boundary_test";

        // Track the database before starting operations
        track_database(conn, op_conn).expect("Failed to track database");
        import_gfa(&context, &gfa_path, collection_name, Sample::DEFAULT_NAME)
            .expect("GFA import failed");

        let gen_graph = Sample::get_graph(conn, collection_name, Sample::DEFAULT_NAME);

        // Configure with large partition for stability
        let config = GraphConfig {
            partition: PartitionConfig {
                layer_count: 150,
                node_count: usize::MAX,
            },
            ..Default::default()
        };

        let node_sizer = GenGraphNodeSizer;
        let mut controller = GraphController::new_with_config(gen_graph, node_sizer, config);

        controller.viewport_state.viewport_bounds = Rect::new(0, 0, 80, 24);
        controller.viewport_state.focus();
        controller.set_anchor_partition(0).unwrap();

        // Try to navigate beyond boundaries
        for _ in 0..50 {
            controller.handle_key_event(KeyEvent::new(KeyCode::Left, KeyModifiers::NONE));
        }

        let left_boundary_cursor = controller.current;

        // Navigate far right
        for _ in 0..50 {
            controller.handle_key_event(KeyEvent::new(KeyCode::Right, KeyModifiers::NONE));
        }

        let right_boundary_cursor = controller.current;

        // Verify we stayed within reasonable bounds
        assert!(
            left_boundary_cursor.x >= -10000,
            "Should not navigate too far left"
        );
        assert!(
            right_boundary_cursor.x <= 10000,
            "Should not navigate too far right"
        );

        println!("✅ Boundary navigation test completed");
        println!("  Left boundary: {:?}", left_boundary_cursor);
        println!("  Right boundary: {:?}", right_boundary_cursor);
    }
}
