use gen_widget::{
    geometry::{BigRect, WorldPos},
    graph_controller::GraphController,
    graph_widget::DefaultNodeSizer,
    layout::VisualDetail,
    testing::mocks::TestGraphs,
};
use ratatui::layout::Rect;

fn main() {
    println!("Testing camera adjustment for cursor positioning...\n");

    // Create a test graph
    let graph = TestGraphs::domain_diamond();
    let sizer = DefaultNodeSizer;

    let mut controller = GraphController::new(&graph, sizer);

    // Set up viewport - simulate a terminal area at position (10, 5) with size 60x30
    controller.viewport_state.viewport_bounds = Rect::new(10, 5, 60, 30);
    let viewport = BigRect::from_corners(WorldPos::new(-30, -15), WorldPos::new(30, 15));

    // Initial setup at Full detail
    println!("=== Initial setup ===");
    controller.set_detail_level(VisualDetail::Full);
    controller
        .rebuild_viewport_graph(viewport, VisualDetail::Full)
        .expect("Failed to update viewport graph");

    // Enable cursor
    controller.enable_cursor();

    // Move cursor to a specific position
    for _ in 0..5 {
        controller.move_cursor_horizontal(1);
    }
    for _ in 0..3 {
        controller.move_cursor_vertical(1);
    }

    let initial_world = controller.viewport_state.cursor.current;
    let initial_terminal = controller.viewport_state.world_to_terminal(initial_world);
    let initial_node = controller.viewport_state.cursor.node_domain_idx;
    let initial_offset = controller.viewport_state.cursor.node_offset;
    let initial_camera = controller.viewport_state.camera_current;

    println!("Before detail change:");
    println!("  Camera: {:?}", initial_camera);
    println!("  Cursor world: {:?}", initial_world);
    println!("  Cursor terminal: {:?}", initial_terminal);
    println!("  Node: {:?}, offset: {:?}", initial_node, initial_offset);
    println!();

    // Change detail level
    println!("=== Changing to Truncated ===");
    controller.set_detail_level(VisualDetail::Truncated);

    // Before update, add debug info
    println!(
        "Camera before update: {:?}",
        controller.viewport_state.camera_current
    );
    println!(
        "Stored viewport pos: {:?}",
        controller.viewport_state.cursor.stored_viewport_pos
    );

    controller
        .rebuild_viewport_graph(viewport, VisualDetail::Truncated)
        .expect("Failed to update after detail change");

    let after_world = controller.viewport_state.cursor.current;
    let after_terminal = controller.viewport_state.world_to_terminal(after_world);
    let after_node = controller.viewport_state.cursor.node_domain_idx;
    let after_offset = controller.viewport_state.cursor.node_offset;
    let after_camera = controller.viewport_state.camera_current;

    println!("\nAfter detail change:");
    println!("  Camera: {:?}", after_camera);
    println!("  Cursor world: {:?}", after_world);
    println!("  Cursor terminal: {:?}", after_terminal);
    println!("  Node: {:?}, offset: {:?}", after_node, after_offset);
    println!();

    // Check results
    println!("=== Results ===");

    if initial_node == after_node {
        println!("✅ Node preserved");
    } else {
        println!(
            "❌ Node changed from {:?} to {:?}",
            initial_node, after_node
        );
    }

    if let (Some((ix, iy)), Some((ax, ay))) = (initial_terminal, after_terminal) {
        if ix == ax && iy == ay {
            println!("✅ Terminal position EXACTLY preserved: ({}, {})", ix, iy);
        } else {
            let dx = (ax as i32 - ix as i32).abs();
            let dy = (ay as i32 - iy as i32).abs();
            if dx <= 1 && dy <= 1 {
                println!(
                    "✅ Terminal position approximately preserved: ({}, {}) -> ({}, {})",
                    ix, iy, ax, ay
                );
            } else {
                println!(
                    "❌ Terminal position changed: ({}, {}) -> ({}, {})",
                    ix, iy, ax, ay
                );
                println!(
                    "   Delta: ({}, {})",
                    ax as i32 - ix as i32,
                    ay as i32 - iy as i32
                );
            }
        }
    } else {
        println!("⚠️  Terminal position not available");
    }

    // Test disperse
    println!("\n=== Testing disperse ===");
    let before_disperse_terminal = controller
        .viewport_state
        .world_to_terminal(controller.viewport_state.cursor.current);

    controller.disperse();
    controller
        .rebuild_viewport_graph(viewport, VisualDetail::Truncated)
        .expect("Failed to update after disperse");

    let after_disperse_terminal = controller
        .viewport_state
        .world_to_terminal(controller.viewport_state.cursor.current);

    if let (Some((bx, by)), Some((ax, ay))) = (before_disperse_terminal, after_disperse_terminal) {
        if bx == ax && by == ay {
            println!(
                "✅ Terminal position preserved after disperse: ({}, {})",
                bx, by
            );
        } else {
            println!(
                "❌ Terminal position changed after disperse: ({}, {}) -> ({}, {})",
                bx, by, ax, ay
            );
        }
    }
}
