use gen_widget::{
    geometry::{BigRect, WorldPos},
    graph_controller::GraphController,
    graph_widget::DefaultNodeSizer,
    layout::VisualDetail,
    testing::mocks::TestGraphs,
};
use ratatui::layout::Rect;

fn main() {
    // Initialize logger
    env_logger::init();

    println!("Testing Y-axis cursor positioning...\n");
    println!("Run with RUST_LOG=gen_widget=trace to see detailed logs\n");

    // Create a test graph with more vertical structure
    let graph = TestGraphs::domain_complex_dag();
    let sizer = DefaultNodeSizer;

    let mut controller = GraphController::new(&graph, sizer);

    // Set up viewport
    controller.viewport_state.viewport_bounds = Rect::new(5, 3, 80, 40);
    let viewport = BigRect::from_corners(WorldPos::new(-40, -20), WorldPos::new(40, 20));

    // Initial setup
    controller.set_detail_level(VisualDetail::Full);
    controller
        .update_viewport_graph(viewport, VisualDetail::Full)
        .expect("Failed to update viewport graph");

    // Enable cursor and move it both horizontally and vertically
    controller.enable_cursor();

    // Move right and up to get to a non-origin position
    for _ in 0..8 {
        controller.move_cursor_horizontal(1);
    }
    for _ in 0..5 {
        controller.move_cursor_vertical(1); // Move UP in world coordinates
    }

    let initial_world = controller.viewport_state.cursor.current;
    let initial_terminal = controller.viewport_state.world_to_terminal(initial_world);

    println!("Initial cursor after movements:");
    println!("  World: {:?}", initial_world);
    println!(
        "  Terminal: {:?} (should be higher on screen, lower Y value)",
        initial_terminal
    );
    println!();

    // Test detail level change
    println!("Changing detail level to test Y-axis preservation...");
    controller.set_detail_level(VisualDetail::Truncated);
    controller
        .update_viewport_graph(viewport, VisualDetail::Truncated)
        .expect("Failed to update after detail change");

    let after_world = controller.viewport_state.cursor.current;
    let after_terminal = controller.viewport_state.world_to_terminal(after_world);

    println!("After detail change:");
    println!("  World: {:?}", after_world);
    println!("  Terminal: {:?}", after_terminal);

    if let (Some((ix, iy)), Some((ax, ay))) = (initial_terminal, after_terminal) {
        if ix == ax && iy == ay {
            println!("✅ Terminal position preserved exactly (X and Y)");
        } else {
            println!("❌ Terminal position changed:");
            println!("   X: {} -> {} (delta: {})", ix, ax, ax as i32 - ix as i32);
            println!("   Y: {} -> {} (delta: {})", iy, ay, ay as i32 - iy as i32);
        }
    }

    println!();

    // Test disperse with Y position
    println!("Testing disperse with non-zero Y position...");
    let before_terminal = controller
        .viewport_state
        .world_to_terminal(controller.viewport_state.cursor.current);

    controller.disperse();
    controller
        .update_viewport_graph(viewport, VisualDetail::Truncated)
        .expect("Failed to update after disperse");

    let after_disperse_terminal = controller
        .viewport_state
        .world_to_terminal(controller.viewport_state.cursor.current);

    println!("Before disperse: Terminal {:?}", before_terminal);
    println!("After disperse: Terminal {:?}", after_disperse_terminal);

    if let (Some((bx, by)), Some((ax, ay))) = (before_terminal, after_disperse_terminal) {
        if bx == ax && by == ay {
            println!("✅ Terminal position preserved after disperse (X and Y)");
        } else {
            println!("❌ Terminal position changed:");
            println!("   X: {} -> {} (delta: {})", bx, ax, ax as i32 - bx as i32);
            println!("   Y: {} -> {} (delta: {})", by, ay, ay as i32 - by as i32);
        }
    }
}
