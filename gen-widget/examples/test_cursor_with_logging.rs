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

    println!("Testing cursor positioning with trace logging...\n");
    println!("Run with RUST_LOG=gen_widget=trace to see detailed logs\n");

    // Create a test graph
    let graph = TestGraphs::domain_diamond();
    let sizer = DefaultNodeSizer;

    let mut controller = GraphController::new(&graph, sizer);

    // Set up viewport
    controller.viewport_state.viewport_bounds = Rect::new(10, 5, 60, 30);
    let viewport = BigRect::from_corners(WorldPos::new(-30, -15), WorldPos::new(30, 15));

    // Initial setup
    controller.set_detail_level(VisualDetail::Full);
    controller
        .update_viewport_graph(viewport, VisualDetail::Full)
        .expect("Failed to update viewport graph");

    // Enable cursor and move it away from origin
    controller.enable_cursor();
    for _ in 0..5 {
        controller.move_cursor_horizontal(1);
    }
    for _ in 0..3 {
        controller.move_cursor_vertical(1);
    }

    let initial_world = controller.viewport_state.cursor.current;
    let initial_terminal = controller.viewport_state.world_to_terminal(initial_world);

    println!("Initial cursor:");
    println!("  World: {:?}", initial_world);
    println!("  Terminal: {:?}", initial_terminal);
    println!();

    // Test detail level change
    println!("Changing detail level from Full to Truncated...");
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
            println!("✅ Terminal position preserved exactly");
        } else {
            println!(
                "❌ Terminal position changed: ({}, {}) -> ({}, {})",
                ix, iy, ax, ay
            );
        }
    }

    println!();

    // Test disperse
    println!("Testing disperse operation...");
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
            println!("✅ Terminal position preserved after disperse");
        } else {
            println!(
                "❌ Terminal position changed: ({}, {}) -> ({}, {})",
                bx, by, ax, ay
            );
        }
    }
}
