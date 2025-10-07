use gen_widget::{
    geometry::{BigRect, WorldPos},
    graph_controller::GraphController,
    graph_widget::DefaultNodeSizer,
    layout::VisualDetail,
    testing::mocks::TestGraphs,
};
use ratatui::layout::Rect;

fn main() {
    eprintln!("Testing cursor terminal position preservation...\n");

    // Create a simple test graph
    let graph = TestGraphs::domain_diamond();
    let sizer = DefaultNodeSizer;

    let mut controller = GraphController::new(&graph, sizer);

    // Set up viewport - simulate a terminal area
    controller.viewport_state.viewport_bounds = Rect::new(5, 5, 40, 20);
    let viewport = BigRect::from_corners(WorldPos::new(-20, -10), WorldPos::new(20, 10));

    // Initial setup at Full detail
    eprintln!("=== Initial setup at Full detail ===");
    controller.set_detail_level(VisualDetail::Full);
    controller
        .update_viewport_graph(viewport, VisualDetail::Full)
        .expect("Failed to update viewport graph");

    // Enable and initialize cursor
    controller.enable_cursor();

    // Move cursor a bit to get away from default position
    controller.move_cursor_horizontal(3);
    controller.move_cursor_vertical(2);

    let initial_world = controller.viewport_state.cursor.current;
    let initial_terminal = controller.viewport_state.world_to_terminal(initial_world);
    let initial_node = controller.viewport_state.cursor.node_domain_idx;

    eprintln!("Initial state:");
    eprintln!("  World pos: {:?}", initial_world);
    eprintln!("  Terminal pos: {:?}", initial_terminal);
    eprintln!("  Node: {:?}\n", initial_node);

    // Change to Truncated detail level
    eprintln!("=== Changing to Truncated detail ===");
    controller.set_detail_level(VisualDetail::Truncated);
    controller
        .update_viewport_graph(viewport, VisualDetail::Truncated)
        .expect("Failed to update after detail change");

    let after_world = controller.viewport_state.cursor.current;
    let after_terminal = controller.viewport_state.world_to_terminal(after_world);
    let after_node = controller.viewport_state.cursor.node_domain_idx;

    eprintln!("After detail change:");
    eprintln!("  World pos: {:?}", after_world);
    eprintln!("  Terminal pos: {:?}", after_terminal);
    eprintln!("  Node: {:?}\n", after_node);

    // Check results
    if initial_node == after_node {
        eprintln!("✅ Node preserved: stayed on {:?}", initial_node);
    } else {
        eprintln!(
            "❌ Node changed: was {:?}, now {:?}",
            initial_node, after_node
        );
    }

    if let (Some((ix, iy)), Some((ax, ay))) = (initial_terminal, after_terminal) {
        let dx = (ax as i32 - ix as i32).abs();
        let dy = (ay as i32 - iy as i32).abs();
        if dx <= 1 && dy <= 1 {
            eprintln!(
                "✅ Terminal position preserved: ({}, {}) -> ({}, {})",
                ix, iy, ax, ay
            );
        } else {
            eprintln!(
                "❌ Terminal position changed: ({}, {}) -> ({}, {})",
                ix, iy, ax, ay
            );
            eprintln!("   Delta: ({}, {})", dx, dy);
        }
    } else {
        eprintln!("⚠️  Terminal position not available");
    }
}
