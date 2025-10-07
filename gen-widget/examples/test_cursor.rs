use gen_widget::{
    geometry::{BigRect, WorldPos},
    graph_controller::GraphController,
    graph_widget::DefaultNodeSizer,
    layout::VisualDetail,
    testing::mocks::TestGraphs,
};

fn main() {
    eprintln!("Testing cursor preservation across detail level changes...\n");

    // Create a simple test graph
    let graph = TestGraphs::domain_diamond();
    let sizer = DefaultNodeSizer;

    let mut controller = GraphController::new(&graph, sizer);

    // Set up viewport
    let viewport = BigRect::from_corners(WorldPos::new(-50, -25), WorldPos::new(50, 25));

    // Initial setup at Full detail
    eprintln!("=== Setting detail level to Full ===");
    controller.set_detail_level(VisualDetail::Full);
    controller
        .rebuild_viewport_graph(viewport, VisualDetail::Full)
        .expect("Failed to update viewport graph");

    // Initialize cursor
    controller.initialize_cursor();
    let initial_cursor = controller.viewport_state.cursor.current;
    let initial_node = controller.viewport_state.cursor.node_domain_idx;
    eprintln!(
        "Initial cursor at {:?} on node {:?}\n",
        initial_cursor, initial_node
    );

    // Change to Truncated detail level
    eprintln!("=== Changing detail level to Truncated ===");
    controller.set_detail_level(VisualDetail::Truncated);
    controller
        .rebuild_viewport_graph(viewport, VisualDetail::Truncated)
        .expect("Failed to update viewport after detail change");

    let after_cursor = controller.viewport_state.cursor.current;
    let after_node = controller.viewport_state.cursor.node_domain_idx;
    eprintln!(
        "After change: cursor at {:?} on node {:?}\n",
        after_cursor, after_node
    );

    // Check if cursor stayed on same node
    if initial_node == after_node {
        eprintln!("✅ SUCCESS: Cursor stayed on same node!");
    } else {
        eprintln!("❌ FAILURE: Cursor moved to different node!");
        eprintln!("   Was on {:?}, now on {:?}", initial_node, after_node);
    }

    // Test disperse operation
    eprintln!("\n=== Testing disperse operation ===");
    controller.initialize_cursor(); // Reset cursor
    let before_disperse_cursor = controller.viewport_state.cursor.current;
    let before_disperse_node = controller.viewport_state.cursor.node_domain_idx;
    eprintln!(
        "Before disperse: cursor at {:?} on node {:?}",
        before_disperse_cursor, before_disperse_node
    );

    controller.disperse();
    controller
        .rebuild_viewport_graph(viewport, VisualDetail::Truncated)
        .expect("Failed to update after disperse");

    let after_disperse_cursor = controller.viewport_state.cursor.current;
    let after_disperse_node = controller.viewport_state.cursor.node_domain_idx;
    eprintln!(
        "After disperse: cursor at {:?} on node {:?}",
        after_disperse_cursor, after_disperse_node
    );

    if before_disperse_node == after_disperse_node {
        eprintln!("✅ SUCCESS: Cursor stayed on same node after disperse!");
    } else {
        eprintln!("❌ FAILURE: Cursor moved to different node after disperse!");
    }
}
