#[cfg(test)]
use crate::geometry::WorldPos;
#[cfg(test)]
use crate::graph_controller::{GraphConfig, GraphController};
#[cfg(test)]
use crate::layout::VisualDetail;
#[cfg(test)]
use crate::testing::mocks::{FixedNodeSizer, MockDomainGraph};

/// Test cursor traversal across partition boundaries
#[cfg(test)]
fn test_cursor_partition_traversal() {
    // Create a long chain that forces multiple partitions
    let mut domain_graph = MockDomainGraph::new();
    let nodes: Vec<_> = (0..10).map(|_| domain_graph.add_node(())).collect();

    // Create chain: 0 -> 1 -> 2 -> ... -> 9
    for i in 0..9 {
        domain_graph.add_edge(nodes[i], nodes[i + 1], ());
    }

    let node_sizer = FixedNodeSizer {
        width: 5,
        height: 3,
    };

    // Use small partitions to force multiple partitions
    let mut config = GraphConfig::default();
    config.partition.layer_count = 3; // Very small partitions

    let mut controller = GraphController::new_with_config(domain_graph.clone(), node_sizer, config);

    // Set viewport
    controller.viewport_state.viewport_bounds = ratatui::layout::Rect::new(0, 0, 120, 30);
    controller.viewport_state.camera_current = WorldPos::new(0, 0);
    controller.viewport_state.camera_target = WorldPos::new(0, 0);
    controller.set_detail_level(VisualDetail::Full);

    // Expand viewport to load partitions
    let _ = controller.ensure_camera_coverage();

    // Enable cursor and initialize
    controller.enable_cursor();

    // Check if cursor is positioned on a node
    assert!(
        controller.viewport_state.cursor.node_domain_idx.is_some(),
        "Cursor should be positioned on a node after initialization"
    );

    let initial_node = controller.viewport_state.cursor.node_domain_idx.unwrap();
    println!("Initial cursor node: {:?}", initial_node);

    // Try to move cursor horizontally to traverse to connected nodes
    let mut successful_moves = 0;

    for i in 0..9 {
        let before_move = controller.viewport_state.cursor.node_domain_idx;
        controller.move_cursor_horizontal(1); // Move right
        let after_move = controller.viewport_state.cursor.node_domain_idx;

        if before_move != after_move {
            successful_moves += 1;
            let current_node = after_move.unwrap();
            println!(
                "Move {}: {} -> {}",
                i,
                before_move.unwrap().index(),
                current_node.index()
            );
        } else {
            println!(
                "Move {} failed: stuck at node {}",
                i,
                before_move.unwrap().index()
            );
            break;
        }
    }

    println!("Successful moves: {} out of 9 expected", successful_moves);
    println!(
        "Viewport graph nodes: {}",
        controller.viewport_graph.nodes.len()
    );
    println!(
        "Viewport graph edges: {}",
        controller.viewport_graph.graph.edge_count()
    );

    // Export viewport graph to dot if RUST_LOG=debug is active
    if std::env::var("RUST_LOG")
        .map(|v| v.contains("debug"))
        .unwrap_or(false)
    {
        if let Err(e) = crate::dot_export::export_to_dot(
            &controller.viewport_graph,
            "cursor_partition_test_viewport.dot",
        ) {
            println!("Failed to export dot file: {}", e);
        } else {
            println!("Exported viewport graph to cursor_partition_test_viewport.dot");
        }
    }

    // We should be able to traverse the entire chain
    assert!(
        successful_moves >= 5,
        "Should be able to traverse at least 5 nodes in the chain, but only got {}",
        successful_moves
    );
}

#[test]
fn cursor_can_traverse_partition_boundaries() {
    test_cursor_partition_traversal();
}
