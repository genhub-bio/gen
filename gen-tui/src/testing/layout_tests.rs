#[cfg(test)]
use crate::graph_controller::{GraphController, WorldBuffer};
#[cfg(test)]
use crate::layout::VisualDetail;
#[cfg(test)]
use crate::plotter::plot_viewport_graph;
#[cfg(test)]
use crate::testing::create_test_terminal;
#[cfg(test)]
use crate::testing::mocks::{FixedNodeSizer, MockDomainGraph, TestRenderers};

/// Helper function to create viewport-based visual snapshots using GraphController
#[cfg(test)]
fn make_snapshot_custom<NS, R>(
    domain_graph: MockDomainGraph,
    viewport_width: u16,
    viewport_height: u16,
    layer_count: usize,
    node_count: usize,
    node_sizer: NS,
    mut renderer: R,
) -> String
where
    NS: crate::plotter::NodeSizer<MockDomainGraph>,
    R: crate::plotter::NodeRenderer<MockDomainGraph>,
{
    use crate::graph_controller::GraphConfig;

    let mut terminal = create_test_terminal(viewport_width, viewport_height);

    // // alternatively - very minimalist node labels:
    // let mut renderer = TestRenderers::minimal();
    // let node_sizer = TestNodeSizers::fixed_1x1();

    // Use configurable partitions for testing
    let mut config = GraphConfig::default();
    config.partition.layer_count = layer_count;
    config.partition.node_count = node_count;
    let mut controller = GraphController::new_with_config(domain_graph.clone(), node_sizer, config);

    let test_viewport = ratatui::layout::Rect::new(0, 0, viewport_width, viewport_height);
    controller.viewport_state.viewport_bounds = test_viewport;

    // Set detail level before the camera
    controller.set_detail_level(VisualDetail::Full);

    let result = terminal.draw(|f| {
        let area = f.area();
        controller.viewport_state.viewport_bounds = area;
        let loaded_partitions = controller.ensure_camera_coverage();
        let partition_indices = loaded_partitions.unwrap_or_default();
        println!(
            "number of partitions loaded: {}, indices: {:?}",
            partition_indices.len(),
            partition_indices
        );

        controller
            .rebuild_viewport_graph()
            .expect("Failed to rebuild viewport graph for snapshot generation");
        let viewport_graph = controller.get_viewport_graph();
        let detail_level = controller.get_detail_level();

        // Export viewport graph to dot if RUST_LOG=debug is active
        if std::env::var("RUST_LOG")
            .map(|v| v.contains("debug"))
            .unwrap_or(false)
        {
            // Generate a filename based on the current test name
            let current_thread = std::thread::current();
            let test_name = current_thread.name().unwrap_or("unknown_test");
            let filename = format!("{}_viewport.dot", test_name);
            if let Err(e) = crate::dot_export::export_to_dot(viewport_graph, &filename) {
                eprintln!("Failed to export dot file {}: {}", filename, e);
            }
        }

        let mut buffer = WorldBuffer::new(f.buffer_mut(), &controller.viewport_state);
        plot_viewport_graph(
            viewport_graph,
            &mut buffer,
            &mut renderer,
            controller.graph(),
            detail_level,
            &crate::theme::current_theme(),
        );
    });

    match result {
        Ok(_) => format!("{}", terminal.backend()),
        Err(e) => format!("Rendering failed: {}", e),
    }
}

/// Helper function to create viewport-based visual snapshots with default node sizer and renderer
#[cfg(test)]
fn make_snapshot(
    domain_graph: MockDomainGraph,
    viewport_width: u16,
    viewport_height: u16,
    layer_count: usize,
    node_count: usize,
) -> String {
    let node_sizer = FixedNodeSizer {
        width: 5,
        height: 3,
    };
    let renderer = TestRenderers::debug();

    make_snapshot_custom(
        domain_graph,
        viewport_width,
        viewport_height,
        layer_count,
        node_count,
        node_sizer,
        renderer,
    )
}

/// Like `make_snapshot`, but rewrites `backward_edges` onto pin nodes via
/// `GraphController::new_with_backward_edges`, so a cyclic domain graph renders as a loop
/// instead of panicking.
#[cfg(test)]
fn make_snapshot_with_backward_edges(
    domain_graph: MockDomainGraph,
    viewport_width: u16,
    viewport_height: u16,
    layer_count: usize,
    node_count: usize,
    backward_edges: &[(petgraph::graph::NodeIndex, petgraph::graph::NodeIndex)],
) -> String {
    use crate::graph_controller::GraphConfig;

    let node_sizer = FixedNodeSizer {
        width: 5,
        height: 3,
    };
    let mut renderer = TestRenderers::debug();

    let mut terminal = create_test_terminal(viewport_width, viewport_height);

    let mut config = GraphConfig::default();
    config.partition.layer_count = layer_count;
    config.partition.node_count = node_count;
    let mut controller = GraphController::new_with_config_and_backward_edges(
        domain_graph.clone(),
        node_sizer,
        config,
        backward_edges,
    );

    let test_viewport = ratatui::layout::Rect::new(0, 0, viewport_width, viewport_height);
    controller.viewport_state.viewport_bounds = test_viewport;
    controller.set_detail_level(VisualDetail::Full);

    let result = terminal.draw(|f| {
        let area = f.area();
        controller.viewport_state.viewport_bounds = area;
        let _ = controller.ensure_camera_coverage();

        controller
            .rebuild_viewport_graph()
            .expect("Failed to rebuild viewport graph for snapshot generation");
        let viewport_graph = controller.get_viewport_graph();
        let detail_level = controller.get_detail_level();

        if std::env::var("RUST_LOG")
            .map(|v| v.contains("debug"))
            .unwrap_or(false)
        {
            let current_thread = std::thread::current();
            let test_name = current_thread.name().unwrap_or("unknown_test");
            let vp_filename = format!("{}_viewport.dot", test_name);
            if let Err(e) = crate::dot_export::export_to_dot(viewport_graph, &vp_filename) {
                eprintln!("Failed to export viewport dot {}: {}", vp_filename, e);
            }
            for (partition_idx, partition) in controller
                .partition_controller
                .partition_table
                .partitions
                .iter()
                .enumerate()
            {
                if let Some(layout) = partition.layouts[detail_level.as_index()].as_ref() {
                    let layout_filename =
                        format!("{}_partition{}_layout.dot", test_name, partition_idx);
                    if let Err(e) = crate::dot_export::export_layout_graph_to_dot(
                        &layout.graph,
                        &layout_filename,
                    ) {
                        eprintln!("Failed to export layout dot {}: {}", layout_filename, e);
                    }
                }
            }
        }

        let mut buffer = WorldBuffer::new(f.buffer_mut(), &controller.viewport_state);
        plot_viewport_graph(
            viewport_graph,
            &mut buffer,
            &mut renderer,
            controller.graph(),
            detail_level,
            &crate::theme::current_theme(),
        );
    });

    match result {
        Ok(_) => format!("{}", terminal.backend()),
        Err(e) => format!("Rendering failed: {}", e),
    }
}

#[test]
fn viewport_visual_regression_simple_chain() {
    let _ = env_logger::try_init();
    // Create a simple chain domain graph: 0 -> 1 -> 2
    let mut domain_graph = MockDomainGraph::new();
    let node_0 = domain_graph.add_node(());
    let node_1 = domain_graph.add_node(());
    let node_2 = domain_graph.add_node(());
    domain_graph.add_edge(node_0, node_1, ());
    domain_graph.add_edge(node_1, node_2, ());

    let snapshot = make_snapshot(domain_graph, 60, 20, 2, 8);
    insta::assert_snapshot!("simple_chain", snapshot);
}

#[test]
fn viewport_visual_regression_diamond() {
    let _ = env_logger::try_init();
    // Create a diamond domain graph: 0 -> {1, 2} -> 3
    let mut domain_graph = MockDomainGraph::new();
    let node_0 = domain_graph.add_node(());
    let node_1 = domain_graph.add_node(());
    let node_2 = domain_graph.add_node(());
    let node_3 = domain_graph.add_node(());
    domain_graph.add_edge(node_0, node_1, ());
    domain_graph.add_edge(node_0, node_2, ());
    domain_graph.add_edge(node_1, node_3, ());
    domain_graph.add_edge(node_2, node_3, ());

    let snapshot = make_snapshot(domain_graph, 60, 20, 2, 8);
    insta::assert_snapshot!("diamond", snapshot);
}

#[test]
fn viewport_visual_regression_single_node() {
    let _ = env_logger::try_init();
    // Create a single node domain graph
    let mut domain_graph = MockDomainGraph::new();
    domain_graph.add_node(());

    let snapshot = make_snapshot(domain_graph, 60, 20, 2, 8);
    insta::assert_snapshot!("single_node", snapshot);
}

#[test]
fn viewport_visual_regression_subcombinatorial_dag() {
    let _ = env_logger::try_init();
    // Create a DAG in which two subsequent layers are not fully connected all-to-all.
    // This tests the challenge of handling complex edge routing between layers.
    let mut domain_graph = MockDomainGraph::new();
    let nodes: Vec<_> = (0..6).map(|_| domain_graph.add_node(())).collect();

    // Create edges: 0->{1,2}, 1->3, 2->{3,4}, 3->5, 4->5
    domain_graph.add_edge(nodes[0], nodes[1], ());
    domain_graph.add_edge(nodes[0], nodes[2], ());
    domain_graph.add_edge(nodes[1], nodes[3], ());
    domain_graph.add_edge(nodes[2], nodes[3], ());
    domain_graph.add_edge(nodes[2], nodes[4], ());
    domain_graph.add_edge(nodes[3], nodes[5], ());
    domain_graph.add_edge(nodes[4], nodes[5], ());

    let snapshot = make_snapshot(domain_graph, 60, 20, 2, 8);
    insta::assert_snapshot!("subcombinatorial_dag", snapshot);
}

#[test]
fn viewport_visual_regression_complex_dag() {
    let _ = env_logger::try_init();
    // Create the original complex DAG structure matching TestGraphs::complex_dag()
    // This is a hierarchical 9-node DAG with multiple levels and convergence points
    let mut domain_graph = MockDomainGraph::new();
    let nodes: Vec<_> = (0..9).map(|_| domain_graph.add_node(())).collect();

    // Create the complex hierarchical structure:
    // 0 -> {1, 2}
    // 1 -> {3, 4}
    // 2 -> {4, 5}
    // 3 -> 6
    // 4 -> {6, 7}
    // 5 -> 7
    // 6 -> 8
    // 7 -> 8
    domain_graph.add_edge(nodes[0], nodes[1], ());
    domain_graph.add_edge(nodes[0], nodes[2], ());
    domain_graph.add_edge(nodes[1], nodes[3], ());
    domain_graph.add_edge(nodes[1], nodes[4], ());
    domain_graph.add_edge(nodes[2], nodes[4], ());
    domain_graph.add_edge(nodes[2], nodes[5], ());
    domain_graph.add_edge(nodes[3], nodes[6], ());
    domain_graph.add_edge(nodes[4], nodes[6], ());
    domain_graph.add_edge(nodes[4], nodes[7], ());
    domain_graph.add_edge(nodes[5], nodes[7], ());
    domain_graph.add_edge(nodes[6], nodes[8], ());
    domain_graph.add_edge(nodes[7], nodes[8], ());

    let snapshot = make_snapshot(domain_graph, 60, 20, 2, 8);
    insta::assert_snapshot!("complex_dag", snapshot);
}

#[test]
fn viewport_multi_partition_boundary_handling() {
    let _ = env_logger::try_init();
    // Create a wide graph that forces multiple partitions to test boundary handling
    let mut domain_graph = MockDomainGraph::new();
    let nodes: Vec<_> = (0..20).map(|_| domain_graph.add_node(())).collect();

    // Create a long chain that should force multiple partitions
    for i in 0..19 {
        domain_graph.add_edge(nodes[i], nodes[i + 1], ());
    }

    let snapshot = make_snapshot(domain_graph, 120, 30, 3, 5); // Wide viewport

    insta::assert_snapshot!("multi_partition_chain", snapshot);
}

#[test]
fn viewport_visual_regression_extended_complex_dag_no_partitioning() {
    let _ = env_logger::try_init();
    // Test 1: No partitioning - everything in one partition
    use crate::testing::mocks::TestGraphs;
    let domain_graph = TestGraphs::domain_complex_dag();

    let snapshot = make_snapshot(domain_graph, 80, 25, usize::MAX, usize::MAX);
    insta::assert_snapshot!("extended_complex_dag_no_partitioning", snapshot);
}

#[test]
fn viewport_visual_regression_extended_complex_dag_layer_partitioning() {
    let _ = env_logger::try_init();
    // Test 2: Layer-based partitioning
    use crate::testing::mocks::TestGraphs;
    let domain_graph = TestGraphs::domain_complex_dag();

    let snapshot = make_snapshot(domain_graph, 80, 25, 3, usize::MAX);
    insta::assert_snapshot!("extended_complex_dag_layer_partitioning", snapshot);
}

// This test is a good example of why we try to go for articulation points:
//  By breaking up the graph between layers that each have multiple nodes
//  suboptimal node orderings are encountered. This is also non-deterministic,
//  hence disabling this test.
//
//  Valid outcome, but ugly:
//
//                        █████
//                    ╭───█N5██───╮
//            █████ ╭─╯   █████   │ █████
//          ╭─█N1██─│─╮           ├─█N7██─╮
//    █████ │ █████ │ ├─╮ █████ ╭─╯ █████ │ █████ █████
//    █N0██─┤       │ │ ├─█N4██─┤         ├─█N8██─█N9██
//    █████ │ █████ ├─│─╯ █████ ╰─╮ █████ │ █████ █████
//          ╰─█N2██─╯ │           ├─█N6██─╯
//            █████   │   █████   │ █████
//                    ╰───█N3██───╯
//                        █████
//
// Ideal outcome:
//                      █████
//                  ╭───█N3██───╮
//            █████ │   █████   │ █████
//          ╭─█N1██─┤           ├─█N6██─╮
//    █████ │ █████ ╰─╮ █████ ╭─╯ █████ │ █████ █████
//    █N0██─┤         ├─█N4██─┤         ├─█N8██─█N9██
//    █████ │ █████ ╭─╯ █████ ╰─╮ █████ │ █████ █████
//          ╰─█N2██─┤           ├─█N7██─╯
//            █████ │   █████   │ █████
//                  ╰───█N5██───╯
//                      █████
#[test]
fn viewport_visual_regression_extended_complex_dag_node_partitioning() {
    let _ = env_logger::try_init();
    // Test 3: Node-based partitioning
    use crate::testing::mocks::TestGraphs;
    let domain_graph = TestGraphs::domain_complex_dag();

    let snapshot = make_snapshot(domain_graph, 80, 25, usize::MAX, 3);
    insta::assert_snapshot!("extended_complex_dag_node_partitioning", snapshot);
}

#[test]
fn viewport_visual_regression_extended_diamond_no_partitioning() {
    let _ = env_logger::try_init();
    // Test 1: No partitioning - everything in one partition
    use crate::testing::mocks::TestGraphs;
    let domain_graph = TestGraphs::domain_extended_diamond();

    let snapshot = make_snapshot(domain_graph, 80, 25, usize::MAX, usize::MAX);
    insta::assert_snapshot!("extended_diamond_no_partitioning", snapshot);
}

#[test]
fn viewport_visual_regression_extended_diamond_layer_partitioning() {
    let _ = env_logger::try_init();
    // Test 2: Layer-based partitioning
    use crate::testing::mocks::TestGraphs;
    let domain_graph = TestGraphs::domain_extended_diamond();

    let snapshot = make_snapshot(domain_graph, 80, 25, 3, usize::MAX);
    insta::assert_snapshot!("extended_diamond_layer_partitioning", snapshot);
}

#[test]
fn viewport_visual_regression_extended_diamond_node_partitioning() {
    let _ = env_logger::try_init();
    // Test 3: Node-based partitioning
    use crate::testing::mocks::TestGraphs;
    let domain_graph = TestGraphs::domain_extended_diamond();

    let snapshot = make_snapshot(domain_graph, 80, 25, usize::MAX, 3);
    insta::assert_snapshot!("extended_diamond_node_partitioning", snapshot);
}

#[test]
fn test_layer_coordinate_alignment_and_ordering() {
    let _ = env_logger::try_init();

    use crate::{
        geometry::WorldPos,
        graph_controller::{GraphConfig, GraphController},
        layout::VisualDetail,
        testing::mocks::{FixedNodeSizer, TestGraphs},
    };

    // Create domain_extended_diamond with partitioning to test layer alignment
    let domain_graph = TestGraphs::domain_extended_diamond();

    let node_sizer = FixedNodeSizer {
        width: 5,
        height: 3,
    };

    // Use partitioning settings that will force partition creation
    let mut config = GraphConfig::default();
    config.partition.layer_count = 2; // Force layer-based partitioning
    config.partition.node_count = 3;
    config.controller.max_loaded_partitions = 5; // Force node-based partitioning as well

    let mut controller = GraphController::new_with_config(domain_graph.clone(), node_sizer, config);

    // Set detail level first
    controller.set_detail_level(VisualDetail::Full);

    // Set camera bounds to cover the unlimited viewport BEFORE creating viewport graph
    controller.viewport_state.viewport_bounds =
        ratatui::layout::Rect::new(0, 0, u16::MAX / 2, u16::MAX / 2);
    controller.viewport_state.camera_current = WorldPos::new(0, 0);
    controller.viewport_state.camera_target = WorldPos::new(0, 0);

    // Get total partition count first to verify all are loaded
    let total_partition_count = controller
        .partition_controller
        .partition_table
        .partitions
        .len();
    println!("Total partitions in graph: {}", total_partition_count);

    // Load all partitions by ensuring camera coverage
    let loaded_partitions = controller.ensure_camera_coverage().unwrap_or_default();
    println!("Number of partitions loaded: {}", loaded_partitions.len());

    // Assert that ALL partitions were loaded, not just multiple
    assert_eq!(
        loaded_partitions.len(),
        total_partition_count,
        "Expected all {} partitions to be loaded, but only {} were loaded",
        total_partition_count,
        loaded_partitions.len()
    );

    // Rebuild the viewport graph with unlimited bounds
    let result = controller.rebuild_viewport_graph();
    assert!(
        result.is_ok(),
        "Failed to rebuild viewport graph: {:?}",
        result
    );

    let viewport_graph = controller.get_viewport_graph();

    // Verify we have layers
    assert!(
        viewport_graph.layer_count() > 0,
        "No layers found in viewport graph"
    );
    println!("Viewport graph has {} layers", viewport_graph.layer_count());

    // Group nodes by layer and collect their x-coordinates
    let mut layer_x_coords: Vec<Vec<i64>> = Vec::new();

    for layer_idx in 0..viewport_graph.layer_count() {
        if let Some(layer_nodes) = viewport_graph.get_layer(layer_idx) {
            let mut x_coords = Vec::new();

            for &domain_node in layer_nodes {
                // Find world position for this domain node
                if let Some(world_pos) = viewport_graph.node_positions.get(&domain_node) {
                    x_coords.push(world_pos.x);
                } else {
                    panic!(
                        "Domain node {:?} in layer {} not found in domain_to_world mapping",
                        domain_node, layer_idx
                    );
                }
            }

            println!(
                "Layer {}: {} nodes with x-coordinates: {:?}",
                layer_idx,
                x_coords.len(),
                x_coords
            );
            layer_x_coords.push(x_coords);
        }
    }

    // Test 1: Verify that all nodes in each layer share the same x-coordinate
    for (layer_idx, x_coords) in layer_x_coords.iter().enumerate() {
        if !x_coords.is_empty() {
            let first_x = x_coords[0];
            for &x in x_coords {
                assert_eq!(
                    x, first_x,
                    "Layer {} has nodes with different x-coordinates: expected all to be {}, but found {}",
                    layer_idx, first_x, x
                );
            }
            println!(
                "Layer {} has consistent x-coordinate: {}",
                layer_idx, first_x
            );
        }
    }

    // Test 2: Verify that x-coordinates are ordered between layers (increasing from layer to layer)
    let layer_x_representatives: Vec<i64> = layer_x_coords
        .iter()
        .filter(|coords| !coords.is_empty())
        .map(|coords| coords[0]) // Take first (they should all be the same per layer)
        .collect();

    for i in 1..layer_x_representatives.len() {
        let prev_x = layer_x_representatives[i - 1];
        let curr_x = layer_x_representatives[i];
        assert!(
            curr_x > prev_x,
            "Layer {} x-coordinate ({}) should be greater than layer {} x-coordinate ({})",
            i,
            curr_x,
            i - 1,
            prev_x
        );
    }

    println!(
        "Layer x-coordinates are properly ordered: {:?}",
        layer_x_representatives
    );

    // Additional verification: Check that we have the expected layer structure for extended diamond
    // Expected structure: [0] -> [1,2] -> [3] -> [4,5] -> [6] -> [7]
    // So we should have layers with node counts roughly matching this pattern
    let layer_sizes: Vec<usize> = layer_x_coords.iter().map(|coords| coords.len()).collect();
    println!("Layer sizes: {:?}", layer_sizes);

    // For extended diamond, we expect some layers to have multiple nodes (the diamond middles)
    let has_multi_node_layers = layer_sizes.iter().any(|&size| size > 1);
    assert!(
        has_multi_node_layers,
        "Expected some layers to have multiple nodes for diamond structure, but all layers have single nodes"
    );

    println!("Test completed successfully - layer coordinate alignment and ordering verified");
}

#[test]
fn viewport_visual_regression_bridge_position_with_variable_node_widths() {
    use crate::{
        layout::VisualDetail,
        plotter::NodeSizer,
        testing::mocks::{MockDomainGraph, TestGraphs, TestRenderers},
    };

    let _ = env_logger::try_init();

    // Custom node sizer with dramatically different widths for middle layer
    #[derive(Debug, Clone)]
    struct VariableWidthSizer;

    impl NodeSizer<MockDomainGraph> for VariableWidthSizer {
        fn get_node_size(
            &self,
            node: &petgraph::stable_graph::NodeIndex<u32>,
            _scale: VisualDetail,
        ) -> (u64, u64) {
            match node.index() {
                0 => (4, 1),  // Start node: medium width
                1 => (15, 2), // Left middle node: very wide
                2 => (2, 1),  // Right middle node: very narrow
                3 => (5, 1),  // End node: medium width
                _ => (3, 1),  // Default
            }
        }

        fn get_dummy_size(&self) -> (u64, u64) {
            (1, 1)
        }
    }

    let node_sizer = VariableWidthSizer;
    let renderer = TestRenderers::debug();

    let snapshot = make_snapshot_custom(
        TestGraphs::domain_diamond(),
        80,
        25,
        2,
        3,
        node_sizer,
        renderer,
    );

    insta::assert_snapshot!("bridge_position_variable_widths", snapshot);
}

#[test]
fn test_skip_layer() {
    let _ = env_logger::try_init();
    use crate::testing::mocks::TestGraphs;

    let domain_graph = TestGraphs::domain_skip_layer();
    let snapshot = make_snapshot(domain_graph, 80, 25, usize::MAX, usize::MAX);

    insta::assert_snapshot!("skip_layer", snapshot);
}

#[test]
fn test_skip_layer_partition_boundary() {
    let _ = env_logger::try_init();
    use crate::testing::mocks::TestGraphs;

    let domain_graph = TestGraphs::domain_skip_layer();
    let snapshot = make_snapshot(domain_graph, 80, 25, 2, usize::MAX);

    insta::assert_snapshot!("skip_layer_partition_boundary", snapshot);
}

#[test]
fn viewport_chain_three_partitions_spanning_edge() {
    let _ = env_logger::try_init();
    // Create a chain graph divided into 3 partitions on layer basis
    // with one edge completely spanning the middle partition.
    //
    // Graph structure with layers:
    // Layer 0: [0]
    // Layer 1: [1]
    // Layer 2: [2]
    // Layer 3: [3]
    // Layer 4: [4]
    // Layer 5: [5]
    //
    // Partitions (layer_count=2):
    // Partition 0: Layers 0-1 (nodes 0, 1)
    // Partition 1 (middle): Layers 2-3 (nodes 2, 3)
    // Partition 2: Layers 4-5 (nodes 4, 5)
    //
    // Regular chain edges: 0->1->2->3->4->5
    // Spanning edge: 1->4 (spans middle partition completely)
    let mut domain_graph = MockDomainGraph::new();
    let nodes: Vec<_> = (0..6).map(|_| domain_graph.add_node(())).collect();

    // Create the chain
    for i in 0..5 {
        domain_graph.add_edge(nodes[i], nodes[i + 1], ());
    }

    // Add the spanning edge that completely skips the middle partition
    // Node 1 is in partition 0 (layer 1), node 4 is in partition 2 (layer 4)
    domain_graph.add_edge(nodes[1], nodes[4], ());

    // Use layer_count=2 to create 3 partitions from 6 layers
    // This creates partition boundaries at layers 2 and 4
    let snapshot = make_snapshot(domain_graph, 100, 30, 2, usize::MAX);

    insta::assert_snapshot!("chain_three_partitions_spanning_edge", snapshot);
}

#[test]
fn viewport_chain_five_partitions_long_spanning_edge() {
    let _ = env_logger::try_init();
    // Create a longer chain graph divided into 5 partitions (3 data + 2 bridge)
    // with one edge completely spanning the middle partitions.
    //
    // With node_count=5 and the spanning edge 1->8, we get:
    // Partition 0: 6 nodes (0-5)     - Data partition
    // Partition 1: 0 nodes           - Bridge partition for edges crossing from 0 to 2
    // Partition 2: 5 nodes (4-8)     - Data partition
    // Partition 3: 0 nodes           - Bridge partition for edges crossing from 2 to 4
    // Partition 4: 3 nodes (7-9)     - Data partition
    //
    // Regular chain edges: 0->1->2->3->4->5->6->7->8->9
    // Long spanning edge: 1->8 (spans bridge partitions 1 and 3, and data partition 2)
    let mut domain_graph = MockDomainGraph::new();
    let nodes: Vec<_> = (0..10).map(|_| domain_graph.add_node(())).collect();

    // Create the chain
    for i in 0..9 {
        domain_graph.add_edge(nodes[i], nodes[i + 1], ());
    }

    // Add the spanning edge that completely skips partitions 1, 2, and 3
    // Node 1 is in partition 0 (layer 1), node 8 is in partition 4 (layer 8)
    domain_graph.add_edge(nodes[1], nodes[8], ());

    // Note: if you cut off the partitions using node_count=5 the test will fail due to
    // a visual artefact, which is concession made when using node_count to create the cut.
    // Topologically, the graph was still correct.
    let snapshot = make_snapshot(domain_graph, 120, 35, 2, usize::MAX);

    insta::assert_snapshot!("chain_five_partitions_long_spanning_edge", snapshot);
}

#[test]
fn viewport_chain_five_partitions_verify_partition_count() {
    let _ = env_logger::try_init();
    use crate::{
        geometry::WorldPos,
        graph_algorithms::find_articulation_points,
        graph_controller::{GraphConfig, GraphController},
        layout::VisualDetail,
        testing::mocks::FixedNodeSizer,
    };

    // Create the same chain graph as above
    let mut domain_graph = MockDomainGraph::new();
    let nodes: Vec<_> = (0..10).map(|_| domain_graph.add_node(())).collect();
    for i in 0..9 {
        domain_graph.add_edge(nodes[i], nodes[i + 1], ());
    }
    domain_graph.add_edge(nodes[1], nodes[8], ());

    // Check articulation points
    let articulation_points = find_articulation_points(&domain_graph);
    println!(
        "Articulation points in 10-node chain: {:?}",
        articulation_points
    );
    println!(
        "Number of articulation points: {}",
        articulation_points.len()
    );

    let node_sizer = FixedNodeSizer {
        width: 5,
        height: 3,
    };

    // Configure for partitioning - use node_count to control partition size
    // With 10 nodes and node_count=5, we get 3 data partitions (6 + 5 + 3 nodes) + 2 bridge partitions
    let mut config = GraphConfig::default();
    config.partition.layer_count = 2;
    config.partition.node_count = 5; // Allow up to 5 nodes per partition

    let mut controller = GraphController::new_with_config(domain_graph.clone(), node_sizer, config);
    controller.set_detail_level(VisualDetail::Full);

    // Set viewport to cover everything
    controller.viewport_state.viewport_bounds =
        ratatui::layout::Rect::new(0, 0, u16::MAX / 2, u16::MAX / 2);
    controller.viewport_state.camera_current = WorldPos::new(0, 0);
    controller.viewport_state.camera_target = WorldPos::new(0, 0);

    // Check how many partitions are actually created
    let total_partition_count = controller
        .partition_controller
        .partition_table
        .partitions
        .len();
    println!("Total partitions created: {}", total_partition_count);

    // Verify we have 5 partitions total (3 data + 2 bridge partitions)
    assert_eq!(
        total_partition_count, 5,
        "Expected 5 partitions (3 data + 2 bridge), but found {}",
        total_partition_count
    );

    // Load all partitions
    let loaded_partitions = controller.ensure_camera_coverage().unwrap_or_default();
    println!("Partitions loaded: {}", loaded_partitions.len());
    assert_eq!(
        loaded_partitions.len(),
        total_partition_count,
        "Expected all partitions to be loaded"
    );

    // Print partition details and verify spanning edge crosses multiple partitions
    let mut data_partition_count = 0;
    for (i, partition) in controller
        .partition_controller
        .partition_table
        .partitions
        .iter()
        .enumerate()
    {
        println!("Partition {}: {} nodes", i, partition.graph.node_count());
        if partition.graph.node_count() > 0 {
            data_partition_count += 1;
        }
    }

    println!(
        "Data partitions: {}, Bridge partitions: {}",
        data_partition_count,
        total_partition_count - data_partition_count
    );

    // The spanning edge 1->8 should cross partitions 1, 2, and 3
    // Node 1 is in partition 0, node 8 is in partition 4
    println!(
        "✓ Confirmed: 5 partitions created (3 data + 2 bridge), spanning edge crosses middle partitions"
    );
}

#[test]
fn test_skip_layer_terminal_stitch_edge_bundles() {
    let _ = env_logger::try_init();

    use crate::{
        geometry::WorldPos,
        graph_controller::{GraphConfig, GraphController},
        layout::VisualDetail,
        testing::mocks::{FixedNodeSizer, TestGraphs},
    };

    let domain_graph = TestGraphs::domain_skip_layer();

    let node_sizer = FixedNodeSizer {
        width: 5,
        height: 3,
    };

    let mut config = GraphConfig::default();
    config.partition.layer_count = 2;
    config.partition.node_count = 3;
    config.controller.max_loaded_partitions = 5;

    let mut controller = GraphController::new_with_config(domain_graph.clone(), node_sizer, config);

    controller.set_detail_level(VisualDetail::Full);

    controller.viewport_state.viewport_bounds =
        ratatui::layout::Rect::new(0, 0, u16::MAX / 2, u16::MAX / 2);
    controller.initialize_cursor();
    controller.viewport_state.camera_current = WorldPos::new(0, 0);
    controller.viewport_state.camera_target = WorldPos::new(0, 0);

    let result = controller.rebuild_viewport_graph();
    assert!(
        result.is_ok(),
        "Failed to rebuild viewport graph: {:?}",
        result
    );

    let viewport_graph = controller.get_viewport_graph();

    // Count edges that lack bundles by iterating through the viewport graph
    let mut edges_without_bundles = 0;
    let mut total_edges = 0;

    for edge in viewport_graph.graph.all_edges() {
        total_edges += 1;
        let (source, target, edge_data) = edge;

        // Check if this edge has an empty bundle (indicating it's from terminal stitch nodes)
        if edge_data.is_empty() {
            edges_without_bundles += 1;
            println!(
                "Edge from {:?} to {:?} lacks bundle: {:?}",
                source, target, edge_data
            );
        }
    }

    println!("Total edges in viewport graph: {}", total_edges);
    println!("Edges without bundles: {}", edges_without_bundles);

    // After filtering out edges with empty bundles in viewport_graph.rs,
    // there should be 0 edges without bundles in the viewport graph
    assert_eq!(
        edges_without_bundles, 0,
        "Expected 0 edges without bundles (they should be filtered out at viewport graph creation), but found {}",
        edges_without_bundles
    );

    println!(
        "Test completed successfully - no unbundled edges in viewport graph (filtered at creation)"
    );
}

#[test]
fn viewport_even_width_node_spacing() {
    let _ = env_logger::try_init();
    // Test that even-width nodes have proper spacing for edge routing
    use crate::{
        layout::VisualDetail,
        plotter::NodeSizer,
        testing::mocks::{MockDomainGraph, TestRenderers},
    };

    // Create a simple diamond graph
    let mut domain_graph = MockDomainGraph::new();
    let node_0 = domain_graph.add_node(());
    let node_1 = domain_graph.add_node(());
    let node_2 = domain_graph.add_node(());
    let node_3 = domain_graph.add_node(());
    domain_graph.add_edge(node_0, node_1, ());
    domain_graph.add_edge(node_0, node_2, ());
    domain_graph.add_edge(node_1, node_3, ());
    domain_graph.add_edge(node_2, node_3, ());

    // Custom node sizer to test asymmetric extent handling
    #[derive(Debug, Clone)]
    struct OddEvenSizer;

    impl NodeSizer<MockDomainGraph> for OddEvenSizer {
        fn get_node_size(
            &self,
            node: &petgraph::stable_graph::NodeIndex<u32>,
            _scale: VisualDetail,
        ) -> (u64, u64) {
            match node.index() {
                0 => (3, 1),
                1 => (6, 1),
                2 => (8, 1),
                3 => (9, 1),
                _ => (4, 1),
            }
        }

        fn get_dummy_size(&self) -> (u64, u64) {
            (1, 1)
        }
    }

    let node_sizer = OddEvenSizer;
    let renderer = TestRenderers::debug();

    let snapshot = make_snapshot_custom(
        domain_graph,
        80,
        25,
        usize::MAX,
        usize::MAX,
        node_sizer,
        renderer,
    );

    insta::assert_snapshot!("even_width_nodes", snapshot);
}

#[test]
fn test_edge_viewport_intersection_endpoints_outside() {
    let _ = env_logger::try_init();

    use petgraph::{
        stable_graph::StableDiGraph,
        visit::{EdgeRef, IntoEdgeReferences},
    };

    use crate::{
        geometry::BigRect,
        layout::{LayoutEngine, VisualDetail},
        partition::{PartitionEdge, PartitionNode},
        testing::mocks::FixedNodeSizer,
    };

    // Create a larger partition graph with multiple nodes to ensure
    // there's enough distance between the endpoints
    let mut partition_graph = StableDiGraph::<PartitionNode, PartitionEdge, u32>::new();

    // Create a chain: 0 -> 1 -> 2 -> 3 -> 4
    // This will space out the nodes significantly
    let nodes: Vec<_> = (0..5)
        .map(|i| partition_graph.add_node(PartitionNode::Data(petgraph::graph::NodeIndex::new(i))))
        .collect();

    for i in 0..4 {
        partition_graph.add_edge(
            nodes[i],
            nodes[i + 1],
            Some((
                petgraph::graph::NodeIndex::new(i),
                petgraph::graph::NodeIndex::new(i + 1),
            )),
        );
    }

    // Create a layout engine
    let mut layout_engine = LayoutEngine::new(&partition_graph, 0);

    // Use small node sizes to ensure clear separation
    let node_sizer = FixedNodeSizer {
        width: 2,
        height: 2,
    };

    // Compute the layout
    let layout = layout_engine
        .compute_layout(&node_sizer, VisualDetail::Full)
        .expect("Failed to compute layout");

    // Get the positions of the first and last data nodes
    let node0_pos = layout
        .graph
        .node_indices()
        .find_map(|idx| {
            let node = layout.graph.node_weight(idx)?;
            if matches!(node.role, crate::layout::NodeRole::Data(n) if n.index() == 0) {
                Some(node.pos)
            } else {
                None
            }
        })
        .expect("Node 0 not found in layout");

    let node4_pos = layout
        .graph
        .node_indices()
        .find_map(|idx| {
            let node = layout.graph.node_weight(idx)?;
            if matches!(node.role, crate::layout::NodeRole::Data(n) if n.index() == 4) {
                Some(node.pos)
            } else {
                None
            }
        })
        .expect("Node 4 not found in layout");

    println!("Node 0 position: {:?}", node0_pos);
    println!("Node 4 position: {:?}", node4_pos);

    // Create a small viewport in the middle of the graph
    // Position it between nodes 1 and 3 to ensure it doesn't overlap with node 0 or 4
    let viewport_center_x = (node0_pos.x + node4_pos.x) / 2;
    let viewport_center_y = (node0_pos.y + node4_pos.y) / 2;

    // Make viewport very small - just 2x2 cells
    let viewport = BigRect::from_coords(
        viewport_center_x - 1,
        viewport_center_y - 1,
        viewport_center_x + 1,
        viewport_center_y + 1,
    );

    println!("Viewport: {:?}", viewport);

    // Count data nodes in the viewport (excluding routing nodes)
    let data_nodes_in_viewport = layout
        .find_nodes_in_rect(viewport)
        .iter()
        .filter(|obj| {
            matches!(
                obj.object_type,
                crate::geometry::SpatialObjectType::DataNode(_)
            )
        })
        .count();

    println!("Data nodes in viewport: {}", data_nodes_in_viewport);

    // Query for edges in the viewport
    let edges_in_viewport = layout.find_edges_in_rect(viewport);
    println!("Edges in viewport: {}", edges_in_viewport.len());

    // Debug: print all edges and their bounding boxes
    for (idx, edge_ref) in layout.graph.edge_references().enumerate() {
        let source = edge_ref.source();
        let target = edge_ref.target();
        let source_node = layout.graph.node_weight(source).unwrap();
        let target_node = layout.graph.node_weight(target).unwrap();
        println!(
            "Edge {}: ({}, {}) to ({}, {})",
            idx, source_node.pos.x, source_node.pos.y, target_node.pos.x, target_node.pos.y
        );
    }

    // The viewport should be in the middle of the chain, containing node 2 (the middle node)
    // but not nodes 0 or 4 (the endpoints). There should be edges crossing through it
    // from routing nodes connecting the chain segments.
    assert!(
        !edges_in_viewport.is_empty(),
        "Expected to find edges crossing through viewport, but found none. \
         This indicates that the spatial indexing is not capturing edges whose \
         endpoints are outside the viewport."
    );

    println!(
        "✓ Successfully found {} edge(s) crossing through viewport",
        edges_in_viewport.len()
    );
}

#[test]
fn test_diagonal_edge_viewport_intersection() {
    let _ = env_logger::try_init();

    use petgraph::stable_graph::StableDiGraph;

    use crate::{
        geometry::BigRect,
        layout::{LayoutEngine, VisualDetail},
        partition::{PartitionEdge, PartitionNode},
        testing::mocks::FixedNodeSizer,
    };

    // Create a diamond graph to test diagonal edges
    // Structure:  0
    //            / \
    //           1   2
    //            \ /
    //             3
    let mut partition_graph = StableDiGraph::<PartitionNode, PartitionEdge, u32>::new();

    let nodes: Vec<_> = (0..4)
        .map(|i| partition_graph.add_node(PartitionNode::Data(petgraph::graph::NodeIndex::new(i))))
        .collect();

    // Add diamond edges
    partition_graph.add_edge(
        nodes[0],
        nodes[1],
        Some((
            petgraph::graph::NodeIndex::new(0),
            petgraph::graph::NodeIndex::new(1),
        )),
    );
    partition_graph.add_edge(
        nodes[0],
        nodes[2],
        Some((
            petgraph::graph::NodeIndex::new(0),
            petgraph::graph::NodeIndex::new(2),
        )),
    );
    partition_graph.add_edge(
        nodes[1],
        nodes[3],
        Some((
            petgraph::graph::NodeIndex::new(1),
            petgraph::graph::NodeIndex::new(3),
        )),
    );
    partition_graph.add_edge(
        nodes[2],
        nodes[3],
        Some((
            petgraph::graph::NodeIndex::new(2),
            petgraph::graph::NodeIndex::new(3),
        )),
    );

    let mut layout_engine = LayoutEngine::new(&partition_graph, 0);
    let node_sizer = FixedNodeSizer {
        width: 3,
        height: 3,
    };

    let layout = layout_engine
        .compute_layout(&node_sizer, VisualDetail::Full)
        .expect("Failed to compute layout");

    // Find node positions
    let find_node = |node_index: usize| {
        layout
            .graph
            .node_indices()
            .find_map(|idx| {
                let node = layout.graph.node_weight(idx)?;
                if matches!(node.role, crate::layout::NodeRole::Data(n) if n.index() == node_index)
                {
                    Some(node.pos)
                } else {
                    None
                }
            })
            .unwrap_or_else(|| panic!("Node {} not found in layout", node_index))
    };

    let node0_pos = find_node(0);
    let node1_pos = find_node(1);
    let node2_pos = find_node(2);
    let node3_pos = find_node(3);

    println!("Node 0 (top): {:?}", node0_pos);
    println!("Node 1 (left): {:?}", node1_pos);
    println!("Node 2 (right): {:?}", node2_pos);
    println!("Node 3 (bottom): {:?}", node3_pos);

    // Create a small viewport positioned in the center of the diamond
    // This should not contain any of the 4 corner nodes, but should
    // intersect the diagonal routing edges between them
    let center_x = (node0_pos.x + node3_pos.x) / 2;
    let center_y = (node1_pos.y + node2_pos.y) / 2;

    // Make a small viewport around the center - expand by 2 units in each direction
    let viewport = BigRect::from_coords(center_x - 2, center_y - 2, center_x + 2, center_y + 2);

    println!("Viewport (center region): {:?}", viewport);

    // Query for nodes - we expect none of the data nodes to be in this tiny viewport
    let data_nodes_in_viewport = layout
        .find_nodes_in_rect(viewport)
        .iter()
        .filter(|obj| {
            matches!(
                obj.object_type,
                crate::geometry::SpatialObjectType::DataNode(_)
            )
        })
        .count();

    println!("Data nodes in viewport: {}", data_nodes_in_viewport);

    // Debug: Print all edges and their positions
    use petgraph::visit::{EdgeRef, IntoEdgeReferences};
    println!("\nAll edges in layout:");
    for edge_ref in layout.graph.edge_references() {
        let source = edge_ref.source();
        let target = edge_ref.target();
        let source_node = layout.graph.node_weight(source).unwrap();
        let target_node = layout.graph.node_weight(target).unwrap();
        let bbox_min_x = source_node.pos.x.min(target_node.pos.x);
        let bbox_max_x = source_node.pos.x.max(target_node.pos.x);
        let bbox_min_y = source_node.pos.y.min(target_node.pos.y);
        let bbox_max_y = source_node.pos.y.max(target_node.pos.y);
        println!(
            "  Edge ({},{}) -> ({},{}) | AABB: x[{},{}] y[{},{}]",
            source_node.pos.x,
            source_node.pos.y,
            target_node.pos.x,
            target_node.pos.y,
            bbox_min_x,
            bbox_max_x,
            bbox_min_y,
            bbox_max_y
        );
    }

    // Query for edges - we expect to find routing edges that cross through this center point
    let edges_in_viewport = layout.find_edges_in_rect(viewport);
    println!("\nEdges found in viewport: {}", edges_in_viewport.len());

    // The spatial indexing should capture edges that cross through the viewport
    // even if both endpoints are outside
    assert!(
        !edges_in_viewport.is_empty(),
        "Expected to find edges crossing through the center of the diamond, but found none. \
         This would indicate that diagonal edges with endpoints outside the viewport \
         are not being captured by spatial indexing."
    );

    println!(
        "✓ Successfully found {} edge(s) crossing through center region",
        edges_in_viewport.len()
    );
}

/// Test for determinism by running the same layout multiple times and comparing snapshots.
/// This test uses the complex_dag graph with node-based partitioning (which forces bridge creation)
/// to ensure that HashMap/HashSet iterations produce consistent results.
#[test]
fn test_layout_determinism_with_partitioning() {
    let _ = env_logger::try_init();
    use crate::testing::mocks::TestGraphs;

    // Generate the same layout 10 times
    let num_iterations = 10;
    let mut snapshots = Vec::new();

    for i in 0..num_iterations {
        // Clone the graph for each iteration since make_snapshot takes ownership
        let graph = TestGraphs::domain_complex_dag();
        let snapshot = make_snapshot(graph, 80, 25, usize::MAX, 3);
        snapshots.push(snapshot);
        log::trace!("Generated snapshot {} for determinism test", i);
    }

    // All snapshots should be identical
    let first = &snapshots[0];
    for (i, snapshot) in snapshots.iter().enumerate().skip(1) {
        assert_eq!(
            first, snapshot,
            "Layout iteration {} produced different output than iteration 0. \
             This indicates non-determinism in the layout algorithm. \
             The difference suggests HashMap or HashSet iteration order is affecting the result.",
            i
        );
    }

    // Also verify against the stored snapshot to ensure the output is correct
    insta::assert_snapshot!("determinism_check_complex_dag_node_partitioning", first);
}

/// Cyclic graphs with two or more backward edges must lay out deterministically. Cycle
/// auto-detection returns the backward edges as a HashSet, whose iteration order is
/// randomized per process; unless the controller sorts them before injecting pins, the
/// pin order (and thus the layout) varies from run to run. Building the same cyclic graph
/// many times and comparing catches a regression of that sort within a single process.
#[test]
fn test_cyclic_layout_determinism_multiple_backward_edges() {
    let _ = env_logger::try_init();

    let mut snapshots = Vec::new();
    for _ in 0..10 {
        let mut graph = cycle_graph(8);
        add_edge_by_index(&mut graph, 6, 3);
        add_edge_by_index(&mut graph, 4, 1);
        snapshots.push(make_snapshot(graph, 120, 25, 1000, 1000));
    }

    let first = &snapshots[0];
    for (i, snapshot) in snapshots.iter().enumerate().skip(1) {
        assert_eq!(
            first, snapshot,
            "Cyclic layout iteration {i} differs from iteration 0 - backward-edge pin \
             injection order is not deterministic (likely HashSet iteration order)."
        );
    }
}

/// Proves crossing-reduction tie handling is deterministic by constructing the same symmetric
/// graph twice, but inserting edges in a different order (which affects DFS-based init order).
/// With the tiebreaker in `gen-sugiyama`, these should render identically.
#[test]
fn test_layout_determinism_across_edge_insertion_order_symmetric_fan() {
    let _ = env_logger::try_init();

    // Graph:
    //   node_0 -> {node_1, node_2, node_3} -> node_4
    // The three middle nodes are perfectly symmetric, so their barycenters tie.
    // Without a deterministic tiebreaker, the middle layer can preserve the DFS visit order.

    let mut graph_1 = MockDomainGraph::new();
    let node_0 = graph_1.add_node(());
    let node_1 = graph_1.add_node(());
    let node_2 = graph_1.add_node(());
    let node_3 = graph_1.add_node(());
    let node_4 = graph_1.add_node(());
    graph_1.add_edge(node_0, node_1, ());
    graph_1.add_edge(node_0, node_2, ());
    graph_1.add_edge(node_0, node_3, ());
    graph_1.add_edge(node_1, node_4, ());
    graph_1.add_edge(node_2, node_4, ());
    graph_1.add_edge(node_3, node_4, ());

    let mut graph_2 = MockDomainGraph::new();
    let node_0 = graph_2.add_node(());
    let node_1 = graph_2.add_node(());
    let node_2 = graph_2.add_node(());
    let node_3 = graph_2.add_node(());
    let node_4 = graph_2.add_node(());
    // Same edges, different insertion order.
    graph_2.add_edge(node_0, node_3, ());
    graph_2.add_edge(node_0, node_1, ());
    graph_2.add_edge(node_0, node_2, ());
    graph_2.add_edge(node_3, node_4, ());
    graph_2.add_edge(node_1, node_4, ());
    graph_2.add_edge(node_2, node_4, ());

    let snapshot1 = make_snapshot(graph_1, 80, 25, usize::MAX, usize::MAX);
    let snapshot2 = make_snapshot(graph_2, 80, 25, usize::MAX, usize::MAX);

    assert_eq!(
        snapshot1, snapshot2,
        "Symmetric fan layout should be identical regardless of edge insertion order"
    );
}

#[test]
fn test_double_chain() {
    let _ = env_logger::try_init();

    // Create a graph with 18 nodes arranged in two chains with a common start and stop node.
    // Chain 1: node_1 -> node_2 -> node_3 -> node_4 -> node_5 -> node_6 -> node_7 -> node_8 -> node_9 -> node_10 (10 nodes)
    // Chain 2: node_1 -> node_12 -> node_13 -> node_14 -> node_15 -> node_16 -> node_17 -> node_18 -> node_19 -> node_10 (10 nodes)
    // Shared nodes: node_1 (start), node_10 (stop)
    // Total unique nodes: 18

    let mut domain_graph = MockDomainGraph::new();

    // Add all nodes (indices 0-17 correspond to node_1-node_10 and node_12-node_19)
    // Using index mapping:
    // 0 -> node_1 (shared start)
    // 1 -> node_2
    // 2 -> node_3
    // 3 -> node_4
    // 4 -> node_5
    // 5 -> node_6
    // 6 -> node_7
    // 7 -> node_8
    // 8 -> node_9
    // 9 -> node_10 (shared stop)
    // 10 -> node_12
    // 11 -> node_13
    // 12 -> node_14
    // 13 -> node_15
    // 14 -> node_16
    // 15 -> node_17
    // 16 -> node_18
    // 17 -> node_19
    let nodes: Vec<_> = (0..18).map(|_| domain_graph.add_node(())).collect();

    // Chain 1: node_1(0) -> node_2(1) -> node_3(2) -> node_4(3) -> node_5(4) -> node_6(5) -> node_7(6) -> node_8(7) -> node_9(8) -> node_10(9)
    for i in 0..9 {
        domain_graph.add_edge(nodes[i], nodes[i + 1], ());
    }

    // Chain 2: node_1(0) -> node_12(10) -> node_13(11) -> node_14(12) -> node_15(13) -> node_16(14) -> node_17(15) -> node_18(16) -> node_19(17) -> node_10(9)
    domain_graph.add_edge(nodes[0], nodes[10], ());
    for i in 10..17 {
        domain_graph.add_edge(nodes[i], nodes[i + 1], ());
    }
    domain_graph.add_edge(nodes[17], nodes[9], ());

    let snapshot = make_snapshot(domain_graph, 120, 40, 5, 20);

    insta::assert_snapshot!("double_chain", snapshot);
}

#[test]
fn test_asymmetric_diamond() {
    let _ = env_logger::try_init();

    // Create an asymmetric diamond graph:
    //   A-B-C-D-E
    //    \     /
    //     --F--
    //
    // Edges:
    // A -> B, B -> C, C -> D, D -> E (main chain)
    // A -> F, F -> E (bypass through single intermediate node)

    let mut domain_graph = MockDomainGraph::new();
    let node_a = domain_graph.add_node(());
    let node_b = domain_graph.add_node(());
    let node_c = domain_graph.add_node(());
    let node_d = domain_graph.add_node(());
    let node_e = domain_graph.add_node(());
    let node_f = domain_graph.add_node(());

    // Main chain: A -> B -> C -> D -> E
    domain_graph.add_edge(node_a, node_b, ());
    domain_graph.add_edge(node_b, node_c, ());
    domain_graph.add_edge(node_c, node_d, ());
    domain_graph.add_edge(node_d, node_e, ());

    // Bypass: A -> F -> E
    domain_graph.add_edge(node_a, node_f, ());
    domain_graph.add_edge(node_f, node_e, ());

    let snapshot = make_snapshot(domain_graph, 80, 25, usize::MAX, usize::MAX);

    insta::assert_snapshot!("asymmetric_diamond", snapshot);
}

#[test]
fn test_asymmetric_diamond_2_1() {
    let _ = env_logger::try_init();

    let mut domain_graph = MockDomainGraph::new();
    let node_a = domain_graph.add_node(());
    let node_b = domain_graph.add_node(());
    let node_c = domain_graph.add_node(());
    let node_d = domain_graph.add_node(());
    let node_e = domain_graph.add_node(());

    domain_graph.add_edge(node_a, node_b, ());
    domain_graph.add_edge(node_b, node_c, ());
    domain_graph.add_edge(node_c, node_d, ());

    domain_graph.add_edge(node_a, node_e, ());
    domain_graph.add_edge(node_e, node_d, ());

    let snapshot = make_snapshot(domain_graph, 80, 25, usize::MAX, usize::MAX);

    insta::assert_snapshot!("asymmetric_diamond_2_1", snapshot);
}

#[test]
fn test_asymmetric_diamond_4_1() {
    let _ = env_logger::try_init();

    // Longer leg: 4 intermediate nodes (6 total), shorter leg: 1 intermediate (2 total)
    // A -> B -> C -> D -> E -> F
    // A -> G -> E

    let mut domain_graph = MockDomainGraph::new();
    let node_a = domain_graph.add_node(());
    let node_b = domain_graph.add_node(());
    let node_c = domain_graph.add_node(());
    let node_d = domain_graph.add_node(());
    let node_e = domain_graph.add_node(());
    let node_f = domain_graph.add_node(());
    let node_g = domain_graph.add_node(());

    domain_graph.add_edge(node_a, node_b, ());
    domain_graph.add_edge(node_b, node_c, ());
    domain_graph.add_edge(node_c, node_d, ());
    domain_graph.add_edge(node_d, node_e, ());
    domain_graph.add_edge(node_e, node_f, ());

    domain_graph.add_edge(node_a, node_g, ());
    domain_graph.add_edge(node_g, node_f, ());

    let snapshot = make_snapshot(domain_graph, 80, 25, usize::MAX, usize::MAX);

    insta::assert_snapshot!("asymmetric_diamond_4_1", snapshot);
}

#[test]
fn test_asymmetric_diamond_3_2() {
    let _ = env_logger::try_init();

    // A -> B -> C -> D -> E
    // A -> F1 -> F2 -> E

    let mut domain_graph = MockDomainGraph::new();
    let node_a = domain_graph.add_node(());
    let node_b = domain_graph.add_node(());
    let node_c = domain_graph.add_node(());
    let node_d = domain_graph.add_node(());
    let node_e = domain_graph.add_node(());
    let node_f1 = domain_graph.add_node(());
    let node_f2 = domain_graph.add_node(());

    domain_graph.add_edge(node_a, node_b, ());
    domain_graph.add_edge(node_b, node_c, ());
    domain_graph.add_edge(node_c, node_d, ());
    domain_graph.add_edge(node_d, node_e, ());

    domain_graph.add_edge(node_a, node_f1, ());
    domain_graph.add_edge(node_f1, node_f2, ());
    domain_graph.add_edge(node_f2, node_e, ());

    let snapshot = make_snapshot(domain_graph, 80, 25, usize::MAX, usize::MAX);

    insta::assert_snapshot!("asymmetric_diamond_3_2", snapshot);
}

#[test]
fn test_asymmetric_diamond_4_2() {
    let _ = env_logger::try_init();

    // A -> B -> C -> D -> E -> F
    // A -> X -> Y -> F

    let mut domain_graph = MockDomainGraph::new();
    let node_a = domain_graph.add_node(());
    let node_b = domain_graph.add_node(());
    let node_c = domain_graph.add_node(());
    let node_d = domain_graph.add_node(());
    let node_e = domain_graph.add_node(());
    let node_f = domain_graph.add_node(());
    let node_x = domain_graph.add_node(());
    let node_y = domain_graph.add_node(());

    domain_graph.add_edge(node_a, node_b, ());
    domain_graph.add_edge(node_b, node_c, ());
    domain_graph.add_edge(node_c, node_d, ());
    domain_graph.add_edge(node_d, node_e, ());
    domain_graph.add_edge(node_e, node_f, ());

    domain_graph.add_edge(node_a, node_x, ());
    domain_graph.add_edge(node_x, node_y, ());
    domain_graph.add_edge(node_y, node_f, ());

    let snapshot = make_snapshot(domain_graph, 80, 25, usize::MAX, usize::MAX);

    insta::assert_snapshot!("asymmetric_diamond_4_2", snapshot);
}

/// Test a complex multi-level DAG with dense intermediate connectivity.
///
/// Uses alphabetic single-character labels (node N → Nth letter A-Z) with 1×1 cells,
/// except N17 (4×1) and N3 (6×1) which are wider to stress horizontal spacing.
#[test]
fn test_complex_multipath_dag() {
    let _ = env_logger::try_init();

    use crate::{
        geometry::WorldPos,
        plotter::{NodeRenderer, NodeSizer},
    };

    #[derive(Debug, Clone)]
    struct AlphabeticSizer;

    impl NodeSizer<MockDomainGraph> for AlphabeticSizer {
        fn get_node_size(
            &self,
            node: &petgraph::stable_graph::NodeIndex<u32>,
            _scale: VisualDetail,
        ) -> (u64, u64) {
            match node.index() {
                17 => (4, 1),
                3 => (6, 1),
                _ => (1, 1),
            }
        }

        fn get_dummy_size(&self) -> (u64, u64) {
            (1, 1)
        }
    }

    #[derive(Debug, Clone)]
    struct AlphabeticRenderer;

    impl NodeRenderer<MockDomainGraph> for AlphabeticRenderer {
        fn render_node(
            &mut self,
            buffer: &mut crate::graph_controller::WorldBuffer,
            area: crate::geometry::WorldRect,
            node_id: &petgraph::stable_graph::NodeIndex<u32>,
            _scale: VisualDetail,
        ) {
            let letter = char::from_u32(b'A' as u32 + node_id.index() as u32).unwrap_or('?');
            let Some(visible) = buffer.calculate_visible_area(area) else {
                return;
            };
            for y in visible.min.y..=visible.max.y {
                let width = (visible.max.x - visible.min.x + 1) as usize;
                let content = letter.to_string().repeat(width);
                buffer.set_string(WorldPos::new(visible.min.x, y), &content);
            }
        }
    }

    let mut domain_graph = MockDomainGraph::new();
    let nodes: Vec<_> = (0..22).map(|_| domain_graph.add_node(())).collect();

    let edges: &[(usize, usize)] = &[
        (0, 10),
        (0, 11),
        (0, 8),
        (0, 19),
        (1, 7),
        (1, 12),
        (1, 13),
        (1, 18),
        (2, 11),
        (2, 10),
        (2, 19),
        (2, 8),
        (3, 21),
        (4, 10),
        (4, 8),
        (4, 11),
        (4, 19),
        (5, 11),
        (5, 19),
        (5, 10),
        (5, 8),
        (6, 21),
        (7, 2),
        (7, 0),
        (7, 5),
        (7, 4),
        (8, 6),
        (9, 7),
        (9, 18),
        (9, 13),
        (9, 12),
        (10, 6),
        (11, 6),
        (12, 2),
        (12, 4),
        (12, 0),
        (12, 5),
        (13, 0),
        (13, 4),
        (13, 2),
        (13, 5),
        (14, 18),
        (14, 13),
        (14, 12),
        (14, 7),
        (15, 1),
        (15, 17),
        (15, 14),
        (15, 9),
        (15, 16),
        (16, 7),
        (16, 12),
        (16, 18),
        (16, 13),
        (17, 6),
        (18, 2),
        (18, 4),
        (18, 0),
        (18, 5),
        (19, 6),
        (20, 3),
        (20, 15),
    ];

    for &(from, to) in edges {
        domain_graph.add_edge(nodes[from], nodes[to], ());
    }

    let snapshot = make_snapshot_custom(
        domain_graph,
        160,
        50,
        usize::MAX,
        usize::MAX,
        AlphabeticSizer,
        AlphabeticRenderer,
    );
    insta::assert_snapshot!("complex_multipath_dag", snapshot);
}

/// Build a dense all-to-all grid with a bypass node.
///
/// Structure (`layer_count` + 2 layers):
/// Layer 0: single source node (index 0)
/// Layers 1..=layer_count: `nodes_per_layer` nodes each, connected all-to-all
///   between consecutive layers
/// Last layer: single sink node
/// Bypass: source -> bypass (highest index) -> sink, skipping the grid entirely
#[cfg(test)]
fn make_grid_all_to_all_with_bypass(layer_count: usize, nodes_per_layer: usize) -> MockDomainGraph {
    let mut domain_graph = MockDomainGraph::new();

    let source = domain_graph.add_node(());

    let grid_layers: Vec<Vec<_>> = (0..layer_count)
        .map(|_| {
            (0..nodes_per_layer)
                .map(|_| domain_graph.add_node(()))
                .collect()
        })
        .collect();

    let sink = domain_graph.add_node(());
    let bypass = domain_graph.add_node(());

    // Source fans out to the first grid layer
    for &node in &grid_layers[0] {
        domain_graph.add_edge(source, node, ());
    }

    // All-to-all between consecutive grid layers
    for pair in grid_layers.windows(2) {
        for &from in &pair[0] {
            for &to in &pair[1] {
                domain_graph.add_edge(from, to, ());
            }
        }
    }

    // Last grid layer fans in to the sink
    for &node in &grid_layers[layer_count - 1] {
        domain_graph.add_edge(node, sink, ());
    }

    // Bypass: source -> bypass -> sink
    domain_graph.add_edge(source, bypass, ());
    domain_graph.add_edge(bypass, sink, ());

    domain_graph
}

#[test]
fn test_grid_all_to_all_with_bypass() {
    let _ = env_logger::try_init();

    let domain_graph = make_grid_all_to_all_with_bypass(4, 4);
    let snapshot = make_snapshot(domain_graph, 120, 40, usize::MAX, usize::MAX);

    insta::assert_snapshot!("grid_all_to_all_with_bypass", snapshot);
}

/// Zoom (disperse/contract adjusts the minimum inter-node distance) must
/// survive dead-space compression: dispersed layouts keep their wider gaps
/// uniformly, and contracting returns to the original layout.
#[test]
fn test_grid_disperse_zoom_preserves_spacing() {
    let _ = env_logger::try_init();

    use crate::{
        geometry::WorldPos,
        graph_controller::{GraphConfig, GraphController},
        testing::mocks::FixedNodeSizer,
    };

    let domain_graph = make_grid_all_to_all_with_bypass(4, 4);
    let node_sizer = FixedNodeSizer {
        width: 5,
        height: 3,
    };
    let mut config = GraphConfig::default();
    config.partition.layer_count = usize::MAX;
    config.partition.node_count = usize::MAX;

    let mut controller = GraphController::new_with_config(domain_graph.clone(), node_sizer, config);
    controller.set_detail_level(VisualDetail::Full);
    controller.viewport_state.viewport_bounds =
        ratatui::layout::Rect::new(0, 0, u16::MAX / 2, u16::MAX / 2);
    controller.viewport_state.camera_current = WorldPos::new(0, 0);
    controller.viewport_state.camera_target = WorldPos::new(0, 0);

    // First grid layer (domain indices 1-4) plus the bypass node (index 18)
    // span all distinct rows of the layout.
    let row_node_indices = [1u32, 2, 3, 4, 18];

    let collect_row_ys = |controller: &mut GraphController<MockDomainGraph, FixedNodeSizer>| {
        controller
            .ensure_camera_coverage()
            .expect("Failed to ensure camera coverage");
        controller
            .rebuild_viewport_graph()
            .expect("Failed to rebuild viewport graph");
        let viewport_graph = controller.get_viewport_graph();
        let mut ys: Vec<i64> = row_node_indices
            .iter()
            .map(|&i| {
                viewport_graph
                    .node_positions
                    .get(&petgraph::graph::NodeIndex::new(i as usize))
                    .expect("Node missing from viewport graph")
                    .y
            })
            .collect();
        ys.sort();
        ys
    };

    let uniform_gaps = |ys: &[i64]| -> Vec<i64> { ys.windows(2).map(|w| w[1] - w[0]).collect() };

    // Default spacing: rows must be uniformly spaced (no dead space)
    let initial_ys = collect_row_ys(&mut controller);
    let initial_gaps = uniform_gaps(&initial_ys);
    assert!(
        initial_gaps.iter().all(|&g| g == initial_gaps[0]),
        "Rows must be uniformly spaced at default zoom, got gaps {:?}",
        initial_gaps
    );

    // Disperse twice: vertex spacing 1 -> 3 -> 5, so row gaps grow to height + spacing
    controller.disperse();
    controller.disperse();

    let dispersed_ys = collect_row_ys(&mut controller);
    let dispersed_gaps = uniform_gaps(&dispersed_ys);
    assert!(
        dispersed_gaps.iter().all(|&g| g == dispersed_gaps[0]),
        "Rows must stay uniformly spaced when dispersed, got gaps {:?}",
        dispersed_gaps
    );
    assert!(
        dispersed_gaps[0] > initial_gaps[0],
        "Dispersing must increase row spacing: {} -> {}",
        initial_gaps[0],
        dispersed_gaps[0]
    );

    // Contract back to the default spacing: layout must return to the original
    controller.contract();
    controller.contract();

    let contracted_ys = collect_row_ys(&mut controller);
    assert_eq!(
        contracted_ys, initial_ys,
        "Contracting back to default spacing must restore the original row positions"
    );
}

/// 5 grid layers of 4 nodes: probes whether the dead space around the bypass
/// node depends on an odd number of layers.
#[test]
fn test_grid_5_layers_by_4_with_bypass() {
    let _ = env_logger::try_init();

    let domain_graph = make_grid_all_to_all_with_bypass(5, 4);
    let snapshot = make_snapshot(domain_graph, 120, 40, usize::MAX, usize::MAX);

    insta::assert_snapshot!("grid_5_layers_by_4_with_bypass", snapshot);
}

/// 4 grid layers of 5 nodes: probes whether the dead space around the bypass
/// node depends on an odd number of nodes per layer.
#[test]
fn test_grid_4_layers_by_5_with_bypass() {
    let _ = env_logger::try_init();

    let domain_graph = make_grid_all_to_all_with_bypass(4, 5);
    let snapshot = make_snapshot(domain_graph, 120, 40, usize::MAX, usize::MAX);

    insta::assert_snapshot!("grid_4_layers_by_5_with_bypass", snapshot);
}

/// Same grid-with-bypass structure, but with variable node widths where the
/// bypass node is wider than every other node.
#[test]
fn test_grid_all_to_all_with_bypass_variable_widths() {
    let _ = env_logger::try_init();

    use crate::plotter::NodeSizer;

    #[derive(Debug, Clone)]
    struct WideBypassSizer;

    impl NodeSizer<MockDomainGraph> for WideBypassSizer {
        fn get_node_size(
            &self,
            node: &petgraph::stable_graph::NodeIndex<u32>,
            _scale: VisualDetail,
        ) -> (u64, u64) {
            match node.index() {
                // Bypass node: wider than everything else
                18 => (13, 3),
                // Source and sink: medium width
                0 | 17 => (5, 3),
                // Grid nodes: alternating widths
                i if i % 2 == 0 => (7, 3),
                _ => (4, 3),
            }
        }

        fn get_dummy_size(&self) -> (u64, u64) {
            (1, 1)
        }
    }

    let domain_graph = make_grid_all_to_all_with_bypass(4, 4);
    let renderer = TestRenderers::debug();

    let snapshot = make_snapshot_custom(
        domain_graph,
        120,
        40,
        usize::MAX,
        usize::MAX,
        WideBypassSizer,
        renderer,
    );

    insta::assert_snapshot!("grid_all_to_all_with_bypass_variable_widths", snapshot);
}

/// Test rendering and positioning of very large nodes during zoom operations.
///
/// This test verifies that nodes with extreme widths (1000+ characters) are rendered
/// correctly without coordinate overflow issues. The test zooms through different
/// detail levels and ensures that:
/// 1. Large nodes don't cause coordinate wraparound or positioning errors
/// 2. Spatial relationships between nodes are maintained (left node < right node)
/// 3. Cursor positioning remains stable during zoom operations
///
/// This test specifically addresses coordinate overflow bugs where very wide nodes
/// could exceed u16::MAX coordinates and cause rendering artifacts.
#[test]
fn test_large_node_rendering_with_zoom() {
    let _ = env_logger::try_init();

    use crate::{
        geometry::WorldRect,
        graph_controller::{GraphConfig, GraphController, WorldBuffer},
        layout::VisualDetail,
        plotter::{NodeRenderer, NodeSizer},
        testing::{create_test_terminal, mocks::MockDomainGraph},
    };

    // 1. Create a 3-node chain graph: 0 -> 1 -> 2
    let mut domain_graph = MockDomainGraph::new();
    let node_0 = domain_graph.add_node(());
    let node_1 = domain_graph.add_node(());
    let node_2 = domain_graph.add_node(());
    domain_graph.add_edge(node_0, node_1, ());
    domain_graph.add_edge(node_1, node_2, ());

    // 2. Custom NodeSizer with adjustable node length
    #[derive(Debug, Clone)]
    struct VariableDetailSizer;

    impl NodeSizer<MockDomainGraph> for VariableDetailSizer {
        fn get_node_size(
            &self,
            node: &petgraph::stable_graph::NodeIndex<u32>,
            scale: VisualDetail,
        ) -> (u64, u64) {
            match scale {
                VisualDetail::Minimal => (1, 1),
                VisualDetail::Truncated => (10, 1),
                VisualDetail::Full => match node.index() {
                    0 => (5, 1),
                    1 => (1000, 1),
                    2 => (5, 1),
                    _ => (1, 1),
                },
            }
        }

        fn get_dummy_size(&self) -> (u64, u64) {
            (1, 1)
        }
    }

    #[derive(Debug, Clone)]
    struct UltrawideRenderer;

    impl NodeRenderer<MockDomainGraph> for UltrawideRenderer {
        fn render_node(
            &mut self,
            buffer: &mut WorldBuffer,
            area: WorldRect,
            node_id: &petgraph::stable_graph::NodeIndex<u32>,
            _scale: VisualDetail,
        ) {
            // Viewport-aware rendering: only render the visible portion of large nodes
            // This is critical for performance with very large nodes (1000+ width)

            let Some(visible_area) = buffer.calculate_visible_area(area) else {
                // Node is completely outside viewport - don't render anything
                return;
            };

            let symbol = format!("{}", node_id.index()).chars().next().unwrap();

            // Only render the visible portion
            for y in visible_area.min.y..=visible_area.max.y {
                // Calculate the visible width for this row
                let visible_width = (visible_area.max.x - visible_area.min.x + 1) as usize;
                let content = symbol.to_string().repeat(visible_width);

                let start_pos = crate::geometry::WorldPos::new(visible_area.min.x, y);
                buffer.set_string(start_pos, &content);
            }
        }
    }

    let viewport_width = 80;
    let viewport_height = 20;
    let mut terminal = create_test_terminal(viewport_width, viewport_height);
    let mut config = GraphConfig::default();
    config.partition.layer_count = usize::MAX;
    config.partition.node_count = usize::MAX;

    let mut controller =
        GraphController::new_with_config(domain_graph.clone(), VariableDetailSizer, config);

    // 4. Starts in minimal level-of-detail
    controller.set_detail_level(VisualDetail::Minimal);

    // 5. Cursor setup (not visible in the snapshots though)
    controller.show_cursor();
    controller.initialize_cursor();
    controller.cursor.set_node(node_0, (0.0, 0.0));

    // Set cursor to the center of the viewport
    let vp_center_x = viewport_width / 2;
    let vp_center_y = viewport_height / 2;
    let initial_cursor_viewport_pos = crate::geometry::ViewportPos::new(vp_center_x, vp_center_y);
    controller
        .cursor
        .set_viewport_pos(initial_cursor_viewport_pos);

    let renderer = UltrawideRenderer;

    // 6. Snapshot 1: Minimal detail level
    let _ = terminal.draw(|f| {
        let area = f.area();
        controller.viewport_state.viewport_bounds = area;

        let widget = crate::graph_widget::GraphWidget::with_renderer(renderer.clone())
            .detail_level(VisualDetail::Minimal)
            .cursor();
        f.render_stateful_widget(widget, area, &mut controller);
    });
    let minimal_snapshot = format!("{}", terminal.backend());
    insta::assert_snapshot!("variable_detail_chain_minimal", minimal_snapshot);

    // 7. Simulate hitting '+' to zoom in (goes to Truncated)
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    let plus_key = KeyEvent::new(KeyCode::Char('+'), KeyModifiers::NONE);
    controller.handle_key_event(plus_key).unwrap();
    controller.trigger_rebuild();

    // Snapshot 2: Truncated detail level
    let _ = terminal.draw(|f| {
        let area = f.area();
        controller.viewport_state.viewport_bounds = area;

        let widget = crate::graph_widget::GraphWidget::with_renderer(renderer.clone())
            .detail_level(controller.get_detail_level())
            .cursor();
        f.render_stateful_widget(widget, area, &mut controller);
    });
    let truncated_snapshot = format!("{}", terminal.backend());
    insta::assert_snapshot!("variable_detail_chain_truncated", truncated_snapshot);

    // 8. Simulate hitting '+' again to zoom in (goes to Full)
    controller.handle_key_event(plus_key).unwrap();
    controller.trigger_rebuild();

    // Snapshot 3: Full detail level
    let _ = terminal.draw(|f| {
        let area = f.area();
        controller.viewport_state.viewport_bounds = area;

        let widget = crate::graph_widget::GraphWidget::with_renderer(renderer.clone())
            .detail_level(controller.get_detail_level())
            .cursor();
        f.render_stateful_widget(widget, area, &mut controller);
    });
    let full_snapshot = format!("{}", terminal.backend());
    insta::assert_snapshot!("variable_detail_chain_full", full_snapshot);

    // 9. Confirms node 1's minimum x is to the right of node 0's maximum x

    let viewport_graph = controller.get_viewport_graph();

    let pos0 = viewport_graph.node_positions.get(&node_0).unwrap();
    let pos1 = viewport_graph.node_positions.get(&node_1).unwrap();

    let node0 = viewport_graph.get_node(pos0).unwrap();
    let node1 = viewport_graph.get_node(pos1).unwrap();

    let rect0 = WorldRect::from_center_and_size(*pos0, node0.size);
    let rect1 = WorldRect::from_center_and_size(*pos1, node1.size);

    assert!(
        rect1.min.x > rect0.max.x,
        "Node 1's minimum x ({}) should be to the right of node 0's maximum x ({})",
        rect1.min.x,
        rect0.max.x
    );

    // Verify that the cursor has maintained its viewport position throughout the zoom operations
    let final_cursor_viewport_pos = controller.cursor.viewport_pos;
    assert_eq!(
        initial_cursor_viewport_pos, final_cursor_viewport_pos,
        "Cursor viewport position should remain stable during zoom operations. Initial: {:?}, Final: {:?}",
        initial_cursor_viewport_pos, final_cursor_viewport_pos
    );
}

/// Test diamond graph with variable width nodes on parallel branches.
/// This tests the horizontal chain redistribution with nodes of different sizes.
#[test]
fn test_diamond_variable_width_parallel_nodes() {
    let _ = env_logger::try_init();

    use crate::plotter::NodeSizer;

    // Create diamond: A -> {B, C} -> D
    let mut domain_graph = MockDomainGraph::new();
    let node_a = domain_graph.add_node(());
    let node_b = domain_graph.add_node(());
    let node_c = domain_graph.add_node(());
    let node_d = domain_graph.add_node(());

    domain_graph.add_edge(node_a, node_b, ());
    domain_graph.add_edge(node_a, node_c, ());
    domain_graph.add_edge(node_b, node_d, ());
    domain_graph.add_edge(node_c, node_d, ());

    // Custom NodeSizer: B=3 wide, C=2 wide, others=5 wide
    #[derive(Debug, Clone)]
    struct VariableWidthSizer;

    impl NodeSizer<MockDomainGraph> for VariableWidthSizer {
        fn get_node_size(
            &self,
            node: &petgraph::stable_graph::NodeIndex<u32>,
            _scale: VisualDetail,
        ) -> (u64, u64) {
            match node.index() {
                1 => (15, 3), // B - 3 units wide (15 chars = 3 * 5-char units)
                2 => (10, 3), // C - 2 units wide (10 chars = 2 * 5-char units)
                _ => (5, 3),  // A and D - 1 unit wide (5 chars)
            }
        }

        fn get_dummy_size(&self) -> (u64, u64) {
            (1, 1)
        }
    }

    let renderer = TestRenderers::debug();
    let node_sizer = VariableWidthSizer;

    let snapshot = make_snapshot_custom(
        domain_graph,
        80,
        25,
        usize::MAX,
        usize::MAX,
        node_sizer,
        renderer,
    );

    insta::assert_snapshot!("diamond_variable_width_parallel", snapshot);
}

#[test]
fn viewport_visual_regression_circular_genome_loop() {
    let _ = env_logger::try_init();
    // a -> b -> c -> d, plus a backward edge d -> a closing the loop, mirroring a
    // circular genome's PATH_END -> PATH_START edge.
    let mut domain_graph = MockDomainGraph::new();
    let a = domain_graph.add_node(());
    let b = domain_graph.add_node(());
    let c = domain_graph.add_node(());
    let d = domain_graph.add_node(());
    domain_graph.add_edge(a, b, ());
    domain_graph.add_edge(b, c, ());
    domain_graph.add_edge(c, d, ());

    let snapshot =
        make_snapshot_with_backward_edges(domain_graph, 80, 20, usize::MAX, usize::MAX, &[(d, a)]);

    insta::assert_snapshot!("circular_genome_loop", snapshot);
}

#[test]
fn viewport_visual_regression_circular_genome_loop_multi_partition() {
    let _ = env_logger::try_init();
    // A longer chain split across several partitions, with a backward edge from the
    // last node back to the first, exercising the multi-partition pin-relay path.
    let mut domain_graph = MockDomainGraph::new();
    let nodes: Vec<_> = (0..6).map(|_| domain_graph.add_node(())).collect();
    for i in 0..5 {
        domain_graph.add_edge(nodes[i], nodes[i + 1], ());
    }

    let snapshot = make_snapshot_with_backward_edges(
        domain_graph,
        160,
        20,
        1,
        usize::MAX,
        &[(nodes[5], nodes[0])],
    );

    insta::assert_snapshot!("circular_genome_loop_multi_partition", snapshot);
}

#[test]
fn backward_edge_pin_layer_insertion_single_partition() {
    let _ = env_logger::try_init();
    // Linear chain 0→1→2→3→4 with a backward edge (4, 0) closing the loop,
    // forced into one partition (layer_count=1000) so the pin/dummy/data interplay
    // is exercised in the simplest possible subgraph.
    let mut domain_graph = MockDomainGraph::new();
    let nodes: Vec<_> = (0..5).map(|_| domain_graph.add_node(())).collect();
    for i in 0..4 {
        domain_graph.add_edge(nodes[i], nodes[i + 1], ());
    }

    let snapshot = make_snapshot_with_backward_edges(
        domain_graph,
        100,
        20,
        1000,
        usize::MAX,
        &[(nodes[4], nodes[0])],
    );

    insta::assert_snapshot!(
        "backward_edge_pin_layer_insertion_single_partition",
        snapshot
    );
}

#[test]
fn backward_edge_pin_layer_insertion_multi_partition() {
    let _ = env_logger::try_init();
    // Same graph split across multiple partitions (layer_count=2) so the dummy-relay
    // path through intermediate stitch chains is also exercised.
    let mut domain_graph = MockDomainGraph::new();
    let nodes: Vec<_> = (0..5).map(|_| domain_graph.add_node(())).collect();
    for i in 0..4 {
        domain_graph.add_edge(nodes[i], nodes[i + 1], ());
    }

    let snapshot = make_snapshot_with_backward_edges(
        domain_graph,
        160,
        20,
        2,
        usize::MAX,
        &[(nodes[4], nodes[0])],
    );

    insta::assert_snapshot!(
        "backward_edge_pin_layer_insertion_multi_partition",
        snapshot
    );
}

#[test]
fn backward_edge_minimal_triangle() {
    let _ = env_logger::try_init();
    // Smallest possible cycle: 3 nodes, backward edge 2 → 0.
    let mut domain_graph = MockDomainGraph::new();
    let nodes: Vec<_> = (0..3).map(|_| domain_graph.add_node(())).collect();
    domain_graph.add_edge(nodes[0], nodes[1], ());
    domain_graph.add_edge(nodes[1], nodes[2], ());

    let snapshot = make_snapshot_with_backward_edges(
        domain_graph,
        80,
        20,
        usize::MAX,
        usize::MAX,
        &[(nodes[2], nodes[0])],
    );

    insta::assert_snapshot!("backward_edge_minimal_triangle", snapshot);
}

#[test]
fn backward_edge_six_node_cycle() {
    let _ = env_logger::try_init();
    // 6-node cycle mirroring the cycle_no_path.gfa fixture from the view-cycles branch.
    // Forward: 0 → 1 → 2 → 3 → 4 → 5, backward edge 5 → 0.
    let mut domain_graph = MockDomainGraph::new();
    let nodes: Vec<_> = (0..6).map(|_| domain_graph.add_node(())).collect();
    for i in 0..5 {
        domain_graph.add_edge(nodes[i], nodes[i + 1], ());
    }

    let snapshot = make_snapshot_with_backward_edges(
        domain_graph,
        120,
        20,
        usize::MAX,
        usize::MAX,
        &[(nodes[5], nodes[0])],
    );

    insta::assert_snapshot!("backward_edge_six_node_cycle", snapshot);
}

#[test]
fn backward_edge_six_node_cycle_multi_partition() {
    let _ = env_logger::try_init();
    // Same 6-node cycle as backward_edge_six_node_cycle, but split across partitions
    // (layer_count=2), exercising the pin-relay path across partition boundaries.
    let mut domain_graph = MockDomainGraph::new();
    let nodes: Vec<_> = (0..6).map(|_| domain_graph.add_node(())).collect();
    for i in 0..5 {
        domain_graph.add_edge(nodes[i], nodes[i + 1], ());
    }

    let snapshot = make_snapshot_with_backward_edges(
        domain_graph,
        160,
        20,
        2,
        usize::MAX,
        &[(nodes[5], nodes[0])],
    );

    insta::assert_snapshot!("backward_edge_six_node_cycle_multi_partition", snapshot);
}

#[test]
fn backward_edge_partial_loop() {
    let _ = env_logger::try_init();
    // Backward edge that doesn't close to the start node.
    // Chain: 0 → 1 → 2 → 3 → 4, backward edge 3 → 1 creates an inner loop.
    let mut domain_graph = MockDomainGraph::new();
    let nodes: Vec<_> = (0..5).map(|_| domain_graph.add_node(())).collect();
    for i in 0..4 {
        domain_graph.add_edge(nodes[i], nodes[i + 1], ());
    }

    let snapshot = make_snapshot_with_backward_edges(
        domain_graph,
        100,
        20,
        usize::MAX,
        usize::MAX,
        &[(nodes[3], nodes[1])],
    );

    insta::assert_snapshot!("backward_edge_partial_loop", snapshot);
}

#[test]
fn backward_edge_diamond_loop() {
    let _ = env_logger::try_init();
    // Diamond 0 → {1, 2} → 3 with a backward edge 3 → 0 creating a loop
    // around the entire structure.
    let mut domain_graph = MockDomainGraph::new();
    let nodes: Vec<_> = (0..4).map(|_| domain_graph.add_node(())).collect();
    domain_graph.add_edge(nodes[0], nodes[1], ());
    domain_graph.add_edge(nodes[0], nodes[2], ());
    domain_graph.add_edge(nodes[1], nodes[3], ());
    domain_graph.add_edge(nodes[2], nodes[3], ());

    let snapshot = make_snapshot_with_backward_edges(
        domain_graph,
        100,
        20,
        usize::MAX,
        usize::MAX,
        &[(nodes[3], nodes[0])],
    );

    insta::assert_snapshot!("backward_edge_diamond_loop", snapshot);
}

#[test]
fn backward_edge_branched_cycle() {
    let _ = env_logger::try_init();
    // Main chain with an external branch that merges in mid-cycle.
    // Chain: 0 → 1 → 2 → 3 → 4, branch: 5 → 2, backward edge 4 → 0.
    let mut domain_graph = MockDomainGraph::new();
    let nodes: Vec<_> = (0..6).map(|_| domain_graph.add_node(())).collect();
    for i in 0..4 {
        domain_graph.add_edge(nodes[i], nodes[i + 1], ());
    }
    domain_graph.add_edge(nodes[5], nodes[2], ());

    let snapshot = make_snapshot_with_backward_edges(
        domain_graph,
        120,
        25,
        usize::MAX,
        usize::MAX,
        &[(nodes[4], nodes[0])],
    );

    insta::assert_snapshot!("backward_edge_branched_cycle", snapshot);
}

#[test]
fn backward_edge_local_loop_multi_partition() {
    let _ = env_logger::try_init();
    // Long chain 0→1→…→7 split across partitions (layer_count=2), with a backward edge
    // 5→2 that only loops over the middle. The bypass should span just the 2→5 region,
    // not the whole graph - contrast with backward_edge_six_node_cycle_multi_partition,
    // whose full-width loop reaches partition 0 and the last partition.
    let mut domain_graph = MockDomainGraph::new();
    let nodes: Vec<_> = (0..8).map(|_| domain_graph.add_node(())).collect();
    for i in 0..7 {
        domain_graph.add_edge(nodes[i], nodes[i + 1], ());
    }

    let snapshot = make_snapshot_with_backward_edges(
        domain_graph,
        200,
        20,
        2,
        usize::MAX,
        &[(nodes[5], nodes[2])],
    );

    insta::assert_snapshot!("backward_edge_local_loop_multi_partition", snapshot);
}

#[test]
fn backward_edge_two_independent_local_loops() {
    let _ = env_logger::try_init();
    // Two disjoint local cycles on one chain split across partitions: 2→0 on the left and
    // 7→5 on the right. Each should render as its own compact bypass over its own region,
    // rather than both stacking as full-width lines competing across the whole canvas.
    let mut domain_graph = MockDomainGraph::new();
    let nodes: Vec<_> = (0..8).map(|_| domain_graph.add_node(())).collect();
    for i in 0..7 {
        domain_graph.add_edge(nodes[i], nodes[i + 1], ());
    }

    let snapshot = make_snapshot_with_backward_edges(
        domain_graph,
        200,
        20,
        2,
        usize::MAX,
        &[(nodes[2], nodes[0]), (nodes[7], nodes[5])],
    );

    insta::assert_snapshot!("backward_edge_two_independent_local_loops", snapshot);
}

// Cycle auto-detection tests (ported from the view-cycles branch). Unlike the
// `backward_edge_*` tests above, these hand a raw cyclic graph to the auto-detecting
// `make_snapshot`/`make_snapshot_pinned` path (`GraphController::new_with_config`) and let
// `cycle_removal::remove_cycles` identify the backward edges, so they exercise detection,
// self-loops, and `pin_source` end to end rather than the explicit-edge entry point.

/// Build a graph from an explicit node count and edge list.
#[cfg(test)]
fn graph_from_edges(node_count: usize, edges: &[(usize, usize)]) -> MockDomainGraph {
    let mut graph = MockDomainGraph::new();
    let nodes: Vec<_> = (0..node_count).map(|_| graph.add_node(())).collect();
    for &(source, target) in edges {
        graph.add_edge(nodes[source], nodes[target], ());
    }
    graph
}

/// Build a single directed cycle `0 -> 1 -> ... -> node_count-1 -> 0`.
#[cfg(test)]
fn cycle_graph(node_count: usize) -> MockDomainGraph {
    let mut graph = MockDomainGraph::new();
    let nodes: Vec<_> = (0..node_count).map(|_| graph.add_node(())).collect();
    for i in 0..node_count {
        graph.add_edge(nodes[i], nodes[(i + 1) % node_count], ());
    }
    graph
}

/// Add an edge between two nodes identified by their positional index.
#[cfg(test)]
fn add_edge_by_index(graph: &mut MockDomainGraph, source: usize, target: usize) {
    let nodes: Vec<_> = graph.node_indices().collect();
    graph.add_edge(nodes[source], nodes[target], ());
}

/// Like `make_snapshot`, but forces `pin_source` to the given node index so cycle detection
/// breaks each cycle relative to that node (see `PartitionConfig::pin_source`).
#[cfg(test)]
fn make_snapshot_pinned(
    domain_graph: MockDomainGraph,
    viewport_width: u16,
    viewport_height: u16,
    layer_count: usize,
    node_count: usize,
    pin_source: usize,
) -> String {
    use crate::graph_controller::GraphConfig;

    let node_sizer = FixedNodeSizer {
        width: 5,
        height: 3,
    };
    let mut renderer = TestRenderers::debug();

    let mut terminal = create_test_terminal(viewport_width, viewport_height);

    let mut config = GraphConfig::default();
    config.partition.layer_count = layer_count;
    config.partition.node_count = node_count;
    config.partition.pin_source = Some(petgraph::graph::NodeIndex::new(pin_source));
    let mut controller = GraphController::new_with_config(domain_graph.clone(), node_sizer, config);

    controller.viewport_state.viewport_bounds =
        ratatui::layout::Rect::new(0, 0, viewport_width, viewport_height);
    controller.set_detail_level(VisualDetail::Full);

    let result = terminal.draw(|f| {
        let area = f.area();
        controller.viewport_state.viewport_bounds = area;
        let _ = controller.ensure_camera_coverage();
        controller
            .rebuild_viewport_graph()
            .expect("Failed to rebuild viewport graph for snapshot generation");
        let viewport_graph = controller.get_viewport_graph();
        let detail_level = controller.get_detail_level();
        let mut buffer = WorldBuffer::new(f.buffer_mut(), &controller.viewport_state);
        plot_viewport_graph(
            viewport_graph,
            &mut buffer,
            &mut renderer,
            controller.graph(),
            detail_level,
            &crate::theme::current_theme(),
        );
    });

    match result {
        Ok(_) => format!("{}", terminal.backend()),
        Err(e) => format!("Rendering failed: {}", e),
    }
}

#[test]
fn cycle_simple_autodetected() {
    let _ = env_logger::try_init();
    // A bare 3-cycle with no explicit backward edge: detection finds the loop-closing edge.
    let snapshot = make_snapshot(
        graph_from_edges(3, &[(0, 1), (1, 2), (2, 0)]),
        80,
        20,
        1000,
        1000,
    );
    insta::assert_snapshot!("cycle_simple_autodetected", snapshot);
}

#[test]
fn cycle_self_loop_autodetected() {
    let _ = env_logger::try_init();
    // A single node with a self-loop (0 -> 0): the pins share the node's partition and the
    // loop stays entirely local. Self-loops were a non-goal of the pin work but fall out of
    // the same-partition path once detection reports them.
    let snapshot = make_snapshot(graph_from_edges(1, &[(0, 0)]), 60, 20, 1000, 1000);
    insta::assert_snapshot!("cycle_self_loop_autodetected", snapshot);
}

#[test]
fn cycle_with_chord_autodetected() {
    let _ = env_logger::try_init();
    // An 8-cycle plus a chord (6 -> 3): two backward edges, each scoped to its own span.
    let mut domain_graph = cycle_graph(8);
    add_edge_by_index(&mut domain_graph, 6, 3);
    let snapshot = make_snapshot(domain_graph, 100, 25, 1000, 1000);
    insta::assert_snapshot!("cycle_with_chord_autodetected", snapshot);
}

#[test]
fn cycle_with_two_chords_autodetected() {
    let _ = env_logger::try_init();
    // An 8-cycle plus two chords (6 -> 3, 4 -> 1): three independent backward edges.
    let mut domain_graph = cycle_graph(8);
    add_edge_by_index(&mut domain_graph, 6, 3);
    add_edge_by_index(&mut domain_graph, 4, 1);
    let snapshot = make_snapshot(domain_graph, 120, 25, 1000, 1000);
    insta::assert_snapshot!("cycle_with_two_chords_autodetected", snapshot);
}

#[test]
fn cycle_pinned_source() {
    let _ = env_logger::try_init();
    // A 12-cycle with node 6 pinned as the source: detection breaks the cycle relative to
    // node 6 rather than at petgraph's default entry point.
    let snapshot = make_snapshot_pinned(cycle_graph(12), 100, 20, 1000, 1000, 6);
    insta::assert_snapshot!("cycle_pinned_source", snapshot);
}

#[test]
fn cycle_pinned_source_partitioned() {
    let _ = env_logger::try_init();
    // Same as `cycle_pinned_source`, but split across partitions to exercise the pin relay.
    let snapshot = make_snapshot_pinned(cycle_graph(12), 160, 20, 2, 8, 6);
    insta::assert_snapshot!("cycle_pinned_source_partitioned", snapshot);
}

/// Render a cyclic graph through a viewport narrower than the graph, so only the partitions
/// covering the camera load. Returns the rendered frame plus the loaded/total partition
/// counts so a test can assert the load stayed partial.
#[cfg(test)]
fn render_partial_view(
    domain_graph: MockDomainGraph,
    viewport_width: u16,
    viewport_height: u16,
    layer_count: usize,
    node_count: usize,
) -> (String, usize, usize) {
    use crate::graph_controller::GraphConfig;

    let node_sizer = FixedNodeSizer {
        width: 5,
        height: 3,
    };
    let mut renderer = TestRenderers::debug();
    let mut terminal = create_test_terminal(viewport_width, viewport_height);

    let mut config = GraphConfig::default();
    config.partition.layer_count = layer_count;
    config.partition.node_count = node_count;
    let mut controller = GraphController::new_with_config(domain_graph, node_sizer, config);
    controller.set_detail_level(VisualDetail::Full);

    let result = terminal.draw(|f| {
        let area = f.area();
        controller.viewport_state.viewport_bounds = area;
        let _ = controller.ensure_camera_coverage();
        controller
            .rebuild_viewport_graph()
            .expect("Failed to rebuild viewport graph for snapshot generation");
        let viewport_graph = controller.get_viewport_graph();
        let detail_level = controller.get_detail_level();
        let mut buffer = WorldBuffer::new(f.buffer_mut(), &controller.viewport_state);
        plot_viewport_graph(
            viewport_graph,
            &mut buffer,
            &mut renderer,
            controller.graph(),
            detail_level,
            &crate::theme::current_theme(),
        );
    });

    let total = controller
        .partition_controller
        .partition_table
        .partitions
        .len();
    let loaded = controller.loaded_partition_count();
    let frame = match result {
        Ok(_) => format!("{}", terminal.backend()),
        Err(e) => format!("Rendering failed: {}", e),
    };
    (frame, loaded, total)
}

#[test]
fn cycle_partial_partition_loading() {
    let _ = env_logger::try_init();
    // A 24-node cycle split into many small partitions (node_count=3 -> 15 partitions),
    // viewed through a viewport far narrower than the graph. Only the leftmost partitions
    // covering the camera load; the full-cycle backward edge's right pin lives in the
    // last (unloaded) partition, yet its bypass must still render across the loaded region
    // and run off the right edge toward the off-screen pin - without crashing on the
    // partitions that were never loaded.
    let (snapshot, loaded, total) = render_partial_view(cycle_graph(24), 40, 20, 2, 3);

    assert!(
        loaded < total,
        "expected a partial load, but {loaded} of {total} partitions loaded"
    );
    assert!(
        snapshot.contains('◀'),
        "the loop's bypass should still render across the loaded partitions"
    );

    insta::assert_snapshot!("cycle_partial_partition_loading", snapshot);
}

#[test]
fn cycle_with_nested_chords() {
    let _ = env_logger::try_init();
    // An 8-cycle with two chords whose spans nest: 7 -> 1 encloses 6 -> 3. Three backward
    // edges total; the inner chord's bypass should sit inside the outer chord's.
    let mut domain_graph = cycle_graph(8);
    add_edge_by_index(&mut domain_graph, 7, 1);
    add_edge_by_index(&mut domain_graph, 6, 3);
    let snapshot = make_snapshot(domain_graph, 120, 25, 1000, 1000);
    insta::assert_snapshot!("cycle_with_nested_chords", snapshot);
}

#[test]
fn cycle_with_overlapping_chords() {
    let _ = env_logger::try_init();
    // Two chords whose spans partially overlap without nesting: 5 -> 1 (spans 1..5) and
    // 7 -> 3 (spans 3..7) share the 3..5 range. Since every loop is biased to the same
    // vertical side, this exercises how two co-located bypasses share channels.
    let mut domain_graph = cycle_graph(8);
    add_edge_by_index(&mut domain_graph, 5, 1);
    add_edge_by_index(&mut domain_graph, 7, 3);
    let snapshot = make_snapshot(domain_graph, 120, 25, 1000, 1000);
    insta::assert_snapshot!("cycle_with_overlapping_chords", snapshot);
}

#[test]
fn cycle_with_chords_shared_target() {
    let _ = env_logger::try_init();
    // Two chords pointing at the same node: 5 -> 2 and 7 -> 2. Both loopbacks land on the
    // same target, exercising two backward edges that share a left endpoint.
    let mut domain_graph = cycle_graph(8);
    add_edge_by_index(&mut domain_graph, 5, 2);
    add_edge_by_index(&mut domain_graph, 7, 2);
    let snapshot = make_snapshot(domain_graph, 120, 25, 1000, 1000);
    insta::assert_snapshot!("cycle_with_chords_shared_target", snapshot);
}

#[test]
fn cycle_with_chord_multi_partition() {
    let _ = env_logger::try_init();
    // A local backward chord whose span crosses partition boundaries (layer_count=2), so its
    // bypass relays through intermediate stitch chains rather than staying inside a single
    // section. Pinning node 0 fixes the ordering (0..9) so the chord 8 -> 3 is unambiguously
    // backward and local (spanning ranks 3..8), alongside the full-width main loop 9 -> 0.
    let mut domain_graph = cycle_graph(10);
    add_edge_by_index(&mut domain_graph, 8, 3);
    let snapshot = make_snapshot_pinned(domain_graph, 200, 25, 2, usize::MAX, 0);
    insta::assert_snapshot!("cycle_with_chord_multi_partition", snapshot);
}
