#[cfg(test)]
use crate::geometry::WorldPos;
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

/// Helper function to create a diamond graph with variable node widths for bridge testing
#[cfg(test)]
#[allow(dead_code)]
fn create_variable_width_diamond_snapshot(
    viewport_width: u16,
    viewport_height: u16,
    layer_count: usize,
    node_count: usize,
) -> String {
    use crate::{
        layout::VisualDetail,
        plotter::NodeSizer,
        testing::mocks::{MockDomainGraph, TestGraphs, TestRenderers},
    };

    // Use the standard diamond graph from TestGraphs
    let domain_graph = TestGraphs::domain_diamond();

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

    // Also implement for reference type
    impl NodeSizer<&MockDomainGraph> for VariableWidthSizer {
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
    NS: for<'a> crate::plotter::NodeSizer<&'a MockDomainGraph>,
    R: for<'a> crate::plotter::NodeRenderer<&'a MockDomainGraph>,
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
    let mut controller = GraphController::new_with_config(&domain_graph, node_sizer, config);

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

        let _ = controller.rebuild_viewport_graph();
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
            &controller.graph,
            detail_level,
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

// New ViewportGraph-based visual regression tests

#[test]
fn viewport_visual_regression_simple_chain() {
    let _ = env_logger::try_init();
    // Create a simple chain domain graph: 0 -> 1 -> 2
    let mut domain_graph = MockDomainGraph::new();
    let n0 = domain_graph.add_node(());
    let n1 = domain_graph.add_node(());
    let n2 = domain_graph.add_node(());
    domain_graph.add_edge(n0, n1, ());
    domain_graph.add_edge(n1, n2, ());

    let snapshot = make_snapshot(domain_graph, 60, 20, 2, 8);
    insta::assert_snapshot!("viewport_simple_chain", snapshot);
}

#[test]
fn viewport_visual_regression_diamond() {
    let _ = env_logger::try_init();
    // Create a diamond domain graph: 0 -> {1, 2} -> 3
    let mut domain_graph = MockDomainGraph::new();
    let n0 = domain_graph.add_node(());
    let n1 = domain_graph.add_node(());
    let n2 = domain_graph.add_node(());
    let n3 = domain_graph.add_node(());
    domain_graph.add_edge(n0, n1, ());
    domain_graph.add_edge(n0, n2, ());
    domain_graph.add_edge(n1, n3, ());
    domain_graph.add_edge(n2, n3, ());

    let snapshot = make_snapshot(domain_graph, 60, 20, 2, 8);
    insta::assert_snapshot!("viewport_diamond", snapshot);
}

#[test]
fn viewport_visual_regression_single_node() {
    let _ = env_logger::try_init();
    // Create a single node domain graph
    let mut domain_graph = MockDomainGraph::new();
    domain_graph.add_node(());

    let snapshot = make_snapshot(domain_graph, 60, 20, 2, 8);
    insta::assert_snapshot!("viewport_single_node", snapshot);
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
    insta::assert_snapshot!("viewport_subcombinatorial_dag", snapshot);
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
    insta::assert_snapshot!("viewport_complex_dag", snapshot);
}

#[test]
#[ignore]
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

    insta::assert_snapshot!("viewport_multi_partition_chain", snapshot);
}

#[test]
fn viewport_visual_regression_extended_complex_dag_no_partitioning() {
    let _ = env_logger::try_init();
    // Test 1: No partitioning - everything in one partition
    use crate::testing::mocks::TestGraphs;
    let domain_graph = TestGraphs::domain_complex_dag();

    let snapshot = make_snapshot(domain_graph, 80, 25, usize::MAX, usize::MAX);
    insta::assert_snapshot!("viewport_extended_complex_dag_no_partitioning", snapshot);
}

#[test]
fn viewport_visual_regression_extended_complex_dag_layer_partitioning() {
    let _ = env_logger::try_init();
    // Test 2: Layer-based partitioning
    use crate::testing::mocks::TestGraphs;
    let domain_graph = TestGraphs::domain_complex_dag();

    let snapshot = make_snapshot(domain_graph, 80, 25, 3, usize::MAX);
    insta::assert_snapshot!("viewport_extended_complex_dag_layer_partitioning", snapshot);
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
#[ignore]
#[test]
fn viewport_visual_regression_extended_complex_dag_node_partitioning() {
    let _ = env_logger::try_init();
    // Test 3: Node-based partitioning
    use crate::testing::mocks::TestGraphs;
    let domain_graph = TestGraphs::domain_complex_dag();

    let snapshot = make_snapshot(domain_graph, 80, 25, usize::MAX, 3);
    insta::assert_snapshot!("viewport_extended_complex_dag_node_partitioning", snapshot);
}

#[test]
fn viewport_visual_regression_extended_diamond_no_partitioning() {
    let _ = env_logger::try_init();
    // Test 1: No partitioning - everything in one partition
    use crate::testing::mocks::TestGraphs;
    let domain_graph = TestGraphs::domain_extended_diamond();

    let snapshot = make_snapshot(domain_graph, 80, 25, usize::MAX, usize::MAX);
    insta::assert_snapshot!("viewport_extended_diamond_no_partitioning", snapshot);
}

#[test]
fn viewport_visual_regression_extended_diamond_layer_partitioning() {
    let _ = env_logger::try_init();
    // Test 2: Layer-based partitioning
    use crate::testing::mocks::TestGraphs;
    let domain_graph = TestGraphs::domain_extended_diamond();

    let snapshot = make_snapshot(domain_graph, 80, 25, 3, usize::MAX);
    insta::assert_snapshot!("viewport_extended_diamond_layer_partitioning", snapshot);
}

#[test]
fn viewport_visual_regression_extended_diamond_node_partitioning() {
    let _ = env_logger::try_init();
    // Test 3: Node-based partitioning
    use crate::testing::mocks::TestGraphs;
    let domain_graph = TestGraphs::domain_extended_diamond();

    let snapshot = make_snapshot(domain_graph, 80, 25, usize::MAX, 3);
    insta::assert_snapshot!("viewport_extended_diamond_node_partitioning", snapshot);
}

#[test]
fn test_layer_coordinate_alignment_and_ordering() {
    let _ = env_logger::try_init();

    use crate::{
        geometry::{BigRect, WorldPos},
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
    config.partition.node_count = 3; // Force node-based partitioning as well

    let mut controller = GraphController::new_with_config(&domain_graph, node_sizer, config);

    // Set detail level first
    controller.set_detail_level(VisualDetail::Full);

    // Create viewport bounds using u16::MAX/2 limits to fit in Ratatui rect
    let max_coord = (u16::MAX / 2) as i64;
    let min_coord = -((u16::MAX / 2) as i64);

    let unlimited_viewport = BigRect::from_coords(min_coord, min_coord, max_coord, max_coord);

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

    // Also implement for reference type
    impl NodeSizer<&MockDomainGraph> for VariableWidthSizer {
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

    insta::assert_snapshot!("viewport_bridge_position_variable_widths", snapshot);
}

#[test]
fn test_skip_layer() {
    let _ = env_logger::try_init();
    use crate::testing::mocks::TestGraphs;

    let domain_graph = TestGraphs::domain_skip_layer();
    let snapshot = make_snapshot(domain_graph, 80, 25, usize::MAX, usize::MAX);

    insta::assert_snapshot!("viewport_skip_layer", snapshot);
}

#[test]
fn test_skip_layer_partition_boundary() {
    let _ = env_logger::try_init();
    use crate::testing::mocks::TestGraphs;

    let domain_graph = TestGraphs::domain_skip_layer();
    let snapshot = make_snapshot(domain_graph, 80, 25, 2, usize::MAX);

    insta::assert_snapshot!("viewport_skip_layer_partition_boundary", snapshot);
}

#[test]
#[ignore]
fn test_skip_layer_terminal_stitch_edge_bundles() {
    let _ = env_logger::try_init();

    use crate::{
        geometry::{BigRect, WorldPos},
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

    let mut controller = GraphController::new_with_config(&domain_graph, node_sizer, config);

    controller.set_detail_level(VisualDetail::Full);

    let max_coord = (u16::MAX / 2) as i64;
    let min_coord = -((u16::MAX / 2) as i64);

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

    // There should be exactly 1 edge without bundles (from terminal stitch nodes)
    // Note: Previously there were 2, but the fix for layer-skipping edges now properly
    // labels routing edges, so only the actual stitch edges remain unlabeled
    assert_eq!(
        edges_without_bundles, 1,
        "Expected exactly 1 edge without bundles (from terminal stitch nodes), but found {}",
        edges_without_bundles
    );

    println!(
        "Test completed successfully - found exactly 1 unbundled edge from terminal stitch nodes"
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
    let n0 = domain_graph.add_node(());
    let n1 = domain_graph.add_node(());
    let n2 = domain_graph.add_node(());
    let n3 = domain_graph.add_node(());
    domain_graph.add_edge(n0, n1, ());
    domain_graph.add_edge(n0, n2, ());
    domain_graph.add_edge(n1, n3, ());
    domain_graph.add_edge(n2, n3, ());

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

    // Implement for reference type
    impl NodeSizer<&MockDomainGraph> for OddEvenSizer {
        fn get_node_size(
            &self,
            node: &petgraph::stable_graph::NodeIndex<u32>,
            _scale: VisualDetail,
        ) -> (u64, u64) {
            match node.index() {
                0 => (4, 1),
                1 => (6, 1),
                2 => (6, 1),
                3 => (8, 1),
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

    insta::assert_snapshot!("viewport_even_width_nodes", snapshot);
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
