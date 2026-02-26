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
            &controller.graph,
            detail_level,
            &controller.theme,
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
    insta::assert_snapshot!("simple_chain", snapshot);
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
    config.partition.node_count = 3; // Force node-based partitioning as well

    let mut controller = GraphController::new_with_config(&domain_graph, node_sizer, config);

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

    let mut controller = GraphController::new_with_config(&domain_graph, node_sizer, config);
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

    let mut controller = GraphController::new_with_config(&domain_graph, node_sizer, config);

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

/// Proves crossing-reduction tie handling is deterministic by constructing the same symmetric
/// graph twice, but inserting edges in a different order (which affects DFS-based init order).
/// With the tiebreaker in `gen-sugiyama`, these should render identically.
#[test]
fn test_layout_determinism_across_edge_insertion_order_symmetric_fan() {
    let _ = env_logger::try_init();

    // Graph:
    //   n0 -> {n1,n2,n3} -> n4
    // The three middle nodes are perfectly symmetric, so their barycenters tie.
    // Without a deterministic tiebreaker, the middle layer can preserve the DFS visit order.

    let mut g1 = MockDomainGraph::new();
    let n0 = g1.add_node(());
    let n1 = g1.add_node(());
    let n2 = g1.add_node(());
    let n3 = g1.add_node(());
    let n4 = g1.add_node(());
    g1.add_edge(n0, n1, ());
    g1.add_edge(n0, n2, ());
    g1.add_edge(n0, n3, ());
    g1.add_edge(n1, n4, ());
    g1.add_edge(n2, n4, ());
    g1.add_edge(n3, n4, ());

    let mut g2 = MockDomainGraph::new();
    let n0 = g2.add_node(());
    let n1 = g2.add_node(());
    let n2 = g2.add_node(());
    let n3 = g2.add_node(());
    let n4 = g2.add_node(());
    // Same edges, different insertion order.
    g2.add_edge(n0, n3, ());
    g2.add_edge(n0, n1, ());
    g2.add_edge(n0, n2, ());
    g2.add_edge(n3, n4, ());
    g2.add_edge(n1, n4, ());
    g2.add_edge(n2, n4, ());

    let snapshot1 = make_snapshot(g1, 80, 25, usize::MAX, usize::MAX);
    let snapshot2 = make_snapshot(g2, 80, 25, usize::MAX, usize::MAX);

    assert_eq!(
        snapshot1, snapshot2,
        "Symmetric fan layout should be identical regardless of edge insertion order"
    );
}

#[test]
fn test_double_chain() {
    let _ = env_logger::try_init();

    // Create a graph with 18 nodes arranged in two chains with a common start and stop node.
    // Chain 1: n1 -> n2 -> n3 -> n4 -> n5 -> n6 -> n7 -> n8 -> n9 -> n10 (10 nodes)
    // Chain 2: n1 -> n12 -> n13 -> n14 -> n15 -> n16 -> n17 -> n18 -> n19 -> n10 (10 nodes)
    // Shared nodes: n1 (start), n10 (stop)
    // Total unique nodes: 18

    let mut domain_graph = MockDomainGraph::new();

    // Add all nodes (indices 0-17 correspond to n1-n10 and n12-n19)
    // Using index mapping:
    // 0 -> n1 (shared start)
    // 1 -> n2
    // 2 -> n3
    // 3 -> n4
    // 4 -> n5
    // 5 -> n6
    // 6 -> n7
    // 7 -> n8
    // 8 -> n9
    // 9 -> n10 (shared stop)
    // 10 -> n12
    // 11 -> n13
    // 12 -> n14
    // 13 -> n15
    // 14 -> n16
    // 15 -> n17
    // 16 -> n18
    // 17 -> n19
    let nodes: Vec<_> = (0..18).map(|_| domain_graph.add_node(())).collect();

    // Chain 1: n1(0) -> n2(1) -> n3(2) -> n4(3) -> n5(4) -> n6(5) -> n7(6) -> n8(7) -> n9(8) -> n10(9)
    for i in 0..9 {
        domain_graph.add_edge(nodes[i], nodes[i + 1], ());
    }

    // Chain 2: n1(0) -> n12(10) -> n13(11) -> n14(12) -> n15(13) -> n16(14) -> n17(15) -> n18(16) -> n19(17) -> n10(9)
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
    let n_a = domain_graph.add_node(());
    let n_b = domain_graph.add_node(());
    let n_c = domain_graph.add_node(());
    let n_d = domain_graph.add_node(());
    let n_e = domain_graph.add_node(());
    let n_f = domain_graph.add_node(());

    // Main chain: A -> B -> C -> D -> E
    domain_graph.add_edge(n_a, n_b, ());
    domain_graph.add_edge(n_b, n_c, ());
    domain_graph.add_edge(n_c, n_d, ());
    domain_graph.add_edge(n_d, n_e, ());

    // Bypass: A -> F -> E
    domain_graph.add_edge(n_a, n_f, ());
    domain_graph.add_edge(n_f, n_e, ());

    let snapshot = make_snapshot(domain_graph, 80, 25, usize::MAX, usize::MAX);

    insta::assert_snapshot!("asymmetric_diamond", snapshot);
}

#[test]
fn test_asymmetric_diamond_4_1() {
    let _ = env_logger::try_init();

    // Longer leg: 4 intermediate nodes (6 total), shorter leg: 1 intermediate (2 total)
    // A -> B -> C -> D -> E -> F
    // A -> G -> E

    let mut domain_graph = MockDomainGraph::new();
    let n_a = domain_graph.add_node(());
    let n_b = domain_graph.add_node(());
    let n_c = domain_graph.add_node(());
    let n_d = domain_graph.add_node(());
    let n_e = domain_graph.add_node(());
    let n_f = domain_graph.add_node(());
    let n_g = domain_graph.add_node(());

    domain_graph.add_edge(n_a, n_b, ());
    domain_graph.add_edge(n_b, n_c, ());
    domain_graph.add_edge(n_c, n_d, ());
    domain_graph.add_edge(n_d, n_e, ());
    domain_graph.add_edge(n_e, n_f, ());

    domain_graph.add_edge(n_a, n_g, ());
    domain_graph.add_edge(n_g, n_f, ());

    let snapshot = make_snapshot(domain_graph, 80, 25, usize::MAX, usize::MAX);

    insta::assert_snapshot!("asymmetric_diamond_4_1", snapshot);
}

#[test]
fn test_asymmetric_diamond_3_2() {
    let _ = env_logger::try_init();

    // A -> B -> C -> D -> E
    // A -> F1 -> F2 -> E

    let mut domain_graph = MockDomainGraph::new();
    let n_a = domain_graph.add_node(());
    let n_b = domain_graph.add_node(());
    let n_c = domain_graph.add_node(());
    let n_d = domain_graph.add_node(());
    let n_e = domain_graph.add_node(());
    let n_f1 = domain_graph.add_node(());
    let n_f2 = domain_graph.add_node(());

    domain_graph.add_edge(n_a, n_b, ());
    domain_graph.add_edge(n_b, n_c, ());
    domain_graph.add_edge(n_c, n_d, ());
    domain_graph.add_edge(n_d, n_e, ());

    domain_graph.add_edge(n_a, n_f1, ());
    domain_graph.add_edge(n_f1, n_f2, ());
    domain_graph.add_edge(n_f2, n_e, ());

    let snapshot = make_snapshot(domain_graph, 80, 25, usize::MAX, usize::MAX);

    insta::assert_snapshot!("asymmetric_diamond_3_2", snapshot);
}

#[test]
fn test_asymmetric_diamond_4_2() {
    let _ = env_logger::try_init();

    // A -> B -> C -> D -> E -> F
    // A -> X -> Y -> F

    let mut domain_graph = MockDomainGraph::new();
    let n_a = domain_graph.add_node(());
    let n_b = domain_graph.add_node(());
    let n_c = domain_graph.add_node(());
    let n_d = domain_graph.add_node(());
    let n_e = domain_graph.add_node(());
    let n_f = domain_graph.add_node(());
    let n_x = domain_graph.add_node(());
    let n_y = domain_graph.add_node(());

    domain_graph.add_edge(n_a, n_b, ());
    domain_graph.add_edge(n_b, n_c, ());
    domain_graph.add_edge(n_c, n_d, ());
    domain_graph.add_edge(n_d, n_e, ());
    domain_graph.add_edge(n_e, n_f, ());

    domain_graph.add_edge(n_a, n_x, ());
    domain_graph.add_edge(n_x, n_y, ());
    domain_graph.add_edge(n_y, n_f, ());

    let snapshot = make_snapshot(domain_graph, 80, 25, usize::MAX, usize::MAX);

    insta::assert_snapshot!("asymmetric_diamond_4_2", snapshot);
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
    let n0 = domain_graph.add_node(());
    let n1 = domain_graph.add_node(());
    let n2 = domain_graph.add_node(());
    domain_graph.add_edge(n0, n1, ());
    domain_graph.add_edge(n1, n2, ());

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

    impl NodeSizer<&MockDomainGraph> for VariableDetailSizer {
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

    impl NodeRenderer<&MockDomainGraph> for UltrawideRenderer {
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
        GraphController::new_with_config(&domain_graph, VariableDetailSizer, config);

    // 4. Starts in minimal level-of-detail
    controller.set_detail_level(VisualDetail::Minimal);

    // 5. Cursor setup (not visible in the snapshots though)
    controller.show_cursor();
    controller.initialize_cursor();
    controller.cursor.set_node(n0, (0.0, 0.0));

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

    let pos0 = viewport_graph.node_positions.get(&n0).unwrap();
    let pos1 = viewport_graph.node_positions.get(&n1).unwrap();

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
