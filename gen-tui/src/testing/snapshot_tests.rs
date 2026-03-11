#![cfg(test)]
use petgraph::graph::NodeIndex;
use ratatui::layout::Rect;

use super::layout_tests::{
    add_edge_by_index, chain_graph, cycle_graph, graph_from_edges, init_test_logging,
};
use crate::{
    dot_export::export_to_dot,
    graph_controller::{GraphConfig, GraphController, WorldBuffer},
    layout::VisualDetail,
    plotter::{NodeRenderer, NodeSizer, plot_viewport_graph},
    testing::{
        create_test_terminal,
        mocks::{FixedNodeSizer, MockDomainGraph, TestGraphs, TestRenderers},
    },
    viewport_graph::CroppedGraph,
};

#[derive(Debug, Clone, Copy)]
struct SnapshotOptions {
    viewport: Rect,
    layer_count: usize,
    node_count: usize,
    detail: VisualDetail,
    pin_source: Option<usize>,
}

impl Default for SnapshotOptions {
    fn default() -> Self {
        Self {
            viewport: Rect::new(0, 0, 60, 20),
            layer_count: 2,
            node_count: 8,
            detail: VisualDetail::Full,
            pin_source: None,
        }
    }
}

fn maybe_export_dot(viewport_graph: &CroppedGraph) {
    let debug_enabled = std::env::var("RUST_LOG")
        .map(|v| v.contains("debug"))
        .unwrap_or(false);

    if !debug_enabled {
        return;
    }

    let thread = std::thread::current();
    let test_name = thread.name().unwrap_or("unknown_test");
    let filename = format!("{}_viewport.dot", test_name);

    if let Err(e) = export_to_dot(viewport_graph, &filename) {
        eprintln!("Failed to export dot file {}: {}", filename, e);
    }
}

/// Helper function to create viewport-based visual snapshots using GraphController.
fn make_snapshot_with<NS, R>(
    domain_graph: MockDomainGraph,
    options: SnapshotOptions,
    node_sizer: NS,
    mut renderer: R,
) -> String
where
    NS: for<'a> NodeSizer<&'a MockDomainGraph>,
    R: for<'a> NodeRenderer<&'a MockDomainGraph>,
{
    init_test_logging();

    let mut terminal = create_test_terminal(options.viewport.width, options.viewport.height);

    let mut config = GraphConfig::default();
    config.partition.layer_count = options.layer_count;
    config.partition.node_count = options.node_count;
    config.partition.pin_source = options.pin_source.map(NodeIndex::new);

    let mut controller = GraphController::new_with_config(&domain_graph, node_sizer, config);
    controller.viewport_state.viewport_bounds = options.viewport;
    controller.set_detail_level(options.detail);

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

        maybe_export_dot(viewport_graph);

        let mut buffer = WorldBuffer::new(f.buffer_mut(), &controller.viewport_state);
        plot_viewport_graph(
            viewport_graph,
            &mut buffer,
            &mut renderer,
            controller.graph(),
            detail_level,
            &controller.theme,
        );
    });

    match result {
        Ok(_) => format!("{}", terminal.backend()),
        Err(e) => format!("Rendering failed: {}", e),
    }
}

fn make_snapshot(domain_graph: MockDomainGraph) -> String {
    make_snapshot_with(
        domain_graph,
        SnapshotOptions::default(),
        FixedNodeSizer {
            width: 5,
            height: 3,
        },
        TestRenderers::debug(),
    )
}

fn make_snapshot_with_options(domain_graph: MockDomainGraph, options: SnapshotOptions) -> String {
    make_snapshot_with(
        domain_graph,
        options,
        FixedNodeSizer {
            width: 5,
            height: 3,
        },
        TestRenderers::debug(),
    )
}

// -----------------------------------------------------------------------------
// Snapshot regression tests
// -----------------------------------------------------------------------------

#[test]
fn simple_chain() {
    let snapshot = make_snapshot(graph_from_edges(3, &[(0, 1), (1, 2)]));
    insta::assert_snapshot!("simple_chain", snapshot);
}

#[test]
fn diamond() {
    let snapshot = make_snapshot(graph_from_edges(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]));
    insta::assert_snapshot!("diamond", snapshot);
}

#[test]
fn single_node() {
    let snapshot = make_snapshot(graph_from_edges(1, &[]));
    insta::assert_snapshot!("single_node", snapshot);
}

#[test]
fn subcombinatorial_dag() {
    let snapshot = make_snapshot(graph_from_edges(
        6,
        &[(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5)],
    ));
    insta::assert_snapshot!("subcombinatorial_dag", snapshot);
}

#[test]
fn complex_dag() {
    let snapshot = make_snapshot(graph_from_edges(
        9,
        &[
            (0, 1),
            (0, 2),
            (1, 3),
            (1, 4),
            (2, 4),
            (2, 5),
            (3, 6),
            (4, 6),
            (4, 7),
            (5, 7),
            (6, 8),
            (7, 8),
        ],
    ));
    insta::assert_snapshot!("complex_dag", snapshot);
}

#[test]
fn viewport_multi_partition_boundary_handling() {
    let snapshot = make_snapshot_with_options(
        chain_graph(20),
        SnapshotOptions {
            viewport: Rect::new(0, 0, 120, 30),
            layer_count: 3,
            node_count: 5,
            ..Default::default()
        },
    );

    insta::assert_snapshot!("multi_partition_chain", snapshot);
}

#[test]
fn extended_complex_dag_no_partitioning() {
    let snapshot = make_snapshot_with_options(
        TestGraphs::domain_complex_dag(),
        SnapshotOptions {
            viewport: Rect::new(0, 0, 80, 25),
            layer_count: usize::MAX,
            node_count: usize::MAX,
            ..Default::default()
        },
    );

    insta::assert_snapshot!("extended_complex_dag_no_partitioning", snapshot);
}

#[test]
fn extended_complex_dag_layer_partitioning() {
    let snapshot = make_snapshot_with_options(
        TestGraphs::domain_complex_dag(),
        SnapshotOptions {
            viewport: Rect::new(0, 0, 80, 25),
            layer_count: 3,
            node_count: usize::MAX,
            ..Default::default()
        },
    );

    insta::assert_snapshot!("extended_complex_dag_layer_partitioning", snapshot);
}

// This test is a good example of why we try to go for articulation points:
//  By breaking up the graph between layers that each have multiple nodes
//  suboptimal node orderings are encountered.
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
fn extended_complex_dag_node_partitioning() {
    let snapshot = make_snapshot_with_options(
        TestGraphs::domain_complex_dag(),
        SnapshotOptions {
            viewport: Rect::new(0, 0, 80, 25),
            layer_count: usize::MAX,
            node_count: 3,
            ..Default::default()
        },
    );

    insta::assert_snapshot!("extended_complex_dag_node_partitioning", snapshot);
}

#[test]
fn extended_diamond_no_partitioning() {
    let snapshot = make_snapshot_with_options(
        TestGraphs::domain_extended_diamond(),
        SnapshotOptions {
            viewport: Rect::new(0, 0, 80, 25),
            layer_count: usize::MAX,
            node_count: usize::MAX,
            ..Default::default()
        },
    );

    insta::assert_snapshot!("extended_diamond_no_partitioning", snapshot);
}

#[test]
fn extended_diamond_layer_partitioning() {
    let snapshot = make_snapshot_with_options(
        TestGraphs::domain_extended_diamond(),
        SnapshotOptions {
            viewport: Rect::new(0, 0, 80, 25),
            layer_count: 3,
            node_count: usize::MAX,
            ..Default::default()
        },
    );

    insta::assert_snapshot!("extended_diamond_layer_partitioning", snapshot);
}

#[test]
fn extended_diamond_node_partitioning() {
    let snapshot = make_snapshot_with_options(
        TestGraphs::domain_extended_diamond(),
        SnapshotOptions {
            viewport: Rect::new(0, 0, 80, 25),
            layer_count: usize::MAX,
            node_count: 3,
            ..Default::default()
        },
    );

    insta::assert_snapshot!("extended_diamond_node_partitioning", snapshot);
}

#[test]
fn bridge_position_with_variable_node_widths() {
    #[derive(Debug, Clone)]
    struct VariableWidthSizer;

    impl NodeSizer<MockDomainGraph> for VariableWidthSizer {
        fn get_node_size(&self, node: &NodeIndex<u32>, _scale: VisualDetail) -> (u64, u64) {
            match node.index() {
                0 => (4, 1),
                1 => (15, 2),
                2 => (2, 1),
                3 => (5, 1),
                _ => (3, 1),
            }
        }

        fn get_dummy_size(&self) -> (u64, u64) {
            (1, 1)
        }
    }

    impl NodeSizer<&MockDomainGraph> for VariableWidthSizer {
        fn get_node_size(&self, node: &NodeIndex<u32>, scale: VisualDetail) -> (u64, u64) {
            <Self as NodeSizer<MockDomainGraph>>::get_node_size(self, node, scale)
        }

        fn get_dummy_size(&self) -> (u64, u64) {
            <Self as NodeSizer<MockDomainGraph>>::get_dummy_size(self)
        }
    }

    let snapshot = make_snapshot_with(
        TestGraphs::domain_diamond(),
        SnapshotOptions {
            viewport: Rect::new(0, 0, 80, 25),
            layer_count: 2,
            node_count: 3,
            ..Default::default()
        },
        VariableWidthSizer,
        TestRenderers::debug(),
    );

    insta::assert_snapshot!("bridge_position_variable_widths", snapshot);
}

#[test]
fn test_skip_layer() {
    let snapshot = make_snapshot_with_options(
        TestGraphs::domain_skip_layer(),
        SnapshotOptions {
            viewport: Rect::new(0, 0, 80, 25),
            layer_count: usize::MAX,
            node_count: usize::MAX,
            ..Default::default()
        },
    );

    insta::assert_snapshot!("skip_layer", snapshot);
}

#[test]
fn test_skip_layer_partition_boundary() {
    let snapshot = make_snapshot_with_options(
        TestGraphs::domain_skip_layer(),
        SnapshotOptions {
            viewport: Rect::new(0, 0, 80, 25),
            layer_count: 2,
            node_count: usize::MAX,
            ..Default::default()
        },
    );

    insta::assert_snapshot!("skip_layer_partition_boundary", snapshot);
}

#[test]
fn viewport_chain_three_partitions_spanning_edge() {
    let mut domain_graph = chain_graph(6);
    add_edge_by_index(&mut domain_graph, 1, 4);

    let snapshot = make_snapshot_with_options(
        domain_graph,
        SnapshotOptions {
            viewport: Rect::new(0, 0, 100, 30),
            layer_count: 2,
            node_count: usize::MAX,
            ..Default::default()
        },
    );

    insta::assert_snapshot!("chain_three_partitions_spanning_edge", snapshot);
}

#[test]
fn viewport_chain_five_partitions_long_spanning_edge() {
    let mut domain_graph = chain_graph(10);
    add_edge_by_index(&mut domain_graph, 1, 8);

    // Note: if you cut off the partitions using node_count=5 the test will fail due to
    // a visual artefact, which is concession made when using node_count to create the cut.
    // Topologically, the graph was still correct.
    let snapshot = make_snapshot_with_options(
        domain_graph,
        SnapshotOptions {
            viewport: Rect::new(0, 0, 120, 35),
            layer_count: 2,
            node_count: usize::MAX,
            ..Default::default()
        },
    );

    insta::assert_snapshot!("chain_five_partitions_long_spanning_edge", snapshot);
}

#[test]
fn viewport_even_width_node_spacing() {
    #[derive(Debug, Clone)]
    struct OddEvenSizer;

    impl NodeSizer<MockDomainGraph> for OddEvenSizer {
        fn get_node_size(&self, node: &NodeIndex<u32>, _scale: VisualDetail) -> (u64, u64) {
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

    impl NodeSizer<&MockDomainGraph> for OddEvenSizer {
        fn get_node_size(&self, node: &NodeIndex<u32>, _scale: VisualDetail) -> (u64, u64) {
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

    let snapshot = make_snapshot_with(
        graph_from_edges(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]),
        SnapshotOptions {
            viewport: Rect::new(0, 0, 80, 25),
            layer_count: usize::MAX,
            node_count: usize::MAX,
            ..Default::default()
        },
        OddEvenSizer,
        TestRenderers::debug(),
    );

    insta::assert_snapshot!("even_width_nodes", snapshot);
}

/// Test for determinism by running the same layout multiple times and comparing snapshots.
/// This test uses the complex_dag graph with node-based partitioning to ensure
/// that HashMap/HashSet iterations produce consistent results.
#[test]
fn test_layout_determinism_with_partitioning() {
    init_test_logging();

    let num_iterations = 10;
    let mut snapshots = Vec::new();

    for i in 0..num_iterations {
        let graph = TestGraphs::domain_complex_dag();
        let snapshot = make_snapshot_with_options(
            graph,
            SnapshotOptions {
                viewport: Rect::new(0, 0, 80, 25),
                layer_count: usize::MAX,
                node_count: 3,
                ..Default::default()
            },
        );
        snapshots.push(snapshot);
        log::trace!("Generated snapshot {} for determinism test", i);
    }

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

    insta::assert_snapshot!("determinism_check_complex_dag_node_partitioning", first);
}

/// Proves crossing-reduction tie handling is deterministic by constructing the same symmetric
/// graph twice, but inserting edges in a different order.
#[test]
fn test_layout_determinism_across_edge_insertion_order_symmetric_fan() {
    init_test_logging();

    let graph_1 = graph_from_edges(5, &[(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4)]);

    let graph_2 = graph_from_edges(5, &[(0, 3), (0, 1), (0, 2), (3, 4), (1, 4), (2, 4)]);

    let options = SnapshotOptions {
        viewport: Rect::new(0, 0, 80, 25),
        layer_count: usize::MAX,
        node_count: usize::MAX,
        ..Default::default()
    };

    let snapshot1 = make_snapshot_with_options(graph_1, options);
    let snapshot2 = make_snapshot_with_options(graph_2, options);

    assert_eq!(
        snapshot1, snapshot2,
        "Symmetric fan layout should be identical regardless of edge insertion order"
    );
}

#[test]
fn test_double_chain() {
    // Shared start node 0 and shared stop node 9.
    let snapshot = make_snapshot_with_options(
        graph_from_edges(
            18,
            &[
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (0, 10),
                (10, 11),
                (11, 12),
                (12, 13),
                (13, 14),
                (14, 15),
                (15, 16),
                (16, 17),
                (17, 9),
            ],
        ),
        SnapshotOptions {
            viewport: Rect::new(0, 0, 120, 40),
            layer_count: 5,
            node_count: 20,
            ..Default::default()
        },
    );

    insta::assert_snapshot!("double_chain", snapshot);
}

#[test]
fn test_asymmetric_diamond() {
    let snapshot = make_snapshot_with_options(
        graph_from_edges(6, &[(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 4)]),
        SnapshotOptions {
            viewport: Rect::new(0, 0, 80, 25),
            layer_count: usize::MAX,
            node_count: usize::MAX,
            ..Default::default()
        },
    );

    insta::assert_snapshot!("asymmetric_diamond", snapshot);
}

#[test]
fn test_asymmetric_diamond_2_1() {
    let snapshot = make_snapshot_with_options(
        graph_from_edges(5, &[(0, 1), (1, 2), (2, 3), (0, 4), (4, 3)]),
        SnapshotOptions {
            viewport: Rect::new(0, 0, 80, 25),
            layer_count: usize::MAX,
            node_count: usize::MAX,
            ..Default::default()
        },
    );

    insta::assert_snapshot!("asymmetric_diamond_2_1", snapshot);
}

#[test]
fn test_asymmetric_diamond_4_1() {
    let snapshot = make_snapshot_with_options(
        graph_from_edges(7, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 6), (6, 5)]),
        SnapshotOptions {
            viewport: Rect::new(0, 0, 80, 25),
            layer_count: usize::MAX,
            node_count: usize::MAX,
            ..Default::default()
        },
    );

    insta::assert_snapshot!("asymmetric_diamond_4_1", snapshot);
}

#[test]
fn test_asymmetric_diamond_3_2() {
    let snapshot = make_snapshot_with_options(
        graph_from_edges(7, &[(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 4)]),
        SnapshotOptions {
            viewport: Rect::new(0, 0, 80, 25),
            layer_count: usize::MAX,
            node_count: usize::MAX,
            ..Default::default()
        },
    );

    insta::assert_snapshot!("asymmetric_diamond_3_2", snapshot);
}

#[test]
fn test_asymmetric_diamond_4_2() {
    let snapshot = make_snapshot_with_options(
        graph_from_edges(
            8,
            &[
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (0, 6),
                (6, 7),
                (7, 5),
            ],
        ),
        SnapshotOptions {
            viewport: Rect::new(0, 0, 80, 25),
            layer_count: usize::MAX,
            node_count: usize::MAX,
            ..Default::default()
        },
    );

    insta::assert_snapshot!("asymmetric_diamond_4_2", snapshot);
}

#[test]
fn test_diamond_variable_width_parallel_nodes() {
    #[derive(Debug, Clone)]
    struct VariableWidthSizer;

    impl NodeSizer<MockDomainGraph> for VariableWidthSizer {
        fn get_node_size(&self, node: &NodeIndex<u32>, _scale: VisualDetail) -> (u64, u64) {
            match node.index() {
                1 => (15, 3),
                2 => (10, 3),
                _ => (5, 3),
            }
        }

        fn get_dummy_size(&self) -> (u64, u64) {
            (1, 1)
        }
    }

    impl NodeSizer<&MockDomainGraph> for VariableWidthSizer {
        fn get_node_size(&self, node: &NodeIndex<u32>, scale: VisualDetail) -> (u64, u64) {
            <Self as NodeSizer<MockDomainGraph>>::get_node_size(self, node, scale)
        }

        fn get_dummy_size(&self) -> (u64, u64) {
            <Self as NodeSizer<MockDomainGraph>>::get_dummy_size(self)
        }
    }

    let snapshot = make_snapshot_with(
        graph_from_edges(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]),
        SnapshotOptions {
            viewport: Rect::new(0, 0, 80, 25),
            layer_count: usize::MAX,
            node_count: usize::MAX,
            ..Default::default()
        },
        VariableWidthSizer,
        TestRenderers::debug(),
    );

    insta::assert_snapshot!("diamond_variable_width_parallel", snapshot);
}

#[test]
fn simple_cycle() {
    let snapshot = make_snapshot(graph_from_edges(3, &[(0, 1), (1, 2), (2, 0)]));
    insta::assert_snapshot!("simple_cycle", snapshot);
}

#[test]
fn pinned_source_cycle() {
    let snapshot = make_snapshot_with_options(
        cycle_graph(12),
        SnapshotOptions {
            viewport: Rect::new(0, 0, 80, 25),
            layer_count: usize::MAX,
            node_count: usize::MAX,
            pin_source: Some(6),
            ..Default::default()
        },
    );

    insta::assert_snapshot!("pinned_source_cycle", snapshot);
}

#[test]
fn pinned_source_cycle_partitioned() {
    let snapshot = make_snapshot_with_options(
        cycle_graph(12),
        SnapshotOptions {
            viewport: Rect::new(0, 0, 80, 25),
            layer_count: 2,
            node_count: 8,
            pin_source: Some(6),
            ..Default::default()
        },
    );

    insta::assert_snapshot!("pinned_source_cycle_partitioned", snapshot);
}

#[test]
fn cycle_with_chord() {
    let mut domain_graph = cycle_graph(8);
    add_edge_by_index(&mut domain_graph, 6, 3);

    let snapshot = make_snapshot_with_options(
        domain_graph,
        SnapshotOptions {
            viewport: Rect::new(0, 0, 80, 25),
            layer_count: usize::MAX,
            node_count: usize::MAX,
            ..Default::default()
        },
    );

    insta::assert_snapshot!("cycle_with_chord", snapshot);
}

#[test]
fn cycle_with_chords() {
    let mut domain_graph = cycle_graph(8);
    add_edge_by_index(&mut domain_graph, 6, 3);
    add_edge_by_index(&mut domain_graph, 4, 1);

    let snapshot = make_snapshot_with_options(
        domain_graph,
        SnapshotOptions {
            viewport: Rect::new(0, 0, 80, 25),
            layer_count: usize::MAX,
            node_count: usize::MAX,
            ..Default::default()
        },
    );

    insta::assert_snapshot!("cycle_with_chords", snapshot);
}

#[test]
fn cycle_with_chord_pinned() {
    let mut domain_graph = cycle_graph(12);
    add_edge_by_index(&mut domain_graph, 6, 3);

    let snapshot = make_snapshot_with_options(
        domain_graph,
        SnapshotOptions {
            viewport: Rect::new(0, 0, 80, 25),
            layer_count: usize::MAX,
            node_count: usize::MAX,
            pin_source: Some(0),
            ..Default::default()
        },
    );

    insta::assert_snapshot!("cycle_with_chord_pinned", snapshot);
}

#[test]
fn cycle_with_chords_pinned() {
    let mut domain_graph = cycle_graph(12);
    add_edge_by_index(&mut domain_graph, 6, 3);
    add_edge_by_index(&mut domain_graph, 4, 1);

    let snapshot = make_snapshot_with_options(
        domain_graph,
        SnapshotOptions {
            viewport: Rect::new(0, 0, 80, 25),
            layer_count: usize::MAX,
            node_count: usize::MAX,
            pin_source: Some(0),
            ..Default::default()
        },
    );

    insta::assert_snapshot!("cycle_with_chords_pinned", snapshot);
}
#[test]
fn self_loop() {
    let snapshot = make_snapshot(graph_from_edges(1, &[(0, 0)]));
    insta::assert_snapshot!("self_loop", snapshot);
}
