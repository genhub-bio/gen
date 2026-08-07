#[cfg(test)]
use petgraph::graph::NodeIndex;
#[cfg(test)]
use ratatui::widgets::StatefulWidget as _;

#[cfg(test)]
use crate::distribute_nodes::{GapSizer, GapSizes};
#[cfg(test)]
use crate::layout::VisualDetail;
#[cfg(test)]
use crate::testing::create_test_terminal;
#[cfg(test)]
use crate::testing::mocks::{FixedNodeSizer, MockDomainGraph, TestGraphs, TestRenderers};

#[cfg(test)]
const TARGET_GAP_COMBINATIONS: [(u64, u64); 6] = [(0, 0), (1, 0), (4, 0), (0, 2), (1, 2), (4, 2)];

#[cfg(test)]
const COMPLEX_DAG_ADDITIONAL_Y_GAP_COMBINATIONS: [(u64, u64); 2] = [(1, 1), (1, 3)];

#[cfg(test)]
fn fixed_gap(gap: u64) -> GapSizer {
    match gap {
        0 => |_| 0,
        1 => |_| 1,
        2 => |_| 2,
        3 => |_| 3,
        4 => |_| 4,
        _ => panic!("unsupported test gap {gap}"),
    }
}

/// Helper function to create viewport-based visual snapshots using `LayoutEngine` + `GraphView`.
/// `backward_edges` rewrites cyclic edges onto pin nodes (see `LayoutEngine::window_for`);
/// pass an empty slice for acyclic fixtures.
#[cfg(test)]
fn make_snapshot_custom<S, R>(
    domain_graph: MockDomainGraph,
    viewport_width: u16,
    viewport_height: u16,
    node_sizer: S,
    renderer: R,
    backward_edges: &[(petgraph::graph::NodeIndex, petgraph::graph::NodeIndex)],
    target_gaps: GapSizes,
) -> String
where
    S: crate::testing::mocks::MockRenderer<MockDomainGraph>,
    R: crate::testing::mocks::MockRenderer<MockDomainGraph>,
{
    use ratatui::widgets::StatefulWidget as _;

    use crate::{
        graph_view::{GraphView, GraphViewState},
        layout_engine::LayoutEngine,
        testing::mocks::MockVisual,
    };

    let mut terminal = create_test_terminal(viewport_width, viewport_height);

    let mut engine = LayoutEngine::new_with_backward_edges(domain_graph.clone(), backward_edges);
    let mut visual = MockVisual::new(node_sizer, renderer);
    visual.detail = VisualDetail::Full;

    let mut state = GraphViewState::default();
    state.gaps = target_gaps;

    let result = terminal.draw(|f| {
        let area = f.area();
        GraphView::new(&mut engine, &visual).render(area, f.buffer_mut(), &mut state);
    });

    match result {
        Ok(_) => terminal.backend().to_string(),
        Err(e) => format!("Rendering failed: {}", e),
    }
}

/// Auto-detect backward edges by running cycle removal relative to an optional pinned
/// source/sink and collecting the edges it rewrites onto pin nodes. On an acyclic graph
/// this always returns an empty list.
#[cfg(test)]
fn detect_backward_edges(
    domain_graph: &MockDomainGraph,
    pin_source: Option<petgraph::graph::NodeIndex>,
    pin_sink: Option<petgraph::graph::NodeIndex>,
) -> Vec<(petgraph::graph::NodeIndex, petgraph::graph::NodeIndex)> {
    crate::cycle_removal::remove_cycles(domain_graph, pin_source, pin_sink)
        .backward_edges
        .into_iter()
        .collect()
}

/// Helper function to create viewport-based visual snapshots with default node sizer and renderer
#[cfg(test)]
fn make_snapshot(
    domain_graph: MockDomainGraph,
    viewport_width: u16,
    viewport_height: u16,
) -> String {
    let node_sizer = FixedNodeSizer {
        width: 5,
        height: 3,
    };
    let renderer = TestRenderers::debug();
    let backward_edges = detect_backward_edges(&domain_graph, None, None);

    make_snapshot_custom(
        domain_graph,
        viewport_width,
        viewport_height,
        node_sizer,
        renderer,
        &backward_edges,
        GapSizes::default(),
    )
}

/// Render an acyclic fixture at an explicit pair of target gaps.
#[cfg(test)]
fn make_snapshot_at_target_gaps(
    domain_graph: MockDomainGraph,
    viewport_width: u16,
    viewport_height: u16,
    target_gaps: (u64, u64),
) -> String {
    let (data_data_x, data_data_y) = target_gaps;

    make_snapshot_at_gap_sizes(
        domain_graph,
        viewport_width,
        viewport_height,
        GapSizes {
            data_data_x: fixed_gap(data_data_x),
            data_data_y: fixed_gap(data_data_y),
            ..GapSizes::default()
        },
    )
}

#[cfg(test)]
fn make_snapshot_at_gap_sizes(
    domain_graph: MockDomainGraph,
    viewport_width: u16,
    viewport_height: u16,
    target_gaps: GapSizes,
) -> String {
    let backward_edges = detect_backward_edges(&domain_graph, None, None);

    make_snapshot_custom(
        domain_graph,
        viewport_width,
        viewport_height,
        FixedNodeSizer {
            width: 5,
            height: 3,
        },
        TestRenderers::debug(),
        &backward_edges,
        target_gaps,
    )
}

/// Like `make_snapshot`, but rewrites `backward_edges` onto pin nodes, so a cyclic domain
/// graph renders as a loop instead of panicking.
#[cfg(test)]
fn make_snapshot_with_backward_edges(
    domain_graph: MockDomainGraph,
    viewport_width: u16,
    viewport_height: u16,
    backward_edges: &[(petgraph::graph::NodeIndex, petgraph::graph::NodeIndex)],
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
        node_sizer,
        renderer,
        backward_edges,
        GapSizes::default(),
    )
}

/// Render a loopback fixture at an explicit pair of target gaps.
#[cfg(test)]
fn make_snapshot_with_backward_edges_at_target_gaps(
    domain_graph: MockDomainGraph,
    viewport_width: u16,
    viewport_height: u16,
    backward_edges: &[(petgraph::graph::NodeIndex, petgraph::graph::NodeIndex)],
    target_gaps: (u64, u64),
) -> String {
    let (data_data_x, data_data_y) = target_gaps;

    make_snapshot_custom(
        domain_graph,
        viewport_width,
        viewport_height,
        FixedNodeSizer {
            width: 5,
            height: 3,
        },
        TestRenderers::debug(),
        backward_edges,
        GapSizes {
            data_data_x: fixed_gap(data_data_x),
            data_data_y: fixed_gap(data_data_y),
            ..GapSizes::default()
        },
    )
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

    let snapshot = make_snapshot(domain_graph, 80, 24);
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

    let snapshot = make_snapshot(domain_graph, 80, 24);
    insta::assert_snapshot!("diamond", snapshot);
}

#[test]
fn target_gap_combinations_diamond() {
    let domain_graph = TestGraphs::domain_diamond();

    for target_gaps @ (gap_x, gap_y) in TARGET_GAP_COMBINATIONS {
        let snapshot = make_snapshot_at_target_gaps(domain_graph.clone(), 80, 24, target_gaps);
        insta::assert_snapshot!(format!("target_gaps_diamond_x_{gap_x}_y_{gap_y}"), snapshot);
    }
}

#[test]
fn viewport_visual_regression_single_node() {
    let _ = env_logger::try_init();
    // Create a single node domain graph
    let mut domain_graph = MockDomainGraph::new();
    domain_graph.add_node(());

    let snapshot = make_snapshot(domain_graph, 80, 24);
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

    let snapshot = make_snapshot(domain_graph, 80, 24);
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

    let snapshot = make_snapshot(domain_graph, 80, 24);
    insta::assert_snapshot!("complex_dag", snapshot);
}

#[test]
fn target_gap_combinations_complex_dag() {
    let domain_graph = TestGraphs::domain_complex_dag();

    for target_gaps @ (gap_x, gap_y) in TARGET_GAP_COMBINATIONS
        .into_iter()
        .chain(COMPLEX_DAG_ADDITIONAL_Y_GAP_COMBINATIONS)
    {
        let snapshot = make_snapshot_at_target_gaps(domain_graph.clone(), 80, 24, target_gaps);
        insta::assert_snapshot!(
            format!("target_gaps_complex_dag_x_{gap_x}_y_{gap_y}"),
            snapshot
        );
    }

    let identity_y = GapSizes {
        data_data_x: |_| 1,
        data_data_y: |gap| gap,
        data_routing_y: |gap| gap,
        routing_routing_y: |gap| gap,
        ..GapSizes::default()
    };
    let snapshot = make_snapshot_at_gap_sizes(domain_graph, 80, 24, identity_y);
    insta::assert_snapshot!("target_gaps_complex_dag_x_1_y_identity", snapshot);
}

#[test]
fn viewport_wide_chain_go_to_node() {
    let _ = env_logger::try_init();
    use ratatui::widgets::StatefulWidget as _;

    use crate::{
        graph_view::{GraphView, GraphViewState},
        layout_engine::LayoutEngine,
        testing::mocks::MockVisual,
    };

    // A wide chain, long enough to require panning to reach the tail.
    let mut domain_graph = MockDomainGraph::new();
    let nodes: Vec<_> = (0..20).map(|_| domain_graph.add_node(())).collect();
    for i in 0..19 {
        domain_graph.add_edge(nodes[i], nodes[i + 1], ());
    }

    let node_sizer = FixedNodeSizer {
        width: 5,
        height: 3,
    };
    let mut engine = LayoutEngine::new(domain_graph.clone());
    let mut visual = MockVisual::new(node_sizer, TestRenderers::debug());
    visual.detail = VisualDetail::Full;
    let mut terminal = create_test_terminal(80, 24);
    let mut state = GraphViewState::default();
    // Snapshot 1: default view, anchored at the start of the chain.
    let _ = terminal.draw(|f| {
        let area = f.area();
        GraphView::new(&mut engine, &visual).render(area, f.buffer_mut(), &mut state);
    });
    let start_snapshot = terminal.backend().to_string();
    insta::assert_snapshot!("wide_chain_start", start_snapshot);

    // Snapshot 2: jump to the last node to see the tail end, rather than widening the
    // viewport to fit all 20 nodes in one shot.
    state.go_to_node(nodes[19], (0.0, 0.0));
    let _ = terminal.draw(|f| {
        let area = f.area();
        GraphView::new(&mut engine, &visual).render(area, f.buffer_mut(), &mut state);
    });
    let end_snapshot = terminal.backend().to_string();
    insta::assert_snapshot!("wide_chain_end", end_snapshot);
}

#[test]
fn viewport_long_chain_middle_window_wormholes() {
    const NODE_BUDGET: usize = 10;
    const NODE_COUNT: usize = 100;

    let domain_graph = TestGraphs::domain_long_chain(NODE_COUNT);
    let anchor = NodeIndex::new(50);
    let mut engine = crate::layout_engine::LayoutEngine::new(domain_graph);
    engine
        .activate_world_at(anchor, NODE_BUDGET, None)
        .expect("should build the middle window");
    let world = engine.active_world().expect("should have an active world");
    assert_eq!(
        world.members().count(),
        NODE_BUDGET,
        "The node budget should count domain nodes, not wormholes"
    );
    assert_eq!(
        world.layout().external_edges.len(),
        2,
        "A middle window in a linear chain should have one wormhole on each side"
    );

    let mut visual = crate::testing::mocks::MockVisual::new(
        FixedNodeSizer {
            width: 5,
            height: 3,
        },
        TestRenderers::debug(),
    );
    visual.detail = VisualDetail::Full;
    let mut state = crate::graph_view::GraphViewState::default();
    state.go_to_node(anchor, (0.5, 0.5));
    state.hide_cursor();
    let mut terminal = create_test_terminal(90, 12);
    terminal
        .draw(|frame| {
            let area = frame.area();
            crate::graph_view::GraphView::new(&mut engine, &visual).render(
                area,
                frame.buffer_mut(),
                &mut state,
            );
        })
        .expect("should render the middle window");

    assert_eq!(
        state.frame.ids().count(),
        NODE_BUDGET,
        "The rendered frame should contain exactly the budgeted domain nodes"
    );
    assert_eq!(
        state.wormhole.len(),
        2,
        "The rendered frame should contain two wormholes in addition to the domain nodes"
    );
    for &(stub, boundary, target) in &state.wormhole {
        let boundary_rect = state
            .frame
            .rect_of(boundary)
            .expect("should place each wormhole boundary node");
        if target.index() < boundary.index() {
            assert!(
                stub.right() < boundary_rect.left(),
                "A predecessor wormhole should be left of its boundary node"
            );
        } else {
            assert!(
                stub.left() > boundary_rect.right(),
                "A successor wormhole should be right of its boundary node"
            );
        }
    }
    insta::assert_snapshot!(
        "long_chain_middle_window_wormholes",
        terminal.backend().to_string()
    );
}

#[test]
fn viewport_visual_regression_extended_complex_dag() {
    let _ = env_logger::try_init();
    use crate::testing::mocks::TestGraphs;
    let domain_graph = TestGraphs::domain_complex_dag();

    let snapshot = make_snapshot(domain_graph, 80, 24);
    insta::assert_snapshot!("extended_complex_dag", snapshot);
}

#[test]
fn viewport_visual_regression_extended_diamond() {
    let _ = env_logger::try_init();
    use crate::testing::mocks::TestGraphs;
    let domain_graph = TestGraphs::domain_extended_diamond();

    let snapshot = make_snapshot(domain_graph, 80, 24);
    insta::assert_snapshot!("extended_diamond", snapshot);
}

#[test]
fn test_layer_coordinate_alignment_and_ordering() {
    let _ = env_logger::try_init();

    use crate::{
        layout::NodeRole,
        layout_engine::LayoutEngine,
        plotter::NodeRenderer,
        testing::mocks::{FixedNodeSizer, MockDomainGraph, TestGraphs},
        viewport_graph::ViewportGraph,
    };

    let domain_graph = TestGraphs::domain_extended_diamond();

    let node_sizer = FixedNodeSizer {
        width: 5,
        height: 3,
    };

    let mut engine = LayoutEngine::new(domain_graph.clone());

    let anchor = engine.default_anchor().expect("should have graph nodes");
    let assembled = engine
        .window_for(anchor, domain_graph.node_count() * 8)
        .expect("should build a window covering the graph");
    let backward_edges = assembled.backward_edges.clone();

    let geometry = {
        let visual = &node_sizer;
        crate::graph_widget::build_window_geometry(
            assembled,
            &GapSizes {
                data_data_x: |_| 1,
                data_data_y: |_| 0,
                ..GapSizes::default()
            },
            |role| match role {
                NodeRole::Data(domain_idx) => {
                    NodeRenderer::<MockDomainGraph>::get_node_size(visual, domain_idx)
                }
                _ => NodeRenderer::<MockDomainGraph>::get_dummy_size(visual),
            },
        )
    };
    let viewport_graph = ViewportGraph::from_window_geometry(&geometry, &backward_edges);
    let viewport_graph = &viewport_graph;

    assert!(viewport_graph.layer_count() > 0);

    // Group nodes by layer and collect their x-coordinates
    let mut layer_x_coords: Vec<Vec<i64>> = Vec::new();

    for layer_idx in 0..viewport_graph.layer_count() {
        if let Some(layer_nodes) = viewport_graph.get_layer(layer_idx) {
            let mut x_coords = Vec::new();

            for &domain_node in layer_nodes {
                let world_pos = viewport_graph
                    .node_positions
                    .get(&domain_node)
                    .expect("should position every layered domain node");
                x_coords.push(world_pos.x);
            }
            layer_x_coords.push(x_coords);
        }
    }

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
        }
    }

    let layer_x_representatives: Vec<i64> = layer_x_coords
        .iter()
        .filter(|coords| !coords.is_empty())
        .map(|coords| coords[0])
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

    let layer_sizes: Vec<usize> = layer_x_coords.iter().map(|coords| coords.len()).collect();
    assert!(layer_sizes.iter().any(|&size| size > 1));
}

#[test]
fn viewport_visual_regression_bridge_position_with_variable_node_widths() {
    use crate::{
        layout::VisualDetail,
        testing::mocks::{MockDomainGraph, MockRenderer, TestGraphs, TestRenderers},
    };

    let _ = env_logger::try_init();

    // Custom node sizer with dramatically different widths for middle layer
    #[derive(Debug, Clone)]
    struct VariableWidthSizer;

    impl MockRenderer<MockDomainGraph> for VariableWidthSizer {
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
        24,
        node_sizer,
        renderer,
        &[],
        GapSizes::default(),
    );

    insta::assert_snapshot!("bridge_position_variable_widths", snapshot);
}

#[test]
fn test_skip_layer() {
    let _ = env_logger::try_init();
    use crate::testing::mocks::TestGraphs;

    let domain_graph = TestGraphs::domain_skip_layer();
    let snapshot = make_snapshot(domain_graph, 80, 24);

    insta::assert_snapshot!("skip_layer", snapshot);
}

#[test]
fn viewport_chain_long_spanning_edge() {
    let _ = env_logger::try_init();
    // A chain with one edge that skips several ranks entirely.
    //
    // Regular chain edges: 0->1->2->3->4->5
    // Spanning edge: 1->4
    let mut domain_graph = MockDomainGraph::new();
    let nodes: Vec<_> = (0..6).map(|_| domain_graph.add_node(())).collect();

    for i in 0..5 {
        domain_graph.add_edge(nodes[i], nodes[i + 1], ());
    }
    domain_graph.add_edge(nodes[1], nodes[4], ());

    let snapshot = make_snapshot(domain_graph, 80, 24);

    insta::assert_snapshot!("chain_long_spanning_edge", snapshot);
}

#[test]
fn viewport_chain_very_long_spanning_edge() {
    let _ = env_logger::try_init();
    // A longer chain with one edge that skips most of the graph.
    //
    // Regular chain edges: 0->1->2->3->4->5->6->7->8->9
    // Long spanning edge: 1->8
    let mut domain_graph = MockDomainGraph::new();
    let nodes: Vec<_> = (0..10).map(|_| domain_graph.add_node(())).collect();

    for i in 0..9 {
        domain_graph.add_edge(nodes[i], nodes[i + 1], ());
    }
    domain_graph.add_edge(nodes[1], nodes[8], ());

    let snapshot = make_snapshot(domain_graph, 80, 24);

    insta::assert_snapshot!("chain_very_long_spanning_edge", snapshot);
}

#[test]
fn test_skip_layer_edges_carry_bundles() {
    let _ = env_logger::try_init();

    use crate::{
        layout::NodeRole,
        layout_engine::LayoutEngine,
        plotter::NodeRenderer,
        testing::mocks::{FixedNodeSizer, TestGraphs},
        viewport_graph::ViewportGraph,
    };

    let domain_graph = TestGraphs::domain_skip_layer();

    let node_sizer = FixedNodeSizer {
        width: 5,
        height: 3,
    };

    let mut engine = LayoutEngine::new(domain_graph.clone());

    let anchor = engine.default_anchor().expect("should have graph nodes");
    let assembled = engine
        .window_for(anchor, domain_graph.node_count() * 8)
        .expect("should build a window covering the graph");
    let backward_edges = assembled.backward_edges.clone();

    let geometry = {
        let visual = &node_sizer;
        crate::graph_widget::build_window_geometry(
            assembled,
            &GapSizes {
                data_data_x: |_| 1,
                data_data_y: |_| 0,
                ..GapSizes::default()
            },
            |role| match role {
                NodeRole::Data(domain_idx) => {
                    NodeRenderer::<MockDomainGraph>::get_node_size(visual, domain_idx)
                }
                _ => NodeRenderer::<MockDomainGraph>::get_dummy_size(visual),
            },
        )
    };
    let viewport_graph = ViewportGraph::from_window_geometry(&geometry, &backward_edges);

    // Every edge in the assembled viewport graph should carry a bundle back to the domain
    // edge(s) it represents.
    let edges_without_bundles = viewport_graph
        .graph
        .all_edges()
        .filter(|(_, _, bundle)| bundle.is_empty())
        .count();
    assert_eq!(
        edges_without_bundles, 0,
        "expected every viewport edge to carry a bundle, but found {}",
        edges_without_bundles
    );
}

#[test]
fn viewport_even_width_node_spacing() {
    let _ = env_logger::try_init();
    // Test that even-width nodes have proper spacing for edge routing
    use crate::{
        layout::VisualDetail,
        testing::mocks::{MockDomainGraph, MockRenderer, TestRenderers},
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

    impl MockRenderer<MockDomainGraph> for OddEvenSizer {
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
        43,
        node_sizer,
        renderer,
        &[],
        GapSizes::default(),
    );

    insta::assert_snapshot!("even_width_nodes", snapshot);
}

/// Test for determinism by running the same layout multiple times and comparing snapshots.
/// This test uses the complex_dag graph to ensure that HashMap/HashSet iterations produce
/// consistent results.
#[test]
fn test_layout_determinism() {
    let _ = env_logger::try_init();
    use crate::testing::mocks::TestGraphs;

    // Generate the same layout 10 times
    let num_iterations = 10;
    let mut snapshots = Vec::new();

    for i in 0..num_iterations {
        // Clone the graph for each iteration since make_snapshot takes ownership
        let graph = TestGraphs::domain_complex_dag();
        let snapshot = make_snapshot(graph, 80, 24);
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
    insta::assert_snapshot!("determinism_check_complex_dag", first);
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

    let snapshot1 = make_snapshot(graph_1, 80, 24);
    let snapshot2 = make_snapshot(graph_2, 80, 24);

    assert_eq!(
        snapshot1, snapshot2,
        "Symmetric fan layout should be identical regardless of edge insertion order"
    );
}

#[test]
fn test_double_chain() {
    let _ = env_logger::try_init();

    // Two ten-node chains share their first and last nodes.

    let mut domain_graph = MockDomainGraph::new();

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

    let snapshot = make_snapshot(domain_graph, 80, 24);

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

    let snapshot = make_snapshot(domain_graph, 80, 24);

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

    let snapshot = make_snapshot(domain_graph, 80, 24);

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

    let snapshot = make_snapshot(domain_graph, 80, 24);

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

    let snapshot = make_snapshot(domain_graph, 80, 24);

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

    let snapshot = make_snapshot(domain_graph, 80, 24);

    insta::assert_snapshot!("asymmetric_diamond_4_2", snapshot);
}

/// Test a complex multi-level DAG with dense intermediate connectivity.
///
/// Uses alphabetic single-character labels (node N → Nth letter A-Z) with 1×1 cells,
/// except N17 (4×1) and N3 (6×1) which are wider to stress horizontal spacing.
#[test]
fn test_complex_multipath_dag() {
    let _ = env_logger::try_init();

    use crate::{geometry::WorldPos, testing::mocks::MockRenderer};

    #[derive(Debug, Clone)]
    struct AlphabeticSizer;

    impl MockRenderer<MockDomainGraph> for AlphabeticSizer {
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

    impl MockRenderer<MockDomainGraph> for AlphabeticRenderer {
        fn render_node(
            &self,
            buffer: &mut crate::viewport_state::WorldBuffer,
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
        132,
        43,
        AlphabeticSizer,
        AlphabeticRenderer,
        &[],
        GapSizes::default(),
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
    let snapshot = make_snapshot(domain_graph, 80, 24);

    insta::assert_snapshot!("grid_all_to_all_with_bypass", snapshot);
}

/// Zoom (disperse/contract adjusts the minimum inter-node distance) must
/// survive dead-space compression: dispersed layouts keep their wider gaps
/// uniformly, and contracting returns to the original layout.
#[test]
fn test_grid_disperse_zoom_preserves_spacing() {
    let _ = env_logger::try_init();

    use crate::{
        layout::NodeRole, layout_engine::LayoutEngine, plotter::NodeRenderer,
        testing::mocks::FixedNodeSizer, viewport_graph::ViewportGraph,
    };

    let domain_graph = make_grid_all_to_all_with_bypass(4, 4);
    let node_sizer = FixedNodeSizer {
        width: 5,
        height: 3,
    };

    let mut engine = LayoutEngine::new(domain_graph.clone());

    // First grid layer (domain indices 1-4) plus the bypass node (index 18)
    // span all distinct rows of the layout.
    let row_node_indices = [1u32, 2, 3, 4, 18];

    let collect_row_ys = |engine: &mut LayoutEngine<MockDomainGraph>, gap_y: u64| {
        let anchor = engine.default_anchor().expect("should have graph nodes");
        // Cover the full graph so boundary doors do not affect row spacing.
        let assembled = engine
            .window_for(anchor, domain_graph.node_count() * 30)
            .expect("should build a window covering the graph");
        let backward_edges = assembled.backward_edges.clone();
        let geometry = {
            let visual = &node_sizer;
            crate::graph_widget::build_window_geometry(
                assembled,
                &GapSizes {
                    data_data_x: |_| 1,
                    data_data_y: fixed_gap(gap_y),
                    ..GapSizes::default()
                },
                |role| match role {
                    NodeRole::Data(domain_idx) => {
                        NodeRenderer::<MockDomainGraph>::get_node_size(visual, domain_idx)
                    }
                    _ => NodeRenderer::<MockDomainGraph>::get_dummy_size(visual),
                },
            )
        };
        let viewport_graph = ViewportGraph::from_window_geometry(&geometry, &backward_edges);
        let mut ys: Vec<i64> = row_node_indices
            .iter()
            .map(|&i| {
                viewport_graph
                    .node_positions
                    .get(&petgraph::graph::NodeIndex::new(i as usize))
                    .expect("should include the node in the viewport graph")
                    .y
            })
            .collect();
        ys.sort();
        ys
    };

    let uniform_gaps = |ys: &[i64]| -> Vec<i64> { ys.windows(2).map(|w| w[1] - w[0]).collect() };

    // Default gap: rows must be uniformly spaced (no dead space)
    let default_gap_y = 0;
    let initial_ys = collect_row_ys(&mut engine, default_gap_y);
    let initial_gaps = uniform_gaps(&initial_ys);
    assert!(
        initial_gaps.iter().all(|&g| g == initial_gaps[0]),
        "Rows must be uniformly spaced at default zoom, got gaps {:?}",
        initial_gaps
    );

    // Disperse: row gaps grow to height + gap_y
    let dispersed_gap_y = default_gap_y + 4;
    let dispersed_ys = collect_row_ys(&mut engine, dispersed_gap_y);
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

    // Contract back to the default gap: layout must return to the original
    let contracted_ys = collect_row_ys(&mut engine, default_gap_y);
    assert_eq!(
        contracted_ys, initial_ys,
        "Contracting back to default gap must restore the original row positions"
    );
}

/// 5 grid layers of 4 nodes: probes whether the dead space around the bypass
/// node depends on an odd number of layers.
#[test]
fn test_grid_5_layers_by_4_with_bypass() {
    let _ = env_logger::try_init();

    let domain_graph = make_grid_all_to_all_with_bypass(5, 4);
    let snapshot = make_snapshot(domain_graph, 80, 24);

    insta::assert_snapshot!("grid_5_layers_by_4_with_bypass", snapshot);
}

/// 4 grid layers of 5 nodes: probes whether the dead space around the bypass
/// node depends on an odd number of nodes per layer.
#[test]
fn test_grid_4_layers_by_5_with_bypass() {
    let _ = env_logger::try_init();

    let domain_graph = make_grid_all_to_all_with_bypass(4, 5);
    let snapshot = make_snapshot(domain_graph, 80, 24);

    insta::assert_snapshot!("grid_4_layers_by_5_with_bypass", snapshot);
}

/// Same grid-with-bypass structure, but with variable node widths where the
/// bypass node is wider than every other node.
#[test]
fn test_grid_all_to_all_with_bypass_variable_widths() {
    let _ = env_logger::try_init();

    use crate::testing::mocks::MockRenderer;

    #[derive(Debug, Clone)]
    struct WideBypassSizer;

    impl MockRenderer<MockDomainGraph> for WideBypassSizer {
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
        132,
        43,
        WideBypassSizer,
        renderer,
        &[],
        GapSizes::default(),
    );

    insta::assert_snapshot!("grid_all_to_all_with_bypass_variable_widths", snapshot);
}

/// Very wide nodes must not overflow coordinates or destabilize the camera during zoom.
#[test]
fn test_large_node_rendering_with_zoom() {
    let _ = env_logger::try_init();

    use ratatui::widgets::StatefulWidget as _;

    use crate::{
        geometry::WorldRect,
        graph_painter::Camera,
        graph_view::{GraphView, GraphViewState},
        layout::VisualDetail,
        layout_engine::LayoutEngine,
        testing::{
            create_test_terminal,
            mocks::{MockDomainGraph, MockRenderer, MockVisual},
        },
        viewport_state::WorldBuffer,
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

    impl MockRenderer<MockDomainGraph> for VariableDetailSizer {
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

    impl MockRenderer<MockDomainGraph> for UltrawideRenderer {
        fn render_node(
            &self,
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

    let viewport_width = 132;
    let viewport_height = 43;
    let mut terminal = create_test_terminal(viewport_width, viewport_height);

    let mut engine = LayoutEngine::new(domain_graph.clone());
    let mut visual = MockVisual::new(VariableDetailSizer, UltrawideRenderer);

    // 4. Starts in minimal level-of-detail
    visual.detail = VisualDetail::Minimal;
    let mut state = GraphViewState::default();
    // 5. Cursor/camera setup, anchored at the center of the viewport.
    let vp_center_x = viewport_width / 2;
    let vp_center_y = viewport_height / 2;
    state.cursor.set_node(node_0, (0.0, 0.0));
    state.show_cursor();
    let initial_anchor_screen = (vp_center_x as i64, vp_center_y as i64);
    state.camera = Some(Camera {
        anchor: node_0,
        anchor_fraction: (0.0, 0.0),
        anchor_screen: initial_anchor_screen,
        hard_zone: 2,
    });

    // 6. Snapshot 1: Minimal detail level
    let _ = terminal.draw(|f| {
        let area = f.area();
        GraphView::new(&mut engine, &visual).render(area, f.buffer_mut(), &mut state);
    });
    let minimal_snapshot = terminal.backend().to_string();
    insta::assert_snapshot!("variable_detail_chain_minimal", minimal_snapshot);

    // 7. Zoom in (goes to Truncated)
    visual.detail = VisualDetail::Truncated;

    // Snapshot 2: Truncated detail level
    let _ = terminal.draw(|f| {
        let area = f.area();
        GraphView::new(&mut engine, &visual).render(area, f.buffer_mut(), &mut state);
    });
    let truncated_snapshot = terminal.backend().to_string();
    insta::assert_snapshot!("variable_detail_chain_truncated", truncated_snapshot);

    // 8. Zoom in again (goes to Full)
    visual.detail = VisualDetail::Full;

    // Snapshot 3: Full detail level
    let _ = terminal.draw(|f| {
        let area = f.area();
        GraphView::new(&mut engine, &visual).render(area, f.buffer_mut(), &mut state);
    });
    let full_snapshot = terminal.backend().to_string();
    insta::assert_snapshot!("variable_detail_chain_full", full_snapshot);

    // 9. Confirms node 1's minimum x is to the right of node 0's maximum x

    let rect0 = state.frame.rect_of(node_0).unwrap();
    let rect1 = state.frame.rect_of(node_1).unwrap();

    assert!(
        rect1.left() > rect0.right(),
        "Node 1's minimum x ({}) should be to the right of node 0's maximum x ({})",
        rect1.left(),
        rect0.right()
    );

    // Verify that the camera's pinned anchor position remained stable throughout the zoom
    // operations - zooming must not silently perturb it.
    let final_anchor_screen = state
        .camera
        .expect("should retain the camera")
        .anchor_screen;
    assert_eq!(
        initial_anchor_screen, final_anchor_screen,
        "Camera anchor screen position should remain stable during zoom operations. Initial: {:?}, Final: {:?}",
        initial_anchor_screen, final_anchor_screen
    );
}

/// Test diamond graph with variable width nodes on parallel branches.
/// This tests the horizontal chain redistribution with nodes of different sizes.
#[test]
fn test_diamond_variable_width_parallel_nodes() {
    let _ = env_logger::try_init();

    use crate::testing::mocks::MockRenderer;

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

    impl MockRenderer<MockDomainGraph> for VariableWidthSizer {
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
        132,
        43,
        node_sizer,
        renderer,
        &[],
        GapSizes::default(),
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

    let snapshot = make_snapshot_with_backward_edges(domain_graph, 80, 24, &[(d, a)]);

    insta::assert_snapshot!("circular_genome_loop", snapshot);
}

#[test]
fn backward_edge_pin_layer_insertion() {
    let _ = env_logger::try_init();
    // Linear chain 0→1→2→3→4 with a backward edge (4, 0) closing the loop, exercising the
    // pin/dummy/data interplay in the simplest possible subgraph.
    let mut domain_graph = MockDomainGraph::new();
    let nodes: Vec<_> = (0..5).map(|_| domain_graph.add_node(())).collect();
    for i in 0..4 {
        domain_graph.add_edge(nodes[i], nodes[i + 1], ());
    }

    let snapshot =
        make_snapshot_with_backward_edges(domain_graph, 132, 43, &[(nodes[4], nodes[0])]);

    insta::assert_snapshot!("backward_edge_pin_layer_insertion", snapshot);
}

#[test]
fn backward_edge_minimal_triangle() {
    let _ = env_logger::try_init();
    // Smallest possible cycle: 3 nodes, backward edge 2 → 0.
    let mut domain_graph = MockDomainGraph::new();
    let nodes: Vec<_> = (0..3).map(|_| domain_graph.add_node(())).collect();
    domain_graph.add_edge(nodes[0], nodes[1], ());
    domain_graph.add_edge(nodes[1], nodes[2], ());

    let snapshot =
        make_snapshot_with_backward_edges(domain_graph, 132, 43, &[(nodes[2], nodes[0])]);

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

    let snapshot =
        make_snapshot_with_backward_edges(domain_graph, 132, 43, &[(nodes[5], nodes[0])]);

    insta::assert_snapshot!("backward_edge_six_node_cycle", snapshot);
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

    let snapshot =
        make_snapshot_with_backward_edges(domain_graph, 132, 43, &[(nodes[3], nodes[1])]);

    insta::assert_snapshot!("backward_edge_partial_loop", snapshot);
}

#[test]
fn target_gap_combinations_partial_loopback() {
    let mut domain_graph = MockDomainGraph::new();
    let nodes: Vec<_> = (0..5).map(|_| domain_graph.add_node(())).collect();
    for i in 0..4 {
        domain_graph.add_edge(nodes[i], nodes[i + 1], ());
    }
    let backward_edges = [(nodes[3], nodes[1])];

    for target_gaps @ (gap_x, gap_y) in TARGET_GAP_COMBINATIONS {
        let snapshot = make_snapshot_with_backward_edges_at_target_gaps(
            domain_graph.clone(),
            132,
            43,
            &backward_edges,
            target_gaps,
        );
        insta::assert_snapshot!(
            format!("target_gaps_partial_loopback_x_{gap_x}_y_{gap_y}"),
            snapshot
        );
    }
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

    let snapshot =
        make_snapshot_with_backward_edges(domain_graph, 132, 43, &[(nodes[3], nodes[0])]);

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

    let snapshot =
        make_snapshot_with_backward_edges(domain_graph, 132, 43, &[(nodes[4], nodes[0])]);

    insta::assert_snapshot!("backward_edge_branched_cycle", snapshot);
}

#[test]
fn backward_edge_local_loop() {
    let _ = env_logger::try_init();
    // Long chain 0→1→…→7 with a backward edge 5→2 that only loops over the middle. The
    // bypass should span just the 2→5 region, not the whole graph - contrast with
    // backward_edge_six_node_cycle's full-width loop.
    let mut domain_graph = MockDomainGraph::new();
    let nodes: Vec<_> = (0..8).map(|_| domain_graph.add_node(())).collect();
    for i in 0..7 {
        domain_graph.add_edge(nodes[i], nodes[i + 1], ());
    }

    let snapshot =
        make_snapshot_with_backward_edges(domain_graph, 132, 43, &[(nodes[5], nodes[2])]);

    insta::assert_snapshot!("backward_edge_local_loop", snapshot);
}

#[test]
fn backward_edge_two_independent_local_loops() {
    let _ = env_logger::try_init();
    // Two disjoint local cycles on one chain: 2→0 on the left and 7→5 on the right. Each
    // should render as its own compact bypass over its own region, rather than both
    // stacking as full-width lines competing across the whole canvas.
    let mut domain_graph = MockDomainGraph::new();
    let nodes: Vec<_> = (0..8).map(|_| domain_graph.add_node(())).collect();
    for i in 0..7 {
        domain_graph.add_edge(nodes[i], nodes[i + 1], ());
    }

    let snapshot = make_snapshot_with_backward_edges(
        domain_graph,
        132,
        43,
        &[(nodes[2], nodes[0]), (nodes[7], nodes[5])],
    );

    insta::assert_snapshot!("backward_edge_two_independent_local_loops", snapshot);
}

// Cycle auto-detection tests. Unlike the `backward_edge_*` tests above, these hand a raw
// cyclic graph to the auto-detecting `make_snapshot`/`make_snapshot_pinned` path and let
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
/// breaks each cycle relative to that node.
#[cfg(test)]
fn make_snapshot_pinned(
    domain_graph: MockDomainGraph,
    viewport_width: u16,
    viewport_height: u16,
    pin_source: usize,
) -> String {
    use ratatui::widgets::StatefulWidget as _;

    use crate::{
        graph_view::{GraphView, GraphViewState},
        layout_engine::LayoutEngine,
        testing::mocks::MockVisual,
    };

    let node_sizer = FixedNodeSizer {
        width: 5,
        height: 3,
    };
    let mut visual = MockVisual::new(node_sizer, TestRenderers::debug());
    visual.detail = VisualDetail::Full;

    let mut terminal = create_test_terminal(viewport_width, viewport_height);

    let pin_source = Some(petgraph::graph::NodeIndex::new(pin_source));
    let backward_edges = detect_backward_edges(&domain_graph, pin_source, None);

    let mut engine = LayoutEngine::new_with_backward_edges(domain_graph.clone(), &backward_edges);

    let mut state = GraphViewState::default();
    let result = terminal.draw(|f| {
        let area = f.area();
        GraphView::new(&mut engine, &visual).render(area, f.buffer_mut(), &mut state);
    });

    match result {
        Ok(_) => terminal.backend().to_string(),
        Err(e) => format!("Rendering failed: {}", e),
    }
}

#[test]
fn cycle_simple_autodetected() {
    let _ = env_logger::try_init();
    // A bare 3-cycle with no explicit backward edge: detection finds the loop-closing edge.
    let snapshot = make_snapshot(graph_from_edges(3, &[(0, 1), (1, 2), (2, 0)]), 132, 43);
    insta::assert_snapshot!("cycle_simple_autodetected", snapshot);
}

#[test]
fn cycle_self_loop_autodetected() {
    let _ = env_logger::try_init();
    // A single node with a self-loop (0 -> 0): the pins land right next to the node and the
    // loop stays entirely local. Self-loops were a non-goal of the pin work but fall out of
    // the same window once detection reports them.
    let snapshot = make_snapshot(graph_from_edges(1, &[(0, 0)]), 80, 24);
    insta::assert_snapshot!("cycle_self_loop_autodetected", snapshot);
}

#[test]
fn cycle_with_chord_autodetected() {
    let _ = env_logger::try_init();
    // An 8-cycle plus a chord (6 -> 3): two backward edges, each scoped to its own span.
    let mut domain_graph = cycle_graph(8);
    add_edge_by_index(&mut domain_graph, 6, 3);
    let snapshot = make_snapshot(domain_graph, 80, 24);
    insta::assert_snapshot!("cycle_with_chord_autodetected", snapshot);
}

#[test]
fn cycle_with_two_chords_autodetected() {
    let _ = env_logger::try_init();
    // An 8-cycle plus two chords (6 -> 3, 4 -> 1): three independent backward edges.
    let mut domain_graph = cycle_graph(8);
    add_edge_by_index(&mut domain_graph, 6, 3);
    add_edge_by_index(&mut domain_graph, 4, 1);
    let snapshot = make_snapshot(domain_graph, 80, 24);
    insta::assert_snapshot!("cycle_with_two_chords_autodetected", snapshot);
}

#[test]
fn cycle_pinned_source() {
    let _ = env_logger::try_init();
    // A 12-cycle with node 6 pinned as the source: detection breaks the cycle relative to
    // node 6 rather than at petgraph's default entry point.
    let snapshot = make_snapshot_pinned(cycle_graph(12), 80, 24, 6);
    insta::assert_snapshot!("cycle_pinned_source", snapshot);
}
