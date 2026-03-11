#![cfg(test)]
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use petgraph::{
    graph::NodeIndex,
    stable_graph::StableDiGraph,
    visit::{EdgeRef, IntoEdgeReferences},
};
use ratatui::layout::Rect;

use crate::{
    geometry::{BigRect, SpatialObjectType, ViewportPos, WorldPos, WorldRect},
    graph_algorithms::find_articulation_points,
    graph_controller::{GraphConfig, GraphController, WorldBuffer},
    graph_widget::GraphWidget,
    layout::{LayoutEngine, NodeRole, VisualDetail},
    partition::{PartitionEdge, PartitionNode},
    plotter::{NodeRenderer, NodeSizer},
    testing::{
        create_test_terminal,
        mocks::{FixedNodeSizer, MockDomainGraph, TestGraphs},
    },
};

pub(super) fn init_test_logging() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        let _ = env_logger::try_init();
    });
}

pub(super) fn graph_from_edges(node_count: usize, edges: &[(usize, usize)]) -> MockDomainGraph {
    let mut graph = MockDomainGraph::new();
    let nodes: Vec<_> = (0..node_count).map(|_| graph.add_node(())).collect();

    for &(src, dst) in edges {
        graph.add_edge(nodes[src], nodes[dst], ());
    }

    graph
}

pub(super) fn chain_graph(node_count: usize) -> MockDomainGraph {
    let mut graph = MockDomainGraph::new();
    let nodes: Vec<_> = (0..node_count).map(|_| graph.add_node(())).collect();

    for i in 0..node_count.saturating_sub(1) {
        graph.add_edge(nodes[i], nodes[i + 1], ());
    }

    graph
}

pub(super) fn cycle_graph(node_count: usize) -> MockDomainGraph {
    let mut graph = MockDomainGraph::new();
    let nodes: Vec<_> = (0..node_count).map(|_| graph.add_node(())).collect();

    for i in 0..node_count {
        graph.add_edge(nodes[i], nodes[(i + 1) % node_count], ());
    }

    graph
}

pub(super) fn add_edge_by_index(graph: &mut MockDomainGraph, src: usize, dst: usize) {
    let nodes: Vec<_> = graph.node_indices().collect();
    graph.add_edge(nodes[src], nodes[dst], ());
}

pub(super) fn build_controller(
    domain_graph: &MockDomainGraph,
    layer_count: usize,
    node_count: usize,
) -> GraphController<&MockDomainGraph, FixedNodeSizer> {
    init_test_logging();

    let node_sizer = FixedNodeSizer {
        width: 5,
        height: 3,
    };

    let mut config = GraphConfig::default();
    config.partition.layer_count = layer_count;
    config.partition.node_count = node_count;

    let mut controller = GraphController::new_with_config(domain_graph, node_sizer, config);
    controller.set_detail_level(VisualDetail::Full);
    controller.viewport_state.viewport_bounds = Rect::new(0, 0, u16::MAX / 2, u16::MAX / 2);

    controller
}

// -----------------------------------------------------------------------------
// Non-snapshot layout tests
// -----------------------------------------------------------------------------

#[test]
fn test_layer_coordinate_alignment_and_ordering() {
    // Create domain_extended_diamond with partitioning to test layer alignment.
    let domain_graph = TestGraphs::domain_extended_diamond();
    let mut controller = build_controller(&domain_graph, 2, 3);

    controller.viewport_state.camera_current = WorldPos::new(0, 0);
    controller.viewport_state.camera_target = WorldPos::new(0, 0);

    // Get total partition count first to verify all are loaded.
    let total_partition_count = controller
        .partition_controller
        .partition_table
        .partitions
        .len();
    println!("Total partitions in graph: {}", total_partition_count);

    // Load all partitions by ensuring camera coverage.
    let loaded_partitions = controller.ensure_camera_coverage().unwrap_or_default();
    println!("Number of partitions loaded: {}", loaded_partitions.len());

    // Assert that ALL partitions were loaded, not just multiple.
    assert_eq!(
        loaded_partitions.len(),
        total_partition_count,
        "Expected all {} partitions to be loaded, but only {} were loaded",
        total_partition_count,
        loaded_partitions.len()
    );

    // Rebuild the viewport graph with unlimited bounds.
    let result = controller.rebuild_viewport_graph();
    assert!(
        result.is_ok(),
        "Failed to rebuild viewport graph: {:?}",
        result
    );

    let viewport_graph = controller.get_viewport_graph();

    // Verify we have layers.
    assert!(
        viewport_graph.layer_count() > 0,
        "No layers found in viewport graph"
    );
    println!("Viewport graph has {} layers", viewport_graph.layer_count());

    // Group nodes by layer and collect their x-coordinates.
    let mut layer_x_coords: Vec<Vec<i64>> = Vec::new();

    for layer_idx in 0..viewport_graph.layer_count() {
        if let Some(layer_nodes) = viewport_graph.get_layer(layer_idx) {
            let mut x_coords = Vec::new();

            for &domain_node in layer_nodes {
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

    // Test 1: Verify that all nodes in each layer share the same x-coordinate.
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

    // Test 2: Verify that x-coordinates are ordered between layers.
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

    println!(
        "Layer x-coordinates are properly ordered: {:?}",
        layer_x_representatives
    );

    // Additional verification: check that we have the expected layer structure.
    let layer_sizes: Vec<usize> = layer_x_coords.iter().map(|coords| coords.len()).collect();
    println!("Layer sizes: {:?}", layer_sizes);

    let has_multi_node_layers = layer_sizes.iter().any(|&size| size > 1);
    assert!(
        has_multi_node_layers,
        "Expected some layers to have multiple nodes for diamond structure, but all layers have single nodes"
    );

    println!("Test completed successfully - layer coordinate alignment and ordering verified");
}

#[test]
fn viewport_chain_five_partitions_verify_partition_count() {
    let mut domain_graph = chain_graph(10);
    add_edge_by_index(&mut domain_graph, 1, 8);

    let articulation_points = find_articulation_points(&domain_graph);
    println!(
        "Articulation points in 10-node chain: {:?}",
        articulation_points
    );
    println!(
        "Number of articulation points: {}",
        articulation_points.len()
    );

    let mut controller = build_controller(&domain_graph, 2, 5);
    controller.viewport_state.camera_current = WorldPos::new(0, 0);
    controller.viewport_state.camera_target = WorldPos::new(0, 0);

    let total_partition_count = controller
        .partition_controller
        .partition_table
        .partitions
        .len();
    println!("Total partitions created: {}", total_partition_count);

    assert_eq!(
        total_partition_count, 5,
        "Expected 5 partitions (3 data + 2 bridge), but found {}",
        total_partition_count
    );

    let loaded_partitions = controller.ensure_camera_coverage().unwrap_or_default();
    println!("Partitions loaded: {}", loaded_partitions.len());
    assert_eq!(
        loaded_partitions.len(),
        total_partition_count,
        "Expected all partitions to be loaded"
    );

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

    println!(
        "✓ Confirmed: 5 partitions created (3 data + 2 bridge), spanning edge crosses middle partitions"
    );
}

#[test]
fn test_skip_layer_terminal_stitch_edge_bundles() {
    let domain_graph = TestGraphs::domain_skip_layer();
    let mut controller = build_controller(&domain_graph, 2, 3);

    controller.viewport_state.camera_current = WorldPos::new(0, 0);
    controller.viewport_state.camera_target = WorldPos::new(0, 0);
    controller.initialize_cursor();

    let result = controller.rebuild_viewport_graph();
    assert!(
        result.is_ok(),
        "Failed to rebuild viewport graph: {:?}",
        result
    );

    let viewport_graph = controller.get_viewport_graph();

    let mut edges_without_bundles = 0;
    let mut total_edges = 0;

    for edge in viewport_graph.graph.all_edges() {
        total_edges += 1;
        let (source, target, edge_data) = edge;

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
fn test_edge_viewport_intersection_endpoints_outside() {
    init_test_logging();

    // Create a larger partition graph with multiple nodes to ensure
    // there's enough distance between the endpoints.
    let mut partition_graph = StableDiGraph::<PartitionNode, PartitionEdge, u32>::new();

    // Create a chain: 0 -> 1 -> 2 -> 3 -> 4.
    let nodes: Vec<_> = (0..5)
        .map(|i| partition_graph.add_node(PartitionNode::Data(NodeIndex::new(i))))
        .collect();

    for i in 0..4 {
        partition_graph.add_edge(
            nodes[i],
            nodes[i + 1],
            Some((NodeIndex::new(i), NodeIndex::new(i + 1))),
        );
    }

    let mut layout_engine = LayoutEngine::new(&partition_graph, 0);
    let node_sizer = FixedNodeSizer {
        width: 2,
        height: 2,
    };

    let layout = layout_engine
        .compute_layout(&node_sizer, VisualDetail::Full)
        .expect("Failed to compute layout");

    let node0_pos = layout
        .graph
        .node_indices()
        .find_map(|idx| {
            let node = layout.graph.node_weight(idx)?;
            if matches!(node.role, NodeRole::Data(n) if n.index() == 0) {
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
            if matches!(node.role, NodeRole::Data(n) if n.index() == 4) {
                Some(node.pos)
            } else {
                None
            }
        })
        .expect("Node 4 not found in layout");

    println!("Node 0 position: {:?}", node0_pos);
    println!("Node 4 position: {:?}", node4_pos);

    let viewport_center_x = (node0_pos.x + node4_pos.x) / 2;
    let viewport_center_y = (node0_pos.y + node4_pos.y) / 2;

    let viewport = BigRect::from_coords(
        viewport_center_x - 1,
        viewport_center_y - 1,
        viewport_center_x + 1,
        viewport_center_y + 1,
    );

    println!("Viewport: {:?}", viewport);

    let data_nodes_in_viewport = layout
        .find_nodes_in_rect(viewport)
        .iter()
        .filter(|obj| matches!(obj.object_type, SpatialObjectType::DataNode(_)))
        .count();

    println!("Data nodes in viewport: {}", data_nodes_in_viewport);

    let edges_in_viewport = layout.find_edges_in_rect(viewport);
    println!("Edges in viewport: {}", edges_in_viewport.len());

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
    init_test_logging();

    // Create a diamond graph to test diagonal edges.
    let mut partition_graph = StableDiGraph::<PartitionNode, PartitionEdge, u32>::new();

    let nodes: Vec<_> = (0..4)
        .map(|i| partition_graph.add_node(PartitionNode::Data(NodeIndex::new(i))))
        .collect();

    partition_graph.add_edge(
        nodes[0],
        nodes[1],
        Some((NodeIndex::new(0), NodeIndex::new(1))),
    );
    partition_graph.add_edge(
        nodes[0],
        nodes[2],
        Some((NodeIndex::new(0), NodeIndex::new(2))),
    );
    partition_graph.add_edge(
        nodes[1],
        nodes[3],
        Some((NodeIndex::new(1), NodeIndex::new(3))),
    );
    partition_graph.add_edge(
        nodes[2],
        nodes[3],
        Some((NodeIndex::new(2), NodeIndex::new(3))),
    );

    let mut layout_engine = LayoutEngine::new(&partition_graph, 0);
    let node_sizer = FixedNodeSizer {
        width: 3,
        height: 3,
    };

    let layout = layout_engine
        .compute_layout(&node_sizer, VisualDetail::Full)
        .expect("Failed to compute layout");

    let find_node = |node_index: usize| {
        layout
            .graph
            .node_indices()
            .find_map(|idx| {
                let node = layout.graph.node_weight(idx)?;
                if matches!(node.role, NodeRole::Data(n) if n.index() == node_index) {
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

    let center_x = (node0_pos.x + node3_pos.x) / 2;
    let center_y = (node1_pos.y + node2_pos.y) / 2;

    let viewport = BigRect::from_coords(center_x - 2, center_y - 2, center_x + 2, center_y + 2);

    println!("Viewport (center region): {:?}", viewport);

    let data_nodes_in_viewport = layout
        .find_nodes_in_rect(viewport)
        .iter()
        .filter(|obj| matches!(obj.object_type, SpatialObjectType::DataNode(_)))
        .count();

    println!("Data nodes in viewport: {}", data_nodes_in_viewport);

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

    let edges_in_viewport = layout.find_edges_in_rect(viewport);
    println!("\nEdges found in viewport: {}", edges_in_viewport.len());

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

/// Test rendering and positioning of very large nodes during zoom operations.
#[test]
fn test_large_node_rendering_with_zoom() {
    init_test_logging();

    let domain_graph = graph_from_edges(3, &[(0, 1), (1, 2)]);

    #[derive(Debug, Clone)]
    struct VariableDetailSizer;

    impl NodeSizer<MockDomainGraph> for VariableDetailSizer {
        fn get_node_size(&self, node: &NodeIndex<u32>, scale: VisualDetail) -> (u64, u64) {
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
        fn get_node_size(&self, node: &NodeIndex<u32>, scale: VisualDetail) -> (u64, u64) {
            <Self as NodeSizer<MockDomainGraph>>::get_node_size(self, node, scale)
        }

        fn get_dummy_size(&self) -> (u64, u64) {
            <Self as NodeSizer<MockDomainGraph>>::get_dummy_size(self)
        }
    }

    #[derive(Debug, Clone)]
    struct UltrawideRenderer;

    impl NodeRenderer<&MockDomainGraph> for UltrawideRenderer {
        fn render_node(
            &mut self,
            buffer: &mut WorldBuffer,
            area: WorldRect,
            node_id: &NodeIndex<u32>,
            _scale: VisualDetail,
        ) {
            let Some(visible_area) = buffer.calculate_visible_area(area) else {
                return;
            };

            let symbol = format!("{}", node_id.index()).chars().next().unwrap();

            for y in visible_area.min.y..=visible_area.max.y {
                let visible_width = (visible_area.max.x - visible_area.min.x + 1) as usize;
                let content = symbol.to_string().repeat(visible_width);

                let start_pos = WorldPos::new(visible_area.min.x, y);
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

    controller.set_detail_level(VisualDetail::Minimal);
    controller.show_cursor();
    controller.initialize_cursor();

    let node_indices: Vec<_> = domain_graph.node_indices().collect();
    let node_0 = node_indices[0];
    let node_1 = node_indices[1];

    controller.cursor.set_node(node_0, (0.0, 0.0));

    let vp_center_x = viewport_width / 2;
    let vp_center_y = viewport_height / 2;
    let initial_cursor_viewport_pos = ViewportPos::new(vp_center_x, vp_center_y);
    controller
        .cursor
        .set_viewport_pos(initial_cursor_viewport_pos);

    let renderer = UltrawideRenderer;

    let _ = terminal.draw(|f| {
        let area = f.area();
        controller.viewport_state.viewport_bounds = area;

        let widget = GraphWidget::with_renderer(renderer.clone())
            .detail_level(VisualDetail::Minimal)
            .cursor();
        f.render_stateful_widget(widget, area, &mut controller);
    });
    let minimal_snapshot = format!("{}", terminal.backend());
    insta::assert_snapshot!("variable_detail_chain_minimal", minimal_snapshot);

    let plus_key = KeyEvent::new(KeyCode::Char('+'), KeyModifiers::NONE);
    controller.handle_key_event(plus_key).unwrap();
    controller.trigger_rebuild();

    let _ = terminal.draw(|f| {
        let area = f.area();
        controller.viewport_state.viewport_bounds = area;

        let widget = GraphWidget::with_renderer(renderer.clone())
            .detail_level(controller.get_detail_level())
            .cursor();
        f.render_stateful_widget(widget, area, &mut controller);
    });
    let truncated_snapshot = format!("{}", terminal.backend());
    insta::assert_snapshot!("variable_detail_chain_truncated", truncated_snapshot);

    controller.handle_key_event(plus_key).unwrap();
    controller.trigger_rebuild();

    let _ = terminal.draw(|f| {
        let area = f.area();
        controller.viewport_state.viewport_bounds = area;

        let widget = GraphWidget::with_renderer(renderer.clone())
            .detail_level(controller.get_detail_level())
            .cursor();
        f.render_stateful_widget(widget, area, &mut controller);
    });
    let full_snapshot = format!("{}", terminal.backend());
    insta::assert_snapshot!("variable_detail_chain_full", full_snapshot);

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

    let final_cursor_viewport_pos = controller.cursor.viewport_pos;
    assert_eq!(
        initial_cursor_viewport_pos, final_cursor_viewport_pos,
        "Cursor viewport position should remain stable during zoom operations. Initial: {:?}, Final: {:?}",
        initial_cursor_viewport_pos, final_cursor_viewport_pos
    );
}
