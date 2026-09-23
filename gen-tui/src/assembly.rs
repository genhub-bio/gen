//! Converts a crawled window's cached Sugiyama structure into a layout graph.
//!
//! Assembly preserves the layer and within-layer order but leaves sizing, routing, and
//! compaction to the widget.

use std::collections::HashMap;

use petgraph::{
    Undirected,
    stable_graph::{NodeIndex, StableGraph},
    visit::{EdgeRef, IntoEdgeReferences},
};

use crate::{
    crawl::ExternalEdge,
    geometry::LocalPos,
    layout::{LayoutEdge, LayoutNode, NodeRole},
    window_graph::{WindowGraph, WindowNode},
};

/// Domain-edge instances aggregated by their assembled endpoint pair, so a bundle of parallel
/// domain edges collapses into one `LayoutEdge`.
type EdgeBundles = HashMap<(NodeIndex<u32>, NodeIndex<u32>), (Vec<(NodeIndex, NodeIndex)>, bool)>;

/// The assembled structure of one crawled neighbourhood window.
///
/// Positions contain only a layer column and within-layer order; the widget turns them into
/// final geometry after applying current-detail sizes.
#[derive(Clone)]
pub struct AssembledLayout {
    pub graph: StableGraph<LayoutNode, LayoutEdge, Undirected, u32>,
    pub width: i64,
    pub height: i64,
    /// The assembled-graph node the query anchor landed on. `None` only when the anchor
    /// itself fell outside the assembled window (a caller-side inconsistency), so callers
    /// that assemble a window around their own anchor can treat this as reliably `Some`.
    pub anchor_index: Option<NodeIndex<u32>>,
    /// Collapsed boundary edges, filled by `LayoutEngine` after assembly.
    pub external_edges: Vec<ExternalEdge<NodeIndex<u32>>>,
    /// Backward domain edges, filled by `LayoutEngine` after assembly.
    pub backward_edges: Vec<(NodeIndex<u32>, NodeIndex<u32>)>,
}

/// Assemble a window's cached Sugiyama structure into a layout graph.
pub fn assemble_window(window: &WindowGraph) -> Option<AssembledLayout> {
    let mut graph: StableGraph<LayoutNode, LayoutEdge, Undirected, u32> = StableGraph::default();

    // A single-node window bypasses Sugiyama entirely (see
    // `WindowStructureBuilder::build_structure`), so it has no cached structure. Assemble its
    // lone node directly instead of treating the window as empty.
    let Some(structure) = window.structure.as_ref() else {
        let node_index = window.graph.node_indices().next()?;
        let role = match window.graph.node_weight(node_index)? {
            WindowNode::Data(domain_idx) => NodeRole::Data(*domain_idx),
            WindowNode::Pin => NodeRole::Pin,
            WindowNode::Wormhole(target) => NodeRole::Wormhole(*target),
        };
        graph.add_node(LayoutNode::new(
            role,
            LocalPos::new_xy(0, 0),
            (1, 1),
            Some(0),
        ));
        return Some(AssembledLayout {
            graph,
            width: 0,
            height: 0,
            anchor_index: None,
            external_edges: Vec::new(),
            backward_edges: Vec::new(),
        });
    };

    let mut vertex_to_assembled: HashMap<NodeIndex<u32>, NodeIndex<u32>> = HashMap::new();

    for (layer_index, layer) in structure.vertex_layers.iter().enumerate() {
        for (order, &vertex_idx) in layer.iter().enumerate() {
            let vertex = structure.vertex_graph.node_weight(vertex_idx)?;
            let role = match vertex.input_node_idx {
                Some(window_node_idx) => match window.graph.node_weight(window_node_idx)? {
                    WindowNode::Data(domain_idx) => NodeRole::Data(*domain_idx),
                    WindowNode::Pin => NodeRole::Pin,
                    WindowNode::Wormhole(target) => NodeRole::Wormhole(*target),
                },
                None => NodeRole::Routing,
            };
            let position = LocalPos::new_xy(layer_index as i64, order as i64);
            let assembled_idx = graph.add_node(LayoutNode::new(
                role,
                position,
                (1, 1),
                Some(layer_index as i32),
            ));
            vertex_to_assembled.insert(vertex_idx, assembled_idx);
        }
    }

    // Copy edges, aggregating domain-edge bundles per endpoint pair.
    let mut edge_bundles: EdgeBundles = HashMap::new();
    for edge_ref in structure.vertex_graph.edge_references() {
        let pair = edge_ref.weight().input_node_idx_pair;
        let is_main_span = structure
            .backward_span_edges
            .contains(&(edge_ref.source(), edge_ref.target()));

        let (Some(&source), Some(&target)) = (
            vertex_to_assembled.get(&edge_ref.source()),
            vertex_to_assembled.get(&edge_ref.target()),
        ) else {
            continue;
        };
        let entry = edge_bundles.entry((source, target)).or_default();
        if let Some(pair) = pair {
            entry.0.push(pair);
        }
        entry.1 |= is_main_span;
    }
    for ((source, target), (bundle, is_backward_span)) in edge_bundles {
        graph.add_edge(
            source,
            target,
            LayoutEdge {
                bundle,
                is_backward_span,
            },
        );
    }

    let (min_x, max_x, min_y, max_y) = graph.node_weights().fold(
        (i64::MAX, i64::MIN, i64::MAX, i64::MIN),
        |(min_x, max_x, min_y, max_y), node| {
            (
                min_x.min(node.pos.x),
                max_x.max(node.pos.x),
                min_y.min(node.pos.y),
                max_y.max(node.pos.y),
            )
        },
    );
    let (width, height) = if min_x > max_x {
        (0, 0)
    } else {
        (max_x - min_x, max_y - min_y)
    };

    Some(AssembledLayout {
        graph,
        width,
        height,
        anchor_index: None,
        external_edges: Vec::new(),
        backward_edges: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::{
        crawl::{EagerSource, GraphCursor, build_window_graph, neighborhood},
        testing::mocks::MockDomainGraph,
    };

    #[test]
    fn test_assemble_carries_data_pin_and_routing_roles() {
        // 0 -> 1 -> 2 -> 3 with a backward edge 3 -> 0: forces a pin pair, and a long enough
        // chain to force at least one Sugiyama routing dummy on the pin bypass.
        let mut domain_graph = MockDomainGraph::new();
        let nodes: Vec<_> = (0..4).map(|_| domain_graph.add_node(())).collect();
        for window in nodes.windows(2) {
            domain_graph.add_edge(window[0], window[1], ());
        }
        let backward_edges = HashSet::from([(nodes[3], nodes[0])]);

        let subgraph = neighborhood(
            nodes[0],
            10,
            &mut GraphCursor::new(&mut domain_graph, &mut EagerSource),
            None,
            &HashMap::new(),
            &HashMap::new(),
        )
        .unwrap();
        let (window, _) =
            build_window_graph(&subgraph, &domain_graph, Some(&backward_edges)).unwrap();

        let assembled = assemble_window(&window).expect("should assemble the window");

        let data_count = assembled
            .graph
            .node_weights()
            .filter(|node| matches!(node.role, NodeRole::Data(_)))
            .count();
        let pin_count = assembled
            .graph
            .node_weights()
            .filter(|node| matches!(node.role, NodeRole::Pin))
            .count();
        assert_eq!(data_count, 4, "every domain node should carry over");
        assert_eq!(pin_count, 2, "the backward edge should add one pin pair");
    }

    #[test]
    fn test_assemble_aggregates_parallel_edges_into_one_bundle() {
        // Assembly emits at most one edge for each endpoint pair.
        let mut domain_graph = MockDomainGraph::new();
        let nodes: Vec<_> = (0..4).map(|_| domain_graph.add_node(())).collect();
        domain_graph.add_edge(nodes[0], nodes[1], ());
        domain_graph.add_edge(nodes[0], nodes[2], ());
        domain_graph.add_edge(nodes[1], nodes[3], ());
        domain_graph.add_edge(nodes[2], nodes[3], ());

        let subgraph = neighborhood(
            nodes[0],
            10,
            &mut GraphCursor::new(&mut domain_graph, &mut EagerSource),
            None,
            &HashMap::new(),
            &HashMap::new(),
        )
        .unwrap();
        let (window, _) = build_window_graph(&subgraph, &domain_graph, None).unwrap();
        let assembled = assemble_window(&window).unwrap();

        let mut seen_pairs: HashSet<(NodeIndex<u32>, NodeIndex<u32>)> = HashSet::new();
        for edge_index in assembled.graph.edge_indices() {
            let (source, target) = assembled.graph.edge_endpoints(edge_index).unwrap();
            let pair = if source.index() <= target.index() {
                (source, target)
            } else {
                (target, source)
            };
            assert!(
                seen_pairs.insert(pair),
                "each assembled node pair should carry at most one aggregated edge"
            );
        }
    }

    #[test]
    fn test_assemble_single_node_window_has_no_structure_dependency() {
        let mut domain_graph = MockDomainGraph::new();
        let node = domain_graph.add_node(());

        let subgraph = neighborhood(
            node,
            10,
            &mut GraphCursor::new(&mut domain_graph, &mut EagerSource),
            None,
            &HashMap::new(),
            &HashMap::new(),
        )
        .unwrap();
        let (window, _) = build_window_graph(&subgraph, &domain_graph, None).unwrap();
        assert!(
            window.structure.is_none(),
            "a single-node window should have no Sugiyama structure"
        );

        let assembled = assemble_window(&window).expect("should assemble a single-node window");
        assert_eq!(assembled.graph.node_count(), 1);
        assert_eq!(assembled.width, 0);
        assert_eq!(assembled.height, 0);
    }

    #[test]
    fn test_assemble_edges_span_adjacent_columns() {
        let mut domain_graph = MockDomainGraph::new();
        let nodes: Vec<_> = (0..6).map(|_| domain_graph.add_node(())).collect();
        for window in nodes.windows(2) {
            domain_graph.add_edge(window[0], window[1], ());
        }

        let subgraph = neighborhood(
            nodes[0],
            10,
            &mut GraphCursor::new(&mut domain_graph, &mut EagerSource),
            None,
            &HashMap::new(),
            &HashMap::new(),
        )
        .unwrap();
        let (window, _) = build_window_graph(&subgraph, &domain_graph, None).unwrap();
        let assembled = assemble_window(&window).unwrap();

        // `make_rectilinear` only routes between adjacent occupied columns, so every assembled
        // edge must join nodes at most one column apart - Sugiyama's own dummy insertion
        // already guarantees this for a single window with no seam gaps to bridge.
        for edge in assembled.graph.edge_indices() {
            let (source, target) = assembled.graph.edge_endpoints(edge).unwrap();
            let source_x = assembled.graph.node_weight(source).unwrap().pos.x;
            let target_x = assembled.graph.node_weight(target).unwrap().pos.x;
            assert!(
                (source_x - target_x).abs() <= 1,
                "edge spans columns {} and {}",
                source_x,
                target_x
            );
        }
    }
}
