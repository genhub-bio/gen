use gen_core::{GraphNodePosition, HashId, NodeIntervalBlock};
use gen_graph::{GenGraph, GraphError, graph_loader};
use intervaltree::IntervalTree;

use crate::{db::GraphConnection, edge::Edge, node::Node};

pub struct ResolvedGraph {
    pub graph: GenGraph,
    pub interval_tree: IntervalTree<i64, NodeIntervalBlock>,
    pub block_group_id: HashId,
}

/// Identify all edges leading to and from a provided node_id in a block_group and merge them into an existing GenGraph.
/// Returns true if the graph was expanded, false if no new edges were added.
pub fn expand(
    conn: &GraphConnection,
    graph: &mut GenGraph,
    block_group_id: &HashId,
    node_id: HashId,
) -> bool {
    let candidate_edges =
        match Edge::edges_for_block_group_node_neighborhood(conn, block_group_id, node_id, None) {
            Ok(edges) => edges,
            Err(_) => return false,
        };
    let unloaded_edge_ids =
        graph_loader::unloaded_edge_ids(graph, candidate_edges.iter().map(|edge| edge.edge.id));
    let unloaded_edges = candidate_edges
        .into_iter()
        .filter(|edge| unloaded_edge_ids.contains(&edge.edge.id))
        .collect::<Vec<_>>();
    if unloaded_edges.is_empty() {
        return false;
    }
    let blocks = match Edge::blocks_from_edges(conn, block_group_id, &unloaded_edges, None) {
        Ok(blocks) => blocks,
        Err(_) => return false,
    };
    let (fragment, _) = Edge::build_graph(&unloaded_edges, &blocks);
    graph_loader::merge_fragment(graph, &fragment);
    true
}

/// From a given position in a graph, find positions a given number of characters away. An
/// expand function can be given that specifies how to expand the input graph when it is exhausted.
/// By default, this function searches along the known graph, and if a matching position cannot be found
/// it will expand until it is not possible to grow the graph anymore.
/// Returns a Result<Vec<GraphNodePosition>>, given a list of matching GraphNodePositions at the requested distance.
pub fn find_offset(
    graph: &mut GenGraph,
    // The position to begin the search from
    anchor: &GraphNodePosition,
    // How far from the anchor the position of interst is. negative means search upstream of the graph position.
    // positive means search downstream.
    distance: i64,
    mut expand: impl FnMut(&mut GenGraph, HashId) -> bool,
) -> Result<Vec<GraphNodePosition>, GraphError> {
    gen_graph::graph_loader::find_offset(graph, anchor, distance, &mut expand)
}

impl ResolvedGraph {
    /// Find a postion in a graph according to a provided coordinate. This is generally utilized by the GenRegion machinery to identify the
    /// positions a user requested.
    pub fn resolve_anchor(
        &self,
        coord: i64,
        conn: &GraphConnection,
    ) -> Result<GraphNodePosition, GraphError> {
        gen_graph::graph_loader::resolve_anchor(
            &self.graph,
            &self.interval_tree,
            coord,
            |node_id| {
                Node::query_nodes_length(conn, &[node_id])
                    .ok()
                    .and_then(|lengths| lengths.get(&node_id).copied())
            },
            |graph, node_id| expand(conn, graph, &self.block_group_id, node_id),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use gen_core::{GraphNode, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand};
    use gen_graph::{GraphEdge, graph_from_interval_tree};

    use super::*;
    use crate::{
        block_group::{BlockGroup, NewBlockGroup},
        block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
        collection::Collection,
        node::Node,
        sample::{NewSample, Sample},
        sequence::Sequence,
        test_helpers::get_connection,
    };

    fn setup_subset_graph() -> (crate::db::GraphConnection, HashId) {
        let conn = get_connection(None).unwrap();
        Collection::get_or_create(&conn, "test").unwrap();
        Sample::get_or_create(
            &conn,
            NewSample {
                name: "test",
                ..Default::default()
            },
        )
        .unwrap();
        let block_group = BlockGroup::create(
            &conn,
            NewBlockGroup {
                collection_name: "test",
                sample_name: "test",
                name: "chr1",
                ..Default::default()
            },
        )
        .unwrap();

        let seq_x = Sequence::new()
            .sequence_type("DNA")
            .sequence("XXXXX")
            .save(&conn)
            .unwrap();
        let seq_y = Sequence::new()
            .sequence_type("DNA")
            .sequence("YYYYY")
            .save(&conn)
            .unwrap();
        let seq_z = Sequence::new()
            .sequence_type("DNA")
            .sequence("ZZZZZ")
            .save(&conn)
            .unwrap();

        let node_x = Node::create(&conn, &seq_x.hash, &HashId::convert_str("node-x")).unwrap();
        let node_y = Node::create(&conn, &seq_y.hash, &HashId::convert_str("node-y")).unwrap();
        let node_z = Node::create(&conn, &seq_z.hash, &HashId::convert_str("node-z")).unwrap();

        let e_start = Edge::create(
            &conn,
            PATH_START_NODE_ID,
            -1,
            Strand::Forward,
            node_x,
            0,
            Strand::Forward,
        )
        .unwrap();
        let e_xy = Edge::create(
            &conn,
            node_x,
            5,
            Strand::Forward,
            node_y,
            0,
            Strand::Forward,
        )
        .unwrap();
        let e_yz = Edge::create(
            &conn,
            node_y,
            5,
            Strand::Forward,
            node_z,
            0,
            Strand::Forward,
        )
        .unwrap();
        let e_end = Edge::create(
            &conn,
            node_z,
            5,
            Strand::Forward,
            PATH_END_NODE_ID,
            0,
            Strand::Forward,
        )
        .unwrap();

        BlockGroupEdge::bulk_create(
            &conn,
            &[
                BlockGroupEdgeData {
                    block_group_id: block_group.id,
                    edge_id: e_start.id,
                    chromosome_index: 0,
                    phased: 0,
                },
                BlockGroupEdgeData {
                    block_group_id: block_group.id,
                    edge_id: e_xy.id,
                    chromosome_index: 0,
                    phased: 0,
                },
                BlockGroupEdgeData {
                    block_group_id: block_group.id,
                    edge_id: e_yz.id,
                    chromosome_index: 0,
                    phased: 0,
                },
                BlockGroupEdgeData {
                    block_group_id: block_group.id,
                    edge_id: e_end.id,
                    chromosome_index: 0,
                    phased: 0,
                },
            ],
        );

        (conn, block_group.id)
    }

    fn subset_interval_tree() -> IntervalTree<i64, NodeIntervalBlock> {
        vec![(
            0..5,
            NodeIntervalBlock {
                node_id: HashId::convert_str("node-y"),
                start: 0,
                end: 5,
                sequence_start: 0,
                sequence_end: 5,
                strand: Strand::Forward,
            },
        )]
        .into_iter()
        .collect()
    }

    fn test_edge(edge_id: &str) -> Vec<GraphEdge> {
        vec![GraphEdge {
            edge_id: HashId::convert_str(edge_id),
            source_strand: Strand::Forward,
            target_strand: Strand::Forward,
            chromosome_index: 0,
            phased: 0,
            created_on: 0,
        }]
    }

    fn variable_length_branched_graph() -> GenGraph {
        let node_aaa = GraphNode {
            node_id: HashId::convert_str("node-aaa"),
            sequence_start: 0,
            sequence_end: 3,
        };
        let node_cc = GraphNode {
            node_id: HashId::convert_str("node-cc"),
            sequence_start: 0,
            sequence_end: 2,
        };
        let node_gggg = GraphNode {
            node_id: HashId::convert_str("node-gggg"),
            sequence_start: 0,
            sequence_end: 4,
        };
        let node_ttt = GraphNode {
            node_id: HashId::convert_str("node-ttt"),
            sequence_start: 0,
            sequence_end: 3,
        };

        let mut graph = GenGraph::new();
        graph.add_edge(node_aaa, node_cc, test_edge("edge-aaa-cc"));
        graph.add_edge(node_aaa, node_gggg, test_edge("edge-aaa-gggg"));
        graph.add_edge(node_cc, node_ttt, test_edge("edge-cc-ttt"));
        graph.add_edge(node_gggg, node_ttt, test_edge("edge-gggg-ttt"));
        graph
    }

    fn position_set(positions: &[GraphNodePosition]) -> HashSet<(HashId, i64)> {
        positions
            .iter()
            .map(|pos| (pos.graph_node.node_id, pos.offset))
            .collect()
    }

    #[test]
    fn test_find_offset_within_subset_node() {
        let (conn, bg_id) = setup_subset_graph();
        let tree = subset_interval_tree();
        let mut graph = graph_from_interval_tree(&tree);

        let y_node = GraphNode {
            node_id: HashId::convert_str("node-y"),
            sequence_start: 0,
            sequence_end: 5,
        };
        let anchor = GraphNodePosition {
            graph_node: y_node,
            offset: 0,
        };

        let result = find_offset(&mut graph, &anchor, 3, |g, nid| {
            expand(&conn, g, &bg_id, nid)
        });
        assert!(
            result.is_ok(),
            "find_offset(3) should succeed: {:?}",
            result.err()
        );
        let positions = result.unwrap();
        assert_eq!(positions.len(), 1);
        assert_eq!(positions[0].graph_node, y_node);
        assert_eq!(positions[0].offset, 3);
    }

    #[test]
    fn test_find_offset_forward_expands_beyond_subset() {
        let (conn, bg_id) = setup_subset_graph();
        let tree = subset_interval_tree();
        let mut graph = graph_from_interval_tree(&tree);

        let y_node = GraphNode {
            node_id: HashId::convert_str("node-y"),
            sequence_start: 0,
            sequence_end: 5,
        };
        let anchor = GraphNodePosition {
            graph_node: y_node,
            offset: 0,
        };

        let result = find_offset(&mut graph, &anchor, 7, |g, nid| {
            expand(&conn, g, &bg_id, nid)
        });
        assert!(
            result.is_ok(),
            "find_offset(7) should expand to Z: {:?}",
            result.err()
        );
        let positions = result.unwrap();
        assert_eq!(positions.len(), 1);
        assert_eq!(
            positions[0].graph_node.node_id,
            HashId::convert_str("node-z")
        );
        assert_eq!(positions[0].offset, 2);
    }

    #[test]
    fn test_find_offset_forward_expands_after_existing_paths_fail() {
        let (conn, bg_id) = setup_subset_graph();
        let mut graph = GenGraph::new();

        let y_node = GraphNode {
            node_id: HashId::convert_str("node-y"),
            sequence_start: 0,
            sequence_end: 5,
        };
        let dead_end = GraphNode {
            node_id: HashId::convert_str("dead-end"),
            sequence_start: 0,
            sequence_end: 1,
        };
        graph.add_edge(
            y_node,
            dead_end,
            vec![GraphEdge {
                edge_id: HashId::convert_str("edge-y-dead"),
                source_strand: Strand::Forward,
                target_strand: Strand::Forward,
                chromosome_index: 0,
                phased: 0,
                created_on: 0,
            }],
        );
        let anchor = GraphNodePosition {
            graph_node: y_node,
            offset: 0,
        };

        let result = find_offset(&mut graph, &anchor, 7, |g, nid| {
            expand(&conn, g, &bg_id, nid)
        });
        assert!(
            result.is_ok(),
            "find_offset(7) should expand from Y after the existing path fails: {:?}",
            result.err()
        );
        let positions = result.unwrap();
        assert_eq!(positions.len(), 1);
        assert_eq!(
            positions[0].graph_node.node_id,
            HashId::convert_str("node-z")
        );
        assert_eq!(positions[0].offset, 2);
    }

    #[test]
    fn test_find_offset_backward_expands_beyond_subset() {
        let (conn, bg_id) = setup_subset_graph();
        let tree = subset_interval_tree();
        let mut graph = graph_from_interval_tree(&tree);

        let y_node = GraphNode {
            node_id: HashId::convert_str("node-y"),
            sequence_start: 0,
            sequence_end: 5,
        };
        let anchor = GraphNodePosition {
            graph_node: y_node,
            offset: 0,
        };

        let result = find_offset(&mut graph, &anchor, -3, |g, nid| {
            expand(&conn, g, &bg_id, nid)
        });
        assert!(
            result.is_ok(),
            "find_offset(-3) should expand to X: {:?}",
            result.err()
        );
        let positions = result.unwrap();
        assert_eq!(positions.len(), 1);
        assert_eq!(
            positions[0].graph_node.node_id,
            HashId::convert_str("node-x")
        );
        assert_eq!(positions[0].offset, 2);
    }

    #[test]
    fn test_find_offset_out_of_bounds_with_expansion() {
        let (conn, bg_id) = setup_subset_graph();
        let tree = subset_interval_tree();
        let mut graph = graph_from_interval_tree(&tree);

        let y_node = GraphNode {
            node_id: HashId::convert_str("node-y"),
            sequence_start: 0,
            sequence_end: 5,
        };
        let anchor = GraphNodePosition {
            graph_node: y_node,
            offset: 0,
        };

        let result = find_offset(&mut graph, &anchor, 100, |g, nid| {
            expand(&conn, g, &bg_id, nid)
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_find_offset_with_fragment_node() {
        let (conn, bg_id) = setup_subset_graph();

        let tree: IntervalTree<i64, NodeIntervalBlock> = vec![(
            0..3,
            NodeIntervalBlock {
                node_id: HashId::convert_str("node-y"),
                start: 0,
                end: 3,
                sequence_start: 0,
                sequence_end: 3,
                strand: Strand::Forward,
            },
        )]
        .into_iter()
        .collect();
        let mut graph = graph_from_interval_tree(&tree);

        let y_frag = GraphNode {
            node_id: HashId::convert_str("node-y"),
            sequence_start: 0,
            sequence_end: 3,
        };
        let anchor = GraphNodePosition {
            graph_node: y_frag,
            offset: 0,
        };

        let result = find_offset(&mut graph, &anchor, 2, |g, nid| {
            expand(&conn, g, &bg_id, nid)
        });
        assert!(result.is_ok());
        let positions = result.unwrap();
        assert_eq!(positions[0].graph_node, y_frag);
        assert_eq!(positions[0].offset, 2);
    }

    #[test]
    fn resolve_anchor_before_fragment_stays_on_same_backing_node() {
        // The case is when we have a node fragment
        // at the beginning of an accession. We don't know how far back to expand
        // the fragment as there are no edges.
        let (conn, bg_id) = setup_subset_graph();

        let tree: IntervalTree<i64, NodeIntervalBlock> = vec![(
            0..3,
            NodeIntervalBlock {
                node_id: HashId::convert_str("node-y"),
                start: 0,
                end: 3,
                sequence_start: 2,
                sequence_end: 5,
                strand: Strand::Forward,
            },
        )]
        .into_iter()
        .collect();
        let graph = graph_from_interval_tree(&tree);
        let resolved = ResolvedGraph {
            graph,
            interval_tree: tree,
            block_group_id: bg_id,
        };

        let position = resolved.resolve_anchor(-2, &conn).unwrap();

        assert_eq!(
            position.graph_node,
            GraphNode {
                node_id: HashId::convert_str("node-y"),
                sequence_start: 2,
                sequence_end: 5,
            }
        );
        assert_eq!(position.offset, -2);
        assert_eq!(position.coordinate(), 0);
    }

    #[test]
    fn test_expand_adds_neighbors() {
        let (conn, bg_id) = setup_subset_graph();
        let tree = subset_interval_tree();
        let mut graph = graph_from_interval_tree(&tree);

        assert_eq!(graph.node_count(), 1);

        let expanded = expand(&conn, &mut graph, &bg_id, HashId::convert_str("node-y"));
        assert!(expanded, "expand should add new nodes");
        assert!(
            graph.node_count() > 1,
            "graph should have more nodes after expansion"
        );

        let node_ids: Vec<HashId> = graph.nodes().map(|n| n.node_id).collect();
        assert!(
            node_ids.contains(&HashId::convert_str("node-x")),
            "X should be added after expanding Y"
        );
        assert!(
            node_ids.contains(&HashId::convert_str("node-z")),
            "Z should be added after expanding Y"
        );
    }

    #[test]
    fn test_find_offset_in_variable_length_branch_finds_middle_nodes() {
        let mut graph = variable_length_branched_graph();
        let aaa_anchor = GraphNodePosition {
            graph_node: GraphNode {
                node_id: HashId::convert_str("node-aaa"),
                sequence_start: 0,
                sequence_end: 3,
            },
            offset: 2,
        };

        let from_aaa = find_offset(&mut graph, &aaa_anchor, 2, |_, _| false).unwrap();
        assert_eq!(
            position_set(&from_aaa),
            HashSet::from([
                (HashId::convert_str("node-cc"), 1),
                (HashId::convert_str("node-gggg"), 1)
            ])
        );

        let ttt_anchor = GraphNodePosition {
            graph_node: GraphNode {
                node_id: HashId::convert_str("node-ttt"),
                sequence_start: 0,
                sequence_end: 3,
            },
            offset: 0,
        };
        let from_ttt = find_offset(&mut graph, &ttt_anchor, -2, |_, _| false).unwrap();
        assert_eq!(
            position_set(&from_ttt),
            HashSet::from([
                (HashId::convert_str("node-cc"), 0),
                (HashId::convert_str("node-gggg"), 2)
            ])
        );
    }

    #[test]
    fn test_find_offset_in_variable_length_branch_returns_single_position_within_node() {
        let mut graph = variable_length_branched_graph();
        let anchor = GraphNodePosition {
            graph_node: GraphNode {
                node_id: HashId::convert_str("node-aaa"),
                sequence_start: 0,
                sequence_end: 3,
            },
            offset: 1,
        };

        let positions = find_offset(&mut graph, &anchor, 1, |_, _| false).unwrap();
        assert_eq!(
            position_set(&positions),
            HashSet::from([(HashId::convert_str("node-aaa"), 2)])
        );
    }

    #[test]
    fn test_find_offset_in_variable_length_branch_finds_different_ttt_offsets() {
        let mut graph = variable_length_branched_graph();
        let anchor = GraphNodePosition {
            graph_node: GraphNode {
                node_id: HashId::convert_str("node-aaa"),
                sequence_start: 0,
                sequence_end: 3,
            },
            offset: 2,
        };

        let positions = find_offset(&mut graph, &anchor, 6, |_, _| false).unwrap();
        assert_eq!(
            position_set(&positions),
            HashSet::from([
                (HashId::convert_str("node-ttt"), 1),
                (HashId::convert_str("node-ttt"), 3)
            ])
        );
    }
}
