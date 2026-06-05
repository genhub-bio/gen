use std::collections::{HashSet, VecDeque};

use gen_core::{HashId, NodeIntervalBlock, is_terminal};
use gen_graph::{GenGraph, GraphError, GraphNode, GraphNodePosition, MergeGraph};
use intervaltree::IntervalTree;
use petgraph::Direction;

use crate::{
    block_group::BlockGroup, block_group_edge::AugmentedEdge, db::GraphConnection, edge::Edge,
    node::Node,
};

pub struct ResolvedGraph {
    pub graph: GenGraph,
    pub interval_tree: IntervalTree<i64, NodeIntervalBlock>,
    pub block_group_id: HashId,
}

pub fn expand(
    conn: &GraphConnection,
    graph: &mut GenGraph,
    block_group_id: &HashId,
    node_id: HashId,
) -> bool {
    let edges_1hop =
        Edge::edges_for_block_group_nodes(conn, block_group_id, &[node_id]).unwrap_or_default();

    let mut neighbor_ids: Vec<HashId> = edges_1hop
        .iter()
        .flat_map(|ae| [ae.edge.source_node_id, ae.edge.target_node_id])
        .filter(|id| *id != node_id)
        .collect();
    neighbor_ids.sort();
    neighbor_ids.dedup();

    let mut all_edges = edges_1hop;
    if !neighbor_ids.is_empty() {
        let neighbor_edges = Edge::edges_for_block_group_nodes(conn, block_group_id, &neighbor_ids)
            .unwrap_or_default();
        for ae in neighbor_edges {
            if !all_edges
                .iter()
                .any(|existing| existing.edge.id == ae.edge.id)
            {
                all_edges.push(ae);
            }
        }
    }

    let existing_edge_ids: HashSet<HashId> = graph
        .all_edges()
        .flat_map(|(_, _, edges)| edges.iter().map(|e| e.edge_id))
        .collect();

    let new_edges: Vec<AugmentedEdge> = all_edges
        .into_iter()
        .filter(|ae| !existing_edge_ids.contains(&ae.edge.id))
        .collect();

    if new_edges.is_empty() {
        return false;
    }

    let fragment = match BlockGroup::get_graph_from_edges(conn, block_group_id, &new_edges) {
        Ok(g) => g,
        Err(_) => return false,
    };
    graph.merge_graph(&fragment);
    true
}

pub fn find_offset(
    graph: &mut GenGraph,
    anchor: &GraphNodePosition,
    distance: i64,
    mut expand: impl FnMut(&mut GenGraph, HashId) -> bool,
) -> Result<Vec<GraphNodePosition>, GraphError> {
    if distance == 0 {
        return Ok(vec![*anchor]);
    }

    match find_offset_with_optional_expansion(graph, anchor, distance, false, &mut expand) {
        Ok(results) => Ok(results),
        Err(GraphError::OutOfBounds(_)) => {
            find_offset_with_optional_expansion(graph, anchor, distance, true, &mut expand)
        }
        Err(err) => Err(err),
    }
}

fn find_offset_with_optional_expansion(
    graph: &mut GenGraph,
    anchor: &GraphNodePosition,
    distance: i64,
    expand_through_existing_paths: bool,
    expand: &mut impl FnMut(&mut GenGraph, HashId) -> bool,
) -> Result<Vec<GraphNodePosition>, GraphError> {
    let forward = distance > 0;
    let mut queue = VecDeque::new();
    let mut visited = HashSet::new();
    let mut results = Vec::new();

    queue.push_back((*anchor, distance));
    visited.insert(anchor.graph_node);

    while let Some((pos, remaining)) = queue.pop_front() {
        let node = pos.graph_node;
        let node_len = node.length();

        if remaining == 0 {
            results.push(pos);
            continue;
        }

        if forward {
            let in_node = node_len - pos.offset;
            if remaining <= in_node {
                results.push(GraphNodePosition {
                    graph_node: node,
                    offset: pos.offset + remaining,
                });
                continue;
            }

            let mut neighbors: Vec<GraphNode> = graph
                .neighbors_directed(node, Direction::Outgoing)
                .collect();

            if (neighbors.is_empty() || expand_through_existing_paths)
                && expand(graph, node.node_id)
            {
                neighbors = graph
                    .neighbors_directed(node, Direction::Outgoing)
                    .collect();
            }

            for neighbor in neighbors {
                if visited.insert(neighbor) {
                    queue.push_back((
                        GraphNodePosition {
                            graph_node: neighbor,
                            offset: 0,
                        },
                        remaining - in_node,
                    ));
                }
            }
        } else {
            let in_node = pos.offset;
            let abs_remaining = -remaining;
            if abs_remaining <= in_node {
                results.push(GraphNodePosition {
                    graph_node: node,
                    offset: pos.offset + remaining,
                });
                continue;
            }

            let mut neighbors: Vec<GraphNode> = graph
                .neighbors_directed(node, Direction::Incoming)
                .collect();

            if (neighbors.is_empty() || expand_through_existing_paths)
                && expand(graph, node.node_id)
            {
                neighbors = graph
                    .neighbors_directed(node, Direction::Incoming)
                    .collect();
            }

            for neighbor in neighbors {
                if visited.insert(neighbor) {
                    queue.push_back((
                        GraphNodePosition {
                            graph_node: neighbor,
                            offset: neighbor.length(),
                        },
                        remaining + in_node,
                    ));
                }
            }
        }
    }

    if results.is_empty() {
        return Err(GraphError::OutOfBounds(distance));
    }

    Ok(results)
}

impl ResolvedGraph {
    pub fn resolve_anchor(
        &self,
        coord: i64,
        conn: &GraphConnection,
    ) -> Result<GraphNodePosition, GraphError> {
        let mut block = None;
        for item in self.interval_tree.query_point(coord) {
            let b = &item.value;
            if !is_terminal(b.node_id) {
                block = Some(*b);
                break;
            }
        }

        if let Some(b) = block {
            let offset = coord - b.start;
            return Ok(GraphNodePosition {
                graph_node: GraphNode {
                    node_id: b.node_id,
                    sequence_start: b.sequence_start,
                    sequence_end: b.sequence_end,
                },
                offset,
            });
        }

        let last_block = self
            .interval_tree
            .iter()
            .map(|item| &item.value)
            .filter(|b| !is_terminal(b.node_id))
            .max_by_key(|b| b.end)
            .copied();

        let before_tree = coord < 0;
        let after_tree = last_block.map(|b| coord >= b.end).unwrap_or(false);

        let boundary_block = if before_tree {
            self.interval_tree
                .iter()
                .map(|item| &item.value)
                .filter(|b| !is_terminal(b.node_id))
                .min_by_key(|b| b.start)
                .copied()
        } else if after_tree {
            last_block
        } else {
            None
        };

        let boundary = boundary_block.ok_or(GraphError::NoPath)?;
        let boundary_node = GraphNode {
            node_id: boundary.node_id,
            sequence_start: boundary.sequence_start,
            sequence_end: boundary.sequence_end,
        };

        let boundary_anchor = GraphNodePosition {
            graph_node: boundary_node,
            offset: if before_tree {
                0
            } else {
                boundary.sequence_end - boundary.sequence_start
            },
        };

        let boundary_distance = if before_tree {
            coord - boundary.start
        } else {
            coord - boundary.end
        };

        let same_node_coordinate = if before_tree {
            boundary.sequence_start + boundary_distance
        } else {
            boundary.sequence_end + boundary_distance
        };
        if same_node_coordinate >= 0
            && let Ok(node_lengths) = Node::query_nodes_length(conn, &[boundary.node_id])
            && same_node_coordinate <= *node_lengths.get(&boundary.node_id).unwrap_or(&0)
        {
            return Ok(GraphNodePosition {
                graph_node: boundary_node,
                offset: boundary_anchor.offset + boundary_distance,
            });
        }

        let mut expanded_graph = self.graph.clone();
        expand(
            conn,
            &mut expanded_graph,
            &self.block_group_id,
            boundary_node.node_id,
        );

        let anchors = find_offset(
            &mut expanded_graph,
            &boundary_anchor,
            boundary_distance,
            |g, nid| expand(conn, g, &self.block_group_id, nid),
        )?;

        anchors.into_iter().next().ok_or(GraphError::NoPath)
    }
}

#[cfg(test)]
mod tests {
    use gen_core::{PATH_END_NODE_ID, PATH_START_NODE_ID, Strand};
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
    #[ignore]
    fn resolve_anchor_before_fragment_stays_on_same_backing_node() {
        // This test should pass one day. The case is when we have a node fragment
        // at the beginning of an accession. We don't know how far back to expand
        // the fragment as there are no edges. I have another PR to refactor the table
        // so this works out in which case
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
}
