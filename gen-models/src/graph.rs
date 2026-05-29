use std::collections::{HashSet, VecDeque};

use gen_core::{HashId, NodeIntervalBlock, PATH_END_NODE_ID, PATH_START_NODE_ID};
use gen_graph::{
    GenGraph, GraphError, GraphNode, GraphNodePosition, MergeGraph, graph_from_interval_tree,
};
use intervaltree::IntervalTree;
use petgraph::Direction;
use rusqlite::params;

use crate::{
    block_group_edge::AugmentedEdge,
    db::GraphConnection,
    edge::{Edge, GroupBlock},
    region::ResolvedGenRegion,
};

pub struct ResolvedGraph {
    pub graph: GenGraph,
    pub interval_tree: IntervalTree<i64, NodeIntervalBlock>,
    pub block_group_id: HashId,
}

fn edges_for_node_in_block_group(
    conn: &GraphConnection,
    block_group_id: &HashId,
    node_id: HashId,
) -> Vec<AugmentedEdge> {
    let sql = "SELECT e.id, e.source_node_id, e.source_coordinate, e.source_strand, \
               e.target_node_id, e.target_coordinate, e.target_strand, \
               bge.chromosome_index, bge.phased, bge.created_on \
               FROM edges e \
               JOIN block_group_edges bge ON e.id = bge.edge_id \
               WHERE bge.block_group_id = ?1 \
               AND (e.source_node_id = ?2 OR e.target_node_id = ?2)";

    let mut stmt = conn.prepare(sql).unwrap();
    let rows = stmt
        .query_map(params![block_group_id, node_id], |row| {
            Ok(AugmentedEdge {
                edge: Edge {
                    id: row.get(0)?,
                    source_node_id: row.get(1)?,
                    source_coordinate: row.get(2)?,
                    source_strand: row.get(3)?,
                    target_node_id: row.get(4)?,
                    target_coordinate: row.get(5)?,
                    target_strand: row.get(6)?,
                },
                chromosome_index: row.get(7)?,
                phased: row.get(8)?,
                created_on: row.get(9)?,
            })
        })
        .unwrap();

    let mut results = Vec::new();
    for row in rows {
        results.push(row.unwrap());
    }
    results
}

pub fn expand(
    conn: &GraphConnection,
    graph: &mut GenGraph,
    block_group_id: &HashId,
    node_id: HashId,
) -> bool {
    let edges_1hop = edges_for_node_in_block_group(conn, block_group_id, node_id);

    let mut neighbor_ids: Vec<HashId> = edges_1hop
        .iter()
        .flat_map(|ae| [ae.edge.source_node_id, ae.edge.target_node_id])
        .filter(|id| *id != node_id)
        .collect();
    neighbor_ids.sort();
    neighbor_ids.dedup();

    let mut all_edges = edges_1hop;
    for neighbor_id in &neighbor_ids {
        let neighbor_edges = edges_for_node_in_block_group(conn, block_group_id, *neighbor_id);
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

    let all_blocks = Edge::blocks_from_edges(conn, &new_edges);

    let new_blocks: Vec<GroupBlock> = all_blocks
        .into_iter()
        .filter(|b| b.node_id != PATH_START_NODE_ID && b.node_id != PATH_END_NODE_ID)
        .collect();

    if new_blocks.is_empty() {
        return false;
    }

    let (fragment, _) = Edge::build_graph(&new_edges, &new_blocks);
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

            if neighbors.is_empty() && expand(graph, node.node_id) {
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

            if neighbors.is_empty() && expand(graph, node.node_id) {
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
        region: &ResolvedGenRegion,
        conn: &GraphConnection,
    ) -> Result<GraphNodePosition, GraphError> {
        let coord = region.start;

        let mut block = None;
        for item in self.interval_tree.query_point(coord) {
            let b = &item.value;
            if b.node_id != PATH_START_NODE_ID && b.node_id != PATH_END_NODE_ID {
                block = Some(*b);
                break;
            }
        }

        if block.is_none() {
            for item in self.interval_tree.query_point(coord - 1) {
                let b = &item.value;
                if b.node_id != PATH_START_NODE_ID && b.node_id != PATH_END_NODE_ID {
                    block = Some(*b);
                    break;
                }
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

        let first_block = self
            .interval_tree
            .iter()
            .map(|item| &item.value)
            .filter(|b| b.node_id != PATH_START_NODE_ID && b.node_id != PATH_END_NODE_ID)
            .min_by_key(|b| b.start)
            .copied();
        let last_block = self
            .interval_tree
            .iter()
            .map(|item| &item.value)
            .filter(|b| b.node_id != PATH_START_NODE_ID && b.node_id != PATH_END_NODE_ID)
            .max_by_key(|b| b.end)
            .copied();

        let before_tree = first_block.map(|b| coord < b.start).unwrap_or(false);
        let after_tree = last_block.map(|b| coord >= b.end).unwrap_or(false);

        let boundary_block = if before_tree {
            first_block
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

pub fn resolve_and_find_offset(
    conn: &GraphConnection,
    region: &ResolvedGenRegion,
    distance: i64,
) -> Result<Vec<GraphNodePosition>, GraphError> {
    let full_tree = region.intervaltree(conn).map_err(|_| GraphError::NoPath)?;

    let filtered: Vec<(std::ops::Range<i64>, NodeIntervalBlock)> = full_tree
        .iter()
        .filter(|item| {
            item.value.node_id != PATH_START_NODE_ID
                && item.value.node_id != PATH_END_NODE_ID
                && item.value.sequence_start < item.value.sequence_end
        })
        .map(|item| (item.range.clone(), item.value))
        .collect();
    let interval_tree: IntervalTree<i64, NodeIntervalBlock> = filtered.into_iter().collect();

    let mut graph = graph_from_interval_tree(&interval_tree);

    let resolved = ResolvedGraph {
        graph: graph.clone(),
        interval_tree,
        block_group_id: region.block_group.id,
    };
    let anchor = resolved.resolve_anchor(region, conn)?;

    find_offset(&mut graph, &anchor, distance, |g, nid| {
        expand(conn, g, &resolved.block_group_id, nid)
    })
}

#[cfg(test)]
mod tests {
    use gen_core::Strand;

    use super::*;
    use crate::{
        accession::Accession,
        block_group::{BlockGroup, NewBlockGroup, PathCache},
        block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
        collection::Collection,
        node::Node,
        path::Path,
        region::ResolvedRegionKind,
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

    fn create_accession(
        conn: &crate::db::GraphConnection,
        block_group_id: HashId,
        name: &str,
        start: i64,
        end: i64,
    ) -> Accession {
        let edges = BlockGroupEdge::edges_for_block_group(conn, &block_group_id);
        let mut by_source: std::collections::HashMap<HashId, &AugmentedEdge> =
            std::collections::HashMap::new();
        for ae in &edges {
            by_source.insert(ae.edge.source_node_id, ae);
        }
        let mut ordered = vec![];
        let mut current = Some(PATH_START_NODE_ID);
        while let Some(src) = current {
            if let Some(ae) = by_source.get(&src) {
                ordered.push(ae.edge.id);
                current = if ae.edge.target_node_id == PATH_END_NODE_ID {
                    None
                } else {
                    Some(ae.edge.target_node_id)
                };
            } else {
                break;
            }
        }
        let path = Path::create(conn, name, &block_group_id, &ordered).unwrap();
        let mut path_cache = PathCache::new(conn);
        let accession =
            BlockGroup::add_accession(conn, &path, name, start, end, &mut path_cache).unwrap();
        Path::delete(conn, name, &block_group_id);
        accession
    }

    fn setup_graph_with_accession(
        conn: &crate::db::GraphConnection,
        block_group_id: HashId,
        name: &str,
    ) -> (Accession, Accession) {
        let acc_full = create_accession(conn, block_group_id, &format!("{name}-full"), 0, 15);
        let acc_y = create_accession(conn, block_group_id, &format!("{name}-y"), 5, 10);
        (acc_full, acc_y)
    }

    fn make_region(
        bg: BlockGroup,
        accession: Accession,
        anchor_start: i64,
        anchor_end: i64,
        feature_length: i64,
        start: i64,
        end: i64,
    ) -> ResolvedGenRegion {
        ResolvedGenRegion {
            block_group: bg,
            path: None,
            accession: Some(accession),
            kind: ResolvedRegionKind::Accession,
            anchor_start,
            anchor_end,
            feature_length,
            start,
            end,
        }
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
    fn test_resolve_and_find_offset_within_node() {
        let (conn, bg_id) = setup_subset_graph();
        let bg = BlockGroup::get_by_id(&conn, &bg_id).unwrap();
        let (acc_full, _) = setup_graph_with_accession(&conn, bg_id, "within");

        // Accession: X[0..5] Y[5..10] Z[10..15], total length 15
        // Position 7 = Y offset 2
        let region = make_region(bg, acc_full, 0, 15, 15, 7, 7);

        let result = resolve_and_find_offset(&conn, &region, 2);
        assert!(result.is_ok(), "distance 2 within Y: {:?}", result.err());
        let positions = result.unwrap();
        assert_eq!(positions.len(), 1);
        assert_eq!(
            positions[0].graph_node.node_id,
            HashId::convert_str("node-y")
        );
        assert_eq!(positions[0].offset, 4); // 2 + 2
    }

    #[test]
    fn test_resolve_and_find_offset_forward_across_nodes() {
        let (conn, bg_id) = setup_subset_graph();
        let bg = BlockGroup::get_by_id(&conn, &bg_id).unwrap();
        let (acc_full, _) = setup_graph_with_accession(&conn, bg_id, "fwd");

        // Position 7 in Y, forward 5 → Z offset 2
        let region = make_region(bg, acc_full, 0, 15, 15, 7, 7);

        let result = resolve_and_find_offset(&conn, &region, 5);
        assert!(
            result.is_ok(),
            "forward 5 from Y should reach Z: {:?}",
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
    fn test_resolve_and_find_offset_backward_across_nodes() {
        let (conn, bg_id) = setup_subset_graph();
        let bg = BlockGroup::get_by_id(&conn, &bg_id).unwrap();
        let (acc_full, _) = setup_graph_with_accession(&conn, bg_id, "bwd");

        // Position 7 in Y, backward 5 → X offset 2
        let region = make_region(bg, acc_full, 0, 15, 15, 7, 7);

        let result = resolve_and_find_offset(&conn, &region, -5);
        assert!(
            result.is_ok(),
            "backward 5 from Y should reach X: {:?}",
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
    fn test_resolve_and_find_offset_out_of_bounds() {
        let (conn, bg_id) = setup_subset_graph();
        let bg = BlockGroup::get_by_id(&conn, &bg_id).unwrap();
        let (acc_full, _) = setup_graph_with_accession(&conn, bg_id, "oob");

        let region = make_region(bg, acc_full, 0, 15, 15, 7, 7);

        let result = resolve_and_find_offset(&conn, &region, 100);
        assert!(result.is_err(), "distance 100 should be out of bounds");
    }

    #[test]
    fn test_resolve_and_find_offset_from_start() {
        let (conn, bg_id) = setup_subset_graph();
        let bg = BlockGroup::get_by_id(&conn, &bg_id).unwrap();
        let (acc_full, _) = setup_graph_with_accession(&conn, bg_id, "start");

        // Position 0 = start of X, forward 12 → Z offset 2
        let region = make_region(bg, acc_full, 0, 15, 15, 0, 0);

        let result = resolve_and_find_offset(&conn, &region, 12);
        assert!(
            result.is_ok(),
            "forward 12 from start of X: {:?}",
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
    fn test_resolve_and_find_offset_from_end() {
        let (conn, bg_id) = setup_subset_graph();
        let bg = BlockGroup::get_by_id(&conn, &bg_id).unwrap();
        let (acc_full, _) = setup_graph_with_accession(&conn, bg_id, "end");

        // Position 14 = Z offset 4, backward 14 → X offset 0
        let region = make_region(bg, acc_full, 0, 15, 15, 14, 14);

        let result = resolve_and_find_offset(&conn, &region, -14);
        assert!(
            result.is_ok(),
            "backward 14 from end of Z: {:?}",
            result.err()
        );
        let positions = result.unwrap();
        assert_eq!(positions.len(), 1);
        assert_eq!(
            positions[0].graph_node.node_id,
            HashId::convert_str("node-x")
        );
        assert_eq!(positions[0].offset, 0);
    }

    #[test]
    fn test_resolve_anchor_negative_bounds() {
        let (conn, bg_id) = setup_subset_graph();
        let bg = BlockGroup::get_by_id(&conn, &bg_id).unwrap();
        let (_, acc_y) = setup_graph_with_accession(&conn, bg_id, "neg");

        // Accession covering Y: accession-relative 0..5
        let interval_tree = acc_y.intervaltree(&conn).unwrap();
        let graph = graph_from_interval_tree(&interval_tree);
        let resolved = ResolvedGraph {
            graph,
            interval_tree,
            block_group_id: bg_id,
        };

        // Query at accession position -3 (3 before the start of the accession)
        let region = make_region(bg, acc_y, 5, 10, 5, -3, -3);

        let anchor = resolved.resolve_anchor(&region, &conn);
        assert!(
            anchor.is_ok(),
            "resolve_anchor(-3) should expand backward: {:?}",
            anchor.err()
        );
        let pos = anchor.unwrap();
        assert_eq!(pos.graph_node.node_id, HashId::convert_str("node-x"));
        assert_eq!(pos.offset, 2);
    }

    #[test]
    fn test_resolve_and_find_offset_accession_within() {
        let (conn, bg_id) = setup_subset_graph();
        let bg = BlockGroup::get_by_id(&conn, &bg_id).unwrap();
        let (_, acc_y) = setup_graph_with_accession(&conn, bg_id, "within-acc");

        // Accession covering Y: accession-relative 0..5
        // Position 2 = Y offset 2, forward 1 → Y offset 3
        let region = make_region(bg, acc_y, 5, 10, 5, 2, 2);

        let result = resolve_and_find_offset(&conn, &region, 1);
        assert!(result.is_ok(), "distance 1 within Y: {:?}", result.err());
        let positions = result.unwrap();
        assert_eq!(positions.len(), 1);
        assert_eq!(
            positions[0].graph_node.node_id,
            HashId::convert_str("node-y")
        );
        assert_eq!(positions[0].offset, 3);
    }

    #[test]
    fn test_resolve_and_find_offset_accession_forward_expand() {
        let (conn, bg_id) = setup_subset_graph();
        let bg = BlockGroup::get_by_id(&conn, &bg_id).unwrap();
        let (_, acc_y) = setup_graph_with_accession(&conn, bg_id, "fwd-acc");

        // Accession covering Y: accession-relative 0..5
        // Position 3 = Y offset 3, forward 5 → should expand to Z offset 3
        let region = make_region(bg, acc_y, 5, 10, 5, 3, 3);

        let result = resolve_and_find_offset(&conn, &region, 5);
        assert!(
            result.is_ok(),
            "forward 5 from Y should reach Z: {:?}",
            result.err()
        );
        let positions = result.unwrap();
        assert_eq!(positions.len(), 1);
        assert_eq!(
            positions[0].graph_node.node_id,
            HashId::convert_str("node-z")
        );
        assert_eq!(positions[0].offset, 3);
    }

    #[test]
    fn test_resolve_and_find_offset_accession_backward_expand() {
        let (conn, bg_id) = setup_subset_graph();
        let bg = BlockGroup::get_by_id(&conn, &bg_id).unwrap();
        let (_, acc_y) = setup_graph_with_accession(&conn, bg_id, "bwd-acc");

        // Accession covering Y: accession-relative 0..5
        // Position 1 = Y offset 1, backward 4 → should expand to X offset 2
        let region = make_region(bg, acc_y, 5, 10, 5, 1, 1);

        let result = resolve_and_find_offset(&conn, &region, -4);
        assert!(
            result.is_ok(),
            "backward 4 from Y should reach X: {:?}",
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
    fn test_resolve_and_find_offset_accession_out_of_bounds() {
        let (conn, bg_id) = setup_subset_graph();
        let bg = BlockGroup::get_by_id(&conn, &bg_id).unwrap();
        let (_, acc_y) = setup_graph_with_accession(&conn, bg_id, "oob-acc");

        let region = make_region(bg, acc_y, 5, 10, 5, 2, 2);

        let result = resolve_and_find_offset(&conn, &region, 100);
        assert!(result.is_err(), "distance 100 should be out of bounds");
    }
}
