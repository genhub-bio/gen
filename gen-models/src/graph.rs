use std::collections::{HashSet, VecDeque};

use gen_core::{HashId, is_end_node, is_start_node};
use gen_graph::{GenGraph, GraphEdge, GraphNode, GraphNodePosition, MergeGraph};
use petgraph::Direction;
use rusqlite::{params, types::Value};
use thiserror::Error;

use crate::{
    block_group_edge::{AugmentedEdge, BlockGroupEdge},
    db::GraphConnection,
    edge::Edge,
    node::Node,
    traits::Query,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GraphPositionAnchor {
    pub coordinate: i64,
    pub node_id: Option<HashId>,
}

#[derive(Debug, Error, PartialEq)]
pub enum GraphTraversalError {
    #[error("anchor position does not resolve to any graph nodes")]
    AnchorNotFound,
    #[error("offset is out of bounds of the graph")]
    OutOfBounds,
    #[error("database error: {0}")]
    DatabaseError(String),
}

pub struct SqlGraphTraversal<'a> {
    conn: &'a GraphConnection,
    graph: GenGraph,
    expanded_nodes: HashSet<(GraphNode, Direction)>,
}

impl<'a> SqlGraphTraversal<'a> {
    pub fn new(conn: &'a GraphConnection, graph: GenGraph) -> Self {
        SqlGraphTraversal {
            conn,
            graph,
            expanded_nodes: HashSet::new(),
        }
    }

    pub fn graph(&self) -> &GenGraph {
        &self.graph
    }

    pub fn find_offset(
        &mut self,
        anchor: GraphPositionAnchor,
        distance: i64,
    ) -> Result<Vec<GraphNodePosition>, GraphTraversalError> {
        let starts = self.resolve_anchor(anchor)?;
        let mut queue = starts
            .into_iter()
            .map(|position| (position.graph_node, position.offset, distance))
            .collect::<VecDeque<_>>();
        let mut visited = HashSet::new();
        let mut results = Vec::new();

        while let Some((node, offset, remaining)) = queue.pop_front() {
            if !visited.insert((node, offset, remaining)) {
                continue;
            }

            if remaining == 0 {
                results.push(GraphNodePosition {
                    graph_node: node,
                    offset,
                });
                continue;
            }

            if remaining > 0 {
                let available = node.length() - offset;
                if remaining <= available {
                    results.push(GraphNodePosition {
                        graph_node: node,
                        offset: offset + remaining,
                    });
                    continue;
                }

                self.expand_endpoint(node, Direction::Outgoing)?;
                for next in self.graph.neighbors_directed(node, Direction::Outgoing) {
                    queue.push_back((next, 0, remaining - available));
                }
            } else {
                if -remaining <= offset {
                    results.push(GraphNodePosition {
                        graph_node: node,
                        offset: offset + remaining,
                    });
                    continue;
                }

                self.expand_endpoint(node, Direction::Incoming)?;
                for previous in self.graph.neighbors_directed(node, Direction::Incoming) {
                    queue.push_back((previous, previous.length(), remaining + offset));
                }
            }
        }

        results.sort();
        results.dedup();
        if results.is_empty() {
            Err(GraphTraversalError::OutOfBounds)
        } else {
            Ok(results)
        }
    }

    fn resolve_anchor(
        &self,
        anchor: GraphPositionAnchor,
    ) -> Result<Vec<GraphNodePosition>, GraphTraversalError> {
        let mut positions = self
            .graph
            .nodes()
            .filter(|node| match anchor.node_id {
                Some(node_id) => {
                    node.node_id == node_id
                        && anchor.coordinate >= node.sequence_start
                        && anchor.coordinate <= node.sequence_end
                }
                None => {
                    self.graph
                        .neighbors_directed(*node, Direction::Incoming)
                        .next()
                        .is_none()
                        && anchor.coordinate >= 0
                        && anchor.coordinate <= node.length()
                }
            })
            .map(|node| GraphNodePosition {
                graph_node: node,
                offset: if anchor.node_id.is_some() {
                    anchor.coordinate - node.sequence_start
                } else {
                    anchor.coordinate
                },
            })
            .collect::<Vec<_>>();

        positions.sort();
        positions.dedup();
        if positions.is_empty() {
            Err(GraphTraversalError::AnchorNotFound)
        } else {
            Ok(positions)
        }
    }

    fn expand_endpoint(
        &mut self,
        node: GraphNode,
        direction: Direction,
    ) -> Result<(), GraphTraversalError> {
        if !self.expanded_nodes.insert((node, direction)) {
            return Ok(());
        }

        let attached_edges = match direction {
            Direction::Outgoing => Edge::query(
                self.conn,
                "SELECT * FROM edges WHERE source_node_id = ?1 AND source_coordinate = ?2;",
                params![node.node_id, node.sequence_end],
            ),
            Direction::Incoming => Edge::query(
                self.conn,
                "SELECT * FROM edges WHERE target_node_id = ?1 AND target_coordinate = ?2;",
                params![node.node_id, node.sequence_start],
            ),
        };
        if attached_edges.is_empty() {
            return Ok(());
        }

        let attached_edges_by_id = attached_edges
            .iter()
            .map(|edge| (edge.id, edge))
            .collect::<std::collections::HashMap<_, _>>();
        let attached_edge_ids = attached_edges
            .iter()
            .map(|edge| edge.id)
            .collect::<Vec<_>>();
        let attached_block_group_edges = BlockGroupEdge::query(
            self.conn,
            "SELECT * FROM block_group_edges WHERE edge_id in rarray(?1);",
            params![std::rc::Rc::new(
                attached_edge_ids
                    .iter()
                    .copied()
                    .map(Value::from)
                    .collect::<Vec<Value>>()
            )],
        );
        let mut block_group_ids = attached_block_group_edges
            .iter()
            .map(|edge| edge.block_group_id)
            .collect::<Vec<_>>();
        block_group_ids.sort();
        block_group_ids.dedup();

        for block_group_id in block_group_ids {
            let edges = BlockGroupEdge::edges_for_block_group(self.conn, &block_group_id);
            let blocks = Edge::blocks_from_edges(self.conn, &edges);
            let (graph, _) = Edge::build_graph(&edges, &blocks);
            self.graph.merge_graph(&graph);
        }

        let attached_augmented_edges = attached_block_group_edges
            .iter()
            .filter_map(|block_group_edge| {
                attached_edges_by_id
                    .get(&block_group_edge.edge_id)
                    .map(|edge| AugmentedEdge {
                        edge: (*edge).clone(),
                        chromosome_index: block_group_edge.chromosome_index,
                        phased: block_group_edge.phased,
                        created_on: block_group_edge.created_on,
                    })
            })
            .collect::<Vec<_>>();
        self.merge_direct_edge_graph(&attached_augmented_edges)?;

        Ok(())
    }

    fn merge_direct_edge_graph(
        &mut self,
        edges: &[AugmentedEdge],
    ) -> Result<(), GraphTraversalError> {
        let node_ids = edges
            .iter()
            .flat_map(|edge| [edge.edge.source_node_id, edge.edge.target_node_id])
            .filter(|node_id| !is_start_node(*node_id) && !is_end_node(*node_id))
            .collect::<Vec<_>>();
        let lengths = Node::query_nodes_length(self.conn, &node_ids)
            .map_err(|error| GraphTraversalError::DatabaseError(error.to_string()))?;

        for edge in edges {
            let source_node = if is_start_node(edge.edge.source_node_id) {
                GraphNode {
                    node_id: edge.edge.source_node_id,
                    sequence_start: 0,
                    sequence_end: 0,
                }
            } else {
                GraphNode {
                    node_id: edge.edge.source_node_id,
                    sequence_start: 0,
                    sequence_end: edge.edge.source_coordinate,
                }
            };
            let target_node = if is_end_node(edge.edge.target_node_id) {
                GraphNode {
                    node_id: edge.edge.target_node_id,
                    sequence_start: 0,
                    sequence_end: 0,
                }
            } else {
                GraphNode {
                    node_id: edge.edge.target_node_id,
                    sequence_start: edge.edge.target_coordinate,
                    sequence_end: *lengths.get(&edge.edge.target_node_id).ok_or_else(|| {
                        GraphTraversalError::DatabaseError(format!(
                            "missing length for node {}",
                            edge.edge.target_node_id
                        ))
                    })?,
                }
            };
            let graph_edge = GraphEdge {
                edge_id: edge.edge.id,
                source_strand: edge.edge.source_strand,
                target_strand: edge.edge.target_strand,
                chromosome_index: edge.chromosome_index,
                phased: edge.phased,
                created_on: edge.created_on,
            };

            if let Some(existing_edges) = self.graph.edge_weight_mut(source_node, target_node) {
                if !existing_edges.contains(&graph_edge) {
                    existing_edges.push(graph_edge);
                }
            } else {
                self.graph
                    .add_edge(source_node, target_node, vec![graph_edge]);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use gen_core::{HashId, Strand};
    use gen_graph::{GenGraph, GraphEdge, GraphNode, GraphNodePosition};

    use crate::{
        block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
        collection::Collection,
        edge::Edge,
        graph::{GraphPositionAnchor, SqlGraphTraversal},
        node::Node,
        sample::{NewSample, Sample},
        sequence::Sequence,
        test_helpers::{create_bg, get_connection},
    };

    fn create_node(conn: &crate::db::GraphConnection, name: &str, sequence: &str) -> HashId {
        let sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence(sequence)
            .save(conn)
            .unwrap();
        Node::create(conn, &sequence.hash, &HashId::convert_str(name)).unwrap()
    }

    fn create_linear_graph(conn: &crate::db::GraphConnection) -> (Vec<HashId>, Vec<Edge>) {
        Collection::get_or_create(conn, "test").unwrap();
        Sample::get_or_create(
            conn,
            NewSample {
                name: "test",
                ..Default::default()
            },
        )
        .unwrap();

        let nodes = vec![
            create_node(conn, "offset-node-a", "AAA"),
            create_node(conn, "offset-node-c", "CCC"),
            create_node(conn, "offset-node-t", "TTT"),
            create_node(conn, "offset-node-g", "GGG"),
        ];
        let edges = vec![
            Edge::create(
                conn,
                nodes[0],
                3,
                Strand::Forward,
                nodes[1],
                0,
                Strand::Forward,
            )
            .unwrap(),
            Edge::create(
                conn,
                nodes[1],
                3,
                Strand::Forward,
                nodes[2],
                0,
                Strand::Forward,
            )
            .unwrap(),
            Edge::create(
                conn,
                nodes[2],
                3,
                Strand::Forward,
                nodes[3],
                0,
                Strand::Forward,
            )
            .unwrap(),
        ];

        (nodes, edges)
    }

    fn graph_node(node_id: HashId) -> GraphNode {
        GraphNode {
            node_id,
            sequence_start: 0,
            sequence_end: 3,
        }
    }

    fn graph_edge(edge: &Edge) -> GraphEdge {
        GraphEdge {
            edge_id: edge.id,
            source_strand: edge.source_strand,
            target_strand: edge.target_strand,
            chromosome_index: 0,
            phased: 0,
            created_on: 0,
        }
    }

    #[test]
    fn traverses_offsets_within_existing_subgraph() {
        let conn = get_connection(None).unwrap();
        let (nodes, edges) = create_linear_graph(&conn);
        let mut graph = GenGraph::new();
        graph.add_edge(
            graph_node(nodes[1]),
            graph_node(nodes[2]),
            vec![graph_edge(&edges[1])],
        );
        let mut traversal = SqlGraphTraversal::new(&conn, graph);

        let positions = traversal
            .find_offset(
                GraphPositionAnchor {
                    coordinate: 0,
                    node_id: Some(nodes[1]),
                },
                4,
            )
            .unwrap();

        assert_eq!(
            positions,
            vec![GraphNodePosition {
                graph_node: graph_node(nodes[2]),
                offset: 1,
            }]
        );
    }

    #[test]
    fn lazily_expands_past_subgraph_endpoint() {
        let conn = get_connection(None).unwrap();
        let (nodes, edges) = create_linear_graph(&conn);
        let block_group = create_bg(&conn, "test", "test", "chr1");
        BlockGroupEdge::bulk_create(
            &conn,
            &edges
                .iter()
                .map(|edge| BlockGroupEdgeData {
                    block_group_id: block_group.id,
                    edge_id: edge.id,
                    chromosome_index: 0,
                    phased: 0,
                })
                .collect::<Vec<_>>(),
        );
        let mut graph = GenGraph::new();
        graph.add_edge(
            graph_node(nodes[1]),
            graph_node(nodes[2]),
            vec![graph_edge(&edges[1])],
        );
        let mut traversal = SqlGraphTraversal::new(&conn, graph);

        let forward = traversal
            .find_offset(
                GraphPositionAnchor {
                    coordinate: 0,
                    node_id: Some(nodes[1]),
                },
                9,
            )
            .unwrap();
        let backward = traversal
            .find_offset(
                GraphPositionAnchor {
                    coordinate: 0,
                    node_id: Some(nodes[1]),
                },
                -2,
            )
            .unwrap();

        assert_eq!(
            forward,
            vec![GraphNodePosition {
                graph_node: graph_node(nodes[3]),
                offset: 3,
            }]
        );
        assert_eq!(
            backward,
            vec![GraphNodePosition {
                graph_node: graph_node(nodes[0]),
                offset: 1,
            }]
        );
    }

    #[test]
    fn returns_error_when_offset_is_out_of_bounds() {
        let conn = get_connection(None).unwrap();
        let (nodes, edges) = create_linear_graph(&conn);
        let block_group = create_bg(&conn, "test", "test", "chr1");
        BlockGroupEdge::bulk_create(
            &conn,
            &edges
                .iter()
                .map(|edge| BlockGroupEdgeData {
                    block_group_id: block_group.id,
                    edge_id: edge.id,
                    chromosome_index: 0,
                    phased: 0,
                })
                .collect::<Vec<_>>(),
        );
        let mut graph = GenGraph::new();
        graph.add_edge(
            graph_node(nodes[1]),
            graph_node(nodes[2]),
            vec![graph_edge(&edges[1])],
        );
        let mut traversal = SqlGraphTraversal::new(&conn, graph);

        let error = traversal
            .find_offset(
                GraphPositionAnchor {
                    coordinate: 0,
                    node_id: Some(nodes[1]),
                },
                10,
            )
            .unwrap_err();

        assert!(error.to_string().contains("out of bounds"));
    }

    #[test]
    fn missing_anchor_node_uses_start_nodes() {
        let conn = get_connection(None).unwrap();
        let (nodes, edges) = create_linear_graph(&conn);
        let mut graph = GenGraph::new();
        graph.add_edge(
            graph_node(nodes[1]),
            graph_node(nodes[2]),
            vec![graph_edge(&edges[1])],
        );
        let mut traversal = SqlGraphTraversal::new(&conn, graph);

        let positions = traversal
            .find_offset(
                GraphPositionAnchor {
                    coordinate: 1,
                    node_id: None,
                },
                2,
            )
            .unwrap();

        assert_eq!(
            positions,
            vec![GraphNodePosition {
                graph_node: graph_node(nodes[1]),
                offset: 3,
            }]
        );
    }
}
