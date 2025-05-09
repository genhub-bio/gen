use itertools::Itertools;
use petgraph::graphmap::DiGraphMap;
use rusqlite::types::Value;
use rusqlite::{params_from_iter, Connection, Result as SQLResult, Row};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, RandomState};
use std::rc::Rc;

use crate::graph::{GenGraph, GraphEdge, GraphNode};
use crate::models::block_group_edge::AugmentedEdge;
use crate::models::node::{Node, PATH_END_NODE_ID, PATH_START_NODE_ID};
use crate::models::sequence::{cached_sequence, Sequence};
use crate::models::strand::Strand;
use crate::models::traits::*;

#[derive(Clone, Debug, Eq, Hash, PartialEq, Deserialize, Serialize, Ord, PartialOrd)]
pub struct Edge {
    pub id: i64,
    pub source_node_id: i64,
    pub source_coordinate: i64,
    pub source_strand: Strand,
    pub target_node_id: i64,
    pub target_coordinate: i64,
    pub target_strand: Strand,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd)]
pub struct EdgeData {
    pub source_node_id: i64,
    pub source_coordinate: i64,
    pub source_strand: Strand,
    pub target_node_id: i64,
    pub target_coordinate: i64,
    pub target_strand: Strand,
}

impl From<&Edge> for EdgeData {
    fn from(item: &Edge) -> Self {
        EdgeData {
            source_node_id: item.source_node_id,
            source_coordinate: item.source_coordinate,
            source_strand: item.source_strand,
            target_node_id: item.target_node_id,
            target_coordinate: item.target_coordinate,
            target_strand: item.target_strand,
        }
    }
}

#[derive(Eq, Hash, PartialEq)]
pub struct BlockKey {
    pub node_id: i64,
    pub coordinate: i64,
}

#[derive(Clone, Debug)]
pub struct GroupBlock {
    pub id: i64,
    pub node_id: i64,
    sequence: Option<String>,
    external_sequence: Option<(String, String)>,
    pub start: i64,
    pub end: i64,
}

impl GroupBlock {
    pub fn new(id: i64, node_id: i64, sequence: &Sequence, start: i64, end: i64) -> Self {
        if sequence.external_sequence {
            GroupBlock {
                id,
                node_id,
                sequence: None,
                external_sequence: Some((sequence.file_path.clone(), sequence.name.clone())),
                start,
                end,
            }
        } else {
            GroupBlock {
                id,
                node_id,
                sequence: Some(sequence.get_sequence(start, end)),
                external_sequence: None,
                start,
                end,
            }
        }
    }
    pub fn sequence(&self) -> String {
        if let Some(sequence) = &self.sequence {
            sequence.to_string()
        } else if let Some((path, name)) = &self.external_sequence {
            cached_sequence(path, name, self.start as usize, self.end as usize).unwrap()
        } else {
            panic!("Sequence or external sequence is not set.")
        }
    }
}

impl Query for Edge {
    type Model = Edge;
    fn process_row(row: &Row) -> Self::Model {
        Edge {
            id: row.get(0).unwrap(),
            source_node_id: row.get(1).unwrap(),
            source_coordinate: row.get(2).unwrap(),
            source_strand: row.get(3).unwrap(),
            target_node_id: row.get(4).unwrap(),
            target_coordinate: row.get(5).unwrap(),
            target_strand: row.get(6).unwrap(),
        }
    }
}

impl Edge {
    #[allow(clippy::too_many_arguments)]
    pub fn create(
        conn: &Connection,
        source_node_id: i64,
        source_coordinate: i64,
        source_strand: Strand,
        target_node_id: i64,
        target_coordinate: i64,
        target_strand: Strand,
    ) -> Edge {
        let query = "INSERT INTO edges (source_node_id, source_coordinate, source_strand, target_node_id, target_coordinate, target_strand) VALUES (?1, ?2, ?3, ?4, ?5, ?6) RETURNING *";
        let id_query = "select id from edges where source_node_id = ?1 and source_coordinate = ?2 and source_strand = ?3 and target_node_id = ?4 and target_coordinate = ?5 and target_strand = ?6;";
        let placeholders: Vec<Value> = vec![
            source_node_id.into(),
            source_coordinate.into(),
            source_strand.into(),
            target_node_id.into(),
            target_coordinate.into(),
            target_strand.into(),
        ];

        let mut stmt = conn.prepare(query).unwrap();
        match stmt.query_row(params_from_iter(&placeholders), |row| {
            Ok(Edge {
                id: row.get(0)?,
                source_node_id: row.get(1)?,
                source_coordinate: row.get(2)?,
                source_strand: row.get(3)?,
                target_node_id: row.get(4)?,
                target_coordinate: row.get(5)?,
                target_strand: row.get(6)?,
            })
        }) {
            Ok(edge) => edge,
            Err(rusqlite::Error::SqliteFailure(err, _details)) => {
                if err.code == rusqlite::ErrorCode::ConstraintViolation {
                    Edge {
                        id: conn
                            .query_row(id_query, params_from_iter(&placeholders), |row| row.get(0))
                            .unwrap(),
                        source_node_id,
                        source_coordinate,
                        source_strand,
                        target_node_id,
                        target_coordinate,
                        target_strand,
                    }
                } else {
                    panic!("something bad happened querying the database")
                }
            }
            Err(_) => {
                panic!("something bad happened querying the database")
            }
        }
    }

    fn edge_from_row(row: &Row) -> SQLResult<Edge> {
        Ok(Edge {
            id: row.get(0)?,
            source_node_id: row.get(1)?,
            source_coordinate: row.get(2)?,
            source_strand: row.get(3)?,
            target_node_id: row.get(4)?,
            target_coordinate: row.get(5)?,
            target_strand: row.get(6)?,
        })
    }

    pub fn bulk_load(conn: &Connection, edge_ids: &[i64]) -> Vec<Edge> {
        let query_edge_ids = edge_ids
            .iter()
            .map(|edge_id| Value::from(*edge_id))
            .collect::<Vec<_>>();
        let query = "select id, source_node_id, source_coordinate, source_strand, target_node_id, target_coordinate, target_strand from edges where id in rarray(?1);";
        Edge::query(conn, query, rusqlite::params!(Rc::new(query_edge_ids)))
    }

    pub fn bulk_create(conn: &Connection, edges: &Vec<EdgeData>) -> Vec<i64> {
        let mut edge_rows = vec![];
        let mut edge_map: HashMap<EdgeData, i64> = HashMap::new();
        for edge in edges {
            let source_strand = format!("\"{0}\"", edge.source_strand);
            let target_strand = format!("\"{0}\"", edge.target_strand);
            let edge_row = format!(
                "({0}, {1}, {2}, {3}, {4}, {5})",
                edge.source_node_id,
                edge.source_coordinate,
                source_strand,
                edge.target_node_id,
                edge.target_coordinate,
                target_strand,
            );
            edge_rows.push(edge_row);
        }
        let formatted_edge_rows = edge_rows.join(", ");

        let select_statement = format!("SELECT * FROM edges WHERE (source_node_id, source_coordinate, source_strand, target_node_id, target_coordinate, target_strand) in ({0});", formatted_edge_rows);
        let existing_edges = Edge::query(conn, &select_statement, rusqlite::params!());
        for edge in existing_edges.iter() {
            edge_map.insert(EdgeData::from(edge), edge.id);
        }

        let existing_edge_set = HashSet::<EdgeData, RandomState>::from_iter(
            existing_edges.into_iter().map(Edge::to_data),
        );
        let mut edges_to_insert = HashSet::new();
        for edge in edges {
            if !existing_edge_set.contains(edge) {
                edges_to_insert.insert(edge);
            }
        }

        let mut edge_rows_to_insert = vec![];
        for edge in edges_to_insert {
            let source_strand = format!("\"{0}\"", edge.source_strand);
            let target_strand = format!("\"{0}\"", edge.target_strand);
            let edge_row = format!(
                "({0}, {1}, {2}, {3}, {4}, {5})",
                edge.source_node_id,
                edge.source_coordinate,
                source_strand,
                edge.target_node_id,
                edge.target_coordinate,
                target_strand,
            );
            edge_rows_to_insert.push(edge_row);
        }

        if !edge_rows_to_insert.is_empty() {
            for chunk in edge_rows_to_insert.chunks(100000) {
                let formatted_edge_rows_to_insert = chunk.join(", ");

                let insert_statement = format!("INSERT INTO edges (source_node_id, source_coordinate, source_strand, target_node_id, target_coordinate, target_strand) VALUES {0} RETURNING *;", formatted_edge_rows_to_insert);
                let mut stmt = conn.prepare(&insert_statement).unwrap();
                let rows = stmt.query_map([], Edge::edge_from_row).unwrap();
                for row in rows {
                    let edge = row.unwrap();
                    edge_map.insert(EdgeData::from(&edge), edge.id);
                }
            }
        }
        edges
            .iter()
            .map(|edge| *edge_map.get(edge).unwrap())
            .collect::<Vec<i64>>()
    }

    pub fn to_data(edge: Edge) -> EdgeData {
        EdgeData {
            source_node_id: edge.source_node_id,
            source_coordinate: edge.source_coordinate,
            source_strand: edge.source_strand,
            target_node_id: edge.target_node_id,
            target_coordinate: edge.target_coordinate,
            target_strand: edge.target_strand,
        }
    }

    fn get_block_boundaries(
        source_edges: Option<&Vec<&Edge>>,
        target_edges: Option<&Vec<&Edge>>,
    ) -> Vec<i64> {
        let mut block_boundary_coordinates = HashSet::new();
        if let Some(actual_source_edges) = source_edges {
            for source_edge in actual_source_edges {
                block_boundary_coordinates.insert(source_edge.source_coordinate);
            }
        }
        if let Some(actual_target_edges) = target_edges {
            for target_edge in actual_target_edges {
                block_boundary_coordinates.insert(target_edge.target_coordinate);
            }
        }

        block_boundary_coordinates
            .into_iter()
            .sorted_by(|c1, c2| Ord::cmp(&c1, &c2))
            .collect::<Vec<i64>>()
    }

    pub fn blocks_from_edges(conn: &Connection, edges: &[AugmentedEdge]) -> Vec<GroupBlock> {
        let mut node_ids = HashSet::new();
        let mut edges_by_source_node_id: HashMap<i64, Vec<&Edge>> = HashMap::new();
        let mut edges_by_target_node_id: HashMap<i64, Vec<&Edge>> = HashMap::new();
        for edge in edges.iter().map(|edge| &edge.edge) {
            if !Node::is_start_node(edge.source_node_id) {
                node_ids.insert(edge.source_node_id);
            }
            edges_by_source_node_id
                .entry(edge.source_node_id)
                .and_modify(|edges| edges.push(edge))
                .or_insert(vec![edge]);

            if !Node::is_end_node(edge.target_node_id) {
                node_ids.insert(edge.target_node_id);
            }
            edges_by_target_node_id
                .entry(edge.target_node_id)
                .and_modify(|edges| edges.push(edge))
                .or_insert(vec![edge]);
        }

        let sequences_by_node_id =
            Node::get_sequences_by_node_ids(conn, &node_ids.iter().copied().collect::<Vec<i64>>());

        let mut blocks = vec![];
        let mut block_index = 0;
        // we sort by keys to exploit the external sequence cache which keeps the most recently used
        // external sequence in memory.
        for (node_id, sequence) in sequences_by_node_id
            .iter()
            .sorted_by_key(|(_node_id, seq)| seq.hash.clone())
        {
            let block_boundaries = Edge::get_block_boundaries(
                edges_by_source_node_id.get(node_id),
                edges_by_target_node_id.get(node_id),
            );

            if !block_boundaries.is_empty() {
                for (start, end) in block_boundaries.clone().into_iter().tuple_windows() {
                    let block = GroupBlock::new(block_index, *node_id, sequence, start, end);
                    blocks.push(block);
                    block_index += 1;
                }
            } else {
                blocks.push(GroupBlock::new(
                    block_index,
                    *node_id,
                    sequence,
                    0,
                    sequence.length,
                ));
                block_index += 1;
            }
        }

        // NOTE: We need a dedicated start node and a dedicated end node for the graph formed by the
        // block group, since different paths in the block group may start or end at different
        // places on sequences.  These two "start sequence" and "end sequence" blocks will serve
        // that role.
        let start_block = GroupBlock::new(
            block_index + 1,
            PATH_START_NODE_ID,
            &Sequence::new().sequence_type("DNA").sequence("").build(),
            0,
            0,
        );
        blocks.push(start_block);
        let end_block = GroupBlock::new(
            block_index + 2,
            PATH_END_NODE_ID,
            &Sequence::new().sequence_type("DNA").sequence("").build(),
            0,
            0,
        );
        blocks.push(end_block);
        blocks
    }

    pub fn build_graph(
        edges: &Vec<AugmentedEdge>,
        blocks: &Vec<GroupBlock>,
    ) -> (GenGraph, HashMap<(i64, i64), Edge>) {
        let blocks_by_start = blocks
            .clone()
            .into_iter()
            .map(|block| {
                (
                    BlockKey {
                        node_id: block.node_id,
                        coordinate: block.start,
                    },
                    block.id,
                )
            })
            .collect::<HashMap<BlockKey, i64>>();
        let blocks_by_end = blocks
            .clone()
            .into_iter()
            .map(|block| {
                (
                    BlockKey {
                        node_id: block.node_id,
                        coordinate: block.end,
                    },
                    block.id,
                )
            })
            .collect::<HashMap<BlockKey, i64>>();
        let block_coordinates = blocks
            .clone()
            .into_iter()
            .map(|block| (block.id, (block.start, block.end)))
            .collect::<HashMap<i64, (i64, i64)>>();

        let mut graph: GenGraph = DiGraphMap::new();
        let mut edges_by_node_pair = HashMap::new();
        for block in blocks {
            graph.add_node(GraphNode {
                block_id: block.id,
                node_id: block.node_id,
                sequence_start: block.start,
                sequence_end: block.end,
            });
        }
        for augmented_edge in edges {
            let edge = &augmented_edge.edge;
            let source_key = BlockKey {
                node_id: edge.source_node_id,
                coordinate: edge.source_coordinate,
            };
            let source_id = blocks_by_end.get(&source_key);
            let target_key = BlockKey {
                node_id: edge.target_node_id,
                coordinate: edge.target_coordinate,
            };
            let target_id = blocks_by_start.get(&target_key);

            if let Some(source_id_value) = source_id {
                if let Some(target_id_value) = target_id {
                    let source_node = GraphNode {
                        block_id: *source_id_value,
                        node_id: edge.source_node_id,
                        sequence_start: block_coordinates[source_id_value].0,
                        sequence_end: block_coordinates[source_id_value].1,
                    };
                    let target_node = GraphNode {
                        block_id: *target_id_value,
                        node_id: edge.target_node_id,
                        sequence_start: block_coordinates[target_id_value].0,
                        sequence_end: block_coordinates[target_id_value].1,
                    };
                    let graph_edge = GraphEdge {
                        edge_id: edge.id,
                        source_strand: edge.source_strand,
                        target_strand: edge.target_strand,
                        chromosome_index: augmented_edge.chromosome_index,
                        phased: augmented_edge.phased,
                    };
                    if let Some(existing_edges) = graph.edge_weight_mut(source_node, target_node) {
                        existing_edges.push(graph_edge);
                    } else {
                        graph.add_edge(source_node, target_node, vec![graph_edge]);
                    }
                    edges_by_node_pair.insert((*source_id_value, *target_id_value), edge.clone());
                }
            }
        }

        (graph, edges_by_node_pair)
    }

    pub fn is_start_edge(&self) -> bool {
        self.source_node_id == PATH_START_NODE_ID
    }

    pub fn is_end_edge(&self) -> bool {
        self.target_node_id == PATH_END_NODE_ID
    }
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    use crate::models::{
        block_group::{BlockGroup, PathChange},
        block_group_edge::BlockGroupEdge,
        collection::Collection,
        path::PathBlock,
        sequence::Sequence,
    };
    use crate::test_helpers::{get_connection, setup_block_group};

    #[test]
    fn test_bulk_create() {
        let conn = &mut get_connection(None);
        Collection::create(conn, "test collection");
        let sequence1 = Sequence::new()
            .sequence_type("DNA")
            .sequence("ATCGATCG")
            .save(conn);
        let node1_id = Node::create(conn, sequence1.hash.as_str(), None);
        let edge1 = EdgeData {
            source_node_id: PATH_START_NODE_ID,
            source_coordinate: -1,
            source_strand: Strand::Forward,
            target_node_id: node1_id,
            target_coordinate: 1,
            target_strand: Strand::Forward,
        };
        let sequence2 = Sequence::new()
            .sequence_type("DNA")
            .sequence("AAAAAAAA")
            .save(conn);
        let node2_id = Node::create(conn, sequence2.hash.as_str(), None);
        let edge2 = EdgeData {
            source_node_id: node1_id,
            source_coordinate: 2,
            source_strand: Strand::Forward,
            target_node_id: node2_id,
            target_coordinate: 3,
            target_strand: Strand::Forward,
        };
        let edge3 = EdgeData {
            source_node_id: node2_id,
            source_coordinate: 4,
            source_strand: Strand::Forward,
            target_node_id: PATH_END_NODE_ID,
            target_coordinate: -1,
            target_strand: Strand::Forward,
        };

        let edge_ids = Edge::bulk_create(conn, &vec![edge1, edge2, edge3]);
        assert_eq!(edge_ids.len(), 3);
        let edges = Edge::bulk_load(conn, &edge_ids);
        assert_eq!(edges.len(), 3);

        let edges_by_source_node_id = edges
            .into_iter()
            .map(|edge| (edge.source_node_id, edge))
            .collect::<HashMap<i64, Edge>>();

        let edge_result1 = edges_by_source_node_id.get(&PATH_START_NODE_ID).unwrap();
        assert_eq!(edge_result1.source_coordinate, -1);
        assert_eq!(edge_result1.target_node_id, node1_id);
        assert_eq!(edge_result1.target_coordinate, 1);
        let edge_result2 = edges_by_source_node_id.get(&node1_id).unwrap();
        assert_eq!(edge_result2.source_coordinate, 2);
        assert_eq!(edge_result2.target_node_id, node2_id);
        assert_eq!(edge_result2.target_coordinate, 3);
        let edge_result3 = edges_by_source_node_id.get(&node2_id).unwrap();
        assert_eq!(edge_result3.source_coordinate, 4);
        assert_eq!(edge_result3.target_node_id, PATH_END_NODE_ID);
        assert_eq!(edge_result3.target_coordinate, -1);
    }

    #[test]
    fn test_bulk_create_returns_edges_in_order() {
        let conn = &mut get_connection(None);
        Collection::create(conn, "test collection");
        let sequence1 = Sequence::new()
            .sequence_type("DNA")
            .sequence("ATCGATCG")
            .save(conn);
        let node1_id = Node::create(conn, sequence1.hash.as_str(), None);
        let edge1 = EdgeData {
            source_node_id: PATH_START_NODE_ID,
            source_coordinate: -1,
            source_strand: Strand::Forward,
            target_node_id: node1_id,
            target_coordinate: 1,
            target_strand: Strand::Forward,
        };
        let sequence2 = Sequence::new()
            .sequence_type("DNA")
            .sequence("AAAAAAAA")
            .save(conn);
        let node2_id = Node::create(conn, sequence2.hash.as_str(), None);
        let edge2 = EdgeData {
            source_node_id: node1_id,
            source_coordinate: 2,
            source_strand: Strand::Forward,
            target_node_id: node2_id,
            target_coordinate: 3,
            target_strand: Strand::Forward,
        };
        let edge3 = EdgeData {
            source_node_id: node2_id,
            source_coordinate: 4,
            source_strand: Strand::Forward,
            target_node_id: PATH_END_NODE_ID,
            target_coordinate: -1,
            target_strand: Strand::Forward,
        };

        let edges = vec![edge2, edge3];
        let edge_ids1 = Edge::bulk_create(conn, &edges);
        assert_eq!(edge_ids1.len(), 2);
        for (index, id) in edge_ids1.iter().enumerate() {
            let binding = Edge::query(
                conn,
                "select * from edges where id = ?1;",
                rusqlite::params!(Value::from(*id)),
            );
            let edge = binding.first().unwrap();
            assert_eq!(EdgeData::from(edge), edges[index]);
        }

        let edges = vec![edge1, edge2, edge3];
        let edge_ids2 = Edge::bulk_create(conn, &edges);
        assert_eq!(edge_ids2[1], edge_ids1[0]);
        assert_eq!(edge_ids2[2], edge_ids1[1]);
        assert_eq!(edge_ids2.len(), 3);
        for (index, id) in edge_ids2.iter().enumerate() {
            // this sort by makes it so the order will not match the input order of the function call
            let binding = Edge::query(
                conn,
                "select * from edges where id = ?1;",
                rusqlite::params!(Value::from(*id)),
            );
            let edge = binding.first().unwrap();
            assert_eq!(EdgeData::from(edge), edges[index]);
        }
    }

    #[test]
    fn test_bulk_create_with_existing_edge() {
        let conn = &mut get_connection(None);
        Collection::create(conn, "test collection");
        let sequence1 = Sequence::new()
            .sequence_type("DNA")
            .sequence("ATCGATCG")
            .save(conn);
        let node1_id = Node::create(conn, sequence1.hash.as_str(), None);
        // NOTE: Create one edge ahead of time to confirm an existing row ID gets returned in the bulk create
        let existing_edge = Edge::create(
            conn,
            PATH_START_NODE_ID,
            -1,
            Strand::Forward,
            node1_id,
            1,
            Strand::Forward,
        );
        assert_eq!(existing_edge.source_node_id, PATH_START_NODE_ID);
        assert_eq!(existing_edge.source_coordinate, -1);
        assert_eq!(existing_edge.target_node_id, node1_id);
        assert_eq!(existing_edge.target_coordinate, 1);

        let edge1 = EdgeData {
            source_coordinate: -1,
            source_node_id: PATH_START_NODE_ID,
            source_strand: Strand::Forward,
            target_node_id: node1_id,
            target_coordinate: 1,
            target_strand: Strand::Forward,
        };
        let sequence2 = Sequence::new()
            .sequence_type("DNA")
            .sequence("AAAAAAAA")
            .save(conn);
        let node2_id = Node::create(conn, sequence2.hash.as_str(), None);
        let edge2 = EdgeData {
            source_node_id: node1_id,
            source_coordinate: 2,
            source_strand: Strand::Forward,
            target_node_id: node2_id,
            target_coordinate: 3,
            target_strand: Strand::Forward,
        };
        let edge3 = EdgeData {
            source_node_id: node2_id,
            source_coordinate: 4,
            source_strand: Strand::Forward,
            target_node_id: PATH_END_NODE_ID,
            target_coordinate: -1,
            target_strand: Strand::Forward,
        };

        let edge_ids = Edge::bulk_create(conn, &vec![edge1, edge2, edge3]);
        assert_eq!(edge_ids.len(), 3);
        let edges = Edge::bulk_load(conn, &edge_ids);
        assert_eq!(edges.len(), 3);

        let edges_by_source_node_id = edges
            .into_iter()
            .map(|edge| (edge.source_node_id, edge))
            .collect::<HashMap<i64, Edge>>();

        let edge_result1 = edges_by_source_node_id.get(&PATH_START_NODE_ID).unwrap();

        assert_eq!(edge_result1.id, existing_edge.id);

        assert_eq!(edge_result1.source_coordinate, -1);
        assert_eq!(edge_result1.target_node_id, node1_id);
        assert_eq!(edge_result1.target_coordinate, 1);
        let edge_result2 = edges_by_source_node_id.get(&node1_id).unwrap();
        assert_eq!(edge_result2.source_coordinate, 2);
        assert_eq!(edge_result2.target_node_id, node2_id);
        assert_eq!(edge_result2.target_coordinate, 3);
        let edge_result3 = edges_by_source_node_id.get(&node2_id).unwrap();
        assert_eq!(edge_result3.source_coordinate, 4);
        assert_eq!(edge_result3.target_node_id, PATH_END_NODE_ID);
        assert_eq!(edge_result3.target_coordinate, -1);
    }

    #[test]
    fn test_blocks_from_edges() {
        let conn = get_connection(None);
        let (block_group_id, path) = setup_block_group(&conn);

        let edges = BlockGroupEdge::edges_for_block_group(&conn, block_group_id);
        let blocks = Edge::blocks_from_edges(&conn, &edges);

        // 4 actual sequences: 10-length ones of all A, all T, all C, all G
        // 2 terminal node blocks (start/end)
        // 6 total
        assert_eq!(blocks.len(), 6);

        let insert_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("NNNN")
            .save(&conn);
        let insert_node_id = Node::create(&conn, insert_sequence.hash.as_str(), None);
        let insert = PathBlock {
            id: 0,
            node_id: insert_node_id,
            block_sequence: insert_sequence.get_sequence(0, 4).to_string(),
            sequence_start: 0,
            sequence_end: 4,
            path_start: 7,
            path_end: 15,
            strand: Strand::Forward,
        };
        let change = PathChange {
            block_group_id,
            path: path.clone(),
            path_accession: None,
            start: 7,
            end: 15,
            block: insert,
            chromosome_index: 0,
            phased: 0,
            preserve_edge: true,
        };
        let tree = path.intervaltree(&conn);
        BlockGroup::insert_change(&conn, &change, &tree).unwrap();
        let mut edges = BlockGroupEdge::edges_for_block_group(&conn, block_group_id);

        let blocks = Edge::blocks_from_edges(&conn, &edges);

        // 2 10-length sequences of all C, all G
        // 1 inserted NNNN sequence
        // 4 split blocks (A and T sequences were split) resulting from the inserted sequence
        // 2 terminal node blocks (start/end)
        // 9 total
        assert_eq!(blocks.len(), 9);

        // Confirm that ordering doesn't matter
        edges.reverse();
        let blocks = Edge::blocks_from_edges(&conn, &edges);

        // 2 10-length sequences of all C, all G
        // 1 inserted NNNN sequence
        // 4 split blocks (A and T sequences were split) resulting from the inserted sequence
        // 2 terminal node blocks (start/end)
        // 9 total
        assert_eq!(blocks.len(), 9);
    }

    #[test]
    fn test_get_block_boundaries() {
        let conn = get_connection(None);
        let template_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("AAAAAAAAAA")
            .save(&conn);
        let template_node_id = Node::create(&conn, template_sequence.hash.as_str(), None);

        let insert_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("NNNN")
            .save(&conn);
        let insert_node_id = Node::create(&conn, insert_sequence.hash.as_str(), None);

        let edge1 = Edge::create(
            &conn,
            template_node_id,
            2,
            Strand::Forward,
            insert_node_id,
            0,
            Strand::Forward,
        );
        let edge2 = Edge::create(
            &conn,
            insert_node_id,
            4,
            Strand::Forward,
            template_node_id,
            3,
            Strand::Forward,
        );

        let boundaries = Edge::get_block_boundaries(Some(&vec![&edge1]), Some(&vec![&edge2]));
        assert_eq!(boundaries, vec![2, 3]);
    }

    #[test]
    fn test_get_block_boundaries_with_two_original_sequences() {
        let conn = get_connection(None);
        let template_sequence1 = Sequence::new()
            .sequence_type("DNA")
            .sequence("AAAAAAAAAA")
            .save(&conn);
        let template1_node_id = Node::create(&conn, template_sequence1.hash.as_str(), None);

        let template_sequence2 = Sequence::new()
            .sequence_type("DNA")
            .sequence("TTTTTTTTTT")
            .save(&conn);
        let template2_node_id = Node::create(&conn, template_sequence2.hash.as_str(), None);

        let insert_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("NNNN")
            .save(&conn);
        let insert_node_id = Node::create(&conn, insert_sequence.hash.as_str(), None);

        let edge1 = Edge::create(
            &conn,
            template1_node_id,
            2,
            Strand::Forward,
            insert_node_id,
            0,
            Strand::Forward,
        );
        let edge2 = Edge::create(
            &conn,
            insert_node_id,
            4,
            Strand::Forward,
            template2_node_id,
            3,
            Strand::Forward,
        );

        let outgoing_boundaries = Edge::get_block_boundaries(Some(&vec![&edge1]), None);
        assert_eq!(outgoing_boundaries, vec![2]);
        let incoming_boundaries = Edge::get_block_boundaries(None, Some(&vec![&edge2]));
        assert_eq!(incoming_boundaries, vec![3]);
    }
}
