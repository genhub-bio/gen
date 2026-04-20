use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
};

use gen_core::{
    HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand, calculate_hash, is_end_node,
    is_start_node, traits::Capnp,
};
use gen_graph::{GenGraph, GraphEdge, GraphNode};
use indexmap::IndexSet;
use itertools::Itertools;
use rusqlite::{Row, params};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    block_group_edge::AugmentedEdge,
    db::GraphConnection,
    gen_models_capnp::edge,
    node::{Node, NodeError},
    sequence::{Sequence, cached_sequence},
    traits::*,
};

#[derive(Clone, Debug, Eq, Hash, PartialEq, Deserialize, Serialize, Ord, PartialOrd)]
pub struct Edge {
    pub id: HashId,
    pub source_node_id: HashId,
    pub source_coordinate: i64,
    pub source_strand: Strand,
    pub target_node_id: HashId,
    pub target_coordinate: i64,
    pub target_strand: Strand,
}

impl<'a> Capnp<'a> for Edge {
    type Builder = edge::Builder<'a>;
    type Reader = edge::Reader<'a>;

    fn write_capnp(&self, builder: &mut Self::Builder) {
        builder.set_id(&self.id.0).unwrap();
        builder.set_source_node_id(&self.source_node_id.0).unwrap();
        builder.set_source_coordinate(self.source_coordinate);
        builder.set_source_strand(self.source_strand.into());
        builder.set_target_node_id(&self.target_node_id.0).unwrap();
        builder.set_target_coordinate(self.target_coordinate);
        builder.set_target_strand(self.target_strand.into());
    }

    fn read_capnp(reader: Self::Reader) -> Self {
        let id: HashId = reader
            .get_id()
            .unwrap()
            .as_slice()
            .unwrap()
            .try_into()
            .unwrap();
        let source_node_id = reader
            .get_source_node_id()
            .unwrap()
            .as_slice()
            .unwrap()
            .try_into()
            .unwrap();
        let source_coordinate = reader.get_source_coordinate();
        let source_strand = reader.get_source_strand().unwrap().into();
        let target_node_id = reader
            .get_target_node_id()
            .unwrap()
            .as_slice()
            .unwrap()
            .try_into()
            .unwrap();
        let target_coordinate = reader.get_target_coordinate();
        let target_strand = reader.get_target_strand().unwrap().into();

        Edge {
            id,
            source_node_id,
            source_coordinate,
            source_strand,
            target_node_id,
            target_coordinate,
            target_strand,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd)]
pub struct EdgeData {
    pub source_node_id: HashId,
    pub source_coordinate: i64,
    pub source_strand: Strand,
    pub target_node_id: HashId,
    pub target_coordinate: i64,
    pub target_strand: Strand,
}

impl EdgeData {
    pub fn id_hash(&self) -> HashId {
        HashId(calculate_hash(&format!(
            "{}:{}:{}:{}:{}:{}",
            self.source_node_id,
            self.source_coordinate,
            self.source_strand,
            self.target_node_id,
            self.target_coordinate,
            self.target_strand,
        )))
    }
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
    pub node_id: HashId,
    pub coordinate: i64,
}

#[derive(Clone, Debug)]
pub struct GroupBlock {
    pub id: i64,
    pub node_id: HashId,
    sequence: Option<String>,
    external_sequence: Option<(String, String)>,
    pub start: i64,
    pub end: i64,
}

impl GroupBlock {
    pub fn new(id: i64, node_id: HashId, sequence: &Sequence, start: i64, end: i64) -> Self {
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

    const TABLE_NAME: &'static str = "edges";

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

#[derive(Debug, Error, PartialEq)]
pub enum EdgeError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] rusqlite::Error),
    #[error("Node creation error: {0}")]
    NodeError(#[from] NodeError),
}

impl Edge {
    #[allow(clippy::too_many_arguments)]
    pub fn create(
        conn: &GraphConnection,
        source_node_id: HashId,
        source_coordinate: i64,
        source_strand: Strand,
        target_node_id: HashId,
        target_coordinate: i64,
        target_strand: Strand,
    ) -> Result<Edge, EdgeError> {
        let hash = HashId(calculate_hash(&format!(
            "{source_node_id}:{source_coordinate}:{source_strand}:{target_node_id}:{target_coordinate}:{target_strand}"
        )));
        let query = "INSERT INTO edges (id, source_node_id, source_coordinate, source_strand, target_node_id, target_coordinate, target_strand) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7);";
        let mut stmt = match conn.prepare(query) {
            Ok(s) => s,
            Err(e) => return Err(EdgeError::DatabaseError(e)),
        };
        match stmt.execute(params![
            hash,
            source_node_id,
            source_coordinate,
            source_strand,
            target_node_id,
            target_coordinate,
            target_strand
        ]) {
            Ok(_) => Ok(Edge {
                id: hash,
                source_node_id,
                source_coordinate,
                source_strand,
                target_node_id,
                target_coordinate,
                target_strand,
            }),
            Err(rusqlite::Error::SqliteFailure(e, _))
                if e.code == rusqlite::ErrorCode::ConstraintViolation =>
            {
                // Edge already exists, return the existing record
                Ok(Edge {
                    id: hash,
                    source_node_id,
                    source_coordinate,
                    source_strand,
                    target_node_id,
                    target_coordinate,
                    target_strand,
                })
            }
            Err(e) => Err(EdgeError::DatabaseError(e)),
        }
    }

    pub fn bulk_create(conn: &GraphConnection, edges: &[EdgeData]) -> Vec<HashId> {
        let edge_ids = edges.iter().map(|edge| edge.id_hash()).collect::<Vec<_>>();
        let query = Edge::query_by_ids(conn, &edge_ids);
        let existing_edges = query.iter().map(|edge| &edge.id).collect::<HashSet<_>>();

        let mut edges_to_insert = IndexSet::new();
        for (index, edge) in edge_ids.iter().enumerate() {
            if !existing_edges.contains(edge) {
                edges_to_insert.insert(&edges[index]);
            }
        }

        let batch_size = max_rows_per_batch(conn, 7);

        for chunk in &edges_to_insert.iter().chunks(batch_size) {
            let mut rows = vec![];
            let mut params: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
            for edge in chunk {
                params.push(Box::new(edge.id_hash()));
                params.push(Box::new(edge.source_node_id));
                params.push(Box::new(edge.source_coordinate));
                params.push(Box::new(edge.source_strand));
                params.push(Box::new(edge.target_node_id));
                params.push(Box::new(edge.target_coordinate));
                params.push(Box::new(edge.target_strand));
                rows.push("(?, ?, ?, ?, ?, ?, ?)");
            }
            let sql = format!(
                "INSERT INTO edges (id, source_node_id, source_coordinate, source_strand, target_node_id, target_coordinate, target_strand) VALUES {};",
                rows.join(",")
            );
            conn.execute(&sql, rusqlite::params_from_iter(params))
                .unwrap();
        }
        edge_ids
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

    pub fn blocks_from_edges(conn: &GraphConnection, edges: &[AugmentedEdge]) -> Vec<GroupBlock> {
        let mut node_ids = IndexSet::new();
        let mut edges_by_source_node_id: HashMap<HashId, Vec<&Edge>> = HashMap::new();
        let mut edges_by_target_node_id: HashMap<HashId, Vec<&Edge>> = HashMap::new();
        for edge in edges.iter().map(|edge| &edge.edge) {
            if !is_start_node(edge.source_node_id) {
                node_ids.insert(edge.source_node_id);
            }
            edges_by_source_node_id
                .entry(edge.source_node_id)
                .and_modify(|edges| edges.push(edge))
                .or_insert(vec![edge]);

            if !is_end_node(edge.target_node_id) {
                node_ids.insert(edge.target_node_id);
            }
            edges_by_target_node_id
                .entry(edge.target_node_id)
                .and_modify(|edges| edges.push(edge))
                .or_insert(vec![edge]);
        }

        let sequences_by_node_id = Node::get_sequences_by_node_ids(
            conn,
            &node_ids.iter().copied().collect::<Vec<HashId>>(),
        );

        let mut blocks = vec![];
        let mut block_index = 0;
        // we sort by keys to exploit the external sequence cache which keeps the most recently used
        // external sequence in memory.
        for (node_id, sequence) in sequences_by_node_id
            .iter()
            .sorted_by_key(|(_node_id, seq)| seq.hash)
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
        let graph_node_for_block = |block: &GroupBlock| GraphNode {
            node_id: block.node_id,
            sequence_start: block.start,
            sequence_end: block.end,
        };
        let blocks_by_start = blocks
            .iter()
            .map(|block| {
                (
                    BlockKey {
                        node_id: block.node_id,
                        coordinate: block.start,
                    },
                    block,
                )
            })
            .collect::<HashMap<BlockKey, &GroupBlock>>();
        let blocks_by_end = blocks
            .iter()
            .map(|block| {
                (
                    BlockKey {
                        node_id: block.node_id,
                        coordinate: block.end,
                    },
                    block,
                )
            })
            .collect::<HashMap<BlockKey, &GroupBlock>>();

        let mut graph = GenGraph::new();
        let mut edges_by_node_pair = HashMap::new();
        for block in blocks {
            graph.add_node(graph_node_for_block(block));
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

            if let Some(source_block) = source_id
                && let Some(target_block) = target_id
            {
                let source_node = graph_node_for_block(source_block);
                let target_node = graph_node_for_block(target_block);
                let graph_edge = GraphEdge {
                    edge_id: edge.id,
                    source_strand: edge.source_strand,
                    target_strand: edge.target_strand,
                    chromosome_index: augmented_edge.chromosome_index,
                    phased: augmented_edge.phased,
                    created_on: augmented_edge.created_on,
                };
                if let Some(existing_edges) = graph.edge_weight_mut(source_node, target_node) {
                    existing_edges.push(graph_edge);
                } else {
                    graph.add_edge(source_node, target_node, vec![graph_edge]);
                }
                edges_by_node_pair.insert((source_block.id, target_block.id), edge.clone());
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
    use gen_core::PathBlock;

    use super::*;
    use crate::{
        block_group::{BlockGroup, PathChange},
        block_group_edge::BlockGroupEdge,
        collection::Collection,
        sequence::Sequence,
        test_helpers::{get_connection, setup_block_group},
    };

    #[test]
    fn test_bulk_create() {
        let conn = &mut get_connection(None).unwrap();
        Collection::create(conn, "test collection").unwrap();
        let sequence1 = Sequence::new()
            .sequence_type("DNA")
            .sequence("ATCGATCG")
            .save(conn);
        let node1_id = Node::create(conn, &sequence1.hash, &HashId::convert_str("1")).unwrap();
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
        let node2_id = Node::create(conn, &sequence2.hash, &HashId::convert_str("2")).unwrap();
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

        let edge_ids = Edge::bulk_create(conn, &[edge1, edge2, edge3]);
        assert_eq!(edge_ids.len(), 3);
        let edges = Edge::query_by_ids(conn, &edge_ids);
        assert_eq!(edges.len(), 3);

        let edges_by_source_node_id = edges
            .into_iter()
            .map(|edge| (edge.source_node_id, edge))
            .collect::<HashMap<_, Edge>>();

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
        let conn = &mut get_connection(None).unwrap();
        Collection::create(conn, "test collection").unwrap();
        let sequence1 = Sequence::new()
            .sequence_type("DNA")
            .sequence("ATCGATCG")
            .save(conn);
        let node1_id = Node::create(conn, &sequence1.hash, &HashId::convert_str("1")).unwrap();
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
        let node2_id = Node::create(conn, &sequence2.hash, &HashId::convert_str("2")).unwrap();
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
            let edge = Edge::get_by_id(conn, id).unwrap();
            assert_eq!(EdgeData::from(&edge), edges[index]);
        }

        let edges = vec![edge1, edge2, edge3];
        let edge_ids2 = Edge::bulk_create(conn, &edges);
        assert_eq!(edge_ids2[1], edge_ids1[0]);
        assert_eq!(edge_ids2[2], edge_ids1[1]);
        assert_eq!(edge_ids2.len(), 3);
        for (index, id) in edge_ids2.iter().enumerate() {
            let edge = Edge::get_by_id(conn, id).unwrap();
            assert_eq!(EdgeData::from(&edge), edges[index]);
        }
    }

    #[test]
    fn test_bulk_create_with_existing_edge() {
        let conn = &mut get_connection(None).unwrap();
        Collection::create(conn, "test collection").unwrap();
        let sequence1 = Sequence::new()
            .sequence_type("DNA")
            .sequence("ATCGATCG")
            .save(conn);
        let node1_id = Node::create(conn, &sequence1.hash, &HashId::convert_str("1")).unwrap();
        // NOTE: Create one edge ahead of time to confirm an existing row ID gets returned in the bulk create
        let existing_edge = Edge::create(
            conn,
            PATH_START_NODE_ID,
            -1,
            Strand::Forward,
            node1_id,
            1,
            Strand::Forward,
        )
        .unwrap();
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
        let node2_id = Node::create(conn, &sequence2.hash, &HashId::convert_str("2")).unwrap();
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

        let edge_ids = Edge::bulk_create(conn, &[edge1, edge2, edge3]);
        assert_eq!(edge_ids.len(), 3);
        let edges = Edge::query_by_ids(conn, &edge_ids);
        assert_eq!(edges.len(), 3);

        let edges_by_source_node_id = edges
            .into_iter()
            .map(|edge| (edge.source_node_id, edge))
            .collect::<HashMap<_, Edge>>();

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
        let conn = get_connection(None).unwrap();
        let (block_group_id, path) = setup_block_group(&conn).unwrap();

        let edges = BlockGroupEdge::edges_for_block_group(&conn, &block_group_id);
        let blocks = Edge::blocks_from_edges(&conn, &edges);

        // 4 actual sequences: 10-length ones of all A, all T, all C, all G
        // 2 terminal node blocks (start/end)
        // 6 total
        assert_eq!(blocks.len(), 6);

        let insert_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("NNNN")
            .save(&conn);
        let insert_node_id =
            Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("1")).unwrap();
        let insert = PathBlock {
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
        let mut edges = BlockGroupEdge::edges_for_block_group(&conn, &block_group_id);

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
        let conn = get_connection(None).unwrap();
        let template_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("AAAAAAAAAA")
            .save(&conn);
        let template_node_id =
            Node::create(&conn, &template_sequence.hash, &HashId::convert_str("1")).unwrap();

        let insert_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("NNNN")
            .save(&conn);
        let insert_node_id =
            Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("2")).unwrap();

        let edge1 = Edge::create(
            &conn,
            template_node_id,
            2,
            Strand::Forward,
            insert_node_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge2 = Edge::create(
            &conn,
            insert_node_id,
            4,
            Strand::Forward,
            template_node_id,
            3,
            Strand::Forward,
        )
        .unwrap();

        let boundaries = Edge::get_block_boundaries(Some(&vec![&edge1]), Some(&vec![&edge2]));
        assert_eq!(boundaries, vec![2, 3]);
    }

    #[test]
    fn test_get_block_boundaries_with_two_original_sequences() {
        let conn = get_connection(None).unwrap();
        let template_sequence1 = Sequence::new()
            .sequence_type("DNA")
            .sequence("AAAAAAAAAA")
            .save(&conn);
        let template1_node_id =
            Node::create(&conn, &template_sequence1.hash, &HashId::convert_str("1")).unwrap();

        let template_sequence2 = Sequence::new()
            .sequence_type("DNA")
            .sequence("TTTTTTTTTT")
            .save(&conn);
        let template2_node_id =
            Node::create(&conn, &template_sequence2.hash, &HashId::convert_str("2")).unwrap();

        let insert_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("NNNN")
            .save(&conn);
        let insert_node_id =
            Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("3")).unwrap();

        let edge1 = Edge::create(
            &conn,
            template1_node_id,
            2,
            Strand::Forward,
            insert_node_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge2 = Edge::create(
            &conn,
            insert_node_id,
            4,
            Strand::Forward,
            template2_node_id,
            3,
            Strand::Forward,
        )
        .unwrap();

        let outgoing_boundaries = Edge::get_block_boundaries(Some(&vec![&edge1]), None);
        assert_eq!(outgoing_boundaries, vec![2]);
        let incoming_boundaries = Edge::get_block_boundaries(None, Some(&vec![&edge2]));
        assert_eq!(incoming_boundaries, vec![3]);
    }

    #[test]
    fn test_edge_capnp_serialization() {
        use capnp::message::TypedBuilder;

        let edge = Edge {
            id: HashId::pad_str(789),
            source_node_id: HashId::convert_str("1"),
            source_coordinate: 10,
            source_strand: Strand::Forward,
            target_node_id: HashId::convert_str("2"),
            target_coordinate: 20,
            target_strand: Strand::Reverse,
        };

        let mut message = TypedBuilder::<edge::Owned>::new_default();
        let mut root = message.init_root();
        edge.write_capnp(&mut root);

        let deserialized = Edge::read_capnp(root.into_reader());
        assert_eq!(edge, deserialized);
    }
}
