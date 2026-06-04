use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
    rc::Rc,
};

use gen_core::{
    HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand, calculate_hash, is_terminal,
    traits::Capnp,
};
use gen_graph::{GenGraph, GraphEdge, GraphNode};
use indexmap::IndexSet;
use itertools::Itertools;
use rusqlite::{Row, params, types::Value};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    block_group_edge::AugmentedEdge,
    db::GraphConnection,
    errors::NodeError,
    gen_models_capnp::edge,
    node::Node,
    sequence::{Sequence, SequenceError, cached_sequence},
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
    external_sequence: Option<(String, String, bool)>,
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
                external_sequence: Some((
                    sequence.file_path.clone(),
                    sequence.name.clone(),
                    sequence.sequence_type.eq_ignore_ascii_case("circular"),
                )),
                start,
                end,
            }
        } else {
            GroupBlock {
                id,
                node_id,
                sequence: Some(sequence.get_sequence(start, end).unwrap()),
                external_sequence: None,
                start,
                end,
            }
        }
    }

    pub fn sequence(&self) -> String {
        if let Some(sequence) = &self.sequence {
            sequence.to_string()
        } else if let Some((path, name, circular)) = &self.external_sequence {
            cached_sequence(path, name, self.start, self.end, *circular).unwrap()
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
    #[error("Sequence error: {0}")]
    SequenceError(#[from] SequenceError),
    #[error("Cannot build blocks from starts {starts:?} and ends {ends:?}")]
    BlockIntervalError {
        starts: HashSet<i64>,
        ends: HashSet<i64>,
    },
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

    pub fn edges_for_block_group_nodes(
        conn: &GraphConnection,
        block_group_id: &HashId,
        node_ids: &[HashId],
    ) -> Result<Vec<AugmentedEdge>, EdgeError> {
        if node_ids.is_empty() {
            return Ok(vec![]);
        }

        let mut edges = vec![];
        let batch_size = max_rows_per_batch(conn, 1);
        let query = "\
            SELECT
                e.id,
                e.source_node_id,
                e.source_coordinate,
                e.source_strand,
                e.target_node_id,
                e.target_coordinate,
                e.target_strand,
                bge.chromosome_index,
                bge.phased,
                bge.created_on
            FROM block_group_edges bge
            JOIN edges e ON e.id = bge.edge_id
            WHERE bge.block_group_id = ?1
              AND (e.source_node_id IN rarray(?2) OR e.target_node_id IN rarray(?2))
            ORDER BY bge.created_on DESC;";

        for chunk in node_ids.chunks(batch_size) {
            let values = chunk.iter().copied().map(Value::from).collect::<Vec<_>>();
            let mut stmt = conn.prepare_cached(query)?;
            let rows = stmt.query_map(params![block_group_id, Rc::new(values)], |row| {
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
            })?;

            for row in rows {
                edges.push(row?);
            }
        }

        Ok(edges)
    }

    fn get_block_intervals(
        starts: &HashSet<i64>,
        ends: &HashSet<i64>,
    ) -> Result<Vec<(i64, i64)>, EdgeError> {
        let coordinates = starts.union(ends).sorted().copied().collect::<Vec<_>>();
        if coordinates.is_empty() {
            return Err(EdgeError::BlockIntervalError {
                starts: starts.clone(),
                ends: ends.clone(),
            });
        }
        if coordinates.len() == 1 {
            return Ok(vec![(coordinates[0], coordinates[0])]);
        }

        Ok(coordinates.into_iter().tuple_windows().collect())
    }

    pub fn blocks_from_edges(
        conn: &GraphConnection,
        block_group_id: &HashId,
        edges: &[AugmentedEdge],
    ) -> Result<Vec<GroupBlock>, EdgeError> {
        let mut node_ids = IndexSet::new();
        let mut starts_by_node_id: HashMap<HashId, HashSet<i64>> = HashMap::new();
        let mut ends_by_node_id: HashMap<HashId, HashSet<i64>> = HashMap::new();
        for edge in edges.iter().map(|edge| &edge.edge) {
            if !is_terminal(edge.source_node_id) {
                node_ids.insert(edge.source_node_id);
            }
            ends_by_node_id
                .entry(edge.source_node_id)
                .or_default()
                .insert(edge.source_coordinate);

            if !is_terminal(edge.target_node_id) {
                node_ids.insert(edge.target_node_id);
            }
            starts_by_node_id
                .entry(edge.target_node_id)
                .or_default()
                .insert(edge.target_coordinate);
        }

        let is_incomplete =
            |node_id: &HashId,
             starts_by_node_id: &HashMap<HashId, HashSet<i64>>,
             ends_by_node_id: &HashMap<HashId, HashSet<i64>>| {
                let has_starts = starts_by_node_id
                    .get(node_id)
                    .is_some_and(|starts| !starts.is_empty());
                let has_ends = ends_by_node_id
                    .get(node_id)
                    .is_some_and(|ends| !ends.is_empty());
                has_starts != has_ends
            };
        let mut queried_node_ids = HashSet::new();
        let mut incomplete_node_ids = node_ids
            .iter()
            .copied()
            .filter(|node_id| is_incomplete(node_id, &starts_by_node_id, &ends_by_node_id))
            .collect::<Vec<_>>();

        while !incomplete_node_ids.is_empty() {
            queried_node_ids.extend(incomplete_node_ids.iter().copied());
            let mut next_incomplete_node_ids = HashSet::new();

            for edge in
                Edge::edges_for_block_group_nodes(conn, block_group_id, &incomplete_node_ids)?
                    .iter()
                    .map(|augmented_edge| &augmented_edge.edge)
            {
                if !is_terminal(edge.source_node_id) {
                    node_ids.insert(edge.source_node_id);
                }
                ends_by_node_id
                    .entry(edge.source_node_id)
                    .or_default()
                    .insert(edge.source_coordinate);

                if !is_terminal(edge.target_node_id) {
                    node_ids.insert(edge.target_node_id);
                }
                starts_by_node_id
                    .entry(edge.target_node_id)
                    .or_default()
                    .insert(edge.target_coordinate);

                for candidate_node_id in [edge.source_node_id, edge.target_node_id] {
                    if !is_terminal(candidate_node_id)
                        && !queried_node_ids.contains(&candidate_node_id)
                        && is_incomplete(&candidate_node_id, &starts_by_node_id, &ends_by_node_id)
                    {
                        next_incomplete_node_ids.insert(candidate_node_id);
                    }
                }
            }

            incomplete_node_ids = next_incomplete_node_ids.into_iter().collect();
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
            let empty_starts = HashSet::new();
            let empty_ends = HashSet::new();
            let starts = starts_by_node_id.get(node_id).unwrap_or(&empty_starts);
            let ends = ends_by_node_id.get(node_id).unwrap_or(&empty_ends);
            let block_intervals = Edge::get_block_intervals(starts, ends)?;

            for (start, end) in block_intervals {
                blocks.push(GroupBlock::new(block_index, *node_id, sequence, start, end));
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
        Ok(blocks)
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
        block_group::{BlockGroup, BlockGroupChange},
        block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
        collection::Collection,
        region::ResolvedGenRegion,
        sequence::Sequence,
        test_helpers::{get_connection, setup_block_group},
    };

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

    #[test]
    fn test_get_block_intervals_splits_sorted_unique_coordinates() {
        let starts = HashSet::from([5, 0, 5]);
        let ends = HashSet::from([10, 3]);

        let intervals = Edge::get_block_intervals(&starts, &ends).unwrap();

        assert_eq!(intervals, vec![(0, 3), (3, 5), (5, 10)]);
    }

    #[test]
    fn test_get_block_intervals_single_coordinate_creates_anchor_block() {
        let starts = HashSet::from([3]);
        let ends = HashSet::new();

        let intervals = Edge::get_block_intervals(&starts, &ends).unwrap();

        assert_eq!(intervals, vec![(3, 3)]);
    }

    #[test]
    fn test_get_block_intervals_errors_without_coordinates() {
        assert!(matches!(
            Edge::get_block_intervals(&HashSet::new(), &HashSet::new()),
            Err(EdgeError::BlockIntervalError { .. })
        ));
    }

    #[test]
    fn test_bulk_create() {
        let conn = &mut get_connection(None).unwrap();
        Collection::create(conn, "test collection").unwrap();
        let sequence1 = Sequence::new()
            .sequence_type("DNA")
            .sequence("ATCGATCG")
            .save(conn)
            .unwrap();
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
            .save(conn)
            .unwrap();
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
            .save(conn)
            .unwrap();
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
            .save(conn)
            .unwrap();
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
    fn test_edges_for_block_group_nodes_filters_by_block_group_and_node_ids() {
        let conn = get_connection(None).unwrap();
        Collection::get_or_create(&conn, "test").unwrap();
        crate::sample::Sample::get_or_create(
            &conn,
            crate::sample::NewSample {
                name: "test",
                ..Default::default()
            },
        )
        .unwrap();
        let bg = BlockGroup::create(
            &conn,
            crate::block_group::NewBlockGroup {
                collection_name: "test",
                sample_name: "test",
                name: "query-bg",
                ..Default::default()
            },
        )
        .unwrap();
        let other_bg = BlockGroup::create(
            &conn,
            crate::block_group::NewBlockGroup {
                collection_name: "test",
                sample_name: "test",
                name: "other-bg",
                ..Default::default()
            },
        )
        .unwrap();

        let seq_a = Sequence::new()
            .sequence_type("DNA")
            .sequence("AAA")
            .save(&conn)
            .unwrap();
        let seq_b = Sequence::new()
            .sequence_type("DNA")
            .sequence("CCC")
            .save(&conn)
            .unwrap();
        let seq_c = Sequence::new()
            .sequence_type("DNA")
            .sequence("GGG")
            .save(&conn)
            .unwrap();
        let n_a = Node::create(&conn, &seq_a.hash, &HashId::convert_str("node-a")).unwrap();
        let n_b = Node::create(&conn, &seq_b.hash, &HashId::convert_str("node-b")).unwrap();
        let n_c = Node::create(&conn, &seq_c.hash, &HashId::convert_str("node-c")).unwrap();

        let source_match =
            Edge::create(&conn, n_a, 3, Strand::Forward, n_b, 0, Strand::Forward).unwrap();
        let target_match =
            Edge::create(&conn, n_c, 3, Strand::Forward, n_a, 0, Strand::Forward).unwrap();
        let no_match =
            Edge::create(&conn, n_b, 3, Strand::Forward, n_c, 0, Strand::Forward).unwrap();
        let other_bg_match = Edge::create(
            &conn,
            PATH_START_NODE_ID,
            -1,
            Strand::Forward,
            n_a,
            0,
            Strand::Forward,
        )
        .unwrap();

        let block_group_edges = [source_match.clone(), target_match.clone(), no_match.clone()]
            .iter()
            .map(|edge| BlockGroupEdgeData {
                block_group_id: bg.id,
                edge_id: edge.id,
                chromosome_index: 0,
                phased: 0,
            })
            .chain(std::iter::once(BlockGroupEdgeData {
                block_group_id: other_bg.id,
                edge_id: other_bg_match.id,
                chromosome_index: 0,
                phased: 0,
            }))
            .collect::<Vec<_>>();
        BlockGroupEdge::bulk_create(&conn, &block_group_edges);

        let edges = Edge::edges_for_block_group_nodes(&conn, &bg.id, &[n_a]).unwrap();
        let edge_ids = edges
            .iter()
            .map(|augmented_edge| augmented_edge.edge.id)
            .collect::<HashSet<_>>();

        assert_eq!(edge_ids.len(), 2);
        assert!(edge_ids.contains(&source_match.id));
        assert!(edge_ids.contains(&target_match.id));
        assert!(!edge_ids.contains(&no_match.id));
        assert!(!edge_ids.contains(&other_bg_match.id));
    }

    #[test]
    fn test_blocks_from_edges_branched_graph() {
        // Branched graph: {AAA,GGG} → TTT → {CCC,ATC}
        // TTT has 2 incoming edges and 2 outgoing edges.
        // All sequences are length 3.
        let conn = get_connection(None).unwrap();
        Collection::get_or_create(&conn, "test").unwrap();
        crate::sample::Sample::get_or_create(
            &conn,
            crate::sample::NewSample {
                name: "test",
                ..Default::default()
            },
        )
        .unwrap();
        let bg = BlockGroup::create(
            &conn,
            crate::block_group::NewBlockGroup {
                collection_name: "test",
                sample_name: "test",
                name: "branched",
                ..Default::default()
            },
        )
        .unwrap();

        let seq_aaa = Sequence::new()
            .sequence_type("DNA")
            .sequence("AAA")
            .save(&conn)
            .unwrap();
        let seq_ggg = Sequence::new()
            .sequence_type("DNA")
            .sequence("GGG")
            .save(&conn)
            .unwrap();
        let seq_ttt = Sequence::new()
            .sequence_type("DNA")
            .sequence("TTT")
            .save(&conn)
            .unwrap();
        let seq_ccc = Sequence::new()
            .sequence_type("DNA")
            .sequence("CCC")
            .save(&conn)
            .unwrap();
        let seq_atc = Sequence::new()
            .sequence_type("DNA")
            .sequence("ATC")
            .save(&conn)
            .unwrap();

        let n_aaa = Node::create(&conn, &seq_aaa.hash, &HashId::convert_str("node-aaa")).unwrap();
        let n_ggg = Node::create(&conn, &seq_ggg.hash, &HashId::convert_str("node-ggg")).unwrap();
        let n_ttt = Node::create(&conn, &seq_ttt.hash, &HashId::convert_str("node-ttt")).unwrap();
        let n_ccc = Node::create(&conn, &seq_ccc.hash, &HashId::convert_str("node-ccc")).unwrap();
        let n_atc = Node::create(&conn, &seq_atc.hash, &HashId::convert_str("node-atc")).unwrap();

        let e_start_aaa = Edge::create(
            &conn,
            PATH_START_NODE_ID,
            -1,
            Strand::Forward,
            n_aaa,
            0,
            Strand::Forward,
        )
        .unwrap();
        let e_start_ggg = Edge::create(
            &conn,
            PATH_START_NODE_ID,
            -1,
            Strand::Forward,
            n_ggg,
            0,
            Strand::Forward,
        )
        .unwrap();
        // Edges: AAA→TTT, GGG→TTT, TTT→CCC, TTT→ATC
        let e_aaa_ttt =
            Edge::create(&conn, n_aaa, 3, Strand::Forward, n_ttt, 0, Strand::Forward).unwrap();
        let e_ggg_ttt =
            Edge::create(&conn, n_ggg, 3, Strand::Forward, n_ttt, 0, Strand::Forward).unwrap();
        let e_ttt_ccc =
            Edge::create(&conn, n_ttt, 3, Strand::Forward, n_ccc, 0, Strand::Forward).unwrap();
        let e_ttt_atc =
            Edge::create(&conn, n_ttt, 3, Strand::Forward, n_atc, 0, Strand::Forward).unwrap();
        let e_ccc_end = Edge::create(
            &conn,
            n_ccc,
            3,
            Strand::Forward,
            PATH_END_NODE_ID,
            -1,
            Strand::Forward,
        )
        .unwrap();
        let e_atc_end = Edge::create(
            &conn,
            n_atc,
            3,
            Strand::Forward,
            PATH_END_NODE_ID,
            -1,
            Strand::Forward,
        )
        .unwrap();

        let block_group_edges = [
            e_start_aaa.clone(),
            e_start_ggg.clone(),
            e_aaa_ttt.clone(),
            e_ggg_ttt.clone(),
            e_ttt_ccc.clone(),
            e_ttt_atc.clone(),
            e_ccc_end.clone(),
            e_atc_end.clone(),
        ]
        .iter()
        .map(|edge| BlockGroupEdgeData {
            block_group_id: bg.id,
            edge_id: edge.id,
            chromosome_index: 0,
            phased: 0,
        })
        .collect::<Vec<_>>();
        BlockGroupEdge::bulk_create(&conn, &block_group_edges);

        let augmented_edges = vec![
            AugmentedEdge {
                edge: e_aaa_ttt,
                chromosome_index: 0,
                phased: 0,
                created_on: 0,
            },
            AugmentedEdge {
                edge: e_ggg_ttt,
                chromosome_index: 0,
                phased: 0,
                created_on: 0,
            },
            AugmentedEdge {
                edge: e_ttt_ccc,
                chromosome_index: 0,
                phased: 0,
                created_on: 0,
            },
            AugmentedEdge {
                edge: e_ttt_atc,
                chromosome_index: 0,
                phased: 0,
                created_on: 0,
            },
        ];

        let blocks = Edge::blocks_from_edges(&conn, &bg.id, &augmented_edges).unwrap();

        // 5 non-terminal nodes: AAA, GGG, TTT, CCC, ATC
        // 2 terminal blocks: START, END
        // Total: 7
        assert_eq!(
            blocks.len(),
            7,
            "expected 7 blocks (5 nodes + 2 terminals), got {}",
            blocks.len()
        );

        // Verify all 5 non-terminal nodes have blocks
        let block_node_ids: Vec<HashId> = blocks
            .iter()
            .filter(|b| {
                b.node_id != gen_core::PATH_START_NODE_ID && b.node_id != gen_core::PATH_END_NODE_ID
            })
            .map(|b| b.node_id)
            .collect();
        assert!(block_node_ids.contains(&n_aaa), "AAA block missing");
        assert!(block_node_ids.contains(&n_ggg), "GGG block missing");
        assert!(block_node_ids.contains(&n_ttt), "TTT block missing");
        assert!(block_node_ids.contains(&n_ccc), "CCC block missing");
        assert!(block_node_ids.contains(&n_atc), "ATC block missing");

        // Verify TTT block has correct boundaries (0..3)
        let ttt_block = blocks.iter().find(|b| b.node_id == n_ttt).unwrap();
        assert_eq!(ttt_block.start, 0, "TTT block start should be 0");
        assert_eq!(ttt_block.end, 3, "TTT block end should be 3");

        // Verify AAA block has correct boundaries (0..3)
        let aaa_block = blocks.iter().find(|b| b.node_id == n_aaa).unwrap();
        assert_eq!(aaa_block.start, 0, "AAA block start should be 0");
        assert_eq!(aaa_block.end, 3, "AAA block end should be 3");

        // Verify CCC block has correct boundaries (0..3)
        let ccc_block = blocks.iter().find(|b| b.node_id == n_ccc).unwrap();
        assert_eq!(ccc_block.start, 0, "CCC block start should be 0");
        assert_eq!(ccc_block.end, 3, "CCC block end should be 3");

        // Verify ATC block has correct boundaries (0..3)
        let atc_block = blocks.iter().find(|b| b.node_id == n_atc).unwrap();
        assert_eq!(atc_block.start, 0, "ATC block start should be 0");
        assert_eq!(atc_block.end, 3, "ATC block end should be 3");
    }

    #[test]
    fn test_blocks_from_edges_uses_edge_coordinates_not_backing_sequence_length() {
        // This test ensures we do not use sequence_start or sequence_end to get coordinates by making
        // a test case where the graph has no nodes that are the sequence length. This was leading to
        // technically incorrect behavior by adding blocks of [0, sequence_length] when that was just
        // a happy coincidence it worked
        let conn = get_connection(None).unwrap();
        Collection::get_or_create(&conn, "test").unwrap();
        crate::sample::Sample::get_or_create(
            &conn,
            crate::sample::NewSample {
                name: "test",
                ..Default::default()
            },
        )
        .unwrap();
        let bg = BlockGroup::create(
            &conn,
            crate::block_group::NewBlockGroup {
                collection_name: "test",
                sample_name: "test",
                name: "slice",
                ..Default::default()
            },
        )
        .unwrap();

        let backing_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("AAATTT")
            .save(&conn)
            .unwrap();
        let node_id =
            Node::create(&conn, &backing_sequence.hash, &HashId::convert_str("slice")).unwrap();

        let into_slice = Edge::create(
            &conn,
            PATH_START_NODE_ID,
            -1,
            Strand::Forward,
            node_id,
            3,
            Strand::Forward,
        )
        .unwrap();
        let out_of_slice = Edge::create(
            &conn,
            node_id,
            4,
            Strand::Forward,
            PATH_END_NODE_ID,
            -1,
            Strand::Forward,
        )
        .unwrap();
        let block_group_edges = [into_slice.clone(), out_of_slice.clone()]
            .iter()
            .map(|edge| BlockGroupEdgeData {
                block_group_id: bg.id,
                edge_id: edge.id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<_>>();
        BlockGroupEdge::bulk_create(&conn, &block_group_edges);

        let augmented_edges = vec![
            AugmentedEdge {
                edge: into_slice,
                chromosome_index: 0,
                phased: 0,
                created_on: 0,
            },
            AugmentedEdge {
                edge: out_of_slice,
                chromosome_index: 0,
                phased: 0,
                created_on: 0,
            },
        ];

        let blocks = Edge::blocks_from_edges(&conn, &bg.id, &augmented_edges).unwrap();
        let node_blocks = blocks
            .iter()
            .filter(|block| block.node_id == node_id)
            .collect::<Vec<_>>();

        assert_eq!(
            node_blocks.len(),
            1,
            "expected one edge-delimited block, got {node_blocks:?}"
        );
        assert_eq!(node_blocks[0].start, 3);
        assert_eq!(node_blocks[0].end, 4);
    }

    #[test]
    fn test_blocks_from_edges_does_not_treat_asymmetric_boundaries_as_incomplete() {
        // It's possible that the passed in set of edges is deliberately incomplete.
        // This asserts we do not expand the graph if node information is present.
        let conn = get_connection(None).unwrap();
        Collection::get_or_create(&conn, "test").unwrap();
        crate::sample::Sample::get_or_create(
            &conn,
            crate::sample::NewSample {
                name: "test",
                ..Default::default()
            },
        )
        .unwrap();
        let bg = BlockGroup::create(
            &conn,
            crate::block_group::NewBlockGroup {
                collection_name: "test",
                sample_name: "test",
                name: "asymmetric",
                ..Default::default()
            },
        )
        .unwrap();

        let seq_src = Sequence::new()
            .sequence_type("DNA")
            .sequence("AAA")
            .save(&conn)
            .unwrap();
        let seq_mid = Sequence::new()
            .sequence_type("DNA")
            .sequence("AAACCCGGG")
            .save(&conn)
            .unwrap();
        let seq_a = Sequence::new()
            .sequence_type("DNA")
            .sequence("CCC")
            .save(&conn)
            .unwrap();
        let seq_b = Sequence::new()
            .sequence_type("DNA")
            .sequence("GGG")
            .save(&conn)
            .unwrap();
        let seq_extra = Sequence::new()
            .sequence_type("DNA")
            .sequence("TTT")
            .save(&conn)
            .unwrap();

        let n_src = Node::create(&conn, &seq_src.hash, &HashId::convert_str("node-src")).unwrap();
        let n_mid = Node::create(&conn, &seq_mid.hash, &HashId::convert_str("node-mid")).unwrap();
        let n_a = Node::create(&conn, &seq_a.hash, &HashId::convert_str("node-a")).unwrap();
        let n_b = Node::create(&conn, &seq_b.hash, &HashId::convert_str("node-b")).unwrap();
        let n_extra =
            Node::create(&conn, &seq_extra.hash, &HashId::convert_str("node-extra")).unwrap();

        let e_src_mid =
            Edge::create(&conn, n_src, 3, Strand::Forward, n_mid, 0, Strand::Forward).unwrap();
        let e_mid_a =
            Edge::create(&conn, n_mid, 3, Strand::Forward, n_a, 0, Strand::Forward).unwrap();
        let e_mid_b =
            Edge::create(&conn, n_mid, 6, Strand::Forward, n_b, 0, Strand::Forward).unwrap();
        // We exclude this edge from graph construction
        let e_mid_extra = Edge::create(
            &conn,
            n_mid,
            9,
            Strand::Forward,
            n_extra,
            0,
            Strand::Forward,
        )
        .unwrap();

        let block_group_edges = [
            e_src_mid.clone(),
            e_mid_a.clone(),
            e_mid_b.clone(),
            e_mid_extra,
        ]
        .iter()
        .map(|edge| BlockGroupEdgeData {
            block_group_id: bg.id,
            edge_id: edge.id,
            chromosome_index: 0,
            phased: 0,
        })
        .collect::<Vec<_>>();
        BlockGroupEdge::bulk_create(&conn, &block_group_edges);

        let augmented_edges = vec![
            AugmentedEdge {
                edge: e_src_mid,
                chromosome_index: 0,
                phased: 0,
                created_on: 0,
            },
            AugmentedEdge {
                edge: e_mid_a,
                chromosome_index: 0,
                phased: 0,
                created_on: 0,
            },
            AugmentedEdge {
                edge: e_mid_b,
                chromosome_index: 0,
                phased: 0,
                created_on: 0,
            },
        ];

        let blocks = Edge::blocks_from_edges(&conn, &bg.id, &augmented_edges).unwrap();
        let mid_intervals = blocks
            .iter()
            .filter(|block| block.node_id == n_mid)
            .map(|block| (block.start, block.end))
            .collect::<Vec<_>>();

        assert_eq!(mid_intervals, vec![(0, 3), (3, 6)]);
        assert!(!blocks.iter().any(|block| block.node_id == n_extra));
    }

    #[test]
    fn test_blocks_from_edges_resolves_incomplete_nodes_created_by_lookup() {
        // If a node is present as just a source or sink node, ensure we look it up to
        // accurately add it to the graph.
        let conn = get_connection(None).unwrap();
        Collection::get_or_create(&conn, "test").unwrap();
        crate::sample::Sample::get_or_create(
            &conn,
            crate::sample::NewSample {
                name: "test",
                ..Default::default()
            },
        )
        .unwrap();
        let bg = BlockGroup::create(
            &conn,
            crate::block_group::NewBlockGroup {
                collection_name: "test",
                sample_name: "test",
                name: "recursive-incomplete",
                ..Default::default()
            },
        )
        .unwrap();

        let seq_a = Sequence::new()
            .sequence_type("DNA")
            .sequence("AAA")
            .save(&conn)
            .unwrap();
        let seq_b = Sequence::new()
            .sequence_type("DNA")
            .sequence("BBB")
            .save(&conn)
            .unwrap();
        let seq_c = Sequence::new()
            .sequence_type("DNA")
            .sequence("CCC")
            .save(&conn)
            .unwrap();
        let seq_d = Sequence::new()
            .sequence_type("DNA")
            .sequence("DDD")
            .save(&conn)
            .unwrap();

        let n_a = Node::create(&conn, &seq_a.hash, &HashId::convert_str("node-a")).unwrap();
        let n_b = Node::create(&conn, &seq_b.hash, &HashId::convert_str("node-b")).unwrap();
        let n_c = Node::create(&conn, &seq_c.hash, &HashId::convert_str("node-c")).unwrap();
        let n_d = Node::create(&conn, &seq_d.hash, &HashId::convert_str("node-d")).unwrap();

        let e_a_b = Edge::create(&conn, n_a, 3, Strand::Forward, n_b, 0, Strand::Forward).unwrap();
        let e_b_c = Edge::create(&conn, n_b, 3, Strand::Forward, n_c, 0, Strand::Forward).unwrap();
        let e_c_d = Edge::create(&conn, n_c, 3, Strand::Forward, n_d, 0, Strand::Forward).unwrap();

        let block_group_edges = [e_a_b.clone(), e_b_c.clone(), e_c_d.clone()]
            .iter()
            .map(|edge| BlockGroupEdgeData {
                block_group_id: bg.id,
                edge_id: edge.id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<_>>();
        BlockGroupEdge::bulk_create(&conn, &block_group_edges);

        let blocks = Edge::blocks_from_edges(
            &conn,
            &bg.id,
            &[AugmentedEdge {
                edge: e_a_b,
                chromosome_index: 0,
                phased: 0,
                created_on: 0,
            }],
        )
        .unwrap();

        let b_block = blocks.iter().find(|block| block.node_id == n_b).unwrap();
        assert_eq!(b_block.start, 0);
        assert_eq!(b_block.end, 3);

        let c_block = blocks.iter().find(|block| block.node_id == n_c).unwrap();
        assert_eq!(c_block.start, 0);
        assert_eq!(c_block.end, 3);
    }

    #[test]
    fn test_bulk_create_with_existing_edge() {
        let conn = &mut get_connection(None).unwrap();
        Collection::create(conn, "test collection").unwrap();
        let sequence1 = Sequence::new()
            .sequence_type("DNA")
            .sequence("ATCGATCG")
            .save(conn)
            .unwrap();
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
            .save(conn)
            .unwrap();
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
        let (block_group_id, path) = setup_block_group(&conn);

        let edges = BlockGroupEdge::edges_for_block_group(&conn, &block_group_id);
        let blocks = Edge::blocks_from_edges(&conn, &block_group_id, &edges).unwrap();

        // 4 actual sequences: 10-length ones of all A, all T, all C, all G
        // 2 terminal node blocks (start/end)
        // 6 total
        assert_eq!(blocks.len(), 6);

        let insert_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("NNNN")
            .save(&conn)
            .unwrap();
        let insert_node_id =
            Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("1")).unwrap();
        let insert = PathBlock {
            node_id: insert_node_id,
            block_sequence: insert_sequence.get_sequence(0, 4).unwrap(),
            sequence_start: 0,
            sequence_end: 4,
            path_start: 7,
            path_end: 15,
            strand: Strand::Forward,
        };
        let region = ResolvedGenRegion::from_path(&conn, block_group_id, &path, 7, 15).unwrap();
        let change = BlockGroupChange {
            region,
            path_accession: None,
            block: insert,
            chromosome_index: 0,
            phased: 0,
            preserve_edge: true,
        };
        BlockGroup::insert_change(&conn, &change).unwrap();
        let mut edges = BlockGroupEdge::edges_for_block_group(&conn, &block_group_id);

        let blocks = Edge::blocks_from_edges(&conn, &block_group_id, &edges).unwrap();

        // 2 10-length sequences of all C, all G
        // 1 inserted NNNN sequence
        // 4 split blocks (A and T sequences were split) resulting from the inserted sequence
        // 2 terminal node blocks (start/end)
        // 9 total
        assert_eq!(blocks.len(), 9);

        // Confirm that ordering doesn't matter
        edges.reverse();
        let blocks = Edge::blocks_from_edges(&conn, &block_group_id, &edges).unwrap();

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
            .save(&conn)
            .unwrap();
        let template_node_id =
            Node::create(&conn, &template_sequence.hash, &HashId::convert_str("1")).unwrap();

        let insert_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("NNNN")
            .save(&conn)
            .unwrap();
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

        let boundaries = get_block_boundaries(Some(&vec![&edge1]), Some(&vec![&edge2]));
        assert_eq!(boundaries, vec![2, 3]);
    }

    #[test]
    fn test_get_block_boundaries_with_two_original_sequences() {
        let conn = get_connection(None).unwrap();
        let template_sequence1 = Sequence::new()
            .sequence_type("DNA")
            .sequence("AAAAAAAAAA")
            .save(&conn)
            .unwrap();
        let template1_node_id =
            Node::create(&conn, &template_sequence1.hash, &HashId::convert_str("1")).unwrap();

        let template_sequence2 = Sequence::new()
            .sequence_type("DNA")
            .sequence("TTTTTTTTTT")
            .save(&conn)
            .unwrap();
        let template2_node_id =
            Node::create(&conn, &template_sequence2.hash, &HashId::convert_str("2")).unwrap();

        let insert_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("NNNN")
            .save(&conn)
            .unwrap();
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

        let outgoing_boundaries = get_block_boundaries(Some(&vec![&edge1]), None);
        assert_eq!(outgoing_boundaries, vec![2]);
        let incoming_boundaries = get_block_boundaries(None, Some(&vec![&edge2]));
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
