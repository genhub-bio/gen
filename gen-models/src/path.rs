use core::ops::Range as RustRange;
use std::{
    collections::{HashMap, HashSet},
    rc::Rc,
};

use gen_core::{
    HASH_ID_SIZE, HashId, NodeIntervalBlock, PATH_END_NODE_ID, PATH_START_NODE_ID, PathBlock,
    Strand, calculate_hash, is_end_node, is_start_node,
    range::{Range, RangeMapping},
    region::{Region, RegionResolutionError, RegionResolver},
    traits::Capnp,
};
use intervaltree::IntervalTree;
use itertools::Itertools;
use rusqlite::{
    Row, params,
    types::{Type, Value as SQLValue},
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    block_group_edge::BlockGroupEdge,
    db::GraphConnection,
    edge::Edge,
    errors::QueryError,
    gen_models_capnp::path as PathCapnp,
    node::Node,
    sequence::{Sequence, SequenceError},
    traits::*,
};

#[derive(Clone, Debug, Eq, Hash, PartialEq, Deserialize, Serialize)]
pub struct Path {
    pub id: HashId,
    pub block_group_id: HashId,
    pub name: String,
    pub created_on: i64,
    pub edge_ids: Vec<HashId>,
}

#[derive(Debug, Error, PartialEq)]
pub enum PathError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] rusqlite::Error),
    #[error("Missing path data: {0}")]
    Missing(String),
    #[error("Duplicate entry with uuid: {0}")]
    Duplicate(String),
    #[error(
        "Malformed path edge-ID BLOB length {actual}; expected a multiple of {hash_id_size} bytes"
    )]
    InvalidEdgeBlobLength { actual: usize, hash_id_size: usize },
    #[error("Problem querying for path: {0}")]
    Query(#[from] QueryError),
    #[error("Problem loading sequence for path: {0}")]
    Sequence(#[from] SequenceError),
}

impl<'a> Capnp<'a> for Path {
    type Builder = PathCapnp::Builder<'a>;
    type Reader = PathCapnp::Reader<'a>;

    fn write_capnp(&self, builder: &mut Self::Builder) {
        builder.set_id(&self.id.0).unwrap();
        builder.set_block_group_id(&self.block_group_id.0).unwrap();
        builder.set_name(&self.name);
        builder.set_created_on(self.created_on);
        let encoded_edge_ids = encode_edge_ids(&self.edge_ids);
        builder.set_edge_ids(encoded_edge_ids.as_slice()).unwrap();
    }

    fn read_capnp(reader: Self::Reader) -> Self {
        let id = reader
            .get_id()
            .unwrap()
            .as_slice()
            .unwrap()
            .try_into()
            .unwrap();
        let block_group_id = reader
            .get_block_group_id()
            .unwrap()
            .as_slice()
            .unwrap()
            .try_into()
            .unwrap();
        let name = reader.get_name().unwrap().to_string().unwrap();
        let created_on = reader.get_created_on();
        let edge_ids = decode_edge_ids(reader.get_edge_ids().unwrap().as_slice().unwrap())
            .expect("should contain a valid path edge-ID array");

        Path {
            id,
            block_group_id,
            name,
            created_on,
            edge_ids,
        }
    }
}

fn encode_edge_ids(edge_ids: &[HashId]) -> Vec<u8> {
    let mut encoded = Vec::with_capacity(edge_ids.len() * HASH_ID_SIZE);
    for edge_id in edge_ids {
        encoded.extend_from_slice(&edge_id.0);
    }
    encoded
}

fn decode_edge_ids(encoded: &[u8]) -> Result<Vec<HashId>, PathError> {
    if !encoded.len().is_multiple_of(HASH_ID_SIZE) {
        return Err(PathError::InvalidEdgeBlobLength {
            actual: encoded.len(),
            hash_id_size: HASH_ID_SIZE,
        });
    }

    Ok(encoded
        .chunks_exact(HASH_ID_SIZE)
        .map(|chunk| {
            HashId(
                chunk
                    .try_into()
                    .expect("should contain one complete edge identifier"),
            )
        })
        .collect())
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct PathData {
    pub name: String,
    pub block_group_id: HashId,
}

// interesting gist here: https://gist.github.com/mbhall88/cd900add6335c96127efea0e0f6a9f48, see if we
// can expand this to ambiguous bases/keep case
pub fn revcomp(seq: &str) -> String {
    String::from_utf8(
        seq.chars()
            .rev()
            .map(|c| -> u8 {
                let is_upper = c.is_ascii_uppercase();
                let rc = c as u8;
                let v = if rc == 78 {
                    // N
                    rc
                } else if rc == 110 {
                    // n
                    rc
                } else if rc & 2 != 0 {
                    // CG
                    rc ^ 4
                } else {
                    // AT
                    rc ^ 21
                };
                if is_upper { v } else { v.to_ascii_lowercase() }
            })
            .collect(),
    )
    .unwrap()
}

#[derive(Clone, Debug)]
pub struct Annotation {
    pub name: String,
    pub start: i64,
    pub end: i64,
}

impl Path {
    #[cfg_attr(
        all(debug_assertions, feature = "profiling"),
        tracing::instrument(skip(conn, edge_ids, block_group_id))
    )]
    pub fn validate_edges(conn: &GraphConnection, edge_ids: &[HashId], block_group_id: &HashId) {
        if edge_ids.is_empty() {
            return;
        }

        // Only load the canonical edges requested by this path. GFA imports can have many paths
        // over one large block group, so scanning that whole block group for every path makes path
        // validation quadratic in practice.
        let requested_edge_ids = edge_ids.iter().copied().collect::<HashSet<_>>();
        let unique_edge_ids = requested_edge_ids.iter().copied().collect::<Vec<_>>();
        let augmented_edges =
            BlockGroupEdge::specific_edges_for_block_group(conn, block_group_id, &unique_edge_ids);
        let block_group_edge_ids = augmented_edges
            .iter()
            .map(|augmented_edge| augmented_edge.edge.id)
            .collect::<HashSet<_>>();

        assert!(
            requested_edge_ids.is_subset(&block_group_edge_ids),
            "Not all edges are in the block group ({block_group_id})"
        );

        let edges_by_id = augmented_edges
            .iter()
            .map(|augmented_edge| (augmented_edge.edge.id, &augmented_edge.edge))
            .collect::<HashMap<_, _>>();

        // Two consecutive edges must share a node
        // Two consecutive edges must not go into and out of a node at the same coordinate
        for (edge1_id, edge2_id) in edge_ids.iter().tuple_windows() {
            let edge1 = edges_by_id.get(edge1_id).unwrap();
            let edge2 = edges_by_id.get(edge2_id).unwrap();
            assert!(
                edge1.target_node_id == edge2.source_node_id,
                "Edges {} and {} don't share the same node ({} vs. {})",
                edge1.id,
                edge2.id,
                edge1.target_node_id,
                edge2.source_node_id
            );

            assert!(
                edge1.target_coordinate < edge2.source_coordinate,
                "Source coordinate {} for edge {} is before target coordinate {} for edge {}",
                edge2.source_coordinate,
                edge2.id,
                edge1.target_coordinate,
                edge1.id
            );

            assert!(
                edge1.target_strand == edge2.source_strand,
                "Strand mismatch between consecutive edges {} and {}",
                edge1.id,
                edge2.id,
            );
        }
    }

    #[cfg_attr(
        all(debug_assertions, feature = "profiling"),
        tracing::instrument(skip(conn, name, block_group_id, edge_ids))
    )]
    pub fn create(
        conn: &GraphConnection,
        name: &str,
        block_group_id: &HashId,
        edge_ids: &[HashId],
    ) -> Result<Path, PathError> {
        Path::validate_edges(conn, edge_ids, block_group_id);
        Self::create_unchecked(conn, name, block_group_id, edge_ids)
    }

    /// Creates a path whose ordered edges have already been validated and persisted by the caller.
    #[cfg_attr(
        all(debug_assertions, feature = "profiling"),
        tracing::instrument(skip(conn, name, block_group_id, edge_ids))
    )]
    pub fn create_unchecked(
        conn: &GraphConnection,
        name: &str,
        block_group_id: &HashId,
        edge_ids: &[HashId],
    ) -> Result<Path, PathError> {
        let hash = HashId(calculate_hash(&format!("{block_group_id}:{name}")));
        let timestamp = chrono::Utc::now().timestamp_nanos_opt().unwrap();
        let encoded_edge_ids = encode_edge_ids(edge_ids);
        let query = "INSERT INTO paths (id, name, block_group_id, created_on, edge_ids) \
                     VALUES (?1, ?2, ?3, ?4, ?5);";
        let mut stmt = conn.prepare(query).unwrap();

        let insert_result = stmt.execute(params![
            hash,
            name,
            block_group_id,
            timestamp,
            encoded_edge_ids
        ]);
        drop(stmt);
        drop(encoded_edge_ids);
        let path = match insert_result {
            Ok(_) => Path {
                id: hash,
                name: name.to_string(),
                block_group_id: *block_group_id,
                created_on: timestamp,
                edge_ids: edge_ids.to_vec(),
            },
            Err(rusqlite::Error::SqliteFailure(err, _))
                if err.code == rusqlite::ErrorCode::ConstraintViolation =>
            {
                Path::get(
                    conn,
                    "SELECT paths.* FROM paths WHERE id = ?1",
                    params![hash],
                )?
            }
            Err(e) => {
                return Err(PathError::DatabaseError(e));
            }
        };

        Ok(path)
    }

    /// Creates a path from concatenated edge-ID bytes prepared by a trusted streaming caller.
    #[cfg_attr(
        all(debug_assertions, feature = "profiling"),
        tracing::instrument(skip(conn, name, block_group_id, encoded_edge_ids))
    )]
    pub fn create_from_encoded_edge_ids_unchecked(
        conn: &GraphConnection,
        name: &str,
        block_group_id: &HashId,
        encoded_edge_ids: &[u8],
    ) -> Result<(), PathError> {
        if !encoded_edge_ids.len().is_multiple_of(HASH_ID_SIZE) {
            return Err(PathError::InvalidEdgeBlobLength {
                actual: encoded_edge_ids.len(),
                hash_id_size: HASH_ID_SIZE,
            });
        }
        let hash = HashId(calculate_hash(&format!("{block_group_id}:{name}")));
        let timestamp = chrono::Utc::now().timestamp_nanos_opt().unwrap();
        let query = "INSERT INTO paths (id, name, block_group_id, created_on, edge_ids) \
                     VALUES (?1, ?2, ?3, ?4, ?5);";
        match conn.execute(
            query,
            params![hash, name, block_group_id, timestamp, encoded_edge_ids],
        ) {
            Ok(_) => Ok(()),
            Err(database_error @ rusqlite::Error::SqliteFailure(sqlite_error, _))
                if sqlite_error.code == rusqlite::ErrorCode::ConstraintViolation =>
            {
                let existing_path = conn.query_row(
                    "SELECT EXISTS(SELECT 1 FROM paths WHERE id = ?1)",
                    params![hash],
                    |row| row.get::<_, bool>(0),
                )?;
                if existing_path {
                    Ok(())
                } else {
                    Err(PathError::DatabaseError(database_error))
                }
            }
            Err(error) => Err(PathError::DatabaseError(error)),
        }
    }

    pub fn delete(conn: &GraphConnection, name: &str, block_group_id: &HashId) {
        let query = "DELETE FROM paths where name = ?1 AND block_group_id = ?2;";
        conn.execute(query, params![name.to_string(), block_group_id])
            .unwrap();
    }

    pub fn get_by_id(conn: &GraphConnection, path_id: &HashId) -> Path {
        Path::get(
            conn,
            "SELECT paths.* FROM paths WHERE id = ?1;",
            params![path_id],
        )
        .unwrap()
    }

    pub fn query_for_collection(conn: &GraphConnection, collection_name: &str) -> Vec<Path> {
        let query = "SELECT paths.* FROM paths JOIN block_groups ON paths.block_group_id = block_groups.id WHERE block_groups.collection_name = ?1";
        Path::query(conn, query, params![collection_name])
    }

    pub fn query_for_collection_and_sample(
        conn: &GraphConnection,
        collection_name: &str,
        sample_name: &str,
    ) -> Vec<Path> {
        let query = "SELECT paths.* FROM paths JOIN block_groups ON paths.block_group_id = block_groups.id WHERE block_groups.collection_name = ?1 AND block_groups.sample_name = ?2";
        Path::query(conn, query, params![collection_name, sample_name])
    }

    fn edges_for_ids(
        conn: &GraphConnection,
        edge_ids: &[HashId],
        history_ref: Option<&str>,
    ) -> Result<Vec<Edge>, PathError> {
        let edges = Edge::query_by_ids(conn, edge_ids, history_ref);
        if edges.len() != edge_ids.len() {
            return Err(PathError::Missing(format!(
                "Expected {} edges while loading a path, found {}",
                edge_ids.len(),
                edges.len()
            )));
        }
        Ok(edges)
    }

    pub fn edges(
        &self,
        conn: &GraphConnection,
        history_ref: Option<&str>,
    ) -> Result<Vec<Edge>, PathError> {
        if let Some(history_ref) = history_ref {
            return Self::edges_for_path(conn, &self.id, Some(history_ref));
        }
        Self::edges_for_ids(conn, &self.edge_ids, None)
    }

    pub fn edges_for_path(
        conn: &GraphConnection,
        path_id: &HashId,
        history_ref: Option<&str>,
    ) -> Result<Vec<Edge>, PathError> {
        let table_name = Self::table_name_with_history_ref(history_ref);
        let query = format!("SELECT paths.* FROM {table_name} paths WHERE id = :path_id");
        let mut query_params: Vec<(&str, &dyn rusqlite::ToSql)> = vec![(":path_id", path_id)];
        if let Some(history_ref) = history_ref.as_ref() {
            query_params.push((":history_ref", history_ref));
        }
        let path = Self::try_query(conn, &query, &query_params[..])?
            .into_iter()
            .next()
            .ok_or_else(|| PathError::Missing(format!("Missing path {path_id}")))?;
        Self::edges_for_ids(conn, &path.edge_ids, history_ref)
    }

    pub fn edges_for_paths(
        conn: &GraphConnection,
        path_ids: Vec<HashId>,
    ) -> Result<HashMap<HashId, Vec<Edge>>, PathError> {
        if path_ids.is_empty() {
            return Ok(HashMap::new());
        }
        let path_id_values = Rc::new(
            path_ids
                .iter()
                .copied()
                .map(SQLValue::from)
                .collect::<Vec<_>>(),
        );
        let paths = Self::try_query(
            conn,
            "WITH requested AS (
                 SELECT value, rowid AS position FROM rarray(?1)
             )
             SELECT paths.* FROM paths
             JOIN requested ON paths.id = requested.value
             ORDER BY requested.position",
            params![path_id_values],
        )?;
        if paths.len() != path_ids.len() {
            return Err(PathError::Missing(format!(
                "Expected {} paths, found {}",
                path_ids.len(),
                paths.len()
            )));
        }

        let edge_ids = paths
            .iter()
            .flat_map(|path| path.edge_ids.iter().copied())
            .collect::<Vec<_>>();
        let edges = Self::edges_for_ids(conn, &edge_ids, None)?;
        let mut edge_offset = 0;
        let mut edges_by_path_id = HashMap::new();
        for path in paths {
            let edge_end = edge_offset + path.edge_ids.len();
            edges_by_path_id.insert(path.id, edges[edge_offset..edge_end].to_vec());
            edge_offset = edge_end;
        }
        Ok(edges_by_path_id)
    }

    pub fn sequence(
        &self,
        conn: &GraphConnection,
        history_ref: Option<&str>,
    ) -> Result<String, PathError> {
        let blocks = self.blocks(conn, history_ref)?;
        Ok(blocks
            .into_iter()
            .map(|block| block.block_sequence)
            .collect::<Vec<_>>()
            .join(""))
    }

    pub fn length(
        &self,
        conn: &GraphConnection,
        history_ref: Option<&str>,
    ) -> Result<i64, PathError> {
        let blocks = self.blocks(conn, history_ref)?;
        let end_block = blocks.last().unwrap();
        // the last block is the terminal node, which starts at the end of the path
        Ok(end_block.path_start)
    }

    pub fn edge_pairs_to_block(
        &self,
        into: Edge,
        out_of: Edge,
        sequences_by_node_id: &HashMap<HashId, Sequence>,
        current_path_length: i64,
    ) -> Result<PathBlock, PathError> {
        let start = into.target_coordinate;
        let end = out_of.source_coordinate;
        let strand = into.target_strand;
        let block_sequence_length = end - start;
        let Some(sequence) = sequences_by_node_id.get(&into.target_node_id) else {
            return Err(PathError::Missing(format!(
                "Missing sequence for node {} while building path {}",
                into.target_node_id, self.id
            )));
        };

        let block_sequence = if strand == Strand::Reverse {
            revcomp(&sequence.get_sequence(start, end)?)
        } else {
            sequence.get_sequence(start, end)?
        };

        Ok(PathBlock {
            node_id: into.target_node_id,
            block_sequence,
            sequence_start: start,
            sequence_end: end,
            path_start: current_path_length,
            path_end: current_path_length + block_sequence_length,
            strand,
        })
    }

    pub fn blocks(
        &self,
        conn: &GraphConnection,
        history_ref: Option<&str>,
    ) -> Result<Vec<PathBlock>, PathError> {
        let edges = self.edges(conn, history_ref)?;

        let mut sequence_node_ids = HashSet::new();
        for edge in &edges {
            if !is_start_node(edge.source_node_id) {
                sequence_node_ids.insert(edge.source_node_id);
            }
            if !is_end_node(edge.target_node_id) {
                sequence_node_ids.insert(edge.target_node_id);
            }
        }
        let sequences_by_node_id = Node::get_sequences_by_node_ids(
            conn,
            &sequence_node_ids.into_iter().collect::<Vec<_>>(),
            history_ref,
        );

        let mut blocks = vec![];
        let mut path_length = 0;

        // NOTE: Adding a "start block" for the dedicated start sequence with a range from i64::MIN
        // to 0 makes interval tree lookups work better.  If the point being looked up is -1 (or
        // below), it will return this block.
        blocks.push(PathBlock {
            node_id: PATH_START_NODE_ID,
            block_sequence: "".to_string(),
            sequence_start: 0,
            sequence_end: 0,
            path_start: i64::MIN + 1,
            path_end: 0,
            strand: Strand::Forward,
        });

        for (into, out_of) in edges.into_iter().tuple_windows() {
            let block =
                self.edge_pairs_to_block(into, out_of, &sequences_by_node_id, path_length)?;
            path_length += block.block_sequence.len() as i64;
            blocks.push(block);
        }

        // NOTE: Adding an "end block" for the dedicated end sequence with a range from the path
        // length to i64::MAX makes interval tree lookups work better.  If the point being looked up
        // is the path length (or higher), it will return this block.
        blocks.push(PathBlock {
            node_id: PATH_END_NODE_ID,
            block_sequence: "".to_string(),
            sequence_start: 0,
            sequence_end: 0,
            path_start: path_length,
            path_end: i64::MAX - 1,
            strand: Strand::Forward,
        });

        Ok(blocks)
    }

    pub fn intervaltree(
        &self,
        conn: &GraphConnection,
    ) -> Result<IntervalTree<i64, NodeIntervalBlock>, PathError> {
        let blocks = self.blocks(conn, None)?;
        let tree: IntervalTree<i64, NodeIntervalBlock> = blocks
            .into_iter()
            .map(|block| {
                (
                    block.path_start..block.path_end,
                    NodeIntervalBlock {
                        node_id: block.node_id,
                        start: block.path_start,
                        end: block.path_end,
                        sequence_start: block.sequence_start,
                        sequence_end: block.sequence_end,
                        strand: block.strand,
                    },
                )
            })
            .collect();
        Ok(tree)
    }

    pub fn find_block_mappings(
        &self,
        conn: &GraphConnection,
        other_path: &Path,
    ) -> Result<Vec<RangeMapping>, PathError> {
        // Given two paths, find the overlapping parts of common nodes/blocks and return a list af
        // mappings from subranges of one path to corresponding shared subranges of the other path
        let our_blocks = self.blocks(conn, None)?;
        let their_blocks = other_path.blocks(conn, None)?;

        let our_node_ids = our_blocks
            .iter()
            .map(|block| block.node_id)
            .collect::<HashSet<_>>();
        let their_node_ids = their_blocks
            .iter()
            .map(|block| block.node_id)
            .collect::<HashSet<_>>();
        let common_node_ids = our_node_ids
            .intersection(&their_node_ids)
            .copied()
            .collect::<HashSet<_>>();

        let mut our_blocks_by_node_id = HashMap::new();
        for block in our_blocks
            .iter()
            .filter(|block| common_node_ids.contains(&block.node_id))
        {
            our_blocks_by_node_id
                .entry(block.node_id)
                .or_insert(vec![])
                .push(block);
        }

        let mut their_blocks_by_node_id = HashMap::new();
        for block in their_blocks
            .iter()
            .filter(|block| common_node_ids.contains(&block.node_id))
        {
            their_blocks_by_node_id
                .entry(block.node_id)
                .or_insert(vec![])
                .push(block);
        }

        let mut mappings = vec![];
        for node_id in common_node_ids {
            let our_blocks = our_blocks_by_node_id.get(&node_id).unwrap();
            let our_sorted_blocks = our_blocks
                .clone()
                .into_iter()
                .sorted_by(|a, b| a.sequence_start.cmp(&b.sequence_start))
                .collect::<Vec<&PathBlock>>();
            let their_blocks = their_blocks_by_node_id.get(&node_id).unwrap();
            let their_sorted_blocks = their_blocks
                .clone()
                .into_iter()
                .sorted_by(|a, b| a.sequence_start.cmp(&b.sequence_start))
                .collect::<Vec<&PathBlock>>();

            for our_block in our_sorted_blocks {
                let mut their_block_index = 0;

                while their_block_index < their_sorted_blocks.len() {
                    let their_block = their_sorted_blocks[their_block_index];
                    if their_block.sequence_end <= our_block.sequence_start {
                        // If their block is before ours, move along to the next one
                        their_block_index += 1;
                    } else {
                        let our_range = Range {
                            start: our_block.sequence_start,
                            end: our_block.sequence_end,
                        };
                        let their_range = Range {
                            start: their_block.sequence_start,
                            end: their_block.sequence_end,
                        };

                        let common_ranges = our_range.overlap(&their_range);
                        if !common_ranges.is_empty() {
                            if common_ranges.len() > 1 {
                                panic!(
                                    "Found more than one common range for blocks with node {node_id}"
                                );
                            }

                            let common_range = &common_ranges[0];
                            let our_start = our_block.path_start
                                + (common_range.start - our_block.sequence_start);
                            let our_end = our_block.path_start
                                + (common_range.end - our_block.sequence_start);
                            let their_start = their_block.path_start
                                + (common_range.start - their_block.sequence_start);
                            let their_end = their_block.path_start
                                + (common_range.end - their_block.sequence_start);

                            let mapping = RangeMapping {
                                source_range: Range {
                                    start: our_start,
                                    end: our_end,
                                },
                                target_range: Range {
                                    start: their_start,
                                    end: their_end,
                                },
                            };
                            mappings.push(mapping);
                        }

                        if their_block.sequence_end < our_block.sequence_end {
                            // If their block ends before ours, move along to the next one
                            their_block_index += 1;
                        } else {
                            break;
                        }
                    }
                }
            }
        }

        Ok(mappings
            .into_iter()
            .sorted_by(|a, b| a.source_range.start.cmp(&b.source_range.start))
            .collect::<Vec<RangeMapping>>())
    }

    pub fn propagate_annotation(
        annotation: Annotation,
        mapping_tree: &IntervalTree<i64, RangeMapping>,
        sequence_length: i64,
    ) -> Option<Annotation> {
        /*
        This method contains the core logic for propagating an annotation from one path to another.
        The core rules are:

        1. If the annotation can be fully propagated to a matching subregion of the other path,
            we propagate it

        2. If only part of the annotation can be propagated to a partial subregion of the other
            path, we propagate just that part and truncate the rest

        3. If the first and last parts of the annotation can be propagated to subregions of the
            other path (but not one or more parts of the middle of the annotation), we propagate the
            entire annotation, including across the parts that don't match those of this path
         */

        // TODO: Add support for different propagation strategies
        // TODO: Handle circular contigs
        let start = annotation.start;
        let end = annotation.end;
        let mappings: Vec<RangeMapping> = mapping_tree
            .query(RustRange { start, end })
            .map(|x| x.value.clone())
            .collect();
        if mappings.is_empty() {
            return None;
        }

        let sorted_mappings: Vec<RangeMapping> = mappings
            .into_iter()
            .sorted_by(|a, b| a.source_range.start.cmp(&b.source_range.start))
            .collect();
        let first_mapping = sorted_mappings.first().unwrap();
        let last_mapping = sorted_mappings.last().unwrap();
        let translated_start = if first_mapping.source_range.contains(start) {
            first_mapping.source_range.translate_index(
                start,
                &first_mapping.target_range,
                sequence_length,
                false,
            )
        } else {
            Ok(first_mapping.target_range.start)
        };

        let translated_end = if last_mapping.source_range.contains(end) {
            last_mapping.source_range.translate_index(
                end,
                &last_mapping.target_range,
                sequence_length,
                false,
            )
        } else {
            Ok(last_mapping.target_range.end)
        };

        if translated_start.is_err() || translated_end.is_err() {
            return None;
        }

        Some(Annotation {
            name: annotation.name,
            start: translated_start.expect("Failed to translate start"),
            end: translated_end.expect("Failed to translate end"),
        })
    }

    pub fn get_mapping_tree(
        &self,
        conn: &GraphConnection,
        path: &Path,
    ) -> Result<IntervalTree<i64, RangeMapping>, PathError> {
        let mappings = self.find_block_mappings(conn, path)?;
        Ok(mappings
            .into_iter()
            .map(|mapping| {
                (
                    mapping.source_range.start..mapping.source_range.end,
                    mapping,
                )
            })
            .collect())
    }

    pub fn propagate_annotations(
        &self,
        conn: &GraphConnection,
        path: &Path,
        annotations: Vec<Annotation>,
    ) -> Result<Vec<Annotation>, PathError> {
        let mapping_tree = self.get_mapping_tree(conn, path)?;
        let sequence_length = path.sequence(conn, None)?.len();
        Ok(annotations
            .into_iter()
            .filter_map(|annotation| {
                Path::propagate_annotation(annotation, &mapping_tree, sequence_length as i64)
            })
            .clone()
            .collect())
    }

    pub fn new_path_with(
        &self,
        conn: &GraphConnection,
        path_start: i64,
        path_end: i64,
        edge_to_new_node: &Edge,
        edge_from_new_node: &Edge,
    ) -> Result<Path, PathError> {
        // Creates a new path from the current one by replacing all edges between path_start and
        // path_end with the input edges that are to and from a new node
        let tree = self.intervaltree(conn)?;
        let block_with_start = tree.query_point(path_start).next().unwrap().value;
        let block_with_end = tree.query_point(path_end).next().unwrap().value;

        let edges = self.edges(conn, None)?;
        let edges_by_source = edges
            .iter()
            .map(|edge| ((edge.source_node_id, edge.source_coordinate), edge))
            .collect::<HashMap<(_, i64), &Edge>>();
        let edges_by_target = edges
            .iter()
            .map(|edge| ((edge.target_node_id, edge.target_coordinate), edge))
            .collect::<HashMap<(_, i64), &Edge>>();

        let edge_before_new_node = if edge_to_new_node.source_node_id == PATH_START_NODE_ID {
            None
        } else {
            let edge = edges_by_target
                .get(&(block_with_start.node_id, block_with_start.sequence_start))
                .unwrap();
            Some(edge)
        };
        let edge_after_new_node = if edge_from_new_node.target_node_id == PATH_END_NODE_ID {
            None
        } else {
            let edge = edges_by_source
                .get(&(block_with_end.node_id, block_with_end.sequence_end))
                .unwrap();
            Some(edge)
        };

        let mut new_edge_ids = vec![];
        if let Some(edge_before_new_node) = edge_before_new_node {
            for edge in &edges {
                new_edge_ids.push(edge.id);
                if edge.id == edge_before_new_node.id {
                    break;
                }
            }
        }

        new_edge_ids.push(edge_to_new_node.id);
        new_edge_ids.push(edge_from_new_node.id);

        if let Some(edge_after_new_node) = edge_after_new_node {
            let mut after_new_node = false;
            for edge in &edges {
                if edge.id == edge_after_new_node.id {
                    after_new_node = true;
                }
                if after_new_node {
                    new_edge_ids.push(edge.id);
                }
            }
        }

        let new_name = format!(
            "{}-start-{}-end-{}-node-{}",
            self.name, path_start, path_end, edge_to_new_node.target_node_id
        );
        Path::create(conn, &new_name, &self.block_group_id, &new_edge_ids)
    }

    pub fn new_path_with_deletion(
        &self,
        conn: &GraphConnection,
        deletion_start: i64,
        deletion_end: i64,
    ) -> Result<Path, PathError> {
        // Creates a new path from the current one by replacing all edges between deletion_start and
        // deletion_end with a single edge spanning the deletion.
        let tree = self.intervaltree(conn)?;
        let block_with_start = tree.query_point(deletion_start).next().unwrap().value;
        let block_with_end = tree.query_point(deletion_end).next().unwrap().value;

        let node_deletion_start = deletion_start - block_with_start.start;
        let node_deletion_end = deletion_end - block_with_end.start;
        let deletion_edge_result = Edge::query(
            conn,
            "SELECT * FROM edges WHERE source_node_id = ?1 AND source_coordinate = ?2 AND target_node_id = ?3 AND target_coordinate = ?4",
            rusqlite::params!(
                SQLValue::from(block_with_start.node_id),
                SQLValue::from(node_deletion_start),
                SQLValue::from(block_with_end.node_id),
                SQLValue::from(node_deletion_end)
            ),
        );

        if deletion_edge_result.is_empty() {
            let error_string = format!(
                "No edge found from node {}:{node_deletion_start} to node {}:{node_deletion_end}",
                block_with_start.node_id, block_with_end.node_id
            );
            return Err(PathError::Query(QueryError::ResultsNotFound(error_string)));
        }

        let deletion_edge = deletion_edge_result[0].clone();

        let edges = self.edges(conn, None)?;
        let edges_by_source = edges
            .iter()
            .map(|edge| ((edge.source_node_id, edge.source_coordinate), edge))
            .collect::<HashMap<(_, i64), &Edge>>();
        let edges_by_target = edges
            .iter()
            .map(|edge| ((edge.target_node_id, edge.target_coordinate), edge))
            .collect::<HashMap<(_, i64), &Edge>>();
        let edge_before_deletion = edges_by_target
            .get(&(block_with_start.node_id, block_with_start.sequence_start))
            .unwrap();
        let edge_after_deletion = edges_by_source
            .get(&(block_with_end.node_id, block_with_end.sequence_end))
            .unwrap();

        let mut new_edge_ids = vec![];
        let mut before_deletion = true;
        let mut after_deletion = false;
        for edge in &edges {
            if before_deletion {
                new_edge_ids.push(edge.id);
                if edge.id == edge_before_deletion.id {
                    before_deletion = false;
                    new_edge_ids.push(deletion_edge.id);
                }
            } else if after_deletion {
                new_edge_ids.push(edge.id);
            } else if edge.id == edge_after_deletion.id {
                after_deletion = true;
                new_edge_ids.push(edge.id);
            }
        }

        let new_name = format!(
            "{}-start-{}-end-{}-node-{}",
            self.name,
            deletion_start,
            deletion_end,
            HashId::convert_str("")
        );

        Path::create(conn, &new_name, &self.block_group_id, &new_edge_ids)
    }

    fn node_blocks_for_range(
        &self,
        intervaltree: &IntervalTree<i64, NodeIntervalBlock>,
        start: i64,
        end: i64,
    ) -> Vec<NodeIntervalBlock> {
        // TODO: Handle start/end values that are in the middle of blocks
        let node_blocks: Vec<NodeIntervalBlock> = intervaltree
            .query(RustRange { start, end })
            .map(|x| x.value)
            .sorted_by(|a, b| a.start.cmp(&b.start))
            .collect();

        if node_blocks.is_empty() {
            return vec![];
        }

        let mut result_node_blocks = vec![];
        let start_offset = if node_blocks[0].start < start {
            start - node_blocks[0].start
        } else {
            0
        };

        let mut consolidated_block = NodeIntervalBlock {
            node_id: node_blocks[0].node_id,
            start: node_blocks[0].start + start_offset,
            end: node_blocks[0].end,
            sequence_start: node_blocks[0].sequence_start + start_offset,
            sequence_end: node_blocks[0].sequence_end,
            strand: node_blocks[0].strand,
        };

        for block in &node_blocks[1..] {
            if consolidated_block.node_id == block.node_id && consolidated_block.end == block.start
            {
                // If the current block is immediately adjacent to the previous one (as recorded
                // in the consolidated block), extend the consolidated block
                consolidated_block = NodeIntervalBlock {
                    node_id: consolidated_block.node_id,
                    start: consolidated_block.start,
                    end: block.end,
                    sequence_start: consolidated_block.sequence_start,
                    sequence_end: block.sequence_end,
                    strand: consolidated_block.strand,
                };
            } else {
                result_node_blocks.push(consolidated_block);
                consolidated_block = *block;
            }
        }

        let end_offset = if consolidated_block.end > end {
            consolidated_block.end - end
        } else {
            0
        };

        result_node_blocks.push(NodeIntervalBlock {
            node_id: consolidated_block.node_id,
            start: consolidated_block.start,
            end: consolidated_block.end - end_offset,
            sequence_start: consolidated_block.sequence_start,
            sequence_end: consolidated_block.sequence_end - end_offset,
            strand: consolidated_block.strand,
        });

        result_node_blocks
    }

    pub fn node_block_partition(
        &self,
        conn: &GraphConnection,
        ranges: Vec<Range>,
    ) -> Result<Vec<NodeIntervalBlock>, PathError> {
        let intervaltree = self.intervaltree(conn)?;
        let mut partitioned_nodes = vec![];
        for range in ranges {
            let node_blocks = self.node_blocks_for_range(&intervaltree, range.start, range.end);
            for node_block in &node_blocks {
                partitioned_nodes.push(*node_block);
            }
        }
        Ok(partitioned_nodes)
    }
}

impl RegionResolver for Path {
    type Connection = GraphConnection;
    type Error = PathError;

    fn resolve(
        region: &Region,
        conn: &Self::Connection,
        collection_name: &str,
        sample_name: &str,
    ) -> Result<Self, RegionResolutionError<Self::Error>> {
        let mut stmt = conn
            .prepare(
                "SELECT paths.*, block_groups.is_default \
             FROM paths \
             JOIN block_groups ON paths.block_group_id = block_groups.id \
             WHERE block_groups.collection_name = ?1 \
               AND block_groups.sample_name = ?2 \
               AND lower(paths.name) = lower(?3)",
            )
            .map_err(PathError::DatabaseError)?;
        let matches = stmt
            .query_map(params![collection_name, sample_name, region.name], |row| {
                Ok((Self::try_process_row(row)?, row.get::<_, bool>(5)?))
            })
            .map_err(PathError::DatabaseError)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(PathError::DatabaseError)?;

        match matches.len() {
            0 => Err(RegionResolutionError::NotFound(region.name.clone())),
            1 => Ok(matches.into_iter().next().unwrap().0),
            _ => {
                let default_matches = matches
                    .into_iter()
                    .filter(|(_, is_default)| *is_default)
                    .collect::<Vec<_>>();
                match default_matches.len() {
                    1 => Ok(default_matches.into_iter().next().unwrap().0),
                    _ => Err(RegionResolutionError::Ambiguous(format!(
                        "multiple paths named {}",
                        region.name
                    ))),
                }
            }
        }
    }
}

impl Query for Path {
    type Model = Path;

    const TABLE_NAME: &'static str = "paths";

    fn process_row(row: &Row) -> Self::Model {
        Self::try_process_row(row).expect("should decode a valid path row")
    }

    fn try_process_row(row: &Row) -> rusqlite::Result<Self::Model> {
        let encoded_edge_ids = row.get::<_, Vec<u8>>(4)?;
        let edge_ids = decode_edge_ids(&encoded_edge_ids).map_err(|error| {
            rusqlite::Error::FromSqlConversionFailure(4, Type::Blob, Box::new(error))
        })?;
        Ok(Path {
            id: row.get(0)?,
            block_group_id: row.get(1)?,
            name: row.get(2)?,
            created_on: row.get(3)?,
            edge_ids,
        })
    }
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use capnp::message::TypedBuilder;
    use chrono::Utc;
    use gen_core::region::RegionResolutionError;

    use super::*;
    use crate::{
        block_group::{BlockGroup, NewBlockGroup},
        block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
        collection::Collection,
        edge::Edge,
        history::dolt::commit_all,
        sample::{NewSample, Sample},
        test_helpers::{create_bg, get_connection},
    };

    fn create_test_block_group(conn: &GraphConnection) -> BlockGroup {
        Sample::get_or_create(
            conn,
            NewSample {
                name: "test-sample",
                ..Default::default()
            },
        )
        .unwrap();
        BlockGroup::create(
            conn,
            NewBlockGroup {
                collection_name: "test collection",
                sample_name: "test-sample",
                name: "test block group",
                ..Default::default()
            },
        )
        .unwrap()
    }

    fn create_repeating_path_edges(
        conn: &GraphConnection,
    ) -> (BlockGroup, Vec<HashId>, Vec<HashId>) {
        Collection::create(conn, "blob storage").unwrap();
        Sample::get_or_create(
            conn,
            NewSample {
                name: "blob-sample",
                ..Default::default()
            },
        )
        .unwrap();
        let block_group = BlockGroup::create(
            conn,
            NewBlockGroup {
                collection_name: "blob storage",
                sample_name: "blob-sample",
                name: "cycle",
                ..Default::default()
            },
        )
        .unwrap();
        let sequence_a = Sequence::new()
            .sequence_type("DNA")
            .sequence("A")
            .save(conn)
            .unwrap();
        let sequence_c = Sequence::new()
            .sequence_type("DNA")
            .sequence("C")
            .save(conn)
            .unwrap();
        let node_a =
            Node::create(conn, &sequence_a.hash, &HashId::convert_str("blob-node-a")).unwrap();
        let node_c =
            Node::create(conn, &sequence_c.hash, &HashId::convert_str("blob-node-c")).unwrap();
        let start_to_a = Edge::create(
            conn,
            PATH_START_NODE_ID,
            0,
            Strand::Forward,
            node_a,
            0,
            Strand::Forward,
        )
        .unwrap();
        let a_to_c =
            Edge::create(conn, node_a, 1, Strand::Forward, node_c, 0, Strand::Forward).unwrap();
        let c_to_a =
            Edge::create(conn, node_c, 1, Strand::Forward, node_a, 0, Strand::Forward).unwrap();
        let c_to_end = Edge::create(
            conn,
            node_c,
            1,
            Strand::Forward,
            PATH_END_NODE_ID,
            0,
            Strand::Forward,
        )
        .unwrap();
        let block_group_edges = [start_to_a.id, a_to_c.id, c_to_a.id, c_to_end.id]
            .into_iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<_>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let repeated = vec![start_to_a.id, a_to_c.id, c_to_a.id, a_to_c.id, c_to_end.id];
        let short = vec![start_to_a.id, a_to_c.id, c_to_end.id];
        (block_group, repeated, short)
    }

    #[test]
    fn test_empty_edge_id_array_encoding_and_decoding() {
        let encoded = encode_edge_ids(&[]);

        assert!(encoded.is_empty());
        assert_eq!(decode_edge_ids(&encoded).unwrap(), Vec::<HashId>::new());
    }

    #[test]
    fn test_ordered_edge_id_array_round_trip() {
        let edge_ids = vec![HashId::pad_str(3), HashId::pad_str(1), HashId::pad_str(2)];

        let encoded = encode_edge_ids(&edge_ids);

        assert_eq!(encoded.len(), edge_ids.len() * HASH_ID_SIZE);
        assert_eq!(decode_edge_ids(&encoded).unwrap(), edge_ids);
    }

    #[test]
    fn test_create_from_encoded_edge_ids_preserves_order_without_decoding_input() {
        let conn = &get_connection(None).unwrap();
        Collection::create(conn, "test collection").unwrap();
        let block_group = create_test_block_group(conn);
        let edge_ids = vec![HashId::pad_str(3), HashId::pad_str(1), HashId::pad_str(3)];
        let encoded_edge_ids = encode_edge_ids(&edge_ids);

        Path::create_from_encoded_edge_ids_unchecked(
            conn,
            "encoded",
            &block_group.id,
            &encoded_edge_ids,
        )
        .unwrap();
        Path::create_from_encoded_edge_ids_unchecked(
            conn,
            "encoded",
            &block_group.id,
            &encoded_edge_ids,
        )
        .unwrap();

        let path_id = HashId(calculate_hash(&format!("{}:encoded", block_group.id)));
        assert_eq!(Path::get_by_id(conn, &path_id).edge_ids, edge_ids);
    }

    #[test]
    fn test_duplicate_edge_ids_preserved_at_distinct_positions() {
        let repeated = HashId::pad_str(42);
        let edge_ids = vec![HashId::pad_str(1), repeated, HashId::pad_str(2), repeated];

        let decoded = decode_edge_ids(&encode_edge_ids(&edge_ids)).unwrap();

        assert_eq!(decoded, edge_ids);
        assert_eq!(decoded[1], decoded[3]);
    }

    #[test]
    fn test_malformed_edge_id_blob_is_rejected() {
        let error = decode_edge_ids(&[0; HASH_ID_SIZE + 1]).unwrap_err();
        assert_eq!(
            error,
            PathError::InvalidEdgeBlobLength {
                actual: HASH_ID_SIZE + 1,
                hash_id_size: HASH_ID_SIZE,
            }
        );

        let conn = &get_connection(None).unwrap();
        Collection::create(conn, "test collection").unwrap();
        let block_group = create_test_block_group(conn);
        conn.execute(
            "INSERT INTO paths (id, block_group_id, name, created_on, edge_ids)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                HashId::convert_str("malformed-path"),
                block_group.id,
                "malformed",
                1_i64,
                vec![0_u8; HASH_ID_SIZE + 1]
            ],
        )
        .unwrap();

        let query_error = Path::try_query(
            conn,
            "SELECT paths.* FROM paths WHERE name = 'malformed'",
            [],
        )
        .unwrap_err();
        let QueryError::DatabaseError(rusqlite::Error::FromSqlConversionFailure(
            column,
            Type::Blob,
            source,
        )) = query_error
        else {
            panic!("expected malformed path BLOB conversion error");
        };
        assert_eq!(column, 4);
        assert_eq!(
            source.downcast_ref::<PathError>(),
            Some(&PathError::InvalidEdgeBlobLength {
                actual: HASH_ID_SIZE + 1,
                hash_id_size: HASH_ID_SIZE,
            })
        );
    }

    #[test]
    fn test_path_create_reconstructs_ordered_repeated_edges_and_sequence() {
        let conn = &get_connection(None).unwrap();
        let (block_group, repeated, _) = create_repeating_path_edges(conn);

        let created = Path::create(conn, "repeated", &block_group.id, &repeated).unwrap();
        let loaded = Path::get_by_id(conn, &created.id);
        let loaded_edge_ids = loaded
            .edges(conn, None)
            .unwrap()
            .into_iter()
            .map(|edge| edge.id)
            .collect::<Vec<_>>();

        assert_eq!(loaded.edge_ids, repeated);
        assert_eq!(loaded_edge_ids, repeated);
        assert_eq!(loaded.sequence(conn, None).unwrap(), "ACAC");
        let encoded = conn
            .query_row(
                "SELECT edge_ids FROM paths WHERE id = ?1",
                params![loaded.id],
                |row| row.get::<_, Vec<u8>>(0),
            )
            .unwrap();
        assert_eq!(encoded, encode_edge_ids(&repeated));
    }

    #[test]
    fn test_multiple_paths_share_canonical_edges_without_path_edges_table() {
        let conn = &get_connection(None).unwrap();
        let (block_group, repeated, _) = create_repeating_path_edges(conn);
        let path_a = Path::create(conn, "path-a", &block_group.id, &repeated).unwrap();
        let path_b = Path::create(conn, "path-b", &block_group.id, &repeated).unwrap();

        let edges_by_path = Path::edges_for_paths(conn, vec![path_a.id, path_b.id]).unwrap();
        let edge_ids_a = edges_by_path[&path_a.id]
            .iter()
            .map(|edge| edge.id)
            .collect::<Vec<_>>();
        let edge_ids_b = edges_by_path[&path_b.id]
            .iter()
            .map(|edge| edge.id)
            .collect::<Vec<_>>();

        assert_eq!(edge_ids_a, repeated);
        assert_eq!(edge_ids_b, repeated);
        let path_edges_table_count = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'path_edges'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap();
        assert_eq!(path_edges_table_count, 0);
    }

    #[test]
    fn test_empty_path_stores_valid_empty_blob() {
        let conn = &get_connection(None).unwrap();
        Collection::create(conn, "test collection").unwrap();
        let block_group = create_test_block_group(conn);

        let path = Path::create(conn, "empty", &block_group.id, &[]).unwrap();
        let encoded = conn
            .query_row(
                "SELECT edge_ids FROM paths WHERE id = ?1",
                params![path.id],
                |row| row.get::<_, Vec<u8>>(0),
            )
            .unwrap();

        assert!(path.edge_ids.is_empty());
        assert!(encoded.is_empty());
        assert_eq!(path.sequence(conn, None).unwrap(), "");
    }

    #[test]
    fn test_historical_path_read_returns_snapshot_edge_array() {
        let conn = &get_connection(None).unwrap();
        let (block_group, repeated, short) = create_repeating_path_edges(conn);
        let path = Path::create(conn, "history", &block_group.id, &repeated).unwrap();
        let first_commit = commit_all(conn, "repeated path").unwrap();

        conn.execute(
            "UPDATE paths SET edge_ids = ?1 WHERE id = ?2",
            params![encode_edge_ids(&short), path.id],
        )
        .unwrap();
        commit_all(conn, "short path").unwrap();

        let first_ref = first_commit.to_string();
        let query = format!(
            "SELECT paths.* FROM {} paths WHERE id = :path_id",
            Path::table_name_with_history_ref(Some(&first_ref))
        );
        let query_params: [(&str, &dyn rusqlite::ToSql); 2] =
            [(":history_ref", &first_ref), (":path_id", &path.id)];
        let historical_path = Path::try_query(conn, &query, &query_params)
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let current_path = Path::get_by_id(conn, &path.id);

        assert_eq!(historical_path.edge_ids, repeated);
        assert_eq!(current_path.edge_ids, short);
        assert_eq!(
            historical_path.sequence(conn, Some(&first_ref)).unwrap(),
            "ACAC"
        );
        assert_eq!(current_path.sequence(conn, None).unwrap(), "AC");
    }

    mod region_resolver {
        use super::*;

        #[test]
        fn resolves_path_by_name_case_insensitively() {
            let conn = &get_connection(None).unwrap();
            Collection::create(conn, "test collection").unwrap();
            let block_group = create_test_block_group(conn);
            let edge = Edge::create(
                conn,
                PATH_START_NODE_ID,
                0,
                Strand::Forward,
                PATH_END_NODE_ID,
                0,
                Strand::Forward,
            )
            .unwrap();
            BlockGroupEdge::bulk_create(
                conn,
                &[BlockGroupEdgeData {
                    block_group_id: block_group.id,
                    edge_id: edge.id,
                    chromosome_index: 0,
                    phased: 0,
                }],
            );
            let path = Path::create(conn, "chr1", &block_group.id, &[edge.id]).unwrap();

            let region = Region::parse("CHR1").unwrap();
            let resolved = Path::resolve(&region, conn, "test collection", "test-sample").unwrap();
            assert_eq!(resolved.id, path.id);
        }

        #[test]
        fn returns_not_found_for_missing_path() {
            let conn = &get_connection(None).unwrap();
            Collection::create(conn, "test collection").unwrap();
            let _ = create_test_block_group(conn);

            let region = Region::parse("missing").unwrap();
            let err = Path::resolve(&region, conn, "test collection", "test-sample").unwrap_err();
            assert!(matches!(
                err,
                RegionResolutionError::NotFound(name) if name == "missing"
            ));
        }

        #[test]
        fn returns_ambiguous_for_multiple_matching_paths() {
            let conn = &get_connection(None).unwrap();
            Collection::create(conn, "test collection").unwrap();
            let block_group = create_test_block_group(conn);
            let edge = Edge::create(
                conn,
                PATH_START_NODE_ID,
                0,
                Strand::Forward,
                PATH_END_NODE_ID,
                0,
                Strand::Forward,
            )
            .unwrap();
            BlockGroupEdge::bulk_create(
                conn,
                &[BlockGroupEdgeData {
                    block_group_id: block_group.id,
                    edge_id: edge.id,
                    chromosome_index: 0,
                    phased: 0,
                }],
            );
            let _ = Path::create(conn, "chr1", &block_group.id, &[edge.id]).unwrap();

            let other_block_group =
                create_bg(conn, "test collection", "test-sample", "other block group");
            let other_edge = Edge::create(
                conn,
                PATH_START_NODE_ID,
                0,
                Strand::Forward,
                PATH_END_NODE_ID,
                0,
                Strand::Forward,
            )
            .unwrap();
            BlockGroupEdge::bulk_create(
                conn,
                &[BlockGroupEdgeData {
                    block_group_id: other_block_group.id,
                    edge_id: other_edge.id,
                    chromosome_index: 0,
                    phased: 0,
                }],
            );
            let _ = Path::create(conn, "CHR1", &other_block_group.id, &[other_edge.id]).unwrap();

            let region = Region::parse("chr1").unwrap();
            let err = Path::resolve(&region, conn, "test collection", "test-sample").unwrap_err();
            assert!(matches!(
                err,
                RegionResolutionError::Ambiguous(name) if name == "multiple paths named chr1"
            ));
        }

        #[test]
        fn resolves_default_block_group_when_multiple_paths_match() {
            let conn = &get_connection(None).unwrap();
            Collection::create(conn, "test collection").unwrap();
            let non_default_block_group = create_test_block_group(conn);
            let default_block_group = BlockGroup::create(
                conn,
                NewBlockGroup {
                    collection_name: "test collection",
                    sample_name: "test-sample",
                    name: "default block group",
                    is_default: true,
                    ..Default::default()
                },
            )
            .unwrap();
            conn.execute(
                "UPDATE block_groups SET is_default = 0 WHERE id = ?1",
                params![non_default_block_group.id],
            )
            .unwrap();

            let non_default_edge = Edge::create(
                conn,
                PATH_START_NODE_ID,
                0,
                Strand::Forward,
                PATH_END_NODE_ID,
                0,
                Strand::Forward,
            )
            .unwrap();
            BlockGroupEdge::bulk_create(
                conn,
                &[BlockGroupEdgeData {
                    block_group_id: non_default_block_group.id,
                    edge_id: non_default_edge.id,
                    chromosome_index: 0,
                    phased: 0,
                }],
            );
            let non_default_path = Path::create(
                conn,
                "chr1",
                &non_default_block_group.id,
                &[non_default_edge.id],
            )
            .unwrap();

            let default_edge = Edge::create(
                conn,
                PATH_START_NODE_ID,
                0,
                Strand::Forward,
                PATH_END_NODE_ID,
                0,
                Strand::Forward,
            )
            .unwrap();
            BlockGroupEdge::bulk_create(
                conn,
                &[BlockGroupEdgeData {
                    block_group_id: default_block_group.id,
                    edge_id: default_edge.id,
                    chromosome_index: 0,
                    phased: 0,
                }],
            );
            let default_path =
                Path::create(conn, "chr1", &default_block_group.id, &[default_edge.id]).unwrap();

            let region = Region::parse("chr1").unwrap();
            let resolved = Path::resolve(&region, conn, "test collection", "test-sample").unwrap();
            assert_eq!(resolved.id, default_path.id);
            assert_ne!(resolved.id, non_default_path.id);
        }

        #[test]
        fn returns_ambiguous_for_multiple_default_block_groups() {
            let conn = &get_connection(None).unwrap();
            Collection::create(conn, "test collection").unwrap();
            let default_block_group_a = create_bg(
                conn,
                "test collection",
                "test-sample",
                "default block group a",
            );
            let default_block_group_b = create_bg(
                conn,
                "test collection",
                "test-sample",
                "default block group b",
            );
            conn.execute(
                "UPDATE block_groups SET is_default = 1 WHERE id = ?1",
                params![default_block_group_a.id],
            )
            .unwrap();
            conn.execute(
                "UPDATE block_groups SET is_default = 1 WHERE id = ?1",
                params![default_block_group_b.id],
            )
            .unwrap();

            let edge_a = Edge::create(
                conn,
                PATH_START_NODE_ID,
                0,
                Strand::Forward,
                PATH_END_NODE_ID,
                0,
                Strand::Forward,
            )
            .unwrap();
            BlockGroupEdge::bulk_create(
                conn,
                &[BlockGroupEdgeData {
                    block_group_id: default_block_group_a.id,
                    edge_id: edge_a.id,
                    chromosome_index: 0,
                    phased: 0,
                }],
            );
            let _ = Path::create(conn, "chr1", &default_block_group_a.id, &[edge_a.id]).unwrap();

            let edge_b = Edge::create(
                conn,
                PATH_START_NODE_ID,
                0,
                Strand::Forward,
                PATH_END_NODE_ID,
                0,
                Strand::Forward,
            )
            .unwrap();
            BlockGroupEdge::bulk_create(
                conn,
                &[BlockGroupEdgeData {
                    block_group_id: default_block_group_b.id,
                    edge_id: edge_b.id,
                    chromosome_index: 0,
                    phased: 0,
                }],
            );
            let _ = Path::create(conn, "chr1", &default_block_group_b.id, &[edge_b.id]).unwrap();

            let region = Region::parse("chr1").unwrap();
            let err = Path::resolve(&region, conn, "test collection", "test-sample").unwrap_err();
            assert!(matches!(
                err,
                RegionResolutionError::Ambiguous(name) if name == "multiple paths named chr1"
            ));
        }
    }

    #[test]
    fn test_path_capnp_serialization() {
        let path = Path {
            id: HashId::pad_str(100),
            block_group_id: HashId::pad_str(50),
            name: "test_path".to_string(),
            created_on: Utc::now().timestamp_nanos_opt().unwrap(),
            edge_ids: vec![HashId::pad_str(1), HashId::pad_str(2)],
        };

        let mut message = TypedBuilder::<PathCapnp::Owned>::new_default();
        let mut root = message.init_root();
        path.write_capnp(&mut root);

        let deserialized = Path::read_capnp(root.into_reader());
        assert_eq!(path, deserialized);
    }

    #[test]
    fn test_path_delete() {
        let conn = &get_connection(None).unwrap();
        Collection::create(conn, "test collection").unwrap();
        let block_group = create_test_block_group(conn);

        // Create first path
        let sequence1 = Sequence::new()
            .sequence_type("DNA")
            .sequence("ATCGATCG")
            .save(conn)
            .unwrap();
        let node1_id = Node::create(conn, &sequence1.hash, &HashId::convert_str("1")).unwrap();
        let edge1 = Edge::create(
            conn,
            PATH_START_NODE_ID,
            -1,
            Strand::Forward,
            node1_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge2 = Edge::create(
            conn,
            node1_id,
            8,
            Strand::Forward,
            PATH_END_NODE_ID,
            -1,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids1 = vec![edge1.id, edge2.id];
        let block_group_edges1 = edge_ids1
            .iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: (*edge_id),
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges1);

        let _ = Path::create(conn, "chr1", &block_group.id, &edge_ids1).unwrap();

        // Create second path
        let sequence2 = Sequence::new()
            .sequence_type("DNA")
            .sequence("TTTTTTTT")
            .save(conn)
            .unwrap();
        let node2_id = Node::create(conn, &sequence2.hash, &HashId::convert_str("2")).unwrap();
        let edge3 = Edge::create(
            conn,
            PATH_START_NODE_ID,
            -1,
            Strand::Forward,
            node2_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge4 = Edge::create(
            conn,
            node2_id,
            8,
            Strand::Forward,
            PATH_END_NODE_ID,
            -1,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids2 = vec![edge3.id, edge4.id];
        let block_group_edges2 = edge_ids2
            .iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: (*edge_id),
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges2);

        let path2 = Path::create(conn, "chr2", &block_group.id, &edge_ids2).unwrap();

        let paths_before = Path::query_for_collection(conn, "test collection");
        assert_eq!(paths_before.len(), 2);

        Path::delete(conn, "chr1", &block_group.id);

        let paths_after = Path::query_for_collection(conn, "test collection");
        assert_eq!(paths_after.len(), 1);
        assert_eq!(paths_after[0], path2);
    }

    #[test]
    fn test_gets_sequence() {
        let conn = &get_connection(None).unwrap();
        Collection::create(conn, "test collection").unwrap();
        let block_group = create_test_block_group(conn);
        let sequence1 = Sequence::new()
            .sequence_type("DNA")
            .sequence("ATCGATCG")
            .save(conn)
            .unwrap();
        let node1_id = Node::create(conn, &sequence1.hash, &HashId::convert_str("1")).unwrap();
        let edge1 = Edge::create(
            conn,
            PATH_START_NODE_ID,
            -123,
            Strand::Forward,
            node1_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let sequence2 = Sequence::new()
            .sequence_type("DNA")
            .sequence("AAAAAAAA")
            .save(conn)
            .unwrap();
        let node2_id = Node::create(conn, &sequence2.hash, &HashId::convert_str("2")).unwrap();
        let edge2 = Edge::create(
            conn,
            node1_id,
            8,
            Strand::Forward,
            node2_id,
            1,
            Strand::Forward,
        )
        .unwrap();
        let sequence3 = Sequence::new()
            .sequence_type("DNA")
            .sequence("CCCCCCCC")
            .save(conn)
            .unwrap();
        let node3_id = Node::create(conn, &sequence3.hash, &HashId::convert_str("3")).unwrap();
        let edge3 = Edge::create(
            conn,
            node2_id,
            8,
            Strand::Forward,
            node3_id,
            1,
            Strand::Forward,
        )
        .unwrap();
        let sequence4 = Sequence::new()
            .sequence_type("DNA")
            .sequence("GGGGGGGG")
            .save(conn)
            .unwrap();
        let node4_id = Node::create(conn, &sequence4.hash, &HashId::convert_str("4")).unwrap();
        let edge4 = Edge::create(
            conn,
            node3_id,
            8,
            Strand::Forward,
            node4_id,
            1,
            Strand::Forward,
        )
        .unwrap();
        let edge5 = Edge::create(
            conn,
            node4_id,
            8,
            Strand::Forward,
            PATH_END_NODE_ID,
            -1,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = vec![edge1.id, edge2.id, edge3.id, edge4.id, edge5.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: *edge_id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let path = Path::create(conn, "chr1", &block_group.id, &edge_ids).unwrap();
        assert_eq!(
            path.sequence(conn, None).unwrap(),
            "ATCGATCGAAAAAAACCCCCCCGGGGGGG"
        );
    }

    #[test]
    fn test_gets_sequence_with_rc() {
        let conn = &get_connection(None).unwrap();
        Collection::create(conn, "test collection").unwrap();
        let block_group = create_test_block_group(conn);
        let sequence1 = Sequence::new()
            .sequence_type("DNA")
            .sequence("ATCGATCG")
            .save(conn)
            .unwrap();
        let node1_id = Node::create(conn, &sequence1.hash, &HashId::convert_str("1")).unwrap();
        let edge5 = Edge::create(
            conn,
            node1_id,
            8,
            Strand::Reverse,
            PATH_END_NODE_ID,
            0,
            Strand::Reverse,
        )
        .unwrap();
        let sequence2 = Sequence::new()
            .sequence_type("DNA")
            .sequence("AAAAAAAA")
            .save(conn)
            .unwrap();
        let node2_id = Node::create(conn, &sequence2.hash, &HashId::convert_str("2")).unwrap();
        let edge4 = Edge::create(
            conn,
            node2_id,
            7,
            Strand::Reverse,
            node1_id,
            0,
            Strand::Reverse,
        )
        .unwrap();
        let sequence3 = Sequence::new()
            .sequence_type("DNA")
            .sequence("CCCCCCCC")
            .save(conn)
            .unwrap();
        let node3_id = Node::create(conn, &sequence3.hash, &HashId::convert_str("3")).unwrap();
        let edge3 = Edge::create(
            conn,
            node3_id,
            7,
            Strand::Reverse,
            node2_id,
            0,
            Strand::Reverse,
        )
        .unwrap();
        let sequence4 = Sequence::new()
            .sequence_type("DNA")
            .sequence("GGGGGGGG")
            .save(conn)
            .unwrap();
        let node4_id = Node::create(conn, &sequence4.hash, &HashId::convert_str("4")).unwrap();
        let edge2 = Edge::create(
            conn,
            node4_id,
            7,
            Strand::Reverse,
            node3_id,
            0,
            Strand::Reverse,
        )
        .unwrap();
        let edge1 = Edge::create(
            conn,
            PATH_START_NODE_ID,
            -1,
            Strand::Reverse,
            node4_id,
            0,
            Strand::Reverse,
        )
        .unwrap();

        let edge_ids = vec![edge1.id, edge2.id, edge3.id, edge4.id, edge5.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: *edge_id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let path = Path::create(conn, "chr1", &block_group.id, &edge_ids).unwrap();
        assert_eq!(
            path.sequence(conn, None).unwrap(),
            "CCCCCCCGGGGGGGTTTTTTTCGATCGAT"
        );
    }

    #[test]
    fn test_reverse_complement() {
        assert_eq!(revcomp("ATCCGG"), "CCGGAT");
        assert_eq!(revcomp("CNNNNA"), "TNNNNG");
        assert_eq!(revcomp("cNNgnAt"), "aTncNNg");
    }

    #[test]
    fn test_intervaltree() {
        let conn = &get_connection(None).unwrap();
        Collection::create(conn, "test collection").unwrap();
        let block_group = create_test_block_group(conn);
        let sequence1 = Sequence::new()
            .sequence_type("DNA")
            .sequence("ATCGATCG")
            .save(conn)
            .unwrap();
        let node1_id = Node::create(conn, &sequence1.hash, &HashId::convert_str("1")).unwrap();
        let edge1 = Edge::create(
            conn,
            PATH_START_NODE_ID,
            -1,
            Strand::Forward,
            node1_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let sequence2 = Sequence::new()
            .sequence_type("DNA")
            .sequence("AAAAAAAA")
            .save(conn)
            .unwrap();
        let node2_id = Node::create(conn, &sequence2.hash, &HashId::convert_str("2")).unwrap();
        let edge2 = Edge::create(
            conn,
            node1_id,
            8,
            Strand::Forward,
            node2_id,
            1,
            Strand::Forward,
        )
        .unwrap();
        let sequence3 = Sequence::new()
            .sequence_type("DNA")
            .sequence("CCCCCCCC")
            .save(conn)
            .unwrap();
        let node3_id = Node::create(conn, &sequence3.hash, &HashId::convert_str("3")).unwrap();
        let edge3 = Edge::create(
            conn,
            node2_id,
            8,
            Strand::Forward,
            node3_id,
            1,
            Strand::Forward,
        )
        .unwrap();
        let sequence4 = Sequence::new()
            .sequence_type("DNA")
            .sequence("GGGGGGGG")
            .save(conn)
            .unwrap();
        let node4_id = Node::create(conn, &sequence4.hash, &HashId::convert_str("4")).unwrap();
        let edge4 = Edge::create(
            conn,
            node3_id,
            8,
            Strand::Forward,
            node4_id,
            1,
            Strand::Forward,
        )
        .unwrap();
        let edge5 = Edge::create(
            conn,
            node4_id,
            8,
            Strand::Forward,
            PATH_END_NODE_ID,
            -1,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = vec![edge1.id, edge2.id, edge3.id, edge4.id, edge5.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: *edge_id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let path = Path::create(conn, "chr1", &block_group.id, &edge_ids).unwrap();
        let tree = path.intervaltree(conn).unwrap();
        let blocks1: Vec<NodeIntervalBlock> = tree.query_point(2).map(|x| x.value).collect();
        assert_eq!(blocks1.len(), 1);
        let block1 = &blocks1[0];
        assert_eq!(block1.node_id, node1_id);
        assert_eq!(block1.sequence_start, 0);
        assert_eq!(block1.sequence_end, 8);
        assert_eq!(block1.start, 0);
        assert_eq!(block1.end, 8);
        assert_eq!(block1.strand, Strand::Forward);

        let blocks2: Vec<NodeIntervalBlock> = tree.query_point(12).map(|x| x.value).collect();
        assert_eq!(blocks2.len(), 1);
        let block2 = &blocks2[0];
        assert_eq!(block2.node_id, node2_id);
        assert_eq!(block2.sequence_start, 1);
        assert_eq!(block2.sequence_end, 8);
        assert_eq!(block2.start, 8);
        assert_eq!(block2.end, 15);
        assert_eq!(block2.strand, Strand::Forward);

        let blocks4: Vec<NodeIntervalBlock> = tree.query_point(25).map(|x| x.value).collect();
        assert_eq!(blocks4.len(), 1);
        let block4 = &blocks4[0];
        assert_eq!(block4.node_id, node4_id);
        assert_eq!(block4.sequence_start, 1);
        assert_eq!(block4.sequence_end, 8);
        assert_eq!(block4.start, 22);
        assert_eq!(block4.end, 29);
        assert_eq!(block4.strand, Strand::Forward);
    }

    #[test]
    fn test_gets_sequence_with_edges_into_node_middles() {
        // Tests that if the edge from the virtual start node goes into the middle of the first
        // node, and the edge to the virtual end node comes from the middle of the last node, the
        // sequence is correctly generated
        let conn = &get_connection(None).unwrap();
        Collection::create(conn, "test collection").unwrap();
        let block_group = create_test_block_group(conn);
        let sequence1 = Sequence::new()
            .sequence_type("DNA")
            .sequence("ATCGATCG")
            .save(conn)
            .unwrap();
        let node1_id = Node::create(conn, &sequence1.hash, &HashId::convert_str("1")).unwrap();
        let edge1 = Edge::create(
            conn,
            PATH_START_NODE_ID,
            -1,
            Strand::Forward,
            node1_id,
            4,
            Strand::Forward,
        )
        .unwrap();
        let sequence2 = Sequence::new()
            .sequence_type("DNA")
            .sequence("AAAAAAAA")
            .save(conn)
            .unwrap();
        let node2_id = Node::create(conn, &sequence2.hash, &HashId::convert_str("2")).unwrap();
        let edge2 = Edge::create(
            conn,
            node1_id,
            8,
            Strand::Forward,
            node2_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let sequence3 = Sequence::new()
            .sequence_type("DNA")
            .sequence("CCCCCCCC")
            .save(conn)
            .unwrap();
        let node3_id = Node::create(conn, &sequence3.hash, &HashId::convert_str("3")).unwrap();
        let edge3 = Edge::create(
            conn,
            node2_id,
            8,
            Strand::Forward,
            node3_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let sequence4 = Sequence::new()
            .sequence_type("DNA")
            .sequence("GGGGGGGG")
            .save(conn)
            .unwrap();
        let node4_id = Node::create(conn, &sequence4.hash, &HashId::convert_str("4")).unwrap();
        let edge4 = Edge::create(
            conn,
            node3_id,
            8,
            Strand::Forward,
            node4_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge5 = Edge::create(
            conn,
            node4_id,
            4,
            Strand::Forward,
            PATH_END_NODE_ID,
            -1,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = vec![edge1.id, edge2.id, edge3.id, edge4.id, edge5.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: *edge_id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let path = Path::create(conn, "chr1", &block_group.id, &edge_ids).unwrap();
        let tree = path.intervaltree(conn).unwrap();
        let blocks1: Vec<NodeIntervalBlock> = tree.query_point(2).map(|x| x.value).collect();
        assert_eq!(blocks1.len(), 1);
        let block1 = &blocks1[0];
        assert_eq!(block1.node_id, node1_id);
        assert_eq!(block1.sequence_start, 4);
        assert_eq!(block1.sequence_end, 8);
        assert_eq!(block1.start, 0);
        assert_eq!(block1.end, 4);
        assert_eq!(block1.strand, Strand::Forward);

        let blocks2: Vec<NodeIntervalBlock> = tree.query_point(10).map(|x| x.value).collect();
        assert_eq!(blocks2.len(), 1);
        let block2 = &blocks2[0];
        assert_eq!(block2.node_id, node2_id);
        assert_eq!(block2.sequence_start, 0);
        assert_eq!(block2.sequence_end, 8);
        assert_eq!(block2.start, 4);
        assert_eq!(block2.end, 12);
        assert_eq!(block2.strand, Strand::Forward);

        let blocks4: Vec<NodeIntervalBlock> = tree.query_point(22).map(|x| x.value).collect();
        assert_eq!(blocks4.len(), 1);
        let block4 = &blocks4[0];
        assert_eq!(block4.node_id, node4_id);
        assert_eq!(block4.sequence_start, 0);
        assert_eq!(block4.sequence_end, 4);
        assert_eq!(block4.start, 20);
        assert_eq!(block4.end, 24);
        assert_eq!(block4.strand, Strand::Forward);

        assert_eq!(
            path.sequence(conn, None).unwrap(),
            "ATCGAAAAAAAACCCCCCCCGGGG"
        );
    }

    #[test]
    fn test_full_block_mapping() {
        /*
            |--------| path: 1 sequence, (0, 8)
            |ATCGATCG|
            |--------| Same path: 1 sequence, (0, 8)

            Mapping: (0, 8) -> (0, 8)
        */
        let conn = &get_connection(None).unwrap();
        Collection::create(conn, "test collection").unwrap();
        let block_group = create_test_block_group(conn);
        let sequence1 = Sequence::new()
            .sequence_type("DNA")
            .sequence("ATCGATCG")
            .save(conn)
            .unwrap();
        let node1_id = Node::create(conn, &sequence1.hash, &HashId::convert_str("1")).unwrap();
        let edge1 = Edge::create(
            conn,
            PATH_START_NODE_ID,
            -1,
            Strand::Forward,
            node1_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge2 = Edge::create(
            conn,
            node1_id,
            8,
            Strand::Forward,
            PATH_END_NODE_ID,
            -1,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = vec![edge1.id, edge2.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: *edge_id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let path = Path::create(conn, "chr1", &block_group.id, &edge_ids).unwrap();

        let mappings = path.find_block_mappings(conn, &path).unwrap();
        assert_eq!(mappings.len(), 1);
        let mapping = &mappings[0];
        assert_eq!(mapping.source_range, mapping.target_range);
        assert_eq!(mapping.source_range.start, 0);
        assert_eq!(mapping.source_range.end, 8);
        assert_eq!(mapping.target_range.start, 0);
        assert_eq!(mapping.target_range.end, 8);
    }

    #[test]
    fn test_no_block_mapping_overlap() {
        /*
            |--------| -> path 1 (one node)
            |ATCGATCG| -> sequence

            |--------| -> path 2 (one node, totally different sequence)
            |TTTTTTTT| -> other sequence

            Mappings: empty
        */
        let conn = &get_connection(None).unwrap();
        Collection::create(conn, "test collection").unwrap();
        let block_group = create_test_block_group(conn);
        let sequence1 = Sequence::new()
            .sequence_type("DNA")
            .sequence("ATCGATCG")
            .save(conn)
            .unwrap();
        let node1_id = Node::create(conn, &sequence1.hash, &HashId::convert_str("1")).unwrap();
        let edge1 = Edge::create(
            conn,
            PATH_START_NODE_ID,
            -1,
            Strand::Forward,
            node1_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge2 = Edge::create(
            conn,
            node1_id,
            8,
            Strand::Forward,
            PATH_END_NODE_ID,
            -1,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = vec![edge1.id, edge2.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: *edge_id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);
        let path1 = Path::create(conn, "chr1", &block_group.id, &edge_ids).unwrap();

        let sequence2 = Sequence::new()
            .sequence_type("DNA")
            .sequence("TTTTTTTT")
            .save(conn)
            .unwrap();
        let node2_id = Node::create(conn, &sequence2.hash, &HashId::convert_str("2")).unwrap();
        let edge3 = Edge::create(
            conn,
            PATH_START_NODE_ID,
            -1,
            Strand::Forward,
            node2_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge4 = Edge::create(
            conn,
            node2_id,
            8,
            Strand::Forward,
            PATH_END_NODE_ID,
            -1,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = vec![edge3.id, edge4.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: *edge_id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let path2 = Path::create(conn, "chr2", &block_group.id, &edge_ids).unwrap();

        let mappings = path1.find_block_mappings(conn, &path2).unwrap();
        assert_eq!(mappings.len(), 0);
    }

    #[test]
    fn test_partial_overlap_block_mapping() {
        /*
            path 1 (one node/sequence):
            |--------|
            |ATCGATCG| -> sequence (0, 8)

            path 2:
            |----| -> (0, 4)
                |--------| -> (4, 12)
            |ATCG| -> shared with path 1
                |TTTTTTTT| -> unrelated sequence

            Mapping: (0, 4) -> (0, 4)
        */
        let conn = &get_connection(None).unwrap();
        Collection::create(conn, "test collection").unwrap();
        let block_group = create_test_block_group(conn);
        let sequence1 = Sequence::new()
            .sequence_type("DNA")
            .sequence("ATCGATCG")
            .save(conn)
            .unwrap();
        let node1_id = Node::create(conn, &sequence1.hash, &HashId::convert_str("1")).unwrap();
        let edge1 = Edge::create(
            conn,
            PATH_START_NODE_ID,
            -1,
            Strand::Forward,
            node1_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge2 = Edge::create(
            conn,
            node1_id,
            8,
            Strand::Forward,
            PATH_END_NODE_ID,
            -1,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = vec![edge1.id, edge2.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: *edge_id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let path1 = Path::create(conn, "chr1", &block_group.id, &edge_ids).unwrap();

        let sequence2 = Sequence::new()
            .sequence_type("DNA")
            .sequence("TTTTTTTT")
            .save(conn)
            .unwrap();
        let node2_id = Node::create(conn, &sequence2.hash, &HashId::convert_str("2")).unwrap();
        let edge3 = Edge::create(
            conn,
            PATH_START_NODE_ID,
            -1,
            Strand::Forward,
            node1_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge4 = Edge::create(
            conn,
            node1_id,
            4,
            Strand::Forward,
            node2_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge5 = Edge::create(
            conn,
            node2_id,
            8,
            Strand::Forward,
            PATH_END_NODE_ID,
            -1,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = vec![edge3.id, edge4.id, edge5.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: *edge_id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let path2 = Path::create(conn, "chr2", &block_group.id, &edge_ids).unwrap();

        assert_eq!(path2.sequence(conn, None).unwrap(), "ATCGTTTTTTTT");

        let mappings = path1.find_block_mappings(conn, &path2).unwrap();
        assert_eq!(mappings.len(), 1);
        let mapping = &mappings[0];
        assert_eq!(mapping.source_range, mapping.target_range);
        assert_eq!(mapping.source_range.start, 0);
        assert_eq!(mapping.source_range.end, 4);
        assert_eq!(mapping.target_range.start, 0);
        assert_eq!(mapping.target_range.end, 4);
    }

    #[test]
    fn test_insertion_block_mapping() {
        /*
            path 1 (one node/sequence):
            |ATCGATCG| -> sequence (0, 8)

            path 2:	Mimics a pure insertion
            |ATCG| -> (0, 4) shared with first half of path 1
                |TTTTTTTT| -> (4, 12) unrelated sequence
                        |ATCG| -> (12, 16) shared with second half of path 1

            Mappings:
            (0, 4) -> (0, 4)
            (4, 8) -> (12, 16)
        */
        let conn = &get_connection(None).unwrap();
        Collection::create(conn, "test collection").unwrap();
        let block_group = create_test_block_group(conn);
        let sequence1 = Sequence::new()
            .sequence_type("DNA")
            .sequence("ATCGATCG")
            .save(conn)
            .unwrap();
        let node1_id = Node::create(conn, &sequence1.hash, &HashId::convert_str("1")).unwrap();
        let edge1 = Edge::create(
            conn,
            PATH_START_NODE_ID,
            -1,
            Strand::Forward,
            node1_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge2 = Edge::create(
            conn,
            node1_id,
            8,
            Strand::Forward,
            PATH_END_NODE_ID,
            -1,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = vec![edge1.id, edge2.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: *edge_id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let path1 = Path::create(conn, "chr1", &block_group.id, &edge_ids).unwrap();

        let sequence2 = Sequence::new()
            .sequence_type("DNA")
            .sequence("TTTTTTTT")
            .save(conn)
            .unwrap();
        let node2_id = Node::create(conn, &sequence2.hash, &HashId::convert_str("2")).unwrap();
        let edge4 = Edge::create(
            conn,
            node1_id,
            4,
            Strand::Forward,
            node2_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge5 = Edge::create(
            conn,
            node2_id,
            8,
            Strand::Forward,
            node1_id,
            4,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = [edge4.id, edge5.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: *edge_id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let path2 = Path::create(
            conn,
            "chr2",
            &block_group.id,
            &[edge1.id, edge4.id, edge5.id, edge2.id],
        )
        .unwrap();

        assert_eq!(path2.sequence(conn, None).unwrap(), "ATCGTTTTTTTTATCG");

        let mappings = path1.find_block_mappings(conn, &path2).unwrap();
        assert_eq!(mappings.len(), 2);
        let mapping1 = &mappings[0];
        assert_eq!(mapping1.source_range, mapping1.target_range);
        assert_eq!(mapping1.source_range.start, 0);
        assert_eq!(mapping1.source_range.end, 4);
        assert_eq!(mapping1.target_range.start, 0);
        assert_eq!(mapping1.target_range.end, 4);

        let mapping2 = &mappings[1];
        assert_eq!(mapping2.source_range.start, 4);
        assert_eq!(mapping2.source_range.end, 8);
        assert_eq!(mapping2.target_range.start, 12);
        assert_eq!(mapping2.target_range.end, 16);
    }

    #[test]
    fn test_replacement_block_mapping() {
        /*
            path 1 (one node/sequence):
            |ATCGATCG| -> sequence (0, 8)

            path 2:	Mimics a replacement
            |AT| -> (0, 2) shared with first two bp of path 1
              |TTTTTTTT| -> (2, 10) unrelated sequence
                      |CG| -> (10, 12) shared with last 2 bp of path 1

            Mappings:
            (0, 2) -> (0, 2)
            (6, 8) -> (10, 12)
        */
        let conn = &get_connection(None).unwrap();
        Collection::create(conn, "test collection").unwrap();
        let block_group = create_test_block_group(conn);
        let sequence1 = Sequence::new()
            .sequence_type("DNA")
            .sequence("ATCGATCG")
            .save(conn)
            .unwrap();
        let node1_id = Node::create(conn, &sequence1.hash, &HashId::convert_str("1")).unwrap();
        let edge1 = Edge::create(
            conn,
            PATH_START_NODE_ID,
            -1,
            Strand::Forward,
            node1_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge2 = Edge::create(
            conn,
            node1_id,
            8,
            Strand::Forward,
            PATH_END_NODE_ID,
            -1,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = [edge1.id, edge2.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: *edge_id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let path1 = Path::create(conn, "chr1", &block_group.id, &edge_ids).unwrap();

        let sequence2 = Sequence::new()
            .sequence_type("DNA")
            .sequence("TTTTTTTT")
            .save(conn)
            .unwrap();
        let node2_id = Node::create(conn, &sequence2.hash, &HashId::convert_str("2")).unwrap();
        let edge4 = Edge::create(
            conn,
            node1_id,
            2,
            Strand::Forward,
            node2_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge5 = Edge::create(
            conn,
            node2_id,
            8,
            Strand::Forward,
            node1_id,
            6,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = [edge4.id, edge5.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: *edge_id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let path2 = Path::create(
            conn,
            "chr2",
            &block_group.id,
            &[edge1.id, edge4.id, edge5.id, edge2.id],
        )
        .unwrap();

        assert_eq!(path2.sequence(conn, None).unwrap(), "ATTTTTTTTTCG");

        let mappings = path1.find_block_mappings(conn, &path2).unwrap();
        assert_eq!(mappings.len(), 2);
        let mapping1 = &mappings[0];
        assert_eq!(mapping1.source_range, mapping1.target_range);
        assert_eq!(mapping1.source_range.start, 0);
        assert_eq!(mapping1.source_range.end, 2);
        assert_eq!(mapping1.target_range.start, 0);
        assert_eq!(mapping1.target_range.end, 2);

        let mapping2 = &mappings[1];
        assert_eq!(mapping2.source_range.start, 6);
        assert_eq!(mapping2.source_range.end, 8);
        assert_eq!(mapping2.target_range.start, 10);
        assert_eq!(mapping2.target_range.end, 12);
    }

    #[test]
    fn test_deletion_block_mapping() {
        /*
            path 1 (one node/sequence):
            |ATCGATCG| -> sequence (0, 8)

            path 2: Mimics a pure deletion
            |AT| -> (0, 2) shared with first two bp of path 1
              |CG| -> (2, 4) shared with last 2 bp of path 1

            Mappings:
            (0, 2) -> (0, 2)
            (6, 8) -> (2, 4)
        */
        let conn = &get_connection(None).unwrap();
        Collection::create(conn, "test collection").unwrap();
        let block_group = create_test_block_group(conn);
        let sequence1 = Sequence::new()
            .sequence_type("DNA")
            .sequence("ATCGATCG")
            .save(conn)
            .unwrap();
        let node1_id = Node::create(conn, &sequence1.hash, &HashId::convert_str("1")).unwrap();
        let edge1 = Edge::create(
            conn,
            PATH_START_NODE_ID,
            -1,
            Strand::Forward,
            node1_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge2 = Edge::create(
            conn,
            node1_id,
            8,
            Strand::Forward,
            PATH_END_NODE_ID,
            -1,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = [edge1.id, edge2.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: *edge_id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let path1 = Path::create(conn, "chr1", &block_group.id, &edge_ids).unwrap();

        let edge4 = Edge::create(
            conn,
            node1_id,
            2,
            Strand::Forward,
            node1_id,
            6,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = [edge4.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: *edge_id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let path2 = Path::create(
            conn,
            "chr2",
            &block_group.id,
            &[edge1.id, edge4.id, edge2.id],
        )
        .unwrap();

        assert_eq!(path2.sequence(conn, None).unwrap(), "ATCG");

        let mappings = path1.find_block_mappings(conn, &path2).unwrap();
        assert_eq!(mappings.len(), 2);
        let mapping1 = &mappings[0];
        assert_eq!(mapping1.source_range, mapping1.target_range);
        assert_eq!(mapping1.source_range.start, 0);
        assert_eq!(mapping1.source_range.end, 2);
        assert_eq!(mapping1.target_range.start, 0);
        assert_eq!(mapping1.target_range.end, 2);

        let mapping2 = &mappings[1];
        assert_eq!(mapping2.source_range.start, 6);
        assert_eq!(mapping2.source_range.end, 8);
        assert_eq!(mapping2.target_range.start, 2);
        assert_eq!(mapping2.target_range.end, 4);
    }

    #[test]
    fn test_two_block_insertion_mapping() {
        /*
            path 1 (two nodes/sequences):
            |ATCGATCG| -> sequence (0, 8)
                    |TTTTTTTT| -> sequence (8, 16)

            path 2: Mimics a pure insertion in the middle of the two blocks
            |ATCGATCG| -> sequence (0, 8)
                    |AAAAAAAA| -> sequence (8, 16)
                            |TTTTTTTT| -> sequence (16, 24)

            Mappings:
            (0, 8) -> (0, 8)
            (8, 16) -> (16, 24)
        */
        let conn = &get_connection(None).unwrap();
        Collection::create(conn, "test collection").unwrap();
        let block_group = create_test_block_group(conn);
        let sequence1 = Sequence::new()
            .sequence_type("DNA")
            .sequence("ATCGATCG")
            .save(conn)
            .unwrap();
        let node1_id = Node::create(conn, &sequence1.hash, &HashId::convert_str("1")).unwrap();
        let sequence2 = Sequence::new()
            .sequence_type("DNA")
            .sequence("TTTTTTTT")
            .save(conn)
            .unwrap();
        let node2_id = Node::create(conn, &sequence2.hash, &HashId::convert_str("2")).unwrap();
        let edge1 = Edge::create(
            conn,
            PATH_START_NODE_ID,
            -1,
            Strand::Forward,
            node1_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge2 = Edge::create(
            conn,
            node1_id,
            8,
            Strand::Forward,
            node2_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge3 = Edge::create(
            conn,
            node2_id,
            8,
            Strand::Forward,
            PATH_END_NODE_ID,
            -1,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = [edge1.id, edge2.id, edge3.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: *edge_id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let path1 = Path::create(conn, "chr1", &block_group.id, &edge_ids).unwrap();

        let sequence3 = Sequence::new()
            .sequence_type("DNA")
            .sequence("AAAAAAAA")
            .save(conn)
            .unwrap();
        let node3_id = Node::create(conn, &sequence3.hash, &HashId::convert_str("3")).unwrap();
        let edge4 = Edge::create(
            conn,
            node1_id,
            8,
            Strand::Forward,
            node3_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge5 = Edge::create(
            conn,
            node3_id,
            8,
            Strand::Forward,
            node2_id,
            0,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = [edge4.id, edge5.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: *edge_id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let path2 = Path::create(
            conn,
            "chr2",
            &block_group.id,
            &[edge1.id, edge4.id, edge5.id, edge3.id],
        )
        .unwrap();

        assert_eq!(
            path2.sequence(conn, None).unwrap(),
            "ATCGATCGAAAAAAAATTTTTTTT"
        );

        let mappings = path1.find_block_mappings(conn, &path2).unwrap();
        assert_eq!(mappings.len(), 2);
        let mapping1 = &mappings[0];
        assert_eq!(mapping1.source_range, mapping1.target_range);
        assert_eq!(mapping1.source_range.start, 0);
        assert_eq!(mapping1.source_range.end, 8);
        assert_eq!(mapping1.target_range.start, 0);
        assert_eq!(mapping1.target_range.end, 8);

        let mapping2 = &mappings[1];
        assert_eq!(mapping2.source_range.start, 8);
        assert_eq!(mapping2.source_range.end, 16);
        assert_eq!(mapping2.target_range.start, 16);
        assert_eq!(mapping2.target_range.end, 24);
    }

    #[test]
    fn test_two_block_replacement_mapping() {
        /*
            path 1 (two nodes/sequences):
            |ATCGATCG| -> sequence (0, 8)
                    |TTTTTTTT| -> sequence (8, 16)

            path 2: Mimics a replacement across the two blocks
            |ATCG| -> sequence (0, 4)
                |AAAAAAAA| -> sequence (4, 12)
                        |TTTT| -> sequence (12, 16)

            Mappings:
            (0, 4) -> (0, 4)
            (12, 16) -> (12, 16)
        */
        let conn = &get_connection(None).unwrap();
        Collection::create(conn, "test collection").unwrap();
        let block_group = create_test_block_group(conn);
        let sequence1 = Sequence::new()
            .sequence_type("DNA")
            .sequence("ATCGATCG")
            .save(conn)
            .unwrap();
        let node1_id = Node::create(conn, &sequence1.hash, &HashId::convert_str("1")).unwrap();
        let sequence2 = Sequence::new()
            .sequence_type("DNA")
            .sequence("TTTTTTTT")
            .save(conn)
            .unwrap();
        let node2_id = Node::create(conn, &sequence2.hash, &HashId::convert_str("2")).unwrap();
        let edge1 = Edge::create(
            conn,
            PATH_START_NODE_ID,
            -1,
            Strand::Forward,
            node1_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge2 = Edge::create(
            conn,
            node1_id,
            8,
            Strand::Forward,
            node2_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge3 = Edge::create(
            conn,
            node2_id,
            8,
            Strand::Forward,
            PATH_END_NODE_ID,
            -1,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = [edge1.id, edge2.id, edge3.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: *edge_id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let path1 = Path::create(conn, "chr1", &block_group.id, &edge_ids).unwrap();

        let sequence3 = Sequence::new()
            .sequence_type("DNA")
            .sequence("AAAAAAAA")
            .save(conn)
            .unwrap();
        let node3_id = Node::create(conn, &sequence3.hash, &HashId::convert_str("3")).unwrap();
        let edge4 = Edge::create(
            conn,
            node1_id,
            4,
            Strand::Forward,
            node3_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge5 = Edge::create(
            conn,
            node3_id,
            8,
            Strand::Forward,
            node2_id,
            4,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = [edge4.id, edge5.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: *edge_id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let path2 = Path::create(
            conn,
            "chr2",
            &block_group.id,
            &[edge1.id, edge4.id, edge5.id, edge3.id],
        )
        .unwrap();

        assert_eq!(path2.sequence(conn, None).unwrap(), "ATCGAAAAAAAATTTT");

        let mappings = path1.find_block_mappings(conn, &path2).unwrap();
        assert_eq!(mappings.len(), 2);
        let mapping1 = &mappings[0];
        assert_eq!(mapping1.source_range, mapping1.target_range);
        assert_eq!(mapping1.source_range.start, 0);
        assert_eq!(mapping1.source_range.end, 4);
        assert_eq!(mapping1.target_range.start, 0);
        assert_eq!(mapping1.target_range.end, 4);

        let mapping2 = &mappings[1];
        assert_eq!(mapping2.source_range.start, 12);
        assert_eq!(mapping2.source_range.end, 16);
        assert_eq!(mapping2.target_range.start, 12);
        assert_eq!(mapping2.target_range.end, 16);
    }

    #[test]
    fn test_two_block_deletion_mapping() {
        /*
            path 1 (two nodes/sequences):
            |ATCGATCG| -> sequence (0, 8)
                    |TTTTTTTT| -> sequence (8, 16)

            path 2: Mimics a deletion across the two blocks
            |ATCG| -> sequence (0, 4)
                |TTTT| -> sequence (4, 8)

            Mappings:
            (0, 4) -> (0, 4)
            (12, 16) -> (4, 8)
        */
        let conn = &get_connection(None).unwrap();
        Collection::create(conn, "test collection").unwrap();
        let block_group = create_test_block_group(conn);
        let sequence1 = Sequence::new()
            .sequence_type("DNA")
            .sequence("ATCGATCG")
            .save(conn)
            .unwrap();
        let node1_id = Node::create(conn, &sequence1.hash, &HashId::convert_str("1")).unwrap();
        let sequence2 = Sequence::new()
            .sequence_type("DNA")
            .sequence("TTTTTTTT")
            .save(conn)
            .unwrap();
        let node2_id = Node::create(conn, &sequence2.hash, &HashId::convert_str("2")).unwrap();
        let edge1 = Edge::create(
            conn,
            PATH_START_NODE_ID,
            -1,
            Strand::Forward,
            node1_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge2 = Edge::create(
            conn,
            node1_id,
            8,
            Strand::Forward,
            node2_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge3 = Edge::create(
            conn,
            node2_id,
            8,
            Strand::Forward,
            PATH_END_NODE_ID,
            -1,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = [edge1.id, edge2.id, edge3.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: *edge_id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let path1 = Path::create(conn, "chr1", &block_group.id, &edge_ids).unwrap();

        let edge4 = Edge::create(
            conn,
            node1_id,
            4,
            Strand::Forward,
            node2_id,
            4,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = [edge4.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: *edge_id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let path2 = Path::create(
            conn,
            "chr2",
            &block_group.id,
            &[edge1.id, edge4.id, edge3.id],
        )
        .unwrap();

        assert_eq!(path2.sequence(conn, None).unwrap(), "ATCGTTTT");

        let mappings = path1.find_block_mappings(conn, &path2).unwrap();
        assert_eq!(mappings.len(), 2);
        let mapping1 = &mappings[0];
        assert_eq!(mapping1.source_range, mapping1.target_range);
        assert_eq!(mapping1.source_range.start, 0);
        assert_eq!(mapping1.source_range.end, 4);
        assert_eq!(mapping1.target_range.start, 0);
        assert_eq!(mapping1.target_range.end, 4);

        let mapping2 = &mappings[1];
        assert_eq!(mapping2.source_range.start, 12);
        assert_eq!(mapping2.source_range.end, 16);
        assert_eq!(mapping2.target_range.start, 4);
        assert_eq!(mapping2.target_range.end, 8);
    }

    #[test]
    fn test_annotation_propagation_full_overlap() {
        /*
            |--------| path: 1 sequence, (0, 8)
            |ATCGATCG|
            |--------| Same path: 1 sequence, (0, 8)

            Mapping: (0, 8) -> (0, 8)
        */
        let conn = &get_connection(None).unwrap();
        Collection::create(conn, "test collection").unwrap();
        let block_group = create_test_block_group(conn);
        let sequence1 = Sequence::new()
            .sequence_type("DNA")
            .sequence("ATCGATCG")
            .save(conn)
            .unwrap();
        let node1_id = Node::create(conn, &sequence1.hash, &HashId::convert_str("1")).unwrap();
        let edge1 = Edge::create(
            conn,
            PATH_START_NODE_ID,
            -1,
            Strand::Forward,
            node1_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge2 = Edge::create(
            conn,
            node1_id,
            8,
            Strand::Forward,
            PATH_END_NODE_ID,
            -1,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = [edge1.id, edge2.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: *edge_id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let path = Path::create(conn, "chr1", &block_group.id, &edge_ids).unwrap();

        let annotation = Annotation {
            name: "foo".to_string(),
            start: 0,
            end: 8,
        };
        let annotations = path
            .propagate_annotations(conn, &path, vec![annotation])
            .unwrap();
        assert_eq!(annotations.len(), 1);
        let result_annotation = &annotations[0];
        assert_eq!(result_annotation.name, "foo");
        assert_eq!(result_annotation.start, 0);
        assert_eq!(result_annotation.end, 8);
    }

    #[test]
    fn test_propagate_annotations_no_overlap() {
        /*
            |--------| -> path 1 (one node)
            |ATCGATCG| -> sequence

            |--------| -> path 2 (one node, totally different sequence)
            |TTTTTTTT| -> other sequence

            Mappings: empty
        */
        let conn = &get_connection(None).unwrap();
        Collection::create(conn, "test collection").unwrap();
        let block_group = create_test_block_group(conn);
        let sequence1 = Sequence::new()
            .sequence_type("DNA")
            .sequence("ATCGATCG")
            .save(conn)
            .unwrap();
        let node1_id = Node::create(conn, &sequence1.hash, &HashId::convert_str("1")).unwrap();
        let edge1 = Edge::create(
            conn,
            PATH_START_NODE_ID,
            -1,
            Strand::Forward,
            node1_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge2 = Edge::create(
            conn,
            node1_id,
            8,
            Strand::Forward,
            PATH_END_NODE_ID,
            -1,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = [edge1.id, edge2.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: *edge_id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let path1 = Path::create(conn, "chr1", &block_group.id, &edge_ids).unwrap();

        let sequence2 = Sequence::new()
            .sequence_type("DNA")
            .sequence("TTTTTTTT")
            .save(conn)
            .unwrap();
        let node2_id = Node::create(conn, &sequence2.hash, &HashId::convert_str("2")).unwrap();
        let edge3 = Edge::create(
            conn,
            PATH_START_NODE_ID,
            -1,
            Strand::Forward,
            node2_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge4 = Edge::create(
            conn,
            node2_id,
            8,
            Strand::Forward,
            PATH_END_NODE_ID,
            -1,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = [edge3.id, edge4.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: *edge_id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let path2 = Path::create(conn, "chr2", &block_group.id, &edge_ids).unwrap();

        let annotation = Annotation {
            name: "foo".to_string(),
            start: 0,
            end: 8,
        };
        let annotations = path1
            .propagate_annotations(conn, &path2, vec![annotation])
            .unwrap();
        assert_eq!(annotations.len(), 0);
    }

    #[test]
    fn test_propagate_annotations_partial_overlap() {
        /*
            path 1 (one node/sequence):
            |--------|
            |ATCGATCG| -> sequence (0, 8)

            path 2:
            |----| -> (0, 4)
                |--------| -> (4, 12)
            |ATCG| -> shared with path 1
                |TTTTTTTT| -> unrelated sequence

            Mapping: (0, 4) -> (0, 4)
        */
        let conn = &get_connection(None).unwrap();
        Collection::create(conn, "test collection").unwrap();
        let block_group = create_test_block_group(conn);
        let sequence1 = Sequence::new()
            .sequence_type("DNA")
            .sequence("ATCGATCG")
            .save(conn)
            .unwrap();
        let node1_id = Node::create(conn, &sequence1.hash, &HashId::convert_str("1")).unwrap();
        let edge1 = Edge::create(
            conn,
            PATH_START_NODE_ID,
            -1,
            Strand::Forward,
            node1_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge2 = Edge::create(
            conn,
            node1_id,
            8,
            Strand::Forward,
            PATH_END_NODE_ID,
            -1,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = vec![edge1.id, edge2.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: *edge_id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let path1 = Path::create(conn, "chr1", &block_group.id, &edge_ids).unwrap();

        let sequence2 = Sequence::new()
            .sequence_type("DNA")
            .sequence("TTTTTTTT")
            .save(conn)
            .unwrap();
        let node2_id = Node::create(conn, &sequence2.hash, &HashId::convert_str("2")).unwrap();
        let edge3 = Edge::create(
            conn,
            PATH_START_NODE_ID,
            -1,
            Strand::Forward,
            node1_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge4 = Edge::create(
            conn,
            node1_id,
            4,
            Strand::Forward,
            node2_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge5 = Edge::create(
            conn,
            node2_id,
            8,
            Strand::Forward,
            PATH_END_NODE_ID,
            -1,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = vec![edge3.id, edge4.id, edge5.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: *edge_id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let path2 = Path::create(conn, "chr2", &block_group.id, &edge_ids).unwrap();

        assert_eq!(path2.sequence(conn, None).unwrap(), "ATCGTTTTTTTT");

        let annotation = Annotation {
            name: "foo".to_string(),
            start: 0,
            end: 8,
        };
        let annotations = path1
            .propagate_annotations(conn, &path2, vec![annotation])
            .unwrap();
        assert_eq!(annotations.len(), 1);
        let result_annotation = &annotations[0];
        assert_eq!(result_annotation.name, "foo");
        assert_eq!(result_annotation.start, 0);
        assert_eq!(result_annotation.end, 4);
    }

    #[test]
    fn test_propagate_annotations_with_insertion() {
        /*
            path 1 (one node/sequence):
            |ATCGATCG| -> sequence (0, 8)

            path 2:	Mimics a pure insertion
            |ATCG| -> (0, 4) shared with first half of path 1
                |TTTTTTTT| -> (4, 12) unrelated sequence
                        |ATCG| -> (12, 16) shared with second half of path 1

            Mappings:
            (0, 4) -> (0, 4)
            (4, 8) -> (12, 16)
        */
        let conn = &get_connection(None).unwrap();
        Collection::create(conn, "test collection").unwrap();
        let block_group = create_test_block_group(conn);
        let sequence1 = Sequence::new()
            .sequence_type("DNA")
            .sequence("ATCGATCG")
            .save(conn)
            .unwrap();
        let node1_id = Node::create(conn, &sequence1.hash, &HashId::convert_str("1")).unwrap();
        let edge1 = Edge::create(
            conn,
            PATH_START_NODE_ID,
            -1,
            Strand::Forward,
            node1_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge2 = Edge::create(
            conn,
            node1_id,
            8,
            Strand::Forward,
            PATH_END_NODE_ID,
            -1,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = vec![edge1.id, edge2.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: *edge_id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let path1 = Path::create(conn, "chr1", &block_group.id, &edge_ids).unwrap();

        let sequence2 = Sequence::new()
            .sequence_type("DNA")
            .sequence("TTTTTTTT")
            .save(conn)
            .unwrap();
        let node2_id = Node::create(conn, &sequence2.hash, &HashId::convert_str("2")).unwrap();
        let edge4 = Edge::create(
            conn,
            node1_id,
            4,
            Strand::Forward,
            node2_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge5 = Edge::create(
            conn,
            node2_id,
            8,
            Strand::Forward,
            node1_id,
            4,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = [edge4.id, edge5.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: *edge_id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let path2 = Path::create(
            conn,
            "chr2",
            &block_group.id,
            &[edge1.id, edge4.id, edge5.id, edge2.id],
        )
        .unwrap();

        assert_eq!(path2.sequence(conn, None).unwrap(), "ATCGTTTTTTTTATCG");

        let annotation = Annotation {
            name: "foo".to_string(),
            start: 0,
            end: 8,
        };

        let annotations = path1
            .propagate_annotations(conn, &path2, vec![annotation])
            .unwrap();
        assert_eq!(annotations.len(), 1);

        // Under the default propagation strategy, the annotation is expanded to cover anything in
        // between parts it covers
        let result_annotation = &annotations[0];
        assert_eq!(result_annotation.name, "foo");
        assert_eq!(result_annotation.start, 0);
        assert_eq!(result_annotation.end, 16);
    }

    #[test]
    fn test_propagate_annotations_with_replacement() {
        /*
            path 1 (one node/sequence):
            |ATCGATCG| -> sequence (0, 8)

            path 2:	Mimics a replacement
            |AT| -> (0, 2) shared with first two bp of path 1
              |TTTTTTTT| -> (2, 10) unrelated sequence
                      |CG| -> (10, 12) shared with last 2 bp of path 1

            Mappings:
            (0, 2) -> (0, 2)
            (6, 8) -> (10, 12)
        */
        let conn = &get_connection(None).unwrap();
        Collection::create(conn, "test collection").unwrap();
        let block_group = create_test_block_group(conn);
        let sequence1 = Sequence::new()
            .sequence_type("DNA")
            .sequence("ATCGATCG")
            .save(conn)
            .unwrap();
        let node1_id = Node::create(conn, &sequence1.hash, &HashId::convert_str("1")).unwrap();
        let edge1 = Edge::create(
            conn,
            PATH_START_NODE_ID,
            -1,
            Strand::Forward,
            node1_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge2 = Edge::create(
            conn,
            node1_id,
            8,
            Strand::Forward,
            PATH_END_NODE_ID,
            -1,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = [edge1.id, edge2.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: *edge_id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let path1 = Path::create(conn, "chr1", &block_group.id, &edge_ids).unwrap();

        let sequence2 = Sequence::new()
            .sequence_type("DNA")
            .sequence("TTTTTTTT")
            .save(conn)
            .unwrap();
        let node2_id = Node::create(conn, &sequence2.hash, &HashId::convert_str("2")).unwrap();
        let edge4 = Edge::create(
            conn,
            node1_id,
            2,
            Strand::Forward,
            node2_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge5 = Edge::create(
            conn,
            node2_id,
            8,
            Strand::Forward,
            node1_id,
            6,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = [edge4.id, edge5.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: *edge_id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let path2 = Path::create(
            conn,
            "chr2",
            &block_group.id,
            &[edge1.id, edge4.id, edge5.id, edge2.id],
        )
        .unwrap();

        assert_eq!(path2.sequence(conn, None).unwrap(), "ATTTTTTTTTCG");

        let annotation = Annotation {
            name: "foo".to_string(),
            start: 0,
            end: 4,
        };

        let annotations = path1
            .propagate_annotations(conn, &path2, vec![annotation])
            .unwrap();
        assert_eq!(annotations.len(), 1);

        // Under the default propagation strategy, the annotation is truncated
        let result_annotation = &annotations[0];
        assert_eq!(result_annotation.name, "foo");
        assert_eq!(result_annotation.start, 0);
        assert_eq!(result_annotation.end, 2);
    }

    #[test]
    fn test_propagate_annotations_with_insertion_across_two_blocks() {
        /*
            path 1 (two nodes/sequences):
            |ATCGATCG| -> sequence (0, 8)
                    |TTTTTTTT| -> sequence (8, 16)

            path 2: Mimics a pure insertion in the middle of the two blocks
            |ATCGATCG| -> sequence (0, 8)
                    |AAAAAAAA| -> sequence (8, 16)
                            |TTTTTTTT| -> sequence (16, 24)

            Mappings:
            (0, 8) -> (0, 8)
            (8, 16) -> (16, 24)
        */
        let conn = &get_connection(None).unwrap();
        Collection::create(conn, "test collection").unwrap();
        let block_group = create_test_block_group(conn);
        let sequence1 = Sequence::new()
            .sequence_type("DNA")
            .sequence("ATCGATCG")
            .save(conn)
            .unwrap();
        let node1_id = Node::create(conn, &sequence1.hash, &HashId::convert_str("1")).unwrap();
        let sequence2 = Sequence::new()
            .sequence_type("DNA")
            .sequence("TTTTTTTT")
            .save(conn)
            .unwrap();
        let node2_id = Node::create(conn, &sequence2.hash, &HashId::convert_str("2")).unwrap();
        let edge1 = Edge::create(
            conn,
            PATH_START_NODE_ID,
            -1,
            Strand::Forward,
            node1_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge2 = Edge::create(
            conn,
            node1_id,
            8,
            Strand::Forward,
            node2_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge3 = Edge::create(
            conn,
            node2_id,
            8,
            Strand::Forward,
            PATH_END_NODE_ID,
            -1,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = [edge1.id, edge2.id, edge3.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: *edge_id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let path1 = Path::create(conn, "chr1", &block_group.id, &edge_ids).unwrap();

        let sequence3 = Sequence::new()
            .sequence_type("DNA")
            .sequence("AAAAAAAA")
            .save(conn)
            .unwrap();
        let node3_id = Node::create(conn, &sequence3.hash, &HashId::convert_str("3")).unwrap();
        let edge4 = Edge::create(
            conn,
            node1_id,
            8,
            Strand::Forward,
            node3_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge5 = Edge::create(
            conn,
            node3_id,
            8,
            Strand::Forward,
            node2_id,
            0,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = [edge4.id, edge5.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: *edge_id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let path2 = Path::create(
            conn,
            "chr2",
            &block_group.id,
            &[edge1.id, edge4.id, edge5.id, edge3.id],
        )
        .unwrap();

        assert_eq!(
            path2.sequence(conn, None).unwrap(),
            "ATCGATCGAAAAAAAATTTTTTTT"
        );

        let annotation = Annotation {
            name: "foo".to_string(),
            start: 0,
            end: 16,
        };

        let annotations = path1
            .propagate_annotations(conn, &path2, vec![annotation])
            .unwrap();
        assert_eq!(annotations.len(), 1);

        // Under the default propagation strategy, the annotation is extended across the inserted
        // region
        let result_annotation = &annotations[0];
        assert_eq!(result_annotation.name, "foo");
        assert_eq!(result_annotation.start, 0);
        assert_eq!(result_annotation.end, 24);
    }

    #[test]
    fn test_propagate_annotations_with_deletion_across_two_blocks() {
        /*
            path 1 (two nodes/sequences):
            |ATCGATCG| -> sequence (0, 8)
                    |TTTTTTTT| -> sequence (8, 16)

            path 2: Mimics a deletion across the two blocks
            |ATCG| -> sequence (0, 4)
                |TTTT| -> sequence (4, 8)

            Mappings:
            (0, 4) -> (0, 4)
            (12, 16) -> (4, 8)
        */
        let conn = &get_connection(None).unwrap();
        Collection::create(conn, "test collection").unwrap();
        let block_group = create_test_block_group(conn);
        let sequence1 = Sequence::new()
            .sequence_type("DNA")
            .sequence("ATCGATCG")
            .save(conn)
            .unwrap();
        let node1_id = Node::create(conn, &sequence1.hash, &HashId::convert_str("1")).unwrap();
        let sequence2 = Sequence::new()
            .sequence_type("DNA")
            .sequence("TTTTTTTT")
            .save(conn)
            .unwrap();
        let node2_id = Node::create(conn, &sequence2.hash, &HashId::convert_str("2")).unwrap();
        let edge1 = Edge::create(
            conn,
            PATH_START_NODE_ID,
            -1,
            Strand::Forward,
            node1_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge2 = Edge::create(
            conn,
            node1_id,
            8,
            Strand::Forward,
            node2_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge3 = Edge::create(
            conn,
            node2_id,
            8,
            Strand::Forward,
            PATH_END_NODE_ID,
            -1,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = [edge1.id, edge2.id, edge3.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: *edge_id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let path1 = Path::create(conn, "chr1", &block_group.id, &edge_ids).unwrap();

        let edge4 = Edge::create(
            conn,
            node1_id,
            4,
            Strand::Forward,
            node2_id,
            4,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = [edge4.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: *edge_id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let path2 = Path::create(
            conn,
            "chr2",
            &block_group.id,
            &[edge1.id, edge4.id, edge3.id],
        )
        .unwrap();

        assert_eq!(path2.sequence(conn, None).unwrap(), "ATCGTTTT");

        let annotation = Annotation {
            name: "foo".to_string(),
            start: 0,
            end: 12,
        };

        let annotations = path1
            .propagate_annotations(conn, &path2, vec![annotation])
            .unwrap();
        assert_eq!(annotations.len(), 1);

        // Under the default propagation strategy, the annotation is truncated
        let result_annotation = &annotations[0];
        assert_eq!(result_annotation.name, "foo");
        assert_eq!(result_annotation.start, 0);
        assert_eq!(result_annotation.end, 4);
    }

    #[test]
    fn test_new_path_with() {
        let conn = &get_connection(None).unwrap();
        Collection::create(conn, "test collection").unwrap();
        let block_group = create_test_block_group(conn);
        let sequence1 = Sequence::new()
            .sequence_type("DNA")
            .sequence("ATCGATCG")
            .save(conn)
            .unwrap();
        let node1_id = Node::create(conn, &sequence1.hash, &HashId::convert_str("1")).unwrap();
        let edge1 = Edge::create(
            conn,
            PATH_START_NODE_ID,
            -123,
            Strand::Forward,
            node1_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let sequence2 = Sequence::new()
            .sequence_type("DNA")
            .sequence("AAAAAAAA")
            .save(conn)
            .unwrap();
        let node2_id = Node::create(conn, &sequence2.hash, &HashId::convert_str("2")).unwrap();
        let edge2 = Edge::create(
            conn,
            node1_id,
            8,
            Strand::Forward,
            node2_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge3 = Edge::create(
            conn,
            node2_id,
            8,
            Strand::Forward,
            PATH_END_NODE_ID,
            -1,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = [edge1.id, edge2.id, edge3.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: *edge_id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let path1 = Path::create(conn, "chr1", &block_group.id, &edge_ids).unwrap();
        assert_eq!(path1.sequence(conn, None).unwrap(), "ATCGATCGAAAAAAAA");

        let sequence3 = Sequence::new()
            .sequence_type("DNA")
            .sequence("CCCCCCCC")
            .save(conn)
            .unwrap();
        let node3_id = Node::create(conn, &sequence3.hash, &HashId::convert_str("3")).unwrap();
        let edge4 = Edge::create(
            conn,
            node1_id,
            4,
            Strand::Forward,
            node3_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge5 = Edge::create(
            conn,
            node3_id,
            8,
            Strand::Forward,
            node2_id,
            3,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = [edge4.id, edge5.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: *edge_id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let path2 = path1.new_path_with(conn, 4, 11, &edge4, &edge5).unwrap();
        assert_eq!(path2.sequence(conn, None).unwrap(), "ATCGCCCCCCCCAAAAA");

        let edge6 = Edge::create(
            conn,
            node1_id,
            4,
            Strand::Forward,
            node3_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge7 = Edge::create(
            conn,
            node3_id,
            8,
            Strand::Forward,
            node1_id,
            7,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = [edge6.id, edge7.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: *edge_id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let path3 = path1.new_path_with(conn, 4, 7, &edge6, &edge7).unwrap();
        assert_eq!(path3.sequence(conn, None).unwrap(), "ATCGCCCCCCCCGAAAAAAAA");
    }

    #[test]
    fn test_new_path_with_deletion() {
        let conn = &get_connection(None).unwrap();
        Collection::create(conn, "test collection").unwrap();
        let block_group = create_test_block_group(conn);
        let sequence1 = Sequence::new()
            .sequence_type("DNA")
            .sequence("ATCGATCG")
            .save(conn)
            .unwrap();
        let node1_id = Node::create(conn, &sequence1.hash, &HashId::convert_str("1")).unwrap();
        let edge1 = Edge::create(
            conn,
            PATH_START_NODE_ID,
            -123,
            Strand::Forward,
            node1_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let sequence2 = Sequence::new()
            .sequence_type("DNA")
            .sequence("AAAAAAAA")
            .save(conn)
            .unwrap();
        let node2_id = Node::create(conn, &sequence2.hash, &HashId::convert_str("2")).unwrap();
        let edge2 = Edge::create(
            conn,
            node1_id,
            8,
            Strand::Forward,
            node2_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge3 = Edge::create(
            conn,
            node2_id,
            8,
            Strand::Forward,
            PATH_END_NODE_ID,
            -1,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = [edge1.id, edge2.id, edge3.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: *edge_id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let path1 = Path::create(conn, "chr1", &block_group.id, &edge_ids).unwrap();
        assert_eq!(path1.sequence(conn, None).unwrap(), "ATCGATCGAAAAAAAA");

        let deletion_edge = Edge::create(
            conn,
            node1_id,
            4,
            Strand::Forward,
            node2_id,
            3,
            Strand::Forward,
        )
        .unwrap();

        let block_group_edge = BlockGroupEdgeData {
            block_group_id: block_group.id,
            edge_id: deletion_edge.id,
            chromosome_index: 0,
            phased: 0,
        };

        BlockGroupEdge::bulk_create(conn, &[block_group_edge]);

        let path2 = path1.new_path_with_deletion(conn, 4, 11).unwrap();
        assert_eq!(path2.sequence(conn, None).unwrap(), "ATCGAAAAA");
    }

    #[test]
    fn test_duplicate_edge_warning() {
        let conn = &get_connection(None).unwrap();
        Collection::create(conn, "test collection").unwrap();
        let block_group = create_test_block_group(conn);
        let sequence1 = Sequence::new()
            .sequence_type("DNA")
            .sequence("ATCGATCG")
            .save(conn)
            .unwrap();
        let node1_id = Node::create(conn, &sequence1.hash, &HashId::convert_str("1")).unwrap();
        let edge1 = Edge::create(
            conn,
            PATH_START_NODE_ID,
            -123,
            Strand::Forward,
            node1_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge2 = Edge::create(
            conn,
            node1_id,
            8,
            Strand::Forward,
            PATH_END_NODE_ID,
            -1,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = vec![edge1.id, edge2.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: *edge_id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        // Should print a warning that there are duplicate edges, but continue
        let _path = Path::create(conn, "chr1", &block_group.id, &edge_ids).unwrap();
    }

    #[test]
    #[should_panic(expected = "Not all edges are in the block group")]
    fn test_edges_must_be_in_path_block_group() {
        let conn = &get_connection(None).unwrap();
        Collection::create(conn, "test collection").unwrap();
        let block_group = create_test_block_group(conn);
        let sequence1 = Sequence::new()
            .sequence_type("DNA")
            .sequence("ATCGATCG")
            .save(conn)
            .unwrap();
        let node1_id = Node::create(conn, &sequence1.hash, &HashId::convert_str("1")).unwrap();
        let edge1 = Edge::create(
            conn,
            PATH_START_NODE_ID,
            -123,
            Strand::Forward,
            node1_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge2 = Edge::create(
            conn,
            node1_id,
            8,
            Strand::Forward,
            PATH_END_NODE_ID,
            -1,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = [edge1.id, edge2.id];
        let _path = Path::create(conn, "chr1", &block_group.id, &edge_ids).unwrap();
    }

    #[test]
    #[should_panic]
    // Panic message is something like "Edges 1 and 2 don't share the same node (3 vs. 4)"
    fn test_consecutive_edges_must_share_a_node() {
        let conn = &get_connection(None).unwrap();
        Collection::create(conn, "test collection").unwrap();
        let block_group = create_test_block_group(conn);
        let sequence1 = Sequence::new()
            .sequence_type("DNA")
            .sequence("ATCGATCG")
            .save(conn)
            .unwrap();
        let node1_id = Node::create(conn, &sequence1.hash, &HashId::convert_str("1")).unwrap();
        let edge1 = Edge::create(
            conn,
            PATH_START_NODE_ID,
            -123,
            Strand::Forward,
            node1_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let sequence2 = Sequence::new()
            .sequence_type("DNA")
            .sequence("AAAAAAAA")
            .save(conn)
            .unwrap();
        let node2_id = Node::create(conn, &sequence2.hash, &HashId::convert_str("2")).unwrap();
        let edge2 = Edge::create(
            conn,
            node2_id,
            8,
            Strand::Forward,
            PATH_END_NODE_ID,
            -1,
            Strand::Forward,
        )
        .unwrap();

        let block_group_edges = vec![
            BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: edge1.id,
                chromosome_index: 0,
                phased: 0,
            },
            BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: edge2.id,
                chromosome_index: 0,
                phased: 0,
            },
        ];
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let edge_ids = vec![edge1.id, edge2.id];

        let _path = Path::create(conn, "chr1", &block_group.id, &edge_ids).unwrap();
    }

    #[test]
    #[should_panic]
    // Panic message is something like "Strand mismatch between consecutive edges 1 and 2"
    fn test_consecutive_edges_must_share_the_same_strand() {
        let conn = &get_connection(None).unwrap();
        Collection::create(conn, "test collection").unwrap();
        let block_group = create_test_block_group(conn);
        let sequence1 = Sequence::new()
            .sequence_type("DNA")
            .sequence("ATCGATCG")
            .save(conn)
            .unwrap();
        let node1_id = Node::create(conn, &sequence1.hash, &HashId::convert_str("1")).unwrap();
        let edge1 = Edge::create(
            conn,
            PATH_START_NODE_ID,
            -123,
            Strand::Forward,
            node1_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let sequence2 = Sequence::new()
            .sequence_type("DNA")
            .sequence("AAAAAAAA")
            .save(conn)
            .unwrap();
        let node2_id = Node::create(conn, &sequence2.hash, &HashId::convert_str("2")).unwrap();
        let edge2 = Edge::create(
            conn,
            node2_id,
            8,
            Strand::Forward,
            PATH_END_NODE_ID,
            -1,
            Strand::Reverse,
        )
        .unwrap();

        let edge_ids = vec![edge1.id, edge2.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: *edge_id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();

        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let _path = Path::create(conn, "chr1", &block_group.id, &edge_ids).unwrap();
    }

    #[test]
    #[should_panic]
    // Panic message is something like "Source coordinate 2 for edge 2 is before target coordinate 4 for edge 1"
    fn test_consecutive_edges_must_have_different_coordinates_on_a_node() {
        let conn = &get_connection(None).unwrap();
        Collection::create(conn, "test collection").unwrap();
        let block_group = create_test_block_group(conn);
        let sequence1 = Sequence::new()
            .sequence_type("DNA")
            .sequence("ATCGATCG")
            .save(conn)
            .unwrap();
        let node1_id = Node::create(conn, &sequence1.hash, &HashId::convert_str("1")).unwrap();
        let edge1 = Edge::create(
            conn,
            PATH_START_NODE_ID,
            -123,
            Strand::Forward,
            node1_id,
            4,
            Strand::Forward,
        )
        .unwrap();
        let sequence2 = Sequence::new()
            .sequence_type("DNA")
            .sequence("AAAAAAAA")
            .save(conn)
            .unwrap();
        let node2_id = Node::create(conn, &sequence2.hash, &HashId::convert_str("2")).unwrap();
        // Source coordinate on node 1 is before target coordinate on node1 for edge1
        let edge2 = Edge::create(
            conn,
            node1_id,
            2,
            Strand::Forward,
            node2_id,
            0,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = vec![edge1.id, edge2.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: *edge_id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let _path = Path::create(conn, "chr1", &block_group.id, &edge_ids).unwrap();
    }

    #[test]
    fn test_node_blocks_for_range() {
        let conn = &get_connection(None).unwrap();
        Collection::create(conn, "test collection").unwrap();
        let block_group = create_test_block_group(conn);
        let sequence1 = Sequence::new()
            .sequence_type("DNA")
            .sequence("ATCGATCG")
            .save(conn)
            .unwrap();
        let node1_id = Node::create(conn, &sequence1.hash, &HashId::convert_str("1")).unwrap();
        let edge1 = Edge::create(
            conn,
            PATH_START_NODE_ID,
            -123,
            Strand::Forward,
            node1_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let sequence2 = Sequence::new()
            .sequence_type("DNA")
            .sequence("AAAAAAAA")
            .save(conn)
            .unwrap();
        let node2_id = Node::create(conn, &sequence2.hash, &HashId::convert_str("2")).unwrap();
        let edge2 = Edge::create(
            conn,
            node1_id,
            8,
            Strand::Forward,
            node2_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge3 = Edge::create(
            conn,
            node2_id,
            8,
            Strand::Forward,
            PATH_END_NODE_ID,
            -1,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = vec![edge1.id, edge2.id, edge3.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: *edge_id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let path = Path::create(conn, "chr1", &block_group.id, &edge_ids).unwrap();

        let intervaltree = path.intervaltree(conn).unwrap();

        let node_blocks1 = path.node_blocks_for_range(&intervaltree, 0, 8);
        let expected_node_blocks1 = vec![NodeIntervalBlock {
            node_id: node1_id,
            start: 0,
            end: 8,
            sequence_start: 0,
            sequence_end: 8,
            strand: Strand::Forward,
        }];
        assert_eq!(node_blocks1, expected_node_blocks1);

        let node_blocks2 = path.node_blocks_for_range(&intervaltree, 0, 4);
        let expected_node_blocks2 = vec![NodeIntervalBlock {
            node_id: node1_id,
            start: 0,
            end: 4,
            sequence_start: 0,
            sequence_end: 4,
            strand: Strand::Forward,
        }];
        assert_eq!(node_blocks2, expected_node_blocks2);

        let node_blocks3 = path.node_blocks_for_range(&intervaltree, 2, 6);
        let expected_node_blocks3 = vec![NodeIntervalBlock {
            node_id: node1_id,
            start: 2,
            end: 6,
            sequence_start: 2,
            sequence_end: 6,
            strand: Strand::Forward,
        }];
        assert_eq!(node_blocks3, expected_node_blocks3);

        let node_blocks4 = path.node_blocks_for_range(&intervaltree, 3, 8);
        let expected_node_blocks4 = vec![NodeIntervalBlock {
            node_id: node1_id,
            start: 3,
            end: 8,
            sequence_start: 3,
            sequence_end: 8,
            strand: Strand::Forward,
        }];
        assert_eq!(node_blocks4, expected_node_blocks4);

        let node_blocks5 = path.node_blocks_for_range(&intervaltree, 6, 10);
        let expected_node_blocks5 = vec![
            NodeIntervalBlock {
                node_id: node1_id,
                start: 6,
                end: 8,
                sequence_start: 6,
                sequence_end: 8,
                strand: Strand::Forward,
            },
            NodeIntervalBlock {
                node_id: node2_id,
                start: 8,
                end: 10,
                sequence_start: 0,
                sequence_end: 2,
                strand: Strand::Forward,
            },
        ];
        assert_eq!(node_blocks5, expected_node_blocks5);

        let node_blocks6 = path.node_blocks_for_range(&intervaltree, 12, 16);
        let expected_node_blocks6 = vec![NodeIntervalBlock {
            node_id: node2_id,
            start: 12,
            end: 16,
            sequence_start: 4,
            sequence_end: 8,
            strand: Strand::Forward,
        }];
        assert_eq!(node_blocks6, expected_node_blocks6);
    }

    #[test]
    fn test_node_blocks_for_range_with_node_parts() {
        let conn = &get_connection(None).unwrap();
        Collection::create(conn, "test collection").unwrap();
        let block_group = create_test_block_group(conn);
        let sequence1 = Sequence::new()
            .sequence_type("DNA")
            .sequence("AAAAAAAA")
            .save(conn)
            .unwrap();
        let node1_id = Node::create(conn, &sequence1.hash, &HashId::convert_str("1")).unwrap();
        let edge1 = Edge::create(
            conn,
            PATH_START_NODE_ID,
            -123,
            Strand::Forward,
            node1_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let sequence2 = Sequence::new()
            .sequence_type("DNA")
            .sequence("TTTTTTTT")
            .save(conn)
            .unwrap();
        let node2_id = Node::create(conn, &sequence2.hash, &HashId::convert_str("2")).unwrap();
        let edge2 = Edge::create(
            conn,
            node1_id,
            5,
            Strand::Forward,
            node2_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge3 = Edge::create(
            conn,
            node2_id,
            8,
            Strand::Forward,
            node1_id,
            6,
            Strand::Forward,
        )
        .unwrap();
        let edge4 = Edge::create(
            conn,
            node1_id,
            8,
            Strand::Forward,
            PATH_END_NODE_ID,
            -1,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = &[edge1.id, edge2.id, edge3.id, edge4.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: *edge_id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let path1 = Path::create(conn, "chr1.1", &block_group.id, &[edge1.id, edge4.id]).unwrap();
        let path2 = Path::create(conn, "chr1.2", &block_group.id, edge_ids).unwrap();

        let intervaltree1 = path1.intervaltree(conn).unwrap();

        let node_blocks1 = path1.node_blocks_for_range(&intervaltree1, 0, 8);
        let expected_node_blocks1 = vec![NodeIntervalBlock {
            node_id: node1_id,
            start: 0,
            end: 8,
            sequence_start: 0,
            sequence_end: 8,
            strand: Strand::Forward,
        }];
        assert_eq!(node_blocks1, expected_node_blocks1);

        let intervaltree2 = path2.intervaltree(conn).unwrap();

        let node_blocks2 = path2.node_blocks_for_range(&intervaltree2, 0, 8);
        let expected_node_blocks2 = vec![
            NodeIntervalBlock {
                node_id: node1_id,
                start: 0,
                end: 5,
                sequence_start: 0,
                sequence_end: 5,
                strand: Strand::Forward,
            },
            NodeIntervalBlock {
                node_id: node2_id,
                start: 5,
                end: 8,
                sequence_start: 0,
                sequence_end: 3,
                strand: Strand::Forward,
            },
        ];
        assert_eq!(node_blocks2, expected_node_blocks2);

        let node_blocks3 = path2.node_blocks_for_range(&intervaltree2, 4, 14);
        let expected_node_blocks3 = vec![
            NodeIntervalBlock {
                node_id: node1_id,
                start: 4,
                end: 5,
                sequence_start: 4,
                sequence_end: 5,
                strand: Strand::Forward,
            },
            NodeIntervalBlock {
                node_id: node2_id,
                start: 5,
                end: 13,
                sequence_start: 0,
                sequence_end: 8,
                strand: Strand::Forward,
            },
            NodeIntervalBlock {
                node_id: node1_id,
                start: 13,
                end: 14,
                sequence_start: 6,
                sequence_end: 7,
                strand: Strand::Forward,
            },
        ];
        assert_eq!(node_blocks3, expected_node_blocks3);
    }
}
