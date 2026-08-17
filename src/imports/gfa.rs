use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader},
    path::Path as FilePath,
    time::{SystemTime, UNIX_EPOCH},
};

use gen_core::{
    HashId, NO_CHROMOSOME_INDEX, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand, is_end_node,
    is_start_node,
};
use gen_graph::{GraphEdge, GraphNode};
use gen_models::{
    block_group::{BlockGroup, NewBlockGroup},
    block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
    collection::Collection,
    db::{DbContext, GraphConnection},
    edge::{Edge, EdgeData},
    errors::{
        BlockGroupError, CollectionError, NodeError, OperationError, PathError, SampleError,
        SequenceError,
    },
    file_types::FileTypes,
    node::Node,
    operations::{OperationFile, OperationInfo, OperationSummary},
    path::Path,
    sample::Sample,
    sequence::Sequence,
    traits::max_rows_per_batch,
};
use itertools::Itertools;
use petgraph::{algo::kosaraju_scc, prelude::UnGraphMap, visit::Dfs};
use rusqlite::ToSql;
use thiserror::Error;

use crate::progress_bar::{get_handler, get_message_bar, get_progress_bar, get_time_elapsed_bar};

const GFA_EDGE_BATCH_SIZE: usize = 50_000;
const GFA_END_SEGMENT_INDEX: u32 = u32::MAX - 1;
const GFA_READ_BUFFER_BYTES: usize = 8 * 1024 * 1024;
const GFA_SEGMENT_BATCH_SIZE: usize = 5_000;
const GFA_SEQUENCE_BATCH_BYTES: usize = 64 * 1024 * 1024;
const GFA_START_SEGMENT_INDEX: u32 = u32::MAX;
const GFA_EDGE_INDEXES_CREATE_SQL: &str = "CREATE UNIQUE INDEX IF NOT EXISTS edge_uidx
     ON edges(source_node_id, source_coordinate, source_strand, target_node_id,
              target_coordinate, target_strand);
     CREATE UNIQUE INDEX IF NOT EXISTS block_group_edges_uidx
     ON block_group_edges(block_group_id, edge_id, chromosome_index, phased);";
const GFA_EDGE_INDEXES_DROP_SQL: &str =
    "DROP INDEX IF EXISTS edge_uidx; DROP INDEX IF EXISTS block_group_edges_uidx;";
const GFA_SEGMENT_STAGING_CREATE_SQL: &str = "DROP TABLE IF EXISTS temp.gfa_import_sequences;
     DROP TABLE IF EXISTS temp.gfa_import_nodes;
     CREATE TEMP TABLE gfa_import_sequences (
         hash BLOB NOT NULL,
         sequence_type TEXT NOT NULL,
         sequence TEXT NOT NULL,
         name TEXT NOT NULL,
         file_path TEXT NOT NULL,
         length INTEGER NOT NULL
     );
     CREATE TEMP TABLE gfa_import_nodes (
         id BLOB NOT NULL,
         sequence_hash BLOB NOT NULL
     );";
const GFA_SEGMENT_STAGING_MATERIALIZE_SQL: &str = "INSERT INTO sequences
         (hash, sequence_type, sequence, name, file_path, length)
     SELECT hash, sequence_type, sequence, name, file_path, length
     FROM temp.gfa_import_sequences
     WHERE TRUE
     ORDER BY hash
     ON CONFLICT(hash) DO NOTHING;
     INSERT INTO nodes (id, sequence_hash)
     SELECT id, sequence_hash
     FROM temp.gfa_import_nodes
     WHERE TRUE
     ORDER BY id
     ON CONFLICT(id) DO NOTHING;";
const GFA_SEGMENT_STAGING_DROP_SQL: &str = "DROP TABLE IF EXISTS temp.gfa_import_nodes;
     DROP TABLE IF EXISTS temp.gfa_import_sequences;";
const GFA_EDGE_STAGING_CREATE_SQL: &str = "DROP TABLE IF EXISTS temp.gfa_import_edges;
     DROP TABLE IF EXISTS temp.gfa_import_block_group_edges;
     CREATE TEMP TABLE gfa_import_edges (
         id BLOB NOT NULL,
         source_node_id BLOB NOT NULL,
         source_coordinate INTEGER NOT NULL,
         source_strand TEXT NOT NULL,
         target_node_id BLOB NOT NULL,
         target_coordinate INTEGER NOT NULL,
         target_strand TEXT NOT NULL
     );
     CREATE TEMP TABLE gfa_import_block_group_edges (
         id BLOB NOT NULL,
         block_group_id BLOB NOT NULL,
         edge_id BLOB NOT NULL,
         chromosome_index INTEGER,
         phased INTEGER NOT NULL,
         created_on INTEGER NOT NULL
     );";
const GFA_EDGE_STAGING_MATERIALIZE_SQL: &str = "INSERT INTO edges
         (id, source_node_id, source_coordinate, source_strand, target_node_id,
          target_coordinate, target_strand)
     SELECT id, source_node_id, source_coordinate, source_strand, target_node_id,
            target_coordinate, target_strand
     FROM temp.gfa_import_edges
     ORDER BY id;
     INSERT INTO block_group_edges
         (id, block_group_id, edge_id, chromosome_index, phased, created_on)
     SELECT id, block_group_id, edge_id, chromosome_index, phased, created_on
     FROM temp.gfa_import_block_group_edges
     ORDER BY id;";
const GFA_EDGE_STAGING_DROP_SQL: &str = "DROP TABLE IF EXISTS temp.gfa_import_edges;
     DROP TABLE IF EXISTS temp.gfa_import_block_group_edges;";

#[derive(Clone, Copy, Debug)]
struct ImportedSegment {
    node_id: HashId,
    length: i64,
}

struct ImportedSegments {
    indices_by_name: HashMap<Box<str>, u32>,
    sparse_indices_by_numeric_name: HashMap<usize, u32>,
    values: Vec<ImportedSegment>,
    link_directions: Vec<u8>,
    indices_by_numeric_name: Vec<u32>,
}

impl ImportedSegments {
    const LINK_SOURCE: u8 = 1;
    const LINK_TARGET: u8 = 2;
    const MISSING_INDEX: u32 = u32::MAX;

    fn new() -> Self {
        Self {
            indices_by_name: HashMap::new(),
            sparse_indices_by_numeric_name: HashMap::new(),
            values: Vec::new(),
            link_directions: Vec::new(),
            indices_by_numeric_name: Vec::new(),
        }
    }

    fn len(&self) -> usize {
        self.values.len()
    }

    fn insert(&mut self, name: &str, segment: ImportedSegment) -> u32 {
        let numeric_name = canonical_numeric_name(name);
        let existing_index = if let Some(numeric_name) = numeric_name {
            self.sparse_indices_by_numeric_name
                .get(&numeric_name)
                .copied()
        } else {
            self.indices_by_name.get(name).copied()
        };
        if let Some(index) = existing_index {
            self.values[index as usize] = segment;
            return index;
        }
        let index = u32::try_from(self.values.len())
            .expect("should validate the imported segment count before insertion");
        if let Some(numeric_name) = numeric_name {
            self.sparse_indices_by_numeric_name
                .insert(numeric_name, index);
        } else {
            self.indices_by_name.insert(Box::<str>::from(name), index);
        }
        self.values.push(segment);
        self.link_directions.push(0);
        index
    }

    fn build_numeric_name_index(&mut self) {
        let max_numeric_name = self.sparse_indices_by_numeric_name.keys().copied().max();
        let Some(max_numeric_name) = max_numeric_name else {
            return;
        };
        if max_numeric_name > self.len().saturating_mul(2).max(1_024) {
            return;
        }
        self.indices_by_numeric_name = vec![Self::MISSING_INDEX; max_numeric_name + 1];
        for (numeric_name, index) in self.sparse_indices_by_numeric_name.drain() {
            self.indices_by_numeric_name[numeric_name] = index;
        }
        self.sparse_indices_by_numeric_name = HashMap::new();
    }

    fn index(&self, name: &str) -> Option<u32> {
        if let Some(numeric_name) = canonical_numeric_name(name) {
            if let Some(index) = self.indices_by_numeric_name.get(numeric_name)
                && *index != Self::MISSING_INDEX
            {
                return Some(*index);
            }
            return self
                .sparse_indices_by_numeric_name
                .get(&numeric_name)
                .copied();
        }
        self.indices_by_name.get(name).copied()
    }

    fn get(&self, name: &str) -> Option<(u32, ImportedSegment)> {
        let index = self.index(name)?;
        Some((index, self.values[index as usize]))
    }

    fn mark_link_source(&mut self, index: u32) {
        self.link_directions[index as usize] |= Self::LINK_SOURCE;
    }

    fn mark_link_target(&mut self, index: u32) {
        self.link_directions[index as usize] |= Self::LINK_TARGET;
    }

    fn iter(&self) -> impl Iterator<Item = (u32, ImportedSegment, u8)> + '_ {
        self.values
            .iter()
            .copied()
            .zip(self.link_directions.iter().copied())
            .enumerate()
            .map(|(index, (segment, link_direction))| {
                (
                    u32::try_from(index).expect("should represent a segment index as u32"),
                    segment,
                    link_direction,
                )
            })
    }
}

fn canonical_numeric_name(name: &str) -> Option<usize> {
    if name.is_empty()
        || !name.bytes().all(|character| character.is_ascii_digit())
        || (name.len() > 1 && name.starts_with('0'))
    {
        return None;
    }
    name.parse().ok()
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct ImportedEdgeKey {
    source_segment_index: u32,
    source_strand: Strand,
    target_segment_index: u32,
    target_strand: Strand,
}

impl ImportedEdgeKey {
    fn new(
        source_segment_index: u32,
        source_strand: Strand,
        target_segment_index: u32,
        target_strand: Strand,
    ) -> Self {
        Self {
            source_segment_index,
            source_strand,
            target_segment_index,
            target_strand,
        }
    }
}

struct GfaConnectivity {
    parents: Vec<u32>,
    component_sizes: Vec<u32>,
    components_have_terminal: Vec<bool>,
}

impl GfaConnectivity {
    fn new(segment_count: usize) -> Self {
        let segment_count = u32::try_from(segment_count)
            .expect("should represent the imported segment count as u32");
        Self {
            parents: (0..segment_count).collect(),
            component_sizes: vec![1; segment_count as usize],
            components_have_terminal: vec![false; segment_count as usize],
        }
    }

    fn root(&mut self, segment_index: u32) -> u32 {
        let mut root = segment_index;
        while self.parents[root as usize] != root {
            root = self.parents[root as usize];
        }
        let mut current = segment_index;
        while self.parents[current as usize] != current {
            let parent = self.parents[current as usize];
            self.parents[current as usize] = root;
            current = parent;
        }
        root
    }

    fn connect(&mut self, source_index: u32, target_index: u32) {
        let mut source_root = self.root(source_index);
        let mut target_root = self.root(target_index);
        if source_root == target_root {
            return;
        }
        if self.component_sizes[source_root as usize] < self.component_sizes[target_root as usize] {
            core::mem::swap(&mut source_root, &mut target_root);
        }
        self.parents[target_root as usize] = source_root;
        self.component_sizes[source_root as usize] += self.component_sizes[target_root as usize];
        self.components_have_terminal[source_root as usize] |=
            self.components_have_terminal[target_root as usize];
    }

    fn mark_terminal(&mut self, segment_index: u32) {
        let root = self.root(segment_index);
        self.components_have_terminal[root as usize] = true;
    }

    fn requires_cycle_breaking(&self) -> bool {
        self.parents.iter().enumerate().any(|(index, parent)| {
            *parent as usize == index
                && self.component_sizes[index] >= 3
                && !self.components_have_terminal[index]
        })
    }
}

struct EdgeBatchWriter<'a> {
    conn: &'a GraphConnection,
    block_group_id: HashId,
    pending_edge_ids: Vec<HashId>,
    pending_edges: Vec<EdgeData>,
    pending_block_group_edges: Vec<BlockGroupEdgeData>,
    imported_edges: HashMap<ImportedEdgeKey, HashId>,
    hash_buffer: String,
    use_staging_tables: bool,
}

impl<'a> EdgeBatchWriter<'a> {
    fn new(
        conn: &'a GraphConnection,
        block_group_id: HashId,
        edge_capacity: usize,
        use_staging_tables: bool,
    ) -> Self {
        Self {
            conn,
            block_group_id,
            pending_edge_ids: Vec::with_capacity(GFA_EDGE_BATCH_SIZE),
            pending_edges: Vec::with_capacity(GFA_EDGE_BATCH_SIZE),
            pending_block_group_edges: Vec::with_capacity(GFA_EDGE_BATCH_SIZE),
            imported_edges: HashMap::with_capacity(edge_capacity),
            hash_buffer: String::new(),
            use_staging_tables,
        }
    }

    fn queue(&mut self, key: ImportedEdgeKey, edge: EdgeData) -> Result<HashId, GFAImportError> {
        if let Some(edge_id) = self.imported_edges.get(&key) {
            return Ok(*edge_id);
        }
        let edge_id = edge.id_hash_with_buffer(&mut self.hash_buffer);
        self.imported_edges.insert(key, edge_id);
        self.pending_edge_ids.push(edge_id);
        self.pending_edges.push(edge);
        if self.pending_edges.len() >= GFA_EDGE_BATCH_SIZE {
            self.flush()?;
        }
        Ok(edge_id)
    }

    fn flush(&mut self) -> Result<(), GFAImportError> {
        if self.pending_edges.is_empty() {
            return Ok(());
        }
        if self.use_staging_tables {
            insert_staged_edges(self.conn, &self.pending_edge_ids, &self.pending_edges)?;
        } else {
            Edge::bulk_create_with_ids(self.conn, &self.pending_edge_ids, &self.pending_edges);
        }
        self.pending_block_group_edges.clear();
        self.pending_block_group_edges
            .extend(
                self.pending_edge_ids
                    .iter()
                    .map(|edge_id| BlockGroupEdgeData {
                        block_group_id: self.block_group_id,
                        edge_id: *edge_id,
                        chromosome_index: NO_CHROMOSOME_INDEX,
                        phased: 0,
                    }),
            );
        if self.use_staging_tables {
            insert_staged_block_group_edges(self.conn, &self.pending_block_group_edges)?;
        } else {
            BlockGroupEdge::bulk_create(self.conn, &self.pending_block_group_edges);
        }
        self.pending_edge_ids.clear();
        self.pending_edges.clear();
        self.pending_block_group_edges.clear();
        Ok(())
    }

    fn create_path(
        &mut self,
        path_name: &str,
        encoded_edge_ids: &[u8],
    ) -> Result<(), GFAImportError> {
        // The streaming parser constructed these edges consecutively from validated segment IDs.
        // Flushing first satisfies the persistence invariant without querying every edge back.
        self.flush()?;
        Path::create_from_encoded_edge_ids_unchecked(
            self.conn,
            path_name,
            &self.block_group_id,
            encoded_edge_ids,
        )?;
        Ok(())
    }
}

#[derive(Debug, Error, PartialEq)]
pub enum GFAImportError {
    #[error("Operation Error: {0}")]
    OperationError(#[from] OperationError),
    #[error("Collection creation error: {0}")]
    CollectionError(#[from] CollectionError),
    #[error("Sample creation error: {0}")]
    SampleError(#[from] SampleError),
    #[error("Node creation error: {0}")]
    NodeError(#[from] NodeError),
    #[error("Path creation error: {0}")]
    PathError(#[from] PathError),
    #[error("Sequence save error: {0}")]
    SequenceError(#[from] SequenceError),
    #[error("Block group creation error: {0}")]
    BlockGroupError(#[from] BlockGroupError),
    #[error("GFA I/O error: {0}")]
    Io(String),
    #[error("Invalid GFA record at line {line}: {reason}")]
    InvalidRecord { line: usize, reason: String },
    #[error("GFA database error: {0}")]
    Database(String),
}

fn invalid_record(line: usize, reason: impl Into<String>) -> GFAImportError {
    GFAImportError::InvalidRecord {
        line,
        reason: reason.into(),
    }
}

fn read_gfa_records(
    gfa_path: &FilePath,
    mut handle_record: impl FnMut(usize, &str) -> Result<(), GFAImportError>,
) -> Result<(), GFAImportError> {
    let file = File::open(gfa_path)
        .map_err(|error| GFAImportError::Io(format!("{}: {error}", gfa_path.display())))?;
    let mut reader = BufReader::with_capacity(GFA_READ_BUFFER_BYTES, file);
    let mut buffer = String::new();
    let mut line_number = 0;
    loop {
        buffer.clear();
        let bytes_read = reader
            .read_line(&mut buffer)
            .map_err(|error| GFAImportError::Io(format!("{}: {error}", gfa_path.display())))?;
        if bytes_read == 0 {
            break;
        }
        line_number += 1;
        let record = buffer.trim_end_matches(['\r', '\n']);
        if !record.is_empty() {
            handle_record(line_number, record)?;
        }
        if buffer.capacity() > GFA_READ_BUFFER_BYTES {
            buffer.shrink_to(GFA_READ_BUFFER_BYTES);
        }
    }
    Ok(())
}

fn required_field<'a>(
    fields: &mut impl Iterator<Item = &'a str>,
    line: usize,
    field_name: &str,
) -> Result<&'a str, GFAImportError> {
    fields
        .next()
        .ok_or_else(|| invalid_record(line, format!("missing {field_name}")))
}

fn parse_direction(direction: &str, line: usize) -> Result<Strand, GFAImportError> {
    match direction {
        "+" => Ok(Strand::Forward),
        "-" => Ok(Strand::Reverse),
        _ => Err(invalid_record(
            line,
            format!("invalid segment direction '{direction}'"),
        )),
    }
}

fn path_steps(path: &str) -> impl Iterator<Item = Result<(&str, Strand), String>> {
    path.split(',').map(|step| {
        let split_index = step
            .len()
            .checked_sub(1)
            .ok_or_else(|| "empty path step".to_string())?;
        let (segment_id, direction) = step.split_at(split_index);
        if segment_id.is_empty() {
            return Err("path step is missing a segment ID".to_string());
        }
        let strand = match direction {
            "+" => Strand::Forward,
            "-" => Strand::Reverse,
            _ => return Err(format!("invalid path step direction '{direction}'")),
        };
        Ok((segment_id, strand))
    })
}

fn walk_steps(walk: &str) -> impl Iterator<Item = Result<(&str, Strand), String>> {
    let mut remaining = walk;
    std::iter::from_fn(move || {
        if remaining.is_empty() {
            return None;
        }
        let (direction, after_direction) = remaining.split_at(1);
        let strand = match direction {
            ">" => Strand::Forward,
            "<" => Strand::Reverse,
            _ => {
                remaining = "";
                return Some(Err(format!("invalid walk direction '{direction}'")));
            }
        };
        let next_direction = after_direction.find(['>', '<']);
        let (segment_id, rest) = if let Some(index) = next_direction {
            after_direction.split_at(index)
        } else {
            (after_direction, "")
        };
        remaining = rest;
        if segment_id.is_empty() {
            remaining = "";
            return Some(Err("walk step is missing a segment ID".to_string()));
        }
        Some(Ok((segment_id, strand)))
    })
}

fn flush_segment_batch(
    conn: &GraphConnection,
    sequences: &mut Vec<Sequence>,
    nodes: &mut Vec<Node>,
) -> Result<(), GFAImportError> {
    if sequences.is_empty() {
        return Ok(());
    }
    insert_staged_sequences(conn, sequences)
        .map_err(|error| GFAImportError::Database(error.to_string()))?;
    insert_staged_nodes(conn, nodes)
        .map_err(|error| GFAImportError::Database(error.to_string()))?;
    sequences.clear();
    nodes.clear();
    Ok(())
}

#[cfg_attr(
    all(debug_assertions, feature = "profiling"),
    tracing::instrument(skip(conn, sequences))
)]
fn insert_staged_sequences(
    conn: &GraphConnection,
    sequences: &[Sequence],
) -> Result<(), rusqlite::Error> {
    let batch_size = max_rows_per_batch(conn, 6);
    for chunk in sequences.chunks(batch_size) {
        let mut sql = String::from(
            "INSERT INTO temp.gfa_import_sequences
             (hash, sequence_type, sequence, name, file_path, length) VALUES ",
        );
        for row_index in 0..chunk.len() {
            if row_index > 0 {
                sql.push(',');
            }
            sql.push_str("(?, ?, ?, ?, ?, ?)");
        }
        sql.push(';');
        let stored_sequences = chunk
            .iter()
            .map(Sequence::stored_sequence)
            .collect::<Vec<_>>();
        let mut parameters = Vec::<&dyn ToSql>::with_capacity(chunk.len() * 6);
        for (sequence, stored_sequence) in chunk.iter().zip(&stored_sequences) {
            parameters.push(&sequence.hash);
            parameters.push(&sequence.sequence_type);
            parameters.push(stored_sequence);
            parameters.push(&sequence.name);
            parameters.push(&sequence.file_path);
            parameters.push(&sequence.length);
        }
        let mut statement = conn.prepare_cached(&sql)?;
        statement.execute(rusqlite::params_from_iter(parameters))?;
    }
    Ok(())
}

#[cfg_attr(
    all(debug_assertions, feature = "profiling"),
    tracing::instrument(skip(conn, nodes))
)]
fn insert_staged_nodes(conn: &GraphConnection, nodes: &[Node]) -> Result<(), rusqlite::Error> {
    let batch_size = max_rows_per_batch(conn, 2);
    for chunk in nodes.chunks(batch_size) {
        let mut sql = String::from("INSERT INTO temp.gfa_import_nodes (id, sequence_hash) VALUES ");
        for row_index in 0..chunk.len() {
            if row_index > 0 {
                sql.push(',');
            }
            sql.push_str("(?, ?)");
        }
        sql.push(';');
        let mut parameters = Vec::<&dyn ToSql>::with_capacity(chunk.len() * 2);
        for node in chunk {
            parameters.push(&node.id);
            parameters.push(&node.sequence_hash);
        }
        let mut statement = conn.prepare_cached(&sql)?;
        statement.execute(rusqlite::params_from_iter(parameters))?;
    }
    Ok(())
}

#[cfg_attr(
    all(debug_assertions, feature = "profiling"),
    tracing::instrument(skip(conn, edge_ids, edges))
)]
fn insert_staged_edges(
    conn: &GraphConnection,
    edge_ids: &[HashId],
    edges: &[EdgeData],
) -> Result<(), GFAImportError> {
    debug_assert_eq!(edge_ids.len(), edges.len());
    let batch_size = max_rows_per_batch(conn, 7);
    for (edge_id_chunk, edge_chunk) in edge_ids.chunks(batch_size).zip(edges.chunks(batch_size)) {
        let mut sql = String::from(
            "INSERT INTO temp.gfa_import_edges
             (id, source_node_id, source_coordinate, source_strand, target_node_id,
              target_coordinate, target_strand) VALUES ",
        );
        for row_index in 0..edge_chunk.len() {
            if row_index > 0 {
                sql.push(',');
            }
            sql.push_str("(?, ?, ?, ?, ?, ?, ?)");
        }
        sql.push(';');
        let mut parameters = Vec::<&dyn ToSql>::with_capacity(edge_chunk.len() * 7);
        for (edge_id, edge) in edge_id_chunk.iter().zip(edge_chunk) {
            parameters.push(edge_id);
            parameters.push(&edge.source_node_id);
            parameters.push(&edge.source_coordinate);
            parameters.push(&edge.source_strand);
            parameters.push(&edge.target_node_id);
            parameters.push(&edge.target_coordinate);
            parameters.push(&edge.target_strand);
        }
        let mut statement = conn
            .prepare_cached(&sql)
            .map_err(|error| GFAImportError::Database(error.to_string()))?;
        statement
            .execute(rusqlite::params_from_iter(parameters))
            .map_err(|error| GFAImportError::Database(error.to_string()))?;
    }
    Ok(())
}

#[cfg_attr(
    all(debug_assertions, feature = "profiling"),
    tracing::instrument(skip(conn, block_group_edges))
)]
fn insert_staged_block_group_edges(
    conn: &GraphConnection,
    block_group_edges: &[BlockGroupEdgeData],
) -> Result<(), GFAImportError> {
    let batch_size = max_rows_per_batch(conn, 6);
    let mut hash_buffer = String::new();
    for chunk in block_group_edges.chunks(batch_size) {
        let timestamp = i64::try_from(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("should calculate a post-epoch timestamp")
                .as_nanos(),
        )
        .expect("should represent the current timestamp as i64 nanoseconds");
        let block_group_edge_ids = chunk
            .iter()
            .map(|block_group_edge| block_group_edge.id_hash_with_buffer(&mut hash_buffer))
            .collect::<Vec<_>>();
        let mut sql = String::from(
            "INSERT INTO temp.gfa_import_block_group_edges
             (id, block_group_id, edge_id, chromosome_index, phased, created_on) VALUES ",
        );
        for row_index in 0..chunk.len() {
            if row_index > 0 {
                sql.push(',');
            }
            sql.push_str("(?, ?, ?, ?, ?, ?)");
        }
        sql.push(';');
        let mut parameters = Vec::<&dyn ToSql>::with_capacity(chunk.len() * 6);
        for (block_group_edge_id, block_group_edge) in block_group_edge_ids.iter().zip(chunk) {
            parameters.push(block_group_edge_id);
            parameters.push(&block_group_edge.block_group_id);
            parameters.push(&block_group_edge.edge_id);
            parameters.push(&block_group_edge.chromosome_index);
            parameters.push(&block_group_edge.phased);
            parameters.push(&timestamp);
        }
        let mut statement = conn
            .prepare_cached(&sql)
            .map_err(|error| GFAImportError::Database(error.to_string()))?;
        statement
            .execute(rusqlite::params_from_iter(parameters))
            .map_err(|error| GFAImportError::Database(error.to_string()))?;
    }
    Ok(())
}

fn create_segment_staging_tables(conn: &GraphConnection) -> Result<(), GFAImportError> {
    conn.execute_batch(GFA_SEGMENT_STAGING_CREATE_SQL)
        .map_err(|error| GFAImportError::Database(error.to_string()))
}

#[cfg_attr(
    all(debug_assertions, feature = "profiling"),
    tracing::instrument(skip(conn))
)]
fn materialize_staged_segments(conn: &GraphConnection) -> Result<(), GFAImportError> {
    conn.execute_batch(GFA_SEGMENT_STAGING_MATERIALIZE_SQL)
        .map_err(|error| GFAImportError::Database(error.to_string()))
}

fn drop_segment_staging_tables(
    conn: &GraphConnection,
    staging_tables_were_created: bool,
) -> Result<(), GFAImportError> {
    if staging_tables_were_created {
        conn.execute_batch(GFA_SEGMENT_STAGING_DROP_SQL)
            .map_err(|error| GFAImportError::Database(error.to_string()))?;
    }
    Ok(())
}

fn create_edge_staging_tables(conn: &GraphConnection) -> Result<(), GFAImportError> {
    conn.execute_batch(GFA_EDGE_STAGING_CREATE_SQL)
        .map_err(|error| GFAImportError::Database(error.to_string()))
}

#[cfg_attr(
    all(debug_assertions, feature = "profiling"),
    tracing::instrument(skip(conn))
)]
fn materialize_staged_edges(conn: &GraphConnection) -> Result<(), GFAImportError> {
    conn.execute_batch(GFA_EDGE_STAGING_MATERIALIZE_SQL)
        .map_err(|error| GFAImportError::Database(error.to_string()))
}

fn drop_edge_staging_tables(
    conn: &GraphConnection,
    staging_tables_were_created: bool,
) -> Result<(), GFAImportError> {
    if staging_tables_were_created {
        conn.execute_batch(GFA_EDGE_STAGING_DROP_SQL)
            .map_err(|error| GFAImportError::Database(error.to_string()))?;
    }
    Ok(())
}

fn complete_import_maintenance(
    import_result: Result<(), GFAImportError>,
    maintenance_results: impl IntoIterator<Item = Result<(), GFAImportError>>,
) -> Result<(), GFAImportError> {
    let maintenance_errors = maintenance_results
        .into_iter()
        .filter_map(Result::err)
        .collect::<Vec<_>>();
    if maintenance_errors.is_empty() {
        return import_result;
    }
    let mut messages = Vec::with_capacity(maintenance_errors.len() + 1);
    if let Err(import_error) = import_result {
        messages.push(import_error.to_string());
    }
    messages.extend(
        maintenance_errors
            .into_iter()
            .map(|error| error.to_string()),
    );
    Err(GFAImportError::Database(messages.join("; ")))
}

fn defer_edge_indexes_for_empty_tables(conn: &GraphConnection) -> Result<bool, GFAImportError> {
    let tables_have_edges = conn
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM edges LIMIT 1)
                    OR EXISTS(SELECT 1 FROM block_group_edges LIMIT 1)",
            [],
            |row| row.get::<_, bool>(0),
        )
        .map_err(|error| GFAImportError::Database(error.to_string()))?;
    if tables_have_edges {
        return Ok(false);
    }
    conn.execute_batch(GFA_EDGE_INDEXES_DROP_SQL)
        .map_err(|error| GFAImportError::Database(error.to_string()))?;
    Ok(true)
}

#[cfg_attr(
    all(debug_assertions, feature = "profiling"),
    tracing::instrument(skip(conn))
)]
fn restore_deferred_edge_indexes(
    conn: &GraphConnection,
    indexes_were_deferred: bool,
) -> Result<(), GFAImportError> {
    if indexes_were_deferred {
        conn.execute_batch(GFA_EDGE_INDEXES_CREATE_SQL)
            .map_err(|error| GFAImportError::Database(error.to_string()))?;
    }
    Ok(())
}

fn create_streamed_path<'a>(
    edge_writer: &mut EdgeBatchWriter<'_>,
    path_name: &str,
    segments: &ImportedSegments,
    steps: impl IntoIterator<Item = Result<(&'a str, Strand), String>>,
    line: usize,
    encoded_edge_ids: &mut Vec<u8>,
    connectivity: &mut GfaConnectivity,
) -> Result<(), GFAImportError> {
    encoded_edge_ids.clear();
    let mut source_node_id = PATH_START_NODE_ID;
    let mut source_coordinate = 0;
    let mut source_strand = Strand::Forward;
    let mut first_segment_index = None;
    let mut source_segment_index = None;
    for step in steps {
        let (segment_id, target_strand) = step.map_err(|reason| invalid_record(line, reason))?;
        let (target_index, target) = segments
            .get(segment_id)
            .ok_or_else(|| invalid_record(line, format!("unknown segment ID '{segment_id}'")))?;
        if let Some(source_index) = source_segment_index {
            connectivity.connect(source_index, target_index);
        } else {
            first_segment_index = Some(target_index);
        }
        let edge = edge_data_from_fields(
            source_node_id,
            source_coordinate,
            source_strand,
            target.node_id,
            target_strand,
        );
        let edge_key = ImportedEdgeKey::new(
            source_segment_index.unwrap_or(GFA_START_SEGMENT_INDEX),
            source_strand,
            target_index,
            target_strand,
        );
        encoded_edge_ids.extend_from_slice(&edge_writer.queue(edge_key, edge)?.0);
        source_node_id = target.node_id;
        source_coordinate = target.length;
        source_strand = target_strand;
        source_segment_index = Some(target_index);
    }
    let edge = edge_data_from_fields(
        source_node_id,
        source_coordinate,
        source_strand,
        PATH_END_NODE_ID,
        Strand::Forward,
    );
    let edge_key = ImportedEdgeKey::new(
        source_segment_index.unwrap_or(GFA_START_SEGMENT_INDEX),
        source_strand,
        GFA_END_SEGMENT_INDEX,
        Strand::Forward,
    );
    encoded_edge_ids.extend_from_slice(&edge_writer.queue(edge_key, edge)?.0);
    if let Some(segment_index) = first_segment_index {
        connectivity.mark_terminal(segment_index);
    }
    if let Some(segment_index) = source_segment_index {
        connectivity.mark_terminal(segment_index);
    }
    edge_writer.create_path(path_name, encoded_edge_ids)?;
    Ok(())
}

#[cfg_attr(
    all(debug_assertions, feature = "profiling"),
    tracing::instrument(skip(conn))
)]
fn break_unbounded_gfa_components(
    conn: &GraphConnection,
    block_group_id: HashId,
) -> Result<(), GFAImportError> {
    let progress_bar = get_handler();
    let bar = progress_bar.add(get_progress_bar(None));
    bar.set_message("Breaking cycles");
    let message_bar = progress_bar.add(get_message_bar());
    let graph = BlockGroup::get_graph(conn, &block_group_id, None)?;
    let mut undirected_graph: UnGraphMap<GraphNode, GraphEdge> = UnGraphMap::new();
    for node in graph.nodes() {
        undirected_graph.add_node(node);
    }
    for (source, target, weights) in graph.all_edges() {
        undirected_graph.add_edge(source, target, weights[0]);
    }
    let connected_components = kosaraju_scc(&undirected_graph);
    drop(undirected_graph);
    let mut new_edges = vec![];
    for subgraph in &connected_components {
        if subgraph.len() >= 3 {
            let mut has_start = false;
            let mut has_end = false;
            for node in subgraph {
                if !has_start && is_start_node(node.node_id) {
                    has_start = true;
                } else if !has_end && is_end_node(node.node_id) {
                    has_end = true;
                }
                if has_start && has_end {
                    break;
                }
            }
            if !has_start && !has_end {
                // Kosaraju returns nodes in arbitrary order. DFS plus rotation gives the cycle a
                // repeatable synthetic entry and exit when the input supplied neither terminal.
                let mut order = vec![];
                let mut depth_first_search = Dfs::new(&graph, subgraph[0]);
                while let Some(node) = depth_first_search.next(&graph) {
                    order.push(node);
                }
                let min_index = order
                    .iter()
                    .enumerate()
                    .min_set_by_key(|(_, node)| node.node_id)[0]
                    .0;
                order.rotate_left(min_index);
                bar.inc(1);
                new_edges.push(edge_data_from_fields(
                    PATH_START_NODE_ID,
                    0,
                    Strand::Forward,
                    order[0].node_id,
                    Strand::Forward,
                ));
                let last_node = order.last().expect("should retain a component node");
                new_edges.push(edge_data_from_fields(
                    last_node.node_id,
                    last_node.sequence_end,
                    Strand::Forward,
                    PATH_END_NODE_ID,
                    Strand::Forward,
                ));
                new_edges.push(edge_data_from_fields(
                    PATH_END_NODE_ID,
                    0,
                    Strand::Forward,
                    PATH_START_NODE_ID,
                    Strand::Forward,
                ));
            } else if has_start && has_end {
                // The component is already bounded by the path or link terminals imported above.
            } else {
                message_bar.set_message(
                    "Path encountered with cycle after start/end node, no cycle breaking will apply.",
                );
            }
        }
    }
    message_bar.finish();
    let new_edge_ids = Edge::bulk_create(conn, &new_edges);
    let new_block_group_edges = new_edge_ids
        .iter()
        .map(|edge_id| BlockGroupEdgeData {
            block_group_id,
            edge_id: *edge_id,
            chromosome_index: NO_CHROMOSOME_INDEX,
            phased: 0,
        })
        .collect::<Vec<_>>();
    BlockGroupEdge::bulk_create(conn, &new_block_group_edges);
    bar.finish();
    Ok(())
}

#[cfg_attr(
    all(debug_assertions, feature = "profiling"),
    tracing::instrument(skip(context, gfa_path, collection_name, sample_name))
)]
pub fn import_gfa(
    context: &DbContext,
    gfa_path: &FilePath,
    collection_name: &str,
    sample_name: &str,
) -> Result<OperationSummary, GFAImportError> {
    let conn = context.graph().conn();
    let progress_bar = get_handler();
    match Collection::create(conn, collection_name) {
        Ok(_) => {}
        Err(CollectionError::Duplicate(_)) => {}
        Err(e) => return Err(GFAImportError::CollectionError(e)),
    }
    match Sample::get_or_create(
        conn,
        gen_models::sample::NewSample {
            name: sample_name,
            ..Default::default()
        },
    ) {
        Ok(_) => {}
        Err(e) => {
            return Err(GFAImportError::SampleError(e));
        }
    }
    let block_group = BlockGroup::create(
        conn,
        NewBlockGroup {
            collection_name,
            sample_name,
            name: "",
            ..Default::default()
        },
    )?;
    let gen_bar = progress_bar.add(get_time_elapsed_bar());
    gen_bar.set_message("Creating Gen objects");

    // The first pass flushes sequence payloads to heap staging tables in bounded batches. Sorting
    // those disk-backed rows once before the primary-key inserts avoids millions of random B-tree
    // writes without retaining the GFA payloads in memory.
    let segment_bar = progress_bar.add(get_progress_bar(None));
    segment_bar.set_message("Importing segments");
    if let Err(staging_error) = create_segment_staging_tables(conn) {
        let setup_error = complete_import_maintenance(
            Err(staging_error),
            [drop_segment_staging_tables(conn, true)],
        )
        .expect_err("should preserve the staging setup error");
        return Err(setup_error);
    }
    let mut segments = ImportedSegments::new();
    let mut sequences = Vec::with_capacity(GFA_SEGMENT_BATCH_SIZE);
    let mut nodes = Vec::with_capacity(GFA_SEGMENT_BATCH_SIZE);
    let mut sequence_batch_bytes = 0_usize;
    let segment_result = (|| -> Result<(), GFAImportError> {
        read_gfa_records(gfa_path, |line, record| {
            let mut fields = record.split_whitespace();
            if fields.next() != Some("S") {
                return Ok(());
            }
            let segment_id = required_field(&mut fields, line, "segment ID")?;
            let input_sequence = required_field(&mut fields, line, "segment sequence")?;
            if !sequences.is_empty()
                && (sequences.len() >= GFA_SEGMENT_BATCH_SIZE
                    || sequence_batch_bytes.saturating_add(input_sequence.len())
                        > GFA_SEQUENCE_BATCH_BYTES)
            {
                flush_segment_batch(conn, &mut sequences, &mut nodes)?;
                sequence_batch_bytes = 0;
            }

            let sequence = Sequence::new()
                .sequence_type("DNA")
                .sequence(input_sequence)
                .build();
            // Node IDs intentionally remain UUIDv7 values so their existing ordering semantics do
            // not change when the persistence calls move from per-row to bounded batches.
            let node = Node {
                id: HashId::uuid7(),
                sequence_hash: sequence.hash,
            };
            if segments.index(segment_id).is_none()
                && segments.len() >= GFA_END_SEGMENT_INDEX as usize
            {
                return Err(invalid_record(
                    line,
                    "GFA contains more than 4,294,967,294 segments",
                ));
            }
            segments.insert(
                segment_id,
                ImportedSegment {
                    node_id: node.id,
                    length: sequence.length,
                },
            );
            sequence_batch_bytes = sequence_batch_bytes.saturating_add(input_sequence.len());
            sequences.push(sequence);
            nodes.push(node);
            if sequences.len() >= GFA_SEGMENT_BATCH_SIZE
                || sequence_batch_bytes >= GFA_SEQUENCE_BATCH_BYTES
            {
                flush_segment_batch(conn, &mut sequences, &mut nodes)?;
                sequence_batch_bytes = 0;
            }
            segment_bar.inc(1);
            Ok(())
        })?;
        flush_segment_batch(conn, &mut sequences, &mut nodes)?;
        materialize_staged_segments(conn)?;
        Ok(())
    })();
    drop(sequences);
    drop(nodes);
    let drop_segment_staging_result = drop_segment_staging_tables(conn, true);
    complete_import_maintenance(segment_result, [drop_segment_staging_result])?;
    segments.build_numeric_name_index();
    segment_bar.finish();

    // The second pass never materializes all links, paths, walks, or canonical edges. Edge batches
    // are persisted as they fill, while a path retains only its ordered ID BLOB until creation.
    let record_bar = progress_bar.add(get_progress_bar(None));
    record_bar.set_message("Importing links and paths");
    // GFA paths usually repeat transitions already declared by link records. Retaining only their
    // compact deterministic IDs prevents those repeated paths from issuing millions of conflict
    // inserts without retaining the much larger edge records themselves.
    // Empty edge tables can rebuild their secondary uniqueness indexes once after the bulk load,
    // which is substantially cheaper than maintaining both indexes for every inserted row.
    let edge_indexes_were_deferred = defer_edge_indexes_for_empty_tables(conn)?;
    let use_staging_tables = edge_indexes_were_deferred;
    if use_staging_tables && let Err(staging_error) = create_edge_staging_tables(conn) {
        let setup_error = complete_import_maintenance(
            Err(staging_error),
            [
                drop_edge_staging_tables(conn, true),
                restore_deferred_edge_indexes(conn, edge_indexes_were_deferred),
            ],
        )
        .expect_err("should preserve the staging setup error");
        return Err(setup_error);
    }
    let mut edge_writer =
        EdgeBatchWriter::new(conn, block_group.id, segments.len(), use_staging_tables);
    let mut encoded_path_edge_ids = Vec::new();
    let mut connectivity = GfaConnectivity::new(segments.len());
    let link_path_result = (|| -> Result<(), GFAImportError> {
        read_gfa_records(gfa_path, |line, record| {
            let mut fields = record.split_whitespace();
            match fields.next() {
                Some("L") => {
                    let source_id = required_field(&mut fields, line, "link source segment ID")?;
                    let source_strand = parse_direction(
                        required_field(&mut fields, line, "link source direction")?,
                        line,
                    )?;
                    let target_id = required_field(&mut fields, line, "link target segment ID")?;
                    let target_strand = parse_direction(
                        required_field(&mut fields, line, "link target direction")?,
                        line,
                    )?;
                    let (source_index, source) = segments.get(source_id).ok_or_else(|| {
                        invalid_record(line, format!("unknown segment ID '{source_id}'"))
                    })?;
                    let (target_index, target) = segments.get(target_id).ok_or_else(|| {
                        invalid_record(line, format!("unknown segment ID '{target_id}'"))
                    })?;
                    segments.mark_link_source(source_index);
                    segments.mark_link_target(target_index);
                    connectivity.connect(source_index, target_index);
                    let edge_key = ImportedEdgeKey::new(
                        source_index,
                        source_strand,
                        target_index,
                        target_strand,
                    );
                    edge_writer.queue(
                        edge_key,
                        edge_data_from_fields(
                            source.node_id,
                            source.length,
                            source_strand,
                            target.node_id,
                            target_strand,
                        ),
                    )?;
                    record_bar.inc(1);
                }
                Some("P") => {
                    let path_name = required_field(&mut fields, line, "path name")?;
                    let steps = required_field(&mut fields, line, "path steps")?;
                    create_streamed_path(
                        &mut edge_writer,
                        path_name,
                        &segments,
                        path_steps(steps),
                        line,
                        &mut encoded_path_edge_ids,
                        &mut connectivity,
                    )?;
                    record_bar.inc(1);
                }
                Some("W") => {
                    let path_name = required_field(&mut fields, line, "walk sample ID")?;
                    let _haplotype_index =
                        required_field(&mut fields, line, "walk haplotype index")?;
                    let _sequence_id = required_field(&mut fields, line, "walk sequence ID")?;
                    let _sequence_start = required_field(&mut fields, line, "walk sequence start")?;
                    let _sequence_end = required_field(&mut fields, line, "walk sequence end")?;
                    let steps = required_field(&mut fields, line, "walk steps")?;
                    create_streamed_path(
                        &mut edge_writer,
                        path_name,
                        &segments,
                        walk_steps(steps),
                        line,
                        &mut encoded_path_edge_ids,
                        &mut connectivity,
                    )?;
                    record_bar.inc(1);
                }
                _ => {}
            }
            Ok(())
        })?;

        for (segment_index, segment, link_direction) in segments.iter() {
            let is_link_source = link_direction & ImportedSegments::LINK_SOURCE != 0;
            let is_link_target = link_direction & ImportedSegments::LINK_TARGET != 0;
            if is_link_source && !is_link_target {
                connectivity.mark_terminal(segment_index);
                let edge_key = ImportedEdgeKey::new(
                    GFA_START_SEGMENT_INDEX,
                    Strand::Forward,
                    segment_index,
                    Strand::Forward,
                );
                edge_writer.queue(
                    edge_key,
                    edge_data_from_fields(
                        PATH_START_NODE_ID,
                        0,
                        Strand::Forward,
                        segment.node_id,
                        Strand::Forward,
                    ),
                )?;
            }
            if is_link_target && !is_link_source {
                connectivity.mark_terminal(segment_index);
                let edge_key = ImportedEdgeKey::new(
                    segment_index,
                    Strand::Forward,
                    GFA_END_SEGMENT_INDEX,
                    Strand::Forward,
                );
                edge_writer.queue(
                    edge_key,
                    edge_data_from_fields(
                        segment.node_id,
                        segment.length,
                        Strand::Forward,
                        PATH_END_NODE_ID,
                        Strand::Forward,
                    ),
                )?;
            }
        }
        edge_writer.flush()?;
        if use_staging_tables {
            materialize_staged_edges(conn)?;
        }
        Ok(())
    })();
    let requires_cycle_breaking = connectivity.requires_cycle_breaking();
    drop(edge_writer);
    drop(encoded_path_edge_ids);
    drop(connectivity);
    drop(segments);
    let drop_staging_result = drop_edge_staging_tables(conn, use_staging_tables);
    let restore_indexes_result = restore_deferred_edge_indexes(conn, edge_indexes_were_deferred);
    complete_import_maintenance(
        link_path_result,
        [drop_staging_result, restore_indexes_result],
    )?;
    record_bar.finish();

    // Most GFA components already touch a path or a natural link endpoint. The compact union-find
    // proof avoids rebuilding the entire persisted graph merely to learn that cycle repair is a
    // no-op; unusual terminal-free components retain the exact existing graph-based behavior.
    if requires_cycle_breaking {
        break_unbounded_gfa_components(conn, block_group.id)?;
    }

    let operation_summary = OperationSummary::new(
        OperationInfo {
            files: vec![
                OperationFile::new(gfa_path.to_str().unwrap().to_string())
                    .set_file_type(FileTypes::GFA),
            ],
            description: "gfa_import".to_string(),
        },
        format!("Imported GFA {path}", path = gfa_path.to_str().unwrap()),
    );
    gen_bar.finish();
    Ok(operation_summary)
}

fn edge_data_from_fields(
    source_node_id: HashId,
    source_coordinate: i64,
    source_strand: Strand,
    target_node_id: HashId,
    target_strand: Strand,
) -> EdgeData {
    EdgeData {
        source_node_id,
        source_coordinate,
        source_strand,
        target_node_id,
        target_coordinate: 0,
        target_strand,
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, io::Write as _, path::PathBuf};

    use gen_core::HASH_ID_SIZE;
    use gen_models::{
        assets::{OperationKind, OperationLog},
        history::{HistoryStore, dolt::DoltHistoryStore},
        operations::commit_operation_summary,
        traits::Query,
    };
    use rusqlite::params;
    use tempfile::NamedTempFile;

    use super::*;
    use crate::test_helpers::setup_gen;

    fn edge_index_count(conn: &GraphConnection) -> i64 {
        conn.query_row(
            "SELECT COUNT(*) FROM sqlite_schema
             WHERE type = 'index' AND name IN ('edge_uidx', 'block_group_edges_uidx')",
            [],
            |row| row.get(0),
        )
        .unwrap()
    }

    fn gfa_staging_table_count(conn: &GraphConnection) -> i64 {
        conn.query_row(
            "SELECT COUNT(*) FROM sqlite_temp_schema
             WHERE type = 'table'
               AND name IN (
                   'gfa_import_sequences', 'gfa_import_nodes',
                   'gfa_import_edges', 'gfa_import_block_group_edges'
               )",
            [],
            |row| row.get(0),
        )
        .unwrap()
    }

    #[test]
    fn test_import_simple_gfa() {
        let mut gfa_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        gfa_path.push("fixtures/simple.gfa");
        let collection_name = "test".to_string();
        let context = setup_gen();
        let conn = context.graph().conn();
        let history_store = DoltHistoryStore::new(conn);
        let operation_summary =
            import_gfa(&context, &gfa_path, &collection_name, Sample::DEFAULT_NAME)
                .expect("should import simple gfa");
        let commit_hash = commit_operation_summary(&context, &operation_summary).unwrap();
        assert_eq!(history_store.current_head().unwrap(), Some(commit_hash));
        let mut operation_logs = OperationLog::all(conn);
        operation_logs.sort_by_key(|operation_log| std::cmp::Reverse(operation_log.created_on));
        assert_eq!(
            operation_logs[0].operation_kind,
            OperationKind::Other("gfa_import".to_string())
        );

        let block_group_id = BlockGroup::get_id(&collection_name, Sample::DEFAULT_NAME, "", None);
        let path = Path::query(
            conn,
            "select * from paths where block_group_id = ?1 AND name = ?2",
            params![block_group_id, "m123"],
        )[0]
        .clone();

        let result = path.sequence(conn, None);
        assert_eq!(result.unwrap(), "ATCGATCGATCGATCGATCGGGAACACACAGAGA");

        let node_count = Node::query(conn, "select * from nodes", rusqlite::params!()).len() as i64;
        assert_eq!(node_count, 6);
    }

    #[test]
    fn test_creates_sample() {
        let mut gfa_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        gfa_path.push("fixtures/simple.gfa");
        let collection_name = "test".to_string();
        let context = setup_gen();
        let conn = context.graph().conn();
        let _ = import_gfa(&context, &gfa_path, &collection_name, "new-sample");
        assert_eq!(
            Sample::get_by_name(conn, "new-sample").unwrap().name,
            "new-sample"
        );
    }

    #[test]
    fn test_import_no_path_gfa() {
        let mut gfa_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        gfa_path.push("fixtures/no_path.gfa");
        let collection_name = "no path".to_string();
        let context = setup_gen();
        let conn = context.graph().conn();
        let _ = import_gfa(&context, &gfa_path, &collection_name, Sample::DEFAULT_NAME);

        let block_group_id = BlockGroup::get_id(&collection_name, Sample::DEFAULT_NAME, "", None);
        let all_sequences = BlockGroup::get_all_sequences(conn, &block_group_id, false).unwrap();
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec!["AAAATTTTGGGGCCCC".to_string()])
        );

        let node_count = Node::query(conn, "select * from nodes", rusqlite::params!()).len() as i64;
        assert_eq!(node_count, 6);
    }

    #[test]
    fn test_import_gfa_with_walk() {
        let mut gfa_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        gfa_path.push("fixtures/walk.gfa");
        let collection_name = "walk".to_string();
        let context = setup_gen();
        let conn = context.graph().conn();
        let _ = import_gfa(&context, &gfa_path, &collection_name, Sample::DEFAULT_NAME);

        let block_group_id = BlockGroup::get_id(&collection_name, Sample::DEFAULT_NAME, "", None);
        let path = Path::query(
            conn,
            "select * from paths where block_group_id = ?1 AND name = ?2",
            params![block_group_id, "291344"],
        )[0]
        .clone();

        let result = path.sequence(conn, None);
        assert_eq!(result.unwrap(), "ACCTACAAATTCAAAC");

        let node_count = Node::query(conn, "select * from nodes", rusqlite::params!()).len() as i64;
        assert_eq!(node_count, 6);
    }

    #[test]
    fn test_import_gfa_with_reverse_strand_edges() {
        let mut gfa_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        gfa_path.push("fixtures/reverse_strand.gfa");
        let collection_name = "test".to_string();
        let context = setup_gen();
        let conn = context.graph().conn();
        let _ = import_gfa(&context, &gfa_path, &collection_name, Sample::DEFAULT_NAME);

        let block_group_id = BlockGroup::get_id(&collection_name, Sample::DEFAULT_NAME, "", None);
        let path = Path::query(
            conn,
            "select * from paths where block_group_id = ?1 AND name = ?2",
            params![block_group_id, "124"],
        )[0]
        .clone();

        let result = path.sequence(conn, None);
        assert_eq!(result.unwrap(), "TATGCCAGCTGCGAATA");

        let node_count = Node::query(conn, "select * from nodes", rusqlite::params!()).len() as i64;
        assert_eq!(node_count, 6);
    }

    #[test]
    fn test_import_gfa_preserves_repeated_path_edges() {
        let gfa_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/gfa/repeated_path_edge.gfa");
        let context = setup_gen();
        let conn = context.graph().conn();

        import_gfa(&context, &gfa_path, "repeated path", Sample::DEFAULT_NAME).unwrap();

        let block_group_id = BlockGroup::get_id("repeated path", Sample::DEFAULT_NAME, "", None);
        let path = Path::query(
            conn,
            "SELECT paths.* FROM paths WHERE block_group_id = ?1 AND name = ?2",
            params![block_group_id, "repeated"],
        )
        .into_iter()
        .next()
        .unwrap();
        let encoded_length = conn
            .query_row(
                "SELECT length(edge_ids) FROM paths WHERE id = ?1",
                params![path.id],
                |row| row.get::<_, i64>(0),
            )
            .unwrap();

        assert_eq!(path.edge_ids.len(), 5);
        assert_eq!(path.edge_ids[1], path.edge_ids[3]);
        assert_eq!(path.sequence(conn, None).unwrap(), "ACAC");
        assert_eq!(encoded_length, 5 * HASH_ID_SIZE as i64);
    }

    #[test]
    fn test_import_gfa_reuses_link_edges_across_paths() {
        let mut gfa_file = NamedTempFile::new().unwrap();
        writeln!(gfa_file, "H\tVN:Z:1.2").unwrap();
        writeln!(gfa_file, "S\t1\tA").unwrap();
        writeln!(gfa_file, "S\t2\tC").unwrap();
        writeln!(gfa_file, "L\t1\t+\t2\t+\t*").unwrap();
        writeln!(gfa_file, "P\tfirst\t1+,2+\t*").unwrap();
        writeln!(gfa_file, "P\tsecond\t1+,2+\t*").unwrap();
        gfa_file.flush().unwrap();
        let context = setup_gen();
        let conn = context.graph().conn();

        import_gfa(
            &context,
            gfa_file.path(),
            "shared path edges",
            Sample::DEFAULT_NAME,
        )
        .unwrap();

        let block_group_id =
            BlockGroup::get_id("shared path edges", Sample::DEFAULT_NAME, "", None);
        let block_group_edge_count = conn
            .query_row(
                "SELECT COUNT(*) FROM block_group_edges WHERE block_group_id = ?1",
                params![block_group_id],
                |row| row.get::<_, i64>(0),
            )
            .unwrap();
        let paths = Path::query(
            conn,
            "SELECT paths.* FROM paths WHERE block_group_id = ?1",
            params![block_group_id],
        );

        assert_eq!(block_group_edge_count, 3);
        assert_eq!(paths.len(), 2);
        assert!(paths.iter().all(|path| path.edge_ids.len() == 3));
        assert_eq!(edge_index_count(conn), 2);
        assert_eq!(gfa_staging_table_count(conn), 0);
    }

    #[test]
    fn test_import_gfa_restores_indexes_after_link_error() {
        let mut gfa_file = NamedTempFile::new().unwrap();
        writeln!(gfa_file, "H\tVN:Z:1.2").unwrap();
        writeln!(gfa_file, "S\t1\tA").unwrap();
        writeln!(gfa_file, "L\t1\t+\tmissing\t+\t*").unwrap();
        gfa_file.flush().unwrap();
        let context = setup_gen();
        let conn = context.graph().conn();

        let result = import_gfa(
            &context,
            gfa_file.path(),
            "invalid link",
            Sample::DEFAULT_NAME,
        );

        assert!(matches!(result, Err(GFAImportError::InvalidRecord { .. })));
        assert_eq!(edge_index_count(conn), 2);
        assert_eq!(gfa_staging_table_count(conn), 0);
    }

    #[test]
    fn test_import_gfa_flushes_multiple_segment_batches() {
        let mut gfa_file = NamedTempFile::new().unwrap();
        for index in 0..=GFA_SEGMENT_BATCH_SIZE {
            writeln!(gfa_file, "S\t{index}\tA{index}").unwrap();
        }
        gfa_file.flush().unwrap();
        let context = setup_gen();
        let conn = context.graph().conn();
        let sequence_count_before = Sequence::all(conn).len();
        let node_count_before = Node::all(conn).len();

        import_gfa(
            &context,
            gfa_file.path(),
            "multiple segment batches",
            Sample::DEFAULT_NAME,
        )
        .unwrap();

        assert_eq!(
            Sequence::all(conn).len(),
            sequence_count_before + GFA_SEGMENT_BATCH_SIZE + 1
        );
        assert_eq!(
            Node::all(conn).len(),
            node_count_before + GFA_SEGMENT_BATCH_SIZE + 1
        );
        assert_eq!(gfa_staging_table_count(conn), 0);
    }

    #[test]
    fn test_import_gfa_discards_staged_segments_after_parse_error() {
        let mut gfa_file = NamedTempFile::new().unwrap();
        for index in 0..GFA_SEGMENT_BATCH_SIZE {
            writeln!(gfa_file, "S\t{index}\tA{index}").unwrap();
        }
        writeln!(gfa_file, "S\tmissing-sequence").unwrap();
        gfa_file.flush().unwrap();
        let context = setup_gen();
        let conn = context.graph().conn();
        let sequence_count_before = Sequence::all(conn).len();
        let node_count_before = Node::all(conn).len();

        let result = import_gfa(
            &context,
            gfa_file.path(),
            "invalid staged segments",
            Sample::DEFAULT_NAME,
        );

        assert!(matches!(result, Err(GFAImportError::InvalidRecord { .. })));
        assert_eq!(Sequence::all(conn).len(), sequence_count_before);
        assert_eq!(Node::all(conn).len(), node_count_before);
        assert_eq!(gfa_staging_table_count(conn), 0);
    }

    #[test]
    fn test_edge_batch_writer_queues_each_canonical_edge_once() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let mut edge_writer =
            EdgeBatchWriter::new(conn, HashId::convert_str("block-group"), 1, false);
        let edge = edge_data_from_fields(
            PATH_START_NODE_ID,
            0,
            Strand::Forward,
            PATH_END_NODE_ID,
            Strand::Forward,
        );
        let edge_key = ImportedEdgeKey::new(
            GFA_START_SEGMENT_INDEX,
            Strand::Forward,
            GFA_END_SEGMENT_INDEX,
            Strand::Forward,
        );

        let first_id = edge_writer.queue(edge_key, edge).unwrap();
        let repeated_id = edge_writer.queue(edge_key, edge).unwrap();

        assert_eq!(first_id, repeated_id, "Repeated edges should use one ID");
        assert_eq!(
            edge_writer.pending_edges.len(),
            1,
            "Repeated edges should create one pending database row"
        );
        assert_eq!(
            edge_writer.imported_edges.len(),
            1,
            "Repeated edges should occupy one deduplication entry"
        );
    }

    #[test]
    fn test_import_anderson_promoters() {
        let mut gfa_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        gfa_path.push("fixtures/anderson_promoters.gfa");
        let collection_name = "anderson promoters".to_string();
        let context = setup_gen();
        let conn = context.graph().conn();
        let _ = import_gfa(&context, &gfa_path, &collection_name, Sample::DEFAULT_NAME);

        let paths = Path::query_for_collection(conn, &collection_name);
        assert_eq!(paths.len(), 20);

        let block_group_id = BlockGroup::get_id(&collection_name, Sample::DEFAULT_NAME, "", None);
        let path = Path::query(
            conn,
            "select * from paths where block_group_id = ?1 AND name = ?2",
            params![block_group_id, "BBa_J23100"],
        )[0]
        .clone();

        let result = path.sequence(conn, None);
        let big_part = "TGCTAGCTACTAGTGAAAGAGGAGAAATACTAGATGGCTTCCTCCGAAGACGTTATCAAAGAGTTCATGCGTTTCAAAGTTCGTATGGAAGGTTCCGTTAACGGTCACGAGTTCGAAATCGAAGGTGAAGGTGAAGGTCGTCCGTACGAAGGTACCCAGACCGCTAAACTGAAAGTTACCAAAGGTGGTCCGCTGCCGTTCGCTTGGGACATCCTGTCCCCGCAGTTCCAGTACGGTTCCAAAGCTTACGTTAAACACCCGGCTGACATCCCGGACTACCTGAAACTGTCCTTCCCGGAAGGTTTCAAATGGGAACGTGTTATGAACTTCGAAGACGGTGGTGTTGTTACCGTTACCCAGGACTCCTCCCTGCAAGACGGTGAGTTCATCTACAAAGTTAAACTGCGTGGTACCAACTTCCCGTCCGACGGTCCGGTTATGCAGAAAAAAACCATGGGTTGGGAAGCTTCCACCGAACGTATGTACCCGGAAGACGGTGCTCTGAAAGGTGAAATCAAAATGCGTCTGAAACTGAAAGACGGTGGTCACTACGACGCTGAAGTTAAAACCACCTACATGGCTAAAAAACCGGTTCAGCTGCCGGGTGCTTACAAAACCGACATCAAACTGGACATCACCTCCCACAACGAAGACTACACCATCGTTGAACAGTACGAACGTGCTGAAGGTCGTCACTCCACCGGTGCTTAATAACGCTGATAGTGCTAGTGTAGATCGCTACTAGAGCCAGGCATCAAATAAAACGAAAGGCTCAGTCGAAAGACTGGGCCTTTCGTTTTATCTGTTGTTTGTCGGTGAACGCTCTCTACTAGAGTCACACTGGCTCACCTTCGGGTGGGCCTTTCTGCGTTTATATACTAGAAGCGGCCGCTGCAGGCTTCCTCGCTCACTGACTCGCTGCGCTCGGTCGTTCGGCTGCGGCGAGCGGTATCAGCTCACTCAAAGGCGGTAATACGGTTATCCACAGAATCAGGGGATAACGCAGGAAAGAACATGTGAGCAAAAGGCCAGCAAAAGGCCAGGAACCGTAAAAAGGCCGCGTTGCTGGCGTTTTTCCATAGGCTCCGCCCCCCTGACGAGCATCACAAAAATCGACGCTCAAGTCAGAGGTGGCGAAACCCGACAGGACTATAAAGATACCAGGCGTTTCCCCCTGGAAGCTCCCTCGTGCGCTCTCCTGTTCCGACCCTGCCGCTTACCGGATACCTGTCCGCCTTTCTCCCTTCGGGAAGCGTGGCGCTTTCTCATAGCTCACGCTGTAGGTATCTCAGTTCGGTGTAGGTCGTTCGCTCCAAGCTGGGCTGTGTGCACGAACCCCCCGTTCAGCCCGACCGCTGCGCCTTATCCGGTAACTATCGTCTTGAGTCCAACCCGGTAAGACACGACTTATCGCCACTGGCAGCAGCCACTGGTAACAGGATTAGCAGAGCGAGGTATGTAGGCGGTGCTACAGAGTTCTTGAAGTGGTGGCCTAACTACGGCTACACTAGAAGGACAGTATTTGGTATCTGCGCTCTGCTGAAGCCAGTTACCTTCGGAAAAAGAGTTGGTAGCTCTTGATCCGGCAAACAAACCACCGCTGGTAGCGGTGGTTTTTTTGTTTGCAAGCAGCAGATTACGCGCAGAAAAAAAGGATCTCAAGAAGATCCTTTGATCTTTTCTACGGGGTCTGACGCTCAGTGGAACGAAAACTCACGTTAAGGGATTTTGGTCATGAGATTATCAAAAAGGATCTTCACCTAGATCCTTTTAAATTAAAAATGAAGTTTTAAATCAATCTAAAGTATATATGAGTAAACTTGGTCTGACAGTTACCAATGCTTAATCAGTGAGGCACCTATCTCAGCGATCTGTCTATTTCGTTCATCCATAGTTGCCTGACTCCCCGTCGTGTAGATAACTACGATACGGGAGGGCTTACCATCTGGCCCCAGTGCTGCAATGATACCGCGAGACCCACGCTCACCGGCTCCAGATTTATCAGCAATAAACCAGCCAGCCGGAAGGGCCGAGCGCAGAAGTGGTCCTGCAACTTTATCCGCCTCCATCCAGTCTATTAATTGTTGCCGGGAAGCTAGAGTAAGTAGTTCGCCAGTTAATAGTTTGCGCAACGTTGTTGCCATTGCTACAGGCATCGTGGTGTCACGCTCGTCGTTTGGTATGGCTTCATTCAGCTCCGGTTCCCAACGATCAAGGCGAGTTACATGATCCCCCATGTTGTGCAAAAAAGCGGTTAGCTCCTTCGGTCCTCCGATCGTTGTCAGAAGTAAGTTGGCCGCAGTGTTATCACTCATGGTTATGGCAGCACTGCATAATTCTCTTACTGTCATGCCATCCGTAAGATGCTTTTCTGTGACTGGTGAGTACTCAACCAAGTCATTCTGAGAATAGTGTATGCGGCGACCGAGTTGCTCTTGCCCGGCGTCAATACGGGATAATACCGCGCCACATAGCAGAACTTTAAAAGTGCTCATCATTGGAAAACGTTCTTCGGGGCGAAAACTCTCAAGGATCTTACCGCTGTTGAGATCCAGTTCGATGTAACCCACTCGTGCACCCAACTGATCTTCAGCATCTTTTACTTTCACCAGCGTTTCTGGGTGAGCAAAAACAGGAAGGCAAAATGCCGCAAAAAAGGGAATAAGGGCGACACGGAAATGTTGAATACTCATACTCTTCCTTTTTCAATATTATTGAAGCATTTATCAGGGTTATTGTCTCATGAGCGGATACATATTTGAATGTATTTAGAAAAATAAACAAATAGGGGTTCCGCGCACATTTCCCCGAAAAGTGCCACCTGACGTCTAAGAAACCATTATTATCATGACATTAACCTATAAAAATAGGCGTATCACGAGGCAGAATTTCAGATAAAAAAAATCCTTAGCTTTCGCTAAGGATGATTTCTGGAATTCGCGGCCGCATCTAGAG";
        let expected_sequence_parts = vec![
            "T",
            "T",
            "G",
            "A",
            "C",
            "G",
            "GCTAGCTCAG",
            "T",
            "CCT",
            "A",
            "GG",
            "T",
            "A",
            "C",
            "A",
            "G",
            big_part,
        ];

        let expected_sequence = expected_sequence_parts.join("");
        assert_eq!(result.unwrap(), expected_sequence);

        let part1 = "T";
        let part3 = "T";
        let part4_5 = vec!["G", "T"];
        let part6 = "A";
        let part7_8 = vec!["C", "T"];
        let part9_10 = vec!["A", "G"];
        let part11 = "GCTAGCTCAG";
        let part12_13 = vec!["T", "C"];
        let part14 = "CCT";
        let part15_16 = vec!["A", "T"];
        let part17 = "GG";
        let part18_19 = vec!["T", "G"];
        let part20 = "A";
        let part21_22 = vec!["T", "C"];
        let part23_24 = vec!["A", "T"];
        let part25_26 = vec!["A", "G"];

        let mut expected_sequences = HashSet::new();
        for part_a in &part4_5 {
            for part_b in &part7_8 {
                for part_c in &part9_10 {
                    for part_d in &part12_13 {
                        for part_e in &part15_16 {
                            for part_f in &part18_19 {
                                for part_g in &part21_22 {
                                    for part_h in &part23_24 {
                                        for part_i in &part25_26 {
                                            let expected_sequence_parts1 = vec![
                                                part1, part3, part_a, part6, part_b, part_c,
                                                part11, part_d, part14, part_e, part17, part_f,
                                                part20, part_g, part_h, part_i, big_part,
                                            ];
                                            let temp_sequence1 = expected_sequence_parts1.join("");
                                            let expected_sequence_parts2 = vec![
                                                part3, part_a, part6, part_b, part_c, part11,
                                                part_d, part14, part_e, part17, part_f, part20,
                                                part_g, part_h, part_i, big_part,
                                            ];
                                            let temp_sequence2 = expected_sequence_parts2.join("");
                                            expected_sequences.insert(temp_sequence1);
                                            expected_sequences.insert(temp_sequence2);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        let all_sequences = BlockGroup::get_all_sequences(conn, &block_group_id, false).unwrap();
        assert_eq!(all_sequences.len(), 1024);
        assert_eq!(all_sequences, expected_sequences);

        let node_count = Node::query(conn, "select * from nodes", rusqlite::params!()).len() as i64;
        assert_eq!(node_count, 28);
    }

    #[test]
    fn test_import_aa_gfa() {
        let mut gfa_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        gfa_path.push("fixtures/aa.gfa");
        let collection_name = "test".to_string();
        let context = setup_gen();
        let conn = context.graph().conn();
        let _ = import_gfa(&context, &gfa_path, &collection_name, Sample::DEFAULT_NAME);

        let block_group_id = BlockGroup::get_id(&collection_name, Sample::DEFAULT_NAME, "", None);
        let path = Path::query(
            conn,
            "select * from paths where block_group_id = ?1 AND name = ?2",
            params![block_group_id, "124"],
        )[0]
        .clone();

        let result = path.sequence(conn, None);
        assert_eq!(result.unwrap(), "AA");

        let all_sequences = BlockGroup::get_all_sequences(conn, &block_group_id, false).unwrap();
        assert_eq!(all_sequences, HashSet::from_iter(vec!["AA".to_string()]));

        let node_count = Node::query(conn, "select * from nodes", rusqlite::params!()).len() as i64;
        assert_eq!(node_count, 4);
    }

    #[test]
    fn test_imports_gfa_with_cycle() {
        let gfa_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/gfa/cycle_no_path.gfa");
        let collection_name = "test".to_string();
        let context = setup_gen();
        let conn = context.graph().conn();
        let _ = import_gfa(&context, &gfa_path, &collection_name, Sample::DEFAULT_NAME);

        let block_group_id = BlockGroup::get_id(&collection_name, Sample::DEFAULT_NAME, "", None);

        let all_sequences = BlockGroup::get_all_sequences(conn, &block_group_id, false).unwrap();
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec!["AAACCCTTTGGGACTCTA".to_string()])
        );
    }

    #[test]
    fn test_breaks_cycle_using_path_node() {
        // here the fixture has a path indicting the cycle starts in the middle of where it would
        // normally be created
        let gfa_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/gfa/cycle_with_path.gfa");
        let collection_name = "test".to_string();
        let context = setup_gen();
        let conn = context.graph().conn();
        let _ = import_gfa(&context, &gfa_path, &collection_name, Sample::DEFAULT_NAME);

        let block_group_id = BlockGroup::get_id(&collection_name, Sample::DEFAULT_NAME, "", None);

        let all_sequences = BlockGroup::get_all_sequences(conn, &block_group_id, false).unwrap();
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec!["TTTGGGACTCTAAAACCC".to_string()])
        );
    }

    #[test]
    fn test_connectivity_requires_fallback_only_for_unbounded_large_components() {
        let mut connectivity = GfaConnectivity::new(6);
        connectivity.connect(0, 1);
        connectivity.connect(1, 2);
        assert!(connectivity.requires_cycle_breaking());

        connectivity.mark_terminal(0);
        assert!(!connectivity.requires_cycle_breaking());

        connectivity.connect(3, 4);
        assert!(!connectivity.requires_cycle_breaking());
        connectivity.connect(4, 5);
        assert!(connectivity.requires_cycle_breaking());
    }

    #[test]
    fn test_imported_segments_use_numeric_index_without_aliasing_names() {
        let mut segments = ImportedSegments::new();
        let numeric_index = segments.insert(
            "1",
            ImportedSegment {
                node_id: HashId::pad_str(1),
                length: 1,
            },
        );
        let leading_zero_index = segments.insert(
            "01",
            ImportedSegment {
                node_id: HashId::pad_str(2),
                length: 2,
            },
        );
        let plus_prefixed_index = segments.insert(
            "+1",
            ImportedSegment {
                node_id: HashId::pad_str(3),
                length: 3,
            },
        );
        let text_index = segments.insert(
            "segment",
            ImportedSegment {
                node_id: HashId::pad_str(4),
                length: 4,
            },
        );

        segments.build_numeric_name_index();

        assert_eq!(segments.index("1"), Some(numeric_index));
        assert_eq!(segments.index("01"), Some(leading_zero_index));
        assert_eq!(segments.index("+1"), Some(plus_prefixed_index));
        assert_eq!(segments.index("segment"), Some(text_index));
        assert_eq!(segments.index("missing"), None);
        assert!(segments.sparse_indices_by_numeric_name.is_empty());
    }
}
