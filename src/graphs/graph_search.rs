use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use gen_core::{HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand};
use gen_graph::{GenGraph, GraphNode, GraphNodeSlice};
use gen_models::{
    db::GraphConnection, locus::GraphLocus, node::Node, sequence::reverse_complement,
};
use petgraph::Direction;
use serde::{Deserialize, Serialize};

/// A position in the graph: a GraphNode (aka Block) plus a local byte offset
/// relative to the start of that GraphNode's sequence slice.
///
/// If the GraphNode has a sequence_start of 100, then offset 5 means sequence
/// position 105 in the underlying stored sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GraphPos {
    pub block: GraphNode,
    /// Local offset within `block`'s sequence slice: `0..=block.length()`.
    pub offset: usize,
}

/// Biological interpretation of the sequences being searched.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum SequenceKind {
    /// Case-sensitive exact byte match.
    ///
    /// No case normalization, no IUPAC expansion, no reverse complement.
    Exact,
    /// Double-stranded DNA search.
    ///
    /// Case-insensitive. Searches the query as provided and as its reverse
    /// complement. Uses IUPAC degenerate matching if the query contains
    /// non-ACGT characters; otherwise uses case-insensitive exact matching.
    #[default]
    Dna,
    /// Single-stranded DNA search.
    ///
    /// Case-insensitive. Searches only the query as provided. Uses IUPAC
    /// degenerate matching if the query contains non-ACGT characters; otherwise uses
    /// case-insensitive exact matching.
    SsDna,
    /// Protein search, case-insensitive.
    Protein,
}

impl SequenceKind {
    /// Compatibility helper for call sites that still use string sequence types.
    pub fn from_seq_type(seq_type: Option<&str>) -> Self {
        match seq_type {
            Some("protein") | Some("Protein") => Self::Protein,
            Some("ssDNA") | Some("ssdna") | Some("DNA") | Some("dna") => Self::SsDna,
            Some("dsDNA") | Some("dsdna") => Self::Dna,
            _ => Self::Dna,
        }
    }

    fn matcher_for_query(self, query: &[u8]) -> fn(u8, u8) -> bool {
        match self {
            Self::Exact => |q: u8, g: u8| q == g,
            Self::Protein => |q, g| q.eq_ignore_ascii_case(&g),
            Self::Dna | Self::SsDna => {
                if query_contains_degenerate_iupac(query) {
                    degenerate_matches
                } else {
                    |q, g| q.eq_ignore_ascii_case(&g)
                }
            }
        }
    }
}

/// Complete search state: which query byte to match next, and where we are in
/// the graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct State {
    /// Index into the query: the next byte that must be matched.
    q_idx: usize,
    block: GraphNode,
    /// Local byte offset within contained sequence slice.
    offset: usize,
}

enum StepResult {
    /// Single successor state.
    Advance(State),
    /// Multiple successor states, usually at a graph junction.
    Branch(Vec<State>),
    /// No match; this path is dead.
    Dead,
}

#[inline]
fn is_degenerate_iupac(byte: u8) -> bool {
    matches!(
        byte.to_ascii_uppercase(),
        b'R' | b'Y' | b'S' | b'W' | b'K' | b'M' | b'B' | b'D' | b'H' | b'V' | b'N'
    )
}

#[inline]
fn query_contains_degenerate_iupac(query: &[u8]) -> bool {
    query.iter().any(|&byte| is_degenerate_iupac(byte))
}

/// IUPAC degenerate query matching against a concrete graph byte.
///
/// Both bytes are uppercased before comparison. Degenerate codes are
/// interpreted in the query only; unknown query codes fall back to exact match.
#[inline]
fn degenerate_matches(query_byte: u8, graph_byte: u8) -> bool {
    let query_byte = query_byte.to_ascii_uppercase();
    let graph_byte = graph_byte.to_ascii_uppercase();

    match query_byte {
        b'A' => graph_byte == b'A',
        b'C' => graph_byte == b'C',
        b'G' => graph_byte == b'G',
        b'T' => graph_byte == b'T',
        b'U' => graph_byte == b'U',
        b'N' => matches!(graph_byte, b'A' | b'C' | b'G' | b'T'),
        b'R' => matches!(graph_byte, b'A' | b'G'),
        b'Y' => matches!(graph_byte, b'C' | b'T'),
        b'S' => matches!(graph_byte, b'G' | b'C'),
        b'W' => matches!(graph_byte, b'A' | b'T'),
        b'K' => matches!(graph_byte, b'G' | b'T'),
        b'M' => matches!(graph_byte, b'A' | b'C'),
        b'B' => matches!(graph_byte, b'C' | b'G' | b'T'),
        b'D' => matches!(graph_byte, b'A' | b'G' | b'T'),
        b'H' => matches!(graph_byte, b'A' | b'C' | b'T'),
        b'V' => matches!(graph_byte, b'A' | b'C' | b'G'),
        other => graph_byte == other,
    }
}

pub struct GenGraphMatcher {
    graph: GenGraph,
    sequence_kind: SequenceKind,
    /// Pre-fetched GraphNode sequence bytes, keyed by `GraphNode::node_id`.
    node_sequences: HashMap<HashId, Vec<u8>>,
}

impl GenGraphMatcher {
    /// Build a double-stranded DNA matcher from a database connection and graph.
    ///
    /// Batch-loads all full node sequences up front. No further database access
    /// occurs during matching.
    pub fn new(conn: &GraphConnection, graph: GenGraph) -> Self {
        Self::new_with_sequence_kind(conn, graph, SequenceKind::Dna)
    }

    /// Build a single-stranded DNA matcher from a database connection and graph.
    ///
    /// Batch-loads all full node sequences up front. No further database access
    /// occurs during matching.
    pub fn new_ssdna(conn: &GraphConnection, graph: GenGraph) -> Self {
        Self::new_with_sequence_kind(conn, graph, SequenceKind::SsDna)
    }

    /// Build a protein matcher from a database connection and graph.
    ///
    /// Batch-loads all full node sequences up front. No further database access
    /// occurs during matching.
    pub fn new_protein(conn: &GraphConnection, graph: GenGraph) -> Self {
        Self::new_with_sequence_kind(conn, graph, SequenceKind::Protein)
    }

    /// Build a matcher from a database connection, graph, and sequence kind.
    ///
    /// Batch-loads all full node sequences up front. No further database access
    /// occurs during matching.
    pub fn new_with_sequence_kind(
        conn: &GraphConnection,
        graph: GenGraph,
        sequence_kind: SequenceKind,
    ) -> Self {
        let node_ids: Vec<HashId> = {
            let mut ids: Vec<HashId> = graph.nodes().map(|n| n.node_id).collect();
            ids.sort_unstable();
            ids.dedup();
            ids
        };

        let mut node_sequences: HashMap<HashId, Vec<u8>> =
            Node::get_sequences_by_node_ids(conn, &node_ids)
                .into_iter()
                .map(|(node_id, seq)| {
                    (
                        node_id,
                        seq.get_sequence(None, None)
                            .expect("failed to get sequence")
                            .into_bytes(),
                    )
                })
                .collect();

        // PATH_START and PATH_END have empty sequences.
        node_sequences.entry(PATH_START_NODE_ID).or_default();
        node_sequences.entry(PATH_END_NODE_ID).or_default();

        Self {
            graph,
            sequence_kind,
            node_sequences,
        }
    }

    pub fn sequence_kind(&self) -> SequenceKind {
        self.sequence_kind
    }

    pub fn set_sequence_kind(&mut self, sequence_kind: SequenceKind) {
        self.sequence_kind = sequence_kind;
    }

    /// Returns `true` if `query` occurs anywhere in the graph using this
    /// matcher's configured sequence kind.
    pub fn contains(&self, query: &[u8]) -> bool {
        if query.is_empty() {
            return true;
        }

        let matcher = self.sequence_kind.matcher_for_query(query);
        if self.contains_query_orientation(query, matcher) {
            return true;
        }

        if self.sequence_kind == SequenceKind::Dna {
            let rc = reverse_complement(query);
            let rc_matcher = self.sequence_kind.matcher_for_query(&rc);
            if self.contains_query_orientation(&rc, rc_matcher) {
                return true;
            }
        }

        false
    }

    /// Returns every match of `query` in the graph using this matcher's
    /// configured sequence kind.
    ///
    /// For `SequenceKind::Dna`, reverse-complement matches are returned in graph
    /// coordinates exactly like forward matches. The returned strand field
    /// records which orientation matched - `Forward` for the query as-provided,
    /// `Reverse` for the reverse-complement.
    pub fn find_all(&self, query: &[u8]) -> Vec<GraphLocus> {
        if query.is_empty() {
            return Vec::new();
        }

        let matcher = self.sequence_kind.matcher_for_query(query);

        if self.sequence_kind == SequenceKind::Dna {
            let mut out = self.find_all_query_orientation(query, matcher, Strand::Forward);
            let rc = reverse_complement(query);
            let rc_matcher = self.sequence_kind.matcher_for_query(&rc);
            out.append(&mut self.find_all_query_orientation(&rc, rc_matcher, Strand::Reverse));
            out
        } else {
            self.find_all_query_orientation(query, matcher, Strand::Unknown)
        }
    }

    /// Returns every match of `query` using a seed index to prune start positions.
    ///
    /// For `SequenceKind::Dna`, also searches the reverse complement and tags
    /// results with `Strand::Forward` or `Strand::Reverse`. Does not expand
    /// IUPAC-degenerate bases. Returns an error if the query contains degenerate
    /// bases, because an exact seed lookup would silently miss valid matches.
    pub fn find_all_with_seed_index(
        &self,
        seed_index: &SeedIndex,
        query: &[u8],
    ) -> Result<Vec<GraphLocus>, SeedIndexSearchError> {
        if query.iter().any(|&byte| is_degenerate_iupac(byte)) {
            return Err(SeedIndexSearchError::UnsupportedQuery);
        }

        if query.is_empty() {
            return Ok(Vec::new());
        }

        let matcher: fn(u8, u8) -> bool = if seed_index.normalized {
            |q, g| q.eq_ignore_ascii_case(&g)
        } else {
            |q: u8, g: u8| q == g
        };

        if self.sequence_kind == SequenceKind::Dna {
            let mut out =
                self.index_search_one_orientation(seed_index, query, matcher, Strand::Forward);
            let rc = reverse_complement(query);
            out.append(&mut self.index_search_one_orientation(
                seed_index,
                &rc,
                matcher,
                Strand::Reverse,
            ));
            Ok(out)
        } else {
            Ok(self.index_search_one_orientation(seed_index, query, matcher, Strand::Unknown))
        }
    }

    fn index_search_one_orientation(
        &self,
        seed_index: &SeedIndex,
        query: &[u8],
        matcher: fn(u8, u8) -> bool,
        strand: Strand,
    ) -> Vec<GraphLocus> {
        if query.len() < seed_index.k {
            return self.find_all_forward(query, matcher, strand);
        }

        let seed: Vec<u8> = if seed_index.normalized {
            query[..seed_index.k]
                .iter()
                .map(|&b| b.to_ascii_uppercase())
                .collect()
        } else {
            query[..seed_index.k].to_vec()
        };

        let Some(positions) = seed_index.table.get(&seed) else {
            return Vec::new();
        };

        let mut out = Vec::new();
        for &pos in positions {
            self.collect_matches_from(pos, query, matcher, &mut out, strand);
        }
        out
    }

    fn find_all_forward(
        &self,
        query: &[u8],
        matcher: fn(u8, u8) -> bool,
        strand: Strand,
    ) -> Vec<GraphLocus> {
        if query.is_empty() {
            return Vec::new();
        }

        self.find_all_query_orientation(query, matcher, strand)
    }

    fn contains_query_orientation(&self, query: &[u8], matcher: fn(u8, u8) -> bool) -> bool {
        self.all_start_positions()
            .into_iter()
            .any(|start| self.contains_from(start, query, matcher))
    }

    fn find_all_query_orientation(
        &self,
        query: &[u8],
        matcher: fn(u8, u8) -> bool,
        strand: Strand,
    ) -> Vec<GraphLocus> {
        let mut out = Vec::new();

        for start in self.all_start_positions() {
            self.collect_matches_from(start, query, matcher, &mut out, strand);
        }

        out
    }

    /// DFS from a single start position; returns `true` on the first complete match.
    fn contains_from(&self, start: GraphPos, query: &[u8], matcher: fn(u8, u8) -> bool) -> bool {
        let initial = State {
            q_idx: 0,
            block: start.block,
            offset: start.offset,
        };

        let mut stack = vec![initial];
        let mut dead: HashSet<State> = HashSet::new();

        while let Some(state) = stack.pop() {
            if state.q_idx == query.len() {
                return true;
            }

            if dead.contains(&state) {
                continue;
            }

            match self.step(state, query[state.q_idx], matcher) {
                StepResult::Advance(next) => stack.push(next),
                StepResult::Branch(nexts) => stack.extend(nexts),
                StepResult::Dead => {
                    dead.insert(state);
                }
            }
        }

        false
    }

    /// DFS from a single start position; appends every complete match to `out`.
    fn collect_matches_from(
        &self,
        start: GraphPos,
        query: &[u8],
        matcher: fn(u8, u8) -> bool,
        out: &mut Vec<GraphLocus>,
        strand: Strand,
    ) {
        #[derive(Clone, Debug)]
        struct TraceState {
            state: State,
            start_offset: usize,
            path: Vec<GraphNode>,
        }

        let initial = TraceState {
            state: State {
                q_idx: 0,
                block: start.block,
                offset: start.offset,
            },
            start_offset: start.offset,
            path: vec![start.block],
        };

        let mut stack = vec![initial];

        while let Some(ts) = stack.pop() {
            if ts.state.q_idx == query.len() {
                let n = ts.path.len();
                let slices = ts
                    .path
                    .into_iter()
                    .enumerate()
                    .map(|(i, node)| GraphNodeSlice {
                        block: node,
                        start: if i == 0 { ts.start_offset } else { 0 },
                        end: if i == n - 1 {
                            ts.state.offset
                        } else {
                            node.length() as usize
                        },
                        strand,
                    })
                    .collect();
                out.push(GraphLocus { slices });
                continue;
            }

            match self.step(ts.state, query[ts.state.q_idx], matcher) {
                StepResult::Advance(next) => {
                    let mut next_ts = ts;

                    if next.block != next_ts.state.block {
                        next_ts.path.push(next.block);
                    }

                    next_ts.state = next;
                    stack.push(next_ts);
                }
                StepResult::Branch(nexts) => {
                    for next in nexts {
                        let mut next_ts = ts.clone();

                        if next.block != next_ts.state.block {
                            next_ts.path.push(next.block);
                        }

                        next_ts.state = next;
                        stack.push(next_ts);
                    }
                }
                StepResult::Dead => {}
            }
        }
    }

    fn step(&self, state: State, query_byte: u8, matcher: fn(u8, u8) -> bool) -> StepResult {
        let text = self.graph_node_text(state.block);

        if state.offset < text.len() {
            if matcher(query_byte, text[state.offset]) {
                return StepResult::Advance(State {
                    q_idx: state.q_idx + 1,
                    block: state.block,
                    offset: state.offset + 1,
                });
            }

            return StepResult::Dead;
        }

        // At node boundary: traverse outgoing neighbors without consuming a
        // query byte.
        let nexts: Vec<State> = self
            .graph
            .neighbors_directed(state.block, Direction::Outgoing)
            .map(|next_node| State {
                q_idx: state.q_idx,
                block: next_node,
                offset: 0,
            })
            .collect();

        match nexts.len() {
            0 => StepResult::Dead,
            1 => StepResult::Advance(nexts[0]),
            _ => StepResult::Branch(nexts),
        }
    }

    /// Enumerate every valid starting position: every byte offset inside every
    /// non-empty node slice.
    fn all_start_positions(&self) -> Vec<GraphPos> {
        let mut starts = Vec::new();

        for node in self.graph.nodes() {
            let len = self.graph_node_text(node).len();
            for offset in 0..len {
                starts.push(GraphPos {
                    block: node,
                    offset,
                });
            }
        }

        starts
    }

    /// Get GraphNode sequence from the local sequence store.
    ///
    /// Panics if `node.node_id` is absent from `node_sequences`.
    pub(crate) fn graph_node_text(&self, node: GraphNode) -> &[u8] {
        let full = self
            .node_sequences
            .get(&node.node_id)
            .unwrap_or_else(|| panic!("missing sequence for node_id {:?}", node.node_id));

        let start = usize::try_from(node.sequence_start).expect("negative sequence_start");
        let end = usize::try_from(node.sequence_end).expect("negative sequence_end");

        &full[start..end]
    }
}

/// Dense k-mer index over the graph, including k-mers that span node boundaries.
///
/// `normalized` controls whether k-mer bytes are uppercased at build time. An
/// index built with `normalized = true` must be searched with normalized queries;
/// an index built with `normalized = false` uses exact byte lookup.
#[derive(Serialize, Deserialize)]
pub struct SeedIndex {
    pub k: usize,
    pub normalized: bool,
    pub table: HashMap<Vec<u8>, Vec<GraphPos>>,
}

/// Bumped whenever the index format or indexing behavior changes incompatibly.
const SEED_INDEX_VERSION: u32 = 2;

/// File header written before the `SeedIndex` payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct SeedIndexHeader {
    version: u32,
    k: usize,
    normalized: bool,
}

/// Errors from indexed search.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SeedIndexSearchError {
    /// The seed index does not support degenerate (IUPAC) query bases.
    #[error("seed index search does not support degenerate (IUPAC) query bases")]
    UnsupportedQuery,
}

/// Errors from `SeedIndex` serialization/deserialization.
#[derive(Debug, thiserror::Error)]
pub enum SeedIndexIoError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("encode error: {0}")]
    Encode(postcard::Error),
    #[error("decode error: {0}")]
    Decode(postcard::Error),
    /// File is shorter than expected.
    #[error("file truncated or malformed")]
    Truncated,
    /// Header `version` does not match `SEED_INDEX_VERSION`.
    #[error("version mismatch: got {got}, expected {expected}")]
    VersionMismatch { got: u32, expected: u32 },
    /// Header `k` does not match the requested k-mer length.
    #[error("k mismatch: got {got}, expected {expected}")]
    KMismatch { got: usize, expected: usize },
}

impl SeedIndex {
    /// Build a k-mer index over the graph.
    ///
    /// When `normalized` is `true`, k-mer bytes are uppercased before insertion
    /// so that the index supports case-insensitive lookup. When `false`, raw
    /// graph bytes are stored and lookup is exact.
    pub fn build(matcher: &GenGraphMatcher, k: usize, normalized: bool) -> Self {
        let mut table: HashMap<Vec<u8>, Vec<GraphPos>> = HashMap::new();

        for start in matcher.all_start_positions() {
            for kmer in Self::collect_kmers_from(matcher, start, k, normalized) {
                table.entry(kmer).or_default().push(start);
            }
        }

        Self {
            k,
            normalized,
            table,
        }
    }

    /// Serialize `self` to bytes: 4-byte little-endian header length, header, payload.
    pub fn to_bytes_with_header(&self) -> Result<Vec<u8>, SeedIndexIoError> {
        let header = SeedIndexHeader {
            version: SEED_INDEX_VERSION,
            k: self.k,
            normalized: self.normalized,
        };

        let header_bytes = postcard::to_allocvec(&header).map_err(SeedIndexIoError::Encode)?;
        let payload_bytes = postcard::to_allocvec(self).map_err(SeedIndexIoError::Encode)?;

        let header_len = header_bytes.len() as u32;
        let mut out = Vec::with_capacity(4 + header_bytes.len() + payload_bytes.len());
        out.extend_from_slice(&header_len.to_le_bytes());
        out.extend_from_slice(&header_bytes);
        out.extend_from_slice(&payload_bytes);
        Ok(out)
    }

    /// Deserialize from bytes produced by `to_bytes_with_header`.
    ///
    /// Returns an error if `version` or `k` do not match expectations.
    /// `normalized` is read from the header and stored on the returned index.
    pub fn from_bytes_with_header(
        bytes: &[u8],
        expected_k: usize,
    ) -> Result<Self, SeedIndexIoError> {
        let (header, payload) = decode_seed_index_header_and_payload(bytes)?;

        validate_seed_index_header_version(&header)?;

        if header.k != expected_k {
            return Err(SeedIndexIoError::KMismatch {
                got: header.k,
                expected: expected_k,
            });
        }

        postcard::from_bytes(payload).map_err(SeedIndexIoError::Decode)
    }

    /// Write index to `path`.
    pub fn save_to_path<P: AsRef<Path>>(&self, path: P) -> Result<(), SeedIndexIoError> {
        let bytes = self.to_bytes_with_header()?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Load index from `path`. `normalized` and `k` are read from the stored header.
    pub fn load_from_path<P: AsRef<Path>>(path: P) -> Result<Self, SeedIndexIoError> {
        let bytes = std::fs::read(path)?;
        let (header, payload) = decode_seed_index_header_and_payload(&bytes)?;

        validate_seed_index_header_version(&header)?;

        postcard::from_bytes(payload).map_err(SeedIndexIoError::Decode)
    }

    /// Collect every string of exactly `k` bytes reachable from `start`.
    ///
    /// Bytes are uppercased when `normalized` is `true`.
    fn collect_kmers_from(
        matcher: &GenGraphMatcher,
        start: GraphPos,
        k: usize,
        normalized: bool,
    ) -> Vec<Vec<u8>> {
        #[derive(Clone)]
        struct Frame {
            block: GraphNode,
            offset: usize,
            buf: Vec<u8>,
        }

        let mut out = Vec::new();
        let mut stack = vec![Frame {
            block: start.block,
            offset: start.offset,
            buf: Vec::with_capacity(k),
        }];

        while let Some(frame) = stack.pop() {
            if frame.buf.len() == k {
                out.push(frame.buf);
                continue;
            }

            let text = matcher.graph_node_text(frame.block);

            if frame.offset < text.len() {
                let mut next = frame.clone();
                let byte = text[frame.offset];
                next.buf.push(if normalized {
                    byte.to_ascii_uppercase()
                } else {
                    byte
                });
                next.offset += 1;
                stack.push(next);
                continue;
            }

            for next_node in matcher
                .graph
                .neighbors_directed(frame.block, Direction::Outgoing)
            {
                let mut next = frame.clone();
                next.block = next_node;
                next.offset = 0;
                stack.push(next);
            }
        }

        out
    }
}

fn decode_seed_index_header_and_payload(
    bytes: &[u8],
) -> Result<(SeedIndexHeader, &[u8]), SeedIndexIoError> {
    if bytes.len() < 4 {
        return Err(SeedIndexIoError::Truncated);
    }

    let header_len = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    let header_end = 4 + header_len;

    if bytes.len() < header_end {
        return Err(SeedIndexIoError::Truncated);
    }

    let header: SeedIndexHeader =
        postcard::from_bytes(&bytes[4..header_end]).map_err(SeedIndexIoError::Decode)?;

    Ok((header, &bytes[header_end..]))
}

fn validate_seed_index_header_version(header: &SeedIndexHeader) -> Result<(), SeedIndexIoError> {
    if header.version != SEED_INDEX_VERSION {
        return Err(SeedIndexIoError::VersionMismatch {
            got: header.version,
            expected: SEED_INDEX_VERSION,
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use gen_models::{block_group::BlockGroup, collection::Collection};

    use super::*;
    use crate::test_helpers::{setup_block_group, setup_gen};

    // The test graph is a single linear path built by setup_block_group:
    //   AAAAAAAAAA → TTTTTTTTTT → CCCCCCCCCC → GGGGGGGGGG  (40 bp total)

    fn test_matcher() -> GenGraphMatcher {
        let ctx = setup_gen();
        let conn = ctx.graph().conn();
        let _ = Collection::create(conn, "test");
        let (block_group_id, _path) = setup_block_group(conn);
        let graph = BlockGroup::get_graph(conn, &block_group_id);
        GenGraphMatcher::new(conn, graph)
    }

    fn build_ssdna_matcher() -> GenGraphMatcher {
        let ctx = setup_gen();
        let conn = ctx.graph().conn();
        let _ = Collection::create(conn, "test");
        let (block_group_id, _path) = setup_block_group(conn);
        let graph = BlockGroup::get_graph(conn, &block_group_id);
        GenGraphMatcher::new_ssdna(conn, graph)
    }

    fn test_protein_matcher() -> GenGraphMatcher {
        let ctx = setup_gen();
        let conn = ctx.graph().conn();
        let _ = Collection::create(conn, "test");
        let (block_group_id, _path) = setup_block_group(conn);
        let graph = BlockGroup::get_graph(conn, &block_group_id);
        GenGraphMatcher::new_protein(conn, graph)
    }

    fn test_exact_matcher() -> GenGraphMatcher {
        let ctx = setup_gen();
        let conn = ctx.graph().conn();
        let _ = Collection::create(conn, "test");
        let (block_group_id, _path) = setup_block_group(conn);
        let graph = BlockGroup::get_graph(conn, &block_group_id);
        GenGraphMatcher::new_with_sequence_kind(conn, graph, SequenceKind::Exact)
    }

    #[test]
    fn reverse_complement_basic() {
        assert_eq!(reverse_complement(b"ACGT"), b"ACGT");
        assert_eq!(reverse_complement(b"AAAA"), b"TTTT");
        assert_eq!(reverse_complement(b"TTTT"), b"AAAA");
        assert_eq!(reverse_complement(b"AACGT"), b"ACGTT");
        assert_eq!(reverse_complement(b""), b"");
    }

    #[test]
    fn reverse_complement_lowercase() {
        assert_eq!(reverse_complement(b"acgt"), b"ACGT");
        assert_eq!(reverse_complement(b"aatt"), b"AATT");
    }

    #[test]
    fn reverse_complement_degenerate_codes() {
        assert_eq!(reverse_complement(b"RYSWKMBVDH"), b"DHBVKMWSRY");
        assert_eq!(reverse_complement(b"ry"), b"RY");
        assert_eq!(reverse_complement(b"ACGR"), b"YCGT");
    }

    #[test]
    fn reverse_complement_uracil() {
        assert_eq!(reverse_complement(b"U"), b"A");
        assert_eq!(reverse_complement(b"AU"), b"AT");
    }

    #[test]
    fn degenerate_matches_standard_bases() {
        assert!(degenerate_matches(b'A', b'A'));
        assert!(degenerate_matches(b'C', b'C'));
        assert!(degenerate_matches(b'G', b'G'));
        assert!(degenerate_matches(b'T', b'T'));

        assert!(!degenerate_matches(b'A', b'C'));
        assert!(!degenerate_matches(b'C', b'G'));
        assert!(!degenerate_matches(b'G', b'T'));
        assert!(!degenerate_matches(b'T', b'A'));
    }

    #[test]
    fn degenerate_matches_degenerate_codes() {
        assert!(degenerate_matches(b'N', b'A'));
        assert!(degenerate_matches(b'N', b'C'));
        assert!(degenerate_matches(b'N', b'G'));
        assert!(degenerate_matches(b'N', b'T'));

        assert!(degenerate_matches(b'R', b'A'));
        assert!(degenerate_matches(b'R', b'G'));
        assert!(!degenerate_matches(b'R', b'C'));
        assert!(!degenerate_matches(b'R', b'T'));

        assert!(degenerate_matches(b'Y', b'C'));
        assert!(degenerate_matches(b'Y', b'T'));
        assert!(!degenerate_matches(b'Y', b'A'));
        assert!(!degenerate_matches(b'Y', b'G'));

        assert!(degenerate_matches(b'B', b'C'));
        assert!(degenerate_matches(b'B', b'G'));
        assert!(degenerate_matches(b'B', b'T'));
        assert!(!degenerate_matches(b'B', b'A'));
    }

    #[test]
    fn degenerate_matches_unknown_falls_back_to_exact() {
        assert!(degenerate_matches(b'X', b'X'));
        assert!(degenerate_matches(b'!', b'!'));
        assert!(!degenerate_matches(b'X', b'A'));
        assert!(!degenerate_matches(b'!', b'A'));
    }

    #[test]
    fn query_contains_degenerate_iupac_detects_only_degenerate_codes() {
        assert!(!query_contains_degenerate_iupac(b"ACGT"));
        assert!(!query_contains_degenerate_iupac(b"acgt"));
        assert!(query_contains_degenerate_iupac(b"ACGN"));
        assert!(query_contains_degenerate_iupac(b"ry"));
    }

    #[test]
    fn contains_exact_within_single_node() {
        let matcher = test_matcher();

        assert!(matcher.contains(b"AAAA"));
        assert!(matcher.contains(b"TTTT"));
        assert!(matcher.contains(b"CCCC"));
        assert!(matcher.contains(b"GGGG"));
    }

    #[test]
    fn contains_spanning_node_boundary() {
        let matcher = test_matcher();

        assert!(matcher.contains(b"AAAATTTT"));
        assert!(matcher.contains(b"TTTTCCCC"));
    }

    #[test]
    fn contains_absent_exact_sequence() {
        let matcher = test_matcher();

        assert!(!matcher.contains(b"ACGT"));
    }

    #[test]
    fn contains_empty_query_always_true() {
        let matcher = test_matcher();

        assert!(matcher.contains(b""));
    }

    #[test]
    fn protein_matcher_does_not_search_reverse_complement() {
        let matcher = test_protein_matcher();

        // AAAAACCCCC is absent. Its reverse complement, GGGGGTTTTT, is present
        // across the G-node and T-node, but protein mode does not search reverse
        // complements.
        assert!(!matcher.contains(b"AAAAACCCCC"));
    }

    #[test]
    fn contains_iupac_n_matches_any_base() {
        let matcher = test_matcher();

        assert!(matcher.contains(b"NNNN"));
    }

    #[test]
    fn protein_matcher_does_not_use_iupac_matching() {
        let matcher = test_protein_matcher();

        assert!(!matcher.contains(b"NNNN"));
        assert!(!matcher.contains(b"RRRR"));
        assert!(!matcher.contains(b"YYYY"));
    }

    #[test]
    fn find_all_single_node_match() {
        let matcher = test_matcher();

        let hits = matcher.find_all(b"AAAAAAAAAA");
        assert_eq!(hits.len(), 2);

        assert!(hits.iter().any(|hit| {
            hit.slices.len() == 1 && hit.slices[0].start == 0 && hit.slices[0].end == 10
        }));
    }

    #[test]
    fn find_all_spanning_match() {
        let matcher = test_matcher();

        let hits = matcher.find_all(b"AAAAATTTTT");
        assert_eq!(hits.len(), 2);

        assert!(hits.iter().any(|hit| {
            hit.slices.len() == 2 && hit.slices[0].start == 5 && hit.slices[1].end == 5
        }));
    }

    #[test]
    fn find_all_no_match_returns_empty() {
        let matcher = test_matcher();

        assert!(matcher.find_all(b"ACGT").is_empty());
    }

    #[test]
    fn find_all_empty_query_returns_empty() {
        let matcher = test_matcher();

        assert!(matcher.find_all(b"").is_empty());
    }

    #[test]
    fn find_all_iupac_n_matches_each_node() {
        let matcher = test_matcher();

        let hits = matcher.find_all(b"NNNNNNNNNN");
        assert_eq!(hits.len(), 62);
    }

    #[test]
    fn seed_index_build_and_find() {
        let matcher = test_matcher();

        let index = SeedIndex::build(&matcher, 4, true);
        assert_eq!(index.k, 4);
        assert!(index.table.contains_key(b"AAAA".as_ref()));
        assert!(index.table.contains_key(b"TTTT".as_ref()));
        assert!(index.table.contains_key(b"CCCC".as_ref()));
        assert!(index.table.contains_key(b"GGGG".as_ref()));
        assert!(index.table.contains_key(b"ATTT".as_ref()));
    }

    #[test]
    fn seed_index_find_all_with_index_matches_forward_search() {
        let matcher = build_ssdna_matcher();

        let index = SeedIndex::build(&matcher, 4, true);
        let query = b"AAAAATTTTT";
        let via_index = matcher.find_all_with_seed_index(&index, query).unwrap();
        assert_eq!(via_index.len(), 1);
    }

    #[test]
    fn seed_index_find_all_absent_query_empty() {
        let matcher = test_matcher();

        let index = SeedIndex::build(&matcher, 4, true);
        let hits = matcher.find_all_with_seed_index(&index, b"ACGT").unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn seed_index_short_query_falls_back_to_forward_find_all() {
        let matcher = build_ssdna_matcher();

        let index = SeedIndex::build(&matcher, 4, true);
        let via_index = matcher.find_all_with_seed_index(&index, b"AA").unwrap();
        assert_eq!(via_index.len(), 9);
    }

    #[test]
    fn seed_index_rejects_degenerate_query() {
        let matcher = test_matcher();

        let index = SeedIndex::build(&matcher, 4, true);
        let err = matcher
            .find_all_with_seed_index(&index, b"ATGN")
            .unwrap_err();
        assert_eq!(err, SeedIndexSearchError::UnsupportedQuery);
    }

    #[test]
    fn seed_index_finds_reverse_complement_via_index() {
        let matcher = test_matcher();
        let index = SeedIndex::build(&matcher, 4, true);

        // The graph contains AAAAAAAAAA (A node) and TTTTTTTTTT (T node).
        // Searching AAAAAAAAAA with DNA mode must find both the forward hit (A
        // node) and the reverse-complement hit (T node, tagged Strand::Reverse).
        let hits = matcher
            .find_all_with_seed_index(&index, b"AAAAAAAAAA")
            .unwrap();
        assert_eq!(hits.len(), 2);

        let has_forward = hits
            .iter()
            .flat_map(|l| l.slices.iter())
            .any(|s| s.strand == Strand::Forward);
        let has_reverse = hits
            .iter()
            .flat_map(|l| l.slices.iter())
            .any(|s| s.strand == Strand::Reverse);
        assert!(has_forward, "expected a forward hit");
        assert!(has_reverse, "expected a reverse-complement hit");
    }

    #[test]
    fn seed_index_short_query_finds_reverse_complement() {
        let matcher = test_matcher();
        let index = SeedIndex::build(&matcher, 4, true);

        // Query shorter than k=4 falls back to a full scan, but DNA mode must
        // still search the RC. "GG" appears 9 times in the G node (forward) and
        // its RC "CC" appears 9 times in the C node (reverse). Total: 18.
        let hits = matcher.find_all_with_seed_index(&index, b"GG").unwrap();
        assert_eq!(hits.len(), 18);

        let has_forward = hits
            .iter()
            .flat_map(|l| l.slices.iter())
            .any(|s| s.strand == Strand::Forward);
        let has_reverse = hits
            .iter()
            .flat_map(|l| l.slices.iter())
            .any(|s| s.strand == Strand::Reverse);
        assert!(has_forward, "expected forward hits");
        assert!(has_reverse, "expected reverse-complement hits");
    }

    #[test]
    fn seed_index_roundtrip_bytes() {
        let matcher = test_matcher();

        let index = SeedIndex::build(&matcher, 4, true);
        let bytes = index.to_bytes_with_header().unwrap();
        let loaded = SeedIndex::from_bytes_with_header(&bytes, 4).unwrap();
        assert_eq!(loaded.k, index.k);
        assert_eq!(loaded.normalized, index.normalized);
        assert_eq!(loaded.table.len(), index.table.len());
    }

    #[test]
    fn seed_index_error_cases() {
        let matcher = test_matcher();

        let index = SeedIndex::build(&matcher, 4, true);
        let bytes = index.to_bytes_with_header().unwrap();

        assert!(matches!(
            SeedIndex::from_bytes_with_header(&bytes[..2], 4),
            Err(SeedIndexIoError::Truncated)
        ));
        assert!(matches!(
            SeedIndex::from_bytes_with_header(&bytes, 8),
            Err(SeedIndexIoError::KMismatch { .. })
        ));
    }

    #[test]
    fn seed_index_save_and_load_path() {
        let matcher = test_matcher();

        let index = SeedIndex::build(&matcher, 4, true);
        let tmp = tempfile::NamedTempFile::new().unwrap();
        index.save_to_path(tmp.path()).unwrap();
        let loaded = SeedIndex::load_from_path(tmp.path()).unwrap();
        assert_eq!(loaded.k, index.k);
        assert_eq!(loaded.normalized, index.normalized);
        assert_eq!(loaded.table.len(), index.table.len());
    }

    // --- Case sensitivity tests ---
    // The test graph contains uppercase-only sequences: AAAAAAAAAA, TTTTTTTTTT,
    // CCCCCCCCCC, GGGGGGGGGG.

    #[test]
    fn exact_matcher_matches_uppercase_query() {
        let matcher = test_exact_matcher();
        assert!(matcher.contains(b"AAAA"));
        assert!(matcher.contains(b"TTTT"));
    }

    #[test]
    fn exact_matcher_rejects_lowercase_query() {
        let matcher = test_exact_matcher();
        assert!(!matcher.contains(b"aaaa"));
        assert!(!matcher.contains(b"tttt"));
        assert!(!matcher.contains(b"cccc"));
        assert!(!matcher.contains(b"gggg"));
    }

    #[test]
    fn exact_matcher_does_not_reverse_complement() {
        let matcher = test_exact_matcher();
        // The graph goes A→T→C→G. GGGGGAAAAA is absent in the forward direction;
        // its RC (TTTTTCCCCC) spans the T→C boundary and IS present. The DNA
        // matcher finds it via RC; Exact must not.
        assert!(test_matcher().contains(b"GGGGGAAAAA")); // DNA finds via RC
        assert!(!matcher.contains(b"GGGGGAAAAA")); // Exact does not
    }

    #[test]
    fn dna_matcher_is_case_insensitive() {
        let matcher = test_matcher();
        assert!(matcher.contains(b"aaaa"));
        assert!(matcher.contains(b"tttt"));
        assert!(matcher.contains(b"cccc"));
        assert!(matcher.contains(b"gggg"));
        assert!(matcher.contains(b"AaAa"));
    }

    #[test]
    fn protein_matcher_is_case_insensitive() {
        let matcher = test_protein_matcher();
        assert!(matcher.contains(b"aaaa"));
        assert!(matcher.contains(b"AAAA"));
        assert!(matcher.contains(b"cccc"));
        assert!(matcher.contains(b"CCCC"));
    }

    #[test]
    fn seed_index_normalized_matches_lowercase_query() {
        let matcher = test_matcher();
        let index = SeedIndex::build(&matcher, 4, true);
        // lowercase query should be normalized and hit the uppercase index keys
        let hits = matcher
            .find_all_with_seed_index(&index, b"aaaaaaaaaa")
            .unwrap();
        assert!(!hits.is_empty());
    }

    #[test]
    fn seed_index_not_normalized_misses_lowercase_query() {
        let matcher = test_exact_matcher();
        let index = SeedIndex::build(&matcher, 4, false);
        // index keys are raw uppercase; lowercase seed won't match
        let hits = matcher
            .find_all_with_seed_index(&index, b"aaaaaaaaaa")
            .unwrap();
        assert!(hits.is_empty());
        // uppercase query does match
        let hits = matcher
            .find_all_with_seed_index(&index, b"AAAAAAAAAA")
            .unwrap();
        assert!(!hits.is_empty());
    }
}
