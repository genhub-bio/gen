use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use gen_core::{HashId, PATH_END_NODE_ID, PATH_START_NODE_ID};
use gen_graph::{GenGraph, GraphNode};
use gen_models::{db::GraphConnection, node::Node};
use petgraph::Direction;
use serde::{Deserialize, Serialize};

/// A position in the graph: a GraphNode (aka Block) plus a local byte offset
/// relative to the start of that GraphNode's sequence slice.
///
/// If the GraphNode has a sequence_start of 100, then offset 5 means sequence
/// position 105 in the underlying stored sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GraphPos {
    pub node: GraphNode,
    /// Local offset within contained sequence slice: `0..=node.length()`.
    pub offset: usize,
}

/// A linear walk through graph space that covers a complete match.
///
/// `blocks[0]` is the first node containing matched bytes;
/// `blocks.last()` is the last. `start_offset` and `end_offset` anchor the
/// match within those boundary nodes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphLocus {
    /// Local offset of the first matched byte inside `blocks[0]`.
    pub start_offset: usize,
    /// Local offset past the last consumed byte in `blocks.last()`.
    pub end_offset: usize,
    /// Ordered sequence of nodes that spell the match. Length is at least 1.
    pub blocks: Vec<GraphNode>,
}

/// Biological interpretation of the sequences being searched.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SequenceKind {
    /// Double-stranded DNA search.
    ///
    /// Searches the query as provided and as its reverse complement. Uses exact
    /// byte matching unless the query contains an IUPAC degenerate nucleotide
    /// code, in which case the query is matched with IUPAC semantics.
    Dna,
    /// Single-stranded DNA search.
    ///
    /// Searches only the query as provided. Uses exact byte matching unless the
    /// query contains an IUPAC degenerate nucleotide code, in which case the
    /// query is matched with IUPAC semantics.
    SsDna,
    /// Protein or other non-nucleotide sequence search.
    ///
    /// Searches only the query as provided with exact byte matching.
    Protein,
}

impl Default for SequenceKind {
    fn default() -> Self {
        Self::Dna
    }
}

impl SequenceKind {
    pub fn uses_iupac_for_query(self, query: &[u8]) -> bool {
        matches!(self, Self::Dna | Self::SsDna) && query_contains_degenerate_iupac(query)
    }

    /// Compatibility helper for call sites that still use string sequence types.
    pub fn from_seq_type(seq_type: Option<&str>) -> Self {
        match seq_type {
            Some("protein") | Some("Protein") => Self::Protein,
            Some("ssDNA") | Some("ssdna") | Some("DNA") | Some("dna") => Self::SsDna,
            Some("dsDNA") | Some("dsdna") => Self::Dna,
            _ => Self::Dna,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ByteMatchMode {
    Exact,
    Iupac,
}

impl ByteMatchMode {
    #[inline]
    fn for_query(sequence_kind: SequenceKind, query: &[u8]) -> Self {
        if sequence_kind.uses_iupac_for_query(query) {
            Self::Iupac
        } else {
            Self::Exact
        }
    }

    #[inline]
    fn matches(self, query_byte: u8, graph_byte: u8) -> bool {
        match self {
            Self::Exact => query_byte == graph_byte,
            Self::Iupac => iupac_matches(query_byte, graph_byte),
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
fn normalize_nucleotide_byte(byte: u8) -> u8 {
    byte.to_ascii_uppercase()
}

#[inline]
fn is_degenerate_iupac(byte: u8) -> bool {
    matches!(
        normalize_nucleotide_byte(byte),
        b'R' | b'Y' | b'S' | b'W' | b'K' | b'M' | b'B' | b'D' | b'H' | b'V' | b'N'
    )
}

#[inline]
fn query_contains_degenerate_iupac(query: &[u8]) -> bool {
    query.iter().any(|&byte| is_degenerate_iupac(byte))
}

/// Case-insensitive IUPAC query matching against a concrete graph byte.
///
/// Degenerate codes are interpreted in the query. Graph bytes are usually
/// expected to be concrete bases, but if the graph contains non-standard bytes,
/// exact fallback is used for unknown query codes.
#[inline]
fn iupac_matches(query_byte: u8, graph_byte: u8) -> bool {
    let query_byte = normalize_nucleotide_byte(query_byte);
    let graph_byte = normalize_nucleotide_byte(graph_byte);

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

/// Compute the reverse complement of a DNA sequence, including IUPAC codes.
fn reverse_complement(seq: &[u8]) -> Vec<u8> {
    seq.iter()
        .rev()
        .map(|&base| match base {
            b'A' | b'a' => b'T',
            b'T' | b't' => b'A',
            b'C' | b'c' => b'G',
            b'G' | b'g' => b'C',
            b'U' | b'u' => b'A',
            b'N' | b'n' => b'N',
            b'R' | b'r' => b'Y',
            b'Y' | b'y' => b'R',
            b'S' | b's' => b'S',
            b'W' | b'w' => b'W',
            b'K' | b'k' => b'M',
            b'M' | b'm' => b'K',
            b'B' | b'b' => b'V',
            b'V' | b'v' => b'B',
            b'D' | b'd' => b'H',
            b'H' | b'h' => b'D',
            _ => base,
        })
        .collect()
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
                .map(|(node_id, seq)| (node_id, seq.get_sequence(None, None).into_bytes()))
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

        let byte_match_mode = ByteMatchMode::for_query(self.sequence_kind, query);
        if self.contains_query_orientation(query, byte_match_mode) {
            return true;
        }

        if self.sequence_kind == SequenceKind::Dna {
            let rc = reverse_complement(query);
            let rc_byte_match_mode = ByteMatchMode::for_query(self.sequence_kind, &rc);
            if self.contains_query_orientation(&rc, rc_byte_match_mode) {
                return true;
            }
        }

        false
    }

    /// Returns every match of `query` in the graph using this matcher's
    /// configured sequence kind.
    ///
    /// For `SequenceKind::Dna`, reverse-complement matches are returned in graph
    /// coordinates exactly like forward matches. The returned `GraphLocus`
    /// currently does not record which query orientation matched.
    pub fn find_all(&self, query: &[u8]) -> Vec<GraphLocus> {
        if query.is_empty() {
            return Vec::new();
        }

        let byte_match_mode = ByteMatchMode::for_query(self.sequence_kind, query);
        let mut out = self.find_all_query_orientation(query, byte_match_mode);

        if self.sequence_kind == SequenceKind::Dna {
            let rc = reverse_complement(query);
            let rc_byte_match_mode = ByteMatchMode::for_query(self.sequence_kind, &rc);
            out.extend(self.find_all_query_orientation(&rc, rc_byte_match_mode));
        }

        out
    }

    /// Returns every exact, forward-only match, using an exact seed index to
    /// prune start positions.
    ///
    /// This method ignores the matcher's `SequenceKind`. It does not perform
    /// reverse-complement search for `SequenceKind::Dna`, and it does not expand
    /// IUPAC-degenerate query seeds.
    ///
    /// Returns an error if the query contains IUPAC-degenerate bases, because an
    /// exact seed lookup would otherwise silently miss valid matches.
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

        if query.len() < seed_index.k {
            return Ok(self.find_all_exact_forward(query));
        }

        let seed = &query[..seed_index.k];
        let Some(positions) = seed_index.table.get(seed) else {
            return Ok(Vec::new());
        };

        let mut out = Vec::new();
        for &pos in positions {
            self.collect_matches_from(pos, query, ByteMatchMode::Exact, &mut out);
        }
        Ok(out)
    }

    fn find_all_exact_forward(&self, query: &[u8]) -> Vec<GraphLocus> {
        if query.is_empty() {
            return Vec::new();
        }

        self.find_all_query_orientation(query, ByteMatchMode::Exact)
    }

    fn contains_query_orientation(&self, query: &[u8], byte_match_mode: ByteMatchMode) -> bool {
        self.all_start_positions()
            .into_iter()
            .any(|start| self.contains_from(start, query, byte_match_mode))
    }

    fn find_all_query_orientation(
        &self,
        query: &[u8],
        byte_match_mode: ByteMatchMode,
    ) -> Vec<GraphLocus> {
        let mut out = Vec::new();

        for start in self.all_start_positions() {
            self.collect_matches_from(start, query, byte_match_mode, &mut out);
        }

        out
    }

    /// DFS from a single start position; returns `true` on the first complete
    /// match.
    fn contains_from(&self, start: GraphPos, query: &[u8], byte_match_mode: ByteMatchMode) -> bool {
        let initial = State {
            q_idx: 0,
            block: start.node,
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

            match self.step(state, query[state.q_idx], byte_match_mode) {
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
        byte_match_mode: ByteMatchMode,
        out: &mut Vec<GraphLocus>,
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
                block: start.node,
                offset: start.offset,
            },
            start_offset: start.offset,
            path: vec![start.node],
        };

        let mut stack = vec![initial];

        while let Some(ts) = stack.pop() {
            if ts.state.q_idx == query.len() {
                out.push(GraphLocus {
                    start_offset: ts.start_offset,
                    end_offset: ts.state.offset,
                    blocks: ts.path,
                });
                continue;
            }

            match self.step(ts.state, query[ts.state.q_idx], byte_match_mode) {
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

    fn step(&self, state: State, query_byte: u8, byte_match_mode: ByteMatchMode) -> StepResult {
        let text = self.graph_node_text(state.block);

        if state.offset < text.len() {
            if byte_match_mode.matches(query_byte, text[state.offset]) {
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
                starts.push(GraphPos { node, offset });
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

/// Dense exact k-mer index over the graph, including k-mers that span node
/// boundaries.
#[derive(Serialize, Deserialize)]
pub struct SeedIndex {
    pub k: usize,
    pub table: HashMap<Vec<u8>, Vec<GraphPos>>,
}

/// Bumped whenever the index format or indexing behavior changes incompatibly.
const SEED_INDEX_VERSION: u32 = 1;

/// File header written before the `SeedIndex` payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct SeedIndexHeader {
    version: u32,
    k: usize,
    case_sensitive: bool,
}

/// Errors from indexed search.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SeedIndexSearchError {
    /// The exact seed index can only be used with exact, non-degenerate queries.
    UnsupportedQuery,
}

impl std::fmt::Display for SeedIndexSearchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedQuery => write!(
                f,
                "exact seed index can only be used with exact, non-degenerate queries"
            ),
        }
    }
}

impl std::error::Error for SeedIndexSearchError {}

/// Errors from `SeedIndex` serialization/deserialization.
#[derive(Debug)]
pub enum SeedIndexIoError {
    Io(std::io::Error),
    Encode(postcard::Error),
    Decode(postcard::Error),
    /// File is shorter than expected.
    Truncated,
    /// Header `version` does not match `SEED_INDEX_VERSION`.
    VersionMismatch {
        got: u32,
        expected: u32,
    },
    /// Header `k` does not match the requested k-mer length.
    KMismatch {
        got: usize,
        expected: usize,
    },
    /// Header `case_sensitive` does not match the requested flag.
    CaseSensitiveMismatch {
        got: bool,
        expected: bool,
    },
}

impl std::fmt::Display for SeedIndexIoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Encode(e) => write!(f, "encode error: {e}"),
            Self::Decode(e) => write!(f, "decode error: {e}"),
            Self::Truncated => write!(f, "file truncated or malformed"),
            Self::VersionMismatch { got, expected } => {
                write!(f, "version mismatch: got {got}, expected {expected}")
            }
            Self::KMismatch { got, expected } => {
                write!(f, "k mismatch: got {got}, expected {expected}")
            }
            Self::CaseSensitiveMismatch { got, expected } => {
                write!(f, "case_sensitive mismatch: got {got}, expected {expected}")
            }
        }
    }
}

impl std::error::Error for SeedIndexIoError {}

impl From<std::io::Error> for SeedIndexIoError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl SeedIndex {
    /// Enumerate all exact strings of length `k` reachable from every start
    /// position, including those that span node boundaries.
    pub fn build(matcher: &GenGraphMatcher, k: usize) -> Self {
        let mut table: HashMap<Vec<u8>, Vec<GraphPos>> = HashMap::new();

        for start in matcher.all_start_positions() {
            for kmer in Self::collect_kmers_from(matcher, start, k) {
                table.entry(kmer).or_default().push(start);
            }
        }

        Self { k, table }
    }

    /// Serialize `self` to bytes: 4-byte little-endian header length, header,
    /// payload.
    ///
    /// `case_sensitive` is stored in the header and validated on load.
    pub fn to_bytes_with_header(&self, case_sensitive: bool) -> Result<Vec<u8>, SeedIndexIoError> {
        let header = SeedIndexHeader {
            version: SEED_INDEX_VERSION,
            k: self.k,
            case_sensitive,
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
    /// Returns an error if the header fields do not match `expected_k` or
    /// `expected_case_sensitive`, or if the version has changed.
    pub fn from_bytes_with_header(
        bytes: &[u8],
        expected_k: usize,
        expected_case_sensitive: bool,
    ) -> Result<Self, SeedIndexIoError> {
        let (header, payload) = decode_seed_index_header_and_payload(bytes)?;

        validate_seed_index_header_version(&header)?;

        if header.k != expected_k {
            return Err(SeedIndexIoError::KMismatch {
                got: header.k,
                expected: expected_k,
            });
        }

        if header.case_sensitive != expected_case_sensitive {
            return Err(SeedIndexIoError::CaseSensitiveMismatch {
                got: header.case_sensitive,
                expected: expected_case_sensitive,
            });
        }

        postcard::from_bytes(payload).map_err(SeedIndexIoError::Decode)
    }

    /// Write index to `path`. Fails if the file cannot be created or written.
    pub fn save_to_path<P: AsRef<Path>>(
        &self,
        path: P,
        case_sensitive: bool,
    ) -> Result<(), SeedIndexIoError> {
        let bytes = self.to_bytes_with_header(case_sensitive)?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Load index from `path`, validating `version` and `case_sensitive`.
    ///
    /// `k` is taken from the stored header; no k-check is performed.
    pub fn load_from_path<P: AsRef<Path>>(
        path: P,
        expected_case_sensitive: bool,
    ) -> Result<Self, SeedIndexIoError> {
        let bytes = std::fs::read(path)?;
        let (header, payload) = decode_seed_index_header_and_payload(&bytes)?;

        validate_seed_index_header_version(&header)?;

        if header.case_sensitive != expected_case_sensitive {
            return Err(SeedIndexIoError::CaseSensitiveMismatch {
                got: header.case_sensitive,
                expected: expected_case_sensitive,
            });
        }

        postcard::from_bytes(payload).map_err(SeedIndexIoError::Decode)
    }

    /// Collect every exact string of exactly `k` bytes reachable by walking
    /// forward through the graph from `start`.
    fn collect_kmers_from(matcher: &GenGraphMatcher, start: GraphPos, k: usize) -> Vec<Vec<u8>> {
        #[derive(Clone)]
        struct Frame {
            node: GraphNode,
            offset: usize,
            buf: Vec<u8>,
        }

        let mut out = Vec::new();
        let mut stack = vec![Frame {
            node: start.node,
            offset: start.offset,
            buf: Vec::with_capacity(k),
        }];

        while let Some(frame) = stack.pop() {
            if frame.buf.len() == k {
                out.push(frame.buf);
                continue;
            }

            let text = matcher.graph_node_text(frame.node);

            if frame.offset < text.len() {
                let mut next = frame.clone();
                next.buf.push(text[frame.offset]);
                next.offset += 1;
                stack.push(next);
                continue;
            }

            for next_node in matcher
                .graph
                .neighbors_directed(frame.node, Direction::Outgoing)
            {
                let mut next = frame.clone();
                next.node = next_node;
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
#[path = "graph_search_tests.rs"]
mod tests;
