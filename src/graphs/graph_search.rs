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
/// compared to the start of that GraphNode. If the GraphNode has a start of 100,
/// then offset 5 means position 105 compared to the start of the sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GraphPos {
    pub node: GraphNode,
    /// Byte offset, local to the node slice (`0..=node.length()`).
    pub offset: usize,
}

/// A linear sequence of nodes in graph space that covers an exact match.
///
/// `nodes[0]` is the first node containing matched bytes;
/// `nodes.last()` is the last.  `start_offset` and `end_offset` anchor the
/// match within those boundary nodes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphLocus {
    /// Local offset of the first matched byte inside `nodes[0]`.
    pub start_offset: usize,
    /// Local offset past the last consumed byte in `nodes.last()` (exclusive).
    pub end_offset: usize,
    /// Ordered sequence of nodes that spell the match (length ≥ 1).
    pub nodes: Vec<GraphNode>,
}

/// Complete search state: which query byte to match next, and where we are in
/// the graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct State {
    /// Index into the query — the next byte that must be matched.
    q_idx: usize,
    node: GraphNode,
    /// Local byte offset within contained sequence slice..
    offset: usize,
}

enum StepResult {
    /// Single successor state.
    Advance(State),
    /// Multiple successor states (junction with several outgoing edges).
    Branch(Vec<State>),
    /// No match; this path is dead.
    Dead,
}

pub struct GenGraphMatcher {
    graph: GenGraph,
    /// Pre-fetched GraphNode sequence bytes, keyed by `GraphNode::node_id`.
    node_sequences: HashMap<HashId, Vec<u8>>,
}

impl GenGraphMatcher {
    /// Build a matcher from a database connection and graph.
    ///
    /// Batch-loads all node sequences up front; no further database access
    /// occurs during matching.
    pub fn new(conn: &GraphConnection, graph: GenGraph) -> Self {
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
        // PATH_START / PATH_END get empty vecs associated to them
        node_sequences.entry(PATH_START_NODE_ID).or_default();
        node_sequences.entry(PATH_END_NODE_ID).or_default();

        Self {
            graph,
            node_sequences,
        }
    }

    /// Returns `true` if `query` occurs anywhere in the graph.
    pub fn contains(&self, query: &[u8]) -> bool {
        if query.is_empty() {
            return true;
        }
        for start in self.all_start_positions() {
            if self.contains_from(start, query) {
                return true;
            }
        }
        false
    }

    /// Returns every exact match of `query` in the graph.
    pub fn find_all(&self, query: &[u8]) -> Vec<GraphLocus> {
        if query.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::new();
        for start in self.all_start_positions() {
            self.collect_matches_from(start, query, &mut out);
        }
        out
    }

    /// Returns every exact match, using `seed_index` to prune the start positions.
    ///
    /// Falls back to `find_all` for queries shorter than `seed_index.k`.
    pub fn find_all_with_seed_index(
        &self,
        seed_index: &SeedIndex,
        query: &[u8],
    ) -> Vec<GraphLocus> {
        if query.is_empty() {
            return Vec::new();
        }
        if query.len() < seed_index.k {
            return self.find_all(query);
        }
        let seed = &query[..seed_index.k];
        let Some(positions) = seed_index.table.get(seed) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for &pos in positions {
            self.collect_matches_from(pos, query, &mut out);
        }
        out
    }

    /// DFS from a single start position; returns `true` on the first complete
    /// match.
    fn contains_from(&self, start: GraphPos, query: &[u8]) -> bool {
        let initial = State {
            q_idx: 0,
            node: start.node,
            offset: start.offset,
        };
        let mut stack = vec![initial];
        // Avoid redundant work by keeping track of deadend states.
        let mut dead: HashSet<State> = HashSet::new();

        while let Some(state) = stack.pop() {
            if state.q_idx == query.len() {
                return true;
            }
            if dead.contains(&state) {
                continue;
            }
            match self.step_exact(state, query[state.q_idx]) {
                StepResult::Advance(next) => stack.push(next),
                StepResult::Branch(nexts) => stack.extend(nexts),
                StepResult::Dead => {
                    dead.insert(state);
                }
            }
        }
        false
    }

    /// DFS from a single start position; appends every complete match to
    /// `out`.
    fn collect_matches_from(&self, start: GraphPos, query: &[u8], out: &mut Vec<GraphLocus>) {
        #[derive(Clone, Debug)]
        struct TraceState {
            state: State,
            start_offset: usize,
            path: Vec<GraphNode>,
        }

        let initial = TraceState {
            state: State {
                q_idx: 0,
                node: start.node,
                offset: start.offset,
            },
            start_offset: start.offset,
            path: vec![start.node],
        };

        let mut stack = vec![initial];

        // In classic intepretations you'd see a global visited-set keyed on
        // keyed on (q_idx, node, offset) here. But that would collapse two
        // distinct paths that converge on the same node at the same query
        // position — e.g. two allele branches with the same sequence that
        // both enter a shared suffix node.
        //
        // Not keeping this visited doesn't impact cycle safety either: we can't
        // get zero-length GraphNodes (except start/stop), so you'll always
        // increase the query index and traversal terminates naturally when
        // the query is exhausted.
        while let Some(ts) = stack.pop() {
            if ts.state.q_idx == query.len() {
                out.push(GraphLocus {
                    start_offset: ts.start_offset,
                    end_offset: ts.state.offset,
                    nodes: ts.path,
                });
                continue;
            }

            match self.step_exact(ts.state, query[ts.state.q_idx]) {
                StepResult::Advance(next) => {
                    let mut next_ts = ts;
                    if next.node != next_ts.state.node {
                        next_ts.path.push(next.node);
                    }
                    next_ts.state = next;
                    stack.push(next_ts);
                }
                StepResult::Branch(nexts) => {
                    for next in nexts {
                        let mut next_ts = ts.clone();
                        if next.node != next_ts.state.node {
                            next_ts.path.push(next.node);
                        }
                        next_ts.state = next;
                        stack.push(next_ts);
                    }
                }
                StepResult::Dead => {}
            }
        }
    }

    fn step_exact(&self, state: State, query_byte: u8) -> StepResult {
        let text = self.graph_node_text(state.node);

        if state.offset < text.len() {
            if text[state.offset] == query_byte {
                return StepResult::Advance(State {
                    q_idx: state.q_idx + 1,
                    node: state.node,
                    offset: state.offset + 1,
                });
            } else {
                return StepResult::Dead;
            }
        }

        // At node boundary: traverse outgoing neighbors.
        let nexts: Vec<State> = self
            .graph
            .neighbors_directed(state.node, Direction::Outgoing)
            .map(|next_node| State {
                q_idx: state.q_idx,
                node: next_node,
                offset: 0,
            })
            .collect();

        match nexts.len() {
            0 => StepResult::Dead,
            1 => StepResult::Advance(nexts[0]),
            _ => StepResult::Branch(nexts),
        }
    }

    /// Enumerate every valid starting position: every byte offset inside
    /// every non-empty node slice.
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
    /// Enumerate all strings of length `k` reachable from every start
    /// position, including those that cross node boundaries.
    pub fn build(matcher: &GenGraphMatcher, k: usize) -> Self {
        let mut table: HashMap<Vec<u8>, Vec<GraphPos>> = HashMap::new();

        for start in matcher.all_start_positions() {
            for kmer in collect_kmers_from(matcher, start, k) {
                table.entry(kmer).or_default().push(start);
            }
        }

        Self { k, table }
    }

    /// Serialize `self` to bytes: 4-byte LE header length, header, payload.
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

        if header.version != SEED_INDEX_VERSION {
            return Err(SeedIndexIoError::VersionMismatch {
                got: header.version,
                expected: SEED_INDEX_VERSION,
            });
        }
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

        let index: Self =
            postcard::from_bytes(&bytes[header_end..]).map_err(SeedIndexIoError::Decode)?;
        Ok(index)
    }

    /// Write index to `path`.  Fails if the file cannot be created or written.
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
        if header.version != SEED_INDEX_VERSION {
            return Err(SeedIndexIoError::VersionMismatch {
                got: header.version,
                expected: SEED_INDEX_VERSION,
            });
        }
        if header.case_sensitive != expected_case_sensitive {
            return Err(SeedIndexIoError::CaseSensitiveMismatch {
                got: header.case_sensitive,
                expected: expected_case_sensitive,
            });
        }
        let index: Self =
            postcard::from_bytes(&bytes[header_end..]).map_err(SeedIndexIoError::Decode)?;
        Ok(index)
    }
}

/// Collect every string of exactly `k` bytes reachable by walking
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

#[cfg(test)]
#[rustfmt::skip]
mod db_tests {
    use std::path::PathBuf;

    use gen_core::{PATH_END_NODE_ID, PATH_START_NODE_ID, Strand};
    use gen_models::{
        block_group::BlockGroup,
        block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
        collection::Collection,
        edge::Edge,
        node::Node,
        sample::Sample,
        sequence::Sequence,
    };

    use super::*;
    use crate::{
        imports::{fasta::import_fasta, gfa::import_gfa},
        test_helpers::{get_connection, get_sample_bg, setup_gen},
        track_database,
        updates::vcf::update_with_vcf,
    };

    /// Build a two-node linear graph:
    ///   PATH_START → seq_a("ACGT") → seq_b("TGCA") → PATH_END
    ///
    /// Returns `(conn, block_group_id)`.
    fn two_node_graph() -> (gen_models::db::GraphConnection, HashId) {
        let conn = get_connection(None).unwrap();
        let seq_a = Sequence::new().sequence_type("DNA").sequence("ACGT").save(&conn);
        let node_a = Node::create(&conn, &seq_a.hash, &HashId::convert_str("test-node-a"));
        let seq_b = Sequence::new().sequence_type("DNA").sequence("TGCA").save(&conn);
        let node_b = Node::create(&conn, &seq_b.hash, &HashId::convert_str("test-node-b"));
        Collection::create(&conn, "test");
        Sample::get_or_create(&conn, "test");
        let bg = BlockGroup::create(&conn, "test", "test", "chr1");
        let e0 = Edge::create(&conn, PATH_START_NODE_ID, 0, Strand::Forward, node_a, 0, Strand::Forward);
        let e1 = Edge::create(&conn, node_a, 4, Strand::Forward, node_b, 0, Strand::Forward);
        let e2 = Edge::create(&conn, node_b, 4, Strand::Forward, PATH_END_NODE_ID, 0, Strand::Forward);
        BlockGroupEdge::bulk_create(&conn, &[
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e0.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e1.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e2.id, chromosome_index: 0, phased: 0 },
        ]);
        (conn, bg.id)
    }

    /// PATH_START / PATH_END sentinel nodes are present in every real graph
    /// but absent from the nodes DB table; construction must handle them.
    #[test]
    fn sentinel_nodes_handled_at_construction() {
        let (conn, bg_id) = two_node_graph();
        let graph = BlockGroup::get_graph(&conn, &bg_id);

        let matcher = GenGraphMatcher::new(&conn, graph);

        assert!(matcher.contains(b"ACGT"));
        assert!(matcher.contains(b"TGCA"));
        assert!(!matcher.contains(b"NNNN"));
    }

    #[test]
    fn cross_junction_match_found_on_db_graph() {
        let (conn, bg_id) = two_node_graph();
        let graph = BlockGroup::get_graph(&conn, &bg_id);
        let matcher = GenGraphMatcher::new(&conn, graph);

        // Spans the A→B junction.
        assert!(matcher.contains(b"CGTT"));
        assert!(matcher.contains(b"GTTG")); // last 2 of A + first 2 of B
        assert!(matcher.contains(b"ACGTTGCA")); // full span of both nodes

        // Should NOT match across the junction.
        assert!(!matcher.contains(b"ACGTACGT"));

        let hits = matcher.find_all(b"CGTT");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].start_offset, 1); // 'C' is at local offset 1 in "ACGT"
        assert_eq!(hits[0].end_offset, 1); // exclusive: one byte consumed in node_b ("T")
        assert_eq!(hits[0].nodes.len(), 2); // spans two nodes
    }

    #[test]
    fn seed_index_works_on_db_graph() {
        let (conn, bg_id) = two_node_graph();
        let graph = BlockGroup::get_graph(&conn, &bg_id);
        let matcher = GenGraphMatcher::new(&conn, graph);

        let idx = SeedIndex::build(&matcher, 4);

        // "CGTT" is a junction-spanning 4-mer.
        assert!(idx.table.contains_key(b"CGTT".as_slice()));

        let hits = matcher.find_all_with_seed_index(&idx, b"CGTT");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].nodes.len(), 2);
    }

    /// Import `simple.fa` and apply `complex.vcf` to produce real variant graphs,
    /// then verify that the matcher correctly resolves each allele branch.
    ///
    /// `complex.vcf` records a multi-allelic site at position 10 of m123
    /// (REF=TCG, ALT=TG,T,TCAG).  The `bar` sample (GT=0/1) is heterozygous
    /// for the reference and the first alternate, producing a diamond:
    ///
    ///   PREFIX("ATCGATCGA") → "TCG" (ref) → SUFFIX("ATCGATCGGGAACACACAGAGA")
    ///                       → "TG"  (alt) → (same suffix)
    ///
    /// The `baz` sample (GT=2/3) holds the two non-reference alternates.
    #[test]
    fn vcf_update_fork_matches_correct_allele() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();
        track_database(conn, op_conn).unwrap();

        let collection = "test";
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/vcfs/complex.vcf");

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            collection,
            Sample::DEFAULT_NAME,
            false,
        )
        .unwrap();
        update_with_vcf(
            &context,
            &vcf_path.to_str().unwrap().to_string(),
            collection,
            "".to_string(),
            None,
            Some(Sample::DEFAULT_NAME),
            false,
        )
        .unwrap();

        // ── bar: ref + alt1 (TCG → TG) ──────────────────────────────────────
        let bar_bg = get_sample_bg(conn, collection, "bar");
        let mut bar_graph = BlockGroup::get_graph(conn, &bar_bg.id);
        BlockGroup::prune_graph(&mut bar_graph);
        let bar = GenGraphMatcher::new(conn, bar_graph);

        // Full allele-path sequences are reachable.
        assert!(bar.contains(b"ATCGATCGATCGATCGATCGGGAACACACAGAGA")); // ref path
        assert!(bar.contains(b"ATCGATCGATGATCGATCGGGAACACACAGAGA")); // alt path (TCG→TG)

        // Cross-junction: allele → shared suffix ("GGG" anchor is unique to suffix).
        assert!(bar.contains(b"TCGATCGATCGGG")); // ref allele "TCG" → suffix
        assert!(bar.contains(b"TGATCGATCGGG")); // alt allele "TG" → suffix

        // No allele that bridges across both branches exists.
        assert!(!bar.contains(b"TCGTGATCG")); // ref immediately followed by alt start
        // baz's alt2 "T" allele is not in bar's graph.
        assert!(!bar.contains(b"TATCGATCGGG"));

        // find_all returns one hit per allele.  Both queries start at the last
        // byte of the PREFIX node (m123[0..10][9]='T'), cross through the allele
        // node, and end inside the SUFFIX node — 3 nodes in each path.
        //
        // Indel normalisation strips the anchor base from the alt allele, so the
        // graph split point is at coordinate 10 of m123, not 9.  m123 is therefore
        // split into [0..10], [10..12] (ref allele "CG"), and [12..34] (suffix).
        let ref_hits = bar.find_all(b"TCGATCGATCGGG");
        assert_eq!(ref_hits.len(), 1);
        assert_eq!(ref_hits[0].nodes.len(), 3); // prefix tail → ref-allele node → suffix node

        let alt_hits = bar.find_all(b"TGATCGATCGGG");
        assert_eq!(alt_hits.len(), 1);
        assert_eq!(alt_hits[0].nodes.len(), 3); // prefix tail → alt-allele node → suffix node

        // ── baz: alt2 ("T") + alt3 ("TCAG"), no reference path ───────────────
        let baz_bg = get_sample_bg(conn, collection, "baz");
        let mut baz_graph = BlockGroup::get_graph(conn, &baz_bg.id);
        BlockGroup::prune_graph(&mut baz_graph);
        let baz = GenGraphMatcher::new(conn, baz_graph);

        assert!(baz.contains(b"ATCGATCGATATCGATCGGGAACACACAGAGA")); // alt2 path
        assert!(baz.contains(b"ATCGATCGATCAGATCGATCGGGAACACACAGAGA")); // alt3 path
        assert!(!baz.contains(b"ATCGATCGATCGATCGATCGGGAACACACAGAGA")); // ref absent

        // Cross-junction for each non-ref allele.
        assert!(baz.contains(b"TATCGATCGGG")); // alt2 "T" → suffix
        assert!(baz.contains(b"TCAGATCGATCGGG")); // alt3 "TCAG" → suffix
    }

    // -----------------------------------------------------------------------
    // Helper: a three-allele site (multi-allelic fork).
    //
    //   PATH_START → PREFIX("CGCGCGCG") → T_NODE("T") → SUFFIX("ATATATATATA") → PATH_END
    //                                   → C_NODE("C") → SUFFIX
    //                                   → G_NODE("G") → SUFFIX
    //
    // Three distinct allele nodes fan out from the same prefix node and
    // reconverge on the same suffix node.
    // -----------------------------------------------------------------------
    fn three_allele_graph() -> (gen_models::db::GraphConnection, HashId) {
        let conn = get_connection(None).unwrap();
        let seq_pre = Sequence::new().sequence_type("DNA").sequence("CGCGCGCG").save(&conn);
        let node_pre = Node::create(&conn, &seq_pre.hash, &HashId::convert_str("tri-prefix"));
        let seq_t = Sequence::new().sequence_type("DNA").sequence("T").save(&conn);
        let node_t = Node::create(&conn, &seq_t.hash, &HashId::convert_str("tri-allele-t"));
        let seq_c = Sequence::new().sequence_type("DNA").sequence("C").save(&conn);
        let node_c = Node::create(&conn, &seq_c.hash, &HashId::convert_str("tri-allele-c"));
        let seq_g = Sequence::new().sequence_type("DNA").sequence("G").save(&conn);
        let node_g = Node::create(&conn, &seq_g.hash, &HashId::convert_str("tri-allele-g"));
        let seq_suf = Sequence::new().sequence_type("DNA").sequence("ATATATATATA").save(&conn);
        let node_suf = Node::create(&conn, &seq_suf.hash, &HashId::convert_str("tri-suffix"));
        Collection::create(&conn, "test");
        Sample::get_or_create(&conn, "test");
        let bg = BlockGroup::create(&conn, "test", "test", "chr1");
        let e0 = Edge::create(&conn, PATH_START_NODE_ID, 0, Strand::Forward, node_pre, 0, Strand::Forward);
        let e_pt = Edge::create(&conn, node_pre, 8, Strand::Forward, node_t, 0, Strand::Forward);
        let e_pc = Edge::create(&conn, node_pre, 8, Strand::Forward, node_c, 0, Strand::Forward);
        let e_pg = Edge::create(&conn, node_pre, 8, Strand::Forward, node_g, 0, Strand::Forward);
        let e_ts = Edge::create(&conn, node_t, 1, Strand::Forward, node_suf, 0, Strand::Forward);
        let e_cs = Edge::create(&conn, node_c, 1, Strand::Forward, node_suf, 0, Strand::Forward);
        let e_gs = Edge::create(&conn, node_g, 1, Strand::Forward, node_suf, 0, Strand::Forward);
        let e5 = Edge::create(&conn, node_suf, 11, Strand::Forward, PATH_END_NODE_ID, 0, Strand::Forward);
        BlockGroupEdge::bulk_create(&conn, &[
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e0.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e_pt.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e_pc.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e_pg.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e_ts.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e_cs.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e_gs.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e5.id, chromosome_index: 0, phased: 0 },
        ]);
        (conn, bg.id)
    }

    /// A three-allele (multi-allelic) site: exactly the matching allele is found
    /// for each of three possible query strings that span the variant position,
    /// and a fourth allele that is absent does not match.
    #[test]
    fn three_allele_site_db() {
        let (conn, bg_id) = three_allele_graph();
        let graph = BlockGroup::get_graph(&conn, &bg_id);
        let matcher = GenGraphMatcher::new(&conn, graph);

        // Each of the three alleles is present in the graph.
        // Last 4 of prefix = "CGCG", allele, first 5 of suffix = "ATATA".
        assert!(matcher.contains(b"CGCGTATATA")); // via T allele
        assert!(matcher.contains(b"CGCGCATATA")); // via C allele
        assert!(matcher.contains(b"CGCGGATATA")); // via G allele

        // "A" is not an allele node in this graph.
        assert!(!matcher.contains(b"CGCGAATATA"));

        // Each query returns exactly one match that spans three nodes.
        let t_hits = matcher.find_all(b"CGCGTATATA");
        assert_eq!(t_hits.len(), 1);
        assert_eq!(t_hits[0].nodes.len(), 3); // prefix → T → suffix

        let c_hits = matcher.find_all(b"CGCGCATATA");
        assert_eq!(c_hits.len(), 1);
        assert_eq!(c_hits[0].nodes.len(), 3);

        let g_hits = matcher.find_all(b"CGCGGATATA");
        assert_eq!(g_hits.len(), 1);
        assert_eq!(g_hits[0].nodes.len(), 3);

        // find_all for the full suffix (entirely within one node) gives one result.
        let suf_hits = matcher.find_all(b"ATATATATATA");
        assert_eq!(suf_hits.len(), 1);
        assert_eq!(suf_hits[0].start_offset, 0);
    }

    /// Two allele nodes that carry the exact same sequence converge on a shared
    /// suffix node.  find_all must return one match per branch — not one total.
    ///
    ///   PATH_START → PREFIX("ACGT") → node_x1("T") → SUFFIX("GCGC") → PATH_END
    ///                               → node_x2("T") → SUFFIX
    ///
    /// A query that spans PREFIX-tail + allele + SUFFIX-head enters SUFFIX at the
    /// same (q_idx, node, offset) state from both branches.  When we kept a
    /// visited-set tracker, it would let the first branch mark that state visited
    /// and silently drop the second branch's match.
    #[test]
    fn duplicate_allele_sequence_returns_two_hits() {
        let conn = get_connection(None).unwrap();
        let seq_pre = Sequence::new().sequence_type("DNA").sequence("ACGT").save(&conn);
        let node_pre = Node::create(&conn, &seq_pre.hash, &HashId::convert_str("dup-prefix"));
        let seq_t = Sequence::new().sequence_type("DNA").sequence("T").save(&conn);
        let node_t1 = Node::create(&conn, &seq_t.hash, &HashId::convert_str("dup-allele-t1"));
        let node_t2 = Node::create(&conn, &seq_t.hash, &HashId::convert_str("dup-allele-t2"));
        let seq_suf = Sequence::new().sequence_type("DNA").sequence("GCGC").save(&conn);
        let node_suf = Node::create(&conn, &seq_suf.hash, &HashId::convert_str("dup-suffix"));
        Collection::create(&conn, "test");
        Sample::get_or_create(&conn, "test");
        let bg = BlockGroup::create(&conn, "test", "test", "chr1");
        let e0 = Edge::create(&conn, PATH_START_NODE_ID, 0, Strand::Forward, node_pre, 0, Strand::Forward);
        let e_t1 = Edge::create(&conn, node_pre, 4, Strand::Forward, node_t1, 0, Strand::Forward);
        let e_t2 = Edge::create(&conn, node_pre, 4, Strand::Forward, node_t2, 0, Strand::Forward);
        let e_s1 = Edge::create(&conn, node_t1, 1, Strand::Forward, node_suf, 0, Strand::Forward);
        let e_s2 = Edge::create(&conn, node_t2, 1, Strand::Forward, node_suf, 0, Strand::Forward);
        let e5 = Edge::create(&conn, node_suf, 4, Strand::Forward, PATH_END_NODE_ID, 0, Strand::Forward);
        BlockGroupEdge::bulk_create(&conn, &[
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e0.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e_t1.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e_t2.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e_s1.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e_s2.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e5.id, chromosome_index: 0, phased: 0 },
        ]);
        let graph = BlockGroup::get_graph(&conn, &bg.id);
        let matcher = GenGraphMatcher::new(&conn, graph);
        assert!(matcher.contains(b"ACGTTGCGC"));
        let hits = matcher.find_all(b"GTTGCGC");
        assert_eq!(hits.len(), 2, "expected one hit per duplicate allele branch");
    }

    /// A fork where one branch uses nodes "AT" then "G", and the other uses
    /// nodes "A" then "TG" — functionally identical but split differently.
    /// Both paths produce the same spelled sequence and must each be reported.
    ///
    ///   PREFIX("CCCC") → node_AT("AT") → node_G("G")  → SUFFIX("TTTT") → PATH_END
    ///                  → node_A("A")  → node_TG("TG") → SUFFIX
    #[test]
    fn split_equivalent_paths_return_two_hits() {
        let conn = get_connection(None).unwrap();
        let seq_pre = Sequence::new().sequence_type("DNA").sequence("CCCC").save(&conn);
        let node_pre = Node::create(&conn, &seq_pre.hash, &HashId::convert_str("split-prefix"));
        let seq_at = Sequence::new().sequence_type("DNA").sequence("AT").save(&conn);
        let node_at = Node::create(&conn, &seq_at.hash, &HashId::convert_str("split-at"));
        let seq_g = Sequence::new().sequence_type("DNA").sequence("G").save(&conn);
        let node_g = Node::create(&conn, &seq_g.hash, &HashId::convert_str("split-g"));
        let seq_a = Sequence::new().sequence_type("DNA").sequence("A").save(&conn);
        let node_a = Node::create(&conn, &seq_a.hash, &HashId::convert_str("split-a"));
        let seq_tg = Sequence::new().sequence_type("DNA").sequence("TG").save(&conn);
        let node_tg = Node::create(&conn, &seq_tg.hash, &HashId::convert_str("split-tg"));
        let seq_suf = Sequence::new().sequence_type("DNA").sequence("TTTT").save(&conn);
        let node_suf = Node::create(&conn, &seq_suf.hash, &HashId::convert_str("split-suffix"));
        Collection::create(&conn, "test");
        Sample::get_or_create(&conn, "test");
        let bg = BlockGroup::create(&conn, "test", "test", "chr1");
        let e0 = Edge::create(&conn, PATH_START_NODE_ID, 0, Strand::Forward, node_pre, 0, Strand::Forward);
        let e_p_at = Edge::create(&conn, node_pre, 4, Strand::Forward, node_at, 0, Strand::Forward);
        let e_p_a = Edge::create(&conn, node_pre, 4, Strand::Forward, node_a, 0, Strand::Forward);
        let e_at_g = Edge::create(&conn, node_at, 2, Strand::Forward, node_g, 0, Strand::Forward);
        let e_a_tg = Edge::create(&conn, node_a, 1, Strand::Forward, node_tg, 0, Strand::Forward);
        let e_g_s = Edge::create(&conn, node_g, 1, Strand::Forward, node_suf, 0, Strand::Forward);
        let e_tg_s = Edge::create(&conn, node_tg, 2, Strand::Forward, node_suf, 0, Strand::Forward);
        let e5 = Edge::create(&conn, node_suf, 4, Strand::Forward, PATH_END_NODE_ID, 0, Strand::Forward);
        BlockGroupEdge::bulk_create(&conn, &[
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e0.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e_p_at.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e_p_a.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e_at_g.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e_a_tg.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e_g_s.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e_tg_s.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e5.id, chromosome_index: 0, phased: 0 },
        ]);
        let graph = BlockGroup::get_graph(&conn, &bg.id);
        let matcher = GenGraphMatcher::new(&conn, graph);
        assert!(matcher.contains(b"CCCCATGTTTT"));
        let hits = matcher.find_all(b"CCATGTT");
        assert_eq!(hits.len(), 2, "expected one hit per structurally-equivalent path");
    }

    /// Import the anderson_promoters.gfa fixture (20 promoter variants, ~1024
    /// paths) and verify that the matcher correctly finds:
    ///
    /// 1. A sequence entirely contained in the shared long terminal node (within-node).
    /// 2. A cross-junction sequence spanning the variable region into the
    ///    shared "GCTAGCTCAG" node (segment 11 in the GFA).
    /// 3. A cross-junction sequence spanning from node 11 into both possible
    ///    downstream single-base nodes (segment 12 = "T", segment 13 = "C").
    /// 4. A sequence that does NOT exist in any path through the graph.
    #[test]
    fn anderson_promoters_from_gfa() {
        let gfa_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/anderson_promoters.gfa");

        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();
        track_database(conn, op_conn).unwrap();

        let collection_name = "anderson promoters";
        import_gfa(&context, &gfa_path, collection_name, Sample::DEFAULT_NAME).unwrap();

        let bg_id = BlockGroup::get_id(collection_name, Sample::DEFAULT_NAME, "");
        let graph = BlockGroup::get_graph(conn, &bg_id);
        let matcher = GenGraphMatcher::new(conn, graph);

        // ── 1. Within the shared terminal long sequence ──────────────────────
        // Segment 27 contains a unique subsequence that appears in all 20 paths.
        assert!(matcher.contains(b"GCTAGCTACTAGTGAAAGAGG"));

        // ── 2. Cross-junction: variable node → shared "GCTAGCTCAG" node ──────
        // Segment 9 = "A"  → segment 11 = "GCTAGCTCAG"
        assert!(matcher.contains(b"AGCTAGCTCAG"));
        // Segment 10 = "G" → segment 11 = "GCTAGCTCAG"
        assert!(matcher.contains(b"GGCTAGCTCAG"));

        // ── 3. Cross-junction: "GCTAGCTCAG" → downstream single-base nodes ──
        // Segment 11 "GCTAGCTCAG" → segment 12 "T"
        assert!(matcher.contains(b"GCTAGCTCAGT"));
        // Segment 11 "GCTAGCTCAG" → segment 13 "C"
        assert!(matcher.contains(b"GCTAGCTCAGC"));
        // Three-node span: segment 11 → segment 12 "T" → segment 14 "CCT"
        assert!(matcher.contains(b"GCTAGCTCAGTCCT"));

        // ── 4. Notebook demo query: −10 box "ctagctcagt" ────────────────────
        // From search_and_navigate.ipynb: QUERY = "ctagctcagt".  The GFA stores
        // sequences in uppercase, so this is "CTAGCTCAGT".  It is a
        // cross-junction match: bytes 1–9 of segment 11 ("GCTAGCTCAG") followed
        // by "T" from segment 12.  All 20 Anderson promoters pass through this
        // junction, so find_all returns exactly one hit (one graph node for each
        // segment) spanning two nodes.
        assert!(matcher.contains(b"CTAGCTCAGT"));

        let minus_10_hits = matcher.find_all(b"CTAGCTCAGT");
        assert_eq!(minus_10_hits.len(), 1);
        assert_eq!(minus_10_hits[0].start_offset, 1); // 'C' is at offset 1 in "GCTAGCTCAG"
        assert_eq!(minus_10_hits[0].end_offset, 1); // exclusive: consumed 1 byte of segment 12
        assert_eq!(minus_10_hits[0].nodes.len(), 2); // segment 11 node → segment 12 node

        // ── 5. Notebook demo query: −35 core "ttgac" ────────────────────────
        // From search_and_navigate.ipynb: MINUS_35 = "ttgac" → "TTGAC".  This
        // spans five consecutive single-base nodes: segment 1 ("T") → segment 3
        // ("T") → segment 4 ("G") → segment 6 ("A") → segment 7 ("C").  Paths
        // that go through segment 5 ("T") instead of segment 4 ("G") do not
        // carry this -35 box, but the graph contains both routes so the match
        // exists.
        assert!(matcher.contains(b"TTGAC"));

        let minus_35_hits = matcher.find_all(b"TTGAC");
        assert_eq!(minus_35_hits.len(), 1);
        assert_eq!(minus_35_hits[0].nodes.len(), 5); // five single-byte nodes

        // ── 6. Absent sequence ───────────────────────────────────────────────
        // No node "A" is immediately downstream of segment 11; only 12 ("T")
        // and 13 ("C") are.
        assert!(!matcher.contains(b"GCTAGCTCAGA"));

        // ── 7. Lowercase does NOT match the Rust matcher ─────────────────────
        // The Python `repo.search()` API normalises the query to uppercase before
        // calling into the matcher (fixing the reproduce_bug.ipynb failure where
        // "ctagctcagt" returned 0 matches).  The Rust matcher itself is
        // intentionally byte-exact; lowercase queries must not match here so
        // that the normalisation responsibility stays clearly at the API boundary.
        assert!(!matcher.contains(b"ctagctcagt"));
    }
    // -----------------------------------------------------------------------
    // Single-node helpers
    // -----------------------------------------------------------------------

    /// Build a trivial graph:  PATH_START → node(sequence) → PATH_END
    fn single_node_graph(sequence: &str) -> (GraphConnection, HashId) {
        let conn = get_connection(None).unwrap();
        let seq = Sequence::new().sequence_type("DNA").sequence(sequence).save(&conn);
        let node = Node::create(&conn, &seq.hash, &HashId::convert_str("node-a"));
        Collection::create(&conn, "test");
        Sample::get_or_create(&conn, "test");
        let bg = BlockGroup::create(&conn, "test", "test", "chr1");
        let len = sequence.len() as i64;
        let e0 = Edge::create(&conn, PATH_START_NODE_ID, 0, Strand::Forward, node, 0, Strand::Forward);
        let e1 = Edge::create(&conn, node, len, Strand::Forward, PATH_END_NODE_ID, 0, Strand::Forward);
        BlockGroupEdge::bulk_create(&conn, &[
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e0.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e1.id, chromosome_index: 0, phased: 0 },
        ]);
        (conn, bg.id)
    }

    // -----------------------------------------------------------------------
    // Single-node tests
    // -----------------------------------------------------------------------

    /// Within-node exact match and correct offset reporting.
    /// Covers both "first occurrence" and "mid-node start offset" cases.
    #[test]
    fn single_node_matches() {
        let (conn, bg_id) = single_node_graph("ACGTACGT");
        let graph = BlockGroup::get_graph(&conn, &bg_id);
        let m = GenGraphMatcher::new(&conn, graph);

        assert!(m.contains(b"ACGT"));
        assert!(m.contains(b"CGTA"));
        assert!(m.contains(b"ACGTACGT"));
        assert!(!m.contains(b"NNNN"));

        let hits = m.find_all(b"CGTA");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].start_offset, 1);
        assert_eq!(hits[0].end_offset, 5);
    }

    /// A repeated k-mer yields one hit per occurrence.
    #[test]
    fn match_multiple_start_positions_in_single_node() {
        // "ACGACG" contains "ACG" at both offset 0 and offset 3.
        let (conn, bg_id) = single_node_graph("ACGACG");
        let graph = BlockGroup::get_graph(&conn, &bg_id);
        let m = GenGraphMatcher::new(&conn, graph);

        let hits = m.find_all(b"ACG");
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].start_offset, 0);
        assert_eq!(hits[1].start_offset, 3);
    }

    /// Edge cases: absent query and empty query.
    #[test]
    fn single_node_edge_cases() {
        let (conn, bg_id) = single_node_graph("ACGT");
        let graph = BlockGroup::get_graph(&conn, &bg_id);
        let m = GenGraphMatcher::new(&conn, graph);

        assert!(!m.contains(b"XXXX"));
        assert!(m.find_all(b"XXXX").is_empty());

        assert!(m.contains(b""));
        assert!(m.find_all(b"").is_empty());
    }

    /// Seed index and short-query fallback on a single node.
    #[test]
    fn seed_index_single_node() {
        let (conn, bg_id) = single_node_graph("ACGTACGT");
        let graph = BlockGroup::get_graph(&conn, &bg_id);
        let m = GenGraphMatcher::new(&conn, graph);

        let idx = SeedIndex::build(&m, 4);

        assert!(idx.table.contains_key(b"ACGT".as_slice()));
        assert!(idx.table.contains_key(b"CGTA".as_slice()));

        let hits = m.find_all_with_seed_index(&idx, b"CGTA");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].start_offset, 1);

        // Queries shorter than k fall back to find_all.
        let short_hits = m.find_all_with_seed_index(&idx, b"ACG");
        assert_eq!(short_hits.len(), 2);
    }

    // -----------------------------------------------------------------------
    // Multi-node topology tests
    // -----------------------------------------------------------------------

    /// Fork without reconvergence: A→B and A→C; each branch is matched
    /// independently by its end-node identity.
    #[test]
    fn exact_match_branches_at_junction() {
        let conn = get_connection(None).unwrap();
        let seq_a = Sequence::new().sequence_type("DNA").sequence("AC").save(&conn);
        let seq_b = Sequence::new().sequence_type("DNA").sequence("GT").save(&conn);
        let seq_c = Sequence::new().sequence_type("DNA").sequence("GA").save(&conn);
        let node_a = Node::create(&conn, &seq_a.hash, &HashId::convert_str("node-a"));
        let node_b = Node::create(&conn, &seq_b.hash, &HashId::convert_str("node-b"));
        let node_c = Node::create(&conn, &seq_c.hash, &HashId::convert_str("node-c"));
        Collection::create(&conn, "test");
        Sample::get_or_create(&conn, "test");
        let bg = BlockGroup::create(&conn, "test", "test", "chr1");
        let e0 = Edge::create(&conn, PATH_START_NODE_ID, 0, Strand::Forward, node_a, 0, Strand::Forward);
        let e1 = Edge::create(&conn, node_a, 2, Strand::Forward, node_b, 0, Strand::Forward);
        let e2 = Edge::create(&conn, node_a, 2, Strand::Forward, node_c, 0, Strand::Forward);
        let e3 = Edge::create(&conn, node_b, 2, Strand::Forward, PATH_END_NODE_ID, 0, Strand::Forward);
        let e4 = Edge::create(&conn, node_c, 2, Strand::Forward, PATH_END_NODE_ID, 0, Strand::Forward);
        BlockGroupEdge::bulk_create(&conn, &[
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e0.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e1.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e2.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e3.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e4.id, chromosome_index: 0, phased: 0 },
        ]);
        let graph = BlockGroup::get_graph(&conn, &bg.id);
        let m = GenGraphMatcher::new(&conn, graph);
        assert!(m.contains(b"ACGT"));
        assert!(m.contains(b"ACGA"));
        assert!(!m.contains(b"ACGG"));
        let hits_b = m.find_all(b"ACGT");
        assert_eq!(hits_b.len(), 1);
        assert_eq!(hits_b[0].nodes.last().unwrap().node_id, node_b);
        let hits_c = m.find_all(b"ACGA");
        assert_eq!(hits_c.len(), 1);
        assert_eq!(hits_c[0].nodes.last().unwrap().node_id, node_c);
    }

    /// Diamond: A→B→D and A→C→D, reconverging on D.
    #[test]
    fn match_over_diamond() {
        let conn = get_connection(None).unwrap();
        let seq_a = Sequence::new().sequence_type("DNA").sequence("AC").save(&conn);
        let seq_b = Sequence::new().sequence_type("DNA").sequence("GT").save(&conn);
        let seq_c = Sequence::new().sequence_type("DNA").sequence("GA").save(&conn);
        let seq_d = Sequence::new().sequence_type("DNA").sequence("TC").save(&conn);
        let node_a = Node::create(&conn, &seq_a.hash, &HashId::convert_str("node-a"));
        let node_b = Node::create(&conn, &seq_b.hash, &HashId::convert_str("node-b"));
        let node_c = Node::create(&conn, &seq_c.hash, &HashId::convert_str("node-c"));
        let node_d = Node::create(&conn, &seq_d.hash, &HashId::convert_str("node-d"));
        Collection::create(&conn, "test");
        Sample::get_or_create(&conn, "test");
        let bg = BlockGroup::create(&conn, "test", "test", "chr1");
        let e0 = Edge::create(&conn, PATH_START_NODE_ID, 0, Strand::Forward, node_a, 0, Strand::Forward);
        let e1 = Edge::create(&conn, node_a, 2, Strand::Forward, node_b, 0, Strand::Forward);
        let e2 = Edge::create(&conn, node_a, 2, Strand::Forward, node_c, 0, Strand::Forward);
        let e3 = Edge::create(&conn, node_b, 2, Strand::Forward, node_d, 0, Strand::Forward);
        let e4 = Edge::create(&conn, node_c, 2, Strand::Forward, node_d, 0, Strand::Forward);
        let e5 = Edge::create(&conn, node_d, 2, Strand::Forward, PATH_END_NODE_ID, 0, Strand::Forward);
        BlockGroupEdge::bulk_create(&conn, &[
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e0.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e1.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e2.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e3.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e4.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e5.id, chromosome_index: 0, phased: 0 },
        ]);
        let graph = BlockGroup::get_graph(&conn, &bg.id);
        let m = GenGraphMatcher::new(&conn, graph);
        assert!(m.contains(b"ACGT"));
        assert!(m.contains(b"ACGA"));
        assert!(m.contains(b"ACGTTC"));
        assert!(m.contains(b"ACGATC"));
        assert!(!m.contains(b"ACGG"));
        let hits_b = m.find_all(b"ACGT");
        assert_eq!(hits_b.len(), 1);
        assert_eq!(hits_b[0].nodes.last().unwrap().node_id, node_b);
        let hits_c = m.find_all(b"ACGA");
        assert_eq!(hits_c.len(), 1);
        assert_eq!(hits_c[0].nodes.last().unwrap().node_id, node_c);
        let hits_bd = m.find_all(b"ACGTTC");
        assert_eq!(hits_bd.len(), 1);
        assert_eq!(hits_bd[0].nodes.last().unwrap().node_id, node_d);
        assert_eq!(hits_bd[0].nodes.len(), 3);
        let hits_cd = m.find_all(b"ACGATC");
        assert_eq!(hits_cd.len(), 1);
        assert_eq!(hits_cd[0].nodes.last().unwrap().node_id, node_d);
        assert_eq!(hits_cd[0].nodes.len(), 3);
    }

    /// Same diamond topology as `match_over_diamond`, but node B's sequence is
    /// entered at coordinate 1, so only its second byte ("C") participates.
    /// The edge `node_a(2) → node_b(1)` creates a GraphNode with
    /// sequence_start=1, sequence_end=2, spelling "C" from the stored "XC".
    #[test]
    fn match_over_diamond_with_offset() {
        let conn = get_connection(None).unwrap();
        let seq_a = Sequence::new().sequence_type("DNA").sequence("AC").save(&conn);
        let seq_b = Sequence::new().sequence_type("DNA").sequence("XC").save(&conn);
        let seq_c = Sequence::new().sequence_type("DNA").sequence("GA").save(&conn);
        let seq_d = Sequence::new().sequence_type("DNA").sequence("TC").save(&conn);
        let node_a = Node::create(&conn, &seq_a.hash, &HashId::convert_str("node-a"));
        let node_b = Node::create(&conn, &seq_b.hash, &HashId::convert_str("node-b"));
        let node_c = Node::create(&conn, &seq_c.hash, &HashId::convert_str("node-c"));
        let node_d = Node::create(&conn, &seq_d.hash, &HashId::convert_str("node-d"));
        Collection::create(&conn, "test");
        Sample::get_or_create(&conn, "test");
        let bg = BlockGroup::create(&conn, "test", "test", "chr1");
        let e0 = Edge::create(&conn, PATH_START_NODE_ID, 0, Strand::Forward, node_a, 0, Strand::Forward);
        let e1 = Edge::create(&conn, node_a, 2, Strand::Forward, node_b, 1, Strand::Forward);
        let e2 = Edge::create(&conn, node_a, 2, Strand::Forward, node_c, 0, Strand::Forward);
        let e3 = Edge::create(&conn, node_b, 2, Strand::Forward, node_d, 0, Strand::Forward);
        let e4 = Edge::create(&conn, node_c, 2, Strand::Forward, node_d, 0, Strand::Forward);
        let e5 = Edge::create(&conn, node_d, 2, Strand::Forward, PATH_END_NODE_ID, 0, Strand::Forward);
        BlockGroupEdge::bulk_create(&conn, &[
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e0.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e1.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e2.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e3.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e4.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e5.id, chromosome_index: 0, phased: 0 },
        ]);
        let graph = BlockGroup::get_graph(&conn, &bg.id);
        let m = GenGraphMatcher::new(&conn, graph);
        assert!(m.contains(b"ACC"));
        assert!(m.contains(b"ACGA"));
        assert!(m.contains(b"ACCTC"));
        assert!(m.contains(b"ACGATC"));
        let hits_b = m.find_all(b"ACC");
        assert_eq!(hits_b.len(), 1);
        assert_eq!(hits_b[0].nodes.last().unwrap().node_id, node_b);
        let hits_c = m.find_all(b"ACGA");
        assert_eq!(hits_c.len(), 1);
        assert_eq!(hits_c[0].nodes.last().unwrap().node_id, node_c);
        let hits_bd = m.find_all(b"ACCTC");
        assert_eq!(hits_bd.len(), 1);
        assert_eq!(hits_bd[0].nodes.last().unwrap().node_id, node_d);
        assert_eq!(hits_bd[0].nodes.len(), 3);
    }

    /// Two diamonds in series: A→{B,C}→D→{E,F}→G.
    #[test]
    fn match_over_double_diamond() {
        let conn = get_connection(None).unwrap();
        let seq_a = Sequence::new().sequence_type("DNA").sequence("AC").save(&conn);
        let seq_b = Sequence::new().sequence_type("DNA").sequence("GT").save(&conn);
        let seq_c = Sequence::new().sequence_type("DNA").sequence("GA").save(&conn);
        let seq_d = Sequence::new().sequence_type("DNA").sequence("TC").save(&conn);
        let seq_e = Sequence::new().sequence_type("DNA").sequence("AT").save(&conn);
        let seq_f = Sequence::new().sequence_type("DNA").sequence("CG").save(&conn);
        let seq_g = Sequence::new().sequence_type("DNA").sequence("AA").save(&conn);
        let node_a = Node::create(&conn, &seq_a.hash, &HashId::convert_str("node-a"));
        let node_b = Node::create(&conn, &seq_b.hash, &HashId::convert_str("node-b"));
        let node_c = Node::create(&conn, &seq_c.hash, &HashId::convert_str("node-c"));
        let node_d = Node::create(&conn, &seq_d.hash, &HashId::convert_str("node-d"));
        let node_e = Node::create(&conn, &seq_e.hash, &HashId::convert_str("node-e"));
        let node_f = Node::create(&conn, &seq_f.hash, &HashId::convert_str("node-f"));
        let node_g = Node::create(&conn, &seq_g.hash, &HashId::convert_str("node-g"));
        Collection::create(&conn, "test");
        Sample::get_or_create(&conn, "test");
        let bg = BlockGroup::create(&conn, "test", "test", "chr1");
        let e0 = Edge::create(&conn, PATH_START_NODE_ID, 0, Strand::Forward, node_a, 0, Strand::Forward);
        let e1 = Edge::create(&conn, node_a, 2, Strand::Forward, node_b, 0, Strand::Forward);
        let e2 = Edge::create(&conn, node_a, 2, Strand::Forward, node_c, 0, Strand::Forward);
        let e3 = Edge::create(&conn, node_b, 2, Strand::Forward, node_d, 0, Strand::Forward);
        let e4 = Edge::create(&conn, node_c, 2, Strand::Forward, node_d, 0, Strand::Forward);
        let e5 = Edge::create(&conn, node_d, 2, Strand::Forward, node_e, 0, Strand::Forward);
        let e6 = Edge::create(&conn, node_d, 2, Strand::Forward, node_f, 0, Strand::Forward);
        let e7 = Edge::create(&conn, node_e, 2, Strand::Forward, node_g, 0, Strand::Forward);
        let e8 = Edge::create(&conn, node_f, 2, Strand::Forward, node_g, 0, Strand::Forward);
        let e9 = Edge::create(&conn, node_g, 2, Strand::Forward, PATH_END_NODE_ID, 0, Strand::Forward);
        BlockGroupEdge::bulk_create(&conn, &[
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e0.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e1.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e2.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e3.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e4.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e5.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e6.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e7.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e8.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e9.id, chromosome_index: 0, phased: 0 },
        ]);
        let graph = BlockGroup::get_graph(&conn, &bg.id);
        let m = GenGraphMatcher::new(&conn, graph);
        assert!(m.contains(b"ACGTTCATAA"));
        assert!(m.contains(b"ACGTTCCGAA"));
        assert!(m.contains(b"ACGATCATAA"));
        assert!(m.contains(b"ACGATCCGAA"));
        assert!(!m.contains(b"ACGG"));
        let hits_be = m.find_all(b"ACGTTCATAA");
        assert_eq!(hits_be.len(), 1);
        assert_eq!(hits_be[0].nodes.last().unwrap().node_id, node_g);
        assert_eq!(hits_be[0].nodes.len(), 5);
        let hits_bf = m.find_all(b"ACGTTCCGAA");
        assert_eq!(hits_bf.len(), 1);
        assert_eq!(hits_bf[0].nodes.len(), 5);
        let hits_ce = m.find_all(b"ACGATCATAA");
        assert_eq!(hits_ce.len(), 1);
        assert_eq!(hits_ce[0].nodes.len(), 5);
        let hits_cf = m.find_all(b"ACGATCCGAA");
        assert_eq!(hits_cf.len(), 1);
        assert_eq!(hits_cf[0].nodes.len(), 5);
    }

    /// Linear 4-node chain spanning three junctions: A→B→C→D.
    #[test]
    fn match_spans_three_junctions() {
        let conn = get_connection(None).unwrap();
        let seq_a = Sequence::new().sequence_type("DNA").sequence("AB").save(&conn);
        let seq_b = Sequence::new().sequence_type("DNA").sequence("CD").save(&conn);
        let seq_c = Sequence::new().sequence_type("DNA").sequence("EF").save(&conn);
        let seq_d = Sequence::new().sequence_type("DNA").sequence("G").save(&conn);
        let node_a = Node::create(&conn, &seq_a.hash, &HashId::convert_str("node-a"));
        let node_b = Node::create(&conn, &seq_b.hash, &HashId::convert_str("node-b"));
        let node_c = Node::create(&conn, &seq_c.hash, &HashId::convert_str("node-c"));
        let node_d = Node::create(&conn, &seq_d.hash, &HashId::convert_str("node-d"));
        Collection::create(&conn, "test");
        Sample::get_or_create(&conn, "test");
        let bg = BlockGroup::create(&conn, "test", "test", "chr1");
        let e0 = Edge::create(&conn, PATH_START_NODE_ID, 0, Strand::Forward, node_a, 0, Strand::Forward);
        let e1 = Edge::create(&conn, node_a, 2, Strand::Forward, node_b, 0, Strand::Forward);
        let e2 = Edge::create(&conn, node_b, 2, Strand::Forward, node_c, 0, Strand::Forward);
        let e3 = Edge::create(&conn, node_c, 2, Strand::Forward, node_d, 0, Strand::Forward);
        let e4 = Edge::create(&conn, node_d, 1, Strand::Forward, PATH_END_NODE_ID, 0, Strand::Forward);
        BlockGroupEdge::bulk_create(&conn, &[
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e0.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e1.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e2.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e3.id, chromosome_index: 0, phased: 0 },
            BlockGroupEdgeData { block_group_id: bg.id, edge_id: e4.id, chromosome_index: 0, phased: 0 },
        ]);
        let graph = BlockGroup::get_graph(&conn, &bg.id);
        let m = GenGraphMatcher::new(&conn, graph);
        assert!(m.contains(b"ABCDEFG"));
        assert!(m.contains(b"BCDEF"));
        assert!(m.contains(b"CDEFG"));
        let hits = m.find_all(b"ABCDEFG");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].nodes.len(), 4);
    }
}
