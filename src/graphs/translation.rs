use std::collections::{HashMap, HashSet, VecDeque};

use gen_annotations::projection::AnnotationSegment;
use gen_core::{
    HashId, NodeIntervalBlock, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand, range::Range,
};
use gen_graph::{GraphNode, all_intermediate_edges};
use gen_models::{
    accession::Accession,
    annotations::Annotation,
    block_group::{BlockGroup, NewBlockGroup},
    block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
    db::{DbContext, GraphConnection},
    edge::Edge,
    errors::OperationError,
    node::Node,
    operations::OperationInfo,
    path::revcomp,
    sample::{NewSample, Sample},
    sequence::Sequence,
    session_operations::{end_operation, start_operation},
};
use petgraph::{graphmap::DiGraphMap, visit::EdgeRef};
use thiserror::Error;

// ── CodonTable ───────────────────────────────────────────────────────────────

// NCBI table 1 – Standard
const STANDARD_TABLE: [u8; 64] =
    build_table(b"FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG");

// NCBI table 2 – Vertebrate Mitochondrial
const VERT_MITO_TABLE: [u8; 64] =
    build_table(b"FFLLSSSSYY**CCWWLLLLPPPPHHQQRRRRIIMMTTTTNNKKSS**VVVVAAAADDEEGGGG");

// NCBI table 4 – Mycoplasma / Spiroplasma (TGA → Trp)
const MYCOPLASMA_TABLE: [u8; 64] =
    build_table(b"FFLLSSSSYY**CCWWLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG");

// NCBI table 5 – Invertebrate Mitochondrial (AAA -> Lys, AGA/AGG -> Ser)
// https://www.ncbi.nlm.nih.gov/Taxonomy/Utils/wprintgc.cgi
const INVERT_MITO_TABLE: [u8; 64] =
    build_table(b"FFLLSSSSYY**CCWWLLLLPPPPHHQQRRRRIIMMTTTTNNKKSSSSVVVVAAAADDEEGGGG");

// NCBI table 6 – Ciliate Nuclear (TAA/TAG → Gln)
const CILIATE_TABLE: [u8; 64] =
    build_table(b"FFLLSSSSYYQQCC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG");

// NCBI table 11 – Bacterial / Archaeal / Plant Plastid
const BACTERIAL_TABLE: [u8; 64] =
    build_table(b"FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG");

const fn nucleotide_index(b: u8) -> Option<usize> {
    match b {
        b'T' | b't' | b'U' | b'u' => Some(0),
        b'C' | b'c' => Some(1),
        b'A' | b'a' => Some(2),
        b'G' | b'g' => Some(3),
        _ => None,
    }
}

// Build a [u8;64] lookup from an NCBI ncbieaa string.
// The NCBI string lists amino acids for codons in the order defined by
// base1=T*16,C*16,A*16,G*16 × base2=T*4,C*4,A*4,G*4 × base3=T,C,A,G.
// We store them indexed as T=0,C=1,A=2,G=3 in the natural (b1*16+b2*4+b3) layout.
const fn build_table(ncbieaa: &[u8]) -> [u8; 64] {
    let mut table = [b'X'; 64];
    let mut i = 0usize;
    while i < 64 {
        // Position i in the NCBI string maps to codon (b1_idx, b2_idx, b3_idx)
        // where T=0,C=1,A=2,G=3 and the order is:
        // b1: repeats of 16 → i/16
        // b2: repeats of 4 within 16 → (i%16)/4
        // b3: position within 4 → i%4
        let b1 = i / 16;
        let b2 = (i % 16) / 4;
        let b3 = i % 4;
        let encoded_idx = b1 * 16 + b2 * 4 + b3;
        table[encoded_idx] = ncbieaa[i];
        i += 1;
    }
    table
}

pub struct CodonTable {
    pub id: u8,
    table: [u8; 64],
}

impl CodonTable {
    pub fn standard() -> Self {
        Self {
            id: 1,
            table: STANDARD_TABLE,
        }
    }

    pub fn ncbi(id: u8) -> Option<Self> {
        let table = match id {
            1 => STANDARD_TABLE,
            2 => VERT_MITO_TABLE,
            4 => MYCOPLASMA_TABLE,
            5 => INVERT_MITO_TABLE,
            6 => CILIATE_TABLE,
            11 => BACTERIAL_TABLE,
            _ => return None,
        };
        Some(Self { id, table })
    }

    pub fn from_ncbi_format(input: &str) -> Result<Self, TranslationError> {
        let mut id: Option<u8> = None;
        let mut ncbieaa_line: Option<String> = None;

        for line in input.lines() {
            let line = line.trim();
            if let Some(rest) = line.strip_prefix("id") {
                let val = rest.trim().trim_matches('"');
                id = Some(
                    val.parse::<u8>()
                        .map_err(|_| TranslationError::CodonTableParse)?,
                );
            } else if let Some(rest) = line.strip_prefix("ncbieaa") {
                let val = rest.trim().trim_matches('"').to_string();
                ncbieaa_line = Some(val);
            }
        }

        let ncbieaa = ncbieaa_line.ok_or(TranslationError::CodonTableParse)?;
        if ncbieaa.len() != 64 {
            return Err(TranslationError::CodonTableParse);
        }

        Ok(Self {
            id: id.unwrap_or(0),
            table: build_table(ncbieaa.as_bytes()),
        })
    }

    fn translate_codon(&self, codon: &[u8]) -> u8 {
        if codon.len() < 3 {
            return b'X';
        }
        let (Some(i1), Some(i2), Some(i3)) = (
            nucleotide_index(codon[0]),
            nucleotide_index(codon[1]),
            nucleotide_index(codon[2]),
        ) else {
            return b'X';
        };
        self.table[i1 * 16 + i2 * 4 + i3]
    }
}

// ── Error ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Error, PartialEq)]
pub enum TranslationError {
    #[error("Cycle detected at node {0}")]
    CycleDetected(HashId),
    #[error("Invalid frame {0}: must be 0, 1, or 2")]
    InvalidFrame(u8),
    #[error("Cannot derive strand: annotation segments have mixed strands")]
    AmbiguousStrand,
    #[error("Annotation has no segments")]
    EmptyAnnotation,
    #[error("Failed to parse NCBI codon table format")]
    CodonTableParse,
    #[error("Sequence error: {0}")]
    Sequence(String),
    #[error("Node error: {0}")]
    NodeError(String),
    #[error("Edge error: {0}")]
    EdgeError(String),
    #[error("BlockGroup error: {0}")]
    BlockGroupError(String),
}

// ── TranslationParams ─────────────────────────────────────────────────────────

pub struct TranslationParams<'a> {
    pub strand: Option<Strand>,
    pub initial_frame: u8,
    pub codon_table: CodonTable,
    pub output_collection_name: &'a str,
    pub output_sample_name: &'a str,
}

impl<'a> TranslationParams<'a> {
    pub fn new(output_collection_name: &'a str, output_sample_name: &'a str) -> Self {
        Self {
            strand: None,
            initial_frame: 0,
            codon_table: CodonTable::standard(),
            output_collection_name,
            output_sample_name,
        }
    }

    pub fn strand(mut self, strand: Strand) -> Self {
        self.strand = Some(strand);
        self
    }

    pub fn initial_frame(mut self, frame: u8) -> Result<Self, TranslationError> {
        if frame > 2 {
            return Err(TranslationError::InvalidFrame(frame));
        }
        self.initial_frame = frame;
        Ok(self)
    }

    pub fn codon_table(mut self, table: CodonTable) -> Self {
        self.codon_table = table;
        self
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// The complete codons a DNA node contributes at a given entry frame, plus the
/// trailing bases that spill into successors.
///
/// `head_skip` leading bases finish a codon begun in a predecessor (handled by an
/// incoming junction). `aa` is the run of fully-contained codons (empty if the
/// node is too short to hold one). `tail` is the trailing partial codon.
struct NodeCodons {
    aa: Vec<u8>,
    tail: Vec<u8>,
}

/// A protein node already wired into the graph, used as the source of the next
/// edge during the codon walk. `PATH_START` and junction nodes are both expressed
/// as a `WalkFrom`.
#[derive(Clone)]
struct WalkFrom {
    id: HashId,
    coord: i64,
    merkle: String,
}

/// Immutable context threaded through the codon walk.
struct WalkCtx<'a> {
    subgraph: &'a DiGraphMap<HashId, ()>,
    dna_by_node: &'a HashMap<HashId, String>,
    protein_node_ids: &'a HashMap<(HashId, u8), HashId>,
    codon_table: &'a CodonTable,
}

/// Wire `from` to `PATH_END` unless it is `PATH_START` (a gene too short to encode
/// a single residue contributes no edge).
fn connect_to_end(
    conn: &GraphConnection,
    bg_id: &HashId,
    from: &WalkFrom,
    ci: i64,
    bg_edges: &mut Vec<BlockGroupEdgeData>,
) -> Result<(), TranslationError> {
    if from.id != PATH_START_NODE_ID {
        bg_edges.push(make_edge(
            conn,
            bg_id,
            from.id,
            from.coord,
            PATH_END_NODE_ID,
            0,
            ci,
        )?);
    }
    Ok(())
}

/// Walk the DNA sub-DAG forward from `from`, entering `node` at offset 0 with
/// `pending` bases (0–2) already collected toward the current codon. Emits one
/// junction node per straddling codon and connects to the next protein anchor (or
/// `PATH_END`). Fully-consumed short nodes are skipped without producing nodes,
/// which is what keeps the protein graph connected.
#[expect(clippy::too_many_arguments, reason = "recursive graph walk state")]
fn codon_walk(
    conn: &GraphConnection,
    bg_id: &HashId,
    ctx: &WalkCtx<'_>,
    from: &WalkFrom,
    pending: &[u8],
    node: HashId,
    ci: i64,
    bg_edges: &mut Vec<BlockGroupEdgeData>,
) -> Result<(), TranslationError> {
    let dna = ctx
        .dna_by_node
        .get(&node)
        .map(|s| s.as_bytes())
        .unwrap_or(&[]);
    let successors: Vec<HashId> = ctx.subgraph.neighbors(node).collect();

    if pending.is_empty() {
        // At a codon boundary entering `node`: frame 0 for this node on this path.
        if let Some(&pid) = ctx.protein_node_ids.get(&(node, 0)) {
            bg_edges.push(make_edge(conn, bg_id, from.id, from.coord, pid, 0, ci)?);
            return Ok(());
        }
        // No complete codon starts here (node shorter than a codon): consume it and
        // continue building the codon in the successors.
        if successors.is_empty() {
            return connect_to_end(conn, bg_id, from, ci, bg_edges);
        }
        let carried = dna.to_vec();
        for succ in &successors {
            codon_walk(conn, bg_id, ctx, from, &carried, *succ, ci, bg_edges)?;
        }
        return Ok(());
    }

    let need = 3 - pending.len();
    if dna.len() < need {
        // `node` is fully consumed by the in-progress codon; keep walking.
        if successors.is_empty() {
            return connect_to_end(conn, bg_id, from, ci, bg_edges);
        }
        let mut carried = pending.to_vec();
        carried.extend_from_slice(dna);
        for succ in &successors {
            codon_walk(conn, bg_id, ctx, from, &carried, *succ, ci, bg_edges)?;
        }
        return Ok(());
    }

    // The codon completes inside `node`.
    let mut codon = pending.to_vec();
    codon.extend_from_slice(&dna[..need]);
    let aa = ctx.codon_table.translate_codon(&codon);
    let aa_char = aa as char;
    // Junction hash derives from the predecessor's hash so synonymous variants that
    // share a predecessor and amino acid collapse to one node.
    let junction_merkle = format!("{}:j:{aa_char}", from.merkle);
    let junction_seq = Sequence::new()
        .sequence_type("AA")
        .sequence(&aa_char.to_string())
        .save(conn)
        .map_err(|e| TranslationError::Sequence(e.to_string()))?;
    let junction_id = Node::create(
        conn,
        &junction_seq.hash,
        &HashId::convert_str(&junction_merkle),
    )
    .map_err(|e| TranslationError::NodeError(e.to_string()))?;
    bg_edges.push(make_edge(
        conn,
        bg_id,
        from.id,
        from.coord,
        junction_id,
        0,
        ci,
    )?);

    let junction_from = WalkFrom {
        id: junction_id,
        coord: 1,
        merkle: junction_merkle,
    };
    // After the codon, `node`'s entry frame on this path equals the bytes carried in.
    let anchor_frame = pending.len() as u8;
    if let Some(&pid) = ctx.protein_node_ids.get(&(node, anchor_frame)) {
        bg_edges.push(make_edge(conn, bg_id, junction_id, 1, pid, 0, ci)?);
        return Ok(());
    }

    // No anchor here (node too short for a full codon after its head): the leftover
    // bases begin the next codon.
    let rem = &dna[need..];
    if successors.is_empty() {
        return connect_to_end(conn, bg_id, &junction_from, ci, bg_edges);
    }
    for succ in &successors {
        codon_walk(conn, bg_id, ctx, &junction_from, rem, *succ, ci, bg_edges)?;
    }
    Ok(())
}

fn make_edge(
    conn: &GraphConnection,
    bg_id: &HashId,
    src: HashId,
    src_coord: i64,
    tgt: HashId,
    tgt_coord: i64,
    chromosome_index: i64,
) -> Result<BlockGroupEdgeData, TranslationError> {
    let edge = Edge::create(
        conn,
        src,
        src_coord,
        Strand::Forward,
        tgt,
        tgt_coord,
        Strand::Forward,
    )
    .map_err(|e| TranslationError::EdgeError(e.to_string()))?;
    Ok(BlockGroupEdgeData {
        block_group_id: *bg_id,
        edge_id: edge.id,
        chromosome_index,
        phased: 0,
    })
}

fn virtual_id(gn: GraphNode) -> HashId {
    HashId::convert_str(&format!(
        "gn:{}:{}:{}",
        gn.node_id, gn.sequence_start, gn.sequence_end
    ))
}

// ── Main function ─────────────────────────────────────────────────────────────

/// The DNA sub-DAG to translate, plus its entry/exit frontiers.
///
/// `graph` keys are "virtual IDs" — one per GraphNode slice. `node_ranges` maps a
/// virtual ID to `(real_node_id, seq_start, seq_end)`; `None` bounds mean "use the
/// full node sequence". `entry_nodes`/`exit_nodes` are the source/sink frontiers
/// (every node directly after the graph start / before the graph end), so parallel
/// branches at the first and last columns are all represented.
struct TranslationSubgraph {
    graph: DiGraphMap<HashId, ()>,
    node_ranges: HashMap<HashId, (HashId, Option<i64>, Option<i64>)>,
    edge_chromosome_indices: HashMap<(HashId, HashId), Vec<i64>>,
    entry_nodes: Vec<HashId>,
    exit_nodes: Vec<HashId>,
}

fn is_terminal(node_id: HashId) -> bool {
    node_id == PATH_START_NODE_ID || node_id == PATH_END_NODE_ID
}

/// Translate the full block group: the whole sub-DAG between the graph start and
/// end nodes, with every parallel branch represented. Stop codons are recorded as
/// `*` and translation continues through them.
pub fn translate_block_group(
    conn: &GraphConnection,
    block_group_id: &HashId,
    params: TranslationParams<'_>,
) -> Result<BlockGroup, TranslationError> {
    let bg = BlockGroup::get_by_id(conn, block_group_id)
        .map_err(|e| TranslationError::BlockGroupError(e.to_string()))?;
    let subgraph = extract_full_graph(conn, block_group_id)?;
    let strand = params.strand.unwrap_or(Strand::Forward);
    let label = HashId::convert_str(&format!("translate-full:{block_group_id}"));
    let bg_name = format!("{}-protein", bg.name);
    translate_core(conn, subgraph, strand, params, &bg_name, label)
}

/// Extract the entire block-group graph (everything between the start and end
/// terminals) as an in-memory sub-DAG. No database writes.
fn extract_full_graph(
    conn: &GraphConnection,
    bg_id: &HashId,
) -> Result<TranslationSubgraph, TranslationError> {
    let gen_graph = BlockGroup::get_graph(conn, bg_id)
        .map_err(|e| TranslationError::BlockGroupError(e.to_string()))?;
    let start_gn = gen_graph
        .nodes()
        .find(|gn| gn.node_id == PATH_START_NODE_ID)
        .ok_or_else(|| TranslationError::BlockGroupError("graph has no start node".into()))?;
    let end_gn = gen_graph
        .nodes()
        .find(|gn| gn.node_id == PATH_END_NODE_ID)
        .ok_or_else(|| TranslationError::BlockGroupError("graph has no end node".into()))?;

    let mut graph: DiGraphMap<HashId, ()> = DiGraphMap::new();
    let mut node_ranges: HashMap<HashId, (HashId, Option<i64>, Option<i64>)> = HashMap::new();
    let mut edge_chromosome_indices: HashMap<(HashId, HashId), Vec<i64>> = HashMap::new();
    let mut entry_set: HashSet<HashId> = HashSet::new();
    let mut exit_set: HashSet<HashId> = HashSet::new();

    for edge_ref in all_intermediate_edges(&gen_graph, start_gn, end_gn) {
        let src = edge_ref.source();
        let tgt = edge_ref.target();
        // Edges touching a terminal mark the entry/exit frontier but are not added
        // to the DNA subgraph (terminals carry no sequence).
        if is_terminal(src.node_id) && !is_terminal(tgt.node_id) {
            entry_set.insert(virtual_id(tgt));
        }
        if is_terminal(tgt.node_id) && !is_terminal(src.node_id) {
            exit_set.insert(virtual_id(src));
        }
        if is_terminal(src.node_id) || is_terminal(tgt.node_id) {
            continue;
        }
        let src_vid = virtual_id(src);
        let tgt_vid = virtual_id(tgt);
        graph.add_node(src_vid);
        graph.add_node(tgt_vid);
        graph.add_edge(src_vid, tgt_vid, ());
        for graph_edge in edge_ref.weight() {
            edge_chromosome_indices
                .entry((src_vid, tgt_vid))
                .or_default()
                .push(graph_edge.chromosome_index);
        }
    }

    // A single-node graph (start → N → end) has no internal edges; capture N too.
    for gn in gen_graph.nodes() {
        if is_terminal(gn.node_id) {
            continue;
        }
        let vid = virtual_id(gn);
        if graph.contains_node(vid) || entry_set.contains(&vid) || exit_set.contains(&vid) {
            graph.add_node(vid);
            node_ranges.insert(
                vid,
                (gn.node_id, Some(gn.sequence_start), Some(gn.sequence_end)),
            );
        }
    }

    Ok(TranslationSubgraph {
        graph,
        node_ranges,
        edge_chromosome_indices,
        entry_nodes: entry_set.into_iter().collect(),
        exit_nodes: exit_set.into_iter().collect(),
    })
}

/// Extract the sub-DAG between an annotation's entry and exit coordinates from the
/// block group's GenGraph, capturing every variant branch in between. No database
/// writes. The entry/exit nodes are trimmed to the annotation's coordinates.
fn extract_annotation(
    conn: &GraphConnection,
    bg_id: &HashId,
    segments: &[AnnotationSegment],
) -> Result<TranslationSubgraph, TranslationError> {
    let gen_graph = BlockGroup::get_graph(conn, bg_id)
        .map_err(|e| TranslationError::BlockGroupError(e.to_string()))?;

    let entry_node_id = segments[0].node_id;
    let entry_coord = segments[0].range.start;
    let last_seg = segments.last().unwrap();
    let exit_node_id = last_seg.node_id;
    let exit_coord = last_seg.range.end;

    // GraphNode slices that contain the annotation's entry/exit coordinates.
    let entry_gn = gen_graph
        .nodes()
        .find(|gn| {
            gn.node_id == entry_node_id
                && gn.sequence_start <= entry_coord
                && gn.sequence_end > entry_coord
        })
        .ok_or_else(|| {
            TranslationError::NodeError(format!(
                "annotation entry coordinate {entry_coord} not found in graph"
            ))
        })?;
    let exit_gn = gen_graph
        .nodes()
        .find(|gn| {
            gn.node_id == exit_node_id
                && gn.sequence_start < exit_coord
                && gn.sequence_end >= exit_coord
        })
        .ok_or_else(|| {
            TranslationError::NodeError(format!(
                "annotation exit coordinate {exit_coord} not found in graph"
            ))
        })?;

    let mut graph: DiGraphMap<HashId, ()> = DiGraphMap::new();
    let mut node_ranges: HashMap<HashId, (HashId, Option<i64>, Option<i64>)> = HashMap::new();
    let mut edge_chromosome_indices: HashMap<(HashId, HashId), Vec<i64>> = HashMap::new();

    let entry_vid = virtual_id(entry_gn);
    let exit_vid = virtual_id(exit_gn);
    let intermediate_edges = all_intermediate_edges(&gen_graph, entry_gn, exit_gn);

    if intermediate_edges.is_empty() {
        // Single-node annotation (no graph splits within it).
        graph.add_node(entry_vid);
        node_ranges.insert(
            entry_vid,
            (entry_gn.node_id, Some(entry_coord), Some(exit_coord)),
        );
        return Ok(TranslationSubgraph {
            graph,
            node_ranges,
            edge_chromosome_indices,
            entry_nodes: vec![entry_vid],
            exit_nodes: vec![exit_vid],
        });
    }

    for edge_ref in &intermediate_edges {
        let src_vid = virtual_id(edge_ref.source());
        let tgt_vid = virtual_id(edge_ref.target());
        graph.add_node(src_vid);
        graph.add_node(tgt_vid);
        graph.add_edge(src_vid, tgt_vid, ());
        for graph_edge in edge_ref.weight() {
            edge_chromosome_indices
                .entry((src_vid, tgt_vid))
                .or_default()
                .push(graph_edge.chromosome_index);
        }
    }

    // DNA range for every node in the subgraph; entry/exit trimmed to the
    // annotation's coordinates.
    for gn in gen_graph.nodes() {
        let vid = virtual_id(gn);
        if !graph.contains_node(vid) {
            continue;
        }
        let seq_start = if gn == entry_gn {
            entry_coord
        } else {
            gn.sequence_start
        };
        let seq_end = if gn == exit_gn {
            exit_coord
        } else {
            gn.sequence_end
        };
        node_ranges.insert(vid, (gn.node_id, Some(seq_start), Some(seq_end)));
    }

    Ok(TranslationSubgraph {
        graph,
        node_ranges,
        edge_chromosome_indices,
        entry_nodes: vec![entry_vid],
        exit_nodes: vec![exit_vid],
    })
}

/// Translate a gene annotation. The variant branches inside the annotated region
/// come from the block group's GenGraph, so a `block_group_id` is required.
pub fn translate_annotation(
    conn: &GraphConnection,
    annotation: &Annotation,
    block_group_id: Option<&HashId>,
    params: TranslationParams<'_>,
) -> Result<BlockGroup, TranslationError> {
    let bg_id = block_group_id.ok_or_else(|| {
        TranslationError::BlockGroupError("translation requires a block group id".into())
    })?;

    let accession_nodes = Accession::get_nodes_by_id(conn, &annotation.accession_id);
    let segments: Vec<AnnotationSegment> = accession_nodes
        .iter()
        .map(AnnotationSegment::from)
        .collect();
    if segments.is_empty() {
        return Err(TranslationError::EmptyAnnotation);
    }

    let strand = match params.strand {
        Some(s) => s,
        None => {
            let first_strand = segments[0].strand;
            if segments
                .iter()
                .any(|s| s.strand != first_strand && s.strand != Strand::Unknown)
            {
                return Err(TranslationError::AmbiguousStrand);
            }
            first_strand
        }
    };

    let subgraph = extract_annotation(conn, bg_id, &segments)?;
    let bg_name = format!("{}-protein", annotation.name);
    translate_core(conn, subgraph, strand, params, &bg_name, annotation.id)
}

/// Translate a coordinate range on a block group's current path into a protein
/// sequence graph, in memory. The path's interval tree maps the path-space
/// `start`/`end` coordinates to node-level entry/exit points; the same
/// `extract_annotation` + `translate_core` pipeline used by `translate_annotation`
/// is then applied to the resulting subgraph.
pub fn translate_path_range(
    conn: &GraphConnection,
    bg_id: &HashId,
    start: i64,
    end: i64,
    params: TranslationParams<'_>,
) -> Result<BlockGroup, TranslationError> {
    let bg = BlockGroup::get_by_id(conn, bg_id)
        .map_err(|e| TranslationError::BlockGroupError(e.to_string()))?;
    let path = BlockGroup::get_current_path(conn, bg_id)
        .map_err(|e| TranslationError::BlockGroupError(e.to_string()))?;
    let tree = path
        .intervaltree(conn)
        .map_err(|e| TranslationError::BlockGroupError(e.to_string()))?;

    let entry_block: NodeIntervalBlock = tree
        .query_point(start)
        .map(|e| e.value)
        .next()
        .ok_or_else(|| {
            TranslationError::NodeError(format!("no node found at path coordinate {start}"))
        })?;
    // Query at `end - 1` so an exclusive end coordinate lands inside the last node.
    let exit_block: NodeIntervalBlock = tree
        .query_point(end - 1)
        .map(|e| e.value)
        .next()
        .ok_or_else(|| {
            TranslationError::NodeError(format!("no node found at path coordinate {end}"))
        })?;

    let entry_seq = entry_block.sequence_start + (start - entry_block.start);
    let exit_seq = exit_block.sequence_start + (end - exit_block.start);

    let strand = params.strand.unwrap_or(entry_block.strand);

    let segments = if entry_block.node_id == exit_block.node_id {
        vec![AnnotationSegment {
            node_id: entry_block.node_id,
            range: Range {
                start: entry_seq,
                end: exit_seq,
            },
            strand,
        }]
    } else {
        vec![
            AnnotationSegment {
                node_id: entry_block.node_id,
                range: Range {
                    start: entry_seq,
                    end: entry_block.sequence_end,
                },
                strand,
            },
            AnnotationSegment {
                node_id: exit_block.node_id,
                range: Range {
                    start: exit_block.sequence_start,
                    end: exit_seq,
                },
                strand,
            },
        ]
    };

    let subgraph = extract_annotation(conn, bg_id, &segments)?;
    let bg_name = format!("{}-protein", bg.name);
    let label = HashId::convert_str(&format!("translate-range:{bg_id}:{start}-{end}"));
    translate_core(conn, subgraph, strand, params, &bg_name, label)
}

// ── Operation wrapper ─────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum TranslationOperationError {
    #[error("Translation failed: {0}")]
    Translation(#[from] TranslationError),
    #[error("Transaction error: {0}")]
    Transaction(String),
    #[error("Operation tracking failed: {0}")]
    OperationTracking(#[from] OperationError),
}

/// Run a translation closure inside a DB transaction and operation record.
///
/// Rolls back both connections on any failure. This is the canonical wrapper
/// used by the CLI, Python, and R bindings — call it instead of hand-rolling
/// `start_operation` / `end_operation` around translation.
pub fn with_translation_operation<F>(
    ctx: &DbContext,
    label: &str,
    f: F,
) -> Result<BlockGroup, TranslationOperationError>
where
    F: FnOnce() -> Result<BlockGroup, TranslationError>,
{
    let graph_conn = ctx.graph().conn();
    let operation_conn = ctx.operations().conn();
    let mut session = start_operation(graph_conn);

    graph_conn
        .execute("BEGIN TRANSACTION", [])
        .map_err(|e| TranslationOperationError::Transaction(e.to_string()))?;
    operation_conn
        .execute("BEGIN TRANSACTION", [])
        .map_err(|e| TranslationOperationError::Transaction(e.to_string()))?;

    let protein_bg = match f() {
        Err(e) => {
            let _ = graph_conn.execute("ROLLBACK TRANSACTION;", []);
            let _ = operation_conn.execute("ROLLBACK TRANSACTION;", []);
            return Err(TranslationOperationError::Translation(e));
        }
        Ok(bg) => bg,
    };

    let summary = format!(
        " {}: protein block group derived from {label}",
        protein_bg.name
    );
    if let Err(e) = end_operation(
        ctx,
        &mut session,
        &OperationInfo {
            files: vec![],
            description: "translate annotation".to_string(),
        },
        &summary,
        None,
    ) {
        let _ = graph_conn.execute("ROLLBACK TRANSACTION;", []);
        let _ = operation_conn.execute("ROLLBACK TRANSACTION;", []);
        return Err(TranslationOperationError::OperationTracking(e));
    }

    graph_conn
        .execute("END TRANSACTION;", [])
        .map_err(|e| TranslationOperationError::Transaction(e.to_string()))?;
    operation_conn
        .execute("END TRANSACTION;", [])
        .map_err(|e| TranslationOperationError::Transaction(e.to_string()))?;

    Ok(protein_bg)
}

/// Shared translation core: fetch sequences, orient by strand, propagate reading
/// frames, then build and persist the protein block group.
///
/// `label_hash` seeds the Merkle hash of nodes with no protein predecessor.
fn translate_core(
    conn: &GraphConnection,
    subgraph: TranslationSubgraph,
    strand: Strand,
    params: TranslationParams<'_>,
    bg_name: &str,
    label_hash: HashId,
) -> Result<BlockGroup, TranslationError> {
    let TranslationSubgraph {
        graph: subgraph,
        node_ranges,
        edge_chromosome_indices,
        entry_nodes,
        exit_nodes,
    } = subgraph;

    // Collect real node IDs needed for sequence fetch.
    let real_node_ids: Vec<HashId> = {
        let mut seen = HashSet::new();
        node_ranges
            .values()
            .filter_map(|(nid, _, _)| if seen.insert(*nid) { Some(*nid) } else { None })
            .collect()
    };
    let sequences_by_node = Node::get_sequences_by_node_ids(conn, &real_node_ids);

    let mut dna_by_node: HashMap<HashId, String> = HashMap::new();
    for vid in subgraph.nodes() {
        let (real_nid, seq_start, seq_end) = node_ranges
            .get(&vid)
            .ok_or_else(|| TranslationError::Sequence(format!("no range for node {vid}")))?;
        let seq = sequences_by_node
            .get(real_nid)
            .ok_or_else(|| TranslationError::Sequence(format!("missing node {real_nid}")))?;
        let slice = seq
            .get_sequence(*seq_start, *seq_end)
            .map_err(|e| TranslationError::Sequence(e.to_string()))?;
        dna_by_node.insert(vid, slice);
    }

    // Reverse strand: rev-comp sequences and flip graph edges; swap entry/exit.
    // Also reverse the direction of keys in edge_chromosome_indices to match the
    // flipped subgraph.
    let (subgraph, dna_by_node, entry_nodes, exit_nodes, edge_chromosome_indices) =
        if strand == Strand::Reverse {
            let rev_dna: HashMap<HashId, String> = dna_by_node
                .iter()
                .map(|(&nid, seq)| (nid, revcomp(seq)))
                .collect();
            let mut rev_graph: DiGraphMap<HashId, ()> = DiGraphMap::new();
            for node in subgraph.nodes() {
                rev_graph.add_node(node);
            }
            for (src, tgt, _) in subgraph.all_edges() {
                rev_graph.add_edge(tgt, src, ());
            }
            let rev_ci: HashMap<(HashId, HashId), Vec<i64>> = edge_chromosome_indices
                .into_iter()
                .map(|((s, t), v)| ((t, s), v))
                .collect();
            (rev_graph, rev_dna, exit_nodes, entry_nodes, rev_ci)
        } else {
            (
                subgraph,
                dna_by_node,
                entry_nodes,
                exit_nodes,
                edge_chromosome_indices,
            )
        };

    // Step 3: frame propagation via Kahn's BFS
    let mut entry_frames: HashMap<HashId, HashSet<u8>> = HashMap::new();
    let mut in_degree: HashMap<HashId, usize> = HashMap::new();

    for node in subgraph.nodes() {
        in_degree.insert(node, 0);
        entry_frames.insert(node, HashSet::new());
    }
    for (_, tgt, _) in subgraph.all_edges() {
        *in_degree.get_mut(&tgt).unwrap() += 1;
    }

    let mut queue: VecDeque<HashId> = VecDeque::new();
    for &entry_node in &entry_nodes {
        entry_frames
            .get_mut(&entry_node)
            .unwrap()
            .insert(params.initial_frame);
        queue.push_back(entry_node);
    }
    let mut processed: HashSet<HashId> = HashSet::new();

    while let Some(node) = queue.pop_front() {
        if processed.contains(&node) {
            continue;
        }
        processed.insert(node);

        // Full-width length: casting to u8 first would corrupt `% 3` for nodes
        // longer than 255 bp (256 % 3 == 1), silently shifting the reading frame.
        let seq_len = dna_by_node.get(&node).map(|s| s.len()).unwrap_or(0);
        let frames: Vec<u8> = entry_frames[&node].iter().copied().collect();

        for frame in frames {
            let exit_frame = ((frame as usize + seq_len) % 3) as u8;
            for successor in subgraph.neighbors(node) {
                entry_frames.get_mut(&successor).unwrap().insert(exit_frame);
                let deg = in_degree.get_mut(&successor).unwrap();
                if *deg > 0 {
                    *deg -= 1;
                }
                if *deg == 0 && !processed.contains(&successor) {
                    queue.push_back(successor);
                }
            }
        }
    }

    for (node, &deg) in &in_degree {
        if deg > 0 && !processed.contains(node) {
            return Err(TranslationError::CycleDetected(*node));
        }
    }

    // Step 4: for each (node, entry_frame) collect the fully-contained codons and
    // the trailing partial. A node shorter than its head_skip + one codon yields no
    // amino acids (its bases belong to junction codons handled during wiring).
    let mut node_codons: HashMap<(HashId, u8), NodeCodons> = HashMap::new();
    for (&node_id, frames) in &entry_frames {
        let seq_bytes = match dna_by_node.get(&node_id) {
            Some(s) => s.as_bytes(),
            None => continue,
        };
        for &frame in frames {
            let head_skip = ((3 - frame as usize) % 3).min(seq_bytes.len());
            let body = &seq_bytes[head_skip..];
            let codon_count = body.len() / 3;
            let split = codon_count * 3;
            let aa: Vec<u8> = body[..split]
                .chunks_exact(3)
                .map(|c| params.codon_table.translate_codon(c))
                .collect();
            let tail = body[split..].to_vec();
            node_codons.insert((node_id, frame), NodeCodons { aa, tail });
        }
    }

    // Step 5 & 6: create protein BlockGroup and persist nodes/edges
    Sample::get_or_create(
        conn,
        NewSample {
            name: params.output_sample_name,
            is_reference: false,
        },
    )
    .map_err(|e| TranslationError::BlockGroupError(e.to_string()))?;

    let protein_bg = BlockGroup::create(
        conn,
        NewBlockGroup {
            collection_name: params.output_collection_name,
            sample_name: params.output_sample_name,
            name: bg_name,
            parent_block_group_id: None,
            is_default: false,
        },
    )
    .map_err(|e| TranslationError::BlockGroupError(e.to_string()))?;

    let bg_id = protein_bg.id;
    let mut bg_edges: Vec<BlockGroupEdgeData> = Vec::new();

    // Merkle-chain hashing: each protein node's hash = hash(sorted_predecessor_protein_hashes + aa_sequence).
    // Two DNA nodes that translate to the same AA and have predecessors with identical
    // protein hashes collapse to the same protein node (synonymous-variant collapse).
    // Process nodes in topological order so predecessor hashes are always available.
    let mut topo_order: Vec<HashId> = Vec::new();
    {
        let mut indegree: HashMap<HashId, usize> = subgraph.nodes().map(|n| (n, 0)).collect();
        for (_, tgt, _) in subgraph.all_edges() {
            *indegree.get_mut(&tgt).unwrap() += 1;
        }
        let mut q: VecDeque<HashId> = indegree
            .iter()
            .filter(|(_, d)| **d == 0)
            .map(|(&n, _)| n)
            .collect();
        while let Some(n) = q.pop_front() {
            topo_order.push(n);
            for succ in subgraph.neighbors(n) {
                let d = indegree.get_mut(&succ).unwrap();
                *d -= 1;
                if *d == 0 {
                    q.push_back(succ);
                }
            }
        }
    }

    // Anchor protein nodes: one per (dna_node, entry_frame) that contains at least
    // one complete codon. Nodes with no complete codon contribute only to junctions
    // and get no anchor. Merkle hashing collapses synonymous variants that share
    // both predecessor protein hashes and amino-acid sequence.
    let mut protein_node_ids: HashMap<(HashId, u8), HashId> = HashMap::new();
    let mut protein_merkle_hashes: HashMap<(HashId, u8), HashId> = HashMap::new();
    let mut protein_aa_len: HashMap<(HashId, u8), i64> = HashMap::new();

    for &node_id in &topo_order {
        let frames: Vec<u8> = match entry_frames.get(&node_id) {
            Some(fs) => fs.iter().copied().collect(),
            None => continue,
        };
        for frame in frames {
            let codons = match node_codons.get(&(node_id, frame)) {
                Some(c) if !c.aa.is_empty() => c,
                _ => continue,
            };
            let aa_str = String::from_utf8_lossy(&codons.aa).into_owned();
            let seq = Sequence::new()
                .sequence_type("AA")
                .sequence(&aa_str)
                .save(conn)
                .map_err(|e| TranslationError::Sequence(e.to_string()))?;

            // Collect protein hashes of all DNA predecessors whose exit frame equals
            // this node's entry frame, then sort for determinism.
            let mut pred_hashes: Vec<String> = Vec::new();
            for src in subgraph.neighbors_directed(node_id, petgraph::Direction::Incoming) {
                let src_len = dna_by_node.get(&src).map(|s| s.len()).unwrap_or(0);
                if let Some(src_frames) = entry_frames.get(&src) {
                    for &src_frame in src_frames {
                        if (src_frame as usize + src_len) % 3 == frame as usize
                            && let Some(h) = protein_merkle_hashes.get(&(src, src_frame))
                        {
                            pred_hashes.push(h.to_string());
                        }
                    }
                }
            }
            pred_hashes.sort();
            if pred_hashes.is_empty() {
                pred_hashes.push(label_hash.to_string());
            }

            let node_hash = HashId::convert_str(&format!("{}:{}", pred_hashes.join(","), aa_str));
            let nid = Node::create(conn, &seq.hash, &node_hash)
                .map_err(|e| TranslationError::NodeError(e.to_string()))?;
            protein_node_ids.insert((node_id, frame), nid);
            protein_merkle_hashes.insert((node_id, frame), node_hash);
            protein_aa_len.insert((node_id, frame), codons.aa.len() as i64);
        }
    }

    // Step 6: wire the protein graph by walking codons across the DNA sub-DAG.
    let exit_set: HashSet<HashId> = exit_nodes.iter().copied().collect();
    let ctx = WalkCtx {
        subgraph: &subgraph,
        dna_by_node: &dna_by_node,
        protein_node_ids: &protein_node_ids,
        codon_table: &params.codon_table,
    };

    // PATH_START → entry. When an entry node is too short to hold a complete codon,
    // walk forward from the start through its trailing bases instead.
    let start_from = WalkFrom {
        id: PATH_START_NODE_ID,
        coord: 0,
        merkle: label_hash.to_string(),
    };
    for &entry_node in &entry_nodes {
        let frames: Vec<u8> = entry_frames
            .get(&entry_node)
            .map(|fs| fs.iter().copied().collect())
            .unwrap_or_default();
        for frame in frames {
            if let Some(&pid) = protein_node_ids.get(&(entry_node, frame)) {
                bg_edges.push(make_edge(conn, &bg_id, PATH_START_NODE_ID, 0, pid, 0, 0)?);
                continue;
            }
            // No anchor: drop the leading partial codon and walk the remainder forward.
            let dna = dna_by_node
                .get(&entry_node)
                .map(|s| s.as_bytes())
                .unwrap_or(&[]);
            let head_skip = ((3 - frame as usize) % 3).min(dna.len());
            let rem = dna[head_skip..].to_vec();
            for succ in subgraph.neighbors(entry_node) {
                for &ci in edge_chromosome_indices
                    .get(&(entry_node, succ))
                    .map(Vec::as_slice)
                    .unwrap_or(&[0])
                {
                    codon_walk(
                        conn,
                        &bg_id,
                        &ctx,
                        &start_from,
                        &rem,
                        succ,
                        ci,
                        &mut bg_edges,
                    )?;
                }
            }
        }
    }

    // Each anchor's outgoing edges: walk its trailing partial codon into successors,
    // and connect exit anchors to PATH_END.
    for (&(node, frame), &pid) in &protein_node_ids {
        let aa_len = protein_aa_len.get(&(node, frame)).copied().unwrap_or(0);
        let from = WalkFrom {
            id: pid,
            coord: aa_len,
            merkle: protein_merkle_hashes
                .get(&(node, frame))
                .map(|h| h.to_string())
                .unwrap_or_default(),
        };
        let tail = node_codons
            .get(&(node, frame))
            .map(|c| c.tail.clone())
            .unwrap_or_default();
        for succ in subgraph.neighbors(node) {
            for &ci in edge_chromosome_indices
                .get(&(node, succ))
                .map(Vec::as_slice)
                .unwrap_or(&[0])
            {
                codon_walk(conn, &bg_id, &ctx, &from, &tail, succ, ci, &mut bg_edges)?;
            }
        }
        if exit_set.contains(&node) {
            bg_edges.push(make_edge(
                conn,
                &bg_id,
                pid,
                aa_len,
                PATH_END_NODE_ID,
                0,
                0,
            )?);
        }
    }

    // The walk can reach a junction or anchor from several paths; drop duplicate
    // (edge, chromosome_index) entries before persisting.
    let mut seen: HashSet<(HashId, i64)> = HashSet::new();
    bg_edges.retain(|e| seen.insert((e.edge_id, e.chromosome_index)));

    BlockGroupEdge::bulk_create(conn, &bg_edges);

    Ok(protein_bg)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use gen_core::{HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand, range::Range};
    use gen_models::{
        accession::{Accession, AccessionSpan, NewAccession},
        annotations::{Annotation, AnnotationGroup},
        block_group::BlockGroup,
        block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
        collection::Collection,
        db::GraphConnection,
        edge::Edge,
        node::Node,
        path::{Path, revcomp},
        sample::Sample,
        sample_lineage::SampleLineage,
        sequence::Sequence,
    };
    use petgraph::algo::{connected_components, has_path_connecting};

    use super::{
        CodonTable, TranslationError, TranslationParams, translate_annotation,
        translate_block_group,
    };
    use crate::test_helpers::{create_bg, get_connection};

    // ── CodonTable unit tests ─────────────────────────────────────────────────

    #[test]
    fn standard_met() {
        assert_eq!(CodonTable::standard().translate_codon(b"ATG"), b'M');
    }

    #[test]
    fn standard_stop_taa() {
        assert_eq!(CodonTable::standard().translate_codon(b"TAA"), b'*');
    }

    #[test]
    fn standard_stop_tag() {
        assert_eq!(CodonTable::standard().translate_codon(b"TAG"), b'*');
    }

    #[test]
    fn standard_stop_tga() {
        assert_eq!(CodonTable::standard().translate_codon(b"TGA"), b'*');
    }

    #[test]
    fn ambiguous_n() {
        assert_eq!(CodonTable::standard().translate_codon(b"NNN"), b'X');
    }

    #[test]
    fn table4_tga_is_trp() {
        let table = CodonTable::ncbi(4).unwrap();
        assert_eq!(table.translate_codon(b"TGA"), b'W');
    }

    #[test]
    fn table6_taa_is_gln() {
        let table = CodonTable::ncbi(6).unwrap();
        assert_eq!(table.translate_codon(b"TAA"), b'Q');
    }

    #[test]
    fn all_organism_tables_translate_correctly() {
        // Controls: ATG (Met) and TTT (Phe) never change across tables.
        // Variable codons: TGA, ATA, AGA, AGG, TAA, TAG, AAA each get reassigned
        // in at least one table per the NCBI reference
        // (https://www.ncbi.nlm.nih.gov/Taxonomy/Utils/wprintgc.cgi).
        let dna = b"ATGTTTTGAATAAGAAGGTAATAGAAA";
        let codons: Vec<&[u8]> = dna.chunks(3).collect();

        let expected: [(u8, &str); 6] = [
            (1, "MF*IRR**K"),  // Standard
            (2, "MFWM****K"),  // Vertebrate Mitochondrial
            (4, "MFWIRR**K"),  // Mycoplasma/Spiroplasma
            (5, "MFWMSS**K"),  // Invertebrate Mitochondrial
            (6, "MF*IRRQQK"),  // Ciliate Nuclear
            (11, "MF*IRR**K"), // Bacterial/Archaeal/Plant Plastid
        ];

        for (id, expected_protein) in expected {
            let table = CodonTable::ncbi(id).unwrap();
            let protein: String = codons
                .iter()
                .map(|codon| table.translate_codon(codon) as char)
                .collect();
            assert_eq!(protein, expected_protein, "mismatch for NCBI table {id}");
        }
    }

    #[test]
    fn ncbi_format_roundtrip() {
        let ncbi_str = r#"
id 1
name "Standard"
ncbieaa  "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
"#;
        let table = CodonTable::from_ncbi_format(ncbi_str).unwrap();
        assert_eq!(table.translate_codon(b"ATG"), b'M');
        assert_eq!(table.translate_codon(b"GAA"), b'E');
    }

    #[test]
    fn ncbi_format_invalid() {
        let result = CodonTable::from_ncbi_format("not a valid table");
        assert!(matches!(result, Err(TranslationError::CodonTableParse)));
    }

    #[test]
    fn ncbi_unknown_id() {
        assert!(CodonTable::ncbi(99).is_none());
    }

    // ── Graph fixtures ────────────────────────────────────────────────────────

    /// Fresh connection seeded with a `test` collection, the default sample, and an
    /// empty block group of the given name.
    fn new_block_group(name: &str) -> (GraphConnection, BlockGroup) {
        let conn = get_connection(None).expect("should open in-memory database");
        Collection::create(&conn, "test").unwrap();
        let bg = create_bg(&conn, "test", Sample::DEFAULT_NAME, name);
        (conn, bg)
    }

    /// Save a DNA sequence as a node. Returns the node id and its length.
    fn mk_node(conn: &GraphConnection, dna: &str, tag: &str) -> (HashId, i64) {
        let seq = Sequence::new()
            .sequence_type("DNA")
            .sequence(dna)
            .save(conn)
            .unwrap();
        let nid = Node::create(
            conn,
            &seq.hash,
            &HashId::convert_str(&format!("{tag}:{}", seq.hash)),
        )
        .unwrap();
        (nid, seq.length)
    }

    /// Create a forward block-group edge between two nodes at the given coordinates.
    fn mk_edge(
        conn: &GraphConnection,
        src: HashId,
        src_coord: i64,
        tgt: HashId,
        tgt_coord: i64,
    ) -> HashId {
        Edge::create(
            conn,
            src,
            src_coord,
            Strand::Forward,
            tgt,
            tgt_coord,
            Strand::Forward,
        )
        .unwrap()
        .id
    }

    fn bge(bg_id: HashId, edge_id: HashId, ci: i64) -> BlockGroupEdgeData {
        BlockGroupEdgeData {
            block_group_id: bg_id,
            edge_id,
            chromosome_index: ci,
            phased: 0,
        }
    }

    /// Create an accession (and matching `gene` annotation) spanning a linear node
    /// chain on a single strand. Each entry is a `(node_id, length)` pair.
    fn accession_annotation(
        conn: &GraphConnection,
        bg_id: &HashId,
        name: &str,
        chain: &[(HashId, i64)],
        strand: Strand,
    ) -> Annotation {
        let spans = chain
            .iter()
            .map(|(node, len)| AccessionSpan {
                node_id: *node,
                range: Range {
                    start: 0,
                    end: *len,
                },
                strand,
            })
            .collect();
        let accession = Accession::create(
            conn,
            &NewAccession {
                name: name.to_string(),
                block_group_id: *bg_id,
                parent_accession_id: None,
                spans,
            },
        )
        .unwrap();
        AnnotationGroup::create(conn, "gene").unwrap();
        Annotation::create(conn, name, "gene", &accession.id, None).unwrap()
    }

    /// Linear forward gene: each DNA segment becomes a node, wired into both the
    /// block-group path and the annotation accession (all at chromosome_index 0).
    fn setup_linear_gene(segments: &[&str]) -> (GraphConnection, Annotation, HashId) {
        let (conn, bg) = new_block_group("test-gene");
        let chain: Vec<(HashId, i64)> = segments
            .iter()
            .enumerate()
            .map(|(i, dna)| mk_node(&conn, dna, &format!("n{i}")))
            .collect();

        // Wire a linear chain START → n1 → … → nk → END at chromosome_index 0.
        let mut edges = Vec::new();
        let mut prev = PATH_START_NODE_ID;
        let mut prev_coord = 0;
        for (node, len) in &chain {
            let e = mk_edge(&conn, prev, prev_coord, *node, 0);
            edges.push(bge(bg.id, e, 0));
            prev = *node;
            prev_coord = *len;
        }
        let e_out = mk_edge(&conn, prev, prev_coord, PATH_END_NODE_ID, 0);
        edges.push(bge(bg.id, e_out, 0));
        BlockGroupEdge::bulk_create(&conn, &edges);

        let annotation = accession_annotation(&conn, &bg.id, "test-gene", &chain, Strand::Forward);
        (conn, annotation, bg.id)
    }

    /// Gene with a variant bubble at the middle node:
    ///   prefix → wt_mid  (chromosome_index 0)
    ///   prefix → alt_mid (chromosome_index 1)
    /// rejoining at the shared suffix. The annotation accession follows the
    /// wild-type path only, as a real VCF edit would leave it.
    fn setup_variant_gene(
        prefix: &str,
        wt_mid: &str,
        alt_mid: &str,
        suffix: &str,
    ) -> (GraphConnection, Annotation, HashId) {
        let (conn, bg) = new_block_group("test-gene");
        let (pre, pre_len) = mk_node(&conn, prefix, "pre");
        let (wt, wt_len) = mk_node(&conn, wt_mid, "wt");
        let (alt, alt_len) = mk_node(&conn, alt_mid, "alt");
        let (suf, suf_len) = mk_node(&conn, suffix, "suf");

        // Block-group graph: wild-type path (ci 0) plus the variant detour (ci 1).
        let e_s_pre = mk_edge(&conn, PATH_START_NODE_ID, 0, pre, 0);
        let e_pre_wt = mk_edge(&conn, pre, pre_len, wt, 0);
        let e_wt_suf = mk_edge(&conn, wt, wt_len, suf, 0);
        let e_suf_end = mk_edge(&conn, suf, suf_len, PATH_END_NODE_ID, 0);
        let e_pre_alt = mk_edge(&conn, pre, pre_len, alt, 0);
        let e_alt_suf = mk_edge(&conn, alt, alt_len, suf, 0);
        BlockGroupEdge::bulk_create(
            &conn,
            &[
                bge(bg.id, e_s_pre, 0),
                bge(bg.id, e_pre_wt, 0),
                bge(bg.id, e_wt_suf, 0),
                bge(bg.id, e_suf_end, 0),
                bge(bg.id, e_pre_alt, 1),
                bge(bg.id, e_alt_suf, 1),
            ],
        );

        let annotation = accession_annotation(
            &conn,
            &bg.id,
            "test-gene",
            &[(pre, pre_len), (wt, wt_len), (suf, suf_len)],
            Strand::Forward,
        );
        (conn, annotation, bg.id)
    }

    /// Two fully-connected columns of two nodes each:
    ///   START → {col1a, col1b} → {col2a, col2b} → END
    /// The stored current path runs START → col1a → col2a → END. There is no
    /// annotation; this fixture exercises whole-block-group translation.
    fn setup_parallel_columns() -> (GraphConnection, HashId) {
        let (conn, bg) = new_block_group("columns");

        // GGT→G, GCT→A, GAT→D, TTT→F (all codon-aligned, no junctions).
        let (c1a, l1a) = mk_node(&conn, "GGT", "c1a");
        let (c1b, l1b) = mk_node(&conn, "GCT", "c1b");
        let (c2a, l2a) = mk_node(&conn, "GAT", "c2a");
        let (c2b, _l2b) = mk_node(&conn, "TTT", "c2b");

        let e_s1a = mk_edge(&conn, PATH_START_NODE_ID, 0, c1a, 0);
        let e_s1b = mk_edge(&conn, PATH_START_NODE_ID, 0, c1b, 0);
        let e_1a2a = mk_edge(&conn, c1a, l1a, c2a, 0);
        let e_1a2b = mk_edge(&conn, c1a, l1a, c2b, 0);
        let e_1b2a = mk_edge(&conn, c1b, l1b, c2a, 0);
        let e_1b2b = mk_edge(&conn, c1b, l1b, c2b, 0);
        let e_2a_end = mk_edge(&conn, c2a, l2a, PATH_END_NODE_ID, 0);
        let e_2b_end = mk_edge(&conn, c2b, l2a, PATH_END_NODE_ID, 0);

        BlockGroupEdge::bulk_create(
            &conn,
            &[
                bge(bg.id, e_s1a, 0),
                bge(bg.id, e_s1b, 1),
                bge(bg.id, e_1a2a, 0),
                bge(bg.id, e_1a2b, 1),
                bge(bg.id, e_1b2a, 1),
                bge(bg.id, e_1b2b, 1),
                bge(bg.id, e_2a_end, 0),
                bge(bg.id, e_2b_end, 1),
            ],
        );
        Path::create(&conn, "columns", &bg.id, &[e_s1a, e_1a2a, e_2a_end]).unwrap();
        (conn, bg.id)
    }

    /// A 1-bp SNP that falls mid-codon, modelled as a real block-group bubble
    /// (the annotation's accession only carries the wild-type path):
    ///   pre  = "ATGG"  → M, then tail "G"
    ///   ref  = "A" (ci 0)   alt = "C" (ci 1)
    ///   post = "ATAA"  → completes the junction codon, then "TAA" = *
    /// Junction codon: G + {A|C} + A → GAA = E (wt) / GCA = A (variant).
    /// The protein graph must stay connected: START → M → {E|A} → * → END.
    fn setup_real_bubble() -> (GraphConnection, Annotation, HashId) {
        let (conn, bg) = new_block_group("bubble");

        let (pre, pre_len) = mk_node(&conn, "ATGG", "pre");
        let (rf, rf_len) = mk_node(&conn, "A", "ref");
        let (alt, alt_len) = mk_node(&conn, "C", "alt");
        let (post, post_len) = mk_node(&conn, "ATAA", "post");

        let e_s_pre = mk_edge(&conn, PATH_START_NODE_ID, 0, pre, 0);
        let e_pre_ref = mk_edge(&conn, pre, pre_len, rf, 0);
        let e_ref_post = mk_edge(&conn, rf, rf_len, post, 0);
        let e_post_e = mk_edge(&conn, post, post_len, PATH_END_NODE_ID, 0);
        let e_pre_alt = mk_edge(&conn, pre, pre_len, alt, 0);
        let e_alt_post = mk_edge(&conn, alt, alt_len, post, 0);

        BlockGroupEdge::bulk_create(
            &conn,
            &[
                bge(bg.id, e_s_pre, 0),
                bge(bg.id, e_pre_ref, 0),
                bge(bg.id, e_ref_post, 0),
                bge(bg.id, e_post_e, 0),
                bge(bg.id, e_pre_alt, 1),
                bge(bg.id, e_alt_post, 1),
            ],
        );
        Path::create(
            &conn,
            "bubble",
            &bg.id,
            &[e_s_pre, e_pre_ref, e_ref_post, e_post_e],
        )
        .unwrap();

        // Accession = wild-type linear path only (no variant edges).
        let annotation = accession_annotation(
            &conn,
            &bg.id,
            "bubble-gene",
            &[(pre, pre_len), (rf, rf_len), (post, post_len)],
            Strand::Forward,
        );
        (conn, annotation, bg.id)
    }

    /// Count protein nodes in the given block group whose amino-acid sequence equals `aa`.
    fn count_protein_nodes_with_aa(conn: &GraphConnection, bg_id: &HashId, aa: &str) -> usize {
        let aug_edges = BlockGroupEdge::edges_for_block_group(conn, bg_id);
        let mut node_ids: HashSet<HashId> = HashSet::new();
        for ae in &aug_edges {
            node_ids.insert(ae.edge.source_node_id);
            node_ids.insert(ae.edge.target_node_id);
        }
        node_ids.retain(|id| *id != PATH_START_NODE_ID && *id != PATH_END_NODE_ID);
        let node_ids_vec: Vec<HashId> = node_ids.into_iter().collect();
        let seqs_by_node = Node::get_sequences_by_node_ids(conn, &node_ids_vec);
        seqs_by_node
            .values()
            .filter(|seq| {
                seq.sequence_type == "AA" && seq.get_sequence(None, None).unwrap_or_default() == aa
            })
            .count()
    }

    /// Whether PATH_END is reachable from PATH_START in the protein graph.
    fn start_reaches_end(conn: &GraphConnection, bg_id: &HashId) -> bool {
        let graph = BlockGroup::get_graph(conn, bg_id).expect("should load protein graph");
        let start = graph.nodes().find(|n| n.node_id == PATH_START_NODE_ID);
        let end = graph.nodes().find(|n| n.node_id == PATH_END_NODE_ID);
        match (start, end) {
            (Some(start), Some(end)) => has_path_connecting(&graph, start, end, None),
            _ => false,
        }
    }

    /// Every full PATH_START → PATH_END amino-acid string, read left to right.
    fn protein_full_paths(conn: &GraphConnection, bg_id: &HashId) -> Vec<String> {
        let mut adj: HashMap<HashId, HashSet<HashId>> = HashMap::new();
        for ae in BlockGroupEdge::edges_for_block_group(conn, bg_id) {
            adj.entry(ae.edge.source_node_id)
                .or_default()
                .insert(ae.edge.target_node_id);
        }
        let mut ids: HashSet<HashId> = adj.keys().copied().collect();
        for tgts in adj.values() {
            ids.extend(tgts.iter().copied());
        }
        ids.remove(&PATH_START_NODE_ID);
        ids.remove(&PATH_END_NODE_ID);
        let id_vec: Vec<HashId> = ids.into_iter().collect();
        let seqs = Node::get_sequences_by_node_ids(conn, &id_vec);
        let seq_of = |n: HashId| {
            seqs.get(&n)
                .map(|s| s.get_sequence(None, None).unwrap_or_default())
                .unwrap_or_default()
        };

        let mut results = Vec::new();
        let mut stack: Vec<(HashId, String)> = vec![(PATH_START_NODE_ID, String::new())];
        while let Some((node, acc)) = stack.pop() {
            if node == PATH_END_NODE_ID {
                results.push(acc);
                continue;
            }
            if let Some(tgts) = adj.get(&node) {
                for &t in tgts {
                    let mut next = acc.clone();
                    if t != PATH_END_NODE_ID {
                        next.push_str(&seq_of(t));
                    }
                    stack.push((t, next));
                }
            }
        }
        results.sort();
        results
    }

    // ── Translation integration tests ─────────────────────────────────────────

    #[test]
    fn translate_simple_forward() {
        // ATG→M, GAA→E, TGA→* : single node, frame 0
        let (conn, annotation, bg_id) = setup_linear_gene(&["ATGGAATGA"]);
        let params = TranslationParams::new("test", Sample::DEFAULT_NAME);
        let protein = translate_annotation(&conn, &annotation, Some(&bg_id), params).unwrap();
        assert_eq!(protein_full_paths(&conn, &protein.id), vec!["ME*"]);
    }

    #[test]
    fn translate_frame1() {
        // initial_frame=1 → head_skip=2, reads "GGAATG" → G,M; tail "A" dropped
        let (conn, annotation, bg_id) = setup_linear_gene(&["ATGGAATGA"]);
        let params = TranslationParams::new("test", Sample::DEFAULT_NAME)
            .initial_frame(1)
            .unwrap();
        let protein = translate_annotation(&conn, &annotation, Some(&bg_id), params).unwrap();
        assert_eq!(protein_full_paths(&conn, &protein.id), vec!["GM"]);
    }

    #[test]
    fn translate_junction_codon() {
        // Node A = "ATGG": ATG→M, tail "G" (1 base)
        // Node B = "AATGA": junction G+"AA"="GAA"→E; B[2..]="TGA"→*
        let (conn, annotation, bg_id) = setup_linear_gene(&["ATGG", "AATGA"]);
        let params = TranslationParams::new("test", Sample::DEFAULT_NAME);
        let protein = translate_annotation(&conn, &annotation, Some(&bg_id), params).unwrap();
        assert_eq!(protein_full_paths(&conn, &protein.id), vec!["ME*"]);
    }

    #[test]
    fn translate_short_node_junction() {
        // A="AT" (2b), B="G" (1b), C="GAATGA" (6b), frame 0
        // Junction ATG=M (B fully consumed); node C: GAA→E, TGA→* → "E*"
        let (conn, annotation, bg_id) = setup_linear_gene(&["AT", "G", "GAATGA"]);
        let params = TranslationParams::new("test", Sample::DEFAULT_NAME);
        let protein = translate_annotation(&conn, &annotation, Some(&bg_id), params).unwrap();
        assert_eq!(protein_full_paths(&conn, &protein.id), vec!["ME*"]);
    }

    #[test]
    fn translate_multi_hop_junction() {
        // A="T" (1b), B="G" (1b), C="GGAATGA" (7b), frame 0
        // Multi-hop junction "T"+"G"+"G" = TGG = W; node C → "E*"
        let (conn, annotation, bg_id) = setup_linear_gene(&["T", "G", "GGAATGA"]);
        let params = TranslationParams::new("test", Sample::DEFAULT_NAME);
        let protein = translate_annotation(&conn, &annotation, Some(&bg_id), params).unwrap();
        assert_eq!(protein_full_paths(&conn, &protein.id), vec!["WE*"]);
    }

    #[test]
    fn translate_table4_tga() {
        // Table 4: TGA → W (Trp), not stop
        let (conn, annotation, bg_id) = setup_linear_gene(&["ATGTGA"]);
        let params = TranslationParams::new("test", Sample::DEFAULT_NAME)
            .codon_table(CodonTable::ncbi(4).unwrap());
        let protein = translate_annotation(&conn, &annotation, Some(&bg_id), params).unwrap();
        assert_eq!(protein_full_paths(&conn, &protein.id), vec!["MW"]);
    }

    #[test]
    fn translate_invalid_frame() {
        let result = TranslationParams::new("test", "default").initial_frame(3);
        assert!(matches!(result, Err(TranslationError::InvalidFrame(3))));
    }

    #[test]
    fn translate_ambiguous_strand_error() {
        let (conn, bg) = new_block_group("test-gene");
        let (n1, l1) = mk_node(&conn, "ATGGAATGA", "n1");
        let (n2, l2) = mk_node(&conn, "CCCGGG", "n2");

        // Forward first segment, reverse second segment → ambiguous strand.
        let spans = vec![
            AccessionSpan {
                node_id: n1,
                range: Range { start: 0, end: l1 },
                strand: Strand::Forward,
            },
            AccessionSpan {
                node_id: n2,
                range: Range { start: 0, end: l2 },
                strand: Strand::Reverse,
            },
        ];
        let accession = Accession::create(
            &conn,
            &NewAccession {
                name: "test-gene".to_string(),
                block_group_id: bg.id,
                parent_accession_id: None,
                spans,
            },
        )
        .unwrap();
        AnnotationGroup::create(&conn, "gene").unwrap();
        let annotation =
            Annotation::create(&conn, "test-gene", "gene", &accession.id, None).unwrap();

        let params = TranslationParams::new("test", Sample::DEFAULT_NAME);
        // Strand is resolved before subgraph extraction, so a valid block-group id is
        // required even though this case errors out first.
        let result = translate_annotation(&conn, &annotation, Some(&bg.id), params);
        assert!(
            matches!(result, Err(TranslationError::AmbiguousStrand)),
            "expected AmbiguousStrand, got {:?}",
            result
        );
    }

    /// A reverse-strand gene whose single node is longer than 255 bp must still
    /// translate to the correct protein, read left to right (N → C). Casting the
    /// node length to `u8` before `% 3` would corrupt the reading frame here.
    #[test]
    fn reverse_strand_long_node_reads_left_to_right() {
        // 90-codon CDS (270 bp): ATG + 88×GCT(Ala) + TAA → "M" + 88×"A" + "*".
        let cds = format!("ATG{}TAA", "GCT".repeat(88));
        let expected = format!("M{}*", "A".repeat(88));
        // Reverse strand: the node stores the reverse complement of the CDS.
        let stored = revcomp(&cds);

        let (conn, bg) = new_block_group("rev");
        let (node, len) = mk_node(&conn, &stored, "rev");
        let e_in = mk_edge(&conn, PATH_START_NODE_ID, 0, node, 0);
        let e_out = mk_edge(&conn, node, len, PATH_END_NODE_ID, 0);
        BlockGroupEdge::bulk_create(&conn, &[bge(bg.id, e_in, 0), bge(bg.id, e_out, 0)]);

        // Accession covers the whole node on the reverse strand.
        let annotation =
            accession_annotation(&conn, &bg.id, "rev-gene", &[(node, len)], Strand::Reverse);

        let params = TranslationParams::new("test", Sample::DEFAULT_NAME);
        let protein = translate_annotation(&conn, &annotation, Some(&bg.id), params).unwrap();

        assert_eq!(
            protein_full_paths(&conn, &protein.id),
            vec![expected],
            "reverse-strand protein should read N → C left to right"
        );
    }

    // ── Variant / chromosome_index tests ──────────────────────────────────────

    /// A non-synonymous SNP in the middle node should produce two parallel protein
    /// paths: wildtype (ci 0) and variant (ci 1).
    ///
    /// DNA layout (frame 0, codon-aligned):
    ///   prefix  "ATG"           → M
    ///   wt_mid  "GAA" (ci = 0)  → E
    ///   alt_mid "CAA" (ci = 1)  → Q  ← different amino acid
    ///   suffix  "TGA"           → *
    #[test]
    fn non_synonymous_snp_creates_variant_protein_path() {
        let (conn, annotation, bg_id) = setup_variant_gene("ATG", "GAA", "CAA", "TGA");
        let params = TranslationParams::new("test", Sample::DEFAULT_NAME);
        let protein = translate_annotation(&conn, &annotation, Some(&bg_id), params).unwrap();

        // Both amino acids must appear as protein nodes.
        assert_eq!(
            count_protein_nodes_with_aa(&conn, &protein.id, "E"),
            1,
            "wildtype E node missing"
        );
        assert_eq!(
            count_protein_nodes_with_aa(&conn, &protein.id, "Q"),
            1,
            "variant Q node missing"
        );

        // The protein graph must contain both the wildtype (ci 0) and variant (ci 1)
        // paths so they remain distinguishable.
        let aug_edges = BlockGroupEdge::edges_for_block_group(&conn, &protein.id);
        assert!(
            aug_edges.iter().any(|e| e.chromosome_index == 1),
            "no chromosome_index=1 edge found in protein graph; variant path was lost"
        );
        assert!(
            aug_edges.iter().any(|e| e.chromosome_index == 0),
            "no chromosome_index=0 edge found in protein graph"
        );
    }

    /// A synonymous SNP (different codon, same amino acid) in the middle node must
    /// produce a SINGLE shared protein node for that amino acid — not two separate
    /// ones — while still recording the variant with a chromosome_index = 1 edge.
    ///
    /// DNA layout (frame 0, codon-aligned):
    ///   prefix  "ATG"           → M
    ///   wt_mid  "GAA" (ci = 0)  → E
    ///   alt_mid "GAG" (ci = 1)  → E  ← same amino acid (synonymous)
    ///   suffix  "TGA"           → *
    #[test]
    fn synonymous_snp_collapses_to_single_protein_node() {
        let (conn, annotation, bg_id) = setup_variant_gene("ATG", "GAA", "GAG", "TGA");
        let params = TranslationParams::new("test", Sample::DEFAULT_NAME);
        let protein = translate_annotation(&conn, &annotation, Some(&bg_id), params).unwrap();

        // Synonymous collapse: exactly one protein node with sequence "E".
        assert_eq!(
            count_protein_nodes_with_aa(&conn, &protein.id, "E"),
            1,
            "synonymous SNP should collapse to a single E node"
        );

        // The variant path is still tracked: chromosome_index = 1 must appear on
        // at least one edge (the two parallel edges from M→E).
        let aug_edges = BlockGroupEdge::edges_for_block_group(&conn, &protein.id);
        assert!(
            aug_edges.iter().any(|e| e.chromosome_index == 1),
            "no chromosome_index=1 edge found; synonymous variant path was lost"
        );
    }

    /// A synonymous SNP at a split-codon junction: both the wildtype and the
    /// variant middle node complete the junction codon to the same amino acid.
    /// They must share a single junction protein node.
    ///
    /// DNA layout (frame 0):
    ///   prefix  "ATGG"          → M, tail "G" (1 base)
    ///   wt_mid  "AA" (ci=0)     → junction G+AA = GAA = E
    ///   alt_mid "AG" (ci=1)     → junction G+AG = GAG = E (synonymous!)
    ///   suffix  "TAA"           → * (shared)
    #[test]
    fn synonymous_junction_snp_collapses_junction_node() {
        let (conn, annotation, bg_id) = setup_variant_gene("ATGG", "AA", "AG", "TAA");
        let params = TranslationParams::new("test", Sample::DEFAULT_NAME);
        let protein = translate_annotation(&conn, &annotation, Some(&bg_id), params).unwrap();

        // Exactly one junction node coding for E.
        assert_eq!(
            count_protein_nodes_with_aa(&conn, &protein.id, "E"),
            1,
            "synonymous junction SNP should collapse to a single E junction node"
        );

        // Variant edge must be present.
        let aug_edges = BlockGroupEdge::edges_for_block_group(&conn, &protein.id);
        assert!(
            aug_edges.iter().any(|e| e.chromosome_index == 1),
            "no chromosome_index=1 edge found; junction variant path was lost"
        );
    }

    /// Full block-group translation must surface every node in the first and last
    /// columns, not collapse them to the single node on the current path.
    #[test]
    fn full_block_group_keeps_parallel_columns() {
        let (conn, bg_id) = setup_parallel_columns();
        let params = TranslationParams::new("test", Sample::DEFAULT_NAME);
        let protein = translate_block_group(&conn, &bg_id, params).unwrap();

        for aa in ["G", "A", "D", "F"] {
            assert_eq!(
                count_protein_nodes_with_aa(&conn, &protein.id, aa),
                1,
                "protein node for {aa} missing; a parallel column was collapsed"
            );
        }
        assert!(
            start_reaches_end(&conn, &protein.id),
            "no PATH_START → PATH_END path in protein graph"
        );
    }

    #[test]
    fn variant_bubble_protein_graph_is_connected() {
        let (conn, annotation, bg_id) = setup_real_bubble();
        let params = TranslationParams::new("test", Sample::DEFAULT_NAME);
        let protein = translate_annotation(&conn, &annotation, Some(&bg_id), params).unwrap();

        // Both the wild-type (E) and variant (A) junction residues must appear.
        assert_eq!(
            count_protein_nodes_with_aa(&conn, &protein.id, "E"),
            1,
            "wild-type junction residue E missing"
        );
        assert_eq!(
            count_protein_nodes_with_aa(&conn, &protein.id, "A"),
            1,
            "variant junction residue A missing"
        );
        // M (start) and * (stop) must be present exactly once each.
        assert_eq!(count_protein_nodes_with_aa(&conn, &protein.id, "M"), 1);
        assert_eq!(count_protein_nodes_with_aa(&conn, &protein.id, "*"), 1);

        // The whole protein graph must be a single connected component with a
        // traversable START → END path.
        assert!(
            start_reaches_end(&conn, &protein.id),
            "protein graph has no PATH_START → PATH_END path (disconnected)"
        );
        let protein_graph =
            BlockGroup::get_graph(&conn, &protein.id).expect("should load protein graph");
        assert_eq!(
            connected_components(&protein_graph),
            1,
            "protein graph is fragmented into multiple components"
        );
    }

    // ── Sample-lineage edit tests ─────────────────────────────────────────────
    //
    // `translate_annotation` operates per block group: the annotation pins the
    // entry/exit nodes, and the sequence between them is read from the block
    // group's own graph. These tests model a parent and a child sample as two
    // block groups that share the CDS nodes, then translate the same annotation
    // against each to check how an edit in the child propagates to the protein.

    /// A deletion *upstream* of the annotated CDS in a child sample must leave the
    /// translated protein unchanged: edits outside the annotation's entry/exit
    /// span are invisible to translation.
    #[test]
    fn upstream_deletion_in_child_keeps_translation_identical() {
        let (conn, parent_bg) = new_block_group("test-gene");

        // CDS node (ATG GAA TGA → M E *) shared by both samples, plus an upstream
        // spacer that exists only in the parent.
        let (upstream, up_len) = mk_node(&conn, "GGGGGG", "upstream");
        let (gene, gene_len) = mk_node(&conn, "ATGGAATGA", "gene");

        // Parent path: START → upstream → gene → END.
        let p_start_up = mk_edge(&conn, PATH_START_NODE_ID, 0, upstream, 0);
        let p_up_gene = mk_edge(&conn, upstream, up_len, gene, 0);
        let gene_end = mk_edge(&conn, gene, gene_len, PATH_END_NODE_ID, 0);
        BlockGroupEdge::bulk_create(
            &conn,
            &[
                bge(parent_bg.id, p_start_up, 0),
                bge(parent_bg.id, p_up_gene, 0),
                bge(parent_bg.id, gene_end, 0),
            ],
        );

        // Annotation covers the CDS node only, recorded on the parent block group.
        let annotation = accession_annotation(
            &conn,
            &parent_bg.id,
            "test-gene",
            &[(gene, gene_len)],
            Strand::Forward,
        );

        // Child sample: the upstream spacer is deleted, so START → gene → END.
        let child_bg = create_bg(&conn, "test", "child", "test-gene");
        SampleLineage::create(&conn, Sample::DEFAULT_NAME, "child").unwrap();
        let c_start_gene = mk_edge(&conn, PATH_START_NODE_ID, 0, gene, 0);
        BlockGroupEdge::bulk_create(
            &conn,
            &[
                bge(child_bg.id, c_start_gene, 0),
                bge(child_bg.id, gene_end, 0),
            ],
        );

        let parent_protein = translate_annotation(
            &conn,
            &annotation,
            Some(&parent_bg.id),
            TranslationParams::new("test", Sample::DEFAULT_NAME),
        )
        .unwrap();
        let child_protein = translate_annotation(
            &conn,
            &annotation,
            Some(&child_bg.id),
            TranslationParams::new("test", "child"),
        )
        .unwrap();

        assert_eq!(protein_full_paths(&conn, &parent_protein.id), vec!["ME*"]);
        assert_eq!(
            protein_full_paths(&conn, &child_protein.id),
            protein_full_paths(&conn, &parent_protein.id),
            "upstream deletion changed the annotated protein",
        );
    }

    /// A point mutation *inside* the annotated CDS in a child sample must change
    /// the translated protein. A non-synonymous SNP (GAA → CAA) in the middle
    /// codon flips E → Q.
    #[test]
    fn point_mutation_in_child_changes_translation() {
        let (conn, parent_bg) = new_block_group("test-gene");

        // Shared start/stop codons; the middle codon differs between samples.
        let (start_codon, start_len) = mk_node(&conn, "ATG", "start"); // M
        let (wt_mid, wt_len) = mk_node(&conn, "GAA", "wt"); // E
        let (alt_mid, alt_len) = mk_node(&conn, "CAA", "alt"); // Q
        let (stop_codon, stop_len) = mk_node(&conn, "TGA", "stop"); // *

        // Parent path: START → ATG → GAA → TGA → END.
        let start_in = mk_edge(&conn, PATH_START_NODE_ID, 0, start_codon, 0);
        let p_start_wt = mk_edge(&conn, start_codon, start_len, wt_mid, 0);
        let p_wt_stop = mk_edge(&conn, wt_mid, wt_len, stop_codon, 0);
        let stop_out = mk_edge(&conn, stop_codon, stop_len, PATH_END_NODE_ID, 0);
        BlockGroupEdge::bulk_create(
            &conn,
            &[
                bge(parent_bg.id, start_in, 0),
                bge(parent_bg.id, p_start_wt, 0),
                bge(parent_bg.id, p_wt_stop, 0),
                bge(parent_bg.id, stop_out, 0),
            ],
        );

        // Annotation = wild-type CDS; entry = ATG, exit = TGA.
        let annotation = accession_annotation(
            &conn,
            &parent_bg.id,
            "test-gene",
            &[
                (start_codon, start_len),
                (wt_mid, wt_len),
                (stop_codon, stop_len),
            ],
            Strand::Forward,
        );

        // Child sample: the middle codon is mutated GAA → CAA (E → Q).
        let child_bg = create_bg(&conn, "test", "child", "test-gene");
        SampleLineage::create(&conn, Sample::DEFAULT_NAME, "child").unwrap();
        let c_start_alt = mk_edge(&conn, start_codon, start_len, alt_mid, 0);
        let c_alt_stop = mk_edge(&conn, alt_mid, alt_len, stop_codon, 0);
        BlockGroupEdge::bulk_create(
            &conn,
            &[
                bge(child_bg.id, start_in, 0),
                bge(child_bg.id, c_start_alt, 0),
                bge(child_bg.id, c_alt_stop, 0),
                bge(child_bg.id, stop_out, 0),
            ],
        );

        let parent_protein = translate_annotation(
            &conn,
            &annotation,
            Some(&parent_bg.id),
            TranslationParams::new("test", Sample::DEFAULT_NAME),
        )
        .unwrap();
        let child_protein = translate_annotation(
            &conn,
            &annotation,
            Some(&child_bg.id),
            TranslationParams::new("test", "child"),
        )
        .unwrap();

        assert_eq!(protein_full_paths(&conn, &parent_protein.id), vec!["ME*"]);
        assert_eq!(protein_full_paths(&conn, &child_protein.id), vec!["MQ*"]);
        assert_ne!(
            protein_full_paths(&conn, &child_protein.id),
            protein_full_paths(&conn, &parent_protein.id),
            "point mutation inside the CDS did not change the protein",
        );
    }

    /// A SNP at the *first* base of the annotated CDS should surface the variant
    /// residue (CTG → L) alongside the wild-type (ATG → M).
    ///
    /// Ignored: `extract_annotation` anchors on the entry node and keeps only
    /// nodes on a path from entry to exit, so a variant allele parallel to the
    /// entry (a boundary SNP) is dropped and the protein comes out wild-type
    /// only. Re-enable once the extraction captures alleles at the entry/exit
    /// coordinate rather than the specific anchor node.
    #[ignore = "known limitation: boundary-base variant parallel to the entry anchor is dropped by extract_annotation"]
    #[test]
    fn boundary_first_base_mutation_in_child_is_captured() {
        let (conn, parent_bg) = new_block_group("test-gene");

        // First CDS base is its own node so a SNP there forms a bubble at entry.
        let (first_wt, _) = mk_node(&conn, "A", "first-wt"); // ATG... → M
        let (first_alt, _) = mk_node(&conn, "C", "first-alt"); // CTG... → L
        let (rest, rest_len) = mk_node(&conn, "TGGAATGA", "rest"); // shared remainder

        // Parent: START → A → TGGAATGA → END  (ATG GAA TGA = M E *).
        let start_first = mk_edge(&conn, PATH_START_NODE_ID, 0, first_wt, 0);
        let first_rest = mk_edge(&conn, first_wt, 1, rest, 0);
        let rest_end = mk_edge(&conn, rest, rest_len, PATH_END_NODE_ID, 0);
        BlockGroupEdge::bulk_create(
            &conn,
            &[
                bge(parent_bg.id, start_first, 0),
                bge(parent_bg.id, first_rest, 0),
                bge(parent_bg.id, rest_end, 0),
            ],
        );

        let annotation = accession_annotation(
            &conn,
            &parent_bg.id,
            "test-gene",
            &[(first_wt, 1), (rest, rest_len)],
            Strand::Forward,
        );

        // Child: first base mutated A→C, kept as a bubble alongside the reference.
        //   START → A → rest → END   (ci 0, reference)
        //   START → C → rest → END   (ci 1, variant)
        let child_bg = create_bg(&conn, "test", "child", "test-gene");
        SampleLineage::create(&conn, Sample::DEFAULT_NAME, "child").unwrap();
        let start_alt = mk_edge(&conn, PATH_START_NODE_ID, 0, first_alt, 0);
        let alt_rest = mk_edge(&conn, first_alt, 1, rest, 0);
        BlockGroupEdge::bulk_create(
            &conn,
            &[
                bge(child_bg.id, start_first, 0),
                bge(child_bg.id, first_rest, 0),
                bge(child_bg.id, start_alt, 1),
                bge(child_bg.id, alt_rest, 1),
                bge(child_bg.id, rest_end, 0),
            ],
        );

        let protein = translate_annotation(
            &conn,
            &annotation,
            Some(&child_bg.id),
            TranslationParams::new("test", "child"),
        )
        .unwrap();

        assert_eq!(
            protein_full_paths(&conn, &protein.id),
            vec!["LE*", "ME*"],
            "first-base variant was dropped from the extracted subgraph",
        );
    }

    /// A SNP at the *last* base of the annotated CDS should surface the variant
    /// residue (TGG → W) alongside the wild-type stop (TGA → *).
    ///
    /// Ignored: same root cause as the first-base case — a variant allele
    /// parallel to the exit anchor falls outside `[entry..exit]` and is dropped,
    /// so the protein comes out wild-type only. Re-enable once the extraction
    /// captures alleles at the entry/exit coordinate rather than the specific
    /// anchor node.
    #[ignore = "known limitation: boundary-base variant parallel to the exit anchor is dropped by extract_annotation"]
    #[test]
    fn boundary_last_base_mutation_in_child_is_captured() {
        let (conn, parent_bg) = new_block_group("test-gene");

        // Last CDS base is its own node so a SNP there forms a bubble at exit.
        let (head, head_len) = mk_node(&conn, "ATGGAATG", "head"); // ATG GAA TG_
        let (last_wt, _) = mk_node(&conn, "A", "last-wt"); // …TGA → *
        let (last_alt, _) = mk_node(&conn, "G", "last-alt"); // …TGG → W

        // Parent: START → ATGGAATG → A → END  (ATG GAA TGA = M E *).
        let start_head = mk_edge(&conn, PATH_START_NODE_ID, 0, head, 0);
        let head_last = mk_edge(&conn, head, head_len, last_wt, 0);
        let last_end = mk_edge(&conn, last_wt, 1, PATH_END_NODE_ID, 0);
        BlockGroupEdge::bulk_create(
            &conn,
            &[
                bge(parent_bg.id, start_head, 0),
                bge(parent_bg.id, head_last, 0),
                bge(parent_bg.id, last_end, 0),
            ],
        );

        let annotation = accession_annotation(
            &conn,
            &parent_bg.id,
            "test-gene",
            &[(head, head_len), (last_wt, 1)],
            Strand::Forward,
        );

        // Child: last base mutated A→G, kept as a bubble alongside the reference.
        //   head → A → END   (ci 0, reference)
        //   head → G → END   (ci 1, variant)
        let child_bg = create_bg(&conn, "test", "child", "test-gene");
        SampleLineage::create(&conn, Sample::DEFAULT_NAME, "child").unwrap();
        let head_alt = mk_edge(&conn, head, head_len, last_alt, 0);
        let alt_end = mk_edge(&conn, last_alt, 1, PATH_END_NODE_ID, 0);
        BlockGroupEdge::bulk_create(
            &conn,
            &[
                bge(child_bg.id, start_head, 0),
                bge(child_bg.id, head_last, 0),
                bge(child_bg.id, last_end, 0),
                bge(child_bg.id, head_alt, 1),
                bge(child_bg.id, alt_end, 1),
            ],
        );

        let protein = translate_annotation(
            &conn,
            &annotation,
            Some(&child_bg.id),
            TranslationParams::new("test", "child"),
        )
        .unwrap();

        assert_eq!(
            protein_full_paths(&conn, &protein.id),
            vec!["ME*", "MEW"],
            "last-base variant was dropped from the extracted subgraph",
        );
    }

    /// SPEC (ignored): a deletion at the 5′ boundary should make the gene start
    /// *later* — never reach upstream to backfill — and the variant branch is
    /// translated in the resulting shifted frame all the way to the annotation's
    /// defined end, passing *through* any premature stop (with a frameshift
    /// warning) rather than truncating at it.
    ///
    ///   wt CDS  "ATGAAATAA"   → ATG AAA TAA            → "MK*"
    ///   child deletes the first base, leaving "TGAAATAA" read in frame 0:
    ///           TGA(*) AAT(N) [AA dropped]             → "*N"
    ///
    /// The frameshift warning is part of the spec but cannot be asserted until
    /// translation surfaces one; for now we only assert the protein paths.
    #[ignore = "spec: 5′ boundary deletion (start gene later, frameshift through stop to defined end) not yet implemented"]
    #[test]
    fn boundary_5prime_deletion_in_child_starts_gene_later() {
        let (conn, parent_bg) = new_block_group("test-gene");

        let (first, _) = mk_node(&conn, "A", "first"); // 5′ base
        let (rest, rest_len) = mk_node(&conn, "TGAAATAA", "rest"); // ATG AAA TAA remainder

        // Parent: START → A → TGAAATAA → END  (ATG AAA TAA = M K *).
        let start_first = mk_edge(&conn, PATH_START_NODE_ID, 0, first, 0);
        let first_rest = mk_edge(&conn, first, 1, rest, 0);
        let rest_end = mk_edge(&conn, rest, rest_len, PATH_END_NODE_ID, 0);
        BlockGroupEdge::bulk_create(
            &conn,
            &[
                bge(parent_bg.id, start_first, 0),
                bge(parent_bg.id, first_rest, 0),
                bge(parent_bg.id, rest_end, 0),
            ],
        );

        let annotation = accession_annotation(
            &conn,
            &parent_bg.id,
            "test-gene",
            &[(first, 1), (rest, rest_len)],
            Strand::Forward,
        );

        // Child: the 5′ base is deleted, reference path kept alongside.
        //   START → A → rest → END   (ci 0, reference)
        //   START → rest → END       (ci 1, deletion skips the first base)
        let child_bg = create_bg(&conn, "test", "child", "test-gene");
        SampleLineage::create(&conn, Sample::DEFAULT_NAME, "child").unwrap();
        let start_rest = mk_edge(&conn, PATH_START_NODE_ID, 0, rest, 0);
        BlockGroupEdge::bulk_create(
            &conn,
            &[
                bge(child_bg.id, start_first, 0),
                bge(child_bg.id, first_rest, 0),
                bge(child_bg.id, start_rest, 1),
                bge(child_bg.id, rest_end, 0),
            ],
        );

        let protein = translate_annotation(
            &conn,
            &annotation,
            Some(&child_bg.id),
            TranslationParams::new("test", "child"),
        )
        .unwrap();

        assert_eq!(
            protein_full_paths(&conn, &protein.id),
            vec!["*N", "MK*"],
            "5′ deletion should start the gene later and translate the shifted frame to the defined end",
        );
    }

    /// SPEC (ignored): an insertion at the 5′ boundary should *grow* the
    /// annotation to cover the inserted bases; the variant branch is translated
    /// in the resulting shifted frame to the defined end. Here a 1 bp insertion
    /// splits the stop apart (readthrough), which should also raise a frameshift
    /// warning.
    ///
    ///   wt CDS  "ATGAAATAA"        → ATG AAA TAA        → "MK*"
    ///   child inserts "G" before the start; grown span "GATGAAATAA" in frame 0:
    ///           GAT(D) GAA(E) ATA(I) [A dropped]        → "DEI"
    #[ignore = "spec: 5′ boundary insertion (grow annotation, frameshift to defined end) not yet implemented"]
    #[test]
    fn boundary_5prime_insertion_in_child_grows_annotation() {
        let (conn, parent_bg) = new_block_group("test-gene");

        let (ins, _) = mk_node(&conn, "G", "ins"); // inserted base
        let (gene, gene_len) = mk_node(&conn, "ATGAAATAA", "gene"); // ATG AAA TAA

        // Parent: START → ATGAAATAA → END  (M K *).
        let start_gene = mk_edge(&conn, PATH_START_NODE_ID, 0, gene, 0);
        let gene_end = mk_edge(&conn, gene, gene_len, PATH_END_NODE_ID, 0);
        BlockGroupEdge::bulk_create(
            &conn,
            &[
                bge(parent_bg.id, start_gene, 0),
                bge(parent_bg.id, gene_end, 0),
            ],
        );

        let annotation = accession_annotation(
            &conn,
            &parent_bg.id,
            "test-gene",
            &[(gene, gene_len)],
            Strand::Forward,
        );

        // Child: "G" inserted before the gene, reference path kept alongside.
        //   START → ATGAAATAA → END        (ci 0, reference)
        //   START → G → ATGAAATAA → END     (ci 1, insertion)
        let child_bg = create_bg(&conn, "test", "child", "test-gene");
        SampleLineage::create(&conn, Sample::DEFAULT_NAME, "child").unwrap();
        let start_ins = mk_edge(&conn, PATH_START_NODE_ID, 0, ins, 0);
        let ins_gene = mk_edge(&conn, ins, 1, gene, 0);
        BlockGroupEdge::bulk_create(
            &conn,
            &[
                bge(child_bg.id, start_gene, 0),
                bge(child_bg.id, gene_end, 0),
                bge(child_bg.id, start_ins, 1),
                bge(child_bg.id, ins_gene, 1),
            ],
        );

        let protein = translate_annotation(
            &conn,
            &annotation,
            Some(&child_bg.id),
            TranslationParams::new("test", "child"),
        )
        .unwrap();

        assert_eq!(
            protein_full_paths(&conn, &protein.id),
            vec!["DEI", "MK*"],
            "5′ insertion should grow the annotation and translate the shifted frame to the defined end",
        );
    }
}
