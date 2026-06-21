use std::collections::{HashMap, HashSet, VecDeque};

use gen_annotations::projection;
use gen_core::{
    HashId, NodeIntervalBlock, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand, is_terminal,
};
use gen_graph::{GraphNode, all_intermediate_edges};
use gen_models::{
    annotations::Annotation,
    block_group::{BlockGroup, NewBlockGroup},
    block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
    db::{DbContext, GraphConnection},
    edge::Edge,
    errors::OperationError,
    node::Node,
    operations::OperationInfo,
    path::revcomp,
    sequence::Sequence,
    session_operations::{end_operation, start_operation},
};
use petgraph::{Direction, graphmap::DiGraphMap, visit::EdgeRef};
use thiserror::Error;

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

// Amino-acid byte every codon table uses for a stop codon. A codon walk ends at
// the first in-frame occurrence of this residue.
const STOP_CODON: u8 = b'*';

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
    #[error("Sequence graph error: {0}")]
    BlockGroupError(String),
    #[error("A sequence graph named '{0}' already exists in this sample")]
    DuplicateBlockGroup(String),
}

pub struct TranslationParams<'a> {
    pub strand: Option<Strand>,
    pub initial_frame: u8,
    pub codon_table: CodonTable,
    pub output_collection_name: &'a str,
    pub name: Option<&'a str>,
}

impl<'a> TranslationParams<'a> {
    pub fn new(output_collection_name: &'a str) -> Self {
        Self {
            strand: None,
            initial_frame: 0,
            codon_table: CodonTable::standard(),
            output_collection_name,
            name: None,
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

    /// Name for the protein sequence graph. Defaults to "{source name} (protein)"
    /// when not set, where the source name is the translated annotation,
    /// region, or sequence graph's name.
    pub fn name(mut self, name: &'a str) -> Self {
        self.name = Some(name);
        self
    }
}

/// The complete codons a DNA node contributes at a given entry frame, plus the
/// trailing bases that spill into successors.
///
/// The node's leading bases that finish a codon begun in a predecessor are
/// handled separately by an incoming junction and are not part of either field
/// here. `aa` is the run of fully-contained codons after that head (empty if the
/// node is too short to hold one), excluding the first in-frame stop codon and
/// anything after it. `tail` is the trailing partial codon, empty once a stop has
/// ended the walk. `is_terminal` is set when a stop codon ended the run: the codon
/// walk ends here and the anchor connects to the shared stop node, which in turn
/// connects to the END sentinel.
struct NodeCodons {
    aa: Vec<u8>,
    tail: Vec<u8>,
    is_terminal: bool,
}

/// A protein node already wired into the graph, used as the source of the next
/// edge during the codon walk. `PATH_START` and junction nodes are both expressed
/// as a `WalkFrom`.
#[derive(Clone)]
struct WalkFrom {
    id: HashId,
    coord: i64,
    identity_hash: String,
}

/// Immutable context threaded through the codon walk.
struct WalkCtx<'a> {
    subgraph: &'a DiGraphMap<HashId, ()>,
    dna_by_node: &'a HashMap<HashId, String>,
    protein_node_ids: &'a HashMap<(HashId, u8), HashId>,
    codon_table: &'a CodonTable,
}

/// A site that needs an edge into the protein graph's `*` stop node, recorded
/// during the codon walk and resolved once every site has been visited and the
/// stop node's final identity hash is known (see its use in `translate_from`).
struct PendingStopEdge {
    predecessor_id: HashId,
    predecessor_coord: i64,
    chromosome_index: i64,
    predecessor_hash: String,
}

/// Create (or fetch, by content hash) a protein node identified by its sorted
/// predecessor identity hashes and amino-acid sequence. Returns the node id and
/// its own identity hash, the latter usable as a predecessor hash for whatever
/// node comes next. `Node::create` and `Sequence::save` are both content-hash
/// idempotent, so two calls with the same inputs converge on the same row.
fn create_protein_node(
    conn: &GraphConnection,
    predecessor_hashes: &[String],
    amino_acids: &str,
) -> Result<(HashId, String), TranslationError> {
    let sequence = Sequence::new()
        .sequence_type("AA")
        .sequence(amino_acids)
        .save(conn)
        .map_err(|e| TranslationError::Sequence(e.to_string()))?;
    let identity_hash = format!("{}:{}", predecessor_hashes.join(","), amino_acids);
    let node_id = Node::create(conn, &sequence.hash, &HashId::convert_str(&identity_hash))
        .map_err(|e| TranslationError::NodeError(e.to_string()))?;
    Ok((node_id, identity_hash))
}

/// Whether `bytes` begins with a complete codon that is an in-frame stop. Used
/// at codon-walk sites where no anchor was found, where the leftover bytes are
/// either too short for a codon or their leading codon is a stop (a non-stop
/// leading codon would already have produced an anchor).
fn starts_with_stop(codon_table: &CodonTable, bytes: &[u8]) -> bool {
    bytes.len() >= 3 && codon_table.translate_codon(&bytes[..3]) == STOP_CODON
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
    stop_edges: &mut Vec<PendingStopEdge>,
) -> Result<(), TranslationError> {
    let dna = ctx
        .dna_by_node
        .get(&node)
        .map(|s| s.as_bytes())
        .unwrap_or(&[]);
    let successors: Vec<HashId> = ctx.subgraph.neighbors(node).collect();

    if pending.is_empty() {
        // At a codon boundary entering `node`: frame 0 for this node on this path.
        if let Some(&protein_id) = ctx.protein_node_ids.get(&(node, 0)) {
            bg_edges.push(make_edge(
                conn, bg_id, from.id, from.coord, protein_id, 0, ci,
            )?);
            return Ok(());
        }
        // No complete codon starts here. If the bytes available so far already begin
        // with a stop, the walk ends here regardless of what successors hold.
        if starts_with_stop(ctx.codon_table, dna) {
            stop_edges.push(PendingStopEdge {
                predecessor_id: from.id,
                predecessor_coord: from.coord,
                chromosome_index: ci,
                predecessor_hash: from.identity_hash.clone(),
            });
            return Ok(());
        }
        // Node shorter than a codon: consume it and continue building the codon in
        // the successors.
        if successors.is_empty() {
            return connect_to_end(conn, bg_id, from, ci, bg_edges);
        }
        let carried = dna.to_vec();
        for succ in &successors {
            codon_walk(
                conn, bg_id, ctx, from, &carried, *succ, ci, bg_edges, stop_edges,
            )?;
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
            codon_walk(
                conn, bg_id, ctx, from, &carried, *succ, ci, bg_edges, stop_edges,
            )?;
        }
        return Ok(());
    }

    // The codon completes inside `node`.
    let mut codon = pending.to_vec();
    codon.extend_from_slice(&dna[..need]);
    let aa = ctx.codon_table.translate_codon(&codon);

    if aa == STOP_CODON {
        // First in-frame stop codon ends the walk: record the edge into the
        // eventual stop node, with no junction node created for this codon.
        stop_edges.push(PendingStopEdge {
            predecessor_id: from.id,
            predecessor_coord: from.coord,
            chromosome_index: ci,
            predecessor_hash: from.identity_hash.clone(),
        });
        return Ok(());
    }

    let aa_char = aa as char;
    // A junction always has exactly one predecessor, so the same predecessor-hash
    // formula as a regular anchor (Step 5) collapses to a single-element list
    // here, keeping junction nodes indistinguishable from anchors once built.
    let (junction_id, junction_hash) = create_protein_node(
        conn,
        std::slice::from_ref(&from.identity_hash),
        &aa_char.to_string(),
    )?;
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
        identity_hash: junction_hash,
    };
    // After the codon, `node`'s entry frame on this path equals the bytes carried in.
    let anchor_frame = pending.len() as u8;
    if let Some(&protein_id) = ctx.protein_node_ids.get(&(node, anchor_frame)) {
        bg_edges.push(make_edge(conn, bg_id, junction_id, 1, protein_id, 0, ci)?);
        return Ok(());
    }

    // No anchor here (node too short for a full codon after its head): the leftover
    // bases begin the next codon, unless they already are a stop on their own.
    let rem = &dna[need..];
    if starts_with_stop(ctx.codon_table, rem) {
        stop_edges.push(PendingStopEdge {
            predecessor_id: junction_from.id,
            predecessor_coord: junction_from.coord,
            chromosome_index: ci,
            predecessor_hash: junction_from.identity_hash.clone(),
        });
        return Ok(());
    }
    if successors.is_empty() {
        return connect_to_end(conn, bg_id, &junction_from, ci, bg_edges);
    }
    for succ in &successors {
        codon_walk(
            conn,
            bg_id,
            ctx,
            &junction_from,
            rem,
            *succ,
            ci,
            bg_edges,
            stop_edges,
        )?;
    }
    Ok(())
}

fn make_edge(
    conn: &GraphConnection,
    bg_id: &HashId,
    source: HashId,
    source_coord: i64,
    target: HashId,
    target_coord: i64,
    chromosome_index: i64,
) -> Result<BlockGroupEdgeData, TranslationError> {
    let edge = Edge::create(
        conn,
        source,
        source_coord,
        Strand::Forward,
        target,
        target_coord,
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

/// The DNA sub-DAG to translate, plus where reading frames are seeded
/// (`entry_nodes`) and where the walk connects straight to `PATH_END`
/// (`exit_nodes`).
///
/// `graph` keys are "virtual IDs" — one per GraphNode slice. `node_ranges` maps a
/// virtual ID to `(real_node_id, seq_start, seq_end)`; `None` bounds mean "use the
/// full node sequence". `extract_full_graph` populates `entry_nodes` with every
/// node directly after the graph start (so every parallel branch at the first
/// column gets seeded), and `exit_nodes` with every node directly before the
/// graph end; `extract_from_entry` always sets `entry_nodes` to a single node,
/// with `exit_nodes` derived the same way from there to the graph end.
struct TranslationSubgraph {
    graph: DiGraphMap<HashId, ()>,
    node_ranges: HashMap<HashId, (HashId, Option<i64>, Option<i64>)>,
    edge_chromosome_indices: HashMap<(HashId, HashId), Vec<i64>>,
    entry_nodes: Vec<HashId>,
    exit_nodes: Vec<HashId>,
}

/// Translate the full sequence graph: the whole sub-DAG between the graph start
/// and end nodes, with every parallel branch represented. Each walk ends at the
/// first in-frame stop codon, which is emitted as a `*` residue and then bounded
/// by the END sentinel. Translation does not read through stops: stochastic
/// readthrough is intentionally not modelled, since a stop that never fully
/// terminates would explode the protein graph.
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
    let bg_name = params
        .name
        .map(str::to_string)
        .unwrap_or_else(|| format!("{} (protein)", bg.name));
    translate_from(
        conn,
        subgraph,
        strand,
        params,
        &bg.sample_name,
        &bg_name,
        label,
    )
}

/// Extract the entire sequence graph (everything between the start and end
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

/// Extract the sub-DAG from a single entry coordinate to the sequence graph's
/// end, capturing every variant branch along the way. No database writes.
///
/// The entry coordinate names exactly one node, the sole starting point for the
/// walk. `codon_walk` ends each branch at its own first in-frame stop codon, so
/// nothing past the entry point is ever consulted.
fn extract_from_entry(
    conn: &GraphConnection,
    bg_id: &HashId,
    entry_node_id: HashId,
    entry_coord: i64,
) -> Result<TranslationSubgraph, TranslationError> {
    let gen_graph = BlockGroup::get_graph(conn, bg_id)
        .map_err(|e| TranslationError::BlockGroupError(e.to_string()))?;

    let entry_gn = gen_graph
        .nodes()
        .find(|gn| {
            gn.node_id == entry_node_id
                && gn.sequence_start <= entry_coord
                && gn.sequence_end > entry_coord
        })
        .ok_or_else(|| {
            TranslationError::NodeError(format!(
                "entry coordinate {entry_coord} not found in graph"
            ))
        })?;
    let end_gn = gen_graph
        .nodes()
        .find(|gn| gn.node_id == PATH_END_NODE_ID)
        .ok_or_else(|| TranslationError::BlockGroupError("graph has no end node".into()))?;

    let mut graph: DiGraphMap<HashId, ()> = DiGraphMap::new();
    let mut node_ranges: HashMap<HashId, (HashId, Option<i64>, Option<i64>)> = HashMap::new();
    let mut edge_chromosome_indices: HashMap<(HashId, HashId), Vec<i64>> = HashMap::new();
    let mut exit_nodes: HashSet<HashId> = HashSet::new();

    let entry_vid = virtual_id(entry_gn);
    let intermediate_edges = all_intermediate_edges(&gen_graph, entry_gn, end_gn);
    if intermediate_edges.is_empty() {
        // A direct edge to the end (no internal sequence beyond the entry, e.g. a
        // single-node annotation) makes the entry its own standalone exit too.
        exit_nodes.insert(entry_vid);
    }
    for edge_ref in &intermediate_edges {
        let src = edge_ref.source();
        let tgt = edge_ref.target();
        if is_terminal(tgt.node_id) && !is_terminal(src.node_id) {
            exit_nodes.insert(virtual_id(src));
        }
        if is_terminal(src.node_id) || is_terminal(tgt.node_id) {
            continue;
        }
        let src_vid = virtual_id(src);
        let tgt_vid = virtual_id(tgt);
        graph.add_node(src_vid);
        graph.add_node(tgt_vid);
        graph.add_edge(src_vid, tgt_vid, ());
        let indices = edge_chromosome_indices
            .entry((src_vid, tgt_vid))
            .or_default();
        for graph_edge in edge_ref.weight() {
            if !indices.contains(&graph_edge.chromosome_index) {
                indices.push(graph_edge.chromosome_index);
            }
        }
    }

    // The entry node itself, or an exit reached only via a direct edge to a
    // terminal, has no non-terminal edge of its own and so is skipped by the
    // loop above; add it directly.
    graph.add_node(entry_vid);
    for &vid in &exit_nodes {
        graph.add_node(vid);
    }

    // DNA range for every node in the subgraph; only the entry node is trimmed,
    // to the entry coordinate.
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
        node_ranges.insert(vid, (gn.node_id, Some(seq_start), Some(gn.sequence_end)));
    }

    Ok(TranslationSubgraph {
        graph,
        node_ranges,
        edge_chromosome_indices,
        entry_nodes: vec![entry_vid],
        exit_nodes: exit_nodes.into_iter().collect(),
    })
}

/// Translate a gene annotation: take the entry coordinate and strand from the
/// annotation's first segment (the transcription-start end of its accession
/// path, see `projection::annotation_segments`) and translate from there, the
/// same way `translate_from_path` does for a raw path coordinate. Translation
/// is not bounded by or spliced across the rest of the annotation's segments:
/// it reads the literal underlying DNA graph from the entry point to its own
/// first in-frame stop codon, the same as every other translation entry point.
pub fn translate_annotation(
    conn: &GraphConnection,
    annotation: &Annotation,
    block_group_id: Option<&HashId>,
    params: TranslationParams<'_>,
) -> Result<BlockGroup, TranslationError> {
    let bg_id = block_group_id.ok_or_else(|| {
        TranslationError::BlockGroupError("translation requires a sequence graph id".into())
    })?;
    let bg = BlockGroup::get_by_id(conn, bg_id)
        .map_err(|e| TranslationError::BlockGroupError(e.to_string()))?;

    let segments = projection::annotation_segments(conn, annotation);
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

    let subgraph = extract_from_entry(conn, bg_id, segments[0].node_id, segments[0].range.start)?;
    let bg_name = params
        .name
        .map(str::to_string)
        .unwrap_or_else(|| format!("{} (protein)", annotation.name));
    translate_from(
        conn,
        subgraph,
        strand,
        params,
        &bg.sample_name,
        &bg_name,
        annotation.id,
    )
}

/// Translate from a single path-space coordinate on a sequence graph's current
/// path into a protein sequence graph, in memory. The coordinate is resolved to
/// a node and node-relative offset via the path's interval tree, then handed to
/// the same entry-point pipeline `translate_annotation` uses.
pub fn translate_from_path(
    conn: &GraphConnection,
    bg_id: &HashId,
    coordinate: i64,
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
        .query_point(coordinate)
        .map(|e| e.value)
        .next()
        .ok_or_else(|| {
            TranslationError::NodeError(format!("no node found at path coordinate {coordinate}"))
        })?;
    let entry_seq = entry_block.sequence_start + (coordinate - entry_block.start);
    let strand = params.strand.unwrap_or(entry_block.strand);

    let subgraph = extract_from_entry(conn, bg_id, entry_block.node_id, entry_seq)?;
    let bg_name = params
        .name
        .map(str::to_string)
        .unwrap_or_else(|| format!("{} (protein)", bg.name));
    let label = HashId::convert_str(&format!("translate-from:{bg_id}:{coordinate}"));
    translate_from(
        conn,
        subgraph,
        strand,
        params,
        &bg.sample_name,
        &bg_name,
        label,
    )
}

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
        " {}: protein sequence graph derived from {label}",
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

/// Shared translation engine every entry point (`translate_annotation`,
/// `translate_from_path`, `translate_block_group`) funnels into once it has built its own
/// `TranslationSubgraph`: fetch sequences, orient by strand, propagate reading
/// frames, then build and persist the protein sequence graph.
///
/// `label_hash` seeds the identity hash of nodes with no protein predecessor.
/// Each node's identity hash also folds in its predecessors' identity hashes, so
/// the protein graph is a Merkle DAG (not a tree: a node can have more than one
/// predecessor, e.g. where a junction or two variant paths reconverge).
fn translate_from(
    conn: &GraphConnection,
    subgraph: TranslationSubgraph,
    strand: Strand,
    params: TranslationParams<'_>,
    sample_name: &str,
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

    // Step 1: fetch the DNA sequence for every node in the subgraph (dna_by_node).
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

    // Step 2: handle reverse-strand orientation. Rev-comp sequences and flip graph
    // edges; swap entry/exit. Also reverse the direction of keys in
    // edge_chromosome_indices to match the flipped subgraph.
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
            let mut aa: Vec<u8> = Vec::with_capacity(codon_count);
            let mut is_terminal = false;
            for codon in body[..split].chunks_exact(3) {
                let amino_acid = params.codon_table.translate_codon(codon);
                if amino_acid == STOP_CODON {
                    // The first in-frame stop ends the walk; it gets its own shared
                    // stop node rather than joining this anchor's sequence, and later
                    // codons in this node are not part of the protein on this path.
                    is_terminal = true;
                    break;
                }
                aa.push(amino_acid);
            }
            // The trailing partial codon only spills into successors when the walk
            // continues past this node.
            let tail = if is_terminal {
                Vec::new()
            } else {
                body[split..].to_vec()
            };
            node_codons.insert(
                (node_id, frame),
                NodeCodons {
                    aa,
                    tail,
                    is_terminal,
                },
            );
        }
    }

    // Create the protein sequence graph that steps 5 and 6 will populate, in the
    // same sample as the DNA sequence graph it was translated from. Not a
    // SampleLineage child of that sample: SampleLineage backs annotation-name
    // resolution across a sample's ancestors (see Annotation::resolve), and the
    // protein/DNA relationship isn't an ancestry relationship in that sense.
    //
    // BlockGroup::create silently no-ops on a name collision instead of erroring
    // (its id is a deterministic hash of collection+sample+name), so check for an
    // existing sequence graph up front rather than risk merging new protein edges
    // into someone else's sequence graph.
    let existing_id = BlockGroup::get_id(params.output_collection_name, sample_name, bg_name, None);
    if BlockGroup::get_by_id(conn, &existing_id).is_ok() {
        return Err(TranslationError::DuplicateBlockGroup(bg_name.to_string()));
    }

    let protein_bg = BlockGroup::create(
        conn,
        NewBlockGroup {
            collection_name: params.output_collection_name,
            sample_name,
            name: bg_name,
            parent_block_group_id: None,
            is_default: false,
        },
    )
    .map_err(|e| TranslationError::BlockGroupError(e.to_string()))?;

    let bg_id = protein_bg.id;
    let mut bg_edges: Vec<BlockGroupEdgeData> = Vec::new();

    // Merkle DAG hashing: each protein node's identity hash =
    // hash(sorted_predecessor_identity_hashes + aa_sequence). Two DNA nodes that
    // translate to the same AA and have predecessors with identical identity
    // hashes collapse to the same protein node (synonymous-variant collapse).
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

    // Step 5: create protein node IDs by content hash (protein_node_ids). One
    // anchor per (dna_node, entry_frame) that contains at least one complete
    // codon. Nodes with no complete codon contribute only to junctions and get no
    // anchor. Hashing on predecessor identity hashes plus amino-acid sequence
    // collapses synonymous variants that share both.
    let mut protein_node_ids: HashMap<(HashId, u8), HashId> = HashMap::new();
    let mut protein_identity_hashes: HashMap<(HashId, u8), String> = HashMap::new();
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

            // Collect identity hashes of all DNA predecessors whose exit frame
            // equals this node's entry frame, then sort for determinism.
            let mut pred_hashes: Vec<String> = Vec::new();
            for src in subgraph.neighbors_directed(node_id, Direction::Incoming) {
                let src_len = dna_by_node.get(&src).map(|s| s.len()).unwrap_or(0);
                if let Some(src_frames) = entry_frames.get(&src) {
                    for &src_frame in src_frames {
                        if (src_frame as usize + src_len) % 3 == frame as usize
                            && let Some(h) = protein_identity_hashes.get(&(src, src_frame))
                        {
                            pred_hashes.push(h.clone());
                        }
                    }
                }
            }
            pred_hashes.sort();
            if pred_hashes.is_empty() {
                pred_hashes.push(label_hash.to_string());
            }

            let (node_id_protein, identity_hash) =
                create_protein_node(conn, &pred_hashes, &aa_str)?;
            protein_node_ids.insert((node_id, frame), node_id_protein);
            protein_identity_hashes.insert((node_id, frame), identity_hash);
            protein_aa_len.insert((node_id, frame), codons.aa.len() as i64);
        }
    }

    // Step 6: wire the protein graph by walking codons across the DNA sub-DAG.
    // The `*` stop node is created once every walk has finished and its full set
    // of predecessors is known (see `stop_edges` below), rather than up front.
    let exit_set: HashSet<HashId> = exit_nodes.iter().copied().collect();
    let mut stop_edges: Vec<PendingStopEdge> = Vec::new();
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
        identity_hash: label_hash.to_string(),
    };
    for &entry_node in &entry_nodes {
        let frames: Vec<u8> = entry_frames
            .get(&entry_node)
            .map(|fs| fs.iter().copied().collect())
            .unwrap_or_default();
        for frame in frames {
            if let Some(&protein_id) = protein_node_ids.get(&(entry_node, frame)) {
                bg_edges.push(make_edge(
                    conn,
                    &bg_id,
                    PATH_START_NODE_ID,
                    0,
                    protein_id,
                    0,
                    0,
                )?);
                continue;
            }
            // No anchor: drop the leading partial codon and walk the remainder
            // forward, unless it is already a stop on its own.
            let dna = dna_by_node
                .get(&entry_node)
                .map(|s| s.as_bytes())
                .unwrap_or(&[]);
            let head_skip = ((3 - frame as usize) % 3).min(dna.len());
            let rem = dna[head_skip..].to_vec();
            if starts_with_stop(&params.codon_table, &rem) {
                stop_edges.push(PendingStopEdge {
                    predecessor_id: start_from.id,
                    predecessor_coord: start_from.coord,
                    chromosome_index: 0,
                    predecessor_hash: start_from.identity_hash.clone(),
                });
                continue;
            }
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
                        &mut stop_edges,
                    )?;
                }
            }
        }
    }

    // Each anchor's outgoing edges: walk its trailing partial codon into successors,
    // and connect exit anchors to PATH_END.
    for (&(node, frame), &protein_id) in &protein_node_ids {
        let aa_len = protein_aa_len.get(&(node, frame)).copied().unwrap_or(0);
        if node_codons
            .get(&(node, frame))
            .is_some_and(|codons| codons.is_terminal)
        {
            // The walk ended at an in-frame stop inside this anchor: record the
            // edge into the eventual stop node, without spilling a tail into
            // successors.
            stop_edges.push(PendingStopEdge {
                predecessor_id: protein_id,
                predecessor_coord: aa_len,
                chromosome_index: 0,
                predecessor_hash: protein_identity_hashes
                    .get(&(node, frame))
                    .cloned()
                    .unwrap_or_default(),
            });
            continue;
        }
        let from = WalkFrom {
            id: protein_id,
            coord: aa_len,
            identity_hash: protein_identity_hashes
                .get(&(node, frame))
                .cloned()
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
                codon_walk(
                    conn,
                    &bg_id,
                    &ctx,
                    &from,
                    &tail,
                    succ,
                    ci,
                    &mut bg_edges,
                    &mut stop_edges,
                )?;
            }
        }
        if exit_set.contains(&node) {
            bg_edges.push(make_edge(
                conn,
                &bg_id,
                protein_id,
                aa_len,
                PATH_END_NODE_ID,
                0,
                0,
            )?);
        }
    }

    // The `*` stop node's identity hash depends on every site that reaches it
    // across the whole walk, so it is only created once `stop_edges` is complete.
    if !stop_edges.is_empty() {
        let mut pred_hashes: Vec<String> = stop_edges
            .iter()
            .map(|edge| edge.predecessor_hash.clone())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        pred_hashes.sort();
        let (stop_id, _) =
            create_protein_node(conn, &pred_hashes, &(STOP_CODON as char).to_string())?;

        let mut stop_exit_chromosome_indices: HashSet<i64> = HashSet::new();
        for edge in &stop_edges {
            bg_edges.push(make_edge(
                conn,
                &bg_id,
                edge.predecessor_id,
                edge.predecessor_coord,
                stop_id,
                0,
                edge.chromosome_index,
            )?);
            stop_exit_chromosome_indices.insert(edge.chromosome_index);
        }
        for ci in stop_exit_chromosome_indices {
            bg_edges.push(make_edge(
                conn,
                &bg_id,
                stop_id,
                1,
                PATH_END_NODE_ID,
                0,
                ci,
            )?);
        }
    }

    // Step 7: dedupe and bulk-insert the edges. The walk can reach a junction or
    // anchor from several paths; drop duplicate (edge, chromosome_index) entries
    // before persisting.
    let mut seen: HashSet<(HashId, i64)> = HashSet::new();
    bg_edges.retain(|e| seen.insert((e.edge_id, e.chromosome_index)));

    BlockGroupEdge::bulk_create(conn, &bg_edges);

    Ok(protein_bg)
}

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
        translate_block_group, translate_from_path,
    };
    use crate::test_helpers::{create_bg, get_connection};

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

    /// Fresh connection seeded with a `test` collection, the default sample, and an
    /// empty block group of the given name.
    fn new_block_group(name: &str) -> (GraphConnection, BlockGroup) {
        let conn = get_connection(None).expect("should open in-memory database");
        Collection::create(&conn, "test").unwrap();
        let bg = create_bg(&conn, "test", Sample::DEFAULT_NAME, name);
        (conn, bg)
    }

    /// Save a DNA sequence as a node. Returns the node id and its length.
    fn build_node(conn: &GraphConnection, dna: &str, tag: &str) -> (HashId, i64) {
        let seq = Sequence::new()
            .sequence_type("DNA")
            .sequence(dna)
            .save(conn)
            .unwrap();
        let node_id = Node::create(
            conn,
            &seq.hash,
            &HashId::convert_str(&format!("{tag}:{}", seq.hash)),
        )
        .unwrap();
        (node_id, seq.length)
    }

    /// Create a forward block-group edge between two nodes at the given coordinates.
    fn build_edge(
        conn: &GraphConnection,
        source: HashId,
        source_coord: i64,
        target: HashId,
        target_coord: i64,
    ) -> HashId {
        Edge::create(
            conn,
            source,
            source_coord,
            Strand::Forward,
            target,
            target_coord,
            Strand::Forward,
        )
        .unwrap()
        .id
    }

    fn build_block_group_edge(
        bg_id: HashId,
        edge_id: HashId,
        chromosome_index: i64,
    ) -> BlockGroupEdgeData {
        BlockGroupEdgeData {
            block_group_id: bg_id,
            edge_id,
            chromosome_index,
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
            .map(|(i, dna)| build_node(&conn, dna, &format!("n{i}")))
            .collect();

        // Wire a linear chain START → n1 → … → nk → END at chromosome_index 0.
        let mut edges = Vec::new();
        let mut prev = PATH_START_NODE_ID;
        let mut prev_coord = 0;
        for (node, len) in &chain {
            let e = build_edge(&conn, prev, prev_coord, *node, 0);
            edges.push(build_block_group_edge(bg.id, e, 0));
            prev = *node;
            prev_coord = *len;
        }
        let e_out = build_edge(&conn, prev, prev_coord, PATH_END_NODE_ID, 0);
        edges.push(build_block_group_edge(bg.id, e_out, 0));
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
        let (pre, pre_len) = build_node(&conn, prefix, "pre");
        let (wt, wt_len) = build_node(&conn, wt_mid, "wt");
        let (alt, alt_len) = build_node(&conn, alt_mid, "alt");
        let (suf, suf_len) = build_node(&conn, suffix, "suf");

        // Block-group graph: wild-type path (ci 0) plus the variant detour (ci 1).
        let e_s_pre = build_edge(&conn, PATH_START_NODE_ID, 0, pre, 0);
        let e_pre_wt = build_edge(&conn, pre, pre_len, wt, 0);
        let e_wt_suf = build_edge(&conn, wt, wt_len, suf, 0);
        let e_suf_end = build_edge(&conn, suf, suf_len, PATH_END_NODE_ID, 0);
        let e_pre_alt = build_edge(&conn, pre, pre_len, alt, 0);
        let e_alt_suf = build_edge(&conn, alt, alt_len, suf, 0);
        BlockGroupEdge::bulk_create(
            &conn,
            &[
                build_block_group_edge(bg.id, e_s_pre, 0),
                build_block_group_edge(bg.id, e_pre_wt, 0),
                build_block_group_edge(bg.id, e_wt_suf, 0),
                build_block_group_edge(bg.id, e_suf_end, 0),
                build_block_group_edge(bg.id, e_pre_alt, 1),
                build_block_group_edge(bg.id, e_alt_suf, 1),
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

    /// Reverse-strand counterpart of `setup_variant_gene`. `prefix`/`wt_mid`/
    /// `alt_mid`/`suffix` are given as the biological CDS pieces, read 5' to 3';
    /// each becomes a node storing the reverse complement of its own piece, laid
    /// out in reverse graph order (the suffix piece nearest PATH_START, the
    /// prefix piece nearest PATH_END), so that translating with strand = Reverse
    /// reconstructs the same CDS, read left to right. The current path follows
    /// the wild-type branch (ci 0). Returns the suffix node's id and length,
    /// which is also the graph's literal entry point (path coordinate 0).
    fn setup_reverse_variant_gene(
        prefix: &str,
        wt_mid: &str,
        alt_mid: &str,
        suffix: &str,
    ) -> (GraphConnection, HashId, HashId, i64) {
        let (conn, bg) = new_block_group("test-gene");
        let (suf, suf_len) = build_node(&conn, &revcomp(suffix), "suf");
        let (wt, wt_len) = build_node(&conn, &revcomp(wt_mid), "wt");
        let (alt, alt_len) = build_node(&conn, &revcomp(alt_mid), "alt");
        let (pre, pre_len) = build_node(&conn, &revcomp(prefix), "pre");

        let e_s_suf = build_edge(&conn, PATH_START_NODE_ID, 0, suf, 0);
        let e_suf_wt = build_edge(&conn, suf, suf_len, wt, 0);
        let e_wt_pre = build_edge(&conn, wt, wt_len, pre, 0);
        let e_pre_end = build_edge(&conn, pre, pre_len, PATH_END_NODE_ID, 0);
        let e_suf_alt = build_edge(&conn, suf, suf_len, alt, 0);
        let e_alt_pre = build_edge(&conn, alt, alt_len, pre, 0);
        BlockGroupEdge::bulk_create(
            &conn,
            &[
                build_block_group_edge(bg.id, e_s_suf, 0),
                build_block_group_edge(bg.id, e_suf_wt, 0),
                build_block_group_edge(bg.id, e_wt_pre, 0),
                build_block_group_edge(bg.id, e_pre_end, 0),
                build_block_group_edge(bg.id, e_suf_alt, 1),
                build_block_group_edge(bg.id, e_alt_pre, 1),
            ],
        );
        Path::create(
            &conn,
            "test-gene",
            &bg.id,
            &[e_s_suf, e_suf_wt, e_wt_pre, e_pre_end],
        )
        .unwrap();

        (conn, bg.id, suf, suf_len)
    }

    /// Two fully-connected columns of two nodes each:
    ///   START → {col1a, col1b} → {col2a, col2b} → END
    /// The stored current path runs START → col1a → col2a → END. There is no
    /// annotation; this fixture exercises whole-block-group translation.
    fn setup_parallel_columns() -> (GraphConnection, HashId) {
        let (conn, bg) = new_block_group("columns");

        // GGT→G, GCT→A, GAT→D, TTT→F (all codon-aligned, no junctions).
        let (c1a, l1a) = build_node(&conn, "GGT", "c1a");
        let (c1b, l1b) = build_node(&conn, "GCT", "c1b");
        let (c2a, l2a) = build_node(&conn, "GAT", "c2a");
        let (c2b, _l2b) = build_node(&conn, "TTT", "c2b");

        let e_s1a = build_edge(&conn, PATH_START_NODE_ID, 0, c1a, 0);
        let e_s1b = build_edge(&conn, PATH_START_NODE_ID, 0, c1b, 0);
        let e_1a2a = build_edge(&conn, c1a, l1a, c2a, 0);
        let e_1a2b = build_edge(&conn, c1a, l1a, c2b, 0);
        let e_1b2a = build_edge(&conn, c1b, l1b, c2a, 0);
        let e_1b2b = build_edge(&conn, c1b, l1b, c2b, 0);
        let e_2a_end = build_edge(&conn, c2a, l2a, PATH_END_NODE_ID, 0);
        let e_2b_end = build_edge(&conn, c2b, l2a, PATH_END_NODE_ID, 0);

        BlockGroupEdge::bulk_create(
            &conn,
            &[
                build_block_group_edge(bg.id, e_s1a, 0),
                build_block_group_edge(bg.id, e_s1b, 1),
                build_block_group_edge(bg.id, e_1a2a, 0),
                build_block_group_edge(bg.id, e_1a2b, 1),
                build_block_group_edge(bg.id, e_1b2a, 1),
                build_block_group_edge(bg.id, e_1b2b, 1),
                build_block_group_edge(bg.id, e_2a_end, 0),
                build_block_group_edge(bg.id, e_2b_end, 1),
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

        let (pre, pre_len) = build_node(&conn, "ATGG", "pre");
        let (rf, rf_len) = build_node(&conn, "A", "ref");
        let (alt, alt_len) = build_node(&conn, "C", "alt");
        let (post, post_len) = build_node(&conn, "ATAA", "post");

        let e_s_pre = build_edge(&conn, PATH_START_NODE_ID, 0, pre, 0);
        let e_pre_ref = build_edge(&conn, pre, pre_len, rf, 0);
        let e_ref_post = build_edge(&conn, rf, rf_len, post, 0);
        let e_post_e = build_edge(&conn, post, post_len, PATH_END_NODE_ID, 0);
        let e_pre_alt = build_edge(&conn, pre, pre_len, alt, 0);
        let e_alt_post = build_edge(&conn, alt, alt_len, post, 0);

        BlockGroupEdge::bulk_create(
            &conn,
            &[
                build_block_group_edge(bg.id, e_s_pre, 0),
                build_block_group_edge(bg.id, e_pre_ref, 0),
                build_block_group_edge(bg.id, e_ref_post, 0),
                build_block_group_edge(bg.id, e_post_e, 0),
                build_block_group_edge(bg.id, e_pre_alt, 1),
                build_block_group_edge(bg.id, e_alt_post, 1),
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

    #[test]
    fn translate_simple_forward() {
        // ATG→M, GAA→E, TGA→* : single node, frame 0
        let (conn, annotation, bg_id) = setup_linear_gene(&["ATGGAATGA"]);
        let params = TranslationParams::new("test");
        let protein = translate_annotation(&conn, &annotation, Some(&bg_id), params).unwrap();
        assert_eq!(protein_full_paths(&conn, &protein.id), vec!["ME*"]);
    }

    #[test]
    fn translate_default_name_is_source_name_protein_suffixed() {
        let (conn, annotation, bg_id) = setup_linear_gene(&["ATGGAATGA"]);
        let params = TranslationParams::new("test");
        let protein = translate_annotation(&conn, &annotation, Some(&bg_id), params).unwrap();
        assert_eq!(protein.name, "test-gene (protein)");
    }

    #[test]
    fn translate_explicit_name_overrides_default() {
        let (conn, annotation, bg_id) = setup_linear_gene(&["ATGGAATGA"]);
        let params = TranslationParams::new("test").name("custom-protein-name");
        let protein = translate_annotation(&conn, &annotation, Some(&bg_id), params).unwrap();
        assert_eq!(protein.name, "custom-protein-name");
    }

    #[test]
    fn translate_duplicate_name_in_sample_errors() {
        let (conn, annotation, bg_id) = setup_linear_gene(&["ATGGAATGA"]);
        let params = TranslationParams::new("test");
        translate_annotation(&conn, &annotation, Some(&bg_id), params).unwrap();

        // Same collection, same sample (it's always the source sample now), same
        // default name: the second translation must error instead of silently
        // merging new protein edges into the first run's sequence graph.
        let params = TranslationParams::new("test");
        let result = translate_annotation(&conn, &annotation, Some(&bg_id), params);
        assert!(matches!(
            result,
            Err(TranslationError::DuplicateBlockGroup(name)) if name == "test-gene (protein)"
        ));
    }

    #[test]
    fn translate_frame1() {
        // initial_frame=1 → head_skip=2, reads "GGAATG" → G,M; tail "A" dropped
        let (conn, annotation, bg_id) = setup_linear_gene(&["ATGGAATGA"]);
        let params = TranslationParams::new("test").initial_frame(1).unwrap();
        let protein = translate_annotation(&conn, &annotation, Some(&bg_id), params).unwrap();
        assert_eq!(protein_full_paths(&conn, &protein.id), vec!["GM"]);
    }

    #[test]
    fn translate_junction_codon() {
        // Node A = "ATGG": ATG→M, tail "G" (1 base)
        // Node B = "AATGA": junction G+"AA"="GAA"→E; B[2..]="TGA"→*
        let (conn, annotation, bg_id) = setup_linear_gene(&["ATGG", "AATGA"]);
        let params = TranslationParams::new("test");
        let protein = translate_annotation(&conn, &annotation, Some(&bg_id), params).unwrap();
        assert_eq!(protein_full_paths(&conn, &protein.id), vec!["ME*"]);
    }

    #[test]
    fn translate_short_node_junction() {
        // A="AT" (2b), B="G" (1b), C="GAATGA" (6b), frame 0
        // Junction ATG=M (B fully consumed); node C: GAA→E, TGA→* → "E*"
        let (conn, annotation, bg_id) = setup_linear_gene(&["AT", "G", "GAATGA"]);
        let params = TranslationParams::new("test");
        let protein = translate_annotation(&conn, &annotation, Some(&bg_id), params).unwrap();
        assert_eq!(protein_full_paths(&conn, &protein.id), vec!["ME*"]);
    }

    #[test]
    fn translate_multi_hop_junction() {
        // A="T" (1b), B="G" (1b), C="GGAATGA" (7b), frame 0
        // Multi-hop junction "T"+"G"+"G" = TGG = W; node C → "E*"
        let (conn, annotation, bg_id) = setup_linear_gene(&["T", "G", "GGAATGA"]);
        let params = TranslationParams::new("test");
        let protein = translate_annotation(&conn, &annotation, Some(&bg_id), params).unwrap();
        assert_eq!(protein_full_paths(&conn, &protein.id), vec!["WE*"]);
    }

    #[test]
    fn translate_table4_tga() {
        // Table 4: TGA → W (Trp), not stop
        let (conn, annotation, bg_id) = setup_linear_gene(&["ATGTGA"]);
        let params = TranslationParams::new("test").codon_table(CodonTable::ncbi(4).unwrap());
        let protein = translate_annotation(&conn, &annotation, Some(&bg_id), params).unwrap();
        assert_eq!(protein_full_paths(&conn, &protein.id), vec!["MW"]);
    }

    #[test]
    fn translate_invalid_frame() {
        let result = TranslationParams::new("test").initial_frame(3);
        assert!(matches!(result, Err(TranslationError::InvalidFrame(3))));
    }

    #[test]
    fn translate_ambiguous_strand_error() {
        let (conn, bg) = new_block_group("test-gene");
        let (n1, l1) = build_node(&conn, "ATGGAATGA", "n1");
        let (n2, l2) = build_node(&conn, "CCCGGG", "n2");

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

        let params = TranslationParams::new("test");
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
        let (node, len) = build_node(&conn, &stored, "rev");
        let e_in = build_edge(&conn, PATH_START_NODE_ID, 0, node, 0);
        let e_out = build_edge(&conn, node, len, PATH_END_NODE_ID, 0);
        BlockGroupEdge::bulk_create(
            &conn,
            &[
                build_block_group_edge(bg.id, e_in, 0),
                build_block_group_edge(bg.id, e_out, 0),
            ],
        );

        // Accession covers the whole node on the reverse strand.
        let annotation =
            accession_annotation(&conn, &bg.id, "rev-gene", &[(node, len)], Strand::Reverse);

        let params = TranslationParams::new("test");
        let protein = translate_annotation(&conn, &annotation, Some(&bg.id), params).unwrap();

        assert_eq!(
            protein_full_paths(&conn, &protein.id),
            vec![expected],
            "reverse-strand protein should read N → C left to right"
        );
    }

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
        let params = TranslationParams::new("test");
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
        let params = TranslationParams::new("test");
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
        let params = TranslationParams::new("test");
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
        let params = TranslationParams::new("test");
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
        let params = TranslationParams::new("test");
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

    /// A junction codon that lands exactly on a stop terminates immediately,
    /// instead of looking for an anchor at the post-stop frame in the next node.
    #[test]
    fn junction_codon_stop_terminates_at_junction() {
        // A = "ATGT": ATG→M, tail "T" (1 byte). B = "AAGG": junction "T"+"AA" = "TAA" = stop.
        let (conn, annotation, bg_id) = setup_linear_gene(&["ATGT", "AAGG"]);
        let params = TranslationParams::new("test");
        let protein = translate_annotation(&conn, &annotation, Some(&bg_id), params).unwrap();
        assert_eq!(protein_full_paths(&conn, &protein.id), vec!["M*"]);
    }

    /// A single-node annotation (the whole annotated span is one node) translates
    /// only the annotation's own node (`wt`); a sibling node (`alt`) that starts
    /// at the same graph position but belongs to a different sample's path is
    /// not part of this walk. Translation continues past the annotation's
    /// declared end into the downstream "suffix" node to its natural stop.
    #[test]
    fn single_node_translates_only_its_own_entry_node() {
        let (conn, parent_bg) = new_block_group("test-gene");
        let (wt, wt_len) = build_node(&conn, "ATG", "wt-first-codon");
        let (alt, alt_len) = build_node(&conn, "CTG", "alt-first-codon");
        let (suffix, suffix_len) = build_node(&conn, "GAATGA", "suffix");

        let start_wt = build_edge(&conn, PATH_START_NODE_ID, 0, wt, 0);
        let wt_suffix = build_edge(&conn, wt, wt_len, suffix, 0);
        let suffix_end = build_edge(&conn, suffix, suffix_len, PATH_END_NODE_ID, 0);
        BlockGroupEdge::bulk_create(
            &conn,
            &[
                build_block_group_edge(parent_bg.id, start_wt, 0),
                build_block_group_edge(parent_bg.id, wt_suffix, 0),
                build_block_group_edge(parent_bg.id, suffix_end, 0),
            ],
        );

        // Annotation covers ONLY the first codon node.
        let annotation = accession_annotation(
            &conn,
            &parent_bg.id,
            "test-gene",
            &[(wt, wt_len)],
            Strand::Forward,
        );

        let child_bg = create_bg(&conn, "test", "child", "test-gene");
        SampleLineage::create(&conn, Sample::DEFAULT_NAME, "child").unwrap();
        let start_alt = build_edge(&conn, PATH_START_NODE_ID, 0, alt, 0);
        let alt_suffix = build_edge(&conn, alt, alt_len, suffix, 0);
        BlockGroupEdge::bulk_create(
            &conn,
            &[
                build_block_group_edge(child_bg.id, start_wt, 0),
                build_block_group_edge(child_bg.id, wt_suffix, 0),
                build_block_group_edge(child_bg.id, suffix_end, 0),
                build_block_group_edge(child_bg.id, start_alt, 1),
                build_block_group_edge(child_bg.id, alt_suffix, 1),
            ],
        );

        let protein = translate_annotation(
            &conn,
            &annotation,
            Some(&child_bg.id),
            TranslationParams::new("test"),
        )
        .unwrap();
        assert_eq!(
            protein_full_paths(&conn, &protein.id),
            vec!["ME*"],
            "translation should only follow the annotation's own entry node"
        );
    }

    /// A gene whose very first codon is a stop codon translates to a single `*`
    /// residue, bounded straight away by the END sentinel.
    #[test]
    fn first_codon_stop_translates_to_lone_stop() {
        // TAA is a stop in the standard table; the rest of the node is never read.
        let (conn, annotation, bg_id) = setup_linear_gene(&["TAAGGGCCC"]);
        let params = TranslationParams::new("test");
        let protein = translate_annotation(&conn, &annotation, Some(&bg_id), params).unwrap();
        assert_eq!(protein_full_paths(&conn, &protein.id), vec!["*"]);
    }

    /// A 1 bp deletion in the middle node shifts the downstream reading frame so the
    /// variant path hits a premature stop and truncates, while the wild-type path
    /// runs to its natural stop.
    ///
    /// DNA layout (frame 0):
    ///   wt  : ATG CCC GTA GGG TAA            → "MPVG*"
    ///   alt : ATG CC. GTAGGGTAA  (1 bp del)
    ///         read in frame: ATG CCG TAG …   → "MP*"  (TAG stop, early)
    #[test]
    fn frameshift_deletion_causes_early_stop() {
        let (conn, annotation, bg_id) = setup_variant_gene("ATG", "CCC", "CC", "GTAGGGTAA");
        let params = TranslationParams::new("test");
        let protein = translate_annotation(&conn, &annotation, Some(&bg_id), params).unwrap();
        assert_eq!(
            protein_full_paths(&conn, &protein.id),
            vec!["MP*", "MPVG*"],
            "frameshift deletion should truncate the variant protein at the premature stop"
        );
    }

    /// A 1 bp deletion that shifts the frame can also read *through* the wild-type
    /// stop and terminate at a later in-frame stop, lengthening the variant protein.
    ///
    /// DNA layout (frame 0):
    ///   wt  : ATG AAA TGA …                  → "MK*"  (stop at codon 3)
    ///   alt : ATG AA. TGACTAAGG  (1 bp del)
    ///         read in frame: ATG AAT GAC TAA → "MND*" (later stop)
    #[test]
    fn frameshift_deletion_causes_late_stop() {
        let (conn, annotation, bg_id) = setup_variant_gene("ATG", "AAA", "AA", "TGACTAAGG");
        let params = TranslationParams::new("test");
        let protein = translate_annotation(&conn, &annotation, Some(&bg_id), params).unwrap();
        assert_eq!(
            protein_full_paths(&conn, &protein.id),
            vec!["MK*", "MND*"],
            "frameshift deletion should read through the wild-type stop to a later one"
        );
    }

    /// One prefix node fanning out to four successors that all complete the same
    /// junction codon position: ATG → M, tail "AA", then junction "AA" + first
    /// base of each successor → AAA/AAG = K, AAC/AAT = N. Each successor's
    /// remaining bytes ("TAA") are themselves an in-frame stop.
    fn setup_junction_fanout() -> (GraphConnection, HashId) {
        let (conn, bg) = new_block_group("fanout");

        let (prefix, prefix_len) = build_node(&conn, "ATGAA", "prefix");
        let (a_node, a_len) = build_node(&conn, "ATAA", "a");
        let (c_node, c_len) = build_node(&conn, "CTAA", "c");
        let (t_node, t_len) = build_node(&conn, "TTAA", "t");
        let (g_node, g_len) = build_node(&conn, "GTAA", "g");

        let e_start = build_edge(&conn, PATH_START_NODE_ID, 0, prefix, 0);
        let e_a = build_edge(&conn, prefix, prefix_len, a_node, 0);
        let e_c = build_edge(&conn, prefix, prefix_len, c_node, 0);
        let e_t = build_edge(&conn, prefix, prefix_len, t_node, 0);
        let e_g = build_edge(&conn, prefix, prefix_len, g_node, 0);
        let e_a_end = build_edge(&conn, a_node, a_len, PATH_END_NODE_ID, 0);
        let e_c_end = build_edge(&conn, c_node, c_len, PATH_END_NODE_ID, 0);
        let e_t_end = build_edge(&conn, t_node, t_len, PATH_END_NODE_ID, 0);
        let e_g_end = build_edge(&conn, g_node, g_len, PATH_END_NODE_ID, 0);

        BlockGroupEdge::bulk_create(
            &conn,
            &[
                build_block_group_edge(bg.id, e_start, 0),
                build_block_group_edge(bg.id, e_a, 0),
                build_block_group_edge(bg.id, e_c, 1),
                build_block_group_edge(bg.id, e_t, 2),
                build_block_group_edge(bg.id, e_g, 3),
                build_block_group_edge(bg.id, e_a_end, 0),
                build_block_group_edge(bg.id, e_c_end, 1),
                build_block_group_edge(bg.id, e_t_end, 2),
                build_block_group_edge(bg.id, e_g_end, 3),
            ],
        );
        Path::create(&conn, "fanout", &bg.id, &[e_start, e_a, e_a_end]).unwrap();
        (conn, bg.id)
    }

    /// Junction codons sharing the same predecessor and amino acid must collapse
    /// to one node: four successors produce only two distinct junction residues
    /// (K and N), so the protein graph must have exactly two junction nodes, not
    /// four.
    #[test]
    fn junction_codons_with_same_amino_acid_deduplicate() {
        let (conn, bg_id) = setup_junction_fanout();
        let params = TranslationParams::new("test");
        let protein = translate_block_group(&conn, &bg_id, params).unwrap();

        assert_eq!(
            count_protein_nodes_with_aa(&conn, &protein.id, "K"),
            1,
            "the two AAA/AAG junctions should collapse to a single K node"
        );
        assert_eq!(
            count_protein_nodes_with_aa(&conn, &protein.id, "N"),
            1,
            "the two AAC/AAT junctions should collapse to a single N node"
        );
    }

    /// Two prefix nodes with different anchor residues (F and P) both converge on
    /// the same successor node and land on the same junction codon: ATG → M, tail
    /// "AA", then junction "AA" + the successor's first base "A" → AAA = K either
    /// way. The successor's remaining bytes ("TAA") are themselves an in-frame
    /// stop.
    fn setup_many_to_one_junction() -> (GraphConnection, HashId) {
        let (conn, bg) = new_block_group("converge");

        let (f_prefix, f_len) = build_node(&conn, "ATGTTTAA", "f-prefix");
        let (p_prefix, p_len) = build_node(&conn, "ATGCCCAA", "p-prefix");
        let (successor, successor_len) = build_node(&conn, "ATAA", "successor");

        let e_start_f = build_edge(&conn, PATH_START_NODE_ID, 0, f_prefix, 0);
        let e_start_p = build_edge(&conn, PATH_START_NODE_ID, 0, p_prefix, 0);
        let e_f = build_edge(&conn, f_prefix, f_len, successor, 0);
        let e_p = build_edge(&conn, p_prefix, p_len, successor, 0);
        let e_end = build_edge(&conn, successor, successor_len, PATH_END_NODE_ID, 0);

        BlockGroupEdge::bulk_create(
            &conn,
            &[
                build_block_group_edge(bg.id, e_start_f, 0),
                build_block_group_edge(bg.id, e_start_p, 1),
                build_block_group_edge(bg.id, e_f, 0),
                build_block_group_edge(bg.id, e_p, 1),
                build_block_group_edge(bg.id, e_end, 0),
            ],
        );
        Path::create(&conn, "converge", &bg.id, &[e_start_f, e_f, e_end]).unwrap();
        (conn, bg.id)
    }

    /// Many-to-one junction convergence: two different predecessor anchors (F and
    /// P) both complete the same junction codon (K) on the same successor node.
    /// An anchor in this position would combine every predecessor's hash into one
    /// node (Step 5's `pred_hashes.join(",")`); a junction instead hashes only its
    /// own single `from`, so each predecessor gets its own junction node here
    /// instead of the one they should share.
    #[test]
    #[ignore = "known limitation: junction nodes hash only their own predecessor, \
                so many-to-one convergence onto the same junction codon does not \
                deduplicate the way anchors do"]
    fn many_to_one_junction_convergence_deduplicates() {
        let (conn, bg_id) = setup_many_to_one_junction();
        let params = TranslationParams::new("test");
        let protein = translate_block_group(&conn, &bg_id, params).unwrap();

        assert_eq!(
            count_protein_nodes_with_aa(&conn, &protein.id, "K"),
            1,
            "both predecessors land on the same junction codon and should share one K node"
        );
    }

    //
    // `translate_annotation` operates per block group: the annotation pins the
    // entry node, and the sequence forward of it is read from the block group's
    // own graph. These tests model a parent and a child sample as two block
    // groups that share the CDS nodes, then translate the same annotation
    // against each to check how an edit in the child propagates to the protein.

    /// A deletion *upstream* of the annotated CDS in a child sample must leave the
    /// translated protein unchanged: edits before the annotation's entry point
    /// are invisible to translation.
    #[test]
    fn upstream_deletion_in_child_keeps_translation_identical() {
        let (conn, parent_bg) = new_block_group("test-gene");

        // CDS node (ATG GAA TGA → M E *) shared by both samples, plus an upstream
        // spacer that exists only in the parent.
        let (upstream, up_len) = build_node(&conn, "GGGGGG", "upstream");
        let (gene, gene_len) = build_node(&conn, "ATGGAATGA", "gene");

        // Parent path: START → upstream → gene → END.
        let p_start_up = build_edge(&conn, PATH_START_NODE_ID, 0, upstream, 0);
        let p_up_gene = build_edge(&conn, upstream, up_len, gene, 0);
        let gene_end = build_edge(&conn, gene, gene_len, PATH_END_NODE_ID, 0);
        BlockGroupEdge::bulk_create(
            &conn,
            &[
                build_block_group_edge(parent_bg.id, p_start_up, 0),
                build_block_group_edge(parent_bg.id, p_up_gene, 0),
                build_block_group_edge(parent_bg.id, gene_end, 0),
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
        let c_start_gene = build_edge(&conn, PATH_START_NODE_ID, 0, gene, 0);
        BlockGroupEdge::bulk_create(
            &conn,
            &[
                build_block_group_edge(child_bg.id, c_start_gene, 0),
                build_block_group_edge(child_bg.id, gene_end, 0),
            ],
        );

        let parent_protein = translate_annotation(
            &conn,
            &annotation,
            Some(&parent_bg.id),
            TranslationParams::new("test"),
        )
        .unwrap();
        let child_protein = translate_annotation(
            &conn,
            &annotation,
            Some(&child_bg.id),
            TranslationParams::new("test"),
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
        let (start_codon, start_len) = build_node(&conn, "ATG", "start"); // M
        let (wt_mid, wt_len) = build_node(&conn, "GAA", "wt"); // E
        let (alt_mid, alt_len) = build_node(&conn, "CAA", "alt"); // Q
        let (stop_codon, stop_len) = build_node(&conn, "TGA", "stop"); // *

        // Parent path: START → ATG → GAA → TGA → END.
        let start_in = build_edge(&conn, PATH_START_NODE_ID, 0, start_codon, 0);
        let p_start_wt = build_edge(&conn, start_codon, start_len, wt_mid, 0);
        let p_wt_stop = build_edge(&conn, wt_mid, wt_len, stop_codon, 0);
        let stop_out = build_edge(&conn, stop_codon, stop_len, PATH_END_NODE_ID, 0);
        BlockGroupEdge::bulk_create(
            &conn,
            &[
                build_block_group_edge(parent_bg.id, start_in, 0),
                build_block_group_edge(parent_bg.id, p_start_wt, 0),
                build_block_group_edge(parent_bg.id, p_wt_stop, 0),
                build_block_group_edge(parent_bg.id, stop_out, 0),
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
        let c_start_alt = build_edge(&conn, start_codon, start_len, alt_mid, 0);
        let c_alt_stop = build_edge(&conn, alt_mid, alt_len, stop_codon, 0);
        BlockGroupEdge::bulk_create(
            &conn,
            &[
                build_block_group_edge(child_bg.id, start_in, 0),
                build_block_group_edge(child_bg.id, c_start_alt, 0),
                build_block_group_edge(child_bg.id, c_alt_stop, 0),
                build_block_group_edge(child_bg.id, stop_out, 0),
            ],
        );

        let parent_protein = translate_annotation(
            &conn,
            &annotation,
            Some(&parent_bg.id),
            TranslationParams::new("test"),
        )
        .unwrap();
        let child_protein = translate_annotation(
            &conn,
            &annotation,
            Some(&child_bg.id),
            TranslationParams::new("test"),
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

    /// A SNP at the *first* base of the annotated CDS translates only the
    /// annotation's own entry node (`first_wt`) and does not surface the
    /// variant residue (CTG → L) carried by the alternate node (`first_alt`)
    /// at that position.
    #[test]
    fn entry_boundary_snp_does_not_surface_alternate_allele() {
        let (conn, parent_bg) = new_block_group("test-gene");

        // First CDS base is its own node so a SNP there forms a bubble at entry.
        let (first_wt, _) = build_node(&conn, "A", "first-wt"); // ATG... → M
        let (first_alt, _) = build_node(&conn, "C", "first-alt"); // CTG... → L
        let (rest, rest_len) = build_node(&conn, "TGGAATGA", "rest"); // shared remainder

        // Parent: START → A → TGGAATGA → END  (ATG GAA TGA = M E *).
        let start_first = build_edge(&conn, PATH_START_NODE_ID, 0, first_wt, 0);
        let first_rest = build_edge(&conn, first_wt, 1, rest, 0);
        let rest_end = build_edge(&conn, rest, rest_len, PATH_END_NODE_ID, 0);
        BlockGroupEdge::bulk_create(
            &conn,
            &[
                build_block_group_edge(parent_bg.id, start_first, 0),
                build_block_group_edge(parent_bg.id, first_rest, 0),
                build_block_group_edge(parent_bg.id, rest_end, 0),
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
        let start_alt = build_edge(&conn, PATH_START_NODE_ID, 0, first_alt, 0);
        let alt_rest = build_edge(&conn, first_alt, 1, rest, 0);
        BlockGroupEdge::bulk_create(
            &conn,
            &[
                build_block_group_edge(child_bg.id, start_first, 0),
                build_block_group_edge(child_bg.id, first_rest, 0),
                build_block_group_edge(child_bg.id, start_alt, 1),
                build_block_group_edge(child_bg.id, alt_rest, 1),
                build_block_group_edge(child_bg.id, rest_end, 0),
            ],
        );

        let protein = translate_annotation(
            &conn,
            &annotation,
            Some(&child_bg.id),
            TranslationParams::new("test"),
        )
        .unwrap();

        assert_eq!(
            protein_full_paths(&conn, &protein.id),
            vec!["ME*"],
            "translation should only follow the annotation's own entry node"
        );
    }

    /// A SNP at the *last* base of the annotated CDS should surface the variant
    /// residue (TGG → W) alongside the wild-type stop (TGA → *).
    #[test]
    fn boundary_last_base_mutation_in_child_is_captured() {
        let (conn, parent_bg) = new_block_group("test-gene");

        // Last CDS base is its own node so a SNP there forms a bubble at exit.
        let (head, head_len) = build_node(&conn, "ATGGAATG", "head"); // ATG GAA TG_
        let (last_wt, _) = build_node(&conn, "A", "last-wt"); // …TGA → *
        let (last_alt, _) = build_node(&conn, "G", "last-alt"); // …TGG → W

        // Parent: START → ATGGAATG → A → END  (ATG GAA TGA = M E *).
        let start_head = build_edge(&conn, PATH_START_NODE_ID, 0, head, 0);
        let head_last = build_edge(&conn, head, head_len, last_wt, 0);
        let last_end = build_edge(&conn, last_wt, 1, PATH_END_NODE_ID, 0);
        BlockGroupEdge::bulk_create(
            &conn,
            &[
                build_block_group_edge(parent_bg.id, start_head, 0),
                build_block_group_edge(parent_bg.id, head_last, 0),
                build_block_group_edge(parent_bg.id, last_end, 0),
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
        let head_alt = build_edge(&conn, head, head_len, last_alt, 0);
        let alt_end = build_edge(&conn, last_alt, 1, PATH_END_NODE_ID, 0);
        BlockGroupEdge::bulk_create(
            &conn,
            &[
                build_block_group_edge(child_bg.id, start_head, 0),
                build_block_group_edge(child_bg.id, head_last, 0),
                build_block_group_edge(child_bg.id, last_end, 0),
                build_block_group_edge(child_bg.id, head_alt, 1),
                build_block_group_edge(child_bg.id, alt_end, 1),
            ],
        );

        let protein = translate_annotation(
            &conn,
            &annotation,
            Some(&child_bg.id),
            TranslationParams::new("test"),
        )
        .unwrap();

        assert_eq!(
            protein_full_paths(&conn, &protein.id),
            vec!["ME*", "MEW"],
            "last-base variant was dropped from the extracted subgraph",
        );
    }

    /// translate_from_path translates only the literal node at the given
    /// coordinate, even at coordinate 0: a parallel branch from PATH_START (D
    /// below) that does not pass through the entry node is not part of the
    /// walk. translate_block_group, which starts from every direct successor
    /// of PATH_START, produces more paths here — the two are intentionally
    /// different operations.
    ///
    ///   START -> A (ci0, current path)         A = "TAA"      (stop alone)
    ///   START -> D (ci1)                        D = "G"        (1 byte)
    ///   D -> P (ci1)                             P = "C"        (1 byte)
    ///   P -> A (ci1)                             (rejoins entry node A)
    ///   P -> Z (ci2)                             Z = "TTTTAA"  (F* in frame 0)
    ///   A -> END (ci0), Z -> END (ci2)
    #[test]
    fn translate_from_path_zero_follows_only_the_entry_node() {
        let (conn, bg) = new_block_group("probe");
        let (a, a_len) = build_node(&conn, "TAA", "a");
        let (d, d_len) = build_node(&conn, "G", "d");
        let (p, p_len) = build_node(&conn, "C", "p");
        let (z, z_len) = build_node(&conn, "TTTTAA", "z");

        let e_start_a = build_edge(&conn, PATH_START_NODE_ID, 0, a, 0);
        let e_start_d = build_edge(&conn, PATH_START_NODE_ID, 0, d, 0);
        let e_d_p = build_edge(&conn, d, d_len, p, 0);
        let e_p_a = build_edge(&conn, p, p_len, a, 0);
        let e_p_z = build_edge(&conn, p, p_len, z, 0);
        let e_a_end = build_edge(&conn, a, a_len, PATH_END_NODE_ID, 0);
        let e_z_end = build_edge(&conn, z, z_len, PATH_END_NODE_ID, 0);

        BlockGroupEdge::bulk_create(
            &conn,
            &[
                build_block_group_edge(bg.id, e_start_a, 0),
                build_block_group_edge(bg.id, e_start_d, 1),
                build_block_group_edge(bg.id, e_d_p, 1),
                build_block_group_edge(bg.id, e_p_a, 1),
                build_block_group_edge(bg.id, e_p_z, 2),
                build_block_group_edge(bg.id, e_a_end, 0),
                build_block_group_edge(bg.id, e_z_end, 2),
            ],
        );
        Path::create(&conn, "probe", &bg.id, &[e_start_a, e_a_end]).unwrap();

        let from_path =
            translate_from_path(&conn, &bg.id, 0, TranslationParams::new("test")).unwrap();

        assert_eq!(
            protein_full_paths(&conn, &from_path.id),
            vec!["*"],
            "translate_from_path(0) should only translate the literal entry node (A), \
             not the unrelated D/P/Z branch reachable from PATH_START"
        );
    }

    // Extra flanking codons so codon-order bugs can't hide behind a coincidental match.
    const REV_PREFIX: &str = "ATGAAACCC"; // M K P
    const REV_SUFFIX: &str = "GGGTTTTGA"; // G F *

    /// Whole-block-group translation on the reverse strand reads every node's
    /// reverse complement and walks the graph back to front, reconstructing the
    /// CDS and both branches of the variant bubble at the middle codon.
    #[test]
    fn reverse_strand_block_group_translation_handles_variant() {
        let (conn, bg_id, _, _) = setup_reverse_variant_gene(REV_PREFIX, "GAA", "CAA", REV_SUFFIX);
        let params = TranslationParams::new("test").strand(Strand::Reverse);
        let protein = translate_block_group(&conn, &bg_id, params).unwrap();

        assert_eq!(
            protein_full_paths(&conn, &protein.id),
            vec!["MKPEGF*", "MKPQGF*"]
        );
    }

    /// Annotation translation on the reverse strand reads every node's reverse
    /// complement and walks the graph back to front from the entry segment,
    /// reconstructing the CDS and both branches of the variant bubble. Strand is
    /// inferred from the accession's own segment, not passed explicitly.
    #[test]
    fn reverse_strand_annotation_translation_handles_variant() {
        let (conn, bg_id, suf, suf_len) =
            setup_reverse_variant_gene(REV_PREFIX, "GAA", "CAA", REV_SUFFIX);
        let annotation = accession_annotation(
            &conn,
            &bg_id,
            "rev-gene",
            &[(suf, suf_len)],
            Strand::Reverse,
        );

        let params = TranslationParams::new("test");
        let protein = translate_annotation(&conn, &annotation, Some(&bg_id), params).unwrap();

        assert_eq!(
            protein_full_paths(&conn, &protein.id),
            vec!["MKPEGF*", "MKPQGF*"]
        );
    }

    /// Path-coordinate translation on the reverse strand reads every node's
    /// reverse complement and walks the graph back to front from the entry
    /// coordinate, reconstructing the CDS and both branches of the variant
    /// bubble.
    #[test]
    fn reverse_strand_path_translation_handles_variant() {
        let (conn, bg_id, _, _) = setup_reverse_variant_gene(REV_PREFIX, "GAA", "CAA", REV_SUFFIX);
        let params = TranslationParams::new("test").strand(Strand::Reverse);
        let protein = translate_from_path(&conn, &bg_id, 0, params).unwrap();

        assert_eq!(
            protein_full_paths(&conn, &protein.id),
            vec!["MKPEGF*", "MKPQGF*"]
        );
    }
}
