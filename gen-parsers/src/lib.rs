use std::collections::HashMap;

use gen_core::{HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand};
use gen_graph::{GenGraph, GraphEdge, GraphNode};
use thiserror::Error;

pub mod blast;
pub mod clustal;
pub mod maf;
pub mod paf;
pub mod psl;

pub use blast::BlastParser;
pub use clustal::ClustalwParser;
pub use maf::MafParser;
pub use paf::PafParser;
pub use psl::PslParser;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ParseError {
    #[error("failed to read alignment input: {0}")]
    Read(String),
    #[error("CLUSTAL alignment must start with CLUSTAL W or CLUSTALW")]
    MissingHeader,
    #[error("CLUSTAL alignment does not contain sequence rows")]
    NoSequences,
    #[error("alignment row {line_number} is missing a sequence fragment")]
    MissingSequenceFragment { line_number: usize },
    #[error("BLAST HSP is missing a Sbjct sequence row")]
    MissingBlastSubject,
    #[error("BLAST output does not contain pairwise HSP alignments")]
    NoBlastAlignments,
    #[error("PAF line {line_number} has {actual} fields but expected at least 12")]
    PafTooFewFields { line_number: usize, actual: usize },
    #[error("PAF line {line_number} is missing required cg:Z CIGAR tag")]
    MissingPafCigar { line_number: usize },
    #[error("PAF line {line_number} has invalid integer in field {field}")]
    InvalidPafInteger { line_number: usize, field: usize },
    #[error("PAF line {line_number} has invalid strand {strand}")]
    InvalidPafStrand { line_number: usize, strand: String },
    #[error("PAF line {line_number} has invalid CIGAR string {cigar}")]
    InvalidCigar { line_number: usize, cigar: String },
    #[error("PSL line {line_number} has {actual} fields but expected at least 21")]
    PslTooFewFields { line_number: usize, actual: usize },
    #[error("PSL line {line_number} has invalid integer in field {field}")]
    InvalidPslInteger { line_number: usize, field: usize },
    #[error("PSL line {line_number} has invalid strand {strand}")]
    InvalidPslStrand { line_number: usize, strand: String },
    #[error(
        "PSL line {line_number} blockCount is {block_count} but blockSizes, qStarts, and tStarts contain {block_sizes}, {query_starts}, and {target_starts} entries"
    )]
    MismatchedPslBlocks {
        line_number: usize,
        block_count: usize,
        block_sizes: usize,
        query_starts: usize,
        target_starts: usize,
    },
    #[error("MAF block ending at line {line_number} does not contain sequence rows")]
    MafBlockWithoutSequences { line_number: usize },
    #[error("MAF line {line_number} has {actual} fields but expected at least 7")]
    MafTooFewFields { line_number: usize, actual: usize },
    #[error("MAF line {line_number} has invalid integer in field {field}")]
    InvalidMafInteger { line_number: usize, field: usize },
    #[error("MAF line {line_number} has invalid strand {strand}")]
    InvalidMafStrand { line_number: usize, strand: String },
    #[error(
        "MAF line {line_number} sequence {name} declares size {declared} but has {actual} non-gap bases"
    )]
    MafSizeMismatch {
        line_number: usize,
        name: String,
        declared: i64,
        actual: usize,
    },
    #[error("aligned sequence {name} has length {actual} but expected {expected}")]
    MismatchedAlignedLength {
        name: String,
        actual: usize,
        expected: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CigarOp {
    Match(i64),
    Insertion(i64),
    Deletion(i64),
    ReferenceSkip(i64),
    SoftClip(i64),
    HardClip(i64),
    Equal(i64),
    Difference(i64),
}

#[derive(Debug)]
pub struct ParsedAlignment {
    pub base_name: String,
    pub sequence_order: Vec<String>,
    pub aligned_sequences: HashMap<String, String>,
    pub graph: GenGraph,
}

impl ParsedAlignment {
    pub fn ungapped_sequence(&self, name: &str) -> Option<String> {
        self.aligned_sequences
            .get(name)
            .map(|sequence| sequence.chars().filter(|base| *base != '-').collect())
    }
}

#[derive(Debug)]
pub struct ParsedMapping {
    pub query_name: String,
    pub query_len: i64,
    pub query_start: i64,
    pub query_end: i64,
    pub strand: Strand,
    pub target_name: String,
    pub target_len: i64,
    pub target_start: i64,
    pub target_end: i64,
    pub matching_bases: i64,
    pub block_len: i64,
    pub mapping_quality: i64,
    pub cigar: Vec<CigarOp>,
    pub tags: HashMap<String, String>,
    pub graph: GenGraph,
}

pub(crate) fn validate_lengths(
    sequence_order: &[String],
    aligned_sequences: &HashMap<String, String>,
) -> Result<(), ParseError> {
    let expected = aligned_sequences
        .get(&sequence_order[0])
        .expect("should have base sequence")
        .len();

    for name in sequence_order.iter().skip(1) {
        let actual = aligned_sequences
            .get(name)
            .expect("should have sequence")
            .len();
        if actual != expected {
            return Err(ParseError::MismatchedAlignedLength {
                name: name.clone(),
                actual,
                expected,
            });
        }
    }

    Ok(())
}

pub(crate) fn build_graph(
    base_name: &str,
    sequence_order: &[String],
    aligned_sequences: &HashMap<String, String>,
) -> GenGraph {
    let base_aligned = aligned_sequences
        .get(base_name)
        .expect("should have base sequence");
    let base_sequence = ungapped(base_aligned);
    let base_node_id = HashId::convert_str(&format!("clustalw:base:{base_name}:{base_sequence}"));
    let runs = alignment_runs(base_aligned, sequence_order, aligned_sequences);

    let mut graph = GenGraph::new();
    let mut previous_by_sequence = vec![terminal_node(PATH_START_NODE_ID); sequence_order.len()];
    let end_node = terminal_node(PATH_END_NODE_ID);
    let mut base_offset = 0_i64;

    for run in runs {
        let base_text = ungapped(&base_aligned[run.start..run.end]);
        let base_node = (!base_text.is_empty()).then(|| {
            let node = GraphNode {
                node_id: base_node_id,
                sequence_start: base_offset,
                sequence_end: base_offset + base_text.len() as i64,
            };
            base_offset += base_text.len() as i64;
            node
        });

        for (sequence_index, name) in sequence_order.iter().enumerate() {
            let aligned = aligned_sequences.get(name).expect("should have sequence");
            let text = ungapped(&aligned[run.start..run.end]);
            let next = if run.kind == RunKind::Invariant || sequence_index == 0 {
                base_node
            } else if text.is_empty() {
                None
            } else if text == base_text {
                base_node
            } else {
                Some(GraphNode {
                    node_id: HashId::convert_str(&format!(
                        "clustalw:variant:{name}:{}:{}:{text}",
                        run.start, run.end
                    )),
                    sequence_start: 0,
                    sequence_end: text.len() as i64,
                })
            };

            if let Some(next_node) = next {
                add_path_edge(
                    &mut graph,
                    previous_by_sequence[sequence_index],
                    next_node,
                    sequence_index as i64,
                );
                previous_by_sequence[sequence_index] = next_node;
            }
        }
    }

    for (sequence_index, previous) in previous_by_sequence.iter().enumerate() {
        add_path_edge(&mut graph, *previous, end_node, sequence_index as i64);
    }

    graph
}

fn alignment_runs(
    base_aligned: &str,
    sequence_order: &[String],
    aligned_sequences: &HashMap<String, String>,
) -> Vec<AlignmentRun> {
    let mut runs = Vec::new();
    let mut start = 0;
    let mut current_kind = None;

    for column in 0..base_aligned.len() {
        let kind = column_kind(column, base_aligned, sequence_order, aligned_sequences);
        if current_kind.is_some_and(|current| current != kind) {
            runs.push(AlignmentRun {
                start,
                end: column,
                kind: current_kind.expect("should have run kind"),
            });
            start = column;
        }
        current_kind = Some(kind);
    }

    if let Some(kind) = current_kind {
        runs.push(AlignmentRun {
            start,
            end: base_aligned.len(),
            kind,
        });
    }

    runs
}

fn column_kind(
    column: usize,
    base_aligned: &str,
    sequence_order: &[String],
    aligned_sequences: &HashMap<String, String>,
) -> RunKind {
    let base = base_aligned.as_bytes()[column];
    if base == b'-' {
        return RunKind::Variant;
    }

    let invariant = sequence_order.iter().all(|name| {
        aligned_sequences
            .get(name)
            .expect("should have sequence")
            .as_bytes()[column]
            == base
    });

    if invariant {
        RunKind::Invariant
    } else {
        RunKind::Variant
    }
}

pub(crate) fn add_path_edge(
    graph: &mut GenGraph,
    source: GraphNode,
    target: GraphNode,
    sequence_index: i64,
) {
    let edge = GraphEdge {
        edge_id: HashId::convert_str(&format!(
            "clustalw:edge:{sequence_index}:{}:{}:{}:{}",
            source.node_id, source.sequence_start, target.node_id, target.sequence_start
        )),
        source_strand: Strand::Forward,
        target_strand: Strand::Forward,
        chromosome_index: sequence_index,
        phased: 0,
        created_on: 0,
    };

    if let Some(edges) = graph.edge_weight_mut(source, target) {
        edges.push(edge);
    } else {
        graph.add_edge(source, target, vec![edge]);
    }
}

pub(crate) fn terminal_node(node_id: HashId) -> GraphNode {
    GraphNode {
        node_id,
        sequence_start: 0,
        sequence_end: 0,
    }
}

pub(crate) fn ungapped(sequence: &str) -> String {
    sequence
        .chars()
        .filter(|character| *character != '-')
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AlignmentRun {
    start: usize,
    end: usize,
    kind: RunKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RunKind {
    Invariant,
    Variant,
}
