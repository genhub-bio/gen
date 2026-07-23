use std::collections::HashMap;

use gen_core::{HashId, Strand};
use gen_graph::{GenGraph, GraphNode, GraphNodeSlice};
use gen_models::locus::GraphLocus;
use petgraph::visit::IntoNodeIdentifiers;

#[derive(Clone, Debug)]
pub struct AnnotationSegment {
    pub node_id: HashId,
    pub start: i64,
    pub end: i64,
    pub strand: Strand,
}

#[derive(Clone, Debug)]
pub struct AnnotationSpan {
    pub id: HashId,
    pub name: String,
    pub segments: Vec<AnnotationSegment>,
}

#[derive(Clone, Debug)]
pub struct AnnotationTrack {
    pub name: String,
    pub annotations: Vec<AnnotationSpan>,
}

impl AnnotationTrack {
    pub fn new(name: impl Into<String>, annotations: Vec<AnnotationSpan>) -> Self {
        AnnotationTrack {
            name: name.into(),
            annotations,
        }
    }
}

pub fn annotation_span_from_graph_locus(locus: &GraphLocus, name: &str) -> AnnotationSpan {
    let segments = locus
        .slices
        .iter()
        .map(|s| AnnotationSegment {
            node_id: s.block.node_id,
            start: s.block.sequence_start + s.start as i64,
            end: s.block.sequence_start + s.end as i64,
            strand: s.strand,
        })
        .collect();
    AnnotationSpan {
        id: HashId::convert_str(name),
        name: name.to_string(),
        segments,
    }
}

/// Map `span`'s per-node segments onto the current graph.
pub fn graph_locus_from_annotation_span(
    span: &AnnotationSpan,
    graph: &GenGraph,
) -> Option<GraphLocus> {
    if span.segments.is_empty() {
        return None;
    }
    let mut node_map: HashMap<HashId, Vec<GraphNode>> = HashMap::new();
    for node in graph.node_identifiers() {
        node_map.entry(node.node_id).or_default().push(node);
    }
    let slices: Option<Vec<GraphNodeSlice>> = span
        .segments
        .iter()
        .map(|seg| {
            let candidates = node_map.get(&seg.node_id)?;
            let block = *candidates
                .iter()
                .find(|block| seg.start < block.sequence_end && block.sequence_start < seg.end)
                .or_else(|| candidates.first())?;
            let start = (seg.start - block.sequence_start).max(0) as usize;
            let end = (seg.end - block.sequence_start).max(0) as usize;
            Some(GraphNodeSlice {
                block,
                start,
                end,
                strand: seg.strand,
            })
        })
        .collect();
    Some(GraphLocus { slices: slices? })
}

/// Compute the display label for a span, appending a strand arrow when all
/// segments share a single non-ambiguous strand.
pub fn span_label_text(span: &AnnotationSpan) -> String {
    let strand = match span.segments.first() {
        Some(seg)
            if !Strand::is_ambiguous(seg.strand)
                && span.segments.iter().all(|s| s.strand == seg.strand) =>
        {
            Some(seg.strand)
        }
        _ => None,
    };
    match strand {
        Some(Strand::Forward) => format!("{}›", span.name),
        Some(Strand::Reverse) => format!("‹{}", span.name),
        _ => span.name.clone(),
    }
}

/// Return `true` if every segment of `span` resolves to the same `GraphNode`
/// fragment in `graph`, i.e. the annotation does not cross a node boundary.
pub fn span_is_single_node(span: &AnnotationSpan, graph: &GenGraph) -> bool {
    let Some(locus) = graph_locus_from_annotation_span(span, graph) else {
        return true;
    };
    match locus.slices.first() {
        Some(first) => locus.slices.iter().all(|slice| slice.block == first.block),
        None => true,
    }
}

/// Return `true` if the annotation `span` should be dropped from the inline
/// overlay in `Truncated` detail level.
///
/// An annotation that crosses a node boundary, or that covers the full width
/// of the single node it lies on, is kept. This way you avoid pileups of many
/// small annotations that lie within the truncated sequence, but still show
/// the annotations that get interrupted by variants since those are relevant.
pub fn span_should_hide_in_truncated(span: &AnnotationSpan, graph: &GenGraph) -> bool {
    let Some(locus) = graph_locus_from_annotation_span(span, graph) else {
        return false;
    };
    let Some(first) = locus.slices.first() else {
        return false;
    };
    if !locus.slices.iter().all(|slice| slice.block == first.block) {
        return false;
    }
    !(first.start == 0 && first.end as i64 >= first.block.length())
}

/// Return `true` if every segment of `span` at `idx` is fully contained within
/// at least one segment from a later span (higher index = shorter = painted on top).
/// Used to count annotations that are completely obscured by other highlights.
pub fn span_covered_by_later(
    span: &AnnotationSpan,
    idx: usize,
    all_spans: &[&AnnotationSpan],
) -> bool {
    if span.segments.is_empty() {
        return false;
    }
    'outer: for seg in &span.segments {
        for later_span in &all_spans[idx + 1..] {
            for other in &later_span.segments {
                if other.node_id == seg.node_id && other.start <= seg.start && other.end >= seg.end
                {
                    continue 'outer;
                }
            }
        }
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use gen_graph::{GenGraph, GraphNode};

    use super::*;

    fn make_node(node_id: &str, seq_start: i64, seq_end: i64) -> GraphNode {
        GraphNode {
            node_id: HashId::convert_str(node_id),
            sequence_start: seq_start,
            sequence_end: seq_end,
        }
    }

    fn make_graph(nodes: &[GraphNode]) -> GenGraph {
        let mut g = GenGraph::new();
        for &n in nodes {
            g.add_node(n);
        }
        g
    }

    #[test]
    fn annotation_span_from_graph_locus_preserves_name_and_coordinates() {
        let node = make_node("n1", 100, 200);
        let locus = GraphLocus {
            slices: vec![GraphNodeSlice {
                block: node,
                start: 5,
                end: 15,
                strand: Strand::Forward,
            }],
        };
        let span = annotation_span_from_graph_locus(&locus, "my_gene");
        assert_eq!(span.name, "my_gene");
        assert_eq!(span.segments.len(), 1);
        let seg = &span.segments[0];
        assert_eq!(seg.node_id, node.node_id);
        assert_eq!(seg.start, 105); // sequence_start + slice.start
        assert_eq!(seg.end, 115); // sequence_start + slice.end
        assert_eq!(seg.strand, Strand::Forward);
    }

    #[test]
    fn graph_locus_from_annotation_span_inverts_to_annotation_span() {
        let node = make_node("n1", 100, 200);
        let graph = make_graph(&[node]);
        let locus = GraphLocus {
            slices: vec![GraphNodeSlice {
                block: node,
                start: 5,
                end: 15,
                strand: Strand::Forward,
            }],
        };
        let span = annotation_span_from_graph_locus(&locus, "");
        let recovered = graph_locus_from_annotation_span(&span, &graph).unwrap();
        assert_eq!(recovered.slices.len(), 1);
        assert_eq!(recovered.slices[0].block, node);
        assert_eq!(recovered.slices[0].start, 5);
        assert_eq!(recovered.slices[0].end, 15);
        assert_eq!(recovered.slices[0].strand, Strand::Forward);
    }

    #[test]
    fn graph_locus_from_annotation_span_returns_none_for_empty_span() {
        let graph = make_graph(&[]);
        let span = AnnotationSpan {
            id: HashId::convert_str("x"),
            name: "x".into(),
            segments: vec![],
        };
        assert!(graph_locus_from_annotation_span(&span, &graph).is_none());
    }

    #[test]
    fn graph_locus_from_annotation_span_returns_none_when_node_missing_from_graph() {
        let node = make_node("n1", 0, 100);
        let graph = make_graph(&[]); // node not in graph
        let locus = GraphLocus {
            slices: vec![GraphNodeSlice {
                block: node,
                start: 0,
                end: 10,
                strand: Strand::Forward,
            }],
        };
        let span = annotation_span_from_graph_locus(&locus, "");
        assert!(graph_locus_from_annotation_span(&span, &graph).is_none());
    }

    /// A node that has been split by a later edit (e.g. a library insertion) shows up as
    /// several disjoint `GraphNode`s in the current graph, all sharing the same `node_id`.
    /// A span with one segment per surviving fragment must resolve each segment to the
    /// specific fragment its range overlaps, not to whichever fragment a naive
    /// `node_id`-keyed lookup happens to keep.
    #[test]
    fn graph_locus_from_annotation_span_resolves_each_segment_to_its_own_fragment() {
        let node_id = HashId::convert_str("split-node");
        let left_fragment = GraphNode {
            node_id,
            sequence_start: 0,
            sequence_end: 395,
        };
        let right_fragment = GraphNode {
            node_id,
            sequence_start: 483,
            sequence_end: 2686,
        };
        // Graph iteration order shouldn't matter; put the right fragment first so a
        // naive `HashMap<node_id, GraphNode>` collect would keep it over the left one.
        let graph = make_graph(&[right_fragment, left_fragment]);

        let span = AnnotationSpan {
            id: HashId::convert_str("source"),
            name: "source".into(),
            segments: vec![
                make_segment("split-node", 0, 395, Strand::Forward),
                make_segment("split-node", 483, 2686, Strand::Forward),
            ],
        };

        let locus = graph_locus_from_annotation_span(&span, &graph).unwrap();
        assert_eq!(locus.slices.len(), 2);
        assert_eq!(locus.slices[0].block, left_fragment);
        assert_eq!(locus.slices[0].start, 0);
        assert_eq!(locus.slices[0].end, 395);
        assert_eq!(locus.slices[1].block, right_fragment);
        assert_eq!(locus.slices[1].start, 0);
        assert_eq!(locus.slices[1].end, 2203); // 2686 - 483, local to the right fragment
    }

    fn make_segment(node_id: &str, start: i64, end: i64, strand: Strand) -> AnnotationSegment {
        AnnotationSegment {
            node_id: HashId::convert_str(node_id),
            start,
            end,
            strand,
        }
    }

    #[test]
    fn span_label_text_appends_forward_arrow() {
        let span = AnnotationSpan {
            id: HashId::convert_str("x"),
            name: "my_gene".into(),
            segments: vec![make_segment("n1", 0, 10, Strand::Forward)],
        };
        assert_eq!(span_label_text(&span), "my_gene›");
    }

    #[test]
    fn span_label_text_prepends_reverse_arrow() {
        let span = AnnotationSpan {
            id: HashId::convert_str("x"),
            name: "my_gene".into(),
            segments: vec![make_segment("n1", 0, 10, Strand::Reverse)],
        };
        assert_eq!(span_label_text(&span), "‹my_gene");
    }

    #[test]
    fn span_label_text_omits_arrow_when_segments_disagree_on_strand() {
        let span = AnnotationSpan {
            id: HashId::convert_str("x"),
            name: "my_gene".into(),
            segments: vec![
                make_segment("n1", 0, 10, Strand::Forward),
                make_segment("n2", 0, 10, Strand::Reverse),
            ],
        };
        assert_eq!(span_label_text(&span), "my_gene");
    }

    #[test]
    fn span_label_text_omits_arrow_for_empty_segments() {
        let span = AnnotationSpan {
            id: HashId::convert_str("x"),
            name: "my_gene".into(),
            segments: vec![],
        };
        assert_eq!(span_label_text(&span), "my_gene");
    }

    #[test]
    fn span_is_single_node_true_for_one_segment() {
        let node = make_node("n1", 0, 10);
        let graph = make_graph(&[node]);
        let span = AnnotationSpan {
            id: HashId::convert_str("x"),
            name: "x".into(),
            segments: vec![make_segment("n1", 0, 10, Strand::Forward)],
        };
        assert!(span_is_single_node(&span, &graph));
    }

    #[test]
    fn span_is_single_node_true_when_all_segments_share_node() {
        let node = make_node("n1", 0, 20);
        let graph = make_graph(&[node]);
        let span = AnnotationSpan {
            id: HashId::convert_str("x"),
            name: "x".into(),
            segments: vec![
                make_segment("n1", 0, 10, Strand::Forward),
                make_segment("n1", 10, 20, Strand::Forward),
            ],
        };
        assert!(span_is_single_node(&span, &graph));
    }

    #[test]
    fn span_is_single_node_false_when_segments_span_multiple_nodes() {
        let node_1 = make_node("n1", 0, 10);
        let node_2 = make_node("n2", 0, 10);
        let graph = make_graph(&[node_1, node_2]);
        let span = AnnotationSpan {
            id: HashId::convert_str("x"),
            name: "x".into(),
            segments: vec![
                make_segment("n1", 0, 10, Strand::Forward),
                make_segment("n2", 0, 10, Strand::Forward),
            ],
        };
        assert!(!span_is_single_node(&span, &graph));
    }

    #[test]
    fn span_is_single_node_true_for_empty_segments() {
        let graph = make_graph(&[]);
        let span = AnnotationSpan {
            id: HashId::convert_str("x"),
            name: "x".into(),
            segments: vec![],
        };
        assert!(span_is_single_node(&span, &graph));
    }

    /// Segments on different fragments of a split node must not count as single-node,
    /// even though they share a `node_id`.
    #[test]
    fn span_is_single_node_false_for_split_fragments() {
        let node_id = HashId::convert_str("split-node");
        let left_fragment = GraphNode {
            node_id,
            sequence_start: 0,
            sequence_end: 10,
        };
        let right_fragment = GraphNode {
            node_id,
            sequence_start: 10,
            sequence_end: 20,
        };
        let graph = make_graph(&[left_fragment, right_fragment]);
        let span = AnnotationSpan {
            id: HashId::convert_str("x"),
            name: "x".into(),
            segments: vec![
                make_segment("split-node", 5, 10, Strand::Forward),
                make_segment("split-node", 10, 15, Strand::Forward),
            ],
        };
        assert!(!span_is_single_node(&span, &graph));
    }

    #[test]
    fn test_span_should_hide_in_truncated_true_for_partial_single_node_span() {
        let node = make_node("n1", 0, 20);
        let graph = make_graph(&[node]);
        let span = AnnotationSpan {
            id: HashId::convert_str("x"),
            name: "x".into(),
            segments: vec![make_segment("n1", 5, 10, Strand::Forward)],
        };
        assert!(span_should_hide_in_truncated(&span, &graph));
    }

    #[test]
    fn test_span_should_hide_in_truncated_false_for_full_width_single_node_span() {
        let node = make_node("n1", 0, 20);
        let graph = make_graph(&[node]);
        let span = AnnotationSpan {
            id: HashId::convert_str("x"),
            name: "x".into(),
            segments: vec![make_segment("n1", 0, 20, Strand::Forward)],
        };
        assert!(!span_should_hide_in_truncated(&span, &graph));
    }

    #[test]
    fn test_span_should_hide_in_truncated_false_for_multi_node_span() {
        let node_1 = make_node("n1", 0, 20);
        let node_2 = make_node("n2", 0, 20);
        let graph = make_graph(&[node_1, node_2]);
        let span = AnnotationSpan {
            id: HashId::convert_str("x"),
            name: "x".into(),
            segments: vec![
                make_segment("n1", 5, 20, Strand::Forward),
                make_segment("n2", 0, 5, Strand::Forward),
            ],
        };
        assert!(!span_should_hide_in_truncated(&span, &graph));
    }

    /// Regression test: an annotation spanning a variant bubble must stay visible in
    /// `Truncated`, even though its segments share one `node_id`.
    #[test]
    fn test_span_should_hide_in_truncated_false_for_bubble_span() {
        let node_id = HashId::convert_str("split-node");
        let before_bubble = GraphNode {
            node_id,
            sequence_start: 0,
            sequence_end: 10,
        };
        let after_bubble = GraphNode {
            node_id,
            sequence_start: 10,
            sequence_end: 20,
        };
        let graph = make_graph(&[before_bubble, after_bubble]);
        let span = AnnotationSpan {
            id: HashId::convert_str("ori"),
            name: "ori".into(),
            segments: vec![
                make_segment("split-node", 5, 10, Strand::Forward),
                make_segment("split-node", 10, 15, Strand::Forward),
            ],
        };
        assert!(!span_should_hide_in_truncated(&span, &graph));
    }
}
