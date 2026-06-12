use std::collections::HashMap;

use gen_core::{HashId, Strand};
use gen_graph::{GenGraph, GraphNodeSlice};
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

pub fn graph_locus_from_annotation_span(
    span: &AnnotationSpan,
    graph: &GenGraph,
) -> Option<GraphLocus> {
    if span.segments.is_empty() {
        return None;
    }
    let node_map: HashMap<_, _> = graph.node_identifiers().map(|n| (n.node_id, n)).collect();
    let slices: Option<Vec<GraphNodeSlice>> = span
        .segments
        .iter()
        .map(|seg| {
            let block = *node_map.get(&seg.node_id)?;
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

impl AnnotationTrack {
    pub fn new(name: impl Into<String>, annotations: Vec<AnnotationSpan>) -> Self {
        AnnotationTrack {
            name: name.into(),
            annotations,
        }
    }
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
}
