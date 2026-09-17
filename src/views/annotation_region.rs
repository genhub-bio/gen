use gen_core::{Strand, region::Region};
use gen_graph::{GenGraph, GraphNodePosition};

use crate::views::annotation_track::{AnnotationSegment, AnnotationSpan};

/// A file annotation projected onto a relative region, together with the position used to
/// center the viewer. The span carries every visible base while the midpoint remains a single
/// navigation target.
#[derive(Clone, Debug)]
pub struct RelativeAnnotationRegion {
    pub span: AnnotationSpan,
    pub midpoint: GraphNodePosition,
}

/// Resolve a relative annotation range, including coordinates before or after the stored feature.
/// Outside coordinates follow the graph from the feature boundary, preserving strand direction.
pub fn resolve_relative_annotation_region(
    span: &AnnotationSpan,
    region: &Region,
    graph: &GenGraph,
) -> Option<RelativeAnnotationRegion> {
    let sliced_span = file_span_for_region(span, region, graph)?;
    let midpoint = file_span_midpoint(span, region, graph)?;
    Some(RelativeAnnotationRegion {
        span: sliced_span,
        midpoint,
    })
}

fn file_span_for_nonnegative_bounds(
    span: &AnnotationSpan,
    start: i64,
    end: i64,
    total_length: i64,
) -> Option<AnnotationSpan> {
    if start < 0 || start > end || end > total_length || start == end {
        return None;
    }

    let mut offset = 0;
    let mut segments = Vec::new();
    for segment in &span.segments {
        let segment_length = (segment.end - segment.start).max(0);
        let segment_start = offset;
        let segment_end = offset + segment_length;
        let clipped_start = start.max(segment_start);
        let clipped_end = end.min(segment_end);
        if clipped_start < clipped_end {
            let local_start = clipped_start - segment_start;
            let local_end = clipped_end - segment_start;
            let (absolute_start, absolute_end) = if segment.strand == Strand::Reverse {
                (segment.end - local_end, segment.end - local_start)
            } else {
                (segment.start + local_start, segment.start + local_end)
            };
            segments.push(AnnotationSegment {
                node_id: segment.node_id,
                start: absolute_start,
                end: absolute_end,
                strand: segment.strand,
            });
        }
        offset = segment_end;
        if offset >= end {
            break;
        }
    }

    (!segments.is_empty()).then_some(AnnotationSpan {
        id: gen_core::HashId::convert_str(&format!("region-search:{}:{start}-{end}", span.name)),
        name: String::new(),
        segments,
    })
}

fn annotation_span_position_at_offset(
    span: &AnnotationSpan,
    offset: i64,
    graph: &GenGraph,
) -> Option<(GraphNodePosition, Strand)> {
    if offset < 0 {
        return None;
    }

    let mut remaining = offset;
    for segment in &span.segments {
        let segment_length = (segment.end - segment.start).max(0);
        if remaining < segment_length {
            let coordinate = if segment.strand == Strand::Reverse {
                segment.end - remaining - 1
            } else {
                segment.start + remaining
            };
            let graph_node = graph
                .nodes()
                .filter(|node| {
                    node.node_id == segment.node_id
                        && node.sequence_start <= coordinate
                        && coordinate < node.sequence_end
                })
                .min_by_key(|node| (node.sequence_end - coordinate).abs())
                .or_else(|| {
                    graph
                        .nodes()
                        .filter(|node| node.node_id == segment.node_id)
                        .min_by_key(|node| (node.sequence_start - coordinate).abs())
                })?;
            return Some((
                GraphNodePosition {
                    graph_node,
                    offset: (coordinate - graph_node.sequence_start).clamp(0, graph_node.length()),
                },
                segment.strand,
            ));
        }
        remaining -= segment_length;
    }
    None
}

fn normalize_graph_position_for_base(
    position: GraphNodePosition,
    graph: &GenGraph,
    direction: i64,
) -> Option<GraphNodePosition> {
    if position.offset >= 0 && position.offset < position.graph_node.length() {
        return Some(position);
    }

    let neighbor = if direction >= 0 {
        graph
            .neighbors_directed(position.graph_node, petgraph::Direction::Outgoing)
            .find(|node| !gen_core::is_terminal(node.node_id))
    } else {
        graph
            .neighbors_directed(position.graph_node, petgraph::Direction::Incoming)
            .find(|node| !gen_core::is_terminal(node.node_id))
    }?;
    Some(GraphNodePosition {
        offset: if direction >= 0 {
            0
        } else {
            neighbor.length() - 1
        },
        graph_node: neighbor,
    })
}

fn graph_positions_at_annotation_offset(
    span: &AnnotationSpan,
    offset: i64,
    graph: &GenGraph,
) -> Option<Vec<(GraphNodePosition, Strand)>> {
    let total_length = span
        .segments
        .iter()
        .map(|segment| (segment.end - segment.start).max(0))
        .sum::<i64>();
    if total_length == 0 {
        return None;
    }
    if offset >= 0 && offset < total_length {
        return annotation_span_position_at_offset(span, offset, graph)
            .map(|(position, strand)| vec![(position, strand)]);
    }

    let anchor_offset = if offset < 0 { 0 } else { total_length - 1 };
    let (anchor, strand) = annotation_span_position_at_offset(span, anchor_offset, graph)?;
    let distance = if offset < 0 {
        offset
    } else {
        offset - anchor_offset
    };
    let graph_distance = if strand == Strand::Reverse {
        -distance
    } else {
        distance
    };
    let mut traversed_graph = graph.clone();
    let positions =
        gen_models::graph::find_offset(&mut traversed_graph, &anchor, graph_distance, |_, _| false)
            .ok()?
            .into_iter()
            .filter_map(|position| {
                normalize_graph_position_for_base(position, &traversed_graph, graph_distance)
                    .map(|position| (position, strand))
            })
            .collect::<Vec<_>>();
    (!positions.is_empty()).then_some(positions)
}

fn graph_position_at_annotation_offset(
    span: &AnnotationSpan,
    offset: i64,
    graph: &GenGraph,
) -> Option<(GraphNodePosition, Strand)> {
    graph_positions_at_annotation_offset(span, offset, graph)?
        .into_iter()
        .next()
}

fn append_graph_position_segment(
    segments: &mut Vec<AnnotationSegment>,
    position: GraphNodePosition,
    strand: Strand,
) {
    append_annotation_segment(
        segments,
        AnnotationSegment {
            node_id: position.graph_node.node_id,
            start: position.coordinate(),
            end: position.coordinate() + 1,
            strand,
        },
    );
}

fn append_annotation_segment(segments: &mut Vec<AnnotationSegment>, segment: AnnotationSegment) {
    if let Some(previous) = segments.last_mut()
        && previous.node_id == segment.node_id
        && previous.strand == segment.strand
        && segment.start <= previous.end
        && segment.end >= previous.start
    {
        previous.start = previous.start.min(segment.start);
        previous.end = previous.end.max(segment.end);
        return;
    }
    segments.push(segment);
}

fn append_annotation_segments(
    segments: &mut Vec<AnnotationSegment>,
    additions: impl IntoIterator<Item = AnnotationSegment>,
) {
    for segment in additions {
        append_annotation_segment(segments, segment);
    }
}

fn file_span_for_relative_range(
    span: &AnnotationSpan,
    start: i64,
    end: i64,
    total_length: i64,
    graph: &GenGraph,
) -> Option<AnnotationSpan> {
    if start > end {
        return None;
    }
    let mut offset = start;
    let mut segments = Vec::new();
    while offset < end {
        if offset >= 0 && offset < total_length {
            let inside_end = end.min(total_length);
            let inside = file_span_for_nonnegative_bounds(span, offset, inside_end, total_length)?;
            append_annotation_segments(&mut segments, inside.segments);
            offset = inside_end;
        } else {
            let outside_end = if offset < 0 { end.min(0) } else { end };
            while offset < outside_end {
                let positions = graph_positions_at_annotation_offset(span, offset, graph)?;
                for (position, strand) in positions {
                    append_graph_position_segment(&mut segments, position, strand);
                }
                offset += 1;
            }
        }
    }
    (!segments.is_empty()).then_some(AnnotationSpan {
        id: gen_core::HashId::convert_str(&format!("region-search:{}:{start}-{end}", span.name)),
        name: String::new(),
        segments,
    })
}

fn file_span_for_region(
    span: &AnnotationSpan,
    region: &Region,
    graph: &GenGraph,
) -> Option<AnnotationSpan> {
    let total_length = span
        .segments
        .iter()
        .map(|segment| (segment.end - segment.start).max(0))
        .sum::<i64>();
    let (start, mut end) = match (region.start, region.end) {
        (None, None) => (0, total_length),
        (Some(start), None) => (start, total_length),
        (Some(start), Some(end)) => (start, end),
        (None, Some(_)) => return None,
    };
    if start == end {
        // Resolver point coordinates have no interval width; retain the exact base for navigation
        // and highlighting as a one-base span.
        end = end.saturating_add(1);
    }
    if start >= 0 && end <= total_length {
        file_span_for_nonnegative_bounds(span, start, end, total_length)
    } else {
        file_span_for_relative_range(span, start, end, total_length, graph)
    }
}

fn file_span_midpoint(
    span: &AnnotationSpan,
    region: &Region,
    graph: &GenGraph,
) -> Option<GraphNodePosition> {
    let total_length = span
        .segments
        .iter()
        .map(|segment| (segment.end - segment.start).max(0))
        .sum::<i64>();
    let (start, mut end) = match (region.start, region.end) {
        (None, None) => (0, total_length),
        (Some(start), None) => (start, total_length),
        (Some(start), Some(end)) => (start, end),
        (None, Some(_)) => return None,
    };
    if start > end {
        return None;
    }
    if start == end {
        end = end.saturating_add(1);
    }
    let midpoint = start.saturating_add(end.saturating_sub(start) / 2);
    graph_position_at_annotation_offset(span, midpoint, graph).map(|(position, _)| position)
}

#[cfg(test)]
mod tests {
    use gen_core::{HashId, Strand, region::Region};
    use gen_graph::{GenGraph, GraphEdge, GraphNode};

    use super::*;

    fn graph_with_split_node() -> GenGraph {
        let node_id = HashId::convert_str("gff-node");
        let first_slice = GraphNode {
            node_id,
            sequence_start: 0,
            sequence_end: 3,
        };
        let second_slice = GraphNode {
            node_id,
            sequence_start: 3,
            sequence_end: 6,
        };
        let mut graph = GenGraph::new();
        graph.add_node(first_slice);
        graph.add_node(second_slice);
        graph.add_edge(
            first_slice,
            second_slice,
            vec![GraphEdge {
                edge_id: HashId::convert_str("gff-edge"),
                source_strand: Strand::Forward,
                target_strand: Strand::Forward,
                chromosome_index: 0,
                phased: 0,
                created_on: 0,
            }],
        );
        graph
    }

    #[test]
    fn test_file_region_negative_offsets_traverse_node_slices_and_strands() {
        let node_id = HashId::convert_str("gff-node");
        let graph = graph_with_split_node();
        let forward_span = AnnotationSpan {
            id: HashId::convert_str("forward"),
            name: "forward".to_string(),
            segments: vec![AnnotationSegment {
                node_id,
                start: 3,
                end: 6,
                strand: Strand::Forward,
            }],
        };
        let forward_slice = resolve_relative_annotation_region(
            &forward_span,
            &Region::parse("forward:-3").expect("should parse negative point"),
            &graph,
        )
        .expect("should traverse to the preceding node slice");
        assert_eq!(forward_slice.span.segments.len(), 1);
        assert_eq!(forward_slice.span.segments[0].start, 0);
        assert_eq!(forward_slice.span.segments[0].end, 1);
        assert_eq!(forward_slice.span.segments[0].strand, Strand::Forward);

        let reverse_span = AnnotationSpan {
            id: HashId::convert_str("reverse"),
            name: "reverse".to_string(),
            segments: vec![AnnotationSegment {
                node_id,
                start: 0,
                end: 3,
                strand: Strand::Reverse,
            }],
        };
        let reverse_slice = resolve_relative_annotation_region(
            &reverse_span,
            &Region::parse("reverse:-3").expect("should parse negative point"),
            &graph,
        )
        .expect("should traverse to the following node slice for reverse strand");
        assert_eq!(reverse_slice.span.segments.len(), 1);
        assert_eq!(reverse_slice.span.segments[0].start, 5);
        assert_eq!(reverse_slice.span.segments[0].end, 6);
        assert_eq!(reverse_slice.span.segments[0].strand, Strand::Reverse);
    }

    #[test]
    fn test_simple_gff_gene_negative_point_projects_from_fixture_coordinates() {
        let simple_gff = include_str!("../../fixtures/simple.gff");
        assert!(simple_gff.contains("ID=gene-a0001"));
        let node_id = HashId::convert_str("simple-gff-node");
        let annotation_node = GraphNode {
            node_id,
            sequence_start: 4,
            sequence_end: 34,
        };
        let prefix_node = GraphNode {
            node_id,
            sequence_start: 0,
            sequence_end: 4,
        };
        let mut graph = GenGraph::new();
        graph.add_node(prefix_node);
        graph.add_node(annotation_node);
        graph.add_edge(
            prefix_node,
            annotation_node,
            vec![GraphEdge {
                edge_id: HashId::convert_str("simple-gff-edge"),
                source_strand: Strand::Forward,
                target_strand: Strand::Forward,
                chromosome_index: 0,
                phased: 0,
                created_on: 0,
            }],
        );
        let span = AnnotationSpan {
            id: HashId::convert_str("gene-a0001"),
            name: "gene-a0001".to_string(),
            // GFF coordinates 5..20 are zero-based half-open 4..20.
            segments: vec![AnnotationSegment {
                node_id,
                start: 4,
                end: 20,
                strand: Strand::Forward,
            }],
        };
        let result = resolve_relative_annotation_region(
            &span,
            &Region::parse("gene-a0001:-3").expect("should parse fixture gene range"),
            &graph,
        )
        .expect("should resolve the fixture gene's negative point");
        assert_eq!(result.span.segments.len(), 1);
        assert_eq!(result.span.segments[0].start, 1);
        assert_eq!(result.span.segments[0].end, 2);
        assert_eq!(result.midpoint.graph_node.sequence_start, 0);
        assert_eq!(result.midpoint.offset, 1);
    }

    #[test]
    fn test_file_region_exact_point_keeps_one_base_span() {
        let graph = graph_with_split_node();
        let span = AnnotationSpan {
            id: HashId::convert_str("point"),
            name: "point".to_string(),
            segments: vec![AnnotationSegment {
                node_id: HashId::convert_str("gff-node"),
                start: 0,
                end: 6,
                strand: Strand::Forward,
            }],
        };
        let result = resolve_relative_annotation_region(
            &span,
            &Region::parse("point:3").expect("should parse exact point"),
            &graph,
        )
        .expect("should resolve exact point");
        assert_eq!(result.span.segments.len(), 1);
        assert_eq!(result.span.segments[0].start, 3);
        assert_eq!(result.span.segments[0].end, 4);
        assert_eq!(result.midpoint.offset, 0);
    }
}
