//! Relative coordinate resolution for annotations selected by an application.
//!
//! This module deliberately starts after annotation lookup and file loading. Callers provide the
//! already selected annotation segments together with the graph in which those segments should be
//! resolved. That keeps source-specific decisions (database annotations, GFF records, BED
//! records, and so on) outside the resolver while giving every source the same coordinate and
//! graph-traversal behavior.

use gen_core::{
    HashId, Strand,
    region::{Region, RegionParseError},
};
use gen_graph::{GenGraph, GraphError, GraphNodePosition};
use thiserror::Error;

use crate::projection::AnnotationSegment;

/// Annotation-relative coordinate bounds and their graph boundary positions.
///
/// The resolver keeps every graph alternative found at a branch. Offsets are zero-based and
/// relative to the annotation's 5-prime-to-3-prime orientation; reverse-strand annotations
/// therefore walk the graph in the opposite direction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedAnnotationRegion {
    /// The uniform strand of the annotation segments.
    pub strand: Strand,
    /// The annotation-relative start offset used for resolution.
    pub start_offset: i64,
    /// The annotation-relative end offset used for resolution.
    pub end_offset: i64,
    /// All graph positions at the requested start offset.
    pub start_anchors: Vec<GraphNodePosition>,
    /// All graph positions at the requested end offset.
    pub end_anchors: Vec<GraphNodePosition>,
}

/// Errors produced while resolving a source-neutral annotation against a graph.
#[derive(Debug, Error)]
pub enum AnnotationRegionError {
    #[error("annotation has no segments")]
    EmptyAnnotation,
    #[error("annotation segments have mixed strands")]
    MixedStrands,
    #[error("annotation strand is not directional")]
    NonDirectionalStrand,
    #[error(transparent)]
    Parse(#[from] RegionParseError),
    #[error("annotation region start {start} is greater than end {end}")]
    InvalidRange { start: i64, end: i64 },
    #[error("annotation anchor node {node_id} is not present at coordinate {coordinate}")]
    MissingAnchor { node_id: HashId, coordinate: i64 },
    #[error(transparent)]
    Graph(#[from] GraphError),
}

/// Resolve an already-normalized annotation-relative region without graph expansion.
///
/// The caller must first apply [`gen_core::region::normalize_user_search_region`] to raw
/// user-facing coordinates. This function preserves zero and negative exact points and should not
/// be given raw positive user coordinates.
pub fn resolve_annotation_region(
    normalized_region: &Region,
    segments: &[AnnotationSegment],
    graph: &mut GenGraph,
) -> Result<ResolvedAnnotationRegion, AnnotationRegionError> {
    resolve_annotation_region_with_expansion(normalized_region, segments, graph, |_, _| false)
}

/// Resolve an already-normalized annotation-relative region with graph expansion.
///
/// The caller must first apply [`gen_core::region::normalize_user_search_region`] to raw
/// user-facing coordinates. This function preserves zero and negative exact points and should not
/// be given raw positive user coordinates. The expansion callback remains owned by the caller,
/// so source lookup and graph loading stay outside this shared resolver.
pub fn resolve_annotation_region_with_expansion<F>(
    normalized_region: &Region,
    segments: &[AnnotationSegment],
    graph: &mut GenGraph,
    mut expand: F,
) -> Result<ResolvedAnnotationRegion, AnnotationRegionError>
where
    F: FnMut(&mut GenGraph, HashId) -> bool,
{
    let strand = annotation_strand(segments)?;
    let ordered_segments = ordered_segments(segments, strand);
    let total_length = annotation_length(&ordered_segments);
    let (start_offset, end_offset) = normalized_region.resolve_relative_bounds(0, total_length)?;
    if start_offset > end_offset {
        return Err(AnnotationRegionError::InvalidRange {
            start: start_offset,
            end: end_offset,
        });
    }
    let start_anchors = resolve_annotation_offset(
        start_offset,
        total_length,
        &ordered_segments,
        graph,
        strand,
        &mut expand,
        true,
    )?;
    let end_anchors = if start_offset == end_offset {
        start_anchors.clone()
    } else {
        resolve_annotation_offset(
            end_offset,
            total_length,
            &ordered_segments,
            graph,
            strand,
            &mut expand,
            false,
        )?
    };

    Ok(ResolvedAnnotationRegion {
        strand,
        start_offset,
        end_offset,
        start_anchors,
        end_anchors,
    })
}

fn annotation_strand(segments: &[AnnotationSegment]) -> Result<Strand, AnnotationRegionError> {
    let first = segments
        .first()
        .ok_or(AnnotationRegionError::EmptyAnnotation)?;
    let strand = first.strand;
    if Strand::is_ambiguous(strand) {
        return Err(AnnotationRegionError::NonDirectionalStrand);
    }
    if segments.iter().any(|segment| segment.strand != strand) {
        return Err(AnnotationRegionError::MixedStrands);
    }
    Ok(strand)
}

fn ordered_segments(segments: &[AnnotationSegment], strand: Strand) -> Vec<AnnotationSegment> {
    // Segment order is the source/path order supplied by the caller. A reverse-strand feature's
    // 5-prime end is its rightmost source segment, so reverse that order instead of comparing
    // coordinates from unrelated node sequence spaces.
    let mut ordered = segments.to_vec();
    if strand == Strand::Reverse {
        ordered.reverse();
    }
    ordered
}

fn annotation_length(segments: &[AnnotationSegment]) -> i64 {
    segments
        .iter()
        .map(|segment| segment.range.end.saturating_sub(segment.range.start))
        .fold(0, i64::saturating_add)
}

fn annotation_positions_at_boundary(
    offset: i64,
    segments: &[AnnotationSegment],
    graph: &GenGraph,
    strand: Strand,
    start_boundary: bool,
) -> Result<Vec<GraphNodePosition>, AnnotationRegionError> {
    let total_length = annotation_length(segments);
    let mut consumed: i64 = 0;
    let mut selected = None;
    for segment in segments {
        let segment_length = segment.range.end.saturating_sub(segment.range.start);
        let segment_end = consumed.saturating_add(segment_length);
        let is_selected = if start_boundary {
            offset < segment_end || (offset == segment_end && segment_end == total_length)
        } else {
            offset <= segment_end && (offset > consumed || offset == 0)
        };
        if is_selected {
            selected = Some((segment, offset.saturating_sub(consumed), segment_length));
            break;
        }
        consumed = segment_end;
    }

    let (segment, local_offset, segment_length) =
        selected.ok_or(AnnotationRegionError::EmptyAnnotation)?;
    let coordinate = match strand {
        Strand::Forward => segment.range.start.saturating_add(local_offset),
        Strand::Reverse => segment.range.end.saturating_sub(local_offset),
        Strand::Unknown | Strand::ImportantButUnknown => {
            return Err(AnnotationRegionError::NonDirectionalStrand);
        }
    };
    let prefer_end_boundary = match strand {
        Strand::Forward if local_offset == 0 || local_offset == segment_length => {
            local_offset == segment_length
        }
        Strand::Reverse if local_offset == 0 || local_offset == segment_length => local_offset == 0,
        // For a coordinate boundary inside one annotation segment, find_offset's cursor is on
        // the side reached from the annotation's 5-prime direction. Segment boundaries above
        // use the source segment's own half-open side instead.
        Strand::Forward => true,
        Strand::Reverse => false,
        Strand::Unknown | Strand::ImportantButUnknown => false,
    };
    let positions = graph
        .nodes()
        .filter(|node| {
            node.node_id == segment.node_id
                && node.sequence_start <= coordinate
                && coordinate <= node.sequence_end
        })
        .map(|node| GraphNodePosition {
            graph_node: node,
            offset: coordinate - node.sequence_start,
        })
        .collect::<Vec<_>>();
    if positions.is_empty() {
        return Err(AnnotationRegionError::MissingAnchor {
            node_id: segment.node_id,
            coordinate,
        });
    }

    let preferred_positions = positions
        .iter()
        .copied()
        .filter(|position| {
            if prefer_end_boundary {
                position.offset == position.graph_node.length()
            } else {
                position.offset != position.graph_node.length()
            }
        })
        .collect::<Vec<_>>();
    if preferred_positions.is_empty() {
        Ok(positions)
    } else {
        Ok(preferred_positions)
    }
}

fn resolve_annotation_offset<F>(
    offset: i64,
    total_length: i64,
    segments: &[AnnotationSegment],
    graph: &mut GenGraph,
    strand: Strand,
    expand: &mut F,
    start_boundary: bool,
) -> Result<Vec<GraphNodePosition>, AnnotationRegionError>
where
    F: FnMut(&mut GenGraph, HashId) -> bool,
{
    if (0..=total_length).contains(&offset) {
        return annotation_positions_at_boundary(offset, segments, graph, strand, start_boundary);
    }

    if offset < 0 {
        let boundary_anchors = annotation_positions_at_boundary(0, segments, graph, strand, true)?;
        return find_graph_positions_from_anchors(
            &boundary_anchors,
            graph,
            graph_distance(offset, strand),
            expand,
        );
    }

    let boundary_anchors =
        annotation_positions_at_boundary(total_length, segments, graph, strand, false)?;
    find_graph_positions_from_anchors(
        &boundary_anchors,
        graph,
        graph_distance(offset.saturating_sub(total_length), strand),
        expand,
    )
}

fn find_graph_positions_from_anchors<F>(
    anchors: &[GraphNodePosition],
    graph: &mut GenGraph,
    distance: i64,
    expand: &mut F,
) -> Result<Vec<GraphNodePosition>, AnnotationRegionError>
where
    F: FnMut(&mut GenGraph, HashId) -> bool,
{
    let mut positions = Vec::new();
    for anchor in anchors {
        positions.extend(gen_models::graph::find_offset(
            graph,
            anchor,
            distance,
            &mut *expand,
        )?);
    }
    positions.sort_unstable();
    positions.dedup();
    Ok(positions)
}

fn graph_distance(annotation_offset: i64, strand: Strand) -> i64 {
    match strand {
        Strand::Forward => annotation_offset,
        Strand::Reverse => annotation_offset.saturating_neg(),
        Strand::Unknown | Strand::ImportantButUnknown => 0,
    }
}

#[cfg(test)]
mod tests {
    use gen_core::{
        HashId, Strand,
        range::Range,
        region::{Region, RegionParseError},
    };
    use gen_graph::{GraphEdge, GraphNode};

    use super::{resolve_annotation_region, resolve_annotation_region_with_expansion};
    use crate::projection::AnnotationSegment;

    fn normalized_region(region: &str) -> Region {
        gen_core::region::normalize_user_search_region(&Region::parse(region).unwrap())
    }

    fn node(name: &str, start: i64, end: i64) -> GraphNode {
        GraphNode {
            node_id: HashId::convert_str(name),
            sequence_start: start,
            sequence_end: end,
        }
    }

    fn graph_with_path(nodes: &[GraphNode]) -> gen_graph::GenGraph {
        let mut graph = gen_graph::GenGraph::new();
        for window in nodes.windows(2) {
            graph.add_edge(window[0], window[1], Vec::<GraphEdge>::new());
        }
        if let Some(last) = nodes.last() {
            graph.add_node(*last);
        }
        graph
    }

    fn segment(name: &str, start: i64, end: i64, strand: Strand) -> AnnotationSegment {
        AnnotationSegment {
            node_id: HashId::convert_str(name),
            range: Range { start, end },
            strand,
        }
    }

    #[test]
    fn test_resolves_forward_negative_point_relative_to_annotation_start() {
        let graph = &mut graph_with_path(&[node("m123", 0, 34)]);
        let segments = [segment("m123", 4, 20, Strand::Forward)];

        let resolved =
            resolve_annotation_region(&normalized_region("gene-a0001:-3"), &segments, graph)
                .unwrap();

        assert_eq!((resolved.start_offset, resolved.end_offset), (-3, -3));
        assert_eq!(resolved.start_anchors.len(), 1);
        assert_eq!(
            resolved.start_anchors[0].graph_node.node_id,
            HashId::convert_str("m123")
        );
        assert_eq!(resolved.start_anchors[0].offset, 1);
        assert_eq!(resolved.end_anchors[0].offset, 1);
    }

    #[test]
    fn test_resolves_zero_points_unchanged_and_positive_points_normalized() {
        let segments = [segment("m123", 4, 20, Strand::Forward)];

        let zero = resolve_annotation_region(
            &normalized_region("gene:0"),
            &segments,
            &mut graph_with_path(&[node("m123", 0, 34)]),
        )
        .unwrap();
        assert_eq!((zero.start_offset, zero.end_offset), (0, 0));
        assert_eq!(
            (zero.start_anchors[0].offset, zero.end_anchors[0].offset),
            (4, 4)
        );

        let positive = resolve_annotation_region(
            &normalized_region("gene:5"),
            &segments,
            &mut graph_with_path(&[node("m123", 0, 34)]),
        )
        .unwrap();
        assert_eq!((positive.start_offset, positive.end_offset), (4, 5));
        assert_eq!(
            (
                positive.start_anchors[0].offset,
                positive.end_anchors[0].offset
            ),
            (8, 9)
        );
    }

    #[test]
    fn test_resolves_reverse_offsets_from_feature_five_prime_start() {
        let segments = [segment("m123", 4, 20, Strand::Reverse)];
        let resolved = resolve_annotation_region(
            &normalized_region("gene:-3"),
            &segments,
            &mut graph_with_path(&[node("m123", 0, 34)]),
        )
        .unwrap();

        assert_eq!((resolved.start_offset, resolved.end_offset), (-3, -3));
        assert_eq!(
            (
                resolved.start_anchors[0].offset,
                resolved.end_anchors[0].offset
            ),
            (23, 23)
        );
    }

    #[test]
    fn test_resolves_reverse_post_feature_points_from_three_prime_boundary() {
        let segments = [segment("m123", 4, 20, Strand::Reverse)];
        let resolved = resolve_annotation_region(
            &normalized_region("gene:17"),
            &segments,
            &mut graph_with_path(&[node("m123", 0, 34)]),
        )
        .unwrap();

        assert_eq!((resolved.start_offset, resolved.end_offset), (16, 17));

        let range = resolve_annotation_region(
            &normalized_region("gene:17-19"),
            &segments,
            &mut graph_with_path(&[node("m123", 0, 34)]),
        )
        .unwrap();
        assert_eq!((range.start_offset, range.end_offset), (16, 19));
        assert_eq!(range.start_anchors[0].offset, 4);
        assert_eq!(range.end_anchors[0].offset, 1);
    }

    #[test]
    fn test_resolves_forward_discontinuous_segments_without_graph_gap_traversal() {
        let segments = [
            segment("m123", 0, 3, Strand::Forward),
            segment("m123", 10, 13, Strand::Forward),
        ];
        let resolved = resolve_annotation_region(
            &normalized_region("gene:5"),
            &segments,
            &mut graph_with_path(&[node("m123", 0, 34)]),
        )
        .unwrap();

        assert_eq!(resolved.start_anchors.len(), 1);
        assert_eq!(resolved.start_anchors[0].coordinate(), 11);
        assert_eq!(resolved.end_anchors.len(), 1);
        assert_eq!(resolved.end_anchors[0].coordinate(), 12);

        let boundary = resolve_annotation_region(
            &normalized_region("gene:4"),
            &segments,
            &mut graph_with_path(&[node("m123", 0, 34)]),
        )
        .unwrap();
        assert_eq!(boundary.start_anchors[0].coordinate(), 10);
        assert_eq!(boundary.end_anchors[0].coordinate(), 11);

        let first_segment = resolve_annotation_region(
            &normalized_region("gene:1-3"),
            &segments,
            &mut graph_with_path(&[node("m123", 0, 34)]),
        )
        .unwrap();
        assert_eq!(first_segment.start_anchors[0].coordinate(), 0);
        assert_eq!(first_segment.end_anchors[0].coordinate(), 3);

        let point = resolve_annotation_region(
            &Region::parse("gene:3").unwrap(),
            &segments,
            &mut graph_with_path(&[node("m123", 0, 34)]),
        )
        .unwrap();
        assert_eq!((point.start_offset, point.end_offset), (3, 3));
        assert_eq!(point.start_anchors, point.end_anchors);
        assert_eq!(point.start_anchors[0].coordinate(), 10);
    }

    #[test]
    fn test_resolves_reverse_discontinuous_segments_without_graph_gap_traversal() {
        let segments = [
            segment("m123", 0, 3, Strand::Reverse),
            segment("m123", 10, 13, Strand::Reverse),
        ];
        let resolved = resolve_annotation_region(
            &normalized_region("gene:5"),
            &segments,
            &mut graph_with_path(&[node("m123", 0, 34)]),
        )
        .unwrap();

        assert_eq!(resolved.start_anchors.len(), 1);
        assert_eq!(resolved.start_anchors[0].coordinate(), 2);
        assert_eq!(resolved.end_anchors.len(), 1);
        assert_eq!(resolved.end_anchors[0].coordinate(), 1);

        let boundary = resolve_annotation_region(
            &normalized_region("gene:4"),
            &segments,
            &mut graph_with_path(&[node("m123", 0, 34)]),
        )
        .unwrap();
        assert_eq!(boundary.start_anchors[0].coordinate(), 3);
        assert_eq!(boundary.end_anchors[0].coordinate(), 2);

        let first_segment = resolve_annotation_region(
            &normalized_region("gene:1-3"),
            &segments,
            &mut graph_with_path(&[node("m123", 0, 34)]),
        )
        .unwrap();
        assert_eq!(first_segment.start_anchors[0].coordinate(), 13);
        assert_eq!(first_segment.end_anchors[0].coordinate(), 10);
    }

    #[test]
    fn test_resolves_forward_discontinuous_post_feature_offsets_from_three_prime_boundary() {
        let segments = [
            segment("m123", 0, 3, Strand::Forward),
            segment("m123", 10, 13, Strand::Forward),
        ];
        let resolved = resolve_annotation_region(
            &normalized_region("gene:8"),
            &segments,
            &mut graph_with_path(&[node("m123", 0, 34)]),
        )
        .unwrap();

        assert_eq!((resolved.start_offset, resolved.end_offset), (7, 8));
        assert_eq!(resolved.start_anchors[0].coordinate(), 14);
        assert_eq!(resolved.end_anchors[0].coordinate(), 15);
    }

    #[test]
    fn test_resolves_annotation_with_caller_graph_expansion_callback() {
        let first = node("first", 0, 3);
        let next = node("next", 0, 3);
        let segments = [segment("first", 0, 3, Strand::Forward)];
        let mut graph = graph_with_path(&[first]);
        let mut expanded = false;

        let resolved = resolve_annotation_region_with_expansion(
            &normalized_region("gene:5"),
            &segments,
            &mut graph,
            |graph, node_id| {
                if node_id == first.node_id && !expanded {
                    graph.add_edge(first, next, Vec::<GraphEdge>::new());
                    expanded = true;
                    true
                } else {
                    false
                }
            },
        )
        .unwrap();

        assert!(expanded);
        assert_eq!(resolved.start_anchors[0].graph_node.node_id, next.node_id);
        assert_eq!(resolved.start_anchors[0].coordinate(), 1);
        assert_eq!(resolved.end_anchors[0].coordinate(), 2);
    }

    #[test]
    fn test_resolves_reverse_discontinuous_post_feature_offsets_from_three_prime_boundary() {
        let segments = [
            segment("m123", 5, 8, Strand::Reverse),
            segment("m123", 15, 18, Strand::Reverse),
        ];
        let resolved = resolve_annotation_region(
            &normalized_region("gene:8"),
            &segments,
            &mut graph_with_path(&[node("m123", 0, 34)]),
        )
        .unwrap();

        assert_eq!((resolved.start_offset, resolved.end_offset), (7, 8));
        assert_eq!(resolved.start_anchors[0].coordinate(), 4);
        assert_eq!(resolved.end_anchors[0].coordinate(), 3);
    }

    #[test]
    fn test_resolves_reverse_segments_in_input_order_from_rightmost_five_prime_segment() {
        let left = node("left", 0, 3);
        let right = node("right", 0, 3);
        let mut graph = gen_graph::GenGraph::new();
        graph.add_edge(left, right, Vec::new());
        let segments = [
            segment("left", 0, 3, Strand::Reverse),
            segment("right", 0, 3, Strand::Reverse),
        ];

        let resolved =
            resolve_annotation_region(&normalized_region("gene:0"), &segments, &mut graph).unwrap();

        assert_eq!(resolved.start_anchors.len(), 1);
        assert_eq!(resolved.start_anchors[0].graph_node.node_id, right.node_id);
        assert_eq!(resolved.start_anchors[0].offset, 3);
        assert_eq!(resolved.end_anchors, resolved.start_anchors);
    }

    #[test]
    fn test_resolves_post_feature_offset_from_three_prime_boundary() {
        let first = node("first", 0, 3);
        let second = node("second", 3, 6);
        let mut graph = gen_graph::GenGraph::new();
        graph.add_edge(first, second, Vec::new());
        let segments = [segment("first", 0, 3, Strand::Forward)];

        let resolved =
            resolve_annotation_region(&normalized_region("gene:4-5"), &segments, &mut graph)
                .unwrap();

        assert_eq!((resolved.start_offset, resolved.end_offset), (3, 5));
        assert_eq!(resolved.start_anchors[0].graph_node, first);
        assert_eq!(resolved.start_anchors[0].offset, 3);
        assert_eq!(resolved.end_anchors[0].graph_node, second);
        assert_eq!(resolved.end_anchors[0].offset, 2);
    }

    #[test]
    fn test_resolves_ranges_across_graph_node_slices() {
        let node_id = HashId::convert_str("split");
        let first = GraphNode {
            node_id,
            sequence_start: 0,
            sequence_end: 3,
        };
        let second = GraphNode {
            node_id,
            sequence_start: 3,
            sequence_end: 6,
        };
        let mut graph = gen_graph::GenGraph::new();
        graph.add_edge(first, second, Vec::new());
        let segments = [AnnotationSegment {
            node_id,
            range: Range { start: 0, end: 6 },
            strand: Strand::Forward,
        }];

        let resolved =
            resolve_annotation_region(&normalized_region("gene:4-6"), &segments, &mut graph)
                .unwrap();

        assert_eq!(resolved.start_anchors.len(), 1);
        assert_eq!(resolved.start_anchors[0].graph_node, first);
        assert_eq!(resolved.start_anchors[0].offset, 3);
        assert_eq!(resolved.end_anchors.len(), 1);
        assert_eq!(resolved.end_anchors[0].graph_node, second);
        assert_eq!(resolved.end_anchors[0].offset, 3);
    }

    #[test]
    fn test_resolves_ranges_across_graph_nodes_and_branch_boundaries() {
        let first = node("first", 0, 4);
        let left = node("left", 0, 3);
        let right = node("right", 0, 5);
        let tail = node("tail", 0, 4);
        let mut graph = gen_graph::GenGraph::new();
        graph.add_edge(first, left, Vec::new());
        graph.add_edge(first, right, Vec::new());
        graph.add_edge(left, tail, Vec::new());
        graph.add_edge(right, tail, Vec::new());
        let segments = [segment("first", 0, 4, Strand::Forward)];

        let resolved = resolve_annotation_region_with_expansion(
            &normalized_region("gene:5-9"),
            &segments,
            &mut graph,
            |_, _| false,
        )
        .unwrap();

        assert_eq!(resolved.start_anchors.len(), 1);
        assert_eq!(
            resolved.start_anchors[0].graph_node.node_id,
            HashId::convert_str("first")
        );
        assert_eq!(resolved.start_anchors[0].offset, 4);
        assert_eq!(resolved.end_anchors.len(), 2);
        assert!(
            resolved
                .end_anchors
                .iter()
                .any(|position| position.graph_node.node_id == HashId::convert_str("tail"))
        );

        let branch_segments = [segment("tail", 0, 4, Strand::Forward)];
        let branch_region = resolve_annotation_region(
            &normalized_region("gene:-3--1"),
            &branch_segments,
            &mut graph,
        )
        .unwrap();
        assert_eq!(branch_region.start_anchors.len(), 2);
        assert!(branch_region.start_anchors.iter().any(|position| {
            position.graph_node.node_id == HashId::convert_str("left") && position.offset == 0
        }));
        assert!(branch_region.start_anchors.iter().any(|position| {
            position.graph_node.node_id == HashId::convert_str("right") && position.offset == 2
        }));
        assert_eq!(
            (branch_region.start_offset, branch_region.end_offset),
            (-3, -1)
        );
    }

    #[test]
    fn test_rejects_open_start_annotation_region() {
        let segments = [segment("m123", 4, 20, Strand::Forward)];
        let error = resolve_annotation_region(
            &normalized_region("gene:..5"),
            &segments,
            &mut graph_with_path(&[node("m123", 0, 34)]),
        )
        .expect_err("open-start annotation regions should match model syntax");

        assert!(matches!(
            error,
            super::AnnotationRegionError::Parse(RegionParseError::InvalidSyntax)
        ));
    }
}
