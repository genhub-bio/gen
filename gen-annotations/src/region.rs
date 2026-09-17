//! Relative coordinate resolution for annotations selected by an application.
//!
//! This module deliberately starts after annotation lookup and file loading. Callers provide the
//! already selected annotation segments together with the graph in which those segments should be
//! resolved. That keeps source-specific decisions (database annotations, GFF records, BED
//! records, and so on) outside the resolver while giving every source the same coordinate and
//! graph-traversal behavior.

pub use gen_core::region::normalize_user_search_region;
use gen_core::{
    HashId, Strand,
    region::{Region, RegionCoordinateSpace, RegionParseError},
};
use gen_graph::{GenGraph, GraphError, GraphNodePosition};
use petgraph::Direction;
use thiserror::Error;

use crate::projection::AnnotationSegment;

/// A region after its annotation-relative offsets have been resolved to graph positions.
///
/// `start_anchors` and `end_anchors` retain every graph alternative found at a branch. The
/// offsets are zero-based half-open offsets in the annotation's 5-prime-to-3-prime orientation;
/// reverse-strand annotations therefore walk the graph in the opposite direction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedAnnotationRegion {
    /// The selected source-neutral segments, ordered from the annotation's 5-prime end.
    pub segments: Vec<AnnotationSegment>,
    /// The uniform strand of the annotation segments.
    pub strand: Strand,
    /// The annotation-relative start offset used for resolution.
    pub start_offset: i64,
    /// The annotation-relative half-open end offset used for resolution.
    pub end_offset: i64,
    /// The graph position at the annotation's 5-prime anchor.
    pub anchor: GraphNodePosition,
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
    #[error("annotation-relative regions cannot have an open start")]
    OpenStart,
    #[error("annotation region start {start} is greater than end {end}")]
    InvalidRange { start: i64, end: i64 },
    #[error("annotation anchor node {node_id} is not present at coordinate {coordinate}")]
    MissingAnchor { node_id: HashId, coordinate: i64 },
    #[error(transparent)]
    Graph(#[from] GraphError),
}

/// Resolve raw user-facing annotation-relative coordinates without graph expansion.
///
/// Positive starts are normalized from one-based to zero-based exactly once. Zero and negative
/// starts are preserved. Call [`resolve_normalized_annotation_region`] when the input has already
/// been normalized.
pub fn resolve_annotation_region(
    region: &Region,
    segments: &[AnnotationSegment],
    graph: &mut GenGraph,
) -> Result<ResolvedAnnotationRegion, AnnotationRegionError> {
    let normalized_region = normalize_user_search_region(region);
    resolve_normalized_annotation_region(&normalized_region, segments, graph)
}

/// Resolve an already-normalized annotation-relative region without graph expansion.
///
/// This function expects internal zero-based coordinates; do not pass raw user-facing positive
/// starts. Use [`resolve_annotation_region`] for raw search coordinates.
pub fn resolve_normalized_annotation_region(
    normalized_region: &Region,
    segments: &[AnnotationSegment],
    graph: &mut GenGraph,
) -> Result<ResolvedAnnotationRegion, AnnotationRegionError> {
    resolve_normalized_annotation_region_with_expansion(
        normalized_region,
        segments,
        graph,
        |_, _| false,
    )
}

/// Resolve raw user-facing annotation-relative coordinates with caller-provided graph expansion.
///
/// The expansion callback is intentionally supplied by the caller because only the application
/// knows how to load missing graph edges. The coordinate, strand, point, range, and branching
/// semantics remain in this shared resolver for both persisted and file-backed annotations. This
/// function normalizes positive starts exactly once; call
/// [`resolve_normalized_annotation_region_with_expansion`] for already-normalized input.
pub fn resolve_annotation_region_with_expansion<F>(
    region: &Region,
    segments: &[AnnotationSegment],
    graph: &mut GenGraph,
    expand: F,
) -> Result<ResolvedAnnotationRegion, AnnotationRegionError>
where
    F: FnMut(&mut GenGraph, HashId) -> bool,
{
    let normalized_region = normalize_user_search_region(region);
    resolve_normalized_annotation_region_with_expansion(&normalized_region, segments, graph, expand)
}

/// Resolve an already-normalized annotation-relative region and permit graph expansion.
///
/// This function expects internal zero-based coordinates; do not pass raw user-facing positive
/// starts. Use [`resolve_annotation_region_with_expansion`] for raw search coordinates.
pub fn resolve_normalized_annotation_region_with_expansion<F>(
    normalized_region: &Region,
    segments: &[AnnotationSegment],
    graph: &mut GenGraph,
    mut expand: F,
) -> Result<ResolvedAnnotationRegion, AnnotationRegionError>
where
    F: FnMut(&mut GenGraph, HashId) -> bool,
{
    if normalized_region.start.is_none() && normalized_region.end.is_some() {
        return Err(AnnotationRegionError::OpenStart);
    }

    let strand = annotation_strand(segments)?;
    let ordered_segments = ordered_segments(segments, strand);
    let anchor = annotation_anchor(&ordered_segments, graph, strand)?;
    let (start_offset, end_offset) = annotation_offsets(normalized_region, &ordered_segments)?;
    let total_length = annotation_length(&ordered_segments);
    let start_anchors = if (0..=total_length).contains(&start_offset) {
        annotation_positions_at_boundary(start_offset, &ordered_segments, graph, strand, true)?
    } else if start_offset < 0 {
        let start_distance = graph_distance(start_offset, strand);
        gen_graph::find_offset(graph, &anchor, start_distance, &mut expand)?
    } else {
        let three_prime_anchor =
            annotation_boundary_anchor(&ordered_segments, graph, strand, false)?;
        let start_distance = graph_distance(start_offset - total_length, strand);
        gen_graph::find_offset(graph, &three_prime_anchor, start_distance, &mut expand)?
    };
    let end_anchors = if (0..=total_length).contains(&end_offset) {
        annotation_positions_at_boundary(end_offset, &ordered_segments, graph, strand, false)?
    } else if end_offset < 0 {
        let end_distance = graph_distance(end_offset, strand);
        gen_graph::find_offset(graph, &anchor, end_distance, &mut expand)?
    } else {
        let three_prime_anchor =
            annotation_boundary_anchor(&ordered_segments, graph, strand, false)?;
        let end_distance = graph_distance(end_offset - total_length, strand);
        gen_graph::find_offset(graph, &three_prime_anchor, end_distance, &mut expand)?
    };
    let resolved_segments = resolve_segments(
        start_offset,
        end_offset,
        &ordered_segments,
        graph,
        strand,
        &anchor,
        &mut expand,
    )?;

    Ok(ResolvedAnnotationRegion {
        segments: resolved_segments,
        strand,
        start_offset,
        end_offset,
        anchor,
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

fn annotation_anchor(
    segments: &[AnnotationSegment],
    graph: &GenGraph,
    strand: Strand,
) -> Result<GraphNodePosition, AnnotationRegionError> {
    let segment = segments
        .first()
        .ok_or(AnnotationRegionError::EmptyAnnotation)?;
    let coordinate = match strand {
        Strand::Forward => segment.range.start,
        Strand::Reverse => segment.range.end,
        Strand::Unknown | Strand::ImportantButUnknown => {
            return Err(AnnotationRegionError::NonDirectionalStrand);
        }
    };

    graph
        .nodes()
        .filter(|node| {
            if node.node_id != segment.node_id {
                return false;
            }
            match strand {
                Strand::Forward => {
                    node.sequence_start <= coordinate && coordinate <= node.sequence_end
                }
                Strand::Reverse => {
                    node.sequence_start <= coordinate && coordinate <= node.sequence_end
                }
                Strand::Unknown | Strand::ImportantButUnknown => false,
            }
        })
        .map(|node| GraphNodePosition {
            graph_node: node,
            offset: coordinate - node.sequence_start,
        })
        .min_by_key(|position| {
            // At a shared boundary, keep the side that contains the annotation's 5-prime
            // anchor: forward starts prefer the outgoing fragment, while reverse starts prefer
            // the incoming fragment. No graph branch is selected here; traversal still returns
            // every alternative from the chosen boundary.
            let at_unwanted_boundary = match strand {
                Strand::Forward => position.offset == position.graph_node.length(),
                Strand::Reverse => position.offset == 0,
                Strand::Unknown | Strand::ImportantButUnknown => false,
            };
            at_unwanted_boundary as u8
        })
        .ok_or(AnnotationRegionError::MissingAnchor {
            node_id: segment.node_id,
            coordinate,
        })
}

fn annotation_boundary_anchor(
    segments: &[AnnotationSegment],
    graph: &GenGraph,
    strand: Strand,
    five_prime: bool,
) -> Result<GraphNodePosition, AnnotationRegionError> {
    let segment = if five_prime {
        segments
            .first()
            .ok_or(AnnotationRegionError::EmptyAnnotation)?
    } else {
        segments
            .last()
            .ok_or(AnnotationRegionError::EmptyAnnotation)?
    };
    let coordinate = match (strand, five_prime) {
        (Strand::Forward, true) | (Strand::Reverse, false) => segment.range.start,
        (Strand::Forward, false) | (Strand::Reverse, true) => segment.range.end,
        (Strand::Unknown | Strand::ImportantButUnknown, _) => {
            return Err(AnnotationRegionError::NonDirectionalStrand);
        }
    };
    let prefer_end_boundary = matches!(
        (strand, five_prime),
        (Strand::Reverse, true) | (Strand::Forward, false)
    );

    graph
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
        .min_by_key(|position| {
            let at_unwanted_boundary = if prefer_end_boundary {
                position.offset == 0
            } else {
                position.offset == position.graph_node.length()
            };
            at_unwanted_boundary as u8
        })
        .ok_or(AnnotationRegionError::MissingAnchor {
            node_id: segment.node_id,
            coordinate,
        })
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

fn resolve_segments<F>(
    start_offset: i64,
    end_offset: i64,
    segments: &[AnnotationSegment],
    graph: &mut GenGraph,
    strand: Strand,
    five_prime_anchor: &GraphNodePosition,
    expand: &mut F,
) -> Result<Vec<AnnotationSegment>, AnnotationRegionError>
where
    F: FnMut(&mut GenGraph, HashId) -> bool,
{
    if start_offset > end_offset {
        return Err(AnnotationRegionError::InvalidRange {
            start: start_offset,
            end: end_offset,
        });
    }

    let total_length = annotation_length(segments);
    let mut resolved = Vec::new();

    if start_offset < 0 {
        let outside_end = end_offset.min(0);
        for offset in start_offset..outside_end {
            let positions = graph_positions_at_offset(
                offset,
                total_length,
                segments,
                graph,
                strand,
                five_prime_anchor,
                expand,
            )?;
            append_graph_position_segments(&mut resolved, positions, strand);
        }
    }

    let inside_start = start_offset.max(0);
    let inside_end = end_offset.min(total_length);
    if inside_start < inside_end {
        append_clipped_segments(&mut resolved, inside_start, inside_end, segments, strand);
    }

    if end_offset > total_length {
        let outside_start = start_offset.max(total_length);
        for offset in outside_start..end_offset {
            let positions = graph_positions_at_offset(
                offset,
                total_length,
                segments,
                graph,
                strand,
                five_prime_anchor,
                expand,
            )?;
            append_graph_position_segments(&mut resolved, positions, strand);
        }
    }

    Ok(resolved)
}

fn append_clipped_segments(
    resolved: &mut Vec<AnnotationSegment>,
    start_offset: i64,
    end_offset: i64,
    segments: &[AnnotationSegment],
    strand: Strand,
) {
    let mut segment_offset: i64 = 0;
    for segment in segments {
        let segment_length = segment.range.end.saturating_sub(segment.range.start);
        let segment_start = segment_offset;
        let segment_end = segment_offset.saturating_add(segment_length);
        let clipped_start = start_offset.max(segment_start);
        let clipped_end = end_offset.min(segment_end);
        if clipped_start < clipped_end {
            let local_start = clipped_start - segment_start;
            let local_end = clipped_end - segment_start;
            let range = if strand == Strand::Reverse {
                gen_core::range::Range {
                    start: segment.range.end - local_end,
                    end: segment.range.end - local_start,
                }
            } else {
                gen_core::range::Range {
                    start: segment.range.start + local_start,
                    end: segment.range.start + local_end,
                }
            };
            resolved.push(AnnotationSegment {
                node_id: segment.node_id,
                range,
                strand: segment.strand,
            });
        }
        segment_offset = segment_end;
        if segment_offset >= end_offset {
            break;
        }
    }
}

fn graph_positions_at_offset<F>(
    offset: i64,
    total_length: i64,
    segments: &[AnnotationSegment],
    graph: &mut GenGraph,
    strand: Strand,
    five_prime_anchor: &GraphNodePosition,
    expand: &mut F,
) -> Result<Vec<GraphNodePosition>, AnnotationRegionError>
where
    F: FnMut(&mut GenGraph, HashId) -> bool,
{
    if (0..total_length).contains(&offset) {
        return direct_graph_positions(offset, segments, graph);
    }

    let (positions, direction) = if offset < 0 {
        // Relative negative coordinates identify bases before the 5-prime boundary. For a
        // reverse feature, that boundary is immediately before the first outside base in graph
        // direction, so offset -1 is distance zero and subsequent bases advance from there.
        let distance = match strand {
            Strand::Forward => offset,
            Strand::Reverse => offset.saturating_neg().saturating_sub(1),
            Strand::Unknown | Strand::ImportantButUnknown => 0,
        };
        (
            gen_graph::find_offset(graph, five_prime_anchor, distance, &mut *expand)?,
            if strand == Strand::Reverse {
                Direction::Outgoing
            } else {
                Direction::Incoming
            },
        )
    } else {
        let three_prime_anchor = annotation_boundary_anchor(segments, graph, strand, false)?;
        // At the 3-prime boundary, the first outside base is distance zero for forward features
        // and one base upstream for reverse features. Endpoint anchors intentionally retain the
        // boundary distances above; this adjustment only affects the concrete selected bases.
        let distance = match strand {
            Strand::Forward => offset.saturating_sub(total_length),
            Strand::Reverse => offset
                .saturating_sub(total_length)
                .saturating_add(1)
                .saturating_neg(),
            Strand::Unknown | Strand::ImportantButUnknown => 0,
        };
        (
            gen_graph::find_offset(graph, &three_prime_anchor, distance, &mut *expand)?,
            if strand == Strand::Reverse {
                Direction::Incoming
            } else {
                Direction::Outgoing
            },
        )
    };
    Ok(normalize_graph_positions_for_base(
        graph, positions, direction,
    ))
}

fn direct_graph_positions(
    offset: i64,
    segments: &[AnnotationSegment],
    graph: &GenGraph,
) -> Result<Vec<GraphNodePosition>, AnnotationRegionError> {
    let mut remaining = offset;
    let segment = segments
        .iter()
        .find(|segment| {
            let length = segment.range.end.saturating_sub(segment.range.start);
            if remaining < length {
                true
            } else {
                remaining -= length;
                false
            }
        })
        .ok_or(AnnotationRegionError::EmptyAnnotation)?;
    let coordinate = if segment.strand == Strand::Reverse {
        segment.range.end - remaining - 1
    } else {
        segment.range.start + remaining
    };
    let positions = graph
        .nodes()
        .filter(|node| {
            node.node_id == segment.node_id
                && node.sequence_start <= coordinate
                && coordinate < node.sequence_end
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
    Ok(positions)
}

fn normalize_graph_positions_for_base(
    graph: &GenGraph,
    positions: Vec<GraphNodePosition>,
    direction: Direction,
) -> Vec<GraphNodePosition> {
    let mut normalized = Vec::new();
    for position in positions {
        if position.offset >= 0 && position.offset < position.graph_node.length() {
            normalized.push(position);
            continue;
        }
        for neighbor in graph
            .neighbors_directed(position.graph_node, direction)
            .filter(|node| !gen_core::is_terminal(node.node_id))
        {
            normalized.push(GraphNodePosition {
                graph_node: neighbor,
                offset: if direction == Direction::Outgoing {
                    0
                } else {
                    neighbor.length() - 1
                },
            });
        }
    }
    normalized.sort_unstable();
    normalized.dedup();
    normalized
}

fn append_graph_position_segments(
    resolved: &mut Vec<AnnotationSegment>,
    positions: Vec<GraphNodePosition>,
    strand: Strand,
) {
    resolved.extend(positions.into_iter().map(|position| AnnotationSegment {
        node_id: position.graph_node.node_id,
        range: gen_core::range::Range {
            start: position.coordinate(),
            end: position.coordinate() + 1,
        },
        strand,
    }));
}

fn annotation_offsets(
    normalized_region: &Region,
    segments: &[AnnotationSegment],
) -> Result<(i64, i64), AnnotationRegionError> {
    let annotation_length = segments
        .iter()
        .map(|segment| segment.range.end.saturating_sub(segment.range.start))
        .fold(0, i64::saturating_add);
    let coordinates = normalized_region.resolve_coordinates(RegionCoordinateSpace::Relative {
        anchor_start: 0,
        anchor_end: annotation_length,
    })?;
    let coordinates = coordinates.into_half_open_point();
    Ok((coordinates.start, coordinates.end))
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
    use gen_core::{HashId, Strand, range::Range, region::Region};
    use gen_graph::{GraphEdge, GraphNode};

    use super::{
        normalize_user_search_region, resolve_annotation_region,
        resolve_annotation_region_with_expansion,
    };
    use crate::projection::AnnotationSegment;

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
    fn test_normalize_user_search_region_only_converts_positive_start() {
        assert_eq!(
            normalize_user_search_region(&Region::parse("gene:5-20").unwrap()),
            Region::parse("gene:4-20").unwrap()
        );
        assert_eq!(
            normalize_user_search_region(&Region::parse("gene:0-0").unwrap()),
            Region::parse("gene:0-0").unwrap()
        );
        assert_eq!(
            normalize_user_search_region(&Region::parse("gene:-3--1").unwrap()),
            Region::parse("gene:-3--1").unwrap()
        );
    }

    #[test]
    fn test_resolves_forward_negative_point_relative_to_annotation_start() {
        let graph = &mut graph_with_path(&[node("m123", 0, 34)]);
        let segments = [segment("m123", 4, 20, Strand::Forward)];

        let resolved =
            resolve_annotation_region(&Region::parse("gene-a0001:-3").unwrap(), &segments, graph)
                .unwrap();

        assert_eq!((resolved.start_offset, resolved.end_offset), (-3, -2));
        assert_eq!(resolved.start_anchors.len(), 1);
        assert_eq!(
            resolved.start_anchors[0].graph_node.node_id,
            HashId::convert_str("m123")
        );
        assert_eq!(resolved.start_anchors[0].offset, 1);
        assert_eq!(resolved.end_anchors[0].offset, 2);
    }

    #[test]
    fn test_resolves_zero_and_positive_points_as_one_base_ranges() {
        let segments = [segment("m123", 4, 20, Strand::Forward)];

        let zero = resolve_annotation_region(
            &Region::parse("gene:0").unwrap(),
            &segments,
            &mut graph_with_path(&[node("m123", 0, 34)]),
        )
        .unwrap();
        assert_eq!((zero.start_offset, zero.end_offset), (0, 1));
        assert_eq!(
            (zero.start_anchors[0].offset, zero.end_anchors[0].offset),
            (4, 5)
        );

        let positive = resolve_annotation_region(
            &Region::parse("gene:5").unwrap(),
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
            &Region::parse("gene:-3").unwrap(),
            &segments,
            &mut graph_with_path(&[node("m123", 0, 34)]),
        )
        .unwrap();

        assert_eq!(resolved.anchor.offset, 20);
        assert_eq!((resolved.start_offset, resolved.end_offset), (-3, -2));
        assert_eq!(
            (
                resolved.start_anchors[0].offset,
                resolved.end_anchors[0].offset
            ),
            (23, 22)
        );
        assert_eq!(resolved.segments.len(), 1);
        assert_eq!(resolved.segments[0].range.start, 22);
        assert_eq!(resolved.segments[0].range.end, 23);
    }

    #[test]
    fn test_resolves_reverse_post_feature_points_from_three_prime_boundary() {
        let segments = [segment("m123", 4, 20, Strand::Reverse)];
        let resolved = resolve_annotation_region(
            &Region::parse("gene:17").unwrap(),
            &segments,
            &mut graph_with_path(&[node("m123", 0, 34)]),
        )
        .unwrap();

        assert_eq!((resolved.start_offset, resolved.end_offset), (16, 17));
        assert_eq!(resolved.segments.len(), 1);
        assert_eq!(resolved.segments[0].range.start, 3);
        assert_eq!(resolved.segments[0].range.end, 4);

        let range = resolve_annotation_region(
            &Region::parse("gene:17-19").unwrap(),
            &segments,
            &mut graph_with_path(&[node("m123", 0, 34)]),
        )
        .unwrap();
        assert_eq!(range.segments.len(), 3);
        assert_eq!(range.segments[0].range.start, 3);
        assert_eq!(range.segments[0].range.end, 4);
        assert_eq!(range.segments[1].range.start, 2);
        assert_eq!(range.segments[1].range.end, 3);
        assert_eq!(range.segments[2].range.start, 1);
        assert_eq!(range.segments[2].range.end, 2);
    }

    #[test]
    fn test_resolves_forward_discontinuous_segments_without_graph_gap_traversal() {
        let segments = [
            segment("m123", 0, 3, Strand::Forward),
            segment("m123", 10, 13, Strand::Forward),
        ];
        let resolved = resolve_annotation_region(
            &Region::parse("gene:5").unwrap(),
            &segments,
            &mut graph_with_path(&[node("m123", 0, 34)]),
        )
        .unwrap();

        assert_eq!(resolved.start_anchors.len(), 1);
        assert_eq!(resolved.start_anchors[0].coordinate(), 11);
        assert_eq!(resolved.end_anchors.len(), 1);
        assert_eq!(resolved.end_anchors[0].coordinate(), 12);
        assert_eq!(resolved.segments.len(), 1);
        assert_eq!(resolved.segments[0].range, Range { start: 11, end: 12 });

        let boundary = resolve_annotation_region(
            &Region::parse("gene:4").unwrap(),
            &segments,
            &mut graph_with_path(&[node("m123", 0, 34)]),
        )
        .unwrap();
        assert_eq!(boundary.start_anchors[0].coordinate(), 10);
        assert_eq!(boundary.end_anchors[0].coordinate(), 11);
        assert_eq!(boundary.segments[0].range, Range { start: 10, end: 11 });

        let first_segment = resolve_annotation_region(
            &Region::parse("gene:1-3").unwrap(),
            &segments,
            &mut graph_with_path(&[node("m123", 0, 34)]),
        )
        .unwrap();
        assert_eq!(first_segment.start_anchors[0].coordinate(), 0);
        assert_eq!(first_segment.end_anchors[0].coordinate(), 3);
        assert_eq!(first_segment.segments[0].range, Range { start: 0, end: 3 });
    }

    #[test]
    fn test_resolves_reverse_discontinuous_segments_without_graph_gap_traversal() {
        let segments = [
            segment("m123", 0, 3, Strand::Reverse),
            segment("m123", 10, 13, Strand::Reverse),
        ];
        let resolved = resolve_annotation_region(
            &Region::parse("gene:5").unwrap(),
            &segments,
            &mut graph_with_path(&[node("m123", 0, 34)]),
        )
        .unwrap();

        assert_eq!(resolved.start_anchors.len(), 1);
        assert_eq!(resolved.start_anchors[0].coordinate(), 2);
        assert_eq!(resolved.end_anchors.len(), 1);
        assert_eq!(resolved.end_anchors[0].coordinate(), 1);
        assert_eq!(resolved.segments.len(), 1);
        assert_eq!(resolved.segments[0].range, Range { start: 1, end: 2 });

        let boundary = resolve_annotation_region(
            &Region::parse("gene:4").unwrap(),
            &segments,
            &mut graph_with_path(&[node("m123", 0, 34)]),
        )
        .unwrap();
        assert_eq!(boundary.start_anchors[0].coordinate(), 3);
        assert_eq!(boundary.end_anchors[0].coordinate(), 2);
        assert_eq!(boundary.segments[0].range, Range { start: 2, end: 3 });

        let first_segment = resolve_annotation_region(
            &Region::parse("gene:1-3").unwrap(),
            &segments,
            &mut graph_with_path(&[node("m123", 0, 34)]),
        )
        .unwrap();
        assert_eq!(first_segment.start_anchors[0].coordinate(), 13);
        assert_eq!(first_segment.end_anchors[0].coordinate(), 10);
        assert_eq!(
            first_segment.segments[0].range,
            Range { start: 10, end: 13 }
        );
    }

    #[test]
    fn test_resolves_forward_discontinuous_post_feature_offsets_from_three_prime_boundary() {
        let segments = [
            segment("m123", 0, 3, Strand::Forward),
            segment("m123", 10, 13, Strand::Forward),
        ];
        let resolved = resolve_annotation_region(
            &Region::parse("gene:8").unwrap(),
            &segments,
            &mut graph_with_path(&[node("m123", 0, 34)]),
        )
        .unwrap();

        assert_eq!((resolved.start_offset, resolved.end_offset), (7, 8));
        assert_eq!(resolved.start_anchors[0].coordinate(), 14);
        assert_eq!(resolved.end_anchors[0].coordinate(), 15);
        assert_eq!(resolved.segments.len(), 1);
        assert_eq!(resolved.segments[0].range, Range { start: 14, end: 15 });
    }

    #[test]
    fn test_resolves_annotation_with_caller_graph_expansion_callback() {
        let first = node("first", 0, 3);
        let next = node("next", 0, 3);
        let segments = [segment("first", 0, 3, Strand::Forward)];
        let mut graph = graph_with_path(&[first]);
        let mut expanded = false;

        let resolved = resolve_annotation_region_with_expansion(
            &Region::parse("gene:5").unwrap(),
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
        assert_eq!(resolved.segments.len(), 1);
        assert_eq!(resolved.segments[0].node_id, next.node_id);
        assert_eq!(resolved.segments[0].range, Range { start: 1, end: 2 });
    }

    #[test]
    fn test_resolves_reverse_discontinuous_post_feature_offsets_from_three_prime_boundary() {
        let segments = [
            segment("m123", 5, 8, Strand::Reverse),
            segment("m123", 15, 18, Strand::Reverse),
        ];
        let resolved = resolve_annotation_region(
            &Region::parse("gene:8").unwrap(),
            &segments,
            &mut graph_with_path(&[node("m123", 0, 34)]),
        )
        .unwrap();

        assert_eq!((resolved.start_offset, resolved.end_offset), (7, 8));
        assert_eq!(resolved.start_anchors[0].coordinate(), 4);
        assert_eq!(resolved.end_anchors[0].coordinate(), 3);
        assert_eq!(resolved.segments.len(), 1);
        assert_eq!(resolved.segments[0].range, Range { start: 3, end: 4 });
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
            resolve_annotation_region(&Region::parse("gene:0").unwrap(), &segments, &mut graph)
                .unwrap();

        assert_eq!(
            resolved.anchor.graph_node.node_id,
            HashId::convert_str("right")
        );
        assert_eq!(resolved.anchor.offset, 3);
        assert_eq!(resolved.segments[0].node_id, HashId::convert_str("right"));
        assert_eq!(resolved.segments[0].range.start, 2);
        assert_eq!(resolved.segments[0].range.end, 3);
    }

    #[test]
    fn test_resolves_post_feature_offset_from_three_prime_boundary() {
        let first = node("first", 0, 3);
        let second = node("second", 3, 6);
        let mut graph = gen_graph::GenGraph::new();
        graph.add_edge(first, second, Vec::new());
        let segments = [segment("first", 0, 3, Strand::Forward)];

        let resolved =
            resolve_annotation_region(&Region::parse("gene:4-5").unwrap(), &segments, &mut graph)
                .unwrap();

        assert_eq!(resolved.segments.len(), 2);
        assert_eq!(resolved.segments[0].node_id, HashId::convert_str("second"));
        assert_eq!(resolved.segments[0].range.start, 3);
        assert_eq!(resolved.segments[0].range.end, 4);
        assert_eq!(resolved.segments[1].range.start, 4);
        assert_eq!(resolved.segments[1].range.end, 5);
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
            resolve_annotation_region(&Region::parse("gene:4-6").unwrap(), &segments, &mut graph)
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
            &Region::parse("gene:5-9").unwrap(),
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
            &Region::parse("gene:-3--1").unwrap(),
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
        assert_eq!(branch_region.segments.len(), 4);
    }

    #[test]
    fn test_rejects_open_start_annotation_region() {
        let segments = [segment("m123", 4, 20, Strand::Forward)];
        let error = resolve_annotation_region(
            &Region::parse("gene:..5").unwrap(),
            &segments,
            &mut graph_with_path(&[node("m123", 0, 34)]),
        )
        .expect_err("open-start annotation regions should match model syntax");

        assert!(matches!(error, super::AnnotationRegionError::OpenStart));
    }
}
