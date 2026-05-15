use std::{
    cmp::{max, min},
    collections::{HashMap, HashSet},
};

use gen_core::{HashId, Strand, is_end_node, is_start_node};
use gen_graph::GenGraph;
use gen_models::locus::{BlockSlice, GraphLocus};
use gen_tui::{
    GraphController, ViewportState, VisualDetail, WorldRect, plotter::NodeSizer,
    theme::current_theme,
};
use petgraph::visit::{IntoNodeIdentifiers, NodeIndexable};
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
};

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
    pub annotation_segments_by_node: HashMap<HashId, Vec<(usize, AnnotationSegment)>>,
    pub min_rows: usize,
    pub max_rows: usize,
}

pub fn graphlocus_to_annotation_span(locus: &GraphLocus, name: &str) -> AnnotationSpan {
    let segments = locus
        .slices
        .iter()
        .map(|s| AnnotationSegment {
            node_id: s.block.node_id,
            start: s.block.sequence_start + s.start as i64,
            end: s.block.sequence_start + s.end as i64,
            strand: locus.strand,
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
    let strand = span.segments.first()?.strand;
    let node_map: HashMap<_, _> = graph.node_identifiers().map(|n| (n.node_id, n)).collect();
    let slices: Option<Vec<BlockSlice>> = span
        .segments
        .iter()
        .map(|seg| {
            let block = *node_map.get(&seg.node_id)?;
            let start = (seg.start - block.sequence_start).max(0) as usize;
            let end = (seg.end - block.sequence_start).max(0) as usize;
            Some(BlockSlice { block, start, end })
        })
        .collect();
    Some(GraphLocus {
        slices: slices?,
        strand,
    })
}

impl AnnotationTrack {
    pub fn new(name: impl Into<String>, annotations: Vec<AnnotationSpan>) -> Self {
        let name = name.into();
        let mut segments_by_node: HashMap<HashId, Vec<(usize, AnnotationSegment)>> = HashMap::new();
        for (idx, annotation) in annotations.iter().enumerate() {
            for segment in &annotation.segments {
                segments_by_node
                    .entry(segment.node_id)
                    .or_default()
                    .push((idx, segment.clone()));
            }
        }
        AnnotationTrack {
            name,
            annotations,
            annotation_segments_by_node: segments_by_node,
            min_rows: 0,
            max_rows: 9,
        }
    }

    /// Draw annotation track at the bottom of `area`. Returns height used, or 0 if nothing visible.
    pub fn draw<S: NodeSizer<GenGraph>>(
        &self,
        buf: &mut Buffer,
        area: Rect,
        controller: &GraphController<GenGraph, S>,
    ) -> u16 {
        let (_, mut segments_by_annotation, mut truncated_count) =
            collect_visual_segments(self, controller);

        if controller.get_detail_level() == VisualDetail::Minimal {
            segments_by_annotation.retain(|_, segs| {
                let all_same_node = segs.windows(2).all(|w| w[0].node_x1 == w[1].node_x1);
                if all_same_node {
                    truncated_count += 1;
                    false
                } else {
                    true
                }
            });
        }

        let visible_ranges: AnnotationVisibleRanges = segments_by_annotation
            .iter()
            .map(|(idx, segs)| {
                let x1 = segs.iter().map(VisualSegment::start_x).min().unwrap_or(0);
                let x2 = segs.iter().map(VisualSegment::end_x).max().unwrap_or(0);
                (*idx, (x1, x2))
            })
            .collect();
        let packed_rows = pack_visible_annotation_rows(&visible_ranges, self, self.max_rows);

        let n = packed_rows.len().max(self.min_rows);
        let height = (n as u16 + 1).max(1);
        if area.height < height {
            return 0;
        }

        let area = Rect {
            x: area.x,
            y: area.y + area.height - height,
            width: area.width,
            height,
        };

        let viewport_state = &controller.viewport_state;

        let theme = current_theme();

        let divider_style = Style::default().fg(theme[0x02]);
        buf.set_string(
            area.x,
            area.y,
            "─".repeat(area.width as usize),
            divider_style,
        );

        if !self.name.is_empty() {
            let header = if truncated_count > 0 {
                format!("{} (+{truncated_count} in truncated regions)", self.name)
            } else {
                self.name.clone()
            };
            buf.set_string(area.x + 1, area.y, header, Style::default().fg(theme[0x07]));
        }

        let inner = Rect {
            x: area.x,
            y: area.y + 1,
            width: area.width,
            height: area.height - 1,
        };

        if inner.width == 0 {
            return height;
        }

        let bg_color = theme[0x00];
        for row in inner.y..inner.y + inner.height {
            buf.set_string(
                inner.x,
                row,
                " ".repeat(inner.width as usize),
                Style::default().bg(bg_color),
            );
        }

        let label_fg_over_bar = bg_color;

        for (row_idx, row_annotations) in packed_rows.iter().take(inner.height as usize).enumerate()
        {
            let annotation_color = if row_idx % 2 == 0 {
                theme[0x0B]
            } else {
                theme[0x0C]
            };
            let annotation_style = Style::default().fg(annotation_color).bg(bg_color);
            let terminal_y = inner.y + row_idx as u16;

            for idx in row_annotations {
                let Some(mut segments) = segments_by_annotation.get(idx).cloned() else {
                    continue;
                };
                segments.sort_by_key(|s| s.start_x());
                let mut prev_end: Option<i64> = None;
                for seg in &segments {
                    let x1 = seg.start_x();
                    let x2 = seg.end_x();
                    place_bar(
                        buf,
                        inner,
                        viewport_state,
                        x1,
                        x2,
                        terminal_y,
                        annotation_style,
                    );
                    if seg.start.1.is_none() {
                        draw_truncation_marker(
                            buf,
                            inner,
                            viewport_state,
                            seg.start.0,
                            terminal_y,
                            annotation_style,
                        );
                    }
                    if seg.end.1.is_none() {
                        draw_truncation_marker(
                            buf,
                            inner,
                            viewport_state,
                            seg.end.0,
                            terminal_y,
                            annotation_style,
                        );
                    }
                    if let Some(prev) = prev_end
                        && x1 - prev > 1
                    {
                        draw_dashed_connector(
                            buf,
                            inner,
                            viewport_state,
                            prev + 1,
                            x1 - 1,
                            terminal_y,
                            annotation_style,
                        );
                    }
                    prev_end = Some(x2);
                }
            }

            for idx in row_annotations {
                let annotation_name = &self.annotations[*idx].name;
                if annotation_name.is_empty() {
                    continue;
                }
                let Some(visual_segs) = segments_by_annotation.get(idx) else {
                    continue;
                };
                let x1 = visual_segs
                    .iter()
                    .map(VisualSegment::start_x)
                    .min()
                    .unwrap_or_default();
                let x2 = visual_segs
                    .iter()
                    .map(VisualSegment::end_x)
                    .max()
                    .unwrap_or_default();

                let span_strand = span_strand(&self.annotations[*idx]);
                place_sticky_label(
                    buf,
                    inner,
                    viewport_state,
                    x1,
                    x2,
                    terminal_y,
                    annotation_name,
                    span_strand,
                    annotation_style,
                    label_fg_over_bar,
                );
            }
        }

        height
    }
}

type AnnotationVisibleRanges = HashMap<usize, (i64, i64)>;

/// Bases from either node boundary that map to precise cells (0-indexed, inclusive).
/// For example, if we truncate 123456789 to 12...89  this number would be 2
const BORDER_BP: i64 = 5;

/// Extra collection margin for labels that can remain visible outside the bar.
const MIN_LABEL_LOOKAHEAD: i64 = 8;

/// Fixed gap between annotation bar and label when label is external.
const LABEL_SIDE_GUTTER: i64 = 1;

/// Fixed gap reserved to the right of each annotation bar when packing rows
const RIGHT_BAR_MARGIN: i64 = 4;

/// How an annotation endpoint maps to the current visual representation.
///
/// `(base_x, Some(offset))` — exact cell at world position `base_x + offset`.
/// `(node_mid_x, None)`     — falls in the truncated interior; draw `░` at `node_mid_x`.
pub type EndpointX = (i64, Option<i64>);

/// One resolved segment of an annotation, covering part of a single graph node.
#[derive(Clone, Debug)]
pub struct VisualSegment {
    /// World X of node left edge.
    pub node_x1: i64,
    /// World X of node right edge.
    pub node_x2: i64,
    /// Resolved start endpoint.
    pub start: EndpointX,
    /// Resolved end endpoint.
    pub end: EndpointX,
}

impl VisualSegment {
    pub fn start_x(&self) -> i64 {
        self.start.0 + self.start.1.unwrap_or(0)
    }

    pub fn end_x(&self) -> i64 {
        self.end.0 + self.end.1.unwrap_or(0)
    }
}

/// Resolve one annotation sequence position to a visual endpoint.
///
/// * `dist_from_start` – bases from the node's first base (≥ 0).
/// * `dist_from_end`   – bases from the node's last base, inclusive (≥ 0; 0 = last base).
///
/// For truncated nodes (`node_len > node_width`, `node_width ≥ 3`):
/// - Positions within `BORDER_BP` bases of either boundary map to a precise border cell.
/// - All other positions map to the node's middle cell with `offset = None`.
///
/// The middle cell (`node_width / 2`) is kept free of border mappings, ensuring
/// `░` indicators only appear for genuinely ambiguous interior positions.
fn resolve_endpoint(
    dist_from_start: i64,
    dist_from_end: i64,
    node_x1: i64,
    node_width: i64,
    node_len: i64,
) -> EndpointX {
    if node_len <= node_width || node_width < 3 {
        let offset = if node_len <= 0 {
            0
        } else {
            (dist_from_start * node_width / node_len).clamp(0, node_width - 1)
        };
        return (node_x1, Some(offset));
    }

    // Truncated node. `half` is the mid-cell index (= 6 for width 13).
    // Left border uses cells 0 .. half-1; right border uses cells half+1 .. end.
    let half = node_width / 2;

    if dist_from_start <= BORDER_BP {
        return (node_x1, Some(dist_from_start.min(half - 1)));
    }

    if dist_from_end <= BORDER_BP {
        return (node_x1, Some(node_width - 1 - dist_from_end.min(half - 1)));
    }

    (node_x1 + half, None)
}

/// Collect visible annotation segments, resolving endpoint positions against
/// the current layout and detail level.
///
/// Returns `(visible_indices_sorted, segments_by_annotation, entirely_truncated_count)`.
/// `entirely_truncated_count` is the number of visible annotations whose every segment
/// falls entirely within a truncated node interior and is therefore visually omitted.
pub fn collect_visual_segments<S: NodeSizer<GenGraph>>(
    track: &AnnotationTrack,
    controller: &GraphController<GenGraph, S>,
) -> (Vec<usize>, HashMap<usize, Vec<VisualSegment>>, usize) {
    if track.annotations.is_empty() {
        return (Vec::new(), HashMap::new(), 0);
    }

    let viewport_state = &controller.viewport_state;
    let viewport_graph = controller.get_viewport_graph();
    let graph = controller.graph();

    let camera_rect = viewport_state.camera_rect();
    let win_start = camera_rect.min.x;
    let win_end = camera_rect.max.x;

    let max_label_len = track
        .annotations
        .iter()
        .map(|annotation| annotation.name.chars().count() as i64)
        .max()
        .unwrap_or(0);

    // Labels can remain visible outside the bar by roughly their own width plus
    // gutter. If collection only looks at the bar, a label can vanish before it
    // has finished sliding out of the viewport.
    let lookahead = MIN_LABEL_LOOKAHEAD.max(max_label_len + LABEL_SIDE_GUTTER * 2 + 1);
    let left_bound = win_start - lookahead;
    let right_bound = win_end + lookahead;

    let mut segments_by_annotation: HashMap<usize, Vec<VisualSegment>> = HashMap::new();
    let mut visible_indices: Vec<usize> = Vec::new();
    let mut visible_set: HashSet<usize> = HashSet::new();

    for (world_pos, domain_idx, layout_node) in viewport_graph.data_nodes() {
        let block = <&GenGraph as NodeIndexable>::from_index(&graph, domain_idx.index());

        if is_start_node(block.node_id) || is_end_node(block.node_id) {
            continue;
        }

        let node_rect = WorldRect::from_center_and_size(world_pos, layout_node.size);
        let x1 = node_rect.min.x;
        let x2 = node_rect.max.x;

        if x2 < left_bound || x1 > right_bound {
            continue;
        }

        let Some(segments) = track.annotation_segments_by_node.get(&block.node_id) else {
            continue;
        };

        let node_len = block.sequence_end - block.sequence_start;
        if node_len <= 0 {
            continue;
        }

        let node_width = layout_node.size.0 as i64;

        for (idx, segment) in segments {
            let overlap_start = max(segment.start, block.sequence_start);
            let overlap_end = min(segment.end, block.sequence_end);

            if overlap_end <= overlap_start {
                continue;
            }

            let seg_start = resolve_endpoint(
                overlap_start - block.sequence_start,
                block.sequence_end - 1 - overlap_start,
                x1,
                node_width,
                node_len,
            );

            let seg_end = resolve_endpoint(
                overlap_end - 1 - block.sequence_start,
                block.sequence_end - overlap_end,
                x1,
                node_width,
                node_len,
            );

            if visible_set.insert(*idx) {
                visible_indices.push(*idx);
            }

            segments_by_annotation
                .entry(*idx)
                .or_default()
                .push(VisualSegment {
                    node_x1: x1,
                    node_x2: x2,
                    start: seg_start,
                    end: seg_end,
                });
        }
    }

    // Filter out entirely truncated annotations (start and end on same node, both in truncated interior).
    let mut entirely_truncated_count = 0;

    visible_indices.retain(|idx| {
        let Some(segs) = segments_by_annotation.get(idx) else {
            return false;
        };

        let (Some(first), Some(last)) = (segs.first(), segs.last()) else {
            return true;
        };

        let is_truncated =
            first.node_x1 == last.node_x1 && first.start.1.is_none() && last.end.1.is_none();

        if is_truncated {
            entirely_truncated_count += 1;
            segments_by_annotation.remove(idx);
            false
        } else {
            true
        }
    });

    (
        visible_indices,
        segments_by_annotation,
        entirely_truncated_count,
    )
}

fn pack_visible_annotation_rows(
    visible_ranges: &AnnotationVisibleRanges,
    track: &AnnotationTrack,
    max_rows: usize,
) -> Vec<Vec<usize>> {
    if visible_ranges.is_empty() || max_rows == 0 {
        return Vec::new();
    }

    let mut annotations: Vec<_> = visible_ranges.iter().collect();
    annotations.sort_by(|a, b| {
        a.1.0
            .cmp(&b.1.0)
            .then_with(|| a.1.1.cmp(&b.1.1))
            .then_with(|| a.0.cmp(b.0))
    });

    let mut rows: Vec<Vec<usize>> = Vec::with_capacity(max_rows);
    let mut row_ends: Vec<i64> = Vec::with_capacity(max_rows);

    for (idx, (start, end)) in annotations {
        let label_len = track.annotations[*idx].name.chars().count() as i64;
        let occupied_start = *start - label_len - 1;
        let occupied_end = *end + RIGHT_BAR_MARGIN;

        let best_row = row_ends
            .iter()
            .enumerate()
            .filter(|(_, row_end)| occupied_start > **row_end)
            .min_by_key(|(_, row_end)| **row_end)
            .map(|(row_idx, _)| row_idx);

        if let Some(row_idx) = best_row {
            rows[row_idx].push(*idx);
            row_ends[row_idx] = occupied_end;
            continue;
        }

        if rows.len() < max_rows {
            rows.push(vec![*idx]);
            row_ends.push(occupied_end);
        }
    }

    rows
}

fn span_strand(span: &AnnotationSpan) -> Option<Strand> {
    let first = span.segments.first()?.strand;
    if Strand::is_ambiguous(first) {
        return None;
    }
    if span.segments.iter().all(|s| s.strand == first) {
        Some(first)
    } else {
        None
    }
}

/// Convert a world X coordinate to a terminal X coordinate.
#[inline]
pub(crate) fn world_x_to_term_x(viewport_state: &ViewportState, world_x: i64) -> i64 {
    let cam_min_x = viewport_state.camera_rect().min.x;
    viewport_state.viewport_bounds.x as i64 + (world_x - cam_min_x)
}

fn draw_truncation_marker(
    buf: &mut Buffer,
    inner: Rect,
    viewport_state: &ViewportState,
    world_x: i64,
    terminal_y: u16,
    style: Style,
) {
    let raw_x = world_x_to_term_x(viewport_state, world_x);
    let area_left = inner.x as i64;
    let area_right = (inner.x + inner.width - 1) as i64;

    if raw_x >= area_left && raw_x <= area_right {
        buf.set_string(raw_x as u16, terminal_y, "░", style);
    }
}

fn place_bar(
    buf: &mut Buffer,
    inner: Rect,
    viewport_state: &ViewportState,
    world_x1: i64,
    world_x2: i64,
    terminal_y: u16,
    style: Style,
) {
    let area_left = inner.x as i64;
    let area_right = (inner.x + inner.width - 1) as i64;
    let raw_x1 = world_x_to_term_x(viewport_state, world_x1);
    let raw_x2 = world_x_to_term_x(viewport_state, world_x2);

    if raw_x2 < area_left || raw_x1 > area_right {
        return;
    }

    let start_x = raw_x1.max(area_left) as u16;
    let end_x = raw_x2.min(area_right) as u16;
    let width = (end_x - start_x + 1) as usize;

    buf.set_string(start_x, terminal_y, "█".repeat(width), style);
}

fn draw_dashed_connector(
    buf: &mut Buffer,
    inner: Rect,
    viewport_state: &ViewportState,
    world_x_start: i64,
    world_x_end: i64,
    terminal_y: u16,
    style: Style,
) {
    if world_x_end <= world_x_start {
        return;
    }

    let area_left = inner.x as i64;
    let area_right = (inner.x + inner.width - 1) as i64;
    let raw_x1 = world_x_to_term_x(viewport_state, world_x_start);
    let raw_x2 = world_x_to_term_x(viewport_state, world_x_end);

    if raw_x2 < area_left || raw_x1 > area_right {
        return;
    }

    let start_x = raw_x1.max(area_left) as u16;
    let end_x = raw_x2.min(area_right) as u16;
    let visible_width = (end_x - start_x + 1) as usize;

    buf.set_string(start_x, terminal_y, "-".repeat(visible_width), style);
}

#[allow(clippy::too_many_arguments)]
fn place_sticky_label(
    buf: &mut Buffer,
    inner: Rect,
    viewport_state: &ViewportState,
    annotation_world_x1: i64,
    annotation_world_x2: i64,
    terminal_y: u16,
    label: &str,
    strand: Option<Strand>,
    normal_style: Style,
    label_fg_over_bar: Color,
) {
    // TODO: this assumes one terminal cell per char. If annotation names can
    // contain wide Unicode or combining marks, switch to unicode-width and
    // grapheme-aware clipping/writing.
    let label_len = label.chars().count() as i64;

    if label_len == 0 || inner.width == 0 {
        return;
    }

    let area_left = inner.x as i64;
    let area_right = (inner.x + inner.width - 1) as i64;

    if area_right < area_left {
        return;
    }

    let annotation_start = annotation_world_x1.min(annotation_world_x2);
    let annotation_end = annotation_world_x1.max(annotation_world_x2);

    let annotation_start_term = world_x_to_term_x(viewport_state, annotation_start);
    let annotation_end_term = world_x_to_term_x(viewport_state, annotation_end);

    let final_start = match strand {
        Some(Strand::Forward) => (annotation_start_term - LABEL_SIDE_GUTTER - label_len)
            .max(area_left + LABEL_SIDE_GUTTER)
            .min(annotation_end_term - label_len),
        Some(Strand::Reverse) => (annotation_end_term + LABEL_SIDE_GUTTER)
            .min(area_right - label_len - LABEL_SIDE_GUTTER)
            .max(annotation_start_term + LABEL_SIDE_GUTTER),
        _ => (annotation_start_term - LABEL_SIDE_GUTTER - label_len)
            .max(area_left + LABEL_SIDE_GUTTER)
            .min(annotation_end_term - label_len),
    };

    let label_with_marker = match strand {
        Some(Strand::Forward) => format!("{label}›"),
        Some(Strand::Reverse) => format!("‹{label}"),
        _ => label.to_string(),
    };

    draw_label_clipped_over_existing(
        buf,
        inner,
        final_start,
        terminal_y,
        &label_with_marker,
        normal_style,
        label_fg_over_bar,
    );
}

fn draw_label_clipped_over_existing(
    buf: &mut Buffer,
    inner: Rect,
    start_x: i64,
    terminal_y: u16,
    label: &str,
    normal_style: Style,
    label_fg_over_bar: Color,
) {
    let area_left = inner.x as i64;
    let area_right = (inner.x + inner.width - 1) as i64;

    if area_right < area_left {
        return;
    }

    let normal_fg = normal_style.fg.unwrap_or(Color::Reset);

    for (i, ch) in label.chars().enumerate() {
        let x = start_x + i as i64;

        if x < area_left || x > area_right {
            continue;
        }

        if ch == ' ' {
            // Preserve the existing cell entirely for spaces in the label, so
            // bars/connectors/backgrounds shine through instead of getting a
            // visible label-colored gap.
            continue;
        }

        let x = x as u16;

        let Some(old_cell) = buf.cell((x, terminal_y)) else {
            continue;
        };

        let old_symbol = old_cell.symbol();

        let style = if old_symbol == "█" || old_symbol == "░" {
            // The annotation bar is drawn as a full block glyph. Its visible
            // color is the cell foreground, so reuse that fg as the label bg.
            Style::default().fg(label_fg_over_bar).bg(old_cell.fg)
        } else {
            // Not over a bar: preserve whatever background is already there.
            Style::default().fg(normal_fg).bg(old_cell.bg)
        };

        buf.set_string(x, terminal_y, ch.to_string(), style);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn track_with_names(names: &[&str]) -> AnnotationTrack {
        AnnotationTrack::new(
            "test",
            names
                .iter()
                .map(|name| AnnotationSpan {
                    id: HashId::convert_str(name),
                    name: (*name).to_string(),
                    segments: Vec::new(),
                })
                .collect(),
        )
    }

    #[test]
    fn pack_visible_annotation_rows_packs_disjoint_annotations_on_same_row() {
        let track = track_with_names(&["abc1", "abc2", "abc3"]);
        let mut visible_ranges = AnnotationVisibleRanges::new();
        visible_ranges.insert(0, (10, 20));
        visible_ranges.insert(1, (31, 41));
        visible_ranges.insert(2, (52, 62));

        let rows = pack_visible_annotation_rows(&visible_ranges, &track, 2);

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0], vec![0, 1, 2]);
    }

    #[test]
    fn pack_visible_annotation_rows_uses_multiple_rows_for_overlaps() {
        let track = track_with_names(&["abc1", "abc2", "abc3"]);
        let mut visible_ranges = AnnotationVisibleRanges::new();
        visible_ranges.insert(0, (10, 20));
        visible_ranges.insert(1, (18, 28));
        visible_ranges.insert(2, (50, 60));

        let rows = pack_visible_annotation_rows(&visible_ranges, &track, 2);

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0], vec![0, 2]);
        assert_eq!(rows[1], vec![1]);
    }

    #[test]
    fn pack_visible_annotation_rows_drops_annotations_when_rows_are_full() {
        let track = track_with_names(&["abc1", "abc2", "abc3"]);
        let mut visible_ranges = AnnotationVisibleRanges::new();
        visible_ranges.insert(0, (10, 20));
        visible_ranges.insert(1, (11, 21));
        visible_ranges.insert(2, (12, 22));

        let rows = pack_visible_annotation_rows(&visible_ranges, &track, 2);

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0], vec![0]);
        assert_eq!(rows[1], vec![1]);
        assert!(!rows.iter().flatten().any(|idx| *idx == 2));
    }
}
