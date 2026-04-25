use std::{
    cmp::{max, min},
    collections::{HashMap, HashSet},
};

use gen_core::{HashId, is_end_node, is_start_node};
use gen_graph::GenGraph;
use gen_tui::{
    GraphController, ViewportState, WorldRect, plotter::NodeSizer, theme::current_theme,
};
use petgraph::visit::NodeIndexable;
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
};

use crate::{graphs::graph_search::GraphLocus, views::gen_graph_widget::GenGraphNodeSizer};

#[derive(Clone, Debug)]
pub struct AnnotationSegment {
    pub node_id: HashId,
    pub start: i64,
    pub end: i64,
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
}

pub fn graphlocus_to_annotation_span(locus: &GraphLocus, name: &str) -> AnnotationSpan {
    let n = locus.blocks.len();
    let segments = locus
        .blocks
        .iter()
        .enumerate()
        .map(|(i, node)| {
            let start = if i == 0 {
                node.sequence_start + locus.start_offset as i64
            } else {
                node.sequence_start
            };
            let end = if i == n - 1 {
                node.sequence_start + locus.end_offset as i64
            } else {
                node.sequence_end
            };
            AnnotationSegment {
                node_id: node.node_id,
                start,
                end,
            }
        })
        .collect();
    AnnotationSpan {
        id: HashId::convert_str(name),
        name: name.to_string(),
        segments,
    }
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
        }
    }
}

// ---------------------------------------------------------------------------
// Endpoint resolution
// ---------------------------------------------------------------------------

/// Bases from either node boundary that map to precise cells (0-indexed, inclusive).
/// Bases 0–4 from the start → cells 0–4; bases 0–4 from the end → cells 8–12.
/// Anything beyond falls in the `...` region (cells 5–7) and is marked truncated.
const BORDER_BP: i64 = 5;

/// Extra collection margin for labels that can remain visible outside the bar.
const MIN_LABEL_LOOKAHEAD: i64 = 8;

/// Preferred empty visual cells on both sides of an embedded label.
const LABEL_SIDE_GUTTER: i64 = 1;

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
        // Not truncated — proportional mapping (same formula as the original code).
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

    // Interior — exact position unknown.
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

            // Start endpoint: overlap_start.
            let seg_start = resolve_endpoint(
                overlap_start - block.sequence_start,
                block.sequence_end - 1 - overlap_start,
                x1,
                node_width,
                node_len,
            );

            // End endpoint: last included base = overlap_end - 1.
            let seg_end = resolve_endpoint(
                overlap_end - 1 - block.sequence_start,
                block.sequence_end - overlap_end,
                x1,
                node_width,
                node_len,
            );

            // Node in viewport range → annotation is visible.
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

/// Calculate the desired height for an annotation track panel.
pub fn annotation_panel_height(track: &AnnotationTrack, max_height: u16) -> u16 {
    if max_height < 2 {
        return 0;
    }

    if track.annotations.is_empty() {
        return if track.name.is_empty() {
            0
        } else {
            2.min(max_height)
        };
    }

    let desired = track.annotations.len().saturating_add(1) as u16;
    let cap = max_height.saturating_div(3).max(3);
    desired.min(cap).min(max_height)
}

/// Draw the annotation track panel below the graph canvas (Frame version).
pub fn draw_annotations_panel(
    frame: &mut ratatui::Frame,
    area: Rect,
    track: &AnnotationTrack,
    controller: &GraphController<GenGraph, GenGraphNodeSizer>,
) {
    draw_annotations_to_buffer(frame.buffer_mut(), area, track, controller);
}

/// Draw the annotation track panel below the graph canvas.
///
/// Writes directly to `buf` at the given `area`. Generic over the node-sizer
/// type so it works with both `GenGraphNodeSizer` and `AnnotationNodeSizer`.
pub fn draw_annotations_to_buffer<S: NodeSizer<GenGraph>>(
    buf: &mut Buffer,
    area: Rect,
    track: &AnnotationTrack,
    controller: &GraphController<GenGraph, S>,
) {
    if area.height < 2 {
        return;
    }

    let theme = current_theme();

    // Collect segments first so the truncated count is available for the header.
    let (visible_indices, segments_by_annotation, truncated_count) =
        collect_visual_segments(track, controller);

    let divider_style = Style::default().fg(theme[0x02]);
    let divider = "─".repeat(area.width as usize);
    buf.set_string(area.x, area.y, divider, divider_style);

    if !track.name.is_empty() {
        let header = if truncated_count > 0 {
            format!("{} (+{truncated_count} in truncated regions)", track.name)
        } else {
            track.name.clone()
        };

        buf.set_string(area.x + 1, area.y, header, Style::default().fg(theme[0x04]));
    }

    let inner = Rect {
        x: area.x,
        y: area.y + 1,
        width: area.width,
        height: area.height - 1,
    };

    if inner.height == 0 || inner.width == 0 || visible_indices.is_empty() {
        return;
    }

    let bg_color = theme[0x00];
    let bg_style = Style::default().bg(bg_color);

    for row in inner.y..inner.y + inner.height {
        let blank = " ".repeat(inner.width as usize);
        buf.set_string(inner.x, row, blank, bg_style);
    }

    let annotation_color = theme[0x0B];
    let annotation_label_style = Style::default().fg(annotation_color).bg(bg_color);
    let annotation_bar_style = Style::default().fg(annotation_color).bg(bg_color);

    // Label text when drawn over a full annotation bar. The bar is represented
    // by a full block glyph whose visible color is its foreground color, so the
    // overlay label uses that foreground as its bg.
    let label_fg_over_bar = bg_color;

    let viewport_state = &controller.viewport_state;
    let mut row = 0u16;

    for idx in &visible_indices {
        if row >= inner.height {
            break;
        }

        let Some(mut segments) = segments_by_annotation.get(idx).cloned() else {
            continue;
        };

        segments.sort_by_key(|s| s.start_x());

        let terminal_y = inner.y + row;
        row += 1;

        let annotation_name = &track.annotations[*idx].name;

        let mut prev_end: Option<i64> = None;

        // Draw bars/connectors/truncation markers first. The label is drawn
        // afterward so it can inspect and replace bar cells.
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
                annotation_bar_style,
            );

            // Overwrite the truncated endpoint cells with `…`.
            if seg.start.1.is_none() {
                draw_truncation_marker(
                    buf,
                    inner,
                    viewport_state,
                    seg.start.0,
                    terminal_y,
                    annotation_label_style,
                );
            }

            if seg.end.1.is_none() {
                draw_truncation_marker(
                    buf,
                    inner,
                    viewport_state,
                    seg.end.0,
                    terminal_y,
                    annotation_label_style,
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
                    annotation_label_style,
                );
            }

            prev_end = Some(x2);
        }

        if !annotation_name.is_empty() {
            let annotation_world_x1 = segments
                .iter()
                .map(VisualSegment::start_x)
                .min()
                .unwrap_or_default();

            let annotation_world_x2 = segments
                .iter()
                .map(VisualSegment::end_x)
                .max()
                .unwrap_or_default();

            place_centered_sticky_label(
                buf,
                inner,
                viewport_state,
                annotation_world_x1,
                annotation_world_x2,
                terminal_y,
                annotation_name,
                annotation_label_style,
                label_fg_over_bar,
            );
        }
    }
}

/// Convert a world X coordinate to a terminal X coordinate without requiring a
/// valid world Y. `world_to_terminal` gates on Y being inside the viewport,
/// which fails when the graph is scrolled far from world Y=0 and causes bars
/// to "flip" to the opposite screen edge via the `unwrap_or` fallback.
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
        buf.set_string(raw_x as u16, terminal_y, "…", style);
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
fn place_centered_sticky_label(
    buf: &mut Buffer,
    inner: Rect,
    viewport_state: &ViewportState,
    annotation_world_x1: i64,
    annotation_world_x2: i64,
    terminal_y: u16,
    label: &str,
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
    let annotation_width = annotation_end - annotation_start + 1;

    // If the feature itself is shorter than the label plus a one-cell gutter
    // on both sides, drawing the label inside the bar mostly hides the feature.
    // Keep those short-feature labels outside, pinned to the left of the bar.
    if annotation_width < label_len + LABEL_SIDE_GUTTER * 2 {
        place_left_external_label(
            buf,
            inner,
            viewport_state,
            annotation_start,
            terminal_y,
            label,
            normal_style,
        );
        return;
    }

    let gutter = LABEL_SIDE_GUTTER;
    let virtual_left = annotation_start - label_len - gutter;
    let virtual_right = annotation_end + gutter + label_len;

    let virtual_left_term = world_x_to_term_x(viewport_state, virtual_left);
    let virtual_right_term = world_x_to_term_x(viewport_state, virtual_right);
    let virtual_center_term = (virtual_left_term + virtual_right_term) / 2;

    let viewport_center = (area_left + area_right) / 2;
    let pinned_start = viewport_center - label_len / 2;
    let pinned_end = pinned_start + label_len - 1;

    let final_start = if virtual_center_term < viewport_center {
        virtual_left_term
    } else if virtual_right_term >= pinned_end {
        pinned_start
    } else {
        virtual_right_term - label_len + 1
    };

    draw_label_clipped_over_existing(
        buf,
        inner,
        final_start,
        terminal_y,
        label,
        normal_style,
        label_fg_over_bar,
    );
}

fn place_left_external_label(
    buf: &mut Buffer,
    inner: Rect,
    viewport_state: &ViewportState,
    annotation_start_world_x: i64,
    terminal_y: u16,
    label: &str,
    normal_style: Style,
) {
    // TODO: this assumes one terminal cell per char. If annotation names can
    // contain wide Unicode or combining marks, switch to unicode-width and
    // grapheme-aware clipping/writing.
    let label_len = label.chars().count() as i64;

    if label_len == 0 || inner.width == 0 {
        return;
    }

    let label_start_world_x = annotation_start_world_x - label_len - 1;
    let label_start_term_x = world_x_to_term_x(viewport_state, label_start_world_x);

    draw_label_clipped_preserving_spaces(
        buf,
        inner,
        label_start_term_x,
        terminal_y,
        label,
        normal_style,
    );
}

fn draw_label_clipped_preserving_spaces(
    buf: &mut Buffer,
    inner: Rect,
    start_x: i64,
    terminal_y: u16,
    label: &str,
    style: Style,
) {
    let area_left = inner.x as i64;
    let area_right = (inner.x + inner.width - 1) as i64;

    if area_right < area_left {
        return;
    }

    for (i, ch) in label.chars().enumerate() {
        if ch == ' ' {
            continue;
        }

        let x = start_x + i as i64;

        if x < area_left || x > area_right {
            continue;
        }

        buf.set_string(x as u16, terminal_y, ch.to_string(), style);
    }
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

        // Snapshot the existing cell before replacing it.
        let Some(old_cell) = buf.cell((x, terminal_y)) else {
            continue;
        };

        let old_symbol = old_cell.symbol();

        let style = if old_symbol == "█" {
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
