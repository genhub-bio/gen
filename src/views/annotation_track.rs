use std::{
    cmp::{max, min},
    collections::HashMap,
};

use gen_core::{HashId, Strand, is_end_node, is_start_node};
use gen_graph::GenGraph;
use gen_tui::{
    CroppedGraph, GraphController, ViewportState, VisualDetail, WorldPos, WorldRect,
    theme::current_theme,
};
use petgraph::visit::NodeIndexable;
use ratatui::{layout::Rect, style::Style};

use crate::views::gen_graph_widget::GenGraphNodeSizer;

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

type AnnotationSegmentsByIndex = HashMap<usize, Vec<(i64, i64)>>;
type AnnotationVisibleRanges = HashMap<usize, (i64, i64)>;
type AnnotationSegmentsResult = (AnnotationVisibleRanges, AnnotationSegmentsByIndex);

/// Collect visible annotation segments by mapping sequence coordinates to world X coordinates.
///
/// This is the gen-tui equivalent of the old `Viewer::collect_annotation_segments`. Instead of
/// reading from `scaled_layout.labels`, it iterates over visible Data nodes in the CroppedGraph
/// and uses `WorldPos` + `LayoutNode.size` to determine each node's X span in world coordinates.
fn collect_annotation_segments(
    track: &AnnotationTrack,
    viewport_graph: &CroppedGraph,
    viewport_state: &ViewportState,
    graph: &GenGraph,
) -> AnnotationSegmentsResult {
    if track.annotations.is_empty() {
        return (HashMap::new(), HashMap::new());
    }

    const HORIZONTAL_LOOKAHEAD: i64 = 8;
    let mut visible_ranges: AnnotationVisibleRanges = HashMap::new();
    let mut segments_by_annotation: AnnotationSegmentsByIndex = HashMap::new();

    let camera_rect = viewport_state.camera_rect();
    let window_start = camera_rect.min.x;
    let window_end = camera_rect.max.x;
    let left_bound = window_start - HORIZONTAL_LOOKAHEAD;
    let right_bound = window_end + HORIZONTAL_LOOKAHEAD;

    for (world_pos, domain_idx, layout_node) in viewport_graph.data_nodes() {
        // Resolve the domain GraphNode from the index
        let block = <&GenGraph as NodeIndexable>::from_index(&graph, domain_idx.index());

        if is_start_node(block.node_id) || is_end_node(block.node_id) {
            continue;
        }

        // Compute the node's world-space X range from its center position and size
        let node_rect = WorldRect::from_center_and_size(world_pos, layout_node.size);
        let x1 = node_rect.min.x;
        let x2 = node_rect.max.x;

        let near_horizontally = x2 >= left_bound && x1 <= right_bound;
        if !near_horizontally {
            continue;
        }

        let Some(segments) = track.annotation_segments_by_node.get(&block.node_id) else {
            continue;
        };

        let node_len = block.sequence_end - block.sequence_start;
        if node_len <= 0 {
            continue;
        }

        // label_len is the inclusive width of the node in world cells
        let label_len = x2 - x1;
        if label_len <= 0 {
            continue;
        }

        for (idx, segment) in segments {
            let overlap_start = max(segment.start, block.sequence_start);
            let overlap_end = min(segment.end, block.sequence_end);
            if overlap_end <= overlap_start {
                continue;
            }

            // Map sequence coordinates to world X coordinates using relative positioning
            // (same algorithm as the annotations branch, but with integer world coordinates)
            let seg_x1 = x1 + (overlap_start - block.sequence_start) * label_len / node_len;
            let seg_x2 = x1 + (overlap_end - block.sequence_start) * label_len / node_len;
            let (seg_x1, seg_x2) = if seg_x2 < seg_x1 {
                (seg_x2, seg_x1)
            } else {
                (seg_x1, seg_x2)
            };

            let is_on_screen = seg_x2 >= window_start && seg_x1 <= window_end;
            if is_on_screen {
                visible_ranges
                    .entry(*idx)
                    .and_modify(|range| {
                        range.0 = range.0.min(seg_x1);
                        range.1 = range.1.max(seg_x2);
                    })
                    .or_insert((seg_x1, seg_x2));
            }

            segments_by_annotation
                .entry(*idx)
                .or_default()
                .push((seg_x1, seg_x2));
        }
    }

    (visible_ranges, segments_by_annotation)
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
        let occupied_end = *end;

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

/// Draw the annotation track panel below the graph canvas.
///
/// This is the gen-tui equivalent of the old `Viewer::draw_annotations_panel`. Instead of
/// rendering through ratatui's `Canvas` widget with braille coordinates, it writes directly
/// to the terminal buffer. The X axis is synchronized with the graph's viewport by converting
/// world X coordinates to terminal X via `ViewportState::world_to_terminal`.
pub fn draw_annotations_panel(
    frame: &mut ratatui::Frame,
    area: Rect,
    track: &AnnotationTrack,
    controller: &GraphController<GenGraph, GenGraphNodeSizer>,
) {
    if area.height < 2 {
        return;
    }

    let theme = current_theme();

    // Draw the divider line with track name
    let divider_style = Style::default().fg(theme[0x02]);
    let divider_y = area.y;
    let divider = "─".repeat(area.width as usize);
    frame
        .buffer_mut()
        .set_string(area.x, divider_y, divider, divider_style);

    if !track.name.is_empty() {
        frame.buffer_mut().set_string(
            area.x + 1,
            divider_y,
            &track.name,
            Style::default().fg(theme[0x04]),
        );
    }

    let inner = Rect {
        x: area.x,
        y: area.y + 1,
        width: area.width,
        height: area.height - 1,
    };

    if inner.height == 0 || inner.width == 0 {
        return;
    }

    // Fill background
    let bg_color = theme[0x00];
    let bg_style = Style::default().bg(bg_color);
    for row in inner.y..inner.y + inner.height {
        let blank = " ".repeat(inner.width as usize);
        frame.buffer_mut().set_string(inner.x, row, blank, bg_style);
    }

    let zoomed_out = controller.get_detail_level() == VisualDetail::Minimal;
    let annotation_color = theme[0x0B];
    let annotation_label_style = Style::default().fg(annotation_color).bg(bg_color);
    let annotation_bar_style = Style::default().bg(annotation_color);
    let annotation_dot_style = Style::default().fg(annotation_color).bg(bg_color);

    let (visible_ranges, segments_by_annotation) = collect_annotation_segments(
        track,
        controller.get_viewport_graph(),
        &controller.viewport_state,
        controller.graph(),
    );
    let packed_rows = pack_visible_annotation_rows(&visible_ranges, track, inner.height as usize);
    if packed_rows.is_empty() {
        return;
    }

    let viewport_state = &controller.viewport_state;

    for (row, row_annotations) in packed_rows.iter().enumerate() {
        // Each annotation row is drawn at a fixed terminal Y within the inner rect
        let terminal_y = inner.y + row as u16;

        for idx in row_annotations {
            let Some(mut segments) = segments_by_annotation.get(idx).cloned() else {
                continue;
            };
            segments.sort_by(|a, b| a.0.cmp(&b.0));

            let annotation_name = &track.annotations[*idx].name;

            // Draw the label to the left of the first segment
            if let Some((first_x1, _)) = segments.first() {
                let label_len = annotation_name.chars().count() as i64;
                let label_world_x = first_x1 - label_len - 1;
                if let Some((term_x, _)) =
                    viewport_state.world_to_terminal(WorldPos::new(label_world_x, 0))
                {
                    // Clamp label to the panel area
                    let label_start = term_x.max(inner.x);
                    if label_start < inner.x + inner.width {
                        frame.buffer_mut().set_string(
                            label_start,
                            terminal_y,
                            annotation_name,
                            annotation_label_style,
                        );
                    }
                }
            }

            let mut prev_end: Option<i64> = None;
            for (x1, x2) in &segments {
                if zoomed_out {
                    // Draw a single dot at the center
                    let center = (x1 + x2) / 2;
                    if let Some((term_x, _)) =
                        viewport_state.world_to_terminal(WorldPos::new(center, 0))
                        && term_x >= inner.x
                        && term_x < inner.x + inner.width
                    {
                        frame.buffer_mut().set_string(
                            term_x,
                            terminal_y,
                            "●",
                            annotation_dot_style,
                        );
                    }
                } else {
                    // Draw a solid bar for the segment
                    place_bar(
                        frame,
                        inner,
                        viewport_state,
                        *x1,
                        *x2,
                        terminal_y,
                        annotation_bar_style,
                    );
                }

                // Draw dashed connectors between disconnected segments of the same annotation
                if !zoomed_out
                    && let Some(prev) = prev_end
                    && x1 - prev > 1
                {
                    draw_dashed_connector(
                        frame,
                        inner,
                        viewport_state,
                        prev + 1,
                        x1 - 1,
                        terminal_y,
                        annotation_label_style,
                    );
                }

                prev_end = Some(*x2);
            }
        }
    }
}

/// Draw a solid bar from world x1 to world x2 at the given terminal y, clipped to the inner rect.
fn place_bar(
    frame: &mut ratatui::Frame,
    inner: Rect,
    viewport_state: &ViewportState,
    world_x1: i64,
    world_x2: i64,
    terminal_y: u16,
    style: Style,
) {
    // Convert world X endpoints to terminal X
    let term_x1 = viewport_state
        .world_to_terminal(WorldPos::new(world_x1, 0))
        .map(|(x, _)| x);
    let term_x2 = viewport_state
        .world_to_terminal(WorldPos::new(world_x2, 0))
        .map(|(x, _)| x);

    // Fall back to inner rect edges if off-screen
    let start_x = term_x1.unwrap_or(inner.x).max(inner.x);
    let end_x = term_x2
        .unwrap_or(inner.x + inner.width - 1)
        .min(inner.x + inner.width - 1);

    if start_x > end_x {
        return;
    }

    let width = (end_x - start_x + 1) as usize;
    let bar = " ".repeat(width);
    frame
        .buffer_mut()
        .set_string(start_x, terminal_y, bar, style);
}

/// Draw a dashed connector from world x_start to world x_end at the given terminal y.
fn draw_dashed_connector(
    frame: &mut ratatui::Frame,
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

    let term_x1 = viewport_state
        .world_to_terminal(WorldPos::new(world_x_start, 0))
        .map(|(x, _)| x);
    let term_x2 = viewport_state
        .world_to_terminal(WorldPos::new(world_x_end, 0))
        .map(|(x, _)| x);

    let start_x = term_x1.unwrap_or(inner.x).max(inner.x);
    let end_x = term_x2
        .unwrap_or(inner.x + inner.width - 1)
        .min(inner.x + inner.width - 1);

    if start_x > end_x {
        return;
    }

    let visible_width = (end_x - start_x + 1) as usize;
    // Compute the offset into the dash pattern based on how far we are from the start
    let pattern_offset =
        (start_x as i64 - term_x1.unwrap_or(start_x) as i64).unsigned_abs() as usize;

    let mut label = String::with_capacity(visible_width);
    for i in 0..visible_width {
        if (pattern_offset + i).is_multiple_of(2) {
            label.push('-');
        } else {
            label.push(' ');
        }
    }

    frame
        .buffer_mut()
        .set_string(start_x, terminal_y, label, style);
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
