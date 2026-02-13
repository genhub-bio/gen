use std::{
    cmp::{max, min},
    collections::{HashMap, HashSet},
};

use gen_core::{HashId, is_end_node, is_start_node};
use ratatui::{
    layout::Rect,
    style::{Color, Style},
    widgets::canvas::{Canvas, Points},
};

use crate::{config::get_theme_color, views::block_group_viewer::Viewer};

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
    annotation_segments_by_node: HashMap<HashId, Vec<(usize, AnnotationSegment)>>,
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

type AnnotationSegmentsByIndex = HashMap<usize, Vec<(f64, f64)>>;
type AnnotationSegmentsResult = (Vec<usize>, AnnotationSegmentsByIndex);

impl<'a> Viewer<'a> {
    fn collect_annotation_segments(&self, track: &AnnotationTrack) -> AnnotationSegmentsResult {
        if track.annotations.is_empty() {
            return (Vec::new(), HashMap::new());
        }

        const HORIZONTAL_LOOKAHEAD: f64 = 8.0;
        let mut visible_indices = Vec::new();
        let mut visible_index_set = HashSet::new();
        let mut segments_by_annotation: AnnotationSegmentsByIndex = HashMap::new();
        let window_start = self.state.offset_x as f64;
        let window_end = window_start + self.state.viewport.width as f64 - 0.5;
        let left_bound = window_start - HORIZONTAL_LOOKAHEAD;
        let right_bound = window_end + HORIZONTAL_LOOKAHEAD;
        let y_min = self.state.offset_y;
        let y_max = self.state.offset_y + self.state.viewport.height as i32;

        for &block in self.scaled_layout.labels.keys() {
            if is_start_node(block.node_id) || is_end_node(block.node_id) {
                continue;
            }

            let Some(&((x1, y), (x2, _))) = self.scaled_layout.labels.get(&block) else {
                continue;
            };
            let y_visible = (y as i32) >= y_min && (y as i32) < y_max;
            let near_horizontally = x2 >= left_bound && x1 <= right_bound;
            if !y_visible || !near_horizontally {
                continue;
            }

            let Some(segments) = track.annotation_segments_by_node.get(&block.node_id) else {
                continue;
            };

            let node_len = block.sequence_end - block.sequence_start;
            if node_len <= 0 {
                continue;
            }

            let label_len = x2 - x1;
            for (idx, segment) in segments {
                let overlap_start = max(segment.start, block.sequence_start);
                let overlap_end = min(segment.end, block.sequence_end);
                if overlap_end <= overlap_start {
                    continue;
                }

                // We do the swap here to ensure seg_x1 is always the left bound, and seg_x2 is always right bound
                let relative_start =
                    (overlap_start - block.sequence_start) as f64 / node_len as f64;
                let relative_end = (overlap_end - block.sequence_start) as f64 / node_len as f64;
                let mut seg_x1 = x1 + relative_start * label_len;
                let mut seg_x2 = x1 + relative_end * label_len;
                if seg_x2 < seg_x1 {
                    std::mem::swap(&mut seg_x1, &mut seg_x2);
                }
                let is_on_screen = seg_x2 >= window_start && seg_x1 <= window_end;
                if is_on_screen && visible_index_set.insert(*idx) {
                    visible_indices.push(*idx);
                }

                segments_by_annotation
                    .entry(*idx)
                    .or_default()
                    .push((seg_x1, seg_x2));
            }
        }

        visible_indices.sort_unstable();
        (visible_indices, segments_by_annotation)
    }

    pub fn annotation_panel_height(&self, track: &AnnotationTrack, max_height: u16) -> u16 {
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

    pub fn draw_annotations_panel(
        &self,
        frame: &mut ratatui::Frame,
        area: Rect,
        track: &AnnotationTrack,
    ) {
        if area.height < 2 {
            return;
        }

        let divider_style = Style::default().fg(get_theme_color("separator").unwrap());
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
                Style::default().fg(get_theme_color("text_muted").unwrap()),
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

        let x_min = self.state.offset_x as f64;
        let x_max = x_min + inner.width as f64 - 1.0 + 1.0 / 2.0;
        let y_min = 0.0;
        let y_max = inner.height as f64 - 1.0 + 3.0 / 4.0;

        let zoomed_out = self.parameters.label_width < 5;
        let annotation_color = get_theme_color("base0b").unwrap_or(Color::Green);
        let annotation_label_style = Style::default().fg(annotation_color);
        let annotation_bar_style = Style::default().bg(annotation_color);

        let (visible_indices, segments_by_annotation) = self.collect_annotation_segments(track);
        if visible_indices.is_empty() {
            return;
        }

        let max_rows = inner.height as usize;
        let row_count = visible_indices.len().min(max_rows);

        let canvas = Canvas::default()
            .background_color(get_theme_color("canvas").unwrap())
            .x_bounds([x_min, x_max])
            .y_bounds([y_min, y_max])
            .paint(|ctx| {
                for (row, idx) in visible_indices.iter().take(row_count).enumerate() {
                    let Some(mut segments) = segments_by_annotation.get(idx).cloned() else {
                        continue;
                    };
                    segments.sort_by(|a, b| a.0.total_cmp(&b.0));
                    let y = (inner.height as i64 - 1 - row as i64) as f64;
                    let annotation_name = &track.annotations[*idx].name;
                    if let Some((first_x1, _)) = segments.first() {
                        let label_offset = annotation_name.chars().count() as f64 + 1.0;
                        let label_x = first_x1 - label_offset;
                        self.place_label(
                            ctx,
                            annotation_name,
                            (label_x, y),
                            annotation_label_style,
                        );
                    }

                    let mut prev_end: Option<f64> = None;
                    for (x1, x2) in segments {
                        if zoomed_out {
                            let center = (x1 + x2) / 2.0;
                            ctx.draw(&Points {
                                coords: &[(center, y)],
                                color: annotation_color,
                            });
                        } else {
                            let start_cell = x1.floor();
                            let end_cell = x2.ceil();
                            let width = ((end_cell - start_cell).max(1.0)) as usize;
                            self.place_bar(ctx, start_cell, y, width, annotation_bar_style);
                        }

                        if !zoomed_out
                            && let Some(prev) = prev_end
                            && x1 - prev > 1.0
                        {
                            self.draw_dashed_connector(
                                ctx,
                                prev + 1.0,
                                x1 - 1.0,
                                y,
                                annotation_label_style,
                            );
                        }

                        prev_end = Some(x2);
                    }
                }
            });

        frame.render_widget(canvas, inner);
    }
}
