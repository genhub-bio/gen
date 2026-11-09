//! A position scale for graph views that align a path on y = 0.
//!
//! The nodes of the aligned path sit side by side along one row, but how many columns a node
//! gets has little to do with how many bases it holds: a truncated node squeezes thousands of
//! bases into a few cells, a minimal one into a single cell, and a full one shows every base.
//! The scale therefore treats the path nodes in view as independent segments of a piecewise
//! axis, each mapping its own range of path positions onto its own columns.
//!
//! The label step is chosen once for the whole axis rather than per segment: the coarsest 1-2-5
//! step that still gets enough well-spaced labels across the view, the same at every detail
//! level. Only its multiples are labelled. There are no unlabelled ticks, since the number of
//! them between two labels would vary with the density of the nodes and would not help a
//! reader interpolate. Every viewer shares this through
//! `GenGraphController::render_to_buffer`.

use core::cmp::Reverse;

use gen_graph::{GraphNode, PackedGenGraph};
use gen_tui::{
    frame_index::FrameIndex,
    plotter::{ColumnRun, NodeRenderer},
    theme::current_theme,
};
use ratatui::{buffer::Buffer, layout::Rect, style::Style};

use crate::views::gen_graph_widget::AlignedPath;

/// Rows the scale draws: a line carrying the tick marks and, under it, the tick labels.
pub const SCALE_ROWS: u16 = 2;

/// Blank rows between the lowest node and the scale when it follows the content.
pub const SCALE_CONTENT_GAP: u16 = 2;

/// Blank cells required between two labels.
const LABEL_GAP: i64 = 6;

/// Columns of view that each wanted label takes, so a wide view is asked for more labels, up to
/// [`MOST_LABELS`].
const COLUMNS_PER_LABEL: i64 = 32;

/// The most labels a view aims for, so that a wide view stays uncrowded.
const MOST_LABELS: usize = 6;

/// How many candidate ticks per column of view a step may offer before the search for a finer
/// step stops. One label needs several columns, so a step that already puts a few candidates
/// in every column has offered every place a label could go, and finer steps would only
/// format millions of positions that fall in the same few columns.
const CANDIDATES_PER_COLUMN: i64 = 4;

/// The scale line: dots between marks and a pipe under each label. Edges are drawn with
/// box-drawing glyphs, so the scale is not mistaken for part of the graph.
const RAIL: &str = ".";
const TICK_MARK: &str = "|";

/// Where the scale sits in the area a viewer draws into.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ScalePlacement {
    /// On the last rows of the area, which is what an interactive viewer wants.
    #[default]
    Bottom,
    /// [`SCALE_CONTENT_GAP`] rows under the lowest node, for static images that crop to their
    /// content.
    BelowContent,
}

/// One stretch of the axis: path positions `domain_start..domain_end` drawn over the columns
/// `range_start..range_end`. The domain is half open, so a position on the boundary of two
/// adjacent segments belongs to the second.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AxisSegment {
    pub domain_start: i64,
    pub domain_end: i64,
    pub range_start: i64,
    pub range_end: i64,
}

impl AxisSegment {
    fn domain_length(&self) -> i64 {
        self.domain_end - self.domain_start
    }

    fn range_length(&self) -> i64 {
        self.range_end - self.range_start
    }

    /// The column that `position` is drawn in, which is always inside the segment.
    pub fn column_of(&self, position: i64) -> i64 {
        let fraction = (position - self.domain_start) as f64 / self.domain_length() as f64;
        let column = self.range_start + (fraction * self.range_length() as f64).floor() as i64;
        column.clamp(self.range_start, self.range_end - 1)
    }
}

/// A tick with its label.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tick {
    pub position: i64,
    pub column: i64,
    pub label: String,
    /// The column the label starts at, which is under the tick unless it was shifted to clear
    /// a neighbor or the edge of the view.
    pub label_start: i64,
}

impl Tick {
    fn label_width(&self) -> i64 {
        self.label.chars().count() as i64
    }
}

/// The segments of the aligned path in view, with the ticks drawn over them.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PathScale {
    pub segments: Vec<AxisSegment>,
    pub ticks: Vec<Tick>,
}

impl PathScale {
    /// Choose ticks for `segments`, keeping only those whose column is inside
    /// `visible_columns`. Segments must be sorted by column and not overlap.
    pub fn new(segments: Vec<AxisSegment>, visible_columns: (i64, i64)) -> Self {
        let (_, ticks) = choose_labels(&segments, visible_columns);
        Self { segments, ticks }
    }

    /// The scale for the path nodes of `frame`, whose rects are relative to the left of `area`.
    /// Each node's columns run from its first base to just past its last, as `renderer` draws
    /// it, so padding or an ellipsis in the node stretches the bases that are shown.
    pub fn from_frame(
        frame: &FrameIndex<GraphNode>,
        aligned_path: &AlignedPath,
        renderer: &impl NodeRenderer<PackedGenGraph>,
        area: Rect,
    ) -> Self {
        let mut segments: Vec<AxisSegment> = frame
            .visible_ids()
            .flat_map(|node| {
                let Some(domain_start) = aligned_path.path_position(&node) else {
                    return Vec::new();
                };
                let length = node.sequence_end - node.sequence_start;
                let Some(rect) = frame.rect_of(node) else {
                    return Vec::new();
                };
                let left = rect.left() + i64::from(area.x);
                // A renderer that doesn't override `map_column` returns the raw offset, which
                // can run past the cells the node is drawn in.
                let last_column = rect.right() - rect.left();
                let clamp_column = |column: i64| column.clamp(0, last_column);
                // A renderer that compresses stretches of the node into single cells says
                // where each stretch is drawn; otherwise the bases map evenly between the
                // columns of the first and last base.
                let runs = renderer.column_runs(&node).unwrap_or_else(|| {
                    vec![ColumnRun {
                        raw_start: 0,
                        raw_end: length,
                        column_start: renderer.map_column(&node, 0),
                        column_end: renderer.map_column(&node, length - 1) + 1,
                    }]
                });
                runs.into_iter()
                    .filter_map(|run| {
                        let range_start = left + clamp_column(run.column_start);
                        let range_end = left + clamp_column(run.column_end - 1) + 1;
                        (run.raw_end > run.raw_start && range_end > range_start).then_some(
                            AxisSegment {
                                domain_start: domain_start + run.raw_start,
                                domain_end: domain_start + run.raw_end,
                                range_start,
                                range_end,
                            },
                        )
                    })
                    .collect()
            })
            .collect();
        segments.sort_by_key(|segment| segment.range_start);
        let mut separate: Vec<AxisSegment> = Vec::with_capacity(segments.len());
        for segment in segments {
            if separate
                .last()
                .is_none_or(|previous| previous.range_end <= segment.range_start)
            {
                separate.push(segment);
            }
        }
        Self::new(
            separate,
            (i64::from(area.x), i64::from(area.x) + i64::from(area.width)),
        )
    }

    /// Draw the rail on `line_row` and the labels on the row under it. The rail runs across
    /// every segment and stops where the path jumps further than the columns between segments
    /// could hold.
    pub fn draw(&self, buffer: &mut Buffer, area: Rect, line_row: u16) {
        let theme = current_theme();
        let line_style = Style::default().fg(theme[0x04]);
        let label_style = Style::default().fg(theme[0x05]);
        let label_row = line_row + 1;
        let mut put = |column: i64, row: u16, symbol: &str, style: Style| {
            let Ok(column) = u16::try_from(column) else {
                return;
            };
            if column >= area.x && column < area.x + area.width && row < area.y + area.height {
                buffer[(column, row)].set_symbol(symbol).set_style(style);
            }
        };
        for (index, segment) in self.segments.iter().enumerate() {
            for column in segment.range_start..segment.range_end {
                put(column, line_row, RAIL, line_style);
            }
            let Some(next) = self.segments.get(index + 1) else {
                continue;
            };
            // Nodes drawn on the row but off the path (an allele replacing a path base) leave
            // a small domain gap that the columns between the segments account for. Skipping
            // more bases than those columns could hold is a real jump in the path.
            let skipped = next.domain_start - segment.domain_end;
            if skipped > next.range_start - segment.range_end {
                continue;
            }
            for column in segment.range_end..next.range_start {
                put(column, line_row, RAIL, line_style);
            }
        }
        for tick in &self.ticks {
            put(tick.column, line_row, TICK_MARK, line_style);
            for (offset, character) in tick.label.chars().enumerate() {
                put(
                    tick.label_start + offset as i64,
                    label_row,
                    character.encode_utf8(&mut [0; 4]),
                    label_style,
                );
            }
        }
    }
}

/// The next finer 1-2-5 step: 10 -> 5 -> 2 -> 1, or `None` below 1.
fn finer_step(step: i64) -> Option<i64> {
    let leading = leading_digit(step);
    let magnitude = step / leading;
    match leading {
        5 => Some(2 * magnitude),
        2 => Some(magnitude),
        _ if magnitude >= 10 => Some(5 * magnitude / 10),
        _ => None,
    }
}

fn leading_digit(value: i64) -> i64 {
    let mut leading = value.max(1);
    while leading >= 10 {
        leading /= 10;
    }
    leading
}

/// Multiples of `step` that fall in `start..end`.
fn multiples_in(start: i64, end: i64, step: i64) -> impl Iterator<Item = i64> {
    let first = (start + step - 1).div_euclid(step) * step;
    (first..end).step_by(step as usize)
}

/// The stretch of `segment`'s domain whose columns may be in `visible_columns`, widened by one
/// position on each side so that rounding never cuts off a tick that is in view.
fn visible_domain(segment: &AxisSegment, visible_columns: (i64, i64)) -> Option<(i64, i64)> {
    let first_column = visible_columns.0.max(segment.range_start);
    let last_column = visible_columns.1.min(segment.range_end);
    if first_column >= last_column {
        return None;
    }
    let positions_per_column = segment.domain_length() as f64 / segment.range_length() as f64;
    let start = segment.domain_start
        + ((first_column - segment.range_start) as f64 * positions_per_column).floor() as i64
        - 1;
    let end = segment.domain_start
        + ((last_column - segment.range_start) as f64 * positions_per_column).ceil() as i64
        + 1;
    Some((start.max(segment.domain_start), end.min(segment.domain_end)))
}

/// How many multiples of `step` lie in the visible stretch of `segments`.
fn candidate_count(segments: &[AxisSegment], step: i64, visible_columns: (i64, i64)) -> i64 {
    segments
        .iter()
        .filter_map(|segment| visible_domain(segment, visible_columns))
        .map(|(start, end)| multiples_in(start, end, step).count() as i64)
        .sum()
}

/// The ticks of `segments` on multiples of `step` that are in view. Only positions whose column
/// is in view are labelled, because formatting the text is what costs.
fn ticks_at_step(segments: &[AxisSegment], step: i64, visible_columns: (i64, i64)) -> Vec<Tick> {
    segments
        .iter()
        .filter_map(|segment| {
            let (start, end) = visible_domain(segment, visible_columns)?;
            Some((segment, start, end))
        })
        .flat_map(|(segment, start, end)| {
            multiples_in(start, end, step)
                .map(|position| (position, segment.column_of(position)))
                .filter(|(_, column)| *column >= visible_columns.0 && *column < visible_columns.1)
                .map(|(position, column)| Tick {
                    position,
                    column,
                    label: format_position(position, step),
                    label_start: 0,
                })
        })
        .collect()
}

/// Pick the label step for the whole axis and place its labels. Steps are tried from coarse
/// to fine, and the first that places as many labels as the width of the view calls for wins;
/// failing that, the step that placed the most. The search also ends once a step offers more
/// candidates than [`CANDIDATES_PER_COLUMN`] allows, which keeps a long sequence squeezed into
/// a few columns from being walked position by position. 1-2-5 steps jump by two to two and a half
/// times, so a view can land above the number it aimed for.
fn choose_labels(segments: &[AxisSegment], visible_columns: (i64, i64)) -> (i64, Vec<Tick>) {
    let wanted = (((visible_columns.1 - visible_columns.0) / COLUMNS_PER_LABEL).max(2) as usize)
        .min(MOST_LABELS);
    let largest = segments.iter().map(|segment| segment.domain_end).max();
    let coarsest = largest.map_or(1, |end| 10_i64.pow(end.max(1).ilog10() + 1));
    let mut step = coarsest;
    let mut best: (i64, Vec<Tick>) = (step, Vec::new());
    let most_candidates = (visible_columns.1 - visible_columns.0).max(1) * CANDIDATES_PER_COLUMN;
    loop {
        if candidate_count(segments, step, visible_columns) > most_candidates {
            return best;
        }
        let placed = place_labels(
            ticks_at_step(segments, step, visible_columns),
            visible_columns,
        );
        if placed.len() >= wanted {
            return (step, placed);
        }
        if placed.len() > best.1.len() {
            best = (step, placed);
        }
        match finer_step(step) {
            Some(finer) => step = finer,
            None => return best,
        }
    }
}

/// Write `position` compactly for ticks `step` apart: thousands as `k` below a million and
/// millions as `M` once the step is coarse enough that the suffix costs at most one decimal.
/// Anything else is written in full with thousands separators, so a coordinate deep into a
/// chromosome stays exact and readable.
pub fn format_position(position: i64, step: i64) -> String {
    let (unit, suffix) = if position.abs() >= 1_000_000 && step >= 100_000 {
        (1_000_000.0, "M")
    } else if (1_000..1_000_000).contains(&position.abs()) && step >= 100 {
        (1_000.0, "k")
    } else {
        return group_thousands(position);
    };
    let decimals = usize::from(step as f64 % unit != 0.0);
    let text = format!("{:.decimals$}", position as f64 / unit);
    let trimmed = if decimals > 0 {
        text.trim_end_matches('0').trim_end_matches('.')
    } else {
        text.as_str()
    };
    format!("{trimmed}{suffix}")
}

/// Write `value` with a comma between each group of three digits.
fn group_thousands(value: i64) -> String {
    let digits = value.unsigned_abs().to_string();
    let mut grouped = String::with_capacity(digits.len() + digits.len() / 3 + 1);
    if value < 0 {
        grouped.push('-');
    }
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            grouped.push(',');
        }
        grouped.push(digit);
    }
    grouped
}

/// How "round" a position is, so that when two labels collide the rounder one survives: a
/// boundary between a dense and a compressed segment then keeps 20k over 19.5k.
fn roundness(position: i64) -> i64 {
    if position == 0 {
        return i64::MAX;
    }
    let mut rest = position.abs();
    let mut zeros = 0;
    while rest % 10 == 0 {
        rest /= 10;
        zeros += 1;
    }
    2 * zeros + i64::from(rest % 5 == 0)
}

/// The first column of a label of `width` cells under `column`: centered when it fits in
/// `bounds`, otherwise shifted to stay inside.
fn label_start(column: i64, width: i64, bounds: (i64, i64)) -> i64 {
    let centered = column - width / 2;
    let latest = (bounds.1 - width).max(bounds.0);
    centered.clamp(bounds.0, latest)
}

/// Place the labels of `candidates` roundest first. Each takes the first of centered,
/// left-aligned or right-aligned that leaves [`LABEL_GAP`] blank cells to the labels already
/// placed and stays in view, and is dropped if none does.
fn place_labels(mut candidates: Vec<Tick>, visible_columns: (i64, i64)) -> Vec<Tick> {
    candidates.sort_by_key(|tick| (Reverse(roundness(tick.position)), tick.column));
    let mut placed: Vec<Tick> = Vec::new();
    for tick in candidates {
        let width = tick.label_width();
        let alignments = [
            label_start(tick.column, width, visible_columns),
            tick.column,
            tick.column + 1 - width,
        ];
        let free = |start: i64| {
            start >= visible_columns.0
                && start + width <= visible_columns.1
                && placed.iter().all(|other| {
                    let other_end = other.label_start + other.label_width();
                    start + width + LABEL_GAP <= other.label_start || other_end + LABEL_GAP <= start
                })
        };
        if let Some(start) = alignments.into_iter().find(|start| free(*start)) {
            placed.push(Tick {
                label_start: start,
                ..tick
            });
        }
    }
    placed.sort_by_key(|tick| tick.column);
    placed
}

#[cfg(test)]
mod tests {
    use ratatui::{buffer::Buffer, layout::Rect};

    use super::{AxisSegment, PathScale, finer_step, format_position};

    fn segment(domain: (i64, i64), range: (i64, i64)) -> AxisSegment {
        AxisSegment {
            domain_start: domain.0,
            domain_end: domain.1,
            range_start: range.0,
            range_end: range.1,
        }
    }

    /// A dense, a moderate and a heavily compressed segment, as a truncated view shows them.
    fn truncated_segments() -> Vec<AxisSegment> {
        vec![
            segment((0, 10), (2, 12)),
            segment((10, 10_010), (14, 24)),
            segment((10_010, 29_903), (26, 62)),
        ]
    }

    #[test]
    fn test_finer_step_walks_down_the_sequence() {
        let walk: Vec<i64> = std::iter::successors(Some(100), |step| finer_step(*step)).collect();
        assert_eq!(walk, [100, 50, 20, 10, 5, 2, 1]);
    }

    #[test]
    fn test_format_position_drops_zeros_and_uses_units() {
        assert_eq!(format_position(0, 5), "0");
        assert_eq!(format_position(1500, 50), "1,500");
        assert_eq!(format_position(2000, 1000), "2k");
        assert_eq!(format_position(2500, 500), "2.5k");
        assert_eq!(format_position(29_900, 100), "29.9k");
        assert_eq!(format_position(3_000_000, 1_000_000), "3M");
        assert_eq!(format_position(2_500_000, 500_000), "2.5M");
        assert_eq!(format_position(48_570_100, 100), "48,570,100");
        assert_eq!(format_position(999, 1), "999");
    }

    #[test]
    fn test_segment_maps_positions_into_its_columns() {
        let segments = truncated_segments();
        assert_eq!(segments[0].column_of(0), 2);
        assert_eq!(segments[0].column_of(9), 11);
        assert_eq!(segments[1].column_of(10), 14);
        assert_eq!(segments[1].column_of(5_010), 19);
        assert_eq!(segments[2].column_of(29_902), 61);
    }

    #[test]
    fn test_labels_share_one_step_across_segments() {
        let scale = PathScale::new(truncated_segments(), (0, 120));
        let positions: Vec<i64> = scale.ticks.iter().map(|tick| tick.position).collect();
        assert!(
            positions.len() >= 2,
            "should label several places: {positions:?}"
        );
        assert!(
            positions.iter().all(|position| position % 5000 == 0),
            "labels {positions:?} should all be multiples of 5000"
        );
    }

    #[test]
    fn test_labels_keep_a_wide_gap() {
        for width in [20, 40, 64, 120, 200] {
            let scale = PathScale::new(truncated_segments(), (0, width));
            let spans: Vec<(i64, i64)> = scale
                .ticks
                .iter()
                .map(|tick| (tick.label_start, tick.label_start + tick.label_width()))
                .collect();
            for pair in spans.windows(2) {
                assert!(
                    pair[0].1 + 6 <= pair[1].0,
                    "labels {pair:?} should have six blank cells between them at width {width}"
                );
            }
        }
    }

    #[test]
    fn test_a_squeezed_segment_with_no_round_position_gets_no_tick() {
        let scale = PathScale::new(
            vec![segment((0, 1000), (0, 60)), segment((1001, 1099), (64, 66))],
            (0, 70),
        );
        assert!(
            scale.ticks.iter().all(|tick| tick.column < 64),
            "should leave the squeezed segment bare: {:?}",
            scale.ticks
        );
    }

    #[test]
    fn test_draw_puts_the_rail_and_labels_on_two_rows() {
        let scale = PathScale::new(vec![segment((0, 100), (0, 40))], (0, 40));
        let area = Rect::new(0, 0, 40, 4);
        let mut buffer = Buffer::empty(area);
        scale.draw(&mut buffer, area, 2);
        let row = |index: u16| -> String {
            (0..40)
                .map(|column| buffer[(column, index)].symbol().to_string())
                .collect()
        };
        assert!(row(2).starts_with('|'), "rail row was {:?}", row(2));
        assert!(row(2).contains('.'));
        assert!(row(3).starts_with('0'), "label row was {:?}", row(3));
        assert!(row(0).trim().is_empty() && row(1).trim().is_empty());
    }

    #[test]
    fn test_rail_stops_at_a_jump_in_path_position() {
        let scale = PathScale {
            segments: vec![segment((0, 10), (0, 5)), segment((50, 60), (8, 12))],
            ticks: Vec::new(),
        };
        let area = Rect::new(0, 0, 12, 2);
        let mut buffer = Buffer::empty(area);
        scale.draw(&mut buffer, area, 0);
        assert_eq!(buffer[(6, 0)].symbol(), " ");
        assert_eq!(buffer[(2, 0)].symbol(), ".");
        assert_eq!(buffer[(9, 0)].symbol(), ".");
    }

    #[test]
    fn test_a_long_sequence_in_few_columns_stops_at_a_coarse_step() {
        let segments = vec![segment((0, 51_304_566), (0, 3))];

        let (step, ticks) = super::choose_labels(&segments, (0, 200));

        assert!(step >= 10_000, "should not descend to step {step}");
        assert!(!ticks.is_empty(), "should still label the axis");
        assert!(
            super::candidate_count(&segments, 1, (0, 200)) > 200 * super::CANDIDATES_PER_COLUMN,
            "step 1 should be refused as too many candidates"
        );
    }

    #[test]
    fn test_ticks_cover_only_the_visible_part_of_a_long_sequence() {
        let segments = vec![segment((0, 51_304_566), (0, 51_304_566))];

        let ticks = super::ticks_at_step(&segments, 10, (25_000_000, 25_000_100));

        assert_eq!(ticks.len(), 10);
        assert!(
            ticks
                .iter()
                .all(|tick| (25_000_000..25_000_100).contains(&tick.position))
        );
    }

    #[test]
    fn test_a_compressed_stretch_keeps_its_ticks_in_its_own_cell() {
        // A first base, a 16 Mb run drawn as one cell, and the base after it, as a compact
        // view draws a long node.
        let segments = vec![
            segment((0, 1), (0, 1)),
            segment((1, 16_000_000), (1, 2)),
            segment((16_000_000, 16_000_001), (2, 3)),
        ];

        let scale = PathScale::new(segments, (0, 10));

        assert!(!scale.ticks.is_empty(), "should label the axis");
        for tick in &scale.ticks {
            let expected = match tick.position {
                0 => 0,
                16_000_000 => 2,
                _ => 1,
            };
            assert_eq!(tick.column, expected, "tick at {}", tick.position);
        }
    }
}
