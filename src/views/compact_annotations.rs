//! Shared annotation geometry with label-aware sequence compression for compact detail.

use std::collections::{BTreeSet, HashMap, HashSet};

use gen_core::{HashId, Strand};
use gen_graph::GraphNode;
use gen_tui::plotter::ColumnRun;
use ratatui::style::Color;
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

use crate::views::gen_graph_widget::{
    ANNOTATION_BAR, ANNOTATION_FORWARD_CAP, ANNOTATION_REVERSE_CAP, ANNOTATION_SINGLE_BASE_MARK,
    AnnotationFlag,
};

/// Whether an end glyph marks where the feature starts or ends, as opposed to the bar that
/// continues into a neighbouring node.
pub(crate) fn is_cap(glyph: char) -> bool {
    glyph != ANNOTATION_BAR
}

/// The glyph a piece draws in `column`. A piece one cell wide holds both of its ends in one
/// cell: a cap always beats the bar that continues into the neighbouring node, and when the cell
/// holds a start and an end cap the directional one stays, since it also tells the strand
/// (`>` forward, `<` reverse). A feature with no strand shows `|`, since neither `[` nor `]`
/// alone would tell the reader to look for the other end.
pub(crate) fn piece_glyph(piece: &AnnotationPiece, column: i64) -> char {
    if piece.left == piece.right {
        return match (is_cap(piece.left_glyph), is_cap(piece.right_glyph)) {
            (false, false) => ANNOTATION_BAR,
            (true, false) => piece.left_glyph,
            (false, true) => piece.right_glyph,
            (true, true) if piece.right_glyph == ANNOTATION_FORWARD_CAP => piece.right_glyph,
            (true, true) if piece.left_glyph == ANNOTATION_REVERSE_CAP => piece.left_glyph,
            (true, true) => ANNOTATION_SINGLE_BASE_MARK,
        };
    }
    if column == piece.left {
        piece.left_glyph
    } else if column == piece.right {
        piece.right_glyph
    } else {
        ANNOTATION_BAR
    }
}

/// Whether `incoming` may replace `existing` in a cell two pieces both draw into: caps are never
/// overdrawn by a bar, and the first of two caps stays.
pub(crate) fn replaces_glyph(existing: char, incoming: char) -> bool {
    !is_cap(existing) && is_cap(incoming)
}

#[derive(Clone, Copy)]
pub(crate) enum AnnotationDetail {
    Compact,
    Full,
}

/// One arrow and its optional name, packed together in display coordinates.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct AnnotationPiece {
    pub id: HashId,
    pub piece: usize,
    pub color: Color,
    pub row: usize,
    /// Row this piece's annotation held on the previous node of the same layer position, which
    /// packing keeps even where that leaves a gap.
    pub pinned_row: Option<usize>,
    pub label_left: bool,
    /// A name wider than the node occupies its own row below the bar.
    pub label_below: bool,
    pub left: i64,
    pub right: i64,
    pub label: Option<String>,
    pub left_glyph: char,
    pub right_glyph: char,
}

impl AnnotationPiece {
    pub fn new(
        flag: &AnnotationFlag,
        left: i64,
        right: i64,
        label: Option<String>,
        node_width: i64,
    ) -> Self {
        let width = label
            .as_deref()
            .map_or(0, |text| UnicodeWidthStr::width(text) as i64);
        let label_below = label.is_some() && width > node_width;
        let label_left = label.is_some() && !label_below && width > right - left - 1;
        let (left_glyph, right_glyph) =
            end_glyphs(flag.strand, flag.continues_left, flag.continues_right);
        Self {
            id: flag.id,
            piece: flag.piece,
            color: flag.color,
            row: 0,
            pinned_row: None,
            label_left,
            label_below,
            left,
            right,
            label,
            left_glyph,
            right_glyph,
        }
    }

    pub fn label_width(&self) -> i64 {
        self.label
            .as_deref()
            .map_or(0, |text| UnicodeWidthStr::width(text) as i64)
    }
}

/// Geometry shared by sequence painting, annotation painting, and cursor navigation.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct NodeAnnotationLayout {
    /// Retained node-local nucleotide offsets, in sequence order.
    pub events: Vec<i64>,
    /// Display columns corresponding to `events`, before horizontal padding.
    pub columns: Vec<i64>,
    pub base_markers: Vec<i64>,
    /// Space before the sequence for labels that overhang to the left.
    pub left_padding: i64,
    pub width: i64,
    /// Odd height with equal space above and below the sequence row for edge alignment.
    pub height: usize,
    pub pieces: Vec<AnnotationPiece>,
}

impl NodeAnnotationLayout {
    pub fn empty(length: i64) -> Self {
        let mut events = if length <= 3 {
            (0..length).collect()
        } else {
            vec![0, length - 1]
        };
        events.sort_unstable();
        let columns = compact_columns(&events);
        Self {
            width: columns.last().copied().unwrap_or(0) + 1,
            columns,
            events,
            height: 1,
            ..Self::default()
        }
    }

    pub fn map_column(&self, raw: i64) -> i64 {
        if self.events.is_empty() {
            return 0;
        }
        let index = self.events.partition_point(|event| *event < raw);
        let column = if index == 0 {
            self.columns[0]
        } else if index == self.events.len() {
            self.columns[index - 1]
        } else if self.events[index] == raw {
            self.columns[index]
        } else {
            // Each skipped run has one period between its retained endpoint bases.
            self.columns[index - 1] + 1
        };
        self.left_padding + column
    }

    /// Each retained base in its own column and each skipped run in the one cell drawn for it,
    /// in offset order, so a scale can place positions where the bases are drawn.
    pub fn column_runs(&self) -> Vec<ColumnRun> {
        let mut runs = Vec::with_capacity(self.events.len() * 2);
        for (index, (&event, &column)) in self.events.iter().zip(&self.columns).enumerate() {
            if index > 0 {
                let previous = self.events[index - 1];
                if event - previous > 1 {
                    let skipped_column = self.left_padding + self.columns[index - 1] + 1;
                    runs.push(ColumnRun {
                        raw_start: previous + 1,
                        raw_end: event,
                        column_start: skipped_column,
                        column_end: skipped_column + 1,
                    });
                }
            }
            runs.push(ColumnRun {
                raw_start: event,
                raw_end: event + 1,
                column_start: self.left_padding + column,
                column_end: self.left_padding + column + 1,
            });
        }
        runs
    }

    pub fn raw_column(&self, column: i64) -> i64 {
        if self.events.is_empty() {
            return 0;
        }
        let index = self
            .columns
            .partition_point(|value| *value < column - self.left_padding);
        self.events[index.min(self.events.len() - 1)]
    }
}

/// A skipped run occupies one period cell; retained nucleotides keep their own columns.
fn compact_columns(events: &[i64]) -> Vec<i64> {
    let mut columns = Vec::with_capacity(events.len());
    let mut column = 0;
    for (index, event) in events.iter().enumerate() {
        if index > 0 {
            column += if *event - events[index - 1] > 1 { 2 } else { 1 };
        }
        columns.push(column);
    }
    columns
}

// Preserve a little sequence context even when no annotation ends on this slice.
fn sequence_anchors(length: i64) -> Vec<i64> {
    if length <= 3 {
        (0..length).collect()
    } else {
        vec![0, length - 1]
    }
}

/// Keep enough real bases at both ends to fit a requested label between the arrow caps.
/// Short segments retain the compact fallback rather than inventing sequence columns.
fn label_anchors(flag: &AnnotationFlag, label: &str) -> Vec<i64> {
    let width = UnicodeWidthStr::width(label) as i64;
    let required = width + 2;
    let length = flag.bar_end - flag.bar_start;
    if width == 0 || length < required {
        return Vec::new();
    }
    let retained = if length > required {
        required - 1
    } else {
        length
    };
    let left_count = (retained + 1) / 2;
    let right_count = retained / 2;
    (flag.bar_start..flag.bar_start + left_count)
        .chain(flag.bar_end - right_count..flag.bar_end)
        .collect()
}

fn end_glyphs(strand: Strand, continues_left: bool, continues_right: bool) -> (char, char) {
    match strand {
        Strand::Forward => (
            if continues_left { ANNOTATION_BAR } else { '[' },
            if continues_right {
                ANNOTATION_BAR
            } else {
                ANNOTATION_FORWARD_CAP
            },
        ),
        Strand::Reverse => (
            if continues_left {
                ANNOTATION_BAR
            } else {
                ANNOTATION_REVERSE_CAP
            },
            if continues_right { ANNOTATION_BAR } else { ']' },
        ),
        _ => (
            if continues_left { ANNOTATION_BAR } else { '[' },
            if continues_right { ANNOTATION_BAR } else { ']' },
        ),
    }
}

/// Shorten at grapheme boundaries while counting terminal cells, including the period.
fn short_name(name: &str, limit: usize) -> String {
    let width = UnicodeWidthStr::width(name);
    if width <= limit {
        return name.to_owned();
    }
    if limit == 0 {
        return String::new();
    }
    let mut result = String::new();
    let mut used = 0;
    for grapheme in name.graphemes(true) {
        let cells = UnicodeWidthStr::width(grapheme);
        if used + cells > limit - 1 {
            break;
        }
        result.push_str(grapheme);
        used += cells;
    }
    result.push('.');
    result
}

pub(crate) fn label_start(left: i64, right: i64, width: i64) -> i64 {
    (left + right + 1 - width).div_euclid(2)
}

pub(crate) fn piece_label_start(piece: &AnnotationPiece, width: i64) -> i64 {
    if piece.label_left {
        piece.left - width
    } else {
        label_start(piece.left, piece.right, width)
    }
}

/// Node-local columns currently on screen, both ends inclusive.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct VisibleColumns {
    pub left: i64,
    pub right: i64,
}

/// Where a label is drawn once the viewport is known: its normal spot, held inside the visible
/// part of its flag. A long flag whose normal spot has panned off-screen keeps its name pinned
/// to the nearest visible edge, and since the clamp does nothing while the normal spot is in
/// view, the label does not jump as panning moves between the two regimes. The columns of an
/// end that is on screen stay free for its cap, and a cut-off end shows no cap, which tells the
/// reader the name belongs to a flag running past the edge. An external label whose flag start
/// is on screen, or a visible part too short for the name, keeps the normal spot.
pub(crate) fn visible_label_start(
    piece: &AnnotationPiece,
    width: i64,
    visible: VisibleColumns,
) -> i64 {
    let normal_start = piece_label_start(piece, width);
    if piece.label_below {
        return normal_start;
    }
    let left_end_visible = piece.left >= visible.left;
    let right_end_visible = piece.right <= visible.right;
    if piece.label_left && left_end_visible {
        return normal_start;
    }
    let usable_left = piece.left.max(visible.left) + i64::from(left_end_visible);
    let usable_right = piece.right.min(visible.right) - i64::from(right_end_visible);
    if usable_right - usable_left + 1 < width {
        return normal_start;
    }
    normal_start.clamp(usable_left, usable_right + 1 - width)
}

/// The order nodes are laid out in: the usual order of first piece, except that a node always
/// follows the nodes holding the preceding pieces of its annotations, so their rows are known
/// when its own are chosen.
fn layout_order(
    nodes: &[(&GraphNode, &Vec<AnnotationFlag>)],
    piece_nodes: &HashMap<(HashId, usize), GraphNode>,
) -> Vec<usize> {
    let index_of: HashMap<GraphNode, usize> = nodes
        .iter()
        .enumerate()
        .map(|(index, (node, _))| (**node, index))
        .collect();
    let mut waiting_on: Vec<HashSet<usize>> = vec![HashSet::new(); nodes.len()];
    let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
    for (index, (_, flags)) in nodes.iter().enumerate() {
        for flag in *flags {
            let Some(previous_piece) = flag.piece.checked_sub(1) else {
                continue;
            };
            let Some(previous) = piece_nodes
                .get(&(flag.id, previous_piece))
                .and_then(|previous_node| index_of.get(previous_node))
                .copied()
                .filter(|previous| *previous != index)
            else {
                continue;
            };
            if waiting_on[index].insert(previous) {
                dependents[previous].push(index);
            }
        }
    }
    let mut ready: BTreeSet<usize> = (0..nodes.len())
        .filter(|index| waiting_on[*index].is_empty())
        .collect();
    let mut remaining: BTreeSet<usize> = (0..nodes.len()).collect();
    let mut order = Vec::with_capacity(nodes.len());
    while let Some(next) = ready.pop_first().or_else(|| remaining.first().copied()) {
        remaining.remove(&next);
        order.push(next);
        for dependent in &dependents[next] {
            waiting_on[*dependent].remove(&next);
            if waiting_on[*dependent].is_empty() && remaining.contains(dependent) {
                ready.insert(*dependent);
            }
        }
        ready.remove(&next);
    }
    order
}

/// Every piece of a span that already has enough room to show its name between the arrow
/// caps gets a label: each independently wide segment can repeat the annotation's name. A
/// segment counts as fitting when its bar length meets the same ``width + 2`` rule
/// [`label_anchors`] uses to keep endpoint bases, so labels on selected pieces never need
/// extra retained bases. A span with no fitting piece, whose name cannot be made to fit by
/// exposing more sequence, falls back as a last resort to its widest piece (the one nearest
/// the middle on a tie). A name wider than its node sits below the flag.
fn select_label_pieces(
    flags_by_node: &HashMap<GraphNode, Vec<AnnotationFlag>>,
    name_limit: usize,
) -> HashMap<HashId, Vec<usize>> {
    let mut pieces_by_annotation: HashMap<HashId, Vec<(usize, i64)>> = HashMap::new();
    let mut names_by_annotation: HashMap<HashId, String> = HashMap::new();
    for flags in flags_by_node.values() {
        for flag in flags {
            pieces_by_annotation
                .entry(flag.id)
                .or_default()
                .push((flag.piece, flag.bar_end - flag.bar_start));
            names_by_annotation
                .entry(flag.id)
                .or_insert_with(|| flag.name.clone());
        }
    }
    pieces_by_annotation
        .into_iter()
        .map(|(id, mut pieces)| {
            pieces.sort_unstable_by_key(|(piece, _)| *piece);
            pieces.dedup_by_key(|(piece, _)| *piece);
            let label_width =
                UnicodeWidthStr::width(short_name(&names_by_annotation[&id], name_limit).as_str())
                    as i64;
            let mut selected: Vec<usize> = pieces
                .iter()
                .filter(|(_, length)| *length >= label_width + 2)
                .map(|(piece, _)| *piece)
                .collect();
            if selected.is_empty() {
                let middle = pieces.len() / 2;
                let largest = pieces.iter().map(|(_, length)| *length).max().unwrap_or(0);
                selected.extend(
                    pieces
                        .iter()
                        .enumerate()
                        .filter(|(_, (_, length))| *length == largest)
                        .min_by_key(|(index, _)| index.abs_diff(middle))
                        .map(|(_, (piece, _))| *piece),
                );
            }
            (id, selected)
        })
        .collect()
}

pub(crate) fn build_layouts_for_detail(
    flags_by_node: &HashMap<GraphNode, Vec<AnnotationFlag>>,
    name_limit: usize,
    detail: AnnotationDetail,
    centered_nodes: &HashSet<GraphNode>,
) -> HashMap<GraphNode, NodeAnnotationLayout> {
    let full = matches!(detail, AnnotationDetail::Full);
    let label_piece = select_label_pieces(flags_by_node, name_limit);
    let mut nodes: Vec<_> = flags_by_node.iter().collect();
    nodes.sort_by_key(|(node, flags)| {
        (
            flags
                .iter()
                .map(|flag| flag.piece)
                .min()
                .unwrap_or(usize::MAX),
            **node,
        )
    });
    let piece_nodes: HashMap<(HashId, usize), GraphNode> = nodes
        .iter()
        .flat_map(|(node, flags)| flags.iter().map(|flag| ((flag.id, flag.piece), **node)))
        .collect();
    let order = layout_order(&nodes, &piece_nodes);
    let mut preferred: HashMap<HashId, usize> = HashMap::new();
    let mut rows_by_piece: HashMap<(HashId, usize), usize> = HashMap::new();
    let mut result = HashMap::new();
    for (node, flags) in order.into_iter().map(|index| nodes[index]) {
        let mut labels: HashMap<_, _> = flags
            .iter()
            .filter(|flag| {
                name_limit > 0
                    && label_piece
                        .get(&flag.id)
                        .is_some_and(|pieces| pieces.contains(&flag.piece))
            })
            .map(|flag| ((flag.id, flag.piece), short_name(&flag.name, name_limit)))
            .collect();
        let mut base_markers: Vec<i64> = flags
            .iter()
            .flat_map(|flag| {
                [
                    (!flag.continues_left).then_some(flag.bar_start),
                    (!flag.continues_right).then_some(flag.bar_end - 1),
                ]
            })
            .flatten()
            .collect();
        base_markers.sort_unstable();
        base_markers.dedup();
        // True annotation ends add ordered columns; continuation boundaries do not.
        let mut events = base_markers.clone();
        // A piece that continues into a neighbouring piece but stops short of the node's edge,
        // such as one exon of a CDS lying inside a node that also holds the next exon, needs
        // columns of its own; only bars reaching the edge run through the continuation.
        for flag in flags {
            if flag.continues_left && flag.bar_start > 0 {
                events.push(flag.bar_start);
            }
            if flag.continues_right && flag.bar_end < node.length() {
                events.push(flag.bar_end - 1);
            }
        }
        if !full {
            for flag in flags {
                if let Some(label) = labels.get(&(flag.id, flag.piece)) {
                    events.extend(label_anchors(flag, label));
                }
            }
        }
        let anchors = sequence_anchors(node.length());
        if !full {
            // Annotation events cannot replace the node's sequence endpoints.
            events.extend(anchors.iter().take(2).copied());
            events.sort_unstable();
            events.dedup();
        }
        for anchor in anchors {
            if events.len() >= node.length().min(3) as usize {
                break;
            }
            if !events.contains(&anchor) {
                events.push(anchor);
            }
        }
        events.sort_unstable();
        events.dedup();
        let columns: Vec<i64> = if full {
            events.clone()
        } else {
            compact_columns(&events)
        };
        let mut ordered: Vec<_> = flags.iter().collect();
        ordered.sort_by_key(|flag| (flag.bar_start, flag.id, flag.piece));
        let mut left_padding = 0;
        let mut pieces = Vec::with_capacity(flags.len());
        for flag in ordered {
            let left_event = if flag.continues_left && flag.bar_start == 0 {
                0
            } else {
                columns[events
                    .binary_search(&flag.bar_start)
                    .expect("should find start event")]
            };
            let right_event = if flag.continues_right && flag.bar_end >= node.length() {
                if full {
                    node.length() - 1
                } else {
                    *columns.last().expect("should have a compact column")
                }
            } else {
                columns[events
                    .binary_search(&(flag.bar_end - 1))
                    .expect("should find end event")]
            };
            let label = labels.remove(&(flag.id, flag.piece));
            let piece = AnnotationPiece::new(flag, left_event, right_event, label, node.length());
            if piece.label_left {
                left_padding = left_padding.max(piece.label_width() - left_event);
            }
            pieces.push(piece);
        }
        let sequence_width = if full {
            node.length().max(1)
        } else {
            columns.last().copied().unwrap_or(0) + 1
        };
        for piece in &mut pieces {
            piece.left += left_padding;
            piece.right += left_padding;
        }
        // Centered nodes sit on the y = 0 row whatever their height, so an annotation may hold
        // its row between two of them; elsewhere a node's y moves with the heights around it.
        if centered_nodes.contains(node) {
            for piece in &mut pieces {
                piece.pinned_row = piece
                    .piece
                    .checked_sub(1)
                    .and_then(|previous_piece| {
                        piece_nodes
                            .get(&(piece.id, previous_piece))
                            .filter(|previous_node| {
                                *previous_node != node && centered_nodes.contains(previous_node)
                            })
                            .and_then(|_| rows_by_piece.get(&(piece.id, previous_piece)))
                    })
                    .copied();
            }
        }
        let rows = pack_pieces(&mut pieces, &mut preferred);
        for piece in &pieces {
            rows_by_piece.insert((piece.id, piece.piece), piece.row);
        }
        result.insert(
            *node,
            NodeAnnotationLayout {
                width: sequence_width + left_padding,
                height: 2 * rows + 1,
                left_padding,
                events,
                columns,
                base_markers,
                pieces,
            },
        );
    }
    result
}

/// Pack each arrow and name into rows. A name wider than its node reserves a second row
/// below its arrow; shorter names share the arrow's row. Pieces with a pinned row
/// are placed first, the annotation that started earlier (more pieces behind it) winning a
/// contested row, and the rest take the free row closest to where their annotation was last.
pub(crate) fn pack_pieces(
    pieces: &mut [AnnotationPiece],
    preferred: &mut HashMap<HashId, usize>,
) -> usize {
    let mut rows: Vec<Vec<(i64, i64)>> = Vec::new();
    let intervals: Vec<(i64, i64)> = pieces
        .iter()
        .map(|piece| {
            (
                if piece.label_below {
                    piece.left
                } else {
                    piece
                        .left
                        .min(piece_label_start(piece, piece.label_width()))
                },
                piece.right,
            )
        })
        .collect();
    let label_intervals: Vec<Option<(i64, i64)>> = pieces
        .iter()
        .map(|piece| {
            piece.label_below.then(|| {
                let start = piece_label_start(piece, piece.label_width());
                (start, start + piece.label_width() - 1)
            })
        })
        .collect();
    let is_interval_free = |rows: &[Vec<(i64, i64)>], row: usize, interval: (i64, i64)| {
        rows.get(row).is_none_or(|occupied| {
            occupied
                .iter()
                .all(|other| interval.1 < other.0 || other.1 < interval.0)
        })
    };
    let is_free = |rows: &[Vec<(i64, i64)>], row: usize, index: usize| {
        is_interval_free(rows, row, intervals[index])
            && label_intervals[index]
                .is_none_or(|interval| is_interval_free(rows, row + 1, interval))
    };
    let mut pinned: Vec<usize> = (0..pieces.len())
        .filter(|index| pieces[*index].pinned_row.is_some())
        .collect();
    pinned.sort_by_key(|index| (core::cmp::Reverse(pieces[*index].piece), pieces[*index].id));
    let mut placed = vec![false; pieces.len()];
    for index in pinned {
        let Some(row) = pieces[index].pinned_row else {
            continue;
        };
        if !is_free(&rows, row, index) {
            continue;
        }
        rows.resize_with(
            rows.len()
                .max(row + 1 + usize::from(label_intervals[index].is_some())),
            Vec::new,
        );
        rows[row].push(intervals[index]);
        if let Some(interval) = label_intervals[index] {
            rows[row + 1].push(interval);
        }
        pieces[index].row = row;
        preferred.insert(pieces[index].id, row);
        placed[index] = true;
    }
    for (index, piece) in pieces.iter_mut().enumerate() {
        if placed[index] {
            continue;
        }
        let wanted = preferred.get(&piece.id).copied().unwrap_or(0);
        let row = (0..=rows.len())
            .filter(|&row| is_free(&rows, row, index))
            .min_by_key(|&row| (row.abs_diff(wanted), row))
            .expect("should have a free new row");
        rows.resize_with(
            rows.len()
                .max(row + 1 + usize::from(label_intervals[index].is_some())),
            Vec::new,
        );
        rows[row].push(intervals[index]);
        if let Some(interval) = label_intervals[index] {
            rows[row + 1].push(interval);
        }
        piece.row = row;
        preferred.insert(piece.id, row);
    }
    rows.len()
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use gen_core::{HashId, Strand};
    use gen_graph::GraphNode;
    use ratatui::style::Color;
    use unicode_width::UnicodeWidthStr;

    use super::{
        AnnotationDetail, AnnotationPiece, NodeAnnotationLayout, VisibleColumns,
        build_layouts_for_detail, end_glyphs, label_start, piece_glyph, piece_label_start,
        replaces_glyph, short_name, visible_label_start,
    };
    use crate::views::gen_graph_widget::{
        ANNOTATION_BAR, ANNOTATION_FORWARD_CAP, ANNOTATION_REVERSE_CAP,
        ANNOTATION_SINGLE_BASE_MARK, AnnotationFlag, LabelPlacement,
    };

    fn build_layouts(
        flags_by_node: &HashMap<GraphNode, Vec<AnnotationFlag>>,
        name_limit: usize,
    ) -> HashMap<GraphNode, NodeAnnotationLayout> {
        build_layouts_for_detail(
            flags_by_node,
            name_limit,
            AnnotationDetail::Compact,
            &HashSet::new(),
        )
    }

    fn node(name: &str) -> GraphNode {
        GraphNode {
            node_id: HashId::convert_str(name),
            sequence_start: 0,
            sequence_end: 20,
        }
    }

    fn flag(name: &str, piece: usize, start: i64, end: i64) -> AnnotationFlag {
        AnnotationFlag {
            id: HashId::convert_str(name),
            piece,
            name: name.to_owned(),
            color: Color::Blue,
            strand: Strand::Forward,
            bar_start: start,
            bar_end: end,
            continues_left: piece > 0,
            continues_right: false,
            show_label: false,
            label: LabelPlacement::Inside,
            lane: 0,
        }
    }

    #[test]
    fn test_nested_and_crossing_endpoints_use_event_order() {
        let nested = node("nested");
        let crossing = node("crossing");
        let flags = HashMap::from([
            (nested, vec![flag("A", 0, 2, 10), flag("B", 0, 4, 6)]),
            (crossing, vec![flag("A", 0, 2, 6), flag("B", 0, 4, 10)]),
        ]);
        let layouts = build_layouts(&flags, 10);
        assert_eq!(layouts[&nested].events, [0, 2, 4, 5, 9, 19]);
        assert_eq!(layouts[&crossing].events, [0, 2, 4, 5, 9, 19]);
        assert_eq!(
            layouts[&nested].pieces[0].right - layouts[&nested].pieces[0].left,
            5
        );
        assert_eq!(layouts[&nested].height, 5);
    }

    #[test]
    fn test_continuation_inside_a_node_claims_its_own_columns() {
        let graph_node = node("touching");
        let flags = HashMap::from([(graph_node, vec![flag("A", 0, 2, 4), flag("A", 1, 5, 7)])]);
        let layout = &build_layouts(&flags, 10)[&graph_node];
        assert_eq!(layout.events, [0, 2, 3, 5, 6, 19]);
        assert!(layout.pieces[0].right < layout.pieces[1].left);
        assert_eq!(layout.height, 3);
        assert!(layout.pieces[0].label.is_none());
        assert!(layout.pieces[1].label.is_some());
        assert_eq!(
            layout.pieces[0].row, layout.pieces[1].row,
            "the second piece's name fills the room to its left without a gap column"
        );
        assert_eq!(layout.map_column(3), layout.left_padding + 3);
        assert_eq!(layout.raw_column(layout.map_column(5)), 5);
    }

    #[test]
    fn test_annotations_without_a_gap_column_share_rows() {
        let graph_node = node("adjacent labels");
        let flags = HashMap::from([(graph_node, vec![flag("A", 0, 2, 4), flag("B", 0, 5, 7)])]);
        let layout = &build_layouts(&flags, 10)[&graph_node];
        assert_eq!(layout.pieces[0].right + 2, layout.pieces[1].left);
        assert_eq!(layout.pieces[0].row, layout.pieces[1].row);
    }

    #[test]
    fn test_adjacent_annotations_keep_distinct_base_columns() {
        let graph_node = node("shared coordinate");
        let flags = HashMap::from([(graph_node, vec![flag("A", 0, 2, 4), flag("B", 0, 4, 7)])]);
        let layout = &build_layouts(&flags, 10)[&graph_node];
        assert_eq!(layout.events, [0, 2, 3, 4, 5, 6, 19]);
        assert!(layout.pieces[0].right < layout.pieces[1].left);
        assert_eq!(layout.height, 3);
    }

    #[test]
    fn test_single_base_and_continuation_cells() {
        let graph_node = node("single base");
        let mut continuation = flag("continued", 1, 0, 20);
        continuation.continues_right = true;
        let flags = HashMap::from([(graph_node, vec![flag("one", 0, 5, 6), continuation])]);
        let layout = &build_layouts(&flags, 10)[&graph_node];
        let one = layout
            .pieces
            .iter()
            .find(|piece| piece.label.as_deref() == Some("one"))
            .unwrap();
        assert_eq!(one.right, one.left);
        assert!(one.label_left);
        assert_eq!((one.left_glyph, one.right_glyph), ('[', '>'));
        let continued = layout
            .pieces
            .iter()
            .find(|piece| piece.id == HashId::convert_str("continued"))
            .unwrap();
        assert_eq!((continued.left_glyph, continued.right_glyph), ('═', '═'));
    }

    #[test]
    fn test_continuing_piece_inside_a_node_stops_at_its_own_ends() {
        let graph_node = node("shared");
        let mut earlier = flag("coding", 1, 2, 7);
        earlier.continues_right = true;
        let mut later = flag("coding", 2, 12, 18);
        later.continues_right = true;
        let flags = HashMap::from([(graph_node, vec![earlier, later])]);

        let layout = &build_layouts(&flags, 0)[&graph_node];

        let mut spans: Vec<(i64, i64)> = layout
            .pieces
            .iter()
            .map(|piece| (piece.left, piece.right))
            .collect();
        spans.sort_unstable();
        assert!(
            spans[0].1 < spans[1].0,
            "the two pieces should sit side by side with a gap, got {spans:?}"
        );
        assert_eq!(
            layout.pieces[0].row, layout.pieces[1].row,
            "pieces that don't overlap should share a row"
        );
    }

    #[test]
    fn test_labeled_internal_segment_keeps_room_for_its_name() {
        let graph_node = node("isolated");
        let flags = HashMap::from([(graph_node, vec![flag("mutation", 0, 5, 6)])]);
        let layout = &build_layouts(&flags, 20)[&graph_node];
        let piece = &layout.pieces[0];
        let name_start = label_start(piece.left, piece.right, 8);
        assert_eq!(piece.left, piece.right);
        assert!(piece.label_left);
        assert_eq!(name_start, label_start(piece.left, piece.right, 8));
        assert_eq!(layout.height, 3);
    }

    #[test]
    fn test_short_internal_continuation_stays_unnamed_while_longer_pieces_repeat_the_name() {
        let internal = node("internal mutation");
        let continuation = node("internal continuation");
        let labeled = node("labeled continuation");
        let last = node("last continuation");
        let mut continuing = flag("span", 1, 0, 20);
        continuing.continues_right = true;
        let mut middle = flag("span", 2, 0, 20);
        middle.continues_right = true;
        let flags = HashMap::from([
            (internal, vec![flag("span", 0, 5, 6)]),
            (continuation, vec![continuing]),
            (labeled, vec![middle]),
            (last, vec![flag("span", 3, 0, 8)]),
        ]);
        let layouts = build_layouts(&flags, 20);
        let piece = &layouts[&internal].pieces[0];
        assert_eq!(piece.label, None);
        assert_eq!(piece.left, piece.right);
        // Every segment wide enough to hold "span" between its caps now names itself.
        for piece in [&layouts[&continuation], &layouts[&labeled], &layouts[&last]] {
            assert_eq!(piece.pieces[0].label.as_deref(), Some("span"));
            assert!(piece.pieces[0].left < piece.pieces[0].right);
        }
    }

    #[test]
    fn test_two_base_node_with_only_spanning_annotations_preserves_both_bases() {
        let short_node = GraphNode {
            node_id: HashId::convert_str("two base continuation"),
            sequence_start: 10,
            sequence_end: 12,
        };
        let mut first = flag("first", 1, 0, 2);
        first.continues_right = true;
        let mut second = flag("second", 1, 0, 2);
        second.continues_right = true;
        let layouts = build_layouts(
            &HashMap::from([
                (short_node, vec![first, second]),
                (node("first host"), vec![flag("first", 2, 0, 10)]),
                (node("second host"), vec![flag("second", 2, 0, 10)]),
            ]),
            20,
        );
        let layout = &layouts[&short_node];

        assert_eq!(layout.events, [0, 1]);
        assert!(layout.base_markers.is_empty());
        assert_eq!(layout.width, 2);
        assert_eq!(layout.height, 5);
        assert!(
            layout
                .pieces
                .iter()
                .all(|piece| { piece.left == 0 && piece.right == 1 && piece.label.is_none() })
        );
        assert_ne!(layout.pieces[0].row, layout.pieces[1].row);
    }

    #[test]
    fn test_label_uses_largest_sequence_piece_instead_of_middle_piece() {
        let first = node("largest first");
        let middle = GraphNode {
            node_id: HashId::convert_str("short middle"),
            sequence_start: 0,
            sequence_end: 2,
        };
        let last = node("short last");
        let mut first_piece = flag("span", 0, 0, 18);
        first_piece.continues_right = true;
        let mut middle_piece = flag("span", 1, 0, 2);
        middle_piece.continues_right = true;
        let layouts = build_layouts(
            &HashMap::from([
                (first, vec![first_piece]),
                (middle, vec![middle_piece]),
                (last, vec![flag("span", 2, 0, 3)]),
            ]),
            20,
        );

        assert_eq!(layouts[&first].pieces[0].label.as_deref(), Some("span"));
        assert!(layouts[&middle].pieces[0].label.is_none());
        assert!(layouts[&last].pieces[0].label.is_none());
    }

    #[test]
    fn test_single_base_slice_uses_local_coordinate_even_with_nonzero_sequence_start() {
        let one_base = GraphNode {
            node_id: HashId::convert_str("one base slice"),
            sequence_start: 100,
            sequence_end: 101,
        };
        let labeled = node("label host");
        let mut first = flag("feature", 0, 0, 1);
        first.continues_right = true;
        let flags = HashMap::from([
            (one_base, vec![first]),
            (labeled, vec![flag("feature", 1, 2, 10)]),
        ]);
        let layout = &build_layouts(&flags, 20)[&one_base];
        assert_eq!(layout.events, [0]);
        assert_eq!(layout.width, 1);
        assert_eq!(layout.pieces[0].left, layout.pieces[0].right);
        assert!(layout.pieces[0].label.is_none());
    }

    #[test]
    fn test_split_piece_keeps_name_color_and_row() {
        let first = node("first");
        let second = node("second");
        let mut continued = flag("A", 1, 0, 7);
        continued.color = Color::Magenta;
        let mut initial = flag("A", 0, 5, 20);
        initial.color = Color::Magenta;
        initial.continues_right = true;
        let flags = HashMap::from([(first, vec![initial]), (second, vec![continued])]);
        let layouts = build_layouts(&flags, 10);
        let first_piece = &layouts[&first].pieces[0];
        let second_piece = &layouts[&second].pieces[0];
        assert_eq!(first_piece.label.as_deref(), Some("A"));
        // Both segments are wide enough to hold the name, so both repeat it.
        assert_eq!(second_piece.label.as_deref(), Some("A"));
        assert_eq!(first_piece.color, second_piece.color);
        assert_eq!(first_piece.row, second_piece.row);
        assert_eq!(first_piece.right_glyph, '═');
        assert_eq!(second_piece.left_glyph, '═');
    }

    #[test]
    fn test_three_piece_span_labels_every_fitting_piece() {
        let first = node("first of three");
        let middle = node("middle of three");
        let last = node("last of three");
        let mut first_piece = flag("span", 0, 2, 20);
        first_piece.continues_right = true;
        let mut middle_piece = flag("span", 1, 0, 20);
        middle_piece.continues_right = true;
        let flags = HashMap::from([
            (first, vec![first_piece]),
            (middle, vec![middle_piece]),
            (last, vec![flag("span", 2, 0, 8)]),
        ]);
        let layouts = build_layouts(&flags, 20);
        // "span" needs six bases (name + two caps), so all three segments qualify.
        assert_eq!(layouts[&first].pieces[0].label.as_deref(), Some("span"));
        assert_eq!(layouts[&first].base_markers, [2]);
        assert_eq!(layouts[&middle].pieces[0].label.as_deref(), Some("span"));
        assert!(layouts[&middle].base_markers.is_empty());
        let labeled = &layouts[&middle].pieces[0];
        let name_start = label_start(labeled.left, labeled.right, 4);
        assert!(!labeled.label_left);
        assert_eq!(name_start, labeled.left + 1);
        assert_eq!(layouts[&last].pieces[0].label.as_deref(), Some("span"));
        assert_eq!(layouts[&last].base_markers, [7]);
    }

    #[test]
    fn test_collision_moves_span_to_nearest_compact_row() {
        let first = node("row first");
        let second = node("row second");
        let flags = HashMap::from([
            (first, vec![flag("A", 0, 0, 20), flag("B", 0, 2, 8)]),
            (second, vec![flag("B", 1, 0, 20), flag("A", 1, 2, 8)]),
        ]);
        let layouts = build_layouts(&flags, 10);
        assert_eq!(layouts[&first].height, 5);
        assert_eq!(layouts[&second].height, 5);
        assert_ne!(
            layouts[&second].pieces[0].row,
            layouts[&second].pieces[1].row
        );
    }

    #[test]
    fn test_label_wider_than_node_uses_row_below_without_padding() {
        let graph_node = GraphNode {
            sequence_end: 4,
            ..node("short sequence")
        };
        let flags = HashMap::from([(graph_node, vec![flag("long feature name", 0, 0, 4)])]);
        for detail in [AnnotationDetail::Compact, AnnotationDetail::Full] {
            let layout =
                &build_layouts_for_detail(&flags, usize::MAX, detail, &HashSet::new())[&graph_node];
            let piece = &layout.pieces[0];
            assert!(piece.label_below);
            assert!(!piece.label_left);
            assert_eq!(layout.left_padding, 0);
            assert_eq!(piece.left, 0);
            assert_eq!(piece.right, layout.width - 1);
            let (expected_width, expected_columns) = if matches!(detail, AnnotationDetail::Full) {
                (4, vec![0, 3])
            } else {
                (3, vec![0, 2])
            };
            assert_eq!(layout.width, expected_width);
            assert_eq!(layout.columns, expected_columns);
            for raw in &layout.events {
                assert_eq!(layout.raw_column(layout.map_column(*raw)), *raw);
            }
            let width = 17;
            assert_eq!(piece_label_start(piece, width), -7);
            assert_eq!(layout.height, 5);
        }
    }

    #[test]
    fn test_compact_label_retains_balanced_endpoint_bases_and_one_period() {
        let graph_node = node("label-sized compression");
        let flags = HashMap::from([(graph_node, vec![flag("feature", 0, 0, 20)])]);
        let layout = &build_layouts(&flags, usize::MAX)[&graph_node];
        assert_eq!(layout.events, [0, 1, 2, 3, 16, 17, 18, 19]);
        assert_eq!(layout.columns, [0, 1, 2, 3, 5, 6, 7, 8]);
        assert_eq!(layout.width, 9);
        assert_eq!(layout.left_padding, 0);
        assert!(!layout.pieces[0].label_left);
        assert_eq!(layout.height, 3);
        for raw in &layout.events {
            assert_eq!(layout.raw_column(layout.map_column(*raw)), *raw);
        }
    }

    #[test]
    fn test_compact_label_at_sequence_capacity_shows_every_base() {
        let graph_node = GraphNode {
            sequence_end: 9,
            ..node("exact label capacity")
        };
        let flags = HashMap::from([(graph_node, vec![flag("feature", 0, 0, 9)])]);
        let layout = &build_layouts(&flags, usize::MAX)[&graph_node];
        assert_eq!(layout.events, (0..9).collect::<Vec<_>>());
        assert_eq!(layout.width, 9);
        assert!(!layout.pieces[0].label_left);
    }

    #[test]
    fn test_compact_compression_uses_terminal_width_and_requested_name_limit() {
        let graph_node = node("unicode label compression");
        let flags = HashMap::from([(graph_node, vec![flag("漢字abc", 0, 0, 20)])]);
        let layouts = build_layouts(&flags, usize::MAX);
        assert_eq!(layouts[&graph_node].width, 9);
        let layouts = build_layouts(&flags, 4);
        assert_eq!(layouts[&graph_node].pieces[0].label.as_deref(), Some("漢."));
        assert_eq!(layouts[&graph_node].width, 5);
        let layouts = build_layouts(&flags, 0);
        assert_eq!(layouts[&graph_node].events, [0, 19]);
        assert!(layouts[&graph_node].pieces[0].label.is_none());
        assert_eq!(layouts[&graph_node].left_padding, 0);
    }

    #[test]
    fn test_left_placement_uses_existing_sequence_space_without_padding() {
        let graph_node = node("existing label space");
        let flags = HashMap::from([(graph_node, vec![flag("a", 0, 0, 20), flag("x", 0, 10, 11)])]);
        for detail in [AnnotationDetail::Compact, AnnotationDetail::Full] {
            let layouts = build_layouts_for_detail(&flags, usize::MAX, detail, &HashSet::new());
            let layout = &layouts[&graph_node];
            let piece = layout
                .pieces
                .iter()
                .find(|piece| piece.label.as_deref() == Some("x"))
                .expect("should carry x label");
            assert!(piece.label_left);
            assert_eq!(layout.left_padding, 0);
            assert_eq!(piece_label_start(piece, 1), piece.left - 1);
        }
    }

    #[test]
    fn test_left_labels_move_together_with_arrows_when_extents_overlap() {
        let graph_node = node("overlapping names");
        let flags = HashMap::from([(
            graph_node,
            vec![
                flag("first long feature", 0, 2, 4),
                flag("second long feature", 0, 5, 7),
            ],
        )]);
        let layouts =
            build_layouts_for_detail(&flags, usize::MAX, AnnotationDetail::Full, &HashSet::new());
        let pieces = &layouts[&graph_node].pieces;
        assert!(pieces.iter().all(|piece| piece.label_left));
        assert_ne!(pieces[0].row, pieces[1].row);
        assert_eq!(layouts[&graph_node].height, 5);
    }

    #[test]
    fn test_overlapping_labels_wider_than_node_reserve_separate_rows() {
        let graph_node = GraphNode {
            sequence_end: 7,
            ..node("short overlapping labels")
        };
        let flags = HashMap::from([(
            graph_node,
            vec![
                flag("first long feature", 0, 2, 4),
                flag("second long feature", 0, 3, 6),
            ],
        )]);
        for detail in [AnnotationDetail::Compact, AnnotationDetail::Full] {
            let layout =
                &build_layouts_for_detail(&flags, usize::MAX, detail, &HashSet::new())[&graph_node];
            assert_eq!(layout.left_padding, 0);
            assert!(layout.pieces.iter().all(|piece| piece.label_below));
            assert_eq!(layout.height, 9);
            assert_eq!(layout.pieces[0].row, 0);
            assert_eq!(layout.pieces[1].row, 2);
        }
    }

    #[test]
    fn test_empty_node_has_single_cell() {
        let empty = NodeAnnotationLayout::empty(0);
        assert_eq!((empty.width, empty.height), (1, 1));
        assert!(empty.pieces.is_empty());
        assert_eq!(empty.map_column(17), 0);
    }

    #[test]
    fn test_sequence_context_without_annotations() {
        for length in 1..=8 {
            let graph_node = GraphNode {
                sequence_start: 100,
                sequence_end: 100 + length,
                ..node("sequence context")
            };
            let layout =
                &build_layouts(&HashMap::from([(graph_node, Vec::new())]), 20)[&graph_node];
            assert_eq!(layout.width, if length <= 3 { length } else { 3 });
            assert_eq!(layout.events.first(), Some(&0));
            assert_eq!(layout.events.last(), Some(&(length - 1)));
            for raw in &layout.events {
                assert_eq!(layout.raw_column(layout.map_column(*raw)), *raw);
            }
        }
    }

    #[test]
    fn test_unannotated_long_node_retains_only_endpoints() {
        let layout = NodeAnnotationLayout::empty(20);
        assert_eq!(layout.events, [0, 19]);
        assert_eq!(layout.columns, [0, 2]);
        assert_eq!(layout.width, 3);
        assert_eq!(layout.map_column(0), 0);
        assert_eq!(layout.map_column(1), 1);
        assert_eq!(layout.map_column(10), 1);
        assert_eq!(layout.map_column(18), 1);
        assert_eq!(layout.map_column(19), 2);
    }

    #[test]
    fn test_end_glyph_table() {
        assert_eq!(end_glyphs(Strand::Forward, false, false), ('[', '>'));
        assert_eq!(end_glyphs(Strand::Forward, true, false), ('═', '>'));
        assert_eq!(end_glyphs(Strand::Forward, false, true), ('[', '═'));
        assert_eq!(end_glyphs(Strand::Reverse, false, false), ('<', ']'));
        assert_eq!(end_glyphs(Strand::Reverse, false, true), ('<', '═'));
        assert_eq!(end_glyphs(Strand::Reverse, true, false), ('═', ']'));
        assert_eq!(end_glyphs(Strand::Unknown, false, false), ('[', ']'));
    }

    #[test]
    fn test_unicode_name_limit() {
        assert_eq!(short_name("abcdefghijk", 10), "abcdefghi.");
        assert_eq!(short_name("e\u{301}e\u{301}e\u{301}", 2), "e\u{301}.");
        assert_eq!(short_name("漢字abc", 4), "漢.");
    }

    fn piece_spanning(left: i64, right: i64, label: &str) -> AnnotationPiece {
        AnnotationPiece {
            id: HashId::convert_str("annotation"),
            piece: 0,
            color: Color::Reset,
            row: 0,
            pinned_row: None,
            label_left: UnicodeWidthStr::width(label) as i64 > right - left - 1,
            label_below: false,
            left,
            right,
            label: Some(label.to_string()),
            left_glyph: '[',
            right_glyph: ']',
        }
    }

    fn visible_start(piece: &AnnotationPiece, left: i64, right: i64) -> i64 {
        visible_label_start(piece, piece.label_width(), VisibleColumns { left, right })
    }

    #[test]
    fn test_visible_label_start_keeps_a_fully_visible_label() {
        let piece = piece_spanning(10, 40, "gene");
        let normal = piece_label_start(&piece, 4);
        assert_eq!(visible_start(&piece, 0, 99), normal);
        assert_eq!(visible_start(&piece, normal, normal + 3), normal);
    }

    #[test]
    fn test_visible_label_start_pins_to_the_right_edge_when_the_flag_runs_off_right() {
        let piece = piece_spanning(10, 500, "gene");
        assert_eq!(visible_start(&piece, 0, 99), 99 - 3);
    }

    #[test]
    fn test_visible_label_start_pins_to_the_left_edge_when_the_flag_runs_off_left() {
        let piece = piece_spanning(10, 500, "gene");
        assert_eq!(visible_start(&piece, 400, 499), 400);
    }

    #[test]
    fn test_visible_label_start_reserves_a_visible_cap_column() {
        let piece = piece_spanning(10, 40, "gene");
        assert_eq!(visible_start(&piece, 0, 16), 13);
        assert_eq!(visible_start(&piece, 36, 60), 36);
    }

    #[test]
    fn test_visible_label_start_rides_the_nearest_edge_when_cut_on_both_sides() {
        let piece = piece_spanning(0, 1000, "gene");
        let normal = piece_label_start(&piece, 4);
        assert_eq!(visible_start(&piece, 100, 199), 199 - 3);
        assert_eq!(visible_start(&piece, normal - 50, normal + 49), normal);
        assert_eq!(visible_start(&piece, 800, 899), 800);
    }

    #[test]
    fn test_visible_label_start_moves_one_column_at_a_time_while_panning() {
        let piece = piece_spanning(0, 1000, "gene");
        let width = 100;
        let mut previous = None;
        for view_left in -50..=996 {
            let start = visible_start(&piece, view_left, view_left + width - 1);
            if let Some(previous_start) = previous {
                let step: i64 = start - previous_start;
                assert!((0..=1).contains(&step), "jumped by {step} at {view_left}");
            }
            previous = Some(start);
        }
    }

    #[test]
    fn test_visible_label_start_keeps_the_normal_spot_when_too_narrow() {
        let piece = piece_spanning(10, 500, "a long name");
        let normal = piece_label_start(&piece, 11);
        assert_eq!(visible_start(&piece, 10, 18), normal);
        assert_eq!(visible_start(&piece, 600, 700), normal);
    }

    #[test]
    fn test_visible_label_start_keeps_an_external_label_while_the_flag_start_shows() {
        let piece = piece_spanning(10, 13, "promoter");
        assert_eq!(visible_start(&piece, 0, 99), 10 - 8);
    }
    fn row_of(layout: &NodeAnnotationLayout, name: &str, piece: usize) -> usize {
        layout
            .pieces
            .iter()
            .find(|candidate| candidate.id == HashId::convert_str(name) && candidate.piece == piece)
            .map(|candidate| candidate.row)
            .expect("should have the piece")
    }

    fn layouts_with_centered_nodes(
        flags: &HashMap<GraphNode, Vec<AnnotationFlag>>,
        centered_nodes: &[GraphNode],
    ) -> HashMap<GraphNode, NodeAnnotationLayout> {
        build_layouts_for_detail(
            flags,
            usize::MAX,
            AnnotationDetail::Compact,
            &centered_nodes.iter().copied().collect(),
        )
    }

    fn glyph_piece(left: i64, right: i64, left_glyph: char, right_glyph: char) -> AnnotationPiece {
        AnnotationPiece {
            left_glyph,
            right_glyph,
            ..piece_spanning(left, right, "")
        }
    }

    #[test]
    fn test_one_cell_piece_shows_a_cap_over_the_continuation_bar() {
        for (left_glyph, right_glyph, expected) in [
            ('[', ANNOTATION_BAR, '['),
            (
                ANNOTATION_REVERSE_CAP,
                ANNOTATION_BAR,
                ANNOTATION_REVERSE_CAP,
            ),
            (
                ANNOTATION_BAR,
                ANNOTATION_FORWARD_CAP,
                ANNOTATION_FORWARD_CAP,
            ),
            (ANNOTATION_BAR, ']', ']'),
            (ANNOTATION_BAR, ANNOTATION_BAR, ANNOTATION_BAR),
        ] {
            let piece = glyph_piece(4, 4, left_glyph, right_glyph);
            assert_eq!(piece_glyph(&piece, 4), expected);
        }
    }

    #[test]
    fn test_one_cell_piece_with_both_caps_keeps_the_directional_one() {
        let forward = glyph_piece(4, 4, '[', ANNOTATION_FORWARD_CAP);
        let reverse = glyph_piece(4, 4, ANNOTATION_REVERSE_CAP, ']');
        let unknown = glyph_piece(4, 4, '[', ']');
        assert_eq!(piece_glyph(&forward, 4), ANNOTATION_FORWARD_CAP);
        assert_eq!(piece_glyph(&reverse, 4), ANNOTATION_REVERSE_CAP);
        assert_eq!(piece_glyph(&unknown, 4), ANNOTATION_SINGLE_BASE_MARK);
    }

    #[test]
    fn test_wider_piece_puts_each_cap_on_its_own_end() {
        let piece = glyph_piece(2, 5, '[', ANNOTATION_FORWARD_CAP);
        let glyphs: Vec<char> = (2..=5).map(|column| piece_glyph(&piece, column)).collect();
        assert_eq!(
            glyphs,
            ['[', ANNOTATION_BAR, ANNOTATION_BAR, ANNOTATION_FORWARD_CAP]
        );
    }

    #[test]
    fn test_a_bar_never_replaces_a_cap() {
        assert!(!replaces_glyph('[', ANNOTATION_BAR));
        assert!(!replaces_glyph(ANNOTATION_FORWARD_CAP, ANNOTATION_BAR));
        assert!(replaces_glyph(ANNOTATION_BAR, '['));
        assert!(!replaces_glyph('[', ANNOTATION_FORWARD_CAP));
        assert!(!replaces_glyph(ANNOTATION_BAR, ANNOTATION_BAR));
    }

    mod row_pinning {
        use super::*;

        /// Another annotation starting on the second node would normally take the row the
        /// continuing annotation had, pushing it down a row.
        fn bubble_flags() -> (
            GraphNode,
            GraphNode,
            HashMap<GraphNode, Vec<AnnotationFlag>>,
        ) {
            let first = node("first");
            let second = node("second");
            let flags = HashMap::from([
                (first, vec![flag("A", 0, 2, 10)]),
                (second, vec![flag("C", 0, 0, 10), flag("A", 1, 2, 10)]),
            ]);
            (first, second, flags)
        }

        #[test]
        fn test_continuing_annotation_keeps_its_row_between_centered_nodes() {
            let (first, second, flags) = bubble_flags();
            let layouts = layouts_with_centered_nodes(&flags, &[first, second]);
            assert_eq!(row_of(&layouts[&first], "A", 0), 0);
            assert_eq!(row_of(&layouts[&second], "A", 1), 0);
            assert_eq!(row_of(&layouts[&second], "C", 0), 1);
        }

        #[test]
        fn test_without_centered_nodes_the_annotation_packs_by_order() {
            let (first, second, flags) = bubble_flags();
            let layouts = layouts_with_centered_nodes(&flags, &[]);
            assert_eq!(row_of(&layouts[&first], "A", 0), 0);
            assert_eq!(row_of(&layouts[&second], "C", 0), 0);
            assert_eq!(row_of(&layouts[&second], "A", 1), 1);
        }

        #[test]
        fn test_pinned_row_keeps_its_gap_and_grows_the_node() {
            let first = node("first");
            let second = node("second");
            let flags = HashMap::from([
                (
                    first,
                    vec![
                        flag("X", 0, 0, 10),
                        flag("Y", 0, 1, 10),
                        flag("A", 0, 2, 10),
                    ],
                ),
                (second, vec![flag("A", 1, 2, 10)]),
            ]);
            let pinned = layouts_with_centered_nodes(&flags, &[first, second]);
            assert_eq!(row_of(&pinned[&first], "A", 0), 2);
            assert_eq!(row_of(&pinned[&second], "A", 1), 2);
            assert_eq!(pinned[&second].height, 2 * 3 + 1);
            let unpinned = layouts_with_centered_nodes(&flags, &[]);
            assert_eq!(unpinned[&second].height, 3);
        }

        #[test]
        fn test_annotation_with_more_pieces_behind_wins_a_contested_row() {
            let earlier = node("earlier");
            let before_a = node("before_a");
            let before_b = node("before_b");
            let joined = node("joined");
            // A and B each held row 1 on the node before the join, B after one more piece.
            let flags = HashMap::from([
                (earlier, vec![flag("B", 0, 0, 10)]),
                (before_b, vec![flag("Y", 0, 0, 10), flag("B", 1, 2, 10)]),
                (before_a, vec![flag("X", 0, 0, 10), flag("A", 0, 2, 10)]),
                (joined, vec![flag("A", 1, 0, 10), flag("B", 2, 0, 10)]),
            ]);
            let layouts = layouts_with_centered_nodes(&flags, &[before_a, before_b, joined]);
            assert_eq!(row_of(&layouts[&before_a], "A", 0), 1);
            assert_eq!(row_of(&layouts[&before_b], "B", 1), 1);
            assert_eq!(row_of(&layouts[&joined], "B", 2), 1);
            assert_ne!(row_of(&layouts[&joined], "A", 1), 1);
        }

        #[test]
        fn test_node_off_the_centered_row_is_not_pinned() {
            let (first, second, flags) = bubble_flags();
            let off_row_second = layouts_with_centered_nodes(&flags, &[first]);
            assert_eq!(row_of(&off_row_second[&second], "A", 1), 1);
            let off_row_first = layouts_with_centered_nodes(&flags, &[second]);
            assert_eq!(row_of(&off_row_first[&second], "A", 1), 1);
        }

        /// Nodes are sorted by first piece and then by hash, so across many names the node
        /// holding the later piece sorts first about half the time.
        #[test]
        fn test_pin_holds_whichever_node_sorts_first() {
            for index in 0..16 {
                let first = node(&format!("first {index}"));
                let second = node(&format!("second {index}"));
                let flags = HashMap::from([
                    (first, vec![flag("A", 0, 2, 10)]),
                    (second, vec![flag("C", 0, 0, 10), flag("A", 1, 2, 10)]),
                ]);
                let layouts = layouts_with_centered_nodes(&flags, &[first, second]);
                assert_eq!(row_of(&layouts[&second], "A", 1), 0, "pair {index}");
            }
        }
    }

    #[test]
    fn test_column_runs_give_each_skipped_stretch_one_cell() {
        let layout = NodeAnnotationLayout::empty(1_000);

        let runs = layout.column_runs();

        let spans: Vec<_> = runs
            .iter()
            .map(|run| (run.raw_start, run.raw_end, run.column_start, run.column_end))
            .collect();
        assert_eq!(spans, [(0, 1, 0, 1), (1, 999, 1, 2), (999, 1_000, 2, 3)]);
    }

    #[test]
    fn test_column_runs_agree_with_map_column() {
        let layout = NodeAnnotationLayout::empty(1_000);

        for run in layout.column_runs() {
            assert_eq!(layout.map_column(run.raw_start), run.column_start);
            assert_eq!(layout.map_column(run.raw_end - 1), run.column_end - 1);
        }
    }
}
