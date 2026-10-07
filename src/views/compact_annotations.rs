//! Shared annotation geometry with label-aware sequence compression for compact detail.

use std::collections::HashMap;

use gen_core::{HashId, Strand};
use gen_graph::GraphNode;
use ratatui::style::Color;
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

use crate::views::gen_graph_widget::{
    ANNOTATION_BAR, ANNOTATION_FORWARD_CAP, ANNOTATION_REVERSE_CAP, AnnotationFlag,
};

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
    pub label_left: bool,
    pub left: i64,
    pub right: i64,
    pub label: Option<String>,
    pub left_glyph: char,
    pub right_glyph: char,
}

impl AnnotationPiece {
    pub fn new(flag: &AnnotationFlag, left: i64, right: i64, label: Option<String>) -> Self {
        let width = label
            .as_deref()
            .map_or(0, |text| UnicodeWidthStr::width(text) as i64);
        let label_left = label.is_some() && width > right - left - 1;
        let (left_glyph, right_glyph) =
            end_glyphs(flag.strand, flag.continues_left, flag.continues_right);
        Self {
            id: flag.id,
            piece: flag.piece,
            color: flag.color,
            row: 0,
            label_left,
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
        let nearest = if index == 0 {
            0
        } else if index == self.events.len()
            || raw - self.events[index - 1] <= self.events[index] - raw
        {
            index - 1
        } else {
            index
        };
        self.left_padding + self.columns[nearest]
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
        vec![0, length - 1, length / 2]
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

/// Build all nodes together so labels favor the largest sequence piece and row preferences
/// can follow a span through its nodes.
pub(crate) fn build_layouts(
    flags_by_node: &HashMap<GraphNode, Vec<AnnotationFlag>>,
    name_limit: usize,
) -> HashMap<GraphNode, NodeAnnotationLayout> {
    build_layouts_for_detail(flags_by_node, name_limit, AnnotationDetail::Compact)
}

fn select_label_pieces(
    flags_by_node: &HashMap<GraphNode, Vec<AnnotationFlag>>,
) -> HashMap<HashId, usize> {
    let mut pieces_by_annotation: HashMap<HashId, Vec<(usize, i64)>> = HashMap::new();
    for flags in flags_by_node.values() {
        for flag in flags {
            pieces_by_annotation
                .entry(flag.id)
                .or_default()
                .push((flag.piece, flag.bar_end - flag.bar_start));
        }
    }
    pieces_by_annotation
        .into_iter()
        .map(|(id, mut pieces)| {
            pieces.sort_unstable_by_key(|(piece, _)| *piece);
            pieces.dedup_by_key(|(piece, _)| *piece);
            let middle = pieces.len() / 2;
            let largest = pieces.iter().map(|(_, length)| *length).max().unwrap_or(0);
            let selected = pieces
                .iter()
                .enumerate()
                .filter(|(_, (_, length))| *length == largest)
                .min_by_key(|(index, _)| index.abs_diff(middle))
                .map(|(_, (piece, _))| *piece)
                .expect("should have a label piece");
            (id, selected)
        })
        .collect()
}

pub(crate) fn build_layouts_for_detail(
    flags_by_node: &HashMap<GraphNode, Vec<AnnotationFlag>>,
    name_limit: usize,
    detail: AnnotationDetail,
) -> HashMap<GraphNode, NodeAnnotationLayout> {
    let full = matches!(detail, AnnotationDetail::Full);
    let label_piece = select_label_pieces(flags_by_node);
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
    let mut preferred: HashMap<HashId, usize> = HashMap::new();
    let mut result = HashMap::new();
    for (node, flags) in nodes {
        let mut labels: HashMap<_, _> = flags
            .iter()
            .filter(|flag| name_limit > 0 && label_piece.get(&flag.id) == Some(&flag.piece))
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
        if !full {
            for flag in flags {
                if let Some(label) = labels.get(&(flag.id, flag.piece)) {
                    events.extend(label_anchors(flag, label));
                }
            }
        }
        for anchor in sequence_anchors(node.length()) {
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
            let left_event = if flag.continues_left {
                0
            } else {
                columns[events
                    .binary_search(&flag.bar_start)
                    .expect("should find start event")]
            };
            let right_event = if flag.continues_right {
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
            let piece = AnnotationPiece::new(flag, left_event, right_event, label);
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
        let rows = pack_pieces(&mut pieces, &mut preferred);
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

/// Pack an arrow and its name as one interval. Names that do not fit inside extend
/// the interval to the left; sequence coordinates remain unchanged.
pub(crate) fn pack_pieces(
    pieces: &mut [AnnotationPiece],
    preferred: &mut HashMap<HashId, usize>,
) -> usize {
    let mut rows: Vec<Vec<(i64, i64)>> = Vec::new();
    for piece in pieces {
        let width = piece.label_width();
        let interval = (
            piece.left.min(piece_label_start(piece, width)),
            piece.right + 1,
        );
        let wanted = preferred.get(&piece.id).copied().unwrap_or(0);
        let row = (0..=rows.len())
            .filter(|&row| {
                rows.get(row).is_none_or(|occupied| {
                    occupied
                        .iter()
                        .all(|other| interval.1 < other.0 || other.1 < interval.0)
                })
            })
            .min_by_key(|&row| (row.abs_diff(wanted), row))
            .expect("should have a free new row");
        rows.resize_with(rows.len().max(row + 1), Vec::new);
        rows[row].push(interval);
        piece.row = row;
        preferred.insert(piece.id, row);
    }
    rows.len()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use gen_core::{HashId, Strand};
    use gen_graph::GraphNode;
    use ratatui::style::Color;

    use super::{
        AnnotationDetail, NodeAnnotationLayout, build_layouts, build_layouts_for_detail,
        end_glyphs, label_start, piece_label_start, short_name,
    };
    use crate::views::gen_graph_widget::{AnnotationFlag, LabelPlacement};

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
        assert_eq!(layouts[&nested].events, [2, 4, 5, 9]);
        assert_eq!(layouts[&crossing].events, [2, 4, 5, 9]);
        assert_eq!(
            layouts[&nested].pieces[0].right - layouts[&nested].pieces[0].left,
            5
        );
        assert_eq!(layouts[&nested].height, 5);
    }

    #[test]
    fn test_continuation_boundary_does_not_claim_a_column() {
        let graph_node = node("touching");
        let flags = HashMap::from([(graph_node, vec![flag("A", 0, 2, 4), flag("A", 1, 5, 7)])]);
        let layout = &build_layouts(&flags, 10)[&graph_node];
        assert_eq!(layout.events, [2, 3, 6]);
        assert_eq!(layout.height, 5);
        assert!(layout.pieces[0].label.is_none());
        assert!(layout.pieces[1].label.is_some());
        assert_ne!(layout.pieces[0].row, layout.pieces[1].row);
        assert_eq!(layout.map_column(3), layout.left_padding + 1);
        assert_eq!(layout.raw_column(layout.map_column(5)), 6);
    }

    #[test]
    fn test_period_gap_allows_disjoint_units_to_share_rows() {
        let graph_node = node("adjacent labels");
        let flags = HashMap::from([(graph_node, vec![flag("A", 0, 2, 4), flag("B", 0, 5, 7)])]);
        let layout = &build_layouts(&flags, 10)[&graph_node];
        assert_eq!(layout.pieces[0].right + 2, layout.pieces[1].left);
        assert_ne!(layout.pieces[0].row, layout.pieces[1].row);
    }

    #[test]
    fn test_adjacent_annotations_keep_distinct_base_columns() {
        let graph_node = node("shared coordinate");
        let flags = HashMap::from([(graph_node, vec![flag("A", 0, 2, 4), flag("B", 0, 4, 7)])]);
        let layout = &build_layouts(&flags, 10)[&graph_node];
        assert_eq!(layout.events, [2, 3, 4, 5, 6]);
        assert!(layout.pieces[0].right < layout.pieces[1].left);
        assert_eq!(layout.height, 5);
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
    fn test_unlabeled_internal_segments_preserve_sequence_context() {
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
        let piece = &layouts[&continuation].pieces[0];
        assert_eq!(piece.label, None);
        assert!(piece.left < piece.right);
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
        assert_eq!(second_piece.label, None);
        assert_eq!(first_piece.color, second_piece.color);
        assert_eq!(first_piece.row, second_piece.row);
        assert_eq!(first_piece.right_glyph, '═');
        assert_eq!(second_piece.left_glyph, '═');
    }

    #[test]
    fn test_three_piece_span_labels_only_middle_piece() {
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
        assert_eq!(layouts[&first].pieces[0].label, None);
        assert_eq!(layouts[&first].base_markers, [2]);
        assert_eq!(layouts[&middle].pieces[0].label.as_deref(), Some("span"));
        assert!(layouts[&middle].base_markers.is_empty());
        let labeled = &layouts[&middle].pieces[0];
        let name_start = label_start(labeled.left, labeled.right, 4);
        assert!(!labeled.label_left);
        assert_eq!(name_start, labeled.left + 1);
        assert_eq!(layouts[&last].pieces[0].label, None);
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
    fn test_labels_add_only_left_padding_without_stretching_sequence_columns() {
        let graph_node = GraphNode {
            sequence_end: 4,
            ..node("short sequence")
        };
        let flags = HashMap::from([(graph_node, vec![flag("long feature name", 0, 0, 4)])]);
        for detail in [AnnotationDetail::Compact, AnnotationDetail::Full] {
            let layout = &build_layouts_for_detail(&flags, usize::MAX, detail)[&graph_node];
            let piece = &layout.pieces[0];
            assert!(piece.label_left);
            assert_eq!(piece.left, layout.left_padding);
            assert_eq!(piece.right, layout.width - 1);
            assert_eq!(layout.width, 4 + layout.left_padding);
            assert_eq!(layout.columns, vec![0, 2, 3]);
            for raw in &layout.events {
                assert_eq!(layout.raw_column(layout.map_column(*raw)), *raw);
            }
            let width = 17;
            assert!(piece_label_start(piece, width) >= 0);
            assert!(piece_label_start(piece, width) + width <= layout.width);
            assert_eq!(layout.height, 3);
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
        assert_eq!(layouts[&graph_node].events, [0, 10, 19]);
        assert!(layouts[&graph_node].pieces[0].label.is_none());
        assert_eq!(layouts[&graph_node].left_padding, 0);
    }

    #[test]
    fn test_left_placement_uses_existing_sequence_space_without_padding() {
        let graph_node = node("existing label space");
        let flags = HashMap::from([(graph_node, vec![flag("a", 0, 0, 20), flag("x", 0, 10, 11)])]);
        for detail in [AnnotationDetail::Compact, AnnotationDetail::Full] {
            let layouts = build_layouts_for_detail(&flags, usize::MAX, detail);
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
        let layouts = build_layouts_for_detail(&flags, usize::MAX, AnnotationDetail::Full);
        let pieces = &layouts[&graph_node].pieces;
        assert!(pieces.iter().all(|piece| piece.label_left));
        assert_ne!(pieces[0].row, pieces[1].row);
        assert_eq!(layouts[&graph_node].height, 5);
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
            assert_eq!(
                layout.width,
                if length <= 3 {
                    length
                } else if length == 4 {
                    4
                } else {
                    5
                }
            );
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
}
