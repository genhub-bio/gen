//! Endpoint-based annotation geometry for truncated graph detail.

use std::collections::HashMap;

use gen_core::{HashId, Strand};
use gen_graph::GraphNode;
use ratatui::style::Color;
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

use crate::views::gen_graph_widget::{
    ANNOTATION_BAR, ANNOTATION_FORWARD_CAP, ANNOTATION_REVERSE_CAP, AnnotationFlag,
};

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CompactPiece {
    pub id: HashId,
    pub piece: usize,
    pub color: Color,
    pub row: usize,
    pub left: i64,
    pub right: i64,
    pub label: Option<String>,
    pub left_glyph: char,
    pub right_glyph: char,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct CompactNode {
    pub events: Vec<i64>,
    pub columns: Vec<i64>,
    pub base_markers: Vec<i64>,
    pub margin: i64,
    pub width: i64,
    pub height: usize,
    pub pieces: Vec<CompactPiece>,
}

impl CompactNode {
    pub fn empty(length: i64) -> Self {
        let mut events = sequence_anchors(length);
        events.sort_unstable();
        Self {
            columns: (0..events.len() as i64).collect(),
            width: events.len().max(1) as i64,
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
        self.margin + self.columns[nearest]
    }

    pub fn raw_column(&self, column: i64) -> i64 {
        if self.events.is_empty() {
            return 0;
        }
        let index = self
            .columns
            .partition_point(|value| *value < column - self.margin);
        self.events[index.min(self.events.len() - 1)]
    }
}

// Preserve a little sequence context even when no annotation ends on this slice.
fn sequence_anchors(length: i64) -> Vec<i64> {
    if length <= 3 {
        (0..length).collect()
    } else {
        vec![0, length - 1, length / 2]
    }
}

fn end_glyphs(strand: Strand, continues_left: bool, continues_right: bool) -> (char, char) {
    match strand {
        Strand::Forward => (
            ANNOTATION_BAR,
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
            ANNOTATION_BAR,
        ),
        _ => (ANNOTATION_BAR, ANNOTATION_BAR),
    }
}

/// Shorten at grapheme boundaries while counting terminal cells, including the ellipsis.
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
    result.push('…');
    result
}

pub(crate) fn label_start(left: i64, right: i64, width: i64) -> i64 {
    (left + right + 1 - width).div_euclid(2)
}

/// Build all nodes together so labels favor the largest sequence piece and row preferences
/// can follow a span through its nodes.
pub(crate) fn build_layouts(
    flags_by_node: &HashMap<GraphNode, Vec<AnnotationFlag>>,
    name_limit: usize,
) -> HashMap<GraphNode, CompactNode> {
    let mut pieces_by_annotation: HashMap<HashId, Vec<(usize, i64)>> = HashMap::new();
    for flags in flags_by_node.values() {
        for flag in flags {
            pieces_by_annotation
                .entry(flag.id)
                .or_default()
                .push((flag.piece, flag.bar_end - flag.bar_start));
        }
    }
    let label_piece: HashMap<HashId, usize> = pieces_by_annotation
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
        .collect();
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
        let mut columns: Vec<i64> = (0..events.len() as i64).collect();
        for flag in flags {
            if label_piece.get(&flag.id) != Some(&flag.piece) {
                continue;
            }
            let left = if flag.continues_left {
                0
            } else {
                events
                    .binary_search(&flag.bar_start)
                    .expect("should find start event")
            };
            let right = if flag.continues_right {
                events.len() - 1
            } else {
                events
                    .binary_search(&(flag.bar_end - 1))
                    .expect("should find end event")
            };
            let name = short_name(&flag.name, name_limit);
            let width = UnicodeWidthStr::width(name.as_str()) as i64;
            let required = width + 1;
            let extra = (required - (columns[right] - columns[left])).max(0);
            for column in &mut columns[if right == left { right + 1 } else { right }..] {
                *column += extra;
            }
        }
        let mut ordered: Vec<_> = flags.iter().collect();
        ordered.sort_by_key(|flag| (flag.bar_start, flag.id, flag.piece));
        let mut margin = 0;
        let mut pieces = Vec::with_capacity(flags.len());
        for flag in ordered {
            let left_event = if flag.continues_left {
                columns[0]
            } else {
                columns[events
                    .binary_search(&flag.bar_start)
                    .expect("should find start event")]
            };
            let right_event = if flag.continues_right {
                *columns.last().expect("should have a compact column")
            } else {
                columns[events
                    .binary_search(&(flag.bar_end - 1))
                    .expect("should find end event")]
            };
            let label = (label_piece.get(&flag.id) == Some(&flag.piece))
                .then(|| short_name(&flag.name, name_limit));
            let single_column_label_width = if left_event == right_event {
                label
                    .as_deref()
                    .map_or(0, |text| UnicodeWidthStr::width(text) as i64 + 1)
            } else {
                0
            };
            let right_event = right_event + single_column_label_width;
            if let Some(text) = &label {
                let width = UnicodeWidthStr::width(text.as_str()) as i64;
                margin = margin.max(-label_start(left_event, right_event, width));
            }
            let (left_glyph, right_glyph) =
                end_glyphs(flag.strand, flag.continues_left, flag.continues_right);
            pieces.push(CompactPiece {
                id: flag.id,
                piece: flag.piece,
                color: flag.color,
                row: 0,
                left: left_event,
                right: right_event,
                label,
                left_glyph,
                right_glyph,
            });
        }
        let mut rows: Vec<Vec<(i64, i64)>> = Vec::new();
        for piece in &mut pieces {
            piece.left += margin;
            piece.right += margin;
            let label_width = piece
                .label
                .as_deref()
                .map_or(0, |text| UnicodeWidthStr::width(text) as i64);
            let label_start = if piece.label.is_some() {
                label_start(piece.left, piece.right, label_width)
            } else {
                piece.left
            };
            let interval = (
                piece.left.min(label_start),
                (piece.right + 1).max(label_start + label_width),
            );
            let wanted = preferred.get(&piece.id).copied().unwrap_or(0);
            let row = (0..=rows.len())
                .filter(|&row| {
                    rows.get(row).is_none_or(|intervals| {
                        intervals
                            .iter()
                            .all(|other| interval.1 < other.0 || other.1 < interval.0)
                    })
                })
                .min_by_key(|&row| (row.abs_diff(wanted), row))
                .expect("should have a free new row");
            if row == rows.len() {
                rows.push(Vec::new());
            }
            rows[row].push(interval);
            piece.row = row;
            preferred.insert(piece.id, row);
        }
        result.insert(
            *node,
            CompactNode {
                width: pieces
                    .iter()
                    .map(|piece| {
                        let label_width = piece
                            .label
                            .as_deref()
                            .map_or(0, |label| UnicodeWidthStr::width(label) as i64);
                        if piece.label.is_some() {
                            (piece.right + 1).max(
                                label_start(piece.left, piece.right, label_width) + label_width,
                            )
                        } else {
                            piece.right + 1
                        }
                    })
                    .max()
                    .unwrap_or(1)
                    .max(margin + columns.last().copied().unwrap_or(0) + 1),
                height: 2 * rows.len() + 1,
                margin,
                events,
                columns,
                base_markers,
                pieces,
            },
        );
    }
    result
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use gen_core::{HashId, Strand};
    use gen_graph::GraphNode;
    use ratatui::style::Color;

    use super::{CompactNode, build_layouts, end_glyphs, label_start, short_name};
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
        assert!(layouts[&nested].pieces[0].right - layouts[&nested].pieces[0].left >= 3);
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
        assert_eq!(layout.map_column(3), layout.margin + 1);
        assert_eq!(layout.raw_column(layout.map_column(5)), 6);
    }

    #[test]
    fn test_adjacent_labels_use_separate_rows() {
        let graph_node = node("adjacent labels");
        let flags = HashMap::from([(graph_node, vec![flag("A", 0, 2, 4), flag("B", 0, 5, 7)])]);
        let layout = &build_layouts(&flags, 10)[&graph_node];
        assert_eq!(layout.pieces[0].right + 1, layout.pieces[1].left);
        assert_ne!(layout.pieces[0].row, layout.pieces[1].row);
    }

    #[test]
    fn test_adjacent_annotations_keep_distinct_base_columns() {
        let graph_node = node("shared coordinate");
        let flags = HashMap::from([(graph_node, vec![flag("A", 0, 2, 4), flag("B", 0, 4, 7)])]);
        let layout = &build_layouts(&flags, 10)[&graph_node];
        assert_eq!(layout.events, [2, 3, 4, 6]);
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
        assert!(one.right > one.left);
        assert_eq!((one.left_glyph, one.right_glyph), ('═', '▶'));
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
        assert!(name_start > piece.left);
        assert!(name_start + 8 <= piece.right);
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
        assert!(name_start > labeled.left);
        assert!(name_start + 4 <= labeled.right);
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
    fn test_empty_node_has_single_cell() {
        let empty = CompactNode::empty(0);
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
            assert_eq!(layout.width, length.min(3));
            assert_eq!(layout.events.first(), Some(&0));
            assert_eq!(layout.events.last(), Some(&(length - 1)));
            for raw in &layout.events {
                assert_eq!(layout.raw_column(layout.map_column(*raw)), *raw);
            }
        }
    }

    #[test]
    fn test_end_glyph_table() {
        assert_eq!(end_glyphs(Strand::Forward, false, false), ('═', '▶'));
        assert_eq!(end_glyphs(Strand::Forward, true, false), ('═', '▶'));
        assert_eq!(end_glyphs(Strand::Forward, false, true), ('═', '═'));
        assert_eq!(end_glyphs(Strand::Reverse, false, false), ('◀', '═'));
        assert_eq!(end_glyphs(Strand::Reverse, false, true), ('◀', '═'));
        assert_eq!(end_glyphs(Strand::Reverse, true, false), ('═', '═'));
        assert_eq!(end_glyphs(Strand::Unknown, false, false), ('═', '═'));
    }

    #[test]
    fn test_unicode_name_limit() {
        assert_eq!(short_name("abcdefghijk", 10), "abcdefghi…");
        assert_eq!(short_name("e\u{301}e\u{301}e\u{301}", 2), "e\u{301}…");
        assert_eq!(short_name("漢字abc", 4), "漢…");
    }
}
