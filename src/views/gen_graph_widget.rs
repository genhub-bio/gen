use std::collections::{HashMap, HashSet, VecDeque};

use gen_core::{
    INDETERMINATE_CHROMOSOME_INDEX, NO_CHROMOSOME_INDEX, PRESERVE_EDIT_SITE_CHROMOSOME_INDEX,
    is_end_node, is_start_node,
};
use gen_graph::{GenGraph, GraphEdge, GraphNode};
use gen_models::{db::GraphConnection, locus::GraphLocus, node::Node, sequence::SequenceError};
use gen_tui::{
    ViewportState,
    geometry::{WorldPos, WorldRect},
    graph_controller::{GraphController, WorldBuffer},
    graph_widget::{GraphWidget, NODE_GLYPH},
    layout::VisualDetail,
    plotter::{NodeRenderer, NodeSizer, PathStyle},
    theme::current_theme,
};
use petgraph::visit::NodeIndexable;
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
};

use crate::views::{
    annotation_track::{
        AnnotationSpan, graph_locus_from_annotation_span, span_covered_by_later, span_label_text,
        span_should_hide_in_truncated,
    },
    graph_overlay::GraphOverlay,
    inline_label_placement::draw_label_near_pos,
};

/// Labels for special start/end nodes
pub mod label {
    pub const START: &str = "╟";
    pub const END: &str = "╢";
}

/// Domain-specific node sizer for GenGraph that calculates visual dimensions
/// based on genomic sequence length.
#[derive(Clone)]
pub struct GenGraphNodeSizer;

impl NodeSizer<GenGraph> for GenGraphNodeSizer {
    /// Calculate how much screen space a GenGraph node needs based on sequence length and level of detail
    fn get_node_size(&self, node: &GraphNode, detail_level: VisualDetail) -> (u64, u64) {
        // Handle special start/end nodes with fixed label sizes (always show full label)
        if is_start_node(node.node_id) {
            return (label::START.chars().count() as u64, 1u64);
        }
        if is_end_node(node.node_id) {
            return (label::END.chars().count() as u64, 1u64);
        }

        let sequence_length = (node.sequence_end - node.sequence_start) as u64;
        match detail_level {
            VisualDetail::Minimal => (1u64, 1u64), // Just a glyph
            VisualDetail::Truncated => (sequence_length.min(13), 1u64), // 13 = 5 border + 3 mid + 5 border
            VisualDetail::Full => (sequence_length, 1u64),              // Full sequence length
        }
    }

    /// Map a raw sequence column to the visual cell it occupies for `detail_level`.
    ///
    /// In `Truncated` mode a node longer than 13 bases is drawn as `AAAAA...BBBBB`, so
    /// interior columns collapse onto the central `...` cell. `Minimal` mode draws a
    /// single glyph, so every column maps to column 0.
    fn clamp_column(&self, node: &GraphNode, raw_col: i64, detail_level: VisualDetail) -> i64 {
        match detail_level {
            VisualDetail::Minimal => 0,
            VisualDetail::Truncated if node.length() > 13 => {
                clamp_truncated_col(raw_col, node.length())
            }
            _ => raw_col,
        }
    }
}

/// Domain-specific node renderer for GenGraph that handles database sequence fetching
/// and genomic sequence visualization with caching.
pub struct GenGraphNodeRenderer<'a> {
    conn: &'a GraphConnection,
    cache: HashMap<GraphNode, String>,
}

impl<'a> GenGraphNodeRenderer<'a> {
    /// Create a new GenGraph node renderer with database connection
    pub fn new(conn: &'a GraphConnection) -> Self {
        Self {
            conn,
            cache: HashMap::new(),
        }
    }

    /// Get the database connection (for accessing from other code)
    pub fn connection(&self) -> &'a GraphConnection {
        self.conn
    }

    /// Fetch sequence for a GenGraph node with caching
    pub fn get_sequence(&mut self, node_key: &GraphNode) -> Result<String, SequenceError> {
        // Check cache first
        if let Some(cached) = self.cache.get(node_key) {
            return Ok(cached.clone());
        }

        // Cache miss - query database
        let (db_node_id, start, end) = (
            node_key.node_id,
            node_key.sequence_start,
            node_key.sequence_end,
        );
        let sequences = Node::get_sequences_by_node_ids(self.conn, &[db_node_id]);
        let sequence = match sequences.get(&db_node_id) {
            Some(seq) => seq.get_sequence(start, end)?,
            None => "?".repeat((end - start).max(0) as usize),
        };

        self.cache.insert(*node_key, sequence.clone());
        Ok(sequence)
    }
}

impl NodeRenderer<GenGraph> for GenGraphNodeRenderer<'_> {
    /// Render a GenGraph node with genomic sequence data and theme styling
    /// The rendering changes based on level of detail:
    /// - Minimal: Simple glyph representation
    /// - Truncated: Sequence with inner truncation if needed
    /// - Full: Complete sequence display
    fn render_node(
        &mut self,
        buffer: &mut WorldBuffer,
        area: WorldRect,
        node_id: &GraphNode,
        detail_level: VisualDetail,
    ) {
        let theme = current_theme();
        let background_style = Style::default().bg(theme[0x05]);
        let text_style = Style::default().bg(theme[0x05]).fg(theme[0x00]);

        buffer.fill_rect(area, ' ');
        buffer.set_char_styled(area.left_center(), ' ', background_style);

        // Handle special start/end nodes (always show full label)
        if is_start_node(node_id.node_id) {
            let edge_style = Style::default().bg(theme[0x00]).fg(theme[0x05]);
            buffer.set_string_styled(area.left_center(), label::START, edge_style);
            return;
        }
        if is_end_node(node_id.node_id) {
            let edge_style = Style::default().bg(theme[0x00]).fg(theme[0x05]);
            buffer.set_string_styled(area.left_center(), label::END, edge_style);
            return;
        }

        match detail_level {
            VisualDetail::Minimal => {
                // Base scale: Just show a simple glyph
                let text_style = Style::default().fg(theme[0x05]).bg(theme[0x00]);
                buffer.set_string_styled(area.left_center(), &NODE_GLYPH.to_string(), text_style);
            }
            VisualDetail::Truncated => {
                // 13 cells: 5 left bases + "..." + 5 right bases.
                let sequence = self
                    .get_sequence(node_id)
                    .unwrap_or_else(|_| "Unknown Sequence".to_string());
                let max_width = 13u32;
                let truncated = inner_truncation(&sequence, max_width);
                buffer.set_string_styled(area.left_center(), &truncated, text_style);
            }
            VisualDetail::Full => {
                // Full scale: Show complete sequence (truncated only by area width)
                let sequence = self
                    .get_sequence(node_id)
                    .unwrap_or_else(|_| "Unknown Sequence".to_string());
                buffer.set_string_styled(area.left_center(), &sequence, text_style);
            }
        }
    }
}

/// Truncate a genomic sequence from the inside, keeping the beginning and end.
///
/// # Arguments
/// * `s` - The sequence string to truncate
/// * `target_length` - Maximum length for the output string
///
/// # Returns
/// A string showing beginning...end if truncation needed, or original if short enough.
pub fn inner_truncation(s: &str, target_length: u32) -> String {
    if s.len() <= target_length as usize {
        return s.to_string();
    } else if target_length < 5 {
        return NODE_GLYPH.to_string(); // ⏺ is U+23FA
    }
    // length - 3 because we need space for the ellipsis
    let left_len = (target_length - 3) / 2 + ((target_length - 3) % 2);
    let right_len = (target_length - 3) / 2;

    let left = &s[..left_len as usize];
    let right = &s[(s.len() - right_len as usize)..];

    format!("{}...{}", left, right)
}

/// Convenience function to create a GenGraph widget with all domain-specific components
/// configured. This provides a simple interface for creating GenGraph visualizations.
///
/// # Arguments
/// * `conn` - Database connection for sequence fetching
///
/// # Returns
/// A configured GraphWidget ready to visualize GenGraph data
pub fn create_gen_graph_widget(
    conn: &GraphConnection,
) -> GraphWidget<'_, GenGraph, GenGraphNodeSizer, GenGraphNodeRenderer<'_>> {
    let renderer = GenGraphNodeRenderer::new(conn);
    GraphWidget::with_renderer(renderer)
}

/// Compute which edges would be removed by `BlockGroup::prune_graph`.
///
/// Mirrors the per-source-node, per-chromosome_index deduplication logic: for each
/// chromosome_index appearing on outgoing edges of a node, the edge with the highest
/// `created_on` is kept; all others are dimmed. Edges with
/// `PRESERVE_EDIT_SITE_CHROMOSOME_INDEX` are always dimmed; edges with
/// `NO_CHROMOSOME_INDEX` or `INDETERMINATE_CHROMOSOME_INDEX` are never dimmed.
fn compute_pruned_edges(graph: &GenGraph) -> HashSet<(GraphNode, GraphNode)> {
    let mut pruned: HashSet<(GraphNode, GraphNode)> = HashSet::new();

    for node in graph.nodes() {
        // chromosome_index -> (source, target, best_created_on)
        let mut edges_by_ci: HashMap<i64, (GraphNode, GraphNode, i64)> = HashMap::new();

        for (source_node, target_node, edge_weights) in graph.edges(node) {
            for edge_weight in edge_weights {
                let GraphEdge {
                    chromosome_index,
                    created_on,
                    ..
                } = *edge_weight;

                if chromosome_index == NO_CHROMOSOME_INDEX
                    || chromosome_index == INDETERMINATE_CHROMOSOME_INDEX
                {
                    continue;
                }
                if chromosome_index == PRESERVE_EDIT_SITE_CHROMOSOME_INDEX {
                    pruned.insert((source_node, target_node));
                    continue;
                }
                edges_by_ci
                    .entry(chromosome_index)
                    .and_modify(|(best_src, best_tgt, best_ts)| {
                        if created_on > *best_ts {
                            pruned.insert((*best_src, *best_tgt));
                            *best_src = source_node;
                            *best_tgt = target_node;
                            *best_ts = created_on;
                        } else {
                            pruned.insert((source_node, target_node));
                        }
                    })
                    .or_insert((source_node, target_node, created_on));
            }
        }
    }

    pruned
}

/// Find nodes that become inaccessible when all pruned edges are removed.
///
/// BFS from all start nodes following only non-pruned edges. Any node not reached
/// is only reachable through pruned (lowlighted) edges and should be dimmed.
fn compute_inaccessible_nodes(
    graph: &GenGraph,
    pruned: &HashSet<(GraphNode, GraphNode)>,
) -> Vec<GraphNode> {
    let mut reachable: HashSet<GraphNode> = HashSet::new();
    let mut queue: VecDeque<GraphNode> =
        graph.nodes().filter(|n| is_start_node(n.node_id)).collect();
    for &node in &queue {
        reachable.insert(node);
    }

    while let Some(node) = queue.pop_front() {
        for (src, tgt, _) in graph.edges(node) {
            if !pruned.contains(&(src, tgt)) && reachable.insert(tgt) {
                queue.push_back(tgt);
            }
        }
    }

    graph.nodes().filter(|n| !reachable.contains(n)).collect()
}

/// Create a configured GraphController for a GenGraph with the standard theme and settings.
///
/// This is the standard way to initialize a graph controller for GenGraph visualization.
/// It applies the application's theme colors, sets the detail level to Truncated, and
/// Starts in free-camera mode (cursor hidden until the user clicks a node or uses keyboard nav).
///
/// # Arguments
/// * `graph` - The GenGraph to visualize
///
/// # Returns
/// A configured GraphController ready for use with `create_gen_graph_widget`
pub fn create_gen_graph_controller(
    graph: GenGraph,
) -> GraphController<GenGraph, GenGraphNodeSizer> {
    let pruned = compute_pruned_edges(&graph);
    let inaccessible = compute_inaccessible_nodes(&graph, &pruned);
    let node_sizer = GenGraphNodeSizer;
    let mut controller = GraphController::new(graph, node_sizer);
    for edge in pruned {
        controller.dim_edge(edge);
    }
    for node in inaccessible {
        controller.dim_node(node);
    }
    controller.set_detail_level(VisualDetail::Truncated);
    controller.hide_cursor();
    controller
}

/// Navigate to an exact byte offset within a node, snapping the camera left.
///
/// Build a GraphNode → (WorldPos, node_size) lookup from the current viewport graph.
pub fn viewport_pos_map<S: NodeSizer<GenGraph>>(
    controller: &GraphController<GenGraph, S>,
) -> HashMap<GraphNode, (WorldPos, (u64, u64))> {
    let graph = controller.graph();
    controller
        .get_viewport_graph()
        .data_nodes()
        .map(|(world_pos, domain_idx, layout_node)| {
            let node = <&GenGraph as NodeIndexable>::from_index(&graph, domain_idx.index());
            (node, (world_pos, layout_node.size))
        })
        .collect()
}

/// Compute the world-space bounding corners of the matched region in a `GraphLocus`.
///
/// Returns `(left_pos, right_pos)` where:
/// - `left_pos` is the world position of the first matched column in `blocks[0]`,
///   with the minimum y across all blocks
/// - `right_pos` is the world position of the last matched column in `blocks[last]`,
///   with the maximum y across all blocks
///
/// Column offsets are clamped with `NodeSizer::clamp_column` so they map correctly in
/// every detail level (e.g. truncated nodes collapse interior columns to the `...` cell).
/// The x bounds are clipped to the visible camera rect so viewport-spanning annotations
/// place their label near the visible portion rather than off-screen.
///
/// Returns `None` if no block in the locus is present in `pos_map` (all off-screen),
/// or if the annotation's x span is entirely outside the camera rect.
pub fn locus_label_bounds(
    locus: &GraphLocus,
    pos_map: &HashMap<GraphNode, (WorldPos, (u64, u64))>,
    detail_level: VisualDetail,
    viewport_state: &ViewportState,
) -> Option<(WorldPos, WorldPos)> {
    let block_world_pos = |block: GraphNode, col_raw: i64| -> Option<WorldPos> {
        let &(center, size) = pos_map.get(&block)?;
        let rect = WorldRect::from_center_and_size(center, size);
        let col = GenGraphNodeSizer.clamp_column(&block, col_raw, detail_level);
        Some(WorldPos::new(rect.min.x + col, center.y))
    };

    let last = locus.slices.len() - 1;

    let left_pos = locus.slices.iter().enumerate().find_map(|(i, s)| {
        let col_raw = if i == 0 { s.start as i64 } else { 0 };
        block_world_pos(s.block, col_raw)
    })?;

    let right_pos = locus.slices.iter().enumerate().rev().find_map(|(i, s)| {
        let col_raw = if i == last {
            s.end.saturating_sub(1) as i64
        } else {
            s.block.length() - 1
        };
        block_world_pos(s.block, col_raw)
    })?;

    let mut y_min = i64::MAX;
    let mut y_max = i64::MIN;
    for s in &locus.slices {
        let Some(&(center, _)) = pos_map.get(&s.block) else {
            continue;
        };
        y_min = y_min.min(center.y);
        y_max = y_max.max(center.y);
    }

    if y_min == i64::MAX {
        return None;
    }

    let cam = viewport_state.camera_rect();
    let clipped_left_x = left_pos.x.max(cam.min.x);
    let clipped_right_x = right_pos.x.min(cam.max.x);
    if clipped_right_x < clipped_left_x {
        return None;
    }

    let left_pos = WorldPos::new(clipped_left_x, y_max);
    let right_pos = WorldPos::new(clipped_right_x, y_min);

    Some((left_pos, right_pos))
}

/// Map a sequence offset to a visual cell column inside a 13-cell truncated node.
///
/// Display layout: `AAAAA...BBBBB` — first 5 bases (cells 0-4), `...` (cells 5-7),
/// last 5 bases (cells 8-12). Interior positions that fall in the `...` region clamp
/// to cell 6 (the centre dot).
fn clamp_truncated_col(value: i64, block_seq_len: i64) -> i64 {
    if value < 5 {
        value
    } else if block_seq_len - value <= 5 {
        13 - (block_seq_len - value)
    } else {
        6
    }
}

/// Highlight only the matched bytes of a `GraphLocus`, using sub-rect tinting.
///
/// - Start node: tinted from `start_offset` to its right edge.
/// - Middle nodes: fully tinted.
/// - End node: tinted from its left edge to `end_offset` (exclusive).
///
/// The stored columns are raw sequence offsets; the controller maps them to visual
/// cells for the current detail level on every rebuild via `NodeSizer::clamp_column`,
/// so the highlight self-heals across detail and zoom changes.
pub fn highlight_locus<S: NodeSizer<GenGraph>>(
    controller: &mut GraphController<GenGraph, S>,
    m: &GraphLocus,
    style: PathStyle,
) {
    for s in &m.slices {
        let col_start = s.start as i64;
        let col_end = s.end.saturating_sub(1) as i64;
        controller.set_cell_highlight(s.block, (col_start, 0), (col_end, 0), style);
    }

    for (s, t) in m.slices.iter().zip(m.slices.iter().skip(1)) {
        controller.set_edge_highlight((s.block, t.block), style);
    }
}

/// Re-register every overlay highlight on `controller`, replacing whatever highlights
/// were previously set.
///
/// Span overlays are applied longest-first so shorter (inner) spans paint on top; any
/// path overlay is applied last so the route paints over the span tints. Callers run this
/// after any change that invalidates clamped highlight columns (zoom, detail change) or,
/// in the live TUI viewers, every frame because the overlay set changes with scrolling.
pub fn reapply_overlays<S: NodeSizer<GenGraph>>(
    controller: &mut GraphController<GenGraph, S>,
    overlays: &[GraphOverlay],
) {
    let mut spans: Vec<(&AnnotationSpan, PathStyle)> = overlays
        .iter()
        .filter_map(|overlay| overlay.span().map(|span| (span, overlay.style)))
        .collect();
    spans.sort_by_key(|(span, _)| {
        -(span
            .segments
            .iter()
            .map(|segment| segment.end - segment.start)
            .sum::<i64>())
    });
    let loci_and_styles: Vec<(GraphLocus, PathStyle)> = spans
        .iter()
        .filter_map(|(span, style)| {
            graph_locus_from_annotation_span(span, controller.graph()).map(|locus| (locus, *style))
        })
        .collect();
    controller.clear_all_highlights();
    for (locus, style) in &loci_and_styles {
        highlight_locus(controller, locus, *style);
    }
    for overlay in overlays {
        if let Some(nodes) = overlay.path_nodes() {
            controller.set_path_highlight(overlay.style, nodes.to_vec());
        }
    }
}

/// Draw floating labels for `overlays` after the graph has been rendered into `buf`.
///
/// Overlays are labelled longest-first so the covered-by-later check matches highlight
/// paint order. A label is suppressed when its span is fully covered by a shorter overlay
/// on top, when it collapses into a truncated node, or when no free cell is found near its
/// span. Returns `true` if any labelled overlay was suppressed, so the caller can show a
/// single "some annotations hidden" hint.
pub fn draw_annotation_labels<S: NodeSizer<GenGraph>>(
    buf: &mut Buffer,
    area: Rect,
    controller: &GraphController<GenGraph, S>,
    overlays: &[GraphOverlay],
) -> bool {
    let mut labeled: Vec<(&AnnotationSpan, PathStyle)> = overlays
        .iter()
        .filter_map(|overlay| {
            overlay
                .span()
                .filter(|span| !span.name.is_empty())
                .map(|span| (span, overlay.style))
        })
        .collect();
    if labeled.is_empty() {
        return false;
    }
    labeled.sort_by_key(|(span, _)| {
        -(span
            .segments
            .iter()
            .map(|segment| segment.end - segment.start)
            .sum::<i64>())
    });

    let span_refs: Vec<&AnnotationSpan> = labeled.iter().map(|(span, _)| *span).collect();
    let pos_map = viewport_pos_map(controller);
    let detail_level = controller.get_detail_level();
    let theme = current_theme();
    let max_distance = if detail_level == VisualDetail::Minimal {
        10
    } else {
        5
    };

    let mut any_hidden = false;
    for (idx, (span, style)) in labeled.iter().enumerate() {
        let Some(locus) = graph_locus_from_annotation_span(span, controller.graph()) else {
            continue;
        };
        if span_covered_by_later(span, idx, &span_refs) {
            any_hidden = true;
            continue;
        }
        if detail_level == VisualDetail::Truncated
            && span_should_hide_in_truncated(span, controller.graph())
        {
            any_hidden = true;
            continue;
        }
        let Some(bounds) =
            locus_label_bounds(&locus, &pos_map, detail_level, &controller.viewport_state)
        else {
            continue;
        };
        let color = match style.color {
            Color::Reset => theme[0x06],
            other => other,
        };
        let label = span_label_text(span);
        if draw_label_near_pos(
            buf,
            area,
            bounds,
            &label,
            color,
            &controller.viewport_state,
            max_distance,
        )
        .is_none()
        {
            any_hidden = true;
        }
    }
    any_hidden
}

#[cfg(test)]
mod tests {
    use gen_tui::{
        geometry::WorldPos,
        viewport_state::{ViewportState, WorldBuffer},
    };
    use ratatui::backend::TestBackend;

    use super::*;

    /// Test coordinate handling for very large genomic sequences
    ///
    /// Genomic sequences can span hundreds of thousands of base pairs, creating
    /// world coordinates that exceed u16::MAX (65,535) when rendered. This test
    /// verifies that the coordinate conversion system handles such large values
    /// correctly without integer overflow or wraparound artifacts.
    #[test]
    fn test_coordinate_overflow_with_large_genomic_sequences() {
        // Set up a viewport for rendering genomic data
        let mut viewport_state = ViewportState::new();
        viewport_state.viewport_bounds = ratatui::layout::Rect::new(0, 0, 80, 20);

        // Position camera to simulate viewing a region of a large genome
        // where sequence coordinates naturally reach high values
        let camera_center = WorldPos::new(40000, 0);
        viewport_state.camera_current = camera_center;
        viewport_state.camera_target = camera_center;

        let backend = TestBackend::new(80, 20);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        let mut buffer = terminal.current_buffer_mut().clone();
        let world_buffer = WorldBuffer::new(&mut buffer, &viewport_state);

        // Test a coordinate representing the end of a 100K base pair genomic sequence
        // Such large sequences are common in genomics (genes, regulatory regions, etc.)
        let large_genomic_pos = WorldPos::new(camera_center.x + 70000, 0); // ~110K coordinate

        // The coordinate conversion should handle large values gracefully:
        // - Return None if outside viewport (correct behavior)
        // - Never wrap around due to u16 overflow (incorrect behavior)
        let result = world_buffer.world_to_viewport(large_genomic_pos);

        assert!(
            result.is_none(),
            "Large genomic coordinates outside viewport should return None, not wrap around"
        );

        // Verify that normal-sized coordinates still work correctly
        let normal_pos = WorldPos::new(camera_center.x, camera_center.y);
        assert!(
            world_buffer.world_to_viewport(normal_pos).is_some(),
            "Coordinates within normal range should convert successfully"
        );
    }

    #[test]
    fn test_inner_truncation_no_truncation_needed() {
        let s = "hello";
        let truncated = inner_truncation(s, 10);
        assert_eq!(truncated, "hello");
    }

    #[test]
    fn test_inner_truncation_truncate_to_odd_length() {
        let s = "hello world";
        let truncated = inner_truncation(s, 7);
        assert_eq!(truncated, "he...ld");
    }

    #[test]
    fn test_inner_truncation_truncate_to_even_length() {
        let s = "hello world";
        let truncated = inner_truncation(s, 8);
        assert_eq!(truncated, "hel...ld");
    }

    #[test]
    fn test_inner_truncation_empty_string() {
        let s = "";
        let truncated = inner_truncation(s, 5);
        assert_eq!(truncated, "");
    }

    #[test]
    fn test_inner_truncation_short_target() {
        let s = "hello world";
        let truncated = inner_truncation(s, 3);
        assert_eq!(truncated, NODE_GLYPH.to_string());
    }

    #[test]
    fn snapshot_zygosity_pruned_edges() {
        use std::path::PathBuf;

        use gen_models::sample::Sample;
        use gen_tui::{graph_widget::GraphWidget, testing::create_test_terminal};
        use ratatui::widgets::StatefulWidget as _;

        use crate::{
            imports::fasta::import_fasta, test_helpers::setup_gen_on_disk, track_database,
            updates::vcf::update_with_vcf,
        };

        let context = setup_gen_on_disk();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();
        track_database(conn, op_conn).unwrap();

        let collection = "test";
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/simple.fa")
            .to_str()
            .unwrap()
            .to_string();
        let vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/simple_zygosity.vcf")
            .to_str()
            .unwrap()
            .to_string();

        import_fasta(
            &context,
            &fasta_path,
            collection,
            Sample::DEFAULT_NAME,
            false,
        )
        .unwrap();
        update_with_vcf(
            &context,
            &vcf_path,
            collection,
            "".to_string(),
            None,
            vec![Sample::DEFAULT_NAME.to_string()],
            false,
        )
        .unwrap();

        let gen_graph = Sample::get_graph(conn, collection, "SAMPLE1").unwrap();
        let mut controller = create_gen_graph_controller(gen_graph);

        let mut terminal = create_test_terminal(120, 30);
        terminal
            .draw(|f| {
                let area = f.area();
                controller.viewport_state.viewport_bounds = area;
                controller.ensure_camera_coverage().unwrap();
                controller.rebuild_viewport_graph().unwrap();
                GraphWidget::with_renderer(GenGraphNodeRenderer::new(conn)).render(
                    area,
                    f.buffer_mut(),
                    &mut controller,
                );
            })
            .unwrap();

        insta::assert_snapshot!("zygosity_pruned_edges", terminal.backend().to_string());
    }

    /// A cell highlight set in `Full` detail must re-clamp onto the truncated cells
    /// when the detail level changes, without any external `reapply_overlays` call.
    ///
    /// The highlight stores raw content columns; the controller maps them through
    /// `GenGraphNodeSizer::clamp_column` on every rebuild. For a 34-base node drawn
    /// as `AAAAA...BBBBB` (13 cells), raw column 2 stays at cell 2 while raw column
    /// 31 (three from the end) lands at cell 10.
    #[test]
    fn cell_highlight_reclamps_on_detail_change() {
        use std::path::PathBuf;

        use gen_core::strand::Strand;
        use gen_graph::GraphNodeSlice;
        use gen_models::{locus::GraphLocus, sample::Sample};
        use ratatui::{layout::Rect, style::Color};

        use crate::{
            imports::fasta::import_fasta, test_helpers::setup_gen_on_disk, track_database,
        };

        let context = setup_gen_on_disk();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();
        track_database(conn, op_conn).unwrap();

        let collection = "test";
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/simple.fa")
            .to_str()
            .unwrap()
            .to_string();
        import_fasta(
            &context,
            &fasta_path,
            collection,
            Sample::DEFAULT_NAME,
            false,
        )
        .unwrap();

        let gen_graph = Sample::get_graph(conn, collection, Sample::DEFAULT_NAME).unwrap();
        let mut controller = create_gen_graph_controller(gen_graph);

        let block = controller
            .graph()
            .nodes()
            .find(|node| !is_start_node(node.node_id) && !is_end_node(node.node_id))
            .expect("should have a data node");
        assert_eq!(block.length(), 34);

        // Highlight raw columns 2..=31 while showing the full sequence.
        controller.set_detail_level(VisualDetail::Full);
        let locus = GraphLocus {
            slices: vec![GraphNodeSlice {
                block,
                start: 2,
                end: 32,
                strand: Strand::Forward,
            }],
        };
        highlight_locus(&mut controller, &locus, PathStyle::new(Color::Yellow));

        // Switch to the truncated view; the stored raw columns must re-clamp.
        controller.set_detail_level(VisualDetail::Truncated);
        controller.viewport_state.viewport_bounds = Rect::new(0, 0, 120, 30);
        controller.ensure_camera_coverage().unwrap();
        controller.rebuild_viewport_graph().unwrap();

        let highlights = controller.get_cell_highlights();
        assert_eq!(highlights.len(), 1, "expected exactly one cell highlight");
        let (_, top_left, bottom_right, _) = highlights[0];
        assert_eq!(top_left.0, 2, "raw column 2 stays at cell 2");
        assert_eq!(
            bottom_right.0, 10,
            "raw column 31 clamps to cell 10 in the truncated node"
        );
    }
}
