//! The graph state and behaviour shared by the inline widget and the full-screen viewer.
//!
//! Both viewers draw the same graph widget from one [`GenGraphController`]; they differ only in
//! what surrounds it (a border and a help line inline, the collection explorer, search bar,
//! panels and status bars full-screen) and in how annotations are drawn beside it. Because the
//! controller owns everything keyed to the loaded graph (the lazily crawled engine, its view
//! state, dimming, overlays, and which batch the annotation groups were loaded for), switching
//! from the inline widget to the full-screen viewer moves the controller over and reloads
//! nothing.

use std::{
    collections::{HashMap, HashSet},
    error::Error,
    sync::Arc,
};

use crossterm::event::{KeyCode, KeyEvent};
use gen_core::{HashId, PATH_START_NODE_ID, is_end_node, is_start_node};
use gen_graph::{GenGraph, GraphNode, PackedGenGraph};
use gen_models::{block_group::BlockGroup, path::Path, port_crawl::LocateError};
use gen_tui::{
    curved::{EdgeShape, EdgeStyle},
    graph_view::{GraphView, GraphViewState},
    layout::VisualDetail,
    layout_engine::{BatchId, LayoutEngine},
    plotter::{LineStyle, NodeRenderer, PathStyle},
    theme::current_theme,
};
use log::warn;
use ratatui::{
    Frame,
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
    widgets::{Clear, StatefulWidget, Widget},
};

use crate::views::{
    annotation_groups::{AnnotationGroupEntry, load_annotation_group_entries},
    annotations::{AnnotationGroupTrackRequest, load_annotations_for_group},
    block_group::{
        active_neighborhood_node_ids, load_block_group_graph, teleport_through_wormhole,
    },
    gen_graph_widget::{
        self, AlignedPath, AnnotationLabels, AnnotationStarts, NodeAnnotationLayer, OverlayInputs,
        SendSyncZoomLevels, TRUNCATED_ZOOM_LEVEL, align_zoom_levels,
        build_send_sync_annotated_zoom_levels, create_send_sync_annotated_gen_graph_engine_lazy,
        draw_annotation_labels, highlight_adhoc_loci, reapply_overlays, update_node_annotations,
    },
    graph_database::GraphDatabase,
    graph_dimming::GraphDimming,
    graph_overlay::{
        AnnotationColorCache, GraphOverlay, OverlaySource, PathMembership, group_track_key,
        has_path_overlay, remove_path_overlay, remove_track_overlays, replace_track_overlays,
        set_path_overlay,
    },
    lazy_graph_source::EagerOrProjectedSource,
    path_scale::{PathScale, SCALE_CONTENT_GAP, SCALE_ROWS, ScalePlacement},
};

/// What a key press did to the graph.
#[derive(Debug, PartialEq, Eq)]
pub enum GraphKeyOutcome {
    /// Something drawn may have changed.
    Redraw,
    /// The cursor stepped through a door into another batch. Redraw, then discard the input
    /// that queued up while the new batch loaded, so held or repeated keys don't carry the
    /// cursor on past the door.
    EnteredDoor,
    /// Nothing changed; the key is unbound or had nothing to act on.
    Ignore,
}

/// The annotation groups reloaded for a newly active batch.
#[derive(Debug, Default)]
pub struct GroupReload {
    /// Groups with at least one annotation in the batch, now drawn.
    pub loaded: Vec<String>,
    /// One message for each group that failed to load.
    pub warnings: Vec<String>,
}

/// What a mouse click on the graph did.
#[derive(Debug, PartialEq, Eq)]
pub enum ClickOutcome {
    /// The click landed on a door and the cursor stepped through it into another batch.
    EnteredDoor,
    /// The cursor moved to the clicked node.
    SelectedNode,
    /// The click hit nothing; the cursor is hidden.
    Missed,
}

/// What [`GenGraphController::sync_active_world`] brought up to date.
#[derive(Debug, Default)]
pub struct WorldSync {
    /// Whether anything drawn changed, so the frame just drawn is stale.
    pub changed: bool,
    /// Set when the active batch changed and its annotation groups were reloaded.
    pub group_reload: Option<GroupReload>,
}

/// The cursor and camera as node-local sequence offsets, which survive a change in how wide
/// the nodes are drawn.
struct ViewOffsets {
    cursor: Option<(GraphNode, i64)>,
    camera: Option<(GraphNode, i64)>,
    /// Whether the camera was following the cursor, so it should keep doing so.
    follows_cursor: bool,
}

/// One block group's lazily loaded graph together with the view of it.
///
/// It owns its database handle rather than borrowing a connection, so the Jupyter and R widgets,
/// which can't hold a borrow, keep one the same way the terminal viewers do.
pub struct GenGraphController {
    database: GraphDatabase,
    history_ref: Option<String>,
    /// Whether opened block groups leave out the edges `BlockGroup::prune_graph` would remove,
    /// instead of loading and dimming them.
    prune_history: bool,
    /// Seeded with a block group's start and grown batch by batch from SQLite, so opening a
    /// large block group never materializes the whole graph.
    engine: LayoutEngine<PackedGenGraph, EagerOrProjectedSource>,
    zoom_levels: SendSyncZoomLevels,
    view_state: GraphViewState<GraphNode>,
    /// Pruned edges and the nodes only they lead into, synced whenever a draw or a door may
    /// have grown the graph.
    dimming: GraphDimming,
    /// Annotation flags the node renderers draw under their nodes, with names inside the flag
    /// when they fit and just left of it otherwise.
    node_annotations: NodeAnnotationLayer,
    /// The path whose nodes the renderers align on y = 0; see
    /// [`Self::with_aligned_current_path`].
    aligned_path: AlignedPath,
    /// Whether opened block groups align their current path on y = 0 (default).
    align_current_path: bool,
    /// Whether the curved edge style can be shown; see [`Self::set_curved_edges_available`].
    curved_edges_available: bool,
    /// Where the position scale of the aligned path is drawn; see [`PathScale`].
    scale_placement: ScalePlacement,
    /// Where annotations start and end on each loaded node, for the `w`/`b`/`e` keys. Rebuilt with the
    /// highlights, so the stops don't depend on flags being drawn.
    annotation_starts: AnnotationStarts,
    /// The node-local sequence offset the cursor and camera were last aimed at, kept because a
    /// coarser zoom level draws several offsets in one cell (the ellipsis of a truncated node)
    /// or moves the cursor onto a neighbouring base, so the column alone can't say where to
    /// land when zooming back in. Set by `go_to_coordinate` and zooming, and only trusted
    /// while the cursor or camera is still on the column it maps to.
    focus_offset: Option<(GraphNode, i64)>,
    /// `None` until a block group is opened.
    block_group: Option<BlockGroup>,
    /// Fetched once per block group; a batch change only reloads their annotations for the
    /// new batch's nodes.
    annotation_group_entries: Vec<AnnotationGroupEntry>,
    /// The batch the annotation groups were last loaded for. A batch is already the
    /// deliberately-constrained local window, so a reload is only needed when it changes (not
    /// on every pan/zoom within the same batch).
    annotation_groups_world: Option<BatchId>,
    /// Paths the `p` key can highlight; the last one is used.
    paths: Vec<PathMembership>,
    /// Annotation, path and search overlays currently loaded, ready for highlight and label
    /// rendering.
    overlays: Vec<GraphOverlay>,
    annotation_colors: AnnotationColorCache,
    /// Whether the highlights registered in `view_state` are stale: the overlays, the zoom
    /// level, or the loaded batch changed since `reapply_overlays` last ran. Highlights persist
    /// between frames, so plain panning and cursor moves skip that work.
    overlays_dirty: bool,
    /// What the highlights and node annotation flags were last built against. A draw rebuilds
    /// them when this no longer matches, even if nothing marked the overlays dirty.
    applied_overlay_inputs: Option<OverlayInputs>,
    /// Floating labels of the search and path overlays, resolved against the graph as last
    /// drawn, and what they were resolved against (`None` once the overlays are rebuilt).
    /// Annotation names are drawn by the node renderers instead.
    annotation_labels: AnnotationLabels,
    labelled_overlay_inputs: Option<OverlayInputs>,
    /// Colors requested for annotation group annotations by id, applied whenever a batch's
    /// groups are loaded: `Some` pins that color, `None` hides the annotation.
    annotation_color_overrides: HashMap<HashId, Option<Color>>,
    /// Annotation groups (by entry id) switched off, which batch reloads skip. Every other
    /// group of the open block group is loaded for each batch.
    disabled_annotation_groups: HashSet<String>,
}

/// A clone gets its own annotation flag layer and renderers, so drawing one never changes
/// what the other draws, and its own database connection, opened on first use.
impl Clone for GenGraphController {
    fn clone(&self) -> Self {
        let aligned_path = self.aligned_path.detached_copy();
        let node_annotations = NodeAnnotationLayer::with_aligned_path(aligned_path.clone());
        node_annotations.set_compact_names(self.node_annotations.compact_names());
        let zoom_levels = align_zoom_levels(
            build_send_sync_annotated_zoom_levels(
                self.database.sequence_source(),
                node_annotations.clone(),
            ),
            &aligned_path,
        );
        Self {
            database: self.database.clone(),
            history_ref: self.history_ref.clone(),
            prune_history: self.prune_history,
            engine: self.engine.clone(),
            zoom_levels,
            view_state: self.view_state.clone(),
            dimming: self.dimming.clone(),
            node_annotations,
            aligned_path,
            align_current_path: self.align_current_path,
            curved_edges_available: self.curved_edges_available,
            scale_placement: self.scale_placement,
            annotation_starts: self.annotation_starts.clone(),
            focus_offset: self.focus_offset,
            block_group: self.block_group.clone(),
            annotation_group_entries: self.annotation_group_entries.clone(),
            annotation_groups_world: self.annotation_groups_world,
            paths: self.paths.clone(),
            overlays: self.overlays.clone(),
            annotation_colors: self.annotation_colors.clone(),
            overlays_dirty: true,
            applied_overlay_inputs: None,
            annotation_labels: AnnotationLabels::default(),
            labelled_overlay_inputs: None,

            annotation_color_overrides: self.annotation_color_overrides.clone(),
            disabled_annotation_groups: self.disabled_annotation_groups.clone(),
        }
    }
}

impl GenGraphController {
    /// A controller with no block group open, showing just a `PATH_START` placeholder.
    pub fn new(database: GraphDatabase, history_ref: Option<String>) -> Self {
        let mut graph = GenGraph::new();
        graph.add_node(GraphNode {
            node_id: PATH_START_NODE_ID,
            sequence_start: 0,
            sequence_end: 0,
        });
        let aligned_path = AlignedPath::new();
        let node_annotations = NodeAnnotationLayer::with_aligned_path(aligned_path.clone());
        node_annotations.set_compact_names(false);
        let (engine, zoom_levels, view_state) = create_send_sync_annotated_gen_graph_engine_lazy(
            graph,
            EagerOrProjectedSource::Eager {
                surviving_routes: GenGraph::new(),
            },
            database.sequence_source(),
            node_annotations.clone(),
            TRUNCATED_ZOOM_LEVEL,
        );
        let zoom_levels = align_zoom_levels(zoom_levels, &aligned_path);
        Self {
            database,
            history_ref,
            prune_history: false,
            engine,
            zoom_levels,
            view_state,
            dimming: GraphDimming::default(),
            node_annotations,
            aligned_path,
            align_current_path: true,
            curved_edges_available: true,
            scale_placement: ScalePlacement::default(),
            annotation_starts: AnnotationStarts::default(),
            focus_offset: None,
            block_group: None,
            annotation_group_entries: Vec::new(),
            annotation_groups_world: None,
            paths: Vec::new(),
            overlays: Vec::new(),
            annotation_colors: AnnotationColorCache::new(),
            overlays_dirty: true,
            applied_overlay_inputs: None,
            annotation_labels: AnnotationLabels::default(),
            labelled_overlay_inputs: None,

            annotation_color_overrides: HashMap::new(),
            disabled_annotation_groups: HashSet::new(),
        }
    }

    /// A controller with `block_group_id` open.
    pub fn for_block_group(
        database: GraphDatabase,
        block_group_id: &HashId,
        history_ref: Option<String>,
    ) -> Result<Self, Box<dyn Error>> {
        let mut controller = Self::new(database, history_ref);
        controller.open_block_group(block_group_id)?;
        Ok(controller)
    }

    /// Leave out the edges `BlockGroup::prune_graph` would remove, and anything only they lead
    /// to, from the block groups opened from now on, instead of loading and dimming them.
    pub fn with_pruned_history(self, prune_history: bool) -> Self {
        Self {
            prune_history,
            ..self
        }
    }

    /// Route and draw edges in `edge_style`, including in block groups opened from now on.
    pub fn with_edge_style(mut self, edge_style: EdgeStyle) -> Self {
        self.view_state.edge_style = edge_style;
        self
    }

    /// How edges are routed and drawn.
    pub fn edge_style(&self) -> EdgeStyle {
        self.view_state.edge_style
    }

    /// Whether the curved edge style can be shown. A terminal needs the Kitty graphics protocol
    /// for it; a PNG always can. Defaults to true.
    pub fn set_curved_edges_available(&mut self, available: bool) {
        self.curved_edges_available = available;
        if !available && self.view_state.edge_style.is_curved() {
            self.view_state.edge_style = EdgeStyle::Rectilinear;
        }
    }

    /// Whether the curved edge style can be shown; see [`Self::set_curved_edges_available`].
    pub fn curved_edges_available(&self) -> bool {
        self.curved_edges_available
    }

    /// Switch between rectilinear and curved edges. Returns false, changing nothing, when the
    /// curved style is not available.
    pub fn toggle_curved_edges(&mut self) -> bool {
        self.view_state.edge_style = match self.view_state.edge_style {
            EdgeStyle::Rectilinear if self.curved_edges_available => EdgeStyle::Curved,
            EdgeStyle::Rectilinear => return false,
            EdgeStyle::Curved => EdgeStyle::Rectilinear,
        };
        true
    }

    /// The edge lines the last draw left for the caller, in cells of the buffer drawn into.
    /// Empty unless the edge style is curved.
    pub fn edge_shapes(&self) -> &[EdgeShape] {
        &self.view_state.edge_shapes
    }

    /// Align the current path of subsequently opened block groups along y = 0. Variants branch
    /// around that axis, preferably above it. Only the path's own edges are loaded.
    pub fn with_aligned_current_path(self, align_current_path: bool) -> Self {
        Self {
            align_current_path,
            ..self
        }
    }

    /// Draw the aligned path's position scale on the last rows of the view (the default), or
    /// just under the lowest node for an image that is cropped to its content.
    pub fn set_scale_placement(&mut self, scale_placement: ScalePlacement) {
        self.scale_placement = scale_placement;
    }

    /// Return where the position scale is drawn in the view.
    pub fn scale_placement(&self) -> ScalePlacement {
        self.scale_placement
    }

    /// Rows `render_to_buffer` takes from the bottom of an area of `height` rows for the
    /// position scale. A view too short to leave the graph a few rows goes without one.
    fn scale_rows(&self, height: u16) -> u16 {
        let rows = SCALE_ROWS
            + match self.scale_placement {
                ScalePlacement::Bottom => 0,
                ScalePlacement::BelowContent => SCALE_CONTENT_GAP,
            };
        if self.aligned_path.is_set() && height >= rows + 4 {
            rows
        } else {
            0
        }
    }

    /// The terminal row the scale's line is drawn on, given the area the graph was drawn in
    /// and what `buf` holds there. Under the content, the scale clears everything drawn in the
    /// graph: nodes, annotation names, edges in the buffer and the curves deferred to the PNG
    /// renderer, which loop well below the lowest node.
    fn scale_line_row(&self, buf: &Buffer, area: Rect, graph_area: Rect) -> u16 {
        let last_graph_row = graph_area.y + graph_area.height - 1;
        match self.scale_placement {
            ScalePlacement::Bottom => area.y + area.height - SCALE_ROWS,
            ScalePlacement::BelowContent => {
                let lowest_node_row = self
                    .view_state
                    .frame
                    .visible_ids()
                    .filter_map(|node| self.view_state.frame.rect_of(node))
                    .map(|rect| rect.bottom())
                    .min()
                    .and_then(|bottom| self.view_state.screen_to_terminal(0, bottom.max(0)))
                    .map_or(last_graph_row, |(_, row)| row);
                let lowest_content_row = lowest_node_row
                    .max(lowest_buffer_row(buf, graph_area))
                    .max(self.lowest_deferred_edge_row(graph_area));
                lowest_content_row.min(last_graph_row) + SCALE_CONTENT_GAP + 1
            }
        }
    }

    /// The lowest terminal row of `graph_area` that a curve deferred to the PNG renderer reaches
    /// (control points included, so a loop is never cut short), or the top row without any.
    fn lowest_deferred_edge_row(&self, graph_area: Rect) -> u16 {
        let last_row = f64::from(graph_area.y + graph_area.height - 1);
        self.view_state
            .edge_shapes
            .iter()
            .flat_map(|edge| match edge {
                EdgeShape::Line(line) => vec![line.from.1, line.to.1],
                EdgeShape::Bezier(curve) => {
                    vec![curve.from.1, curve.control1.1, curve.control2.1, curve.to.1]
                }
                EdgeShape::Arrow(mark) => vec![mark.center.1],
                EdgeShape::NodeGlow(glow) => vec![glow.bottom_right.1],
                EdgeShape::SymbolGlow(glow) => vec![glow.center.1],
            })
            .fold(f64::from(graph_area.y), f64::max)
            .floor()
            .min(last_row) as u16
    }

    /// Replace the graph with `block_group_id`'s seed at full detail, with no overlays,
    /// paths, or annotation groups loaded.
    pub fn open_block_group(&mut self, block_group_id: &HashId) -> Result<(), Box<dyn Error>> {
        let history_ref = self.history_ref.as_deref();
        let block_group =
            BlockGroup::get_by_id(self.database.connection()?, block_group_id, history_ref)?;
        let loaded = load_block_group_graph(
            &mut self.database,
            block_group_id,
            history_ref,
            self.prune_history,
        )?;
        let edge_style = self.view_state.edge_style;
        (self.engine, self.zoom_levels, self.view_state) =
            create_send_sync_annotated_gen_graph_engine_lazy(
                loaded.graph,
                loaded.source,
                self.database.sequence_source(),
                self.node_annotations.clone(),
                TRUNCATED_ZOOM_LEVEL,
            );
        self.view_state.edge_style = edge_style;
        self.zoom_levels = align_zoom_levels(self.zoom_levels.clone(), &self.aligned_path);
        // The position scale reads left to right, so the aligned path's nodes must be laid out in
        // path order even where recombination edges make the graph cyclic.
        let ordering_path = self.aligned_path.clone();
        self.engine
            .set_order_hint(Arc::new(move |node| ordering_path.path_position(&node)));
        self.aligned_path.set(if self.align_current_path {
            let conn = self.database.connection()?;
            BlockGroup::get_current_path(conn, block_group_id, history_ref)
                .ok()
                .map(|path| PathMembership::load(conn, &path.id, history_ref))
        } else {
            None
        });
        self.dimming = GraphDimming::default();
        self.annotation_group_entries =
            load_annotation_group_entries(self.database.connection()?, &block_group, history_ref);
        self.block_group = Some(block_group);
        self.annotation_groups_world = None;
        self.paths.clear();
        self.overlays.clear();
        self.node_annotations.replace(HashMap::new());
        self.focus_offset = None;

        self.overlays_dirty = true;
        Ok(())
    }

    /// The database the open block group is read from, for callers that query it alongside
    /// the view (e.g. listing a notebook graph's annotations).
    pub fn database_mut(&mut self) -> &mut GraphDatabase {
        &mut self.database
    }

    /// Color or hide annotation group annotations by id from the next load on: `Some` pins
    /// that color, `None` hides the annotation. Groups already loaded for the active batch
    /// are loaded again on the next sync.
    pub fn set_annotation_color_overrides(&mut self, overrides: HashMap<HashId, Option<Color>>) {
        self.annotation_color_overrides = overrides;
        self.annotation_groups_world = None;
    }

    /// The open block group's annotation groups, whether or not they are switched on.
    pub fn annotation_group_entries(&self) -> &[AnnotationGroupEntry] {
        &self.annotation_group_entries
    }

    /// Switch annotation group `group_id` (an entry id) on or off. Switching it off removes
    /// its overlays and keeps later batches from loading it; switching it on loads it for the
    /// active batch at the next sync.
    pub fn set_annotation_group_enabled(&mut self, group_id: &str, enabled: bool) {
        if enabled {
            if self.disabled_annotation_groups.remove(group_id) {
                self.annotation_groups_world = None;
            }
        } else if self.disabled_annotation_groups.insert(group_id.to_string()) {
            remove_track_overlays(&mut self.overlays, &group_track_key(group_id));
            self.overlays_dirty = true;
        }
    }

    /// The loaded graph's nodes other than the start and end sentinels.
    pub fn loaded_node_ids(&self) -> HashSet<HashId> {
        self.engine
            .graph()
            .nodes()
            .map(|node| node.node_id)
            .filter(|&node_id| !is_start_node(node_id) && !is_end_node(node_id))
            .collect()
    }

    /// Pin `color` on the overlay span `id`, so automatic coloring never repaints it.
    pub fn pin_annotation_color(&mut self, id: HashId, color: Color) {
        self.annotation_colors.pin(id, color);
        self.overlays_dirty = true;
    }

    /// Switch to zoom level `index`, keeping the cursor and the camera on the same sequence
    /// offsets so the view stays oriented. Node widths differ per level, so fractions of a
    /// node's width are converted through the node-local offset they stand for.
    fn change_zoom(&mut self, index: usize) {
        self.change_view(index, self.node_annotations.compact_names());
    }

    /// Switch to zoom level `index` with truncated nodes showing or hiding annotation names.
    /// Names change how wide nodes are drawn, so the cursor and camera are carried over the
    /// same way as for a zoom.
    fn change_view(&mut self, index: usize, compact_names: bool) {
        let index = index.min(self.zoom_levels.len() - 1);
        if index == self.view_state.zoom_index
            && compact_names == self.node_annotations.compact_names()
        {
            return;
        }
        let offsets = self.capture_view_offsets();
        self.node_annotations.set_compact_names(compact_names);
        gen_graph_widget::apply_zoom_level(&mut self.view_state, index, &self.zoom_levels);
        self.restore_view_offsets(offsets);
        self.overlays_dirty = true;
    }

    /// The column of `node` the cursor is on as the current renderer draws it. The frame may
    /// still hold the previous draw's rects right after a zoom or a re-layout, so the cursor's
    /// fraction is read against the renderer's node width, which is what set it.
    fn cursor_column(&self, node: &GraphNode) -> i64 {
        let renderer = &self.zoom_levels[self.view_state.zoom_index].1;
        (self.view_state.cursor.fractional.0 * column_span(renderer, node)).round() as i64
    }

    /// The node-local sequence offset the current renderer draws at `column` of `node`.
    ///
    /// The aimed-at offset stands in for a column only while that column is still where it
    /// was aimed (or where the cursor snapped it); something else may have moved the view
    /// since, e.g. a caller driving the view state directly.
    fn offset_at_column(&self, node: &GraphNode, column: i64) -> i64 {
        let renderer = &self.zoom_levels[self.view_state.zoom_index].1;
        self.focus_offset
            .filter(|(saved_node, saved_raw)| {
                let last_column = (renderer.get_node_size(node).0 as i64 - 1).max(0);
                let mapped = renderer.map_column(node, *saved_raw).min(last_column);
                let snapped = renderer
                    .cursor_columns(node)
                    .map_or(mapped, |columns| columns.nearest(mapped));
                saved_node == node && (column == mapped || column == snapped)
            })
            .map_or_else(|| renderer.raw_column(node, column), |(_, raw)| raw)
    }

    /// Where the cursor and camera sit as sequence offsets, taken before something changes the
    /// columns those offsets are drawn at. Pair with [`Self::restore_view_offsets`].
    fn capture_view_offsets(&mut self) -> ViewOffsets {
        // The camera follows the cursor when it is anchored on it and the cursor is drawn or
        // was just aimed at; otherwise it holds whatever is in the middle of the view.
        let cursor_node = self.view_state.cursor.node;
        let aimed_node = self.focus_offset.map(|(node, _)| node);
        let follows_cursor = self.view_state.camera.is_some_and(|camera| {
            Some(camera.anchor) == cursor_node
                && (self.view_state.is_cursor_visible() || aimed_node == Some(camera.anchor))
        });
        if !follows_cursor {
            self.view_state.keep_camera_anchor_on_screen();
        }

        let cursor = cursor_node.and_then(|node| {
            self.view_state.frame.rect_of(node)?;
            Some((
                node,
                self.offset_at_column(&node, self.cursor_column(&node)),
            ))
        });
        let camera = self.view_state.camera.map(|camera| {
            let span = column_span(
                &self.zoom_levels[self.view_state.zoom_index].1,
                &camera.anchor,
            );
            let column = (camera.anchor_fraction.0 * span).round() as i64;
            (camera.anchor, self.offset_at_column(&camera.anchor, column))
        });
        ViewOffsets {
            cursor,
            camera,
            follows_cursor,
        }
    }

    /// Put the cursor and camera back on the columns their offsets are drawn at now, so each
    /// stays on the screen cell it was on.
    fn restore_view_offsets(&mut self, offsets: ViewOffsets) {
        let ViewOffsets {
            cursor,
            camera,
            follows_cursor,
        } = offsets;
        let renderer = &self.zoom_levels[self.view_state.zoom_index].1;
        if let Some((node, raw)) = cursor {
            let column = column_of_offset(renderer, &node, raw);
            self.view_state.cursor.fractional.0 = fraction_of_column(renderer, &node, column);
            self.view_state.rest_cursor_on_column(node, column);
        }
        if let (Some((anchor, raw)), Some(camera)) = (camera, self.view_state.camera.as_mut()) {
            let column = column_of_offset(renderer, &anchor, raw);
            camera.anchor_fraction.0 = fraction_of_column(renderer, &anchor, column);
        }
        if follows_cursor && cursor.is_some() {
            self.view_state.follow_cursor_after_snap();
        }
        self.focus_offset = match (cursor, camera) {
            (Some(cursor), _) if follows_cursor => Some(cursor),
            (_, camera) => camera,
        };
    }

    /// Show or hide annotation names in the truncated view, keeping the cursor and camera on the
    /// sequence they were on. Full detail names every annotation either way.
    pub fn show_annotation_names(&mut self, shown: bool) {
        self.change_view(self.view_state.zoom_index, shown);
    }

    /// Whether the truncated view names annotations; see [`Self::show_annotation_names`].
    pub fn annotation_names_shown(&self) -> bool {
        self.node_annotations.compact_names()
    }

    /// Step one zoom level in. Between minimal and full detail the truncated view has two
    /// steps, names hidden and then shown, so zooming in reveals the annotation names before
    /// it widens every node to its sequence.
    pub fn zoom_in(&mut self) {
        let index = self.view_state.zoom_index;
        match (self.detail_level(), self.node_annotations.compact_names()) {
            (VisualDetail::Truncated, false) => self.change_view(index, true),
            (VisualDetail::Minimal, _) => self.change_view(index + 1, false),
            _ => self.change_zoom(index + 1),
        }
    }

    /// Step one zoom level out, back through the truncated view's two name steps.
    pub fn zoom_out(&mut self) {
        let index = self.view_state.zoom_index;
        match (self.detail_level(), self.node_annotations.compact_names()) {
            (VisualDetail::Truncated, true) => self.change_view(index, false),
            (VisualDetail::Full, _) => self.change_view(index.saturating_sub(1), true),
            _ => self.change_zoom(index.saturating_sub(1)),
        }
    }

    /// Jump straight to the first zoom level drawing nodes at `detail`.
    pub fn set_detail_level(&mut self, detail: VisualDetail) {
        if let Some(index) = self
            .zoom_levels
            .iter()
            .position(|(level, _, _)| *level == detail)
        {
            self.change_zoom(index);
        }
    }

    /// Move the camera by a drag of `dx` by `dy` cells, following the pointer one to one.
    pub fn pan(&mut self, dx: i16, dy: i16) {
        self.focus_offset = None;
        self.view_state.move_by_terminal(dx, dy);
        self.view_state.rebase_camera_to_closest_node();
    }

    /// Apply a click at a cell of the area last drawn: a door takes the cursor into the batch
    /// behind it, a node takes the cursor, and anything else hides it.
    pub fn click(&mut self, column: u16, row: u16) -> ClickOutcome {
        if let Some((boundary, target)) = self.view_state.wormhole_hit(column, row) {
            self.teleport_through_wormhole(boundary, target);
            return ClickOutcome::EnteredDoor;
        }
        if self.view_state.handle_click(column, row) {
            ClickOutcome::SelectedNode
        } else {
            ClickOutcome::Missed
        }
    }

    /// Show `coordinate` of node `node_id` at the current detail level, centred when `center` is
    /// set and otherwise against the left edge, with the cursor on it. The block holding it is
    /// located even where the crawl hasn't reached yet, and the batch around it opened. Fails
    /// when the block group lacks that position, or when edits cut it off from the start of a
    /// pruned view, so callers can tell the user which.
    ///
    /// Taking a node coordinate rather than a block keeps positions from elsewhere (a search
    /// result, an annotation) usable, since each graph carves its blocks by its own edges.
    pub fn go_to_coordinate(
        &mut self,
        node_id: HashId,
        coordinate: i64,
        center: bool,
    ) -> Result<(), LocateError> {
        let (source, graph) = self.engine.source_and_graph_mut();
        let node = source.locate(graph, node_id, coordinate)?;
        let offset = coordinate - node.sequence_start;
        let node_budget = self
            .engine
            .neighborhood_node_budget(self.view_state.last_area_width() as usize);
        self.engine
            .activate_batch_containing(node, node_budget)
            .map_err(|error| LocateError::Unavailable(error.to_string()))?;
        // The cursor is placed by fractions of the node's drawn width, which differs per detail
        // level, so the column comes from the renderer.
        let renderer = &self.zoom_levels[self.view_state.zoom_index].1;
        let column = column_of_offset(renderer, &node, offset);
        let fraction = fraction_of_column(renderer, &node, column);
        self.view_state.go_to_node(node, (fraction, 0.5));
        self.view_state.rest_cursor_on_column(node, column);
        self.focus_offset = Some((node, offset));
        if !center {
            self.view_state.queue_snap_left();
        }
        Ok(())
    }

    /// The open block group, if any.
    pub fn block_group(&self) -> Option<&BlockGroup> {
        self.block_group.as_ref()
    }

    pub fn engine(&self) -> &LayoutEngine<PackedGenGraph, EagerOrProjectedSource> {
        &self.engine
    }

    pub fn view_state(&self) -> &GraphViewState<GraphNode> {
        &self.view_state
    }

    pub fn view_state_mut(&mut self) -> &mut GraphViewState<GraphNode> {
        &mut self.view_state
    }

    pub fn zoom_levels(&self) -> &SendSyncZoomLevels {
        &self.zoom_levels
    }

    /// The detail level the current zoom step draws nodes at.
    pub fn detail_level(&self) -> VisualDetail {
        self.zoom_levels[self.view_state.zoom_index].0
    }

    pub fn node_annotations(&self) -> &NodeAnnotationLayer {
        &self.node_annotations
    }

    pub fn overlays(&self) -> &[GraphOverlay] {
        &self.overlays
    }

    /// The overlays, for a change the next draw should show.
    pub fn overlays_mut(&mut self) -> &mut Vec<GraphOverlay> {
        self.overlays_dirty = true;
        &mut self.overlays
    }

    /// The engine alongside the overlays, for loading overlays scoped to the active batch.
    pub fn engine_and_overlays_mut(
        &mut self,
    ) -> (
        &LayoutEngine<PackedGenGraph, EagerOrProjectedSource>,
        &mut Vec<GraphOverlay>,
    ) {
        self.overlays_dirty = true;
        (&self.engine, &mut self.overlays)
    }

    /// The loaded graph, view state and overlays together, for a jump that both moves the
    /// cursor and highlights where it lands.
    pub fn graph_view_and_overlays_mut(
        &mut self,
    ) -> (
        &GenGraph,
        &mut GraphViewState<GraphNode>,
        &mut Vec<GraphOverlay>,
    ) {
        self.overlays_dirty = true;
        (
            self.engine.graph(),
            &mut self.view_state,
            &mut self.overlays,
        )
    }

    /// The annotation groups currently drawn.
    pub fn loaded_annotation_groups(&self) -> impl Iterator<Item = &str> {
        self.overlays
            .iter()
            .filter_map(|overlay| match &overlay.source {
                OverlaySource::Track(key) => key.strip_prefix("group:"),
                _ => None,
            })
            .collect::<HashSet<_>>()
            .into_iter()
    }

    /// Add a path the `p` key can highlight. Only its edge membership is fetched; the
    /// highlight itself is resolved against the loaded graph whenever the overlays reapply.
    pub fn add_path(&mut self, path: &Path) {
        match self.database.connection() {
            Ok(conn) => self.paths.push(PathMembership::load(
                conn,
                &path.id,
                self.history_ref.as_deref(),
            )),
            Err(error) => warn!("Failed to open the graph database: {error}"),
        }
    }

    /// Show or hide the highlight of the last added path, or of the block group's current
    /// path when none was added. Returns whether anything changed.
    pub fn toggle_path(&mut self) -> bool {
        if has_path_overlay(&self.overlays) {
            remove_path_overlay(&mut self.overlays);
            self.overlays_dirty = true;
            return true;
        }
        if self.paths.is_empty()
            && let Some(block_group) = &self.block_group
        {
            let current_path = self
                .database
                .connection()
                .map_err(|error| error.to_string())
                .and_then(|conn| {
                    BlockGroup::get_current_path(conn, &block_group.id, self.history_ref.as_deref())
                        .map_err(|error| error.to_string())
                });
            match current_path {
                Ok(path) => self.add_path(&path),
                Err(error) => warn!("Failed to query path: {error}"),
            }
        }
        let Some(path) = self.paths.last().filter(|path| !path.is_empty()).cloned() else {
            return false;
        };
        let path_style = PathStyle::new(current_theme()[0x09])
            .with_line_style(LineStyle::Bold)
            .with_merge_glyphs(true);
        set_path_overlay(&mut self.overlays, path_style, path);
        self.overlays_dirty = true;
        true
    }

    /// Apply a graph key: zoom, the path toggle, annotation stops, and cursor
    /// navigation, where a door reached by the cursor opens the batch behind it.
    pub fn handle_key(&mut self, key: KeyEvent) -> GraphKeyOutcome {
        match key.code {
            KeyCode::Char('p') => {
                if self.toggle_path() {
                    GraphKeyOutcome::Redraw
                } else {
                    GraphKeyOutcome::Ignore
                }
            }
            KeyCode::Char('s') => {
                if self.toggle_curved_edges() {
                    GraphKeyOutcome::Redraw
                } else {
                    GraphKeyOutcome::Ignore
                }
            }
            KeyCode::Char('+') | KeyCode::Char('=') => {
                self.zoom_in();
                GraphKeyOutcome::Redraw
            }
            KeyCode::Char('-') => {
                self.zoom_out();
                GraphKeyOutcome::Redraw
            }
            KeyCode::Char(key_char @ ('w' | 'b' | 'e'))
                if self.detail_level() != VisualDetail::Minimal =>
            {
                // Reaching the edge of the loaded batch leaves the cursor where it is, like an
                // arrow key with nothing beyond it.
                let annotation_starts = &self.annotation_starts;
                let renderer = &self.zoom_levels[self.view_state.zoom_index].1;
                match self
                    .view_state
                    .move_cursor_to_stop(key_char != 'b', |node| {
                        let raw = if key_char == 'e' {
                            annotation_starts.ends_on(&node)
                        } else {
                            annotation_starts.on(&node)
                        };
                        raw.into_iter()
                            .map(|column| renderer.map_column(&node, column))
                            .collect()
                    }) {
                    Ok(()) => GraphKeyOutcome::Redraw,
                    Err(_) => GraphKeyOutcome::Ignore,
                }
            }
            _ => match self.view_state.handle_key_event(key) {
                Ok(Some((boundary, target))) => {
                    self.teleport_through_wormhole(boundary, target);
                    GraphKeyOutcome::EnteredDoor
                }
                // `handle_key_event` reports keys it doesn't bind the same way as a successful
                // move, so only the navigation keys it binds are worth a redraw.
                Ok(None) if is_navigation_key(key.code) => GraphKeyOutcome::Redraw,
                Ok(None) | Err(_) => GraphKeyOutcome::Ignore,
            },
        }
    }

    /// Step through a door from `boundary` to `target`; see [`teleport_through_wormhole`].
    pub fn teleport_through_wormhole(&mut self, boundary: GraphNode, target: GraphNode) {
        teleport_through_wormhole(&mut self.engine, &mut self.view_state, boundary, target);
    }

    /// Frame the cursor node again at its current fraction on the next draw. A viewer that
    /// takes over the controller calls this, since the camera was placed for the previous
    /// viewer's area.
    pub fn reframe_on_cursor(&mut self) {
        if let Some(cursor_node) = self.view_state.cursor.node {
            let fraction = self.view_state.cursor.fractional;
            self.view_state.go_to_node(cursor_node, fraction);
        }
    }

    /// Bring everything keyed to the loaded graph up to date: dimming for whatever the crawl
    /// added, and on a batch change the annotation groups for the new batch. Call it before a
    /// draw, since a door may have grown the graph, and after one, since rendering can crawl.
    pub fn sync_active_world(&mut self) -> WorldSync {
        let dimming_changed = self.dimming.sync(
            self.engine.graph(),
            self.engine.source(),
            &mut self.view_state,
        );
        // Highlights and annotation flags drawn for a graph that has since grown or moved to
        // another batch are stale too.
        let overlays_stale = self
            .applied_overlay_inputs
            .is_some_and(|inputs| inputs != OverlayInputs::current(&self.engine, &self.view_state));
        let changed = dimming_changed || overlays_stale;
        let current_world = self.engine.active_batch();
        if self.block_group.is_none() || current_world == self.annotation_groups_world {
            return WorldSync {
                changed,
                group_reload: None,
            };
        }
        let node_ids = active_neighborhood_node_ids(&self.engine);
        if node_ids.is_empty() {
            return WorldSync {
                changed,
                group_reload: None,
            };
        }
        let group_reload = self.load_annotation_groups(&node_ids);
        self.annotation_groups_world = current_world;
        self.overlays_dirty = true;
        WorldSync {
            changed: true,
            group_reload: Some(group_reload),
        }
    }

    /// Replace every annotation group overlay with the groups' annotations on `node_ids`,
    /// keeping the path, file and search overlays.
    fn load_annotation_groups(&mut self, node_ids: &HashSet<HashId>) -> GroupReload {
        let mut reload = GroupReload::default();
        if self.block_group.is_none() {
            return reload;
        }
        let conn = match self.database.connection() {
            Ok(conn) => conn,
            Err(error) => {
                reload
                    .warnings
                    .push(format!("Failed to open the graph database: {error}"));
                return reload;
            }
        };
        self.overlays.retain(
            |overlay| !matches!(&overlay.source, OverlaySource::Track(key) if key.starts_with("group:")),
        );
        for entry in &self.annotation_group_entries {
            if self.disabled_annotation_groups.contains(&entry.id) {
                continue;
            }
            let spans = match load_annotations_for_group(&AnnotationGroupTrackRequest {
                conn,
                history_ref: self.history_ref.as_deref(),
                entry,
                projection_graph: self.engine.graph(),
                node_ids,
            }) {
                Ok(spans) => spans,
                Err(error) => {
                    reload.warnings.push(format!(
                        "Failed to load annotations for group {}: {error}",
                        entry.id
                    ));
                    continue;
                }
            };
            let spans: Vec<_> = spans
                .into_iter()
                .filter(|span| match self.annotation_color_overrides.get(&span.id) {
                    Some(None) => false,
                    Some(Some(color)) => {
                        self.annotation_colors.pin(span.id, *color);
                        true
                    }
                    None => true,
                })
                .collect();
            if spans.is_empty() {
                continue;
            }
            reload.loaded.push(entry.id.clone());
            replace_track_overlays(&mut self.overlays, &group_track_key(&entry.id), spans);
        }
        reload
    }

    /// Draw the graph and its annotation names into `area`.
    pub fn render(&mut self, frame: &mut Frame, area: Rect, style: Style) {
        self.render_to_buffer(frame.buffer_mut(), area, style);
    }

    /// Draw one finished frame into `buf` for a viewer that draws on request rather than in an
    /// event loop (the notebook and R widgets): bring the loaded graph up to date, draw, and
    /// draw again if drawing crawled in more of the graph, so the frame never shows stale
    /// dimming or annotations.
    pub fn render_settled(&mut self, buf: &mut Buffer, area: Rect, style: Style) -> WorldSync {
        let before = self.sync_active_world();
        self.render_to_buffer(buf, area, style);
        let after = self.sync_active_world();
        if after.changed {
            Clear.render(area, buf);
            self.render_to_buffer(buf, area, style);
        }
        WorldSync {
            changed: before.changed || after.changed,
            group_reload: after.group_reload.or(before.group_reload),
        }
    }

    /// Draw the graph and its annotation names into `area` of `buf`.
    pub fn render_to_buffer(&mut self, buf: &mut Buffer, full_area: Rect, style: Style) {
        let scale_rows = self.scale_rows(full_area.height);
        let area = Rect {
            height: full_area.height - scale_rows,
            ..full_area
        };
        // Re-register overlay highlights and refill the node annotation flags only when the
        // overlay set, the zoom level, the loaded graph, or how annotations are drawn changed
        // since they were last registered.
        let overlay_inputs = OverlayInputs::current(&self.engine, &self.view_state);
        if self.overlays_dirty || self.applied_overlay_inputs != Some(overlay_inputs) {
            reapply_overlays(
                &self.engine,
                &mut self.view_state,
                &self.zoom_levels,
                &mut self.overlays,
                &mut self.annotation_colors,
            );
            self.annotation_starts = AnnotationStarts::new(&self.engine, &self.overlays);
            update_node_annotations(&self.node_annotations, &self.engine, &self.overlays);
            highlight_adhoc_loci(
                self.engine.graph(),
                &mut self.view_state,
                &self.zoom_levels,
                &self.overlays,
            );
            self.overlays_dirty = false;
            self.applied_overlay_inputs = Some(overlay_inputs);
            self.labelled_overlay_inputs = None;
        }

        let active_renderer = &self.zoom_levels[self.view_state.zoom_index].1;
        let view = GraphView::new(&mut self.engine, active_renderer).style(style);
        StatefulWidget::render(view, area, buf, &mut self.view_state);

        // Resolve the search and path labels against the graph as drawn, which may have just
        // grown.
        let label_inputs = OverlayInputs::current(&self.engine, &self.view_state);
        if self.labelled_overlay_inputs != Some(label_inputs) {
            let floating_overlays: Vec<GraphOverlay> = self
                .overlays
                .iter()
                .filter(|overlay| !overlay.source.is_annotation())
                .cloned()
                .collect();
            self.annotation_labels =
                AnnotationLabels::new(self.engine.graph(), self.detail_level(), &floating_overlays)
                    .with_compact_layout(self.node_annotations.clone());
            self.labelled_overlay_inputs = Some(label_inputs);
        }
        draw_annotation_labels(buf, area, &self.view_state, &self.annotation_labels);
        if scale_rows > 0 {
            // The rows under the graph are outside the area the graph view styles, so give them
            // the same canvas background.
            let scale_area = Rect {
                y: area.y + area.height,
                height: scale_rows,
                ..full_area
            };
            buf.set_style(scale_area, style);
            let scale = PathScale::from_frame(
                &self.view_state.frame,
                &self.aligned_path,
                &self.zoom_levels[self.view_state.zoom_index].1,
                area,
            );
            scale.draw(buf, full_area, self.scale_line_row(buf, full_area, area));
        }
    }

    /// Draw the graph alone into `area`, without the cursor or annotation names.
    pub fn render_plain(&mut self, frame: &mut Frame, area: Rect) {
        self.view_state.hide_cursor();
        let active_renderer = &self.zoom_levels[self.view_state.zoom_index].1;
        let view = GraphView::new(&mut self.engine, active_renderer);
        frame.render_stateful_widget(view, area, &mut self.view_state);
    }
}

/// The lowest row of `area` in `buf` holding a visible glyph, or the top row of an empty area.
fn lowest_buffer_row(buf: &Buffer, area: Rect) -> u16 {
    (area.y..area.y + area.height)
        .rev()
        .find(|&row| {
            (area.x..area.x + area.width).any(|column| buf[(column, row)].symbol().trim() != "")
        })
        .unwrap_or(area.y)
}

/// The column of `node`, as `renderer` draws it, that node-local `offset` is shown at: the
/// ellipsis for an offset inside a truncated node's hidden middle, the single glyph at
/// minimal detail.
fn column_of_offset(
    renderer: &impl NodeRenderer<PackedGenGraph>,
    node: &GraphNode,
    offset: i64,
) -> i64 {
    let last_column = (renderer.get_node_size(node).0 as i64 - 1).max(0);
    renderer.map_column(node, offset).clamp(0, last_column)
}

/// The span of `node`'s columns, from its first to its last: what a fraction of the node's
/// width is a fraction of, as in `WorldRect::point_at_fraction`.
fn column_span(renderer: &impl NodeRenderer<PackedGenGraph>, node: &GraphNode) -> f64 {
    renderer.get_node_size(node).0.saturating_sub(1).max(1) as f64
}

/// The fraction of `node`'s width at which `column` sits.
fn fraction_of_column(
    renderer: &impl NodeRenderer<PackedGenGraph>,
    node: &GraphNode,
    column: i64,
) -> f64 {
    (column as f64 / column_span(renderer, node)).clamp(0.0, 1.0)
}

/// The keys `GraphViewState::handle_key_event` moves the cursor with.
fn is_navigation_key(code: KeyCode) -> bool {
    matches!(
        code,
        KeyCode::Left
            | KeyCode::Right
            | KeyCode::Up
            | KeyCode::Down
            | KeyCode::Char('h' | 'j' | 'k' | 'l')
    )
}

#[cfg(test)]
mod tests {
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use gen_core::Workspace;
    use gen_models::{db::get_connection, path::Path};
    use gen_tui::{
        EdgeStyle, VerticalAnchor,
        curved::{BezierCurve, EdgeShape},
        plotter::LineStyle,
    };
    use ratatui::{
        Terminal,
        backend::TestBackend,
        buffer::Buffer,
        layout::Rect,
        style::{Color, Style},
    };

    use super::{GenGraphController, GraphKeyOutcome, lowest_buffer_row};
    use crate::views::{
        gen_graph_widget::{FULL_ZOOM_LEVEL, TRUNCATED_ZOOM_LEVEL},
        graph_database::GraphDatabase,
        graph_overlay::has_path_overlay,
        lazy_graph_source::tests::setup_labelled_chain_block_group,
        path_scale::ScalePlacement,
    };

    fn draw(controller: &mut GenGraphController, terminal: &mut Terminal<TestBackend>) {
        controller.sync_active_world();
        terminal
            .draw(|frame| {
                controller.render(frame, frame.area(), Style::default());
            })
            .expect("should draw the graph");
        controller.sync_active_world();
    }

    #[test]
    fn test_controller_can_be_held_by_the_python_and_r_widgets() {
        fn assert_owned_and_shareable<T: Send + Sync + 'static>() {}
        assert_owned_and_shareable::<GenGraphController>();
    }

    #[test]
    fn test_path_pinning_does_not_change_node_attachment() {
        let directory = tempfile::tempdir().expect("should create a temporary directory");
        let database_path = directory.path().join("graph.db");
        let (block_group_id, edge_ids) =
            setup_labelled_chain_block_group(&database_path, &["x", "y"]);
        let connection = get_connection(&database_path).expect("should connect to the database");
        let workspace = Workspace::from_current_dir();
        let mut controller = GenGraphController::for_block_group(
            GraphDatabase::for_connection(&connection, &workspace)
                .expect("should open the graph database"),
            &block_group_id,
            None,
        )
        .expect("should load the block group");
        let node = controller
            .engine
            .graph()
            .nodes()
            .next()
            .expect("should have a graph node");
        assert_eq!(
            controller.zoom_levels[FULL_ZOOM_LEVEL]
                .1
                .vertical_anchor(&node),
            VerticalAnchor::Top
        );
        assert!(!controller.zoom_levels[FULL_ZOOM_LEVEL].1.is_aligned(&node));
        Path::create(&connection, "chain", &block_group_id, &edge_ids)
            .expect("should create a path");
        controller
            .open_block_group(&block_group_id)
            .expect("should reopen the graph");
        assert_eq!(
            controller.zoom_levels[FULL_ZOOM_LEVEL]
                .1
                .vertical_anchor(&node),
            VerticalAnchor::Top
        );
        assert!(controller.zoom_levels[FULL_ZOOM_LEVEL].1.is_aligned(&node));
        let cloned = controller.clone();
        assert_eq!(
            cloned.zoom_levels[FULL_ZOOM_LEVEL].1.vertical_anchor(&node),
            VerticalAnchor::Top
        );
        controller = controller.with_aligned_current_path(false);
        controller
            .open_block_group(&block_group_id)
            .expect("should reopen without path alignment");
        assert_eq!(
            controller.zoom_levels[FULL_ZOOM_LEVEL]
                .1
                .vertical_anchor(&node),
            VerticalAnchor::Top
        );
        assert!(!controller.zoom_levels[FULL_ZOOM_LEVEL].1.is_aligned(&node));
        assert_eq!(
            cloned.zoom_levels[FULL_ZOOM_LEVEL].1.vertical_anchor(&node),
            VerticalAnchor::Top
        );
    }

    fn buffer_rows(buffer: &Buffer) -> Vec<String> {
        let area = buffer.area;
        (area.y..area.y + area.height)
            .map(|row| {
                (area.x..area.x + area.width)
                    .map(|column| buffer[(column, row)].symbol())
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_path_scale_follows_the_aligned_path() {
        let directory = tempfile::tempdir().expect("should create a temporary directory");
        let database_path = directory.path().join("graph.db");
        let (block_group_id, edge_ids) =
            setup_labelled_chain_block_group(&database_path, &["x", "y"]);
        let connection = get_connection(&database_path).expect("should connect to the database");
        let workspace = Workspace::from_current_dir();
        Path::create(&connection, "chain", &block_group_id, &edge_ids)
            .expect("should create a path");
        let mut controller = GenGraphController::for_block_group(
            GraphDatabase::for_connection(&connection, &workspace)
                .expect("should open the graph database"),
            &block_group_id,
            None,
        )
        .expect("should load the block group");
        controller.view_state.zoom_index = FULL_ZOOM_LEVEL;
        let area = Rect::new(0, 0, 40, 12);

        let mut buffer = Buffer::empty(area);
        controller.render_settled(&mut buffer, area, Style::default());
        let rows = buffer_rows(&buffer);
        assert!(
            rows[10].contains('|'),
            "should draw tick marks on the second to last row: {rows:#?}"
        );
        assert!(
            rows[11].contains('0'),
            "should label the first base on the last row: {rows:#?}"
        );

        controller.set_scale_placement(ScalePlacement::BelowContent);
        let mut buffer = Buffer::empty(area);
        controller.render_settled(&mut buffer, area, Style::default());
        let rows = buffer_rows(&buffer);
        let line_row = rows
            .iter()
            .position(|row| row.contains('|'))
            .expect("should draw the scale");
        let node_row = rows
            .iter()
            .rposition(|row| row.contains("AAAAA"))
            .expect("should draw the nodes");
        assert_eq!(
            line_row,
            node_row + 3,
            "should leave two blank rows: {rows:#?}"
        );
        assert!(rows[line_row + 1].contains('0'));

        let mut unaligned = controller.with_aligned_current_path(false);
        unaligned
            .open_block_group(&block_group_id)
            .expect("should reopen without path alignment");
        let mut buffer = Buffer::empty(area);
        unaligned.render_settled(&mut buffer, area, Style::default());
        assert!(
            buffer_rows(&buffer).iter().all(|row| !row.contains('|')),
            "should draw no scale without an aligned path"
        );
    }

    #[test]
    fn test_scale_under_the_content_clears_deferred_curves() {
        let directory = tempfile::tempdir().expect("should create a temporary directory");
        let database_path = directory.path().join("graph.db");
        let (block_group_id, edge_ids) =
            setup_labelled_chain_block_group(&database_path, &["x", "y"]);
        let connection = get_connection(&database_path).expect("should connect to the database");
        let workspace = Workspace::from_current_dir();
        Path::create(&connection, "chain", &block_group_id, &edge_ids)
            .expect("should create a path");
        let mut controller = GenGraphController::for_block_group(
            GraphDatabase::for_connection(&connection, &workspace)
                .expect("should open the graph database"),
            &block_group_id,
            None,
        )
        .expect("should load the block group");
        controller.view_state.zoom_index = FULL_ZOOM_LEVEL;
        controller.set_scale_placement(ScalePlacement::BelowContent);
        let area = Rect::new(0, 0, 40, 30);
        let mut buffer = Buffer::empty(area);
        controller.render_settled(&mut buffer, area, Style::default());
        let graph_area = Rect::new(0, 0, 40, 26);
        let row_without_curve = controller.scale_line_row(&buffer, area, graph_area);

        controller
            .view_state
            .edge_shapes
            .push(EdgeShape::Bezier(BezierCurve {
                from: (2.0, 3.0),
                control1: (2.0, 20.5),
                control2: (30.0, 20.5),
                to: (30.0, 3.0),
                color: Color::White,
                line_style: LineStyle::Normal,
            }));
        let row_with_curve = controller.scale_line_row(&buffer, area, graph_area);
        assert!(
            row_with_curve > row_without_curve && row_with_curve > 20,
            "should draw the scale under a curve that loops below the nodes: \
             {row_without_curve} then {row_with_curve}"
        );
    }

    #[test]
    fn test_lowest_buffer_row_ignores_blank_rows() {
        let area = Rect::new(0, 0, 10, 6);
        let mut buffer = Buffer::empty(area);
        assert_eq!(lowest_buffer_row(&buffer, area), 0);
        buffer[(3, 4)].set_symbol("x");
        assert_eq!(lowest_buffer_row(&buffer, area), 4);
    }

    #[test]
    fn test_navigation_keys_do_not_reapply_overlays() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("graph.db");
        let (block_group_id, _) = setup_labelled_chain_block_group(&db_path, &["x", "y", "z"]);
        let conn = get_connection(&db_path).unwrap();
        let workspace = Workspace::from_current_dir();
        let mut controller = GenGraphController::for_block_group(
            GraphDatabase::for_connection(&conn, &workspace)
                .expect("should open the graph database"),
            &block_group_id,
            None,
        )
        .expect("should load the block group");
        let mut terminal =
            Terminal::new(TestBackend::new(80, 12)).expect("should create a test terminal");
        draw(&mut controller, &mut terminal);
        if controller.overlays_dirty {
            draw(&mut controller, &mut terminal);
        }
        assert!(!controller.overlays_dirty);

        let press = |code| KeyEvent::new(code, KeyModifiers::NONE);
        assert_eq!(
            controller.handle_key(press(KeyCode::Char('x'))),
            GraphKeyOutcome::Ignore
        );
        assert_eq!(
            controller.handle_key(press(KeyCode::Right)),
            GraphKeyOutcome::Redraw
        );
        assert!(!controller.overlays_dirty);
        assert_eq!(
            controller.handle_key(press(KeyCode::Char('+'))),
            GraphKeyOutcome::Redraw
        );
        assert!(controller.overlays_dirty);
        // With no paths added and no current path stored, `p` has nothing to toggle.
        assert_eq!(
            controller.handle_key(press(KeyCode::Char('p'))),
            GraphKeyOutcome::Ignore
        );
        for style in [EdgeStyle::Curved, EdgeStyle::Rectilinear] {
            assert_eq!(
                controller.handle_key(press(KeyCode::Char('s'))),
                GraphKeyOutcome::Redraw
            );
            assert_eq!(controller.view_state.edge_style, style);
        }
        assert_eq!(
            controller.handle_key(press(KeyCode::Char('d'))),
            GraphKeyOutcome::Ignore
        );
    }

    #[test]
    fn test_path_toggle_falls_back_to_the_current_path() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("graph.db");
        let (block_group_id, edge_ids) =
            setup_labelled_chain_block_group(&db_path, &["x", "y", "z"]);
        let conn = get_connection(&db_path).unwrap();
        Path::create(&conn, "chain", &block_group_id, &edge_ids).unwrap();
        let workspace = Workspace::from_current_dir();
        let mut controller = GenGraphController::for_block_group(
            GraphDatabase::for_connection(&conn, &workspace)
                .expect("should open the graph database"),
            &block_group_id,
            None,
        )
        .expect("should load the block group");

        assert!(controller.toggle_path());
        assert!(has_path_overlay(controller.overlays()));
        assert!(controller.toggle_path());
        assert!(!has_path_overlay(controller.overlays()));
    }

    #[test]
    fn test_open_block_group_starts_over() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("graph.db");
        let labels: Vec<String> = (0..80).map(|index| format!("n{index}")).collect();
        let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();
        let (block_group_id, edge_ids) = setup_labelled_chain_block_group(&db_path, &label_refs);
        let conn = get_connection(&db_path).unwrap();
        let path = Path::create(&conn, "chain", &block_group_id, &edge_ids).unwrap();
        let workspace = Workspace::from_current_dir();
        let mut controller = GenGraphController::new(
            GraphDatabase::for_connection(&conn, &workspace)
                .expect("should open the graph database"),
            None,
        );
        assert!(controller.block_group().is_none());
        controller
            .open_block_group(&block_group_id)
            .expect("should open the block group");
        controller.add_path(&path);
        controller.toggle_path();
        controller.handle_key(KeyEvent::new(KeyCode::Char('+'), KeyModifiers::NONE));
        let mut terminal =
            Terminal::new(TestBackend::new(12, 12)).expect("should create a test terminal");
        draw(&mut controller, &mut terminal);
        assert!(controller.engine().graph().node_count() > 2);

        controller
            .open_block_group(&block_group_id)
            .expect("should open the block group again");

        assert!(controller.engine().graph().node_count() <= 2);
        assert_eq!(controller.engine().active_batch(), None);
        assert_eq!(controller.view_state().zoom_index, TRUNCATED_ZOOM_LEVEL);
        assert!(controller.overlays().is_empty());
    }

    #[test]
    fn test_block_group_with_one_node_opens_at_normal_detail() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("graph.db");
        let (block_group_id, _) = setup_labelled_chain_block_group(&db_path, &["x"]);
        let conn = get_connection(&db_path).unwrap();
        let workspace = Workspace::from_current_dir();

        let controller = GenGraphController::for_block_group(
            GraphDatabase::for_connection(&conn, &workspace)
                .expect("should open the graph database"),
            &block_group_id,
            None,
        )
        .expect("should load the block group");

        assert_eq!(controller.view_state().zoom_index, TRUNCATED_ZOOM_LEVEL);
        assert!(controller.engine().graph().node_count() <= 2);
    }

    /// A chain of five-base nodes at full detail with annotations on some of them, the way
    /// either viewer shows a block group's annotation groups.
    mod annotated_chain {
        use gen_annotations::projection::AnnotationSegment;
        use gen_core::{HashId, PATH_START_NODE_ID, Strand, range::Range};
        use gen_graph::GraphNode;
        use gen_models::{db::GraphConnection, port_crawl::LocateError};
        use gen_tui::{
            geometry::WorldRect,
            layout::VisualDetail,
            plotter::{LineStyle, PathStyle},
        };
        use petgraph::Direction;
        use ratatui::{buffer::Buffer, layout::Rect, style::Color};

        use super::*;
        use crate::views::{
            annotation_track::AnnotationSpan,
            gen_graph_controller::ClickOutcome,
            gen_graph_widget::{FULL_ZOOM_LEVEL, apply_zoom_level},
            graph_overlay::{GraphOverlay, OverlayContent, OverlaySource, group_track_key},
        };

        /// The active batch's nodes along the chain, in order from its start.
        fn active_chain(controller: &GenGraphController) -> Vec<GraphNode> {
            let graph = controller.engine().graph();
            let mut node = graph
                .nodes()
                .find(|node| node.node_id == PATH_START_NODE_ID)
                .expect("should load the chain's start");
            let mut chain = Vec::new();
            while let Some(next) = graph.neighbors_directed(node, Direction::Outgoing).next() {
                if !controller.engine().active_contains(next) {
                    break;
                }
                chain.push(next);
                node = next;
            }
            chain
        }

        /// Annotate bases 1-3 of each of `nodes`, forward, one annotation per node.
        fn annotate(controller: &mut GenGraphController, nodes: &[GraphNode]) {
            for (index, node) in nodes.iter().enumerate() {
                controller.overlays_mut().push(GraphOverlay {
                    content: OverlayContent::Span(AnnotationSpan {
                        id: HashId::convert_str(&format!("feature {index}")),
                        name: format!("f{index}"),
                        segments: vec![AnnotationSegment {
                            node_id: node.node_id,
                            range: Range {
                                start: node.sequence_start + 1,
                                end: node.sequence_start + 4,
                            },
                            strand: Strand::Forward,
                        }],
                    }),
                    source: OverlaySource::Track("features".to_string()),
                    style: PathStyle {
                        color: Color::Reset,
                        line_style: LineStyle::Normal,
                        merge_glyphs: true,
                    },
                });
            }
        }

        /// A controller over an 80-node chain, zoomed to full detail and drawn once so its
        /// first batch is loaded.
        fn full_detail_chain(
            conn: &GraphConnection,
            workspace: &Workspace,
            block_group_id: &HashId,
            terminal: &mut Terminal<TestBackend>,
        ) -> GenGraphController {
            let mut controller = GenGraphController::for_block_group(
                GraphDatabase::for_connection(conn, workspace)
                    .expect("should open the graph database"),
                block_group_id,
                None,
            )
            .expect("should load the block group");
            let zoom_levels = controller.zoom_levels().clone();
            apply_zoom_level(controller.view_state_mut(), FULL_ZOOM_LEVEL, &zoom_levels);
            draw(&mut controller, terminal);
            controller
        }

        fn chain_block_group(db_path: &std::path::Path) -> HashId {
            let labels: Vec<String> = (0..80).map(|index| format!("n{index}")).collect();
            let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();
            setup_labelled_chain_block_group(db_path, &label_refs).0
        }

        /// The cursor's rect and the screen row it sits on.
        fn cursor_rect_and_row(controller: &GenGraphController) -> (WorldRect, i64) {
            let view_state = controller.view_state();
            let node = view_state.cursor.node.expect("should have a cursor node");
            let rect = view_state
                .frame
                .rect_of(node)
                .expect("should place the cursor node");
            (rect, rect.point_at_fraction(view_state.cursor.fractional).y)
        }

        /// The sequence row of an annotated node, above its flag lanes.
        fn sequence_row(rect: WorldRect) -> i64 {
            rect.max.y
        }

        #[test]
        fn test_truncated_zoom_snaps_to_endpoint_and_restores_base() {
            let dir = tempfile::tempdir().unwrap();
            let db_path = dir.path().join("graph.db");
            let block_group_id = chain_block_group(&db_path);
            let conn = get_connection(&db_path).unwrap();
            let workspace = Workspace::from_current_dir();
            let mut terminal =
                Terminal::new(TestBackend::new(40, 12)).expect("should create a test terminal");
            let mut controller =
                full_detail_chain(&conn, &workspace, &block_group_id, &mut terminal);
            controller.zoom_out();
            assert_eq!(controller.view_state().zoom_index, FULL_ZOOM_LEVEL - 1);
            controller.zoom_in();
            assert_eq!(controller.view_state().zoom_index, FULL_ZOOM_LEVEL);

            let chain = active_chain(&controller);
            let target = chain[1];
            annotate(&mut controller, &[target]);
            draw(&mut controller, &mut terminal);
            controller.view_state_mut().go_to_node(target, (0.5, 0.5));
            draw(&mut controller, &mut terminal);
            let (full_rect, _) = cursor_rect_and_row(&controller);
            let full_column = full_rect
                .point_at_fraction(controller.view_state().cursor.fractional)
                .x
                - full_rect.min.x;
            let raw = controller.zoom_levels()[FULL_ZOOM_LEVEL]
                .1
                .raw_column(&target, full_column);

            controller.zoom_out();
            assert_eq!(controller.detail_level(), VisualDetail::Truncated);
            draw(&mut controller, &mut terminal);
            let (compact_rect, _) = cursor_rect_and_row(&controller);
            let compact_column = compact_rect
                .point_at_fraction(controller.view_state().cursor.fractional)
                .x
                - compact_rect.min.x;
            assert_eq!(
                compact_column,
                controller.zoom_levels()[FULL_ZOOM_LEVEL - 1]
                    .1
                    .map_column(&target, raw)
            );

            controller.zoom_in();
            draw(&mut controller, &mut terminal);
            let (restored_rect, _) = cursor_rect_and_row(&controller);
            let restored = restored_rect
                .point_at_fraction(controller.view_state().cursor.fractional)
                .x
                - restored_rect.min.x;
            assert_eq!(
                restored,
                controller.zoom_levels()[FULL_ZOOM_LEVEL]
                    .1
                    .map_column(&target, raw)
            );

            controller.zoom_out();
            draw(&mut controller, &mut terminal);
            assert_eq!(controller.detail_level(), VisualDetail::Truncated);
        }

        #[test]
        fn test_zoom_steps_through_hidden_then_shown_names_in_the_truncated_view() {
            let dir = tempfile::tempdir().unwrap();
            let db_path = dir.path().join("graph.db");
            let block_group_id = chain_block_group(&db_path);
            let conn = get_connection(&db_path).unwrap();
            let workspace = Workspace::from_current_dir();
            let mut terminal =
                Terminal::new(TestBackend::new(40, 12)).expect("should create a test terminal");
            let mut controller = GenGraphController::for_block_group(
                GraphDatabase::for_connection(&conn, &workspace)
                    .expect("should open the graph database"),
                &block_group_id,
                None,
            )
            .expect("should load the block group");
            draw(&mut controller, &mut terminal);
            let step = |controller: &GenGraphController| {
                (
                    controller.detail_level(),
                    controller.node_annotations().compact_names(),
                )
            };
            let hidden = (VisualDetail::Truncated, false);
            let shown = (VisualDetail::Truncated, true);
            assert_eq!(step(&controller), hidden, "should open without names");

            controller.zoom_in();
            assert_eq!(step(&controller), shown);
            controller.zoom_in();
            assert_eq!(controller.detail_level(), VisualDetail::Full);
            controller.zoom_out();
            assert_eq!(step(&controller), shown);
            controller.zoom_out();
            assert_eq!(step(&controller), hidden);
            controller.zoom_out();
            assert_eq!(controller.detail_level(), VisualDetail::Minimal);
            controller.zoom_in();
            assert_eq!(step(&controller), hidden);
        }

        #[test]
        fn test_truncated_zoom_stays_available_without_annotations() {
            let dir = tempfile::tempdir().unwrap();
            let db_path = dir.path().join("graph.db");
            let block_group_id = chain_block_group(&db_path);
            let conn = get_connection(&db_path).unwrap();
            let workspace = Workspace::from_current_dir();
            let mut terminal =
                Terminal::new(TestBackend::new(40, 12)).expect("should create a test terminal");
            let mut controller =
                full_detail_chain(&conn, &workspace, &block_group_id, &mut terminal);
            assert!(controller.overlays_mut().is_empty());

            controller.zoom_out();
            draw(&mut controller, &mut terminal);
            assert_eq!(controller.detail_level(), VisualDetail::Truncated);
            // The truncated view's hidden-names step comes before minimal.
            controller.zoom_out();
            controller.zoom_out();
            assert_eq!(controller.detail_level(), VisualDetail::Minimal);
            controller.zoom_in();
            assert_eq!(controller.detail_level(), VisualDetail::Truncated);
        }

        #[test]
        fn test_known_path_keeps_sequence_cursor_on_top_when_zooming() {
            let directory = tempfile::tempdir().expect("should create temporary directory");
            let database_path = directory.path().join("graph.db");
            let (block_group_id, edge_ids) =
                setup_labelled_chain_block_group(&database_path, &["x", "y", "z"]);
            let connection =
                get_connection(&database_path).expect("should connect to the database");
            Path::create(&connection, "chain", &block_group_id, &edge_ids)
                .expect("should create the reference path");
            let workspace = Workspace::from_current_dir();
            let mut terminal =
                Terminal::new(TestBackend::new(60, 15)).expect("should create a terminal");
            let mut controller =
                full_detail_chain(&connection, &workspace, &block_group_id, &mut terminal);
            let chain = active_chain(&controller);
            annotate(&mut controller, &chain[..3]);
            controller.view_state_mut().go_to_node(chain[1], (0.5, 0.0));
            for compact in [false, true, false] {
                if compact {
                    controller.zoom_out();
                } else {
                    controller.zoom_in();
                }
                draw(&mut controller, &mut terminal);
                let (rect, row) = cursor_rect_and_row(&controller);
                assert!(
                    rect.height() > 0,
                    "should retain annotation rows below the sequence"
                );
                assert_eq!(row, rect.max.y, "cursor must stay on the top sequence row");
                let node = controller
                    .view_state()
                    .cursor
                    .node
                    .expect("should have cursor node");
                let renderer = &controller.zoom_levels()[controller.view_state().zoom_index].1;
                assert_eq!(renderer.vertical_anchor(&node), VerticalAnchor::Top);
                let content = renderer.content_rect(&node);
                assert_eq!(rect.min.y + content.max.y, row);
            }
        }

        #[test]
        fn test_cursor_stays_on_the_sequence_row_of_annotated_nodes() {
            let dir = tempfile::tempdir().unwrap();
            let db_path = dir.path().join("graph.db");
            let block_group_id = chain_block_group(&db_path);
            let conn = get_connection(&db_path).unwrap();
            let workspace = Workspace::from_current_dir();
            let mut terminal =
                Terminal::new(TestBackend::new(40, 12)).expect("should create a test terminal");
            let mut controller =
                full_detail_chain(&conn, &workspace, &block_group_id, &mut terminal);
            let chain = active_chain(&controller);
            annotate(&mut controller, &chain[..3]);
            draw(&mut controller, &mut terminal);

            // A go-to aimed at a flag lane, as a door entry or a caller's fraction may be.
            controller.view_state_mut().go_to_node(chain[1], (0.5, 0.0));
            draw(&mut controller, &mut terminal);
            let (rect, row) = cursor_rect_and_row(&controller);
            assert!(
                rect.height() > 0,
                "the annotated node should have flag lanes"
            );
            assert_eq!(
                row,
                sequence_row(rect),
                "a go-to should land on the sequence row"
            );
            let view_state = controller.view_state();
            let cursor_x = rect.point_at_fraction(view_state.cursor.fractional).x;
            let annotation_cell = view_state
                .screen_to_terminal(cursor_x, row - 1)
                .expect("should draw the annotation row under the cursor");
            assert_ne!(terminal.backend().buffer()[annotation_cell].symbol(), "⌃");

            // The chain has nothing above or below, so up and down leave the cursor put
            // instead of moving it onto a flag lane.
            let press = |code| KeyEvent::new(code, KeyModifiers::NONE);
            for code in [KeyCode::Up, KeyCode::Down, KeyCode::Down] {
                controller.handle_key(press(code));
                draw(&mut controller, &mut terminal);
                let (rect, row) = cursor_rect_and_row(&controller);
                assert_eq!(
                    row,
                    sequence_row(rect),
                    "{code:?} should keep the sequence row"
                );
            }
            for code in [KeyCode::Right, KeyCode::Right, KeyCode::Left] {
                controller.handle_key(press(code));
                draw(&mut controller, &mut terminal);
                let (rect, row) = cursor_rect_and_row(&controller);
                assert_eq!(
                    row,
                    sequence_row(rect),
                    "{code:?} should keep the sequence row"
                );
            }

            // A click on a flag lane selects the column above it on the sequence row.
            let (rect, _) = cursor_rect_and_row(&controller);
            let (column, flag_row) = controller
                .view_state()
                .screen_to_terminal(rect.left() + 2, rect.bottom())
                .expect("should draw the lower flag lane on screen");
            assert!(controller.view_state_mut().handle_click(column, flag_row));
            let (rect, row) = cursor_rect_and_row(&controller);
            assert_eq!(
                row,
                sequence_row(rect),
                "a click should land on the sequence row"
            );
        }

        /// Whether the cursor's node was drawn inside the viewport on the last draw.
        fn cursor_on_screen(controller: &GenGraphController) -> bool {
            let view_state = controller.view_state();
            let (rect, row) = cursor_rect_and_row(controller);
            let column = rect.point_at_fraction(view_state.cursor.fractional).x;
            view_state.screen_to_terminal(column, row).is_some()
        }

        #[test]
        fn test_opening_view_is_truncated_with_cursor_on_the_left_most_node() {
            let dir = tempfile::tempdir().unwrap();
            let db_path = dir.path().join("graph.db");
            let block_group_id = chain_block_group(&db_path);
            let conn = get_connection(&db_path).unwrap();
            let workspace = Workspace::from_current_dir();
            let mut terminal =
                Terminal::new(TestBackend::new(40, 12)).expect("should create a test terminal");
            let mut controller = GenGraphController::for_block_group(
                GraphDatabase::for_connection(&conn, &workspace)
                    .expect("should open the graph database"),
                &block_group_id,
                None,
            )
            .expect("should load the block group");
            draw(&mut controller, &mut terminal);

            assert_eq!(controller.detail_level(), VisualDetail::Truncated);
            let view_state = controller.view_state();
            let cursor_node = view_state.cursor.node.expect("should place the cursor");
            let cursor_rect = view_state
                .frame
                .rect_of(cursor_node)
                .expect("should place the cursor node");
            assert!(
                view_state
                    .frame
                    .visible_ids()
                    .filter_map(|node| view_state.frame.rect_of(node))
                    .all(|rect| rect.min.x >= cursor_rect.min.x),
                "should start the cursor on the left-most node"
            );
        }

        #[test]
        fn test_annotation_jumps_reach_starts_off_screen() {
            let dir = tempfile::tempdir().unwrap();
            let db_path = dir.path().join("graph.db");
            let block_group_id = chain_block_group(&db_path);
            let conn = get_connection(&db_path).unwrap();
            let workspace = Workspace::from_current_dir();
            let press = |code| KeyEvent::new(code, KeyModifiers::NONE);
            let mut terminal =
                Terminal::new(TestBackend::new(20, 12)).expect("should create a test terminal");
            let mut controller =
                full_detail_chain(&conn, &workspace, &block_group_id, &mut terminal);
            let chain = active_chain(&controller);
            let (near, far) = (chain[1], chain[chain.len() - 3]);
            annotate(&mut controller, &[near, far]);
            draw(&mut controller, &mut terminal);
            assert!(
                !controller
                    .view_state()
                    .frame
                    .visible_ids()
                    .any(|node| node == far),
                "the far annotation should start off screen"
            );

            let mut stops = Vec::new();
            while controller.handle_key(press(KeyCode::Char('w'))) == GraphKeyOutcome::Redraw {
                draw(&mut controller, &mut terminal);
                assert!(
                    cursor_on_screen(&controller),
                    "the camera should follow the cursor"
                );
                stops.push(controller.view_state().cursor.node);
            }
            assert_eq!(stops, vec![Some(near), Some(far)], "w");

            stops.clear();
            while controller.handle_key(press(KeyCode::Char('b'))) == GraphKeyOutcome::Redraw {
                draw(&mut controller, &mut terminal);
                assert!(
                    cursor_on_screen(&controller),
                    "the camera should follow the cursor"
                );
                stops.push(controller.view_state().cursor.node);
            }
            assert_eq!(stops, vec![Some(near)], "b");
        }

        /// The cursor's screen column and row.
        fn cursor_cell(controller: &GenGraphController) -> (i64, i64) {
            let (rect, row) = cursor_rect_and_row(controller);
            let column = rect
                .point_at_fraction(controller.view_state().cursor.fractional)
                .x;
            (column, row)
        }

        #[test]
        fn test_annotation_jump_centers_the_cursor() {
            let dir = tempfile::tempdir().unwrap();
            let db_path = dir.path().join("graph.db");
            let block_group_id = chain_block_group(&db_path);
            let conn = get_connection(&db_path).unwrap();
            let workspace = Workspace::from_current_dir();
            let press = |code| KeyEvent::new(code, KeyModifiers::NONE);
            let mut terminal =
                Terminal::new(TestBackend::new(20, 12)).expect("should create a test terminal");
            let mut controller =
                full_detail_chain(&conn, &workspace, &block_group_id, &mut terminal);
            let chain = active_chain(&controller);
            let (near, far) = (chain[1], chain[chain.len() - 3]);
            annotate(&mut controller, &[near, far]);
            draw(&mut controller, &mut terminal);

            assert_eq!(
                controller.handle_key(press(KeyCode::Char('w'))),
                GraphKeyOutcome::Redraw
            );
            draw(&mut controller, &mut terminal);
            assert_eq!(controller.view_state().cursor.node, Some(near));
            let before = cursor_cell(&controller);

            assert_eq!(
                controller.handle_key(press(KeyCode::Char('w'))),
                GraphKeyOutcome::Redraw
            );
            draw(&mut controller, &mut terminal);
            assert_eq!(controller.view_state().cursor.node, Some(far));
            assert_eq!(
                cursor_cell(&controller),
                before,
                "every jump should center the cursor"
            );
        }

        #[test]
        fn test_compact_annotation_jumps_and_end_key() {
            let dir = tempfile::tempdir().unwrap();
            let db_path = dir.path().join("graph.db");
            let block_group_id = chain_block_group(&db_path);
            let conn = get_connection(&db_path).unwrap();
            let workspace = Workspace::from_current_dir();
            let mut terminal =
                Terminal::new(TestBackend::new(20, 12)).expect("should create a test terminal");
            let mut controller =
                full_detail_chain(&conn, &workspace, &block_group_id, &mut terminal);
            let chain = active_chain(&controller);
            let (near, far) = (chain[1], chain[chain.len() - 3]);
            annotate(&mut controller, &[near, far]);
            draw(&mut controller, &mut terminal);
            controller.zoom_out();
            draw(&mut controller, &mut terminal);
            assert_eq!(controller.detail_level(), VisualDetail::Truncated);
            let press = |key| KeyEvent::new(KeyCode::Char(key), KeyModifiers::NONE);

            for (key, target) in [
                ('w', near),
                ('w', far),
                ('b', near),
                ('e', near),
                ('e', far),
            ] {
                assert_eq!(controller.handle_key(press(key)), GraphKeyOutcome::Redraw);
                draw(&mut controller, &mut terminal);
                assert_eq!(
                    controller.view_state().cursor.node,
                    Some(target),
                    "jump {key}"
                );
                assert_eq!(cursor_cell(&controller), (10, 6));
            }
        }

        /// Draw the graph into a buffer the way the notebook and R widgets do.
        fn render_text(controller: &mut GenGraphController, width: u16, height: u16) -> String {
            let area = Rect::new(0, 0, width, height);
            let mut buffer = Buffer::empty(area);
            controller.render_settled(&mut buffer, area, Style::default());
            buffer.content().iter().map(|cell| cell.symbol()).collect()
        }

        #[test]
        fn test_clone_draws_its_own_annotation_flags() {
            let dir = tempfile::tempdir().unwrap();
            let db_path = dir.path().join("graph.db");
            let block_group_id = chain_block_group(&db_path);
            let conn = get_connection(&db_path).unwrap();
            let workspace = Workspace::from_current_dir();
            let mut terminal =
                Terminal::new(TestBackend::new(60, 16)).expect("should create a test terminal");
            let mut original = full_detail_chain(&conn, &workspace, &block_group_id, &mut terminal);
            assert!(!render_text(&mut original, 60, 16).contains('═'));

            let mut clone = original.clone();
            let nodes = active_chain(&clone);
            annotate(&mut clone, &nodes);
            assert!(render_text(&mut clone, 60, 16).contains('═'));

            assert!(
                !render_text(&mut original, 60, 16).contains('═'),
                "flags packed for the clone should not show up in the original"
            );
        }

        #[test]
        fn test_disabled_annotation_group_loses_its_overlays() {
            let dir = tempfile::tempdir().unwrap();
            let db_path = dir.path().join("graph.db");
            let block_group_id = chain_block_group(&db_path);
            let conn = get_connection(&db_path).unwrap();
            let workspace = Workspace::from_current_dir();
            let mut terminal =
                Terminal::new(TestBackend::new(60, 16)).expect("should create a test terminal");
            let mut controller =
                full_detail_chain(&conn, &workspace, &block_group_id, &mut terminal);
            let nodes = active_chain(&controller);
            annotate(&mut controller, &nodes);
            for overlay in controller.overlays_mut() {
                overlay.source = OverlaySource::Track(group_track_key("genes"));
            }

            controller.set_annotation_group_enabled("genes", false);

            assert!(controller.overlays().is_empty());
            assert!(!render_text(&mut controller, 60, 16).contains('═'));
        }

        #[test]
        fn test_click_on_a_door_opens_the_batch_behind_it() {
            let dir = tempfile::tempdir().unwrap();
            let db_path = dir.path().join("graph.db");
            let block_group_id = chain_block_group(&db_path);
            let conn = get_connection(&db_path).unwrap();
            let workspace = Workspace::from_current_dir();
            let mut terminal =
                Terminal::new(TestBackend::new(40, 12)).expect("should create a test terminal");
            let mut controller =
                full_detail_chain(&conn, &workspace, &block_group_id, &mut terminal);
            let first_batch = controller.engine().active_batch();

            // Pan right until a door onto the next batch is on screen.
            let mut door_cell = None;
            for _ in 0..400 {
                render_text(&mut controller, 40, 12);
                let view_state = controller.view_state();
                door_cell = view_state.wormhole.iter().find_map(|(rect, _, _)| {
                    let center = rect.center();
                    view_state.screen_to_terminal(center.x, center.y)
                });
                if door_cell.is_some() {
                    break;
                }
                controller.pan(-4, 0);
            }
            let (column, row) = door_cell.expect("should reach a door by panning right");

            assert_eq!(controller.click(column, row), ClickOutcome::EnteredDoor);
            render_text(&mut controller, 40, 12);
            assert_ne!(controller.engine().active_batch(), first_batch);
        }

        #[test]
        fn test_go_to_reaches_a_node_the_crawl_has_not() {
            let dir = tempfile::tempdir().unwrap();
            let db_path = dir.path().join("graph.db");
            let block_group_id = chain_block_group(&db_path);
            let conn = get_connection(&db_path).unwrap();
            let workspace = Workspace::from_current_dir();
            let mut terminal =
                Terminal::new(TestBackend::new(40, 12)).expect("should create a test terminal");
            let mut controller =
                full_detail_chain(&conn, &workspace, &block_group_id, &mut terminal);
            let far_node_id = HashId::convert_str("n75");
            assert!(
                !controller
                    .engine()
                    .graph()
                    .nodes()
                    .any(|node| node.node_id == far_node_id),
                "the first batch should not reach the end of the chain"
            );

            controller
                .go_to_coordinate(far_node_id, 2, true)
                .expect("should go to the far node");
            render_text(&mut controller, 40, 12);

            let active: Vec<GraphNode> = controller
                .engine()
                .active_world()
                .expect("should have an active batch")
                .members()
                .collect();
            assert!(active.iter().any(|node| node.node_id == far_node_id));
            assert!(matches!(
                controller.go_to_coordinate(HashId::convert_str("elsewhere"), 2, true),
                Err(LocateError::NotFound)
            ));
        }
    }
}
