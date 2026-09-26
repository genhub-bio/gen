//! The graph state and behaviour shared by the inline widget and the full-screen viewer.
//!
//! Both viewers draw the same graph widget from one [`GenGraphController`]; they differ only in
//! what surrounds it (a border and a help line inline, the collection explorer, search bar,
//! panels and status bars full-screen) and in how annotations are drawn beside it. Because the
//! controller owns everything keyed to the loaded graph (the lazily crawled engine, its view
//! state, dimming, overlays, and which batch the annotation groups were loaded for), switching
//! from the inline widget to the full-screen viewer moves the controller over and reloads
//! nothing.

use std::{collections::HashSet, error::Error};

use crossterm::event::{KeyCode, KeyEvent};
use gen_core::{HashId, PATH_START_NODE_ID, Workspace};
use gen_graph::{GenGraph, GraphNode};
use gen_models::{block_group::BlockGroup, db::GraphConnection, path::Path};
use gen_tui::{
    crawl::EagerSource,
    graph_view::{GraphView, GraphViewState},
    layout::VisualDetail,
    layout_engine::{BatchId, LayoutEngine},
    plotter::{LineStyle, PathStyle},
    theme::current_theme,
};
use log::warn;
use ratatui::{Frame, layout::Rect, style::Style};

use crate::views::{
    annotation_groups::{AnnotationGroupEntry, load_annotation_group_entries},
    annotations::{AnnotationGroupTrackRequest, load_annotations_for_group},
    block_group::{
        active_neighborhood_node_ids, load_block_group_graph, teleport_through_wormhole,
    },
    gen_graph_widget::{
        self, AnnotationLabels, AnnotationStarts, NodeAnnotationLayer, OverlayInputs, ZoomLevels,
        create_annotated_gen_graph_engine_lazy, draw_annotation_connectors, draw_annotation_labels,
        reapply_overlays, update_node_annotations,
    },
    graph_dimming::GraphDimming,
    graph_overlay::{
        AnnotationColorCache, GraphOverlay, OverlaySource, PathMembership, group_track_key,
        has_path_overlay, remove_path_overlay, replace_track_overlays, set_path_overlay,
    },
    lazy_graph_source::EagerOrSqlSource,
};

/// How annotation names are drawn around the graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnnotationDisplay {
    /// Every annotation gets a floating label near its nodes.
    FloatingLabels,
    /// At full detail annotations are drawn as flags under their nodes, and only the names
    /// with no room there float. Other detail levels float every label.
    FlagsUnderNodes,
}

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

/// What [`GenGraphController::sync_active_world`] brought up to date.
#[derive(Debug, Default)]
pub struct WorldSync {
    /// Whether anything drawn changed, so the frame just drawn is stale.
    pub changed: bool,
    /// Set when the active batch changed and its annotation groups were reloaded.
    pub group_reload: Option<GroupReload>,
}

/// One block group's lazily loaded graph together with the view of it.
pub struct GenGraphController<'a> {
    conn: &'a GraphConnection,
    workspace: &'a Workspace,
    history_ref: Option<&'a str>,
    /// Seeded with a block group's start and grown batch by batch from SQLite, so opening a
    /// large block group never materializes the whole graph.
    engine: LayoutEngine<GenGraph, EagerOrSqlSource>,
    zoom_levels: ZoomLevels<'a>,
    view_state: GraphViewState<GraphNode>,
    /// Pruned edges and the nodes only they lead into, synced whenever a draw or a door may
    /// have grown the graph.
    dimming: GraphDimming,
    /// Annotation flags drawn under nodes at full detail. Only filled when a viewer draws with
    /// [`AnnotationDisplay::FlagsUnderNodes`].
    node_annotations: NodeAnnotationLayer,
    /// Where annotations start on each loaded node, for the `w`/`b` keys. Rebuilt with the
    /// highlights whatever the annotation display, so the stops don't depend on flags being
    /// drawn.
    annotation_starts: AnnotationStarts,
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
    applied_overlay_inputs: Option<(OverlayInputs, AnnotationDisplay)>,
    /// The overlays whose names found no room under their node at the last flag refill;
    /// `None` when drawing with [`AnnotationDisplay::FloatingLabels`].
    floating_overlays: Option<Vec<GraphOverlay>>,
    /// Floating labels resolved against the graph as last drawn, and what they were resolved
    /// against (`None` once the overlays are rebuilt).
    annotation_labels: AnnotationLabels,
    labelled_overlay_inputs: Option<(OverlayInputs, AnnotationDisplay)>,
}

impl<'a> GenGraphController<'a> {
    /// A controller with no block group open, showing just a `PATH_START` placeholder.
    pub fn new(
        conn: &'a GraphConnection,
        workspace: &'a Workspace,
        history_ref: Option<&'a str>,
    ) -> Self {
        let mut graph = GenGraph::new();
        graph.add_node(GraphNode {
            node_id: PATH_START_NODE_ID,
            sequence_start: 0,
            sequence_end: 0,
        });
        let node_annotations = NodeAnnotationLayer::new();
        let (engine, zoom_levels, view_state) = create_annotated_gen_graph_engine_lazy(
            graph,
            EagerOrSqlSource::Eager(EagerSource),
            (conn, workspace),
            node_annotations.clone(),
        );
        Self {
            conn,
            workspace,
            history_ref,
            engine,
            zoom_levels,
            view_state,
            dimming: GraphDimming::default(),
            node_annotations,
            annotation_starts: AnnotationStarts::default(),
            block_group: None,
            annotation_group_entries: Vec::new(),
            annotation_groups_world: None,
            paths: Vec::new(),
            overlays: Vec::new(),
            annotation_colors: AnnotationColorCache::new(),
            overlays_dirty: true,
            applied_overlay_inputs: None,
            floating_overlays: None,
            annotation_labels: AnnotationLabels::default(),
            labelled_overlay_inputs: None,
        }
    }

    /// A controller with `block_group_id` open.
    pub fn for_block_group(
        conn: &'a GraphConnection,
        workspace: &'a Workspace,
        block_group_id: &HashId,
        history_ref: Option<&'a str>,
    ) -> Result<Self, Box<dyn Error>> {
        let mut controller = Self::new(conn, workspace, history_ref);
        controller.open_block_group(block_group_id)?;
        Ok(controller)
    }

    /// Replace the graph with `block_group_id`'s, starting over from its seed at the default
    /// zoom level with no overlays, paths or annotation groups loaded.
    pub fn open_block_group(&mut self, block_group_id: &HashId) -> Result<(), Box<dyn Error>> {
        let block_group = BlockGroup::get_by_id(self.conn, block_group_id, self.history_ref)?;
        let (graph, source) =
            load_block_group_graph(self.conn, self.workspace, block_group_id, self.history_ref)?;
        (self.engine, self.zoom_levels, self.view_state) = create_annotated_gen_graph_engine_lazy(
            graph,
            source,
            (self.conn, self.workspace),
            self.node_annotations.clone(),
        );
        self.dimming = GraphDimming::default();
        self.annotation_group_entries =
            load_annotation_group_entries(self.conn, &block_group, self.history_ref);
        self.block_group = Some(block_group);
        self.annotation_groups_world = None;
        self.paths.clear();
        self.overlays.clear();
        self.overlays_dirty = true;
        Ok(())
    }

    /// The open block group, if any.
    pub fn block_group(&self) -> Option<&BlockGroup> {
        self.block_group.as_ref()
    }

    pub fn engine(&self) -> &LayoutEngine<GenGraph, EagerOrSqlSource> {
        &self.engine
    }

    pub fn view_state(&self) -> &GraphViewState<GraphNode> {
        &self.view_state
    }

    pub fn view_state_mut(&mut self) -> &mut GraphViewState<GraphNode> {
        &mut self.view_state
    }

    pub fn zoom_levels(&self) -> &ZoomLevels<'a> {
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
        &LayoutEngine<GenGraph, EagerOrSqlSource>,
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
        self.paths
            .push(PathMembership::load(self.conn, &path.id, self.history_ref));
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
            match BlockGroup::get_current_path(self.conn, &block_group.id, self.history_ref) {
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

    /// Apply a graph key: zoom, the path toggle, annotation stops at full detail, and cursor
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
            KeyCode::Char('+') | KeyCode::Char('=') => {
                gen_graph_widget::zoom_in(&mut self.view_state, &self.zoom_levels);
                self.overlays_dirty = true;
                GraphKeyOutcome::Redraw
            }
            KeyCode::Char('-') => {
                gen_graph_widget::zoom_out(&mut self.view_state, &self.zoom_levels);
                self.overlays_dirty = true;
                GraphKeyOutcome::Redraw
            }
            // Annotation starts are only known in screen columns where the annotations are
            // drawn under their nodes, at full detail.
            KeyCode::Char(key_char @ ('w' | 'b')) if self.detail_level() == VisualDetail::Full => {
                // Reaching the edge of the loaded batch leaves the cursor where it is, like an
                // arrow key with nothing beyond it.
                let annotation_starts = &self.annotation_starts;
                match self
                    .view_state
                    .move_cursor_to_stop(key_char == 'w', |node| annotation_starts.on(&node))
                {
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
        let overlays_stale = self.applied_overlay_inputs.is_some_and(|(inputs, _)| {
            inputs != OverlayInputs::current(&self.engine, &self.view_state)
        });
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
        self.overlays.retain(
            |overlay| !matches!(&overlay.source, OverlaySource::Track(key) if key.starts_with("group:")),
        );
        for entry in &self.annotation_group_entries {
            let spans = match load_annotations_for_group(&AnnotationGroupTrackRequest {
                conn: self.conn,
                history_ref: self.history_ref,
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
            if spans.is_empty() {
                continue;
            }
            reload.loaded.push(entry.id.clone());
            replace_track_overlays(&mut self.overlays, &group_track_key(&entry.id), spans);
        }
        reload
    }

    /// Draw the graph and its annotation names into `area`, returning whether any name was
    /// left out for lack of room.
    pub fn render(
        &mut self,
        frame: &mut Frame,
        area: Rect,
        annotation_display: AnnotationDisplay,
        style: Style,
    ) -> bool {
        // Re-register overlay highlights and refill the node annotation flags only when the
        // overlay set, the zoom level, the loaded graph, or how annotations are drawn changed
        // since they were last registered.
        let overlay_inputs = (
            OverlayInputs::current(&self.engine, &self.view_state),
            annotation_display,
        );
        if self.overlays_dirty || self.applied_overlay_inputs != Some(overlay_inputs) {
            reapply_overlays(
                &self.engine,
                &mut self.view_state,
                &self.zoom_levels,
                &mut self.overlays,
                &mut self.annotation_colors,
            );
            self.annotation_starts = AnnotationStarts::new(&self.engine, &self.overlays);
            // Names with no room under their node fall back to floating labels.
            self.floating_overlays = match annotation_display {
                AnnotationDisplay::FlagsUnderNodes => Some(update_node_annotations(
                    &self.node_annotations,
                    &self.engine,
                    &self.overlays,
                )),
                AnnotationDisplay::FloatingLabels => None,
            };
            self.overlays_dirty = false;
            self.applied_overlay_inputs = Some(overlay_inputs);
            self.labelled_overlay_inputs = None;
        }

        let active_renderer = &self.zoom_levels[self.view_state.zoom_index].1;
        let view = GraphView::new(&mut self.engine, active_renderer).style(style);
        frame.render_stateful_widget(view, area, &mut self.view_state);

        // Resolve floating labels against the graph as drawn, which may have just grown. At
        // full detail with flags under nodes, only the names that found no room there float.
        let detail_level = self.detail_level();
        let flags_drawn = self.floating_overlays.is_some() && detail_level == VisualDetail::Full;
        let label_inputs = (
            OverlayInputs::current(&self.engine, &self.view_state),
            annotation_display,
        );
        if self.labelled_overlay_inputs != Some(label_inputs) {
            let labelled_overlays = match &self.floating_overlays {
                Some(floating_overlays) if flags_drawn => floating_overlays,
                _ => &self.overlays,
            };
            self.annotation_labels =
                AnnotationLabels::new(self.engine.graph(), detail_level, labelled_overlays);
            self.labelled_overlay_inputs = Some(label_inputs);
        }
        if flags_drawn {
            draw_annotation_connectors(
                frame.buffer_mut(),
                area,
                &self.view_state.frame,
                &self.node_annotations,
                None,
            );
        }
        draw_annotation_labels(
            frame.buffer_mut(),
            area,
            &self.view_state,
            &self.annotation_labels,
        )
    }

    /// Draw the graph alone into `area`, without the cursor or annotation names.
    pub fn render_plain(&mut self, frame: &mut Frame, area: Rect) {
        self.view_state.hide_cursor();
        let active_renderer = &self.zoom_levels[self.view_state.zoom_index].1;
        let view = GraphView::new(&mut self.engine, active_renderer);
        frame.render_stateful_widget(view, area, &mut self.view_state);
    }
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
    use ratatui::{Terminal, backend::TestBackend, style::Style};

    use super::{AnnotationDisplay, GenGraphController, GraphKeyOutcome};
    use crate::views::{
        gen_graph_widget::DEFAULT_ZOOM_LEVEL, graph_overlay::has_path_overlay,
        lazy_graph_source::tests::setup_labelled_chain_block_group,
    };

    fn draw(
        controller: &mut GenGraphController,
        terminal: &mut Terminal<TestBackend>,
        display: AnnotationDisplay,
    ) {
        controller.sync_active_world();
        terminal
            .draw(|frame| {
                controller.render(frame, frame.area(), display, Style::default());
            })
            .expect("should draw the graph");
        controller.sync_active_world();
    }

    #[test]
    fn test_navigation_keys_do_not_reapply_overlays() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("graph.db");
        let (block_group_id, _) = setup_labelled_chain_block_group(&db_path, &["x", "y", "z"]);
        let conn = get_connection(&db_path).unwrap();
        let workspace = Workspace::from_current_dir();
        let mut controller =
            GenGraphController::for_block_group(&conn, &workspace, &block_group_id, None)
                .expect("should load the block group");
        let mut terminal =
            Terminal::new(TestBackend::new(80, 12)).expect("should create a test terminal");
        draw(
            &mut controller,
            &mut terminal,
            AnnotationDisplay::FloatingLabels,
        );
        if controller.overlays_dirty {
            draw(
                &mut controller,
                &mut terminal,
                AnnotationDisplay::FloatingLabels,
            );
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
        let mut controller =
            GenGraphController::for_block_group(&conn, &workspace, &block_group_id, None)
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
        let mut controller = GenGraphController::new(&conn, &workspace, None);
        assert!(controller.block_group().is_none());
        controller
            .open_block_group(&block_group_id)
            .expect("should open the block group");
        controller.add_path(&path);
        controller.toggle_path();
        controller.handle_key(KeyEvent::new(KeyCode::Char('+'), KeyModifiers::NONE));
        let mut terminal =
            Terminal::new(TestBackend::new(12, 12)).expect("should create a test terminal");
        draw(
            &mut controller,
            &mut terminal,
            AnnotationDisplay::FloatingLabels,
        );
        assert!(controller.engine().graph().node_count() > 2);

        controller
            .open_block_group(&block_group_id)
            .expect("should open the block group again");

        assert!(controller.engine().graph().node_count() <= 2);
        assert_eq!(controller.engine().active_batch(), None);
        assert_eq!(controller.view_state().zoom_index, DEFAULT_ZOOM_LEVEL);
        assert!(controller.overlays().is_empty());
    }

    /// A chain of five-base nodes at full detail with annotations on some of them, the way
    /// either viewer shows a block group's annotation groups.
    mod annotated_chain {
        use gen_core::{HashId, PATH_START_NODE_ID, Strand};
        use gen_graph::GraphNode;
        use gen_models::db::GraphConnection;
        use gen_tui::{
            geometry::WorldRect,
            plotter::{LineStyle, PathStyle},
        };
        use petgraph::Direction;
        use ratatui::style::Color;

        use super::*;
        use crate::views::{
            annotation_track::{AnnotationSegment, AnnotationSpan},
            gen_graph_widget::{FULL_ZOOM_LEVEL, apply_zoom_level},
            graph_overlay::{GraphOverlay, OverlayContent, OverlaySource},
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
                            start: node.sequence_start + 1,
                            end: node.sequence_start + 4,
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
        fn full_detail_chain<'a>(
            conn: &'a GraphConnection,
            workspace: &'a Workspace,
            block_group_id: &HashId,
            terminal: &mut Terminal<TestBackend>,
            display: AnnotationDisplay,
        ) -> GenGraphController<'a> {
            let mut controller =
                GenGraphController::for_block_group(conn, workspace, block_group_id, None)
                    .expect("should load the block group");
            let zoom_levels = controller.zoom_levels().clone();
            apply_zoom_level(controller.view_state_mut(), FULL_ZOOM_LEVEL, &zoom_levels);
            draw(&mut controller, terminal, display);
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

        /// The sequence row of an annotated node: its middle row, with a flag lane on either
        /// side.
        fn sequence_row(rect: WorldRect) -> i64 {
            rect.center().y
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
            let display = AnnotationDisplay::FlagsUnderNodes;
            let mut controller =
                full_detail_chain(&conn, &workspace, &block_group_id, &mut terminal, display);
            let chain = active_chain(&controller);
            annotate(&mut controller, &chain[..3]);
            draw(&mut controller, &mut terminal, display);

            // A go-to aimed at a flag lane, as a door entry or a caller's fraction may be.
            controller.view_state_mut().go_to_node(chain[1], (0.5, 0.0));
            draw(&mut controller, &mut terminal, display);
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
            let caret = view_state
                .screen_to_terminal(cursor_x, row - 1)
                .expect("should draw the row under the cursor on screen");
            assert_eq!(
                terminal.backend().buffer()[caret].symbol(),
                "⌃",
                "the caret should sit just under the sequence row"
            );

            // The chain has nothing above or below, so up and down leave the cursor put
            // instead of moving it onto a flag lane.
            let press = |code| KeyEvent::new(code, KeyModifiers::NONE);
            for code in [KeyCode::Up, KeyCode::Down, KeyCode::Down] {
                controller.handle_key(press(code));
                draw(&mut controller, &mut terminal, display);
                let (rect, row) = cursor_rect_and_row(&controller);
                assert_eq!(
                    row,
                    sequence_row(rect),
                    "{code:?} should keep the sequence row"
                );
            }
            for code in [KeyCode::Right, KeyCode::Right, KeyCode::Left] {
                controller.handle_key(press(code));
                draw(&mut controller, &mut terminal, display);
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
        fn test_annotation_jumps_reach_starts_off_screen() {
            let dir = tempfile::tempdir().unwrap();
            let db_path = dir.path().join("graph.db");
            let block_group_id = chain_block_group(&db_path);
            let conn = get_connection(&db_path).unwrap();
            let workspace = Workspace::from_current_dir();
            let press = |code| KeyEvent::new(code, KeyModifiers::NONE);
            // The inline widget floats every name; the full-screen viewer draws flags.
            for display in [
                AnnotationDisplay::FloatingLabels,
                AnnotationDisplay::FlagsUnderNodes,
            ] {
                let mut terminal =
                    Terminal::new(TestBackend::new(20, 12)).expect("should create a test terminal");
                let mut controller =
                    full_detail_chain(&conn, &workspace, &block_group_id, &mut terminal, display);
                let chain = active_chain(&controller);
                let (near, far) = (chain[1], chain[chain.len() - 3]);
                annotate(&mut controller, &[near, far]);
                draw(&mut controller, &mut terminal, display);
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
                    draw(&mut controller, &mut terminal, display);
                    assert!(
                        cursor_on_screen(&controller),
                        "{display:?}: the camera should follow the cursor"
                    );
                    stops.push(controller.view_state().cursor.node);
                }
                assert_eq!(stops, vec![Some(near), Some(far)], "{display:?}: w");

                stops.clear();
                while controller.handle_key(press(KeyCode::Char('b'))) == GraphKeyOutcome::Redraw {
                    draw(&mut controller, &mut terminal, display);
                    assert!(
                        cursor_on_screen(&controller),
                        "{display:?}: the camera should follow the cursor"
                    );
                    stops.push(controller.view_state().cursor.node);
                }
                assert_eq!(stops, vec![Some(near)], "{display:?}: b");
            }
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
        fn test_annotation_jump_off_screen_keeps_the_cursor_cell() {
            let dir = tempfile::tempdir().unwrap();
            let db_path = dir.path().join("graph.db");
            let block_group_id = chain_block_group(&db_path);
            let conn = get_connection(&db_path).unwrap();
            let workspace = Workspace::from_current_dir();
            let press = |code| KeyEvent::new(code, KeyModifiers::NONE);
            let display = AnnotationDisplay::FloatingLabels;
            let mut terminal =
                Terminal::new(TestBackend::new(20, 12)).expect("should create a test terminal");
            let mut controller =
                full_detail_chain(&conn, &workspace, &block_group_id, &mut terminal, display);
            let chain = active_chain(&controller);
            let (near, far) = (chain[1], chain[chain.len() - 3]);
            annotate(&mut controller, &[near, far]);
            draw(&mut controller, &mut terminal, display);

            assert_eq!(
                controller.handle_key(press(KeyCode::Char('w'))),
                GraphKeyOutcome::Redraw
            );
            draw(&mut controller, &mut terminal, display);
            assert_eq!(controller.view_state().cursor.node, Some(near));
            let before = cursor_cell(&controller);

            assert_eq!(
                controller.handle_key(press(KeyCode::Char('w'))),
                GraphKeyOutcome::Redraw
            );
            draw(&mut controller, &mut terminal, display);
            assert_eq!(controller.view_state().cursor.node, Some(far));
            assert_eq!(
                cursor_cell(&controller),
                before,
                "the world should move under a cursor jumping off screen"
            );
        }
    }
}
