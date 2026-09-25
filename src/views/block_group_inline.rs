use std::{collections::HashSet, error::Error, io, panic, time::Duration};

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind};
use gen_core::{HashId, Workspace};
use gen_graph::{GenGraph, GraphNode};
use gen_models::{block_group::BlockGroup, db::GraphConnection, path::Path};
use gen_tui::{
    graph_view::{GraphView, GraphViewState},
    layout::VisualDetail,
    layout_engine::{BatchId, LayoutEngine},
    plotter::{LineStyle, PathStyle},
    theme::current_theme,
};
use ratatui::{
    Terminal, TerminalOptions, Viewport,
    prelude::*,
    widgets::{Block, Borders},
};

use crate::views::{
    annotation_groups::{AnnotationGroupEntry, load_annotation_group_entries},
    annotations::{AnnotationGroupTrackRequest, load_annotations_for_group},
    block_group::{
        active_neighborhood_node_ids, load_block_group_graph, teleport_through_wormhole,
    },
    gen_graph_widget::{
        self, NodeAnnotationLayer, ZoomLevels, create_annotated_gen_graph_engine_lazy,
        draw_annotation_labels, reapply_overlays,
    },
    graph_dimming::GraphDimming,
    graph_overlay::{
        AnnotationColorCache, GraphOverlay, PathMembership, group_track_key, has_path_overlay,
        remove_path_overlay, replace_track_overlays, set_path_overlay,
    },
    lazy_graph_source::{BlockGroupBounds, EagerOrSqlSource},
};

#[derive(Debug)]
pub enum AppEvent {
    KeyPress(KeyEvent),
    Resize(u16, u16),
}

pub trait EventSource {
    fn poll_next(&mut self, timeout: Duration) -> Option<AppEvent>;
}

/// Reads key/resize events straight from the terminal. Rendering is event-driven (no
/// animation to advance), so this blocks on `poll` rather than waking at a fixed rate.
pub struct CrosstermEventSource;

impl EventSource for CrosstermEventSource {
    fn poll_next(&mut self, timeout: Duration) -> Option<AppEvent> {
        if event::poll(timeout).unwrap_or(false) {
            match event::read().unwrap() {
                Event::Key(k) if k.kind == KeyEventKind::Press => {
                    return Some(AppEvent::KeyPress(k));
                }
                Event::Resize(w, h) => {
                    return Some(AppEvent::Resize(w, h));
                }
                _ => {}
            }
        }
        None
    }
}

pub struct InlineGenGraphState<'a> {
    /// Seeded like the full viewer's engine and grown batch by batch from SQLite, so opening a
    /// large block group inline never materializes the whole graph.
    engine: LayoutEngine<GenGraph, EagerOrSqlSource>,
    zoom_levels: ZoomLevels<'a>,
    view_state: GraphViewState<GraphNode>,
    /// Pruned edges and the nodes only they lead into, synced whenever a draw or a door may
    /// have grown the graph.
    dimming: GraphDimming,
    paths: Vec<PathMembership>,
    conn: &'a GraphConnection,
    history_ref: Option<&'a str>,
    /// Where the block group starts and ends, resolved once so annotation loading never needs
    /// the whole graph to tell.
    block_group_bounds: BlockGroupBounds,
    /// Fetched once; a batch change only reloads their annotations for the new batch's nodes.
    annotation_group_entries: Vec<AnnotationGroupEntry>,
    /// Annotation and path overlays currently loaded, ready for highlight + label rendering.
    overlays: Vec<GraphOverlay>,
    annotation_colors: AnnotationColorCache,
    /// The batch the annotation groups were last loaded for. A batch is already the
    /// deliberately-constrained local window, so a reload is only needed when it changes (not
    /// on every pan/zoom within the same batch).
    annotation_groups_world: Option<BatchId>,
    /// Whether the highlights registered in `view_state` are stale: the overlays, the zoom
    /// level, or the loaded batch changed since `reapply_overlays` last ran. Highlights persist
    /// between frames, so plain panning and cursor moves skip that work.
    overlays_dirty: bool,
}

impl<'a> InlineGenGraphState<'a> {
    pub fn new(
        conn: &'a GraphConnection,
        workspace: &'a Workspace,
        block_group_id: &HashId,
        history_ref: Option<&'a str>,
    ) -> Result<Self, Box<dyn Error>> {
        let block_group = BlockGroup::get_by_id(conn, block_group_id, history_ref)?;
        let (graph, source) = load_block_group_graph(conn, workspace, block_group_id, history_ref)?;
        // Annotations are drawn as floating labels here, so the flag layer stays empty.
        let (engine, zoom_levels, mut view_state) = create_annotated_gen_graph_engine_lazy(
            graph,
            source,
            (conn, workspace),
            NodeAnnotationLayer::new(),
        );
        view_state.show_cursor();
        let annotation_group_entries =
            load_annotation_group_entries(conn, &block_group, history_ref);
        let block_group_bounds =
            BlockGroupBounds::for_view(conn, engine.graph(), block_group_id, history_ref);
        Ok(Self {
            engine,
            zoom_levels,
            view_state,
            dimming: GraphDimming::default(),
            paths: Vec::new(),
            conn,
            history_ref,
            block_group_bounds,
            annotation_group_entries,
            overlays: Vec::new(),
            annotation_colors: AnnotationColorCache::new(),
            annotation_groups_world: None,
            overlays_dirty: true,
        })
    }

    /// Add a path the `p` key can highlight. Only its edge membership is fetched; the
    /// highlight itself is resolved against the loaded graph whenever the overlays reapply.
    pub fn add_path(&mut self, path: &Path) {
        self.paths
            .push(PathMembership::load(self.conn, &path.id, self.history_ref));
    }

    fn load_annotation_groups(&mut self, node_ids: &HashSet<HashId>) {
        // Drop the annotation overlays but keep the path overlay across batch changes.
        self.overlays.retain(|overlay| overlay.path().is_some());
        for entry in &self.annotation_group_entries {
            let Ok(entry_spans) = load_annotations_for_group(&AnnotationGroupTrackRequest {
                conn: self.conn,
                history_ref: self.history_ref,
                entry,
                projection_graph: self.engine.graph(),
                bounds: &self.block_group_bounds,
                node_ids,
            }) else {
                continue;
            };
            replace_track_overlays(&mut self.overlays, &group_track_key(&entry.id), entry_spans);
        }
    }

    /// Bring everything keyed to the loaded graph up to date: dimming for whatever the crawl
    /// added, and on a batch change the annotation groups and path highlight for the new
    /// batch. Returns whether anything drawn changed.
    fn sync_active_world(&mut self) -> bool {
        let dimming_changed = self.dimming.sync(
            self.engine.graph(),
            self.engine.source(),
            &mut self.view_state,
        );
        let current_world = self.engine.active_batch();
        if current_world == self.annotation_groups_world {
            return dimming_changed;
        }
        let node_ids = active_neighborhood_node_ids(&self.engine);
        if node_ids.is_empty() {
            return dimming_changed;
        }
        self.load_annotation_groups(&node_ids);
        self.annotation_groups_world = current_world;
        self.overlays_dirty = true;
        true
    }
}

/// What a key press asks of the event loop.
#[derive(Debug, PartialEq)]
enum KeyOutcome {
    Exit { upgrade_requested: bool },
    Redraw,
    Ignore,
}

/// Apply `key` to `state`. A door reached by the cursor opens the batch behind it, the same
/// way the full viewer's keyboard navigation does.
fn handle_key(state: &mut InlineGenGraphState, key: KeyEvent) -> KeyOutcome {
    match key.code {
        KeyCode::Esc | KeyCode::Char('q') | KeyCode::Enter => KeyOutcome::Exit {
            upgrade_requested: false,
        },
        KeyCode::Char('f') => KeyOutcome::Exit {
            upgrade_requested: true,
        },
        KeyCode::Char('p') => {
            if has_path_overlay(&state.overlays) {
                remove_path_overlay(&mut state.overlays);
            } else if let Some(last_path) = state.paths.last().cloned() {
                let path_style = PathStyle::new(current_theme()[0x09])
                    .with_line_style(LineStyle::Bold)
                    .with_merge_glyphs(true);
                set_path_overlay(&mut state.overlays, path_style, last_path);
            } else {
                return KeyOutcome::Ignore;
            }
            state.overlays_dirty = true;
            KeyOutcome::Redraw
        }
        KeyCode::Char('+') | KeyCode::Char('=') => {
            gen_graph_widget::zoom_in(&mut state.view_state, &state.zoom_levels);
            state.overlays_dirty = true;
            KeyOutcome::Redraw
        }
        KeyCode::Char('-') => {
            gen_graph_widget::zoom_out(&mut state.view_state, &state.zoom_levels);
            state.overlays_dirty = true;
            KeyOutcome::Redraw
        }
        _ => match state.view_state.handle_key_event(key) {
            Ok(Some((boundary, target))) => {
                teleport_through_wormhole(
                    &mut state.engine,
                    &mut state.view_state,
                    boundary,
                    target,
                );
                KeyOutcome::Redraw
            }
            // `handle_key_event` reports keys it doesn't bind the same way as a successful
            // move, so only the navigation keys it binds are worth a redraw.
            Ok(None) if is_navigation_key(key.code) => KeyOutcome::Redraw,
            Ok(None) | Err(_) => KeyOutcome::Ignore,
        },
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

/// Draw one frame, then once more if drawing claimed a batch whose dimming, annotation
/// groups, or path highlight differ from what was just drawn.
fn draw_synced<B: Backend>(
    terminal: &mut Terminal<B>,
    state: &mut InlineGenGraphState,
) -> Result<(), B::Error> {
    // A door handled since the last draw may already have grown the graph.
    state.sync_active_world();
    terminal.draw(|frame| render_inline(frame, state))?;
    if state.sync_active_world() {
        terminal.draw(|frame| render_inline(frame, state))?;
    }
    Ok(())
}

/// Run the widget until the user exits, returning whether they asked for the full viewer.
fn run_inline_event_loop<B: Backend>(
    terminal: &mut Terminal<B>,
    state: &mut InlineGenGraphState,
    events: &mut impl EventSource,
) -> Result<bool, B::Error> {
    draw_synced(terminal, state)?;
    loop {
        // Rendering is event-driven (no animation to advance): block indefinitely until the
        // next input event wakes us.
        let Some(event) = events.poll_next(Duration::from_secs(3600)) else {
            continue;
        };
        match event {
            AppEvent::KeyPress(key) => match handle_key(state, key) {
                KeyOutcome::Exit { upgrade_requested } => return Ok(upgrade_requested),
                KeyOutcome::Redraw => {}
                KeyOutcome::Ignore => continue,
            },
            // Width is picked up automatically from `frame.area()` on the next render; the
            // inline viewport's height stays fixed.
            AppEvent::Resize(_, _) => {}
        }
        draw_synced(terminal, state)?;
    }
}

/// Display an inline widget for a `BlockGroup`'s graph, with annotations loaded.
///
/// The widget appears inline in the terminal without taking over the entire screen. Like the
/// full viewer, it loads the graph lazily one batch at a time and steps between batches
/// through their doors.
///
/// # Controls
/// * Arrow keys: Navigate cursor between nodes and pan the view
/// * +/-: Zoom in/out (Minimal → Truncated → Full)
/// * p: Toggle the highlight of the last of `paths`
/// * f: Switch to the full-screen viewer
/// * q/Enter/Esc: Exit the widget
///
/// # Returns
/// * `Ok(true)` if the user requested to transition to full-screen view
/// * `Ok(false)` if completed successfully and exited
pub fn show_inline_block_group_widget(
    conn: &GraphConnection,
    workspace: &Workspace,
    block_group_id: HashId,
    paths: Vec<Path>,
    height: u16,
    history_ref: Option<&str>,
) -> io::Result<bool> {
    let mut state = InlineGenGraphState::new(conn, workspace, &block_group_id, history_ref)
        .map_err(|error| io::Error::other(error.to_string()))?;
    for path in paths {
        state.add_path(&path);
    }

    let terminal_result = panic::catch_unwind(|| {
        ratatui::init_with_options(TerminalOptions {
            viewport: Viewport::Inline(height),
        })
    });

    match terminal_result {
        Ok(mut terminal) => {
            let upgrade_requested =
                run_inline_event_loop(&mut terminal, &mut state, &mut CrosstermEventSource)?;

            // Final render without border -> capture the viewport area
            let viewport_area = terminal.get_frame().area();

            terminal.draw(|frame| render_final(frame, &mut state))?;

            // For inline viewports, we need to manually restore terminal state
            // (ratatui::restore() loses the cursor which resets cursor position incorrectly.

            // Position cursor at the end of the viewport BEFORE restoring terminal mode
            let target_line = viewport_area.y + viewport_area.height;
            let _ =
                crossterm::execute!(std::io::stdout(), crossterm::cursor::MoveTo(0, target_line));

            // Now restore terminal modes manually (show cursor, disable raw mode)
            let _ = crossterm::execute!(std::io::stdout(), crossterm::cursor::Show);
            let _ = crossterm::terminal::disable_raw_mode();

            std::io::Write::flush(&mut std::io::stdout()).ok();

            Ok(upgrade_requested)
        }
        Err(_) => {
            eprintln!("Interactive terminal not available, omitting visualization.");
            Ok(false)
        }
    }
}

/// Draw the inline widget with a border and controls help
fn render_inline(frame: &mut Frame, state: &mut InlineGenGraphState) {
    let area = frame.area();

    // Ratatui layout (not graph layout) - split main area for graph box and controls
    let main_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(1)])
        .split(area);

    let block = Block::default().borders(Borders::ALL);
    let inner_area = block.inner(main_layout[0]);

    // Render the border and content
    frame.render_widget(block, main_layout[0]);

    // Re-register overlay highlights only when the overlay set, zoom level, or loaded batch
    // changed since they were last registered.
    if state.overlays_dirty {
        reapply_overlays(
            &state.engine,
            &mut state.view_state,
            &state.zoom_levels,
            &mut state.overlays,
            &mut state.annotation_colors,
        );
        state.overlays_dirty = false;
    }

    // Create the GenGraph view
    let active_renderer = &state.zoom_levels[state.view_state.zoom_index].1;
    let view = GraphView::new(&mut state.engine, active_renderer);

    // Render the graph view
    frame.render_stateful_widget(view, inner_area, &mut state.view_state);

    // Draw floating annotation labels after the graph.
    let detail_level = state.zoom_levels[state.view_state.zoom_index].0;
    let any_hidden = draw_annotation_labels(
        frame.buffer_mut(),
        inner_area,
        &state.engine,
        &state.view_state,
        &state.zoom_levels,
        &state.overlays,
    );
    let hidden_legend = any_hidden.then(|| {
        if detail_level == VisualDetail::Full {
            "* some annotations hidden due to space constraints"
        } else {
            "* zoom in for more features"
        }
    });
    draw_controls_help(frame, main_layout[1], state, hidden_legend);
}

/// Draw the final plot after the widget is done
fn render_final(frame: &mut Frame, state: &mut InlineGenGraphState) {
    let area = frame.area().offset(ratatui::layout::Offset { x: 0, y: -1 });
    // The final render omits the cursor overlay.
    state.view_state.hide_cursor();

    // Create the GenGraph view
    let active_renderer = &state.zoom_levels[state.view_state.zoom_index].1;
    let view = GraphView::new(&mut state.engine, active_renderer);

    // Render the graph view
    frame.render_stateful_widget(view, area, &mut state.view_state);
}

/// Draw the bottom controls line. When `hidden_legend` is set, it's right-aligned on the
/// same line and the path-visibility shortcut is dropped to make room for it.
fn draw_controls_help(
    frame: &mut Frame,
    area: Rect,
    state: &mut InlineGenGraphState,
    hidden_legend: Option<&str>,
) {
    let help_text = if hidden_legend.is_some() {
        "←→↑↓: Nav | +/-: Zoom | f: Full window | q: Exit".to_string()
    } else if has_path_overlay(&state.overlays) {
        "←→↑↓: Nav | +/-: Zoom | f: Full window | p: Hide Path | q: Exit".to_string()
    } else {
        "←→↑↓: Nav | +/-: Zoom | f: Full window | p: Show Path | q: Exit".to_string()
    };

    let buf = frame.buffer_mut();
    buf.set_string(
        area.x,
        area.y,
        &help_text,
        Style::default().fg(Color::Yellow),
    );

    if let Some(legend) = hidden_legend {
        let help_width = help_text.chars().count() as u16;
        let legend_width = legend.chars().count() as u16;
        let legend_x = area.right().saturating_sub(legend_width);
        if legend_x > area.x + help_width {
            buf.set_string(
                legend_x,
                area.y,
                legend,
                Style::default().fg(current_theme()[0x09]),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use crossterm::event::KeyModifiers;
    use gen_models::db::get_connection;
    use ratatui::backend::TestBackend;

    use super::*;
    use crate::views::{
        gen_graph_widget::DEFAULT_ZOOM_LEVEL,
        lazy_graph_source::tests::{setup_circular_block_group, setup_labelled_chain_block_group},
    };

    /// Replays a fixed list of key presses, then exits.
    struct ScriptedEvents(VecDeque<KeyCode>);

    impl EventSource for ScriptedEvents {
        fn poll_next(&mut self, _timeout: Duration) -> Option<AppEvent> {
            let code = self.0.pop_front().unwrap_or(KeyCode::Char('q'));
            Some(AppEvent::KeyPress(KeyEvent::new(code, KeyModifiers::NONE)))
        }
    }

    fn chain_labels(count: usize) -> Vec<String> {
        (0..count).map(|index| format!("n{index}")).collect()
    }

    fn run_script(state: &mut InlineGenGraphState, width: u16, keys: Vec<KeyCode>) -> bool {
        let mut terminal =
            Terminal::new(TestBackend::new(width, 12)).expect("should create a test terminal");
        run_inline_event_loop(&mut terminal, state, &mut ScriptedEvents(keys.into()))
            .expect("should run the inline event loop")
    }

    #[test]
    fn test_inline_state_loads_only_the_first_batch_of_a_long_chain() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("graph.db");
        let labels = chain_labels(80);
        let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();
        let (block_group_id, _) = setup_labelled_chain_block_group(&db_path, &label_refs);
        let conn = get_connection(&db_path).unwrap();
        let workspace = Workspace::from_current_dir();

        let mut state = InlineGenGraphState::new(&conn, &workspace, &block_group_id, None)
            .expect("should load the block group");
        assert_eq!(state.view_state.zoom_index, DEFAULT_ZOOM_LEVEL);
        assert!(state.engine.graph().node_count() <= 2);

        assert!(!run_script(&mut state, 12, Vec::new()));
        let loaded = state.engine.graph().node_count();
        assert!(
            loaded > 2 && loaded < labels.len(),
            "only the first batch should be loaded, got {loaded} nodes"
        );
    }

    #[test]
    fn test_inline_keyboard_door_leaves_the_first_batch() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("graph.db");
        let labels = chain_labels(80);
        let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();
        let (block_group_id, _) = setup_labelled_chain_block_group(&db_path, &label_refs);
        let conn = get_connection(&db_path).unwrap();
        let workspace = Workspace::from_current_dir();
        let mut state = InlineGenGraphState::new(&conn, &workspace, &block_group_id, None)
            .expect("should load the block group");

        let mut terminal =
            Terminal::new(TestBackend::new(12, 12)).expect("should create a test terminal");
        draw_synced(&mut terminal, &mut state).expect("should draw the first batch");
        let first_batch = state.engine.active_batch();
        assert!(first_batch.is_some());

        let right = KeyEvent::new(KeyCode::Right, KeyModifiers::NONE);
        for _ in 0..2000 {
            if handle_key(&mut state, right) == KeyOutcome::Redraw {
                draw_synced(&mut terminal, &mut state).expect("should draw");
            }
            if state.engine.active_batch() != first_batch {
                break;
            }
        }
        let active_batch = state.engine.active_batch();
        assert!(active_batch.is_some());
        assert_ne!(active_batch, first_batch);
    }

    #[test]
    fn test_inline_navigation_keys_do_not_reapply_overlays() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("graph.db");
        let (block_group_id, _) = setup_labelled_chain_block_group(&db_path, &["x", "y", "z"]);
        let conn = get_connection(&db_path).unwrap();
        let workspace = Workspace::from_current_dir();
        let mut state = InlineGenGraphState::new(&conn, &workspace, &block_group_id, None)
            .expect("should load the block group");
        run_script(&mut state, 80, Vec::new());
        assert!(!state.overlays_dirty);

        let press = |code| KeyEvent::new(code, KeyModifiers::NONE);
        assert_eq!(
            handle_key(&mut state, press(KeyCode::Char('x'))),
            KeyOutcome::Ignore
        );
        assert_eq!(
            handle_key(&mut state, press(KeyCode::Right)),
            KeyOutcome::Redraw
        );
        assert!(!state.overlays_dirty);
        assert_eq!(
            handle_key(&mut state, press(KeyCode::Char('+'))),
            KeyOutcome::Redraw
        );
        assert!(state.overlays_dirty);
        // With no paths added, `p` has nothing to toggle.
        assert_eq!(
            handle_key(&mut state, press(KeyCode::Char('p'))),
            KeyOutcome::Ignore
        );
    }

    #[test]
    fn test_inline_path_toggle_highlights_a_lazily_loaded_graph() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("graph.db");
        let (block_group_id, edge_ids) =
            setup_labelled_chain_block_group(&db_path, &["x", "y", "z"]);
        let conn = get_connection(&db_path).unwrap();
        let path = Path::create(&conn, "chain", &block_group_id, &edge_ids).unwrap();
        let workspace = Workspace::from_current_dir();
        let mut state = InlineGenGraphState::new(&conn, &workspace, &block_group_id, None)
            .expect("should load the block group");
        state.add_path(&path);

        run_script(&mut state, 80, vec![KeyCode::Char('p')]);

        assert!(has_path_overlay(&state.overlays));
        assert!(!state.view_state.highlights.styles.is_empty());
    }

    #[test]
    fn test_inline_path_toggle_on_a_circular_block_group_does_not_panic() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("graph.db");
        let (block_group_id, edge_ids) = setup_circular_block_group(&db_path);
        let conn = get_connection(&db_path).unwrap();
        let path = Path::create(&conn, "circle", &block_group_id, &edge_ids).unwrap();
        let workspace = Workspace::from_current_dir();
        let mut state = InlineGenGraphState::new(&conn, &workspace, &block_group_id, None)
            .expect("should load the block group");
        state.add_path(&path);

        run_script(&mut state, 80, vec![KeyCode::Char('p'), KeyCode::Char('p')]);

        assert!(!has_path_overlay(&state.overlays));
    }
}
