use std::{
    io::{self, Write},
    panic,
    time::Duration,
};

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind};
use gen_core::{HashId, Workspace};
use gen_diff::graph::DiffGenGraph;
use gen_graph::{GraphNode, PackedGenGraph};
use gen_models::{block_group::BlockGroup, db::GraphConnection, path::Path};
use gen_tui::{
    graph_view::{GraphView, GraphViewState},
    highlight_style::PathStyle,
    layout_engine::LayoutEngine,
};
use ratatui::{
    Terminal, TerminalOptions, Viewport,
    prelude::*,
    widgets::{Block, Borders, Clear, Paragraph},
};

use crate::views::{
    annotation_files::AnnotationFileEntry,
    annotations::{AnnotationFileTrackRequest, load_annotation_file_track},
    block_group::{
        active_neighborhood_coordinate_window, discard_pending_input, expand_query_window,
    },
    diff_annotations::DiffAnnotations,
    diff_graph::{
        DiffGraphComponent, DiffLegend, apply_diff_highlights, build_diff_graph_component,
        handle_diff_key,
    },
    gen_graph_controller::{GenGraphController, GraphKeyOutcome},
    gen_graph_widget::ZoomLevels,
    graph_database::GraphDatabase,
    graph_overlay::{file_track_overlays, has_path_overlay},
    helpers::style_text,
    path_scale::ScalePlacement,
    png_export::write_cropped_png,
};

#[derive(Debug)]
pub enum AppEvent {
    KeyPress(KeyEvent),
    Resize(u16, u16),
}

pub trait EventSource {
    fn poll_next(&mut self, timeout: Duration) -> Option<AppEvent>;

    /// Drop every event already waiting, returning whether one of them was a resize.
    fn discard_pending(&mut self) -> bool;
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

    fn discard_pending(&mut self) -> bool {
        discard_pending_input().unwrap_or(false)
    }
}

/// How the inline widget was left.
pub enum InlineOutcome {
    /// The user closed the widget.
    Closed,
    /// The user asked for the full-screen viewer, which keeps drawing this controller.
    OpenFullViewer(Box<GenGraphController>),
}

/// What a key press asks of the event loop.
#[derive(Debug, PartialEq)]
enum KeyOutcome {
    Exit {
        upgrade_requested: bool,
    },
    Redraw,
    /// Redraw, then discard the input queued while the door's batch loaded.
    EnteredDoor,
    Ignore,
}

/// Apply `key` to `controller`: the inline widget's own exit keys, then the graph keys every
/// viewer shares.
fn handle_key(controller: &mut GenGraphController, key: KeyEvent) -> KeyOutcome {
    match key.code {
        KeyCode::Esc | KeyCode::Char('q') | KeyCode::Enter => KeyOutcome::Exit {
            upgrade_requested: false,
        },
        KeyCode::Char('f') => KeyOutcome::Exit {
            upgrade_requested: true,
        },
        _ => match controller.handle_key(key) {
            GraphKeyOutcome::Redraw => KeyOutcome::Redraw,
            GraphKeyOutcome::EnteredDoor => KeyOutcome::EnteredDoor,
            GraphKeyOutcome::Ignore => KeyOutcome::Ignore,
        },
    }
}

/// Draw one frame, then once more if drawing claimed a batch whose dimming, annotation
/// groups, or path highlight differ from what was just drawn.
fn draw_synced<B: Backend>(
    terminal: &mut Terminal<B>,
    controller: &mut GenGraphController,
) -> Result<(), B::Error> {
    // A door handled since the last draw may already have grown the graph.
    controller.sync_active_world();
    terminal.draw(|frame| render_inline(frame, controller))?;
    if controller.sync_active_world().changed {
        terminal.draw(|frame| render_inline(frame, controller))?;
    }
    Ok(())
}

/// Run the widget until the user exits, returning whether they asked for the full viewer.
fn run_inline_event_loop<B: Backend>(
    terminal: &mut Terminal<B>,
    controller: &mut GenGraphController,
    events: &mut impl EventSource,
) -> Result<bool, B::Error> {
    draw_synced(terminal, controller)?;
    loop {
        // Rendering is event-driven (no animation to advance): block indefinitely until the
        // next input event wakes us.
        let Some(event) = events.poll_next(Duration::from_secs(3600)) else {
            continue;
        };
        let entered_door = match event {
            AppEvent::KeyPress(key) => match handle_key(controller, key) {
                KeyOutcome::Exit { upgrade_requested } => return Ok(upgrade_requested),
                KeyOutcome::Redraw => false,
                KeyOutcome::EnteredDoor => true,
                KeyOutcome::Ignore => continue,
            },
            // Width is picked up automatically from `frame.area()` on the next render; the
            // inline viewport's height stays fixed.
            AppEvent::Resize(_, _) => false,
        };
        draw_synced(terminal, controller)?;
        // Keys pressed while the new batch loaded were aimed at the old screen.
        if entered_door && events.discard_pending() {
            draw_synced(terminal, controller)?;
        }
    }
}

/// Build the controller the inline widget draws. The plot stays in the scrollback, which cannot
/// hold the Kitty overlay that curved edges are drawn on, so the controller never offers them and
/// `s` does nothing. The full viewer sets this again for itself when `f` hands it the controller.
fn inline_controller(
    conn: &GraphConnection,
    workspace: &Workspace,
    block_group_id: &HashId,
    history_ref: Option<&str>,
) -> io::Result<GenGraphController> {
    let mut controller = GraphDatabase::for_connection(conn, workspace)
        .and_then(|database| {
            GenGraphController::for_block_group(
                database,
                block_group_id,
                history_ref.map(String::from),
            )
        })
        .map_err(|error| io::Error::other(error.to_string()))?;
    controller.set_curved_layout_available(false);
    // The inline widget follows the keyboard cursor from the start.
    controller.view_state_mut().show_cursor();
    Ok(controller)
}

/// Display an inline widget for a `BlockGroup`'s graph, with annotations loaded: the ones
/// stored in the database and the imported annotation `files`.
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
/// * `Ok(InlineOutcome::OpenFullViewer(_))` if the user requested the full-screen view,
///   carrying the controller for it to keep drawing
/// * `Ok(InlineOutcome::Closed)` if completed successfully and exited
pub fn show_inline_block_group_widget<'a>(
    conn: &'a GraphConnection,
    workspace: &'a Workspace,
    block_group_id: HashId,
    paths: Vec<Path>,
    height: u16,
    history_ref: Option<&'a str>,
    files: &[AnnotationFileEntry],
) -> io::Result<InlineOutcome> {
    let mut controller = inline_controller(conn, workspace, &block_group_id, history_ref)?;
    for path in paths {
        controller.add_path(&path);
    }

    let terminal_result = panic::catch_unwind(|| {
        ratatui::init_with_options(TerminalOptions {
            viewport: Viewport::Inline(height),
        })
    });

    match terminal_result {
        Ok(mut terminal) => {
            // Annotation files are read for the nodes already loaded, so they follow the
            // first draw.
            if !files.is_empty() {
                draw_synced(&mut terminal, &mut controller)?;
                show_annotation_files(
                    &mut controller,
                    conn,
                    workspace,
                    &block_group_id,
                    files,
                    history_ref,
                );
            }
            let upgrade_requested =
                run_inline_event_loop(&mut terminal, &mut controller, &mut CrosstermEventSource)?;

            // Final render without border -> capture the viewport area
            let viewport_area = terminal.get_frame().area();

            terminal.draw(|frame| render_final(frame, &mut controller))?;
            leave_inline_terminal(viewport_area);

            if upgrade_requested {
                Ok(InlineOutcome::OpenFullViewer(Box::new(controller)))
            } else {
                Ok(InlineOutcome::Closed)
            }
        }
        Err(_) => {
            eprintln!("Interactive terminal not available, omitting visualization.");
            Ok(InlineOutcome::Closed)
        }
    }
}

/// Hand the terminal back after an inline widget, leaving its last frame in the scrollback.
fn leave_inline_terminal(viewport_area: Rect) {
    // For inline viewports, we need to manually restore terminal state
    // (ratatui::restore() loses the cursor which resets cursor position incorrectly.

    // Position cursor at the end of the viewport BEFORE restoring terminal mode
    let target_line = viewport_area.y + viewport_area.height;
    let _ = crossterm::execute!(std::io::stdout(), crossterm::cursor::MoveTo(0, target_line));

    // Now restore terminal modes manually (show cursor, disable raw mode)
    let _ = crossterm::execute!(std::io::stdout(), crossterm::cursor::Show);
    let _ = crossterm::terminal::disable_raw_mode();

    std::io::Write::flush(&mut std::io::stdout()).ok();
}

/// A sample diff held in memory with its layout engine, the way the full-screen diff viewer
/// holds it. A diff graph is not stored in a block group, so it has no lazy crawl to drive.
struct InlineDiffView<'a> {
    annotations: DiffAnnotations,
    component: DiffGraphComponent,
    engine: LayoutEngine<PackedGenGraph>,
    zoom_levels: ZoomLevels<'a>,
    view_state: GraphViewState<GraphNode>,
    legend: Line<'static>,
}

impl<'a> InlineDiffView<'a> {
    fn new(
        conn: &'a GraphConnection,
        workspace: &'a Workspace,
        diff_graph: &DiffGenGraph,
        legend: &DiffLegend,
        annotations: DiffAnnotations,
    ) -> Self {
        let component = build_diff_graph_component(diff_graph, String::new());
        let (engine, zoom_levels, mut view_state) =
            annotations.create_engine(component.graph.clone(), (conn, workspace));
        apply_diff_highlights(&mut view_state, &component);
        // The inline widget follows the keyboard cursor from the start.
        view_state.show_cursor();
        Self {
            annotations,
            engine,
            zoom_levels,
            view_state,
            legend: legend.line(component.has_modified_marks()),
            component,
        }
    }

    /// Draw the graph on the terminal's own background, like the inline block-group widget, so
    /// the plot sits in the scrollback instead of looking like a separate screen.
    fn render(&mut self, frame: &mut Frame, area: Rect) {
        self.annotations.prepare(
            &self.engine,
            &mut self.view_state,
            &self.zoom_levels,
            &self.component,
        );
        let renderer = &self.zoom_levels[self.view_state.zoom_index].1;
        let view = GraphView::new(&mut self.engine, renderer).style(Style::default());
        frame.render_stateful_widget(view, area, &mut self.view_state);
    }

    /// Draw the diff inside a border with the controls line below it.
    fn render_inline(&mut self, frame: &mut Frame) {
        let main_layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(0), Constraint::Length(2)])
            .split(frame.area());
        let block = Block::default().borders(Borders::ALL);
        let inner_area = block.inner(main_layout[0]);
        frame.render_widget(block, main_layout[0]);
        self.render(frame, inner_area);
        draw_diff_controls_help(frame, main_layout[1], &self.legend);
    }

    /// Draw the final plot without the border or cursor, for the scrollback.
    fn render_final(&mut self, frame: &mut Frame) {
        let area = frame.area().offset(ratatui::layout::Offset { x: 0, y: -1 });
        self.view_state.hide_cursor();
        self.render(frame, area);
    }
}

/// Run the diff widget until the user exits, returning whether they asked for the full viewer.
fn run_inline_diff_event_loop<B: Backend>(
    terminal: &mut Terminal<B>,
    view: &mut InlineDiffView<'_>,
    events: &mut impl EventSource,
) -> Result<bool, B::Error> {
    terminal.draw(|frame| view.render_inline(frame))?;
    loop {
        // Rendering is event-driven: block until the next input event wakes us.
        let Some(event) = events.poll_next(Duration::from_secs(3600)) else {
            continue;
        };
        if let AppEvent::KeyPress(key) = event {
            match key.code {
                KeyCode::Esc | KeyCode::Char('q') | KeyCode::Enter => return Ok(false),
                KeyCode::Char('f') => return Ok(true),
                _ => handle_diff_key(&mut view.view_state, &view.zoom_levels, key),
            }
        }
        terminal.draw(|frame| view.render_inline(frame))?;
    }
}

/// Display an inline sample diff using the standard added/removed highlights.
///
/// The widget appears inline in the terminal without taking over the entire screen. Use the
/// arrow keys to navigate, +/- to zoom, `f` to ask for the full-screen diff viewer and
/// q/Enter/Esc to exit.
///
/// # Returns
/// * `Ok(true)` if the user requested the full-screen view
/// * `Ok(false)` if completed successfully and exited
pub fn show_inline_diff_graph_widget(
    conn: &GraphConnection,
    workspace: &Workspace,
    diff_graph: &DiffGenGraph,
    legend: &DiffLegend,
    annotations: DiffAnnotations,
    height: u16,
) -> io::Result<bool> {
    let mut view = InlineDiffView::new(conn, workspace, diff_graph, legend, annotations);
    let terminal_result = panic::catch_unwind(|| {
        ratatui::init_with_options(TerminalOptions {
            viewport: Viewport::Inline(height),
        })
    });

    match terminal_result {
        Ok(mut terminal) => {
            let upgrade_requested =
                run_inline_diff_event_loop(&mut terminal, &mut view, &mut CrosstermEventSource)?;
            let viewport_area = terminal.get_frame().area();
            terminal.draw(|frame| view.render_final(frame))?;
            leave_inline_terminal(viewport_area);
            Ok(upgrade_requested)
        }
        Err(_) => {
            eprintln!("Interactive terminal not available, omitting visualization.");
            Ok(false)
        }
    }
}

/// Size of a rendered plot in terminal cells; the PNG is this many cells at the exporter's cell size.
#[derive(Debug, Clone, Copy)]
pub struct PlotSize {
    pub columns: u16,
    pub rows: u16,
    /// Image pixels per logical pixel. A scale of 2 writes the same plot on twice as many
    /// pixels, which stays sharp on high-density screens.
    pub scale: usize,
}

/// Write a PNG of a `BlockGroup`'s graph, with its stored annotations and the imported annotation
/// `files`, without needing a terminal.
pub fn write_block_group_png(
    conn: &GraphConnection,
    workspace: &Workspace,
    block_group_id: HashId,
    history_ref: Option<&str>,
    files: &[AnnotationFileEntry],
    size: PlotSize,
    output: &mut impl Write,
) -> io::Result<()> {
    // Same defaults as the Python plot: retired edit-site edges are left out of the graph.
    let database = GraphDatabase::for_connection(conn, workspace)
        .map_err(|error| io::Error::other(error.to_string()))?;
    let mut controller = GenGraphController::new(database, history_ref.map(String::from))
        .with_pruned_history(true)
        .with_layout_mode(gen_tui::LayoutMode::Curved);
    controller
        .open_block_group(&block_group_id)
        .map_err(|error| io::Error::other(error.to_string()))?;
    controller.set_scale_placement(ScalePlacement::BelowContent);
    let area = Rect::new(0, 0, size.columns, size.rows);
    let mut buffer = Buffer::empty(area);
    draw_flags_under_nodes(&mut controller, &mut buffer, area);
    // Annotation files are read for the nodes already loaded, so they follow the first draw.
    let redraw = show_annotation_files(
        &mut controller,
        conn,
        workspace,
        &block_group_id,
        files,
        history_ref,
    );
    if redraw {
        Clear.render(area, &mut buffer);
        draw_flags_under_nodes(&mut controller, &mut buffer, area);
    }
    write_cropped_png(&buffer, controller.overlay_primitives(), size.scale, output)
}

/// Write a PNG of a sample diff using the standard added/removed highlights.
pub fn write_diff_graph_png(
    conn: &GraphConnection,
    workspace: &Workspace,
    diff_graph: &DiffGenGraph,
    mut annotations: DiffAnnotations,
    size: PlotSize,
    output: &mut impl Write,
) -> io::Result<()> {
    let component = build_diff_graph_component(diff_graph, String::new());
    let (mut engine, zoom_levels, mut view_state) =
        annotations.create_engine(component.graph.clone(), (conn, workspace));
    apply_diff_highlights(&mut view_state, &component);
    annotations.prepare(&engine, &mut view_state, &zoom_levels, &component);
    let area = Rect::new(0, 0, size.columns, size.rows);
    let mut buffer = Buffer::empty(area);
    view_state.layout_mode = gen_tui::LayoutMode::Curved;
    let view = GraphView::new(&mut engine, &zoom_levels[view_state.zoom_index].1);
    StatefulWidget::render(view, area, &mut buffer, &mut view_state);
    write_cropped_png(&buffer, &view_state.overlay_primitives, size.scale, output)
}

fn draw_flags_under_nodes(controller: &mut GenGraphController, buffer: &mut Buffer, area: Rect) {
    controller.render_settled(buffer, area, Style::default());
}

/// Show `files` as overlay tracks over the loaded graph, returning whether any file added
/// annotations. A file that can't be read is skipped so it doesn't hide the others.
fn show_annotation_files(
    controller: &mut GenGraphController,
    conn: &GraphConnection,
    workspace: &Workspace,
    block_group_id: &HashId,
    files: &[AnnotationFileEntry],
    history_ref: Option<&str>,
) -> bool {
    let Ok(block_group) = BlockGroup::get_by_id(conn, block_group_id, history_ref) else {
        return false;
    };
    let node_filter = controller.loaded_node_ids();
    let query_window =
        active_neighborhood_coordinate_window(controller.engine()).map(expand_query_window);
    let mut shown = false;
    for entry in files {
        let Ok(loaded) = load_annotation_file_track(&AnnotationFileTrackRequest {
            conn,
            history_ref,
            workspace,
            collection_name: &block_group.collection_name,
            sample_name: &block_group.sample_name,
            block_group_name: Some(&block_group.name),
            query_window,
            node_filter: &node_filter,
            entry,
        }) else {
            continue;
        };
        let overlays = file_track_overlays(loaded.track);
        shown |= !overlays.is_empty();
        controller.overlays_mut().extend(overlays);
    }
    shown
}

/// Draw the inline widget with a border and controls help
fn render_inline(frame: &mut Frame, controller: &mut GenGraphController) {
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

    controller.render(frame, inner_area, Style::default());
    draw_controls_help(frame, main_layout[1], controller);
}

/// Draw the final plot after the widget is done
fn render_final(frame: &mut Frame, controller: &mut GenGraphController) {
    let area = frame.area().offset(ratatui::layout::Offset { x: 0, y: -1 });
    // The final render omits the cursor overlay.
    controller.render_plain(frame, area);
}

/// Draw the bottom controls line.
fn draw_controls_help(frame: &mut Frame, area: Rect, controller: &GenGraphController) {
    let help_text = if has_path_overlay(controller.overlays()) {
        "*←↓↑→/hjkl,wbe* nav | *+/-* zoom | *f* full window | *p* hide path | *q* exit"
    } else {
        "*←↓↑→/hjkl,wbe* nav | *+/-* zoom | *f* full window | *p* show path | *q* exit"
    };
    let key_style = Style::default().fg(Color::Yellow).bold();
    let text_style = Style::default().fg(Color::Yellow);
    frame.render_widget(
        Paragraph::new(style_text(help_text, text_style, key_style)),
        area,
    );
}

/// The controls line with the diff's color legend on the line below it.
fn draw_diff_controls_help(frame: &mut Frame, area: Rect, legend: &Line<'static>) {
    let key_style = Style::default().fg(Color::Yellow).bold();
    let text_style = Style::default().fg(Color::Yellow);
    let controls = style_text(
        "*←↓↑→/hjkl* nav | *+/-* zoom | *f* full window | *q* exit",
        text_style,
        key_style,
    );
    frame.render_widget(Paragraph::new(vec![controls, legend.clone()]), area);
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use crossterm::event::KeyModifiers;
    use gen_core::{DoltHashId, Strand};
    use gen_diff::graph::{DiffChange, DiffChangeKind, DiffGraphEdge, DiffGraphNode};
    use gen_graph::GraphEdge;
    use gen_models::db::get_connection;
    use gen_tui::LayoutMode;
    use petgraph::Direction::Incoming;
    use ratatui::backend::TestBackend;

    use super::*;
    use crate::{
        test_helpers::setup_gen,
        views::{
            gen_graph_widget::TRUNCATED_ZOOM_LEVEL,
            lazy_graph_source::tests::{
                setup_circular_block_group, setup_labelled_chain_block_group,
            },
        },
    };

    /// Replays a fixed list of key presses, then exits. The whole remaining script counts as
    /// already queued, so discarding pending input drops all of it.
    struct ScriptedEvents(VecDeque<KeyCode>);

    impl EventSource for ScriptedEvents {
        fn poll_next(&mut self, _timeout: Duration) -> Option<AppEvent> {
            let code = self.0.pop_front().unwrap_or(KeyCode::Char('q'));
            Some(AppEvent::KeyPress(KeyEvent::new(code, KeyModifiers::NONE)))
        }

        fn discard_pending(&mut self) -> bool {
            self.0.clear();
            false
        }
    }

    fn chain_labels(count: usize) -> Vec<String> {
        (0..count).map(|index| format!("n{index}")).collect()
    }

    fn run_script(controller: &mut GenGraphController, width: u16, keys: Vec<KeyCode>) -> bool {
        let mut terminal =
            Terminal::new(TestBackend::new(width, 12)).expect("should create a test terminal");
        run_inline_event_loop(&mut terminal, controller, &mut ScriptedEvents(keys.into()))
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

        let mut controller = GenGraphController::for_block_group(
            GraphDatabase::for_connection(&conn, &workspace)
                .expect("should open the graph database"),
            &block_group_id,
            None,
        )
        .expect("should load the block group");
        assert_eq!(controller.view_state().zoom_index, TRUNCATED_ZOOM_LEVEL);
        assert!(controller.engine().graph().node_count() <= 2);

        assert!(!run_script(&mut controller, 12, Vec::new()));
        let loaded = controller.engine().graph().node_count();
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
        let mut controller = GenGraphController::for_block_group(
            GraphDatabase::for_connection(&conn, &workspace)
                .expect("should open the graph database"),
            &block_group_id,
            None,
        )
        .expect("should load the block group");

        let mut terminal =
            Terminal::new(TestBackend::new(12, 12)).expect("should create a test terminal");
        draw_synced(&mut terminal, &mut controller).expect("should draw the first batch");
        let first_batch = controller.engine().active_batch();
        assert!(first_batch.is_some());

        let right = KeyEvent::new(KeyCode::Right, KeyModifiers::NONE);
        for _ in 0..2000 {
            if handle_key(&mut controller, right) != KeyOutcome::Ignore {
                draw_synced(&mut terminal, &mut controller).expect("should draw");
            }
            if controller.engine().active_batch() != first_batch {
                break;
            }
        }
        let active_batch = controller.engine().active_batch();
        assert!(active_batch.is_some());
        assert_ne!(active_batch, first_batch);
    }

    #[test]
    fn test_inline_door_discards_keys_queued_while_loading() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("graph.db");
        let labels = chain_labels(80);
        let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();
        let (block_group_id, _) = setup_labelled_chain_block_group(&db_path, &label_refs);
        let conn = get_connection(&db_path).unwrap();
        let workspace = Workspace::from_current_dir();
        let mut controller = GenGraphController::for_block_group(
            GraphDatabase::for_connection(&conn, &workspace)
                .expect("should open the graph database"),
            &block_group_id,
            None,
        )
        .expect("should load the block group");
        controller.view_state_mut().show_cursor();

        let mut events = ScriptedEvents(vec![KeyCode::Right; 2000].into());
        let mut terminal =
            Terminal::new(TestBackend::new(12, 12)).expect("should create a test terminal");
        run_inline_event_loop(&mut terminal, &mut controller, &mut events)
            .expect("should run the inline event loop");

        assert!(
            events.0.is_empty(),
            "the presses left after the door should be discarded"
        );
        // Nothing past the door moved the cursor: it is still on the door's target, whose
        // predecessor lies in the batch the door led out of.
        let engine = controller.engine();
        let cursor = controller
            .view_state()
            .cursor
            .node
            .expect("should keep a cursor");
        let cursor_batch = engine.batch_of(cursor);
        assert!(cursor_batch.is_some());
        let predecessor = engine
            .graph()
            .neighbors_directed(cursor, Incoming)
            .next()
            .expect("should have a predecessor");
        assert_ne!(engine.batch_of(predecessor), cursor_batch);
    }

    #[test]
    fn test_inline_s_key_keeps_the_rectilinear_layout_mode() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("graph.db");
        let (block_group_id, _) = setup_labelled_chain_block_group(&db_path, &["x", "y", "z"]);
        let conn = get_connection(&db_path).unwrap();
        let workspace = Workspace::from_current_dir();
        let mut controller = inline_controller(&conn, &workspace, &block_group_id, None)
            .expect("should build the inline controller");
        assert!(!controller.curved_layout_available());
        assert_eq!(controller.layout_mode(), LayoutMode::Rectilinear);

        let press = KeyEvent::new(KeyCode::Char('s'), KeyModifiers::NONE);
        assert_eq!(handle_key(&mut controller, press), KeyOutcome::Ignore);
        assert_eq!(controller.layout_mode(), LayoutMode::Rectilinear);
    }

    #[test]
    fn test_inline_exit_keys_leave_the_widget() {
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

        let press = |code| KeyEvent::new(code, KeyModifiers::NONE);
        assert_eq!(
            handle_key(&mut controller, press(KeyCode::Char('f'))),
            KeyOutcome::Exit {
                upgrade_requested: true
            }
        );
        assert_eq!(
            handle_key(&mut controller, press(KeyCode::Esc)),
            KeyOutcome::Exit {
                upgrade_requested: false
            }
        );
        assert_eq!(
            handle_key(&mut controller, press(KeyCode::Char('x'))),
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
        let mut controller = GenGraphController::for_block_group(
            GraphDatabase::for_connection(&conn, &workspace)
                .expect("should open the graph database"),
            &block_group_id,
            None,
        )
        .expect("should load the block group");
        controller.add_path(&path);

        run_script(&mut controller, 80, vec![KeyCode::Char('p')]);

        assert!(has_path_overlay(controller.overlays()));
        assert!(!controller.view_state().highlights.styles.is_empty());
    }

    #[test]
    fn test_inline_path_toggle_on_a_circular_block_group_does_not_panic() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("graph.db");
        let (block_group_id, edge_ids) = setup_circular_block_group(&db_path);
        let conn = get_connection(&db_path).unwrap();
        let path = Path::create(&conn, "circle", &block_group_id, &edge_ids).unwrap();
        let workspace = Workspace::from_current_dir();
        let mut controller = GenGraphController::for_block_group(
            GraphDatabase::for_connection(&conn, &workspace)
                .expect("should open the graph database"),
            &block_group_id,
            None,
        )
        .expect("should load the block group");
        controller.add_path(&path);

        run_script(
            &mut controller,
            80,
            vec![KeyCode::Char('p'), KeyCode::Char('p')],
        );

        assert!(!has_path_overlay(controller.overlays()));
    }

    #[test]
    fn test_write_block_group_png_emits_png() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("graph.db");
        let (block_group_id, _) = setup_labelled_chain_block_group(&db_path, &["ACGT", "TTGA"]);
        let conn = get_connection(&db_path).unwrap();
        let workspace = Workspace::from_current_dir();

        let mut bytes = Vec::new();
        write_block_group_png(
            &conn,
            &workspace,
            block_group_id,
            None,
            &[],
            PlotSize {
                columns: 40,
                rows: 10,
                scale: 1,
            },
            &mut bytes,
        )
        .unwrap();

        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
    }

    fn png_dimensions(bytes: &[u8]) -> (u32, u32) {
        let read = |range: core::ops::Range<usize>| {
            u32::from_be_bytes(bytes[range].try_into().expect("should read four bytes"))
        };
        (read(16..20), read(20..24))
    }

    #[test]
    fn test_write_block_group_png_at_scale_two_doubles_the_pixels() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("graph.db");
        let (block_group_id, _) = setup_labelled_chain_block_group(&db_path, &["ACGT", "TTGA"]);
        let conn = get_connection(&db_path).unwrap();
        let workspace = Workspace::from_current_dir();
        let write = |scale: usize| {
            let mut bytes = Vec::new();
            write_block_group_png(
                &conn,
                &workspace,
                block_group_id,
                None,
                &[],
                PlotSize {
                    columns: 40,
                    rows: 10,
                    scale,
                },
                &mut bytes,
            )
            .unwrap();
            png_dimensions(&bytes)
        };

        let (width, height) = write(1);
        assert_eq!(write(2), (width * 2, height * 2));
    }

    fn added_chain_diff() -> DiffGenGraph {
        let node = |id: i64, start: i64, end: i64| DiffGraphNode {
            node: GraphNode {
                node_id: HashId::pad_str(id),
                sequence_start: start,
                sequence_end: end,
            },
            change: DiffChange::new(DiffChangeKind::Added, Some(DoltHashId([100; 20]))),
        };
        let mut graph = DiffGenGraph::new();
        graph.add_edge(
            node(1, 0, 4),
            node(2, 4, 8),
            vec![DiffGraphEdge {
                edge: GraphEdge {
                    edge_id: HashId::pad_str(10),
                    source_strand: Strand::Forward,
                    target_strand: Strand::Forward,
                    chromosome_index: 0,
                    phased: 0,
                    created_on: 0,
                },
                change: DiffChange::new(DiffChangeKind::Added, Some(DoltHashId([100; 20]))),
            }],
        );
        graph
    }

    fn run_diff_script(keys: Vec<KeyCode>) -> bool {
        let context = setup_gen();
        let mut view = InlineDiffView::new(
            context.graph().conn(),
            context.workspace(),
            &added_chain_diff(),
            &DiffLegend::between("query", "base"),
            DiffAnnotations::none(),
        );
        let mut terminal =
            Terminal::new(TestBackend::new(40, 12)).expect("should create a test terminal");
        run_inline_diff_event_loop(&mut terminal, &mut view, &mut ScriptedEvents(keys.into()))
            .expect("should run the inline diff event loop")
    }

    #[test]
    fn test_inline_diff_exit_keys_close_and_f_asks_for_the_full_viewer() {
        assert!(!run_diff_script(vec![KeyCode::Char('q')]));
        assert!(!run_diff_script(vec![KeyCode::Right, KeyCode::Esc]));
        assert!(run_diff_script(vec![KeyCode::Right, KeyCode::Char('f')]));
    }

    #[test]
    fn test_inline_diff_draws_added_nodes_with_the_added_highlight() {
        let context = setup_gen();
        let mut view = InlineDiffView::new(
            context.graph().conn(),
            context.workspace(),
            &added_chain_diff(),
            &DiffLegend::between("query", "base"),
            DiffAnnotations::none(),
        );
        let mut terminal =
            Terminal::new(TestBackend::new(40, 12)).expect("should create a test terminal");

        terminal
            .draw(|frame| view.render_inline(frame))
            .expect("should draw the diff");

        let buffer = terminal.backend().buffer();
        assert!(
            buffer.content().iter().any(|cell| cell.fg == Color::Green),
            "added nodes and edges should be drawn in the added color"
        );
        assert_eq!(
            buffer[(0, 0)].symbol(),
            "┌",
            "the inline frame has a border"
        );
    }

    #[test]
    fn test_inline_diff_final_render_has_no_border_or_cursor() {
        let context = setup_gen();
        let mut view = InlineDiffView::new(
            context.graph().conn(),
            context.workspace(),
            &added_chain_diff(),
            &DiffLegend::between("query", "base"),
            DiffAnnotations::none(),
        );
        let mut terminal =
            Terminal::new(TestBackend::new(40, 12)).expect("should create a test terminal");

        terminal
            .draw(|frame| view.render_final(frame))
            .expect("should draw the final diff");

        assert_ne!(terminal.backend().buffer()[(0, 0)].symbol(), "┌");
        assert!(!view.view_state.is_cursor_visible());
    }

    #[test]
    fn test_inline_diff_zoom_keys_change_the_zoom_level_and_the_drawing() {
        let context = setup_gen();
        let mut view = InlineDiffView::new(
            context.graph().conn(),
            context.workspace(),
            &added_chain_diff(),
            &DiffLegend::between("query", "base"),
            DiffAnnotations::none(),
        );
        let mut terminal =
            Terminal::new(TestBackend::new(60, 12)).expect("should create a test terminal");
        terminal
            .draw(|frame| view.render_inline(frame))
            .expect("should draw the diff");
        let start = view.view_state.zoom_index;
        let before = terminal.backend().buffer().clone();

        run_inline_diff_event_loop(
            &mut terminal,
            &mut view,
            &mut ScriptedEvents(VecDeque::from([KeyCode::Char('+'), KeyCode::Char('+')])),
        )
        .expect("should run the inline diff event loop");

        assert_eq!(
            view.view_state.zoom_index,
            start + 2,
            "each + zooms in one level"
        );
        assert_ne!(
            terminal.backend().buffer(),
            &before,
            "zooming in changes how the diff is drawn"
        );

        run_inline_diff_event_loop(
            &mut terminal,
            &mut view,
            &mut ScriptedEvents(VecDeque::from([KeyCode::Char('-')])),
        )
        .expect("should run the inline diff event loop");
        assert_eq!(view.view_state.zoom_index, start + 1);
    }

    #[test]
    fn test_inline_diff_names_what_the_colors_mean_in_the_help_lines() {
        let context = setup_gen();
        let mut view = InlineDiffView::new(
            context.graph().conn(),
            context.workspace(),
            &added_chain_diff(),
            &DiffLegend::between("unknown", "reference"),
            DiffAnnotations::none(),
        );
        let mut terminal =
            Terminal::new(TestBackend::new(70, 12)).expect("should create a test terminal");

        terminal
            .draw(|frame| view.render_inline(frame))
            .expect("should draw the diff");

        let buffer = terminal.backend().buffer();
        let row_text = |row: u16| -> String {
            (0..70)
                .map(|column| buffer[(column, row)].symbol())
                .collect()
        };
        let title = row_text(0);
        let legend = row_text(11);
        assert!(
            !title.contains("green"),
            "the frame title should not carry the legend, got {title:?}"
        );
        assert!(
            legend.contains("green = only in unknown")
                && legend.contains("red = only in reference"),
            "the last help line should explain the colors, got {legend:?}"
        );
    }

    #[test]
    fn test_inline_diff_keeps_the_terminal_background() {
        let context = setup_gen();
        let mut view = InlineDiffView::new(
            context.graph().conn(),
            context.workspace(),
            &added_chain_diff(),
            &DiffLegend::changes(),
            DiffAnnotations::none(),
        );
        let mut terminal =
            Terminal::new(TestBackend::new(40, 12)).expect("should create a test terminal");

        terminal
            .draw(|frame| view.render_inline(frame))
            .expect("should draw the diff");

        let theme_background = gen_tui::theme::current_theme()[0x00];
        assert!(
            terminal
                .backend()
                .buffer()
                .content()
                .iter()
                .all(|cell| cell.bg != theme_background),
            "the inline plot should not paint the theme background over the terminal's own"
        );
    }
}
