use std::{
    collections::HashSet,
    io::{Error, Result},
    panic,
    time::Duration,
};

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use gen_core::HashId;
use gen_graph::{GenGraph, GraphNode};
use gen_models::{block_group::BlockGroup, db::GraphConnection, path::Path};
use gen_tui::{
    graph_view::{GraphView, GraphViewState},
    layout::VisualDetail,
    layout_engine::{LayoutEngine, WorldKey},
    plotter::{LineStyle, PathStyle},
    theme::current_theme,
};
use ratatui::{
    TerminalOptions, Viewport,
    prelude::*,
    widgets::{Block, Borders},
};

use crate::views::{
    annotation_groups::load_annotation_group_entries,
    annotations::{AnnotationGroupTrackRequest, load_annotations_for_group},
    block_group::active_neighborhood_node_ids,
    gen_graph_widget::{
        self, DEFAULT_ZOOM_LEVEL, ZoomLevels, build_zoom_levels, draw_annotation_labels,
        reapply_overlays,
    },
    graph_overlay::{
        AnnotationColorCache, GraphOverlay, group_track_key, has_path_overlay, remove_path_overlay,
        replace_track_overlays, set_path_overlay,
    },
};

/// Get path nodes for a path and map it to GraphNodes in the current graph
fn get_path_nodes(
    conn: &GraphConnection,
    path: &Path,
    graph: &GenGraph,
) -> std::io::Result<Vec<gen_graph::GraphNode>> {
    crate::views::helpers::project_path_nodes(conn, path, graph).map_err(Error::other)
}

#[derive(Debug)]
pub enum AppEvent {
    KeyPress(crossterm::event::KeyEvent),
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
    engine: LayoutEngine<GenGraph>,
    zoom_levels: ZoomLevels<'a>,
    view_state: GraphViewState<GraphNode>,
    paths: Vec<Vec<gen_graph::GraphNode>>,
    conn: &'a GraphConnection,
    block_group_id: Option<HashId>,
    history_ref: Option<&'a str>,
    /// Annotation and path overlays currently loaded, ready for highlight + label rendering.
    overlays: Vec<GraphOverlay>,
    annotation_colors: AnnotationColorCache,
    annotation_groups_loaded: bool,
    /// The active neighborhood the annotation groups were last loaded for - the crawled
    /// neighborhood is already the deliberately-constrained local window, so a reload is
    /// only needed when it changes (not on every pan/zoom within the same neighborhood).
    annotation_groups_world: Option<WorldKey<GraphNode>>,
}

impl<'a> InlineGenGraphState<'a> {
    pub fn new(
        graph: &GenGraph,
        conn: &'a GraphConnection,
        block_group_id: Option<HashId>,
        history_ref: Option<&'a str>,
    ) -> Self {
        let zoom_levels = build_zoom_levels(conn);
        let engine = LayoutEngine::new(graph.clone());
        let mut view_state = GraphViewState::default();
        gen_graph_widget::apply_zoom_level(&mut view_state, DEFAULT_ZOOM_LEVEL, &zoom_levels);
        view_state.show_cursor();
        let paths = Vec::new();
        Self {
            engine,
            zoom_levels,
            view_state,
            paths,
            conn,
            block_group_id,
            history_ref,
            overlays: Vec::new(),
            annotation_colors: AnnotationColorCache::new(),
            annotation_groups_loaded: false,
            annotation_groups_world: None,
        }
    }

    /// Add a path to the widget, starting from a Path object
    pub fn add_path(&mut self, path: &Path, conn: &'a GraphConnection) -> Result<()> {
        let path_nodes = get_path_nodes(conn, path, self.engine.graph())?;
        self.paths.push(path_nodes);
        Ok(())
    }

    fn load_annotation_groups(&mut self, node_ids: &HashSet<HashId>) {
        let (Some(block_group_id), conn) = (self.block_group_id, self.conn) else {
            return;
        };
        let Ok(block_group) = BlockGroup::get_by_id(conn, &block_group_id, self.history_ref) else {
            return;
        };
        // Drop the annotation overlays but keep the path overlay across viewport reloads.
        self.overlays
            .retain(|overlay| overlay.path_nodes().is_some());
        for entry in load_annotation_group_entries(conn, &block_group, self.history_ref) {
            let Ok(entry_spans) = load_annotations_for_group(&AnnotationGroupTrackRequest {
                conn,
                history_ref: self.history_ref,
                current_block_group: &block_group,
                entry: &entry,
                node_ids,
            }) else {
                continue;
            };
            replace_track_overlays(&mut self.overlays, &group_track_key(&entry.id), entry_spans);
        }
    }
}

/// Reload annotation groups for the active crawled neighborhood if it has changed since
/// the last load (or nothing has been loaded yet). Returns whether a reload happened, so
/// the caller knows to redraw immediately rather than waiting for the next input event.
fn maybe_reload_annotation_groups(state: &mut InlineGenGraphState) -> bool {
    let current_world = state.engine.active_world_key();
    if current_world != state.annotation_groups_world {
        state.annotation_groups_loaded = false;
    }
    if state.annotation_groups_loaded {
        return false;
    }
    let node_ids = active_neighborhood_node_ids(&state.engine);
    if node_ids.is_empty() {
        return false;
    }
    state.load_annotation_groups(&node_ids);
    state.annotation_groups_loaded = true;
    state.annotation_groups_world = current_world;
    true
}

/// Display an inline GenGraph widget with interactive controls
///
/// This function creates an interactive inline terminal widget that displays a GenGraph
/// with full navigation and zoom controls. The widget appears inline in the terminal
/// without taking over the entire screen.
///
/// # Controls
/// * Arrow keys: Navigate cursor between nodes and pan the view
/// * +/-: Zoom in/out (Minimal → Truncated → Full)
/// * q/Enter/Esc: Exit the widget (auto-exports DOT file if RUST_LOG is set)
///
/// # Arguments
/// * `graph` - The GenGraph to visualize
/// * `conn` - Database connection for sequence data
/// * `paths` - Paths to highlight when asked to
/// * `height` - Height of the inline viewport (in terminal rows, typically 10-20)
///
/// # Returns
/// * `Ok(true)` if the user requested to transition to full-screen view
/// * `Ok(false)` if completed successfully and exited
///
pub fn show_inline_gen_graph_widget(
    conn: &GraphConnection,
    graph: &GenGraph,
    paths: Vec<Path>,
    height: u16,
) -> Result<bool> {
    show_inline_widget(conn, graph, paths, height, None, None)
}

/// Display an inline widget for a `BlockGroup`'s graph, with annotations loaded.
///
/// See [`show_inline_gen_graph_widget`] for controls and return value.
pub fn show_inline_block_group_widget(
    conn: &GraphConnection,
    block_group_id: HashId,
    paths: Vec<Path>,
    height: u16,
    history_ref: Option<&str>,
) -> Result<bool> {
    let graph = BlockGroup::get_graph(conn, &block_group_id, history_ref).map_err(Error::other)?;
    show_inline_widget(
        conn,
        &graph,
        paths,
        height,
        Some(block_group_id),
        history_ref,
    )
}

fn show_inline_widget(
    conn: &GraphConnection,
    graph: &GenGraph,
    paths: Vec<Path>,
    height: u16,
    block_group_id: Option<HashId>,
    history_ref: Option<&str>,
) -> Result<bool> {
    let terminal_result = panic::catch_unwind(|| {
        ratatui::init_with_options(TerminalOptions {
            viewport: Viewport::Inline(height),
        })
    });

    match terminal_result {
        Ok(mut terminal) => {
            let mut state = InlineGenGraphState::new(graph, conn, block_group_id, history_ref);
            for path in paths {
                state.add_path(&path, conn)?;
            }
            let mut events = CrosstermEventSource;
            let mut upgrade_requested = false;

            terminal.draw(|frame| {
                render_inline(frame, &mut state);
            })?;
            // After the first draw the viewport is populated. Load (or reload) annotation
            // groups using the viewport node IDs, and redraw immediately if anything loaded
            // rather than waiting for the next input event.
            if maybe_reload_annotation_groups(&mut state) {
                terminal.draw(|frame| {
                    render_inline(frame, &mut state);
                })?;
            }

            loop {
                // Rendering is event-driven (no animation to advance): block indefinitely
                // until the next input event wakes us.
                let Some(event) = events.poll_next(Duration::from_secs(3600)) else {
                    continue;
                };

                match event {
                    AppEvent::KeyPress(key) => {
                        // Intercept quit signal and path highlighting
                        match key.code {
                            KeyCode::Esc | KeyCode::Char('q') | KeyCode::Enter => {
                                break;
                            }
                            KeyCode::Char('f') => {
                                upgrade_requested = true;
                                break;
                            }
                            KeyCode::Char('p') => {
                                // Toggle the path overlay; reapply_overlays repaints it.
                                if has_path_overlay(&state.overlays) {
                                    remove_path_overlay(&mut state.overlays);
                                } else if let Some(last_path) = state.paths.last().cloned() {
                                    let path_style = PathStyle::new(current_theme()[0x09])
                                        .with_line_style(LineStyle::Bold)
                                        .with_merge_glyphs(true);
                                    set_path_overlay(&mut state.overlays, path_style, last_path);
                                } else {
                                    eprintln!("No paths available for path highlighting");
                                }
                            }
                            KeyCode::Char('+') | KeyCode::Char('=') => {
                                gen_graph_widget::zoom_in(
                                    &mut state.view_state,
                                    &state.zoom_levels,
                                );
                            }
                            KeyCode::Char('-') => {
                                gen_graph_widget::zoom_out(
                                    &mut state.view_state,
                                    &state.zoom_levels,
                                );
                            }
                            _ => {
                                let _ = state.view_state.handle_key_event(key);
                            }
                        }
                    }
                    AppEvent::Resize(_w, _h) => {
                        // Width is picked up automatically from `frame.area()` on the
                        // next render; the inline viewport's height stays fixed.
                    }
                }

                terminal.draw(|frame| {
                    render_inline(frame, &mut state);
                })?;
                // Camera-moving actions (pan, zoom, navigation) can bring new nodes into the
                // viewport; reload annotation groups and redraw immediately when that happens.
                if maybe_reload_annotation_groups(&mut state) {
                    terminal.draw(|frame| {
                        render_inline(frame, &mut state);
                    })?;
                }
            }

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

    // Re-register overlay highlights before rendering, in case the overlay set or zoom
    // level changed since the last frame.
    reapply_overlays(
        &state.engine,
        &mut state.view_state,
        &state.zoom_levels,
        &mut state.overlays,
        &mut state.annotation_colors,
    );

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
    use gen_core::HashId;
    use petgraph::graphmap::DiGraphMap;

    use super::*;
    use crate::{graph::GraphNode, test_helpers::get_connection};

    #[test]
    fn test_inline_state_creation() {
        let conn = get_connection(None).expect("Failed to get test database connection");
        let mut graph = DiGraphMap::new();

        // Add a simple test node
        let node = GraphNode {
            node_id: HashId::pad_str(1),
            sequence_start: 0,
            sequence_end: 10,
        };
        graph.add_node(node);

        let state = InlineGenGraphState::new(&graph, &conn, None, None);
        assert_eq!(state.view_state.zoom_index, DEFAULT_ZOOM_LEVEL);
    }
}
