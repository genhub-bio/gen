use std::{
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
    layout_engine::LayoutEngine,
    plotter::{LineStyle, PathStyle},
    theme::current_theme,
};
use ratatui::{
    TerminalOptions, Viewport,
    prelude::*,
    widgets::{Block, Borders},
};

use crate::views::gen_graph_widget::{self, DEFAULT_ZOOM_LEVEL, ZoomLevels, build_zoom_levels};

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
}

impl<'a> InlineGenGraphState<'a> {
    pub fn new(graph: &GenGraph, conn: &'a GraphConnection) -> Self {
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
        }
    }

    /// Add a path to the widget, starting from a Path object
    pub fn add_path(&mut self, path: &Path, conn: &'a GraphConnection) -> Result<()> {
        let path_nodes = get_path_nodes(conn, path, self.engine.graph())?;
        self.paths.push(path_nodes);
        Ok(())
    }
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
    show_inline_widget(conn, graph, paths, height)
}

/// Display an inline widget for a `BlockGroup`'s graph.
///
/// TODO(annotation-port): pre-unvendor-sugiyama this loaded annotation-group overlays via
/// `load_annotation_groups`/`reapply_overlays`/`draw_annotation_labels` (main's #203 inline
/// highlight rendering). That machinery was built on the deleted `GraphController`/`NodeSizer`
/// and needs porting onto `LayoutEngine`/`GraphViewState` before it can be restored - for now
/// this only renders the graph, with no annotation overlays.
pub fn show_inline_block_group_widget(
    conn: &GraphConnection,
    block_group_id: HashId,
    paths: Vec<Path>,
    height: u16,
    history_ref: Option<&str>,
) -> Result<bool> {
    let graph = BlockGroup::get_graph(conn, &block_group_id, history_ref).map_err(Error::other)?;
    show_inline_widget(conn, &graph, paths, height)
}

fn show_inline_widget(
    conn: &GraphConnection,
    graph: &GenGraph,
    paths: Vec<Path>,
    height: u16,
) -> Result<bool> {
    let terminal_result = panic::catch_unwind(|| {
        ratatui::init_with_options(TerminalOptions {
            viewport: Viewport::Inline(height),
        })
    });

    match terminal_result {
        Ok(mut terminal) => {
            let mut state = InlineGenGraphState::new(graph, conn);
            for path in paths {
                state.add_path(&path, conn)?;
            }
            let mut events = CrosstermEventSource;
            let mut upgrade_requested = false;

            terminal.draw(|frame| {
                render_inline(frame, &mut state);
            })?;

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
                                // Toggle path highlighting
                                let path_style = PathStyle::new(current_theme()[0x09])
                                    .with_line_style(LineStyle::Bold)
                                    .with_merge_glyphs(true);

                                if state.view_state.has_highlight(&path_style) {
                                    state.view_state.clear_highlight(&path_style);
                                } else if let Some(last_path) = state.paths.last() {
                                    state
                                        .view_state
                                        .set_path_highlight(path_style, last_path.clone());
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

    // Create the GenGraph view
    let active_renderer = &state.zoom_levels[state.view_state.zoom_index].1;
    let view = GraphView::new(&mut state.engine, active_renderer);

    // Render the graph view
    frame.render_stateful_widget(view, inner_area, &mut state.view_state);
    draw_controls_help(frame, main_layout[1], state);
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

fn draw_controls_help(frame: &mut Frame, area: Rect, state: &mut InlineGenGraphState) {
    let help_text = if state.view_state.highlights.styles.is_empty() {
        "←→↑↓: Nav | +/-: Zoom | f: Full window | p: Show Path | q: Exit".to_string()
    } else {
        "←→↑↓: Nav | +/-: Zoom | f: Full window | p: Hide Path | q: Exit".to_string()
    };

    let paragraph =
        ratatui::widgets::Paragraph::new(help_text).style(Style::default().fg(Color::Yellow));

    frame.render_widget(paragraph, area);
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

        let state = InlineGenGraphState::new(&graph, &conn);
        assert_eq!(state.view_state.zoom_index, DEFAULT_ZOOM_LEVEL);
    }
}
