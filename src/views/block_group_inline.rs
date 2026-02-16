use std::{
    io::{Error, Result},
    panic,
    time::{Duration, Instant},
};

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use gen_graph::GenGraph;
use gen_models::{db::GraphConnection, path::Path};
use gen_tui::{
    graph_controller::GraphController,
    layout::VisualDetail,
    plotter::{LineStyle, PathStyle},
    theme::Theme,
};
use ratatui::{
    TerminalOptions, Viewport,
    prelude::*,
    style::Color,
    widgets::{Block, Borders},
};

use crate::{
    config::get_theme_color,
    views::gen_graph_widget::{GenGraphNodeSizer, create_gen_graph_widget},
};

/// Get path nodes for a path and map it to GraphNodes in the current graph
fn get_path_nodes(
    conn: &GraphConnection,
    path: &Path,
    graph: &GenGraph,
) -> std::io::Result<Vec<gen_graph::GraphNode>> {
    use gen_core::{PATH_END_NODE_ID, PATH_START_NODE_ID};
    use gen_graph::project_path;

    // Get the path blocks from the database
    let path_blocks = path.blocks(conn);

    // Project the path blocks onto the current graph state
    let projected_path = project_path(graph, &path_blocks);

    // Filter out terminal nodes (start and end) and convert to GraphNodes
    let path_nodes: Vec<gen_graph::GraphNode> = projected_path
        .iter()
        .filter_map(|(node, _)| {
            // Filter out terminal nodes
            if node.node_id != PATH_START_NODE_ID && node.node_id != PATH_END_NODE_ID {
                Some(*node)
            } else {
                None
            }
        })
        .collect();

    if path_nodes.is_empty() {
        return Err(Error::other(
            "Path nodes not found in current graph state".to_string(),
        ));
    }

    Ok(path_nodes)
}

#[derive(Debug)]
pub enum AppEvent {
    Tick,
    KeyPress(crossterm::event::KeyEvent),
    Resize(u16, u16),
}

pub trait EventSource {
    fn poll_next(&mut self, timeout: Duration) -> Option<AppEvent>;
}

pub struct TickEventSource {
    tick_rate: Duration,
    last_tick: Instant,
}

impl TickEventSource {
    pub fn new(tick_rate: Duration) -> Self {
        Self {
            tick_rate,
            last_tick: Instant::now(),
        }
    }
}

impl EventSource for TickEventSource {
    fn poll_next(&mut self, timeout: Duration) -> Option<AppEvent> {
        let remaining = self
            .tick_rate
            .checked_sub(self.last_tick.elapsed())
            .unwrap_or(Duration::ZERO);

        let wait = remaining.min(timeout);

        if event::poll(wait).unwrap_or(false) {
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

        if self.last_tick.elapsed() >= self.tick_rate {
            self.last_tick = Instant::now();
            return Some(AppEvent::Tick);
        }

        None
    }
}

pub struct InlineGenGraphState<'a> {
    controller: GraphController<&'a GenGraph, GenGraphNodeSizer>,
    conn: &'a GraphConnection,
    paths: Vec<Vec<gen_graph::GraphNode>>,
}

impl<'a> InlineGenGraphState<'a> {
    pub fn new(graph: &'a GenGraph, conn: &'a GraphConnection) -> Self {
        let node_sizer = GenGraphNodeSizer;
        let mut graph_controller = GraphController::new(graph, node_sizer).with_theme(Theme {
            canvas: Color::Reset,
            node_fg: get_theme_color("text").unwrap(),
            node_bg: get_theme_color("node").unwrap(),
            edge_fg: get_theme_color("edge").unwrap(),
            edge_bg: Color::Reset,
            cursor_fg: get_theme_color("cursor_fg").unwrap(),
            cursor_bg: get_theme_color("cursor_bg").unwrap(),
            highlight: get_theme_color("cursor_highlight").unwrap(),
        });
        graph_controller.set_detail_level(VisualDetail::Truncated);
        graph_controller.show_cursor();
        let paths = Vec::new();
        Self {
            controller: graph_controller,
            conn,
            paths,
        }
    }

    /// Add a path to the widget, starting from a Path object
    pub fn add_path(&mut self, path: &Path, conn: &'a GraphConnection) -> Result<()> {
        let path_nodes = get_path_nodes(conn, path, self.controller.graph)?;
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
            // Set up tick-based event loop for 60 FPS (16ms per frame)
            let tick_rate = Duration::from_millis(16);
            let mut events = TickEventSource::new(tick_rate);
            let mut last_frame_time = Instant::now();
            let mut upgrade_requested = false;

            loop {
                // Process events with a reasonable timeout
                if let Some(event) = events.poll_next(Duration::from_millis(250)) {
                    match event {
                        AppEvent::Tick => {
                            // Calculate time delta since last frame
                            let now = Instant::now();
                            let frame_delta = now.duration_since(last_frame_time);
                            last_frame_time = now;

                            // Draw the frame
                            terminal.draw(|frame| {
                                let area = frame.area();
                                // Calculate the actual widget area first
                                let main_layout = Layout::default()
                                    .direction(Direction::Vertical)
                                    .constraints([Constraint::Min(0), Constraint::Length(1)])
                                    .split(area);
                                let block = Block::default().borders(Borders::ALL);
                                let inner_area = block.inner(main_layout[0]);

                                // Set viewport bounds to the actual inner area before updating animations
                                state.controller.viewport_state.viewport_bounds = inner_area;

                                // Update animations with frame delta for smooth camera and cursor animations
                                state.controller.update_animations(frame_delta);

                                render_inline(frame, &mut state);
                            })?;
                        }
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
                                    let path_style = PathStyle::new(
                                        get_theme_color("base09")
                                            .expect("Theme should use base16 system"),
                                    )
                                    .with_line_style(LineStyle::Bold)
                                    .with_merge_glyphs(true);

                                    if state.controller.has_highlight(&path_style) {
                                        state.controller.clear_highlight(&path_style);
                                    } else if let Some(last_path) = state.paths.last() {
                                        state
                                            .controller
                                            .set_path_highlight(path_style, last_path.clone());
                                    } else {
                                        eprintln!("No paths available for path highlighting");
                                    }
                                }
                                _ => {
                                    let _ = state.controller.handle_key_event(key);
                                }
                            }
                        }
                        AppEvent::Resize(w, _h) => {
                            // Update viewport width, but keep the fixed inline height
                            state.controller.viewport_state.viewport_bounds.width = w;
                        }
                    }
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

    // Set viewport bounds and focus for the current area
    state.controller.viewport_state.viewport_bounds = inner_area;
    state.controller.viewport_state.focus();

    // Create the GenGraph widget with current level of detail
    let detail_level = state.controller.get_detail_level();
    let widget = create_gen_graph_widget(state.conn)
        .detail_level(detail_level)
        .cursor();

    // Render the graph widget
    frame.render_stateful_widget(widget, inner_area, &mut state.controller);
    draw_controls_help(frame, main_layout[1], state);
}

/// Draw the final plot after the widget is done
fn render_final(frame: &mut Frame, state: &mut InlineGenGraphState) {
    let area = frame.area();
    // Set viewport bounds and focus for the current area
    state.controller.viewport_state.viewport_bounds =
        area.offset(ratatui::layout::Offset { x: 0, y: -1 });
    state.controller.viewport_state.blur();

    // Create the GenGraph widget with current level of detail
    let detail_level = state.controller.get_detail_level();
    let widget = create_gen_graph_widget(state.conn).detail_level(detail_level);

    // Render the graph widget
    frame.render_stateful_widget(
        widget,
        area.offset(ratatui::layout::Offset { x: 0, y: -1 }),
        &mut state.controller,
    );
}

fn draw_controls_help(frame: &mut Frame, area: Rect, state: &mut InlineGenGraphState) {
    let help_text = if state.controller.highlights.is_empty() {
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
            block_id: 1,
            node_id: HashId::pad_str(1),
            sequence_start: 0,
            sequence_end: 10,
        };
        graph.add_node(node);

        let state = InlineGenGraphState::new(&graph, &conn);
        assert_eq!(state.controller.get_detail_level(), VisualDetail::Truncated);
    }
}
