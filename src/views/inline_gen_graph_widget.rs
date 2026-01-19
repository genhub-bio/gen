use std::{
    io::Result,
    panic,
    time::{Duration, Instant},
};

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use gen_graph::GenGraph;
use gen_models::db::GraphConnection;
use gen_tui::{graph_controller::GraphController, layout::VisualDetail, theme::get_theme_color};
use log::info;
use ratatui::{
    TerminalOptions, Viewport,
    prelude::*,
    widgets::{Block, Borders},
};

use crate::views::gen_graph_widget::{
    GenGraphNodeRenderer, GenGraphNodeSizer, GenGraphPathHighlighter, create_gen_graph_widget,
};

/// Get the current block group ID for path highlighting
fn get_current_block_group_id(conn: &GraphConnection) -> Option<String> {
    // Try to get the most recent block group - this is a simplified approach
    // In a real implementation, you might want to track this differently
    conn.query_row(
        "SELECT name FROM block_groups ORDER BY rowid DESC LIMIT 1",
        [],
        |row| row.get(0),
    )
    .ok()
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
    renderer: GenGraphNodeRenderer<'a>,
}

impl<'a> InlineGenGraphState<'a> {
    pub fn new(graph: &'a GenGraph, conn: &'a GraphConnection) -> Self {
        let node_sizer = GenGraphNodeSizer;
        let mut graph_controller = GraphController::new(graph, node_sizer);
        graph_controller.set_detail_level(VisualDetail::Minimal);
        graph_controller.show_cursor();
        Self {
            controller: graph_controller,
            conn,
            renderer: GenGraphNodeRenderer::new(conn),
        }
    }
}

/// Display an inline GenGraph widget with interactive controls
///
/// This function creates an interactive inline terminal widget that displays a GenGraph
/// with full navigation and zoom controls. The widget appears inline in the terminal
/// without taking over the entire screen.
///
/// If the terminal doesn't support interactive features (like in some development environments),
/// it will fall back to a text-based representation.
///
/// # Controls
/// * Arrow keys: Navigate cursor between nodes and pan the view
/// * +/-: Zoom in/out (Minimal → Truncated → Full)
/// * Home: Reset view to origin
/// * q/Enter/Esc: Exit the widget (auto-exports DOT file if RUST_LOG is set)
///
/// # Arguments
/// * `graph` - The GenGraph to visualize
/// * `conn` - Database connection for sequence data
/// * `height` - Height of the inline viewport (in terminal rows, typically 10-20)
///
/// # Returns
/// * `Ok(())` if completed successfully
///
pub fn show_inline_gen_graph_widget(
    graph: &GenGraph,
    conn: &GraphConnection,
    height: u16,
) -> Result<()> {
    // Try to initialize the terminal - if it fails, fall back to text mode
    let terminal_result = panic::catch_unwind(|| {
        ratatui::init_with_options(TerminalOptions {
            viewport: Viewport::Inline(height),
        })
    });

    match terminal_result {
        Ok(terminal) => {
            // Interactive mode - full widget functionality
            show_interactive_widget(terminal, graph, conn, height)
        }
        Err(_) => {
            // Fallback mode - text-based representation
            eprintln!("Interactive terminal not available, falling back to text mode");
            show_text_fallback(graph, conn)
        }
    }
}

/// Interactive widget implementation with tick-based event loop
fn show_interactive_widget(
    mut terminal: ratatui::Terminal<ratatui::backend::CrosstermBackend<std::io::Stdout>>,
    graph: &GenGraph,
    conn: &GraphConnection,
    _height: u16,
) -> Result<()> {
    let mut state = InlineGenGraphState::new(graph, conn);

    // Set up tick-based event loop for 60 FPS (16ms per frame)
    let tick_rate = Duration::from_millis(16);
    let mut events = TickEventSource::new(tick_rate);
    let mut last_frame_time = Instant::now();

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
                        // Calculate the actual widget area first
                        let area = frame.area();
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
                        KeyCode::Esc | KeyCode::Char('q') => {
                            break;
                        }
                        KeyCode::Char('p') => {
                            // Toggle path highlighting
                            if let Some(block_group_id) = get_current_block_group_id(state.conn) {
                                let error_color =
                                    get_theme_color("error").unwrap_or(ratatui::style::Color::Red);
                                match state.renderer.toggle_path_highlight(
                                    &mut state.controller,
                                    &block_group_id,
                                    error_color,
                                ) {
                                    Ok(highlighting_enabled) => {
                                        info!(
                                            "Path highlighting: {}",
                                            if highlighting_enabled {
                                                "enabled"
                                            } else {
                                                "disabled"
                                            }
                                        );
                                    }
                                    Err(err) => {
                                        eprintln!("Failed to toggle path highlighting: {}", err);
                                    }
                                }
                            } else {
                                eprintln!("No block group available for path highlighting");
                            }
                        }
                        _ => {
                            let _ = state.controller.handle_key_event(key);
                        }
                    }
                }
                AppEvent::Resize(w, h) => {
                    // Update viewport bounds to match new terminal size
                    state.controller.viewport_state.viewport_bounds =
                        ratatui::layout::Rect::new(0, 0, w, h);
                }
            }
        }
    }

    // Final render without border
    terminal.draw(|frame| render_final(frame, &mut state))?;

    ratatui::restore();

    Ok(())
}

/// Core static plotting functionality for GenGraph
///
/// This function provides the core static text rendering of a GenGraph using TestBackend.
/// It can be called directly for --static flag or by fallback scenarios.
///
/// # Arguments
/// * `graph` - The GenGraph to visualize
/// * `conn` - Database connection for sequence data
/// * `show_comparison` - Whether to show both Minimal and Full scale renders
/// * `bounds` - Optional custom bounds (bottom_left, top_right), uses auto-calculated if None
///
/// # Returns
/// * `Ok(())` if rendering succeeded
pub fn plot_static(
    graph: &GenGraph,
    conn: &GraphConnection,
    detail_level: Option<VisualDetail>,
) -> Result<()> {
    use gen_tui::plotter::plot_graph_to_string;

    let node_sizer = GenGraphNodeSizer;
    let renderer = GenGraphNodeRenderer::new(conn);
    let mut controller = GraphController::new(graph, node_sizer);
    controller.set_detail_level(detail_level.unwrap_or(VisualDetail::Minimal));
    controller.show_cursor();

    let plot_string = plot_graph_to_string(&mut controller, renderer, detail_level, None, None);

    match plot_string {
        Ok((plot_string, width, _height)) => {
            println!("┌{}┐", "─".repeat(width as usize));
            for line in plot_string.lines() {
                println!("│{}│", line);
            }
            println!("└{}┘", "─".repeat(width as usize));
        }
        Err(e) => {
            eprintln!("Failed to render graph: {}", e);
            // Fallback to basic info
            println!("Graph nodes: {}", graph.node_count());
            println!("Graph edges: {}", graph.edge_count());
            return Err(std::io::Error::other(e));
        }
    }

    Ok(())
}

/// Text-based fallback for environments without interactive terminal support
/// Uses TestBackend to render graph and provide visual debugging output
fn show_text_fallback(graph: &GenGraph, conn: &GraphConnection) -> Result<()> {
    println!("GenGraph Text View (with TestBackend visualization)");
    println!("==================================================");

    plot_static(graph, conn, Some(VisualDetail::Full))?;

    println!("Try running this in a regular terminal for the full interactive experience.");

    Ok(())
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
    draw_gen_graph(frame, inner_area, state);
    draw_controls_help(frame, main_layout[1], state);
}

/// Draw the final plot after the widget is done
fn render_final(frame: &mut Frame, state: &mut InlineGenGraphState) {
    let area = frame.area();

    // Final layout - split main area for graph box and potential future final message
    let main_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(1)])
        .split(area);

    // Add margin to offset content to where it was with border (just for the graph area)
    let padded_area = Rect {
        x: main_layout[0].x + 1,                         // Offset by border width
        y: main_layout[0].y + 1,                         // Offset by border height
        width: main_layout[0].width.saturating_sub(2),   // Reduce by both sides
        height: main_layout[0].height.saturating_sub(2), // Reduce by top and bottom
    };

    // For final render, we can now render the actual graph
    draw_gen_graph(frame, padded_area, state);
    // draw_final_message(frame, main_layout[1], state); // Disabled - scaffolding preserved for future use
}

fn draw_gen_graph(frame: &mut Frame, area: Rect, state: &mut InlineGenGraphState) {
    // Set viewport bounds and focus for the current area
    state.controller.viewport_state.viewport_bounds = area;
    state.controller.viewport_state.focus();

    // Create the GenGraph widget with current level of detail
    let detail_level = state.controller.get_detail_level();
    let widget = create_gen_graph_widget(state.conn)
        .detail_level(detail_level)
        .cursor();

    // Render the graph widget
    frame.render_stateful_widget(widget, area, &mut state.controller);
}

fn draw_controls_help(frame: &mut Frame, area: Rect, state: &mut InlineGenGraphState) {
    // Check if path highlighting is currently enabled
    let _path_status = if state.controller.get_path_highlights().is_empty() {
        "Path: Off"
    } else {
        "Path: On"
    };

    let help_text = if state.controller.get_path_highlights().is_empty() {
        "←→↑↓: Navigate/Pan | +/-: Zoom | Path: Off | q/Esc: Exit".to_string()
    } else {
        "←→↑↓: Navigate/Pan | +/-: Zoom | Path: On | q/Esc: Exit".to_string()
    };

    let paragraph =
        ratatui::widgets::Paragraph::new(help_text).style(Style::default().fg(Color::Yellow));
    frame.render_widget(paragraph, area);
}

#[allow(dead_code)]
fn draw_final_message(frame: &mut Frame, area: Rect, _state: &mut InlineGenGraphState) {
    let final_text = "GenGraph viewing completed".to_string();
    let paragraph =
        ratatui::widgets::Paragraph::new(final_text).style(Style::default().fg(Color::Green));
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
        assert_eq!(state.controller.get_detail_level(), VisualDetail::Minimal);
    }
}
