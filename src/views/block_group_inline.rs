use std::{
    collections::HashSet,
    io::{Error, Result},
    panic,
    time::{Duration, Instant},
};

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use gen_core::HashId;
use gen_graph::GenGraph;
use gen_models::{block_group::BlockGroup, db::GraphConnection, path::Path};
use gen_tui::{
    graph_controller::GraphController,
    layout::VisualDetail,
    plotter::{LineStyle, PathStyle},
    theme::current_theme,
};
use ratatui::{
    TerminalOptions, Viewport,
    prelude::*,
    style::Color,
    widgets::{Block, Borders},
};

use crate::views::{
    annotation_groups::load_annotation_group_entries,
    annotations::{AnnotationGroupTrackRequest, load_annotations_for_group},
    block_group::extract_viewport_node_ids,
    gen_graph_widget::{
        GenGraphNodeSizer, create_gen_graph_widget, draw_annotation_labels, reapply_overlays,
    },
    graph_overlay::{
        GraphOverlay, group_track_key, has_path_overlay, remove_path_overlay,
        replace_track_overlays, set_path_overlay,
    },
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
    let path_blocks = path
        .blocks(conn)
        .map_err(|err| Error::other(format!("Failed to load path blocks: {err}")))?;

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
    controller: GraphController<GenGraph, GenGraphNodeSizer>,
    conn: &'a GraphConnection,
    paths: Vec<Vec<gen_graph::GraphNode>>,
    block_group_id: Option<HashId>,
    /// Annotation and path overlays currently loaded, ready for highlight + label rendering.
    overlays: Vec<GraphOverlay>,
    annotation_groups_loaded: bool,
}

impl<'a> InlineGenGraphState<'a> {
    pub fn new(
        graph: &GenGraph,
        conn: &'a GraphConnection,
        block_group_id: Option<HashId>,
    ) -> Self {
        let node_sizer = GenGraphNodeSizer;
        let mut graph_controller = GraphController::new(graph.clone(), node_sizer);
        graph_controller.set_detail_level(VisualDetail::Truncated);
        graph_controller.show_cursor();
        Self {
            controller: graph_controller,
            conn,
            paths: Vec::new(),
            block_group_id,
            overlays: Vec::new(),
            annotation_groups_loaded: false,
        }
    }

    /// Add a path to the widget, starting from a Path object
    pub fn add_path(&mut self, path: &Path, conn: &'a GraphConnection) -> Result<()> {
        let path_nodes = get_path_nodes(conn, path, self.controller.graph())?;
        self.paths.push(path_nodes);
        Ok(())
    }

    fn load_annotation_groups(&mut self, node_ids: &HashSet<HashId>) {
        let (Some(block_group_id), conn) = (self.block_group_id, self.conn) else {
            return;
        };
        let Ok(block_group) = BlockGroup::get_by_id(conn, &block_group_id) else {
            return;
        };
        // Drop the annotation overlays but keep the path overlay across viewport reloads.
        self.overlays
            .retain(|overlay| overlay.path_nodes().is_some());
        for entry in load_annotation_group_entries(conn, &block_group) {
            let Ok(entry_spans) = load_annotations_for_group(&AnnotationGroupTrackRequest {
                conn,
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

/// Display an inline widget for a generic GenGraph with interactive controls
///
/// This function creates an interactive inline terminal widget that displays a GenGraph
/// with full navigation and zoom controls. The widget appears inline in the terminal
/// without taking over the entire screen.
///
/// Use [`show_inline_block_group_widget`] instead when the graph belongs to a `BlockGroup`,
/// so that annotations can be loaded.
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
    show_inline_widget(conn, graph, paths, height, None)
}

/// Display an inline widget for a `BlockGroup`'s graph, with annotations loaded.
///
/// See [`show_inline_gen_graph_widget`] for controls and return value.
pub fn show_inline_block_group_widget(
    conn: &GraphConnection,
    block_group_id: HashId,
    paths: Vec<Path>,
    height: u16,
) -> Result<bool> {
    let graph = BlockGroup::get_graph(conn, &block_group_id).map_err(Error::other)?;
    show_inline_widget(conn, &graph, paths, height, Some(block_group_id))
}

fn show_inline_widget(
    conn: &GraphConnection,
    graph: &GenGraph,
    paths: Vec<Path>,
    height: u16,
    block_group_id: Option<HashId>,
) -> Result<bool> {
    let terminal_result = panic::catch_unwind(|| {
        ratatui::init_with_options(TerminalOptions {
            viewport: Viewport::Inline(height),
        })
    });

    match terminal_result {
        Ok(mut terminal) => {
            let mut state = InlineGenGraphState::new(graph, conn, block_group_id);
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

                            // Re-apply stored overlays before drawing.
                            reapply_overlays(&mut state.controller, &state.overlays);

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

                            // After the first draw the viewport is populated. Load (or reload)
                            // annotation groups using the viewport node IDs. On subsequent frames,
                            // invalidate when the camera has moved far enough to change the node set.
                            if state.controller.detect_motion() {
                                state.annotation_groups_loaded = false;
                            }
                            if !state.annotation_groups_loaded {
                                let node_ids = extract_viewport_node_ids(&state.controller);
                                if !node_ids.is_empty() {
                                    state.load_annotation_groups(&node_ids);
                                    state.annotation_groups_loaded = true;
                                }
                            }
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
                                    // Toggle the path overlay; reapply_highlights repaints it.
                                    if has_path_overlay(&state.overlays) {
                                        remove_path_overlay(&mut state.overlays);
                                    } else if let Some(nodes) = state.paths.last().cloned() {
                                        let style = PathStyle::new(current_theme()[0x09])
                                            .with_line_style(LineStyle::Bold)
                                            .with_merge_glyphs(true);
                                        set_path_overlay(&mut state.overlays, style, nodes);
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

    // Draw floating annotation labels after the graph.
    let any_annotations_hidden = draw_annotation_labels(
        frame.buffer_mut(),
        inner_area,
        &state.controller,
        &state.overlays,
    );

    let hidden_legend = any_annotations_hidden.then(|| {
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

        let state = InlineGenGraphState::new(&graph, &conn, None);
        assert_eq!(state.controller.get_detail_level(), VisualDetail::Truncated);
    }
}
