use crate::config::col;
use crate::graph::{project_path, GenGraph, GraphNode};
use crate::models::node::Node;
use crate::models::path::Path;
use crate::models::sequence::Sequence;
use crate::views::block_layout::{BaseLayout, ScaledLayout};

use crossterm::event::{KeyCode, KeyEvent};
use log::info;
use log::warn;
use petgraph::graphmap::DiGraphMap;
use petgraph::Direction;
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
    text::Span,
    widgets::canvas::{Canvas, Line, Points},
    widgets::{Block, Widget},
};
use rusqlite::Connection;
use std::collections::{HashMap, HashSet};

/// Labels used in the graph visualization (selected, not-selected)
/// the trick is to get them to align with the braille characters
/// we use to draw lines:
/// ⣿●⣿*⣿∘⣿⦿⣿○⣿◉⣿⏣⣿⏸⣿⏹⣿⏺⣿⏼⣿⎔⣿
pub mod label {
    pub const START: &str = "Start >";
    pub const END: &str = "> End";
    pub const NODE: &str = "⏺";
    pub const SELECTED: &str = "█";
}

/// Used for scrolling through the graph.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum NavDirection {
    Left,
    Right,
    Up,
    Down,
}

/// Defines the discrete zoom levels available in the viewer
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZoomLevel {
    // Scale-based zoom levels for both extremes
    ScaleOut(u32), // Scale-based zooming with smaller scale (zooming out)
    // Label-based zoom levels from lowest to highest detail
    Minimal, // label_width = 0 (no sequence shown)
    Low,     // label_width = 5
    Default, // label_width = 11 (default)
    High,    // label_width = 100
    Full,    // label_width = u32::MAX
    // Scale-based zoom levels at the maximum label detail
    ScaleIn(u32), // Scale-based zooming with larger scale (zooming in)
}

/// Holds parameters that don't change when you scroll.
/// - `label_width` = how many characters to show at most in each block label. If 0, labels are not shown.
/// - `scale` = data units per 1 terminal cell.  
///   - If `scale` = 1.0, each cell is 1 data unit.  
///   - If `scale` = 2.0, each cell is 2 data units (you see *more* data).  
///   - If `scale` = 0.5, each cell is 0.5 data units (you see *less* data, zoomed in).
/// - `aspect_ratio` = width / height of a terminal cell in data units.
/// - `edge_style` = draw the edges as straight lines or as splines.
pub struct PlotParameters {
    pub label_width: u32,
    pub scale: u32,
    pub aspect_ratio: f32,
    pub edge_style: EdgeStyle,
}

impl Default for PlotParameters {
    fn default() -> Self {
        PlotParameters {
            label_width: 11,
            scale: 4,
            aspect_ratio: 0.5,
            edge_style: EdgeStyle::Straight,
        }
    }
}

impl PlotParameters {
    /// Get the current zoom level based on label_width and scale settings
    pub fn get_zoom_level(&self) -> ZoomLevel {
        match self.label_width {
            0 => {
                if self.scale < PlotParameters::default().scale {
                    ZoomLevel::ScaleOut(self.scale)
                } else {
                    ZoomLevel::Minimal
                }
            }
            5 => ZoomLevel::Low,
            w if w == PlotParameters::default().label_width => ZoomLevel::Default,
            100 => ZoomLevel::High,
            u32::MAX => {
                if self.scale > PlotParameters::default().scale {
                    ZoomLevel::ScaleIn(self.scale)
                } else {
                    ZoomLevel::Full
                }
            }
            _ => ZoomLevel::Default, // Fallback
        }
    }

    /// Set parameters according to the specified zoom level
    pub fn set_zoom_level(&mut self, level: ZoomLevel) {
        // Set label width and scale based on zoom level
        match level {
            ZoomLevel::ScaleOut(scale) => {
                self.label_width = 0;
                self.scale = scale;
            }
            ZoomLevel::Minimal => {
                self.label_width = 0;
                self.scale = PlotParameters::default().scale;
            }
            ZoomLevel::Low => {
                self.label_width = 5;
                self.scale = PlotParameters::default().scale;
            }
            ZoomLevel::Default => {
                self.label_width = PlotParameters::default().label_width;
                self.scale = PlotParameters::default().scale;
            }
            ZoomLevel::High => {
                self.label_width = 100;
                self.scale = PlotParameters::default().scale;
            }
            ZoomLevel::Full => {
                self.label_width = u32::MAX;
                self.scale = PlotParameters::default().scale;
            }
            ZoomLevel::ScaleIn(scale) => {
                self.label_width = u32::MAX;
                self.scale = scale;
            }
        }
    }

    /// Get the next zoom level when zooming in
    pub fn zoom_in_level(&self) -> ZoomLevel {
        match self.get_zoom_level() {
            ZoomLevel::ScaleOut(scale) => {
                if scale + 2 >= PlotParameters::default().scale {
                    ZoomLevel::Minimal // Switch back to label-based zooming
                } else {
                    ZoomLevel::ScaleOut(scale + 2) // Continue scale-based zooming
                }
            }
            ZoomLevel::Minimal => ZoomLevel::Low,
            ZoomLevel::Low => ZoomLevel::Default,
            ZoomLevel::Default => ZoomLevel::High,
            ZoomLevel::High => ZoomLevel::Full,
            ZoomLevel::Full => ZoomLevel::ScaleIn(PlotParameters::default().scale + 2),
            ZoomLevel::ScaleIn(scale) => ZoomLevel::ScaleIn(scale + 2),
        }
    }

    /// Get the next zoom level when zooming out
    pub fn zoom_out_level(&self) -> ZoomLevel {
        match self.get_zoom_level() {
            ZoomLevel::ScaleOut(scale) => {
                if scale <= 2 {
                    ZoomLevel::ScaleOut(2) // Minimum scale
                } else {
                    ZoomLevel::ScaleOut(scale - 2)
                }
            }
            ZoomLevel::Minimal => ZoomLevel::ScaleOut(PlotParameters::default().scale - 2),
            ZoomLevel::Low => ZoomLevel::Minimal,
            ZoomLevel::Default => ZoomLevel::Low,
            ZoomLevel::High => ZoomLevel::Default,
            ZoomLevel::Full => ZoomLevel::High,
            ZoomLevel::ScaleIn(scale) => {
                if scale - 2 <= PlotParameters::default().scale {
                    ZoomLevel::Full // Switch back to label-based zooming
                } else {
                    ZoomLevel::ScaleIn(scale - 2) // Continue scale-based zooming
                }
            }
        }
    }
}

pub enum EdgeStyle {
    Straight,
    Spline,
}

/// Holds parameters that do change as the user scrolls through the graph.
/// This includes multiple coordinate systems for the window and the canvas,
/// and we need to keep a 1:1 mapping between systems to avoid glitches.
///
/// From the terminal's perspective, the viewport is the inner area of the scrolling canvas area.
/// Terminal coordinates are referenced from the top-left corner of the terminal, with the y-axis
/// pointing downwards.
///
/// From the data's perspective, the viewport is a moving window defined by a width, height, and
/// offset from the data origin. When offset_x = 0 and offset_y = 0, the bottom-left corner of the
/// viewport has data coordinates (0,0) and the y-axis points upwards.
///
/// The edges are drawn using Unicode Braille characters, which are 2x4 pixels in size.
/// Data coordinates (x,y) are converted to braille dot coordinates using the following formula:
/// resolution_x = width * 2  
/// resolution_y = height * 4
///  x = ((x - left) * (resolution_x - 1.0) / (right - left));
///  y = ((top - y) * (resolution_y - 1.0) / (top - bottom));
///
/// The braille characters are localized using their bottom-left corner, and the other dots have
/// the following offsets to the bottom-left corner:
///
/// (0, 0.75)  (0.5, 0.75)
/// (0, 0.5)   (0.5, 0.5)  
/// (0, 0.25)  (0.5, 0.25)
/// (0, 0)     (0.5, 0)   
///
/// The top right corner of the viewport is equal to (x_max, y_max), but this is offset from the
/// top right terminal cell by (0.5,0.75) units. This is why we define the upper x and y bounds
/// as offset_x + (viewport_width - 1) + 1.0/2.0 and offset_y + (viewport_height - 1) + 3.0/4.0.
/// If you don't get this right you get visual glitches from rounding errors.
///
/// TODO: State is really overdue for a refactor where each coordinate system is a Rect-like struct.
///  - The Ratatui Rect is too small (u16) for world coordinates.
pub struct State {
    pub offset_x: i32,
    pub offset_y: i32,
    pub viewport: Rect,
    pub world: ((f64, f64), (f64, f64)), // (min_x, min_y), (max_x, max_y)
    pub world_viewport: ((f64, f64), (f64, f64)), // (min_x, min_y), (max_x, max_y)
    pub selected_block: Option<GraphNode>,
    pub first_render: bool,
    pub show_splash_screen: bool,
}
impl Default for State {
    fn default() -> Self {
        State {
            offset_x: 0,
            offset_y: 0,
            viewport: Rect::new(0, 0, 0, 0),
            world: ((0.0, 0.0), (0.0, 0.0)),
            world_viewport: ((0.0, 0.0), (0.0, 0.0)),
            selected_block: None,
            first_render: true,
            show_splash_screen: false,
        }
    }
}

pub struct Viewer<'a> {
    pub block_graph: &'a GenGraph,
    pub conn: &'a Connection,
    pub base_layout: BaseLayout,
    pub scaled_layout: ScaledLayout,
    pub node_sequences: HashMap<i64, Sequence>,
    pub state: State,
    pub parameters: PlotParameters,
    pub origin_block: Option<GraphNode>,
    view_block: Block<'a>,
    pub has_focus: bool,
    highlights: Vec<(Color, DiGraphMap<GraphNode, ()>)>,
}

impl<'a> Viewer<'a> {
    pub fn new(
        block_graph: &'a GenGraph,
        conn: &'a Connection,
        plot_parameters: PlotParameters,
    ) -> Viewer<'a> {
        let mut new_viewer = Self::with_origin(
            block_graph,
            conn,
            plot_parameters,
            (Node::get_start_node(), 0),
        );

        // If we're being asked to view an empty graph, show the splash screen
        if block_graph.node_count() == 1 {
            if let Some(node) = block_graph.nodes().next() {
                if node.node_id == crate::models::node::PATH_START_NODE_ID && node.block_id == -1 {
                    new_viewer.state.show_splash_screen = true;
                }
            }
        }
        new_viewer
    }

    pub fn with_origin(
        block_graph: &'a GenGraph,
        conn: &'a Connection,
        plot_parameters: PlotParameters,
        origin: (Node, i64),
    ) -> Viewer<'a> {
        let base_layout = BaseLayout::with_origin(block_graph, origin.clone());

        // Get the sequences for the nodes in the base_layout subgraph
        let node_sequences = Node::get_sequences_by_node_ids(
            conn,
            &base_layout
                .layout_graph
                .nodes()
                .map(|node| node.node_id)
                .collect::<HashSet<i64>>()
                .into_iter()
                .collect::<Vec<i64>>(),
        )
        .into_iter()
        .collect::<HashMap<i64, Sequence>>();

        // Stretch and scale the base layout to account for the sequence labels and plot parameters
        let scaled_layout = ScaledLayout::from_base_layout(&base_layout, &plot_parameters);

        // Capture the origin block from the base_layout's nodes (matching on node_id)
        let origin_block = base_layout
            .layout_graph
            .nodes()
            .find(|node| node.node_id == origin.0.id);

        // If using with_origin, the splash screen is never shown as we have a specific focus
        let initial_state = State {
            show_splash_screen: false,
            ..Default::default()
        };

        Viewer {
            block_graph,
            conn,
            base_layout,
            scaled_layout,
            node_sequences,
            state: initial_state,
            parameters: plot_parameters,
            origin_block,                 // Gen block
            view_block: Block::default(), //Ratatui block (TODO: make Viewer a proper widget with nesting, or find a better name)
            has_focus: false,
            highlights: Vec::new(),
        }
    }

    /// Highlight a specific path in the graph.
    pub fn show_path(&mut self, path: &Path, color: Color) -> Result<(), String> {
        let path_blocks = path.blocks(self.conn);

        // Project the path blocks onto the graph to get GraphNodes and strands
        let projected_path = project_path(self.block_graph, &path_blocks);

        // Filter out the start/end nodes
        let path_nodes: Vec<GraphNode> = projected_path
            .iter()
            .filter_map(|(node, _)| (!Node::is_terminal(node.node_id)).then_some(*node))
            .collect();

        // Build a linear subgraph
        let mut highlight_graph = DiGraphMap::<GraphNode, ()>::new();
        match path_nodes.len() {
            0 => {
                return Err(format!(
                    "Path {name} is not translatable to current graph.",
                    name = &path.name
                ))
            }
            1 => {
                _ = highlight_graph.add_node(path_nodes[0]);
            }
            _ => {
                for i in 0..path_nodes.len() - 1 {
                    let source = path_nodes[i];
                    let target = path_nodes[i + 1];
                    _ = highlight_graph.add_edge(source, target, ());
                }
            }
        }
        self.highlights.push((color, highlight_graph));
        Ok(())
    }

    /// Check if we currently have highlights of a specific color.
    pub fn has_highlight(&self, color: Color) -> bool {
        self.highlights.iter().any(|(c, _)| *c == color)
    }

    /// Clear the highlight from the graph.
    pub fn clear_highlight(&mut self, color: Color) {
        self.highlights.retain(|(c, _)| *c != color);
    }

    /// Refresh based on changed parameters or layout.
    pub fn refresh(&mut self) {
        self.scaled_layout
            .refresh(&self.base_layout, &self.parameters);
        self.state.world = self.compute_bounding_box();
    }

    /// Set the Ratatui Block in which to draw the graph.
    /// This is useful for styling but the name is unfortunate.
    pub fn set_block(&mut self, block: Block<'a>) {
        self.view_block = block;
    }

    /// Check if a block is visible in the viewport.
    pub fn is_block_visible(&self, block: GraphNode) -> bool {
        if let Some(((x1, y), (x2, _))) = self.scaled_layout.labels.get(&block) {
            // Check vertical overlap first (simpler)
            let y_visible = (*y as i32) >= self.state.offset_y
                && (*y as i32) < self.state.offset_y + self.state.viewport.height as i32;

            if !y_visible {
                return false;
            }

            // Check horizontal overlap
            // Block is visible if either:
            // 1. Start point is in viewport
            // 2. End point is in viewport
            // 3. Block spans the entire viewport
            let x1_in_view = (*x1 as i32) >= self.state.offset_x
                && (*x1 as i32) < self.state.offset_x + self.state.viewport.width as i32;
            let x2_in_view = (*x2 as i32) >= self.state.offset_x
                && (*x2 as i32) < self.state.offset_x + self.state.viewport.width as i32;
            let spans_viewport = (*x1 as i32) <= self.state.offset_x
                && (*x2 as i32) >= self.state.offset_x + self.state.viewport.width as i32;

            return x1_in_view || x2_in_view || spans_viewport;
        }
        false
    }

    /// Unselect the currently selected block if it's not visible in the viewport.
    pub fn unselect_if_not_visible(&mut self) {
        if let Some(selected_block) = self.state.selected_block {
            if !self.is_block_visible(selected_block) {
                self.state.selected_block = None;
            }
        }
    }

    /// Center the viewport on a specific block.
    pub fn center_on_block(&mut self, block: GraphNode) -> Result<(f64, f64), String> {
        if let Some(((start, y), (end, _))) = self.scaled_layout.labels.get(&block) {
            let cursor_x = (start + end) / 2.0;
            let cursor_y = *y;
            self.update_scroll_for_cursor(cursor_x, cursor_y);
            Ok((cursor_x, cursor_y))
        } else {
            Err(format!("Block ID {:?} not found in layout", block))
        }
    }

    fn draw_edge(
        &self,
        ctx: &mut ratatui::widgets::canvas::Context,
        edge: &(GraphNode, GraphNode),
        color: Color,
    ) {
        let ((x1, y1), (x2, y2)) = self.scaled_layout.lines[edge];

        // Select the exact braille row to hit depending on whether the edge is going up or down
        // Since there's no "middle" row, we choose between the 2nd and 3rd row of dots.
        //       o o  ^
        //   └-> o o -┘
        //   ┌-> o o -┐
        //       o o  v

        let (y1, y2) = if (y2 - y1).abs() < f64::EPSILON {
            (y1 + 1.0 / 4.0, y2 + 1.0 / 4.0)
        } else if y2 > y1 {
            (y1 + 2.0 / 4.0, y2 + 1.0 / 4.0)
        } else {
            (y1 + 1.0 / 4.0, y2 + 2.0 / 4.0)
        };

        // Leave one dot column of space between the edge and the block labels
        let x1 = x1 - 1.0 / 2.0;
        let x2 = x2 - 1.0;

        // Bounds to clip lines to
        let ((x_min, y_min), (x_max, y_max)) = self.state.world_viewport;
        match self.parameters.edge_style {
            EdgeStyle::Straight => {
                if let Some(((x1c, y1c), (x2c, y2c))) =
                    clip_line((x1, y1), (x2, y2), (x_min, y_min), (x_max, y_max))
                {
                    ctx.draw(&Line {
                        x1: x1c,
                        y1: y1c,
                        x2: x2c,
                        y2: y2c,
                        color,
                    });
                }
            }
            EdgeStyle::Spline => {
                // Bezier curves are always contained within the box defined by their endpoints,
                // so we reject any curves that don't have a bounding box that intersects the viewport.
                if !rectangles_intersect(
                    (x1, y1),
                    (x2, y2),
                    (self.state.offset_x as f64, self.state.offset_y as f64),
                    (
                        (self.state.offset_x + (self.state.viewport.width - 1) as i32) as f64,
                        (self.state.offset_y + (self.state.viewport.height - 1) as i32) as f64,
                    ),
                ) {
                    return;
                }
                // The maximum resolution we can attain is 2 dots per character,
                // and x positions are restricted to the braille grid (which simplifies the math).
                // x1 = 0.0, x2 = 2.0
                // [x.][..][x.] => (2.0 - 0.0) * 2 + 1 = 5 dots (ends included)
                // x1 = 0.0, x2 = 2.5
                // [x.][..][.x] => (2.5 - 0.0) * 2 + 1 = 6 dots
                //
                // We do want to cap the number of points at 64 for very long edges.

                let num_points = u32::min(64, ((x2 - x1) * 2.0 + 1.0) as u32);
                let curve_points = generate_cubic_bezier_curve((x1, y1), (x2, y2), num_points);

                // Draw lines between consecutive points of the curve
                for window in curve_points.windows(2) {
                    if let Some(((x1c, y1c), (x2c, y2c))) =
                        clip_line(window[0], window[1], (x_min, y_min), (x_max, y_max))
                    {
                        ctx.draw(&Line {
                            x1: x1c,
                            y1: y1c,
                            x2: x2c,
                            y2: y2c,
                            color,
                        });
                    }
                }
            }
        }
    }

    /// Get a block's sequence and truncate it to fit within a given width.
    fn make_label(&self, block: &GraphNode, max_width: u32) -> String {
        // At label width 0, don't even attempt to show any sequence information
        if self.parameters.label_width == 0 {
            return label::NODE.to_string().to_string();
        }

        let label = if let Some(sequence) = self.node_sequences.get(&block.node_id) {
            inner_truncation(
                sequence
                    .get_sequence(block.sequence_start, block.sequence_end)
                    .as_str(),
                max_width,
            )
        } else {
            // If for some reason we weren't able to get the actual sequence,
            // do show a placeholder of the correct length.
            let seq_len = (block.sequence_end - block.sequence_start).unsigned_abs() as u32;
            "?".repeat(max_width.min(seq_len) as usize)
        };

        label
    }

    /// Print a block label at the given position.
    fn place_label(
        &self,
        ctx: &mut ratatui::widgets::canvas::Context,
        label: &str,
        pos: (f64, f64),
        style: ratatui::style::Style,
    ) {
        // Show more visual information if we are in debug mode
        let debug = std::env::var("RUST_LOG").is_ok_and(|v| v.contains("debug"));

        let (x, y) = pos;

        // Clip to the viewport
        let clipped_label = clip_label(
            label,
            x as isize,
            (self.state.offset_x + 1) as isize,
            (self.state.viewport.width - 1) as usize,
        );

        if clipped_label.is_empty() {
            return;
        }

        ctx.print(
            f64::max(x, self.state.offset_x as f64),
            y,
            Span::styled(clipped_label, style),
        );

        if debug {
            ctx.print(
                x,
                y + 1.0,
                Span::styled(
                    format!("↓({:.1},{:.1})", x, y),
                    Style::default().fg(Color::Red),
                ),
            );
        }
    }

    /// Draw and render blocks and lines to a canvas through a scrollable window.
    /// TODO: turn this into the render function of a custom stateful widget
    pub fn draw(&mut self, frame: &mut ratatui::Frame, area: Rect) {
        // Show more visual information if we are in debug mode
        let debug = std::env::var("RUST_LOG").is_ok_and(|v| v.contains("debug"));

        // Define the viewport from the perspective of the terminal
        let canvas_block = self.view_block.clone();
        let viewport = canvas_block.inner(area);

        // Splashscreen logic
        if self.state.show_splash_screen && self.state.selected_block.is_none() {
            let splashscreen_lines: Vec<&str> = vec![
                " ██████╗ ███████╗███╗   ██╗",
                "██╔════╝ ██╔════╝████╗  ██║",
                "██║  ███╗█████╗  ██╔██╗ ██║",
                "██║   ██║██╔══╝  ██║╚██╗██║",
                "╚██████╔╝███████╗██║ ╚████║",
                " ╚═════╝ ╚══════╝╚═╝  ╚═══╝",
            ];
            let splashscreen_height = splashscreen_lines.len() as u16;
            let splashscreen_width = splashscreen_lines
                .iter()
                .map(|s| s.chars().count())
                .max()
                .unwrap_or(0) as u16;

            if viewport.width >= splashscreen_width && viewport.height >= splashscreen_height {
                let start_x = viewport.x + (viewport.width - splashscreen_width) / 2;
                let start_y = viewport.y + (viewport.height - splashscreen_height) / 2;

                let splash_canvas = Canvas::default()
                    .background_color(col("canvas").unwrap())
                    .block(canvas_block)
                    .x_bounds([
                        viewport.x as f64,
                        (viewport.x + viewport.width - 1) as f64 + 0.5,
                    ])
                    .y_bounds([
                        viewport.y as f64,
                        (viewport.y + viewport.height - 1) as f64 + 0.75,
                    ])
                    .paint(|ctx| {
                        for (i, line) in splashscreen_lines.iter().enumerate() {
                            ctx.print(
                                start_x as f64,
                                (start_y + (splashscreen_height - 1 - i as u16)) as f64,
                                Span::styled(*line, Style::default().fg(col("base07").unwrap())),
                            );
                        }
                    });
                frame.render_widget(splash_canvas, area);
                return;
            }
        }

        // Check if the viewport has changed size, and if so, update the offset to keep our reference
        // frame intact.
        if self.state.viewport != viewport {
            let x_diff = viewport.x as i32 - self.state.viewport.x as i32;
            let y_diff = viewport.y as i32 - self.state.viewport.y as i32;
            let height_diff = viewport.height as i32 - self.state.viewport.height as i32;

            // Adjust offsets based on viewport changes
            self.state.offset_x -= x_diff;
            // Accommodate inverted y-axis between terminal and data coordinates
            self.state.offset_y += y_diff - height_diff;

            info!(
                "viewport dimensions changed to {:?} (was {:?})",
                viewport, self.state.viewport
            );
            self.state.viewport = viewport;
        }

        // Set initial scroll offset if not already set, aligning the origin block:
        if self.state.first_render
            && self.state.viewport.width > 0
            && self.state.viewport.height > 0
        {
            if let Some(origin) = self.origin_block {
                if Node::is_start_node(origin.node_id) {
                    // Find the first non-start/end node by looking at outgoing neighbors of the start node
                    self.state.selected_block = self
                        .base_layout
                        .layout_graph
                        .neighbors_directed(origin, Direction::Outgoing)
                        .find(|node| {
                            !Node::is_start_node(node.node_id) && !Node::is_end_node(node.node_id)
                        });
                } else {
                    self.state.selected_block = Some(origin);
                }
                self.state.world = self.compute_bounding_box();
                if let Some(center_block) = self.state.selected_block {
                    self.center_on_block(center_block).unwrap();
                }
            }
            self.state.first_render = false;
        }

        // Define the viewport from the world perspective
        // TODO: make this a function implemented on the State struct
        // to ensure consistency of the coordinate systems
        let x_min = self.state.offset_x as f64;
        let y_min = self.state.offset_y as f64;
        let x_max = x_min + self.state.viewport.width as f64 - 1.0;
        let y_max = y_min + self.state.viewport.height as f64 - 1.0;
        // Add the 1/2 and 3/4 units to account for the braille cell size on the right and top
        let x_max = x_max + 1.0 / 2.0;
        let y_max = y_max + 3.0 / 4.0;
        self.state.world_viewport = ((x_min, y_min), (x_max, y_max));

        // Make sure we always have a block selected when we come into focus
        if self.state.selected_block.is_none() && self.has_focus {
            self.select_center_block();
        }

        let canvas = Canvas::default()
            .background_color(col("canvas").unwrap())
            .block(canvas_block)
            // Adjust the x_bounds and y_bounds by the scroll offsets.
            .x_bounds([x_min, x_max])
            .y_bounds([y_min, y_max])
            .paint(|ctx| {
                if debug {
                    self.draw_debug_background(ctx, (x_min, x_max), (y_min, y_max));
                }

                // Draw all edges (does not consider highlights)
                for &(source, target) in self.scaled_layout.lines.keys() {
                    self.draw_edge(ctx, &(source, target), col("edge").unwrap());
                }

                // Print the labels
                for &block in self.scaled_layout.labels.keys() {
                    // Skip the start and end dummy nodes, we draw those as "arrows" on the blocks
                    // to which they are connected.
                    if Node::is_start_node(block.node_id) || Node::is_end_node(block.node_id) {
                        continue;
                    }

                    // Skip blocks that are not in the visible area at all
                    if !self.is_block_visible(block) {
                        continue;
                    }

                    let ((x, y), (x2, _y2)) = self.scaled_layout.labels[&block];
                    let mut label = self.make_label(&block, (x2 - x) as u32);

                    // Style the label
                    // This depends on the zoom level (at lower zoom levels we only show glyphs)
                    // and whether the block is selected or not.
                    let is_selected = Some(block) == self.state.selected_block;
                    let is_glyph = label == label::NODE;
                    let style = match (is_glyph, is_selected) {
                        (true, true) => {
                            label = label::SELECTED.to_string();
                            Style::default().fg(col("highlight").unwrap())
                        }
                        (true, false) => Style::default().fg(col("text").unwrap()),
                        (false, true) => Style::default()
                            .fg(col("text_muted").unwrap())
                            .bg(col("highlight").unwrap()),
                        (false, false) => Style::default()
                            .fg(col("text").unwrap())
                            .bg(col("node").unwrap()),
                    };

                    self.place_label(ctx, &label, (x, y), style);

                    // Place a start arrow meta-label if the block is connected to the start dummy node
                    if self
                        .base_layout
                        .layout_graph
                        .neighbors_directed(block, Direction::Incoming)
                        .any(|neighbor| Node::is_start_node(neighbor.node_id))
                    {
                        let x3 = x - 1.0 - (label::START.chars().count() as f64);
                        self.place_label(
                            ctx,
                            label::START,
                            (x3, y),
                            Style::default().fg(col("text_muted").unwrap()),
                        );
                    }

                    // Draw an end arrow if the block is connected to the end dummy node
                    if self
                        .base_layout
                        .layout_graph
                        .neighbors_directed(block, Direction::Outgoing)
                        .any(|neighbor| Node::is_end_node(neighbor.node_id))
                    {
                        let x3 = x2 + 1.0;
                        self.place_label(
                            ctx,
                            label::END,
                            (x3, y),
                            Style::default().fg(col("text_muted").unwrap()),
                        );
                    }
                }
            });
        frame.render_widget(canvas, area);

        // Create a buffer of the same size as the canvas to layer on top of the buffer in the frame
        let mut highlights_buf = Buffer::empty(area);
        let ((x_min, y_min), (x_max, y_max)) = self.state.world_viewport;
        Canvas::default()
            .x_bounds([x_min, x_max])
            .y_bounds([y_min, y_max])
            .background_color(col("canvas").unwrap())
            .paint(|ctx| {
                for (color, highlight_graph) in &self.highlights {
                    for (source, target, _) in highlight_graph.all_edges() {
                        if self.scaled_layout.lines.contains_key(&(source, target)) {
                            self.draw_edge(ctx, &(source, target), *color);
                        }
                    }
                }
            })
            .render(area, &mut highlights_buf);

        // Overlay the highlights buffer on top of the frame
        for y in area.y..area.y + area.height {
            for x in area.x..area.x + area.width {
                let highlight_cell = highlights_buf[(x, y)].clone();
                if highlight_cell.symbol() != " " {
                    frame.buffer_mut()[(x, y)].set_symbol(highlight_cell.symbol());
                    frame.buffer_mut()[(x, y)].set_style(highlight_cell.style());
                }
            }
        }

        // Compute more of the base layout if we're getting close to the ends.
        self.auto_expand();
    }

    /// Draw debug overlay showing grid and boundaries
    fn draw_debug_background(
        &self,
        ctx: &mut ratatui::widgets::canvas::Context,
        (x_min, x_max): (f64, f64),
        (y_min, y_max): (f64, f64),
    ) {
        // Draw a grid of braille dots to test alignment
        for x in (x_min.floor() as i32)..=(x_max.ceil() as i32) {
            if x % 10 != 0 {
                continue;
            }
            for y in (y_min.floor() as i32)..=(y_max.ceil() as i32) {
                if y % 5 != 0 {
                    continue;
                }
                let color =
                    if (x / 10 % 2 == 0 && y / 5 % 2 != 0) || (x / 10 % 2 != 0 && y / 5 % 2 == 0) {
                        Color::Blue
                    } else {
                        Color::Red
                    };
                ctx.draw(&Points {
                    coords: &[(x as f64 + 1.0 / 2.0, y as f64 + 3.0 / 4.0)],
                    color,
                });
            }
        }

        // Paint bottom left and top right corners
        ctx.draw(&Line {
            x1: x_min,
            y1: y_min,
            x2: x_min + 3.0 / 2.0,
            y2: y_min,
            color: Color::Magenta,
        });
        ctx.draw(&Line {
            x1: x_min,
            y1: y_min,
            x2: x_min,
            y2: y_min + 3.0 / 4.0,
            color: Color::Magenta,
        });
        ctx.draw(&Line {
            x1: x_max,
            y1: y_max,
            x2: x_max - 3.0 / 2.0,
            y2: y_max,
            color: Color::Green,
        });
        ctx.draw(&Line {
            x1: x_max,
            y1: y_max,
            x2: x_max,
            y2: y_max - 3.0 / 4.0,
            color: Color::Green,
        });
    }

    /// Check the viewport bounds against the world bounds and trigger expansion if needed.
    pub fn auto_expand(&mut self) {
        let ((x_min, _), (x_max, _)) = self.state.world;
        // Check if we're a screen width away from the left/right boundary and expand if needed
        // - if we can't expand any further, this will do nothing
        if (x_min as i32) > (self.state.offset_x - self.state.viewport.width as i32) {
            self.base_layout.expand_left();
        }
        if (x_max as i32) < (self.state.offset_x + 2 * self.state.viewport.width as i32) {
            self.base_layout.expand_right();
        }
    }

    /// Cycle through nodes in a specified direction.
    pub fn move_selection(&mut self, direction: NavDirection) {
        // Get the position of the currently selected block, or return early if none
        let current_point = match self
            .state
            .selected_block
            .as_ref()
            .and_then(|selected| self.base_layout.node_positions.get(selected).cloned())
        {
            Some(position) => position,
            None => return,
        };

        // Try to find a node in the specified direction closest to the current point.
        if let Some(new_selection) = self.find_closest_block_in_direction(current_point, direction)
        {
            self.state.selected_block = Some(new_selection);
            if let Err(e) = self.center_on_block(new_selection) {
                warn!("Viewer - error finding block to switch to: {}", e);
            }
        }
    }

    /// Snap the cursor to the closest label in the given direction.
    pub fn snap_cursor(&mut self, direction: NavDirection) {
        // For now, assume the cursor is at the center of the viewport
        // TODO: actually track the cursor position in the state
        let current_point = (
            self.state.offset_x as f64 + self.state.viewport.width as f64 / 2.0,
            self.state.offset_y as f64 + self.state.viewport.height as f64 / 2.0,
        );

        let mut closest_candidate: Option<(GraphNode, f64)> = None;

        for (block, ((x1, y), (x2, _))) in &self.scaled_layout.labels {
            // Skip start and end nodes
            if Node::is_start_node(block.node_id) || Node::is_end_node(block.node_id) {
                continue;
            }

            // Calculate distances
            let dx = current_point.0 - (*x1 + *x2) / 2.0;
            let dy = current_point.1 - *y;

            // Check if this label is a candidate based on direction
            let is_candidate = match direction {
                // Check if the label is in the appropriate quadrant based on direction
                // Using the lines x=y and x=-y to divide the space into quadrants
                NavDirection::Up => dy < 0.0 && dy.abs() > dx.abs(), // Upper quadrant: |y| > |x|
                NavDirection::Down => dy > 0.0 && dy.abs() > dx.abs(), // Lower quadrant: |y| > |x|
                NavDirection::Left => dx > 0.0 && dx.abs() > dy.abs(), // Left quadrant: |x| > |y|
                NavDirection::Right => dx < 0.0 && dx.abs() > dy.abs(), // Right quadrant: |x| > |y|
            };

            if !is_candidate {
                continue;
            }

            // Calculate Euclidean distance
            let distance = (dx * dx + dy * dy).sqrt();

            // Update closest candidate if this one is closer
            if let Some((_, best_dist)) = closest_candidate {
                if distance < best_dist {
                    closest_candidate = Some((*block, distance));
                }
            } else {
                closest_candidate = Some((*block, distance));
            }
        }

        // If we found a candidate, update the selection and center on it
        if let Some((new_selection, _)) = closest_candidate {
            self.state.selected_block = Some(new_selection);
        }
    }

    /// Helper function to find the closest node in the given direction
    /// - This operates in base_layout coordinates, not scaled_layout!
    fn find_closest_block_in_direction(
        &self,
        current_point: (f64, f64),
        direction: NavDirection,
    ) -> Option<GraphNode> {
        let mut closest_candidate: Option<(GraphNode, f64)> = None;

        for (&node, &node_pos) in &self.base_layout.node_positions {
            // Skip the currently selected block and any start/end node
            if Some(node) == self.state.selected_block
                || Node::is_start_node(node.node_id)
                || Node::is_end_node(node.node_id)
            {
                continue;
            }

            let dx = node_pos.0 - current_point.0;
            let dy = node_pos.1 - current_point.1;

            // Depending on the direction, decide if this node is a candidate
            //   - Up/Down: only consider nodes that are vertically aligned
            //   - Left/Right: consider all nodes in the correct half of the screen
            if !match direction {
                NavDirection::Up => node_pos.1 > current_point.1 && dx.abs() < f64::EPSILON,
                NavDirection::Down => node_pos.1 < current_point.1 && dx.abs() < f64::EPSILON,
                NavDirection::Left => dx < 0.0,
                NavDirection::Right => dx > 0.0,
            } {
                continue;
            }

            // Calculate Euclidean distance
            let distance = (dx * dx + dy * dy).sqrt();

            // Keep track if it's closer than any previous candidate
            if let Some((_, best_dist)) = closest_candidate {
                // When scrolling horizontally, break ties by preferring the down direction
                // (otherwise it looks random to the user)
                if (direction == NavDirection::Left || direction == NavDirection::Right)
                    && (distance - best_dist).abs() < f64::EPSILON
                    && dy < 0.0
                {
                    closest_candidate = Some((node, distance));
                } else if distance < best_dist {
                    // No tie-breaking needed
                    closest_candidate = Some((node, distance));
                }
            } else {
                closest_candidate = Some((node, distance));
            }
        }

        // Return the node with the minimum distance in the chosen direction
        closest_candidate.map(|(n, _)| n)
    }

    /// Select the block closest to the center of the viewport using coordinates from scaled_layout.
    pub fn select_center_block(&mut self) -> Option<GraphNode> {
        let center = (
            self.state.offset_x as f64 + self.state.viewport.width as f64 / 2.0,
            self.state.offset_y as f64 + self.state.viewport.height as f64 / 2.0,
        );
        let mut best_candidate: Option<(GraphNode, f64)> = None;
        for (node, &((start, y), (end, _))) in self.scaled_layout.labels.iter() {
            let candidate_center = ((start + end) / 2.0, y);
            let dx = candidate_center.0 - center.0;
            let dy = candidate_center.1 - center.1;
            let dist = (dx * dx + dy * dy).sqrt();
            best_candidate = match best_candidate {
                Some((_, best)) if dist < best => Some((*node, dist)),
                None => Some((*node, dist)),
                other => other,
            };
        }
        if let Some((node, _)) = best_candidate {
            self.state.selected_block = Some(node);
            Some(node)
        } else {
            None
        }
    }

    fn compute_bounding_box(&self) -> ((f64, f64), (f64, f64)) {
        let labels = &self.scaled_layout.labels;

        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for ((x, _), (x2, _)) in labels.values() {
            xs.push(*x);
            xs.push(*x2);
        }
        for ((_, y), (_, _)) in labels.values() {
            ys.push(*y);
        }
        let world_min_x = xs.iter().cloned().fold(f64::INFINITY, f64::min);
        let world_max_x = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let world_min_y = ys.iter().cloned().fold(f64::INFINITY, f64::min);
        let world_max_y = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        ((world_min_x, world_min_y), (world_max_x, world_max_y))
    }

    /// Update scroll offset based on the cursor position (world coordinates of the selected label).
    /// This method computes the world bounds from all labels and clamps the viewport's offset.
    /// On the initial call (first_render), it remains centered. Afterwards, we allow the cursor
    /// to drift within a tolerance range vertically before scrolling.
    pub fn update_scroll_for_cursor(&mut self, cursor_x: f64, cursor_y: f64) {
        let margin = 10.0;
        let vp_width = self.state.viewport.width as f64;
        let vp_height = self.state.viewport.height as f64;
        let bandwidth = 0.4;

        let ((world_min_x, world_min_y), (world_max_x, world_max_y)) = self.state.world;
        let min_x = world_min_x - margin;
        let max_x = world_max_x + margin;
        let min_y = world_min_y - margin;
        let max_y = world_max_y + margin;

        let total_width = max_x - min_x;
        let total_height = max_y - min_y;

        // If it's the initial placement, there is only one allowed position.
        if self.state.first_render {
            let desired_x = cursor_x - vp_width / 2.0;
            let desired_y = cursor_y - vp_height / 2.0;

            let new_offset_x = if total_width >= vp_width {
                desired_x.clamp(min_x, max_x - vp_width)
            } else {
                min_x - (vp_width - total_width) / 2.0
            };
            let new_offset_y = if total_height >= vp_height {
                desired_y.clamp(min_y, max_y - vp_height)
            } else {
                min_y - (vp_height - total_height) / 2.0
            };

            self.state.offset_x = new_offset_x.round() as i32;
            self.state.offset_y = new_offset_y.round() as i32;

            return;
        }

        // In later iterations we treat vertical and horizontal movement differently.
        // Horizontal still clamps to one point
        let desired_x = cursor_x - vp_width / 2.0;
        let new_offset_x = if total_width >= vp_width {
            desired_x.clamp(min_x, max_x - vp_width)
        } else {
            min_x - (vp_width - total_width) / 2.0
        };

        // Vertical is centering and clamping to a range of y-coordinates.
        let current_offset_y = self.state.offset_y as f64;
        let top_boundary = current_offset_y + bandwidth * vp_height;
        let bottom_boundary = current_offset_y + (1.0 - bandwidth) * vp_height;

        let mut desired_y = current_offset_y;
        if cursor_y < top_boundary {
            desired_y -= top_boundary - cursor_y;
        } else if cursor_y > bottom_boundary {
            desired_y += cursor_y - bottom_boundary;
        }

        // Clamp vertically
        if total_height >= vp_height {
            desired_y = desired_y.clamp(min_y, max_y - vp_height);
        } else {
            desired_y = min_y - (vp_height - total_height) / 2.0;
        }

        self.state.offset_x = new_offset_x.round() as i32;
        self.state.offset_y = desired_y.round() as i32;
    }

    /// Get the terminal coordinates of a block's center point
    fn get_block_terminal_coords(&self, block: GraphNode) -> Option<(f64, f64)> {
        if let Some(((start, y), (end, _))) = self.scaled_layout.labels.get(&block) {
            let block_center_x = (start + end) / 2.0;
            let block_center_y = *y;

            // Convert from world coordinates to terminal coordinates
            let terminal_x = block_center_x - self.state.offset_x as f64;
            let terminal_y = block_center_y - self.state.offset_y as f64;

            Some((terminal_x, terminal_y))
        } else {
            None
        }
    }

    pub fn handle_input(&mut self, key: KeyEvent) {
        // Panning a cursor with shift/alt was too inconsistent
        // (e.g. on mac os, alt-left/right is intercepted by the system)
        // so now we make it automatic based on blocks being cut off on the left/right.
        const SCROLL_STEP_X: i32 = 12;

        match key.code {
            KeyCode::Left => {
                if let Some(selected_block) = self.state.selected_block {
                    let ((block_x1_f, _), (_, _)) =
                        self.scaled_layout.labels.get(&selected_block).unwrap();
                    // Selected block is cut off on the left.
                    // TODO: make this more about the next block being visible
                    if (block_x1_f.round() as i32) < self.state.offset_x {
                        self.state.offset_x -= SCROLL_STEP_X;
                    } else {
                        self.move_selection(NavDirection::Left);
                    }
                } else {
                    self.state.offset_x -= self.state.viewport.width as i32 / 3;
                }
            }
            KeyCode::Right => {
                if let Some(selected_block) = self.state.selected_block {
                    let ((_, _), (block_x2_f, _)) =
                        self.scaled_layout.labels.get(&selected_block).unwrap();
                    // Selected block is cut off on the right.
                    if (block_x2_f.round() as i32)
                        > self.state.offset_x + self.state.viewport.width as i32
                    {
                        self.state.offset_x += SCROLL_STEP_X;
                    } else {
                        self.move_selection(NavDirection::Right);
                    }
                } else {
                    self.state.offset_x += self.state.viewport.width as i32 / 3;
                }
            }
            KeyCode::Up => {
                self.move_selection(NavDirection::Up);
            }
            KeyCode::Down => {
                self.move_selection(NavDirection::Down);
            }
            // Zooming in and out
            KeyCode::Char('+') | KeyCode::Char('=') => {
                // Record terminal coordinates of selected block before zoom
                let terminal_coords = self
                    .state
                    .selected_block
                    .and_then(|block| self.get_block_terminal_coords(block));

                // Increase how much of the sequence is shown in each block label.
                self.parameters
                    .set_zoom_level(self.parameters.zoom_in_level());

                // If no block is selected, try to select the center block
                if self.state.selected_block.is_none() {
                    self.select_center_block();
                }

                // Recalculate the layout.
                self.scaled_layout
                    .refresh(&self.base_layout, &self.parameters);
                self.state.world = self.compute_bounding_box();

                // Adjust viewport to maintain terminal coordinates of selected block
                if let Some((old_x, old_y)) = terminal_coords {
                    if let Some(block) = self.state.selected_block {
                        if let Some(((start, y), (end, _))) = self.scaled_layout.labels.get(&block)
                        {
                            let new_center_x = (start + end) / 2.0;
                            let new_center_y = *y;

                            // Calculate new offsets to maintain terminal coordinates
                            self.state.offset_x = (new_center_x - old_x).round() as i32;
                            self.state.offset_y = (new_center_y - old_y).round() as i32;
                        }
                    }
                }
            }
            KeyCode::Char('-') | KeyCode::Char('_') => {
                // Record terminal coordinates of selected block before zoom
                let terminal_coords = self
                    .state
                    .selected_block
                    .and_then(|block| self.get_block_terminal_coords(block));

                // Decrease how much of the sequence is shown in each block label.
                self.parameters
                    .set_zoom_level(self.parameters.zoom_out_level());

                if self.state.selected_block.is_none() {
                    self.select_center_block();
                }

                self.scaled_layout
                    .refresh(&self.base_layout, &self.parameters);
                self.state.world = self.compute_bounding_box();

                // Adjust viewport to maintain terminal coordinates of selected block
                if let Some((old_x, old_y)) = terminal_coords {
                    if let Some(block) = self.state.selected_block {
                        if let Some(((start, y), (end, _))) = self.scaled_layout.labels.get(&block)
                        {
                            let new_center_x = (start + end) / 2.0;
                            let new_center_y = *y;

                            // Calculate new offsets to maintain terminal coordinates
                            self.state.offset_x = (new_center_x - old_x).round() as i32;
                            self.state.offset_y = (new_center_y - old_y).round() as i32;
                        }
                    }
                }
            }
            KeyCode::Char('s') | KeyCode::Char('S') => {
                // Toggle between straight lines and splines
                self.parameters.edge_style = match self.parameters.edge_style {
                    EdgeStyle::Straight => EdgeStyle::Spline,
                    EdgeStyle::Spline => EdgeStyle::Straight,
                };
                self.scaled_layout
                    .refresh(&self.base_layout, &self.parameters);
            }
            _ => {}
        }
    }

    pub fn get_status_line() -> String {
        "*◀ ▼ ▲ ▶* scroll | *+/-* zoom | *s* edge style".to_string()
    }
}

/// Truncate a string to a certain length, adding an ellipsis in the middle
fn inner_truncation(s: &str, target_length: u32) -> String {
    let input_length = s.chars().count() as u32;
    if input_length <= target_length {
        return s.to_string();
    } else if target_length < 5 {
        return label::NODE.to_string(); // ○ is U+25CB; ● is U+25CF
    }
    // length - 3 because we need space for the ellipsis
    let left_len = (target_length - 3) / 2 + ((target_length - 3) % 2);
    let right_len = (target_length - 3) - left_len;

    format!(
        "{}...{}",
        &s[..left_len as usize],
        &s[input_length as usize - right_len as usize..]
    )
}

/// Clips a string to a specific window, indicating that it has been clipped.
/// - If the string is empty, it returns an empty string.
/// - If the string is shorter than the window, it returns the string.
/// - If the string is longer than the window, it clips the string and replaces the last character with a period.
/// - If the string is not within the window at all, it returns an empty string.
pub fn clip_label(
    label: &str,
    label_start: isize,
    window_start: isize,
    window_width: usize,
) -> String {
    if label.is_empty() {
        return "".to_string();
    }

    let label_end = label_start + label.chars().count() as isize - 1;
    let window_end = window_start + window_width as isize - 1;
    if label_end < window_start || label_start > window_end {
        return "".to_string();
    }

    if label_start >= window_start && label_end <= window_end {
        return label.to_string();
    }

    let mut clipped = label.to_string();

    // Process the right side first so we don't lose alignment:
    if label_end > window_end {
        let delta_right = label_end - window_end;
        // Make sure we don't try to cut in the middle of a multibyte character
        let character_cutoff = (label.chars().count() as isize - delta_right - 1) as usize;
        let byte_cutoff = label
            .char_indices()
            .nth(character_cutoff)
            .map(|(i, _)| i)
            .unwrap_or(label.len());
        clipped.replace_range(byte_cutoff.., "…");
    }
    if window_start > label_start {
        let delta_left = window_start - label_start;
        let character_cutoff = delta_left as usize + 1;
        let byte_cutoff = label
            .char_indices()
            .nth(character_cutoff)
            .map(|(i, _)| i)
            .unwrap_or(label.len());
        clipped.replace_range(..byte_cutoff, "…");
    }

    clipped
}

/// Clip a line given as two points to a specific window, also given by two points
/// - If the line is completely outside the window, it returns None.
/// - If the line is completely inside the window, it returns the original line.
/// - If the line is partially inside the window, it clips the line to the window.
///
/// This may be made more efficient through bitwise comparisons (see Cohen-Sutherland line clipping algorithm)
pub fn clip_line(
    (x1, y1): (f64, f64),   // Line start
    (x2, y2): (f64, f64),   // Line end
    (wx1, wy1): (f64, f64), // Window top left
    (wx2, wy2): (f64, f64), // Window bottom right
) -> Option<((f64, f64), (f64, f64))> {
    let mut t0 = 0.0;
    let mut t1 = 1.0;
    let dx = x2 - x1;
    let dy = y2 - y1;

    let clip = |p: f64, q: f64, t0: &mut f64, t1: &mut f64| -> bool {
        if p == 0.0 {
            return q >= 0.0;
        }
        let r = q / p;
        if p < 0.0 {
            if r > *t1 {
                return false;
            }
            if r > *t0 {
                *t0 = r;
            }
        } else {
            if r < *t0 {
                return false;
            }
            if r < *t1 {
                *t1 = r;
            }
        }
        true
    };

    if clip(-dx, x1 - wx1, &mut t0, &mut t1)
        && clip(dx, wx2 - x1, &mut t0, &mut t1)
        && clip(-dy, y1 - wy1, &mut t0, &mut t1)
        && clip(dy, wy2 - y1, &mut t0, &mut t1)
    {
        let nx1 = x1 + t0 * dx;
        let ny1 = y1 + t0 * dy;
        let nx2 = x1 + t1 * dx;
        let ny2 = y1 + t1 * dy;
        Some(((nx1, ny1), (nx2, ny2)))
    } else {
        None
    }
}

/// Check if two rectangles intersect.
/// - Each rectangle is defined by any two opposite corners.
/// - Returns true if the rectangles overlap or touch, false otherwise.
pub fn rectangles_intersect(
    (x1, y1): (f64, f64), // First corner of rectangle 1
    (x2, y2): (f64, f64), // Opposite corner of rectangle 1
    (x3, y3): (f64, f64), // First corner of rectangle 2
    (x4, y4): (f64, f64), // Opposite corner of rectangle 2
) -> bool {
    // For each axis, one rectangle's maximum must be >= other's minimum
    // and one rectangle's minimum must be <= other's maximum
    x1.max(x2) >= x3.min(x4)
        && x3.max(x4) >= x1.min(x2)
        && y1.max(y2) >= y3.min(y4)
        && y3.max(y4) >= y1.min(y2)
}

/// Generate a cubic bezier curve between two points A and B, given a resolution value.
/// - Control points 0 and 3 are equal to A and B.
/// - Control point 1 is halfway between A and B, at the same height as A.
/// - Control point 2 is halfway between A and B, at the same height as B.
///
/// The function returns resolution + 2 points:
/// - First point is exactly A
/// - Last point is exactly B
/// - For resolution=0: returns [A, B]
/// - For resolution=1: returns [A, midpoint, B] where midpoint is the true curve midpoint at t=0.5
/// - For resolution>1: returns [A, ...resolution points along the curve..., B]
pub fn generate_cubic_bezier_curve(
    a: (f64, f64),
    b: (f64, f64),
    num_points: u32,
) -> Vec<(f64, f64)> {
    let (ax, ay) = a;
    let (bx, by) = b;
    // Define control points following Graphviz's style:
    // p0: a, p1: midpoint between a and b at the same height as a,
    // p2: midpoint between a and b at the same height as b, p3: b
    let p0 = a;
    let p1 = (((ax + bx) / 2.0), ay);
    let p2 = (((ax + bx) / 2.0), by);
    let p3 = b;

    let mut points = Vec::with_capacity(num_points as usize);
    // First point is exactly a
    points.push(a);

    // Calculate intermediate points
    for i in 1..num_points - 1 {
        let t = i as f64 / ((num_points - 1) as f64);
        let one_minus_t = 1.0_f64 - t;
        let x = one_minus_t.powi(3) * p0.0
            + 3.0_f64 * one_minus_t.powi(2) * t * p1.0
            + 3.0_f64 * one_minus_t * t.powi(2) * p2.0
            + t.powi(3) * p3.0;
        let y = one_minus_t.powi(3) * p0.1
            + 3.0_f64 * one_minus_t.powi(2) * t * p1.1
            + 3.0_f64 * one_minus_t * t.powi(2) * p2.1
            + t.powi(3) * p3.1;
        points.push(round_to_braille_grid((x, y)));
        // TODO: figure out why this doesn't work to make the curve look better
        //points.push((x, y));
    }

    // Last point is exactly b
    points.push(b);
    points
}

/// Round a (f64, f64) to the nearest 1/2 and 1/4.
pub fn round_to_braille_grid((x, y): (f64, f64)) -> (f64, f64) {
    let (cell_x, x_offset) = (x.floor(), x.fract());
    let (cell_y, y_offset) = (y.floor(), y.fract());

    let x_rounded = cell_x + (x_offset * 2.0).round() / 2.0;
    let y_rounded = cell_y + (y_offset * 4.0).round() / 4.0;
    (x_rounded, y_rounded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inner_truncation_no_truncation_needed() {
        let s = "hello";
        let truncated = inner_truncation(s, 5);
        assert_eq!(truncated, "hello");
    }

    #[test]
    fn test_inner_truncation_truncate_to_odd_length() {
        let s = "hello world";
        let truncated = inner_truncation(s, 5);
        assert_eq!(truncated, "h...d");
    }

    #[test]
    fn test_inner_truncation_truncate_to_even_length() {
        let s = "hello world";
        let truncated = inner_truncation(s, 6);
        assert_eq!(truncated, "he...d");
    }

    #[test]
    fn test_inner_truncation_empty_string() {
        let s = "";
        let truncated = inner_truncation(s, 5);
        assert_eq!(truncated, "");
    }

    #[allow(clippy::too_many_arguments)]
    fn test_clip_line_helper(
        x1: f64,
        y1: f64,
        x2: f64,
        y2: f64,
        wx1: f64,
        wy1: f64,
        wx2: f64,
        wy2: f64,
        expected: Option<((f64, f64), (f64, f64))>,
    ) {
        let clipped = clip_line((x1, y1), (x2, y2), (wx1, wy1), (wx2, wy2));
        assert_eq!(clipped, expected);
    }

    #[test]
    fn test_clip_line_outside() {
        test_clip_line_helper(0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, None);
    }

    #[test]
    fn test_clip_line_inside() {
        test_clip_line_helper(
            0.0,
            0.0,
            1.0,
            1.0,
            -1.0,
            -1.0,
            1.5,
            1.5,
            Some(((0.0, 0.0), (1.0, 1.0))),
        );
    }

    #[test]
    fn test_clip_line_partial() {
        test_clip_line_helper(
            0.0,
            0.0,
            1.0,
            1.0,
            0.5,
            0.5,
            1.5,
            1.5,
            Some(((0.5, 0.5), (1.0, 1.0))),
        );
    }

    #[test]
    fn test_clip_label_multibyte_character() {
        // str.len() counts bytes, not characters (this is a bug in the original implementation)
        let clipped = clip_label("●", 160, -6, 168);
        assert_eq!(clipped, "●");
    }

    #[test]
    fn test_clip_label_negative_offset() {
        //  -2   0 1 2 3 4 5 6 7 8 9
        //  [        A B C]D E
        let clipped = clip_label("ABCDE", 2, -2, 7);
        assert_eq!(clipped, "AB…");
    }

    #[test]
    fn test_clip_label_internal() {
        // 0 1 2 3 4 5 6 7 8 9
        //  [  A B C D E  ]
        let clipped = clip_label("ABCDE", 2, 1, 7);
        assert_eq!(clipped, "ABCDE");
    }

    #[test]
    fn test_clip_label_external() {
        // 0 1 2 3 4 5 6 7 8 9
        //     A B  [     ]
        let clipped = clip_label("AB", 2, 5, 3);
        assert_eq!(clipped, "");
    }

    #[test]
    fn test_clip_label_left() {
        // 0 1 2 3 4 5 6 7 8 9
        //     A B[C D E F G H]
        let clipped = clip_label("ABCDEFGH", 2, 4, 10);
        assert_eq!(clipped, "…DEFGH");
    }

    #[test]
    fn test_clip_label_right() {
        // 0 1 2 3 4 5 6 7 8 9
        //[    A B C D E]F G H
        let clipped = clip_label("ABCDEFGH", 2, 0, 7);
        assert_eq!(clipped, "ABCD…");
    }

    #[test]
    fn test_clip_label_both() {
        // 0 1 2 3 4 5 6 7 8 9
        //     A B[C D E]F G H
        let clipped = clip_label("ABCDEFGH", 2, 4, 3);
        assert_eq!(clipped, "…D…");
    }

    #[test]
    fn test_rectangles_intersect() {
        // Overlapping rectangles (corners in standard order)
        assert!(rectangles_intersect(
            (0.0, 0.0),
            (2.0, 2.0),
            (1.0, 1.0),
            (3.0, 3.0)
        ));

        // Overlapping rectangles (corners in reverse order)
        assert!(rectangles_intersect(
            (2.0, 2.0),
            (0.0, 0.0), // bottom-right to top-left
            (3.0, 3.0),
            (1.0, 1.0) // bottom-right to top-left
        ));

        // Touching rectangles (edge) with mixed corner order
        assert!(rectangles_intersect(
            (2.0, 2.0),
            (0.0, 0.0), // reversed
            (2.0, 0.0),
            (4.0, 2.0) // standard
        ));

        // Touching rectangles (corner) with diagonal corners
        assert!(rectangles_intersect(
            (0.0, 2.0),
            (2.0, 0.0), // top-right to bottom-left
            (2.0, 2.0),
            (4.0, 4.0) // standard
        ));

        // Non-intersecting rectangles with mixed corners
        assert!(!rectangles_intersect(
            (1.0, 1.0),
            (0.0, 0.0), // reversed
            (3.0, 3.0),
            (2.0, 2.0) // reversed
        ));

        // One rectangle inside another with diagonal corners
        assert!(rectangles_intersect(
            (0.0, 4.0),
            (4.0, 0.0), // top-right to bottom-left
            (1.0, 2.0),
            (2.0, 1.0) // bottom-right to top-left
        ));
    }
}
