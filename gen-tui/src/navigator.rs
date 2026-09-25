use std::hash::Hash;

use ratatui::{buffer::Buffer, layout::Rect, style::Style};

use crate::{
    frame_index::{Direction, FrameIndex},
    geometry::{Point, WorldPos, WorldRect, floor_half},
    graph_widget::style_cursor_cell,
    theme::current_theme,
    viewport_state::{ViewportState, WorldBuffer},
};

const NO_NEXT_LAYER_ERR: &str = "No next layer";
const NO_PREVIOUS_LAYER_ERR: &str = "No previous layer";

/// Semantic-only cursor state: node identity + fractional offset within it + visibility.
/// Its screen position is always derived from the current `FrameIndex`
/// (`frame.rect_of(cursor.node)` + fractional offset), never stored - a stored copy would go
/// stale on zoom, resize, camera movement, or a new window.
#[derive(Debug, Clone, Copy)]
pub struct CursorState<N> {
    pub node: Option<N>,
    /// Fractional position within the node rectangle (0.0 to 1.0 on each axis, relative to
    /// the rendered rect's bottom-left). Its meaning shifts across detail levels - documented
    /// known limitation, not fixed here.
    pub fractional: (f64, f64),
    pub visible: bool,
}

impl<N> Default for CursorState<N> {
    fn default() -> Self {
        Self {
            node: None,
            fractional: (0.0, 0.0),
            visible: false,
        }
    }
}

impl<N: Copy + Eq + Hash> CursorState<N> {
    /// Set the tracked node and its fractional offset.
    pub fn set_node(&mut self, node: N, fractional: (f64, f64)) {
        self.node = Some(node);
        self.fractional = fractional;
    }

    /// Put the cursor on `row` of `rect` (counted up from its bottom), keeping its column.
    pub fn hold_to_row(&mut self, rect: WorldRect, row: u64) {
        let y = rect.bottom().saturating_add_unsigned(row).min(rect.top());
        self.fractional.1 = rect.fraction_of(WorldPos::new(rect.left(), y)).1;
    }

    /// Put the cursor on the only row its node's renderer lets it sit on, if the node is placed
    /// in `frame` and restricted to one (see `NodeRenderer::cursor_row`).
    pub fn hold_to_cursor_row(&mut self, frame: &FrameIndex<N>) {
        let Some(node) = self.node else {
            return;
        };
        if let (Some(rect), Some(row)) = (frame.rect_of(node), frame.cursor_row(node)) {
            self.hold_to_row(rect, row);
        }
    }
}

/// Cursor navigation logic. Runs against a `FrameIndex` (screen rects + layer adjacency),
/// so `frame.neighbor(node, direction)` is a direct query instead of geometric probing.
pub struct Navigator;

impl Navigator {
    /// Move the cursor horizontally by `delta` screen cells: within the current node if the
    /// result stays in bounds, otherwise jump to the adjacent node in the next/previous layer,
    /// landing at that node's near edge.
    pub fn move_horizontal<N: Copy + Eq + Hash>(
        cursor: &mut CursorState<N>,
        delta: i64,
        frame: &FrameIndex<N>,
    ) -> Result<(), String> {
        let node = cursor.node.ok_or("No node associated with cursor")?;
        let rect = frame.rect_of(node).ok_or("Node not found in frame")?;
        let current = rect.point_at_fraction(cursor.fractional);
        let new_x = current.x + delta;

        if new_x >= rect.left() && new_x <= rect.right() {
            cursor.fractional = rect.fraction_of(WorldPos::new(new_x, current.y));
            cursor.hold_to_cursor_row(frame);
            return Ok(());
        }

        let direction = if delta > 0 {
            Direction::Right
        } else {
            Direction::Left
        };
        match frame.neighbor(node, direction) {
            Some(target) => {
                let target_frac_x = if delta > 0 { 0.0 } else { 1.0 };
                cursor.set_node(target, (target_frac_x, cursor.fractional.1));
                cursor.hold_to_cursor_row(frame);
                Ok(())
            }
            None => Err(if delta > 0 {
                NO_NEXT_LAYER_ERR.to_string()
            } else {
                NO_PREVIOUS_LAYER_ERR.to_string()
            }),
        }
    }

    /// Move the cursor vertically by `delta` screen cells: within the current node if the
    /// result stays in bounds, otherwise jump to the nearest node in the same layer, landing
    /// at that node's near edge (fractional-x preserved). A node whose renderer holds the
    /// cursor to one row has no other row to move to, so the move always jumps, landing on
    /// the target's own cursor row when it has one.
    pub fn move_vertical<N: Copy + Eq + Hash>(
        cursor: &mut CursorState<N>,
        delta: i64,
        frame: &FrameIndex<N>,
    ) -> Result<(), String> {
        let node = cursor.node.ok_or("No node associated with cursor")?;
        let rect = frame.rect_of(node).ok_or("Node not found in frame")?;
        let current = rect.point_at_fraction(cursor.fractional);
        let new_y = current.y + delta;

        if frame.cursor_row(node).is_none() && new_y >= rect.bottom() && new_y <= rect.top() {
            cursor.fractional = rect.fraction_of(WorldPos::new(current.x, new_y));
            return Ok(());
        }

        let direction = if delta > 0 {
            Direction::Up
        } else {
            Direction::Down
        };
        match frame.neighbor(node, direction) {
            Some(target) => {
                let target_frac_y = if delta > 0 { 0.0 } else { 1.0 };
                cursor.set_node(target, (cursor.fractional.0, target_frac_y));
                cursor.hold_to_cursor_row(frame);
                Ok(())
            }
            None => Err("No node found in same layer in that direction".to_string()),
        }
    }

    /// Move the cursor to the nearest stop column past it in `direction` (`Left` or
    /// `Right`), such as where an annotation starts. `stops(node)` lists a node's stops as
    /// columns counted from its rect's left edge. The search walks the nodes the way
    /// `move_horizontal` does: the rest of the current node, then `frame.neighbor(node,
    /// direction)` and on, so a fork is resolved exactly as stepping the cursor across it
    /// would be. The cursor lands on the node's cursor row (see `NodeRenderer::cursor_row`),
    /// or its middle row when the renderer leaves every row open.
    /// When the walk runs out of placed nodes the cursor is left unchanged and an error is
    /// returned.
    pub fn move_to_stop<N: Copy + Eq + Hash>(
        cursor: &mut CursorState<N>,
        direction: Direction,
        frame: &FrameIndex<N>,
        stops: impl Fn(N) -> Vec<i64>,
    ) -> Result<(), String> {
        let forward = match direction {
            Direction::Right => true,
            Direction::Left => false,
            Direction::Up | Direction::Down => {
                return Err("Stops are only searched horizontally".to_string());
            }
        };
        let mut node = cursor.node.ok_or("No node associated with cursor")?;
        let rect = frame.rect_of(node).ok_or("Node not found in frame")?;
        // Column of the cursor within the node being searched; stops must lie strictly past it.
        let mut from = rect.point_at_fraction(cursor.fractional).x - rect.left();
        loop {
            let rect = frame.rect_of(node).ok_or("Node not found in frame")?;
            let columns = stops(node)
                .into_iter()
                .filter(|column| (0..=rect.width()).contains(column));
            let target = if forward {
                columns.filter(|column| *column > from).min()
            } else {
                columns.filter(|column| *column < from).max()
            };
            if let Some(column) = target {
                let x = rect
                    .fraction_of(WorldPos::new(rect.left() + column, rect.bottom()))
                    .0;
                cursor.set_node(node, (x, 0.5));
                cursor.hold_to_cursor_row(frame);
                return Ok(());
            }
            node = frame.neighbor(node, direction).ok_or(if forward {
                NO_NEXT_LAYER_ERR
            } else {
                NO_PREVIOUS_LAYER_ERR
            })?;
            let width = frame
                .rect_of(node)
                .ok_or("Node not found in frame")?
                .width();
            from = if forward { -1 } else { width + 1 };
        }
    }
}

/// Draws the cursor overlay: restyles the cursor's cell and draws `⌃` below it.
/// A view without a cursor (not visible, or no node placed in `frame`) simply skips drawing.
pub struct CursorOverlay;

impl CursorOverlay {
    pub fn render<N: Copy + Eq + Hash>(
        area: Rect,
        buf: &mut Buffer,
        cursor: &CursorState<N>,
        frame: &FrameIndex<N>,
    ) {
        if !cursor.visible {
            return;
        }
        let Some(node) = cursor.node else {
            return;
        };
        let Some(rect) = frame.rect_of(node) else {
            return;
        };

        // Build a throwaway ViewportState whose camera maps `frame`'s screen-space (origin at
        // area's bottom-left, Y-up) directly onto `area`'s terminal cells.
        let half_width = floor_half(area.width as i64);
        let half_height = floor_half(area.height as i64);
        let mut viewport_state = ViewportState::new();
        viewport_state.camera_current = WorldPos::new(half_width, half_height);
        viewport_state.viewport_bounds = area;

        let theme = current_theme();
        let indicator_style = Style::default().fg(theme[0x0B]);
        let mut cursor_buffer = WorldBuffer::new(buf, &viewport_state);

        let Point { x, y } = rect.point_at_fraction(cursor.fractional);
        style_cursor_cell(&mut cursor_buffer, WorldPos::new(x, y), &theme);
        cursor_buffer.set_char_styled(WorldPos::new(x, y - 1), '⌃', indicator_style);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_index::PlacedNode;

    fn frame_with(rects: Vec<(u32, WorldRect, i32)>) -> FrameIndex<u32> {
        let placed = rects
            .into_iter()
            .map(|(id, rect, layer)| PlacedNode {
                id,
                rect,
                layer,
                cursor_row: None,
            })
            .collect();
        FrameIndex::build(placed, WorldRect::from_coords(0, 0, 100, 100))
    }

    #[test]
    fn cursor_indicators_use_base0b() {
        let area = Rect::new(0, 0, 20, 10);
        let frame = frame_with(vec![(0, WorldRect::from_coords(5, 3, 8, 5), 0)]);
        let cursor = CursorState {
            node: Some(0),
            fractional: (0.5, 0.5),
            visible: true,
        };
        let mut fine_buffer = Buffer::empty(area);
        CursorOverlay::render(area, &mut fine_buffer, &cursor, &frame);
        let caret = (area.top()..area.bottom())
            .flat_map(|y| (area.left()..area.right()).map(move |x| (x, y)))
            .find_map(|position| {
                let cell = &fine_buffer[position];
                (cell.symbol() == "⌃").then_some(cell)
            })
            .expect("should draw the cursor caret");
        assert_eq!(caret.fg, current_theme()[0x0B]);
    }

    #[test]
    fn test_move_horizontal_intra_node() {
        let frame = frame_with(vec![(0, WorldRect::from_coords(0, 0, 4, 2), 0)]);
        let mut cursor = CursorState {
            node: Some(0),
            fractional: (0.0, 0.5),
            visible: true,
        };
        Navigator::move_horizontal(&mut cursor, 1, &frame).expect("should move within node");
        assert!(cursor.fractional.0 > 0.0);
        assert_eq!(cursor.node, Some(0));
    }

    #[test]
    fn test_move_horizontal_jumps_to_adjacent_layer() {
        let frame = frame_with(vec![
            (0, WorldRect::from_coords(0, 0, 4, 2), 0),
            (1, WorldRect::from_coords(10, 0, 14, 2), 1),
        ]);
        let mut cursor = CursorState {
            node: Some(0),
            fractional: (1.0, 0.5),
            visible: true,
        };
        Navigator::move_horizontal(&mut cursor, 1, &frame).expect("should jump to next layer");
        assert_eq!(cursor.node, Some(1));
        assert_eq!(cursor.fractional.0, 0.0);
    }

    #[test]
    fn test_move_horizontal_errors_at_boundary() {
        let frame = frame_with(vec![(0, WorldRect::from_coords(0, 0, 4, 2), 0)]);
        let mut cursor = CursorState {
            node: Some(0),
            fractional: (1.0, 0.5),
            visible: true,
        };
        let result = Navigator::move_horizontal(&mut cursor, 1000, &frame);
        assert!(result.is_err());
    }

    #[test]
    fn test_move_vertical_jumps_to_same_layer() {
        let frame = frame_with(vec![
            (0, WorldRect::from_coords(0, 0, 4, 2), 0),
            (1, WorldRect::from_coords(0, 10, 4, 12), 0),
        ]);
        let mut cursor = CursorState {
            node: Some(0),
            fractional: (0.3, 1.0),
            visible: true,
        };
        Navigator::move_vertical(&mut cursor, 1, &frame).expect("should jump within layer");
        assert_eq!(cursor.node, Some(1));
        assert_eq!(cursor.fractional.1, 0.0);
    }

    mod stops {
        use super::*;

        /// A fork: `first` in layer 0, then `lower` and `upper` side by side in layer 1.
        /// `upper`'s stop is further left on screen than `lower`'s, so only the cursor's
        /// neighbour rule decides which branch a stop search follows.
        fn fork(first_row: i64) -> FrameIndex<u32> {
            frame_with(vec![
                (0, WorldRect::from_coords(0, first_row, 8, first_row + 2), 0),
                (1, WorldRect::from_coords(12, 0, 20, 2), 1),
                (2, WorldRect::from_coords(12, 10, 20, 12), 1),
            ])
        }

        fn stops(node: u32) -> Vec<i64> {
            match node {
                0 => vec![2, 6],
                1 => vec![5],
                _ => vec![1],
            }
        }

        fn cursor_at(node: u32, column_fraction: f64) -> CursorState<u32> {
            CursorState {
                node: Some(node),
                fractional: (column_fraction, 0.0),
                visible: true,
            }
        }

        fn column(cursor: &CursorState<u32>, frame: &FrameIndex<u32>) -> (u32, i64) {
            let node = cursor.node.expect("should have a node");
            let rect = frame.rect_of(node).expect("should place the node");
            (
                node,
                rect.point_at_fraction(cursor.fractional).x - rect.left(),
            )
        }

        #[test]
        fn test_move_to_stop_steps_through_stops_in_the_current_node() {
            let frame = fork(0);
            let mut cursor = cursor_at(0, 0.0);
            Navigator::move_to_stop(&mut cursor, Direction::Right, &frame, stops)
                .expect("should find a stop");
            assert_eq!(column(&cursor, &frame), (0, 2));
            assert_eq!(cursor.fractional.1, 0.5, "should land on the sequence row");
            Navigator::move_to_stop(&mut cursor, Direction::Right, &frame, stops)
                .expect("should find a stop");
            assert_eq!(column(&cursor, &frame), (0, 6));
            Navigator::move_to_stop(&mut cursor, Direction::Left, &frame, stops)
                .expect("should find a stop");
            assert_eq!(column(&cursor, &frame), (0, 2));
        }

        #[test]
        fn test_move_to_stop_follows_the_cursor_neighbor_at_a_fork() {
            // From a node level with `lower`, the search takes `lower` even though `upper`'s
            // stop sits further left on screen.
            let frame = fork(0);
            let mut cursor = cursor_at(0, 1.0);
            Navigator::move_to_stop(&mut cursor, Direction::Right, &frame, stops)
                .expect("should find a stop");
            assert_eq!(column(&cursor, &frame), (1, 5));

            // Raised level with `upper`, the same search takes `upper`.
            let frame = fork(10);
            let mut cursor = cursor_at(0, 1.0);
            Navigator::move_to_stop(&mut cursor, Direction::Right, &frame, stops)
                .expect("should find a stop");
            assert_eq!(column(&cursor, &frame), (2, 1));

            Navigator::move_to_stop(&mut cursor, Direction::Left, &frame, stops)
                .expect("should find a stop");
            assert_eq!(column(&cursor, &frame), (0, 6));
        }

        #[test]
        fn test_move_to_stop_lands_on_the_cursor_row() {
            // Two 5-row nodes whose renderer holds the cursor to their second row from the
            // bottom, the way a sequence row sits between annotation lanes.
            let placed = [(0, 0), (1, 12)]
                .into_iter()
                .map(|(id, left)| PlacedNode {
                    id,
                    rect: WorldRect::from_coords(left, 0, left + 8, 4),
                    layer: id as i32,
                    cursor_row: Some(1),
                })
                .collect();
            let frame = FrameIndex::build(placed, WorldRect::from_coords(0, 0, 100, 100));
            let mut cursor = cursor_at(0, 1.0);
            Navigator::move_to_stop(&mut cursor, Direction::Right, &frame, stops)
                .expect("should find a stop");
            assert_eq!(column(&cursor, &frame), (1, 5));
            let rect = frame.rect_of(1).expect("should place the node");
            assert_eq!(
                rect.point_at_fraction(cursor.fractional).y,
                rect.bottom() + 1
            );
        }

        #[test]
        fn test_move_to_stop_stops_at_the_last_placed_node() {
            let frame = fork(0);
            let mut cursor = cursor_at(1, 1.0);
            let result = Navigator::move_to_stop(&mut cursor, Direction::Right, &frame, stops);
            assert!(result.is_err(), "should find no stop past the last layer");
            assert_eq!(
                column(&cursor, &frame),
                (1, 8),
                "should leave the cursor in place"
            );
        }
    }
}
