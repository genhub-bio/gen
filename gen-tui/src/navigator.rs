use std::hash::Hash;

use ratatui::{buffer::Buffer, layout::Rect, style::Style};

use crate::{
    frame_index::{Direction, FrameIndex},
    geometry::{Point, WorldPos, floor_half},
    graph_widget::style_cursor_cell,
    theme::current_theme,
    viewport_state::{ViewportState, WorldBuffer},
};

const NO_NEXT_LAYER_ERR: &str = "No next layer";
const NO_PREVIOUS_LAYER_ERR: &str = "No previous layer";

/// Semantic-only cursor state: node identity + fractional offset within it + visibility +
/// coarse mode. Its screen position is always derived from the current `FrameIndex`
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
    pub coarse_mode: bool,
}

impl<N> Default for CursorState<N> {
    fn default() -> Self {
        Self {
            node: None,
            fractional: (0.0, 0.0),
            visible: false,
            coarse_mode: true,
        }
    }
}

impl<N: Copy + Eq + Hash> CursorState<N> {
    /// Set the tracked node and its fractional offset.
    pub fn set_node(&mut self, node: N, fractional: (f64, f64)) {
        self.node = Some(node);
        self.fractional = fractional;
    }
}

/// Cursor navigation logic. Runs against a `FrameIndex` (screen rects + layer adjacency),
/// so `frame.neighbor(node, direction)` is a direct query instead of geometric probing.
pub struct Navigator;

impl Navigator {
    /// Move the cursor horizontally by `delta` screen cells: within the current node if the
    /// result stays in bounds, otherwise jump to the adjacent node in the next/previous layer,
    /// landing at that node's near edge. In coarse mode, a missing adjacent layer clamps the
    /// cursor to the current node's edge instead of failing.
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
                Ok(())
            }
            None if cursor.coarse_mode => {
                let edge_x = if delta > 0 { 1.0 } else { 0.0 };
                cursor.fractional = (edge_x, cursor.fractional.1);
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
    /// at that node's near edge (fractional-x preserved).
    pub fn move_vertical<N: Copy + Eq + Hash>(
        cursor: &mut CursorState<N>,
        delta: i64,
        frame: &FrameIndex<N>,
    ) -> Result<(), String> {
        let node = cursor.node.ok_or("No node associated with cursor")?;
        let rect = frame.rect_of(node).ok_or("Node not found in frame")?;
        let current = rect.point_at_fraction(cursor.fractional);
        let new_y = current.y + delta;

        if new_y >= rect.bottom() && new_y <= rect.top() {
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
                Ok(())
            }
            None => Err("No node found in same layer in that direction".to_string()),
        }
    }
}

/// Draws the cursor overlay: coarse mode restyles every cell of the node rect and draws
/// `⟨`/`⟩` flanking glyphs at mid-height; fine mode restyles one cell and draws `⌃` above it.
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

        if cursor.coarse_mode {
            for y in rect.bottom()..=rect.top() {
                for x in rect.left()..=rect.right() {
                    style_cursor_cell(&mut cursor_buffer, WorldPos::new(x, y), &theme);
                }
            }
            let ymid = (rect.bottom() + rect.top()) / 2;
            cursor_buffer.set_char_styled(
                WorldPos::new(rect.left() - 1, ymid),
                '⟨',
                indicator_style,
            );
            cursor_buffer.set_char_styled(
                WorldPos::new(rect.right() + 1, ymid),
                '⟩',
                indicator_style,
            );
        } else {
            let Point { x, y } = rect.point_at_fraction(cursor.fractional);
            style_cursor_cell(&mut cursor_buffer, WorldPos::new(x, y), &theme);
            cursor_buffer.set_char_styled(WorldPos::new(x, y - 1), '⌃', indicator_style);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::WorldRect;

    fn frame_with(rects: Vec<(u32, WorldRect, i32)>) -> FrameIndex<u32> {
        let placed = rects
            .into_iter()
            .map(|(id, rect, layer)| crate::frame_index::PlacedNode { id, rect, layer })
            .collect();
        FrameIndex::build(placed, WorldRect::from_coords(0, 0, 100, 100))
    }

    #[test]
    fn cursor_indicators_use_base0b() {
        let area = Rect::new(0, 0, 20, 10);
        let frame = frame_with(vec![(0, WorldRect::from_coords(5, 3, 8, 5), 0)]);
        let mut cursor = CursorState {
            node: Some(0),
            fractional: (0.5, 0.5),
            visible: true,
            coarse_mode: true,
        };
        let mut coarse_buffer = Buffer::empty(area);

        CursorOverlay::render(area, &mut coarse_buffer, &cursor, &frame);

        let mut chevrons = 0;
        for y in area.top()..area.bottom() {
            for x in area.left()..area.right() {
                let cell = &coarse_buffer[(x, y)];
                if matches!(cell.symbol(), "⟨" | "⟩") {
                    assert_eq!(cell.fg, current_theme()[0x0B]);
                    chevrons += 1;
                }
            }
        }
        assert_eq!(chevrons, 2);

        cursor.coarse_mode = false;
        let mut fine_buffer = Buffer::empty(area);
        CursorOverlay::render(area, &mut fine_buffer, &cursor, &frame);
        let caret = (area.top()..area.bottom())
            .flat_map(|y| (area.left()..area.right()).map(move |x| (x, y)))
            .find_map(|position| {
                let cell = &fine_buffer[position];
                (cell.symbol() == "⌃").then_some(cell)
            })
            .expect("should draw the fine cursor caret");
        assert_eq!(caret.fg, current_theme()[0x0B]);
    }

    #[test]
    fn test_move_horizontal_intra_node() {
        let frame = frame_with(vec![(0, WorldRect::from_coords(0, 0, 4, 2), 0)]);
        let mut cursor = CursorState {
            node: Some(0),
            fractional: (0.0, 0.5),
            visible: true,
            coarse_mode: false,
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
            coarse_mode: false,
        };
        Navigator::move_horizontal(&mut cursor, 1, &frame).expect("should jump to next layer");
        assert_eq!(cursor.node, Some(1));
        assert_eq!(cursor.fractional.0, 0.0);
    }

    #[test]
    fn test_move_horizontal_coarse_mode_clamps_at_boundary() {
        let frame = frame_with(vec![(0, WorldRect::from_coords(0, 0, 4, 2), 0)]);
        let mut cursor = CursorState {
            node: Some(0),
            fractional: (1.0, 0.5),
            visible: true,
            coarse_mode: true,
        };
        Navigator::move_horizontal(&mut cursor, 1000, &frame).expect("should clamp in coarse mode");
        assert_eq!(cursor.node, Some(0));
        assert_eq!(cursor.fractional, (1.0, 0.5));
    }

    #[test]
    fn test_move_horizontal_fine_mode_errors_at_boundary() {
        let frame = frame_with(vec![(0, WorldRect::from_coords(0, 0, 4, 2), 0)]);
        let mut cursor = CursorState {
            node: Some(0),
            fractional: (1.0, 0.5),
            visible: true,
            coarse_mode: false,
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
            coarse_mode: false,
        };
        Navigator::move_vertical(&mut cursor, 1, &frame).expect("should jump within layer");
        assert_eq!(cursor.node, Some(1));
        assert_eq!(cursor.fractional.1, 0.0);
    }
}
