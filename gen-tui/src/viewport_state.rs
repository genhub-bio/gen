use ratatui::{buffer::Buffer, layout::Rect, style::Style};
use unicode_segmentation::UnicodeSegmentation as _;
use unicode_width::UnicodeWidthStr as _;

use crate::geometry::{ViewportPos, WorldPos, WorldRect};

/// Controller for the Graph widget, managing camera and viewport bounds.
#[derive(Clone)]
pub struct ViewportState {
    /// Camera's current world offset (viewport center in world coordinates).
    pub camera_current: WorldPos,

    /// Viewport bounds in screen coordinates
    pub viewport_bounds: Rect,
}

impl Default for ViewportState {
    fn default() -> Self {
        Self::new()
    }
}

impl ViewportState {
    pub fn new() -> Self {
        ViewportState {
            camera_current: WorldPos::ZERO,
            viewport_bounds: Rect::new(0, 0, 0, 0), // Will be set during rendering
        }
    }

    /// Convert a world-space position to viewport coordinates (cell indices).
    /// Returns `Some(ViewportPos)` if inside [0..width) × [0..height), otherwise `None`.
    pub fn world_to_viewport(&self, world: WorldPos) -> Option<ViewportPos> {
        // Use camera_rect().min as origin for correct calculation
        let origin = self.camera_rect().min;
        let relative_x = world.x - origin.x;
        let relative_y = world.y - origin.y;

        // Check against viewport dimensions before casting to avoid u16 wraparound.
        if relative_x >= 0
            && relative_y >= 0
            && relative_x < self.viewport_bounds.width as i64
            && relative_y < self.viewport_bounds.height as i64
        {
            // Safe to convert since we've verified the values fit within viewport bounds
            let screen_x = relative_x as u16;
            let screen_y = relative_y as u16;
            return Some(ViewportPos::new(screen_x, screen_y));
        }
        None
    }

    /// Inverse of `world_to_viewport`: given a viewport cell, return the corresponding world pos.
    #[cfg(test)]
    pub(crate) fn viewport_to_world(&self, screen: ViewportPos) -> WorldPos {
        // Use camera_rect().min as origin for correct calculation
        let origin = self.camera_rect().min;
        WorldPos::new(origin.x + screen.x as i64, origin.y + screen.y as i64)
    }

    /// Convert a world position to terminal buffer coordinates.
    /// Returns `Some((x, y))` if the position is visible in the terminal, otherwise `None`.
    /// Handles viewport offset, large coordinates, and Y-axis flipping.
    pub(crate) fn world_to_terminal(&self, world_pos: WorldPos) -> Option<(u16, u16)> {
        // Handle uninitialized viewport bounds
        if self.viewport_bounds.width == 0 || self.viewport_bounds.height == 0 {
            return None;
        }

        // Calculate viewport-relative coordinates directly
        let origin = self.camera_rect().min;
        let relative_x = world_pos.x - origin.x;
        let relative_y = world_pos.y - origin.y;

        // Check if the position is within the visible viewport area
        if relative_x >= 0
            && relative_y >= 0
            && relative_x < self.viewport_bounds.width as i64
            && relative_y < self.viewport_bounds.height as i64
        {
            // Safe to convert since we've verified the values fit in u16
            let viewport_x = relative_x as u16;
            let viewport_y = relative_y as u16;

            // Convert to terminal coordinates with viewport offset and Y-axis flip
            let terminal_x = self.viewport_bounds.x + viewport_x;
            let terminal_y = self.viewport_bounds.y
                + self
                    .viewport_bounds
                    .height
                    .saturating_sub(1)
                    .saturating_sub(viewport_y);
            Some((terminal_x, terminal_y))
        } else {
            None
        }
    }

    /// Convert a continuous world coordinate to continuous buffer coordinates (x right, y down,
    /// cell `(column, row)` spanning `[column, column + 1) x [row, row + 1)`), without clipping
    /// to the viewport.
    pub(crate) fn world_to_terminal_continuous(&self, world: (f64, f64)) -> (f64, f64) {
        let origin = self.camera_rect().min;
        (
            f64::from(self.viewport_bounds.x) + world.0 - origin.x as f64,
            f64::from(self.viewport_bounds.y) + f64::from(self.viewport_bounds.height)
                - (world.1 - origin.y as f64),
        )
    }

    /// Returns the area the camera sees, i.e. the viewport in world coordinates
    pub(crate) fn camera_rect(&self) -> WorldRect {
        let center = self.camera_current;
        let size = (
            self.viewport_bounds.width as u64,
            self.viewport_bounds.height as u64,
        );
        WorldRect::from_center_and_size(center, size)
    }
}

/// A helper struct for writing characters directly to the viewport buffer using world coordinates.
/// Handles coordinate conversion and automatic clipping.
pub struct WorldBuffer<'a> {
    buffer: &'a mut Buffer,
    viewport_state: &'a ViewportState,
}

impl<'a> WorldBuffer<'a> {
    /// Create a new WorldBuffer
    pub fn new(buffer: &'a mut Buffer, state: &'a ViewportState) -> Self {
        Self {
            buffer,
            viewport_state: state,
        }
    }

    /// Get the currently visible world rectangle.
    pub fn visible_world_area(&self) -> WorldRect {
        self.viewport_state.camera_rect()
    }

    /// The world cells that map onto the terminal, or `None` while the viewport has no area.
    pub(crate) fn visible_cells(&self) -> Option<WorldRect> {
        let bounds = self.viewport_state.viewport_bounds;
        if bounds.width == 0 || bounds.height == 0 {
            return None;
        }
        let origin = self.viewport_state.camera_rect().min;
        Some(WorldRect::from_coords(
            origin.x,
            origin.y,
            origin.x + i64::from(bounds.width) - 1,
            origin.y + i64::from(bounds.height) - 1,
        ))
    }

    /// Intersect a world-space region with the current visible viewport area.
    #[cfg(test)]
    pub(crate) fn calculate_visible_area(&self, world_area: WorldRect) -> Option<WorldRect> {
        self.visible_world_area().intersection(&world_area)
    }

    /// Convert a world position to a viewport position
    pub fn world_to_viewport(&self, world_pos: WorldPos) -> Option<ViewportPos> {
        self.viewport_state.world_to_viewport(world_pos)
    }

    /// See [`ViewportState::world_to_terminal_continuous`].
    pub(crate) fn world_to_terminal_continuous(&self, world: (f64, f64)) -> (f64, f64) {
        self.viewport_state.world_to_terminal_continuous(world)
    }

    /// Set a single character at the specified world position.
    pub fn set_char(&mut self, world_pos: WorldPos, ch: char) {
        self.set_char_styled(world_pos, ch, Style::default())
    }

    /// Set a single character with style at the specified world position.
    pub fn set_char_styled(&mut self, world_pos: WorldPos, ch: char, style: Style) {
        let Some((buffer_x, buffer_y)) = self.viewport_state.world_to_terminal(world_pos) else {
            return;
        };

        if let Some(cell) = self.buffer.cell_mut((buffer_x, buffer_y)) {
            cell.set_char(ch);
            cell.set_style(style);
        }
    }

    /// Set a string starting at the specified world position, advancing horizontally.
    pub(crate) fn set_string(&mut self, world_pos: WorldPos, text: &str) {
        self.set_string_styled(world_pos, text, Style::default())
    }

    /// Set a string with style starting at the specified world position, advancing horizontally.
    ///
    /// Only the characters that land in the viewport are visited, so the cost follows the screen
    /// width rather than the length of the text (a node can be megabases long).
    pub fn set_string_styled(&mut self, world_pos: WorldPos, text: &str, style: Style) {
        let Some(visible) = self.visible_cells() else {
            return;
        };
        if world_pos.y < visible.min.y || world_pos.y > visible.max.y {
            return;
        }
        let first_column = world_pos.x.max(visible.min.x);
        let skipped = usize::try_from(first_column - world_pos.x).unwrap_or(0);
        let visible_columns = usize::try_from(visible.max.x - first_column + 1).unwrap_or(0);
        for (offset, ch) in text.chars().skip(skipped).take(visible_columns).enumerate() {
            let char_world_pos = WorldPos::new(first_column + offset as i64, world_pos.y);
            self.set_char_styled(char_world_pos, ch, style);
        }
    }

    /// Draw terminal graphemes at their cell widths, including wide and combining characters.
    pub fn set_graphemes_styled(&mut self, world_pos: WorldPos, text: &str, style: Style) {
        let mut column = world_pos.x;
        for grapheme in text.graphemes(true) {
            let width = grapheme.width() as i64;
            if let Some((buffer_x, buffer_y)) = self
                .viewport_state
                .world_to_terminal(WorldPos::new(column, world_pos.y))
                && let Some(cell) = self.buffer.cell_mut((buffer_x, buffer_y))
            {
                cell.set_symbol(grapheme);
                cell.set_style(style);
            }
            for offset in 1..width {
                self.set_char_styled(WorldPos::new(column + offset, world_pos.y), ' ', style);
            }
            column += width;
        }
    }

    /// Fill a rectangular area in world coordinates with the specified character.
    pub fn fill_rect(&mut self, world_rect: WorldRect, ch: char) {
        self.fill_rect_styled(world_rect, ch, Style::default())
    }

    /// Fill a rectangular area in world coordinates with the specified character and style.
    pub fn fill_rect_styled(&mut self, world_rect: WorldRect, ch: char, style: Style) {
        let Some(visible) = self
            .visible_cells()
            .and_then(|visible| visible.intersection(&world_rect))
        else {
            return;
        };
        for y in visible.min.y..=visible.max.y {
            for x in visible.min.x..=visible.max.x {
                self.set_char_styled(WorldPos::new(x, y), ch, style);
            }
        }
    }

    /// Get a single character at the specified world position.
    /// Returns Some(char) if the position is within viewport bounds, None otherwise.
    pub fn get_char(&self, world_pos: WorldPos) -> Option<char> {
        let (buffer_x, buffer_y) = self.viewport_state.world_to_terminal(world_pos)?;
        let cell = self.buffer.cell((buffer_x, buffer_y)).unwrap();
        Some(cell.symbol().chars().next().unwrap_or(' '))
    }

    /// Get a single character and its style at the specified world position.
    /// Returns Some((char, Style)) if the position is within viewport bounds, None otherwise.
    pub fn get_char_styled(&self, world_pos: WorldPos) -> Option<(char, Style)> {
        let (buffer_x, buffer_y) = self.viewport_state.world_to_terminal(world_pos)?;
        let cell = self.buffer.cell((buffer_x, buffer_y)).unwrap();
        let ch = cell.symbol().chars().next().unwrap_or(' ');
        Some((ch, cell.style()))
    }
}

#[cfg(test)]
mod tests {
    use ratatui::{buffer::Buffer, layout::Rect};

    use super::*;

    #[test]
    fn test_coordinate_conversions() {
        let mut state = ViewportState::new();
        state.camera_current = WorldPos::new(100, 100);
        state.viewport_bounds = Rect::new(0, 0, 10, 10);

        // Camera rect origin: (100, 100) - ((10-1)/2, (10-1)/2) = (100, 100) - (4, 4) = (96, 96)
        let origin = state.camera_rect().min;
        assert_eq!(origin, WorldPos::new(96, 96));

        // World to viewport conversion
        // world_pos - origin = (104, 103) - (96, 96) = (8, 7)
        let world_pos = WorldPos::new(104, 103);
        let viewport_pos = state.world_to_viewport(world_pos);
        assert_eq!(viewport_pos, Some(ViewportPos::new(8, 7)));

        // Viewport to world conversion
        // origin + screen_pos = (96, 96) + (5, 2) = (101, 98)
        let screen_pos = ViewportPos::new(5, 2);
        let world_pos = state.viewport_to_world(screen_pos);
        assert_eq!(world_pos, WorldPos::new(101, 98));

        // Round-trip test
        let original = WorldPos::new(99, 104);
        if let Some(screen) = state.world_to_viewport(original) {
            let back_to_world = state.viewport_to_world(screen);
            assert_eq!(back_to_world, original);
        }
    }

    fn small_viewport_state() -> ViewportState {
        ViewportState {
            camera_current: WorldPos::new(5, 0),
            viewport_bounds: Rect::new(0, 0, 11, 1),
            ..ViewportState::default()
        }
    }

    #[test]
    fn test_set_string_styled_draws_only_the_visible_part_of_a_long_string() {
        let state = small_viewport_state();
        let mut buffer = Buffer::empty(state.viewport_bounds);
        let text: String = "ACGT".chars().cycle().take(2_000_000).collect();
        WorldBuffer::new(&mut buffer, &state).set_string_styled(
            WorldPos::new(-1_000_000, 0),
            &text,
            Style::default(),
        );
        // Column 0 of the screen is world x = 0, which is 1_000_000 characters in: a multiple
        // of four, so the string reads "ACGT..." from there.
        let row: String = (0..11)
            .map(|x| buffer[(x, 0)].symbol().to_string())
            .collect();
        assert_eq!(row, "ACGTACGTACG");
    }

    #[test]
    fn test_set_string_styled_clips_at_both_edges() {
        let state = small_viewport_state();
        let mut buffer = Buffer::empty(state.viewport_bounds);
        WorldBuffer::new(&mut buffer, &state).set_string_styled(
            WorldPos::new(8, 0),
            "ACGTACGT",
            Style::default(),
        );
        let row: String = (0..11)
            .map(|x| buffer[(x, 0)].symbol().to_string())
            .collect();
        assert_eq!(row, "        ACG");
    }

    #[test]
    fn test_fill_rect_styled_fills_only_the_visible_part_of_a_huge_rect() {
        let state = small_viewport_state();
        let mut buffer = Buffer::empty(state.viewport_bounds);
        WorldBuffer::new(&mut buffer, &state).fill_rect_styled(
            WorldRect::from_coords(-5_000_000_000, 0, 5_000_000_000, 0),
            '#',
            Style::default(),
        );
        let row: String = (0..11)
            .map(|x| buffer[(x, 0)].symbol().to_string())
            .collect();
        assert_eq!(row, "###########");
    }

    #[test]
    fn test_world_to_viewport_does_not_wrap_large_coordinates() {
        let mut state = ViewportState::new();
        state.viewport_bounds = Rect::new(0, 0, 80, 20);
        state.camera_current = WorldPos::new(40_000, 0);

        // This point is far outside the visible range but would wrap into range with a u16 cast.
        // With viewport width 80 and camera origin ~39961, adding 131_082 makes relative_x = 131_121,
        // which wraps to 49 if cast to u16.
        let wrapped_candidate = WorldPos::new(171_082, 0);
        assert_eq!(state.world_to_viewport(wrapped_candidate), None);
        assert_eq!(state.world_to_terminal(wrapped_candidate), None);

        // Visible points still convert correctly.
        let visible = state.camera_current;
        assert!(state.world_to_viewport(visible).is_some());
        assert!(state.world_to_terminal(visible).is_some());
    }

    #[test]
    fn test_world_buffer_writer_coordinate_conversion() {
        let mut state = ViewportState::new();
        state.camera_current = WorldPos::new(10, 10);
        state.viewport_bounds = Rect::new(0, 0, 20, 10);

        let area = Rect::new(0, 0, 20, 10);
        let mut buffer = Buffer::empty(area);
        let mut writer = WorldBuffer::new(&mut buffer, &state);

        // Write at world position (15, 12)
        // Camera is at (10, 10) with viewport 20x10
        // Origin = (10, 10) - ((20-1)/2, (10-1)/2) = (10, 10) - (9, 4) = (1, 6)
        // So world (15, 12) - origin (1, 6) = viewport (14, 6)
        writer.set_char(WorldPos::new(15, 12), 'X');

        // Verify it was written correctly
        // The Y-axis is flipped: viewport y=6 becomes buffer y = 10-1-6 = 3
        assert_eq!(buffer[(14, 3)].symbol(), "X");
    }

    #[test]
    fn test_world_buffer_writer_string_operations() {
        let mut state = ViewportState::new();
        state.viewport_bounds = Rect::new(0, 0, 20, 10); // Set viewport bounds
        let area = Rect::new(0, 0, 20, 10);
        let mut buffer = Buffer::empty(area);
        let mut writer = WorldBuffer::new(&mut buffer, &state);

        // Write a string
        writer.set_string(WorldPos::new(5, 3), "Hello");

        // Verify each character
        assert_eq!(writer.get_char(WorldPos::new(5, 3)), Some('H'));
        assert_eq!(writer.get_char(WorldPos::new(6, 3)), Some('e'));
        assert_eq!(writer.get_char(WorldPos::new(7, 3)), Some('l'));
        assert_eq!(writer.get_char(WorldPos::new(8, 3)), Some('l'));
        assert_eq!(writer.get_char(WorldPos::new(9, 3)), Some('o'));
    }

    #[test]
    fn test_world_buffer_graphemes_keep_terminal_columns() {
        let mut state = ViewportState::new();
        state.viewport_bounds = Rect::new(0, 0, 20, 10);
        let mut buffer = Buffer::empty(state.viewport_bounds);
        let mut writer = WorldBuffer::new(&mut buffer, &state);
        writer.set_graphemes_styled(
            WorldPos::new(0, 0),
            "e\u{301}漢B",
            ratatui::style::Style::default(),
        );
        assert_eq!(writer.get_char(WorldPos::new(0, 0)), Some('e'));
        assert_eq!(writer.get_char(WorldPos::new(1, 0)), Some('漢'));
        assert_eq!(writer.get_char(WorldPos::new(2, 0)), Some(' '));
        assert_eq!(writer.get_char(WorldPos::new(3, 0)), Some('B'));
        drop(writer);
        let (column, row) = state.world_to_terminal(WorldPos::new(0, 0)).unwrap();
        assert_eq!(buffer[(column, row)].symbol(), "e\u{301}");
    }

    #[test]
    fn test_world_buffer_writer_clipping() {
        let state = ViewportState::new();
        let area = Rect::new(0, 0, 5, 5);
        let mut buffer = Buffer::empty(area);
        let mut writer = WorldBuffer::new(&mut buffer, &state);

        // Try to write outside bounds
        writer.set_char(WorldPos::new(10, 10), 'X');

        // Should not crash and character should not appear
        // (All buffer positions should remain at default)
        for y in 0..5 {
            for x in 0..5 {
                assert_eq!(buffer[(x, y)].symbol(), " ");
            }
        }
    }

    #[test]
    fn test_world_buffer_writer_read_write_consistency() {
        let mut state = ViewportState::new();
        state.viewport_bounds = Rect::new(0, 0, 8, 8); // Set viewport bounds
        let area = Rect::new(0, 0, 8, 8);
        let mut buffer = Buffer::empty(area);
        let mut writer = WorldBuffer::new(&mut buffer, &state);

        // Write and read back
        // With camera at (0,0) and viewport 8x8, visible range is [-4, 4) x [-4, 4)
        writer.set_char(WorldPos::new(2, 3), 'A');
        assert_eq!(writer.get_char(WorldPos::new(2, 3)), Some('A'));

        // Read from empty position
        assert_eq!(writer.get_char(WorldPos::new(1, 1)), Some(' '));

        // Read from outside viewport
        assert_eq!(writer.get_char(WorldPos::new(20, 20)), None);
    }
}
