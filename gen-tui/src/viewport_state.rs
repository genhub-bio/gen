use ratatui::{buffer::Buffer, layout::Rect, style::Style};

use crate::geometry::{ViewportPos, WorldPos, WorldRect};

/// Controller for the Graph widget, managing camera and viewport bounds.
#[derive(Clone)]
pub struct ViewportState {
    /// Camera's current world offset (viewport center in world coordinates).
    pub camera_current: WorldPos,

    /// Viewport bounds in screen coordinates
    pub viewport_bounds: Rect,

    /// Whether this viewport has input focus; only then do scroll events move the camera.
    pub has_focus: bool,
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
            has_focus: true, // Enable focus by default for keyboard input
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
    pub fn viewport_to_world(&self, screen: ViewportPos) -> WorldPos {
        // Use camera_rect().min as origin for correct calculation
        let origin = self.camera_rect().min;
        WorldPos::new(origin.x + screen.x as i64, origin.y + screen.y as i64)
    }

    /// Convert a world position to terminal buffer coordinates.
    /// Returns `Some((x, y))` if the position is visible in the terminal, otherwise `None`.
    /// Handles viewport offset, large coordinates, and Y-axis flipping.
    pub fn world_to_terminal(&self, world_pos: WorldPos) -> Option<(u16, u16)> {
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

    /// Returns the area the camera sees, i.e. the viewport in world coordinates
    pub fn camera_rect(&self) -> WorldRect {
        let center = self.camera_current;
        let size = (
            self.viewport_bounds.width as u64,
            self.viewport_bounds.height as u64,
        );
        WorldRect::from_center_and_size(center, size)
    }

    /// Give this viewport input focus. Scroll events and key events will move the camera.
    pub fn focus(&mut self) {
        self.has_focus = true;
    }

    /// Remove input focus. Scroll events are ignored.
    #[cfg(test)]
    pub(crate) fn blur(&mut self) {
        self.has_focus = false;
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

    /// Intersect a world-space region with the current visible viewport area.
    #[cfg(test)]
    pub(crate) fn calculate_visible_area(&self, world_area: WorldRect) -> Option<WorldRect> {
        self.visible_world_area().intersection(&world_area)
    }

    /// Convert a world position to a viewport position
    pub fn world_to_viewport(&self, world_pos: WorldPos) -> Option<ViewportPos> {
        self.viewport_state.world_to_viewport(world_pos)
    }

    /// Convert a viewport position to world coordinates
    pub fn viewport_to_world(&self, screen: ViewportPos) -> WorldPos {
        self.viewport_state.viewport_to_world(screen)
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
    pub fn set_string_styled(&mut self, world_pos: WorldPos, text: &str, style: Style) {
        for (i, ch) in text.chars().enumerate() {
            let char_world_pos = WorldPos::new(world_pos.x + i as i64, world_pos.y);
            self.set_char_styled(char_world_pos, ch, style);
        }
    }

    /// Fill a rectangular area in world coordinates with the specified character.
    pub fn fill_rect(&mut self, world_rect: WorldRect, ch: char) {
        self.fill_rect_styled(world_rect, ch, Style::default())
    }

    /// Fill a rectangular area in world coordinates with the specified character and style.
    pub fn fill_rect_styled(&mut self, world_rect: WorldRect, ch: char, style: Style) {
        for y in world_rect.min.y..=world_rect.max.y {
            for x in world_rect.min.x..=world_rect.max.x {
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
    fn test_focus_management() {
        let mut state = ViewportState::new();
        assert!(state.has_focus); // Focus is enabled by default for keyboard input

        state.blur();
        assert!(!state.has_focus);

        state.focus();
        assert!(state.has_focus);
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
