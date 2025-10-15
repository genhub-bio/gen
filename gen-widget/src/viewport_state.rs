use std::time::Duration;

use ratatui::{buffer::Buffer, layout::Rect, style::Style};

use crate::{
    animation::Animation,
    geometry::{ViewportPos, WorldPos, WorldRect},
};

/// Controller for the Graph widget, managing camera and cursor positions, animations, and zones.
pub struct ViewportState {
    /// Dead Zone as a fraction of viewport radius (0.0 to 1.0)
    /// 0.0 = no dead zone, 1.0 = dead zone extends to viewport edge
    pub dead_zone_fraction: (f32, f32),

    /// Soft Zone as a fraction of viewport radius (0.0 to 1.0)
    /// Must be >= dead_zone_fraction. 0.0 = no soft zone, 1.0 = soft zone extends to viewport edge
    pub soft_zone_fraction: (f32, f32),

    /// Optional world boundaries: camera_current is clamped to this rect if present.
    pub world_bounds: Option<WorldRect>,

    /// Camera's current and target world offsets.
    pub camera_current: WorldPos,
    pub camera_target: WorldPos,
    pub camera_anim: Option<Animation>,

    /// Viewport bounds in screen coordinates
    pub viewport_bounds: Rect,

    /// Whether this viewport has input focus; only then do scroll events move the camera.
    pub has_focus: bool,

    /// Whether we are currently panning; when true, multizone cursor/camera pushing is disabled
    pub panning: bool,
}

impl Default for ViewportState {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper function to clamp coordinates within world bounds if present
fn clamp_to_bounds(pos: WorldPos, bounds: Option<WorldRect>) -> WorldPos {
    if let Some(rect) = bounds {
        WorldPos::new(
            pos.x.clamp(rect.min.x, rect.max.x),
            pos.y.clamp(rect.min.y, rect.max.y),
        )
    } else {
        pos
    }
}

impl ViewportState {
    pub fn new() -> Self {
        ViewportState {
            dead_zone_fraction: (0.6, 0.6), // 60% of viewport radius for dead zone
            soft_zone_fraction: (0.8, 0.8), // 80% of viewport radius for soft zone
            world_bounds: None,
            camera_current: WorldPos::ZERO,
            camera_target: WorldPos::ZERO,
            camera_anim: None,
            has_focus: true, // Enable focus by default for keyboard input
            viewport_bounds: Rect::new(0, 0, 0, 0), // Will be set during rendering
            panning: false,
        }
    }

    /// Move the camera to a new world-space offset over `duration` (smoothly).
    /// Clamps the final target to `world_bounds` if present.
    pub fn move_camera_to(&mut self, target: WorldPos, duration: Duration) {
        let start = self.camera_current;
        let end = clamp_to_bounds(target, self.world_bounds);
        self.camera_target = end;
        self.camera_anim = Some(Animation::new(
            start,
            end,
            duration,
            tachyonfx::Interpolation::CubicOut,
        ));
    }

    /// Handle a scroll event. `dx`/`dy` are deltas in world units.
    pub fn handle_mouse_scroll(&mut self, dx: i64, dy: i64, duration: Duration) {
        if !self.has_focus {
            return;
        }
        // Enable panning mode when scrolling
        self.panning = true;
        let new_target = WorldPos::new(self.camera_target.x + dx, self.camera_target.y + dy);
        self.move_camera_to(new_target, duration);
    }

    /// Check if a world position is currently visible in the viewport
    pub fn is_visible(&self, world_pos: WorldPos) -> bool {
        self.world_to_viewport(world_pos).is_some()
    }

    /// Get the world rectangle that corresponds to the current viewport
    pub fn visible_world_rect(&self) -> WorldRect {
        let top_left = self.viewport_to_world(ViewportPos::ZERO);
        let bottom_right = self.viewport_to_world(ViewportPos::new(
            self.viewport_bounds.width.saturating_sub(1),
            self.viewport_bounds.height.saturating_sub(1),
        ));

        WorldRect::from_corners(top_left, bottom_right)
    }

    /// Convert a world-space position to viewport coordinates (cell indices).
    /// Returns `Some(ViewportPos)` if inside [0..width) × [0..height), otherwise `None`.
    pub fn world_to_viewport(&self, world: WorldPos) -> Option<ViewportPos> {
        // Subtract origin (1:1 mapping, no zoom scaling).
        let origin = self.camera_origin_world();
        let relative = world - origin;

        // Check if position is within viewport bounds
        if relative.x >= 0 && relative.y >= 0 {
            let screen_x = relative.x as u16;
            let screen_y = relative.y as u16;
            if screen_x < self.viewport_bounds.width && screen_y < self.viewport_bounds.height {
                return Some(ViewportPos::new(screen_x, screen_y));
            }
        }
        None
    }

    /// Inverse of `world_to_viewport`: given a viewport cell, return the corresponding world pos.
    pub fn viewport_to_world(&self, screen: ViewportPos) -> WorldPos {
        // Add origin (1:1 mapping, no zoom scaling).
        let origin = self.camera_origin_world();
        WorldPos::new(origin.x + screen.x as i64, origin.y + screen.y as i64)
    }

    /// Convert a world position to terminal buffer coordinates.
    /// Returns `Some((x, y))` if the position is visible in the terminal, otherwise `None`.
    /// Handles viewport offset and Y-axis flipping (world Y+ is up, terminal Y+ is down).
    pub fn world_to_terminal(&self, world_pos: WorldPos) -> Option<(u16, u16)> {
        // Handle uninitialized viewport bounds
        if self.viewport_bounds.width == 0 || self.viewport_bounds.height == 0 {
            return None;
        }

        // First convert to viewport coordinates
        if let Some(viewport_pos) = self.world_to_viewport(world_pos) {
            // Check bounds
            if viewport_pos.x < self.viewport_bounds.width
                && viewport_pos.y < self.viewport_bounds.height
            {
                // Convert to terminal coordinates with viewport offset and Y-axis flip
                let terminal_x = self.viewport_bounds.x + viewport_pos.x;
                let terminal_y =
                    self.viewport_bounds.y + (self.viewport_bounds.height - 1 - viewport_pos.y);
                Some((terminal_x, terminal_y))
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Convert terminal buffer coordinates to world position.
    /// Inverse of `world_to_terminal`. Handles viewport offset and Y-axis flipping.
    pub fn terminal_to_world(&self, terminal_x: u16, terminal_y: u16) -> Option<WorldPos> {
        // Handle uninitialized viewport bounds
        if self.viewport_bounds.width == 0 || self.viewport_bounds.height == 0 {
            return None;
        }

        // Check if terminal coordinates are within our viewport bounds
        if terminal_x >= self.viewport_bounds.x
            && terminal_x < self.viewport_bounds.x + self.viewport_bounds.width
            && terminal_y >= self.viewport_bounds.y
            && terminal_y < self.viewport_bounds.y + self.viewport_bounds.height
        {
            // Convert to viewport coordinates (remove offset and flip Y-axis)
            let viewport_x = terminal_x - self.viewport_bounds.x;
            let viewport_y =
                (self.viewport_bounds.height - 1) - (terminal_y - self.viewport_bounds.y);

            // Convert to world coordinates
            let viewport_pos = ViewportPos::new(viewport_x, viewport_y);
            Some(self.viewport_to_world(viewport_pos))
        } else {
            None
        }
    }

    // TODO: this should replace visible_world_rect
    /// Returns the area the camera sees, i.e. the viewport in world coordinates
    pub fn camera_rect(&self) -> crate::geometry::WorldRect {
        let center = self.camera_current;
        let size = (
            self.viewport_bounds.width as u64,
            self.viewport_bounds.height as u64,
        );
        crate::geometry::WorldRect::from_center_and_size(center, size)
    }

    // TODO remove this function, replace calls with calls to camera_rect().min()
    /// Returns the top-left world coordinate for viewport (0,0).
    pub fn camera_origin_world(&self) -> WorldPos {
        // Center of viewport in world coords = camera_current (no anchor needed)
        // Subtract half the viewport size (1:1 mapping, no zoom scaling) to get the top-left.
        let half_screen = WorldPos::new(
            self.viewport_bounds.width as i64 / 2i64,
            self.viewport_bounds.height as i64 / 2i64,
        );
        self.camera_current - half_screen
    }

    /// Give this viewport input focus. Scroll events and key events will move the camera.
    pub fn focus(&mut self) {
        self.has_focus = true;
    }

    /// Remove input focus. Scroll events are ignored.
    pub fn blur(&mut self) {
        self.has_focus = false;
    }

    /// Disable panning mode and reactivate multizone cursor/camera behavior
    pub fn stop_panning(&mut self) {
        self.panning = false;
    }

    /// Set the dead zone as a fraction of viewport radius.
    /// 0.0 = no dead zone, 1.0 = dead zone extends to viewport edge.
    /// Values are automatically clamped to [0.0, 1.0] to prevent underflow.
    pub fn set_dead_zone_fraction(&mut self, x_fraction: f32, y_fraction: f32) {
        self.dead_zone_fraction = (x_fraction.clamp(0.0, 1.0), y_fraction.clamp(0.0, 1.0));
    }

    /// Set the soft zone as a fraction of viewport radius.
    /// 0.0 = no soft zone, 1.0 = soft zone extends to viewport edge.
    /// Values are automatically clamped to [dead_zone_fraction, 1.0] to ensure soft >= dead.
    pub fn set_soft_zone_fraction(&mut self, x_fraction: f32, y_fraction: f32) {
        self.soft_zone_fraction = (
            x_fraction.clamp(self.dead_zone_fraction.0, 1.0),
            y_fraction.clamp(self.dead_zone_fraction.1, 1.0),
        );
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

    /// Get the viewport size from the state
    pub fn get_viewport_size(&self) -> (u16, u16) {
        (
            self.viewport_state.viewport_bounds.width,
            self.viewport_state.viewport_bounds.height,
        )
    }

    /// Get the viewport area from the state
    pub fn viewport_area(&self) -> Rect {
        self.viewport_state.viewport_bounds
    }

    /// Get the visible world rect from the state
    pub fn visible_world_rect(&self) -> WorldRect {
        self.viewport_state.visible_world_rect()
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
    /// Returns true if the character was written (within viewport bounds), false otherwise.
    pub fn set_char(&mut self, world_pos: WorldPos, ch: char) -> bool {
        self.set_char_styled(world_pos, ch, Style::default())
    }

    /// Set a single character with style at the specified world position.
    /// Returns true if the character was written (within viewport bounds), false otherwise.
    pub fn set_char_styled(&mut self, world_pos: WorldPos, ch: char, style: Style) -> bool {
        let viewport_size = self.get_viewport_size();
        if let Some(viewport_pos) = self.viewport_state.world_to_viewport(world_pos)
            && viewport_pos.x < viewport_size.0
            && viewport_pos.y < viewport_size.1
        {
            let viewport_area = self.viewport_area();
            let buffer_x = viewport_area.x + viewport_pos.x;
            // Flip Y-axis: convert from world coordinates (upward Y) to terminal buffer coordinates (downward Y)
            let buffer_y = viewport_area.y + (viewport_size.1 - 1 - viewport_pos.y);

            // Ensure we're within the buffer bounds
            if buffer_x < viewport_area.right() && buffer_y < viewport_area.bottom() {
                let cell = self.buffer.cell_mut((buffer_x, buffer_y)).unwrap();
                cell.set_char(ch);
                cell.set_style(style);
                return true;
            }
        }
        false
    }

    /// Set a string starting at the specified world position, advancing horizontally.
    /// Returns the number of characters actually written (may be less due to clipping).
    pub fn set_string(&mut self, world_pos: WorldPos, text: &str) -> usize {
        self.set_string_styled(world_pos, text, Style::default())
    }

    /// Set a string with style starting at the specified world position, advancing horizontally.
    /// Returns the number of characters actually written (may be less due to clipping).
    pub fn set_string_styled(&mut self, world_pos: WorldPos, text: &str, style: Style) -> usize {
        let mut written = 0;
        for (i, ch) in text.chars().enumerate() {
            let char_world_pos = WorldPos::new(world_pos.x + i as i64, world_pos.y);
            if self.set_char_styled(char_world_pos, ch, style) {
                written += 1;
            }
            // Continue trying to write characters even if some are out of bounds
            // This allows partial strings to render when only part of the string is visible
        }
        written
    }

    /// Set a string vertically starting at the specified world position, advancing downward.
    /// Returns the number of characters actually written (may be less due to clipping).
    pub fn set_string_vertical(&mut self, world_pos: WorldPos, text: &str) -> usize {
        self.set_string_vertical_styled(world_pos, text, Style::default())
    }

    /// Set a string vertically with style starting at the specified world position, advancing downward.
    /// Returns the number of characters actually written (may be less due to clipping).
    pub fn set_string_vertical_styled(
        &mut self,
        world_pos: WorldPos,
        text: &str,
        style: Style,
    ) -> usize {
        let mut written = 0;
        for (i, ch) in text.chars().enumerate() {
            let char_world_pos = WorldPos::new(world_pos.x, world_pos.y - i as i64);
            if self.set_char_styled(char_world_pos, ch, style) {
                written += 1;
            }
            // Continue trying to write characters even if some are out of bounds
            // This allows partial strings to render when only part of the string is visible
        }
        written
    }

    /// Fill a rectangular area in world coordinates with the specified character.
    /// Returns the number of characters actually written.
    pub fn fill_rect(&mut self, world_rect: WorldRect, ch: char) -> usize {
        self.fill_rect_styled(world_rect, ch, Style::default())
    }

    /// Fill a rectangular area in world coordinates with the specified character and style.
    /// Returns the number of characters actually written.
    pub fn fill_rect_styled(&mut self, world_rect: WorldRect, ch: char, style: Style) -> usize {
        let mut written = 0;
        for y in world_rect.min.y..=world_rect.max.y {
            for x in world_rect.min.x..=world_rect.max.x {
                if self.set_char_styled(WorldPos::new(x, y), ch, style) {
                    written += 1;
                }
            }
        }
        written
    }

    /// Clear a single cell back to default (space character, default style).
    /// Returns true if the cell was cleared (within viewport bounds), false otherwise.
    pub fn clear_cell(&mut self, world_pos: WorldPos) -> bool {
        self.set_char_styled(world_pos, ' ', Style::default())
    }

    /// Clear a rectangular region back to default (space characters, default style).
    /// Returns the number of cells actually cleared.
    pub fn clear_rect(&mut self, world_rect: WorldRect) -> usize {
        self.fill_rect_styled(world_rect, ' ', Style::default())
    }

    /// Clear the entire visible viewport area back to default.
    /// Returns the number of cells actually cleared.
    pub fn clear_visible(&mut self) -> usize {
        let visible_rect = self.viewport_state.visible_world_rect();
        self.clear_rect(visible_rect)
    }

    /// Set multiple characters at once with relative offsets from a base position.
    /// Each entry in `chars` is (relative_offset, character).
    /// Returns the number of characters actually written.
    pub fn set_chars(&mut self, base_pos: WorldPos, chars: &[(WorldPos, char)]) -> usize {
        let mut written = 0;
        for (offset, ch) in chars {
            let world_pos = WorldPos::new(base_pos.x + offset.x, base_pos.y + offset.y);
            if self.set_char(world_pos, *ch) {
                written += 1;
            }
        }
        written
    }

    /// Set multiple characters with styles at once with relative offsets from a base position.
    /// Each entry in `chars` is (relative_offset, character, style).
    /// Returns the number of characters actually written.
    pub fn set_chars_styled(
        &mut self,
        base_pos: WorldPos,
        chars: &[(WorldPos, char, Style)],
    ) -> usize {
        let mut written = 0;
        for (offset, ch, style) in chars {
            let world_pos = WorldPos::new(base_pos.x + offset.x, base_pos.y + offset.y);
            if self.set_char_styled(world_pos, *ch, *style) {
                written += 1;
            }
        }
        written
    }

    /// Get all content within a world rectangle as a 2D vector.
    /// Returns Vec<Vec<(char, Style)>> where outer vec is rows, inner vec is columns.
    /// Missing or out-of-bounds cells are returned as (' ', Style::default()).
    pub fn get_rect_content(&self, world_rect: WorldRect) -> Vec<Vec<(char, Style)>> {
        let mut result = Vec::new();
        for y in world_rect.min.y..=world_rect.max.y {
            let mut row = Vec::new();
            for x in world_rect.min.x..=world_rect.max.x {
                let world_pos = WorldPos::new(x, y);
                let content = self
                    .get_char_styled(world_pos)
                    .unwrap_or((' ', Style::default()));
                row.push(content);
            }
            result.push(row);
        }
        result
    }

    /// Set content for an entire rectangle from a 2D array.
    /// `content` is organized as content[row][col] = (char, style).
    /// Returns the number of characters actually written.
    pub fn set_rect_content(
        &mut self,
        world_rect: WorldRect,
        content: &[Vec<(char, Style)>],
    ) -> usize {
        let mut written = 0;
        let rect_height = (world_rect.max.y - world_rect.min.y + 1) as usize;
        let rect_width = (world_rect.max.x - world_rect.min.x + 1) as usize;

        for (row_idx, row) in content.iter().enumerate().take(rect_height) {
            for (col_idx, (ch, style)) in row.iter().enumerate().take(rect_width) {
                let world_pos = WorldPos::new(
                    world_rect.min.x + col_idx as i64,
                    world_rect.min.y + row_idx as i64,
                );
                if self.set_char_styled(world_pos, *ch, *style) {
                    written += 1;
                }
            }
        }
        written
    }

    /// Check if any non-space content exists in a region.
    /// Returns true if any cell contains a character other than ' ' (space).
    pub fn has_content(&self, world_rect: WorldRect) -> bool {
        for y in world_rect.min.y..=world_rect.max.y {
            for x in world_rect.min.x..=world_rect.max.x {
                let world_pos = WorldPos::new(x, y);
                if let Some(ch) = self.get_char(world_pos)
                    && ch != ' '
                {
                    return true;
                }
            }
        }
        false
    }

    /// Count non-space content in a region.
    /// Returns the number of cells that contain characters other than ' ' (space).
    pub fn count_content(&self, world_rect: WorldRect) -> usize {
        let mut count = 0;
        for y in world_rect.min.y..=world_rect.max.y {
            for x in world_rect.min.x..=world_rect.max.x {
                let world_pos = WorldPos::new(x, y);
                if let Some(ch) = self.get_char(world_pos)
                    && ch != ' '
                {
                    count += 1;
                }
            }
        }
        count
    }

    /// Get the viewport size in cells
    pub fn viewport_size(&self) -> (u16, u16) {
        self.get_viewport_size()
    }

    /// Get the current camera position in world coordinates
    pub fn camera_position(&self) -> WorldPos {
        self.viewport_state.camera_current
    }

    /// Get current camera position (alias for camera_position for consistency)
    pub fn get_camera_position(&self) -> WorldPos {
        self.camera_position()
    }

    /// Get a single character at the specified world position.
    /// Returns Some(char) if the position is within viewport bounds, None otherwise.
    pub fn get_char(&self, world_pos: WorldPos) -> Option<char> {
        let viewport_size = self.get_viewport_size();
        if let Some(viewport_pos) = self.viewport_state.world_to_viewport(world_pos)
            && viewport_pos.x < viewport_size.0
            && viewport_pos.y < viewport_size.1
        {
            let viewport_area = self.viewport_area();
            let buffer_x = viewport_area.x + viewport_pos.x;
            // Flip Y-axis: convert from world coordinates (upward Y) to terminal buffer coordinates (downward Y)
            let buffer_y = viewport_area.y + (viewport_size.1 - 1 - viewport_pos.y);

            // Ensure we're within the buffer bounds
            if buffer_x < viewport_area.right() && buffer_y < viewport_area.bottom() {
                let cell = self.buffer.cell((buffer_x, buffer_y)).unwrap();
                return Some(cell.symbol().chars().next().unwrap_or(' '));
            }
        }
        None
    }

    /// Get a single character and its style at the specified world position.
    /// Returns Some((char, Style)) if the position is within viewport bounds, None otherwise.
    pub fn get_char_styled(&self, world_pos: WorldPos) -> Option<(char, Style)> {
        let viewport_size = self.get_viewport_size();
        if let Some(viewport_pos) = self.world_to_viewport(world_pos)
            && viewport_pos.x < viewport_size.0
            && viewport_pos.y < viewport_size.1
        {
            let viewport_area = self.viewport_area();
            let buffer_x = viewport_area.x + viewport_pos.x;
            // Flip Y-axis: convert from world coordinates (upward Y) to terminal buffer coordinates (downward Y)
            let buffer_y = viewport_area.y + (viewport_size.1 - 1 - viewport_pos.y);

            // Ensure we're within the buffer bounds
            if buffer_x < viewport_area.right() && buffer_y < viewport_area.bottom() {
                let cell = self.buffer.cell((buffer_x, buffer_y)).unwrap();
                let ch = cell.symbol().chars().next().unwrap_or(' ');
                return Some((ch, cell.style()));
            }
        }
        None
    }

    /// Get a horizontal string starting at the specified world position.
    /// Returns the string up to the specified length or until an out-of-bounds position is encountered.
    pub fn get_string(&self, world_pos: WorldPos, max_length: usize) -> String {
        let mut result = String::new();
        for i in 0..max_length {
            let char_world_pos = WorldPos::new(world_pos.x + i as i64, world_pos.y);
            if let Some(ch) = self.get_char(char_world_pos) {
                result.push(ch);
            } else {
                break;
            }
        }
        result
    }

    /// Get a vertical string starting at the specified world position.
    /// Returns the string up to the specified length or until an out-of-bounds position is encountered.
    pub fn get_string_vertical(&self, world_pos: WorldPos, max_length: usize) -> String {
        let mut result = String::new();
        for i in 0..max_length {
            let char_world_pos = WorldPos::new(world_pos.x, world_pos.y - i as i64);
            if let Some(ch) = self.get_char(char_world_pos) {
                result.push(ch);
            } else {
                break;
            }
        }
        result
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

        // World to viewport conversion
        // Test a position that's within the viewport bounds [0, 10) x [0, 10)
        let world_pos = WorldPos::new(104, 103); // This will be at viewport (9, 8)
        let viewport_pos = state.world_to_viewport(world_pos);
        assert_eq!(viewport_pos, Some(ViewportPos::new(9, 8)));

        // Viewport to world conversion
        let screen_pos = ViewportPos::new(5, 2);
        let world_pos = state.viewport_to_world(screen_pos);
        assert_eq!(world_pos, WorldPos::new(100, 97));

        // Round-trip test
        let original = WorldPos::new(99, 104);
        if let Some(screen) = state.world_to_viewport(original) {
            let back_to_world = state.viewport_to_world(screen);
            assert_eq!(back_to_world, original);
        }
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
    fn test_panning_behavior() {
        let mut state = ViewportState::new();
        // Set zone fractions to enable multizone logic
        state.set_dead_zone_fraction(0.2, 0.2);
        state.set_soft_zone_fraction(0.4, 0.4);

        assert!(!state.panning);

        // Simulate scroll event
        state.handle_mouse_scroll(5, 3, Duration::from_millis(100));

        // Should enable panning mode
        assert!(state.panning);

        // Test that scroll is ignored when not focused
        state.blur();
        let old_target = state.camera_target;
        state.handle_mouse_scroll(10, 10, Duration::from_millis(100));
        assert_eq!(state.camera_target, old_target);

        // Stop panning
        state.stop_panning();
        assert!(!state.panning);
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
        // Origin = (10, 10) - (10, 5) = (0, 5)
        // So world (15, 12) - origin (0, 5) = viewport (15, 7)
        writer.set_char(WorldPos::new(15, 12), 'X');

        // Verify it was written correctly
        // The Y-axis is flipped: viewport y=7 becomes buffer y = 10-1-7 = 2
        assert_eq!(buffer[(15, 2)].symbol(), "X");
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
