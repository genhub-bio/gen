use std::time::Duration;

use tachyonfx::{Interpolatable, Interpolation};

#[cfg(test)]
use crate::geometry::ViewportPos;
use crate::{
    cursor::Cursor,
    geometry::{WorldPos, clamp_to_bounds},
    graph_controller::ViewportState,
    viewport_graph::CroppedGraph,
};

impl Interpolatable<WorldPos> for WorldPos {
    fn lerp(&self, target: &WorldPos, alpha: f32) -> WorldPos {
        let x = self.x as f64 + ((target.x as f64 - self.x as f64) * alpha as f64);
        let y = self.y as f64 + ((target.y as f64 - self.y as f64) * alpha as f64);
        WorldPos::new(x.round() as i64, y.round() as i64)
    }
}

/// Animation for smooth interpolation between two positions.
#[derive(Debug, Clone)]
pub struct Animation {
    pub start: WorldPos,
    pub end: WorldPos,
    pub duration: Duration,
    pub elapsed: Duration,
    pub interpolation: Interpolation,
}

impl Animation {
    /// Construct a new animation from `start` to `end` over `duration`.
    /// Uses the specified interpolation curve (e.g., Linear, EaseInOut).
    pub fn new(
        start: WorldPos,
        end: WorldPos,
        duration: Duration,
        interpolation: Interpolation,
    ) -> Self {
        Animation {
            start,
            end,
            duration,
            elapsed: Duration::ZERO,
            interpolation,
        }
    }

    /// Advance the animation by time `delta`. Returns the interpolated position.
    /// When the timer completes, `is_complete()` will return `true`.
    pub fn update(&mut self, delta: Duration) -> WorldPos {
        self.elapsed = self.elapsed.saturating_add(delta);

        // Calculate progress as a value between 0.0 and 1.0
        let progress = if self.duration.as_micros() > 0 {
            (self.elapsed.as_micros() as f32 / self.duration.as_micros() as f32).min(1.0)
        } else {
            1.0
        };

        // Use the Interpolatable trait to interpolate between start and end positions
        self.start.tween(&self.end, progress, self.interpolation)
    }

    /// Returns `true` if the animation has finished.
    pub fn is_complete(&self) -> bool {
        self.elapsed >= self.duration
    }

    /// Returns the final target position.
    pub fn target(&self) -> WorldPos {
        self.end
    }
}

impl ViewportState {
    /// Update animations and camera-follow logic. Call once per frame/tick.
    ///
    /// # Arguments
    /// - `delta`: elapsed time since the last update (e.g., from a timer).
    /// - `viewport_size`: size of the viewport (width, height) in terminal cells.
    /// - `cursor`: mutable reference to the ViewportCursor for animation updates
    /// - `viewport_graph`: reference to viewport graph for coordinate conversions
    ///
    /// Behavior:
    /// 1. Advance any ongoing `cursor_anim` and update `cursor_current`.
    /// 2. Advance any ongoing `camera_anim` and update `camera_current`.
    /// 3. If no camera animation is active, apply three-zone camera logic based on `cursor_current`:
    /// ```text
    /// ┌─────────────────────────────────────┐      
    /// │ Hard zone (cursor cannot enter)     │      
    /// │     ┌──────────────────────────┐    │      
    /// │     │ Soft zones(pushes cursor)│    │      
    /// │     │   ┌──────────────────┐   │    │     
    /// │     │   │  Dead zone       │   │    │      
    /// │     │   │  (cursor free    │   │    │      
    /// │     │   │    to move)      │   │    │      
    /// │     │   └──────────────────┘   │    │      
    /// │     │                          │    │      
    /// │     └──────────────────────────┘    │      
    /// │                                     │      
    /// └─────────────────────────────────────┘
    /// ```
    ///    - If `cursor_current` lies inside the **Dead Zone**, do nothing.
    ///    - Else if `cursor_current` lies inside the **Soft Zone**, smoothly interpolate the camera so that
    ///      the cursor is brought to the Soft Zone boundary. (We start a new `camera_anim` for this.)
    ///    - Else (cursor is outside Soft Zone, i.e. in **Hard Zone** or beyond): immediately "snap" the camera
    ///      so that the cursor lies exactly on the Soft Zone boundary. (No interpolation.)
    /// 4. Always clamp `camera_current` to `world_bounds` if present.
    pub fn update(&mut self, delta: Duration, cursor: &mut Cursor, viewport_graph: &CroppedGraph) {
        // Advance camera animation (if present)
        if let Some(anim) = &mut self.camera_anim {
            let new_cam = anim.update(delta);
            self.camera_current = new_cam;
            if anim.is_complete() {
                self.camera_anim = None;
            }
        }

        // 3. Only if no camera animation is underway and not panning, apply three-zone following
        if self.camera_anim.is_none() && !self.panning {
            // Use the authoritative viewport bounds from state, not the parameter
            let width = self.viewport_bounds.width;
            let height = self.viewport_bounds.height;

            // If viewport is too small for configured zones, use minimal zones instead
            let min_viewport_dimension = width.min(height);
            let (soft_zone, hard_zone) = if self.soft_zone > min_viewport_dimension / 2 {
                // Viewport too small for configured zones - use minimal zones
                (1, 1)
            } else {
                (self.soft_zone, self.hard_zone)
            };

            // Terminal cells are ~2x taller than wide, so Y uses half the cell values
            let soft_zone_y = soft_zone / 2;
            let hard_zone_y = hard_zone / 2;

            // Define nested rectangles for zone boundaries
            // Dead zone: innermost rectangle where cursor moves freely
            let dead_zone = ratatui::layout::Rect {
                x: soft_zone,
                y: soft_zone_y,
                width: width.saturating_sub(2 * soft_zone),
                height: height.saturating_sub(2 * soft_zone_y),
            };

            // Soft zone boundary: middle rectangle for smooth camera following
            let soft_zone_rect = ratatui::layout::Rect {
                x: hard_zone,
                y: hard_zone_y,
                width: width.saturating_sub(2 * hard_zone),
                height: height.saturating_sub(2 * hard_zone_y),
            };

            let (cursor_vp_x, cursor_vp_y) =
                if let Some(world) = cursor.to_world_pos(viewport_graph) {
                    // Calculate position relative to camera rect top-left
                    let cam_rect = self.camera_rect();
                    (world.x - cam_rect.min.x, world.y - cam_rect.min.y)
                } else {
                    let vp = cursor.viewport_pos();
                    (vp.x as i64, vp.y as i64)
                };

            // Determine which zone the cursor is in manually since we have i64 coords
            // Zone rects are in viewport coordinates (0-based)
            let in_dead_zone = cursor_vp_x >= dead_zone.x as i64
                && cursor_vp_x < (dead_zone.x + dead_zone.width) as i64
                && cursor_vp_y >= dead_zone.y as i64
                && cursor_vp_y < (dead_zone.y + dead_zone.height) as i64;

            let in_soft_zone = cursor_vp_x >= soft_zone_rect.x as i64
                && cursor_vp_x < (soft_zone_rect.x + soft_zone_rect.width) as i64
                && cursor_vp_y >= soft_zone_rect.y as i64
                && cursor_vp_y < (soft_zone_rect.y + soft_zone_rect.height) as i64;

            let mut desired_cam = self.camera_current;
            let mut needs_smooth_follow = false;
            let mut needs_snap = false;

            if in_dead_zone {
                // Cursor is in dead zone - no camera adjustment needed
            } else if in_soft_zone {
                // Cursor is in soft zone (outside dead, inside soft boundary)
                // Smooth follow: calculate exact amount to move cursor to dead zone edge

                // Handle X-axis
                if cursor_vp_x < dead_zone.x as i64 {
                    // Cursor is left of dead zone - move camera left to push cursor right
                    let distance = dead_zone.x as i64 - cursor_vp_x;
                    desired_cam.x -= distance;
                    needs_smooth_follow = true;
                } else if cursor_vp_x >= (dead_zone.x + dead_zone.width) as i64 {
                    // Cursor is right of dead zone - move camera right to push cursor left
                    let distance = cursor_vp_x - (dead_zone.x + dead_zone.width - 1) as i64;
                    desired_cam.x += distance;
                    needs_smooth_follow = true;
                }

                // Handle Y-axis
                if cursor_vp_y < dead_zone.y as i64 {
                    // Cursor is above dead zone - move camera up to push cursor down
                    let distance = dead_zone.y as i64 - cursor_vp_y;
                    desired_cam.y -= distance;
                    needs_smooth_follow = true;
                } else if cursor_vp_y >= (dead_zone.y + dead_zone.height) as i64 {
                    // Cursor is below dead zone - move camera down to push cursor up
                    let distance = cursor_vp_y - (dead_zone.y + dead_zone.height - 1) as i64;
                    desired_cam.y += distance;
                    needs_smooth_follow = true;
                }
            } else {
                // Cursor is in hard zone (outside soft boundary) or beyond viewport
                // Immediate snap: move cursor exactly to soft zone boundary

                // Handle X-axis
                if cursor_vp_x < soft_zone_rect.x as i64 {
                    // Cursor is left of soft zone - snap camera to bring cursor to left edge of soft zone
                    let distance = soft_zone_rect.x as i64 - cursor_vp_x;
                    desired_cam.x -= distance;
                    needs_snap = true;
                } else if cursor_vp_x >= (soft_zone_rect.x + soft_zone_rect.width) as i64 {
                    // Cursor is right of soft zone - snap camera to bring cursor to right edge of soft zone
                    let distance =
                        cursor_vp_x - (soft_zone_rect.x + soft_zone_rect.width - 1) as i64;
                    desired_cam.x += distance;
                    needs_snap = true;
                }

                // Handle Y-axis
                if cursor_vp_y < soft_zone_rect.y as i64 {
                    // Cursor is above soft zone - snap camera to bring cursor to top edge of soft zone
                    let distance = soft_zone_rect.y as i64 - cursor_vp_y;
                    desired_cam.y -= distance;
                    needs_snap = true;
                } else if cursor_vp_y >= (soft_zone_rect.y + soft_zone_rect.height) as i64 {
                    // Cursor is below soft zone - snap camera to bring cursor to bottom edge of soft zone
                    let distance =
                        cursor_vp_y - (soft_zone_rect.y + soft_zone_rect.height - 1) as i64;
                    desired_cam.y += distance;
                    needs_snap = true;
                }
            }

            // Apply camera movement
            if needs_smooth_follow || needs_snap {
                // Clamp to world bounds if they exist
                let clamped = if self.world_bounds.is_some() {
                    clamp_to_bounds(desired_cam, self.world_bounds)
                } else {
                    desired_cam
                };

                if needs_snap {
                    // Hard zone - immediate snap (no animation)
                    self.camera_current = clamped;
                    self.camera_target = clamped;
                } else {
                    // Soft zone - smooth animation
                    self.camera_anim = Some(Animation::new(
                        self.camera_current,
                        clamped,
                        Duration::from_millis(200),
                        Interpolation::CubicOut,
                    ));
                    self.camera_target = clamped;
                }
            }
        }

        // Ensure camera_current remains within world_bounds (if any).
        if self.camera_anim.is_none() && self.world_bounds.is_some() {
            let final_clamped = clamp_to_bounds(self.camera_current, self.world_bounds);
            self.camera_current = final_clamped;
        }

        // Synchronize cursor's viewport position after any camera changes
        let camera_rect = self.camera_rect();
        let cursor_vp_before = cursor.viewport_pos();
        let _ = cursor.update(viewport_graph, camera_rect);
        let cursor_vp_after = cursor.viewport_pos();
        if cursor_vp_before != cursor_vp_after {
            log::trace!(
                "viewport_state.update: cursor viewport changed from ({}, {}) to ({}, {}) after cursor.update(), camera_rect.min.y={}",
                cursor_vp_before.x,
                cursor_vp_before.y,
                cursor_vp_after.x,
                cursor_vp_after.y,
                camera_rect.min.y
            );
        }

        // Safety clamp: ensure cursor never escapes viewport bounds
        // This is a defensive measure - the camera following logic should prevent escapes,
        // but if it fails (e.g., due to rapid cursor movement), we forcibly snap the camera
        if self.viewport_bounds.width > 0 && self.viewport_bounds.height > 0 {
            let cursor_vp = cursor.viewport_pos();
            let width = self.viewport_bounds.width;
            let height = self.viewport_bounds.height;

            let mut clamped = false;

            // Clamp X axis
            if cursor_vp.x >= width {
                let overshoot_x = (cursor_vp.x - width + 1) as i64;
                self.camera_current.x += overshoot_x;
                self.camera_target.x += overshoot_x;
                clamped = true;
                log::warn!(
                    "Safety clamp: cursor escaped right (x={} >= {}), moved camera right by {}",
                    cursor_vp.x,
                    width,
                    overshoot_x
                );
            }

            // Clamp Y axis
            if cursor_vp.y >= height {
                let overshoot_y = (cursor_vp.y - height + 1) as i64;
                self.camera_current.y += overshoot_y;
                self.camera_target.y += overshoot_y;
                clamped = true;
                log::warn!(
                    "Safety clamp: cursor escaped bottom (y={} >= {}), moved camera down by {}",
                    cursor_vp.y,
                    height,
                    overshoot_y
                );
            }

            // If we clamped, update cursor position
            if clamped {
                let camera_rect = self.camera_rect();
                let _ = cursor.update(viewport_graph, camera_rect);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec2_animation() {
        let mut anim = Animation::new(
            WorldPos::new(0, 0),
            WorldPos::new(10, 10),
            Duration::from_secs(1),
            Interpolation::Linear,
        );

        // Start position
        assert_eq!(anim.update(Duration::ZERO), WorldPos::new(0, 0));
        assert!(!anim.is_complete());

        // Quarter way
        assert_eq!(anim.update(Duration::from_millis(250)), WorldPos::new(3, 3));
        assert!(!anim.is_complete());

        // Half way
        assert_eq!(anim.update(Duration::from_millis(250)), WorldPos::new(5, 5));
        assert!(!anim.is_complete());

        // Three quarters
        assert_eq!(anim.update(Duration::from_millis(250)), WorldPos::new(8, 8));
        assert!(!anim.is_complete());

        // Complete
        assert_eq!(
            anim.update(Duration::from_millis(250)),
            WorldPos::new(10, 10)
        );
        assert!(anim.is_complete());
    }

    #[test]
    fn test_dead_zone_behavior() {
        use ratatui::layout::Rect;

        use crate::{
            cursor::Cursor, graph_controller::ViewportState, viewport_graph::CroppedGraph,
        };

        let mut state = ViewportState::new();
        state.hard_zone = 2; // Hard zone: 2 cells from edge
        state.soft_zone = 4; // Soft zone: 4 cells from edge (dead zone is implicit)

        let viewport_size = (10_u16, 10_u16);
        state.viewport_bounds = Rect::new(0, 0, viewport_size.0, viewport_size.1);

        let mut cursor = Cursor::new();
        cursor.set_viewport_pos(ViewportPos::new(5, 5)); // Center of viewport

        let camera_before = state.camera_current;
        let viewport_graph = CroppedGraph::empty();

        state.update(Duration::from_millis(16), &mut cursor, &viewport_graph);

        // Camera should not move
        assert_eq!(state.camera_current, camera_before);
        assert!(state.camera_anim.is_none());
    }

    #[test]
    fn test_soft_zone_behavior() {
        use ratatui::layout::Rect;

        use crate::{
            cursor::Cursor, graph_controller::ViewportState, viewport_graph::CroppedGraph,
        };

        let mut state = ViewportState::new();
        state.hard_zone = 2; // Hard zone: 2 cells from edge
        state.soft_zone = 4; // Soft zone: 4 cells from edge

        let viewport_size = (10_u16, 10_u16);
        state.viewport_bounds = Rect::new(0, 0, viewport_size.0, viewport_size.1);

        let mut cursor = Cursor::new();

        // Viewport center in screen coords: (5, 5) == (0, 0) in world coords
        // Hard zone: 2 cells from edge = radius 3 from center, so hard zone outer is (2,2) to (8,8) in screen coords
        // Soft zone: 4 cells from edge = radius 1 from center, so soft zone outer is (4,4) to (6,6) in screen coords (dead zone implicit)
        // Place cursor at screen coordinates (7, 5) (just outside soft zone, inside hard zone)
        cursor.set_viewport_pos(ViewportPos::new(7, 5));

        let viewport_graph = CroppedGraph::empty();
        state.update(Duration::from_millis(16), &mut cursor, &viewport_graph);

        // Camera animation should start
        assert!(state.camera_anim.is_some());
    }

    #[test]
    fn test_hard_zone_behavior() {
        use ratatui::layout::Rect;

        use crate::{
            cursor::Cursor, graph_controller::ViewportState, viewport_graph::CroppedGraph,
        };

        let mut state = ViewportState::new();
        state.hard_zone = 2; // Hard zone: 2 cells from edge
        state.soft_zone = 4; // Soft zone: 4 cells from edge

        let viewport_size = (10_u16, 10_u16);
        state.viewport_bounds = Rect::new(0, 0, viewport_size.0, viewport_size.1);

        let mut cursor = Cursor::new();

        // Place cursor outside hard zone (beyond outer boundary)
        // Hard zone outer is (2,2) to (8,8) in screen coords
        // Place cursor at screen (9, 5) (outside hard zone)
        cursor.set_viewport_pos(ViewportPos::new(9, 5));
        let camera_before = state.camera_current;

        let viewport_graph = CroppedGraph::empty();
        state.update(Duration::from_millis(16), &mut cursor, &viewport_graph);

        // Camera should snap immediately (no animation)
        assert!(state.camera_anim.is_none());
        assert_ne!(state.camera_current, camera_before);
    }

    #[test]
    fn test_soft_zone_behavior_detailed() {
        use ratatui::layout::Rect;

        use crate::{
            cursor::Cursor, graph_controller::ViewportState, viewport_graph::CroppedGraph,
        };

        let mut state = ViewportState::new();
        state.hard_zone = 2; // Hard zone: 2 cells from edge
        state.soft_zone = 4; // Soft zone: 4 cells from edge

        let viewport_size = (10_u16, 10_u16);
        state.viewport_bounds = Rect::new(0, 0, viewport_size.0, viewport_size.1);

        let mut cursor = Cursor::new();

        // Place cursor outside of soft zone, inside hard zone: viewport (7, 5)
        cursor.set_viewport_pos(ViewportPos::new(7, 5));
        state.camera_current = WorldPos::ZERO;
        state.camera_target = WorldPos::ZERO;
        state.camera_anim = None;

        let viewport_graph = CroppedGraph::empty();
        state.update(Duration::from_millis(16), &mut cursor, &viewport_graph);

        // Camera should start moving since cursor is in soft zone but outside dead zone
        assert!(
            state.camera_anim.is_some(),
            "Camera animation should start for soft zone behavior"
        );
        let target = state.camera_target;
        assert!(
            target.x != 0 || target.y != 0,
            "Camera should move when cursor is in soft zone, target: {:?}",
            target
        );
    }
}
