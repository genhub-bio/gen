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

/// Zone classification for cursor following behavior
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Zone {
    Dead, // Far from edges - no camera movement
    Soft, // Medium distance - smooth following
    Hard, // Close to edges - immediate snap
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
        if self.camera_anim.is_none()
            && !self.panning
            && cursor.to_world_pos(viewport_graph).is_some()
        {
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

            let cursor_viewport = cursor.viewport_pos();

            // Calculate distance from each edge (in cells)
            let dist_from_left = cursor_viewport.x;
            let dist_from_right = width.saturating_sub(cursor_viewport.x).saturating_sub(1);
            let dist_from_top = cursor_viewport.y;
            let dist_from_bottom = height.saturating_sub(cursor_viewport.y).saturating_sub(1);

            // Find closest edge distance for each axis
            let min_x_dist = dist_from_left.min(dist_from_right);
            let min_y_dist = dist_from_top.min(dist_from_bottom);

            // Determine zone independently for X and Y based on distance from nearest edge
            // Note: Terminal cells are ~2x taller than wide, so Y uses half the cell values
            let x_zone = if min_x_dist >= soft_zone {
                Zone::Dead
            } else if min_x_dist >= hard_zone {
                Zone::Soft
            } else {
                Zone::Hard
            };

            let soft_zone_y = soft_zone / 2;
            let hard_zone_y = hard_zone / 2;

            let y_zone = if min_y_dist >= soft_zone_y {
                Zone::Dead
            } else if min_y_dist >= hard_zone_y {
                Zone::Soft
            } else {
                Zone::Hard
            };

            let mut desired_cam = self.camera_current;
            let mut needs_smooth_follow = false;
            let mut needs_snap = false;

            // Handle X-axis
            match x_zone {
                Zone::Dead => {
                    // Cursor free to move - no camera adjustment needed
                }
                Zone::Soft => {
                    // Smooth follow: push cursor back toward dead zone
                    // Calculate how far into soft zone (0.0 = at dead boundary, 1.0 = at hard boundary)
                    let soft_zone_width = soft_zone - hard_zone;
                    if soft_zone_width > 0 {
                        let dist_into_soft = soft_zone - min_x_dist;
                        let progress = dist_into_soft as f64 / soft_zone_width as f64;

                        // Determine direction: which edge are we closest to?
                        let shift = (dist_into_soft as f64 * progress) as i64;
                        if dist_from_left < dist_from_right {
                            // Closer to left edge - push camera left to move cursor right in viewport
                            desired_cam.x -= shift;
                        } else {
                            // Closer to right edge - push camera right to move cursor left in viewport
                            desired_cam.x += shift;
                        }
                        needs_smooth_follow = true;
                    }
                }
                Zone::Hard => {
                    // Immediate snap: move cursor exactly to soft zone boundary
                    let target_dist = soft_zone;
                    let current_dist = min_x_dist;
                    let snap_amount = (target_dist - current_dist) as i64;

                    if dist_from_left < dist_from_right {
                        desired_cam.x -= snap_amount;
                    } else {
                        desired_cam.x += snap_amount;
                    }
                    needs_snap = true;
                }
            }

            // Handle Y-axis (uses half the cell values due to aspect ratio)
            match y_zone {
                Zone::Dead => {}
                Zone::Soft => {
                    let soft_zone_width = soft_zone_y - hard_zone_y;
                    if soft_zone_width > 0 {
                        let dist_into_soft = soft_zone_y - min_y_dist;
                        let progress = dist_into_soft as f64 / soft_zone_width as f64;
                        let shift = (dist_into_soft as f64 * progress) as i64;

                        if dist_from_top < dist_from_bottom {
                            desired_cam.y -= shift;
                        } else {
                            desired_cam.y += shift;
                        }
                        needs_smooth_follow = true;
                    }
                }
                Zone::Hard => {
                    let target_dist = soft_zone_y;
                    let current_dist = min_y_dist;
                    let snap_amount = (target_dist - current_dist) as i64;

                    if dist_from_top < dist_from_bottom {
                        desired_cam.y -= snap_amount;
                    } else {
                        desired_cam.y += snap_amount;
                    }
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
        let _ = cursor.update(viewport_graph, camera_rect);
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
        use crate::{
            cursor::Cursor, graph_controller::ViewportState, viewport_graph::CroppedGraph,
        };

        let mut state = ViewportState::new();
        state.hard_zone = 2; // Hard zone: 2 cells from edge
        state.soft_zone = 4; // Soft zone: 4 cells from edge (dead zone is implicit)

        let viewport_size = (10_u16, 10_u16);
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
        use crate::{
            cursor::Cursor, graph_controller::ViewportState, viewport_graph::CroppedGraph,
        };

        let mut state = ViewportState::new();
        state.hard_zone = 2; // Hard zone: 2 cells from edge
        state.soft_zone = 4; // Soft zone: 4 cells from edge

        let viewport_size = (10_u16, 10_u16);
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
        use crate::{
            cursor::Cursor, graph_controller::ViewportState, viewport_graph::CroppedGraph,
        };

        let mut state = ViewportState::new();
        state.hard_zone = 2; // Hard zone: 2 cells from edge
        state.soft_zone = 4; // Soft zone: 4 cells from edge

        let viewport_size = (10_u16, 10_u16);
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
        use crate::{
            cursor::Cursor, graph_controller::ViewportState, viewport_graph::CroppedGraph,
        };

        let mut state = ViewportState::new();
        state.hard_zone = 2; // Hard zone: 2 cells from edge
        state.soft_zone = 4; // Soft zone: 4 cells from edge

        let viewport_size = (10_u16, 10_u16);
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
