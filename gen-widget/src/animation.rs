use std::time::Duration;

use tachyonfx::{Interpolatable, Interpolation};

use crate::{
    cursor::Cursor,
    geometry::{ViewportPos, WorldPos, clamp_to_bounds},
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
    pub fn update(
        &mut self,
        delta: Duration,
        viewport_size: (u16, u16),
        cursor: &mut Cursor,
        viewport_graph: &CroppedGraph,
    ) {
        // Advance camera animation (if present)
        if let Some(anim) = &mut self.camera_anim {
            let new_cam = anim.update(delta);
            self.camera_current = new_cam;
            if anim.is_complete() {
                self.camera_anim = None;
            }
        }

        // 3. Only if no camera animation is underway and not panning, apply three-zone following
        // IMPORTANT: Only follow cursor if it has a valid world position (node tracking initialized)
        if self.camera_anim.is_none()
            && !self.panning
            && cursor.to_world_pos(viewport_graph).is_some()
        {
            // Use the cursor's stored viewport position directly (already synchronized above)
            let cursor_viewport = cursor.viewport_pos();

            // Compute the center of the viewport in viewport units
            let viewport_center = ViewportPos::new(viewport_size.0 / 2, viewport_size.1 / 2);

            // Calculate actual zone sizes from fractions (clamped to safe range)
            // Double the fraction so 1.0 = full radius from center to edge
            let dead_zone_fraction_x = (self.dead_zone_fraction.0 * 0.5).clamp(0.0, 0.5);
            let dead_zone_fraction_y = (self.dead_zone_fraction.1 * 0.5).clamp(0.0, 0.5);
            let soft_zone_fraction_x =
                (self.soft_zone_fraction.0 * 0.5).clamp(dead_zone_fraction_x, 0.5);
            let soft_zone_fraction_y =
                (self.soft_zone_fraction.1 * 0.5).clamp(dead_zone_fraction_y, 0.5);

            let dead_zone = ViewportPos::new(
                (viewport_size.0 as f32 * dead_zone_fraction_x) as u16,
                (viewport_size.1 as f32 * dead_zone_fraction_y) as u16,
            );
            let soft_zone = ViewportPos::new(
                (viewport_size.0 as f32 * soft_zone_fraction_x) as u16,
                (viewport_size.1 as f32 * soft_zone_fraction_y) as u16,
            );

            // Dead Zone bounds in viewport coordinates: [center - dead_zone, center + dead_zone]
            let dz_min = ViewportPos::new(
                viewport_center.x.saturating_sub(dead_zone.x),
                viewport_center.y.saturating_sub(dead_zone.y),
            );
            let dz_max = viewport_center + dead_zone;

            // Soft Zone bounds in viewport coordinates: [center - soft_zone, center + soft_zone]
            let sz_min = ViewportPos::new(
                viewport_center.x.saturating_sub(soft_zone.x),
                viewport_center.y.saturating_sub(soft_zone.y),
            );
            let sz_max = viewport_center + soft_zone;

            // Check where cursor lies in viewport coordinates:
            if (dz_min.x <= cursor_viewport.x && cursor_viewport.x <= dz_max.x)
                && (dz_min.y <= cursor_viewport.y && cursor_viewport.y <= dz_max.y)
            {
                // Cursor is inside Dead Zone: do nothing.
            } else if (sz_min.x <= cursor_viewport.x && cursor_viewport.x <= sz_max.x)
                && (sz_min.y <= cursor_viewport.y && cursor_viewport.y <= sz_max.y)
            {
                // Cursor is between Dead Zone and Soft Zone → inside Soft Zone band.
                // We want to smoothly move the camera to keep the cursor from reaching
                // the soft zone boundary. The camera should follow with some lag.

                let mut desired_cam = self.camera_current;
                let mut needs_update = false;

                // X-axis
                if cursor_viewport.x < dz_min.x {
                    // Cursor is left of Dead Zone, but still inside Soft Zone
                    // Calculate how far into the soft zone we are (0.0 = at dead zone, 1.0 = at soft zone boundary)
                    let progress =
                        (dz_min.x - cursor_viewport.x) as f64 / (dz_min.x - sz_min.x) as f64;
                    // Move camera proportionally to keep cursor away from soft boundary
                    let shift_viewport = ((dz_min.x - cursor_viewport.x) as f64 * progress) as u16;
                    let shift_world = shift_viewport as i64; // 1:1 mapping, no zoom scaling
                    desired_cam.x -= shift_world;
                    needs_update = true;
                } else if cursor_viewport.x > dz_max.x {
                    // Cursor is right of Dead Zone
                    let progress =
                        (cursor_viewport.x - dz_max.x) as f64 / (sz_max.x - dz_max.x) as f64;
                    let shift_viewport = ((cursor_viewport.x - dz_max.x) as f64 * progress) as u16;
                    let shift_world = shift_viewport as i64; // 1:1 mapping, no zoom scaling
                    desired_cam.x += shift_world;
                    needs_update = true;
                }

                // Y-axis
                if cursor_viewport.y < dz_min.y {
                    let progress =
                        (dz_min.y - cursor_viewport.y) as f64 / (dz_min.y - sz_min.y) as f64;
                    let shift_viewport = ((dz_min.y - cursor_viewport.y) as f64 * progress) as u16;
                    let shift_world = shift_viewport as i64; // 1:1 mapping, no zoom scaling
                    desired_cam.y -= shift_world;
                    needs_update = true;
                } else if cursor_viewport.y > dz_max.y {
                    let progress =
                        (cursor_viewport.y - dz_max.y) as f64 / (sz_max.y - dz_max.y) as f64;
                    let shift_viewport = ((cursor_viewport.y - dz_max.y) as f64 * progress) as u16;
                    let shift_world = shift_viewport as i64; // 1:1 mapping, no zoom scaling
                    desired_cam.y += shift_world;
                    needs_update = true;
                }

                // Start a smooth animation only if we need to update
                if needs_update {
                    let start = self.camera_current;
                    let end = clamp_to_bounds(desired_cam, self.world_bounds);
                    self.camera_anim = Some(Animation::new(
                        start,
                        end,
                        Duration::from_millis(200),
                        Interpolation::CubicOut,
                    ));
                    // Update camera_target immediately so that subsequent checks
                    // know where we're heading.
                    self.camera_target = end;
                }
            } else {
                // Cursor is outside Soft Zone entirely → Hard Zone or beyond.
                // Immediately "snap" so that cursor lies exactly on the Soft Zone boundary.

                let mut snapped_cam = self.camera_current;

                // X-axis clamp: if cursor_viewport.x < sz_min.x, snap to sz_min.x; if cursor_viewport.x > sz_max.x, snap to sz_max.x.
                if cursor_viewport.x < sz_min.x {
                    let shift_viewport = cursor_viewport.x as i32 - sz_min.x as i32;
                    let shift_world = shift_viewport as i64; // 1:1 mapping, no zoom scaling
                    snapped_cam.x += shift_world;
                } else if cursor_viewport.x > sz_max.x {
                    let shift_viewport = cursor_viewport.x as i32 - sz_max.x as i32;
                    let shift_world = shift_viewport as i64; // 1:1 mapping, no zoom scaling
                    snapped_cam.x += shift_world;
                }

                // Y-axis clamp
                if cursor_viewport.y < sz_min.y {
                    let shift_viewport = cursor_viewport.y as i32 - sz_min.y as i32;
                    let shift_world = shift_viewport as i64; // 1:1 mapping, no zoom scaling
                    snapped_cam.y += shift_world;
                } else if cursor_viewport.y > sz_max.y {
                    let shift_viewport = cursor_viewport.y as i32 - sz_max.y as i32;
                    let shift_world = shift_viewport as i64; // 1:1 mapping, no zoom scaling
                    snapped_cam.y += shift_world;
                }

                // Immediately apply (no interpolation)
                let clamped = clamp_to_bounds(snapped_cam, self.world_bounds);
                self.camera_current = clamped;
                self.camera_target = clamped;
            }
        }

        // Ensure camera_current remains within world_bounds (if any).
        if self.camera_anim.is_none() {
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
        state.dead_zone_fraction = (0.2, 0.2); // 20% of viewport
        state.soft_zone_fraction = (0.4, 0.4); // 40% of viewport

        let viewport_size = (10_u16, 10_u16);
        let mut cursor = Cursor::new();
        cursor.set_viewport_pos(ViewportPos::new(5, 5)); // Center of viewport

        let camera_before = state.camera_current;
        let viewport_graph = CroppedGraph::empty();

        state.update(
            Duration::from_millis(16),
            viewport_size,
            &mut cursor,
            &viewport_graph,
        );

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
        state.dead_zone_fraction = (0.2, 0.2);
        state.soft_zone_fraction = (0.4, 0.4);

        let viewport_size = (10_u16, 10_u16);
        let mut cursor = Cursor::new();

        // Viewport center in screen coords: (5, 5) == (0, 0) in world coords
        // Dead zone radius: 0.2 * 0.5 * 10 = 1, so dead zone is (4,4) to (6,6) in screen coords
        // Soft zone radius: 0.4 * 0.5 * 10 = 2, so soft zone is (3,3) to (7,7) in screen coords
        // Place cursor at screen coordinates (7, 5) (just outside dead zone, inside soft zone)
        cursor.set_viewport_pos(ViewportPos::new(7, 5));

        let viewport_graph = CroppedGraph::empty();
        state.update(
            Duration::from_millis(16),
            viewport_size,
            &mut cursor,
            &viewport_graph,
        );

        // Camera animation should start
        assert!(state.camera_anim.is_some());
    }

    #[test]
    fn test_hard_zone_behavior() {
        use crate::{
            cursor::Cursor, graph_controller::ViewportState, viewport_graph::CroppedGraph,
        };

        let mut state = ViewportState::new();
        state.dead_zone_fraction = (0.2, 0.2); // 20% of viewport
        state.soft_zone_fraction = (0.4, 0.4); // 40% of viewport

        let viewport_size = (10_u16, 10_u16);
        let mut cursor = Cursor::new();

        // Place cursor outside soft zone (in hard zone)
        // Soft zone is (3,3) to (7,7) in screen coords
        // Place cursor at screen (8, 5) (outside soft zone)
        cursor.set_viewport_pos(ViewportPos::new(8, 5));
        let camera_before = state.camera_current;

        let viewport_graph = CroppedGraph::empty();
        state.update(
            Duration::from_millis(16),
            viewport_size,
            &mut cursor,
            &viewport_graph,
        );

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
        state.dead_zone_fraction = (0.2, 0.2); // 20% of viewport
        state.soft_zone_fraction = (0.4, 0.4); // 40% of viewport

        let viewport_size = (10_u16, 10_u16);
        let mut cursor = Cursor::new();

        // Place cursor outside of dead zone, inside soft zone: viewport (7, 5)
        cursor.set_viewport_pos(ViewportPos::new(7, 5));
        state.camera_current = WorldPos::ZERO;
        state.camera_target = WorldPos::ZERO;
        state.camera_anim = None;

        let viewport_graph = CroppedGraph::empty();
        state.update(
            Duration::from_millis(16),
            viewport_size,
            &mut cursor,
            &viewport_graph,
        );

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
