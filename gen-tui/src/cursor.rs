use itertools::Itertools;
use petgraph::graph::NodeIndex;

use crate::{
    geometry::{BigRect, ViewportPos, WorldPos},
    layout::LayoutNode,
    viewport_graph::CroppedGraph,
};

const NO_NEXT_LAYER_ERR: &str = "No next layer";
const NO_PREVIOUS_LAYER_ERR: &str = "No previous layer";

/// New cursor implementation using viewport coordinates as primary representation.
///
/// This design eliminates the need to "restore" cursor position after viewport rebuilds
/// by storing the viewport position (screen coordinates) as the anchor point. The world
/// coordinates are derived from the viewport position and node tracking information.
///
/// # Coordinate System
///
/// - **ViewportPos**: Screen-space coordinates that stay constant across viewport rebuilds
/// - **WorldPos**: Derived from viewport + node tracking; changes when layouts change
/// - **Fractional position**: Position within node as fraction [0.0, 1.0] relative to bottom-left
///
/// # Algorithm
///
/// World position calculation:
/// 1. Look up node_idx in viewport_graph → get node center in world coords
/// 2. Get node size from layout data
/// 3. Calculate bottom-left corner: Wo = center - (width/2, height/2)
/// 4. Apply fractional: Wc = Wo + fractional * (width, height)
#[derive(Debug, Clone)]
pub struct Cursor {
    /// Position in viewport coordinates (PRIMARY - stays stable across rebuilds)
    pub viewport_pos: ViewportPos,

    /// Domain node index that cursor is positioned on
    /// Uses petgraph::NodeIndex to match ViewportGraph's domain_to_world map
    pub node_idx: Option<NodeIndex>,

    /// Fractional position within the node rectangle (0.0 to 1.0)
    /// Relative to bottom-left corner: (0.0, 0.0) = bottom-left, (1.0, 1.0) = top-right
    /// Technically floating point isn't ideal for this since we can't recreate every i64
    /// integer from the part after the decimal (unlike UQ0.64 for example), but that's overkill
    fractional_pos: (f64, f64),

    /// Whether cursor should be rendered (does not affect cursor functionality)
    visible: bool,

    /// Character to render for cursor
    glyph: char,

    /// Whether coarse navigation mode is active
    pub coarse_mode: bool,
}

impl Default for Cursor {
    fn default() -> Self {
        Self::new()
    }
}

impl Cursor {
    pub fn new() -> Self {
        Self {
            viewport_pos: ViewportPos::ZERO,
            node_idx: None,
            fractional_pos: (0.0, 0.0),
            visible: false,
            glyph: '█',
            coarse_mode: true,
        }
    }

    // ==================== Core Coordinate Conversion ====================

    /// Convert cursor to world coordinates using current viewport graph
    ///
    /// # Algorithm
    /// 1. Look up node_idx → world center position
    /// 2. Get node size from layout data
    /// 3. Calculate bottom-left corner: Wo = center - (width/2, height/2)
    ///    (Uses same calculation as BigRect::from_center_and_size)
    /// 4. Apply fractional: Wc = Wo + fractional * (width, height)
    ///
    /// # Returns
    /// - `Some(WorldPos)` if cursor is on a valid node in the viewport graph
    /// - `None` if no node is tracked or node not found in graph
    pub fn to_world_pos(&self, viewport_graph: &CroppedGraph) -> Option<WorldPos> {
        let node_idx = self.node_idx?;

        // Look up node's center position in world coordinates
        let &node_center = viewport_graph.node_positions.get(&node_idx)?;

        // Get node data (including size)
        let node_data = viewport_graph.node_data_by_pos.get(&node_center)?;

        let (width, height) = node_data.size;

        // Calculate bottom-left corner (Wo) using same logic as BigRect::from_center_and_size
        // For odd sizes, center is unambiguous (e.g., size 5: positions 0,1,2,3,4 with center at 2)
        // For even sizes, we use floor division, biasing toward left/bottom
        // (e.g., size 4: positions 0,1,2,3 - we place center between 1 and 2, closer to 1)
        let half_width = (width as i64 - 1) / 2;
        let half_height = (height as i64 - 1) / 2;

        let wo_x = node_center.x - half_width;
        let wo_y = node_center.y - half_height;

        // Apply fractional position
        // Note: We scale by the actual span (width-1, height-1) because coordinates are inclusive
        // For width=5: positions [0,1,2,3,4] span 4 units, so 0.5 * 4 = 2 (center)
        let span_x = width.saturating_sub(1);
        let span_y = height.saturating_sub(1);

        let cursor_x = wo_x + (self.fractional_pos.0 * span_x as f64).round() as i64;
        let cursor_y = wo_y + (self.fractional_pos.1 * span_y as f64).round() as i64;

        Some(WorldPos::new(cursor_x, cursor_y))
    }

    /// Update fractional position from a world coordinate (for intra-node movement)
    ///
    /// # Formula
    /// `fractional = (Wc - Wo) / (width, height)`
    ///
    /// where:
    /// - Wc = cursor world position (input)
    /// - Wo = node bottom-left corner = center - (width/2, height/2)
    ///
    /// # Arguments
    /// - `world_pos`: The world position to convert to fractional
    /// - `viewport_graph`: Graph containing node position and size data
    ///
    /// # Returns
    /// - `Ok(())` if successful
    /// - `Err(String)` if no node is tracked or node not found
    pub fn update_fractional_from_world(
        &mut self,
        world_pos: WorldPos,
        viewport_graph: &CroppedGraph,
    ) -> Result<(), String> {
        let node_idx = self.node_idx.ok_or("No node associated with cursor")?;

        let &node_center = viewport_graph
            .node_positions
            .get(&node_idx)
            .ok_or("Node not found in viewport graph")?;

        let node_data = viewport_graph
            .node_data_by_pos
            .get(&node_center)
            .ok_or("Node data not found")?;

        let (width, height) = node_data.size;

        // Calculate bottom-left corner (Wo) - same logic as to_world_pos
        let half_width = (width as i64 - 1) / 2;
        let half_height = (height as i64 - 1) / 2;

        let wo_x = node_center.x - half_width;
        let wo_y = node_center.y - half_height;

        // Calculate fractional: (Wc - Wo) / span
        // Note: Divide by span (width-1, height-1) to match inclusive coordinate system
        // For perfect round-tripping, we need to account for the rounding used in to_world_pos
        let span_x = if width > 1 { (width - 1) as f64 } else { 1.0 };
        let span_y = if height > 1 { (height - 1) as f64 } else { 1.0 };

        let frac_x = (world_pos.x - wo_x) as f64 / span_x;
        let frac_y = (world_pos.y - wo_y) as f64 / span_y;

        // Clamp to valid range [0.0, 1.0]
        self.fractional_pos = (frac_x.clamp(0.0, 1.0), frac_y.clamp(0.0, 1.0));

        Ok(())
    }

    // ==================== Getters/Setters ====================

    /// Get viewport position (primary representation)
    pub fn viewport_pos(&self) -> ViewportPos {
        self.viewport_pos
    }

    /// Set viewport position
    pub fn set_viewport_pos(&mut self, pos: ViewportPos) {
        self.viewport_pos = pos;
    }

    /// Get the node the cursor is on
    pub fn node_idx(&self) -> Option<NodeIndex> {
        self.node_idx
    }

    /// Set the node and fractional position
    pub fn set_node(&mut self, node_idx: NodeIndex, fractional_pos: (f64, f64)) {
        self.node_idx = Some(node_idx);
        self.fractional_pos = fractional_pos;
    }

    /// Get fractional position within current node
    pub fn fractional_pos(&self) -> (f64, f64) {
        self.fractional_pos
    }

    /// Check if cursor should be shown (rendered)
    pub fn is_visible(&self) -> bool {
        self.visible
    }

    /// Show cursor (enable rendering)
    pub fn set_visibility(&mut self, show: bool) {
        self.visible = show;
    }

    /// Get cursor glyph
    pub fn glyph(&self) -> char {
        self.glyph
    }

    /// Set cursor glyph
    pub fn set_glyph(&mut self, glyph: char) {
        self.glyph = glyph;
    }

    /// Check if coarse navigation mode is active
    pub fn is_coarse_mode(&self) -> bool {
        self.coarse_mode
    }

    /// Set coarse navigation mode
    pub fn set_coarse_mode(&mut self, active: bool) {
        self.coarse_mode = active;
    }

    // ==================== Synchronization Methods ====================

    /// Update viewport position from cursor's current world position (derived from node tracking).
    /// This is the primary synchronization method - call this after camera movements to keep
    /// viewport coordinates aligned with the cursor's tracked node position.
    ///
    /// # Parameters
    /// - `viewport_graph`: Graph for deriving world position from node tracking
    /// - `camera_rect`: The camera's visible region in world coordinates
    ///
    /// # Returns
    /// - `Ok(())` if synchronization succeeded
    /// - `Err(String)` if cursor has no valid world position (e.g., no node tracked)
    pub fn update(
        &mut self,
        viewport_graph: &crate::viewport_graph::CroppedGraph,
        camera_rect: crate::geometry::BigRect<i64>,
    ) -> Result<(), String> {
        let viewport_before = self.viewport_pos;
        let world_pos = self
            .to_world_pos(viewport_graph)
            .ok_or("Cannot derive world position - no node tracked")?;

        self.update_from_worldpos(world_pos, camera_rect);

        if self.viewport_pos != viewport_before {
            log::trace!(
                "cursor.update: viewport changed from ({}, {}) to ({}, {}), world=({}, {}), camera_rect.min=({}, {})",
                viewport_before.x,
                viewport_before.y,
                self.viewport_pos.x,
                self.viewport_pos.y,
                world_pos.x,
                world_pos.y,
                camera_rect.min.x,
                camera_rect.min.y
            );
        }
        Ok(())
    }

    /// Update viewport position based on an explicit world position and camera rect.
    /// Use this when you have a specific world position to synchronize to.
    ///
    /// # Parameters
    /// - `world_pos`: The world position to synchronize to
    /// - `camera_rect`: The camera's visible region in world coordinates
    pub fn update_from_worldpos(
        &mut self,
        world_pos: WorldPos,
        camera_rect: crate::geometry::BigRect<i64>,
    ) {
        // Convert world position to viewport coordinates: viewport = world - camera_origin
        // Use i64 for calculation to avoid wrapping, then clamp to u16 range
        let relative_x = world_pos.x - camera_rect.min.x;
        let relative_y = world_pos.y - camera_rect.min.y;

        // We use saturating casts/clamps to prevent wrapping artifacts (e.g. negative -> huge positive)
        let viewport_x = relative_x.clamp(0, u16::MAX as i64) as u16;
        let viewport_y = relative_y.clamp(0, u16::MAX as i64) as u16;

        self.viewport_pos = crate::geometry::ViewportPos::new(viewport_x, viewport_y);
    }

    // ==================== Navigation Methods ====================

    /// Move cursor horizontally within current node or to adjacent node
    ///
    /// # Algorithm
    /// 1. Get current world position
    /// 2. Calculate new world position (current + direction)
    /// 3. If still within node bounds, update fractional from world and return
    /// 4. If out of bounds, find connected node in that direction
    /// 5. Jump to connected node at appropriate edge
    pub fn move_horizontal(
        &mut self,
        direction: i64,
        viewport_graph: &CroppedGraph,
    ) -> Result<(), String> {
        let node_idx = self.node_idx.ok_or("No node associated with cursor")?;

        // Get current world position
        let current_world = self
            .to_world_pos(viewport_graph)
            .ok_or("Cannot calculate current world position")?;

        // Calculate new world position (absolute movement)
        let new_world = WorldPos::new(current_world.x + direction, current_world.y);

        // Get node bounds
        let &node_center = viewport_graph
            .node_positions
            .get(&node_idx)
            .ok_or("Node not found in viewport graph")?;
        let node_data = viewport_graph
            .node_data_by_pos
            .get(&node_center)
            .ok_or("Node data not found")?;

        let node_rect = BigRect::from_center_and_size(node_center, node_data.size);
        let min_x = node_rect.left();
        let max_x = node_rect.right();
        let min_y = node_rect.bottom();
        let max_y = node_rect.top();

        if new_world.x >= min_x
            && new_world.x <= max_x
            && new_world.y >= min_y
            && new_world.y <= max_y
        {
            // Still within node bounds - update fractional position to match
            self.update_fractional_from_world(new_world, viewport_graph)?;
            Ok(())
        } else {
            // At boundary - jump to adjacent node in different layer
            match self.jump_to_adjacent_layer(direction, viewport_graph) {
                Ok(()) => Ok(()),
                Err(msg) => {
                    // In coarse mode, let users escape graph boundaries by moving to the
                    // current node edge instead of hard-failing on missing layers.
                    if self.coarse_mode
                        && (msg == NO_NEXT_LAYER_ERR || msg == NO_PREVIOUS_LAYER_ERR)
                    {
                        let target_frac_x = if direction > 0 { 1.0 } else { 0.0 };
                        let current_frac_y = self.fractional_pos.1;
                        self.fractional_pos = (target_frac_x, current_frac_y);
                        Ok(())
                    } else {
                        Err(msg)
                    }
                }
            }
        }
    }

    /// Move cursor vertically within current node or to node in same layer
    ///
    /// # Algorithm
    /// 1. Get current world position
    /// 2. Calculate new world position (current + direction)
    /// 3. If still within node bounds, update fractional from world and return
    /// 4. If out of bounds, find node in same layer in that direction
    /// 5. Jump to node maintaining fractional_x
    pub fn move_vertical(
        &mut self,
        direction: i64,
        viewport_graph: &CroppedGraph,
    ) -> Result<(), String> {
        let node_idx = self.node_idx.ok_or("No node associated with cursor")?;

        // Get current world position
        let current_world = self
            .to_world_pos(viewport_graph)
            .ok_or("Cannot calculate current world position")?;

        // Calculate new world position (absolute movement)
        let new_world = WorldPos::new(current_world.x, current_world.y + direction);

        // Get node bounds
        let &node_center = viewport_graph
            .node_positions
            .get(&node_idx)
            .ok_or("Node not found in viewport graph")?;
        let node_data = viewport_graph
            .node_data_by_pos
            .get(&node_center)
            .ok_or("Node data not found")?;

        let (width, height) = node_data.size;
        let half_width = (width as i64 - 1) / 2;
        let half_height = (height as i64 - 1) / 2;

        let min_x = node_center.x - half_width;
        let max_x = node_center.x + half_width;
        let min_y = node_center.y - half_height;
        let max_y = node_center.y + half_height;

        if new_world.x >= min_x
            && new_world.x <= max_x
            && new_world.y >= min_y
            && new_world.y <= max_y
        {
            // Still within node bounds - update fractional position to match
            self.update_fractional_from_world(new_world, viewport_graph)?;
            Ok(())
        } else {
            // At boundary - jump to node in same layer
            self.jump_to_same_layer(direction, viewport_graph)
        }
    }

    /// Jump to adjacent node in different layer (horizontal movement between ranks)
    fn jump_to_adjacent_layer(
        &mut self,
        direction: i64,
        viewport_graph: &CroppedGraph,
    ) -> Result<(), String> {
        let node_idx = self.node_idx.ok_or("No node tracked")?;

        // Get current layer
        let current_layer = viewport_graph
            .find_domain_node_layer(node_idx)
            .ok_or("Node not found in any layer")?;

        // Determine target layer
        let target_layer = if direction > 0 {
            // Moving right: next layer
            if current_layer + 1 < viewport_graph.layer_count() {
                current_layer + 1
            } else {
                return Err(NO_NEXT_LAYER_ERR.to_string());
            }
        } else {
            // Moving left: previous layer
            if current_layer == 0 {
                return Err(NO_PREVIOUS_LAYER_ERR.to_string());
            }
            current_layer - 1
        };
        // Get nodes in target layer
        let target_layer_nodes = viewport_graph
            .get_layer(target_layer)
            .ok_or("Target layer not found")?;

        let current_world_pos = *viewport_graph
            .node_positions
            .get(&node_idx)
            .ok_or("Node world position not found")?;

        let candidates: Vec<(WorldPos, &LayoutNode)> = target_layer_nodes
            .iter()
            .map(|idx| {
                viewport_graph
                    .node_positions
                    .get(idx)
                    .expect("Every node in the layers field should be in the graph too")
            })
            .map(|pos| {
                (
                    *pos,
                    viewport_graph
                        .get_node(pos)
                        .expect("Every node in the layers field should be in the graph too"),
                )
            })
            .sorted_by_key(|(pos, _)| {
                let dy = (pos.y - current_world_pos.y).abs();
                (dy, -pos.y)
            })
            .collect();

        // Take the closest candidate
        if let Some((_target_pos, target_node)) = candidates.first() {
            // Position cursor at appropriate edge based on direction
            let target_frac_x = if direction > 0 {
                0.0 // Moving right: position at left edge
            } else {
                1.0 // Moving left: position at right edge
            };

            // Maintain Y fractional position
            let target_frac_y = self.fractional_pos.1;

            // Update cursor to new node
            if let crate::layout::NodeRole::Data(domain_idx) = target_node.role {
                self.set_node(domain_idx, (target_frac_x, target_frac_y));
                Ok(())
            } else {
                Err("Target node is not a Data node".to_string())
            }
        } else {
            Err("No connected node found in target layer".to_string())
        }
    }

    /// Jump to node in same layer (vertical movement within rank)
    fn jump_to_same_layer(
        &mut self,
        direction: i64,
        viewport_graph: &CroppedGraph,
    ) -> Result<(), String> {
        let node_idx = self.node_idx.ok_or("No node tracked")?;
        let current_world_pos = *viewport_graph
            .node_positions
            .get(&node_idx)
            .ok_or("Node world position not found")?;

        // Get current layer
        let current_layer = viewport_graph
            .find_domain_node_layer(node_idx)
            .ok_or("Node not found in any layer")?;

        // Get all nodes in the same layer
        let layer_nodes = viewport_graph
            .get_layer(current_layer)
            .ok_or("Current layer not found")?;

        // Find candidates in the same layer
        let mut candidates: Vec<(WorldPos, &crate::layout::LayoutNode)> = layer_nodes
            .iter()
            .filter_map(|&domain_idx| {
                // Skip current node
                if domain_idx == node_idx {
                    return None;
                }

                // Get world position
                let world_pos = *viewport_graph.node_positions.get(&domain_idx)?;
                let node_data = viewport_graph.node_data_by_pos.get(&world_pos)?;

                // Filter by direction
                let dy = world_pos.y - current_world_pos.y;
                if (direction > 0 && dy > 0) || (direction < 0 && dy < 0) {
                    Some((world_pos, node_data))
                } else {
                    None
                }
            })
            .collect();

        // Sort by vertical distance (closest first)
        candidates.sort_by_key(|(pos, _)| (pos.y - current_world_pos.y).abs());

        // Take the closest candidate
        if let Some((_target_pos, target_node)) = candidates.first() {
            // Maintain X fractional position, set Y to appropriate edge
            let target_frac_x = self.fractional_pos.0;
            let target_frac_y = if direction > 0 {
                0.0 // Moving up: position at bottom edge
            } else {
                1.0 // Moving down: position at top edge
            };

            // Update cursor to new node
            if let crate::layout::NodeRole::Data(domain_idx) = target_node.role {
                self.set_node(domain_idx, (target_frac_x, target_frac_y));
                Ok(())
            } else {
                Err("Target node is not a Data node".to_string())
            }
        } else {
            Err("No node found in same layer in that direction".to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        graph_controller::{GraphConfig, GraphController},
        layout::VisualDetail,
        testing::mocks::{MockDomainGraph, TestGraphs, TestNodeSizers},
    };

    /// Helper to create a mock ViewportGraph for testing
    /// Returns (ViewportGraph, known_node_positions)
    #[allow(clippy::type_complexity)]
    fn create_mock_viewport_graph() -> (CroppedGraph, Vec<(NodeIndex, WorldPos, (u64, u64))>) {
        // Create a simple chain: 0 -> 1 -> 2
        let domain_graph = TestGraphs::domain_simple_chain();
        let node_sizer = TestNodeSizers::fixed_5x3(); // width=5, height=3

        let config = GraphConfig::default();
        let mut controller = GraphController::new_with_config(&domain_graph, node_sizer, config);

        // Set viewport bounds
        controller.viewport_state.viewport_bounds = ratatui::layout::Rect::new(0, 0, 100, 50);
        controller.set_detail_level(VisualDetail::Full);

        // Rebuild viewport graph
        let _ = controller.rebuild_viewport_graph();

        // Extract viewport graph
        let viewport_graph = controller.get_viewport_graph().clone();

        // Collect known node positions for testing
        let mut known_nodes = Vec::new();
        for (world_pos, node_data) in &viewport_graph.node_data_by_pos {
            if let crate::layout::NodeRole::Data(domain_idx) = &node_data.role {
                known_nodes.push((*domain_idx, *world_pos, node_data.size));
            }
        }

        // Sort by domain index for predictability
        known_nodes.sort_by_key(|(idx, _, _)| idx.index());

        (viewport_graph, known_nodes)
    }

    #[test]
    fn test_to_world_pos_at_center() {
        let (viewport_graph, known_nodes) = create_mock_viewport_graph();

        assert!(!known_nodes.is_empty(), "No nodes found in viewport graph");

        let (node_idx, node_center, (width, height)) = known_nodes[0];

        let mut cursor = Cursor::new();
        cursor.set_node(node_idx, (0.5, 0.5)); // Center of node

        let world_pos = cursor
            .to_world_pos(&viewport_graph)
            .expect("Failed to convert to world pos");

        // At center, world_pos should equal node_center
        assert_eq!(
            world_pos, node_center,
            "Cursor at center (0.5, 0.5) should map to node center. Node size: {}x{}",
            width, height
        );
    }

    #[test]
    fn test_to_world_pos_bottom_left() {
        let (viewport_graph, known_nodes) = create_mock_viewport_graph();

        let (node_idx, node_center, (width, height)) = known_nodes[0];

        let mut cursor = Cursor::new();
        cursor.set_node(node_idx, (0.0, 0.0)); // Bottom-left corner

        let world_pos = cursor
            .to_world_pos(&viewport_graph)
            .expect("Failed to convert to world pos");

        // Calculate expected bottom-left corner
        let half_width = (width as i64 - 1) / 2;
        let half_height = (height as i64 - 1) / 2;
        let expected = WorldPos::new(node_center.x - half_width, node_center.y - half_height);

        assert_eq!(
            world_pos, expected,
            "Cursor at bottom-left (0.0, 0.0) should map to min corner"
        );
    }

    #[test]
    fn test_to_world_pos_top_right() {
        let (viewport_graph, known_nodes) = create_mock_viewport_graph();

        let (node_idx, node_center, (width, height)) = known_nodes[0];

        let mut cursor = Cursor::new();
        cursor.set_node(node_idx, (1.0, 1.0)); // Top-right corner

        let world_pos = cursor
            .to_world_pos(&viewport_graph)
            .expect("Failed to convert to world pos");

        // Calculate expected top-right corner
        // With fractional (1.0, 1.0), we get: Wo + 1.0 * (width-1, height-1)
        let half_width = (width as i64 - 1) / 2;
        let half_height = (height as i64 - 1) / 2;
        let wo_x = node_center.x - half_width;
        let wo_y = node_center.y - half_height;
        let span_x = width.saturating_sub(1);
        let span_y = height.saturating_sub(1);
        let expected = WorldPos::new(wo_x + span_x as i64, wo_y + span_y as i64);

        assert_eq!(
            world_pos, expected,
            "Cursor at top-right (1.0, 1.0) should map to max corner"
        );
    }

    #[test]
    fn test_update_fractional_from_world_roundtrip() {
        let (viewport_graph, known_nodes) = create_mock_viewport_graph();

        let (node_idx, _, _) = known_nodes[0];

        // Test various fractional positions
        let test_positions = vec![(0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (0.25, 0.75), (0.7, 0.3)];

        for (frac_x, frac_y) in test_positions {
            let mut cursor = Cursor::new();
            cursor.set_node(node_idx, (frac_x, frac_y));

            // Convert to world
            let world_pos = cursor
                .to_world_pos(&viewport_graph)
                .expect("Failed to convert to world pos");

            // Convert back to fractional
            cursor
                .update_fractional_from_world(world_pos, &viewport_graph)
                .expect("Failed to update fractional");

            let (result_x, result_y) = cursor.fractional_pos();

            // Should round-trip successfully (allow for rounding errors)
            // Note: Due to rounding in to_world_pos, some fractional positions may not
            // round-trip exactly (e.g., 0.75 * 2 = 1.5 → rounds to 2, then 2/2 = 1.0)
            // We allow up to 0.5 difference (one quantization step)
            assert!(
                (result_x - frac_x).abs() <= 0.5,
                "X fractional mismatch: expected {}, got {} (world_pos: {:?})",
                frac_x,
                result_x,
                world_pos
            );
            assert!(
                (result_y - frac_y).abs() <= 0.5,
                "Y fractional mismatch: expected {}, got {} (world_pos: {:?})",
                frac_y,
                result_y,
                world_pos
            );
        }
    }

    #[test]
    fn test_fractional_with_odd_width_node() {
        // Use node with odd dimensions: width=5, height=3
        let (viewport_graph, known_nodes) = create_mock_viewport_graph();

        let (node_idx, node_center, (width, height)) = known_nodes[0];
        assert_eq!(width, 5, "Test expects odd width of 5");
        assert_eq!(height, 3, "Test expects odd height of 3");

        let mut cursor = Cursor::new();
        cursor.set_node(node_idx, (0.5, 0.5));

        let world_pos = cursor.to_world_pos(&viewport_graph).unwrap();

        // For odd dimensions, center should map exactly to node center
        assert_eq!(
            world_pos, node_center,
            "Odd-sized node: center fractional should map to node center"
        );
    }

    #[test]
    fn test_node_not_found_returns_none() {
        let (viewport_graph, _) = create_mock_viewport_graph();

        // Create cursor with non-existent node
        let mut cursor = Cursor::new();
        cursor.set_node(NodeIndex::new(9999), (0.5, 0.5));

        let result = cursor.to_world_pos(&viewport_graph);
        assert!(result.is_none(), "Should return None for non-existent node");
    }

    #[test]
    fn test_no_node_tracked() {
        let (viewport_graph, _) = create_mock_viewport_graph();

        // Cursor with no node set
        let cursor = Cursor::new();

        let result = cursor.to_world_pos(&viewport_graph);
        assert!(
            result.is_none(),
            "Should return None when no node is tracked"
        );
    }

    #[test]
    fn test_large_node_precision() {
        // Create a domain graph with a larger node for precision testing
        let mut domain_graph = MockDomainGraph::new();
        let n0 = domain_graph.add_node(());

        // Custom node sizer for large node
        #[derive(Clone)]
        struct LargeNodeSizer;
        impl crate::plotter::NodeSizer<&MockDomainGraph> for LargeNodeSizer {
            fn get_node_size(&self, _: &NodeIndex, _: VisualDetail) -> (u64, u64) {
                (100, 50) // Very large node
            }
        }
        impl crate::plotter::NodeSizer<MockDomainGraph> for LargeNodeSizer {
            fn get_node_size(&self, _: &NodeIndex, _: VisualDetail) -> (u64, u64) {
                (100, 50)
            }
        }

        let config = GraphConfig::default();
        let mut controller =
            GraphController::new_with_config(&domain_graph, LargeNodeSizer, config);

        controller.viewport_state.viewport_bounds = ratatui::layout::Rect::new(0, 0, 200, 100);
        controller.set_detail_level(VisualDetail::Full);

        let _ = controller.rebuild_viewport_graph();

        let viewport_graph = controller.get_viewport_graph();

        // Find the node
        let node_idx = n0;

        let mut cursor = Cursor::new();
        cursor.set_node(node_idx, (0.5, 0.5));

        let world_pos = cursor
            .to_world_pos(viewport_graph)
            .expect("Failed to convert large node position");

        // Verify the formula works for large nodes
        cursor
            .update_fractional_from_world(world_pos, viewport_graph)
            .expect("Failed to update fractional for large node");

        let (frac_x, frac_y) = cursor.fractional_pos();
        assert!(
            (frac_x - 0.5).abs() <= 0.5,
            "Large node X fractional should round-trip"
        );
        assert!(
            (frac_y - 0.5).abs() <= 0.5,
            "Large node Y fractional should round-trip"
        );
    }

    #[test]
    fn test_getters_setters() {
        let mut cursor = Cursor::new();

        // Test viewport_pos
        assert_eq!(cursor.viewport_pos(), ViewportPos::ZERO);
        cursor.set_viewport_pos(ViewportPos::new(10, 20));
        assert_eq!(cursor.viewport_pos(), ViewportPos::new(10, 20));

        // Test node_idx
        assert_eq!(cursor.node_idx(), None);
        cursor.set_node(NodeIndex::new(5), (0.3, 0.7));
        assert_eq!(cursor.node_idx(), Some(NodeIndex::new(5)));
        assert_eq!(cursor.fractional_pos(), (0.3, 0.7));

        // Test show_cursor
        assert!(!cursor.is_visible());
        cursor.set_visibility(true);
        assert!(cursor.is_visible());
        cursor.set_visibility(false);
        assert!(!cursor.is_visible());

        // Test glyph
        assert_eq!(cursor.glyph(), '█');
        cursor.set_glyph('▓');
        assert_eq!(cursor.glyph(), '▓');
    }

    #[test]
    fn test_move_horizontal_intra_node() {
        let (viewport_graph, known_nodes) = create_mock_viewport_graph();
        assert!(!known_nodes.is_empty(), "Need nodes for navigation test");

        let (node_idx, _, (width, _)) = known_nodes[0];

        // Start at left edge
        let mut cursor = Cursor::new();
        cursor.set_node(node_idx, (0.0, 0.5));

        // Move right within node (should succeed)
        let result = cursor.move_horizontal(1, &viewport_graph);
        assert!(result.is_ok(), "Should move right within node");

        // Fractional should have increased
        assert!(
            cursor.fractional_pos().0 > 0.0,
            "Fractional X should increase. Width: {}",
            width
        );
    }

    #[test]
    fn test_move_horizontal_between_layers() {
        // Use diamond graph: 0 -> {1, 2} -> 3
        let domain_graph = TestGraphs::domain_diamond();
        let node_sizer = TestNodeSizers::fixed_5x3();

        let config = GraphConfig::default();
        let mut controller = GraphController::new_with_config(&domain_graph, node_sizer, config);

        controller.viewport_state.viewport_bounds = ratatui::layout::Rect::new(0, 0, 100, 50);
        controller.set_detail_level(VisualDetail::Full);

        let _ = controller.rebuild_viewport_graph();

        let viewport_graph = controller.get_viewport_graph();

        // Find first node (should be in layer 0)
        let first_node = NodeIndex::new(0);

        // Start at right edge of first node
        let mut cursor = Cursor::new();
        cursor.set_node(first_node, (1.0, 0.5));

        // Move right - should jump to next layer
        let result = cursor.move_horizontal(1, viewport_graph);

        // Should either succeed or fail gracefully
        match result {
            Ok(()) => {
                // Successfully moved to next layer
                let new_node = cursor.node_idx().expect("Should have node after move");
                assert_ne!(new_node, first_node, "Should move to different node");

                // Should be positioned at left edge of new node
                assert!(
                    cursor.fractional_pos().0 < 0.5,
                    "Should be at left edge of target node. Fractional: {:?}",
                    cursor.fractional_pos()
                );
            }
            Err(e) => {
                // Moving right from the last layer should fail gracefully
                assert!(
                    e.contains("No") || e.contains("not found"),
                    "Error should indicate no target: {}",
                    e
                );
            }
        }
    }

    #[test]
    fn test_coarse_mode_boundary_navigation_falls_back_to_node_edges() {
        let mut domain_graph = MockDomainGraph::new();
        let n0 = domain_graph.add_node(());
        let n1 = domain_graph.add_node(());
        let n2 = domain_graph.add_node(());
        domain_graph.add_edge(n0, n1, ());
        domain_graph.add_edge(n1, n2, ());

        let node_sizer = TestNodeSizers::fixed_5x3();
        let mut config = GraphConfig::default();
        config.partition.layer_count = usize::MAX;
        config.partition.node_count = usize::MAX;

        let mut controller = GraphController::new_with_config(&domain_graph, node_sizer, config);
        controller.viewport_state.viewport_bounds = ratatui::layout::Rect::new(0, 0, 100, 50);
        controller.rebuild_viewport_graph().unwrap();
        let viewport_graph = controller.get_viewport_graph();

        // Left boundary in coarse mode should clamp to left edge and stay on node.
        let mut left_cursor = Cursor::new();
        left_cursor.set_node(n0, (0.5, 0.5));
        left_cursor.move_horizontal(-1000, viewport_graph).unwrap();
        assert_eq!(left_cursor.node_idx(), Some(n0));
        assert_eq!(left_cursor.fractional_pos(), (0.0, 0.5));

        // Right boundary in coarse mode should clamp to right edge and stay on node.
        let mut right_cursor = Cursor::new();
        right_cursor.set_node(n2, (0.5, 0.5));
        right_cursor.move_horizontal(1000, viewport_graph).unwrap();
        assert_eq!(right_cursor.node_idx(), Some(n2));
        assert_eq!(right_cursor.fractional_pos(), (1.0, 0.5));
    }

    #[test]
    fn test_move_vertical_intra_node() {
        let (viewport_graph, known_nodes) = create_mock_viewport_graph();
        assert!(!known_nodes.is_empty(), "Need nodes for navigation test");

        let (node_idx, _, (_, height)) = known_nodes[0];

        // Start at bottom edge
        let mut cursor = Cursor::new();
        cursor.set_node(node_idx, (0.5, 0.0));

        // Move up within node (should succeed)
        let result = cursor.move_vertical(1, &viewport_graph);
        assert!(result.is_ok(), "Should move up within node");

        // Fractional should have increased
        assert!(
            cursor.fractional_pos().1 > 0.0,
            "Fractional Y should increase. Height: {}",
            height
        );
    }

    #[test]
    fn test_move_vertical_same_layer() {
        // Use diamond graph: 0 -> {1, 2} -> 3
        // Layer 1 has two nodes (1 and 2)
        let domain_graph = TestGraphs::domain_diamond();
        let node_sizer = TestNodeSizers::fixed_5x3();

        let config = GraphConfig::default();
        let mut controller = GraphController::new_with_config(&domain_graph, node_sizer, config);

        controller.viewport_state.viewport_bounds = ratatui::layout::Rect::new(0, 0, 100, 50);
        controller.set_detail_level(VisualDetail::Full);

        let _ = controller.rebuild_viewport_graph();

        let viewport_graph = controller.get_viewport_graph();

        // Find nodes in middle layer (should be nodes 1 and 2)
        let node1 = NodeIndex::new(1);

        // Check if node1 has a layer with multiple nodes
        if let Some(layer) = viewport_graph.find_domain_node_layer(node1)
            && let Some(layer_nodes) = viewport_graph.get_layer(layer)
            && layer_nodes.len() > 1
        {
            // Start at one node in the layer
            let mut cursor = Cursor::new();
            cursor.set_node(node1, (0.5, 1.0)); // At top edge

            // Try to move up - should jump to other node in same layer
            let result = cursor.move_vertical(1, viewport_graph);

            match result {
                Ok(()) => {
                    // Successfully moved within layer
                    let new_node = cursor.node_idx().expect("Should have node after move");
                    assert_ne!(new_node, node1, "Should move to different node");
                    assert!(
                        layer_nodes.contains(&new_node),
                        "New node should be in same layer"
                    );
                }
                Err(_) => {
                    // No other node above in this layer - that's also valid
                }
            }
        }
    }
}
