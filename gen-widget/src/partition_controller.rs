use std::{
    cmp::{max, min},
    collections::HashSet,
    hash::Hash,
};

use petgraph::visit::{
    EdgeIndexable, GraphBase, IntoEdgeReferences, IntoNeighborsDirected, IntoNodeIdentifiers,
    NodeCount, NodeIndexable, Visitable,
};

use crate::{
    geometry::{BigRect, LocalPos, Point, WorldPos, WorldRect},
    layout::VisualDetail,
    partition_table::{PartitionConfig, PartitionIdxError, PartitionTable},
    plotter::NodeSizer,
    standalone_sugiyama::VERTEX_SPACING_DEFAULT,
};

/// Configuration for partition controller behavior
#[derive(Debug, Clone)]
pub struct ControllerConfig {
    /// Maximum number of partitions to keep loaded in memory
    pub max_loaded_partitions: usize,
}

impl Default for ControllerConfig {
    fn default() -> Self {
        Self {
            max_loaded_partitions: 10,
        }
    }
}

// TODO: remove the anchor partition field, it was meant
// to stabilize coordiantes as partitions were being loaded to the left
// of the current partition. However, with the way we now refresh layouts
// it wouldn't have had any benefit.
/// Domain-agnostic partition navigation and caching manager
///
/// This struct manages the lifecycle of partitions in a virtualized viewport:
/// - Handles loading/unloading of partitions based on viewport position
/// - Manages the sliding window cache of rendered partitions
/// - Provides coordinate conversion between local and world space
/// - Handles scale (level of detail) changes and layout computation
pub struct PartitionController<G, S>
where
    G: GraphBase + Clone,
    S: NodeSizer<G>,
{
    pub partition_table: PartitionTable<G>,
    pub current_detail_level: VisualDetail,
    pub node_sizer: S,

    // Dynamic partition layout management
    anchor_partition_idx: usize, // contains the origin
    loaded_partition_indices: HashSet<usize>,
    max_loaded_partitions: usize,
    original_graph: G,
    scale_change_needs_viewport_reset: bool,

    // Vertex spacing for layout computation
    vertex_spacing: f64,
}

impl<G, S> PartitionController<G, S>
where
    G: GraphBase
        + Clone
        + EdgeIndexable
        + NodeIndexable
        + NodeCount
        + Visitable
        + IntoNodeIdentifiers
        + IntoEdgeReferences
        + IntoNeighborsDirected,
    G::NodeId: Copy + Eq + Hash + Ord,
    G::EdgeId: Clone,
    for<'b> &'b G: IntoNodeIdentifiers + IntoEdgeReferences + IntoNeighborsDirected,
    for<'b> &'b G::NodeId: Hash + Ord,
    for<'b> &'b G::EdgeId: Clone,
    S: NodeSizer<G>,
{
    // TODO: a builder pattern may be nice here to support sensible defaults and multiple graph types
    /// Create a new PartitionController starting from a graph, and the following parameters:
    /// - graph: The graph to partition (note: G is often instantiated as &SomeGraph, taking ownership of a reference)
    /// - node_sizer: Function object to determine node sizes at different scales
    pub fn new(graph: G, node_sizer: S) -> Self
    where
        <G as petgraph::visit::GraphBase>::NodeId: std::fmt::Debug,
    {
        Self::new_with_config(
            graph,
            node_sizer,
            PartitionConfig::default(),
            ControllerConfig::default(),
        )
    }

    /// Create a new PartitionController with custom configuration
    /// - graph: The graph to partition
    /// - node_sizer: Function object to determine node sizes at different scales
    /// - partition_config: Configuration for partition behavior
    /// - controller_config: Configuration for memory management
    pub fn new_with_config(
        graph: G,
        node_sizer: S,
        partition_config: PartitionConfig,
        controller_config: ControllerConfig,
    ) -> Self
    where
        <G as petgraph::visit::GraphBase>::NodeId: std::fmt::Debug,
    {
        Self {
            partition_table: PartitionTable::new_with_config(
                graph,
                partition_config.layer_count,
                partition_config.node_count,
            ),
            current_detail_level: VisualDetail::Minimal,
            node_sizer,
            anchor_partition_idx: 0,
            loaded_partition_indices: HashSet::new(),
            max_loaded_partitions: controller_config.max_loaded_partitions,
            original_graph: graph,
            scale_change_needs_viewport_reset: false,
            vertex_spacing: VERTEX_SPACING_DEFAULT,
        }
    }

    /// Set the anchor partition and reset coordinate system
    pub fn set_anchor_partition(&mut self, partition_idx: usize) -> Result<(), String> {
        if partition_idx >= self.partition_table.partitions.len() {
            return Err(format!("Partition index {} out of bounds", partition_idx));
        }
        self.ensure_partition_loaded(partition_idx)?;
        self.anchor_partition_idx = partition_idx;

        // Update the partition table's anchor partition
        self.partition_table.set_anchor_partition(partition_idx)?;

        Ok(())
    }

    /// Ensure a partition is loaded. This computes the layout if needed and updates the
    /// partition table's internal width tracking. Layouts are computed for all rendering scales
    /// at the same time (base, full, truncated).
    pub fn ensure_partition_loaded(&mut self, partition_idx: usize) -> Result<(), String> {
        if self.loaded_partition_indices.contains(&partition_idx) {
            return Ok(()); // Already loaded
        }

        self.partition_table.load_partition(
            partition_idx,
            &self.node_sizer,
            &self.original_graph,
            self.vertex_spacing,
        )?;

        self.loaded_partition_indices.insert(partition_idx);

        // Evict old partitions if we exceed the limit
        self.evict_distant_partitions();

        Ok(())
    }

    pub fn get_partition_rect(&self, partition_idx: usize) -> Result<BigRect<i64>, String> {
        self.partition_table
            .get_partition_rect(partition_idx, self.current_detail_level)
    }

    /// Load partitions to cover the given query rectangle using an interleaved loading approach
    /// Loads partitions in even-odd pairs from the anchor outward
    /// Returns a sorted list of partition indices that cover the query rectangle
    pub fn load_partitions_for_rect(
        &mut self,
        query_rect: WorldRect,
    ) -> Result<Vec<usize>, String> {
        let mut covered_partitions = Vec::new();

        // First ensure the anchor partition is loaded
        self.ensure_partition_loaded(self.anchor_partition_idx)?;
        let anchor_rect = self
            .get_partition_rect(self.anchor_partition_idx)
            .expect("Anchor partition was loaded");

        // Add anchor partition to covered list
        covered_partitions.push(self.anchor_partition_idx);

        // Left side: use descending pairs pattern
        // Sections (even) and bridges (odds) alternate
        // because you need bridges to join sections together
        // and you need sections to compute the bridge
        if query_rect.left() < anchor_rect.left() && self.anchor_partition_idx > 0 {
            for partition_idx in DescendingPartitionOrder::new(self.anchor_partition_idx) {
                self.ensure_partition_loaded(partition_idx)?;
                covered_partitions.push(partition_idx);

                // Only break if we just finished a bridge partition
                if partition_idx % 2 == 1 {
                    // Check the bounds of the section partition to my left
                    let partition_rect = self
                        .get_partition_rect(partition_idx - 1)
                        .expect("Bridges are built between sections");
                    if partition_rect.left() <= query_rect.left() {
                        log::trace!("finished expanding leftwards");
                        break;
                    }
                }
            }
        }
        // Right side: use ascending pairs pattern
        if query_rect.right() > anchor_rect.right() {
            let max_partitions = self.partition_table.partitions.len();

            for partition_idx in
                AscendingPartitionOrder::new(self.anchor_partition_idx).take_while(|&idx| {
                    // Don't try to load partitions beyond array bounds
                    if idx >= max_partitions {
                        return false;
                    }
                    // Don't try to load bridge partitions that don't have a right section
                    if idx % 2 == 1 {
                        // Bridge partition - check that right section exists
                        idx + 1 < max_partitions
                    } else {
                        true
                    }
                })
            {
                self.ensure_partition_loaded(partition_idx)?;
                covered_partitions.push(partition_idx);

                if partition_idx % 2 == 1 {
                    // check the bounds of the section partition to my right
                    let partition_rect = self
                        .get_partition_rect(partition_idx + 1)
                        .expect("Bridges are built between sections");
                    if partition_rect.right() >= query_rect.right() {
                        log::trace!("finished expanding rightwards");
                        break;
                    }
                }
            }
        }

        // Sort the covered partitions and remove duplicates
        covered_partitions.sort_unstable();
        covered_partitions.dedup();

        Ok(covered_partitions)
    }

    /// Remove partitions that are far from the current viewport
    fn evict_distant_partitions(&mut self) {
        if self.loaded_partition_indices.len() <= self.max_loaded_partitions {
            return;
        }

        let widths = self
            .partition_table
            .get_widths_tree(self.current_detail_level);

        // TODO make evictions use the camera or cursor to decide what to kick out
        let reference_offset = widths.prefix_sum(self.anchor_partition_idx, 0);

        // Find partitions furthest from reference and evict them
        let mut partitions_by_distance: Vec<_> = self
            .loaded_partition_indices
            .iter()
            .map(|&idx| {
                let offset = widths.prefix_sum(idx, 0);
                let distance = (offset - reference_offset).abs();
                (idx, distance)
            })
            .collect();

        partitions_by_distance.sort_by_key(|(_, distance)| *distance);

        // Keep the closest partitions, evict the rest
        let to_evict: Vec<_> = partitions_by_distance
            .into_iter()
            .skip(self.max_loaded_partitions)
            .map(|(idx, _)| idx)
            .collect();

        for idx in to_evict {
            self.partition_table.unload_partition(idx);
            self.loaded_partition_indices.remove(&idx);
        }
    }

    /// Convert local (within a partition) coordinates to world coordinates (across all partitions)
    pub fn local_to_world(&self, local_pos: LocalPos) -> WorldPos {
        self.partition_table
            .local_to_world(local_pos, self.current_detail_level)
    }

    /// Convert world coordinates (across all partitions) to local coordinates (within a partition)
    pub fn world_to_local(&self, world_pos: WorldPos) -> Result<LocalPos, PartitionIdxError> {
        self.partition_table
            .world_to_local(world_pos, self.current_detail_level)
    }

    /// Change the current level of detail
    pub fn set_detail_level(&mut self, detail_level: VisualDetail) {
        if detail_level != self.current_detail_level {
            self.current_detail_level = detail_level;

            // Signal that viewport needs to be reset for the new coordinate system
            // Note: We don't clear layouts since each partition already has all 3 scales computed
            self.scale_change_needs_viewport_reset = true;

            // TODO: adjust origin or perform a transformation to have
            // the cursor stay in the same position on the screen
        }
    }

    /// Calculate the total bounds needed to display all partitions
    pub fn calculate_total_bounds(&mut self) -> Result<BigRect<i64>, String> {
        if self.partition_table.partitions.is_empty() {
            return Err("No partitions available".to_string());
        }

        // Start by loading all partitions to get their actual dimensions
        self.load_all_partitions()?;

        if self.loaded_partition_indices.is_empty() {
            return Err("No partitions could be loaded".to_string());
        }

        let mut min_x = i64::MAX;
        let mut max_x = i64::MIN;
        let mut min_y = i64::MAX;
        let mut max_y = i64::MIN;

        let metrics = self
            .partition_table
            .get_scale_data(self.current_detail_level);
        let widths = &metrics.widths;
        let heights = &metrics.heights;

        for &partition_idx in &self.loaded_partition_indices {
            let start_x = widths.prefix_sum(partition_idx, 0);
            let end_x = start_x
                + if partition_idx == 0 {
                    widths.prefix_sum(0, 0)
                } else {
                    widths.prefix_sum(partition_idx, 0) - widths.prefix_sum(partition_idx - 1, 0)
                };

            min_x = min(min_x, start_x);
            max_x = max(max_x, end_x);

            let height = heights[partition_idx];
            let y_offset = if partition_idx == 0 {
                self.partition_table
                    .get_scale_data(self.current_detail_level)
                    .rise
                    .prefix_sum(0, 0)
            } else {
                self.partition_table
                    .get_scale_data(self.current_detail_level)
                    .rise
                    .prefix_sum(partition_idx, 0)
                    - self
                        .partition_table
                        .get_scale_data(self.current_detail_level)
                        .rise
                        .prefix_sum(partition_idx - 1, 0)
            };

            min_y = min(min_y, y_offset);
            max_y = max(max_y, y_offset + height);
        }

        if min_y == i64::MAX {
            return Err("No nodes found in any partition".to_string());
        }

        Ok(BigRect::from_corners(
            Point::new(min_x, min_y),
            Point::new(max_x, max_y),
        ))
    }

    /// Load all partitions with proper relative positioning using ascending partition order
    fn load_all_partitions(&mut self) -> Result<(), String> {
        // First ensure the anchor partition is loaded
        self.ensure_partition_loaded(self.anchor_partition_idx)?;
        log::debug!("Loaded anchor partition {}", self.anchor_partition_idx);

        let total_partitions = self.partition_table.partitions.len();

        // Load partitions to the left of anchor using descending order
        if self.anchor_partition_idx > 0 {
            for partition_idx in DescendingPartitionOrder::new(self.anchor_partition_idx) {
                self.ensure_partition_loaded(partition_idx)?;
                log::debug!("Ensured partition {} was loaded (left side)", partition_idx);
            }
        }

        // Load partitions to the right of anchor using ascending order
        for partition_idx in AscendingPartitionOrder::new(self.anchor_partition_idx)
            .take_while(|&idx| idx < total_partitions)
        {
            self.ensure_partition_loaded(partition_idx)?;
            log::debug!(
                "Ensured partition {} was loaded (right side)",
                partition_idx
            );
        }

        Ok(())
    }

    /// Get the current anchor partition index
    pub fn get_anchor_partition(&self) -> usize {
        self.anchor_partition_idx
    }

    /// Check if viewport needs to be reset due to scale change, and clear the flag
    pub fn check_and_clear_viewport_reset_flag(&mut self) -> bool {
        let needs_reset = self.scale_change_needs_viewport_reset;
        self.scale_change_needs_viewport_reset = false;
        needs_reset
    }

    /// Get the number of currently loaded partitions
    pub fn loaded_partition_count(&self) -> usize {
        self.loaded_partition_indices.len()
    }

    /// Get information about all loaded partitions (idx, start_x, width, height)
    pub fn get_loaded_partitions_info(&self) -> Vec<(usize, i64, i64, i64)> {
        self.loaded_partition_indices
            .iter()
            .filter_map(|&idx| {
                let widths = self
                    .partition_table
                    .get_widths_tree(self.current_detail_level);
                let start_x = widths.prefix_sum(idx, 0);
                self.partition_table
                    .get_layout(idx, self.current_detail_level)
                    .map(|layout| (idx, start_x, layout.width, layout.height))
            })
            .collect()
    }

    /// Get the current vertex spacing
    pub fn get_vertex_spacing(&self) -> f64 {
        self.vertex_spacing
    }

    /// Set the vertex spacing
    pub fn set_vertex_spacing(&mut self, spacing: f64) {
        self.vertex_spacing = spacing;
    }

    /// Increment the vertex spacing by the given amount
    pub fn increment_vertex_spacing(&mut self, increment: f64) {
        self.vertex_spacing += increment;
    }

    /// Clear all layouts while keeping partitions and layer data
    pub fn clear_all_layouts(&mut self) {
        self.partition_table.clear_all_layouts();
        self.loaded_partition_indices.clear();
    }
}

/// Iterator for ascending partition computation order (for rightwards expansion from anchor)
/// Each element represents a single partition, even = sections, odd = bridges
/// Pattern: (even, odd) with odd = even - 1, i.e. 2 steps forward, one step backwards
/// Start_idx must be even and larger than 0.
/// Example for start_idx=0: [2, 1, 4, 3, 6, 5, ...]
/// This maintains section-before-bridge dependency while expanding rightward.
pub struct AscendingPartitionOrder {
    current_x: usize,
    partition_bridge: bool, // false = section (2x), true = bridge (2x-1)
}

impl AscendingPartitionOrder {
    pub fn new(start_idx: usize) -> Self {
        assert!(start_idx.is_multiple_of(2), "start_idx must be even");
        Self {
            current_x: start_idx / 2 + 1, // Start from next x
            partition_bridge: false,
        }
    }
}

impl Iterator for AscendingPartitionOrder {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let x = self.current_x;

        let result = if !self.partition_bridge {
            self.partition_bridge = true;
            2 * x
        } else {
            self.current_x += 1;
            self.partition_bridge = false;
            2 * x - 1
        };
        Some(result)
    }
}

/// Iterator for descending partition computation order (for rightwards expansion from anchor)
/// Each element represents a single partition, even = sections, odd = bridges
/// Pattern: (even, odd) with odd = even + 1, i.e. 2 steps backwards, one step backwards
/// Start_idx must be even and larger than 0.
/// Example for start_idx=2: [2, 1, 4, 3, 6, 5, ...]
/// This maintains section-before-bridge dependency while expanding rightward.
pub struct DescendingPartitionOrder {
    current_x: Option<usize>,
    is_bridge: bool,
}

impl DescendingPartitionOrder {
    pub fn new(start_idx: usize) -> Self {
        assert!(
            start_idx > 0 && start_idx.is_multiple_of(2),
            "start_idx must be even and > 0"
        );

        Self {
            current_x: Some(start_idx / 2),
            is_bridge: false,
        }
    }
}

impl Iterator for DescendingPartitionOrder {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let x = self.current_x?;

        // Pattern: load left section first, then bridge to connect it
        // From anchor 4: load 2 (section), then 3 (bridge between 2 and 4), then 0, then 1
        let result = if !self.is_bridge {
            // Check if there's a section to the left
            if x == 0 {
                return None;
            }
            // Load section to the left: 2*(x-1) = 2*x - 2
            self.is_bridge = true;
            2 * x - 2
        } else {
            // Load bridge between left section and current: 2*x - 1
            self.is_bridge = false;
            self.current_x = x.checked_sub(1);
            2 * x - 1
        };

        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_descending_partition_order_pattern() {
        let result: Vec<usize> = DescendingPartitionOrder::new(6).collect();

        let section_positions: Vec<usize> = result
            .iter()
            .enumerate()
            .filter(|&(_, &idx)| idx % 2 == 0)
            .map(|(pos, _)| pos)
            .collect();

        let bridge_positions: Vec<usize> = result
            .iter()
            .enumerate()
            .filter(|&(_, &idx)| idx % 2 == 1)
            .map(|(pos, _)| pos)
            .collect();

        // Each section should come before its corresponding bridge
        for i in 0..section_positions.len().min(bridge_positions.len()) {
            assert!(
                section_positions[i] < bridge_positions[i],
                "Section at position {} should come before bridge at position {}",
                section_positions[i],
                bridge_positions[i]
            );
        }
    }

    #[test]
    fn test_ascending_partition_order_pattern() {
        let result: Vec<usize> = AscendingPartitionOrder::new(0)
            .take_while(|&x| x <= 6)
            .collect();
        assert_eq!(result, vec![2, 1, 4, 3, 6, 5]);

        // Find actual positions in the sequence where bridges and sections appear
        let bridge_positions: Vec<usize> = result
            .iter()
            .enumerate()
            .filter(|&(_, &idx)| idx % 2 == 1)
            .map(|(pos, _)| pos)
            .collect();

        let section_positions: Vec<usize> = result
            .iter()
            .enumerate()
            .filter(|&(_, &idx)| idx % 2 == 0)
            .map(|(pos, _)| pos)
            .collect();

        // For ascending pattern: sections come first, then bridges in each pair
        // Sequence [2, 1, 4, 3, 6, 5] -> sections at [0, 2, 4], bridges at [1, 3, 5]
        assert_eq!(section_positions, vec![0, 2, 4]);
        assert_eq!(bridge_positions, vec![1, 3, 5]);

        // Each section should come before its corresponding bridge in each pair
        for i in 0..section_positions.len().min(bridge_positions.len()) {
            assert!(
                section_positions[i] < bridge_positions[i],
                "Section at position {} should come before bridge at position {}",
                section_positions[i],
                bridge_positions[i]
            );
        }
    }

    #[test]
    fn test_load_partitions_for_rect_basic() {
        use crate::{
            geometry::BigRect,
            testing::mocks::{FixedNodeSizer, TestGraphs},
        };

        // Create a domain graph for testing
        let domain_graph = TestGraphs::domain_complex_dag();
        let node_sizer = FixedNodeSizer {
            width: 10,
            height: 5,
        };

        // Create partition controller
        let mut controller = PartitionController::new(&domain_graph, node_sizer);

        // Test basic coverage - small rectangle within anchor partition
        let small_rect = BigRect::from_coords(-5, -10, 5, 10);
        let result = controller.load_partitions_for_rect(small_rect);
        assert!(
            result.is_ok(),
            "load_partitions_for_rect should succeed for small rectangle"
        );

        let covered_partitions = result.unwrap();
        assert!(
            !covered_partitions.is_empty(),
            "Should return covered partitions"
        );
        assert!(
            covered_partitions.contains(&controller.anchor_partition_idx),
            "Should include anchor partition"
        );

        // Verify anchor partition is loaded
        assert!(
            controller
                .loaded_partition_indices
                .contains(&controller.anchor_partition_idx)
        );
    }

    #[test]
    fn test_load_partitions_for_rect_expansion() {
        use crate::{
            geometry::BigRect,
            testing::mocks::{FixedNodeSizer, TestGraphs},
        };

        // Create a domain graph with enough nodes to create multiple partitions
        let domain_graph = TestGraphs::domain_complex_dag();
        let node_sizer = FixedNodeSizer {
            width: 10,
            height: 5,
        };

        // Create partition controller
        let mut controller = PartitionController::new(&domain_graph, node_sizer);

        // Test coverage that should require loading multiple partitions
        // Use a large rectangle that extends beyond the anchor partition
        let large_rect = BigRect::from_coords(-50, -20, 50, 20);
        let result = controller.load_partitions_for_rect(large_rect);
        assert!(
            result.is_ok(),
            "load_partitions_for_rect should succeed for large rectangle"
        );

        let covered_partitions = result.unwrap();
        assert!(
            !covered_partitions.is_empty(),
            "Should return covered partitions"
        );
        assert!(
            covered_partitions.contains(&controller.anchor_partition_idx),
            "Should include anchor partition"
        );

        // Verify anchor partition is loaded
        assert!(
            controller
                .loaded_partition_indices
                .contains(&controller.anchor_partition_idx)
        );
    }

    #[test]
    fn test_load_partitions_for_rect_left_expansion() {
        use crate::{
            geometry::BigRect,
            testing::mocks::{FixedNodeSizer, TestGraphs},
        };

        let domain_graph = TestGraphs::domain_complex_dag();
        let node_sizer = FixedNodeSizer {
            width: 15,
            height: 8,
        };

        let mut controller = PartitionController::new(&domain_graph, node_sizer);

        // Set anchor to a partition that's not at index 0 to enable left expansion
        if controller.partition_table.partitions.len() > 2 {
            let _ = controller.set_anchor_partition(2);

            // Test rectangle that extends to the left, requiring left expansion
            let left_rect = BigRect::from_coords(-100, -15, -10, 15);
            let result = controller.load_partitions_for_rect(left_rect);
            assert!(
                result.is_ok(),
                "load_partitions_for_rect should succeed for left expansion"
            );

            let covered_partitions = result.unwrap();
            assert!(
                !covered_partitions.is_empty(),
                "Should return covered partitions"
            );
            assert!(
                covered_partitions.contains(&controller.anchor_partition_idx),
                "Should include anchor partition"
            );

            // Should have loaded partitions to the left of anchor
            assert!(controller.loaded_partition_indices.len() > 1);
        }
    }

    #[test]
    fn test_load_all_partitions_proper_order() {
        use crate::{
            partition_table::PartitionConfig,
            testing::mocks::{FixedNodeSizer, TestGraphs},
        };

        // Create a domain graph and force small partitions to create multiple partitions
        let domain_graph = TestGraphs::domain_complex_dag();
        let node_sizer = FixedNodeSizer {
            width: 10,
            height: 5,
        };

        // Create partition controller with small partitions to force multiple partitions
        let partition_config = PartitionConfig {
            layer_count: 1, // Small layer count
            node_count: 3,  // Small node count to force multiple partitions
        };
        let config = ControllerConfig {
            max_loaded_partitions: 20,
        };

        let mut controller = PartitionController::new_with_config(
            &domain_graph,
            node_sizer,
            partition_config,
            config,
        );

        // Should create multiple partitions from 10-node graph
        assert!(
            controller.partition_table.partitions.len() > 1,
            "Should create multiple partitions"
        );

        // Test the load_all_partitions function
        let result = controller.load_all_partitions();
        assert!(result.is_ok(), "load_all_partitions should succeed");

        // Verify all partitions are loaded (including empty bridge partitions)
        // Bridge partitions (odd indices) may be empty but are needed to connect sections
        let total_partitions = controller.partition_table.partitions.len();

        assert_eq!(
            controller.loaded_partition_indices.len(),
            total_partitions,
            "All partitions (including empty bridges) should be loaded"
        );

        // Verify anchor partition is loaded
        assert!(
            controller
                .loaded_partition_indices
                .contains(&controller.anchor_partition_idx),
            "Anchor partition should be loaded"
        );

        // Test calculate_total_bounds which uses load_all_partitions
        let bounds_result = controller.calculate_total_bounds();
        assert!(
            bounds_result.is_ok(),
            "calculate_total_bounds should succeed after loading all partitions"
        );

        let bounds = bounds_result.unwrap();

        // Verify bounds make sense (should have positive width at least)
        // Note: Height may be zero if all partitions are empty/bridges
        assert!(
            bounds.min.x < bounds.max.x,
            "Bounds should have positive width: min.x={}, max.x={}",
            bounds.min.x,
            bounds.max.x
        );
        assert!(
            bounds.min.y <= bounds.max.y,
            "Bounds Y coordinates should be valid: min.y={}, max.y={}",
            bounds.min.y,
            bounds.max.y
        );
    }
}
