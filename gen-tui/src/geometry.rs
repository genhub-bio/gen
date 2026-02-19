use std::ops::{Add, Div, Mul, Sub};

use petgraph::graph::NodeIndex;
use rstar::{AABB, RTreeObject};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

// Import node-related types for spatial object variants
use crate::partition::StitchSide;

/// Groups together a domain index (index in original graph) and a layout index
/// (index in the unified layout graph) for nodes in the spatial index
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayoutNodeIndex {
    // Index in the original graph (domain-specific and ommited for added dummy nodes)
    pub domain: Option<NodeIndex<u32>>,
    // Index in the unified layout graph
    pub layout: NodeIndex<u32>,
}

impl LayoutNodeIndex {
    pub fn new(domain_index: Option<NodeIndex<u32>>, layout_index: NodeIndex<u32>) -> Self {
        Self {
            domain: domain_index,
            layout: layout_index,
        }
    }

    /// Create for a data node (has both domain and layout indices)
    pub fn data(domain_index: NodeIndex<u32>, layout_index: NodeIndex<u32>) -> Self {
        Self::new(Some(domain_index), layout_index)
    }

    /// Create for a routing or stitch node (only has layout index)
    pub fn routing(layout_index: NodeIndex<u32>) -> Self {
        Self::new(None, layout_index)
    }
}

/// Generic 2D coordinate type at the basis of our 3 coordinate systems.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Point<T> {
    pub x: T,
    pub y: T,
}

impl<T> Point<T> {
    /// Construct from raw coordinates
    pub fn new(x: T, y: T) -> Self {
        Point { x, y }
    }
}

impl<T> From<(T, T)> for Point<T> {
    fn from(tuple: (T, T)) -> Self {
        Point::new(tuple.0, tuple.1)
    }
}

impl<T: Copy> From<Point<T>> for (T, T) {
    fn from(point: Point<T>) -> Self {
        (point.x, point.y)
    }
}

impl<T: Copy> From<Point<T>> for [T; 2] {
    fn from(point: Point<T>) -> Self {
        [point.x, point.y]
    }
}

// allow `p1 + p2` and `p1 - p2`
impl<T: Add<Output = T>> Add for Point<T> {
    type Output = Point<T>;

    fn add(self, other: Point<T>) -> Point<T> {
        Point::new(self.x + other.x, self.y + other.y)
    }
}
impl<T: Sub<Output = T>> Sub for Point<T> {
    type Output = Point<T>;

    fn sub(self, other: Point<T>) -> Point<T> {
        Point::new(self.x - other.x, self.y - other.y)
    }
}
impl<T: Mul<Output = T>> Mul for Point<T> {
    type Output = Point<T>;

    fn mul(self, other: Point<T>) -> Point<T> {
        Point::new(self.x * other.x, self.y * other.y)
    }
}
impl<T: Div<Output = T>> Div for Point<T> {
    type Output = Point<T>;

    fn div(self, other: Point<T>) -> Point<T> {
        Point::new(self.x / other.x, self.y / other.y)
    }
}

// Scalar multiplication and division
impl<T: Mul<Output = T> + Copy> Mul<T> for Point<T> {
    type Output = Point<T>;

    fn mul(self, scalar: T) -> Point<T> {
        Point::new(self.x * scalar, self.y * scalar)
    }
}
impl<T: Div<Output = T> + Copy> Div<T> for Point<T> {
    type Output = Point<T>;

    fn div(self, scalar: T) -> Point<T> {
        Point::new(self.x / scalar, self.y / scalar)
    }
}

// Addition and subtraction with tuples
impl<T: Add<Output = T>> Add<(T, T)> for Point<T> {
    type Output = Point<T>;

    fn add(self, other: (T, T)) -> Point<T> {
        Point::new(self.x + other.0, self.y + other.1)
    }
}
impl<T: Sub<Output = T>> Sub<(T, T)> for Point<T> {
    type Output = Point<T>;

    fn sub(self, other: (T, T)) -> Point<T> {
        Point::new(self.x - other.0, self.y - other.1)
    }
}

// Type aliases for common coordinate systems
/// World coordinates (absolute or relative)
pub type WorldPos = Point<i64>;
/// Coordinates relative to the viewport, in viewport units
pub type ViewportPos = Point<u16>;
/// Coordinates relative to the layout object (e.g. when chunking up the world))
pub type LayoutPos = Point<i64>;

// Origin shortcuts:
impl Point<i64> {
    pub const ZERO: Point<i64> = Point { x: 0, y: 0 };
}
impl Point<u16> {
    pub const ZERO: Point<u16> = Point { x: 0, y: 0 };
}
impl Point<i32> {
    pub const ZERO: Point<i32> = Point { x: 0, y: 0 };
}

// Ratatui Rects are limited to u16 coordinates, which isn't enough for our needs,
// hence: BigRect, available with world, local, and viewport coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BigRect<T> {
    pub min: Point<T>,
    pub max: Point<T>,
}

impl<T> BigRect<T> {
    /// Create from two corner points.
    pub fn from_corners(a: Point<T>, b: Point<T>) -> Self
    where
        T: Ord + Copy + Serialize + DeserializeOwned,
    {
        // Regardless of which corners are passed in, we store
        // the bottom left and top right corners.
        let min = Point::new(a.x.min(b.x), a.y.min(b.y));
        let max = Point::new(a.x.max(b.x), a.y.max(b.y));
        BigRect { min, max }
    }

    /// Create from raw coordinates.
    /// Automatically corrects ordering if min/max values are swapped.
    pub fn from_coords(min_x: T, min_y: T, max_x: T, max_y: T) -> Self
    where
        T: Ord + Copy,
    {
        BigRect {
            min: Point::new(min_x.min(max_x), min_y.min(max_y)),
            max: Point::new(min_x.max(max_x), min_y.max(max_y)),
        }
    }
}

impl<T> BigRect<T>
where
    T: Copy
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + PartialOrd
        + From<i32>
        + Serialize
        + DeserializeOwned
        + Ord,
{
    /// Left side x coordinate
    pub fn left(&self) -> T {
        self.min.x
    }

    /// Right side x coordinate
    pub fn right(&self) -> T {
        self.max.x
    }

    /// Bottom side y coordinate
    pub fn bottom(&self) -> T {
        self.min.y
    }

    /// Top side y coordinate
    pub fn top(&self) -> T {
        self.max.y
    }

    /// Mid‐point of the rectangle.
    pub fn center(&self) -> Point<T> {
        let two = T::from(2);
        Point::new(
            (self.min.x + self.max.x) / two,
            (self.min.y + self.max.y) / two,
        )
    }

    /// Midpoint of the left edge.
    pub fn left_center(&self) -> Point<T> {
        let two = T::from(2);
        Point::new(self.min.x, (self.min.y + self.max.y) / two)
    }

    /// Midpoint of the right edge.
    pub fn right_center(&self) -> Point<T> {
        let two = T::from(2);
        Point::new(self.max.x, (self.min.y + self.max.y) / two)
    }

    /// Width  = max.x – min.x
    pub fn width(&self) -> T {
        self.max.x - self.min.x
    }

    /// Height = max.y – min.y
    pub fn height(&self) -> T {
        self.max.y - self.min.y
    }

    /// Size vector = (width, height)
    pub fn size(&self) -> Point<T> {
        Point::new(self.width(), self.height())
    }

    /// Create a rectangle from a center point and size
    /// Uses floor division for left/bottom edges and ensures correct total size
    pub fn from_center_and_size(center: Point<T>, size: (u64, u64)) -> Self
    where
        T: From<i64> + std::ops::Sub<Output = T> + std::ops::Add<Output = T> + Copy,
    {
        // For odd sizes, center is unambiguous (e.g., size 5: positions 0,1,2,3,4 with center at 2)
        // For even sizes, we use floor division, biasing toward left/bottom
        // (e.g., size 4: positions 0,1,2,3 - we place center between 1 and 2, closer to 1)
        let width = size.0 as i64;
        let height = size.1 as i64;

        // Calculate half sizes using floor division
        let half_width_left = (width - 1) / 2;
        let half_height_bottom = (height - 1) / 2;

        // Calculate the min corner (left-bottom)
        let min_corner = Point::new(
            center.x - T::from(half_width_left),
            center.y - T::from(half_height_bottom),
        );

        // Calculate max corner directly from min + size - 1
        let max_corner = Point::new(
            min_corner.x + T::from(width - 1),
            min_corner.y + T::from(height - 1),
        );

        Self {
            min: min_corner,
            max: max_corner,
        }
    }

    /// All four corners in [min, top_right, max, bottom_left] order.
    pub fn corners(&self) -> [Point<T>; 4] {
        [
            self.min,
            Point::new(self.max.x, self.min.y),
            self.max,
            Point::new(self.min.x, self.max.y),
        ]
    }

    /// Does this rect (inclusive) contain the point?
    pub fn contains(&self, p: Point<T>) -> bool {
        p.x >= self.min.x && p.x <= self.max.x && p.y >= self.min.y && p.y <= self.max.y
    }

    /// Do these two rects overlap?
    pub fn intersects(&self, other: &Self) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
    }

    /// Compute the intersection of two rectangles.
    /// Returns Some(intersection_rect) if they overlap, None otherwise.
    pub fn intersection(&self, other: &Self) -> Option<Self> {
        if !self.intersects(other) {
            return None;
        }

        let min_x = self.min.x.max(other.min.x);
        let min_y = self.min.y.max(other.min.y);
        let max_x = self.max.x.min(other.max.x);
        let max_y = self.max.y.min(other.max.y);

        Some(BigRect::from_corners(
            Point::new(min_x, min_y),
            Point::new(max_x, max_y),
        ))
    }

    /// Proportional resize about the center using a floating-point scale.
    pub fn resize(&self, scale: f64) -> Self
    where
        T: Into<i64> + From<i64> + Add<Output = T> + Sub<Output = T> + Copy,
    {
        let center = self.center();

        // Chain the conversions: T -> i64 -> f64
        let width_float = self.width().into() as f64;
        let height_float = self.height().into() as f64;

        let new_width = (width_float * scale).round() as u64;
        let new_height = (height_float * scale).round() as u64;

        BigRect::from_center_and_size(center, (new_width, new_height))
    }

    /// Transform the rectangle by applying a function to the corner points.
    pub fn transform<F>(&self, f: F) -> Self
    where
        F: Fn(Point<T>) -> Point<T>,
    {
        BigRect::from_corners(f(self.min), f(self.max))
    }

    /// Find the closest cell in this rectangle to the given point
    /// Useful for snapping cursor positions to valid cells within the rectangle
    pub fn find_closest_cell(&self, p: Point<T>) -> Point<T> {
        Point::new(
            p.x.clamp(self.min.x, self.max.x),
            p.y.clamp(self.min.y, self.max.y),
        )
    }
}

// Type aliases for common rectangle types
pub type WorldRect = BigRect<i64>;
pub type ViewportRect = BigRect<u16>;

/// Clamp a world-coordinate vector to optional bounds. If `bounds` is `None`,
/// simply returns the input unchanged.
pub fn clamp_to_bounds(pos: WorldPos, bounds: Option<WorldRect>) -> WorldPos {
    bounds.map_or(pos, |rect| {
        WorldPos::new(
            pos.x.clamp(rect.min.x, rect.max.x),
            pos.y.clamp(rect.min.y, rect.max.y),
        )
    })
}

/// Type of spatial object in the unified RTree
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpatialObjectType {
    DataNode(NodeIndex<u32>),    // A data graph node with original node index
    RoutingNode(NodeIndex<u32>), // A routing node with node index for on-the-fly glyph computation
    StitchNode(StitchSide),      // A stitch node with side information
    HorizontalEdge,              // Horizontal edge segment
    VerticalEdge,                // Vertical edge segment
    AngledEdge,                  // Edge segment that is neither horizontal nor vertical
}

/// Represents a spatial object (node or edge) in the layout's spatial index.
/// This allows for efficient viewport culling and spatial queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayoutObject {
    pub rect: BigRect<i64>,
    pub object_type: SpatialObjectType,
    pub primary_node: LayoutNodeIndex,
    pub secondary_node: Option<LayoutNodeIndex>, // For edges, this is the target node
}

impl RTreeObject for LayoutObject {
    type Envelope = AABB<[i64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        AABB::from_corners(
            [self.rect.min.x, self.rect.min.y],
            [self.rect.max.x, self.rect.max.y],
        )
    }
}

impl LayoutObject {
    /// Create a data node rectangle (with specified size)
    pub fn node(
        center: LayoutPos,
        size: (u64, u64),
        layout_idx: NodeIndex<u32>,
        domain_idx: NodeIndex<u32>,
    ) -> Self {
        // Use the same centering logic as BigRect::from_center_and_size to ensure consistency
        // between layout positioning and rendering
        let rect = BigRect::from_center_and_size(center, size);
        Self {
            rect,
            object_type: SpatialObjectType::DataNode(domain_idx),
            primary_node: LayoutNodeIndex::data(domain_idx, layout_idx),
            secondary_node: None,
        }
    }

    /// Create a routing node 1x1 square
    pub fn routing_node(center: LayoutPos, layout_idx: NodeIndex<u32>) -> Self {
        Self {
            rect: BigRect::from_coords(center.x, center.y, center.x + 1, center.y + 1),
            object_type: SpatialObjectType::RoutingNode(layout_idx),
            primary_node: LayoutNodeIndex::routing(layout_idx),
            secondary_node: None,
        }
    }

    /// Create a stitch node 1x1 square
    pub fn stitch_node(
        center: LayoutPos,
        layout_idx: NodeIndex<u32>,
        stitch_side: StitchSide,
    ) -> Self {
        Self {
            rect: BigRect::from_coords(center.x, center.y, center.x + 1, center.y + 1),
            object_type: SpatialObjectType::StitchNode(stitch_side),
            primary_node: LayoutNodeIndex::routing(layout_idx),
            secondary_node: None,
        }
    }

    /// Create a line segment (1 unit tall or wide)
    /// Takes layout indices for source and target nodes
    pub fn line(
        start: LayoutPos,
        end: LayoutPos,
        source_layout: NodeIndex<u32>,
        target_layout: NodeIndex<u32>,
    ) -> Self {
        let (min_x, max_x) = if start.x <= end.x {
            (start.x, end.x)
        } else {
            (end.x, start.x)
        };
        let (min_y, max_y) = if start.y <= end.y {
            (start.y, end.y)
        } else {
            (end.y, start.y)
        };

        let object_type = if start.x == end.x {
            SpatialObjectType::VerticalEdge
        } else if start.y == end.y {
            SpatialObjectType::HorizontalEdge
        } else {
            SpatialObjectType::AngledEdge
        };

        // For edges, we use routing indices (no domain index) for both endpoints
        Self {
            rect: BigRect::from_coords(min_x, min_y, max_x, max_y),
            object_type,
            primary_node: LayoutNodeIndex::routing(source_layout),
            secondary_node: Some(LayoutNodeIndex::routing(target_layout)),
        }
    }

    /// Check if this is a node object (any type of node)
    pub fn is_node(&self) -> bool {
        matches!(
            self.object_type,
            SpatialObjectType::DataNode(_)
                | SpatialObjectType::RoutingNode(_)
                | SpatialObjectType::StitchNode(_)
        )
    }

    /// Check if this is an edge object
    pub fn is_edge(&self) -> bool {
        matches!(
            self.object_type,
            SpatialObjectType::HorizontalEdge
                | SpatialObjectType::VerticalEdge
                | SpatialObjectType::AngledEdge
        )
    }

    /// Get the layout node index if this is a node object
    pub fn get_node(&self) -> Result<NodeIndex<u32>, String> {
        match self.object_type {
            SpatialObjectType::DataNode(_)
            | SpatialObjectType::RoutingNode(_)
            | SpatialObjectType::StitchNode(_) => Ok(self.primary_node.layout),
            _ => Err(format!(
                "Object is not a node, it's a {:?}",
                self.object_type
            )),
        }
    }

    /// Get the edge endpoints (layout indices) if this is an edge object
    pub fn get_edge(&self) -> Result<(NodeIndex<u32>, NodeIndex<u32>), String> {
        match self.object_type {
            SpatialObjectType::HorizontalEdge
            | SpatialObjectType::VerticalEdge
            | SpatialObjectType::AngledEdge => match self.secondary_node {
                Some(target) => Ok((self.primary_node.layout, target.layout)),
                None => Err("Edge object missing target node".to_string()),
            },
            _ => Err(format!(
                "Object is not an edge, it's a {:?}",
                self.object_type
            )),
        }
    }

    /// Get the center point of the object
    pub fn get_center(&self) -> LayoutPos {
        self.rect.center()
    }

    pub fn get_left_side(&self) -> i64 {
        self.rect.min.x
    }

    pub fn get_right_side(&self) -> i64 {
        self.rect.max.x
    }

    pub fn is_horizontal(&self) -> bool {
        self.object_type == SpatialObjectType::HorizontalEdge
    }

    pub fn is_vertical(&self) -> bool {
        self.object_type == SpatialObjectType::VerticalEdge
    }

    pub fn node_idx(&self) -> NodeIndex<u32> {
        self.primary_node.layout
    }
}

pub type PartitionIndex = usize; // Index of the partition within the partition table

/// A position that includes both the partition index and local coordinates within that partition.
/// This ensures type safety by making it impossible to use coordinates from one partition
/// in the context of another partition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LocalPos {
    pub partition_idx: PartitionIndex,
    pub x: i64,
    pub y: i64,
}

impl LocalPos {
    pub fn new(partition_idx: PartitionIndex, pos: LayoutPos) -> Self {
        Self {
            partition_idx,
            x: pos.x,
            y: pos.y,
        }
    }

    pub fn new_xy(partition_idx: PartitionIndex, x: i64, y: i64) -> Self {
        Self {
            partition_idx,
            x,
            y,
        }
    }

    /// Get the position as a LayoutPos
    pub fn pos(&self) -> LayoutPos {
        LayoutPos::new(self.x, self.y)
    }

    /// Get the position as a Point (referenced to the partition its in)
    pub fn point(&self) -> Point<i64> {
        Point::new(self.x, self.y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_arithmetic() {
        let p1 = Point::new(3, 4);
        let p2 = Point::new(1, 2);

        assert_eq!(p1 + p2, Point::new(4, 6));
        assert_eq!(p1 - p2, Point::new(2, 2));
        assert_eq!(p1 * p2, Point::new(3, 8));
        assert_eq!(p1 / p2, Point::new(3, 2));

        // Scalar operations
        assert_eq!(p1 * 2, Point::new(6, 8));
        assert_eq!(p1 / 2, Point::new(1, 2));
    }

    #[test]
    fn test_rect_operations() {
        let rect = BigRect::from_coords(0, 0, 10, 10);

        assert_eq!(rect.center(), Point::new(5, 5));
        assert_eq!(rect.width(), 10);
        assert_eq!(rect.height(), 10);
        assert!(rect.contains(Point::new(5, 5)));
        assert!(!rect.contains(Point::new(15, 15)));

        let rect2 = BigRect::from_coords(5, 5, 15, 15);
        assert!(rect.intersects(&rect2));

        let intersection = rect.intersection(&rect2).unwrap();
        assert_eq!(intersection.min, Point::new(5, 5));
        assert_eq!(intersection.max, Point::new(10, 10));
    }

    #[test]
    fn test_clamp_to_bounds() {
        let bounds = Some(WorldRect::from_coords(0, 0, 10, 10));

        assert_eq!(
            clamp_to_bounds(WorldPos::new(5, 5), bounds),
            WorldPos::new(5, 5)
        );
        assert_eq!(
            clamp_to_bounds(WorldPos::new(-5, -5), bounds),
            WorldPos::new(0, 0)
        );
        assert_eq!(
            clamp_to_bounds(WorldPos::new(15, 15), bounds),
            WorldPos::new(10, 10)
        );

        // No bounds
        assert_eq!(
            clamp_to_bounds(WorldPos::new(100, 100), None),
            WorldPos::new(100, 100)
        );
    }

    #[test]
    fn test_local_pos() {
        let local = LocalPos::new_xy(1, 50, 100);

        assert_eq!(local.partition_idx, 1);
        assert_eq!(local.pos(), LayoutPos::new(50, 100));
    }

    #[test]
    fn test_from_center_and_size() {
        // Test odd width and height (5x5)
        let rect1 = BigRect::from_center_and_size(Point::new(10i64, 10i64), (5, 5));
        assert_eq!(rect1.min, Point::new(8, 8));
        assert_eq!(rect1.max, Point::new(12, 12));
        // The rectangle spans 5 discrete positions: [8,9,10,11,12]
        // Width as calculated by the width() method: max.x - min.x = 12 - 8 = 4
        // This is correct for inclusive coordinates where actual span = width() + 1

        // Test even width and height (4x6)
        let rect2 = BigRect::from_center_and_size(Point::new(10i64, 10i64), (4, 6));
        assert_eq!(rect2.min, Point::new(9, 8));
        assert_eq!(rect2.max, Point::new(12, 13));
        // Width span is 4 cells: [9,10,11,12]
        // Height span is 6 cells: [8,9,10,11,12,13]

        // Test size 1x1
        let rect3 = BigRect::from_center_and_size(Point::new(0i64, 0i64), (1, 1));
        assert_eq!(rect3.min, Point::new(0, 0));
        assert_eq!(rect3.max, Point::new(0, 0));
        // Single cell at position [0,0]

        // Verify that left() and right() work correctly
        let rect4 = BigRect::from_center_and_size(Point::new(100i64, 50i64), (10, 5));
        assert_eq!(rect4.left(), 96); // 100 - (10-1)/2 = 100 - 4 = 96
        assert_eq!(rect4.right(), 105); // 96 + (10-1) = 96 + 9 = 105
        // Spans 10 cells: [96,97,98,99,100,101,102,103,104,105]
    }
}
