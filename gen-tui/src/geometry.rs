use std::ops::{Add, Div, Mul, Sub};

use serde::{Deserialize, Serialize, de::DeserializeOwned};

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
        let half_width_left = floor_half(width);
        let half_height_bottom = floor_half(height);

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

    /// Find the closest cell in this rectangle to the given point
    /// Useful for snapping cursor positions to valid cells within the rectangle
    pub fn find_closest_cell(&self, p: Point<T>) -> Point<T> {
        Point::new(
            p.x.clamp(self.min.x, self.max.x),
            p.y.clamp(self.min.y, self.max.y),
        )
    }
}

/// Floor-divided half-extent of a length `n` (e.g. width or height), biasing the center
/// toward the left/bottom for even lengths. Shared by every place that centers a rect or a
/// camera view around a point given only a size.
pub fn floor_half(n: i64) -> i64 {
    (n - 1) / 2
}

// Type aliases for common rectangle types
pub type WorldRect = BigRect<i64>;

impl BigRect<i64> {
    /// The point within this rect at fractional offset `frac` (0.0-1.0 on each axis, relative
    /// to the bottom-left corner), rounded to the nearest cell.
    pub fn point_at_fraction(&self, frac: (f64, f64)) -> Point<i64> {
        Point::new(
            self.left() + (frac.0 * self.width() as f64).round() as i64,
            self.bottom() + (frac.1 * self.height() as f64).round() as i64,
        )
    }

    /// Inverse of `point_at_fraction`: the fractional offset of `p` within this rect, clamped
    /// to [0.0, 1.0]. Guards against a zero-span rect (a 1-cell-wide node) by treating it as a
    /// span of 1.0.
    pub fn fraction_of(&self, p: Point<i64>) -> (f64, f64) {
        let span_x = self.width().max(1) as f64;
        let span_y = self.height().max(1) as f64;
        (
            ((p.x - self.left()) as f64 / span_x).clamp(0.0, 1.0),
            ((p.y - self.bottom()) as f64 / span_y).clamp(0.0, 1.0),
        )
    }
}

/// A position in one window's local coordinate space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LocalPos {
    pub x: i64,
    pub y: i64,
}

impl LocalPos {
    pub fn new(point: Point<i64>) -> Self {
        Self {
            x: point.x,
            y: point.y,
        }
    }

    pub fn new_xy(x: i64, y: i64) -> Self {
        Self { x, y }
    }

    /// Return this position as a point.
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
