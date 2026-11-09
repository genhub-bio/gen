//! The trait through which an application sizes and draws its nodes, plus the column
//! mapping types that trait exposes.

use std::sync::Arc;

use petgraph::visit::GraphBase;

use crate::{geometry::WorldRect, layout::VerticalAnchor, world_buffer::WorldBuffer};

/// A stretch of a node's raw offsets and the columns it is drawn in. A stretch drawn as one
/// cell, such as an omitted run of bases, has `column_end == column_start + 1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColumnRun {
    pub raw_start: i64,
    pub raw_end: i64,
    pub column_start: i64,
    pub column_end: i64,
}

/// Sequence columns available to the cursor, excluding decorative padding and omitted runs.
#[derive(Clone, Debug)]
pub enum CursorColumns {
    /// Every column in this inclusive range is a sequence character.
    Range(i64, i64),
    /// Only these ordered columns contain retained sequence characters.
    Positions(Vec<i64>),
}

impl CursorColumns {
    /// First and last accessible column, inclusive.
    pub fn bounds(&self) -> (i64, i64) {
        match self {
            Self::Range(first, last) => (*first, *last),
            Self::Positions(columns) => (
                *columns.first().unwrap_or(&0),
                *columns.last().unwrap_or(&0),
            ),
        }
    }

    /// Closest accessible column, preferring the left column on ties.
    pub fn nearest(&self, column: i64) -> i64 {
        match self {
            Self::Range(first, last) => column.clamp(*first, *last),
            Self::Positions(columns) => columns
                .iter()
                .copied()
                .min_by_key(|candidate| candidate.abs_diff(column))
                .unwrap_or(0),
        }
    }

    /// First accessible column at or beyond a requested column in the travel direction.
    pub fn next(&self, column: i64, forward: bool) -> Option<i64> {
        match self {
            Self::Range(first, last) => (*first..=*last).contains(&column).then_some(column),
            Self::Positions(columns) => {
                if forward {
                    columns
                        .iter()
                        .copied()
                        .find(|candidate| *candidate >= column)
                } else {
                    columns
                        .iter()
                        .rev()
                        .copied()
                        .find(|candidate| *candidate <= column)
                }
            }
        }
    }
}

/// Domain-side trait bundling node sizing and rendering for a single, fixed
/// level of detail. The application picks which concrete `NodeRenderer`
/// is active (e.g. per zoom level) and borrows it immutably for rendering.
pub trait NodeRenderer<G>
where
    G: GraphBase,
{
    /// Get the dimensions (width, height) for a node.
    fn get_node_size(&self, node: &G::NodeId) -> (u64, u64);

    /// Row used for layout alignment and edge attachment. Top-anchored renderers draw their
    /// sequence on the top row and grow decoration downward. Changes invalidate cached sizes.
    fn vertical_anchor(&self, _node: &G::NodeId) -> VerticalAnchor {
        VerticalAnchor::Center
    }

    /// Map a raw node-local coordinate to a displayed column.
    fn map_column(&self, _node: &G::NodeId, raw: i64) -> i64 {
        raw
    }

    /// Map a displayed column back to a raw node-local coordinate.
    fn raw_column(&self, _node: &G::NodeId, column: i64) -> i64 {
        column
    }

    /// How the node's raw offsets are spread over its columns, in offset order, when the
    /// renderer compresses some stretches into a single cell. `None` means the offsets map
    /// evenly from `map_column`'s first to last column, so a scale along the node can
    /// interpolate. A renderer that compresses must answer, or such a scale misplaces every
    /// position that follows a compressed stretch.
    fn column_runs(&self, _node: &G::NodeId) -> Option<Vec<ColumnRun>> {
        None
    }

    /// Get default dimensions for dummy/routing nodes
    fn get_dummy_size(&self) -> (u64, u64) {
        (1, 1)
    }

    /// Render a node in its allocated world area.
    fn render_node(&self, buffer: &mut WorldBuffer, area: WorldRect, node_id: &G::NodeId);

    /// Whether `node` is drawn. An invisible node is still placed in the layout, so its edges
    /// meet at it, but it is drawn as a junction of those edges rather than as a node.
    fn is_visible(&self, _node: &G::NodeId) -> bool {
        true
    }

    /// The one row of `node` the cursor may sit on, counted up from the bottom of its rect, or
    /// `None` to let the cursor use every row. A renderer that draws decoration around a node's
    /// content (e.g. annotation lanes above and below a sequence row) names the content row, so
    /// cursor moves, clicks, and jumps always land on it.
    fn cursor_row(&self, _node: &G::NodeId) -> Option<u64> {
        None
    }

    /// Columns containing sequence characters; `None` permits every column of the node.
    fn cursor_columns(&self, _node: &G::NodeId) -> Option<CursorColumns> {
        None
    }

    /// The node-local sequence rectangle that may receive highlights and dimming.
    /// Decoration outside this rectangle leaves previously painted edges untouched.
    fn content_rect(&self, node: &G::NodeId) -> WorldRect {
        let (width, height) = self.get_node_size(node);
        let (left, right) = self
            .cursor_columns(node)
            .map_or((0, width.saturating_sub(1) as i64), |columns| {
                columns.bounds()
            });
        let (bottom, top) = self
            .cursor_row(node)
            .map_or((0, height.saturating_sub(1) as i64), |row| {
                (row as i64, row as i64)
            });
        WorldRect::from_coords(left, bottom, right, top)
    }

    /// Whether the attachment row of `node` should sit at y = 0. Layout aligns nodes across layers while
    /// preserving within-layer spacing and straight routing chains where feasible. A renderer
    /// marks e.g. the reference sequence so the main path reads as a straight line.
    fn is_aligned(&self, _node: &G::NodeId) -> bool {
        false
    }

    /// A counter that changes whenever `get_node_size`, `get_dummy_size`, or `vertical_anchor` may answer
    /// differently than before. Views reuse a window's routed geometry until it changes, so a
    /// renderer whose sizes depend on mutable state (e.g. annotation lanes under each node) must
    /// bump it on every real change; a renderer with fixed sizes keeps the default.
    fn size_generation(&self) -> u64 {
        0
    }
}

/// Forward rendering through a boxed renderer.
impl<G, T> NodeRenderer<G> for Box<T>
where
    G: GraphBase,
    T: NodeRenderer<G> + ?Sized,
{
    fn get_node_size(&self, node: &G::NodeId) -> (u64, u64) {
        (**self).get_node_size(node)
    }

    fn vertical_anchor(&self, node: &G::NodeId) -> VerticalAnchor {
        (**self).vertical_anchor(node)
    }

    fn get_dummy_size(&self) -> (u64, u64) {
        (**self).get_dummy_size()
    }

    fn map_column(&self, node: &G::NodeId, raw: i64) -> i64 {
        (**self).map_column(node, raw)
    }

    fn raw_column(&self, node: &G::NodeId, column: i64) -> i64 {
        (**self).raw_column(node, column)
    }

    fn column_runs(&self, node: &G::NodeId) -> Option<Vec<ColumnRun>> {
        (**self).column_runs(node)
    }

    fn render_node(&self, buffer: &mut WorldBuffer, area: WorldRect, node_id: &G::NodeId) {
        (**self).render_node(buffer, area, node_id);
    }

    fn is_visible(&self, node: &G::NodeId) -> bool {
        (**self).is_visible(node)
    }

    fn cursor_row(&self, node: &G::NodeId) -> Option<u64> {
        (**self).cursor_row(node)
    }

    fn cursor_columns(&self, node: &G::NodeId) -> Option<CursorColumns> {
        (**self).cursor_columns(node)
    }

    fn content_rect(&self, node: &G::NodeId) -> WorldRect {
        (**self).content_rect(node)
    }

    fn is_aligned(&self, node: &G::NodeId) -> bool {
        (**self).is_aligned(node)
    }

    fn size_generation(&self) -> u64 {
        (**self).size_generation()
    }
}

/// Forward rendering through a shared renderer.
impl<G, T> NodeRenderer<G> for Arc<T>
where
    G: GraphBase,
    T: NodeRenderer<G> + ?Sized,
{
    fn get_node_size(&self, node: &G::NodeId) -> (u64, u64) {
        (**self).get_node_size(node)
    }

    fn vertical_anchor(&self, node: &G::NodeId) -> VerticalAnchor {
        (**self).vertical_anchor(node)
    }

    fn get_dummy_size(&self) -> (u64, u64) {
        (**self).get_dummy_size()
    }

    fn map_column(&self, node: &G::NodeId, raw: i64) -> i64 {
        (**self).map_column(node, raw)
    }

    fn raw_column(&self, node: &G::NodeId, column: i64) -> i64 {
        (**self).raw_column(node, column)
    }

    fn column_runs(&self, node: &G::NodeId) -> Option<Vec<ColumnRun>> {
        (**self).column_runs(node)
    }

    fn render_node(&self, buffer: &mut WorldBuffer, area: WorldRect, node_id: &G::NodeId) {
        (**self).render_node(buffer, area, node_id);
    }

    fn is_visible(&self, node: &G::NodeId) -> bool {
        (**self).is_visible(node)
    }

    fn cursor_row(&self, node: &G::NodeId) -> Option<u64> {
        (**self).cursor_row(node)
    }

    fn cursor_columns(&self, node: &G::NodeId) -> Option<CursorColumns> {
        (**self).cursor_columns(node)
    }

    fn content_rect(&self, node: &G::NodeId) -> WorldRect {
        (**self).content_rect(node)
    }

    fn is_aligned(&self, node: &G::NodeId) -> bool {
        (**self).is_aligned(node)
    }

    fn size_generation(&self) -> u64 {
        (**self).size_generation()
    }
}
