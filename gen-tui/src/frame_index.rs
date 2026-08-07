use std::{collections::HashMap, hash::Hash};

use crate::geometry::{Point, WorldRect};

/// Horizontal/vertical navigation direction over placed nodes: `Left`/`Right` cross to the
/// adjacent layer, `Up`/`Down` move within the current layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Left,
    Right,
    Up,
    Down,
}

/// One node the painter assigned coordinates to: its signed screen rect (the camera can
/// place nodes at negative screen coordinates) plus the layer navigation needs.
/// `layer` mirrors `LayoutNode::layer`.
#[derive(Debug, Clone, Copy)]
pub struct PlacedNode<N> {
    pub id: N,
    pub rect: WorldRect,
    pub layer: i32,
}

/// The per-frame product of a `GraphPainter::render` call: every node the painter placed,
/// including ones outside the visible area, plus a visible-hit subset restricted to the
/// clipped, clickable area. Rebuilt from scratch every render; nothing world-space survives
/// a frame past this. Indexing placed (not just painted) nodes matters: navigation from the
/// rightmost visible node, camera rebasing during a drag, and closest-node queries over empty
/// screen regions all need nodes just outside the viewport.
#[derive(Debug, Clone)]
pub struct FrameIndex<N> {
    placed: Vec<PlacedNode<N>>,
    by_id: HashMap<N, usize>,
    visible: Vec<usize>,
}

impl<N> Default for FrameIndex<N> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<N> FrameIndex<N> {
    /// An index over no nodes at all, for an empty graph or a render that produced nothing.
    pub fn empty() -> Self {
        Self {
            placed: Vec::new(),
            by_id: HashMap::new(),
            visible: Vec::new(),
        }
    }

    /// Whether any node was placed at all.
    pub fn is_empty(&self) -> bool {
        self.placed.is_empty()
    }
}

impl<N: Copy + Eq + Hash> FrameIndex<N> {
    /// Build a frame index from every placed node plus the clip area (in the same signed
    /// screen-coordinate space as each node's `rect`). Nodes intersecting `area` become the
    /// visible-hit subset; nodes entirely outside it stay indexed but are not returned by
    /// `hit`/`closest`.
    pub fn build(placed: Vec<PlacedNode<N>>, area: WorldRect) -> Self {
        let mut by_id = HashMap::with_capacity(placed.len());
        let mut visible = Vec::new();
        for (index, node) in placed.iter().enumerate() {
            by_id.insert(node.id, index);
            if node.rect.intersects(&area) {
                visible.push(index);
            }
        }
        Self {
            placed,
            by_id,
            visible,
        }
    }

    /// The screen rect a node was placed at, whether or not it is currently visible.
    pub fn rect_of(&self, id: N) -> Option<WorldRect> {
        self.by_id.get(&id).map(|&index| self.placed[index].rect)
    }

    /// The layer a node was placed in, whether or not it is currently visible.
    pub fn layer_of(&self, id: N) -> Option<i32> {
        self.by_id.get(&id).map(|&index| self.placed[index].layer)
    }

    /// Every node the painter placed, whether or not it is currently visible.
    pub fn ids(&self) -> impl Iterator<Item = N> + '_ {
        self.placed.iter().map(|node| node.id)
    }

    /// Every node the painter placed that intersected the render area on the last frame - the
    /// same subset used by `hit`/`closest`. Lets a second view over the same frame mirror
    /// which nodes are on screen in this one.
    pub fn visible_ids(&self) -> impl Iterator<Item = N> + '_ {
        self.visible.iter().map(|&index| self.placed[index].id)
    }

    /// The node whose visible rect contains `pos`, plus the fractional offset (0.0-1.0 on
    /// each axis, relative to the rect's bottom-left) of `pos` within that rect. Only
    /// considers the visible subset, matching click-to-select semantics.
    pub fn hit(&self, pos: Point<i64>) -> Option<(N, (f64, f64))> {
        self.visible.iter().find_map(|&index| {
            let node = &self.placed[index];
            if !node.rect.contains(pos) {
                return None;
            }
            Some((node.id, node.rect.fraction_of(pos)))
        })
    }

    /// The visible node whose rect is closest to `pos` (nearest-cell distance). Used for
    /// free-pan camera rebasing onto whatever is currently on screen.
    pub fn closest(&self, pos: Point<i64>) -> Option<N> {
        self.visible
            .iter()
            .min_by_key(|&&index| {
                let rect = self.placed[index].rect;
                let closest_cell = rect.find_closest_cell(pos);
                let dx = closest_cell.x - pos.x;
                let dy = closest_cell.y - pos.y;
                dx * dx + dy * dy
            })
            .map(|&index| self.placed[index].id)
    }

    /// The neighbouring placed node in `direction`, considering every placed node (not just
    /// the visible subset) so navigation can reach nodes just outside the viewport.
    /// `Left`/`Right` pick the nearest node in the nearest adjacent layer in that direction;
    /// `Up`/`Down` pick the nearest node within the same layer.
    pub fn neighbor(&self, id: N, direction: Direction) -> Option<N> {
        let index = *self.by_id.get(&id)?;
        let origin = self.placed[index];

        match direction {
            Direction::Left | Direction::Right => {
                let mut candidates: Vec<&PlacedNode<N>> = self
                    .placed
                    .iter()
                    .filter(|node| {
                        if direction == Direction::Right {
                            node.layer > origin.layer
                        } else {
                            node.layer < origin.layer
                        }
                    })
                    .collect();
                candidates.sort_by_key(|node| {
                    let layer_distance = (node.layer - origin.layer).abs();
                    let cross_distance = (node.rect.center().y - origin.rect.center().y).abs();
                    (layer_distance, cross_distance)
                });
                candidates.first().map(|node| node.id)
            }
            Direction::Up | Direction::Down => {
                let mut candidates: Vec<&PlacedNode<N>> = self
                    .placed
                    .iter()
                    .filter(|node| {
                        node.layer == origin.layer
                            && if direction == Direction::Up {
                                node.rect.center().y > origin.rect.center().y
                            } else {
                                node.rect.center().y < origin.rect.center().y
                            }
                    })
                    .collect();
                candidates
                    .sort_by_key(|node| (node.rect.center().y - origin.rect.center().y).abs());
                candidates.first().map(|node| node.id)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::WorldPos;

    fn placed(id: u32, rect: WorldRect, layer: i32) -> PlacedNode<u32> {
        PlacedNode { id, rect, layer }
    }

    fn small_layout() -> Vec<PlacedNode<u32>> {
        vec![
            placed(0, WorldRect::from_coords(0, 0, 2, 2), 0),
            placed(1, WorldRect::from_coords(10, 0, 12, 2), 1),
            placed(2, WorldRect::from_coords(10, 10, 12, 12), 1),
            // Placed but off the (0,0)-(20,20) clip area used below.
            placed(3, WorldRect::from_coords(30, 0, 32, 2), 2),
        ]
    }

    #[test]
    fn test_rect_of_and_visibility() {
        let area = WorldRect::from_coords(0, 0, 20, 20);
        let index = FrameIndex::build(small_layout(), area);

        assert_eq!(index.rect_of(0), Some(WorldRect::from_coords(0, 0, 2, 2)));
        // Node 3 is placed but outside the clip area, so it is not hit-testable...
        assert_eq!(index.hit(WorldPos::new(31, 1)), None);
        // ...yet its rect is still indexed (offscreen navigation still needs it).
        assert_eq!(index.rect_of(3), Some(WorldRect::from_coords(30, 0, 32, 2)));
    }

    #[test]
    fn test_visible_ids_matches_hit_testable_subset() {
        let area = WorldRect::from_coords(0, 0, 20, 20);
        let index = FrameIndex::build(small_layout(), area);

        let mut visible: Vec<u32> = index.visible_ids().collect();
        visible.sort_unstable();
        // Node 3 is placed outside the clip area, so it is excluded from visible_ids just
        // like it is from hit/closest.
        assert_eq!(visible, vec![0, 1, 2]);
    }

    #[test]
    fn test_hit_returns_fractional_offset() {
        let area = WorldRect::from_coords(0, 0, 20, 20);
        let index = FrameIndex::build(small_layout(), area);

        let (id, frac) = index.hit(WorldPos::new(1, 1)).expect("should hit node 0");
        assert_eq!(id, 0);
        assert_eq!(frac, (0.5, 0.5));

        let bottom_left = index
            .hit(WorldPos::new(0, 0))
            .expect("should hit the corner");
        assert_eq!(bottom_left.1, (0.0, 0.0));
    }

    #[test]
    fn test_closest_prefers_nearest_visible_node() {
        let area = WorldRect::from_coords(0, 0, 20, 20);
        let index = FrameIndex::build(small_layout(), area);

        // Closer to node 1 (10,0)-(12,2) than node 2 (10,10)-(12,12).
        assert_eq!(index.closest(WorldPos::new(11, 3)), Some(1));
        // Node 3 is offscreen (outside the clip area), so it is never returned by closest()
        // even though it is nominally nearer on the x-axis than nodes 1/2.
        assert_eq!(index.closest(WorldPos::new(100, 0)), Some(1));
    }

    #[test]
    fn test_neighbor_crosses_layers_and_stays_within_layer() {
        let area = WorldRect::from_coords(0, 0, 40, 40);
        let index = FrameIndex::build(small_layout(), area);

        // From node 0 (layer 0), moving right lands in layer 1: two candidates (1 and 2),
        // node 1 is closer vertically.
        assert_eq!(index.neighbor(0, Direction::Right), Some(1));
        // From node 1, moving up (same layer) reaches node 2.
        assert_eq!(index.neighbor(1, Direction::Up), Some(2));
        // From node 1, moving left has no layer-0 candidate other than node 0.
        assert_eq!(index.neighbor(1, Direction::Left), Some(0));
        // From node 0, moving up has no same-layer candidate.
        assert_eq!(index.neighbor(0, Direction::Up), None);
    }

    #[test]
    fn test_empty_index() {
        let index: FrameIndex<u32> = FrameIndex::empty();
        assert!(index.is_empty());
        assert_eq!(index.rect_of(0), None);
        assert_eq!(index.hit(WorldPos::ZERO), None);
        assert_eq!(index.closest(WorldPos::ZERO), None);
    }
}
