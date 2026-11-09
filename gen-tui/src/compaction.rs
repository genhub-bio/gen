//! Orthogonal drawing compaction via constraint-graph longest-path passes.
//!
//! This is only the constraint solver; [`crate::layout_compaction`] is the size-aware
//! compaction of positioned layouts that builds the constraints it solves.
//!
//! The solver is axis-agnostic: the caller builds spacing constraints from the
//! drawing geometry and passes them in. The same algorithm handles x and y.
//! See [`crate::layout_compaction::compact_layout_with_vertical_placement`] for how constraints
//! are derived from a layout graph.

use std::collections::{HashMap, HashSet};

use petgraph::{
    Direction,
    algo::tarjan_scc,
    graph::{DiGraph, NodeIndex},
    visit::EdgeRef,
};

pub type ItemId = usize;

/// `position + gap`, on the assumption that a real layout never needs a coordinate wide
/// enough to overflow `i64`.
fn checked_gap(position: i64, gap: i64) -> i64 {
    position
        .checked_add(gap)
        .expect("should fit compacted layout coordinate")
}

/// A lower-bound spacing constraint: `position[to] >= position[from] + gap`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Constraint {
    pub from: ItemId,
    pub to: ItemId,
    pub gap: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompactionError {
    UnknownItem {
        item: ItemId,
    },
    /// The constraints contain a cycle whose positive total gap makes the system infeasible.
    CycleDetected,
}

/// Placement result for a single axis, holding all three coordinate maps and the span.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AxisPlacement {
    /// Tightest placement toward the low side (left / top).
    pub low: HashMap<ItemId, i64>,
    /// Tightest placement toward the high side (right / bottom).
    pub high: HashMap<ItemId, i64>,
    /// Centered placement: `floor((low + high) / 2)` for each item.
    pub centered: HashMap<ItemId, i64>,
    /// Maximum coordinate from the low-side pass; defines the axis span.
    pub span: i64,
}

/// Item ids in encounter order, each mapped to the dense position the solver works on. The
/// solver visits every constraint several times, so it indexes vectors instead of hashing ids.
struct ItemIndex {
    items: Vec<ItemId>,
    /// `None` when the items are exactly `0..items.len()` in order, which is how the layout
    /// numbers its segments, so an id is its own position.
    positions: Option<HashMap<ItemId, usize>>,
}

impl ItemIndex {
    fn new(item_ids: impl IntoIterator<Item = ItemId>) -> Self {
        let mut items: Vec<ItemId> = item_ids.into_iter().collect();
        if items
            .iter()
            .enumerate()
            .all(|(position, &id)| position == id)
        {
            return Self {
                items,
                positions: None,
            };
        }
        // Deduplicate while preserving encounter order.
        let mut seen = HashSet::new();
        items.retain(|id| seen.insert(*id));
        let positions = items
            .iter()
            .enumerate()
            .map(|(position, &id)| (id, position))
            .collect();
        Self {
            items,
            positions: Some(positions),
        }
    }

    fn len(&self) -> usize {
        self.items.len()
    }

    fn position(&self, id: ItemId) -> Result<usize, CompactionError> {
        let position = match &self.positions {
            None => (id < self.items.len()).then_some(id),
            Some(positions) => positions.get(&id).copied(),
        };
        position.ok_or(CompactionError::UnknownItem { item: id })
    }

    fn dense_edges(
        &self,
        constraints: &[Constraint],
    ) -> Result<Vec<(usize, usize, i64)>, CompactionError> {
        constraints
            .iter()
            .map(|constraint| {
                Ok((
                    self.position(constraint.from)?,
                    self.position(constraint.to)?,
                    constraint.gap,
                ))
            })
            .collect()
    }

    fn label(&self, distances: &[i64]) -> HashMap<ItemId, i64> {
        self.items
            .iter()
            .copied()
            .zip(distances.iter().copied())
            .collect()
    }
}

/// Shifts distances so the minimum coordinate is 0.
fn normalized(mut distances: Vec<i64>) -> Vec<i64> {
    if let Some(&minimum) = distances.iter().min()
        && minimum != 0
    {
        distances
            .iter_mut()
            .for_each(|distance| *distance -= minimum);
    }
    distances
}

/// Least distance of each of `count` items from the start of the axis that satisfies every
/// `(from, to, gap)` edge.
fn longest_paths(count: usize, edges: &[(usize, usize, i64)]) -> Result<Vec<i64>, CompactionError> {
    let mut graph: DiGraph<(), i64> = DiGraph::with_capacity(count, edges.len());
    let nodes: Vec<NodeIndex> = (0..count).map(|_| graph.add_node(())).collect();
    for &(from, to, gap) in edges {
        graph.add_edge(nodes[from], nodes[to], gap);
    }

    // Initialize all distances to 0; disconnected items stay at 0.
    let mut dist = vec![0i64; count];

    // Strongly connected components come out sinks first, so walking them in reverse visits every
    // predecessor before its successors. Most items form a component of their own and need a
    // single relaxation of their outgoing edges; only the small cycles that paired upper and
    // lower bounds create are iterated, and only over their own constraints.
    let components = tarjan_scc(&graph);
    let mut component_of = vec![0usize; count];
    for (component_index, component) in components.iter().enumerate() {
        for node in component {
            component_of[node.index()] = component_index;
        }
    }
    for (component_index, component) in components.iter().enumerate().rev() {
        if component.len() > 1 {
            relax_cycle(&graph, component, &component_of, &mut dist)?;
        }
        for &node in component {
            for edge in graph.edges_directed(node, Direction::Outgoing) {
                let target = edge.target().index();
                let candidate = checked_gap(dist[node.index()], *edge.weight());
                if component_of[target] == component_index {
                    if component.len() == 1 && candidate > dist[target] {
                        return Err(CompactionError::CycleDetected);
                    }
                    continue;
                }
                if candidate > dist[target] {
                    dist[target] = candidate;
                }
            }
        }
    }
    Ok(dist)
}

/// Raises the distances of one strongly connected component to the least values that satisfy
/// every constraint between its members, by repeated relaxation. A component of `n` items needs
/// at most `n - 1` passes, so a constraint still violated after that closes a cycle whose gaps
/// add up to more than zero.
fn relax_cycle(
    graph: &DiGraph<(), i64>,
    component: &[NodeIndex],
    component_of: &[usize],
    dist: &mut [i64],
) -> Result<(), CompactionError> {
    let component_index = component_of[component[0].index()];
    let constraints: Vec<(usize, usize, i64)> = component
        .iter()
        .flat_map(|&node| {
            graph
                .edges_directed(node, Direction::Outgoing)
                .filter(|edge| component_of[edge.target().index()] == component_index)
                .map(|edge| (node.index(), edge.target().index(), *edge.weight()))
                .collect::<Vec<_>>()
        })
        .collect();
    for _ in 1..component.len() {
        let mut changed = false;
        for &(from, to, gap) in &constraints {
            let candidate = checked_gap(dist[from], gap);
            if candidate > dist[to] {
                dist[to] = candidate;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    if constraints
        .iter()
        .any(|&(from, to, gap)| checked_gap(dist[from], gap) > dist[to])
    {
        return Err(CompactionError::CycleDetected);
    }
    Ok(())
}

/// Runs the low-side solver, then a mirrored high-side pass, and returns all
/// three coordinate maps (low, high, centered) together with the span.
///
/// Reversing every constraint `from → to, gap` to
/// `to → from, gap` and solving gives each item's minimum distance from the
/// far end of the axis. Converting back with `high[id] = span - reversed[id]`
/// yields the rightmost (or bottommost) valid coordinate.
///
/// Both placements satisfy every lower bound, so their coordinate-wise midpoint does too.
/// Integer division preserves this because twice each gap is even.
///
/// # Errors
///
/// Propagates any [`CompactionError`] from the underlying low-side passes.
pub fn compact_axis_with_centering<I>(
    item_ids: I,
    constraints: &[Constraint],
) -> Result<AxisPlacement, CompactionError>
where
    I: IntoIterator<Item = ItemId>,
{
    let index = ItemIndex::new(item_ids);
    let edges = index.dense_edges(constraints)?;
    let low = normalized(longest_paths(index.len(), &edges)?);
    let span = low.iter().copied().max().unwrap_or(0);

    let reversed: Vec<(usize, usize, i64)> = edges
        .iter()
        .map(|&(from, to, gap)| (to, from, gap))
        .collect();
    let rev_low = normalized(longest_paths(index.len(), &reversed)?);

    let high: Vec<i64> = rev_low.iter().map(|&rev_pos| span - rev_pos).collect();
    let centered: Vec<i64> = low
        .iter()
        .zip(&high)
        .map(|(&lo, &hi)| (lo + hi).div_euclid(2))
        .collect();
    let placement = AxisPlacement {
        low: index.label(&low),
        high: index.label(&high),
        centered: index.label(&centered),
        span,
    };

    debug_assert!(
        constraints_are_satisfied(&placement.centered, constraints),
        "centered placement violates a constraint — this is a bug in the solver"
    );

    Ok(placement)
}

/// Returns `true` if every constraint `position[to] >= position[from] + gap`
/// is satisfied by `positions`.
pub fn constraints_are_satisfied(
    positions: &HashMap<ItemId, i64>,
    constraints: &[Constraint],
) -> bool {
    constraints
        .iter()
        .all(|c| match (positions.get(&c.from), positions.get(&c.to)) {
            (Some(&f), Some(&t)) => t >= f + c.gap,
            _ => false,
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Computes the smallest valid integer position for every item.
    ///
    /// A constraint `from → to, gap` requires `position[to] >= position[from] + gap`.
    ///
    /// For acyclic separation constraints, each item's
    /// minimum position is the length of the longest path from any source to that item. Processing
    /// nodes in topological order guarantees that every predecessor has already been relaxed before
    /// its outgoing edges are visited, so a single forward pass suffices.
    ///
    /// Signed reverse constraints encode upper bounds and introduce feasible cycles. Those systems
    /// use repeated longest-path relaxation instead; only a positive-total cycle is infeasible.
    ///
    /// # Errors
    ///
    /// Returns an error when a constraint references an unknown item or when the constraint system
    /// contains an infeasible positive-total cycle.
    fn compact_low<I>(
        item_ids: I,
        constraints: &[Constraint],
    ) -> Result<HashMap<ItemId, i64>, CompactionError>
    where
        I: IntoIterator<Item = ItemId>,
    {
        let index = ItemIndex::new(item_ids);
        let edges = index.dense_edges(constraints)?;
        let distances = normalized(longest_paths(index.len(), &edges)?);
        Ok(index.label(&distances))
    }

    /// Reverses every constraint: `from → to, gap` becomes `to → from, gap`.
    /// Used to build the high-side pass input.
    fn reverse_constraints(constraints: &[Constraint]) -> Vec<Constraint> {
        constraints
            .iter()
            .map(|c| Constraint {
                from: c.to,
                to: c.from,
                gap: c.gap,
            })
            .collect()
    }

    /// Chain: A → B → C.  Each node's position equals the sum of gaps along
    /// the only path reaching it.
    #[test]
    fn test_chain() {
        let cs = vec![
            Constraint {
                from: 0,
                to: 1,
                gap: 1,
            },
            Constraint {
                from: 1,
                to: 2,
                gap: 2,
            },
        ];
        let pos = compact_low(0..3, &cs).unwrap();
        assert_eq!(pos[&0], 0);
        assert_eq!(pos[&1], 1);
        assert_eq!(pos[&2], 3);
    }

    /// Diamond: two paths to D; the longer one (A→B→D = 11) wins over the
    /// shorter (A→C→D = 6).
    #[test]
    fn test_diamond() {
        let cs = vec![
            Constraint {
                from: 0,
                to: 1,
                gap: 1,
            },
            Constraint {
                from: 0,
                to: 2,
                gap: 5,
            },
            Constraint {
                from: 1,
                to: 3,
                gap: 10,
            },
            Constraint {
                from: 2,
                to: 3,
                gap: 1,
            },
        ];
        let pos = compact_low(0..4, &cs).unwrap();
        assert_eq!(pos[&0], 0);
        assert_eq!(pos[&1], 1);
        assert_eq!(pos[&2], 5);
        assert_eq!(pos[&3], 11);
    }

    /// Item C has no constraints so it stays at the initial value of 0.
    #[test]
    fn test_disconnected_item() {
        let cs = vec![Constraint {
            from: 0,
            to: 1,
            gap: 3,
        }];
        let pos = compact_low(0..3, &cs).unwrap();
        assert_eq!(pos[&0], 0);
        assert_eq!(pos[&1], 3);
        assert_eq!(pos[&2], 0);
    }

    #[test]
    fn test_cycle() {
        let cs = vec![
            Constraint {
                from: 0,
                to: 1,
                gap: 1,
            },
            Constraint {
                from: 1,
                to: 0,
                gap: 1,
            },
        ];
        assert!(matches!(
            compact_low([0, 1], &cs),
            Err(CompactionError::CycleDetected)
        ));
    }

    #[test]
    fn signed_reverse_constraint_caps_distance() {
        let constraints = [
            Constraint {
                from: 0,
                to: 1,
                gap: 2,
            },
            Constraint {
                from: 1,
                to: 0,
                gap: -4,
            },
            Constraint {
                from: 2,
                to: 1,
                gap: 100,
            },
        ];

        let placement = compact_axis_with_centering(0..3, &constraints).unwrap();

        assert_eq!(placement.centered[&1] - placement.centered[&0], 3);
        assert!(constraints_are_satisfied(&placement.centered, &constraints));
    }

    #[test]
    fn test_unknown_item() {
        let cs = vec![Constraint {
            from: 0,
            to: 1,
            gap: 1,
        }];
        assert!(matches!(
            compact_low([0usize], &cs),
            Err(CompactionError::UnknownItem { item: 1 })
        ));
    }

    /// Two constraints between the same pair: only the larger gap is binding.
    #[test]
    fn test_multiple_constraints_same_pair() {
        let cs = vec![
            Constraint {
                from: 0,
                to: 1,
                gap: 1,
            },
            Constraint {
                from: 0,
                to: 1,
                gap: 4,
            },
        ];
        let pos = compact_low([0, 1], &cs).unwrap();
        assert_eq!(pos[&0], 0);
        assert_eq!(pos[&1], 4);
    }

    /// B has slack [1, 9] and must land at 5.
    ///
    /// Graph:  A──(10)──D
    ///          └─(1)─B─(1)─┘
    ///
    /// Low:  A=0, B=1, D=10  (B forced right by A→B=1)
    /// High: A=0, B=9, D=10  (B forced left from D by D→B=1 reversed)
    #[test]
    fn test_centered_slack() {
        // A=0, B=1, D=2
        let items = vec![0usize, 1, 2];
        let cs = vec![
            Constraint {
                from: 0,
                to: 2,
                gap: 10,
            },
            Constraint {
                from: 0,
                to: 1,
                gap: 1,
            },
            Constraint {
                from: 1,
                to: 2,
                gap: 1,
            },
        ];
        let p = compact_axis_with_centering(items, &cs).unwrap();

        assert_eq!(p.low[&0], 0);
        assert_eq!(p.low[&1], 1);
        assert_eq!(p.low[&2], 10);

        assert_eq!(p.high[&0], 0);
        assert_eq!(p.high[&1], 9);
        assert_eq!(p.high[&2], 10);

        assert_eq!(p.centered[&0], 0);
        assert_eq!(p.centered[&1], 5);
        assert_eq!(p.centered[&2], 10);

        assert!(constraints_are_satisfied(&p.centered, &cs));
    }

    #[test]
    fn test_reverse_constraints() {
        let cs = vec![Constraint {
            from: 1,
            to: 2,
            gap: 5,
        }];
        let rev = reverse_constraints(&cs);
        assert_eq!(
            rev[0],
            Constraint {
                from: 2,
                to: 1,
                gap: 5,
            }
        );
    }

    #[test]
    fn test_constraints_are_satisfied() {
        let cs = vec![Constraint {
            from: 0,
            to: 1,
            gap: 3,
        }];
        let mut pos = HashMap::new();
        pos.insert(0, 0);
        pos.insert(1, 3);
        assert!(constraints_are_satisfied(&pos, &cs));
        *pos.get_mut(&1).unwrap() = 2;
        assert!(!constraints_are_satisfied(&pos, &cs));
    }

    #[test]
    fn test_a_feasible_cycle_in_a_chain_pushes_its_successors() {
        // Items 1 and 2 are tethered 3 apart in both directions; item 0 precedes them and
        // item 3 follows item 2.
        let cs = vec![
            Constraint {
                from: 0,
                to: 1,
                gap: 2,
            },
            Constraint {
                from: 1,
                to: 2,
                gap: 3,
            },
            Constraint {
                from: 2,
                to: 1,
                gap: -3,
            },
            Constraint {
                from: 2,
                to: 3,
                gap: 1,
            },
        ];
        let pos = compact_low(0..4, &cs).expect("should solve a feasible cycle");
        assert_eq!((pos[&0], pos[&1], pos[&2], pos[&3]), (0, 2, 5, 6));
    }

    #[test]
    fn test_a_positive_self_loop_is_infeasible() {
        let cs = vec![Constraint {
            from: 0,
            to: 0,
            gap: 1,
        }];
        assert_eq!(compact_low(0..1, &cs), Err(CompactionError::CycleDetected));
    }

    #[test]
    fn test_a_zero_gap_self_loop_is_ignored() {
        let cs = vec![Constraint {
            from: 0,
            to: 0,
            gap: 0,
        }];
        let pos = compact_low(0..1, &cs).expect("should accept a zero-gap self loop");
        assert_eq!(pos[&0], 0);
    }

    #[test]
    fn test_items_with_arbitrary_ids_and_duplicates_keep_encounter_order() {
        let cs = vec![
            Constraint {
                from: 30,
                to: 10,
                gap: 4,
            },
            Constraint {
                from: 10,
                to: 20,
                gap: 2,
            },
        ];
        let pos = compact_low([30, 10, 20, 10], &cs).expect("should solve sparse ids");
        assert_eq!((pos[&30], pos[&10], pos[&20]), (0, 4, 6));
        assert_eq!(pos.len(), 3);
    }

    #[test]
    fn test_centering_agrees_with_the_low_and_high_passes_for_sparse_ids() {
        let cs = vec![Constraint {
            from: 7,
            to: 9,
            gap: 3,
        }];
        let placement =
            compact_axis_with_centering([7, 9, 5], &cs).expect("should solve sparse ids");
        assert_eq!(placement.low[&9], 3);
        assert_eq!(placement.high[&7], 0);
        assert_eq!(placement.span, 3);
        assert_eq!(placement.centered[&5], 1);
    }
}
