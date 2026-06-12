//! Orthogonal drawing compaction via constraint-graph longest-path passes.
//!
//! The solver is axis-agnostic: the caller builds spacing constraints from the
//! drawing geometry and passes them in. The same algorithm handles x and y.
//! See [`crate::distribute_nodes::compact_layout`] for how constraints are
//! derived from a layout graph.

use std::collections::{HashMap, HashSet};

use petgraph::{Direction, algo::toposort, graph::DiGraph, visit::EdgeRef};

pub type ItemId = usize;

/// A lower-bound spacing constraint: `position[to] >= position[from] + gap`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Constraint {
    pub from: ItemId,
    pub to: ItemId,
    pub gap: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompactionError {
    NegativeGap { constraint: Constraint },
    UnknownItem { item: ItemId },
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

/// Computes the smallest valid integer position for every item.
///
/// A constraint `from → to, gap` encodes the lower-bound inequality
///     position[to] >= position[from] + gap.
///
/// **Why topological order?**  Each item's minimum position is the length of the
/// longest path from any source to that item in the constraint graph.  Processing
/// nodes in topological order guarantees that every predecessor has already been
/// relaxed before we relax its outgoing edges, so a single forward pass suffices.
///
/// **Why edge relaxation?**  Standard longest-path on a DAG: for each directed
/// edge `u → v` with weight `w`, update `dist[v] = max(dist[v], dist[u] + w)`.
/// The final `dist[v]` is the length of the longest path reaching `v`, which is
/// the minimum coordinate for `v` that satisfies all constraints.
///
/// # Errors
///
/// Returns an error when a constraint has a negative gap, references an item
/// not present in `item_ids`, or when the constraint graph contains a cycle.
pub fn compact_low<I>(
    item_ids: I,
    constraints: &[Constraint],
) -> Result<HashMap<ItemId, i64>, CompactionError>
where
    I: IntoIterator<Item = ItemId>,
{
    // Deduplicate while preserving encounter order.
    let items: Vec<ItemId> = {
        let mut seen = HashSet::new();
        item_ids.into_iter().filter(|id| seen.insert(*id)).collect()
    };

    let mut graph: DiGraph<ItemId, i64> = DiGraph::new();
    let mut id_to_node = HashMap::new();

    // Add every item as a node before adding edges so unknown-item checks work.
    for &id in &items {
        let node = graph.add_node(id);
        id_to_node.insert(id, node);
    }

    for &c in constraints {
        if c.gap < 0 {
            return Err(CompactionError::NegativeGap { constraint: c });
        }
        let &from_node = id_to_node
            .get(&c.from)
            .ok_or(CompactionError::UnknownItem { item: c.from })?;
        let &to_node = id_to_node
            .get(&c.to)
            .ok_or(CompactionError::UnknownItem { item: c.to })?;
        graph.add_edge(from_node, to_node, c.gap);
    }

    let topo = toposort(&graph, None).map_err(|_| CompactionError::CycleDetected)?;

    // Initialize all distances to 0; disconnected items stay at 0.
    let mut dist = vec![0i64; graph.node_count()];

    for node in topo {
        let d = dist[node.index()];
        for edge in graph.edges_directed(node, Direction::Outgoing) {
            let t = edge.target().index();
            let candidate = d + *edge.weight();
            if candidate > dist[t] {
                dist[t] = candidate;
            }
        }
    }

    let mut result: HashMap<ItemId, i64> = id_to_node
        .iter()
        .map(|(&id, &node)| (id, dist[node.index()]))
        .collect();

    // Normalize so the minimum coordinate is 0.
    if let Some(&min_val) = result.values().min()
        && min_val != 0
    {
        result.values_mut().for_each(|v| *v -= min_val);
    }

    Ok(result)
}

/// Runs the low-side solver, then a mirrored high-side pass, and returns all
/// three coordinate maps (low, high, centered) together with the span.
///
/// **High-side pass**: reversing every constraint `from → to, gap` to
/// `to → from, gap` and solving gives each item's minimum distance from the
/// far end of the axis.  Converting back with `high[id] = span - reversed[id]`
/// yields the rightmost (or bottommost) valid coordinate.
///
/// **Why midpoint satisfies constraints**: for any constraint `u → v, gap`,
/// the low-side pass guarantees `low[v] >= low[u] + gap` and the high-side
/// pass guarantees `high[v] >= high[u] + gap`.  Therefore
///   centered[v] - centered[u]
///     = floor((low[v]+high[v])/2) - floor((low[u]+high[u])/2)
///     >= floor((low[u]+gap + high[u]+gap) / 2) - floor((low[u]+high[u])/2)
///     = gap.
/// Integer floor division preserves this because the 2*gap term is even.
///
/// # Errors
///
/// Propagates any [`CompactionError`] from the underlying low-side passes.
pub fn compact_axis_with_centering<I>(
    item_ids: I,
    constraints: &[Constraint],
) -> Result<AxisPlacement, CompactionError>
where
    I: IntoIterator<Item = ItemId> + Clone,
{
    let low = compact_low(item_ids.clone(), constraints)?;
    let span = low.values().copied().max().unwrap_or(0);

    let reversed = reverse_constraints(constraints);
    let rev_low = compact_low(item_ids, &reversed)?;

    let high: HashMap<ItemId, i64> = rev_low
        .iter()
        .map(|(&id, &rev_pos)| (id, span - rev_pos))
        .collect();

    let centered: HashMap<ItemId, i64> = low
        .iter()
        .map(|(&id, &lo)| (id, (lo + high[&id]).div_euclid(2)))
        .collect();

    debug_assert!(
        constraints_are_satisfied(&centered, constraints),
        "centered placement violates a constraint — this is a bug in the solver"
    );

    Ok(AxisPlacement {
        low,
        high,
        centered,
        span,
    })
}

/// Reverses every constraint: `from → to, gap` becomes `to → from, gap`.
/// Used to build the high-side pass input.
pub fn reverse_constraints(constraints: &[Constraint]) -> Vec<Constraint> {
    constraints
        .iter()
        .map(|c| Constraint {
            from: c.to,
            to: c.from,
            gap: c.gap,
        })
        .collect()
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
    fn test_negative_gap() {
        let c = Constraint {
            from: 0,
            to: 1,
            gap: -1,
        };
        assert!(matches!(
            compact_low([0, 1], &[c]),
            Err(CompactionError::NegativeGap { .. })
        ));
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
}
