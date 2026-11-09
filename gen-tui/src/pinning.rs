//! Which pins an edge needs, given the sides it attaches to and where its endpoints fall in the
//! cycle-removal ordering.
//!
//! Layers run left to right, so an edge from the east of its source to the west of its target is
//! only drawable directly when the source comes first. Every other combination is carried around
//! the nodes by a left pin (west of both ends) or a right pin (east of both ends), or by both
//! with a long span between them. Left and right pins are inserted after the layered layout has
//! placed the data endpoints; up and down bend pins participate in that layout directly.
//!
//! An edge read right to left gets a down pin, and a straight edge that would be drawn on top of it
//! gets an up pin, so layout can place their bends separately.

use std::collections::HashSet;

use LegEnd::{BendPin, HairPin, LeftPin, RightPin, Source, Target};
use petgraph::graph::NodeIndex;

use crate::{
    edge_ports::{EdgePorts, NodeSide, PortedEdge},
    layout::PinSide,
};

/// One end of a leg in a [`PinPlan`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegEnd {
    Source,
    Target,
    LeftPin,
    RightPin,
    /// The up or down pin that bends a plan's one straight edge away from its twin.
    BendPin,
    /// The hairpin node beside the one node of a hairpin.
    HairPin,
}

/// A window-graph edge of a plan, oriented the way the layout reads it (earlier layer first).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Leg {
    pub from: LegEnd,
    pub to: LegEnd,
}

/// The window-graph edges standing in for one domain edge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PinPlan {
    pub legs: Vec<Leg>,
    pub left_pin: bool,
    pub right_pin: bool,
    /// The side of the up or down pin between the endpoints, when the plan has one.
    pub bend_pin: Option<PinSide>,
    /// The side of the node a hairpin turns around on, when the plan is one.
    pub hairpin_side: Option<NodeSide>,
    /// Whether an edge drawn with curves is carried around its nodes by routing nodes inserted
    /// after layout (see [`crate::loopback_pins::insert_horizontal_bend_nodes`]) instead of by left
    /// and right pins that join the layered layout.
    pub horizontal_bend_nodes: bool,
}

impl PinPlan {
    /// Whether the plan routes through a left, right, bend or hairpin node, or through the routing nodes
    /// inserted after layout (`horizontal_bend_nodes`). False only for a plain leg or an arc.
    pub fn has_pins(&self) -> bool {
        self.left_pin
            || self.right_pin
            || self.horizontal_bend_nodes
            || self.bend_pin.is_some()
            || self.hairpin_side.is_some()
    }

    /// The plan of an edge with a single straight leg, routed through a pin on `side` instead.
    fn bent(&self, side: PinSide) -> Option<Self> {
        let [leg] = self.legs[..] else {
            return None;
        };
        Some(Self {
            legs: vec![
                Leg {
                    from: leg.from,
                    to: BendPin,
                },
                Leg {
                    from: BendPin,
                    to: leg.to,
                },
            ],
            left_pin: false,
            right_pin: false,
            bend_pin: Some(side),
            hairpin_side: None,
            horizontal_bend_nodes: false,
        })
    }

    /// The leg joining the left pin to the right pin, when the plan has both.
    #[cfg(test)]
    pub fn span(&self) -> Option<Leg> {
        self.legs
            .iter()
            .copied()
            .find(|leg| leg.from == LegEnd::LeftPin && leg.to == LegEnd::RightPin)
    }
}

/// How an edge constrains the order of its endpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderingConstraint {
    /// The source must come before the target.
    SourceFirst,
    /// The target must come before the source.
    TargetFirst,
}

/// The order an edge asks for, or `None` when it attaches to the same side at both ends.
pub fn ordering_constraint(ports: EdgePorts) -> Option<OrderingConstraint> {
    match (ports.source, ports.target) {
        (NodeSide::East, NodeSide::West) => Some(OrderingConstraint::SourceFirst),
        (NodeSide::West, NodeSide::East) => Some(OrderingConstraint::TargetFirst),
        _ => None,
    }
}

/// The legs and pins that replace an edge. `target_follows_source` is the endpoints' relative
/// order in the cycle-removal ordering; an edge from a node to itself never has its target
/// following its source.
pub fn plan_horizontal_bends(ports: EdgePorts, target_follows_source: bool) -> PinPlan {
    let leg = |from, to| Leg { from, to };
    match (ports.source, ports.target) {
        (NodeSide::East, NodeSide::West) if target_follows_source => PinPlan {
            legs: vec![leg(Source, Target)],
            left_pin: false,
            right_pin: false,
            bend_pin: None,
            hairpin_side: None,
            horizontal_bend_nodes: false,
        },
        (NodeSide::East, NodeSide::West) => PinPlan {
            legs: vec![
                leg(LeftPin, Target),
                leg(Source, RightPin),
                leg(LeftPin, RightPin),
            ],
            left_pin: true,
            right_pin: true,
            bend_pin: None,
            hairpin_side: None,
            horizontal_bend_nodes: false,
        },
        (NodeSide::West, NodeSide::East) if target_follows_source => PinPlan {
            legs: vec![
                leg(LeftPin, Source),
                leg(Target, RightPin),
                leg(LeftPin, RightPin),
            ],
            left_pin: true,
            right_pin: true,
            bend_pin: None,
            hairpin_side: None,
            horizontal_bend_nodes: false,
        },
        (NodeSide::West, NodeSide::East) => PinPlan {
            legs: vec![leg(Target, Source)],
            left_pin: false,
            right_pin: false,
            bend_pin: None,
            hairpin_side: None,
            horizontal_bend_nodes: false,
        },
        (NodeSide::West, NodeSide::West) => PinPlan {
            legs: vec![leg(LeftPin, Source), leg(LeftPin, Target)],
            left_pin: true,
            right_pin: false,
            bend_pin: None,
            hairpin_side: None,
            horizontal_bend_nodes: false,
        },
        (NodeSide::East, NodeSide::East) => PinPlan {
            legs: vec![leg(Source, RightPin), leg(Target, RightPin)],
            left_pin: false,
            right_pin: true,
            bend_pin: None,
            hairpin_side: None,
            horizontal_bend_nodes: false,
        },
    }
}

/// The plan of an edge drawn with curves. The edge joins the layered layout as a single leg that
/// runs the way the layout reads, with no pins, so it takes no room of its own there. When its
/// sides disagree with that order, a west side on the earlier node or an east side on the later
/// one, or when it leaves a node and returns to it on the other side, routing nodes beside its
/// nodes carry it around them once the layers are placed. A self-edge has no leg at all.
pub fn plan_curved_bends(ports: EdgePorts, target_follows_source: bool, is_loop: bool) -> PinPlan {
    let straight = plan_horizontal_bends(ports, target_follows_source);
    if !straight.left_pin && !straight.right_pin {
        return straight;
    }
    let leg = match (is_loop, target_follows_source) {
        (true, _) => None,
        (false, true) => Some(Leg {
            from: Source,
            to: Target,
        }),
        (false, false) => Some(Leg {
            from: Target,
            to: Source,
        }),
    };
    PinPlan {
        legs: leg.into_iter().collect(),
        left_pin: false,
        right_pin: false,
        bend_pin: None,
        hairpin_side: None,
        horizontal_bend_nodes: true,
    }
}

/// The plan of a hairpin: an edge that leaves a side of a node and returns to the same side. One
/// leg joins the node to a hairpin node beside that side, which is drawn as the turnaround. Leaving
/// and returning share the node's one attachment row on that side, so the router sees a single
/// edge and places the pin among the other edges that compete for the port.
pub fn plan_hairpin(side: NodeSide) -> PinPlan {
    let leg = match side {
        NodeSide::East => Leg {
            from: Source,
            to: HairPin,
        },
        NodeSide::West => Leg {
            from: HairPin,
            to: Source,
        },
    };
    PinPlan {
        legs: vec![leg],
        left_pin: false,
        right_pin: false,
        bend_pin: None,
        hairpin_side: Some(side),
        horizontal_bend_nodes: false,
    }
}

/// The node sides that have a hairpin: an edge whose two ends are the same node and the same side.
pub fn hairpin_sides(domain_edges: &[PortedEdge<NodeIndex>]) -> HashSet<(NodeIndex, NodeSide)> {
    domain_edges
        .iter()
        .filter(|edge| edge.source == edge.target && edge.ports.source == edge.ports.target)
        .map(|edge| (edge.source, edge.ports.source))
        .collect()
}

/// The plan of a hairpin that the renderer draws as an arc beside its node: no legs and no pin,
/// because the arc needs no layout space of its own.
pub fn plan_hairpin_arc() -> PinPlan {
    PinPlan {
        legs: Vec::new(),
        left_pin: false,
        right_pin: false,
        bend_pin: None,
        hairpin_side: None,
        horizontal_bend_nodes: false,
    }
}

/// Plan the vertical bends (up and down pins) before Sugiyama, for the straight edges that
/// need a visible direction. An edge read right to left always gets a down pin, so it can bend away from the layout's left-to-right flow and carry an arrow. Its
/// twin, an edge `A -> B` laid out left to right between the same two nodes, would be drawn on top
/// of it, so the twin gets an up pin. The bend direction follows each pin's placed position.
/// `ends[index]` is the (source, target) of `plans[index]`.
pub fn plan_vertical_bends<K: PartialEq>(plans: &mut [PinPlan], ends: &[(K, K)]) {
    let is_straight = |plan: &PinPlan, leg: Leg| !plan.has_pins() && plan.legs == [leg];
    let forward = Leg {
        from: Source,
        to: Target,
    };
    let reversed = Leg {
        from: Target,
        to: Source,
    };
    let reversed_indices: Vec<usize> = (0..plans.len())
        .filter(|&index| is_straight(&plans[index], reversed))
        .collect();
    for &reversed_index in &reversed_indices {
        let (source, target) = &ends[reversed_index];
        let twin = (0..plans.len()).find(|&index| {
            is_straight(&plans[index], forward)
                && ends[index].0 == *target
                && ends[index].1 == *source
        });
        if let Some(twin_index) = twin {
            plans[twin_index] = plans[twin_index]
                .bent(PinSide::Up)
                .expect("should bend a plan with one leg");
        }
        plans[reversed_index] = plans[reversed_index]
            .bent(PinSide::Down)
            .expect("should bend a plan with one leg");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hairpin_is_one_leg_to_a_hairpin_node_on_its_side() {
        let east = plan_hairpin(NodeSide::East);
        assert_eq!(
            east.legs,
            vec![Leg {
                from: Source,
                to: HairPin
            }]
        );
        assert_eq!(east.hairpin_side, Some(NodeSide::East));
        assert!(east.has_pins() && !east.left_pin && !east.right_pin && east.bend_pin.is_none());

        let west = plan_hairpin(NodeSide::West);
        assert_eq!(
            west.legs,
            vec![Leg {
                from: HairPin,
                to: Source
            }]
        );
        assert_eq!(west.hairpin_side, Some(NodeSide::West));
    }

    #[test]
    fn test_curved_plan_is_one_leg_in_layout_order_without_pins() {
        for ports in [WEST_WEST, EAST_EAST] {
            let forward = plan_curved_bends(ports, true, false);
            assert!(forward.horizontal_bend_nodes && !forward.left_pin && !forward.right_pin);
            assert_eq!(
                forward.legs,
                vec![Leg {
                    from: Source,
                    to: Target
                }]
            );
            let backward = plan_curved_bends(ports, false, false);
            assert_eq!(
                backward.legs,
                vec![Leg {
                    from: Target,
                    to: Source
                }]
            );
        }
    }

    #[test]
    fn test_curved_plan_leaves_edges_that_read_in_order_alone() {
        assert_eq!(
            plan_curved_bends(EdgePorts::default(), true, false),
            plan_horizontal_bends(EdgePorts::default(), true)
        );
        assert!(!plan_curved_bends(WEST_EAST, false, false).horizontal_bend_nodes);
    }

    #[test]
    fn test_curved_loop_between_opposite_sides_has_no_leg() {
        for ports in [WEST_EAST, EdgePorts::default()] {
            let plan = plan_curved_bends(ports, ports == WEST_EAST, true);
            assert!(plan.horizontal_bend_nodes && plan.legs.is_empty());
        }
    }

    const WEST_WEST: EdgePorts = EdgePorts::new(NodeSide::West, NodeSide::West);
    const EAST_EAST: EdgePorts = EdgePorts::new(NodeSide::East, NodeSide::East);
    const WEST_EAST: EdgePorts = EdgePorts::new(NodeSide::West, NodeSide::East);

    #[test]
    fn test_default_edge_needs_pins_only_against_the_order() {
        let forward = plan_horizontal_bends(EdgePorts::default(), true);
        assert!(!forward.has_pins());
        assert_eq!(forward.legs.len(), 1);

        let backward = plan_horizontal_bends(EdgePorts::default(), false);
        assert!(backward.left_pin && backward.right_pin);
        assert_eq!(backward.legs.len(), 3);
        assert!(backward.span().is_some());
    }

    #[test]
    fn test_west_to_west_uses_one_left_pin_in_either_order() {
        for target_follows_source in [true, false] {
            let plan = plan_horizontal_bends(WEST_WEST, target_follows_source);
            assert!(plan.left_pin && !plan.right_pin);
            assert_eq!(
                plan.legs,
                vec![
                    Leg {
                        from: LegEnd::LeftPin,
                        to: LegEnd::Source
                    },
                    Leg {
                        from: LegEnd::LeftPin,
                        to: LegEnd::Target
                    },
                ]
            );
        }
    }

    #[test]
    fn test_east_to_east_uses_one_right_pin_in_either_order() {
        for target_follows_source in [true, false] {
            let plan = plan_horizontal_bends(EAST_EAST, target_follows_source);
            assert!(!plan.left_pin && plan.right_pin);
            assert_eq!(plan.legs.len(), 2);
        }
    }

    #[test]
    fn test_west_to_east_loops_around_both_when_the_target_follows() {
        let plan = plan_horizontal_bends(WEST_EAST, true);
        assert!(plan.left_pin && plan.right_pin);
        assert!(plan.legs.contains(&Leg {
            from: LegEnd::LeftPin,
            to: LegEnd::Source
        }));
        assert!(plan.legs.contains(&Leg {
            from: LegEnd::Target,
            to: LegEnd::RightPin
        }));
        assert!(plan.span().is_some());
    }

    #[test]
    fn test_west_to_east_is_a_reversed_straight_edge_when_the_target_comes_first() {
        let plan = plan_horizontal_bends(WEST_EAST, false);
        assert!(!plan.has_pins());
        assert_eq!(
            plan.legs,
            vec![Leg {
                from: LegEnd::Target,
                to: LegEnd::Source
            }]
        );
    }

    #[test]
    fn test_only_mixed_sides_constrain_the_order() {
        assert_eq!(
            ordering_constraint(EdgePorts::default()),
            Some(OrderingConstraint::SourceFirst)
        );
        assert_eq!(
            ordering_constraint(WEST_EAST),
            Some(OrderingConstraint::TargetFirst)
        );
        assert_eq!(ordering_constraint(WEST_WEST), None);
        assert_eq!(ordering_constraint(EAST_EAST), None);
    }

    #[test]
    fn test_reversed_edge_bends_down_without_a_twin() {
        let mut plans = vec![plan_horizontal_bends(WEST_EAST, false)];
        plan_vertical_bends(&mut plans, &[(1, 0)]);
        assert_eq!(plans[0].bend_pin, Some(PinSide::Down));
    }

    #[test]
    fn test_opposite_straight_edges_bend_apart() {
        let mut plans = vec![
            plan_horizontal_bends(EdgePorts::default(), true),
            plan_horizontal_bends(WEST_EAST, false),
            plan_horizontal_bends(EdgePorts::default(), true),
        ];
        let ends = [(0, 1), (1, 0), (1, 2)];
        plan_vertical_bends(&mut plans, &ends);

        assert_eq!(plans[0].bend_pin, Some(PinSide::Up));
        assert_eq!(plans[1].bend_pin, Some(PinSide::Down));
        assert_eq!(plans[2].bend_pin, None);
        assert_eq!(plans[1].legs.len(), 2);
    }
}
