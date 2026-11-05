//! Which side of its two nodes an edge attaches to.
//!
//! Layouts read left to right, so an edge normally leaves the east side of its source and
//! arrives at the west side of its target. An edge that attaches elsewhere, such as the two
//! ends of a head-to-head repeat, is carried around its nodes by pins; see [`crate::pinning`].

use serde::{Deserialize, Serialize};

/// A side of a node: the end the layout reads first (west) or last (east).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum NodeSide {
    West,
    East,
}

/// The sides an edge attaches to at its source and at its target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct EdgePorts {
    pub source: NodeSide,
    pub target: NodeSide,
}

impl EdgePorts {
    pub const fn new(source: NodeSide, target: NodeSide) -> Self {
        Self { source, target }
    }
}

impl Default for EdgePorts {
    /// East to west, the way an edge reads in a plain left-to-right layout.
    fn default() -> Self {
        Self::new(NodeSide::East, NodeSide::West)
    }
}

/// A domain edge named in full: both endpoints and the sides it attaches to. Two edges between
/// the same pair of nodes that differ in their ports (for a sequence graph, in their strands) are
/// different edges, so this is what highlights are keyed on rather than a node pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EdgeKey<N> {
    pub source: N,
    pub target: N,
    pub ports: EdgePorts,
}

impl<N> EdgeKey<N> {
    pub const fn new(source: N, target: N, ports: EdgePorts) -> Self {
        Self {
            source,
            target,
            ports,
        }
    }
}

/// Which edge a highlight names.
///
/// `Exact` lights only the edge with those ports. `Between` is for callers that know only the
/// two nodes: it lights the regular west-to-east edge (default ports) between them, or, when
/// there is no such edge but exactly one edge joins the pair, that one. It never lights every
/// parallel edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeSelector<N> {
    Exact(EdgeKey<N>),
    Between { source: N, target: N },
}

impl<N: Copy> EdgeSelector<N> {
    pub const fn source(&self) -> N {
        match self {
            Self::Exact(key) => key.source,
            Self::Between { source, .. } => *source,
        }
    }

    pub const fn target(&self) -> N {
        match self {
            Self::Exact(key) => key.target,
            Self::Between { target, .. } => *target,
        }
    }
}

impl<N> From<EdgeKey<N>> for EdgeSelector<N> {
    fn from(key: EdgeKey<N>) -> Self {
        Self::Exact(key)
    }
}

impl<N> From<(N, N)> for EdgeSelector<N> {
    fn from((source, target): (N, N)) -> Self {
        Self::Between { source, target }
    }
}
