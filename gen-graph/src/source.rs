use core::convert::Infallible;

use petgraph::{Direction, visit::GraphBase};

/// Supplies missing adjacency to a retained, partially loaded graph.
///
/// Completion state belongs to the supplied graph. Keep graph and source together, including
/// when cloning them, and keep the backing store unchanged for the lifetime of that state.
pub trait GraphSource<G: GraphBase> {
    /// The backing-store failure reported while loading adjacency.
    type Error: core::fmt::Display;

    /// Whether this side may still have unloaded edges. A known dead end is complete.
    fn is_frontier(&self, node: G::NodeId, direction: Direction) -> bool;

    /// Load all immediate incoming and outgoing edges of the requested nodes in place.
    ///
    /// Endpoint nodes are inserted with stable identities and may remain frontier. Repeated
    /// requests must be safe. Successful sides may remain loaded if another side fails;
    /// callers must stop traversal on failure and may retry the request.
    fn expand_frontier(&mut self, graph: &mut G, frontier: &[G::NodeId])
    -> Result<(), Self::Error>;
}

/// A source for a graph whose adjacency is already fully loaded.
#[derive(Clone, Copy, Debug, Default)]
pub struct EagerSource;

impl<G: GraphBase> GraphSource<G> for EagerSource {
    type Error = Infallible;

    fn is_frontier(&self, _node: G::NodeId, _direction: Direction) -> bool {
        false
    }

    fn expand_frontier(
        &mut self,
        _graph: &mut G,
        _frontier: &[G::NodeId],
    ) -> Result<(), Self::Error> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use petgraph::{Direction, graphmap::DiGraphMap};

    use super::{EagerSource, GraphSource};

    #[test]
    fn test_eager_source_preserves_a_complete_graph() {
        let mut graph = DiGraphMap::<usize, ()>::new();
        graph.add_edge(1, 2, ());
        let mut source = EagerSource;
        for node in graph.nodes() {
            for direction in [Direction::Incoming, Direction::Outgoing] {
                assert!(!GraphSource::<DiGraphMap<usize, ()>>::is_frontier(
                    &source, node, direction
                ));
            }
        }
        source.expand_frontier(&mut graph, &[1, 2]).unwrap();
        assert_eq!(graph.nodes().collect::<Vec<_>>(), vec![1, 2]);
        assert_eq!(graph.all_edges().collect::<Vec<_>>(), vec![(1, 2, &())]);
    }
}
