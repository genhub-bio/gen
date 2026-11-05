use core::ops::{Deref, DerefMut};
use std::hash::Hash;

use petgraph::{
    Directed, Direction,
    graphmap::{DiGraphMap, NeighborsDirected},
    visit::{
        Data, EdgeRef, GraphBase, IntoEdgeReferences, IntoEdges, IntoNeighbors,
        IntoNeighborsDirected, IntoNodeIdentifiers, NodeCount, NodeIndexable, Visitable,
    },
};

/// Identifies one parallel edge by its endpoints and its own payload, so the id stays valid when
/// a graph is rebuilt and never depends on the payload's position in its `Vec`.
pub type PackedEdgeId<N, W> = (N, N, W);

/// A `DiGraphMap` whose `Vec` edge payloads are exposed to petgraph as one edge per element.
///
/// A graph map merges parallel edges into a single node-pair entry. Consumers that need to tell
/// parallel edges apart (e.g. a forward-strand and a reverse-strand edge between the same two
/// nodes, which attach to different ports when drawn) traverse this wrapper instead. Node traits
/// pass through unchanged, so `NodeId` is still the graph map's node type, and neighbours stay
/// deduplicated per node pair.
#[derive(Clone, Debug)]
pub struct PackedGraphMap<N, W>
where
    N: Copy + Eq + Hash + Ord,
{
    inner: DiGraphMap<N, Vec<W>>,
}

impl<N, W> PackedGraphMap<N, W>
where
    N: Copy + Eq + Hash + Ord,
    W: Copy + PartialEq,
{
    /// The payload of the edge `id` names, or `None` if the graph no longer holds it.
    pub fn edge_payload(&self, id: &PackedEdgeId<N, W>) -> Option<&W> {
        self.inner
            .edge_weight(id.0, id.1)?
            .iter()
            .find(|payload| **payload == id.2)
    }
}

impl<N, W> From<DiGraphMap<N, Vec<W>>> for PackedGraphMap<N, W>
where
    N: Copy + Eq + Hash + Ord,
{
    fn from(inner: DiGraphMap<N, Vec<W>>) -> Self {
        Self { inner }
    }
}

impl<N, W> Default for PackedGraphMap<N, W>
where
    N: Copy + Eq + Hash + Ord,
{
    fn default() -> Self {
        Self::from(DiGraphMap::new())
    }
}

impl<N, W> Deref for PackedGraphMap<N, W>
where
    N: Copy + Eq + Hash + Ord,
{
    type Target = DiGraphMap<N, Vec<W>>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<N, W> DerefMut for PackedGraphMap<N, W>
where
    N: Copy + Eq + Hash + Ord,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<N, W> GraphBase for PackedGraphMap<N, W>
where
    N: Copy + Eq + Hash + Ord,
    W: Copy + PartialEq,
{
    type EdgeId = PackedEdgeId<N, W>;
    type NodeId = N;
}

impl<N, W> Data for PackedGraphMap<N, W>
where
    N: Copy + Eq + Hash + Ord,
    W: Copy + PartialEq,
{
    type EdgeWeight = W;
    type NodeWeight = N;
}

impl<N, W> Visitable for PackedGraphMap<N, W>
where
    N: Copy + Eq + Hash + Ord,
    W: Copy + PartialEq,
{
    type Map = <DiGraphMap<N, Vec<W>> as Visitable>::Map;

    fn visit_map(&self) -> Self::Map {
        self.inner.visit_map()
    }

    fn reset_map(&self, map: &mut Self::Map) {
        self.inner.reset_map(map);
    }
}

impl<N, W> NodeCount for PackedGraphMap<N, W>
where
    N: Copy + Eq + Hash + Ord,
    W: Copy + PartialEq,
{
    fn node_count(&self) -> usize {
        self.inner.node_count()
    }
}

impl<N, W> NodeIndexable for PackedGraphMap<N, W>
where
    N: Copy + Eq + Hash + Ord,
    W: Copy + PartialEq,
{
    fn node_bound(&self) -> usize {
        self.inner.node_bound()
    }

    fn to_index(&self, node: N) -> usize {
        NodeIndexable::to_index(&self.inner, node)
    }

    fn from_index(&self, index: usize) -> N {
        NodeIndexable::from_index(&self.inner, index)
    }
}

/// One parallel edge of a [`PackedGraphMap`].
#[derive(Debug)]
pub struct PackedEdgeRef<'a, N, W> {
    id: PackedEdgeId<N, W>,
    weight: &'a W,
}

impl<N: Copy, W: Copy> Clone for PackedEdgeRef<'_, N, W> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<N: Copy, W: Copy> Copy for PackedEdgeRef<'_, N, W> {}

impl<N: Copy + PartialEq, W: Copy + PartialEq> EdgeRef for PackedEdgeRef<'_, N, W> {
    type EdgeId = PackedEdgeId<N, W>;
    type NodeId = N;
    type Weight = W;

    fn source(&self) -> N {
        self.id.0
    }

    fn target(&self) -> N {
        self.id.1
    }

    fn weight(&self) -> &W {
        self.weight
    }

    fn id(&self) -> Self::EdgeId {
        self.id
    }
}

impl<'a, N, W> IntoNodeIdentifiers for &'a PackedGraphMap<N, W>
where
    N: Copy + Eq + Hash + Ord,
    W: Copy + PartialEq,
{
    type NodeIdentifiers = <&'a DiGraphMap<N, Vec<W>> as IntoNodeIdentifiers>::NodeIdentifiers;

    fn node_identifiers(self) -> Self::NodeIdentifiers {
        (&self.inner).node_identifiers()
    }
}

impl<'a, N, W> IntoNeighbors for &'a PackedGraphMap<N, W>
where
    N: Copy + Eq + Hash + Ord,
    W: Copy + PartialEq,
{
    type Neighbors = <&'a DiGraphMap<N, Vec<W>> as IntoNeighbors>::Neighbors;

    fn neighbors(self, node: N) -> Self::Neighbors {
        self.inner.neighbors(node)
    }
}

impl<'a, N, W> IntoNeighborsDirected for &'a PackedGraphMap<N, W>
where
    N: Copy + Eq + Hash + Ord,
    W: Copy + PartialEq,
{
    type NeighborsDirected = NeighborsDirected<'a, N, Directed>;

    fn neighbors_directed(self, node: N, direction: Direction) -> Self::NeighborsDirected {
        self.inner.neighbors_directed(node, direction)
    }
}

impl<'a, N, W> IntoEdgeReferences for &'a PackedGraphMap<N, W>
where
    N: Copy + Eq + Hash + Ord,
    W: Copy + PartialEq,
{
    type EdgeRef = PackedEdgeRef<'a, N, W>;
    type EdgeReferences = Box<dyn Iterator<Item = PackedEdgeRef<'a, N, W>> + 'a>;

    fn edge_references(self) -> Self::EdgeReferences {
        Box::new(
            self.inner
                .all_edges()
                .flat_map(|(source, target, payloads)| {
                    payloads.iter().map(move |weight| PackedEdgeRef {
                        id: (source, target, *weight),
                        weight,
                    })
                }),
        )
    }
}

impl<'a, N, W> IntoEdges for &'a PackedGraphMap<N, W>
where
    N: Copy + Eq + Hash + Ord,
    W: Copy + PartialEq,
{
    type Edges = Box<dyn Iterator<Item = PackedEdgeRef<'a, N, W>> + 'a>;

    fn edges(self, node: N) -> Self::Edges {
        Box::new(
            self.inner
                .edges(node)
                .flat_map(|(source, target, payloads)| {
                    payloads.iter().map(move |weight| PackedEdgeRef {
                        id: (source, target, *weight),
                        weight,
                    })
                }),
        )
    }
}

#[cfg(test)]
mod tests {
    use petgraph::{
        Direction,
        algo::toposort,
        graphmap::DiGraphMap,
        visit::{EdgeRef, IntoEdgeReferences, IntoEdges, IntoNeighborsDirected, NodeCount},
    };

    use super::PackedGraphMap;

    fn two_parallel_edges() -> PackedGraphMap<u32, char> {
        let mut graph = DiGraphMap::<u32, Vec<char>>::new();
        graph.add_edge(1, 2, vec!['a', 'b']);
        graph.add_edge(2, 3, vec!['c']);
        PackedGraphMap::from(graph)
    }

    #[test]
    fn test_parallel_payloads_become_separate_edges() {
        let graph = two_parallel_edges();

        let ids = (&graph)
            .edge_references()
            .map(|edge| edge.id())
            .collect::<Vec<_>>();

        assert_eq!(ids, vec![(1, 2, 'a'), (1, 2, 'b'), (2, 3, 'c')]);
    }

    #[test]
    fn test_edge_payload_resolves_an_id_and_survives_growth() {
        let mut graph = two_parallel_edges();
        let id = (1, 2, 'b');

        graph
            .edge_weight_mut(1, 2)
            .expect("should hold the edge")
            .insert(0, 'z');

        assert_eq!(graph.edge_payload(&id), Some(&'b'));
        assert_eq!(graph.edge_payload(&(1, 2, 'q')), None);
    }

    #[test]
    fn test_edges_lists_the_parallel_payloads_leaving_a_node() {
        let graph = two_parallel_edges();

        let ids = (&graph).edges(1).map(|edge| edge.id()).collect::<Vec<_>>();

        assert_eq!(ids, vec![(1, 2, 'a'), (1, 2, 'b')]);
    }

    #[test]
    fn test_neighbors_stay_deduplicated_per_node_pair() {
        let graph = two_parallel_edges();

        let outgoing = (&graph)
            .neighbors_directed(1, Direction::Outgoing)
            .collect::<Vec<_>>();

        assert_eq!(outgoing, vec![2]);
        assert_eq!(graph.node_count(), 3);
        assert!(toposort(&graph, None).is_ok());
    }
}
