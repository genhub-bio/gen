//! Collapse domain junctions for display without moving pruning decisions between sources.
//!
//! A projected connection can depend on two independent edits. Its ordinary edge bundle cannot
//! express that both edits must survive, so route survival is tracked alongside the metadata.
//! Reference continuations carry no additional edit choice when entering an edit or leaving an
//! insertion cycle. In those cases the replacement inherits only the edit leg's records.

use std::collections::{HashMap, hash_map::Entry};

use gen_core::{
    HashId, INDETERMINATE_CHROMOSOME_INDEX, NO_CHROMOSOME_INDEX,
    PRESERVE_EDIT_SITE_CHROMOSOME_INDEX,
};
use gen_graph::{GenGraph, GraphEdge, GraphNode};
use petgraph::{Direction, algo::kosaraju_scc, graphmap::DiGraphMap};

// Classify records at their original source before contraction combines independent edit legs.
// Match BlockGroup::prune_graph: every projection of the newest origin per chromosome survives.
struct EdgePruning {
    newest_by_chromosome: HashMap<i64, (i64, HashId)>,
}

impl EdgePruning {
    fn new(graph: &GenGraph, source: GraphNode) -> Self {
        let mut newest_by_chromosome: HashMap<i64, (i64, HashId)> = HashMap::new();
        for (_, _, graph_edges) in graph.edges(source) {
            for graph_edge in graph_edges {
                if graph_edge.chromosome_index == NO_CHROMOSOME_INDEX
                    || graph_edge.chromosome_index == INDETERMINATE_CHROMOSOME_INDEX
                    || graph_edge.chromosome_index == PRESERVE_EDIT_SITE_CHROMOSOME_INDEX
                {
                    continue;
                }
                match newest_by_chromosome.entry(graph_edge.chromosome_index) {
                    Entry::Vacant(entry) => {
                        entry.insert((graph_edge.created_on, graph_edge.edge_id));
                    }
                    Entry::Occupied(mut entry) => {
                        if graph_edge.created_on > entry.get().0 {
                            entry.insert((graph_edge.created_on, graph_edge.edge_id));
                        }
                    }
                }
            }
        }
        Self {
            newest_by_chromosome,
        }
    }

    fn retains(&self, graph_edge: &GraphEdge) -> bool {
        let chromosome_index = graph_edge.chromosome_index;
        if chromosome_index == NO_CHROMOSOME_INDEX
            || chromosome_index == INDETERMINATE_CHROMOSOME_INDEX
        {
            return true;
        }
        if chromosome_index == PRESERVE_EDIT_SITE_CHROMOSOME_INDEX {
            return false;
        }
        self.newest_by_chromosome
            .get(&chromosome_index)
            .is_some_and(|&(_, edge_id)| edge_id == graph_edge.edge_id)
    }
}

pub(crate) struct DisplayProjection {
    pub(crate) graph: GenGraph,
    /// Connections with a surviving original route. Empty bundles prevent a second pruning pass
    /// from comparing records that originally belonged to different source nodes.
    pub(crate) surviving_routes: GenGraph,
}

#[derive(Clone)]
struct ProjectedConnection {
    records: Vec<GraphEdge>,
    survives: bool,
}

impl ProjectedConnection {
    fn merge(&mut self, other: &Self) {
        for record in &other.records {
            if !self.records.contains(record) {
                self.records.push(*record);
            }
        }
        self.survives |= other.survives;
    }

    fn through(&self, outgoing: &Self) -> Self {
        let mut combined = self.clone();
        combined.merge(outgoing);
        combined.survives = self.survives && outgoing.survives;
        combined
    }
}

fn is_junction(node: &GraphNode) -> bool {
    !gen_core::is_terminal(node.node_id) && node.sequence_start == node.sequence_end
}

/// Project Cartesian endpoint pairs and retain their original pruning dependencies.
pub(crate) fn project_junctions(graph: &GenGraph) -> DisplayProjection {
    let component_by_node: HashMap<_, _> = kosaraju_scc(graph)
        .into_iter()
        .enumerate()
        .flat_map(|(component, nodes)| nodes.into_iter().map(move |node| (node, component)))
        .collect();
    let mut connections = DiGraphMap::<GraphNode, ProjectedConnection>::new();
    for node in graph.nodes() {
        connections.add_node(node);
        let pruning = EdgePruning::new(graph, node);
        for (_, target, records) in graph.edges(node) {
            connections.add_edge(
                node,
                target,
                ProjectedConnection {
                    records: records.clone(),
                    survives: records.is_empty()
                        || records.iter().any(|record| pruning.retains(record)),
                },
            );
        }
    }

    for junction in graph.nodes().filter(is_junction) {
        let incoming: Vec<_> = connections
            .neighbors_directed(junction, Direction::Incoming)
            .filter(|source| *source != junction)
            .map(|source| (source, connections[(source, junction)].clone()))
            .collect();
        let outgoing: Vec<_> = connections
            .neighbors_directed(junction, Direction::Outgoing)
            .filter(|target| *target != junction)
            .map(|target| (target, connections[(junction, target)].clone()))
            .collect();

        for (source, entrance) in &incoming {
            for (target, exit) in &outgoing {
                if source == target {
                    continue;
                }
                let enters_by_continuation = source.node_id == junction.node_id
                    && source.sequence_end == junction.sequence_start;
                let exits_by_continuation = target.node_id == junction.node_id
                    && target.sequence_start == junction.sequence_end;
                let returns_to_insertion =
                    component_by_node.get(source) == component_by_node.get(&junction);
                let replacement = if enters_by_continuation && !exits_by_continuation {
                    exit.clone()
                } else if !enters_by_continuation && exits_by_continuation && returns_to_insertion {
                    // Match the eager insertion exit: a return carries the edit's chromosome
                    // records through its anchor. An acyclic edit must still respect its exit
                    // continuation, otherwise contraction can restore a replaced reference base.
                    entrance.clone()
                } else {
                    entrance.through(exit)
                };
                if let Some(existing) = connections.edge_weight_mut(*source, *target) {
                    existing.merge(&replacement);
                } else {
                    connections.add_edge(*source, *target, replacement);
                }
            }
        }
        connections.remove_node(junction);
    }

    // Preserve input order so contraction does not change layout tie-breaking.
    let retained_nodes: Vec<_> = graph.nodes().filter(|node| !is_junction(node)).collect();
    let node_order: HashMap<_, _> = retained_nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (*node, index))
        .collect();
    let mut projection = DisplayProjection {
        graph: GenGraph::new(),
        surviving_routes: GenGraph::new(),
    };
    for node in &retained_nodes {
        projection.graph.add_node(*node);
        projection.surviving_routes.add_node(*node);
    }
    for source in retained_nodes {
        let mut targets: Vec<_> = connections
            .neighbors_directed(source, Direction::Outgoing)
            .collect();
        targets.sort_unstable_by_key(|target| node_order[target]);
        for target in targets {
            let connection = &connections[(source, target)];
            projection
                .graph
                .add_edge(source, target, connection.records.clone());
            if connection.survives {
                projection
                    .surviving_routes
                    .add_edge(source, target, Vec::new());
            }
        }
    }
    projection
}

#[cfg(test)]
mod tests {
    use gen_core::{
        HashId, NO_CHROMOSOME_INDEX, PATH_START_NODE_ID, PRESERVE_EDIT_SITE_CHROMOSOME_INDEX,
        Strand,
    };
    use gen_graph::{GenGraph, GraphEdge, GraphNode};

    use super::project_junctions;

    fn node(label: &str, start: i64, end: i64) -> GraphNode {
        GraphNode {
            node_id: HashId::convert_str(label),
            sequence_start: start,
            sequence_end: end,
        }
    }

    fn edge(label: &str, chromosome_index: i64, created_on: i64) -> GraphEdge {
        GraphEdge {
            edge_id: HashId::convert_str(label),
            source_strand: Strand::Forward,
            target_strand: Strand::Forward,
            chromosome_index,
            phased: 1,
            created_on,
        }
    }

    fn insertion_graph(reference_chromosome_index: i64) -> GenGraph {
        let mut graph = GenGraph::new();
        let upstream = node("reference", 0, 2);
        let junction = node("reference", 2, 2);
        let downstream = node("reference", 2, 4);
        let inserted = node("inserted", 0, 3);
        let marker = vec![edge("marker", reference_chromosome_index, 2)];
        let start = GraphNode {
            node_id: PATH_START_NODE_ID,
            sequence_start: 0,
            sequence_end: 0,
        };
        graph.add_edge(start, upstream, vec![edge("start", NO_CHROMOSOME_INDEX, 0)]);
        graph.add_edge(upstream, junction, marker.clone());
        graph.add_edge(junction, downstream, marker);
        graph.add_edge(junction, inserted, vec![edge("enter-insertion", 1, 2)]);
        graph.add_edge(inserted, junction, vec![edge("leave-insertion", 1, 2)]);
        graph
    }

    #[test]
    fn test_insertion_entrance_and_return_inherit_the_edit_chromosome() {
        for reference_chromosome_index in [0, PRESERVE_EDIT_SITE_CHROMOSOME_INDEX] {
            let graph = insertion_graph(reference_chromosome_index);
            let projection = project_junctions(&graph);
            let upstream = node("reference", 0, 2);
            let inserted = node("inserted", 0, 3);
            let downstream = node("reference", 2, 4);
            assert_eq!(
                projection.graph.edge_weight(upstream, inserted),
                Some(&vec![edge("enter-insertion", 1, 2)]),
                "a reference-copy marker should not assign its chromosome to the inserted allele"
            );
            assert_eq!(
                projection.graph.edge_weight(inserted, downstream),
                Some(&vec![edge("leave-insertion", 1, 2)]),
                "the insertion return should carry its own origin and chromosome through the anchor"
            );
            assert!(
                projection
                    .surviving_routes
                    .contains_edge(upstream, inserted)
            );
            assert!(
                projection
                    .surviving_routes
                    .contains_edge(inserted, downstream)
            );
            assert_eq!(
                projection
                    .surviving_routes
                    .contains_edge(upstream, downstream),
                reference_chromosome_index == 0,
                "only an ordinary reference-copy marker should keep the unedited route lit"
            );
            assert!(graph.contains_node(node("reference", 2, 2)));
            assert!(!projection.graph.contains_node(node("reference", 2, 2)));
        }
    }

    #[test]
    fn test_insertion_return_retains_all_chromosome_records() {
        let mut graph = insertion_graph(PRESERVE_EDIT_SITE_CHROMOSOME_INDEX);
        let junction = node("reference", 2, 2);
        let inserted = node("inserted", 0, 3);
        graph
            .edge_weight_mut(inserted, junction)
            .unwrap()
            .push(edge("leave-insertion", 0, 2));
        let projection = project_junctions(&graph);
        assert_eq!(
            projection
                .graph
                .edge_weight(inserted, node("reference", 2, 4)),
            graph.edge_weight(inserted, junction),
            "projection should preserve the complete return bundle"
        );
    }

    #[test]
    fn test_acyclic_edit_does_not_restore_a_replaced_reference_continuation() {
        let mut graph = insertion_graph(PRESERVE_EDIT_SITE_CHROMOSOME_INDEX);
        graph.remove_edge(node("reference", 2, 2), node("inserted", 0, 3));
        let projection = project_junctions(&graph);
        assert!(
            projection
                .graph
                .contains_edge(node("inserted", 0, 3), node("reference", 2, 4))
        );
        assert!(
            !projection
                .surviving_routes
                .contains_edge(node("inserted", 0, 3), node("reference", 2, 4)),
            "a marker-only exit should still block an acyclic route into a replaced base"
        );
    }

    #[test]
    fn test_downstream_timestamp_cannot_revive_a_superseded_entrance() {
        let source = node("source", 0, 2);
        let junction = node("junction", 1, 1);
        let downstream = node("downstream", 0, 2);
        let replacement = node("replacement", 0, 2);
        let mut graph = GenGraph::new();
        graph.add_edge(source, junction, vec![edge("old-entrance", 0, 1)]);
        graph.add_edge(source, replacement, vec![edge("new-entrance", 0, 2)]);
        graph.add_edge(junction, downstream, vec![edge("later-exit", 0, 100)]);
        let projection = project_junctions(&graph);
        assert!(projection.graph.contains_edge(source, downstream));
        assert!(
            !projection
                .surviving_routes
                .contains_edge(source, downstream)
        );
        assert!(
            projection
                .surviving_routes
                .contains_edge(source, replacement)
        );
    }

    #[test]
    fn test_superseded_exit_stays_dim_after_sources_are_collapsed() {
        let source = node("source", 0, 2);
        let junction = node("junction", 1, 1);
        let retired = node("retired", 0, 2);
        let current = node("current", 0, 2);
        let mut graph = GenGraph::new();
        graph.add_edge(source, junction, vec![edge("entrance", 0, 1)]);
        graph.add_edge(junction, retired, vec![edge("old-exit", 0, 2)]);
        graph.add_edge(junction, current, vec![edge("new-exit", 0, 3)]);
        let projection = project_junctions(&graph);
        assert!(projection.graph.contains_edge(source, retired));
        assert!(!projection.surviving_routes.contains_edge(source, retired));
        assert!(projection.surviving_routes.contains_edge(source, current));
    }

    #[test]
    fn test_shared_connection_survives_when_one_chromosome_remains_current() {
        let source = node("source", 0, 2);
        let reference = node("reference", 0, 2);
        let variant = node("variant", 0, 2);
        let mut graph = GenGraph::new();
        graph.add_edge(
            source,
            reference,
            vec![edge("old-0", 0, 1), edge("reference-1", 1, 1)],
        );
        graph.add_edge(source, variant, vec![edge("new-0", 0, 2)]);
        let projection = project_junctions(&graph);
        assert!(projection.surviving_routes.contains_edge(source, reference));
        assert!(projection.surviving_routes.contains_edge(source, variant));
    }
}
