//! Dims what later edits superseded in a viewer's `GenGraph`: the edges `BlockGroup::prune_graph`
//! would remove, and the nodes that only those edges lead into.
//!
//! A lazily crawled graph is only partly loaded, and a batch reached by a jump may have no loaded
//! path back to `PATH_START` at all, so nothing here searches from the start or looks past what
//! is loaded. Each verdict comes from one node's own edges, and is only drawn once the side it
//! depends on is complete (see [`GraphSource::is_frontier`]):
//!
//! - An edge is pruned when a newer sibling with the same chromosome index leaves the same source
//!   node, or when it is an edit-site marker. Every outgoing edge of a node leaves from its end
//!   port, so once that side is complete all the siblings are loaded.
//! - A node is dimmed when its incoming side is complete and every incoming edge is pruned or
//!   comes from a dimmed node. A node with an unloaded incoming edge stays lit, since the missing
//!   edge may be the one that reaches it.
//!
//! Loading more of the graph never adds edges to a complete side, so neither verdict can change
//! once drawn. [`GraphDimming`] keeps what it has decided and only examines new nodes, nodes whose
//! sides became complete, and the successors of anything newly pruned or dimmed.
//!
//! Over a fully loaded graph this dims the same edges as `prune_graph`, and the same nodes as its
//! reachability pass from `PATH_START`, except that a cycle is never dimmed as a whole: each
//! member keeps the next one lit. That keeps a circular genome lit once its `PATH_START`
//! attachment is dropped (see `lazy_graph_source`), at the cost of also keeping lit a cycle that
//! only pruned edges lead into.

use std::collections::{HashMap, HashSet, hash_map::Entry};

use gen_core::{
    INDETERMINATE_CHROMOSOME_INDEX, NO_CHROMOSOME_INDEX, PRESERVE_EDIT_SITE_CHROMOSOME_INDEX,
    is_start_node,
};
use gen_graph::{GenGraph, GraphEdge, GraphNode};
use gen_tui::{crawl::GraphSource, graph_view::GraphViewState};
use petgraph::{Direction, visit::NodeIndexable as _};

/// The dimming decided so far for one viewer's graph, mirrored into its `GraphViewState`
/// lowlights. Viewers keep one next to their `LayoutEngine` and call [`Self::sync`] whenever the
/// crawl may have grown the graph.
#[derive(Clone, Debug, Default)]
pub struct GraphDimming {
    /// How many of the graph's nodes, in insertion order, have been seen.
    seen_nodes: usize,
    /// Seen nodes with outgoing edges still unloaded, so which of those are pruned is open.
    pending_sources: HashSet<GraphNode>,
    /// Seen nodes with incoming edges still unloaded, so they cannot be dimmed yet.
    pending_targets: HashSet<GraphNode>,
    pruned_edges: HashSet<(GraphNode, GraphNode)>,
    dimmed_nodes: HashSet<GraphNode>,
}

impl GraphDimming {
    /// Dim whatever became decidable since the last call, given what `source` has loaded into
    /// `graph`, and return whether any lowlight in `view_state` changed. Cheap when nothing
    /// changed: it only rechecks the nodes still waiting on unloaded edges.
    pub fn sync<S: GraphSource<GenGraph>>(
        &mut self,
        graph: &GenGraph,
        source: &S,
        view_state: &mut GraphViewState<GraphNode>,
    ) -> bool {
        let mut changed = false;
        // Nodes are only ever added by the crawl, so fewer of them means a different graph.
        if graph.node_count() < self.seen_nodes {
            *self = Self::default();
            view_state.highlights.edge_lowlights.clear();
            view_state.highlights.node_lowlights.clear();
            changed = true;
        }
        for index in self.seen_nodes..graph.node_count() {
            let node = graph.from_index(index);
            self.pending_sources.insert(node);
            self.pending_targets.insert(node);
        }
        self.seen_nodes = graph.node_count();

        let mut candidates: Vec<GraphNode> = Vec::new();
        let judged_sources: Vec<GraphNode> = self
            .pending_sources
            .iter()
            .copied()
            .filter(|node| !source.is_frontier(*node, Direction::Outgoing))
            .collect();
        for node in judged_sources {
            self.pending_sources.remove(&node);
            for edge in superseded_edges(graph, node) {
                if self.pruned_edges.insert(edge) {
                    view_state.dim_edge(edge);
                    candidates.push(edge.1);
                    changed = true;
                }
            }
        }
        self.pending_targets.retain(|node| {
            let is_open = source.is_frontier(*node, Direction::Incoming);
            if !is_open {
                candidates.push(*node);
            }
            is_open
        });

        while let Some(node) = candidates.pop() {
            if self.dimmed_nodes.contains(&node)
                || self.pending_targets.contains(&node)
                || is_start_node(node.node_id)
                || !self.is_only_entered_through_dimming(graph, node)
            {
                continue;
            }
            self.dimmed_nodes.insert(node);
            view_state.dim_node(node);
            changed = true;
            candidates.extend(graph.neighbors_directed(node, Direction::Outgoing));
        }
        changed
    }

    /// Whether every edge into `node` is pruned or comes from a dimmed node. A self-loop does not
    /// count, since a node cannot be what reaches it. With no incoming edges at all, nothing
    /// reaches the node, as `prune_graph`'s reachability pass would also find.
    fn is_only_entered_through_dimming(&self, graph: &GenGraph, node: GraphNode) -> bool {
        graph
            .neighbors_directed(node, Direction::Incoming)
            .all(|predecessor| {
                predecessor == node
                    || self.pruned_edges.contains(&(predecessor, node))
                    || self.dimmed_nodes.contains(&predecessor)
            })
    }
}

/// The outgoing edges of `source` that `BlockGroup::prune_graph` removes: every edit-site marker,
/// and for each chromosome index all but the newest edge. Edges without a chromosome index, or
/// with an indeterminate one, are always kept.
fn superseded_edges(graph: &GenGraph, source: GraphNode) -> Vec<(GraphNode, GraphNode)> {
    let mut superseded = Vec::new();
    let mut newest_by_chromosome: HashMap<i64, ((GraphNode, GraphNode), i64)> = HashMap::new();
    for (edge_source, target, graph_edges) in graph.edges(source) {
        let edge = (edge_source, target);
        for &GraphEdge {
            chromosome_index,
            created_on,
            ..
        } in graph_edges
        {
            if chromosome_index == NO_CHROMOSOME_INDEX
                || chromosome_index == INDETERMINATE_CHROMOSOME_INDEX
            {
                continue;
            }
            if chromosome_index == PRESERVE_EDIT_SITE_CHROMOSOME_INDEX {
                superseded.push(edge);
                continue;
            }
            match newest_by_chromosome.entry(chromosome_index) {
                Entry::Vacant(entry) => {
                    entry.insert((edge, created_on));
                }
                Entry::Occupied(mut entry) => {
                    let (newest_edge, newest_created_on) = entry.get_mut();
                    if created_on > *newest_created_on {
                        superseded.push(*newest_edge);
                        *newest_edge = edge;
                        *newest_created_on = created_on;
                    } else {
                        superseded.push(edge);
                    }
                }
            }
        }
    }
    superseded
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use gen_core::{HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand};
    use gen_graph::{GenGraph, GraphNode};
    use gen_models::{
        block_group::{BlockGroup, NewBlockGroup},
        block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
        collection::Collection,
        db::get_connection,
        edge::Edge,
        node::Node,
        sample::{NewSample, Sample},
        sequence::Sequence,
    };
    use gen_tui::{graph_view::GraphViewState, layout_engine::LayoutEngine};

    use super::GraphDimming;
    use crate::views::lazy_graph_source::{SqlGraphSource, seed_block_group_graph};

    /// `(source label, source coordinate, target label, target coordinate)`, where "start" and
    /// "end" name the path sentinels.
    type EdgeSpec = (String, i64, String, i64);

    const NODE_LENGTH: i64 = 5;

    fn node_id_for(label: &str) -> HashId {
        match label {
            "start" => PATH_START_NODE_ID,
            "end" => PATH_END_NODE_ID,
            _ => HashId::convert_str(label),
        }
    }

    fn whole_node(label: &str) -> GraphNode {
        GraphNode {
            node_id: node_id_for(label),
            sequence_start: 0,
            sequence_end: NODE_LENGTH,
        }
    }

    fn sentinel(node_id: HashId) -> GraphNode {
        GraphNode {
            node_id,
            sequence_start: 0,
            sequence_end: 0,
        }
    }

    fn link(source: &str, target: &str) -> EdgeSpec {
        let source_coordinate = if source == "start" { 0 } else { NODE_LENGTH };
        (source.to_string(), source_coordinate, target.to_string(), 0)
    }

    /// Store an on-disk block group whose `edge_batches` are recorded one after another, all on
    /// chromosome index 0, so an edge in a later batch supersedes its siblings from earlier ones.
    fn store_block_group(
        db_path: &Path,
        labels: &[String],
        edge_batches: &[Vec<EdgeSpec>],
    ) -> HashId {
        let conn = get_connection(db_path).expect("should open the test database");
        Collection::get_or_create(&conn, "test").expect("should create the collection");
        Sample::get_or_create(
            &conn,
            NewSample {
                name: "test",
                ..Default::default()
            },
        )
        .expect("should create the sample");
        let block_group = BlockGroup::create(
            &conn,
            NewBlockGroup {
                collection_name: "test",
                sample_name: "test",
                name: "chr1",
                ..Default::default()
            },
        )
        .expect("should create the block group");
        for label in labels {
            let sequence = Sequence::new()
                .sequence_type("DNA")
                .sequence("AAAAA")
                .save(&conn)
                .expect("should save the sequence");
            Node::create(&conn, &sequence.hash, &node_id_for(label))
                .expect("should create the node");
        }
        for edges in edge_batches {
            let block_group_edges: Vec<BlockGroupEdgeData> = edges
                .iter()
                .map(|(source, source_coordinate, target, target_coordinate)| {
                    let edge = Edge::create(
                        &conn,
                        node_id_for(source),
                        *source_coordinate,
                        Strand::Forward,
                        node_id_for(target),
                        *target_coordinate,
                        Strand::Forward,
                    )
                    .expect("should create the edge");
                    BlockGroupEdgeData {
                        block_group_id: block_group.id,
                        edge_id: edge.id,
                        chromosome_index: 0,
                        phased: 0,
                    }
                })
                .collect();
            BlockGroupEdge::bulk_create(&conn, &block_group_edges);
        }
        block_group.id
    }

    /// A 30-node reference chain whose link out of `n20` was first replaced by the two-node
    /// branch `old_a -> old_b`, and later by the single node `new`, both rejoining at `n21`.
    fn store_superseded_branch(db_path: &Path) -> HashId {
        let mut labels: Vec<String> = (0..30).map(|index| format!("n{index}")).collect();
        labels.extend(["old_a", "old_b", "new"].map(String::from));
        let mut reference = vec![link("start", "n0"), link("n29", "end")];
        for index in 0..29 {
            reference.push(link(&format!("n{index}"), &format!("n{}", index + 1)));
        }
        let old_branch = vec![
            link("n20", "old_a"),
            link("old_a", "old_b"),
            link("old_b", "n21"),
        ];
        let new_branch = vec![link("n20", "new"), link("new", "n21")];
        store_block_group(db_path, &labels, &[reference, old_branch, new_branch])
    }

    fn is_dimmed_node(view_state: &GraphViewState<GraphNode>, node: GraphNode) -> bool {
        view_state.highlights.node_lowlights.contains(&node)
    }

    fn is_dimmed_edge(view_state: &GraphViewState<GraphNode>, source: &str, target: &str) -> bool {
        view_state
            .highlights
            .edge_lowlights
            .contains(&(whole_node(source), whole_node(target)))
    }

    #[test]
    fn test_jumped_to_batch_far_from_start_dims_only_its_pruned_branch() {
        let dir = tempfile::tempdir().expect("should create a temporary directory");
        let db_path = dir.path().join("graph.db");
        let block_group_id = store_superseded_branch(&db_path);
        let mut seed = GenGraph::new();
        let anchor = whole_node("n20");
        seed.add_node(anchor);
        let mut engine =
            LayoutEngine::new_with_source(seed, SqlGraphSource::new(db_path, block_group_id));
        engine
            .activate_batch_containing(anchor, 12)
            .expect("should claim a batch around the jump target");
        assert!(
            !engine.graph().contains_node(sentinel(PATH_START_NODE_ID)),
            "the batch should have no loaded connection back to PATH_START"
        );
        let world = engine.active_world().expect("should have an active world");
        for label in ["old_a", "old_b", "new", "n19", "n21"] {
            assert!(
                world.contains(whole_node(label)),
                "{label} should be in the jumped-to batch"
            );
        }

        let mut view_state = GraphViewState::default();
        let mut dimming = GraphDimming::default();
        assert!(dimming.sync(engine.graph(), engine.source(), &mut view_state));

        assert!(is_dimmed_edge(&view_state, "n20", "old_a"));
        assert!(is_dimmed_edge(&view_state, "n20", "n21"));
        assert!(!is_dimmed_edge(&view_state, "n20", "new"));
        assert!(is_dimmed_node(&view_state, whole_node("old_a")));
        assert!(
            is_dimmed_node(&view_state, whole_node("old_b")),
            "old_b is only entered from the dimmed old_a"
        );
        let superseded = [node_id_for("old_a"), node_id_for("old_b")];
        let wrongly_dimmed: Vec<GraphNode> = world
            .members()
            .filter(|node| !superseded.contains(&node.node_id))
            .filter(|node| is_dimmed_node(&view_state, *node))
            .collect();
        assert!(
            wrongly_dimmed.is_empty(),
            "no reference or new node should be dimmed, got {wrongly_dimmed:?}"
        );
    }

    #[test]
    fn test_fully_crawled_pruned_branch_is_dimmed_once() {
        let dir = tempfile::tempdir().expect("should create a temporary directory");
        let db_path = dir.path().join("graph.db");
        let block_group_id = store_superseded_branch(&db_path);
        let start = sentinel(PATH_START_NODE_ID);
        let mut seed = GenGraph::new();
        seed.add_node(start);
        let mut engine =
            LayoutEngine::new_with_source(seed, SqlGraphSource::new(db_path, block_group_id));
        engine
            .activate_batch_containing(start, 100)
            .expect("should claim the whole graph as one batch");

        let mut view_state = GraphViewState::default();
        let mut dimming = GraphDimming::default();
        assert!(dimming.sync(engine.graph(), engine.source(), &mut view_state));

        let mut dimmed_nodes = view_state.highlights.node_lowlights.clone();
        dimmed_nodes.sort();
        let mut expected_nodes = vec![whole_node("old_a"), whole_node("old_b")];
        expected_nodes.sort();
        assert_eq!(dimmed_nodes, expected_nodes);
        let mut dimmed_edges = view_state.highlights.edge_lowlights.clone();
        dimmed_edges.sort();
        let mut expected_edges = vec![
            (whole_node("n20"), whole_node("old_a")),
            (whole_node("n20"), whole_node("n21")),
        ];
        expected_edges.sort();
        assert_eq!(dimmed_edges, expected_edges);

        assert!(
            !dimming.sync(engine.graph(), engine.source(), &mut view_state),
            "a sync with nothing newly loaded should change nothing"
        );
        assert_eq!(view_state.highlights.node_lowlights.len(), 2);
    }

    #[test]
    fn test_circular_seed_is_not_dimmed() {
        let dir = tempfile::tempdir().expect("should create a temporary directory");
        let db_path = dir.path().join("graph.db");
        let labels: Vec<String> = ["x", "y", "z"].map(String::from).to_vec();
        let edges = vec![
            link("start", "x"),
            link("x", "y"),
            link("y", "z"),
            link("z", "end"),
            link("z", "x"),
            ("end".to_string(), 0, "start".to_string(), 0),
        ];
        let block_group_id = store_block_group(&db_path, &labels, &[edges]);
        let conn = get_connection(&db_path).expect("should open the test database");
        let seed = seed_block_group_graph(&conn, &block_group_id);
        let anchor = whole_node("x");
        assert!(
            seed.contains_node(anchor),
            "a circular group should seed on x"
        );
        let mut engine =
            LayoutEngine::new_with_source(seed, SqlGraphSource::new(db_path, block_group_id));
        engine
            .activate_batch_containing(anchor, 10)
            .expect("should claim the whole circle as one batch");

        let mut view_state = GraphViewState::default();
        GraphDimming::default().sync(engine.graph(), engine.source(), &mut view_state);

        for label in ["x", "y", "z"] {
            assert!(
                engine.active_contains(whole_node(label)),
                "{label} should be in the batch"
            );
            assert!(
                !is_dimmed_node(&view_state, whole_node(label)),
                "{label} should stay lit"
            );
        }
    }
}
