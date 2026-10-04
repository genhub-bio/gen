//! Plans the edges an edit writes from the ports it spans.
//!
//! A port is a point `(node_id, coordinate)` between two bases of a node, or at one of its ends.
//! A deletion is an edge from its left port to its right port; an insertion is an edge into its
//! allele and an edge out of it. Marker edges split a node at a port inside it, so the unedited
//! route stays in the graph under its own chromosome index.
//!
//! Every combination of adjacent edits is an edge of its own: where routes already arrive at the
//! left port, the edit also starts from each route's source, and where routes leave the right
//! port it also ends at each of their targets. Two touching deletions `(1,2)` and `(2,3)` also
//! write `(1,3)`. An edit with `preserve_edge` false re-adds the edges it continues with
//! `PRESERVE_EDIT_SITE_CHROMOSOME_INDEX`, retiring them.

use std::collections::{HashMap, HashSet};

use gen_core::{
    HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, PRESERVE_EDIT_SITE_CHROMOSOME_INDEX, Strand,
    is_terminal,
};
use indexmap::IndexSet;

use crate::{
    block_group::BlockGroupChange,
    block_group_edge::AugmentedEdgeData,
    db::GraphConnection,
    edge::{BlockKey, Edge, EdgeData},
    errors::EdgeError,
};

/// A port an edit's first edge leaves from or its last edge leads to, and the stored edge whose
/// route it continues, if any.
#[derive(Clone, Copy, Debug)]
struct Endpoint {
    port: BlockKey,
    edge: Option<AugmentedEdgeData>,
}

/// The edges of one block group, stored and planned, indexed by the ports they meet.
///
/// `BlockGroup::insert_changes` keeps one per block group and adds each change's edges before
/// planning the next, so edits that meet within one batch combine as if applied one at a time.
pub struct EdgeLookup {
    block_group_id: HashId,
    loaded_node_ids: HashSet<HashId>,
    retired: HashSet<EdgeData>,
    arriving: HashMap<BlockKey, IndexSet<AugmentedEdgeData>>,
    leaving: HashMap<BlockKey, IndexSet<AugmentedEdgeData>>,
    first_arrival: HashMap<HashId, i64>,
    last_departure: HashMap<HashId, i64>,
}

impl EdgeLookup {
    pub fn new(block_group_id: HashId) -> Self {
        EdgeLookup {
            block_group_id,
            loaded_node_ids: HashSet::new(),
            retired: HashSet::new(),
            arriving: HashMap::new(),
            leaving: HashMap::new(),
            first_arrival: HashMap::new(),
            last_departure: HashMap::new(),
        }
    }

    fn index(&mut self, augmented_edge_data: AugmentedEdgeData) {
        let edge = augmented_edge_data.edge_data;
        if augmented_edge_data.chromosome_index == PRESERVE_EDIT_SITE_CHROMOSOME_INDEX
            && !is_marker(&edge)
        {
            self.retired.insert(edge);
        }
        self.leaving
            .entry(BlockKey::new(edge.source_node_id, edge.source_coordinate))
            .or_default()
            .insert(augmented_edge_data);
        self.arriving
            .entry(BlockKey::new(edge.target_node_id, edge.target_coordinate))
            .or_default()
            .insert(augmented_edge_data);
        self.first_arrival
            .entry(edge.target_node_id)
            .and_modify(|first| *first = (*first).min(edge.target_coordinate))
            .or_insert(edge.target_coordinate);
        self.last_departure
            .entry(edge.source_node_id)
            .and_modify(|last| *last = (*last).max(edge.source_coordinate))
            .or_insert(edge.source_coordinate);
    }

    /// Loads a node's stored edges the first time an edit touches it.
    fn load(&mut self, conn: &GraphConnection, node_id: HashId) -> Result<(), EdgeError> {
        if !self.loaded_node_ids.insert(node_id) {
            return Ok(());
        }
        for stored in
            Edge::edges_for_block_group_nodes(conn, &self.block_group_id, &[node_id], None)?
        {
            self.index(AugmentedEdgeData {
                edge_data: EdgeData::from(&stored.edge),
                chromosome_index: stored.chromosome_index,
                phased: stored.phased,
            });
        }
        Ok(())
    }

    /// The live edges meeting `port` in `edges`, leaving out markers and retired edges.
    fn live(&self, edges: Option<&IndexSet<AugmentedEdgeData>>) -> Vec<AugmentedEdgeData> {
        edges
            .into_iter()
            .flatten()
            .filter(|edge| !is_marker(&edge.edge_data) && !self.retired.contains(&edge.edge_data))
            .copied()
            .collect()
    }

    fn arriving(&self, port: BlockKey) -> Vec<AugmentedEdgeData> {
        self.live(self.arriving.get(&port))
    }

    fn leaving(&self, port: BlockKey) -> Vec<AugmentedEdgeData> {
        self.live(self.leaving.get(&port))
    }

    /// Whether some route reaches `port` along its own node, so the port can start an edge.
    fn has_sequence_before(&self, port: BlockKey) -> bool {
        match port.node_id {
            PATH_START_NODE_ID => true,
            PATH_END_NODE_ID => false,
            node_id => self
                .first_arrival
                .get(&node_id)
                .is_some_and(|first| *first < port.coordinate),
        }
    }

    /// Whether some route continues from `port` along its own node, so an edge can end there.
    fn has_sequence_after(&self, port: BlockKey) -> bool {
        match port.node_id {
            PATH_END_NODE_ID => true,
            PATH_START_NODE_ID => false,
            node_id => self
                .last_departure
                .get(&node_id)
                .is_some_and(|last| *last > port.coordinate),
        }
    }

    /// Whether every marker at `port` is retired, so the node's own sequence no longer runs
    /// through it.
    fn is_split_retired(&self, port: BlockKey) -> bool {
        let mut markers = self
            .leaving
            .get(&port)
            .into_iter()
            .flatten()
            .filter(|edge| is_marker(&edge.edge_data))
            .peekable();
        markers.peek().is_some()
            && markers.all(|marker| marker.chromosome_index == PRESERVE_EDIT_SITE_CHROMOSOME_INDEX)
    }

    /// Loads the nodes at the far end of every edge meeting `port`.
    fn load_neighbors(&mut self, conn: &GraphConnection, port: BlockKey) -> Result<(), EdgeError> {
        let neighbors = self
            .arriving(port)
            .iter()
            .map(|edge| edge.edge_data.source_node_id)
            .chain(
                self.leaving(port)
                    .iter()
                    .map(|edge| edge.edge_data.target_node_id),
            )
            .collect::<Vec<_>>();
        for node_id in neighbors {
            self.load(conn, node_id)?;
        }
        Ok(())
    }

    /// Where an edit starting at `port` attaches its first edge: the port itself when a route
    /// runs along the node into it, and the source of every live edge arriving there.
    fn sources(&self, port: BlockKey, rules: &EditRules) -> Vec<Endpoint> {
        let mut sources = vec![];
        if self.has_sequence_before(port) {
            sources.push(Endpoint { port, edge: None });
        }
        for arriving in self.arriving(port) {
            let source = BlockKey::new(
                arriving.edge_data.source_node_id,
                arriving.edge_data.source_coordinate,
            );
            if rules.allows(source.node_id) && !rules.returns_to_end(source) {
                sources.push(Endpoint {
                    port: source,
                    edge: Some(arriving),
                });
            }
        }
        sources
    }

    /// Where an edit ending at `port` leads its last edge: the port itself when the node's
    /// sequence continues from it, and the target of every live edge leaving there.
    fn targets(&self, port: BlockKey, rules: &EditRules) -> Vec<Endpoint> {
        let mut targets = vec![];
        for leaving in self.leaving(port) {
            let target = BlockKey::new(
                leaving.edge_data.target_node_id,
                leaving.edge_data.target_coordinate,
            );
            if rules.allows(target.node_id) && !rules.returns_to_start(target) {
                targets.push(Endpoint {
                    port: target,
                    edge: Some(leaving),
                });
            }
        }
        // An earlier edit replaced the node's sequence after the port for every chromosome copy,
        // so the edit continues into that edit's allele or deletion instead.
        let leads_elsewhere = !targets.is_empty();
        if self.has_sequence_after(port) && !(leads_elsewhere && self.is_split_retired(port)) {
            targets.insert(0, Endpoint { port, edge: None });
        }
        targets
    }

    /// Plans the edges `change` writes to replace the sequence between `start` and `end` with its
    /// block, and records them for the edits planned after it.
    pub fn plan(
        &mut self,
        conn: &GraphConnection,
        start: BlockKey,
        end: BlockKey,
        change: &BlockGroupChange,
    ) -> Result<Vec<AugmentedEdgeData>, EdgeError> {
        self.load(conn, start.node_id)?;
        self.load(conn, end.node_id)?;
        let is_deletion = change.block.sequence_start == change.block.sequence_end;
        let is_insertion = start == end;
        if is_deletion && is_insertion {
            return Ok(vec![]);
        }
        self.load_neighbors(conn, start)?;
        self.load_neighbors(conn, end)?;
        let rules = EditRules {
            change,
            allele: (!is_deletion).then_some(change.block.node_id),
            start,
            end,
        };
        let variant = |source: BlockKey, target: BlockKey| AugmentedEdgeData {
            edge_data: EdgeData {
                source_node_id: source.node_id,
                source_coordinate: source.coordinate,
                source_strand: Strand::Forward,
                target_node_id: target.node_id,
                target_coordinate: target.coordinate,
                target_strand: Strand::Forward,
            },
            chromosome_index: change.chromosome_index,
            phased: change.phased,
        };

        let mut new_edges = IndexSet::new();
        for port in [start, end] {
            if !is_terminal(port.node_id)
                && self.has_sequence_before(port)
                && self.has_sequence_after(port)
            {
                new_edges.insert(AugmentedEdgeData {
                    chromosome_index: rules.marker_chromosome_index(),
                    phased: 0,
                    ..variant(port, port)
                });
            }
        }

        let sources = self.sources(start, &rules);
        let mut targets = self.targets(end, &rules);
        if is_insertion && !self.has_sequence_before(start) {
            // The path enters the next block through an allele, so the insertion sits at the port
            // that allele was entered from and leads into every route leaving that port.
            for source in &sources {
                for target in self.targets(source.port, &rules) {
                    if !targets.iter().any(|known| known.port == target.port) {
                        targets.push(target);
                    }
                }
            }
        }
        let mut continued = vec![];
        match rules.allele {
            None => {
                for source in &sources {
                    for target in &targets {
                        if is_combined_deletion(source, target) {
                            new_edges.insert(variant(source.port, target.port));
                            continued.extend(source.edge);
                            continued.extend(target.edge);
                        }
                    }
                }
            }
            Some(node_id) => {
                for source in &sources {
                    new_edges.insert(variant(
                        source.port,
                        BlockKey::new(node_id, change.block.sequence_start),
                    ));
                    continued.extend(source.edge);
                }
                for target in &targets {
                    new_edges.insert(variant(
                        BlockKey::new(node_id, change.block.sequence_end),
                        target.port,
                    ));
                    continued.extend(target.edge);
                }
            }
        }
        if !change.preserve_edge {
            let written = new_edges
                .iter()
                .map(|edge| edge.edge_data)
                .collect::<HashSet<_>>();
            for retired in continued
                .into_iter()
                .filter(|continued| !written.contains(&continued.edge_data))
            {
                new_edges.insert(AugmentedEdgeData {
                    edge_data: retired.edge_data,
                    chromosome_index: PRESERVE_EDIT_SITE_CHROMOSOME_INDEX,
                    phased: 0,
                });
            }
        }

        let new_edges = new_edges.into_iter().collect::<Vec<_>>();
        for edge in &new_edges {
            self.index(*edge);
        }
        Ok(new_edges)
    }
}

/// Whether the edge is a marker: both endpoints are the same node and coordinate.
fn is_marker(edge: &EdgeData) -> bool {
    edge.source_node_id == edge.target_node_id && edge.source_coordinate == edge.target_coordinate
}

/// Whether a deletion from `source` to `target` is an edge to write: never one running backwards
/// along its node.
fn is_combined_deletion(source: &Endpoint, target: &Endpoint) -> bool {
    !(source.port.node_id == target.port.node_id
        && source.port.coordinate >= target.port.coordinate)
}

/// What the fan-out of one edit depends on besides its ports.
struct EditRules<'a> {
    change: &'a BlockGroupChange,
    /// The inserted node, or `None` for a deletion.
    allele: Option<HashId>,
    start: BlockKey,
    end: BlockKey,
}

impl EditRules<'_> {
    fn marker_chromosome_index(&self) -> i64 {
        if self.change.preserve_edge {
            0
        } else {
            PRESERVE_EDIT_SITE_CHROMOSOME_INDEX
        }
    }

    /// Whether the edit may continue a route whose far end is on `node_id`: not its own allele
    /// applied again.
    fn allows(&self, node_id: HashId) -> bool {
        Some(node_id) != self.allele
    }

    /// Whether a route to `target` re-enters the start node at or before the edit's start, which
    /// would close a cycle through the edit.
    fn returns_to_start(&self, target: BlockKey) -> bool {
        self.start.node_id == target.node_id && target.coordinate <= self.start.coordinate
    }

    /// Whether a route from `source` leaves the end node at or after the edit's end.
    fn returns_to_end(&self, source: BlockKey) -> bool {
        self.end.node_id == source.node_id && source.coordinate >= self.end.coordinate
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use gen_core::{
        HashId, NO_CHROMOSOME_INDEX, PATH_END_NODE_ID, PATH_START_NODE_ID,
        PRESERVE_EDIT_SITE_CHROMOSOME_INDEX, PathBlock, Strand,
    };

    use crate::{
        block_group::{BlockGroup, BlockGroupChange},
        block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
        db::GraphConnection,
        edge::Edge,
        node::Node,
        path::Path,
        region::ResolvedGenRegion,
        sequence::Sequence,
        test_helpers::{get_connection, setup_block_group, test_workspace},
    };

    type EdgeKey = (HashId, i64, HashId, i64);

    // The test block group is `A`, `T`, `C` and `G` (10 bases each) in a row, and each edit is
    // made against that path. Tests list the edges the edits add, leaving out markers, and count
    // the sequences the block group spells.

    fn node(letter: &str) -> HashId {
        HashId::convert_str(&format!("test-{letter}-node"))
    }

    /// Edits `path` from `start` to `end` with `bases`, a deletion when empty, and returns the
    /// id of the allele's node.
    fn edit(
        conn: &GraphConnection,
        block_group_id: &HashId,
        path: &Path,
        (start, end, bases): (i64, i64, &str),
        preserve_edge: bool,
    ) -> HashId {
        let (node_id, length) = if bases.is_empty() {
            (HashId::convert_str(""), 0)
        } else {
            let sequence = Sequence::new()
                .sequence_type("DNA")
                .sequence(bases)
                .save(conn)
                .unwrap();
            let name = HashId::convert_str(&format!("edit-{bases}-{start}"));
            (
                Node::create(conn, &sequence.hash, &name).unwrap(),
                sequence.length,
            )
        };
        let change = BlockGroupChange {
            region: ResolvedGenRegion::from_path(conn, *block_group_id, path, start, end).unwrap(),
            path_accession: None,
            block: PathBlock {
                node_id,
                block_sequence: bases.to_string(),
                sequence_start: 0,
                sequence_end: length,
                path_start: start,
                path_end: end,
                strand: Strand::Forward,
            },
            chromosome_index: NO_CHROMOSOME_INDEX,
            phased: 0,
            preserve_edge,
        };
        BlockGroup::insert_change(conn, test_workspace(), &change).unwrap();
        node_id
    }

    fn edge_ids(conn: &GraphConnection, block_group_id: &HashId) -> HashSet<HashId> {
        BlockGroupEdge::edges_for_block_group(conn, block_group_id, None)
            .iter()
            .map(|augmented_edge| augmented_edge.edge.id)
            .collect()
    }

    /// The edges stored since `before`, leaving out markers.
    fn written(
        conn: &GraphConnection,
        block_group_id: &HashId,
        before: &HashSet<HashId>,
    ) -> HashSet<EdgeKey> {
        BlockGroupEdge::edges_for_block_group(conn, block_group_id, None)
            .iter()
            .filter(|augmented_edge| !before.contains(&augmented_edge.edge.id))
            .map(|augmented_edge| {
                let edge = &augmented_edge.edge;
                (
                    edge.source_node_id,
                    edge.source_coordinate,
                    edge.target_node_id,
                    edge.target_coordinate,
                )
            })
            .filter(|(source, source_coordinate, target, target_coordinate)| {
                (source, source_coordinate) != (target, target_coordinate)
            })
            .collect()
    }

    fn sequence_count(conn: &GraphConnection, block_group_id: &HashId) -> usize {
        BlockGroup::get_all_sequences(conn, test_workspace(), block_group_id, true)
            .unwrap()
            .len()
    }

    /// A deletion adds one edge, from where the node before it ends into the node it cuts.
    #[test]
    fn test_deletion_adds_one_edge() {
        let conn = &get_connection(None).unwrap();
        let (block_group_id, path) = setup_block_group(conn);
        let before = edge_ids(conn, &block_group_id);
        edit(conn, &block_group_id, &path, (10, 12, ""), true);
        assert_eq!(sequence_count(conn, &block_group_id), 2);
        assert_eq!(
            written(conn, &block_group_id, &before),
            HashSet::from([(node("a"), 10, node("t"), 2)])
        );
    }

    /// Two deletions that touch each add their own edge, and the route through both is an edge
    /// of its own: four sequences.
    #[test]
    fn test_touching_deletions_add_each_deletion_and_their_combination() {
        let conn = &get_connection(None).unwrap();
        let (block_group_id, path) = setup_block_group(conn);
        let before = edge_ids(conn, &block_group_id);
        for range in [(8, 12, ""), (12, 16, "")] {
            edit(conn, &block_group_id, &path, range, true);
        }
        let (a, t) = (node("a"), node("t"));
        assert_eq!(sequence_count(conn, &block_group_id), 4);
        assert_eq!(
            written(conn, &block_group_id, &before),
            HashSet::from([(a, 8, t, 2), (t, 2, t, 6), (a, 8, t, 6)])
        );
    }

    /// A deletion followed by an insertion or substitution where it ends keeps both edits on
    /// their own and adds the route through both: four sequences.
    #[test]
    fn test_deletion_and_touching_edit_add_each_edit_and_their_combination() {
        for (start, end, bases) in [(12, 12, "GG"), (12, 13, "A")] {
            let conn = &get_connection(None).unwrap();
            let (block_group_id, path) = setup_block_group(conn);
            let before = edge_ids(conn, &block_group_id);
            edit(conn, &block_group_id, &path, (8, 12, ""), true);
            let edit_node_id = edit(conn, &block_group_id, &path, (start, end, bases), true);
            let (a, t) = (node("a"), node("t"));
            assert_eq!(sequence_count(conn, &block_group_id), 4, "{bases}");
            assert_eq!(
                written(conn, &block_group_id, &before),
                HashSet::from([
                    (a, 8, t, 2),
                    (a, 8, edit_node_id, 0),
                    (t, 2, edit_node_id, 0),
                    (edit_node_id, bases.len() as i64, t, end - 10),
                ]),
                "{bases}"
            );
        }
    }

    /// Two deletions that do not touch add only their own edges, and no route through both.
    #[test]
    fn test_non_touching_deletions_add_no_combination() {
        let conn = &get_connection(None).unwrap();
        let (block_group_id, path) = setup_block_group(conn);
        let before = edge_ids(conn, &block_group_id);
        for range in [(10, 12, ""), (20, 22, "")] {
            edit(conn, &block_group_id, &path, range, true);
        }
        let (a, t, c) = (node("a"), node("t"), node("c"));
        assert_eq!(sequence_count(conn, &block_group_id), 4);
        assert_eq!(
            written(conn, &block_group_id, &before),
            HashSet::from([(a, 10, t, 2), (t, 10, c, 2)])
        );
    }

    /// The same deletion made twice adds its edge once.
    #[test]
    fn test_repeated_deletion_adds_its_edge_once() {
        let conn = &get_connection(None).unwrap();
        let (block_group_id, path) = setup_block_group(conn);
        let before = edge_ids(conn, &block_group_id);
        for _ in 0..2 {
            edit(conn, &block_group_id, &path, (10, 12, ""), true);
        }
        assert_eq!(sequence_count(conn, &block_group_id), 2);
        assert_eq!(
            written(conn, &block_group_id, &before),
            HashSet::from([(node("a"), 10, node("t"), 2)])
        );
    }

    /// A homozygous edit retires the edge into the replaced bases and adds only its own
    /// allele, so the block group spells one sequence.
    #[test]
    fn test_homozygous_edit_retires_the_replaced_edge() {
        let conn = &get_connection(None).unwrap();
        let (block_group_id, path) = setup_block_group(conn);
        let before = edge_ids(conn, &block_group_id);
        let allele = edit(conn, &block_group_id, &path, (10, 12, "NNNN"), false);
        let (a, t) = (node("a"), node("t"));
        assert_eq!(sequence_count(conn, &block_group_id), 1);
        assert_eq!(
            written(conn, &block_group_id, &before),
            HashSet::from([(a, 10, allele, 0), (allele, 4, t, 2)])
        );
        assert!(
            BlockGroupEdge::edges_for_block_group(conn, &block_group_id, None)
                .iter()
                .filter(|edge| edge.chromosome_index == PRESERVE_EDIT_SITE_CHROMOSOME_INDEX)
                .any(|edge| (
                    edge.edge.source_node_id,
                    edge.edge.source_coordinate,
                    edge.edge.target_node_id,
                    edge.edge.target_coordinate
                ) == (a, 10, t, 0))
        );
    }

    /// A deletion where three routes arrive at a node adds an edge from each of them.
    #[test]
    fn test_deletion_adds_an_edge_from_every_route_arriving_at_the_node() {
        let conn = &get_connection(None).unwrap();
        let (block_group_id, path) = setup_block_group(conn);
        let (a, t) = (node("a"), node("t"));
        let mut alternatives = vec![];
        for (label, bases) in [
            ("first-alternative", "GGGG"),
            ("second-alternative", "CCCC"),
        ] {
            let sequence = Sequence::new()
                .sequence_type("DNA")
                .sequence(bases)
                .save(conn)
                .unwrap();
            let node_id = Node::create(conn, &sequence.hash, &HashId::convert_str(label)).unwrap();
            let into = Edge::create(conn, a, 10, Strand::Forward, node_id, 0, Strand::Forward);
            let out = Edge::create(conn, node_id, 4, Strand::Forward, t, 0, Strand::Forward);
            BlockGroupEdge::bulk_create(
                conn,
                &[into.unwrap().id, out.unwrap().id].map(|edge_id| BlockGroupEdgeData {
                    block_group_id,
                    edge_id,
                    chromosome_index: NO_CHROMOSOME_INDEX,
                    phased: 0,
                }),
            );
            alternatives.push(node_id);
        }
        let before = edge_ids(conn, &block_group_id);
        edit(conn, &block_group_id, &path, (10, 12, ""), true);
        assert_eq!(sequence_count(conn, &block_group_id), 6);
        assert_eq!(
            written(conn, &block_group_id, &before),
            HashSet::from([
                (a, 10, t, 2),
                (alternatives[0], 4, t, 2),
                (alternatives[1], 4, t, 2),
            ])
        );
    }

    /// A deletion at either end of the contig attaches to the start or end marker node.
    #[test]
    fn test_deletion_at_a_contig_end_attaches_to_the_marker_node() {
        for (start, end, expected) in [
            (0, 2, (PATH_START_NODE_ID, 0, node("a"), 2)),
            (38, 40, (node("g"), 8, PATH_END_NODE_ID, 0)),
        ] {
            let conn = &get_connection(None).unwrap();
            let (block_group_id, path) = setup_block_group(conn);
            let before = edge_ids(conn, &block_group_id);
            edit(conn, &block_group_id, &path, (start, end, ""), true);
            assert_eq!(sequence_count(conn, &block_group_id), 2, "{start}-{end}");
            assert_eq!(
                written(conn, &block_group_id, &before),
                HashSet::from([expected]),
                "{start}-{end}"
            );
        }
    }

    /// An insertion at either end of the contig sits between the start or end marker node and
    /// the node beside it.
    #[test]
    fn test_insertion_at_a_contig_end_sits_next_to_the_marker_node() {
        for (position, before_edge, after_edge) in [
            (0, (PATH_START_NODE_ID, 0), (node("a"), 0)),
            (40, (node("g"), 10), (PATH_END_NODE_ID, 0)),
        ] {
            let conn = &get_connection(None).unwrap();
            let (block_group_id, path) = setup_block_group(conn);
            let before = edge_ids(conn, &block_group_id);
            let allele = edit(
                conn,
                &block_group_id,
                &path,
                (position, position, "GG"),
                true,
            );
            assert_eq!(sequence_count(conn, &block_group_id), 2, "{position}");
            assert_eq!(
                written(conn, &block_group_id, &before),
                HashSet::from([
                    (before_edge.0, before_edge.1, allele, 0),
                    (allele, 2, after_edge.0, after_edge.1),
                ]),
                "{position}"
            );
        }
    }
}
