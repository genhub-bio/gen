//! Plans the edges an edit writes from the ports it spans.
//!
//! A port is a point on a node between two of its bases, in front of its first base, or behind its
//! last: `(node_id, coordinate)`. An edit replaces the sequence between a left port and a right
//! port with an allele, which is a new node or, for a deletion, nothing. A deletion is one variant
//! edge from its left port to its right port; an allele is an edge into it and an edge out of it.
//! Marker edges split the node at ports inside it, so the unedited route stays in the graph with
//! a chromosome index of its own.
//!
//! Every combination of adjacent edits is a stored edge of its own, so a later edit, or another
//! sample, can give it its own chromosome index or retire it. Where a route already arrives at the
//! edit's left port (the end of an allele, a deletion ending there, the end of the node before),
//! the edit also starts with an edge from that route's source; where routes already leave its
//! right port, it also ends with an edge to each of their targets. At a node's first base or
//! behind its last there is no sequence to attach to, so only those edges are written: deleting a
//! whole node joins every source arriving at its first base to every target leaving its last.
//!
//! Two adjacent deletions are no exception: deleting `(1,2)` and then `(2,3)` also writes `(1,3)`,
//! the route through both. That edge is the same as a deletion of `[1,3)` in one step, so the
//! graph does not tell the two apart, and a run of k touching deletions stores an edge for each of
//! its k(k+1)/2 contiguous runs.
//!
//! `update sequence` names no side of an insertion made where another allele is already
//! inserted, so the new insertion goes in front of the earlier one: it leads into it like into
//! every other route leaving the port, but does not follow it, which would close a cycle.
//!
//! An edit that does not keep the unedited route (`preserve_edge` false, a homozygous call)
//! retires the stored edges it starts from or ends at in this way, by adding them to the block
//! group again with `PRESERVE_EDIT_SITE_CHROMOSOME_INDEX`: they led past the edit's locus without
//! its allele.

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

/// The ports an edit replaces the sequence between: every port its first base may follow and
/// every port its last base may precede. An insertion has the same ports on both sides.
#[derive(Clone, Debug, Default)]
pub struct EditSpan {
    pub starts: Vec<BlockKey>,
    pub ends: Vec<BlockKey>,
    /// Whether the span is a coordinate on a linear path rather than a named feature. An
    /// insertion at a path coordinate belongs to the point between the path's blocks, so it
    /// leads into every route leaving that point even when the path enters the next block
    /// through an allele.
    pub along_path: bool,
}

/// A port an edit's first edge leaves from or its last edge leads to, and the stored edge whose
/// route it continues, if any.
#[derive(Clone, Copy, Debug)]
struct Endpoint {
    port: BlockKey,
    edge: Option<AugmentedEdgeData>,
}

/// The edges of one block group, stored and planned, indexed by the ports they meet.
///
/// `BlockGroup::insert_changes` keeps one per block group while it plans a batch of changes, and
/// adds each change's edges before planning the next, so edits that meet within one VCF or one
/// library combine as they would have if applied one at a time. A node's stored edges are loaded
/// the first time an edit touches that node.
pub struct EdgeLookup {
    block_group_id: HashId,
    loaded_node_ids: HashSet<HashId>,
    known: HashSet<AugmentedEdgeData>,
    retired: HashSet<EdgeData>,
    arriving: HashMap<BlockKey, IndexSet<AugmentedEdgeData>>,
    leaving: HashMap<BlockKey, IndexSet<AugmentedEdgeData>>,
    first_arrival: HashMap<HashId, i64>,
    last_departure: HashMap<HashId, i64>,
    entries_by_node_id: HashMap<HashId, IndexSet<BlockKey>>,
}

impl EdgeLookup {
    pub fn new(block_group_id: HashId) -> Self {
        EdgeLookup {
            block_group_id,
            loaded_node_ids: HashSet::new(),
            known: HashSet::new(),
            retired: HashSet::new(),
            arriving: HashMap::new(),
            leaving: HashMap::new(),
            first_arrival: HashMap::new(),
            last_departure: HashMap::new(),
            entries_by_node_id: HashMap::new(),
        }
    }

    /// Records edges planned but not stored yet, so later edits in the batch see them.
    pub fn add(&mut self, edges: &[AugmentedEdgeData]) {
        for edge in edges {
            self.index(*edge);
        }
    }

    fn index(&mut self, augmented_edge_data: AugmentedEdgeData) {
        if !self.known.insert(augmented_edge_data) {
            return;
        }
        let edge = augmented_edge_data.edge_data;
        if augmented_edge_data.chromosome_index == PRESERVE_EDIT_SITE_CHROMOSOME_INDEX
            && !augmented_edge_data.edge_data.is_marker()
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
        if edge.source_node_id != edge.target_node_id {
            self.entries_by_node_id
                .entry(edge.target_node_id)
                .or_default()
                .insert(BlockKey::new(edge.source_node_id, edge.source_coordinate));
        }
    }

    fn load(&mut self, conn: &GraphConnection, node_id: HashId) -> Result<(), EdgeError> {
        if !self.loaded_node_ids.insert(node_id) {
            return Ok(());
        }
        for augmented_edge in
            Edge::edges_for_block_group_nodes(conn, &self.block_group_id, &[node_id], None)?
        {
            self.index(AugmentedEdgeData::from(&augmented_edge));
        }
        Ok(())
    }

    /// The live edges arriving at `port` other than markers: retired edges lead nowhere.
    fn arriving(&self, port: BlockKey) -> Vec<AugmentedEdgeData> {
        self.arriving
            .get(&port)
            .into_iter()
            .flatten()
            .filter(|edge| !edge.edge_data.is_marker() && !self.retired.contains(&edge.edge_data))
            .copied()
            .collect()
    }

    /// The live edges leaving `port` other than markers.
    fn leaving(&self, port: BlockKey) -> Vec<AugmentedEdgeData> {
        self.leaving
            .get(&port)
            .into_iter()
            .flatten()
            .filter(|edge| !edge.edge_data.is_marker() && !self.retired.contains(&edge.edge_data))
            .copied()
            .collect()
    }

    /// Whether some route reaches `port` along its own node, so the port can start an edge.
    fn has_sequence_before(&self, port: BlockKey) -> bool {
        if port.node_id == PATH_START_NODE_ID {
            return true;
        }
        if port.node_id == PATH_END_NODE_ID {
            return false;
        }
        self.first_arrival
            .get(&port.node_id)
            .is_some_and(|first| *first < port.coordinate)
    }

    /// Whether some route continues from `port` along its own node, so an edge can end there.
    fn has_sequence_after(&self, port: BlockKey) -> bool {
        if port.node_id == PATH_END_NODE_ID {
            return true;
        }
        if port.node_id == PATH_START_NODE_ID {
            return false;
        }
        self.last_departure
            .get(&port.node_id)
            .is_some_and(|last| *last > port.coordinate)
    }

    /// Whether every marker at `port` is retired: an earlier edit replaced the node's sequence
    /// on one side of the port for every chromosome copy, so the node's own sequence no longer
    /// runs through it.
    fn is_split_retired(&self, port: BlockKey) -> bool {
        let mut markers = self
            .leaving
            .get(&port)
            .into_iter()
            .flatten()
            .filter(|edge| edge.edge_data.is_marker())
            .peekable();
        markers.peek().is_some()
            && markers.all(|marker| marker.chromosome_index == PRESERVE_EDIT_SITE_CHROMOSOME_INDEX)
    }

    /// Loads the nodes at the far end of every edge meeting `port`, so their entries are known.
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

    /// The alleles already inserted at the point an insertion at `port` goes. `update sequence`
    /// names no side of them to insert on, so the new insertion goes in front of them: it leads
    /// into each of them like into every other route leaving the port, but does not follow them,
    /// which would close a cycle. Inside a node they are entered from the port itself; at a
    /// node's first base, from the end of a node leading straight into the port.
    fn siblings(&self, port: BlockKey) -> HashSet<HashId> {
        let entry_ports = if self.has_sequence_before(port) {
            vec![port]
        } else {
            self.arriving(port)
                .iter()
                .filter(|edge| edge.edge_data.source_node_id != port.node_id)
                .map(|edge| {
                    BlockKey::new(
                        edge.edge_data.source_node_id,
                        edge.edge_data.source_coordinate,
                    )
                })
                .collect()
        };
        self.arriving(port)
            .iter()
            .map(|edge| edge.edge_data.source_node_id)
            .filter(|node_id| {
                *node_id != port.node_id
                    && self.entries_by_node_id.get(node_id).is_some_and(|entries| {
                        entries.iter().any(|entry| entry_ports.contains(entry))
                    })
            })
            .collect()
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
            if !rules.allows(source.node_id)
                || rules.siblings.contains(&source.node_id)
                || rules.returns_to_end(source)
            {
                continue;
            }
            sources.push(Endpoint {
                port: source,
                edge: Some(arriving),
            });
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
            if !rules.allows(target.node_id) || rules.returns_to_start(target) {
                continue;
            }
            targets.push(Endpoint {
                port: target,
                edge: Some(leaving),
            });
        }
        // Where an earlier edit replaced the node's sequence after the port for every chromosome
        // copy, the edit continues into that edit's allele or deletion instead.
        let leads_elsewhere = !targets.is_empty();
        if self.has_sequence_after(port) && !(leads_elsewhere && self.is_split_retired(port)) {
            targets.insert(0, Endpoint { port, edge: None });
        }
        targets
    }

    /// Plans the edges `change` writes to replace the sequence in `span` with its block, and
    /// records them for the edits planned after it.
    pub fn plan(
        &mut self,
        conn: &GraphConnection,
        span: &EditSpan,
        change: &BlockGroupChange,
    ) -> Result<Vec<AugmentedEdgeData>, EdgeError> {
        for port in span.starts.iter().chain(&span.ends) {
            self.load(conn, port.node_id)?;
        }
        let starts = span.starts.iter().copied().collect::<IndexSet<_>>();
        let ends = span.ends.iter().copied().collect::<IndexSet<_>>();
        let is_deletion = change.block.sequence_start == change.block.sequence_end;
        let is_insertion = starts == ends;
        if is_deletion && is_insertion {
            // Deleting nothing changes nothing.
            return Ok(vec![]);
        }
        let mut siblings = HashSet::new();
        for port in starts.iter().chain(&ends) {
            self.load_neighbors(conn, *port)?;
            if is_insertion {
                siblings.extend(self.siblings(*port));
            }
        }
        let rules = EditRules {
            change,
            allele: (!is_deletion).then_some(change.block.node_id),
            siblings,
            starts: &starts,
            ends: &ends,
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
        for port in starts.iter().chain(&ends) {
            if !is_terminal(port.node_id)
                && self.has_sequence_before(*port)
                && self.has_sequence_after(*port)
            {
                new_edges.insert(AugmentedEdgeData {
                    chromosome_index: rules.marker_chromosome_index(),
                    phased: 0,
                    ..variant(*port, *port)
                });
            }
        }

        let sources = starts
            .iter()
            .flat_map(|start| self.sources(*start, &rules))
            .collect::<Vec<_>>();
        let mut targets = ends
            .iter()
            .flat_map(|end| self.targets(*end, &rules))
            .collect::<Vec<_>>();
        if is_insertion && span.along_path {
            // The path enters the block after the insertion through an allele, so the insertion
            // sits at the port that allele was entered from and leads into every route leaving
            // that port, not only the allele it was addressed through.
            for start in starts
                .iter()
                .filter(|start| !self.has_sequence_before(**start))
            {
                for source in self.sources(*start, &rules) {
                    for target in self.targets(source.port, &rules) {
                        if !targets.iter().any(|known| known.port == target.port) {
                            targets.push(target);
                        }
                    }
                }
            }
        }
        let mut continued = vec![];
        match rules.allele {
            None => {
                for source in &sources {
                    for target in &targets {
                        if !is_combined_deletion(source, target) {
                            continue;
                        }
                        new_edges.insert(variant(source.port, target.port));
                        continued.extend(source.edge);
                        continued.extend(target.edge);
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
        self.add(&new_edges);
        Ok(new_edges)
    }
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
    /// For an insertion, the alternatives already inserted at its point, which it goes in front
    /// of.
    siblings: HashSet<HashId>,
    starts: &'a IndexSet<BlockKey>,
    ends: &'a IndexSet<BlockKey>,
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
    /// applied again. Chromosome index and phase play no part, since two inputs cannot be assumed
    /// to give the same index the same haplotype, and library fan-out is unphased.
    fn allows(&self, node_id: HashId) -> bool {
        Some(node_id) != self.allele
    }

    /// Whether a route to `target` re-enters a node the edit starts on at or before its start,
    /// closing a cycle through the edit. An edit chained onto the end of an earlier insertion
    /// ends where that insertion returns to, and leaves from.
    fn returns_to_start(&self, target: BlockKey) -> bool {
        self.starts
            .iter()
            .any(|start| start.node_id == target.node_id && target.coordinate <= start.coordinate)
    }

    /// Whether a route from `source` leaves a node the edit ends on at or after its end, the
    /// mirror of [`EditRules::returns_to_start`].
    fn returns_to_end(&self, source: BlockKey) -> bool {
        self.ends
            .iter()
            .any(|end| end.node_id == source.node_id && source.coordinate >= end.coordinate)
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
        edge::Edge,
        node::Node,
        region::ResolvedGenRegion,
        sequence::Sequence,
        test_helpers::{get_connection, setup_block_group, test_workspace},
    };

    // The test block group is `A` (10 bases), `T` (10), `C` (10) and `G` (10) in a row, and each
    // edit is made against that path. Every test lists the edges the edits add, leaving out the
    // zero-width edit-site markers, and counts the sequences the block group spells.

    /// A deletion adds one edge, from where the node before it ends into the node it cuts.
    #[test]
    fn test_deletion_adds_one_edge() {
        let conn = &get_connection(None).unwrap();
        let (block_group_id, path) = setup_block_group(conn);
        let a_node_id = HashId::convert_str("test-a-node");
        let t_node_id = HashId::convert_str("test-t-node");
        let before = BlockGroupEdge::edges_for_block_group(conn, &block_group_id, None)
            .iter()
            .map(|augmented_edge| augmented_edge.edge.id)
            .collect::<HashSet<_>>();
        let mut edit_node_ids = vec![];
        let (start, end, bases) = (10, 12, "");
        let (node_id, length) = if bases.is_empty() {
            (HashId::convert_str(""), 0)
        } else {
            let sequence = Sequence::new()
                .sequence_type("DNA")
                .sequence(bases)
                .save(conn)
                .unwrap();
            let node_id = Node::create(
                conn,
                &sequence.hash,
                &HashId::convert_str(&format!("edit-{bases}-{start}")),
            )
            .unwrap();
            (node_id, sequence.length)
        };
        edit_node_ids.push(node_id);
        let block = PathBlock {
            node_id,
            block_sequence: bases.to_string(),
            sequence_start: 0,
            sequence_end: length,
            path_start: start,
            path_end: end,
            strand: Strand::Forward,
        };
        let region = ResolvedGenRegion::from_path(conn, block_group_id, &path, start, end).unwrap();
        let change = BlockGroupChange {
            region,
            path_accession: None,
            block,
            chromosome_index: NO_CHROMOSOME_INDEX,
            phased: 0,
            preserve_edge: true,
        };
        BlockGroup::insert_change(conn, test_workspace(), &change).unwrap();
        let written = BlockGroupEdge::edges_for_block_group(conn, &block_group_id, None)
            .iter()
            .filter(|augmented_edge| !before.contains(&augmented_edge.edge.id))
            .map(|augmented_edge| {
                (
                    augmented_edge.edge.source_node_id,
                    augmented_edge.edge.source_coordinate,
                    augmented_edge.edge.target_node_id,
                    augmented_edge.edge.target_coordinate,
                )
            })
            .filter(
                |(source_node_id, source_coordinate, target_node_id, target_coordinate)| {
                    (source_node_id, source_coordinate) != (target_node_id, target_coordinate)
                },
            )
            .collect::<HashSet<_>>();
        let sequences =
            BlockGroup::get_all_sequences(conn, test_workspace(), &block_group_id, true).unwrap();
        assert_eq!(sequences.len(), 2);
        assert_eq!(written, HashSet::from([(a_node_id, 10, t_node_id, 2)]));
    }

    /// Two deletions that touch each add their own edge, and the route through both is an edge
    /// of its own: four sequences.
    #[test]
    fn test_touching_deletions_add_each_deletion_and_their_combination() {
        let conn = &get_connection(None).unwrap();
        let (block_group_id, path) = setup_block_group(conn);
        let a_node_id = HashId::convert_str("test-a-node");
        let t_node_id = HashId::convert_str("test-t-node");
        let before = BlockGroupEdge::edges_for_block_group(conn, &block_group_id, None)
            .iter()
            .map(|augmented_edge| augmented_edge.edge.id)
            .collect::<HashSet<_>>();
        let mut edit_node_ids = vec![];
        for (start, end, bases) in [(8, 12, ""), (12, 16, "")] {
            let (node_id, length) = if bases.is_empty() {
                (HashId::convert_str(""), 0)
            } else {
                let sequence = Sequence::new()
                    .sequence_type("DNA")
                    .sequence(bases)
                    .save(conn)
                    .unwrap();
                let node_id = Node::create(
                    conn,
                    &sequence.hash,
                    &HashId::convert_str(&format!("edit-{bases}-{start}")),
                )
                .unwrap();
                (node_id, sequence.length)
            };
            edit_node_ids.push(node_id);
            let block = PathBlock {
                node_id,
                block_sequence: bases.to_string(),
                sequence_start: 0,
                sequence_end: length,
                path_start: start,
                path_end: end,
                strand: Strand::Forward,
            };
            let region =
                ResolvedGenRegion::from_path(conn, block_group_id, &path, start, end).unwrap();
            let change = BlockGroupChange {
                region,
                path_accession: None,
                block,
                chromosome_index: NO_CHROMOSOME_INDEX,
                phased: 0,
                preserve_edge: true,
            };
            BlockGroup::insert_change(conn, test_workspace(), &change).unwrap();
        }
        let written = BlockGroupEdge::edges_for_block_group(conn, &block_group_id, None)
            .iter()
            .filter(|augmented_edge| !before.contains(&augmented_edge.edge.id))
            .map(|augmented_edge| {
                (
                    augmented_edge.edge.source_node_id,
                    augmented_edge.edge.source_coordinate,
                    augmented_edge.edge.target_node_id,
                    augmented_edge.edge.target_coordinate,
                )
            })
            .filter(
                |(source_node_id, source_coordinate, target_node_id, target_coordinate)| {
                    (source_node_id, source_coordinate) != (target_node_id, target_coordinate)
                },
            )
            .collect::<HashSet<_>>();
        let sequences =
            BlockGroup::get_all_sequences(conn, test_workspace(), &block_group_id, true).unwrap();
        assert_eq!(sequences.len(), 4);
        assert_eq!(
            written,
            HashSet::from([
                (a_node_id, 8, t_node_id, 2),
                (t_node_id, 2, t_node_id, 6),
                (a_node_id, 8, t_node_id, 6),
            ])
        );
    }

    /// A deletion followed by an insertion or substitution where it ends keeps both edits on
    /// their own and adds the route through both: four sequences.
    #[test]
    fn test_deletion_and_touching_edit_add_each_edit_and_their_combination() {
        for (start, end, bases) in [(12, 12, "GG"), (12, 13, "A")] {
            let conn = &get_connection(None).unwrap();
            let (block_group_id, path) = setup_block_group(conn);
            let a_node_id = HashId::convert_str("test-a-node");
            let t_node_id = HashId::convert_str("test-t-node");
            let before = BlockGroupEdge::edges_for_block_group(conn, &block_group_id, None)
                .iter()
                .map(|augmented_edge| augmented_edge.edge.id)
                .collect::<HashSet<_>>();
            let mut edit_node_ids = vec![];
            for (start, end, bases) in [(8, 12, ""), (start, end, bases)] {
                let (node_id, length) = if bases.is_empty() {
                    (HashId::convert_str(""), 0)
                } else {
                    let sequence = Sequence::new()
                        .sequence_type("DNA")
                        .sequence(bases)
                        .save(conn)
                        .unwrap();
                    let node_id = Node::create(
                        conn,
                        &sequence.hash,
                        &HashId::convert_str(&format!("edit-{bases}-{start}")),
                    )
                    .unwrap();
                    (node_id, sequence.length)
                };
                edit_node_ids.push(node_id);
                let block = PathBlock {
                    node_id,
                    block_sequence: bases.to_string(),
                    sequence_start: 0,
                    sequence_end: length,
                    path_start: start,
                    path_end: end,
                    strand: Strand::Forward,
                };
                let region =
                    ResolvedGenRegion::from_path(conn, block_group_id, &path, start, end).unwrap();
                let change = BlockGroupChange {
                    region,
                    path_accession: None,
                    block,
                    chromosome_index: NO_CHROMOSOME_INDEX,
                    phased: 0,
                    preserve_edge: true,
                };
                BlockGroup::insert_change(conn, test_workspace(), &change).unwrap();
            }
            let written = BlockGroupEdge::edges_for_block_group(conn, &block_group_id, None)
                .iter()
                .filter(|augmented_edge| !before.contains(&augmented_edge.edge.id))
                .map(|augmented_edge| {
                    (
                        augmented_edge.edge.source_node_id,
                        augmented_edge.edge.source_coordinate,
                        augmented_edge.edge.target_node_id,
                        augmented_edge.edge.target_coordinate,
                    )
                })
                .filter(
                    |(source_node_id, source_coordinate, target_node_id, target_coordinate)| {
                        (source_node_id, source_coordinate) != (target_node_id, target_coordinate)
                    },
                )
                .collect::<HashSet<_>>();
            let sequences =
                BlockGroup::get_all_sequences(conn, test_workspace(), &block_group_id, true)
                    .unwrap();
            assert_eq!(sequences.len(), 4, "{bases}");
            let edit_node_id = edit_node_ids[1];
            assert_eq!(
                written,
                HashSet::from([
                    (a_node_id, 8, t_node_id, 2),
                    (a_node_id, 8, edit_node_id, 0),
                    (t_node_id, 2, edit_node_id, 0),
                    (edit_node_id, bases.len() as i64, t_node_id, end - 10),
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
        let a_node_id = HashId::convert_str("test-a-node");
        let t_node_id = HashId::convert_str("test-t-node");
        let before = BlockGroupEdge::edges_for_block_group(conn, &block_group_id, None)
            .iter()
            .map(|augmented_edge| augmented_edge.edge.id)
            .collect::<HashSet<_>>();
        let mut edit_node_ids = vec![];
        for (start, end, bases) in [(10, 12, ""), (20, 22, "")] {
            let (node_id, length) = if bases.is_empty() {
                (HashId::convert_str(""), 0)
            } else {
                let sequence = Sequence::new()
                    .sequence_type("DNA")
                    .sequence(bases)
                    .save(conn)
                    .unwrap();
                let node_id = Node::create(
                    conn,
                    &sequence.hash,
                    &HashId::convert_str(&format!("edit-{bases}-{start}")),
                )
                .unwrap();
                (node_id, sequence.length)
            };
            edit_node_ids.push(node_id);
            let block = PathBlock {
                node_id,
                block_sequence: bases.to_string(),
                sequence_start: 0,
                sequence_end: length,
                path_start: start,
                path_end: end,
                strand: Strand::Forward,
            };
            let region =
                ResolvedGenRegion::from_path(conn, block_group_id, &path, start, end).unwrap();
            let change = BlockGroupChange {
                region,
                path_accession: None,
                block,
                chromosome_index: NO_CHROMOSOME_INDEX,
                phased: 0,
                preserve_edge: true,
            };
            BlockGroup::insert_change(conn, test_workspace(), &change).unwrap();
        }
        let written = BlockGroupEdge::edges_for_block_group(conn, &block_group_id, None)
            .iter()
            .filter(|augmented_edge| !before.contains(&augmented_edge.edge.id))
            .map(|augmented_edge| {
                (
                    augmented_edge.edge.source_node_id,
                    augmented_edge.edge.source_coordinate,
                    augmented_edge.edge.target_node_id,
                    augmented_edge.edge.target_coordinate,
                )
            })
            .filter(
                |(source_node_id, source_coordinate, target_node_id, target_coordinate)| {
                    (source_node_id, source_coordinate) != (target_node_id, target_coordinate)
                },
            )
            .collect::<HashSet<_>>();
        let c_node_id = HashId::convert_str("test-c-node");
        let sequences =
            BlockGroup::get_all_sequences(conn, test_workspace(), &block_group_id, true).unwrap();
        assert_eq!(sequences.len(), 4);
        assert_eq!(
            written,
            HashSet::from([(a_node_id, 10, t_node_id, 2), (t_node_id, 10, c_node_id, 2)])
        );
    }

    /// The same deletion made twice adds its edge once.
    #[test]
    fn test_repeated_deletion_adds_its_edge_once() {
        let conn = &get_connection(None).unwrap();
        let (block_group_id, path) = setup_block_group(conn);
        let a_node_id = HashId::convert_str("test-a-node");
        let t_node_id = HashId::convert_str("test-t-node");
        let before = BlockGroupEdge::edges_for_block_group(conn, &block_group_id, None)
            .iter()
            .map(|augmented_edge| augmented_edge.edge.id)
            .collect::<HashSet<_>>();
        let mut edit_node_ids = vec![];
        for (start, end, bases) in [(10, 12, ""), (10, 12, "")] {
            let (node_id, length) = if bases.is_empty() {
                (HashId::convert_str(""), 0)
            } else {
                let sequence = Sequence::new()
                    .sequence_type("DNA")
                    .sequence(bases)
                    .save(conn)
                    .unwrap();
                let node_id = Node::create(
                    conn,
                    &sequence.hash,
                    &HashId::convert_str(&format!("edit-{bases}-{start}")),
                )
                .unwrap();
                (node_id, sequence.length)
            };
            edit_node_ids.push(node_id);
            let block = PathBlock {
                node_id,
                block_sequence: bases.to_string(),
                sequence_start: 0,
                sequence_end: length,
                path_start: start,
                path_end: end,
                strand: Strand::Forward,
            };
            let region =
                ResolvedGenRegion::from_path(conn, block_group_id, &path, start, end).unwrap();
            let change = BlockGroupChange {
                region,
                path_accession: None,
                block,
                chromosome_index: NO_CHROMOSOME_INDEX,
                phased: 0,
                preserve_edge: true,
            };
            BlockGroup::insert_change(conn, test_workspace(), &change).unwrap();
        }
        let written = BlockGroupEdge::edges_for_block_group(conn, &block_group_id, None)
            .iter()
            .filter(|augmented_edge| !before.contains(&augmented_edge.edge.id))
            .map(|augmented_edge| {
                (
                    augmented_edge.edge.source_node_id,
                    augmented_edge.edge.source_coordinate,
                    augmented_edge.edge.target_node_id,
                    augmented_edge.edge.target_coordinate,
                )
            })
            .filter(
                |(source_node_id, source_coordinate, target_node_id, target_coordinate)| {
                    (source_node_id, source_coordinate) != (target_node_id, target_coordinate)
                },
            )
            .collect::<HashSet<_>>();
        let sequences =
            BlockGroup::get_all_sequences(conn, test_workspace(), &block_group_id, true).unwrap();
        assert_eq!(sequences.len(), 2);
        assert_eq!(written, HashSet::from([(a_node_id, 10, t_node_id, 2)]));
    }

    /// A second insertion at the same point goes in front of the first, leading into it: it does
    /// not follow it, which would close a cycle. Four sequences.
    #[test]
    fn test_second_insertion_at_one_point_goes_in_front_of_the_first() {
        let conn = &get_connection(None).unwrap();
        let (block_group_id, path) = setup_block_group(conn);
        let a_node_id = HashId::convert_str("test-a-node");
        let t_node_id = HashId::convert_str("test-t-node");
        let before = BlockGroupEdge::edges_for_block_group(conn, &block_group_id, None)
            .iter()
            .map(|augmented_edge| augmented_edge.edge.id)
            .collect::<HashSet<_>>();
        let mut edit_node_ids = vec![];
        for (start, end, bases) in [(10, 10, "GG"), (10, 10, "CC")] {
            let (node_id, length) = if bases.is_empty() {
                (HashId::convert_str(""), 0)
            } else {
                let sequence = Sequence::new()
                    .sequence_type("DNA")
                    .sequence(bases)
                    .save(conn)
                    .unwrap();
                let node_id = Node::create(
                    conn,
                    &sequence.hash,
                    &HashId::convert_str(&format!("edit-{bases}-{start}")),
                )
                .unwrap();
                (node_id, sequence.length)
            };
            edit_node_ids.push(node_id);
            let block = PathBlock {
                node_id,
                block_sequence: bases.to_string(),
                sequence_start: 0,
                sequence_end: length,
                path_start: start,
                path_end: end,
                strand: Strand::Forward,
            };
            let region =
                ResolvedGenRegion::from_path(conn, block_group_id, &path, start, end).unwrap();
            let change = BlockGroupChange {
                region,
                path_accession: None,
                block,
                chromosome_index: NO_CHROMOSOME_INDEX,
                phased: 0,
                preserve_edge: true,
            };
            BlockGroup::insert_change(conn, test_workspace(), &change).unwrap();
        }
        let written = BlockGroupEdge::edges_for_block_group(conn, &block_group_id, None)
            .iter()
            .filter(|augmented_edge| !before.contains(&augmented_edge.edge.id))
            .map(|augmented_edge| {
                (
                    augmented_edge.edge.source_node_id,
                    augmented_edge.edge.source_coordinate,
                    augmented_edge.edge.target_node_id,
                    augmented_edge.edge.target_coordinate,
                )
            })
            .filter(
                |(source_node_id, source_coordinate, target_node_id, target_coordinate)| {
                    (source_node_id, source_coordinate) != (target_node_id, target_coordinate)
                },
            )
            .collect::<HashSet<_>>();
        let sequences =
            BlockGroup::get_all_sequences(conn, test_workspace(), &block_group_id, true).unwrap();
        assert_eq!(sequences.len(), 4);
        let first_node_id = edit_node_ids[0];
        let second_node_id = edit_node_ids[1];
        assert_eq!(
            written,
            HashSet::from([
                (a_node_id, 10, first_node_id, 0),
                (first_node_id, 2, t_node_id, 0),
                (a_node_id, 10, second_node_id, 0),
                (second_node_id, 2, t_node_id, 0),
                (second_node_id, 2, first_node_id, 0),
            ])
        );
    }

    /// A homozygous edit retires the edge into the replaced bases and adds only its own
    /// allele, so the block group spells one sequence.
    #[test]
    fn test_homozygous_edit_retires_the_replaced_edge() {
        let conn = &get_connection(None).unwrap();
        let (block_group_id, path) = setup_block_group(conn);
        let a_node_id = HashId::convert_str("test-a-node");
        let t_node_id = HashId::convert_str("test-t-node");
        let before = BlockGroupEdge::edges_for_block_group(conn, &block_group_id, None)
            .iter()
            .map(|augmented_edge| augmented_edge.edge.id)
            .collect::<HashSet<_>>();
        let mut edit_node_ids = vec![];
        let (start, end, bases) = (10, 12, "NNNN");
        let (node_id, length) = if bases.is_empty() {
            (HashId::convert_str(""), 0)
        } else {
            let sequence = Sequence::new()
                .sequence_type("DNA")
                .sequence(bases)
                .save(conn)
                .unwrap();
            let node_id = Node::create(
                conn,
                &sequence.hash,
                &HashId::convert_str(&format!("edit-{bases}-{start}")),
            )
            .unwrap();
            (node_id, sequence.length)
        };
        edit_node_ids.push(node_id);
        let block = PathBlock {
            node_id,
            block_sequence: bases.to_string(),
            sequence_start: 0,
            sequence_end: length,
            path_start: start,
            path_end: end,
            strand: Strand::Forward,
        };
        let region = ResolvedGenRegion::from_path(conn, block_group_id, &path, start, end).unwrap();
        let change = BlockGroupChange {
            region,
            path_accession: None,
            block,
            chromosome_index: NO_CHROMOSOME_INDEX,
            phased: 0,
            preserve_edge: false,
        };
        BlockGroup::insert_change(conn, test_workspace(), &change).unwrap();
        let written = BlockGroupEdge::edges_for_block_group(conn, &block_group_id, None)
            .iter()
            .filter(|augmented_edge| !before.contains(&augmented_edge.edge.id))
            .map(|augmented_edge| {
                (
                    augmented_edge.edge.source_node_id,
                    augmented_edge.edge.source_coordinate,
                    augmented_edge.edge.target_node_id,
                    augmented_edge.edge.target_coordinate,
                )
            })
            .filter(
                |(source_node_id, source_coordinate, target_node_id, target_coordinate)| {
                    (source_node_id, source_coordinate) != (target_node_id, target_coordinate)
                },
            )
            .collect::<HashSet<_>>();
        let sequences =
            BlockGroup::get_all_sequences(conn, test_workspace(), &block_group_id, true).unwrap();
        assert_eq!(sequences.len(), 1);
        let retired = BlockGroupEdge::edges_for_block_group(conn, &block_group_id, None)
            .iter()
            .filter(|augmented_edge| {
                augmented_edge.chromosome_index == PRESERVE_EDIT_SITE_CHROMOSOME_INDEX
            })
            .map(|augmented_edge| {
                (
                    augmented_edge.edge.source_node_id,
                    augmented_edge.edge.source_coordinate,
                    augmented_edge.edge.target_node_id,
                    augmented_edge.edge.target_coordinate,
                )
            })
            .collect::<HashSet<_>>();
        let allele_node_id = edit_node_ids[0];
        assert_eq!(
            written,
            HashSet::from([
                (a_node_id, 10, allele_node_id, 0),
                (allele_node_id, 4, t_node_id, 2)
            ])
        );
        assert!(retired.contains(&(a_node_id, 10, t_node_id, 0)));
    }

    /// A deletion where three routes arrive at a node adds an edge from each of them.
    #[test]
    fn test_deletion_adds_an_edge_from_every_route_arriving_at_the_node() {
        let conn = &get_connection(None).unwrap();
        let (block_group_id, path) = setup_block_group(conn);
        let a_node_id = HashId::convert_str("test-a-node");
        let t_node_id = HashId::convert_str("test-t-node");
        let mut alternative_node_ids = vec![];
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
            let into = Edge::create(
                conn,
                a_node_id,
                10,
                Strand::Forward,
                node_id,
                0,
                Strand::Forward,
            )
            .unwrap();
            let out = Edge::create(
                conn,
                node_id,
                4,
                Strand::Forward,
                t_node_id,
                0,
                Strand::Forward,
            )
            .unwrap();
            BlockGroupEdge::bulk_create(
                conn,
                &[into.id, out.id].map(|edge_id| BlockGroupEdgeData {
                    block_group_id,
                    edge_id,
                    chromosome_index: NO_CHROMOSOME_INDEX,
                    phased: 0,
                }),
            );
            alternative_node_ids.push(node_id);
        }
        let before = BlockGroupEdge::edges_for_block_group(conn, &block_group_id, None)
            .iter()
            .map(|augmented_edge| augmented_edge.edge.id)
            .collect::<HashSet<_>>();
        let mut edit_node_ids = vec![];
        let (start, end, bases) = (10, 12, "");
        let (node_id, length) = if bases.is_empty() {
            (HashId::convert_str(""), 0)
        } else {
            let sequence = Sequence::new()
                .sequence_type("DNA")
                .sequence(bases)
                .save(conn)
                .unwrap();
            let node_id = Node::create(
                conn,
                &sequence.hash,
                &HashId::convert_str(&format!("edit-{bases}-{start}")),
            )
            .unwrap();
            (node_id, sequence.length)
        };
        edit_node_ids.push(node_id);
        let block = PathBlock {
            node_id,
            block_sequence: bases.to_string(),
            sequence_start: 0,
            sequence_end: length,
            path_start: start,
            path_end: end,
            strand: Strand::Forward,
        };
        let region = ResolvedGenRegion::from_path(conn, block_group_id, &path, start, end).unwrap();
        let change = BlockGroupChange {
            region,
            path_accession: None,
            block,
            chromosome_index: NO_CHROMOSOME_INDEX,
            phased: 0,
            preserve_edge: true,
        };
        BlockGroup::insert_change(conn, test_workspace(), &change).unwrap();
        let written = BlockGroupEdge::edges_for_block_group(conn, &block_group_id, None)
            .iter()
            .filter(|augmented_edge| !before.contains(&augmented_edge.edge.id))
            .map(|augmented_edge| {
                (
                    augmented_edge.edge.source_node_id,
                    augmented_edge.edge.source_coordinate,
                    augmented_edge.edge.target_node_id,
                    augmented_edge.edge.target_coordinate,
                )
            })
            .filter(
                |(source_node_id, source_coordinate, target_node_id, target_coordinate)| {
                    (source_node_id, source_coordinate) != (target_node_id, target_coordinate)
                },
            )
            .collect::<HashSet<_>>();
        let sequences =
            BlockGroup::get_all_sequences(conn, test_workspace(), &block_group_id, true).unwrap();
        assert_eq!(sequences.len(), 6);
        assert_eq!(
            written,
            HashSet::from([
                (a_node_id, 10, t_node_id, 2),
                (alternative_node_ids[0], 4, t_node_id, 2),
                (alternative_node_ids[1], 4, t_node_id, 2),
            ])
        );
    }

    /// A deletion at either end of the contig attaches to the start or end marker node.
    #[test]
    fn test_deletion_at_a_contig_end_attaches_to_the_marker_node() {
        for (start, end, expected) in [
            (
                0,
                2,
                (PATH_START_NODE_ID, 0, HashId::convert_str("test-a-node"), 2),
            ),
            (
                38,
                40,
                (HashId::convert_str("test-g-node"), 8, PATH_END_NODE_ID, 0),
            ),
        ] {
            let conn = &get_connection(None).unwrap();
            let (block_group_id, path) = setup_block_group(conn);
            let before = BlockGroupEdge::edges_for_block_group(conn, &block_group_id, None)
                .iter()
                .map(|augmented_edge| augmented_edge.edge.id)
                .collect::<HashSet<_>>();
            let mut edit_node_ids = vec![];
            let bases = "";
            let (node_id, length) = if bases.is_empty() {
                (HashId::convert_str(""), 0)
            } else {
                let sequence = Sequence::new()
                    .sequence_type("DNA")
                    .sequence(bases)
                    .save(conn)
                    .unwrap();
                let node_id = Node::create(
                    conn,
                    &sequence.hash,
                    &HashId::convert_str(&format!("edit-{bases}-{start}")),
                )
                .unwrap();
                (node_id, sequence.length)
            };
            edit_node_ids.push(node_id);
            let block = PathBlock {
                node_id,
                block_sequence: bases.to_string(),
                sequence_start: 0,
                sequence_end: length,
                path_start: start,
                path_end: end,
                strand: Strand::Forward,
            };
            let region =
                ResolvedGenRegion::from_path(conn, block_group_id, &path, start, end).unwrap();
            let change = BlockGroupChange {
                region,
                path_accession: None,
                block,
                chromosome_index: NO_CHROMOSOME_INDEX,
                phased: 0,
                preserve_edge: true,
            };
            BlockGroup::insert_change(conn, test_workspace(), &change).unwrap();
            let written = BlockGroupEdge::edges_for_block_group(conn, &block_group_id, None)
                .iter()
                .filter(|augmented_edge| !before.contains(&augmented_edge.edge.id))
                .map(|augmented_edge| {
                    (
                        augmented_edge.edge.source_node_id,
                        augmented_edge.edge.source_coordinate,
                        augmented_edge.edge.target_node_id,
                        augmented_edge.edge.target_coordinate,
                    )
                })
                .filter(
                    |(source_node_id, source_coordinate, target_node_id, target_coordinate)| {
                        (source_node_id, source_coordinate) != (target_node_id, target_coordinate)
                    },
                )
                .collect::<HashSet<_>>();
            let sequences =
                BlockGroup::get_all_sequences(conn, test_workspace(), &block_group_id, true)
                    .unwrap();
            assert_eq!(sequences.len(), 2, "{start}-{end}");
            assert_eq!(written, HashSet::from([expected]), "{start}-{end}");
        }
    }

    /// An insertion at either end of the contig sits between the start or end marker node and
    /// the node beside it.
    #[test]
    fn test_insertion_at_a_contig_end_sits_next_to_the_marker_node() {
        for (position, before_edge, after_edge) in [
            (
                0,
                (PATH_START_NODE_ID, 0),
                (HashId::convert_str("test-a-node"), 0),
            ),
            (
                40,
                (HashId::convert_str("test-g-node"), 10),
                (PATH_END_NODE_ID, 0),
            ),
        ] {
            let start = position;
            let end = position;
            let conn = &get_connection(None).unwrap();
            let (block_group_id, path) = setup_block_group(conn);
            let before = BlockGroupEdge::edges_for_block_group(conn, &block_group_id, None)
                .iter()
                .map(|augmented_edge| augmented_edge.edge.id)
                .collect::<HashSet<_>>();
            let mut edit_node_ids = vec![];
            let bases = "GG";
            let (node_id, length) = if bases.is_empty() {
                (HashId::convert_str(""), 0)
            } else {
                let sequence = Sequence::new()
                    .sequence_type("DNA")
                    .sequence(bases)
                    .save(conn)
                    .unwrap();
                let node_id = Node::create(
                    conn,
                    &sequence.hash,
                    &HashId::convert_str(&format!("edit-{bases}-{start}")),
                )
                .unwrap();
                (node_id, sequence.length)
            };
            edit_node_ids.push(node_id);
            let block = PathBlock {
                node_id,
                block_sequence: bases.to_string(),
                sequence_start: 0,
                sequence_end: length,
                path_start: start,
                path_end: end,
                strand: Strand::Forward,
            };
            let region =
                ResolvedGenRegion::from_path(conn, block_group_id, &path, start, end).unwrap();
            let change = BlockGroupChange {
                region,
                path_accession: None,
                block,
                chromosome_index: NO_CHROMOSOME_INDEX,
                phased: 0,
                preserve_edge: true,
            };
            BlockGroup::insert_change(conn, test_workspace(), &change).unwrap();
            let written = BlockGroupEdge::edges_for_block_group(conn, &block_group_id, None)
                .iter()
                .filter(|augmented_edge| !before.contains(&augmented_edge.edge.id))
                .map(|augmented_edge| {
                    (
                        augmented_edge.edge.source_node_id,
                        augmented_edge.edge.source_coordinate,
                        augmented_edge.edge.target_node_id,
                        augmented_edge.edge.target_coordinate,
                    )
                })
                .filter(
                    |(source_node_id, source_coordinate, target_node_id, target_coordinate)| {
                        (source_node_id, source_coordinate) != (target_node_id, target_coordinate)
                    },
                )
                .collect::<HashSet<_>>();
            let sequences =
                BlockGroup::get_all_sequences(conn, test_workspace(), &block_group_id, true)
                    .unwrap();
            assert_eq!(sequences.len(), 2, "{position}");
            let edit_node_id = edit_node_ids[0];
            assert_eq!(
                written,
                HashSet::from([
                    (before_edge.0, before_edge.1, edit_node_id, 0),
                    (edit_node_id, 2, after_edge.0, after_edge.1),
                ]),
                "{position}"
            );
        }
    }
}
