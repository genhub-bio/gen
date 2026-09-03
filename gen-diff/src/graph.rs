//! Builds a connected, renderable graph describing changes between two states.
//!
//! This module is the graph-construction half of `gen-diff`.
//! [`crate::operations`] first uses Dolt's table-diff functions to find affected
//! block groups, edges, and nodes and to associate those rows with operations.
//! It then calls `build_block_group_diff` for each affected block-group ID.
//! The resulting [`BlockGroupDiff`] is consumed by the CLI, patch preview,
//! GenHub, and the diff TUI to show unchanged context together with added,
//! removed, and modified graph elements.
//!
//! # Source and target terminology
//!
//! At a merge boundary, the **source** is the branch the changes are coming
//! from and the **target** is the branch they are being merged into. The
//! historical names in this crate use `source` and `target` differently:
//! `source_ref` is the graph's comparison baseline and `target_ref` is the
//! changed state being displayed. Consequently, a caller showing “merge source
//! branch into target branch” passes the merge target as the first/internal
//! source endpoint and the merge source as the second/internal target endpoint.
//! This is the source/target swap visible in merge-oriented callers.
//!
//! The same distinction applies to `source_block_group` and
//! `target_block_group` in [`BlockGroupDiff`]. They record whether the requested
//! block-group ID exists in the baseline and changed database states,
//! respectively; they do not identify the incoming and receiving branches of a
//! merge. Source/target on a [`GraphEdge`] is a third, unrelated use: it is only
//! the direction of that edge.
//!
//! # Block-group membership and effective graph snapshots
//!
//! Every block group, including one created for a child sample, owns a complete
//! set of `block_group_edges`. Copied memberships are stored as rows belonging
//! to the child block group, and its parent relationship is recorded separately
//! by `parent_block_group_id`. The child graph therefore does not need its
//! parent in order to load or reconstruct its stored state.
//!
//! A parent is selected only to make an added or removed child graph
//! biologically meaningful. The changed rows would show a child's entire graph
//! being removed or created. Thus, comparing to the parent shows the edits that
//! introduced in the child. `load_block_group_comparison_graphs` therefore
//! keeps two concepts separate:
//!
//! - `source_block_group` and `target_block_group` preserve actual membership
//!   of the requested ID and determine whether the block group is added,
//!   removed, or modified.
//! - `source_effective_block_group_id` and
//!   `target_effective_block_group_id` select the graph used for display. When
//!   the requested ID is absent, this may be the child's parent ID.
//!
//! The corresponding effective history ref comes from the endpoint where the
//! selected row is known to exist. Thus a target-only child uses its own graph
//! at `target_ref` and its parent graph at that same endpoint as the source
//! baseline; a source-only child uses the symmetric source-side pair. This
//! selection affects only the rendered graph. It never fills in the missing
//! `source_block_group` or `target_block_group`.
//!
//! Dolt's copied `block_group_edges` rows contain enough information to
//! reconstruct a new child's graph from the table diff. The effective parent is
//! not a workaround for incomplete diff rows. It supplies the inherited
//! comparison baseline, and `effective_edge_changes_for_source` filters copied
//! memberships that Dolt reports as added but that already exist in that
//! selected parent graph.
//!
//! # Graph-building approach
//!
//! Graph construction has three steps:
//!
//! 1. Load the selected endpoint edge sets and convert edge coordinates into
//!    [`GraphNode`] sequence slices. Coordinates are slices of the backing
//!    sequence, not positions in graph space.
//! 2. Split both sides at the union of their slice boundaries. For example,
//!    `N[0..10]` on one side and `N[0..4] + N[4..10]` on the other become the
//!    same two comparison units. Real path edges are retained,
//!    persistence-only edit-site marker edges are omitted, and neutral
//!    continuation edges reconnect slices that belong to one covered span.
//! 3. Union the normalized endpoint graphs into a [`DiffGenGraph`]. Presence
//!    and Dolt operation metadata determine each node's and edge's
//!    [`DiffChange`]. Start/end nodes and unchanged context remain in the result
//!    because the renderer needs a connected graph for layout.
//!
//! Loading both endpoint graphs makes path membership and marker filtering
//! straightforward. An alternative is to load the baseline once and apply the
//! `block_group_edges` diff rows to reconstruct the changed graph; the copied
//! child rows are sufficient for that and it may be preferable if endpoint
//! loading becomes a measured bottleneck. Another alternative is a literal
//! database-membership diff with an empty missing side, but that intentionally
//! gives up the lineage-aware view. The current effective-ID/ref variables could
//! also be replaced by an explicit endpoint-snapshot struct resolved once
//! upstream; that would preserve the behavior while making the storage choice
//! less indirect.

use std::collections::{HashMap, HashSet};

use gen_core::{
    DoltHashId, HashId, NO_CHROMOSOME_INDEX, PATH_END_NODE_ID, PATH_START_NODE_ID,
    PRESERVE_EDIT_SITE_CHROMOSOME_INDEX, Strand, is_terminal,
};
use gen_graph::{GenGraph, GraphEdge, GraphNode};
use gen_models::{
    block_group::BlockGroup,
    block_group_edge::{AugmentedEdge, BlockGroupEdge},
    db::GraphConnection,
};
use petgraph::graphmap::DiGraphMap;

use crate::operations::{BlockGroupEdgeChange, BlockGroupEdgeKey, NodeChange, OperationDiffError};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum DiffChangeKind {
    Unchanged,
    Added,
    Removed,
    Modified,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct DiffChange {
    pub kind: DiffChangeKind,
    pub operation: Option<DoltHashId>,
}

impl DiffChange {
    /// Creates the neutral annotation used for graph elements shared by both endpoints.
    ///
    /// Input builders and merge helpers use this constructor so unchanged
    /// context has neither a change kind nor misleading operation attribution.
    pub const fn unchanged() -> Self {
        Self {
            kind: DiffChangeKind::Unchanged,
            operation: None,
        }
    }

    /// Creates an annotation from the change kind and attribution collected from Dolt.
    ///
    /// Input-graph construction uses this common constructor before the source
    /// and target annotations are consolidated into the rendered diff graph.
    pub const fn new(kind: DiffChangeKind, operation: Option<DoltHashId>) -> Self {
        Self { kind, operation }
    }

    /// Returns explicit attribution or the comparison's fallback operation.
    ///
    /// Change-merging code uses this when every rendered change must be tied to
    /// an operation even if the table-diff row did not provide one.
    pub fn operation_or(self, default_operation: DoltHashId) -> DoltHashId {
        self.operation.unwrap_or(default_operation)
    }
}

#[derive(Clone, Debug)]
pub struct BlockGroupDiff {
    pub id: HashId,
    pub source_block_group: Option<BlockGroup>,
    pub target_block_group: Option<BlockGroup>,
    pub graph: DiffGenGraph,
}

/// Describes how a block group differs between the comparison endpoints.
///
/// This is computed from block-group membership at the source and target. It is
/// intentionally distinct from [`DiffChangeKind`], which describes individual
/// graph elements and can therefore also be unchanged.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum BlockGroupChangeKind {
    Added,
    Removed,
    Modified,
}

impl BlockGroupDiff {
    /// Returns the block group's change classification for this comparison.
    ///
    /// `None` indicates an invalid diff with no block group at either endpoint.
    /// The classification is derived rather than stored so it cannot disagree
    /// with `source_block_group` and `target_block_group`. CLI and TUI views use
    /// the result to label the complete block-group diff.
    pub const fn change_kind(&self) -> Option<BlockGroupChangeKind> {
        match (
            self.source_block_group.is_some(),
            self.target_block_group.is_some(),
        ) {
            (false, true) => Some(BlockGroupChangeKind::Added),
            (true, false) => Some(BlockGroupChangeKind::Removed),
            (true, true) => Some(BlockGroupChangeKind::Modified),
            (false, false) => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct DiffGraphNode {
    pub node: GraphNode,
    pub change: DiffChange,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct DiffGraphEdge {
    pub edge: GraphEdge,
    pub change: DiffChange,
}

pub type DiffGenGraph = DiGraphMap<DiffGraphNode, Vec<DiffGraphEdge>>;

pub struct DiffGenGraphRef<'a>(pub &'a DiffGenGraph);

impl<'a> From<&'a DiffGenGraph> for DiffGenGraphRef<'a> {
    /// Wraps a borrowed diff graph for conversion without cloning its topology.
    ///
    /// Generic graph consumers use this adapter before stripping diff
    /// annotations through the `GenGraph` conversion.
    fn from(graph: &'a DiffGenGraph) -> Self {
        Self(graph)
    }
}

impl<'a> From<DiffGenGraphRef<'a>> for GenGraph {
    /// Projects a diff graph into the ordinary graph shape used by layout code.
    ///
    /// The conversion keeps nodes and edges but deliberately drops change
    /// annotations; the diff TUI uses the resulting `GenGraph` for layout while
    /// retaining the annotated graph for styling.
    fn from(val: DiffGenGraphRef<'a>) -> Self {
        let mut graph = GenGraph::new();
        for node in val.0.nodes() {
            graph.add_node(node.node);
        }
        for (src, dest, edges) in val.0.all_edges() {
            let mapped_edges = edges.iter().map(|edge| edge.edge).collect::<Vec<_>>();
            graph.add_edge(src.node, dest.node, mapped_edges);
        }
        graph
    }
}

impl From<DiffGraphNode> for GraphNode {
    /// Removes a node's diff annotation for consumers that only need graph topology.
    ///
    /// Conversion and rendering helpers use this projection instead of
    /// duplicating field access at each generic-graph boundary.
    fn from(node: DiffGraphNode) -> Self {
        node.node
    }
}

impl From<DiffGraphEdge> for GraphEdge {
    /// Removes an edge's diff annotation for consumers that only need graph topology.
    ///
    /// Conversion and rendering helpers use this projection when passing diff
    /// edges through APIs written for ordinary graph edges.
    fn from(edge: DiffGraphEdge) -> Self {
        edge.edge
    }
}

pub(crate) struct BlockGroupDiffInputs<'a> {
    pub(crate) source_ref: &'a str,
    pub(crate) target_ref: &'a str,
    pub(crate) default_operation: DoltHashId,
    pub(crate) edge_changes_by_block_group: &'a HashMap<HashId, Vec<BlockGroupEdgeChange>>,
    pub(crate) block_group_operations_by_id: &'a HashMap<HashId, HashSet<DoltHashId>>,
    pub(crate) node_changes: &'a [NodeChange],
}

/// Builds the renderable diff for one block group identified by table-diff discovery.
///
/// `build_dolt_operation_diff` calls this once per affected block group. It
/// resolves normalized endpoint input graphs, unifies them, and omits a
/// source-and-target block group when the resulting graph has no visible
/// changes, avoiding empty entries in command and TUI output.
pub(crate) fn build_block_group_diff(
    graph_conn: &GraphConnection,
    block_group_id: HashId,
    inputs: &BlockGroupDiffInputs<'_>,
) -> Result<Option<BlockGroupDiff>, OperationDiffError> {
    let Some(comparison_graphs) =
        load_block_group_comparison_graphs(graph_conn, block_group_id, inputs)?
    else {
        return Ok(None);
    };
    let graph = build_unified_diff_graph(&comparison_graphs, inputs.default_operation);
    if !diff_graph_has_changes(&graph)
        && comparison_graphs.source_block_group.is_some()
        && comparison_graphs.target_block_group.is_some()
    {
        return Ok(None);
    }

    Ok(Some(BlockGroupDiff {
        id: block_group_id,
        source_block_group: comparison_graphs.source_block_group,
        target_block_group: comparison_graphs.target_block_group,
        graph,
    }))
}

/// Loads comparable source and target input graphs for one block-group diff.
///
/// `build_block_group_diff` uses this preparation step to keep endpoint
/// membership separate from the lineage-aware graph chosen for display. It
/// loads complete endpoint edges, filters copied child memberships, aligns
/// sequence-slice boundaries, and attaches Dolt operation attribution because
/// the unified builder needs matching graph units and real path membership.
fn load_block_group_comparison_graphs(
    graph_conn: &GraphConnection,
    block_group_id: HashId,
    inputs: &BlockGroupDiffInputs<'_>,
) -> Result<Option<BlockGroupComparisonGraphs>, OperationDiffError> {
    let target_block_group =
        BlockGroup::get_by_id(graph_conn, &block_group_id, Some(inputs.target_ref)).ok();
    let source_block_group =
        BlockGroup::get_by_id(graph_conn, &block_group_id, Some(inputs.source_ref)).ok();
    if source_block_group.is_none() && target_block_group.is_none() {
        return Ok(None);
    }
    // Added or removed child block groups only exist on one endpoint. The
    // opposite endpoint is represented by the child's parent so the diff
    // compares the child graph against the graph it diverged from.
    let source_parent_block_group_id = target_block_group
        .as_ref()
        .and_then(|block_group| block_group.parent_block_group_id);
    let target_parent_block_group_id = source_block_group
        .as_ref()
        .and_then(|block_group| block_group.parent_block_group_id);
    let source_effective_block_group_id = source_block_group
        .as_ref()
        .map(|block_group| block_group.id)
        .or(source_parent_block_group_id);
    let target_effective_block_group_id = target_block_group
        .as_ref()
        .map(|block_group| block_group.id)
        .or(target_parent_block_group_id);
    // When the block group exists on the endpoint, read it from that endpoint's
    // ref. When it does not, the effective id is the parent id discovered from
    // the other endpoint's child row, so the parent must be read from that
    // endpoint's ref instead.
    let source_effective_history_ref = if source_block_group.is_some() {
        Some(inputs.source_ref)
    } else {
        Some(inputs.target_ref)
    };
    let target_effective_history_ref = if target_block_group.is_some() {
        Some(inputs.target_ref)
    } else {
        Some(inputs.source_ref)
    };
    // Load both endpoint edge sets directly instead of replaying diff rows over
    // the source. Diff rows are still useful for attribution, but they are not
    // a complete graph description:
    //
    //   new sample with no sequence edits:
    //     source(parent):  start -> A -> end
    //     target(child):   start -> A -> end
    //
    // The copied child edges may have the same edge keys as the parent, so
    // replaying only changed keys can make the target look empty and hide the
    // created sample. Endpoint reads keep presence-only samples visible.
    //
    //   insertion/deletion edits:
    //     source: A -> B
    //     target: A -> I -> B, plus internal edit-site marker edges
    //
    // The target endpoint tells us which path edges are real graph choices and
    // lets the input builder filter marker edges before rendering.
    let source_edges = load_block_group_edges(
        graph_conn,
        source_effective_block_group_id,
        source_effective_history_ref,
    );
    let raw_block_group_edge_changes = inputs
        .edge_changes_by_block_group
        .get(&block_group_id)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let block_group_edge_changes =
        effective_edge_changes_for_source(&source_edges, raw_block_group_edge_changes);
    let target_edges = load_block_group_edges(
        graph_conn,
        target_effective_block_group_id,
        target_effective_history_ref,
    );
    let source_spans = spans_from_edges(&source_edges);
    let target_spans = spans_from_edges(&target_edges);
    let operation_change_lookup = collect_input_operation_changes(
        block_group_id,
        inputs,
        &block_group_edge_changes,
        &source_spans,
        &target_spans,
    );
    let (source_nodes, target_nodes) =
        split_nodes_at_shared_boundaries(&source_spans, &target_spans);
    let source_graph = build_diff_input_graph(
        source_nodes,
        &source_spans,
        &source_edges,
        &operation_change_lookup.source_node_operation_by_id,
        &operation_change_lookup.source_edge_operation_by_id,
    );
    let reconstructed_target_graph = build_diff_input_graph(
        target_nodes,
        &target_spans,
        &target_edges,
        &operation_change_lookup.target_node_operation_by_id,
        &operation_change_lookup.target_edge_operation_by_id,
    );

    Ok(Some(BlockGroupComparisonGraphs {
        source_block_group,
        target_block_group,
        source_graph,
        reconstructed_target_graph,
    }))
}

/// Loads all augmented path edges for an optional block group at one history ref.
///
/// `load_block_group_comparison_graphs` uses an empty vector for a genuinely missing
/// endpoint and a complete edge set otherwise. Centralizing that choice keeps
/// absence handling out of the span and input-graph builders.
fn load_block_group_edges(
    graph_conn: &GraphConnection,
    block_group_id: Option<HashId>,
    history_ref: Option<&str>,
) -> Vec<AugmentedEdge> {
    block_group_id.map_or_else(Vec::new, |block_group_id| {
        BlockGroupEdge::edges_for_block_group(graph_conn, &block_group_id, history_ref)
    })
}

/// Derives the sequence slices covered by an endpoint's path edges.
///
/// The input loader uses these spans as lightweight graph nodes, pairing
/// observed entry and exit coordinates without fetching sequence text. This
/// preserves enough endpoint membership for later normalization and rendering
/// while keeping attribution attached to the original edges.
fn spans_from_edges(edges: &[AugmentedEdge]) -> HashSet<GraphNode> {
    // A diff graph node only needs a backing node id and a sequence slice. Do
    // not call Edge::blocks_from_edges here: that query fetches sequence text
    // so GroupBlock::sequence can work, but the diff graph resolves sequence
    // text later through the regular GraphNode rendering path.
    //
    //   endpoint edges:
    //     start -> A@0
    //     A@10 -> end
    //
    //   edge-derived span:
    //     A[0..10]
    //
    // Keeping this as edge-derived spans lets edge attribution stay attached
    // until the graph builder has both the display nodes and real path edges.
    let mut starts_by_node_id = HashMap::<HashId, HashSet<i64>>::new();
    let mut ends_by_node_id = HashMap::<HashId, HashSet<i64>>::new();
    for edge in edges.iter().map(|edge| &edge.edge) {
        if !is_terminal(edge.source_node_id) {
            ends_by_node_id
                .entry(edge.source_node_id)
                .or_default()
                .insert(edge.source_coordinate);
        }
        if !is_terminal(edge.target_node_id) {
            starts_by_node_id
                .entry(edge.target_node_id)
                .or_default()
                .insert(edge.target_coordinate);
        }
    }

    let mut node_ids = starts_by_node_id.keys().copied().collect::<HashSet<_>>();
    node_ids.extend(ends_by_node_id.keys().copied());
    let mut spans = HashSet::new();
    for node_id in node_ids {
        let empty_starts = HashSet::new();
        let empty_ends = HashSet::new();
        let starts = starts_by_node_id.get(&node_id).unwrap_or(&empty_starts);
        let ends = ends_by_node_id.get(&node_id).unwrap_or(&empty_ends);
        for (sequence_start, sequence_end) in coordinate_spans(starts, ends) {
            spans.insert(GraphNode {
                node_id,
                sequence_start,
                sequence_end,
            });
        }
    }
    spans
}

/// Turns observed entry and exit coordinates into ordered sequence intervals.
///
/// `spans_from_edges` uses adjacent coordinates rather than graph-space
/// positions because `GraphNode` ranges slice the backing sequence. The
/// resulting intervals are the endpoint-local spans later split at shared
/// source/target boundaries.
fn coordinate_spans(starts: &HashSet<i64>, ends: &HashSet<i64>) -> Vec<(i64, i64)> {
    let mut coordinates = starts.union(ends).copied().collect::<Vec<_>>();
    coordinates.sort();
    coordinates.dedup();
    if coordinates.len() <= 1 {
        return coordinates
            .first()
            .copied()
            .map(|coordinate| vec![(coordinate, coordinate)])
            .unwrap_or_default();
    }
    coordinates
        .windows(2)
        .map(|window| (window[0], window[1]))
        .collect()
}

/// Normalizes endpoint nodes to the same sequence-slice boundaries.
///
/// `load_block_group_comparison_graphs` uses the paired result so the unified graph
/// can compare identical `GraphNode` keys. It takes the union of boundaries
/// from both endpoints because an edit visible on only one side must still
/// expose the corresponding slice on the other.
fn split_nodes_at_shared_boundaries(
    source_spans: &HashSet<GraphNode>,
    target_spans: &HashSet<GraphNode>,
) -> (HashSet<GraphNode>, HashSet<GraphNode>) {
    let boundaries_by_node_id = collect_shared_boundaries(source_spans, target_spans);
    (
        split_spans_at_boundaries(source_spans, &boundaries_by_node_id),
        split_spans_at_boundaries(target_spans, &boundaries_by_node_id),
    )
}

/// Collects every endpoint boundary for each backing sequence node.
///
/// `split_nodes_at_shared_boundaries` uses this union as the shared cut
/// plan, ensuring source-only and target-only boundaries are applied
/// symmetrically before graph elements are classified.
fn collect_shared_boundaries(
    source_spans: &HashSet<GraphNode>,
    target_spans: &HashSet<GraphNode>,
) -> HashMap<HashId, HashSet<i64>> {
    let mut boundaries_by_node_id = HashMap::new();
    for span in source_spans.iter().chain(target_spans) {
        let boundaries = boundaries_by_node_id
            .entry(span.node_id)
            .or_insert_with(HashSet::new);
        boundaries.insert(span.sequence_start);
        boundaries.insert(span.sequence_end);
    }
    boundaries_by_node_id
}

/// Splits existing endpoint spans according to the shared boundary plan.
///
/// The normalization step uses this helper to divide only sequence already
/// present in that endpoint; it never fills gaps. That preserves membership
/// while giving `build_unified_diff_graph` equal-sized units to mark as added,
/// removed, or unchanged.
fn split_spans_at_boundaries(
    spans: &HashSet<GraphNode>,
    boundaries_by_node_id: &HashMap<HashId, HashSet<i64>>,
) -> HashSet<GraphNode> {
    let mut nodes = HashSet::new();
    for span in spans {
        // Split both endpoint graphs on the union of observed edge boundaries
        // for each backing node. GraphNode coordinates are sequence slices, not
        // path coordinates, so this only divides already-present sequence spans.
        //
        //   source before VCF edit:
        //     [0................................34]
        //
        //   target after deletion/insertion:
        //     [0..3] [4..10] [10..............34] and inserted [AGA]
        //
        //   source display nodes:
        //     [0..3] [3..4] [4..10] [10.......34]
        //
        // Without this split, the deleted [3..4] base is hidden inside a large
        // source block and cannot be colored as removed.
        if span.sequence_start == span.sequence_end {
            nodes.insert(*span);
            continue;
        }
        let mut boundaries = boundaries_by_node_id
            .get(&span.node_id)
            .into_iter()
            .flat_map(|boundaries| boundaries.iter())
            .filter(|coordinate| {
                **coordinate >= span.sequence_start && **coordinate <= span.sequence_end
            })
            .copied()
            .collect::<Vec<_>>();
        boundaries.sort();
        boundaries.dedup();
        if boundaries.len() < 2 {
            nodes.insert(*span);
            continue;
        }
        for window in boundaries.windows(2) {
            let sequence_start = window[0];
            let sequence_end = window[1];
            if sequence_start == sequence_end {
                continue;
            }
            nodes.insert(GraphNode {
                node_id: span.node_id,
                sequence_start,
                sequence_end,
            });
        }
    }
    nodes
}

/// Removes copied child-edge changes already represented by the effective source graph.
///
/// `load_block_group_comparison_graphs` uses this filter when a missing child
/// endpoint is displayed through its parent. Dolt reports the child's copied
/// memberships as additions, but suppressing keys already in that parent keeps
/// inherited paths from receiving false change attribution.
fn effective_edge_changes_for_source(
    source_edges: &[AugmentedEdge],
    edge_changes: &[BlockGroupEdgeChange],
) -> Vec<BlockGroupEdgeChange> {
    let source_edge_keys = source_edges
        .iter()
        .map(BlockGroupEdgeKey::from)
        .collect::<HashSet<_>>();
    edge_changes
        .iter()
        .copied()
        .filter(
            |edge_change| match (edge_change.old_key, edge_change.new_key) {
                (None, Some(new_key)) => !source_edge_keys.contains(&new_key),
                (Some(old_key), None) => source_edge_keys.contains(&old_key),
                (Some(old_key), Some(new_key)) => {
                    source_edge_keys.contains(&old_key) || !source_edge_keys.contains(&new_key)
                }
                (None, None) => false,
            },
        )
        .collect()
}

/// Builds endpoint-specific operation attribution for nodes and edges in one input graph.
///
/// `load_block_group_comparison_graphs` passes this lookup to both input builders.
/// It intersects Dolt changes with the block group's represented operations and
/// endpoint membership so unrelated table rows do not color this graph.
fn collect_input_operation_changes(
    block_group_id: HashId,
    inputs: &BlockGroupDiffInputs<'_>,
    changed_edge_ids: &[BlockGroupEdgeChange],
    source_spans: &HashSet<GraphNode>,
    target_spans: &HashSet<GraphNode>,
) -> InputOperationChangeLookup {
    let mut change_lookup = InputOperationChangeLookup::default();
    let Some(block_group_operations) = inputs.block_group_operations_by_id.get(&block_group_id)
    else {
        return change_lookup;
    };
    let source_node_ids = span_node_ids(source_spans);
    let target_node_ids = span_node_ids(target_spans);
    for node_change in inputs.node_changes {
        if let (Some(old_node_id), Some(operation)) =
            (node_change.old_node_id(), node_change.old_operation())
            && block_group_operations.contains(&operation)
            && source_node_ids.contains(&old_node_id)
            && !target_node_ids.contains(&old_node_id)
        {
            change_lookup
                .source_node_operation_by_id
                .insert(old_node_id, operation);
        }
        if let (Some(new_node_id), Some(operation)) =
            (node_change.new_node_id(), node_change.new_operation())
            && block_group_operations.contains(&operation)
            && target_node_ids.contains(&new_node_id)
            && !source_node_ids.contains(&new_node_id)
        {
            change_lookup
                .target_node_operation_by_id
                .insert(new_node_id, operation);
        }
    }

    for changed_edge_id in changed_edge_ids {
        if let (Some(old_edge_id), Some(operation)) = (
            changed_edge_id.old_edge_id(),
            changed_edge_id.old_operation(),
        ) {
            change_lookup
                .source_edge_operation_by_id
                .insert(old_edge_id, operation);
        }
        if let (Some(new_edge_id), Some(operation)) = (
            changed_edge_id.new_edge_id(),
            changed_edge_id.new_operation(),
        ) {
            change_lookup
                .target_edge_operation_by_id
                .insert(new_edge_id, operation);
        }
    }
    change_lookup
}

/// Extracts non-terminal backing node IDs from an input's sequence spans.
///
/// `collect_input_operation_changes` uses these sets to attribute a node
/// only when its membership actually changes between the two endpoint graphs.
fn span_node_ids(spans: &HashSet<GraphNode>) -> HashSet<HashId> {
    spans
        .iter()
        .filter(|span| !is_terminal(span.node_id))
        .map(|span| span.node_id)
        .collect()
}

/// Unifies normalized endpoint inputs into the connected graph rendered as a diff.
///
/// `build_block_group_diff` uses this after endpoint loading, and focused graph
/// tests exercise it directly. It unions node and edge identities, then merges
/// presence and operation annotations instead of discarding unchanged context;
/// the retained context and terminal structure are required by CLI and TUI
/// layout while changed elements receive added, removed, or modified styling.
pub(crate) fn build_unified_diff_graph(
    comparison_graphs: &BlockGroupComparisonGraphs,
    default_operation: DoltHashId,
) -> DiffGenGraph {
    let mut merged_nodes = comparison_graphs
        .source_graph
        .node_changes
        .keys()
        .chain(
            comparison_graphs
                .reconstructed_target_graph
                .node_changes
                .keys(),
        )
        .copied()
        .collect::<Vec<_>>();
    merged_nodes.sort();
    merged_nodes.dedup();

    let mut merged_edge_keys = comparison_graphs
        .source_graph
        .edge_changes
        .keys()
        .chain(
            comparison_graphs
                .reconstructed_target_graph
                .edge_changes
                .keys(),
        )
        .copied()
        .collect::<Vec<_>>();
    merged_edge_keys.sort();
    merged_edge_keys.dedup();
    let source_edge_pairs = annotated_edge_pairs(&comparison_graphs.source_graph);
    let target_edge_pairs = annotated_edge_pairs(&comparison_graphs.reconstructed_target_graph);

    let mut diff_graph = DiffGenGraph::new();
    for node in merged_nodes {
        diff_graph.add_node(diff_graph_node(node, comparison_graphs, default_operation));
    }

    let mut merged_edges_by_node_pair = HashMap::<(GraphNode, GraphNode), Vec<GraphEdgeKey>>::new();
    for edge_key in merged_edge_keys {
        merged_edges_by_node_pair
            .entry((edge_key.source, edge_key.target))
            .or_default()
            .push(edge_key);
    }
    let mut merged_edge_pairs = merged_edges_by_node_pair.into_iter().collect::<Vec<_>>();
    merged_edge_pairs.sort_by_key(|((source, target), _)| (*source, *target));
    for ((source, target), edge_keys) in merged_edge_pairs {
        let source_node = diff_graph_node(source, comparison_graphs, default_operation);
        let target_node = diff_graph_node(target, comparison_graphs, default_operation);
        let unchanged_endpoints = source_node.change.kind == DiffChangeKind::Unchanged
            && target_node.change.kind == DiffChangeKind::Unchanged;
        let diff_edges = edge_keys
            .iter()
            .copied()
            .map(|edge_key| DiffGraphEdge {
                edge: graph_edge_for_key(comparison_graphs, edge_key),
                change: merge_input_edge_changes(
                    comparison_graphs
                        .source_graph
                        .edge_changes
                        .get(&edge_key)
                        .copied(),
                    comparison_graphs
                        .reconstructed_target_graph
                        .edge_changes
                        .get(&edge_key)
                        .copied(),
                    unchanged_endpoints,
                    source_edge_pairs.contains(&(source, target)),
                    target_edge_pairs.contains(&(source, target)),
                    default_operation,
                ),
            })
            .collect::<Vec<_>>();
        diff_graph.add_edge(source_node, target_node, diff_edges);
    }

    diff_graph
}

/// Reports whether a unified graph contains any visible node or edge change.
///
/// `build_block_group_diff` uses this scan to omit modified block groups whose
/// table rows normalize to unchanged graph content, while still retaining
/// membership-only additions and removals.
fn diff_graph_has_changes(graph: &DiffGenGraph) -> bool {
    graph
        .nodes()
        .any(|node| node.change.kind != DiffChangeKind::Unchanged)
        || graph.all_edges().any(|(_, _, edges)| {
            edges
                .iter()
                .any(|edge| edge.change.kind != DiffChangeKind::Unchanged)
        })
}

/// Creates one unified node by reconciling its endpoint annotations.
///
/// `build_unified_diff_graph` calls this for every normalized node and for edge
/// endpoints. Using one helper guarantees that topology keys and displayed node
/// styling share the same presence and attribution rules.
fn diff_graph_node(
    node: GraphNode,
    comparison_graphs: &BlockGroupComparisonGraphs,
    default_operation: DoltHashId,
) -> DiffGraphNode {
    let source_change = comparison_graphs
        .source_graph
        .node_changes
        .get(&node)
        .copied();
    let target_change = comparison_graphs
        .reconstructed_target_graph
        .node_changes
        .get(&node)
        .copied();
    DiffGraphNode {
        node,
        change: merge_input_changes(source_change, target_change, default_operation),
    }
}

/// Collects endpoint pairs that have at least one annotated edge.
///
/// Edge merging uses these pairs to recognize topology that survives with a
/// different edge identity. Comparing pairs as well as complete edge keys lets
/// the renderer describe that case as modification instead of unrelated
/// removal and addition.
fn annotated_edge_pairs(input_graph: &DiffInputGraph) -> HashSet<(GraphNode, GraphNode)> {
    input_graph
        .edge_changes
        .keys()
        .map(GraphEdgeKey::node_pair)
        .collect()
}

/// Retrieves the concrete edge represented by a merged identity key.
///
/// `build_unified_diff_graph` prefers the target copy so current metadata is
/// rendered, then falls back to the source for removed edges. The key was
/// collected from one of those inputs, so absence from both is an invariant
/// violation.
fn graph_edge_for_key(
    comparison_graphs: &BlockGroupComparisonGraphs,
    edge_key: GraphEdgeKey,
) -> GraphEdge {
    input_graph_edge_for_key(&comparison_graphs.reconstructed_target_graph, edge_key)
        .or_else(|| input_graph_edge_for_key(&comparison_graphs.source_graph, edge_key))
        .expect("should contain graph edge for merged diff key")
}

/// Finds the exact edge for a key within one annotated input graph.
///
/// `graph_edge_for_key` uses this helper because `GenGraph` stores multiple
/// edges per node pair. Matching the identity fields selects the edge whose
/// change annotation participated in the unified key set.
fn input_graph_edge_for_key(
    input_graph: &DiffInputGraph,
    edge_key: GraphEdgeKey,
) -> Option<GraphEdge> {
    input_graph
        .graph
        .edge_weight(edge_key.source, edge_key.target)?
        .iter()
        .copied()
        .find(|edge| edge_key.matches_edge(*edge))
}

/// Reconciles source and target presence plus attribution for one graph element.
///
/// Unified-node construction and the default edge path use this function.
/// Presence on only one side determines addition or removal; elements on both
/// sides become modified only when either input carries change metadata, which
/// keeps shared layout context neutral.
fn merge_input_changes(
    source_change: Option<DiffChange>,
    target_change: Option<DiffChange>,
    default_operation: DoltHashId,
) -> DiffChange {
    match (source_change, target_change) {
        (None, None) => DiffChange::unchanged(),
        (None, Some(change)) => changed(DiffChangeKind::Added, change.operation, default_operation),
        (Some(change), None) => {
            changed(DiffChangeKind::Removed, change.operation, default_operation)
        }
        (Some(source_change), Some(target_change))
            if source_change.kind == DiffChangeKind::Removed =>
        {
            changed(
                DiffChangeKind::Removed,
                source_change.operation.or(target_change.operation),
                default_operation,
            )
        }
        (Some(source_change), Some(target_change))
            if target_change.kind == DiffChangeKind::Added =>
        {
            changed(
                DiffChangeKind::Added,
                target_change.operation.or(source_change.operation),
                default_operation,
            )
        }
        (Some(source_change), Some(target_change))
            if source_change.operation.is_some()
                || target_change.operation.is_some()
                || source_change.kind != DiffChangeKind::Unchanged
                || target_change.kind != DiffChangeKind::Unchanged =>
        {
            changed(
                DiffChangeKind::Modified,
                target_change.operation.or(source_change.operation),
                default_operation,
            )
        }
        (Some(_), Some(_)) => DiffChange::unchanged(),
    }
}

/// Applies edge-specific reconciliation on top of the general input rules.
///
/// `build_unified_diff_graph` uses endpoint-pair membership and node status to
/// distinguish real path changes from neutral continuation edges introduced by
/// normalization. This prevents insertions from coloring surviving context as
/// removed while still exposing deletions and edge replacements.
fn merge_input_edge_changes(
    source_change: Option<DiffChange>,
    target_change: Option<DiffChange>,
    edge_has_unchanged_endpoints: bool,
    source_has_pair: bool,
    target_has_pair: bool,
    default_operation: DoltHashId,
) -> DiffChange {
    match (source_change, target_change) {
        // Source-only continuation edges are context when both endpoint slices
        // survive in the target. They should stay grey for insertions:
        //
        //   source: A -------- B
        //   target: A -> I -> B
        //
        // Deletions still render red because their source-only edges touch a
        // removed node:
        //
        //   source: A -> G -> B
        //   target: A ------> B
        (Some(_), None) if edge_has_unchanged_endpoints => DiffChange::unchanged(),
        (None, Some(change)) if source_has_pair => {
            merge_pair_matched_edge_change(change, default_operation)
        }
        (Some(change), None) if target_has_pair => {
            merge_pair_matched_edge_change(change, default_operation)
        }
        (source_change, target_change) => {
            merge_input_changes(source_change, target_change, default_operation)
        }
    }
}

/// Classifies a one-sided edge as a replacement when its node pair survives.
///
/// `merge_input_edge_changes` uses this for edge-identity churn between the
/// same normalized endpoints. Promoting attributed changes to `Modified`
/// avoids rendering one biological connection as an unrelated add/remove pair.
fn merge_pair_matched_edge_change(change: DiffChange, default_operation: DoltHashId) -> DiffChange {
    if change.kind == DiffChangeKind::Unchanged && change.operation.is_none() {
        DiffChange::unchanged()
    } else {
        changed(
            DiffChangeKind::Modified,
            change.operation,
            default_operation,
        )
    }
}

/// Creates an attributed non-neutral change, filling missing Dolt attribution.
///
/// Input and edge merge helpers use the comparison's default operation so
/// every colored element can be grouped by operation in downstream views.
fn changed(
    kind: DiffChangeKind,
    operation: Option<DoltHashId>,
    default_operation: DoltHashId,
) -> DiffChange {
    DiffChange::new(kind, Some(operation.unwrap_or(default_operation)))
}

/// Builds one endpoint's connected, annotated input graph from normalized nodes.
///
/// `load_block_group_comparison_graphs` uses this for both endpoints, while graph
/// reconstruction tests call it directly. It maps persisted edges onto exact
/// sequence-slice boundaries, drops edit-site markers that are not path
/// choices, adds terminal and within-span continuation topology for layout, and
/// records operation attribution separately for later source/target merging.
pub(crate) fn build_diff_input_graph(
    mut input_nodes: HashSet<GraphNode>,
    continuation_spans: &HashSet<GraphNode>,
    input_edges: &[AugmentedEdge],
    node_operations_by_id: &HashMap<HashId, DoltHashId>,
    edge_operations_by_id: &HashMap<HashId, DoltHashId>,
) -> DiffInputGraph {
    let mut graph = GenGraph::new();
    let mut node_changes = HashMap::new();
    let mut edge_changes = HashMap::new();

    input_nodes.insert(path_start_graph_node());
    input_nodes.insert(path_end_graph_node());
    for node in &input_nodes {
        graph.add_node(*node);
        node_changes.insert(
            *node,
            node_operations_by_id
                .get(&node.node_id)
                .map_or_else(DiffChange::unchanged, |operation| {
                    DiffChange::new(DiffChangeKind::Modified, Some(*operation))
                }),
        );
    }

    let blocks_by_start = input_nodes
        .iter()
        .map(|node| ((node.node_id, node.sequence_start), *node))
        .collect::<HashMap<_, _>>();
    let blocks_by_end = input_nodes
        .iter()
        .map(|node| ((node.node_id, node.sequence_end), *node))
        .collect::<HashMap<_, _>>();

    for augmented_edge in input_edges {
        // Edit-site marker edges preserve coordinates for later graph
        // operations. They are not path choices, so rendering them as diff
        // edges makes inserted/deleted sequence look unchanged.
        if augmented_edge.chromosome_index == PRESERVE_EDIT_SITE_CHROMOSOME_INDEX {
            continue;
        }
        let source_node = blocks_by_end
            .get(&(
                augmented_edge.edge.source_node_id,
                augmented_edge.edge.source_coordinate,
            ))
            .copied();
        let target_node = blocks_by_start
            .get(&(
                augmented_edge.edge.target_node_id,
                augmented_edge.edge.target_coordinate,
            ))
            .copied();
        if let (Some(source_node), Some(target_node)) = (source_node, target_node) {
            let graph_edge = GraphEdge {
                edge_id: augmented_edge.edge.id,
                source_strand: augmented_edge.edge.source_strand,
                target_strand: augmented_edge.edge.target_strand,
                chromosome_index: augmented_edge.chromosome_index,
                phased: augmented_edge.phased,
                created_on: augmented_edge.created_on,
            };
            if let Some(existing_edges) = graph.edge_weight_mut(source_node, target_node) {
                existing_edges.push(graph_edge);
            } else {
                graph.add_edge(source_node, target_node, vec![graph_edge]);
            }
            edge_changes.insert(
                GraphEdgeKey::new(source_node, target_node, graph_edge),
                edge_operations_by_id
                    .get(&graph_edge.edge_id)
                    .map_or_else(DiffChange::unchanged, |operation| {
                        DiffChange::new(DiffChangeKind::Modified, Some(*operation))
                    }),
            );
        }
    }

    add_continuation_edges(
        &input_nodes,
        continuation_spans,
        &mut graph,
        &mut edge_changes,
    );
    remove_unconnected_input_nodes(&mut graph, &mut node_changes);

    DiffInputGraph {
        graph,
        node_changes,
        edge_changes,
    }
}

/// Returns the canonical start terminal used in every input and unified graph.
///
/// The input builder and graph-layout tests use this shared value so endpoint
/// graphs remain connected to the same synthetic path boundary.
pub(crate) fn path_start_graph_node() -> GraphNode {
    GraphNode {
        node_id: PATH_START_NODE_ID,
        sequence_start: 0,
        sequence_end: 0,
    }
}

/// Returns the canonical end terminal used in every input and unified graph.
///
/// The input builder and graph-layout tests use this shared value so endpoint
/// graphs remain connected to the same synthetic path boundary.
pub(crate) fn path_end_graph_node() -> GraphNode {
    GraphNode {
        node_id: PATH_END_NODE_ID,
        sequence_start: 0,
        sequence_end: 0,
    }
}

/// Reconnects normalized slices that came from one continuous endpoint span.
///
/// `build_diff_input_graph` invokes this after mapping persisted edges.
/// It adds neutral synthetic edges only between adjacent slices covered by the
/// same original span, preserving renderable connectivity without bridging
/// genuine deletions or endpoint gaps.
fn add_continuation_edges(
    input_nodes: &HashSet<GraphNode>,
    continuation_spans: &HashSet<GraphNode>,
    graph: &mut GenGraph,
    edge_changes: &mut HashMap<GraphEdgeKey, DiffChange>,
) {
    for span in continuation_spans {
        // Delayed block construction starts from the endpoint edges and keeps
        // each endpoint's real covered spans separate before splitting them
        // into display nodes. Add neutral continuation edges only inside a
        // span that exists in this endpoint:
        //
        //   endpoint span [0..10] split to [0..3] [3..4] [4..10]
        //     add grey continuations 0..3 -> 3..4 -> 4..10
        //
        //   endpoint spans [0..3] and [4..10]
        //     do not synthesize 0..3 -> 3..4 -> 4..10
        //
        // This is the point where the graph knows enough to split display
        // nodes without asking a second question about whether a boundary came
        // from the source or target path.
        if is_terminal(span.node_id) || span.sequence_start == span.sequence_end {
            continue;
        }
        let mut nodes = input_nodes
            .iter()
            .copied()
            .filter(|node| {
                node.node_id == span.node_id
                    && node.sequence_start >= span.sequence_start
                    && node.sequence_end <= span.sequence_end
            })
            .collect::<Vec<_>>();
        nodes.sort_by_key(|node| (node.sequence_start, node.sequence_end));
        for window in nodes.windows(2) {
            let source_node = window[0];
            let target_node = window[1];
            if source_node.sequence_end != target_node.sequence_start {
                continue;
            }
            let synthetic_edge = GraphEdge {
                edge_id: HashId::convert_str(&format!(
                    "diff-continuation:{}:{}:{}:{}:{}",
                    span.node_id,
                    source_node.sequence_start,
                    source_node.sequence_end,
                    target_node.sequence_start,
                    target_node.sequence_end
                )),
                source_strand: Strand::Forward,
                target_strand: Strand::Forward,
                chromosome_index: NO_CHROMOSOME_INDEX,
                phased: 0,
                created_on: 0,
            };
            if let Some(existing_edges) = graph.edge_weight_mut(source_node, target_node) {
                if existing_edges.contains(&synthetic_edge) {
                    continue;
                }
                existing_edges.push(synthetic_edge);
            } else {
                graph.add_edge(source_node, target_node, vec![synthetic_edge]);
            }
            edge_changes.insert(
                GraphEdgeKey::new(source_node, target_node, synthetic_edge),
                DiffChange::unchanged(),
            );
        }
    }
}

/// Removes non-terminal slices that no mapped or continuation edge reaches.
///
/// `build_diff_input_graph` uses this cleanup after edge construction.
/// Coordinate normalization can produce candidate slices unsupported by actual
/// endpoint topology; pruning them keeps the unified graph from displaying
/// isolated artifacts while retaining terminal anchors.
fn remove_unconnected_input_nodes(
    graph: &mut GenGraph,
    node_changes: &mut HashMap<GraphNode, DiffChange>,
) {
    let mut connected_nodes = HashSet::new();
    for (source, target, _) in graph.all_edges() {
        connected_nodes.insert(source);
        connected_nodes.insert(target);
    }
    let disconnected_nodes = graph
        .nodes()
        .filter(|node| !is_terminal(node.node_id) && !connected_nodes.contains(node))
        .collect::<Vec<_>>();
    for node in disconnected_nodes {
        graph.remove_node(node);
        node_changes.remove(&node);
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd)]
pub(crate) struct GraphEdgeKey {
    source: GraphNode,
    target: GraphNode,
    edge_id: HashId,
    source_strand: Strand,
    target_strand: Strand,
    chromosome_index: i64,
    phased: i64,
}

impl GraphEdgeKey {
    /// Creates the stable edge identity used to union and annotate endpoint graphs.
    ///
    /// Input construction and continuation-edge insertion use this key instead
    /// of the full persisted record so volatile `created_on` metadata does not
    /// turn an otherwise identical connection into a structural diff.
    pub(crate) fn new(source: GraphNode, target: GraphNode, edge: GraphEdge) -> Self {
        Self {
            source,
            target,
            edge_id: edge.edge_id,
            source_strand: edge.source_strand,
            target_strand: edge.target_strand,
            chromosome_index: edge.chromosome_index,
            phased: edge.phased,
        }
    }

    /// Returns topology-only endpoints for pair-level edge matching.
    ///
    /// `annotated_edge_pairs` uses this projection to detect connections that
    /// survive even when their persisted edge identities differ.
    fn node_pair(&self) -> (GraphNode, GraphNode) {
        (self.source, self.target)
    }

    /// Tests whether a concrete edge has the structural identity represented by this key.
    ///
    /// `input_graph_edge_for_key` uses this after selecting a node pair because
    /// a `GenGraph` edge weight may contain multiple parallel edges.
    fn matches_edge(self, edge: GraphEdge) -> bool {
        self.edge_id == edge.edge_id
            && self.source_strand == edge.source_strand
            && self.target_strand == edge.target_strand
            && self.chromosome_index == edge.chromosome_index
            && self.phased == edge.phased
    }
}

pub(crate) struct BlockGroupComparisonGraphs {
    pub(crate) source_block_group: Option<BlockGroup>,
    pub(crate) target_block_group: Option<BlockGroup>,
    pub(crate) source_graph: DiffInputGraph,
    pub(crate) reconstructed_target_graph: DiffInputGraph,
}

pub(crate) struct DiffInputGraph {
    pub(crate) graph: GenGraph,
    pub(crate) node_changes: HashMap<GraphNode, DiffChange>,
    pub(crate) edge_changes: HashMap<GraphEdgeKey, DiffChange>,
}

#[derive(Debug, Default)]
struct InputOperationChangeLookup {
    source_node_operation_by_id: HashMap<HashId, DoltHashId>,
    target_node_operation_by_id: HashMap<HashId, DoltHashId>,
    source_edge_operation_by_id: HashMap<HashId, DoltHashId>,
    target_edge_operation_by_id: HashMap<HashId, DoltHashId>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block_group(id: i64) -> BlockGroup {
        BlockGroup {
            id: HashId::pad_str(id),
            collection_name: "collection".to_string(),
            sample_name: "sample".to_string(),
            name: "block-group".to_string(),
            created_on: 0,
            parent_block_group_id: None,
            is_default: false,
        }
    }

    #[test]
    fn test_block_group_change_kind_is_derived_from_endpoint_membership() {
        let block_group = block_group(1);
        let block_group_id = block_group.id;
        let diff = |source_block_group, target_block_group| BlockGroupDiff {
            id: block_group_id,
            source_block_group,
            target_block_group,
            graph: DiffGenGraph::new(),
        };

        assert_eq!(
            diff(None, Some(block_group.clone())).change_kind(),
            Some(BlockGroupChangeKind::Added)
        );
        assert_eq!(
            diff(Some(block_group.clone()), None).change_kind(),
            Some(BlockGroupChangeKind::Removed)
        );
        assert_eq!(
            diff(Some(block_group.clone()), Some(block_group)).change_kind(),
            Some(BlockGroupChangeKind::Modified)
        );
        assert_eq!(diff(None, None).change_kind(), None);
    }

    #[test]
    fn test_split_nodes_at_shared_boundaries_exposes_removed_slice() {
        let node_id = HashId::convert_str("reference-node");
        let source_spans = HashSet::from([GraphNode {
            node_id,
            sequence_start: 0,
            sequence_end: 10,
        }]);
        let target_spans = HashSet::from([
            GraphNode {
                node_id,
                sequence_start: 0,
                sequence_end: 3,
            },
            GraphNode {
                node_id,
                sequence_start: 5,
                sequence_end: 10,
            },
        ]);

        let (source_nodes, target_nodes) =
            split_nodes_at_shared_boundaries(&source_spans, &target_spans);
        let removed_slice = GraphNode {
            node_id,
            sequence_start: 3,
            sequence_end: 5,
        };

        assert_eq!(
            source_nodes,
            HashSet::from([
                GraphNode {
                    node_id,
                    sequence_start: 0,
                    sequence_end: 3,
                },
                removed_slice,
                GraphNode {
                    node_id,
                    sequence_start: 5,
                    sequence_end: 10,
                },
            ])
        );
        assert_eq!(target_nodes, target_spans);
        assert!(source_nodes.contains(&removed_slice));
        assert!(!target_nodes.contains(&removed_slice));
    }

    #[test]
    fn diff_graph_to_gen_graph_maps_nodes_and_edges() {
        let node_a = DiffGraphNode {
            node: GraphNode {
                node_id: HashId::pad_str(1),
                sequence_start: 0,
                sequence_end: 5,
            },
            change: DiffChange {
                kind: DiffChangeKind::Added,
                operation: Some(DoltHashId([10; 20])),
            },
        };
        let node_b = DiffGraphNode {
            node: GraphNode {
                node_id: HashId::pad_str(2),
                sequence_start: 5,
                sequence_end: 10,
            },
            change: DiffChange::unchanged(),
        };
        let mut diff_graph = DiffGenGraph::new();
        diff_graph.add_node(node_a);
        diff_graph.add_node(node_b);
        let edge = DiffGraphEdge {
            edge: GraphEdge {
                edge_id: HashId::pad_str(9),
                source_strand: Strand::Forward,
                target_strand: Strand::Forward,
                chromosome_index: 0,
                phased: 0,
                created_on: 0,
            },
            change: DiffChange {
                kind: DiffChangeKind::Added,
                operation: Some(DoltHashId([10; 20])),
            },
        };
        diff_graph.add_edge(node_a, node_b, vec![edge]);

        let graph: GenGraph = DiffGenGraphRef(&diff_graph).into();
        assert_eq!(graph.nodes().count(), 2);
        assert_eq!(graph.all_edges().count(), 1);
        let weights = graph
            .all_edges()
            .next()
            .map(|(_, _, edges)| edges.clone())
            .expect("graph edge");
        assert_eq!(weights, vec![edge.edge]);
    }
}
