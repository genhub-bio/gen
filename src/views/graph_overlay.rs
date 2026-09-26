use std::collections::{HashMap, HashSet};

use gen_core::{HashId, is_terminal};
use gen_graph::{GenGraph, GraphNode};
use gen_models::{db::GraphConnection, edge::Edge, path::Path};
use gen_tui::{
    plotter::{LineStyle, PathStyle},
    theme::current_theme,
};
use itertools::Itertools;
use ratatui::style::Color;

use crate::views::annotation_track::AnnotationSpan;

/// Remembers the color last chosen for each annotation id by the greedy conflict-avoiding
/// assignment in `gen_graph_widget::reapply_overlays`. That pass reruns every frame (the
/// live TUI viewers repaint whenever the overlay set changes with scrolling), so without
/// this cache an annotation's color could reshuffle between frames even when nothing near
/// it actually changed. Owned alongside the `Vec<GraphOverlay>` it colors, by whichever
/// viewer (full-screen, inline, Jupyter) owns that list.
#[derive(Clone, Default)]
pub struct AnnotationColorCache {
    colors: HashMap<HashId, Color>,
    /// Cursor into the theme accent slots, advanced each time a never-before-seen
    /// annotation needs a color. A simple rotation, rather than deriving a color from the
    /// annotation's id hash, means two never-conflicting annotations seen back to back
    /// reliably get different colors instead of occasionally landing on the same hash.
    next_index: usize,
}

impl AnnotationColorCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn get(&self, id: &HashId) -> Option<Color> {
        self.colors.get(id).copied()
    }

    pub(crate) fn set(&mut self, id: HashId, color: Color) {
        self.colors.insert(id, color);
    }

    /// The next color in rotation through `accents`, advancing the cursor so the color
    /// after it is a fresh one next time.
    pub(crate) fn next_color(&mut self, accents: &[Color; 8]) -> Color {
        let color = accents[self.next_index % accents.len()];
        self.next_index += 1;
        color
    }
}

/// A stored path reduced to what locating it in a partially loaded graph needs: the ids of
/// the edges it runs along, and the node ranges it covers between consecutive edges.
///
/// Every viewer's path highlight is built from this. The lazy viewers only hold the part of a
/// block group the crawl has reached, so rather than projecting the whole path (which needs
/// the whole graph, and a path-only graph fragment would carve nodes differently from the
/// crawl), the highlight is re-derived from whatever is loaded each time overlays are
/// reapplied. Loading more of the graph therefore extends the highlight without refetching
/// the path.
#[derive(Clone, Debug, Default)]
pub struct PathMembership {
    edge_ids: HashSet<HashId>,
    node_ranges: HashMap<HashId, Vec<(i64, i64)>>,
}

/// The part of a path present in a loaded graph: its non-sentinel nodes, and the graph edges
/// between them that the path runs along.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LoadedPathRoute {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<(GraphNode, GraphNode)>,
}

impl PathMembership {
    /// Fetch `path_id`'s edges once and index them for membership checks.
    pub fn load(conn: &GraphConnection, path_id: &HashId, history_ref: Option<&str>) -> Self {
        Self::from_edges(&Path::edges_for_path(conn, path_id, history_ref))
    }

    /// Index a path's ordered edges. The range a path covers on a node runs from where one
    /// edge arrives to where the next one leaves, as in `Path::coordinate_blocks`.
    pub fn from_edges(edges: &[Edge]) -> Self {
        let mut node_ranges: HashMap<HashId, Vec<(i64, i64)>> = HashMap::new();
        for (into, out_of) in edges.iter().tuple_windows() {
            node_ranges
                .entry(into.target_node_id)
                .or_default()
                .push((into.target_coordinate, out_of.source_coordinate));
        }
        Self {
            edge_ids: edges.iter().map(|edge| edge.id).collect(),
            node_ranges,
        }
    }

    /// Whether the path has no edges at all.
    pub fn is_empty(&self) -> bool {
        self.edge_ids.is_empty()
    }

    /// Whether a single path range covers the slice of `node_id` from `start` to `end`.
    fn covers(&self, node_id: HashId, start: i64, end: i64) -> bool {
        self.node_ranges.get(&node_id).is_some_and(|ranges| {
            ranges
                .iter()
                .any(|&(range_start, range_end)| range_start <= start && end <= range_end)
        })
    }

    /// Whether the graph edge `source -> target` joins two adjacent slices of the same node
    /// inside a range the path runs through. Such an edge carries the id of whatever edge
    /// split the node rather than a path edge, but the path still continues across it.
    fn continues_within_node(&self, source: GraphNode, target: GraphNode) -> bool {
        source.node_id == target.node_id
            && source.sequence_end == target.sequence_start
            && self.covers(source.node_id, source.sequence_start, target.sequence_end)
    }

    /// The part of this path present in `graph`: graph edges carrying one of the path's edge
    /// ids or continuing the path within a node, and the nodes they connect. Sentinel nodes
    /// and the edges touching them are left out, since they are never drawn as path content.
    pub fn loaded_route(&self, graph: &GenGraph) -> LoadedPathRoute {
        let mut route = LoadedPathRoute::default();
        let mut seen_nodes: HashSet<GraphNode> = HashSet::new();
        for (source, target, graph_edges) in graph.all_edges() {
            let on_path = graph_edges
                .iter()
                .any(|graph_edge| self.edge_ids.contains(&graph_edge.edge_id))
                || self.continues_within_node(source, target);
            if !on_path {
                continue;
            }
            for node in [source, target] {
                if !is_terminal(node.node_id) && seen_nodes.insert(node) {
                    route.nodes.push(node);
                }
            }
            if !is_terminal(source.node_id) && !is_terminal(target.node_id) {
                route.edges.push((source, target));
            }
        }
        route
    }
}

/// How a `GraphOverlay` was added, and the name/key it's addressable by (if any).
///
/// Shared by the full-screen viewer, the inline viewer, and the Jupyter widget so all
/// three manage overlay display through the same vocabulary. TUI viewers construct
/// `Track` overlays for loaded annotation files/groups, the single `Path` overlay, and
/// `Search` overlays for explicit region-search highlights. `Adhoc` remains available
/// for Jupyter highlights, while `Annotation` is that widget's keyed annotation API.
#[derive(Clone)]
pub enum OverlaySource {
    /// Loaded as a member of a named track: an annotation file or annotation group.
    Track(String),
    /// One annotation added on its own, keyed by its own name (Jupyter widget only).
    Annotation(String),
    /// A highlight with no track/annotation identity, such as a Jupyter ad-hoc annotation.
    Adhoc,
    /// The currently selected TUI region-search highlight.
    Search,
    /// The current path highlight. At most one path overlay is present at a time.
    Path,
}

impl OverlaySource {
    /// Whether this overlay is an annotation, as opposed to a search result or a path.
    pub fn is_annotation(&self) -> bool {
        matches!(self, Self::Track(_) | Self::Annotation(_) | Self::Adhoc)
    }
}

/// What a `GraphOverlay` paints.
///
/// An annotation span supplies bars, connectors, and labels; a path highlights the loaded
/// nodes and edges it runs along. Both are managed together and repainted every frame.
#[derive(Clone)]
pub enum OverlayContent {
    /// An annotation span, painted as bars, connectors, and labels.
    Span(AnnotationSpan),
    /// A stored path, painted over whichever of its nodes and edges are loaded.
    Path(PathMembership),
}

/// Something painted onto the graph canvas as a colour highlight: an annotation span
/// (with or without a label) or the current path.
#[derive(Clone)]
pub struct GraphOverlay {
    pub content: OverlayContent,
    pub source: OverlaySource,
    pub style: PathStyle,
}

impl GraphOverlay {
    /// The annotation span this overlay paints, or `None` if it is a path overlay.
    pub fn span(&self) -> Option<&AnnotationSpan> {
        match &self.content {
            OverlayContent::Span(span) => Some(span),
            OverlayContent::Path(_) => None,
        }
    }

    /// The path this overlay paints, or `None` if it is a span overlay.
    pub fn path(&self) -> Option<&PathMembership> {
        match &self.content {
            OverlayContent::Path(path) => Some(path),
            OverlayContent::Span(_) => None,
        }
    }
}

/// Colour an annotation span from a hash of its own id, so a given annotation's colour
/// is stable regardless of load order or how many other annotations are in view (only 8
/// accent colours exist, so repeats across unrelated annotations are expected).
pub fn stable_span_color(span: &AnnotationSpan) -> Color {
    current_theme()[0x08 + (span.id.0[0] as usize % 8)]
}

/// Replace every overlay belonging to track `key` with freshly loaded `spans`, each
/// coloured from its own stable per-id hash.
pub fn replace_track_overlays(
    overlays: &mut Vec<GraphOverlay>,
    key: &str,
    spans: Vec<AnnotationSpan>,
) {
    remove_track_overlays(overlays, key);
    for span in spans {
        let style = PathStyle::new(stable_span_color(&span))
            .with_line_style(LineStyle::Bold)
            .with_merge_glyphs(true);
        overlays.push(GraphOverlay {
            content: OverlayContent::Span(span),
            source: OverlaySource::Track(key.to_string()),
            style,
        });
    }
}

/// Remove every overlay belonging to track `key`.
pub fn remove_track_overlays(overlays: &mut Vec<GraphOverlay>, key: &str) {
    overlays.retain(|o| !matches!(&o.source, OverlaySource::Track(k) if k == key));
}

/// Replace the current path overlay (if any) with `path` styled by `style`.
pub fn set_path_overlay(overlays: &mut Vec<GraphOverlay>, style: PathStyle, path: PathMembership) {
    remove_path_overlay(overlays);
    overlays.push(GraphOverlay {
        content: OverlayContent::Path(path),
        source: OverlaySource::Path,
        style,
    });
}

/// Remove the current path overlay, if any.
pub fn remove_path_overlay(overlays: &mut Vec<GraphOverlay>) {
    overlays.retain(|o| !matches!(o.source, OverlaySource::Path));
}

/// Whether a path overlay is currently present.
pub fn has_path_overlay(overlays: &[GraphOverlay]) -> bool {
    overlays
        .iter()
        .any(|o| matches!(o.source, OverlaySource::Path))
}

/// Track key for an annotation file, loaded via the TUI viewers' sidebar file toggle.
pub fn file_track_key(id: &HashId) -> String {
    format!("file:{id}")
}

/// Track key for an annotation group, loaded via the TUI viewers' sidebar group toggle
/// or auto-loaded for the current viewport.
pub fn group_track_key(group_id: &str) -> String {
    format!("group:{group_id}")
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use gen_core::{HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand};
    use gen_graph::{GenGraph, GraphEdge, GraphNode};
    use gen_models::{db::get_connection, edge::Edge, path::Path};
    use gen_tui::layout_engine::LayoutEngine;

    use super::{LoadedPathRoute, PathMembership};
    use crate::views::lazy_graph_source::{
        SqlGraphSource, seed_block_group_graph,
        tests::{setup_circular_block_group, setup_labelled_chain_block_group},
    };

    fn slice(label: &str, sequence_start: i64, sequence_end: i64) -> GraphNode {
        GraphNode {
            node_id: HashId::convert_str(label),
            sequence_start,
            sequence_end,
        }
    }

    fn sentinel(node_id: HashId) -> GraphNode {
        GraphNode {
            node_id,
            sequence_start: 0,
            sequence_end: 0,
        }
    }

    fn edge(name: &str, source: (HashId, i64), target: (HashId, i64)) -> Edge {
        Edge {
            id: HashId::convert_str(name),
            source_node_id: source.0,
            source_coordinate: source.1,
            source_strand: Strand::Forward,
            target_node_id: target.0,
            target_coordinate: target.1,
            target_strand: Strand::Forward,
        }
    }

    fn connect(graph: &mut GenGraph, source: GraphNode, target: GraphNode, edge: &Edge) {
        graph.add_edge(
            source,
            target,
            vec![GraphEdge {
                edge_id: edge.id,
                source_strand: edge.source_strand,
                target_strand: edge.target_strand,
                chromosome_index: 0,
                phased: 0,
                created_on: 0,
            }],
        );
    }

    fn route_sets(
        route: &LoadedPathRoute,
    ) -> (HashSet<GraphNode>, HashSet<(GraphNode, GraphNode)>) {
        (
            route.nodes.iter().copied().collect(),
            route.edges.iter().copied().collect(),
        )
    }

    #[test]
    fn test_loaded_route_follows_path_edges_and_continues_within_split_nodes() {
        let [x, y, z, w] = ["x", "y", "z", "w"].map(HashId::convert_str);
        let start_to_x = edge("start-x", (PATH_START_NODE_ID, 0), (x, 0));
        let x_to_y = edge("x-y", (x, 5), (y, 0));
        let y_to_z = edge("y-z", (y, 5), (z, 0));
        let z_to_end = edge("z-end", (z, 5), (PATH_END_NODE_ID, 0));
        let y_split = edge("y-split", (y, 2), (y, 2));
        let y_to_w = edge("y-w", (y, 2), (w, 0));
        let w_to_z = edge("w-z", (w, 5), (z, 0));
        let y_repeat = edge("y-repeat", (y, 5), (y, 0));

        let x_slice = slice("x", 0, 5);
        let y_left = slice("y", 0, 2);
        let y_right = slice("y", 2, 5);
        let z_slice = slice("z", 0, 5);
        let w_slice = slice("w", 0, 5);
        let mut graph = GenGraph::new();
        connect(
            &mut graph,
            sentinel(PATH_START_NODE_ID),
            x_slice,
            &start_to_x,
        );
        connect(&mut graph, x_slice, y_left, &x_to_y);
        connect(&mut graph, y_left, y_right, &y_split);
        connect(&mut graph, y_right, z_slice, &y_to_z);
        connect(&mut graph, z_slice, sentinel(PATH_END_NODE_ID), &z_to_end);
        connect(&mut graph, y_left, w_slice, &y_to_w);
        connect(&mut graph, w_slice, z_slice, &w_to_z);
        connect(&mut graph, y_right, y_left, &y_repeat);

        let membership = PathMembership::from_edges(&[start_to_x, x_to_y, y_to_z, z_to_end]);
        let (nodes, edges) = route_sets(&membership.loaded_route(&graph));

        assert_eq!(
            nodes,
            HashSet::from([x_slice, y_left, y_right, z_slice]),
            "the branch node and the sentinels should not be highlighted"
        );
        assert_eq!(
            edges,
            HashSet::from([(x_slice, y_left), (y_left, y_right), (y_right, z_slice)]),
            "the path should continue across y's split, but not along the branch, the repeat \
             edge back into y, or the sentinel attachments"
        );
    }

    #[test]
    fn test_loaded_route_covers_only_loaded_edges_and_extends_with_new_batches() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("graph.db");
        let labels = ["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9"];
        let (block_group_id, edge_ids) = setup_labelled_chain_block_group(&db_path, &labels);
        let conn = get_connection(&db_path).unwrap();
        let path = Path::create(&conn, "chain", &block_group_id, &edge_ids).unwrap();
        let membership = PathMembership::load(&conn, &path.id, None);

        let seed = seed_block_group_graph(&conn, &block_group_id);
        let mut engine = LayoutEngine::new_with_source(
            seed,
            SqlGraphSource::new(db_path.clone(), block_group_id),
        );
        engine
            .activate_batch_containing(sentinel(PATH_START_NODE_ID), 3)
            .expect("should claim a first batch around PATH_START");

        let chain_edges: HashSet<(GraphNode, GraphNode)> = labels
            .windows(2)
            .map(|pair| (slice(pair[0], 0, 5), slice(pair[1], 0, 5)))
            .collect();
        let (_, first_edges) = route_sets(&membership.loaded_route(engine.graph()));
        assert!(!first_edges.is_empty());
        assert!(first_edges.is_subset(&chain_edges));
        assert!(
            first_edges.len() < chain_edges.len(),
            "a partial crawl should only highlight the path edges it has loaded"
        );
        for (source, target) in &first_edges {
            assert!(engine.graph().contains_edge(*source, *target));
        }

        let next_anchor = engine
            .graph()
            .nodes()
            .find(|node| engine.batch_of(*node).is_none())
            .expect("the crawl should have loaded a neighbour outside the first batch");
        let nodes_before = engine.graph().node_count();
        engine
            .activate_batch_containing(next_anchor, 3)
            .expect("should claim a second batch");
        assert!(engine.graph().node_count() > nodes_before);

        let (_, second_edges) = route_sets(&membership.loaded_route(engine.graph()));
        assert!(second_edges.is_subset(&chain_edges));
        assert!(
            first_edges.is_subset(&second_edges) && second_edges.len() > first_edges.len(),
            "the same membership should extend over the newly loaded batch"
        );
    }

    #[test]
    fn test_loaded_route_on_circular_seed_does_not_panic() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("graph.db");
        let (block_group_id, edge_ids) = setup_circular_block_group(&db_path);
        let conn = get_connection(&db_path).unwrap();
        let path = Path::create(&conn, "linear", &block_group_id, &edge_ids).unwrap();
        let membership = PathMembership::load(&conn, &path.id, None);

        let seed = seed_block_group_graph(&conn, &block_group_id);
        assert!(
            !seed.nodes().any(|node| node.node_id == PATH_START_NODE_ID),
            "a circular seed has no PATH_START to walk from"
        );
        assert_eq!(membership.loaded_route(&seed), LoadedPathRoute::default());

        let mut engine =
            LayoutEngine::new_with_source(seed, SqlGraphSource::new(db_path, block_group_id));
        engine
            .activate_batch_containing(slice("x", 0, 5), 10)
            .expect("should crawl the circular block group");
        let (_, edges) = route_sets(&membership.loaded_route(engine.graph()));
        assert_eq!(
            edges,
            HashSet::from([
                (slice("x", 0, 5), slice("y", 0, 5)),
                (slice("y", 0, 5), slice("z", 0, 5)),
            ]),
            "the real z -> x closure is not part of the stored path"
        );
    }
}
