use std::collections::HashMap;

use gen_core::{HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, PathBlock};
use gen_graph::{GenGraph, GraphNode, project_path};
use gen_tui::{plotter::PathStyle, theme::current_theme};
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

/// Project `path_blocks` onto `graph` and return the route's non-terminal nodes, dropping
/// the synthetic path start/end nodes.
///
/// Shared by every viewer that turns a stored `Path` into a path overlay. Returns an empty
/// vector when the path does not project onto any real node in the current graph state;
/// callers decide how to report that (each viewer has its own error type).
pub fn project_path_overlay_nodes(graph: &GenGraph, path_blocks: &[PathBlock]) -> Vec<GraphNode> {
    project_path(graph, path_blocks)
        .into_iter()
        .filter(|(node, _)| node.node_id != PATH_START_NODE_ID && node.node_id != PATH_END_NODE_ID)
        .map(|(node, _)| node)
        .collect()
}

/// How a `GraphOverlay` was added, and the name/key it's addressable by (if any).
///
/// Shared by the full-screen viewer, the inline viewer, and the Jupyter widget so all
/// three manage overlay display through the same vocabulary. The TUI viewers only
/// ever construct `Track` overlays (one per loaded annotation file or annotation group)
/// plus the single `Path` overlay; `Annotation` and `Adhoc` exist for the Jupyter
/// widget's per-annotation `add_annotation`/`highlight_match` API, which the TUI viewers
/// don't expose.
#[derive(Clone)]
pub enum OverlaySource {
    /// Loaded as a member of a named track: an annotation file or annotation group.
    Track(String),
    /// One annotation added on its own, keyed by its own name (Jupyter widget only).
    Annotation(String),
    /// A highlight with no track/annotation identity (Jupyter widget only).
    Adhoc,
    /// The current path highlight. At most one path overlay is present at a time.
    Path,
}

/// What a `GraphOverlay` paints.
///
/// A path is just another overlay: at the controller level it and an annotation span
/// are both entries in one highlight list (a `HighlightKind::Path` vs `Cells`/`Edge`),
/// so both are repainted the same way every frame. They differ only in payload — a span
/// is a labelled sub-node byte range, a path is a route through whole nodes.
#[derive(Clone)]
pub enum OverlayContent {
    /// A labelled annotation span, painted as a sub-node cell highlight.
    Span(AnnotationSpan),
    /// A route through whole nodes, painted as a connected path highlight.
    Path(Vec<GraphNode>),
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
    pub fn path_nodes(&self) -> Option<&[GraphNode]> {
        match &self.content {
            OverlayContent::Path(nodes) => Some(nodes),
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
        let style = PathStyle::new(stable_span_color(&span));
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

/// Replace the current path overlay (if any) with `path_nodes` styled by `style`.
pub fn set_path_overlay(
    overlays: &mut Vec<GraphOverlay>,
    style: PathStyle,
    path_nodes: Vec<GraphNode>,
) {
    remove_path_overlay(overlays);
    overlays.push(GraphOverlay {
        content: OverlayContent::Path(path_nodes),
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
