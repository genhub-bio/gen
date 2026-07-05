use gen_core::HashId;
use gen_graph::GraphNode;
use gen_tui::{plotter::PathStyle, theme::current_theme};
use ratatui::style::Color;

use crate::views::annotation_track::AnnotationSpan;

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
