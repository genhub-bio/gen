use gen_core::HashId;
use gen_tui::{plotter::PathStyle, theme::current_theme};
use ratatui::style::Color;

use crate::views::annotation_track::AnnotationSpan;

/// How a `GraphOverlay` was added, and the name/key it's addressable by (if any).
///
/// Shared by the full-screen viewer, the inline viewer, and the Jupyter widget so all
/// three manage annotation display through the same vocabulary. The TUI viewers only
/// ever construct `Track` overlays (one per loaded annotation file or annotation group);
/// `Annotation` and `Adhoc` exist for the Jupyter widget's per-annotation
/// `add_annotation`/`highlight_match` API, which the TUI viewers don't expose.
#[derive(Clone)]
pub enum OverlaySource {
    /// Loaded as a member of a named track: an annotation file or annotation group.
    Track(String),
    /// One annotation added on its own, keyed by its own name (Jupyter widget only).
    Annotation(String),
    /// A highlight with no track/annotation identity (Jupyter widget only).
    Adhoc,
}

/// A span painted onto the graph canvas as a colour highlight, with or without a label.
#[derive(Clone)]
pub struct GraphOverlay {
    pub span: AnnotationSpan,
    pub source: OverlaySource,
    pub style: PathStyle,
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
            span,
            source: OverlaySource::Track(key.to_string()),
            style,
        });
    }
}

/// Remove every overlay belonging to track `key`.
pub fn remove_track_overlays(overlays: &mut Vec<GraphOverlay>, key: &str) {
    overlays.retain(|o| !matches!(&o.source, OverlaySource::Track(k) if k == key));
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
