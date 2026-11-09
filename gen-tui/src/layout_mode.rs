//! The mode that selects rectilinear or curved window layout and drawing.

/// Selects how a window is laid out and drawn. The mode coordinates edge routing, layer spacing,
/// compaction and rendering together rather than only changing how edges look.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum LayoutMode {
    /// Axis-aligned routed edges drawn with box-drawing characters.
    #[default]
    Rectilinear,
    /// Cubic Bézier curves with horizontal entry and exit tangents, bypassing the edge router.
    Curved,
}

impl LayoutMode {
    /// Whether edges skip the router.
    pub const fn is_curved(self) -> bool {
        matches!(self, Self::Curved)
    }
}
