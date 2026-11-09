//! Primitives drawn over the character-cell buffer by bitmap renderers.

use ratatui::style::Color;

use crate::highlight_style::LineStyle;

/// One straight edge segment, in buffer cell units: the origin is the top-left corner of the
/// buffer, x grows right and y grows down, and cell `(column, row)` spans
/// `[column, column + 1) x [row, row + 1)`. Endpoints on a node sit on its border.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StraightSegment {
    pub from: (f64, f64),
    pub to: (f64, f64),
    pub color: Color,
    pub line_style: LineStyle,
}

/// A cubic edge curve in buffer cell units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BezierCurve {
    pub from: (f64, f64),
    pub control1: (f64, f64),
    pub control2: (f64, f64),
    pub to: (f64, f64),
    pub color: Color,
    pub line_style: LineStyle,
}

/// A direction arrowhead for bitmap renderers, centered on a point that need not be a cell's
/// center, in buffer cell units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArrowMark {
    pub center: (f64, f64),
    /// Direction of travel in buffer cell coordinates.
    pub direction: (f64, f64),
    pub color: Color,
}

/// The footprint of highlighted node cells (a whole node or a sub-range of one) for bitmap
/// renderers, which surround it with a glow like the one around highlighted edges. Corners are
/// in buffer cell units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodeGlow {
    pub(crate) top_left: (f64, f64),
    pub bottom_right: (f64, f64),
    /// Whether the highlight continues into unhighlighted cells of the same node on that side,
    /// where the glow must stop at the cell boundary instead of spreading past it.
    pub(crate) clip_left: bool,
    pub(crate) clip_right: bool,
    pub color: Color,
}

/// The footprint of a highlighted single-glyph node, as drawn at minimal detail, for bitmap
/// renderers. The glyph is a filled disc smaller than its cell, so a rectangular [`NodeGlow`]
/// would sit away from it; instead the renderer redraws the disc slightly larger, with the same
/// layered glow a highlighted edge has. Center in buffer cell units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SymbolGlow {
    pub center: (f64, f64),
    pub color: Color,
}

/// One graphic drawn over the character-cell buffer: lines, Bezier curves, arrows, node glows
/// and symbol glows. The cells keep the text, and these primitives accompany them in buffer
/// cell units so terminal graphics (Kitty) and PNG rendering draw the same picture at pixel
/// resolution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OverlayPrimitive {
    Line(StraightSegment),
    Bezier(BezierCurve),
    Arrow(ArrowMark),
    NodeGlow(NodeGlow),
    SymbolGlow(SymbolGlow),
}

impl BezierCurve {
    /// The same curve with every point passed through `transform`, e.g. world to terminal
    /// coordinates.
    pub(crate) fn mapped(self, transform: impl Fn((f64, f64)) -> (f64, f64)) -> Self {
        Self {
            from: transform(self.from),
            control1: transform(self.control1),
            control2: transform(self.control2),
            to: transform(self.to),
            ..self
        }
    }

    /// The same curve traversed from its other end.
    pub(crate) fn reversed(self) -> Self {
        Self {
            from: self.to,
            control1: self.control2,
            control2: self.control1,
            to: self.from,
            ..self
        }
    }
}
