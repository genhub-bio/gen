//! Colors and line weights that highlight paths, edges and node cells.

use ratatui::style::Color;

use crate::geometry::WorldPos;

/// Line style for path highlighting
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LineStyle {
    /// Normal weight box-drawing characters
    Normal,
    /// Heavy weight box-drawing characters
    Bold,
    /// Dashed box-drawing characters
    Dashed,
}

/// Style specification for highlighted paths
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PathStyle {
    /// Color to use for tinting (Color::Reset means brighten instead)
    pub color: Color,
    /// Line weight style for edges and routing nodes
    pub line_style: LineStyle,
    /// Whether to merge glyphs with the glyph already drawn in the cell or replace them outright
    pub merge_glyphs: bool,
}

pub(crate) type CellHighlight = (WorldPos, (i64, i64), (i64, i64), PathStyle);

impl PathStyle {
    /// Create a new PathStyle with default settings
    pub fn new(color: Color) -> Self {
        Self {
            color,
            line_style: LineStyle::Normal,
            merge_glyphs: false,
        }
    }

    /// Set the line style
    pub fn with_line_style(mut self, line_style: LineStyle) -> Self {
        self.line_style = line_style;
        self
    }

    /// Set whether to merge glyphs
    pub fn with_merge_glyphs(mut self, merge_glyphs: bool) -> Self {
        self.merge_glyphs = merge_glyphs;
        self
    }
}
