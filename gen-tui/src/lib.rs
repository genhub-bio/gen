//! A Ratatui widget to render very large graphs, originally developed for
//! the Gen version control system for graph genome sequences.

pub mod assembly;
pub mod compaction;
pub mod crawl;
pub mod cross_coordinates;
pub mod curved;
pub mod cycle_removal;
pub mod dot_export;
pub mod edge_ports;
pub mod edge_router;
pub mod frame_index;
pub mod geometry;
pub mod graph_painter;
pub mod graph_view;
#[cfg(feature = "crossterm")]
pub mod kitty_graphics;
pub mod layout;
pub mod layout_compaction;
pub mod layout_engine;
pub(crate) mod loopback_pins;
pub mod navigator;
pub mod pinning;
pub mod pixel_overlay;
pub mod plotter;
pub mod png_export;
pub mod routed_window_graph;
pub mod testing;
pub mod theme;
pub mod viewport_state;
pub mod window_geometry;
pub mod window_graph;

pub use curved::{
    ArrowMark, BezierCurve, LayoutMode, NodeGlow, OverlayPrimitive, StraightSegment, SymbolGlow,
};
pub use edge_ports::{EdgePorts, EdgeSelector, NodeSide, PortedEdge};
pub use geometry::WorldPos;
pub use graph_view::{GraphView, GraphViewState};
pub use layout::{LayoutEdge, LayoutNode, NodeRole, PinSide, VerticalAnchor, VisualDetail};
pub use plotter::{CursorColumns, LineStyle, PathStyle};
pub use png_export::{
    DEFAULT_SCALE, MAX_SCALE, PngExportError, PngRenderer, RasterConfig, color_to_rgb,
};
pub use theme::{Theme, current_theme, set_theme};
