//! A Ratatui widget to render very large graphs, originally developed for
//! the Gen version control system for graph genome sequences.

pub mod assembly;
pub mod compaction;
pub mod crawl;
pub mod cross_coordinates;
pub mod cycle_removal;
pub mod diagonal;
pub mod distribute_nodes;
pub mod dot_export;
pub mod edge_ports;
pub mod edge_router;
pub mod frame_index;
pub mod geometry;
pub mod graph_painter;
pub mod graph_view;
pub mod graph_widget;
pub mod layout;
pub mod layout_engine;
pub(crate) mod loopback_pins;
pub mod navigator;
pub mod pinning;
pub mod plotter;
pub mod png_export;
pub mod testing;
pub mod theme;
pub mod viewport_graph;
pub mod viewport_state;
pub mod window_graph;

pub use diagonal::{
    ArrowMark, BezierCurve, DeferredEdge, DiagonalDrawing, DiagonalLine, EdgeStyle, NodeGlow,
    SymbolGlow,
};
pub use edge_ports::{EdgeKey, EdgePorts, EdgeSelector, NodeSide};
pub use geometry::WorldPos;
pub use graph_view::{GraphView, GraphViewState};
pub use layout::{LayoutEdge, LayoutNode, NodeRole, PinSide, VerticalAnchor, VisualDetail};
pub use plotter::{CursorColumns, LineStyle, PathStyle};
pub use png_export::{
    DEFAULT_SCALE, MAX_SCALE, PngConfig, PngExportError, PngRenderer, color_to_rgb,
};
pub use theme::{Theme, current_theme, set_theme};
