//! A Ratatui widget to render very large graphs, originally developed for
//! the Gen version control system for graph genome sequences.

pub(crate) mod assembly;
pub(crate) mod compaction;
pub mod crawl;
pub(crate) mod cross_coordinates;
pub mod curved_edges;
pub(crate) mod cycle_removal;
pub mod edge_ports;
pub(crate) mod edge_router;
pub mod frame_index;
pub mod geometry;
pub mod graph_painter;
pub mod graph_view;
pub mod highlight_style;
#[cfg(feature = "crossterm")]
pub mod kitty_graphics;
pub mod layout;
pub mod layout_compaction;
pub mod layout_engine;
pub mod layout_mode;
pub(crate) mod loopback_pins;
pub(crate) mod navigator;
pub mod node_renderer;
pub mod overlay_primitive;
pub(crate) mod pinning;
pub mod pixel_overlay;
pub mod plotter;
pub mod raster_config;
pub(crate) mod routed_window_graph;
pub mod testing;
pub mod theme;
pub mod window_geometry;
pub(crate) mod window_graph;
pub mod world_buffer;

pub use edge_ports::{EdgePorts, EdgeSelector, NodeSide, PortedEdge};
pub use geometry::WorldPos;
pub use graph_view::{GraphView, GraphViewState};
pub use highlight_style::{LineStyle, PathStyle};
pub use layout::{LayoutEdge, LayoutNode, NodeRole, PinSide, VerticalAnchor, VisualDetail};
pub use layout_mode::LayoutMode;
pub use node_renderer::CursorColumns;
pub use overlay_primitive::{
    ArrowMark, BezierCurve, NodeGlow, OverlayPrimitive, StraightSegment, SymbolGlow,
};
pub use raster_config::{RasterConfig, color_to_rgb};
pub use theme::{Theme, current_theme, set_theme};
