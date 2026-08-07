//! A Ratatui widget to render very large graphs, originally developed for
//! the Gen version control system for graph genome sequences.

pub mod assembly;
pub mod compaction;
pub mod crawl;
pub mod cross_coordinates;
pub mod cycle_removal;
pub mod distribute_nodes;
pub mod dot_export;
pub mod edge_router;
pub mod frame_index;
pub mod geometry;
pub mod graph_painter;
pub mod graph_view;
pub mod graph_widget;
pub mod layout;
pub mod layout_engine;
pub mod navigator;
pub mod plotter;
pub mod testing;
pub mod theme;
pub mod viewport_graph;
pub mod viewport_state;
pub mod window_graph;

pub use geometry::WorldPos;
pub use graph_view::{GraphView, GraphViewState};
pub use layout::{LayoutEdge, LayoutNode, NodeRole, VisualDetail};
pub use plotter::{LineStyle, PathStyle};
pub use theme::{Theme, current_theme, set_theme};
