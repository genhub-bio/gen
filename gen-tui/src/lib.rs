//! A Ratatui widget to render very large graphs, originally developed for
//! the Gen version control system for graph genome sequences.

pub mod animation;
pub mod cursor;
pub mod distribute_nodes;
pub mod dot_export;
pub mod edge_router; // Rust port of edge routing
pub mod geometry;
pub mod graph_algorithms;
pub mod graph_controller;
pub mod graph_widget;
pub mod layout;
pub mod partition;
pub mod partition_controller;
pub mod partition_table;
pub mod plotter;
pub mod testing;
pub mod theme;
pub mod viewport_graph;
pub mod viewport_state;

pub use geometry::{WorldPos, WorldRect};
pub use graph_algorithms::find_articulation_points;
pub use graph_controller::GraphController;
pub use graph_widget::GraphWidget;
pub use layout::{LayoutEdge, LayoutEngine, LayoutNode, NodeRole, VisualDetail};
pub use partition::{PartitionEdge, PartitionNode};
pub use partition_controller::PartitionController;
pub use plotter::{LineStyle, PathStyle};
pub use theme::{Theme, current_theme, set_theme};
pub use viewport_graph::CroppedGraph;
pub use viewport_state::ViewportState;
