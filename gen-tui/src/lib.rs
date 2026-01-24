//! A Ratatui widget to render very large graphs, originally developed for
//! the Gen version control system for graph genome sequences.

pub mod animation;
pub mod color_utils;
pub mod cursor;
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
pub mod theme;
pub mod viewport_state;

pub mod testing;
pub mod viewport_graph;

// Tests that are no longer relevant (involve deprecated infrastructure):
// #[cfg(test)]
// mod cursor_test;
//
// #[cfg(test)]
// mod layer_navigation_test;
//
// #[cfg(test)]
// mod cursor_partition_test;
//
// #[cfg(test)]
// mod test_cursor_positioning;
//
// #[cfg(test)]
// mod test_cursor_restoration;
//
// #[cfg(test)]
// mod partition_bundle_test;

pub use graph_algorithms::find_articulation_points;
pub use graph_controller::GraphController;
pub use graph_widget::GraphWidget;
pub use layout::{LayoutEdge, LayoutEngine, LayoutNode};
pub use partition::{PartitionEdge, PartitionNode};
pub use partition_controller::PartitionController;
