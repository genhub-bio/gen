// Standardized testing infrastructure for the widget system
//
// This module provides a clean testing approach focused on practical layout testing:
// - Layout testing: Single layouts in isolation (algorithm testing, coordinates, visual regression)
// - Mock infrastructure: Standardized graphs, node sizers, and renderers for consistent testing
// - Utility functions: Common testing patterns and quick verification methods

pub mod edge_label_validation_test;
pub mod graph_validation;
pub mod layout_tests;
pub mod mocks;

// Re-export main testing APIs
pub use graph_validation::{
    GraphValidationResult, assert_valid_layout_graph, validate_layout_graph,
};
pub use mocks::{MockDomainGraph, TestGraphs, TestNodeSizers, TestRenderers};
use ratatui::{Terminal, backend::TestBackend};

// Common test utilities
use crate::graph_controller::ViewportState;

/// Common assertion utilities for testing
pub struct TestAssertions;

impl TestAssertions {
    /// Assert that two rendered outputs are visually different
    pub fn assert_different_layouts(output1: &str, output2: &str) {
        assert_ne!(
            output1.trim(),
            output2.trim(),
            "Expected different visual outputs but they were identical"
        );
    }

    /// Assert that rendered output contains expected content
    pub fn assert_contains_content(output: &str, expected: &str) {
        assert!(
            output.contains(expected),
            "Expected output to contain '{}', but it didn't.\nActual output:\n{}",
            expected,
            output
        );
    }

    /// Assert that coordinates are within expected bounds
    pub fn assert_coordinates_in_bounds(coords: (i64, i64), min: (i64, i64), max: (i64, i64)) {
        assert!(
            coords.0 >= min.0 && coords.0 <= max.0 && coords.1 >= min.1 && coords.1 <= max.1,
            "Coordinates {:?} are outside expected bounds {:?} - {:?}",
            coords,
            min,
            max
        );
    }
}

/// Create a standardized test terminal with specified dimensions
pub fn create_test_terminal(width: u16, height: u16) -> Terminal<TestBackend> {
    let backend = TestBackend::new(width, height);
    Terminal::new(backend).expect("Failed to create test terminal")
}

/// Create a standardized test viewport state
pub fn create_test_viewport(width: u16, height: u16) -> ViewportState {
    let mut state = ViewportState::new();
    state.viewport_bounds = ratatui::layout::Rect::new(0, 0, width, height);
    state.focus(); // Enable focus for testing interactions
    state
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assertion_utilities() {
        // Test different layouts assertion
        TestAssertions::assert_different_layouts("layout A", "layout B");

        // Test contains content assertion
        TestAssertions::assert_contains_content("Hello World", "World");

        // Test coordinates bounds assertion
        TestAssertions::assert_coordinates_in_bounds((5, 5), (0, 0), (10, 10));
    }

    #[test]
    fn test_terminal_creation() {
        let terminal = create_test_terminal(80, 24);
        assert_eq!(terminal.size().unwrap().width, 80);
        assert_eq!(terminal.size().unwrap().height, 24);
    }

    #[test]
    fn test_viewport_creation() {
        let viewport = create_test_viewport(100, 50);
        assert_eq!(viewport.viewport_bounds.width, 100);
        assert_eq!(viewport.viewport_bounds.height, 50);
        assert!(viewport.has_focus);
    }
}
