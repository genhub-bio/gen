//! Shared layout fixtures, renderers, validation, and snapshot tests.

pub mod graph_validation;
pub mod layout_tests;
pub mod mocks;
pub mod navigation_tests;
#[cfg(test)]
mod subsetting_tests;

pub use graph_validation::{
    GraphValidationResult, assert_valid_layout_graph, validate_layout_graph,
};
pub use mocks::{MockDomainGraph, TestGraphs, TestNodeSizers, TestRenderers};
use ratatui::{Terminal, backend::TestBackend};

/// Create a test terminal with the requested dimensions.
pub fn create_test_terminal(width: u16, height: u16) -> Terminal<TestBackend> {
    let backend = TestBackend::new(width, height);
    Terminal::new(backend).expect("should create a test terminal")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_terminal_creation() {
        let terminal = create_test_terminal(80, 24);
        assert_eq!(terminal.size().unwrap().width, 80);
        assert_eq!(terminal.size().unwrap().height, 24);
    }
}
