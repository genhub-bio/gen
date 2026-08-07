// Standardized mock objects for consistent testing across the widget system

use std::collections::HashMap;

use petgraph::{graph::NodeIndex, stable_graph::StableDiGraph, visit::GraphBase};
use ratatui::style::{Color, Style};

use crate::{
    geometry::WorldRect, graph_widget::NODE_GLYPH, layout::VisualDetail, plotter::NodeRenderer,
    viewport_state::WorldBuffer,
};

// Type alias for the common test graph type
pub type MockDomainGraph = StableDiGraph<(), ()>;
pub type MockNodeId = NodeIndex<u32>;

/// Test-only mirror of the production `NodeRenderer`, carrying the detail level as an
/// explicit argument rather than internal state. Implementors typically override only
/// the sizing methods or only `render_node`; [`MockVisual`] composes two implementors
/// so a test can vary sizing and rendering independently.
pub trait MockRenderer<G: GraphBase> {
    fn get_node_size(&self, node: &G::NodeId, detail_level: VisualDetail) -> (u64, u64) {
        let _ = (node, detail_level);
        (1, 1)
    }

    fn get_dummy_size(&self) -> (u64, u64) {
        (1, 1)
    }

    fn render_node(
        &self,
        buffer: &mut WorldBuffer,
        area: WorldRect,
        node_id: &G::NodeId,
        detail_level: VisualDetail,
    ) {
        let _ = (buffer, area, node_id, detail_level);
    }
}

/// Combines a sizing-focused and a rendering-focused [`MockRenderer`] into a single
/// [`NodeRenderer`], threading a stored detail level into both.
pub struct MockVisual<S, R> {
    pub sizer: S,
    pub renderer: R,
    pub detail: VisualDetail,
}

impl<S, R> MockVisual<S, R> {
    pub fn new(sizer: S, renderer: R) -> Self {
        Self {
            sizer,
            renderer,
            detail: VisualDetail::Truncated,
        }
    }
}

impl<G, S, R> NodeRenderer<G> for MockVisual<S, R>
where
    G: GraphBase,
    S: MockRenderer<G>,
    R: MockRenderer<G>,
{
    fn get_node_size(&self, node: &G::NodeId) -> (u64, u64) {
        self.sizer.get_node_size(node, self.detail)
    }

    fn get_dummy_size(&self) -> (u64, u64) {
        self.sizer.get_dummy_size()
    }

    fn render_node(&self, buffer: &mut WorldBuffer, area: WorldRect, node_id: &G::NodeId) {
        self.renderer
            .render_node(buffer, area, node_id, self.detail);
    }
}

/// Collection of standardized test graphs for consistent testing
pub struct TestGraphs;

impl TestGraphs {
    /// Create a corresponding domain graph for testing (simple chain)
    pub fn domain_simple_chain() -> MockDomainGraph {
        let mut graph = MockDomainGraph::new();
        let a = graph.add_node(());
        let b = graph.add_node(());
        let c = graph.add_node(());

        graph.add_edge(a, b, ());
        graph.add_edge(b, c, ());

        graph
    }

    /// Create a corresponding domain graph for testing (diamond)
    pub fn domain_diamond() -> MockDomainGraph {
        let mut graph = MockDomainGraph::new();
        let a = graph.add_node(());
        let b = graph.add_node(());
        let c = graph.add_node(());
        let d = graph.add_node(());

        // Diamond structure: A -> B -> D, A -> C -> D
        graph.add_edge(a, b, ());
        graph.add_edge(a, c, ());
        graph.add_edge(b, d, ());
        graph.add_edge(c, d, ());

        graph
    }

    /// Create a complex DAG with multiple levels and branches (domain graph)
    pub fn domain_complex_dag() -> MockDomainGraph {
        let mut graph = MockDomainGraph::new();
        let nodes: Vec<_> = (0..10).map(|_| graph.add_node(())).collect();

        // Create the same structure as the complex DAG in mocks, but extended
        graph.add_edge(nodes[0], nodes[1], ());
        graph.add_edge(nodes[0], nodes[2], ());
        graph.add_edge(nodes[1], nodes[3], ());
        graph.add_edge(nodes[1], nodes[4], ());
        graph.add_edge(nodes[2], nodes[4], ());
        graph.add_edge(nodes[2], nodes[5], ());
        graph.add_edge(nodes[3], nodes[6], ());
        graph.add_edge(nodes[4], nodes[6], ());
        graph.add_edge(nodes[4], nodes[7], ());
        graph.add_edge(nodes[5], nodes[7], ());

        // Add new nodes 8 and 9
        graph.add_edge(nodes[6], nodes[8], ());
        graph.add_edge(nodes[7], nodes[8], ());
        graph.add_edge(nodes[8], nodes[9], ());

        graph
    }

    /// Create an extended diamond structure (domain graph)
    pub fn domain_extended_diamond() -> MockDomainGraph {
        let mut graph = MockDomainGraph::new();
        let nodes: Vec<_> = (0..8).map(|_| graph.add_node(())).collect();

        // Extended diamond structure:
        // 0 -> {1, 2} -> 3 -> {4, 5} -> 6 -> 7
        // This creates multiple diamond patterns in sequence

        // First diamond: 0 -> {1, 2} -> 3
        graph.add_edge(nodes[0], nodes[1], ());
        graph.add_edge(nodes[0], nodes[2], ());
        graph.add_edge(nodes[1], nodes[3], ());
        graph.add_edge(nodes[2], nodes[3], ());

        // Second diamond: 3 -> {4, 5} -> 6
        graph.add_edge(nodes[3], nodes[4], ());
        graph.add_edge(nodes[3], nodes[5], ());
        graph.add_edge(nodes[4], nodes[6], ());
        graph.add_edge(nodes[5], nodes[6], ());

        // Final connection: 6 -> 7
        graph.add_edge(nodes[6], nodes[7], ());

        graph
    }

    /// Creates a graph with layer-skipping connections for testing edge routing
    /// Structure: 0 -> {1, 3}, 1 -> 3, 3 -> {4, 5} -> 6 -> 7
    pub fn domain_skip_layer() -> MockDomainGraph {
        let mut graph = MockDomainGraph::new();
        let nodes: Vec<_> = (0..7).map(|_| graph.add_node(())).collect();

        graph.add_edge(nodes[0], nodes[1], ()); // 0 -> 1
        graph.add_edge(nodes[0], nodes[2], ()); // 0 -> 3 (skip layer)
        graph.add_edge(nodes[1], nodes[2], ()); // 1 -> 3
        graph.add_edge(nodes[2], nodes[3], ()); // 3 -> 4
        graph.add_edge(nodes[2], nodes[4], ()); // 3 -> 5
        graph.add_edge(nodes[3], nodes[5], ()); // 4 -> 6
        graph.add_edge(nodes[4], nodes[5], ()); // 5 -> 6
        graph.add_edge(nodes[5], nodes[6], ()); // 6 -> 7

        graph
    }

    /// Create a single node graph (domain graph)
    pub fn domain_single_node() -> MockDomainGraph {
        let mut graph = MockDomainGraph::new();
        graph.add_node(());
        graph
    }

    /// Create a bridge graph with clear articulation points: A-B-C where B is articulation point
    pub fn domain_bridge_graph() -> MockDomainGraph {
        let mut graph = MockDomainGraph::new();
        let a = graph.add_node(());
        let b = graph.add_node(());
        let c = graph.add_node(());

        graph.add_edge(a, b, ());
        graph.add_edge(b, c, ());

        graph
    }

    /// Create a star graph where center is articulation point
    pub fn domain_star_graph() -> MockDomainGraph {
        let mut graph = MockDomainGraph::new();
        let center = graph.add_node(());
        let leaf1 = graph.add_node(());
        let leaf2 = graph.add_node(());
        let leaf3 = graph.add_node(());

        graph.add_edge(center, leaf1, ());
        graph.add_edge(center, leaf2, ());
        graph.add_edge(center, leaf3, ());

        graph
    }

    /// Create a complex graph with multiple articulation points
    pub fn domain_articulation_graph() -> MockDomainGraph {
        let mut graph = MockDomainGraph::new();
        let nodes: Vec<_> = (0..7).map(|_| graph.add_node(())).collect();

        // Structure: 0-1-2-3-4
        //               |   |
        //               5   6
        // Here nodes 1, 2, and 3 should be articulation points
        graph.add_edge(nodes[0], nodes[1], ());
        graph.add_edge(nodes[1], nodes[2], ());
        graph.add_edge(nodes[2], nodes[3], ());
        graph.add_edge(nodes[3], nodes[4], ());
        graph.add_edge(nodes[1], nodes[5], ());
        graph.add_edge(nodes[3], nodes[6], ());

        graph
    }

    /// A plain `0 -> 1 -> ... -> length - 1` chain, for subsetting-logic tests exercising the
    /// simplest possible topology at a realistic size.
    pub fn domain_long_chain(length: usize) -> MockDomainGraph {
        let mut graph = MockDomainGraph::new();
        let nodes: Vec<_> = (0..length).map(|_| graph.add_node(())).collect();
        for pair in nodes.windows(2) {
            graph.add_edge(pair[0], pair[1], ());
        }
        graph
    }

    /// `diamonds` diamonds chained end to end: `0 -> {1, 2} -> 3 -> {4, 5} -> 6 -> ...`. Each
    /// diamond after the first reuses the previous one's join node as its own fork node, adding
    /// exactly 3 new nodes (two branches plus a join) and 4 new edges.
    pub fn domain_diamond_chain(diamonds: usize) -> MockDomainGraph {
        let mut graph = MockDomainGraph::new();
        let mut fork = graph.add_node(());
        for _ in 0..diamonds {
            let branch_a = graph.add_node(());
            let branch_b = graph.add_node(());
            let join = graph.add_node(());
            graph.add_edge(fork, branch_a, ());
            graph.add_edge(fork, branch_b, ());
            graph.add_edge(branch_a, join, ());
            graph.add_edge(branch_b, join, ());
            fork = join;
        }
        graph
    }

    /// A "strut": `layers` layers of `width` nodes each, with every node in one layer connected
    /// to every node in the next (a chain of dense many-to-many blocks - `width = 2` gives a
    /// chain of 2-by-2 blocks, 4 edges per layer transition).
    pub fn domain_strut(layers: usize, width: usize) -> MockDomainGraph {
        let mut graph = MockDomainGraph::new();
        let mut layer_nodes: Vec<Vec<NodeIndex>> = Vec::with_capacity(layers);
        for _ in 0..layers {
            layer_nodes.push((0..width).map(|_| graph.add_node(())).collect());
        }
        for pair in layer_nodes.windows(2) {
            for &source in &pair[0] {
                for &target in &pair[1] {
                    graph.add_edge(source, target, ());
                }
            }
        }
        graph
    }

    /// A `rows` by `cols` grid where node `(row, col)` connects to its right neighbour
    /// `(row, col + 1)` and its down neighbour `(row + 1, col)` when in bounds - a DAG (required
    /// by the Sugiyama-based layout pipeline) that still gives every interior node the 4
    /// connections (2 in, 2 out) of an ordinary undirected grid. Returns the graph plus a
    /// `(row, col) -> NodeIndex` lookup so callers can pick a specific anchor (a corner, the
    /// center, an edge midpoint) by grid position instead of raw node insertion order.
    pub fn domain_grid(rows: usize, cols: usize) -> (MockDomainGraph, Vec<Vec<NodeIndex>>) {
        let mut graph = MockDomainGraph::new();
        let mut node_at: Vec<Vec<NodeIndex>> = Vec::with_capacity(rows);
        for _ in 0..rows {
            node_at.push((0..cols).map(|_| graph.add_node(())).collect());
        }
        for row in 0..rows {
            for col in 0..cols {
                if col + 1 < cols {
                    graph.add_edge(node_at[row][col], node_at[row][col + 1], ());
                }
                if row + 1 < rows {
                    graph.add_edge(node_at[row][col], node_at[row + 1][col], ());
                }
            }
        }
        (graph, node_at)
    }
}

/// Collection of standardized node sizers for testing
pub struct TestNodeSizers;

impl TestNodeSizers {
    /// Fixed size sizer - always returns the same size regardless of scale
    pub fn fixed_1x1() -> FixedNodeSizer {
        FixedNodeSizer {
            width: 1,
            height: 1,
        }
    }

    /// Fixed size sizer - medium sized nodes
    pub fn fixed_5x3() -> FixedNodeSizer {
        FixedNodeSizer {
            width: 5,
            height: 3,
        }
    }

    /// Scale-aware sizer that mimics genomic sequence sizing
    pub fn scale_aware() -> ScaleAwareNodeSizer {
        ScaleAwareNodeSizer::new()
    }

    /// Variable size sizer based on node index (for testing different node sizes)
    pub fn variable() -> VariableNodeSizer {
        VariableNodeSizer::new()
    }
}

/// Simple fixed-size node sizer for predictable testing
#[derive(Debug, Clone)]
pub struct FixedNodeSizer {
    pub width: u64,
    pub height: u64,
}

// Detail-aware sizing for pairing inside a MockVisual.
impl<G: GraphBase> MockRenderer<G> for FixedNodeSizer {
    fn get_node_size(&self, _node: &G::NodeId, _detail_level: VisualDetail) -> (u64, u64) {
        (self.width, self.height)
    }
}

// Full renderer for direct use as a controller's renderer (renders a plain glyph; used
// by tests that measure layout rather than rendered output).
impl<G: GraphBase> NodeRenderer<G> for FixedNodeSizer {
    fn get_node_size(&self, _node: &G::NodeId) -> (u64, u64) {
        (self.width, self.height)
    }

    fn render_node(&self, buffer: &mut WorldBuffer, area: WorldRect, _node_id: &G::NodeId) {
        buffer.set_char(area.center(), NODE_GLYPH);
    }
}

/// Scale-aware node sizer that changes size based on selected level of detail (scale)
#[derive(Debug, Clone)]
pub struct ScaleAwareNodeSizer {
    base_size: (u64, u64),
    full_multiplier: (u64, u64),
    truncated_size: (u64, u64),
}

impl ScaleAwareNodeSizer {
    pub fn new() -> Self {
        Self {
            base_size: (1, 1),
            full_multiplier: (10, 3),
            truncated_size: (5, 2),
        }
    }
}

impl Default for ScaleAwareNodeSizer {
    fn default() -> Self {
        Self::new()
    }
}

impl<G: GraphBase> MockRenderer<G> for ScaleAwareNodeSizer {
    fn get_node_size(&self, _node: &G::NodeId, scale: VisualDetail) -> (u64, u64) {
        match scale {
            VisualDetail::Minimal => self.base_size,
            VisualDetail::Full => self.full_multiplier,
            VisualDetail::Truncated => self.truncated_size,
        }
    }
}

/// Variable node sizer that returns different sizes based on node index
#[derive(Debug, Clone)]
pub struct VariableNodeSizer {
    size_map: HashMap<u32, (u64, u64)>,
}

impl VariableNodeSizer {
    pub fn new() -> Self {
        let mut size_map = HashMap::new();
        size_map.insert(0u32, (2, 1)); // Small
        size_map.insert(1u32, (5, 2)); // Medium
        size_map.insert(2u32, (8, 3)); // Large
        // Default to medium size for other nodes

        Self { size_map }
    }
}

impl Default for VariableNodeSizer {
    fn default() -> Self {
        Self::new()
    }
}

impl MockRenderer<MockDomainGraph> for VariableNodeSizer {
    fn get_node_size(&self, node: &MockNodeId, _scale: VisualDetail) -> (u64, u64) {
        self.size_map
            .get(&(node.index() as u32))
            .copied()
            .unwrap_or((5, 2))
    }
}

/// Collection of standardized node renderers for testing
pub struct TestRenderers;

impl TestRenderers {
    /// Debug renderer that shows node indices
    pub fn debug() -> DebugNodeRenderer {
        DebugNodeRenderer::new()
    }

    /// Minimal renderer that just shows a symbol
    pub fn minimal() -> MinimalNodeRenderer {
        MinimalNodeRenderer::new('●')
    }

    /// Mock genomic renderer that shows fake DNA sequences
    pub fn mock_genomic() -> MockGenomicRenderer {
        MockGenomicRenderer::new()
    }
}

/// Debug renderer that displays node indices and boundaries
#[derive(Debug, Clone)]
pub struct DebugNodeRenderer {
    background_char: char,
    text_style: Style,
    bg_style: Style,
}

impl DebugNodeRenderer {
    pub fn new() -> Self {
        Self {
            background_char: '█',
            text_style: Style::default().fg(Color::White).bg(Color::Blue),
            bg_style: Style::default().fg(Color::Blue).bg(Color::Blue),
        }
    }
}

impl Default for DebugNodeRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl MockRenderer<MockDomainGraph> for DebugNodeRenderer {
    fn render_node(
        &self,
        buffer: &mut WorldBuffer,
        area: WorldRect,
        node_id: &NodeIndex<u32>,
        _scale: VisualDetail,
    ) {
        // Fill background
        for y in area.min.y..=area.max.y {
            for x in area.min.x..=area.max.x {
                buffer.set_char_styled(
                    crate::geometry::WorldPos::new(x, y),
                    self.background_char,
                    self.bg_style,
                );
            }
        }

        // Render node index in center
        let label = format!("N{}", node_id.index());
        let center = area.center();
        let label_start =
            crate::geometry::WorldPos::new(center.x - (label.len() as i64) / 2, center.y);
        buffer.set_string_styled(label_start, &label, self.text_style);
    }
}

/// Minimal renderer that just shows a single character
#[derive(Debug, Clone)]
pub struct MinimalNodeRenderer {
    symbol: char,
    style: Style,
}

impl MinimalNodeRenderer {
    pub fn new(symbol: char) -> Self {
        Self {
            symbol,
            style: Style::default(),
        }
    }

    pub fn with_style(mut self, style: Style) -> Self {
        self.style = style;
        self
    }
}

// Generic implementation for all graph types
impl<G> MockRenderer<G> for MinimalNodeRenderer
where
    G: petgraph::visit::GraphBase,
    G::NodeId: std::fmt::Debug,
{
    fn render_node(
        &self,
        buffer: &mut WorldBuffer,
        area: WorldRect,
        _node_id: &G::NodeId,
        _scale: VisualDetail,
    ) {
        let center = area.center();
        buffer.set_char_styled(center, self.symbol, self.style);
    }
}

/// Mock genomic renderer that shows fake DNA sequences based on scale
#[derive(Debug, Clone)]
pub struct MockGenomicRenderer {
    sequences: HashMap<u32, String>,
}

impl MockGenomicRenderer {
    pub fn new() -> Self {
        let mut sequences = HashMap::new();
        sequences.insert(0, "ATCG".to_string());
        sequences.insert(1, "GCTA".to_string());
        sequences.insert(2, "TGCA".to_string());
        sequences.insert(3, "CGAT".to_string());

        Self { sequences }
    }

    fn get_sequence(&self, node_id: &NodeIndex<u32>) -> String {
        self.sequences
            .get(&(node_id.index() as u32))
            .cloned()
            .unwrap_or_else(|| format!("SEQ{}", node_id.index()))
    }
}

impl Default for MockGenomicRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl MockRenderer<MockDomainGraph> for MockGenomicRenderer {
    fn render_node(
        &self,
        buffer: &mut WorldBuffer,
        area: WorldRect,
        node_id: &NodeIndex<u32>,
        scale: VisualDetail,
    ) {
        let center = area.center();

        match scale {
            VisualDetail::Minimal => {
                // Just show node symbol
                buffer.set_char(center, '⏺');
            }
            VisualDetail::Full => {
                // Show full sequence
                let sequence = self.get_sequence(node_id);
                let start_pos = crate::geometry::WorldPos::new(
                    center.x - (sequence.len() as i64) / 2,
                    center.y,
                );
                buffer.set_string(start_pos, &sequence);
            }
            VisualDetail::Truncated => {
                // Show truncated sequence
                let sequence = self.get_sequence(node_id);
                let truncated = if sequence.len() > 3 {
                    format!("{}...", &sequence[..3])
                } else {
                    sequence
                };
                let start_pos = crate::geometry::WorldPos::new(
                    center.x - (truncated.len() as i64) / 2,
                    center.y,
                );
                buffer.set_string(start_pos, &truncated);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_sizers() {
        let fixed = TestNodeSizers::fixed_1x1();
        let node_id: NodeIndex<u32> = NodeIndex::new(0);

        let size_base: (u64, u64) =
            <FixedNodeSizer as MockRenderer<MockDomainGraph>>::get_node_size(
                &fixed,
                &node_id,
                VisualDetail::Minimal,
            );
        assert_eq!(size_base, (1, 1));

        let scale_aware = TestNodeSizers::scale_aware();
        let size_base_aware: (u64, u64) =
            <ScaleAwareNodeSizer as MockRenderer<MockDomainGraph>>::get_node_size(
                &scale_aware,
                &node_id,
                VisualDetail::Minimal,
            );
        let size_full_aware: (u64, u64) =
            <ScaleAwareNodeSizer as MockRenderer<MockDomainGraph>>::get_node_size(
                &scale_aware,
                &node_id,
                VisualDetail::Full,
            );
        assert_eq!(size_base_aware, (1, 1));
        assert_eq!(size_full_aware, (10, 3));
    }
}
