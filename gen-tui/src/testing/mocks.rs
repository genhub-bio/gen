// Standardized mock objects for consistent testing across the widget system

use std::collections::HashMap;

use petgraph::{graph::NodeIndex, stable_graph::StableDiGraph};
use ratatui::style::{Color, Style};

use crate::{
    geometry::WorldRect,
    graph_controller::WorldBuffer,
    layout::VisualDetail,
    partition::{PartitionEdge, PartitionNode, StitchSide},
    plotter::{NodeRenderer, NodeSizer},
};

// Type aliases for common test graph types
pub type MockPartitionGraph = StableDiGraph<PartitionNode, PartitionEdge, u32>;
pub type MockDomainGraph = StableDiGraph<(), ()>;
pub type MockNodeId = NodeIndex<u32>;

/// Collection of standardized test graphs for consistent testing
pub struct TestGraphs;

impl TestGraphs {
    /// Simple linear chain: A -> B -> C (returns partition graph)
    pub fn simple_chain() -> MockPartitionGraph {
        let mut graph = MockPartitionGraph::new();

        // Add stitch nodes
        let left_stitch = graph.add_node(PartitionNode::Stitch(StitchSide::Left));
        let right_stitch = graph.add_node(PartitionNode::Stitch(StitchSide::Right));

        // Add data nodes
        let a = graph.add_node(PartitionNode::Data(NodeIndex::new(0)));
        let b = graph.add_node(PartitionNode::Data(NodeIndex::new(1)));
        let c = graph.add_node(PartitionNode::Data(NodeIndex::new(2)));

        // Connect stitch nodes to data nodes
        graph.add_edge(left_stitch, a, None);
        graph.add_edge(c, right_stitch, None);

        // Connect data nodes
        graph.add_edge(a, b, Some((NodeIndex::new(0), NodeIndex::new(1))));
        graph.add_edge(b, c, Some((NodeIndex::new(1), NodeIndex::new(2))));

        graph
    }

    /// Diamond pattern: A -> B -> D, A -> C -> D (returns partition graph)
    pub fn diamond() -> MockPartitionGraph {
        let mut graph = MockPartitionGraph::new();

        // Add stitch nodes
        let left_stitch = graph.add_node(PartitionNode::Stitch(StitchSide::Left));
        let right_stitch = graph.add_node(PartitionNode::Stitch(StitchSide::Right));

        // Add data nodes
        let a = graph.add_node(PartitionNode::Data(NodeIndex::new(0)));
        let b = graph.add_node(PartitionNode::Data(NodeIndex::new(1)));
        let c = graph.add_node(PartitionNode::Data(NodeIndex::new(2)));
        let d = graph.add_node(PartitionNode::Data(NodeIndex::new(3)));

        // Connect stitch nodes to data nodes
        graph.add_edge(left_stitch, a, None);
        graph.add_edge(d, right_stitch, None);

        // Connect data nodes
        graph.add_edge(a, b, Some((NodeIndex::new(0), NodeIndex::new(1))));
        graph.add_edge(a, c, Some((NodeIndex::new(0), NodeIndex::new(2))));
        graph.add_edge(b, d, Some((NodeIndex::new(1), NodeIndex::new(3))));
        graph.add_edge(c, d, Some((NodeIndex::new(2), NodeIndex::new(3))));

        graph
    }

    /// Complex DAG with multiple levels and branches (returns partition graph)
    pub fn complex_dag() -> MockPartitionGraph {
        let mut graph = MockPartitionGraph::new();

        // Add stitch nodes
        let left_stitch = graph.add_node(PartitionNode::Stitch(StitchSide::Left));
        let right_stitch = graph.add_node(PartitionNode::Stitch(StitchSide::Right));

        // Add data nodes
        let nodes: Vec<_> = (0..10)
            .map(|i| graph.add_node(PartitionNode::Data(NodeIndex::new(i))))
            .collect();

        // Connect left stitch to the first node (node 0 is the root)
        graph.add_edge(left_stitch, nodes[0], None);

        // Connect the sink nodes (nodes 6 and 7) to the right stitch
        graph.add_edge(nodes[9], right_stitch, None);

        // Create a complex hierarchical structure
        graph.add_edge(
            nodes[0],
            nodes[1],
            Some((NodeIndex::new(0), NodeIndex::new(1))),
        );
        graph.add_edge(
            nodes[0],
            nodes[2],
            Some((NodeIndex::new(0), NodeIndex::new(2))),
        );
        graph.add_edge(
            nodes[1],
            nodes[3],
            Some((NodeIndex::new(1), NodeIndex::new(3))),
        );
        graph.add_edge(
            nodes[1],
            nodes[4],
            Some((NodeIndex::new(1), NodeIndex::new(4))),
        );
        graph.add_edge(
            nodes[2],
            nodes[4],
            Some((NodeIndex::new(2), NodeIndex::new(4))),
        );
        graph.add_edge(
            nodes[2],
            nodes[5],
            Some((NodeIndex::new(2), NodeIndex::new(5))),
        );
        graph.add_edge(
            nodes[3],
            nodes[6],
            Some((NodeIndex::new(3), NodeIndex::new(6))),
        );
        graph.add_edge(
            nodes[4],
            nodes[6],
            Some((NodeIndex::new(4), NodeIndex::new(6))),
        );
        graph.add_edge(
            nodes[4],
            nodes[7],
            Some((NodeIndex::new(4), NodeIndex::new(7))),
        );
        graph.add_edge(
            nodes[5],
            nodes[7],
            Some((NodeIndex::new(5), NodeIndex::new(7))),
        );

        graph.add_edge(
            nodes[6],
            nodes[8],
            Some((NodeIndex::new(6), NodeIndex::new(8))),
        );
        graph.add_edge(
            nodes[7],
            nodes[8],
            Some((NodeIndex::new(7), NodeIndex::new(8))),
        );
        graph.add_edge(
            nodes[8],
            nodes[9],
            Some((NodeIndex::new(8), NodeIndex::new(9))),
        );

        graph
    }

    /// Single node (edge case testing)
    pub fn single_node() -> MockPartitionGraph {
        let mut graph = MockPartitionGraph::new();

        // Add stitch nodes
        let left_stitch = graph.add_node(PartitionNode::Stitch(StitchSide::Left));
        let right_stitch = graph.add_node(PartitionNode::Stitch(StitchSide::Right));

        // Add single data node
        let node = graph.add_node(PartitionNode::Data(NodeIndex::new(0)));

        // Connect stitch nodes to the single data node
        graph.add_edge(left_stitch, node, None);
        graph.add_edge(node, right_stitch, None);

        graph
    }

    /// Empty graph (edge case testing)
    pub fn empty() -> MockPartitionGraph {
        MockPartitionGraph::new()
    }

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

impl NodeSizer<MockPartitionGraph> for FixedNodeSizer {
    fn get_node_size(&self, _node: &MockNodeId, _scale: VisualDetail) -> (u64, u64) {
        (self.width, self.height)
    }

    fn get_dummy_size(&self) -> (u64, u64) {
        (1, 1)
    }
}

// Also implement for partition graph reference (for LayoutEngine)
impl NodeSizer<&MockPartitionGraph> for FixedNodeSizer {
    fn get_node_size(&self, _node: &MockNodeId, _scale: VisualDetail) -> (u64, u64) {
        (self.width, self.height)
    }

    fn get_dummy_size(&self) -> (u64, u64) {
        (1, 1)
    }
}

// Implement for domain graph (for GraphController)
impl NodeSizer<MockDomainGraph> for FixedNodeSizer {
    fn get_node_size(
        &self,
        _node: &petgraph::stable_graph::NodeIndex,
        _scale: VisualDetail,
    ) -> (u64, u64) {
        (self.width, self.height)
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

impl NodeSizer<MockPartitionGraph> for ScaleAwareNodeSizer {
    fn get_node_size(&self, _node: &MockNodeId, scale: VisualDetail) -> (u64, u64) {
        match scale {
            VisualDetail::Minimal => self.base_size,
            VisualDetail::Full => self.full_multiplier,
            VisualDetail::Truncated => self.truncated_size,
        }
    }
}

// Also implement for partition graph reference (for LayoutEngine)
impl NodeSizer<&MockPartitionGraph> for ScaleAwareNodeSizer {
    fn get_node_size(&self, _node: &MockNodeId, scale: VisualDetail) -> (u64, u64) {
        match scale {
            VisualDetail::Minimal => self.base_size,
            VisualDetail::Full => self.full_multiplier,
            VisualDetail::Truncated => self.truncated_size,
        }
    }
}

// Implement for domain graph (for GraphController)
impl NodeSizer<MockDomainGraph> for ScaleAwareNodeSizer {
    fn get_node_size(
        &self,
        _node: &petgraph::stable_graph::NodeIndex,
        scale: VisualDetail,
    ) -> (u64, u64) {
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

impl NodeSizer<MockPartitionGraph> for VariableNodeSizer {
    fn get_node_size(&self, node: &MockNodeId, _scale: VisualDetail) -> (u64, u64) {
        self.size_map
            .get(&(node.index() as u32))
            .copied()
            .unwrap_or((5, 2))
    }
}

// Also implement for partition graph reference (for LayoutEngine)
impl NodeSizer<&MockPartitionGraph> for VariableNodeSizer {
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

impl NodeRenderer<MockDomainGraph> for DebugNodeRenderer {
    fn render_node(
        &mut self,
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
impl<G> NodeRenderer<G> for MinimalNodeRenderer
where
    G: petgraph::visit::GraphBase,
    G::NodeId: std::fmt::Debug,
{
    fn render_node(
        &mut self,
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

impl NodeRenderer<MockDomainGraph> for MockGenomicRenderer {
    fn render_node(
        &mut self,
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
    fn test_graph_creation() {
        let simple = TestGraphs::simple_chain();
        assert_eq!(simple.node_count(), 5); // 3 data nodes + 2 stitch nodes
        assert_eq!(simple.edge_count(), 4); // 2 data edges + 2 stitch edges

        let diamond = TestGraphs::diamond();
        assert_eq!(diamond.node_count(), 6); // 4 data nodes + 2 stitch nodes
        assert_eq!(diamond.edge_count(), 6); // 4 data edges + 2 stitch edges
    }

    #[test]
    fn test_node_sizers() {
        let fixed = TestNodeSizers::fixed_1x1();
        let node_id: NodeIndex<u32> = NodeIndex::new(0);

        // Test with MockPartitionGraph type annotation
        let size_base: (u64, u64) =
            <FixedNodeSizer as NodeSizer<MockPartitionGraph>>::get_node_size(
                &fixed,
                &node_id,
                VisualDetail::Minimal,
            );
        assert_eq!(size_base, (1, 1));

        let scale_aware = TestNodeSizers::scale_aware();
        let size_base_aware: (u64, u64) =
            <ScaleAwareNodeSizer as NodeSizer<MockPartitionGraph>>::get_node_size(
                &scale_aware,
                &node_id,
                VisualDetail::Minimal,
            );
        let size_full_aware: (u64, u64) =
            <ScaleAwareNodeSizer as NodeSizer<MockPartitionGraph>>::get_node_size(
                &scale_aware,
                &node_id,
                VisualDetail::Full,
            );
        assert_eq!(size_base_aware, (1, 1));
        assert_eq!(size_full_aware, (10, 3));
    }
}
