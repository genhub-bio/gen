use std::collections::HashMap;

use gen_tui::{
    geometry::WorldRect,
    graph_controller::WorldBuffer,
    graph_widget::NODE_GLYPH,
    layout::VisualDetail,
    plotter::{NodeRenderer, NodeSizer},
};
use petgraph::graphmap::DiGraphMap;
use ratatui::style::{Color, Style};

/// A node representing a GFA segment.
/// Mirrors `GraphNode` from gen-graph but without any database dependency.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct GfaNode {
    pub segment_id: u64,
    pub sequence_start: i64,
    pub sequence_end: i64,
}

impl GfaNode {
    pub fn length(&self) -> i64 {
        self.sequence_end - self.sequence_start
    }
}

/// Graph type for GFA-based visualization.
pub type GfaGraph = DiGraphMap<GfaNode, ()>;

/// Parse a GFA file string into a graph and a segment-id → sequence table.
///
/// Each S-line becomes a node with `sequence_start = 0`, `sequence_end = seq.len()`.
/// L-lines become directed edges (from → to).
/// Segment names that parse as `u64` are used directly; others are hashed.
pub fn parse_gfa(gfa: &str) -> (GfaGraph, HashMap<u64, String>) {
    let mut graph = GfaGraph::new();
    let mut sequences: HashMap<u64, String> = HashMap::new();
    let mut name_to_node: HashMap<String, GfaNode> = HashMap::new();

    for line in gfa.lines() {
        let fields: Vec<&str> = line.split('\t').collect();
        match fields.first().copied() {
            Some("S") if fields.len() >= 3 => {
                let name = fields[1];
                let seq = fields[2];
                let id = name.parse::<u64>().unwrap_or_else(|_| {
                    let mut h = 0u64;
                    for b in name.bytes() {
                        h = h.wrapping_mul(31).wrapping_add(b as u64);
                    }
                    h
                });
                let node = GfaNode {
                    segment_id: id,
                    sequence_start: 0,
                    sequence_end: seq.len() as i64,
                };
                graph.add_node(node);
                sequences.insert(id, seq.to_string());
                name_to_node.insert(name.to_string(), node);
            }
            Some("L") if fields.len() >= 5 => {
                let from_name = fields[1];
                let to_name = fields[3];
                if let (Some(&from), Some(&to)) =
                    (name_to_node.get(from_name), name_to_node.get(to_name))
                {
                    graph.add_edge(from, to, ());
                }
            }
            _ => {}
        }
    }

    (graph, sequences)
}

/// Node sizer for GFA graphs — sizes nodes based on sequence length.
/// Mirrors `GenGraphNodeSizer` from gen_graph_widget.rs.
#[derive(Clone)]
pub struct GfaNodeSizer;

impl NodeSizer<&GfaGraph> for GfaNodeSizer {
    fn get_node_size(&self, node: &GfaNode, detail_level: VisualDetail) -> (u64, u64) {
        let len = node.length() as u64;
        match detail_level {
            VisualDetail::Minimal => (1, 1),
            VisualDetail::Truncated => (len.min(12).max(1), 1),
            VisualDetail::Full => (len.max(1), 1),
        }
    }
}

/// Node renderer for GFA graphs — fetches sequences from the in-memory table.
/// Mirrors `GenGraphNodeRenderer` from gen_graph_widget.rs.
#[derive(Clone)]
pub struct GfaNodeRenderer {
    sequences: HashMap<u64, String>,
    cache: HashMap<GfaNode, String>,
}

impl GfaNodeRenderer {
    pub fn new(sequences: HashMap<u64, String>) -> Self {
        Self {
            sequences,
            cache: HashMap::new(),
        }
    }

    fn get_sequence(&mut self, node: &GfaNode) -> String {
        if let Some(cached) = self.cache.get(node) {
            return cached.clone();
        }
        let seq = self
            .sequences
            .get(&node.segment_id)
            .map(|s| {
                s.get(node.sequence_start as usize..node.sequence_end as usize)
                    .unwrap_or(s)
                    .to_string()
            })
            .unwrap_or_else(|| "?".repeat(node.length().max(0) as usize));
        self.cache.insert(*node, seq.clone());
        seq
    }
}

impl NodeRenderer<&GfaGraph> for GfaNodeRenderer {
    fn render_node(
        &mut self,
        buffer: &mut WorldBuffer,
        area: WorldRect,
        node_id: &GfaNode,
        detail_level: VisualDetail,
    ) {
        let node_bg = Color::Blue;
        let node_fg = Color::White;
        let node_style = Style::default().bg(node_bg).fg(node_fg);

        buffer.fill_rect(area, ' ');

        match detail_level {
            VisualDetail::Minimal => {
                let style = Style::default().fg(node_fg).bg(node_bg);
                buffer.set_string_styled(area.left_center(), &NODE_GLYPH.to_string(), style);
            }
            VisualDetail::Truncated => {
                let sequence = self.get_sequence(node_id);
                let truncated = inner_truncation(&sequence, 12);
                buffer.set_string_styled(area.left_center(), &truncated, node_style);
            }
            VisualDetail::Full => {
                let sequence = self.get_sequence(node_id);
                buffer.set_string_styled(area.left_center(), &sequence, node_style);
            }
        }
    }
}

/// Truncate a sequence string from the inside, keeping beginning and end.
/// Mirrors `inner_truncation` from gen_graph_widget.rs.
pub fn inner_truncation(s: &str, target_length: u32) -> String {
    if s.len() <= target_length as usize {
        return s.to_string();
    } else if target_length < 5 {
        return NODE_GLYPH.to_string();
    }
    let left_len = (target_length - 3) / 2 + ((target_length - 3) % 2);
    let right_len = (target_length - 3) / 2;
    let left = &s[..left_len as usize];
    let right = &s[(s.len() - right_len as usize)..];
    format!("{}...{}", left, right)
}
