use crate::GenGraph;

pub trait MergeGraph {
    fn merge_graph(&mut self, other: &GenGraph);
}

impl MergeGraph for GenGraph {
    fn merge_graph(&mut self, other: &GenGraph) {
        for node in other.nodes() {
            self.add_node(node);
        }

        for (source, target, edges) in other.all_edges() {
            if let Some(existing_edges) = self.edge_weight_mut(source, target) {
                for edge in edges.iter().copied() {
                    if !existing_edges.contains(&edge) {
                        existing_edges.push(edge);
                    }
                }
            } else {
                self.add_edge(source, target, edges.clone());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use gen_core::{HashId, Strand};

    use super::*;
    use crate::{GraphEdge, GraphNode};

    #[test]
    fn merges_gen_graphs_and_preserves_distinct_edges() {
        let start = GraphNode {
            node_id: HashId::convert_str("merge-start"),
            sequence_start: 0,
            sequence_end: 0,
        };
        let middle = GraphNode {
            node_id: HashId::convert_str("merge-middle"),
            sequence_start: 0,
            sequence_end: 5,
        };
        let end = GraphNode {
            node_id: HashId::convert_str("merge-end"),
            sequence_start: 0,
            sequence_end: 0,
        };
        let chrom_index0_edge = GraphEdge {
            edge_id: HashId::convert_str("merge-edge-ci0"),
            source_strand: Strand::Forward,
            target_strand: Strand::Forward,
            chromosome_index: 0,
            phased: 0,
            created_on: 1,
        };
        let chrom_index1_edge = GraphEdge {
            edge_id: HashId::convert_str("merge-edge-ci1"),
            source_strand: Strand::Forward,
            target_strand: Strand::Forward,
            chromosome_index: 1,
            phased: 1,
            created_on: 2,
        };

        let mut graph_a = GenGraph::new();
        graph_a.add_edge(start, middle, vec![chrom_index0_edge]);

        let mut graph_b = GenGraph::new();
        graph_b.add_edge(start, middle, vec![chrom_index1_edge]);
        graph_b.add_edge(middle, end, vec![chrom_index0_edge]);

        graph_a.merge_graph(&graph_b);

        assert_eq!(graph_a.edge_weight(start, middle).unwrap().len(), 2);
        let graph_a_edges = graph_a.edge_weight(start, middle).unwrap();
        assert!(graph_a_edges.contains(&chrom_index0_edge));
        assert!(graph_a_edges.contains(&chrom_index1_edge));
        assert_eq!(
            graph_a.edge_weight(middle, end).unwrap(),
            &vec![chrom_index0_edge]
        );
    }

    #[test]
    fn merges_gen_graphs_without_duplicating_identical_edges() {
        let start = GraphNode {
            node_id: HashId::convert_str("dedup-start"),
            sequence_start: 0,
            sequence_end: 0,
        };
        let end = GraphNode {
            node_id: HashId::convert_str("dedup-end"),
            sequence_start: 0,
            sequence_end: 0,
        };
        let edge = GraphEdge {
            edge_id: HashId::convert_str("dedup-edge"),
            source_strand: Strand::Forward,
            target_strand: Strand::Forward,
            chromosome_index: 0,
            phased: 0,
            created_on: 1,
        };

        let mut graph_a = GenGraph::new();
        graph_a.add_edge(start, end, vec![edge]);

        let mut graph_b = GenGraph::new();
        graph_b.add_edge(start, end, vec![edge]);

        graph_a.merge_graph(&graph_b);

        assert_eq!(graph_a.edge_weight(start, end).unwrap(), &vec![edge]);
    }

    #[test]
    fn merges_gen_graphs_by_graph_node_value() {
        // Ensure when merging the same graph into itself, it dedupes nodes but will keep multiple edges between identical nodes
        // between graphs
        let source_a = GraphNode {
            node_id: HashId::convert_str("shared-node"),
            sequence_start: 0,
            sequence_end: 5,
        };
        let target_a = GraphNode {
            node_id: HashId::convert_str("target-node"),
            sequence_start: 0,
            sequence_end: 5,
        };
        let source_b = GraphNode {
            node_id: source_a.node_id,
            sequence_start: source_a.sequence_start,
            sequence_end: source_a.sequence_end,
        };
        let target_b = GraphNode {
            node_id: target_a.node_id,
            sequence_start: target_a.sequence_start,
            sequence_end: target_a.sequence_end,
        };
        let edge_a = GraphEdge {
            edge_id: HashId::convert_str("logical-edge-a"),
            source_strand: Strand::Forward,
            target_strand: Strand::Forward,
            chromosome_index: 0,
            phased: 0,
            created_on: 1,
        };
        let edge_b = GraphEdge {
            edge_id: HashId::convert_str("logical-edge-b"),
            source_strand: Strand::Forward,
            target_strand: Strand::Forward,
            chromosome_index: 1,
            phased: 0,
            created_on: 2,
        };

        let mut graph_a = GenGraph::new();
        graph_a.add_edge(source_a, target_a, vec![edge_a]);

        let mut graph_b = GenGraph::new();
        graph_b.add_edge(source_b, target_b, vec![edge_b]);

        graph_a.merge_graph(&graph_b);

        assert_eq!(graph_a.nodes().count(), 2);
        assert!(graph_a.nodes().any(|node| node == source_a));
        assert!(graph_a.nodes().any(|node| node == target_a));
        assert_eq!(graph_a.edge_weight(source_a, target_a).unwrap().len(), 2);
    }
}
