//! Show each physical edge once when the graph stores both of its spellings.
//!
//! An edge read from the other side is its mirror: `a+ -> b+` is `b- -> a-`, and `a+ -> b-` is
//! `b+ -> a-`. Edits such as a recombination can leave both spellings stored, and drawing or
//! exporting both shows one junction twice. A spelling is dropped only when its mirror is present,
//! so an edge stored alone is always kept.

use std::collections::HashSet;

use gen_core::Strand;
use gen_graph::{GenGraph, GraphEdge, GraphNode};

type EdgeSpelling = (GraphNode, GraphNode, Strand, Strand);

fn spelling(source: GraphNode, target: GraphNode, edge: &GraphEdge) -> EdgeSpelling {
    (source, target, edge.source_strand, edge.target_strand)
}

/// Strands other than forward and reverse have no complement, so they never pair up.
fn mirror_of((source, target, source_strand, target_strand): EdgeSpelling) -> Option<EdgeSpelling> {
    let has_complement = |strand: Strand| matches!(strand, Strand::Forward | Strand::Reverse);
    (has_complement(source_strand) && has_complement(target_strand)).then(|| {
        (
            target,
            source,
            target_strand.complement(),
            source_strand.complement(),
        )
    })
}

/// Drop the spelling of each mirrored pair that `surviving_routes` does not keep, then the one
/// that reads against the forward strand, then the later one. Passing `None` ranks on the last two.
pub(crate) fn collapse_mirror_edges(graph: &mut GenGraph, surviving_routes: Option<&GenGraph>) {
    let present: HashSet<EdgeSpelling> = graph
        .all_edges()
        .flat_map(|(source, target, edges)| {
            edges.iter().map(move |edge| spelling(source, target, edge))
        })
        .collect();
    let rank = |(source, target, source_strand, _): EdgeSpelling| {
        let routed = surviving_routes.is_some_and(|routes| routes.contains_edge(source, target));
        (!routed, source_strand != Strand::Forward, source, target)
    };

    let mut emptied = vec![];
    for (source, target, edges) in graph.all_edges_mut() {
        edges.retain(|edge| {
            let this = spelling(source, target, edge);
            let Some(mirror) = mirror_of(this).filter(|mirror| *mirror != this) else {
                return true;
            };
            !present.contains(&mirror) || rank(this) < rank(mirror)
        });
        if edges.is_empty() {
            emptied.push((source, target));
        }
    }
    for (source, target) in emptied {
        graph.remove_edge(source, target);
    }
}

#[cfg(test)]
mod tests {
    use gen_core::HashId;

    use super::*;

    fn node(label: &str) -> GraphNode {
        GraphNode {
            node_id: HashId::convert_str(label),
            sequence_start: 0,
            sequence_end: 4,
        }
    }

    fn edge(label: &str, source_strand: Strand, target_strand: Strand) -> GraphEdge {
        GraphEdge {
            edge_id: HashId::convert_str(label),
            source_strand,
            target_strand,
            chromosome_index: 0,
            phased: 0,
            created_on: 0,
        }
    }

    fn strands(graph: &GenGraph, source: GraphNode, target: GraphNode) -> Vec<(Strand, Strand)> {
        graph
            .edge_weight(source, target)
            .map(|edges| {
                edges
                    .iter()
                    .map(|edge| (edge.source_strand, edge.target_strand))
                    .collect()
            })
            .unwrap_or_default()
    }

    #[test]
    fn test_mirror_pair_keeps_the_forward_spelling() {
        let (first, second) = (node("first"), node("second"));
        let mut graph = GenGraph::new();
        graph.add_edge(
            first,
            second,
            vec![edge("a", Strand::Forward, Strand::Forward)],
        );
        graph.add_edge(
            second,
            first,
            vec![edge("b", Strand::Reverse, Strand::Reverse)],
        );
        collapse_mirror_edges(&mut graph, None);
        assert_eq!(
            strands(&graph, first, second),
            [(Strand::Forward, Strand::Forward)]
        );
        assert!(!graph.contains_edge(second, first));
    }

    #[test]
    fn test_inversion_mirror_pair_collapses_to_one_edge() {
        let (first, second) = (node("first"), node("second"));
        let mut graph = GenGraph::new();
        graph.add_edge(
            first,
            second,
            vec![edge("a", Strand::Forward, Strand::Reverse)],
        );
        graph.add_edge(
            second,
            first,
            vec![edge("b", Strand::Forward, Strand::Reverse)],
        );
        collapse_mirror_edges(&mut graph, None);
        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    fn test_edge_without_its_mirror_is_kept() {
        let (first, second) = (node("first"), node("second"));
        let mut graph = GenGraph::new();
        graph.add_edge(
            first,
            second,
            vec![
                edge("a", Strand::Forward, Strand::Forward),
                edge("b", Strand::Reverse, Strand::Forward),
                edge("c", Strand::Forward, Strand::Reverse),
                edge("d", Strand::Reverse, Strand::Reverse),
            ],
        );
        collapse_mirror_edges(&mut graph, None);
        assert_eq!(strands(&graph, first, second).len(), 4);
    }

    #[test]
    fn test_self_mirror_edge_and_unknown_strands_are_kept() {
        let (first, second) = (node("first"), node("second"));
        let mut graph = GenGraph::new();
        graph.add_edge(
            first,
            first,
            vec![edge("a", Strand::Forward, Strand::Reverse)],
        );
        graph.add_edge(
            first,
            second,
            vec![edge("b", Strand::Unknown, Strand::Unknown)],
        );
        graph.add_edge(
            second,
            first,
            vec![edge("c", Strand::Unknown, Strand::Unknown)],
        );
        collapse_mirror_edges(&mut graph, None);
        assert_eq!(graph.edge_count(), 3);
    }

    #[test]
    fn test_surviving_route_decides_which_spelling_is_kept() {
        let (first, second) = (node("first"), node("second"));
        let mut graph = GenGraph::new();
        graph.add_edge(
            first,
            second,
            vec![edge("a", Strand::Forward, Strand::Forward)],
        );
        graph.add_edge(
            second,
            first,
            vec![edge("b", Strand::Reverse, Strand::Reverse)],
        );
        let mut routes = GenGraph::new();
        routes.add_edge(second, first, vec![]);
        collapse_mirror_edges(&mut graph, Some(&routes));
        assert!(graph.contains_edge(second, first));
        assert!(!graph.contains_edge(first, second));
    }
}
