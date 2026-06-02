use gen_core::{PATH_END_NODE_ID, PATH_START_NODE_ID};
use gen_graph::GraphNode;
use gen_parsers::{ClustalwParser, ParsedAlignment};
use petgraph::visit::{EdgeRef as _, IntoEdgeReferences as _};
use similar_asserts::assert_eq;

fn parse_single(input: &str) -> ParsedAlignment {
    let mut parser = ClustalwParser::new(input.as_bytes());
    let alignment = parser
        .next()
        .expect("should emit one alignment")
        .expect("should parse CLUSTAL alignment");
    assert!(
        parser.next().is_none(),
        "CLUSTAL parser should emit one alignment per file"
    );
    alignment
}

fn non_terminal_nodes(alignment: &ParsedAlignment) -> Vec<GraphNode> {
    let mut nodes = alignment
        .graph
        .nodes()
        .filter(|node| node.node_id != PATH_START_NODE_ID && node.node_id != PATH_END_NODE_ID)
        .collect::<Vec<_>>();
    nodes.sort();
    nodes
}

fn has_edge_for_sequence(
    alignment: &ParsedAlignment,
    source: GraphNode,
    target: GraphNode,
    index: i64,
) -> bool {
    alignment.graph.edge_references().any(|edge| {
        edge.source() == source
            && edge.target() == target
            && edge
                .weight()
                .iter()
                .any(|metadata| metadata.chromosome_index == index)
    })
}

#[test]
fn parses_clustal_header_without_w() {
    let input = "\
CLUSTAL 2.1 multiple sequence alignment

SeqA      ACGT
SeqB      ACGT
          ****
";

    let alignment = parse_single(input);

    assert_eq!(alignment.base_name, "SeqA");
    assert_eq!(alignment.sequence_order, vec!["SeqA", "SeqB"]);
    assert_eq!(
        alignment.ungapped_sequence("SeqB"),
        Some("ACGT".to_string())
    );
}

#[test]
fn parses_pairwise_alignment_into_base_relative_graph() {
    let input = "\
CLUSTAL W (2.1) multiple sequence alignment

SeqA      MAGAASAVAAALAAA--AAGAAATAAAG
SeqB      MAGAASAVAAALAAAGAAAGAAATAAAG
          *************   ************
";

    let alignment = parse_single(input);

    assert_eq!(alignment.base_name, "SeqA");
    assert_eq!(alignment.sequence_order, vec!["SeqA", "SeqB"]);
    assert_eq!(
        alignment
            .aligned_sequences
            .get("SeqA")
            .expect("should contain SeqA"),
        "MAGAASAVAAALAAA--AAGAAATAAAG"
    );
    assert_eq!(
        alignment.ungapped_sequence("SeqA"),
        Some("MAGAASAVAAALAAAAAGAAATAAAG".to_string())
    );

    let nodes = non_terminal_nodes(&alignment);
    let mut lengths = nodes.iter().map(GraphNode::length).collect::<Vec<_>>();
    lengths.sort();
    assert_eq!(lengths, vec![2, 11, 15]);
}

#[test]
fn parses_multi_block_alignment_and_keeps_sequence_order() {
    let input = "\
CLUSTAL W (1.83) multiple sequence alignment

Seq1            ATGCCTAGCTAGCTAGCATCGATCGATCGATCGATCGTACGATCGATCGATCGATCGATC 60
Seq2            ATGCCTAGCTAGCTAGC---TCGATCGATCGATCGATCGTACGATCGATCGATCGATC---- 53
Seq3            ATGCCT---------GC---TCGATCGATCGATC----GTACGATCGATCGATCGATCGATC 50
                ******         **   ****** ***        **********************

Seq1            TACGATCGATCGTACGTA 78
Seq2            ----ATCGATCGTACGTA 69
Seq3            TACGATCGATCGTACGTA 68
                    **********
";

    let alignment = parse_single(input);

    assert_eq!(alignment.sequence_order, vec!["Seq1", "Seq2", "Seq3"]);
    assert_eq!(alignment.base_name, "Seq1");
    assert_eq!(
        alignment
            .ungapped_sequence("Seq1")
            .expect("should contain Seq1")
            .len(),
        78
    );
    assert_eq!(
        alignment
            .ungapped_sequence("Seq2")
            .expect("should contain Seq2")
            .len(),
        69
    );
    assert_eq!(
        alignment
            .ungapped_sequence("Seq3")
            .expect("should contain Seq3")
            .len(),
        64
    );
}

#[test]
fn deletion_paths_skip_base_variant_nodes() {
    let input = "\
CLUSTALW multiple sequence alignment

Base      ACGT
Sample    A-GT
          * **
";

    let alignment = parse_single(input);
    let nodes = non_terminal_nodes(&alignment);
    assert_eq!(nodes.len(), 3);

    let mut base_nodes = nodes;
    base_nodes.sort_by_key(|node| node.sequence_start);
    let source = base_nodes[0];
    let deleted_base_node = base_nodes[1];
    let target = base_nodes[2];

    assert!(
        has_edge_for_sequence(&alignment, source, deleted_base_node, 0),
        "base path should include the deleted base node"
    );
    assert!(
        has_edge_for_sequence(&alignment, source, target, 1),
        "sample path should skip the deleted base node"
    );
}

#[test]
fn rejects_mismatched_aligned_lengths() {
    let input = "\
CLUSTAL W multiple sequence alignment

SeqA      ACGT
SeqB      ACG
";

    let mut parser = ClustalwParser::new(input.as_bytes());
    let error = parser
        .next()
        .expect("should emit parse result")
        .expect_err("should reject malformed alignment");
    assert_eq!(
        error.to_string(),
        "aligned sequence SeqB has length 3 but expected 4"
    );
}
