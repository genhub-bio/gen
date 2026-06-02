use gen_core::{PATH_END_NODE_ID, PATH_START_NODE_ID};
use gen_graph::GraphNode;
use gen_parsers::{BlastParser, ParsedAlignment};
use petgraph::visit::{EdgeRef as _, IntoEdgeReferences as _};
use similar_asserts::assert_eq;

fn parse_all(input: &str) -> Vec<ParsedAlignment> {
    BlastParser::new(input.as_bytes())
        .collect::<Result<Vec<_>, _>>()
        .expect("should parse BLAST pairwise output")
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
fn parses_default_pairwise_hsp_into_base_relative_graph() {
    let input = "\
BLASTN 2.16.0+

Query= SeqA

>SeqB
Length=28

 Score = 52.8 bits (28),  Expect = 2e-09
 Identities = 26/28 (93%), Gaps = 2/28 (7%)
 Strand=Plus/Plus

Query  1   MAGAASAVAAALAAA--AAGAAATAAAG  26
           |||||||||||||||  |||||||||||
Sbjct  1   MAGAASAVAAALAAAGAAAGAAATAAAG  28
";

    let alignments = parse_all(input);
    assert_eq!(alignments.len(), 1);
    let alignment = &alignments[0];

    assert_eq!(alignment.base_name, "SeqA");
    assert_eq!(alignment.sequence_order, vec!["SeqA", "SeqB"]);
    assert_eq!(
        alignment
            .aligned_sequences
            .get("SeqA")
            .expect("should contain query"),
        "MAGAASAVAAALAAA--AAGAAATAAAG"
    );
    assert_eq!(
        alignment.ungapped_sequence("SeqA"),
        Some("MAGAASAVAAALAAAAAGAAATAAAG".to_string())
    );

    let mut lengths = non_terminal_nodes(alignment)
        .iter()
        .map(GraphNode::length)
        .collect::<Vec<_>>();
    lengths.sort();
    assert_eq!(lengths, vec![2, 11, 15]);
}

#[test]
fn parses_multiple_hsps_as_multiple_alignments() {
    let input = "\
BLASTN 2.16.0+

Query= QueryOne

>SubjectOne
Length=10

 Score = 10.0 bits (5),  Expect = 0.01

Query  1  ACGT  4
          ||||
Sbjct  5  ACGT  8

 Score = 8.0 bits (4),  Expect = 0.02

Query  7  GG-T  9
          || |
Sbjct  1  GGAT  4
";

    let alignments = parse_all(input);

    assert_eq!(alignments.len(), 2);
    assert_eq!(alignments[0].base_name, "QueryOne");
    assert_eq!(alignments[0].sequence_order, vec!["QueryOne", "SubjectOne"]);
    assert_eq!(
        alignments[1]
            .aligned_sequences
            .get("QueryOne")
            .expect("should contain query"),
        "GG-T"
    );
}

#[test]
fn deletion_paths_skip_query_variant_nodes() {
    let input = "\
BLASTN 2.16.0+

Query= Base

>Sample

 Score = 8.0 bits (4),  Expect = 0.02

Query  1  ACGT  4
          | ||
Sbjct  1  A-GT  3
";

    let alignments = parse_all(input);
    let alignment = &alignments[0];
    let mut nodes = non_terminal_nodes(alignment);
    assert_eq!(nodes.len(), 3);
    nodes.sort_by_key(|node| node.sequence_start);
    let source = nodes[0];
    let deleted_query_node = nodes[1];
    let target = nodes[2];

    assert!(
        has_edge_for_sequence(alignment, source, deleted_query_node, 0),
        "query path should include the deleted query node"
    );
    assert!(
        has_edge_for_sequence(alignment, source, target, 1),
        "subject path should skip the deleted query node"
    );
}

#[test]
fn rejects_hsp_missing_subject_sequence() {
    let input = "\
BLASTN 2.16.0+

Query= SeqA

>SeqB

 Score = 10.0 bits (5),  Expect = 0.01

Query  1  ACGT  4
";

    let mut parser = BlastParser::new(input.as_bytes());
    let error = parser
        .next()
        .expect("should emit parse result")
        .expect_err("should reject incomplete HSP");
    assert_eq!(
        error.to_string(),
        "BLAST HSP is missing a Sbjct sequence row"
    );
}
