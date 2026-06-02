use gen_core::{PATH_END_NODE_ID, PATH_START_NODE_ID, Strand};
use gen_graph::GraphNode;
use gen_parsers::{CigarOp, PafParser, ParsedMapping};
use petgraph::visit::{EdgeRef as _, IntoEdgeReferences as _};
use similar_asserts::assert_eq;

fn parse_all(input: &str) -> Vec<ParsedMapping> {
    PafParser::new(input.as_bytes())
        .collect::<Result<Vec<_>, _>>()
        .expect("should parse PAF mappings")
}

fn non_terminal_nodes(mapping: &ParsedMapping) -> Vec<GraphNode> {
    let mut nodes = mapping
        .graph
        .nodes()
        .filter(|node| node.node_id != PATH_START_NODE_ID && node.node_id != PATH_END_NODE_ID)
        .collect::<Vec<_>>();
    nodes.sort();
    nodes
}

fn has_edge_for_sequence(
    mapping: &ParsedMapping,
    source: GraphNode,
    target: GraphNode,
    index: i64,
) -> bool {
    mapping.graph.edge_references().any(|edge| {
        edge.source() == source
            && edge.target() == target
            && edge
                .weight()
                .iter()
                .any(|metadata| metadata.chromosome_index == index)
    })
}

#[test]
fn parses_paf_with_cigar_into_target_relative_graph() {
    let input =
        "query1\t100\t10\t24\t+\ttarget1\t200\t50\t63\t12\t14\t60\tcg:Z:4M2I3M1D5M\ttp:A:P\n";

    let mappings = parse_all(input);
    assert_eq!(mappings.len(), 1);
    let mapping = &mappings[0];

    assert_eq!(mapping.query_name, "query1");
    assert_eq!(mapping.query_len, 100);
    assert_eq!(mapping.query_start, 10);
    assert_eq!(mapping.query_end, 24);
    assert_eq!(mapping.strand, Strand::Forward);
    assert_eq!(mapping.target_name, "target1");
    assert_eq!(mapping.target_len, 200);
    assert_eq!(mapping.target_start, 50);
    assert_eq!(mapping.target_end, 63);
    assert_eq!(mapping.matching_bases, 12);
    assert_eq!(mapping.block_len, 14);
    assert_eq!(mapping.mapping_quality, 60);
    assert_eq!(
        mapping.cigar,
        vec![
            CigarOp::Match(4),
            CigarOp::Insertion(2),
            CigarOp::Match(3),
            CigarOp::Deletion(1),
            CigarOp::Match(5),
        ]
    );
    assert_eq!(
        mapping.tags.get("tp").expect("should retain optional tag"),
        "A:P"
    );

    let mut lengths = non_terminal_nodes(mapping)
        .iter()
        .map(GraphNode::length)
        .collect::<Vec<_>>();
    lengths.sort();
    assert_eq!(lengths, vec![1, 2, 3, 4, 5]);
}

#[test]
fn query_path_uses_insertions_and_skips_target_deletions() {
    let input = "query1\t100\t10\t24\t+\ttarget1\t200\t50\t63\t12\t14\t60\tcg:Z:4M2I3M1D5M\n";
    let mappings = parse_all(input);
    let mapping = &mappings[0];
    let mut target_nodes = non_terminal_nodes(mapping)
        .into_iter()
        .filter(|node| node.sequence_start >= 50)
        .collect::<Vec<_>>();
    target_nodes.sort_by_key(|node| node.sequence_start);

    let target_match_left = target_nodes[0];
    let target_match_middle = target_nodes[1];
    let target_deletion = target_nodes[2];
    let target_match_right = target_nodes[3];
    let query_insertion = non_terminal_nodes(mapping)
        .into_iter()
        .find(|node| node.sequence_start == 14 && node.sequence_end == 16)
        .expect("should contain query insertion node");

    assert!(
        has_edge_for_sequence(mapping, target_match_left, query_insertion, 1),
        "query path should enter inserted query interval"
    );
    assert!(
        has_edge_for_sequence(mapping, query_insertion, target_match_middle, 1),
        "query path should leave inserted query interval"
    );
    assert!(
        has_edge_for_sequence(mapping, target_match_middle, target_deletion, 0),
        "target path should include deleted target interval"
    );
    assert!(
        has_edge_for_sequence(mapping, target_match_middle, target_match_right, 1),
        "query path should skip deleted target interval"
    );
}

#[test]
fn parses_multiple_paf_lines_and_reverse_strand() {
    let input = "\
query1\t100\t0\t4\t+\ttarget1\t200\t0\t4\t4\t4\t60\tcg:Z:4M
query2\t80\t5\t10\t-\ttarget2\t90\t30\t35\t5\t5\t255\tcg:Z:5=
";

    let mappings = parse_all(input);

    assert_eq!(mappings.len(), 2);
    assert_eq!(mappings[0].strand, Strand::Forward);
    assert_eq!(mappings[1].strand, Strand::Reverse);
    assert_eq!(mappings[1].cigar, vec![CigarOp::Equal(5)]);
}

#[test]
fn rejects_paf_without_cigar_tag() {
    let input = "query1\t100\t10\t24\t+\ttarget1\t200\t50\t63\t12\t14\t60\n";

    let mut parser = PafParser::new(input.as_bytes());
    let error = parser
        .next()
        .expect("should emit parse result")
        .expect_err("should reject PAF without CIGAR");
    assert_eq!(
        error.to_string(),
        "PAF line 1 is missing required cg:Z CIGAR tag"
    );
}
