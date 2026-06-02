use gen_core::{PATH_END_NODE_ID, PATH_START_NODE_ID, Strand};
use gen_graph::GraphNode;
use gen_parsers::{ParsedMapping, PslParser};
use petgraph::visit::{EdgeRef as _, IntoEdgeReferences as _};
use similar_asserts::assert_eq;

fn parse_all(input: &str) -> Vec<ParsedMapping> {
    PslParser::new(input.as_bytes())
        .collect::<Result<Vec<_>, _>>()
        .expect("should parse PSL mappings")
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
fn parses_psl_blocks_into_target_relative_graph() {
    let input = "\
psLayout version 3

match mis- rep. N's Q gap Q gap T gap T gap strand Q         Q    Q     Q   T         T    T     T   block blockSizes qStarts  tStarts
----- ---- ---- --- ----- ----- ----- ----- ------ -------- ---- ----- --- -------- ---- ----- --- ----- ---------- -------- --------
59    9    0    0   1     823   1     96    +      query1    1200 10    33  target1   5000 100   128 2     10,13,     10,20,   100,115,
";

    let mappings = parse_all(input);
    assert_eq!(mappings.len(), 1);
    let mapping = &mappings[0];

    assert_eq!(mapping.query_name, "query1");
    assert_eq!(mapping.query_len, 1200);
    assert_eq!(mapping.query_start, 10);
    assert_eq!(mapping.query_end, 33);
    assert_eq!(mapping.strand, Strand::Forward);
    assert_eq!(mapping.target_name, "target1");
    assert_eq!(mapping.target_len, 5000);
    assert_eq!(mapping.target_start, 100);
    assert_eq!(mapping.target_end, 128);
    assert_eq!(mapping.matching_bases, 59);
    assert_eq!(mapping.block_len, 23);
    assert_eq!(mapping.mapping_quality, 0);
    assert_eq!(mapping.tags.get("psl:misMatches"), Some(&"9".to_string()));

    let mut lengths = non_terminal_nodes(mapping)
        .iter()
        .map(GraphNode::length)
        .collect::<Vec<_>>();
    lengths.sort();
    assert_eq!(lengths, vec![5, 10, 13]);
}

#[test]
fn query_path_uses_query_gap_and_skips_target_gap() {
    let input = "23 0 0 0 1 2 1 5 + query1 100 10 35 target1 200 50 78 2 10,13, 10,22, 50,65,\n";

    let mappings = parse_all(input);
    let mapping = &mappings[0];
    let mut target_nodes = non_terminal_nodes(mapping)
        .into_iter()
        .filter(|node| node.sequence_start >= 50)
        .collect::<Vec<_>>();
    target_nodes.sort_by_key(|node| node.sequence_start);

    let target_left = target_nodes[0];
    let target_gap = target_nodes[1];
    let target_right = target_nodes[2];
    let query_gap = non_terminal_nodes(mapping)
        .into_iter()
        .find(|node| node.sequence_start == 20 && node.sequence_end == 22)
        .expect("should contain query insertion interval");

    assert!(
        has_edge_for_sequence(mapping, target_left, target_gap, 0),
        "target path should include target-only gap interval"
    );
    assert!(
        has_edge_for_sequence(mapping, target_left, query_gap, 1),
        "query path should include query-only gap interval"
    );
    assert!(
        has_edge_for_sequence(mapping, query_gap, target_right, 1),
        "query path should rejoin the next aligned target block"
    );
}

#[test]
fn normalizes_negative_query_block_starts() {
    let input =
        "38 0 0 0 1 14 1 17 - query1 61 4 56 target1 1000 100 155 2 20,18, 5,39, 100,137,\n";

    let mappings = parse_all(input);
    let mapping = &mappings[0];

    assert_eq!(mapping.strand, Strand::Reverse);
    assert_eq!(mapping.query_start, 4);
    assert_eq!(mapping.query_end, 56);
    assert!(
        non_terminal_nodes(mapping)
            .iter()
            .any(|node| node.sequence_start == 22 && node.sequence_end == 36),
        "minus-strand query gap should use normalized forward query coordinates"
    );
}

#[test]
fn rejects_psl_with_mismatched_block_lists() {
    let input = "23 0 0 0 0 0 0 0 + query1 100 10 33 target1 200 50 73 2 10,13, 10, 50,65,\n";

    let mut parser = PslParser::new(input.as_bytes());
    let error = parser
        .next()
        .expect("should emit parse result")
        .expect_err("should reject malformed PSL block lists");
    assert_eq!(
        error.to_string(),
        "PSL line 1 blockCount is 2 but blockSizes, qStarts, and tStarts contain 2, 1, and 2 entries"
    );
}
