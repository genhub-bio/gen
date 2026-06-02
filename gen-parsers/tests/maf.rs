use gen_core::{PATH_END_NODE_ID, PATH_START_NODE_ID};
use gen_graph::GraphNode;
use gen_parsers::{MafParser, ParsedAlignment};
use petgraph::visit::{EdgeRef as _, IntoEdgeReferences as _};
use similar_asserts::assert_eq;

fn parse_all(input: &str) -> Vec<ParsedAlignment> {
    MafParser::new(input.as_bytes())
        .collect::<Result<Vec<_>, _>>()
        .expect("should parse MAF alignments")
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
fn parses_maf_block_into_base_relative_graph() {
    let input = "\
##maf version=1 scoring=tba.v8
# generated example

a score=23262.0
s hg16.chr7    27707221 13 + 158545518 gcagctgaaaaca
s panTro1.chr6 28869787 13 + 161576975 gcagctgaaaaca
s mm4.chr6     53310102 13 + 151104725 ACAGCTGAAAATA
";

    let alignments = parse_all(input);
    assert_eq!(alignments.len(), 1);
    let alignment = &alignments[0];

    assert_eq!(alignment.base_name, "hg16.chr7");
    assert_eq!(
        alignment.sequence_order,
        vec!["hg16.chr7", "panTro1.chr6", "mm4.chr6"]
    );
    assert_eq!(
        alignment
            .aligned_sequences
            .get("hg16.chr7")
            .expect("should contain reference"),
        "GCAGCTGAAAACA"
    );

    let mut lengths = non_terminal_nodes(alignment)
        .iter()
        .map(GraphNode::length)
        .collect::<Vec<_>>();
    lengths.sort();
    assert_eq!(lengths, vec![1, 1, 1, 1, 1, 10]);
}

#[test]
fn parses_multiple_blocks_and_expands_dot_notation() {
    let input = "\
track name=sample
##maf version=1

a score=1
s ref.chr1 0 4 + 100 ACGT
s alt.chr1 5 4 + 100 ....

a score=2
s ref.chr1 10 3 + 100 GG-T
s alt.chr1 20 4 + 100 ..AT
";

    let alignments = parse_all(input);

    assert_eq!(alignments.len(), 2);
    assert_eq!(
        alignments[0]
            .aligned_sequences
            .get("alt.chr1")
            .expect("should contain first alternate"),
        "ACGT"
    );
    assert_eq!(
        alignments[1]
            .aligned_sequences
            .get("alt.chr1")
            .expect("should contain second alternate"),
        "GGAT"
    );
}

#[test]
fn deletion_paths_skip_base_variant_nodes() {
    let input = "\
##maf version=1

a score=3
s ref 0 4 + 10 ACGT
s alt 0 3 + 10 A-GT
";

    let alignments = parse_all(input);
    let alignment = &alignments[0];
    let mut nodes = non_terminal_nodes(alignment);
    assert_eq!(nodes.len(), 3);
    nodes.sort_by_key(|node| node.sequence_start);
    let source = nodes[0];
    let deleted_base_node = nodes[1];
    let target = nodes[2];

    assert!(
        has_edge_for_sequence(alignment, source, deleted_base_node, 0),
        "base path should include deleted base interval"
    );
    assert!(
        has_edge_for_sequence(alignment, source, target, 1),
        "alternate path should skip deleted base interval"
    );
}

#[test]
fn rejects_maf_size_mismatch() {
    let input = "\
##maf version=1

a score=4
s ref 0 4 + 10 ACGT
s alt 0 4 + 10 A-GT
";

    let mut parser = MafParser::new(input.as_bytes());
    let error = parser
        .next()
        .expect("should emit parse result")
        .expect_err("should reject malformed MAF size");
    assert_eq!(
        error.to_string(),
        "MAF line 5 sequence alt declares size 4 but has 3 non-gap bases"
    );
}
