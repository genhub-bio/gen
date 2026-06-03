use std::{collections::HashMap, io::BufRead};

use crate::{ParseError, ParsedAlignment, build_graph, validate_lengths};

pub struct ClustalwParser<R> {
    reader: Option<R>,
}

impl<R> ClustalwParser<R> {
    pub const fn new(reader: R) -> Self {
        Self {
            reader: Some(reader),
        }
    }
}

impl<R> Iterator for ClustalwParser<R>
where
    R: BufRead,
{
    type Item = Result<ParsedAlignment, ParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.reader.take().map(parse_reader)
    }
}

fn parse_reader<R>(reader: R) -> Result<ParsedAlignment, ParseError>
where
    R: BufRead,
{
    let mut lines = reader.lines();
    let Some(header) = lines.next() else {
        return Err(ParseError::MissingHeader);
    };
    let header = header.map_err(|error| ParseError::Read(error.to_string()))?;
    if !is_clustal_header(&header) {
        return Err(ParseError::MissingHeader);
    }

    let mut sequence_order = Vec::new();
    let mut aligned_sequences = HashMap::new();
    let mut block = Vec::new();

    for (line_index, line) in lines.enumerate() {
        let line_number = line_index + 2;
        let line = line.map_err(|error| ParseError::Read(error.to_string()))?;
        if line.trim().is_empty() {
            flush_block(&mut block, &mut aligned_sequences)?;
            continue;
        }
        if is_conservation_line(&line) {
            continue;
        }

        let mut fields = line.split_whitespace();
        let name = fields
            .next()
            .ok_or(ParseError::MissingSequenceFragment { line_number })?;
        let fragment = fields
            .next()
            .ok_or(ParseError::MissingSequenceFragment { line_number })?;

        let name = name.to_string();
        if !aligned_sequences.contains_key(&name) {
            sequence_order.push(name.clone());
        }

        let count = fields.next().and_then(|field| field.parse::<usize>().ok());
        block.push(BlockRow {
            name: name.clone(),
            fragment: fragment.to_ascii_uppercase(),
            count,
        });
    }
    flush_block(&mut block, &mut aligned_sequences)?;

    if sequence_order.is_empty() {
        return Err(ParseError::NoSequences);
    }

    validate_lengths(&sequence_order, &aligned_sequences)?;
    let base_name = sequence_order[0].clone();
    let graph = build_graph(&base_name, &sequence_order, &aligned_sequences);

    Ok(ParsedAlignment {
        base_name,
        sequence_order,
        aligned_sequences,
        graph,
    })
}

fn flush_block(
    block: &mut Vec<BlockRow>,
    aligned_sequences: &mut HashMap<String, String>,
) -> Result<(), ParseError> {
    if block.is_empty() {
        return Ok(());
    }

    let expected = block
        .iter()
        .map(|row| row.fragment.len())
        .max()
        .expect("should have block rows");
    let has_counts = block.iter().any(|row| row.count.is_some());
    for row in block.drain(..) {
        let actual = row.fragment.len();
        if actual != expected && !has_counts {
            return Err(ParseError::MismatchedAlignedLength {
                name: row.name,
                actual,
                expected,
            });
        }

        let sequence = aligned_sequences.entry(row.name).or_default();
        sequence.push_str(&row.fragment);
        for _ in actual..expected {
            sequence.push('-');
        }
    }

    Ok(())
}

fn is_clustal_header(header: &str) -> bool {
    let header = header.trim_start();
    header.starts_with("CLUSTAL")
}

fn is_conservation_line(line: &str) -> bool {
    line.starts_with(char::is_whitespace)
        && line
            .trim()
            .chars()
            .all(|character| matches!(character, '*' | ':' | '.' | ' '))
}

#[derive(Debug)]
struct BlockRow {
    name: String,
    fragment: String,
    count: Option<usize>,
}

#[cfg(test)]
mod tests {
    use gen_core::{PATH_END_NODE_ID, PATH_START_NODE_ID};
    use gen_graph::GraphNode;
    use petgraph::visit::{EdgeRef as _, IntoEdgeReferences as _};
    use similar_asserts::assert_eq;

    use crate::{ClustalwParser, ParsedAlignment};

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
}
