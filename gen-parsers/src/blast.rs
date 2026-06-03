use std::{collections::HashMap, io::BufRead, marker::PhantomData};

use crate::{ParseError, ParsedAlignment, build_graph, validate_lengths};

pub struct BlastParser<R> {
    alignments: std::vec::IntoIter<Result<ParsedAlignment, ParseError>>,
    _reader: PhantomData<R>,
}

impl<R> BlastParser<R>
where
    R: BufRead,
{
    pub fn new(reader: R) -> Self {
        let alignments = parse_blast_reader(reader);
        Self {
            alignments: alignments.into_iter(),
            _reader: PhantomData,
        }
    }
}

impl<R> Iterator for BlastParser<R> {
    type Item = Result<ParsedAlignment, ParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.alignments.next()
    }
}

fn parse_blast_reader<R>(reader: R) -> Vec<Result<ParsedAlignment, ParseError>>
where
    R: BufRead,
{
    let lines = match reader.lines().collect::<Result<Vec<_>, _>>() {
        Ok(lines) => lines,
        Err(error) => return vec![Err(ParseError::Read(error.to_string()))],
    };

    let mut query_name = String::from("Query");
    let mut subject_name = String::from("Sbjct");
    let mut query_parts = Vec::new();
    let mut subject_parts = Vec::new();
    let mut results = Vec::new();
    let mut saw_score = false;

    for line in lines {
        let trimmed = line.trim();
        if let Some(name) = trimmed.strip_prefix("Query=") {
            query_name = name.trim().to_string();
            continue;
        }
        if let Some(name) = trimmed.strip_prefix('>') {
            flush_blast_hsp(
                &mut results,
                &query_name,
                &subject_name,
                &mut query_parts,
                &mut subject_parts,
            );
            subject_name = name
                .split_whitespace()
                .next()
                .unwrap_or("Sbjct")
                .to_string();
            saw_score = false;
            continue;
        }
        if trimmed.starts_with("Score =") {
            flush_blast_hsp(
                &mut results,
                &query_name,
                &subject_name,
                &mut query_parts,
                &mut subject_parts,
            );
            saw_score = true;
            continue;
        }
        if let Some(fragment) = blast_alignment_fragment(trimmed, "Query") {
            query_parts.push(fragment.to_ascii_uppercase());
            continue;
        }
        if let Some(fragment) = blast_alignment_fragment(trimmed, "Sbjct") {
            subject_parts.push(fragment.to_ascii_uppercase());
        }
    }

    flush_blast_hsp(
        &mut results,
        &query_name,
        &subject_name,
        &mut query_parts,
        &mut subject_parts,
    );

    if results.is_empty() && saw_score {
        results.push(Err(ParseError::MissingBlastSubject));
    } else if results.is_empty() {
        results.push(Err(ParseError::NoBlastAlignments));
    }

    results
}

fn flush_blast_hsp(
    results: &mut Vec<Result<ParsedAlignment, ParseError>>,
    query_name: &str,
    subject_name: &str,
    query_parts: &mut Vec<String>,
    subject_parts: &mut Vec<String>,
) {
    if query_parts.is_empty() && subject_parts.is_empty() {
        return;
    }
    if subject_parts.is_empty() {
        results.push(Err(ParseError::MissingBlastSubject));
        query_parts.clear();
        return;
    }

    let query_sequence = query_parts.join("");
    let subject_sequence = subject_parts.join("");
    query_parts.clear();
    subject_parts.clear();

    let sequence_order = vec![query_name.to_string(), subject_name.to_string()];
    let mut aligned_sequences = HashMap::new();
    aligned_sequences.insert(query_name.to_string(), query_sequence);
    aligned_sequences.insert(subject_name.to_string(), subject_sequence);

    let result = validate_lengths(&sequence_order, &aligned_sequences).map(|()| ParsedAlignment {
        base_name: query_name.to_string(),
        sequence_order: sequence_order.clone(),
        graph: build_graph(query_name, &sequence_order, &aligned_sequences),
        aligned_sequences,
    });
    results.push(result);
}

fn blast_alignment_fragment<'a>(line: &'a str, label: &str) -> Option<&'a str> {
    let mut fields = line.split_whitespace();
    if fields.next()? != label {
        return None;
    }
    let _start = fields.next()?;
    fields.next()
}

#[cfg(test)]
mod tests {
    use gen_core::{PATH_END_NODE_ID, PATH_START_NODE_ID};
    use gen_graph::GraphNode;
    use petgraph::visit::{EdgeRef as _, IntoEdgeReferences as _};
    use similar_asserts::assert_eq;

    use crate::{BlastParser, ParsedAlignment};

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
}
