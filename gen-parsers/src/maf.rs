use std::{collections::HashMap, io::BufRead, marker::PhantomData};

use gen_core::Strand;

use crate::{ParseError, ParsedAlignment, build_graph, validate_lengths};

pub struct MafParser<R> {
    alignments: std::vec::IntoIter<Result<ParsedAlignment, ParseError>>,
    _reader: PhantomData<R>,
}

impl<R> MafParser<R>
where
    R: BufRead,
{
    pub fn new(reader: R) -> Self {
        let alignments = parse_maf_reader(reader);
        Self {
            alignments: alignments.into_iter(),
            _reader: PhantomData,
        }
    }
}

impl<R> Iterator for MafParser<R> {
    type Item = Result<ParsedAlignment, ParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.alignments.next()
    }
}

fn parse_maf_reader<R>(reader: R) -> Vec<Result<ParsedAlignment, ParseError>>
where
    R: BufRead,
{
    let mut results = Vec::new();
    let mut rows = Vec::new();
    let mut in_alignment = false;

    for (line_index, line) in reader.lines().enumerate() {
        let line_number = line_index + 1;
        let line = match line {
            Ok(line) => line,
            Err(error) => {
                results.push(Err(ParseError::Read(error.to_string())));
                continue;
            }
        };
        let trimmed = line.trim();

        if trimmed.is_empty() {
            flush_maf_block(&mut results, &mut rows, line_number, &mut in_alignment);
            continue;
        }
        if should_skip_maf_line(trimmed) {
            continue;
        }
        if trimmed.starts_with("a ") || trimmed == "a" {
            flush_maf_block(&mut results, &mut rows, line_number, &mut in_alignment);
            in_alignment = true;
            continue;
        }
        if trimmed.starts_with("s ") {
            match parse_maf_sequence_row(trimmed, line_number) {
                Ok(row) => rows.push(row),
                Err(error) => results.push(Err(error)),
            }
        }
    }

    flush_maf_block(&mut results, &mut rows, 0, &mut in_alignment);
    results
}

fn should_skip_maf_line(line: &str) -> bool {
    line.starts_with('#')
        || line.starts_with("track ")
        || line.starts_with("i ")
        || line.starts_with("e ")
        || line.starts_with("q ")
}

fn flush_maf_block(
    results: &mut Vec<Result<ParsedAlignment, ParseError>>,
    rows: &mut Vec<MafSequenceRow>,
    line_number: usize,
    in_alignment: &mut bool,
) {
    if !*in_alignment {
        return;
    }
    if rows.is_empty() {
        results.push(Err(ParseError::MafBlockWithoutSequences { line_number }));
        *in_alignment = false;
        return;
    }

    let result = build_maf_alignment(rows);
    results.push(result);
    rows.clear();
    *in_alignment = false;
}

fn parse_maf_sequence_row(line: &str, line_number: usize) -> Result<MafSequenceRow, ParseError> {
    let fields = line.split_whitespace().collect::<Vec<_>>();
    if fields.len() < 7 {
        return Err(ParseError::MafTooFewFields {
            line_number,
            actual: fields.len(),
        });
    }

    let name = fields[1].to_string();
    let _start = parse_maf_i64(fields[2], line_number, 3)?;
    let size = parse_maf_i64(fields[3], line_number, 4)?;
    let _strand = match fields[4] {
        "+" => Strand::Forward,
        "-" => Strand::Reverse,
        strand => {
            return Err(ParseError::InvalidMafStrand {
                line_number,
                strand: strand.to_string(),
            });
        }
    };
    let _source_size = parse_maf_i64(fields[5], line_number, 6)?;
    let text = fields[6].to_ascii_uppercase();
    let actual = text.chars().filter(|character| *character != '-').count();
    if actual != size as usize {
        return Err(ParseError::MafSizeMismatch {
            line_number,
            name,
            declared: size,
            actual,
        });
    }

    Ok(MafSequenceRow { name, text })
}

fn parse_maf_i64(field: &str, line_number: usize, field_number: usize) -> Result<i64, ParseError> {
    field.parse().map_err(|_| ParseError::InvalidMafInteger {
        line_number,
        field: field_number,
    })
}

fn build_maf_alignment(rows: &[MafSequenceRow]) -> Result<ParsedAlignment, ParseError> {
    let base = rows.first().expect("should have MAF sequence row");
    let mut sequence_order = Vec::new();
    let mut aligned_sequences = HashMap::new();

    for row in rows {
        sequence_order.push(row.name.clone());
        let text = if row.name == base.name {
            row.text.clone()
        } else {
            expand_maf_dots(&base.text, &row.text)
        };
        aligned_sequences.insert(row.name.clone(), text);
    }

    validate_lengths(&sequence_order, &aligned_sequences)?;
    let graph = build_graph(&base.name, &sequence_order, &aligned_sequences);
    Ok(ParsedAlignment {
        base_name: base.name.clone(),
        sequence_order,
        graph,
        aligned_sequences,
    })
}

fn expand_maf_dots(base: &str, text: &str) -> String {
    base.chars()
        .zip(text.chars())
        .map(|(base_character, character)| {
            if character == '.' {
                base_character
            } else {
                character
            }
        })
        .collect()
}

#[derive(Clone, Debug)]
struct MafSequenceRow {
    name: String,
    text: String,
}

#[cfg(test)]
mod tests {
    use gen_core::{PATH_END_NODE_ID, PATH_START_NODE_ID};
    use gen_graph::GraphNode;
    use petgraph::visit::{EdgeRef as _, IntoEdgeReferences as _};
    use similar_asserts::assert_eq;

    use crate::{MafParser, ParsedAlignment};

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
}
