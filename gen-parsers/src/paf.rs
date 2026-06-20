use std::{collections::HashMap, io::BufRead, marker::PhantomData};

use gen_core::{HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand};
use gen_graph::{GenGraph, GraphNode};

use crate::{CigarOp, ParseError, ParsedMapping, add_path_edge, terminal_node};

pub struct PafParser<R> {
    mappings: std::vec::IntoIter<Result<ParsedMapping, ParseError>>,
    _reader: PhantomData<R>,
}

impl<R> PafParser<R>
where
    R: BufRead,
{
    pub fn new(reader: R) -> Self {
        let mappings = parse_paf_reader(reader);
        Self {
            mappings: mappings.into_iter(),
            _reader: PhantomData,
        }
    }
}

impl<R> Iterator for PafParser<R> {
    type Item = Result<ParsedMapping, ParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.mappings.next()
    }
}

fn parse_paf_reader<R>(reader: R) -> Vec<Result<ParsedMapping, ParseError>>
where
    R: BufRead,
{
    reader
        .lines()
        .enumerate()
        .filter_map(|(line_index, line)| {
            let line_number = line_index + 1;
            let line = match line {
                Ok(line) => line,
                Err(error) => return Some(Err(ParseError::Read(error.to_string()))),
            };
            if line.trim().is_empty() {
                return None;
            }
            Some(parse_paf_line(&line, line_number))
        })
        .collect()
}

fn parse_paf_line(line: &str, line_number: usize) -> Result<ParsedMapping, ParseError> {
    let fields = line.split('\t').collect::<Vec<_>>();
    if fields.len() < 12 {
        return Err(ParseError::PafTooFewFields {
            line_number,
            actual: fields.len(),
        });
    }

    let query_name = fields[0].to_string();
    let query_len = parse_paf_i64(fields[1], line_number, 2)?;
    let query_start = parse_paf_i64(fields[2], line_number, 3)?;
    let query_end = parse_paf_i64(fields[3], line_number, 4)?;
    let strand = match fields[4] {
        "+" => Strand::Forward,
        "-" => Strand::Reverse,
        strand => {
            return Err(ParseError::InvalidPafStrand {
                line_number,
                strand: strand.to_string(),
            });
        }
    };
    let target_name = fields[5].to_string();
    let target_len = parse_paf_i64(fields[6], line_number, 7)?;
    let target_start = parse_paf_i64(fields[7], line_number, 8)?;
    let target_end = parse_paf_i64(fields[8], line_number, 9)?;
    let matching_bases = parse_paf_i64(fields[9], line_number, 10)?;
    let block_len = parse_paf_i64(fields[10], line_number, 11)?;
    let mapping_quality = parse_paf_i64(fields[11], line_number, 12)?;

    let mut tags = HashMap::new();
    let mut cigar = None;
    for field in fields.iter().skip(12) {
        let mut parts = field.splitn(3, ':');
        let Some(key) = parts.next() else {
            continue;
        };
        let Some(tag_type) = parts.next() else {
            continue;
        };
        let Some(value) = parts.next() else {
            continue;
        };
        tags.insert(key.to_string(), format!("{tag_type}:{value}"));
        if key == "cg" && tag_type == "Z" {
            cigar = Some(parse_cigar(value, line_number)?);
        }
    }

    let Some(cigar) = cigar else {
        return Err(ParseError::MissingPafCigar { line_number });
    };

    let mut mapping = ParsedMapping {
        query_name,
        query_len,
        query_start,
        query_end,
        strand,
        target_name,
        target_len,
        target_start,
        target_end,
        matching_bases,
        block_len,
        mapping_quality,
        cigar,
        tags,
        graph: GenGraph::new(),
    };
    mapping.graph = build_paf_graph(&mapping);
    Ok(mapping)
}

fn parse_paf_i64(field: &str, line_number: usize, field_number: usize) -> Result<i64, ParseError> {
    field.parse().map_err(|_| ParseError::InvalidPafInteger {
        line_number,
        field: field_number,
    })
}

fn parse_cigar(cigar: &str, line_number: usize) -> Result<Vec<CigarOp>, ParseError> {
    let mut ops = Vec::new();
    let mut len = 0_i64;
    let mut has_digits = false;

    for character in cigar.chars() {
        if let Some(digit) = character.to_digit(10) {
            has_digits = true;
            len = len * 10 + i64::from(digit);
            continue;
        }

        if !has_digits || len == 0 {
            return Err(ParseError::InvalidCigar {
                line_number,
                cigar: cigar.to_string(),
            });
        }

        let op = match character {
            'M' => CigarOp::Match(len),
            'I' => CigarOp::Insertion(len),
            'D' => CigarOp::Deletion(len),
            'N' => CigarOp::ReferenceSkip(len),
            'S' => CigarOp::SoftClip(len),
            'H' => CigarOp::HardClip(len),
            '=' => CigarOp::Equal(len),
            'X' => CigarOp::Difference(len),
            _ => {
                return Err(ParseError::InvalidCigar {
                    line_number,
                    cigar: cigar.to_string(),
                });
            }
        };
        ops.push(op);
        len = 0;
        has_digits = false;
    }

    if has_digits || ops.is_empty() {
        return Err(ParseError::InvalidCigar {
            line_number,
            cigar: cigar.to_string(),
        });
    }

    Ok(ops)
}

fn build_paf_graph(mapping: &ParsedMapping) -> GenGraph {
    let mut graph = GenGraph::new();
    let mut target_previous = terminal_node(PATH_START_NODE_ID);
    let mut query_previous = terminal_node(PATH_START_NODE_ID);
    let end_node = terminal_node(PATH_END_NODE_ID);
    let mut target_offset = mapping.target_start;
    let mut query_offset = mapping.query_start;
    let target_node_id = HashId::convert_str(&format!(
        "paf:target:{}:{}",
        mapping.target_name, mapping.target_len
    ));
    let query_node_id = HashId::convert_str(&format!(
        "paf:query:{}:{}",
        mapping.query_name, mapping.query_len
    ));

    for op in &mapping.cigar {
        match *op {
            CigarOp::Match(len) | CigarOp::Equal(len) | CigarOp::Difference(len) => {
                let node = GraphNode {
                    node_id: target_node_id,
                    sequence_start: target_offset,
                    sequence_end: target_offset + len,
                };
                add_path_edge(&mut graph, target_previous, node, 0);
                add_path_edge(&mut graph, query_previous, node, 1);
                target_previous = node;
                query_previous = node;
                target_offset += len;
                query_offset += len;
            }
            CigarOp::Insertion(len) => {
                let node = GraphNode {
                    node_id: query_node_id,
                    sequence_start: query_offset,
                    sequence_end: query_offset + len,
                };
                add_path_edge(&mut graph, query_previous, node, 1);
                query_previous = node;
                query_offset += len;
            }
            CigarOp::Deletion(len) | CigarOp::ReferenceSkip(len) => {
                let node = GraphNode {
                    node_id: target_node_id,
                    sequence_start: target_offset,
                    sequence_end: target_offset + len,
                };
                add_path_edge(&mut graph, target_previous, node, 0);
                target_previous = node;
                target_offset += len;
            }
            CigarOp::SoftClip(len) => {
                query_offset += len;
            }
            CigarOp::HardClip(_) => {}
        }
    }

    add_path_edge(&mut graph, target_previous, end_node, 0);
    add_path_edge(&mut graph, query_previous, end_node, 1);
    graph
}

#[cfg(test)]
mod tests {
    use gen_core::{PATH_END_NODE_ID, PATH_START_NODE_ID, Strand};
    use gen_graph::GraphNode;
    use petgraph::visit::{EdgeRef as _, IntoEdgeReferences as _};
    use similar_asserts::assert_eq;

    use crate::{CigarOp, PafParser, ParsedMapping};

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
}
