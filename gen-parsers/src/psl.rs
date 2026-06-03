use std::{collections::HashMap, io::BufRead, marker::PhantomData};

use gen_core::{HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand};
use gen_graph::{GenGraph, GraphNode};

use crate::{ParseError, ParsedMapping, add_path_edge, terminal_node};

pub struct PslParser<R> {
    mappings: std::vec::IntoIter<Result<ParsedMapping, ParseError>>,
    _reader: PhantomData<R>,
}

impl<R> PslParser<R>
where
    R: BufRead,
{
    pub fn new(reader: R) -> Self {
        let mappings = parse_psl_reader(reader);
        Self {
            mappings: mappings.into_iter(),
            _reader: PhantomData,
        }
    }
}

impl<R> Iterator for PslParser<R> {
    type Item = Result<ParsedMapping, ParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.mappings.next()
    }
}

fn parse_psl_reader<R>(reader: R) -> Vec<Result<ParsedMapping, ParseError>>
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
            if should_skip_psl_line(&line) {
                return None;
            }
            Some(parse_psl_line(&line, line_number))
        })
        .collect()
}

fn should_skip_psl_line(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.is_empty()
        || trimmed.starts_with("psLayout")
        || trimmed.starts_with("match ")
        || trimmed.starts_with("-----")
        || trimmed.starts_with("browser ")
        || trimmed.starts_with("track ")
}

fn parse_psl_line(line: &str, line_number: usize) -> Result<ParsedMapping, ParseError> {
    let fields = line.split_whitespace().collect::<Vec<_>>();
    if fields.len() < 21 {
        return Err(ParseError::PslTooFewFields {
            line_number,
            actual: fields.len(),
        });
    }

    let matches = parse_psl_i64(fields[0], line_number, 1)?;
    let mismatches = parse_psl_i64(fields[1], line_number, 2)?;
    let rep_matches = parse_psl_i64(fields[2], line_number, 3)?;
    let n_count = parse_psl_i64(fields[3], line_number, 4)?;
    let query_insert_count = parse_psl_i64(fields[4], line_number, 5)?;
    let query_insert_bases = parse_psl_i64(fields[5], line_number, 6)?;
    let target_insert_count = parse_psl_i64(fields[6], line_number, 7)?;
    let target_insert_bases = parse_psl_i64(fields[7], line_number, 8)?;
    let strand = parse_psl_strand(fields[8], line_number)?;
    let query_name = fields[9].to_string();
    let query_len = parse_psl_i64(fields[10], line_number, 11)?;
    let query_start = parse_psl_i64(fields[11], line_number, 12)?;
    let query_end = parse_psl_i64(fields[12], line_number, 13)?;
    let target_name = fields[13].to_string();
    let target_len = parse_psl_i64(fields[14], line_number, 15)?;
    let target_start = parse_psl_i64(fields[15], line_number, 16)?;
    let target_end = parse_psl_i64(fields[16], line_number, 17)?;
    let block_count = parse_psl_usize(fields[17], line_number, 18)?;
    let block_sizes = parse_psl_i64_list(fields[18], line_number, 19)?;
    let query_starts = parse_psl_i64_list(fields[19], line_number, 20)?;
    let target_starts = parse_psl_i64_list(fields[20], line_number, 21)?;

    if block_sizes.len() != block_count
        || query_starts.len() != block_count
        || target_starts.len() != block_count
    {
        return Err(ParseError::MismatchedPslBlocks {
            line_number,
            block_count,
            block_sizes: block_sizes.len(),
            query_starts: query_starts.len(),
            target_starts: target_starts.len(),
        });
    }

    let blocks = block_sizes
        .iter()
        .zip(query_starts.iter())
        .zip(target_starts.iter())
        .map(|((size, query_start), target_start)| PslBlock {
            size: *size,
            query_start: normalize_psl_query_start(*query_start, *size, query_len, strand),
            target_start: *target_start,
        })
        .collect::<Vec<_>>();
    let mut tags = HashMap::new();
    tags.insert("psl:misMatches".to_string(), mismatches.to_string());
    tags.insert("psl:repMatches".to_string(), rep_matches.to_string());
    tags.insert("psl:nCount".to_string(), n_count.to_string());
    tags.insert("psl:qNumInsert".to_string(), query_insert_count.to_string());
    tags.insert(
        "psl:qBaseInsert".to_string(),
        query_insert_bases.to_string(),
    );
    tags.insert(
        "psl:tNumInsert".to_string(),
        target_insert_count.to_string(),
    );
    tags.insert(
        "psl:tBaseInsert".to_string(),
        target_insert_bases.to_string(),
    );

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
        matching_bases: matches,
        block_len: block_sizes.iter().sum(),
        mapping_quality: 0,
        cigar: Vec::new(),
        tags,
        graph: GenGraph::new(),
    };
    mapping.graph = build_psl_graph(&mapping, &blocks);
    Ok(mapping)
}

fn parse_psl_i64(field: &str, line_number: usize, field_number: usize) -> Result<i64, ParseError> {
    field.parse().map_err(|_| ParseError::InvalidPslInteger {
        line_number,
        field: field_number,
    })
}

fn parse_psl_usize(
    field: &str,
    line_number: usize,
    field_number: usize,
) -> Result<usize, ParseError> {
    field.parse().map_err(|_| ParseError::InvalidPslInteger {
        line_number,
        field: field_number,
    })
}

fn parse_psl_i64_list(
    field: &str,
    line_number: usize,
    field_number: usize,
) -> Result<Vec<i64>, ParseError> {
    field
        .trim_end_matches(',')
        .split(',')
        .filter(|value| !value.is_empty())
        .map(|value| parse_psl_i64(value, line_number, field_number))
        .collect()
}

fn parse_psl_strand(field: &str, line_number: usize) -> Result<Strand, ParseError> {
    match field.chars().next() {
        Some('+') => Ok(Strand::Forward),
        Some('-') => Ok(Strand::Reverse),
        _ => Err(ParseError::InvalidPslStrand {
            line_number,
            strand: field.to_string(),
        }),
    }
}

fn normalize_psl_query_start(query_start: i64, size: i64, query_len: i64, strand: Strand) -> i64 {
    if strand == Strand::Reverse {
        query_len - query_start - size
    } else {
        query_start
    }
}

fn build_psl_graph(mapping: &ParsedMapping, blocks: &[PslBlock]) -> GenGraph {
    let mut graph = GenGraph::new();
    let mut target_previous = terminal_node(PATH_START_NODE_ID);
    let mut query_previous = terminal_node(PATH_START_NODE_ID);
    let end_node = terminal_node(PATH_END_NODE_ID);
    let target_node_id = HashId::convert_str(&format!(
        "psl:target:{}:{}",
        mapping.target_name, mapping.target_len
    ));
    let query_node_id = HashId::convert_str(&format!(
        "psl:query:{}:{}",
        mapping.query_name, mapping.query_len
    ));

    for (index, block) in blocks.iter().enumerate() {
        if let Some(previous_block) = index
            .checked_sub(1)
            .and_then(|previous| blocks.get(previous))
        {
            let target_gap_start = previous_block.target_start + previous_block.size;
            let target_gap_end = block.target_start;
            let (query_gap_start, query_gap_end) =
                psl_query_gap(previous_block, block, mapping.strand);

            if target_gap_end > target_gap_start {
                let node = GraphNode {
                    node_id: target_node_id,
                    sequence_start: target_gap_start,
                    sequence_end: target_gap_end,
                };
                add_path_edge(&mut graph, target_previous, node, 0);
                target_previous = node;
            }

            if query_gap_end > query_gap_start {
                let node = GraphNode {
                    node_id: query_node_id,
                    sequence_start: query_gap_start,
                    sequence_end: query_gap_end,
                };
                add_path_edge(&mut graph, query_previous, node, 1);
                query_previous = node;
            }
        }

        let node = GraphNode {
            node_id: target_node_id,
            sequence_start: block.target_start,
            sequence_end: block.target_start + block.size,
        };
        add_path_edge(&mut graph, target_previous, node, 0);
        add_path_edge(&mut graph, query_previous, node, 1);
        target_previous = node;
        query_previous = node;
    }

    add_path_edge(&mut graph, target_previous, end_node, 0);
    add_path_edge(&mut graph, query_previous, end_node, 1);
    graph
}

fn psl_query_gap(previous_block: &PslBlock, block: &PslBlock, strand: Strand) -> (i64, i64) {
    if strand == Strand::Reverse {
        (block.query_start + block.size, previous_block.query_start)
    } else {
        (
            previous_block.query_start + previous_block.size,
            block.query_start,
        )
    }
}

#[derive(Clone, Copy, Debug)]
struct PslBlock {
    size: i64,
    query_start: i64,
    target_start: i64,
}

#[cfg(test)]
mod tests {
    use gen_core::{PATH_END_NODE_ID, PATH_START_NODE_ID, Strand};
    use gen_graph::GraphNode;
    use petgraph::visit::{EdgeRef as _, IntoEdgeReferences as _};
    use similar_asserts::assert_eq;

    use crate::{ParsedMapping, PslParser};

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
        let input =
            "23 0 0 0 1 2 1 5 + query1 100 10 35 target1 200 50 78 2 10,13, 10,22, 50,65,\n";

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
}
