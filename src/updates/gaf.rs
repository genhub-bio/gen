use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, Read, Write},
    num::ParseIntError,
    path::Path,
};

use csv::Error as CsvError;
use gen_core::{HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand};
use gen_graph::{GraphNode, GraphNodeSlice};
use gen_models::{
    ModelSelectError,
    block_group::BlockGroup,
    block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
    db::DbContext,
    edge::{Edge, EdgeData},
    errors::{NodeError, OperationError, SampleError, SequenceError},
    file_types::FileTypes,
    node::Node,
    operations::{OperationFile, OperationInfo, OperationSummary},
    sample::Sample,
    sequence::Sequence,
};
use regex::Regex;
use thiserror::Error;

use crate::read_lines;

#[derive(Debug, serde::Deserialize)]
struct CSVRow {
    id: Option<String>,
    left: String,
    sequence: String,
    right: String,
}

const GEN_PREFIX: &str = "_gen_";

#[derive(Debug, Error)]
pub enum GafUpdateError {
    #[error("Input csv missing headers. Headers should be id,left,sequence,right.")]
    MissingHeaders,
    #[error("CSV parse error: {0}")]
    Csv(#[from] CsvError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Operation error: {0}")]
    Operation(#[from] OperationError),
    #[error("Sample error: {0}")]
    Sample(#[from] SampleError),
    #[error("Failed to parse expected number in GAF file: {0}")]
    ParseInt(#[from] ParseIntError),
    #[error("Line did not match expected GAF format: {0}")]
    InvalidGafLine(String),
    #[error("GAF path segment is not a Gen segment id (node.start.end): {0}")]
    InvalidGafSegment(String),
    #[error("Node creation error: {0}")]
    NodeError(#[from] NodeError),
    #[error("Sequence save error: {0}")]
    SequenceError(#[from] SequenceError),
    #[error("Selector error: {0}")]
    ModelSelect(#[from] ModelSelectError),
}

/// One GAF row: the twelve mandatory columns plus the optional `tag:type:value` fields.
#[derive(Clone, Debug, PartialEq)]
pub struct GafAlignment {
    pub query_name: String,
    pub query_length: i64,
    pub query_start: i64,
    pub query_end: i64,
    /// `Strand::Unknown` for the `*` of an unmapped row.
    pub strand: Strand,
    /// The path column as written, such as `>a.0.5<b.3.9`, or `*` for an unmapped row.
    pub path: String,
    pub path_length: i64,
    pub path_start: i64,
    pub path_end: i64,
    pub residue_match: i64,
    pub align_block_len: i64,
    pub mapq: i64,
    tags: HashMap<String, String>,
}

impl GafAlignment {
    /// The value of a tag such as `dv` or `cg`, without its type letter.
    pub fn tag(&self, name: &str) -> Option<&str> {
        self.tags.get(name).map(String::as_str)
    }

    pub fn cigar(&self) -> Option<&str> {
        self.tag("cg")
    }

    /// Minigraph marks the chosen alignment `tp:A:P`; rows without the tag count as primary.
    pub fn is_primary(&self) -> bool {
        self.tag("tp").is_none_or(|kind| kind == "P")
    }
}

/// Parse one GAF row. The cigar is optional so unmapped and cigar-less rows parse; callers that
/// need a cigar check [`GafAlignment::cigar`].
pub fn parse_gaf_line(line: &str) -> Result<GafAlignment, GafUpdateError> {
    let invalid = || GafUpdateError::InvalidGafLine(line.to_string());
    let fields: Vec<&str> = line.split('\t').collect();
    if fields.len() < 12 || fields[..12].iter().any(|field| field.is_empty()) {
        return Err(invalid());
    }
    let number = |index: usize| fields[index].parse::<i64>().map_err(|_| invalid());
    let strand = match fields[4] {
        "+" => Strand::Forward,
        "-" => Strand::Reverse,
        "*" => Strand::Unknown,
        _ => return Err(invalid()),
    };
    let tags = fields[12..]
        .iter()
        .filter_map(|field| {
            let mut parts = field.splitn(3, ':');
            let name = parts.next()?;
            parts.next()?;
            Some((name.to_string(), parts.next()?.to_string()))
        })
        .collect();
    Ok(GafAlignment {
        query_name: fields[0].to_string(),
        query_length: number(1)?,
        query_start: number(2)?,
        query_end: number(3)?,
        strand,
        path: fields[5].to_string(),
        path_length: number(6)?,
        path_start: number(7)?,
        path_end: number(8)?,
        residue_match: number(9)?,
        align_block_len: number(10)?,
        mapq: number(11)?,
        tags,
    })
}

/// Split a GAF path column into ordered `(strand, segment name)` visits: `>` reads a segment
/// forward and `<` reverse. A path without orientation marks is a single stable id, read forward.
pub fn parse_gaf_path(path: &str) -> Vec<(Strand, String)> {
    let orient_id_re = Regex::new(r"(?x)(?P<orient>[><])(?P<node>[^><]+(:\d+-\d+)?)").unwrap();
    if path.starts_with(['>', '<']) {
        orient_id_re
            .captures_iter(path)
            .map(|sub_match| {
                let orientation = if &sub_match["orient"] == ">" {
                    Strand::Forward
                } else {
                    Strand::Reverse
                };
                (orientation, sub_match["node"].to_string())
            })
            .collect()
    } else {
        vec![(Strand::Forward, path.to_string())]
    }
}

/// Parse the GFA export's `<node hash>.<start>.<end>` segment id into the graph node it names,
/// splitting from the right on the last two dots.
pub fn parse_gaf_segment(segment: &str) -> Option<GraphNode> {
    let mut parts = segment.rsplitn(3, '.');
    let sequence_end = parts.next()?.parse::<i64>().ok()?;
    let sequence_start = parts.next()?.parse::<i64>().ok()?;
    let node_id = HashId::try_from(parts.next()?).ok()?;
    (sequence_start <= sequence_end).then_some(GraphNode {
        node_id,
        sequence_start,
        sequence_end,
    })
}

/// The walk an alignment follows, clipped to its aligned interval `[path_start, path_end)`.
///
/// The path interval counts bases in walk order, so a reverse visit covers the mirrored range of
/// its segment. Each slice carries the whole segment as its block and the clipped range as local
/// offsets, matching the loci a sequence search returns; visits the interval does not reach are
/// dropped. An unmapped row (`*` path) has no walk.
pub fn clip_gaf_walk(alignment: &GafAlignment) -> Result<Vec<GraphNodeSlice>, GafUpdateError> {
    if alignment.path == "*" {
        return Ok(vec![]);
    }
    let mut cursor = 0;
    let mut slices = vec![];
    for (strand, name) in parse_gaf_path(&alignment.path) {
        let segment = parse_gaf_segment(&name)
            .ok_or_else(|| GafUpdateError::InvalidGafSegment(name.clone()))?;
        let length = segment.sequence_end - segment.sequence_start;
        let low = alignment.path_start.max(cursor);
        let high = alignment.path_end.min(cursor + length);
        if low < high {
            let (start, end) = if strand == Strand::Reverse {
                (length - (high - cursor), length - (low - cursor))
            } else {
                (low - cursor, high - cursor)
            };
            slices.push(GraphNodeSlice {
                block: segment,
                start: start as usize,
                end: end as usize,
                strand,
            });
        }
        cursor += length;
    }
    Ok(slices)
}

pub fn transform_csv_to_fasta<R, W>(reader: R, writer: &mut W) -> Result<(), GafUpdateError>
where
    R: Read,
    W: Write,
{
    let csv_bufreader = BufReader::new(reader);

    let mut csv_reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(csv_bufreader);
    let headers = match csv_reader.headers() {
        Ok(headers) => headers.clone(),
        Err(_) => return Err(GafUpdateError::MissingHeaders),
    };
    for (index, result) in csv_reader.records().enumerate() {
        let record = result?;
        let row: CSVRow = record.deserialize(Some(&headers))?;
        let id = row
            .id
            .clone()
            .unwrap_or_else(|| format!("{GEN_PREFIX}{index}"));
        if !row.left.is_empty() {
            writeln!(writer, ">{id}_left\n{left}", left = row.left)?;
        }
        if !row.right.is_empty() {
            writeln!(writer, ">{id}_right\n{right}", right = row.right)?;
        }
    }

    Ok(())
}

pub fn update_with_gaf<P>(
    context: &DbContext,
    gaf_path: P,
    csv_path: P,
    collection_name: &str,
    sample_name: &str,
    parent_sample: Option<&str>,
) -> Result<OperationSummary, GafUpdateError>
where
    P: AsRef<Path> + Clone,
{
    // Given a gaf, this will incorporate the alignment into the specified graph, creating new nodes.

    let conn = context.graph().conn();
    let parent_samples = parent_sample
        .map(|parent_sample| vec![parent_sample.to_string()])
        .unwrap_or_default();
    let sample_name =
        Sample::get_or_create_child(conn, collection_name, sample_name, parent_samples)?.name;

    let mut node_lengths: HashMap<String, (HashId, i64)> = HashMap::new();

    let mut get_node_info = |node_id: &str| -> Option<(HashId, i64)> {
        if let Some(node_info) = node_lengths.get(node_id) {
            return Some(*node_info);
        }

        // Current exports name segments node.start.end; older ones node.start.
        let node_id = match parse_gaf_segment(node_id) {
            Some(graph_node) => graph_node.node_id.to_string(),
            None => node_id.rsplitn(2, '.').last().unwrap().to_string(),
        };
        let id = HashId::try_from(node_id.as_str()).ok()?;
        let length = Node::query_nodes_length(conn, &[id])
            .ok()?
            .get(&id)
            .copied()?;
        node_lengths.insert(node_id, (id, length));
        Some((id, length))
    };

    let query_re = Regex::new(
        r"(?x)
        ^(?P<query_id>.+)_(left|right)$
        ",
    )
    .unwrap();

    let csv_file = File::open(csv_path).map_err(GafUpdateError::Io)?;
    let csv_bufreader = BufReader::new(csv_file);

    let mut csv_reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(csv_bufreader);
    let headers = match csv_reader.headers() {
        Ok(headers) => headers.clone(),
        Err(_) => return Err(GafUpdateError::MissingHeaders),
    };
    let mut change_spec = HashMap::new();
    for (index, result) in csv_reader.records().enumerate() {
        let record = result.map_err(GafUpdateError::Csv)?;
        let row: CSVRow = record
            .deserialize(Some(&headers))
            .map_err(GafUpdateError::Csv)?;
        change_spec.insert(
            row.id
                .clone()
                .unwrap_or_else(|| format!("{GEN_PREFIX}{index}")),
            row,
        );
    }

    let mut gaf_changes: HashMap<String, HashMap<String, (HashId, Strand, i64)>> = HashMap::new();

    let lines = read_lines(gaf_path.as_ref()).map_err(GafUpdateError::Io)?;
    for line in lines {
        let line = line.map_err(GafUpdateError::Io)?;
        let entry = parse_gaf_line(&line)?;
        // This update locates its insertion by walking alignments, which only rows with a
        // cigar provide.
        if entry.cigar().is_none() {
            return Err(GafUpdateError::InvalidGafLine(line.clone()));
        }
        let mut node_start = entry.path_start;
        let segments = parse_gaf_path(&entry.path);
        let query = entry.query_name.clone();
        if let Some(id_re) = query_re.captures(&query) {
            let query_id = id_re["query_id"].to_string();
            if change_spec.contains_key(&query_id) {
                let mut strand: Option<Strand> = None;
                let mut node_id: Option<_> = None;
                let query_key;
                if query.ends_with("left") {
                    query_key = "left";
                    let mut matches = entry.residue_match;
                    for (segment_strand, segment_id) in segments.iter() {
                        let Some((segment_node_id, node_length)) = get_node_info(segment_id) else {
                            continue;
                        };
                        if node_length >= matches {
                            strand = Some(*segment_strand);
                            node_id = Some(segment_node_id);
                            node_start = matches;
                            break;
                        }
                        matches -= node_length;
                    }
                } else if query.ends_with("right") {
                    query_key = "right";
                    let (segment_strand, segment_id) = segments.first().unwrap();
                    let Some((segment_node_id, _node_length)) = get_node_info(segment_id) else {
                        continue;
                    };
                    strand = Some(*segment_strand);
                    node_id = Some(segment_node_id);
                } else {
                    continue;
                };

                if let Some(node_id) = node_id
                    && let Some(strand) = strand
                {
                    gaf_changes
                        .entry(query_id)
                        .and_modify(|change| {
                            change
                                .entry(query_key.to_string())
                                .or_insert((node_id, strand, node_start));
                        })
                        .or_insert_with(|| {
                            let mut change = HashMap::new();
                            change.insert(query_key.to_string(), (node_id, strand, node_start));
                            change
                        });
                }
            }
        }
    }

    let mut change_count = 0;
    for (path_id, path_changes) in gaf_changes.iter() {
        if let Some(change) = change_spec.get(path_id) {
            change_count += 1;
            let sequence = Sequence::new()
                .sequence(&change.sequence)
                .sequence_type("DNA")
                .save(conn)?;
            let seq_node = Node::create(
                conn,
                &sequence.hash,
                &HashId::convert_str(&format!(
                    "{left_node_info:?}->{hash}->{right_node_info:?}",
                    left_node_info = path_changes.get("left").unwrap_or(&(
                        HashId::convert_str(""),
                        Strand::Unknown,
                        -1
                    )),
                    hash = sequence.hash,
                    right_node_info = path_changes.get("right").unwrap_or(&(
                        HashId::convert_str(""),
                        Strand::Unknown,
                        -1
                    )),
                )),
            )?;

            let mut new_edges = vec![];

            if change.left.is_empty() && change.right.is_empty() {
                panic!("Invalid change specification");
            } else if change.left.is_empty() {
                // we are inserting at the far left side, so our right node mapping is actually
                // where we want to be
                let (node, strand, pos) = path_changes["right"];
                new_edges.push(EdgeData {
                    source_node_id: PATH_START_NODE_ID,
                    source_coordinate: 0,
                    source_strand: Strand::Forward,
                    target_node_id: seq_node,
                    target_coordinate: 0,
                    target_strand: Strand::Forward,
                });
                new_edges.push(EdgeData {
                    source_node_id: seq_node,
                    source_coordinate: sequence.length,
                    source_strand: Strand::Forward,
                    target_node_id: node,
                    target_coordinate: pos,
                    target_strand: strand,
                });
            } else if change.right.is_empty() {
                // we are inserting at the far right side
                let (node, strand, pos) = path_changes["left"];
                new_edges.push(EdgeData {
                    source_node_id: node,
                    source_coordinate: pos,
                    source_strand: strand,
                    target_node_id: seq_node,
                    target_coordinate: 0,
                    target_strand: Strand::Forward,
                });
                new_edges.push(EdgeData {
                    source_node_id: seq_node,
                    source_coordinate: sequence.length,
                    source_strand: Strand::Forward,
                    target_node_id: PATH_END_NODE_ID,
                    target_coordinate: 0,
                    target_strand: Strand::Forward,
                });
            } else {
                // normal insert
                let (node, strand, pos) = path_changes["left"];
                new_edges.push(EdgeData {
                    source_node_id: node,
                    source_coordinate: pos,
                    source_strand: strand,
                    target_node_id: seq_node,
                    target_coordinate: 0,
                    target_strand: Strand::Forward,
                });

                let (node, strand, pos) = path_changes["right"];
                new_edges.push(EdgeData {
                    source_node_id: seq_node,
                    source_coordinate: sequence.length,
                    source_strand: Strand::Forward,
                    target_node_id: node,
                    target_coordinate: pos,
                    target_strand: strand,
                });
            }

            let edge_ids = Edge::bulk_create(conn, &new_edges);
            // Inserted edges belong to every block group in the target sample. Selecting that row
            // set directly also avoids joins whose nullable side never constrained the base rows.
            let block_groups = BlockGroup::select(conn)
                .collection_name(collection_name)
                .sample_name(&sample_name)
                .load()?;
            for block_group in &block_groups {
                let new_block_group_edges = edge_ids
                    .iter()
                    .map(|edge_id| BlockGroupEdgeData {
                        block_group_id: block_group.id,
                        edge_id: *edge_id,
                        chromosome_index: 0,
                        phased: 0,
                    })
                    .collect::<Vec<_>>();
                BlockGroupEdge::bulk_create(conn, &new_block_group_edges);
            }
        }
    }

    Ok(OperationSummary::new(
        OperationInfo {
            files: vec![
                OperationFile::new(gaf_path.as_ref().to_str().unwrap().to_string())
                    .set_file_type(FileTypes::GAF),
            ],
            description: "insert_via_gaf".to_string(),
        },
        format!("{change_count} updates."),
    ))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use gen_graph::{GraphEdge, GraphNode};
    use gen_models::{db::GraphConnection, node::NodeSelect, sequence::SequenceSelect};
    use petgraph::Direction;

    use super::*;
    use crate::{imports::gfa::import_gfa, test_helpers::setup_gen};

    fn nodes_for_sequence(conn: &GraphConnection, sequence: &str) -> Vec<Node> {
        Node::select(conn)
            .join_filtered_on(
                NodeSelect::SequenceHash,
                SequenceSelect::Hash,
                Sequence::select(conn).sequence(sequence),
            )
            .load()
            .expect("should query nodes for the sequence")
    }

    mod test_transform {
        use super::*;
        #[test]
        fn test_transforms_to_fasta() {
            let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/chr22_insert.csv");
            let mut csv_file = File::open(path).unwrap();
            let mut buffer = Vec::new();
            transform_csv_to_fasta(&mut csv_file, &mut buffer).unwrap();
            let results = String::from_utf8(buffer).unwrap();
            assert_eq!(
                results,
                "\
            >test_left\n\
            atgggagtataattttagatagtgaagatttctgtattcaaatgccacat\n\
            >test_right\n\
            acacagaaaaaggcaggcagagaaaataacaaggataaagacactgaagt\n\
            >2node_span_left\n\
            GACCTTATCTTTTAAAAATATAaaaaaaTTTTTACATTAATTACTTCCAAAATAGAGATCAGTTGCATACAAATGGCAGGTCACC\n\
            >2node_span_right\n\
            atacctttctgctcttgtcagacaattaaggggtctttgaatacttcagccctaataatttgcttcctaacatacatattgcagtgctt\n\
            >left_extreme_right\n\
            GAATTCTTGTGTTTATATAATAAGATGTCCTATAATTTCTGTTTGGAATA\n\
            >right_extreme_left\n\
            GGAGATTACAAATTTGCAAACCTCAGCTGCTCTCATTTTATGCTTTCACC\n"
            )
        }

        #[test]
        fn test_prefixes_entries_without_id() {
            let mut input = "id,left,sequence,right\n,aaa,ttt,ccc".as_bytes();
            let mut buffer = Vec::new();
            transform_csv_to_fasta(&mut input, &mut buffer).unwrap();
            let results = String::from_utf8(buffer).unwrap();
            assert_eq!(
                results,
                format!(
                    "\
            >{GEN_PREFIX}0_left\n\
            aaa\n\
            >{GEN_PREFIX}0_right\n\
            ccc\n"
                )
            );
        }

        #[test]
        fn test_prefixes_entries_with_extremes() {
            let mut input =
                "id,left,sequence,right\nextreme_left,aaa,ttt,\nextreme_right,,ccc,ggg".as_bytes();
            let mut buffer = Vec::new();
            transform_csv_to_fasta(&mut input, &mut buffer).unwrap();
            let results = String::from_utf8(buffer).unwrap();
            assert_eq!(
                results,
                "\
            >extreme_left_left\n\
            aaa\n\
            >extreme_right_right\n\
            ggg\n"
                    .to_string()
            );
        }
    }

    #[test]
    #[ignore]
    fn test_insertion_from_gaf() {
        let context = setup_gen();
        let conn = context.graph().conn();

        let collection = "test".to_string();

        let gfa_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/chr22_het.gfa");
        let csv_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/chr22_insert.csv");

        let _ = import_gfa(&context, &gfa_path, &collection, "");
        let gaf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/chr22_het.gaf");
        update_with_gaf(&context, gaf_path, csv_path, "test", "child", None).unwrap();
        let graph = Sample::get_graph(
            conn,
            crate::test_helpers::test_workspace(),
            "test",
            "child",
            None,
        )
        .unwrap();

        let query = nodes_for_sequence(conn, "AATCGAATCG");
        let insert_node_id = query.first().unwrap().id;
        let insert_node = graph
            .nodes()
            .filter(|node| node.node_id == insert_node_id)
            .collect::<Vec<GraphNode>>();
        let insert_node = insert_node.first().unwrap();
        let left_node_id = graph
            .nodes()
            .filter(|node| node.node_id == HashId::convert_str("138"))
            .collect::<Vec<GraphNode>>();
        let left_node = left_node_id.first().unwrap();
        let right_node_id = graph
            .nodes()
            .filter(|node| node.node_id == HashId::convert_str("140"))
            .collect::<Vec<GraphNode>>();
        let right_node = right_node_id.first().unwrap();

        // Here we should be making an edge from our left node -> insert, and an edge from insert -> right node. .edges gives us outgoing edges
        // only so that is why we use the insert_node for the right_edge lookup.

        let left_edges: Vec<(GraphNode, GraphNode, &Vec<GraphEdge>)> =
            graph.edges(*left_node).collect();
        let right_edges: Vec<(GraphNode, GraphNode, &Vec<GraphEdge>)> =
            graph.edges(*insert_node).collect();
        assert!(
            left_edges
                .iter()
                .filter(|(source, dest, _)| source.node_id == left_node.node_id
                    && dest.node_id == insert_node.node_id)
                .collect::<Vec<_>>()
                .len()
                == 1
        );
        assert!(
            right_edges
                .iter()
                .filter(|(source, dest, _)| source.node_id == insert_node.node_id
                    && dest.node_id == right_node.node_id)
                .collect::<Vec<_>>()
                .len()
                == 1
        );
    }

    #[test]
    #[ignore]
    fn test_insertion_from_gaf_extremes() {
        let context = setup_gen();
        let conn = context.graph().conn();

        let collection = "test".to_string();

        let gfa_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/chr22_het.gfa");
        let csv_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/chr22_insert.csv");

        let _ = import_gfa(&context, &gfa_path, &collection, "");
        let gaf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/chr22_het.gaf");
        update_with_gaf(&context, gaf_path, csv_path, "test", "child", None).unwrap();
        let graph = Sample::get_graph(
            conn,
            crate::test_helpers::test_workspace(),
            "test",
            "child",
            None,
        )
        .unwrap();

        // we should end up with a new edge putting our insert to the beginning of the graph, which is node 3.
        let query = nodes_for_sequence(conn, "aaa");
        let insert_node_id = query.first().unwrap().id;
        let start_node_id = graph
            .nodes()
            .filter(|node| node.node_id == HashId::convert_str("3"))
            .collect::<Vec<GraphNode>>();

        let incoming_edges: Vec<(GraphNode, GraphNode, &Vec<GraphEdge>)> = graph
            .edges_directed(start_node_id[0], Direction::Incoming)
            .collect();

        // This checks that we have an incoming edge from our new insert to the old end of the graph
        assert_eq!(incoming_edges[1].0.node_id, insert_node_id);

        let query = nodes_for_sequence(conn, "ttt");
        let insert_node_id = query.first().unwrap().id;
        let end_node_id = graph
            .nodes()
            .filter(|node| node.node_id == HashId::convert_str("1001"))
            .collect::<Vec<GraphNode>>();

        let edges: Vec<(GraphNode, GraphNode, &Vec<GraphEdge>)> = graph
            .edges_directed(end_node_id[0], Direction::Outgoing)
            .collect();

        // This checks that we have an outgoing edge from the end of the old graph to our insert
        assert_eq!(edges[1].1.node_id, insert_node_id);
    }

    mod test_parser {
        use super::*;

        fn segment(label: &str, start: i64, end: i64) -> String {
            format!("{}.{start}.{end}", HashId::convert_str(label))
        }

        fn row(path: &str, path_start: i64, path_end: i64, tags: &str) -> String {
            format!("read\t30\t0\t30\t+\t{path}\t30\t{path_start}\t{path_end}\t30\t30\t60\t{tags}")
        }

        fn slice(
            label: &str,
            start: i64,
            end: i64,
            local: (usize, usize),
            strand: Strand,
        ) -> GraphNodeSlice {
            GraphNodeSlice {
                block: parse_gaf_segment(&segment(label, start, end)).unwrap(),
                start: local.0,
                end: local.1,
                strand,
            }
        }

        #[test]
        fn test_segment_splits_from_the_right_on_two_dots() {
            let node = parse_gaf_segment(&segment("a", 3, 9)).unwrap();
            assert_eq!(node.node_id, HashId::convert_str("a"));
            assert_eq!((node.sequence_start, node.sequence_end), (3, 9));
        }

        #[test]
        fn test_bad_segment_names_do_not_parse() {
            assert!(parse_gaf_segment("12.0").is_none());
            assert!(parse_gaf_segment("notahash.0.5").is_none());
            assert!(parse_gaf_segment(&format!("{}.x.5", HashId::convert_str("a"))).is_none());
            assert!(parse_gaf_segment(&segment("a", 9, 3)).is_none());
        }

        #[test]
        fn test_line_parses_without_cigar_and_reads_tags() {
            let alignment = parse_gaf_line(&row(">x", 0, 5, "tp:A:S\tdv:f:0.05")).unwrap();
            assert_eq!(alignment.cigar(), None);
            assert_eq!(alignment.tag("dv"), Some("0.05"));
            assert!(!alignment.is_primary());
            let alignment = parse_gaf_line(&row(">x", 0, 5, "cg:Z:5=\tds:Z::5")).unwrap();
            assert_eq!(alignment.cigar(), Some("5="));
            assert!(alignment.is_primary());
        }

        #[test]
        fn test_unmapped_row_parses_with_no_walk() {
            let alignment = parse_gaf_line("read\t30\t0\t30\t*\t*\t0\t0\t0\t0\t0\t0").unwrap();
            assert_eq!(alignment.strand, Strand::Unknown);
            assert!(clip_gaf_walk(&alignment).unwrap().is_empty());
        }

        #[test]
        fn test_short_or_nonnumeric_line_is_invalid() {
            assert!(matches!(
                parse_gaf_line("read\t30\t0"),
                Err(GafUpdateError::InvalidGafLine(_))
            ));
            assert!(matches!(
                parse_gaf_line("read\tx\t0\t30\t+\t>a\t30\t0\t30\t30\t30\t60"),
                Err(GafUpdateError::InvalidGafLine(_))
            ));
        }

        #[test]
        fn test_path_visits_keep_orientation_and_order() {
            assert_eq!(
                parse_gaf_path(">a<b>c"),
                vec![
                    (Strand::Forward, "a".to_string()),
                    (Strand::Reverse, "b".to_string()),
                    (Strand::Forward, "c".to_string()),
                ]
            );
            assert_eq!(
                parse_gaf_path("stable"),
                vec![(Strand::Forward, "stable".to_string())]
            );
        }

        #[test]
        fn test_forward_walk_clips_both_ends() {
            let path = format!(
                ">{}>{}>{}",
                segment("a", 0, 10),
                segment("b", 0, 10),
                segment("c", 4, 14)
            );
            let alignment = parse_gaf_line(&row(&path, 3, 25, "")).unwrap();
            assert_eq!(
                clip_gaf_walk(&alignment).unwrap(),
                vec![
                    slice("a", 0, 10, (3, 10), Strand::Forward),
                    slice("b", 0, 10, (0, 10), Strand::Forward),
                    slice("c", 4, 14, (0, 5), Strand::Forward),
                ]
            );
        }

        #[test]
        fn test_reverse_first_and_last_visits_mirror_the_clip() {
            let path = format!(
                "<{}>{}<{}",
                segment("a", 0, 10),
                segment("b", 0, 10),
                segment("c", 4, 14)
            );
            let alignment = parse_gaf_line(&row(&path, 3, 25, "")).unwrap();
            assert_eq!(
                clip_gaf_walk(&alignment).unwrap(),
                vec![
                    slice("a", 0, 10, (0, 7), Strand::Reverse),
                    slice("b", 0, 10, (0, 10), Strand::Forward),
                    slice("c", 4, 14, (5, 10), Strand::Reverse),
                ]
            );
        }

        #[test]
        fn test_visits_outside_the_interval_are_dropped() {
            let path = format!(
                ">{}>{}>{}",
                segment("a", 0, 10),
                segment("b", 0, 10),
                segment("c", 0, 10)
            );
            let alignment = parse_gaf_line(&row(&path, 10, 20, "")).unwrap();
            assert_eq!(
                clip_gaf_walk(&alignment).unwrap(),
                vec![slice("b", 0, 10, (0, 10), Strand::Forward)]
            );
        }

        #[test]
        fn test_walk_with_a_bad_segment_is_an_error() {
            let alignment = parse_gaf_line(&row(">oops.0.5", 0, 5, "")).unwrap();
            assert!(matches!(
                clip_gaf_walk(&alignment),
                Err(GafUpdateError::InvalidGafSegment(_))
            ));
        }
    }
}
