use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, Read, Write},
    num::ParseIntError,
    path::Path,
    rc::Rc,
};

use csv::Error as CsvError;
use gen_core::{HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand};
use gen_models::{
    block_group::BlockGroup,
    block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
    db::DbContext,
    edge::{Edge, EdgeData},
    errors::{NodeError, OperationError, SampleError, SequenceError},
    file_types::FileTypes,
    node::Node,
    operations::{OperationFile, OperationInfo},
    sample::Sample,
    sequence::Sequence,
    traits::*,
};
use regex::Regex;
use rusqlite::{params, types::Value};
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
    #[error("Node creation error: {0}")]
    NodeError(#[from] NodeError),
    #[error("Sequence save error: {0}")]
    SequenceError(#[from] SequenceError),
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
) -> Result<(), GafUpdateError>
where
    P: AsRef<Path> + Clone,
{
    // Given a gaf, this will incorporate the alignment into the specified graph, creating new nodes.

    let conn = context.graph().conn();
    let mut session = gen_models::session_operations::start_operation(conn);

    let parent_samples = parent_sample
        .map(|parent_sample| vec![parent_sample.to_string()])
        .unwrap_or_default();
    let sample_name =
        Sample::get_or_create_child(conn, collection_name, sample_name, parent_samples)?.name;

    let mut node_lengths: HashMap<String, (HashId, i64)> = HashMap::new();

    let mut get_node_info = |node_id: &str| -> (HashId, i64) {
        *node_lengths.entry(node_id.to_string()).or_insert_with(|| {
            let node_info : Vec<&str> = node_id.rsplitn(2, '.').collect();
            let node_id = *node_info.last().unwrap();
            let id = HashId::try_from(node_id).unwrap();
            let mut stmt = conn.prepare_cached("select s.length from nodes n left join sequences s on (s.hash = n.sequence_hash) where n.id = ?1;").unwrap();
            let res = stmt.query_row([id], |row| row.get(0)).unwrap();
            (id, res)
        })
    };

    // our GFA export encodes segments like node_id.sequence_start
    let re = Regex::new(
        r"(?x)
        ^
        (?P<query_name>[^\t]+)
        \t
        (?P<query_length>\d+)
        \t
        (?P<query_start>\d+)
        \t
        (?P<query_end>\d+)
        \t
        (?P<strand>[+-])
        \t
        (?P<path>[^\t]+)
        \t
        (?P<path_length>\d+)
        \t
        (?P<path_start>\d+)
        \t
        (?P<path_end>\d+)
        \t
        (?P<residue_match>\d+)
        \t
        (?P<align_block_len>\d+)
        \t
        (?P<mapq>\d+)
        .+
        cg:Z:(?P<cigar>[^\t]+)
        ",
    )
    .unwrap();

    let query_re = Regex::new(
        r"(?x)
        ^(?P<query_id>.+)_(left|right)$
        ",
    )
    .unwrap();

    let orient_id_re = Regex::new(r"(?x)(?P<orient>[><])(?P<node>[^><]+(:\d+-\d+)?)").unwrap();

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
        let entry = re
            .captures(&line)
            .ok_or_else(|| GafUpdateError::InvalidGafLine(line.clone()))?;
        let aln_path = &entry["path"];
        let mut node_start: i64 = entry["path_start"]
            .parse::<i64>()
            .map_err(GafUpdateError::ParseInt)?;
        let mut segments = vec![];
        if [">", "<"].iter().any(|s| aln_path.starts_with(*s)) {
            // orient id
            for sub_match in orient_id_re.captures_iter(aln_path) {
                let orientation = if &sub_match["orient"] == ">" {
                    Strand::Forward
                } else {
                    Strand::Reverse
                };
                let node = sub_match["node"].to_string();
                segments.push((orientation, node));
            }
        } else {
            // we're a stable id
            segments.push((Strand::Forward, aln_path.to_string()));
        }
        let query = entry["query_name"].to_string();
        if let Some(id_re) = query_re.captures(&query) {
            let query_id = id_re["query_id"].to_string();
            if change_spec.contains_key(&query_id) {
                let mut strand: Option<Strand> = None;
                let mut node_id: Option<_> = None;
                let query_key;
                if query.ends_with("left") {
                    query_key = "left";
                    let mut matches = entry["residue_match"]
                        .parse::<i64>()
                        .map_err(GafUpdateError::ParseInt)?;
                    for (segment_strand, segment_id) in segments.iter() {
                        let (segment_node_id, node_length) = get_node_info(segment_id);
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
                    let (segment_node_id, _node_length) = get_node_info(segment_id);
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
            let mut bg_nodes = vec![];

            if change.left.is_empty() && change.right.is_empty() {
                panic!("Invalid change specification");
            } else if change.left.is_empty() {
                // we are inserting at the far left side, so our right node mapping is actually
                // where we want to be
                let (node, strand, pos) = path_changes["right"];
                bg_nodes.push(Value::from(node));
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
                bg_nodes.push(Value::from(node));
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
                bg_nodes.push(Value::from(node));
                new_edges.push(EdgeData {
                    source_node_id: node,
                    source_coordinate: pos,
                    source_strand: strand,
                    target_node_id: seq_node,
                    target_coordinate: 0,
                    target_strand: Strand::Forward,
                });

                let (node, strand, pos) = path_changes["right"];
                bg_nodes.push(Value::from(node));
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
            let bgs = BlockGroup::query(
                conn,
                "select distinct bg.* from block_groups bg left join block_group_edges bge on (bg.id = bge.block_group_id) left join edges e on (e.id = bge.edge_id and (e.source_node_id in rarray(?3) or e.target_node_id in rarray(?3))) where collection_name = ?1 and sample_name = ?2",
                params!(collection_name.to_string(), sample_name, Rc::new(bg_nodes)),
            );
            for bg in bgs.iter() {
                let new_block_group_edges = edge_ids
                    .iter()
                    .map(|edge_id| BlockGroupEdgeData {
                        block_group_id: bg.id,
                        edge_id: *edge_id,
                        chromosome_index: 0,
                        phased: 0,
                    })
                    .collect::<Vec<_>>();
                BlockGroupEdge::bulk_create(conn, &new_block_group_edges);
            }
        }
    }

    gen_models::session_operations::end_operation(
        context,
        &mut session,
        &OperationInfo {
            files: vec![OperationFile {
                file_path: gaf_path.as_ref().to_str().unwrap().to_string(),
                file_type: FileTypes::GAF,
            }],
            description: "insert_via_gaf".to_string(),
        },
        &format!("{change_count} updates."),
        None,
    )
    .map_err(GafUpdateError::Operation)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use gen_graph::{GraphEdge, GraphNode};
    use gen_models::traits::Query;
    use petgraph::Direction;

    use super::*;
    use crate::{imports::gfa::import_gfa, test_helpers::setup_gen, track_database};

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
                format!(
                    "\
            >extreme_left_left\n\
            aaa\n\
            >extreme_right_right\n\
            ggg\n"
                )
            );
        }
    }

    #[test]
    #[ignore]
    fn test_insertion_from_gaf() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();

        let collection = "test".to_string();

        let gfa_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/chr22_het.gfa");
        let csv_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/chr22_insert.csv");

        let _ = import_gfa(&context, &gfa_path, &collection, "");
        let gaf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/chr22_het.gaf");
        update_with_gaf(&context, gaf_path, csv_path, "test", "child", None).unwrap();
        let graph = Sample::get_graph(conn, "test", "child");

        let query = Node::query(
            conn,
            "select n.* from nodes n left join sequences s on (n.sequence_hash = s.hash) where s.sequence = ?1",
            params!("AATCGAATCG".to_string()),
        );
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
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();

        let collection = "test".to_string();

        let gfa_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/chr22_het.gfa");
        let csv_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/chr22_insert.csv");

        let _ = import_gfa(&context, &gfa_path, &collection, "");
        let gaf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/chr22_het.gaf");
        update_with_gaf(&context, gaf_path, csv_path, "test", "child", None).unwrap();
        let graph = Sample::get_graph(conn, "test", "child");

        // we should end up with a new edge putting our insert to the beginning of the graph, which is node 3.
        let query = Node::query(
            conn,
            "select n.* from nodes n left join sequences s on (n.sequence_hash = s.hash) where s.sequence = ?1",
            params!("aaa".to_string()),
        );
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

        let query = Node::query(
            conn,
            "select n.* from nodes n left join sequences s on (n.sequence_hash = s.hash) where s.sequence = ?1",
            params!("ttt".to_string()),
        );
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
}
