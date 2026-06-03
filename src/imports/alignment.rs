use std::{
    collections::HashSet,
    io::{BufRead, BufReader},
    path::PathBuf,
};

use flate2::read::MultiGzDecoder;
use gen_core::{HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand};
use gen_graph::GraphNode;
use gen_models::{
    assets::AssetUri,
    block_group::{BlockGroup, NewBlockGroup},
    block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
    collection::Collection,
    db::DbContext,
    edge::Edge,
    errors::{
        BlockGroupError, CollectionError, EdgeError, FileAdditionError, NodeError, OperationError,
        PathError, SampleError, SequenceError,
    },
    file_types::FileTypes,
    node::Node,
    operations::{Operation, OperationFile, OperationInfo},
    path::Path,
    sample::{NewSample, Sample},
    sample_lineage::SampleLineage,
    sequence::Sequence,
    session_operations::{end_operation, start_operation},
    traits::Query as _,
};
use gen_parsers::{ClustalwParser, ParseError, ParsedAlignment};
use noodles::bgzf;
use petgraph::visit::{EdgeRef as _, IntoEdgeReferences as _};
use thiserror::Error;

use crate::progress_bar::{add_saving_operation_bar, get_handler, get_progress_bar};

#[derive(Debug, Error)]
pub enum AlignmentImportError {
    #[error("Operation Error: {0}")]
    OperationError(#[from] OperationError),
    #[error("Asset Error: {0}")]
    FileAdditionError(#[from] FileAdditionError),
    #[error("IO Error: {0}")]
    IOError(#[from] std::io::Error),
    #[error("Collection creation error: {0}")]
    CollectionError(#[from] CollectionError),
    #[error("Sample creation error: {0}")]
    SampleError(#[from] SampleError),
    #[error("Sample lineage error: {0}")]
    SampleLineageError(#[from] rusqlite::Error),
    #[error("Block group write error: {0}")]
    BlockGroupError(#[from] BlockGroupError),
    #[error("Edge write error: {0}")]
    EdgeError(#[from] EdgeError),
    #[error("Node creation error: {0}")]
    NodeError(#[from] NodeError),
    #[error("Path creation error: {0}")]
    PathError(#[from] PathError),
    #[error("Sequence save error: {0}")]
    SequenceError(#[from] SequenceError),
    #[error("Alignment parse error: {0}")]
    ParseError(#[from] ParseError),
    #[error("Alignment file reader did not reach EOF before checksum was requested")]
    MissingChecksum,
}

pub fn import_alignment_aln(
    context: &DbContext,
    alignment_path: &String,
    collection_name: &str,
    sample_override: Option<&str>,
) -> Result<Operation, AlignmentImportError> {
    let conn = context.graph().conn();
    let progress_bar = get_handler();
    let mut session = start_operation(conn);
    let path = PathBuf::from(alignment_path);

    let asset_uri = <dyn AssetUri>::new(context.workspace(), alignment_path);
    let file = asset_uri.reader(context.workspace())?;
    let checksum_handle = file.checksum_handle();
    let reader_stream: Box<dyn BufRead> = match path.extension().and_then(|ext| ext.to_str()) {
        Some("gz") => Box::new(BufReader::new(MultiGzDecoder::new(file))),
        Some("bgz") => Box::new(bgzf::io::Reader::new(file)),
        _ => Box::new(BufReader::new(file)),
    };

    let collection = match Collection::create(conn, collection_name) {
        Ok(collection) => collection,
        Err(CollectionError::Duplicate(collection)) => collection,
        Err(e) => return Err(AlignmentImportError::CollectionError(e)),
    };

    let _ = progress_bar.println("Parsing alignment");
    let bar = progress_bar.add(get_progress_bar(None));
    bar.set_message("Alignments Processed.");
    let mut summary = String::new();
    for alignment in ClustalwParser::new(reader_stream) {
        let alignment = alignment?;
        persist_alignment(conn, &collection.name, &alignment, sample_override)?;
        summary.push_str(&format!(
            " {}: {} aligned sequences.\n",
            alignment.base_name,
            alignment.sequence_order.len()
        ));
        bar.inc(1);
    }
    bar.finish();

    let checksum_override = checksum_handle
        .checksum()
        .ok_or(AlignmentImportError::MissingChecksum)?;
    let bar = add_saving_operation_bar(&progress_bar);
    let op = end_operation(
        context,
        &mut session,
        &OperationInfo {
            files: vec![
                OperationFile::new(alignment_path.to_string())
                    .set_file_type(FileTypes::None)
                    .set_checksum_override(checksum_override),
            ],
            description: "alignment_import".to_string(),
        },
        &summary,
        None,
    )
    .map_err(AlignmentImportError::OperationError);
    bar.finish();
    op
}

fn persist_alignment(
    conn: &gen_models::db::GraphConnection,
    collection_name: &str,
    alignment: &ParsedAlignment,
    sample_override: Option<&str>,
) -> Result<(), AlignmentImportError> {
    let graph = build_import_graph(conn, collection_name, alignment)?;
    let sequence_names = parse_alignment_sequence_names(alignment, sample_override);
    persist_alignment_samples(conn, &sequence_names)?;

    let parent_sequence_name = sequence_names
        .first()
        .expect("should contain at least one alignment sequence");
    let parent_block_group = BlockGroup::create(
        conn,
        NewBlockGroup {
            collection_name,
            sample_name: &parent_sequence_name.sample_name,
            name: &parent_sequence_name.block_group_name,
            ..Default::default()
        },
    )?;
    persist_alignment_path(
        conn,
        &graph,
        &parent_block_group.id,
        parent_sequence_name,
        0,
    )?;

    for (index, sequence_name) in sequence_names.iter().enumerate().skip(1) {
        let block_group = BlockGroup::create(
            conn,
            NewBlockGroup {
                collection_name,
                sample_name: &sequence_name.sample_name,
                name: &sequence_name.block_group_name,
                parent_block_group_id: Some(&parent_block_group.id),
                ..Default::default()
            },
        )?;
        block_group.copy_contents_from(conn, &parent_block_group)?;
        delete_paths_for_block_group(conn, &block_group.id);
        persist_alignment_path(conn, &graph, &block_group.id, sequence_name, index as i64)?;
    }

    Ok(())
}

fn persist_alignment_path(
    conn: &gen_models::db::GraphConnection,
    graph: &gen_graph::GenGraph,
    block_group_id: &HashId,
    sequence_name: &AlignmentSequenceName,
    chromosome_index: i64,
) -> Result<(), AlignmentImportError> {
    let (edge_ids, block_group_edges) =
        persist_path_edges(conn, graph, block_group_id, chromosome_index)?;
    BlockGroupEdge::bulk_create(conn, &block_group_edges);
    Path::create(conn, &sequence_name.sample_name, block_group_id, &edge_ids)?;
    Ok(())
}

fn delete_paths_for_block_group(conn: &gen_models::db::GraphConnection, block_group_id: &HashId) {
    for path in Path::query(
        conn,
        "select * from paths where block_group_id = ?1 order by name;",
        rusqlite::params![block_group_id],
    ) {
        Path::delete(conn, &path.name, block_group_id);
    }
}

fn parse_alignment_sequence_names(
    alignment: &ParsedAlignment,
    sample_override: Option<&str>,
) -> Vec<AlignmentSequenceName> {
    alignment
        .sequence_order
        .iter()
        .map(|name| parse_alignment_sequence_name(name, sample_override))
        .collect()
}

fn parse_alignment_sequence_name(
    name: &str,
    sample_override: Option<&str>,
) -> AlignmentSequenceName {
    if let Some(sample_name) = sample_override {
        return AlignmentSequenceName {
            sample_name: sample_name.to_string(),
            block_group_name: name.to_string(),
        };
    }

    match name.split_once('.') {
        Some((sample_name, block_group_name))
            if !sample_name.is_empty() && !block_group_name.is_empty() =>
        {
            AlignmentSequenceName {
                sample_name: sample_name.to_string(),
                block_group_name: block_group_name.to_string(),
            }
        }
        _ => AlignmentSequenceName {
            sample_name: name.to_string(),
            block_group_name: name.to_string(),
        },
    }
}

fn persist_alignment_samples(
    conn: &gen_models::db::GraphConnection,
    sequence_names: &[AlignmentSequenceName],
) -> Result<(), AlignmentImportError> {
    let parent_sample = sequence_names
        .first()
        .expect("should contain at least one alignment sequence");

    for name in sequence_names {
        match Sample::get_or_create(
            conn,
            NewSample {
                name: &name.sample_name,
                ..Default::default()
            },
        ) {
            Ok(_) => {}
            Err(SampleError::Duplicate(_)) => {}
            Err(e) => return Err(AlignmentImportError::SampleError(e)),
        }
    }

    for child_sample in sequence_names.iter().skip(1) {
        if parent_sample.sample_name != child_sample.sample_name {
            SampleLineage::create(conn, &parent_sample.sample_name, &child_sample.sample_name)?;
        }
    }

    Ok(())
}

fn persist_path_edges(
    conn: &gen_models::db::GraphConnection,
    graph: &gen_graph::GenGraph,
    block_group_id: &HashId,
    chromosome_index: i64,
) -> Result<(Vec<HashId>, Vec<BlockGroupEdgeData>), AlignmentImportError> {
    let mut edge_ids = Vec::new();
    let mut block_group_edges = Vec::new();
    let mut current = terminal_node(PATH_START_NODE_ID);

    while current.node_id != PATH_END_NODE_ID {
        let Some((target, metadata)) = next_path_edge(graph, current, chromosome_index) else {
            break;
        };
        let stored_edge = Edge::create(
            conn,
            current.node_id,
            current.sequence_end,
            metadata.source_strand,
            target.node_id,
            target.sequence_start,
            metadata.target_strand,
        )?;
        edge_ids.push(stored_edge.id);
        block_group_edges.push(BlockGroupEdgeData {
            block_group_id: *block_group_id,
            edge_id: stored_edge.id,
            chromosome_index,
            phased: metadata.phased,
        });
        current = target;
    }

    Ok((edge_ids, block_group_edges))
}

fn next_path_edge(
    graph: &gen_graph::GenGraph,
    source: GraphNode,
    chromosome_index: i64,
) -> Option<(GraphNode, gen_graph::GraphEdge)> {
    graph.edge_references().find_map(|edge| {
        if edge.source() != source {
            return None;
        }
        edge.weight()
            .iter()
            .find(|metadata| metadata.chromosome_index == chromosome_index)
            .copied()
            .map(|metadata| (edge.target(), metadata))
    })
}

fn build_import_graph(
    conn: &gen_models::db::GraphConnection,
    collection_name: &str,
    alignment: &ParsedAlignment,
) -> Result<gen_graph::GenGraph, AlignmentImportError> {
    let base_aligned = alignment
        .aligned_sequences
        .get(&alignment.base_name)
        .expect("should contain base sequence");
    let base_sequence = ungapped(base_aligned);
    let base_sequence_record = Sequence::new()
        .sequence_type("DNA")
        .sequence(&base_sequence)
        .save(conn)?;
    let base_node_id = Node::create(
        conn,
        &base_sequence_record.hash,
        &HashId::convert_str(&format!(
            "alignment:{collection_name}:{}:{}",
            alignment.base_name, base_sequence_record.hash
        )),
    )?;

    let mut graph = gen_graph::GenGraph::new();
    let mut previous_by_sequence =
        vec![terminal_node(PATH_START_NODE_ID); alignment.sequence_order.len()];
    let end_node = terminal_node(PATH_END_NODE_ID);
    let mut base_offset = 0_i64;
    let runs = alignment_runs(alignment);
    let mut created_variant_nodes = HashSet::new();

    for run in runs {
        let base_text = ungapped(&base_aligned[run.start..run.end]);
        let base_node = (!base_text.is_empty()).then(|| {
            let node = GraphNode {
                node_id: base_node_id,
                sequence_start: base_offset,
                sequence_end: base_offset + base_text.len() as i64,
            };
            base_offset += base_text.len() as i64;
            node
        });

        for (sequence_index, name) in alignment.sequence_order.iter().enumerate() {
            let aligned = alignment
                .aligned_sequences
                .get(name)
                .expect("should contain aligned sequence");
            let text = ungapped(&aligned[run.start..run.end]);
            let next = if run.kind == AlignmentRunKind::Invariant || sequence_index == 0 {
                base_node
            } else if text.is_empty() {
                None
            } else if text == base_text {
                base_node
            } else {
                let node_id = HashId::convert_str(&format!(
                    "alignment:{collection_name}:{name}:{}:{}:{text}",
                    run.start, run.end
                ));
                if created_variant_nodes.insert(node_id) {
                    let sequence = Sequence::new()
                        .sequence_type("DNA")
                        .sequence(&text)
                        .save(conn)?;
                    Node::create(conn, &sequence.hash, &node_id)?;
                }
                Some(GraphNode {
                    node_id,
                    sequence_start: 0,
                    sequence_end: text.len() as i64,
                })
            };

            if let Some(next_node) = next {
                add_import_edge(
                    &mut graph,
                    previous_by_sequence[sequence_index],
                    next_node,
                    sequence_index as i64,
                );
                previous_by_sequence[sequence_index] = next_node;
            }
        }
    }

    for (sequence_index, previous) in previous_by_sequence.iter().enumerate() {
        add_import_edge(&mut graph, *previous, end_node, sequence_index as i64);
    }

    Ok(graph)
}

fn add_import_edge(
    graph: &mut gen_graph::GenGraph,
    source: GraphNode,
    target: GraphNode,
    sequence_index: i64,
) {
    let edge = gen_graph::GraphEdge {
        edge_id: HashId::convert_str(&format!(
            "alignment:edge:{sequence_index}:{}:{}:{}:{}",
            source.node_id, source.sequence_end, target.node_id, target.sequence_start
        )),
        source_strand: Strand::Forward,
        target_strand: Strand::Forward,
        chromosome_index: sequence_index,
        phased: 0,
        created_on: 0,
    };

    if let Some(edges) = graph.edge_weight_mut(source, target) {
        edges.push(edge);
    } else {
        graph.add_edge(source, target, vec![edge]);
    }
}

fn alignment_runs(alignment: &ParsedAlignment) -> Vec<AlignmentRun> {
    let base_aligned = alignment
        .aligned_sequences
        .get(&alignment.base_name)
        .expect("should contain base sequence");
    let mut runs = Vec::new();
    let mut start = 0;
    let mut current_kind = None;

    for column in 0..base_aligned.len() {
        let kind = alignment_column_kind(alignment, column);
        if current_kind.is_some_and(|current| current != kind) {
            runs.push(AlignmentRun {
                start,
                end: column,
                kind: current_kind.expect("should have run kind"),
            });
            start = column;
        }
        current_kind = Some(kind);
    }

    if let Some(kind) = current_kind {
        runs.push(AlignmentRun {
            start,
            end: base_aligned.len(),
            kind,
        });
    }

    runs
}

fn alignment_column_kind(alignment: &ParsedAlignment, column: usize) -> AlignmentRunKind {
    let base_aligned = alignment
        .aligned_sequences
        .get(&alignment.base_name)
        .expect("should contain base sequence");
    let base = base_aligned.as_bytes()[column];
    if base == b'-' {
        return AlignmentRunKind::Variant;
    }

    let invariant = alignment.sequence_order.iter().all(|name| {
        alignment
            .aligned_sequences
            .get(name)
            .expect("should contain aligned sequence")
            .as_bytes()[column]
            == base
    });

    if invariant {
        AlignmentRunKind::Invariant
    } else {
        AlignmentRunKind::Variant
    }
}

fn terminal_node(node_id: HashId) -> GraphNode {
    GraphNode {
        node_id,
        sequence_start: 0,
        sequence_end: 0,
    }
}

fn ungapped(sequence: &str) -> String {
    sequence
        .chars()
        .filter(|character| *character != '-')
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AlignmentRun {
    start: usize,
    end: usize,
    kind: AlignmentRunKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AlignmentRunKind {
    Invariant,
    Variant,
}

#[derive(Debug, Eq, PartialEq)]
struct AlignmentSequenceName {
    sample_name: String,
    block_group_name: String,
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use gen_models::{
        block_group::BlockGroup, path::Path, sample_lineage::SampleLineage, traits::Query as _,
    };

    use crate::{
        imports::alignment::import_alignment_aln, test_helpers::setup_gen, track_database,
    };

    #[test]
    fn imports_aln_alignment_as_samples_with_base_lineage() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();
        track_database(conn, op_conn).unwrap();

        let aln_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/alignments/simple.aln");

        import_alignment_aln(
            &context,
            &aln_path
                .to_str()
                .expect("should have fixture path")
                .to_string(),
            "test",
            None,
        )
        .unwrap();

        let base_block_group_id = BlockGroup::get_id("test", "SeqA", "chr1", None);
        let base_paths = Path::query(
            conn,
            "select * from paths where block_group_id = ?1 order by name;",
            rusqlite::params![base_block_group_id],
        );
        assert_eq!(
            base_paths
                .iter()
                .map(|path| path.name.as_str())
                .collect::<Vec<_>>(),
            vec!["SeqA"]
        );
        assert_eq!(base_paths[0].sequence(conn).unwrap(), "ACGT");

        let child_block_group_id =
            BlockGroup::get_id("test", "SeqB", "chr1", Some(&base_block_group_id));
        let child_block_group = BlockGroup::get_by_id(conn, &child_block_group_id).unwrap();
        assert_eq!(
            child_block_group.parent_block_group_id,
            Some(base_block_group_id)
        );
        let child_paths = Path::query(
            conn,
            "select * from paths where block_group_id = ?1 order by name;",
            rusqlite::params![child_block_group_id],
        );
        assert_eq!(
            child_paths
                .iter()
                .map(|path| path.name.as_str())
                .collect::<Vec<_>>(),
            vec!["SeqB"]
        );
        assert_eq!(child_paths[0].sequence(conn).unwrap(), "AGT");
        let second_child_block_group_id =
            BlockGroup::get_id("test", "SeqC", "chr1", Some(&base_block_group_id));
        let second_child_block_group =
            BlockGroup::get_by_id(conn, &second_child_block_group_id).unwrap();
        assert_eq!(
            second_child_block_group.parent_block_group_id,
            Some(base_block_group_id)
        );
        let second_child_paths = Path::query(
            conn,
            "select * from paths where block_group_id = ?1 order by name;",
            rusqlite::params![second_child_block_group_id],
        );
        assert_eq!(
            second_child_paths
                .iter()
                .map(|path| path.name.as_str())
                .collect::<Vec<_>>(),
            vec!["SeqC"]
        );
        assert_eq!(second_child_paths[0].sequence(conn).unwrap(), "ACG");
        assert_eq!(
            SampleLineage::get_children(conn, "SeqA"),
            vec!["SeqB", "SeqC"]
        );
        assert_eq!(SampleLineage::get_parents(conn, "SeqB"), vec!["SeqA"]);
        assert_eq!(SampleLineage::get_parents(conn, "SeqC"), vec!["SeqA"]);
    }

    #[test]
    fn imports_plain_alignment_names_as_sample_and_block_group_names() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();
        track_database(conn, op_conn).unwrap();

        let aln_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/alignments/plain_names.aln");

        import_alignment_aln(
            &context,
            &aln_path
                .to_str()
                .expect("should have fixture path")
                .to_string(),
            "test",
            None,
        )
        .unwrap();

        let base_block_group_id = BlockGroup::get_id("test", "Base", "Base", None);
        let base_paths = Path::query(
            conn,
            "select * from paths where block_group_id = ?1 order by name;",
            rusqlite::params![base_block_group_id],
        );
        assert_eq!(
            base_paths
                .iter()
                .map(|path| path.name.as_str())
                .collect::<Vec<_>>(),
            vec!["Base"]
        );

        let child_block_group_id =
            BlockGroup::get_id("test", "Child", "Child", Some(&base_block_group_id));
        let child_block_group = BlockGroup::get_by_id(conn, &child_block_group_id).unwrap();
        assert_eq!(
            child_block_group.parent_block_group_id,
            Some(base_block_group_id)
        );
        let child_paths = Path::query(
            conn,
            "select * from paths where block_group_id = ?1 order by name;",
            rusqlite::params![child_block_group_id],
        );
        assert_eq!(
            child_paths
                .iter()
                .map(|path| path.name.as_str())
                .collect::<Vec<_>>(),
            vec!["Child"]
        );
        assert_eq!(SampleLineage::get_children(conn, "Base"), vec!["Child"]);
    }

    #[test]
    fn imports_alignment_with_sample_override_and_raw_block_group_names() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();
        track_database(conn, op_conn).unwrap();

        let aln_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/alignments/plain_names.aln");

        import_alignment_aln(
            &context,
            &aln_path
                .to_str()
                .expect("should have fixture path")
                .to_string(),
            "test",
            Some("override"),
        )
        .unwrap();

        let base_block_group_id = BlockGroup::get_id("test", "override", "Base", None);
        let base_paths = Path::query(
            conn,
            "select * from paths where block_group_id = ?1 order by name;",
            rusqlite::params![base_block_group_id],
        );
        assert_eq!(
            base_paths
                .iter()
                .map(|path| path.name.as_str())
                .collect::<Vec<_>>(),
            vec!["override"]
        );

        let child_block_group_id =
            BlockGroup::get_id("test", "override", "Child", Some(&base_block_group_id));
        let child_block_group = BlockGroup::get_by_id(conn, &child_block_group_id).unwrap();
        assert_eq!(
            child_block_group.parent_block_group_id,
            Some(base_block_group_id)
        );
        let child_paths = Path::query(
            conn,
            "select * from paths where block_group_id = ?1 order by name;",
            rusqlite::params![child_block_group_id],
        );
        assert_eq!(
            child_paths
                .iter()
                .map(|path| path.name.as_str())
                .collect::<Vec<_>>(),
            vec!["override"]
        );
        assert!(SampleLineage::get_children(conn, "override").is_empty());
    }
}
