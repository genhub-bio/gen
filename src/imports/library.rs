use std::collections::HashMap;

use anyhow::Result;
use gen_core::{PATH_END_NODE_ID, PATH_START_NODE_ID, Strand};
use gen_models::{
    accession::{Accession, AccessionEdge, AccessionEdgeData, AccessionPath},
    annotations::{Annotation, AnnotationExtra, AnnotationGroupSample, PartExtra},
    block_group::{BlockGroup, NewBlockGroup},
    collection::Collection,
    db::DbContext,
    errors::{BlockGroupError, CollectionError, OperationError},
    file_types::FileTypes,
    operations::{Operation, OperationFile, OperationInfo},
    sample::Sample,
    session_operations,
};
use thiserror::Error;

use crate::graphs::combinatorial_library::{
    CombinatorialLibraryCreationError, CombinatorialLibraryParseError, SequencePart, create_library,
};

#[derive(Error, Debug)]
pub enum LibraryImportError {
    #[error("No changes were made to the library")]
    NoChanges,
    #[error("Failed to import library")]
    ImportFailed(String),
    #[error("Operation Error: {0}")]
    OperationError(#[from] OperationError),
    #[error("Failed to parse library files")]
    FileParse(CombinatorialLibraryParseError),
    #[error("Failed to create library")]
    LibraryCreation(CombinatorialLibraryCreationError),
    #[error("Block group creation error: {0}")]
    BlockGroupError(#[from] BlockGroupError),
}

impl From<CombinatorialLibraryParseError> for LibraryImportError {
    fn from(err: CombinatorialLibraryParseError) -> Self {
        LibraryImportError::FileParse(err)
    }
}

impl From<CombinatorialLibraryCreationError> for LibraryImportError {
    fn from(err: CombinatorialLibraryCreationError) -> Self {
        LibraryImportError::LibraryCreation(err)
    }
}

pub fn import_library(
    context: &DbContext,
    collection_name: &str,
    sample: &str,
    library_name: &str,
    parts_list: Vec<Vec<SequencePart>>,
    parts_file_path: Option<&str>,
    library_file_path: Option<&str>,
) -> Result<Operation, LibraryImportError> {
    let conn = context.graph().conn();
    let mut session = session_operations::start_operation(conn);
    match Collection::create(conn, collection_name) {
        Ok(_) => {}
        Err(CollectionError::Duplicate(_)) => {}
        Err(e) => {
            return Err(LibraryImportError::ImportFailed(format!(
                "Failed to get or create collection: {e}"
            )));
        }
    }

    match Sample::get_or_create(
        conn,
        gen_models::sample::NewSample {
            name: sample,
            ..Default::default()
        },
    ) {
        Ok(_) => {}
        Err(e) => {
            return Err(LibraryImportError::ImportFailed(format!(
                "Failed to get or create sample: {e}"
            )));
        }
    }
    let new_block_group = BlockGroup::create(
        conn,
        NewBlockGroup {
            collection_name,
            sample_name: sample,
            name: library_name,
            ..Default::default()
        },
    )?;

    let (_chunk, part_nodes) =
        create_library(conn, new_block_group.id, library_name, parts_list, true)?;

    AnnotationGroupSample::create(conn, library_name, sample)
        .map_err(|e| LibraryImportError::BlockGroupError(e.into()))?;
    create_part_annotations(conn, new_block_group.id, library_name, &part_nodes)
        .map_err(LibraryImportError::BlockGroupError)?;

    let mut files = vec![];
    if let Some(library_file_path) = library_file_path {
        files.push(OperationFile::new(library_file_path.to_string()).set_file_type(FileTypes::CSV));
    }
    if let Some(parts_file_path) = parts_file_path {
        files.push(OperationFile::new(parts_file_path.to_string()).set_file_type(FileTypes::Fasta));
    }

    let summary_str = format!("{library_name} created.\n");
    let op = session_operations::end_operation(
        context,
        &mut session,
        &OperationInfo {
            files,
            description: "library_csv_import".to_string(),
        },
        &summary_str,
        None,
    )?;

    Ok(op)
}

pub(crate) fn create_part_annotations(
    conn: &gen_models::db::GraphConnection,
    block_group_id: gen_core::HashId,
    sample_name: &str,
    part_nodes: &[(gen_core::HashId, SequencePart)],
) -> Result<(), BlockGroupError> {
    let mut name_counts: HashMap<String, usize> = HashMap::new();
    for (node_id, part) in part_nodes {
        let count = name_counts.entry(part.name.clone()).or_insert(0);
        *count += 1;
        let annotation_name = if *count == 1 {
            part.name.clone()
        } else {
            format!("{}_{count}", part.name)
        };
        let accession = Accession::get_or_create(conn, &annotation_name, &block_group_id, None)?;
        let ann_start = part.annotation_start.unwrap_or(0);
        let ann_end = part.annotation_end.unwrap_or(part.sequence_length);
        let edge_ids = AccessionEdge::bulk_create(
            conn,
            &[
                AccessionEdgeData {
                    source_node_id: PATH_START_NODE_ID,
                    source_coordinate: -1,
                    source_strand: Strand::Forward,
                    target_node_id: *node_id,
                    target_coordinate: ann_start,
                    target_strand: Strand::Forward,
                    chromosome_index: 0,
                },
                AccessionEdgeData {
                    source_node_id: *node_id,
                    source_coordinate: ann_end,
                    source_strand: Strand::Forward,
                    target_node_id: PATH_END_NODE_ID,
                    target_coordinate: -1,
                    target_strand: Strand::Forward,
                    chromosome_index: 0,
                },
            ],
        );
        AccessionPath::create(conn, &accession.id, &edge_ids)?;
        let fasta = part.fasta_extra.clone();
        let extra_part = part.metadata.as_deref().and_then(|s| {
            serde_json::from_str(s)
                .ok()
                .map(|v| PartExtra { metadata: Some(v) })
        });
        let extra = if fasta.is_some() || extra_part.is_some() {
            Some(AnnotationExtra {
                fasta,
                part: extra_part,
                ..Default::default()
            })
        } else {
            None
        };
        Annotation::get_or_create(
            conn,
            &annotation_name,
            sample_name,
            &accession.id,
            extra.as_ref(),
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, path::PathBuf};

    use gen_models::{
        accession::Accession, annotations::Annotation, block_group::BlockGroup, node::Node,
    };

    use super::*;
    use crate::{
        graphs::combinatorial_library::parse_library, test_helpers::setup_gen, track_database,
    };

    #[test]
    fn imports_a_library() -> Result<()> {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();
        let collection = "test";

        let binding = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/affix_parts.fa");
        let parts_path = binding.to_str().unwrap();

        let binding = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/affix_layout.csv");
        let library_path = binding.to_str().unwrap();

        let parts_list = parse_library(parts_path, library_path)?;

        let _ = import_library(
            &context,
            collection,
            Sample::DEFAULT_NAME,
            "library graph",
            parts_list,
            Some(parts_path),
            Some(library_path),
        );

        let block_groups = Sample::get_block_groups(conn, collection, Sample::DEFAULT_NAME);
        let block_group = &block_groups[0];

        let mut expected_sequences = HashSet::new();
        for part1 in &[
            "TCTAGAGAAAGAGGGGACAAACTAG",
            "TCTAGAGAAAGACAGGACCCACTAG",
            "TCTAGAGAAAGATCCGATGTACTAG",
            "TCTAGAGAAAGATTAGACAAACTAG",
            "TCTAGAGAAAGAAGGGACAGACTAG",
            "TCTAGAGAAAGACATGACGTACTAG",
            "TCTAGAGAAAGATAGGAGACACTAG",
            "TCTAGAGAAAGAAGAGACTCACTAG",
        ] {
            for part2 in &["ATGCGTAAAGGAGAAGAACTTTAA", "ATGAGTAAGGGTGAAGAGCTGTAA"] {
                expected_sequences.insert(format!("{part1}{part2}"));
            }
        }

        let actual_sequences = BlockGroup::get_all_sequences(conn, &block_group.id, false).unwrap();
        assert_eq!(actual_sequences, expected_sequences);

        let current_path = BlockGroup::get_current_path(conn, &block_group.id).unwrap();
        assert_eq!(
            current_path.sequence(conn).unwrap(),
            "TCTAGAGAAAGAGGGGACAAACTAGATGCGTAAAGGAGAAGAACTTTAA"
        );

        Ok(())
    }

    #[test]
    fn one_column_of_parts() -> Result<()> {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();
        let collection = "test";

        let binding = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/parts.fa");
        let parts_path = binding.to_str().unwrap();

        let binding =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/single_column_design.csv");
        let library_path = binding.to_str().unwrap();

        let parts_list = parse_library(parts_path, library_path)?;

        let _ = import_library(
            &context,
            collection,
            Sample::DEFAULT_NAME,
            "m123",
            parts_list,
            Some(parts_path),
            Some(library_path),
        );

        let block_groups = Sample::get_block_groups(conn, collection, Sample::DEFAULT_NAME);
        let block_group = &block_groups[0];

        let all_sequences = BlockGroup::get_all_sequences(conn, &block_group.id, false).unwrap();
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec![
                "AAAA".to_string(),
                "TAAT".to_string(),
                "CAAC".to_string(),
            ])
        );

        Ok(())
    }

    #[test]
    fn two_columns_of_same_parts() -> Result<()> {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();
        let collection = "test";

        let binding = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/parts.fa");
        let parts_path = binding.to_str().unwrap();

        let binding =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/design_reusing_parts.csv");
        let library_path = binding.to_str().unwrap();

        let parts_list = parse_library(parts_path, library_path)?;

        let _ = import_library(
            &context,
            collection,
            Sample::DEFAULT_NAME,
            "m123",
            parts_list,
            Some(parts_path),
            Some(library_path),
        );

        let block_groups = Sample::get_block_groups(conn, collection, Sample::DEFAULT_NAME);
        let block_group = &block_groups[0];

        let mut expected_sequences = vec![];
        for part1 in ["AAAA", "TAAT", "CAAC"].iter() {
            for part2 in ["AAAA", "TAAT", "CAAC"].iter() {
                expected_sequences.push(part1.to_string() + part2);
            }
        }
        let all_sequences = BlockGroup::get_all_sequences(conn, &block_group.id, false).unwrap();
        assert_eq!(
            all_sequences,
            expected_sequences
                .into_iter()
                .map(|x| x.to_string())
                .collect()
        );

        Ok(())
    }

    #[test]
    fn annotations_created_for_all_parts() -> Result<()> {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();
        let collection = "test";
        let library_name = "m123";

        let binding = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/parts.fa");
        let parts_path = binding.to_str().unwrap();

        let binding =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/design_reusing_parts.csv");
        let library_path = binding.to_str().unwrap();

        let parts_list = parse_library(parts_path, library_path)?;

        import_library(
            &context,
            collection,
            Sample::DEFAULT_NAME,
            library_name,
            parts_list,
            Some(parts_path),
            Some(library_path),
        )?;

        let annotations = Annotation::query_by_group(conn, library_name).unwrap();
        let mut names: Vec<_> = annotations.iter().map(|a| a.name.as_str()).collect();
        names.sort();
        assert_eq!(names, ["p1", "p1_2", "p2", "p2_2", "p3", "p3_2"]);

        Ok(())
    }

    #[test]
    fn annotations_created_for_distinct_columns() -> Result<()> {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();
        let collection = "test";
        let library_name = "m123";

        let binding = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/parts.fa");
        let parts_path = binding.to_str().unwrap();

        let binding =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/combinatorial_design.csv");
        let library_path = binding.to_str().unwrap();

        let parts_list = parse_library(parts_path, library_path)?;

        import_library(
            &context,
            collection,
            Sample::DEFAULT_NAME,
            library_name,
            parts_list,
            Some(parts_path),
            Some(library_path),
        )?;

        let annotations = Annotation::query_by_group(conn, library_name).unwrap();
        let mut names: Vec<_> = annotations.iter().map(|a| a.name.as_str()).collect();
        names.sort();
        assert_eq!(names, ["cds1", "cds2", "cds3", "p1", "p2", "p3"]);

        Ok(())
    }

    #[test]
    fn annotation_offsets_from_fasta_qualifiers() -> Result<()> {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();
        let collection = "test";
        let library_name = "offset_lib";

        let binding = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/parts_with_annotation_offsets.fa");
        let parts_path = binding.to_str().unwrap();
        let binding =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/single_column_design.csv");
        let library_path = binding.to_str().unwrap();

        let parts_list = parse_library(parts_path, library_path)?;
        import_library(
            &context,
            collection,
            Sample::DEFAULT_NAME,
            library_name,
            parts_list,
            Some(parts_path),
            Some(library_path),
        )?;

        let annotations = Annotation::query_by_group(conn, library_name).unwrap();
        let mut annotations_by_name: std::collections::HashMap<_, _> = annotations
            .into_iter()
            .map(|a| (a.name.clone(), a))
            .collect();

        // p1 has [GEN_annotation_start=1][GEN_annotation_end=3] — covers "AA" (chars 1..3 of "AAAA")
        let p1 = annotations_by_name.remove("p1").unwrap();
        let p1_edges = Accession::get_edges_by_id(conn, &p1.accession_id);
        assert_eq!(
            p1_edges[0].target_coordinate, 1,
            "p1 annotation should start at offset 1"
        );
        assert_eq!(
            p1_edges[1].source_coordinate, 3,
            "p1 annotation should end at offset 3"
        );
        let p1_node_id = p1_edges[0].target_node_id;
        let p1_sequences = Node::get_sequences_by_node_ids(conn, &[p1_node_id]);
        let p1_seq = p1_sequences[&p1_node_id]
            .get_sequence(1_i64, 3_i64)
            .unwrap();
        assert_eq!(
            p1_seq, "AA",
            "p1 annotation sequence should be the inner slice"
        );

        // GEN_annotation_start/end must not appear as modifiers on the stored annotation
        assert!(
            p1.extra
                .as_ref()
                .and_then(|e| e.fasta.as_ref())
                .map(|f| f
                    .modifiers
                    .iter()
                    .all(|m| m.key != "GEN_annotation_start" && m.key != "GEN_annotation_end"))
                .unwrap_or(true),
            "GEN_annotation_start/end should be stripped from stored FastaExtra"
        );

        // p2 has no offsets — covers the full sequence "TAAT"
        let p2 = annotations_by_name.remove("p2").unwrap();
        let p2_edges = Accession::get_edges_by_id(conn, &p2.accession_id);
        assert_eq!(
            p2_edges[0].target_coordinate, 0,
            "p2 annotation should start at 0"
        );
        assert_eq!(
            p2_edges[1].source_coordinate, 4,
            "p2 annotation should end at sequence length"
        );

        // p3 has [GEN_annotation_end=3] only — covers "CAA" (chars 0..3 of "CAAC")
        let p3 = annotations_by_name.remove("p3").unwrap();
        let p3_edges = Accession::get_edges_by_id(conn, &p3.accession_id);
        assert_eq!(
            p3_edges[0].target_coordinate, 0,
            "p3 annotation should start at 0"
        );
        assert_eq!(
            p3_edges[1].source_coordinate, 3,
            "p3 annotation should end at offset 3"
        );
        let p3_node_id = p3_edges[0].target_node_id;
        let p3_sequences = Node::get_sequences_by_node_ids(conn, &[p3_node_id]);
        let p3_seq = p3_sequences[&p3_node_id]
            .get_sequence(0_i64, 3_i64)
            .unwrap();
        assert_eq!(
            p3_seq, "CAA",
            "p3 annotation sequence should stop before last char"
        );

        Ok(())
    }

    #[test]
    fn annotation_offsets_from_sequence_part() -> Result<()> {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();
        let collection = "test";
        let library_name = "offset_sp_lib";

        use crate::graphs::combinatorial_library::SequencePart;

        let parts_list = vec![vec![SequencePart {
            name: "cds".to_string(),
            sequence: "ATGATAA".to_string(),
            sequence_length: 7,
            fasta_extra: None,
            metadata: None,
            annotation_start: Some(3),
            annotation_end: Some(6),
        }]];

        import_library(
            &context,
            collection,
            Sample::DEFAULT_NAME,
            library_name,
            parts_list,
            None,
            None,
        )?;

        let annotations = Annotation::query_by_group(conn, library_name).unwrap();
        assert_eq!(annotations.len(), 1);
        let cds = &annotations[0];
        let edges = Accession::get_edges_by_id(conn, &cds.accession_id);
        assert_eq!(
            edges[0].target_coordinate, 3,
            "annotation_start should be 3"
        );
        assert_eq!(edges[1].source_coordinate, 6, "annotation_end should be 6");

        let node_id = edges[0].target_node_id;
        let sequences = Node::get_sequences_by_node_ids(conn, &[node_id]);
        let seq = sequences[&node_id].get_sequence(3_i64, 6_i64).unwrap();
        // "ATGATAA"[3..6] = "ATA"
        assert_eq!(
            seq, "ATA",
            "annotation should cover the middle slice of the part sequence"
        );

        Ok(())
    }
}
