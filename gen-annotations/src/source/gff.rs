//! GFF parsing and translation adapters.
//!
//! The source of the GFF file comes from the caller, this file works off a readable input.
//! The readers here carry out parsing that input, matching identifiers, and returning an
//! Annotation object for use downstream.

use std::{
    io::{BufRead, Cursor, Read},
    sync::Arc,
};

use gen_core::{HashId, NodeIntervalBlock};
use gen_models::{
    accession::Accession,
    annotations::{Annotation, AnnotationExtra, GffAttribute, GffExtra},
    interval_tree::IntervalTreeSource as _,
};
use intervaltree::IntervalTree;
use noodles::gff;

use super::{AnnotationTranslationContext, FileAnnotationError};
use crate::{gff::gff_attribute_value_to_string, translate::gff::translate_gff};

/// Parse one identifier's matching GFF records into the existing annotation model.
pub fn parse_gff_annotation<R>(
    identifier: impl Into<String>,
    reader: R,
) -> Result<Annotation, FileAnnotationError>
where
    R: Read + BufRead,
{
    let identifier = identifier.into();
    let records = gff::io::Reader::new(reader)
        .record_bufs()
        .filter_map(|result| match result {
            Ok(record) => matching_gff_record(&record, &identifier).then_some(Ok(record)),
            Err(error) => Some(Err(error)),
        })
        .collect::<Result<Vec<_>, _>>()?;
    parse_gff_annotation_records(identifier, records)
}

/// Build an annotation from records already matched by an upstream lookup provider.
pub fn parse_gff_annotation_records<I>(
    identifier: impl Into<String>,
    records: I,
) -> Result<Annotation, FileAnnotationError>
where
    I: IntoIterator<Item = gff::feature::RecordBuf>,
{
    let identifier = identifier.into();
    let records: Vec<_> = records.into_iter().collect();
    let first = records.first().ok_or(FileAnnotationError::Empty)?;
    let gff = GffExtra {
        source: Some(first.source().to_string()),
        ty: first.ty().to_string(),
        score: first.score().map(|score| score.to_string()),
        phase: first.phase().map(|phase| match phase {
            gff::feature::record::Phase::Zero => "0".to_string(),
            gff::feature::record::Phase::One => "1".to_string(),
            gff::feature::record::Phase::Two => "2".to_string(),
        }),
        attributes: records
            .iter()
            .flat_map(|record| {
                record
                    .attributes()
                    .as_ref()
                    .iter()
                    .map(|(tag, value)| GffAttribute {
                        key: String::from_utf8_lossy(tag.as_ref()).into_owned(),
                        values: value
                            .iter()
                            .map(|item| String::from_utf8_lossy(item.as_ref()).into_owned())
                            .collect(),
                    })
            })
            .collect(),
    };
    let id = HashId::convert_str(&identifier);
    Ok(Annotation {
        id,
        name: identifier,
        group: "gff".to_string(),
        accession_id: id,
        cached_interval_tree: None,
        extra: Some(AnnotationExtra {
            gff: Some(gff),
            ..AnnotationExtra::default()
        }),
    })
}

/// Translate matching GFF records and cache their graph intervals on the annotation.
///
/// Returns the associated accession.
pub fn translate_gff_annotation<R>(
    context: &AnnotationTranslationContext<'_>,
    annotation: &mut Annotation,
    reader: R,
) -> Result<Accession, FileAnnotationError>
where
    R: Read + BufRead,
{
    let records = gff::io::Reader::new(reader)
        .record_bufs()
        .filter_map(|result| match result {
            Ok(record) => matching_gff_record(&record, &annotation.name).then_some(Ok(record)),
            Err(error) => Some(Err(error)),
        })
        .collect::<Result<Vec<_>, _>>()?;
    translate_gff_annotation_records(context, annotation, records)
}

/// Translate selected GFF records and cache their graph intervals on the annotation.
///
/// Returns the associated accession.
pub fn translate_gff_annotation_records<I>(
    context: &AnnotationTranslationContext<'_>,
    annotation: &mut Annotation,
    records: I,
) -> Result<Accession, FileAnnotationError>
where
    I: IntoIterator<Item = gff::feature::RecordBuf>,
{
    let mut input = Vec::new();
    {
        let mut writer = gff::io::Writer::new(&mut input);
        for record in records {
            writer.write_record(&record)?;
        }
    }
    let mut translated = Vec::new();
    translate_gff(
        context.conn,
        context.workspace,
        context.collection_name,
        context.sample_name,
        context.history_ref,
        Cursor::new(input),
        &mut translated,
    )
    .map_err(|error| FileAnnotationError::Translation(error.to_string()))?;
    let interval_tree = Arc::new(translated_gff_interval_tree(&translated)?);
    annotation.set_cached_interval_tree(Some(Arc::clone(&interval_tree)));
    let mut accession = Accession {
        id: annotation.accession_id,
        name: annotation.name.clone(),
        block_group_id: context.block_group_id,
        parent_accession_id: None,
        cached_interval_tree: None,
    };
    accession.set_cached_interval_tree(Some(interval_tree));
    Ok(accession)
}

fn matching_gff_record(record: &gff::feature::RecordBuf, identifier: &str) -> bool {
    ["Name", "ID", "gene", "gene_name", "db_xref"]
        .iter()
        .any(|key| {
            gff_attribute_value_to_string(record.attributes(), key)
                .is_some_and(|value| value.eq_ignore_ascii_case(identifier))
        })
}

fn translated_gff_interval_tree(
    translated: &[u8],
) -> Result<IntervalTree<i64, NodeIntervalBlock>, FileAnnotationError> {
    let mut reader = gff::io::Reader::new(Cursor::new(translated));
    let mut position = 0;
    let mut blocks = Vec::new();
    for result in reader.record_bufs() {
        let record = result?;
        let node_name = record.reference_sequence_name().to_string();
        let node_id = HashId::try_from(node_name.as_str())
            .map_err(|_| FileAnnotationError::InvalidNode(node_name.clone()))?;
        let start = record.start().get() as i64 - 1;
        let end = record.end().get() as i64;
        if end <= start {
            continue;
        }
        let block = NodeIntervalBlock {
            node_id,
            start: position,
            end: position + end - start,
            sequence_start: start,
            sequence_end: end,
            strand: record.strand().into(),
        };
        position = block.end;
        blocks.push(block);
    }
    if blocks.is_empty() {
        return Err(FileAnnotationError::Empty);
    }
    Ok(blocks
        .into_iter()
        .map(|block| (block.start..block.end, block))
        .collect())
}

#[cfg(test)]
mod tests {
    use std::{
        fs::File,
        io::{BufReader, Cursor},
        sync::Arc,
    };

    use gen_core::{HashId, NodeIntervalBlock, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand};
    use gen_graph::{GraphNode, GraphNodeSlice};
    use gen_models::{
        accession::Accession,
        annotations::Annotation,
        block_group::{BlockGroup, NewBlockGroup},
        block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
        edge::Edge,
        interval_tree::IntervalTreeSource as _,
        locus::GraphLocus,
        node::Node,
        path::Path,
        region::ResolvedGenRegion,
        sample::Sample,
        sequence::Sequence,
    };
    use intervaltree::IntervalTree;

    use super::{AnnotationTranslationContext, parse_gff_annotation, translate_gff_annotation};
    use crate::{
        parse_bed_annotation,
        translate::{bed::translate_bed, gff::translate_gff},
        translate_bed_annotation,
    };

    #[test]
    fn test_gff_annotation_matches_metadata_and_resolves_slices() {
        let conn = crate::test_helpers::get_connection();
        crate::test_helpers::setup_test_data(&conn);
        let block_group = Sample::get_block_groups(&conn, "test", Sample::DEFAULT_NAME, None)
            .into_iter()
            .find(|block_group| block_group.name == "m123")
            .expect("should find test block group");
        let context = AnnotationTranslationContext {
            conn: &conn,
            workspace: crate::test_helpers::test_workspace(),
            collection_name: "test",
            sample_name: Sample::DEFAULT_NAME,
            block_group_id: block_group.id,
            history_ref: None,
        };
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/simple.gff");
        let mut annotation = parse_gff_annotation(
            "gene-a0001",
            BufReader::new(File::open(path).expect("should open simple GFF fixture")),
        )
        .expect("should parse the matching GFF record");
        assert_eq!(annotation.name, "gene-a0001");
        assert_eq!(annotation.id, HashId::convert_str("gene-a0001"));
        assert_eq!(annotation.accession_id, annotation.id);
        assert_eq!(annotation.group, "gff");
        let metadata = annotation
            .extra
            .as_ref()
            .and_then(|extra| extra.gff.as_ref())
            .expect("should preserve GFF metadata");
        let first_attribute = metadata
            .attributes
            .first()
            .expect("should preserve the GFF ID attribute");
        assert_eq!(first_attribute.key, "ID");
        assert_eq!(first_attribute.values, vec!["gene-a0001".to_string()]);
        let accession = translate_gff_annotation(
            &context,
            &mut annotation,
            BufReader::new(File::open(path).expect("should reopen simple GFF fixture")),
        )
        .expect("should translate the matching GFF record");
        assert_eq!(accession.id, annotation.accession_id);
        let annotation_tree = annotation
            .intervaltree(&conn)
            .expect("should get the cached GFF annotation tree");
        let accession_tree = accession
            .intervaltree(&conn)
            .expect("should get the cached GFF accession tree");
        assert!(Arc::ptr_eq(&annotation_tree, &accession_tree));
        assert!(Arc::ptr_eq(
            &annotation_tree,
            annotation
                .cached_interval_tree()
                .expect("should find the cached GFF annotation tree")
        ));
        assert!(Arc::ptr_eq(
            &accession_tree,
            accession
                .cached_interval_tree()
                .expect("should find the cached GFF accession tree")
        ));
        assert_eq!(annotation_tree.iter().count(), 2);
        assert_eq!(annotation.length(&conn).expect("should get GFF length"), 16);
        assert!(
            Accession::select(&conn)
                .id(accession.id)
                .load()
                .expect("should query absent file accession")
                .is_empty()
        );

        let mut accession_without_cache = accession.clone();
        accession_without_cache.set_cached_interval_tree(None);
        let resolved = ResolvedGenRegion::from_annotation(
            &conn,
            &annotation,
            &accession_without_cache,
            12,
            15,
        )
        .expect("should resolve a cached GFF annotation without accession nodes")
        .find_graph_positions(&conn, crate::test_helpers::test_workspace(), 0, 0)
        .expect("should resolve graph positions across translated slices");
        assert_eq!(resolved.anchor_end, 16);
        assert_eq!(resolved.feature_length, 16);
        assert_eq!(
            resolved
                .start_anchors
                .as_ref()
                .expect("should have start anchors")[0]
                .graph_node
                .node_id,
            HashId::try_from("b59698a422128d20462c44537b2d23ef")
                .expect("should parse the first translated node id")
        );
        assert_eq!(
            resolved
                .end_anchors
                .as_ref()
                .expect("should have end anchors")[0]
                .graph_node
                .node_id,
            HashId::try_from("6b460b727030cae3bae7cf389074d4ba")
                .expect("should parse the second translated node id")
        );

        let mut intervals = annotation_tree
            .iter()
            .map(|entry| entry.value)
            .collect::<Vec<_>>();
        intervals.sort_by_key(|interval| interval.start);
        assert_eq!(
            intervals
                .iter()
                .map(|interval| (
                    interval.node_id,
                    interval.sequence_start,
                    interval.sequence_end,
                    interval.strand,
                ))
                .collect::<Vec<_>>(),
            [
                (
                    HashId::try_from("b59698a422128d20462c44537b2d23ef")
                        .expect("should parse the first translated node id"),
                    4,
                    17,
                    Strand::Forward,
                ),
                (
                    HashId::try_from("6b460b727030cae3bae7cf389074d4ba")
                        .expect("should parse the second translated node id"),
                    0,
                    3,
                    Strand::Forward,
                ),
            ],
            "cached GFF spans should be local to each projected node",
        );
    }

    // Edge traversal diagrams for the split-node GFF and BED tests below. Each case is a separate
    // path input with nodes B then A. Positive stored ranges select bases first. The edge strand
    // shown is the source/target endpoint orientation for that traversal.
    //
    // Stored node A:
    //   index: 012345678
    //   DNA:   AACCGGTTA
    //   span:  ^--------^ [0,9)
    //
    // Stored node B:
    //   index: 012345678
    //   DNA:   CCGTTGCAT
    //   span:  ^-------^ [0,8); stored index 8 (T) is excluded.
    //
    // Case 1: path edges traverse B then A; edge endpoints at B and A have strand +.
    //   path index:    01234567890123456
    //   path nodes:    BBBBBBBBAAAAAAAAA
    //   path spans:    ^-------^--------^
    //   path DNA:      CCGTTGCAAACCGGTTA
    //
    // Case 2: path edges traverse B then A; edge endpoints at B and A have strand -. Path DNA
    // shown is negative strand. It is stored as positive strand
    //   path index:    01234567890123456
    //   path nodes:    BBBBBBBBAAAAAAAAA
    //   path spans:    ^-------^--------^
    //   path DNA:      TGCAACGGTAACCGGTT
    fn create_two_node_test_block_group(
        conn: &gen_models::db::GraphConnection,
        name: &str,
        strand: Strand,
    ) -> (BlockGroup, HashId, HashId) {
        let block_group = BlockGroup::create(
            conn,
            NewBlockGroup {
                collection_name: "test",
                sample_name: Sample::DEFAULT_NAME,
                name,
                parent_block_group_id: None,
                is_default: false,
            },
        )
        .expect("should create test block group");
        let sequence_a_text = "AACCGGTTA";
        let sequence_b_text = "CCGTTGCAT";
        let sequence_a = Sequence::new()
            .sequence_type("DNA")
            .sequence(sequence_a_text)
            .save(conn)
            .expect("should save first path sequence");
        let node_a_id = HashId::convert_str(&format!("{name}-path-node-a"));
        Node::create(conn, &sequence_a.hash, &node_a_id).expect("should create first path node");
        let sequence_b = Sequence::new()
            .sequence_type("DNA")
            .sequence(sequence_b_text)
            .save(conn)
            .expect("should save second path sequence");
        let node_b_id = HashId::convert_str(&format!("{name}-path-node-b"));
        Node::create(conn, &sequence_b.hash, &node_b_id).expect("should create second path node");

        let edge_start = Edge::create(
            conn,
            PATH_START_NODE_ID,
            0,
            Strand::Forward,
            node_b_id,
            0,
            strand,
        )
        .expect("should create path start edge");
        let edge_between = Edge::create(conn, node_b_id, 8, strand, node_a_id, 0, strand)
            .expect("should connect path nodes");
        let edge_end = Edge::create(
            conn,
            node_a_id,
            9,
            strand,
            PATH_END_NODE_ID,
            0,
            Strand::Forward,
        )
        .expect("should create path end edge");
        let edge_ids = [edge_start.id, edge_between.id, edge_end.id];
        BlockGroupEdge::bulk_create(
            conn,
            &edge_ids
                .iter()
                .map(|edge_id| BlockGroupEdgeData {
                    block_group_id: block_group.id,
                    edge_id: *edge_id,
                    chromosome_index: 0,
                    phased: 0,
                })
                .collect::<Vec<_>>(),
        );
        let path =
            Path::create(conn, name, &block_group.id, &edge_ids).expect("should create test path");
        let sequences_by_node_id = Node::get_sequences_by_node_ids(
            conn,
            crate::test_helpers::test_workspace(),
            &[node_a_id, node_b_id],
            None,
        );
        assert_eq!(
            sequences_by_node_id[&node_a_id]
                .get_sequence(None, None)
                .expect("should fetch stored sequence for first node"),
            sequence_a_text,
            "node DNA should stay in its stored 5′-to-3′ orientation"
        );
        assert_eq!(
            sequences_by_node_id[&node_b_id]
                .get_sequence(None, None)
                .expect("should fetch stored sequence for second node"),
            sequence_b_text,
            "node DNA should stay in its stored 5′-to-3′ orientation"
        );
        let path_sequence = path
            .sequence(conn, crate::test_helpers::test_workspace(), None)
            .expect("should load test path sequence");
        assert_eq!(
            path_sequence,
            if strand == Strand::Reverse {
                "TGCAACGGTAACCGGTT"
            } else {
                "CCGTTGCAAACCGGTTA"
            },
            "path retrieval should orient node DNA from the path strand"
        );
        assert_eq!(
            &path_sequence[7..9],
            if strand == Strand::Reverse {
                "GT"
            } else {
                "AA"
            },
            "the split-node feature fixture should spell its strand-specific path bases"
        );

        (block_group, node_a_id, node_b_id)
    }

    fn create_reverse_test_block_group(
        conn: &gen_models::db::GraphConnection,
    ) -> (BlockGroup, HashId, HashId) {
        create_two_node_test_block_group(conn, "reverse", Strand::Reverse)
    }

    fn create_forward_test_block_group(
        conn: &gen_models::db::GraphConnection,
    ) -> (BlockGroup, HashId, HashId) {
        create_two_node_test_block_group(conn, "forward", Strand::Forward)
    }

    // Single-node reverse-fetch fixture used by both cached-feature regressions:
    //   index tens:   000000000011
    //   index ones:   012345678901
    //   stored DNA:   AAATTTCCCGGG
    //   raw block:       ^---^ [3,7), strand -
    //
    //   feature offset: 0123
    //   feature DNA:    GAAA
    //   query [0,2):    ^-^ = GA; it maps to raw [5,7) TC.
    // Unchanged raw bounds [3,5) would fetch TT and return AA.
    fn create_cached_reverse_test_feature(
        conn: &gen_models::db::GraphConnection,
    ) -> (Annotation, Accession, HashId) {
        let sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("AAATTTCCCGGG")
            .save(conn)
            .expect("should store sequence in original orientation");
        let node_id = HashId::convert_str("reverse-fetch-test-node");
        Node::create(conn, &sequence.hash, &node_id).expect("should create test node");
        let block_group = Sample::get_block_groups(conn, "test", Sample::DEFAULT_NAME, None)
            .into_iter()
            .find(|block_group| block_group.name == "m123")
            .expect("should find test block group");
        let interval_tree: IntervalTree<i64, NodeIntervalBlock> = [(
            0..4,
            NodeIntervalBlock {
                node_id,
                start: 0,
                end: 4,
                sequence_start: 3,
                sequence_end: 7,
                strand: Strand::Reverse,
            },
        )]
        .into_iter()
        .collect();
        let interval_tree = Arc::new(interval_tree);
        let annotation_id = HashId::convert_str("reverse-fetch-test-annotation");
        let mut annotation = Annotation {
            id: annotation_id,
            name: "reverse-fetch-test-annotation".to_string(),
            group: "test".to_string(),
            accession_id: annotation_id,
            cached_interval_tree: None,
            extra: None,
        };
        annotation.set_cached_interval_tree(Some(Arc::clone(&interval_tree)));
        let mut accession = Accession {
            id: annotation_id,
            name: annotation.name.clone(),
            block_group_id: block_group.id,
            parent_accession_id: None,
            cached_interval_tree: None,
        };
        accession.set_cached_interval_tree(Some(interval_tree));

        (annotation, accession, node_id)
    }

    #[test]
    fn test_reverse_graph_locus_fetches_reverse_complement_from_original_node_dna() {
        let conn = crate::test_helpers::get_connection();
        crate::test_helpers::setup_test_data(&conn);
        let (annotation, accession, node_id) = create_cached_reverse_test_feature(&conn);
        let workspace = crate::test_helpers::test_workspace();
        let stored_sequences = Node::get_sequences_by_node_ids(&conn, workspace, &[node_id], None);
        assert_eq!(
            stored_sequences[&node_id]
                .get_sequence(None, None)
                .expect("should fetch stored node DNA"),
            "AAATTTCCCGGG",
            "stored node DNA should not be pre-reversed for reverse retrieval"
        );
        assert_eq!(
            stored_sequences[&node_id]
                .get_sequence(Some(3), Some(7))
                .expect("should fetch raw node slice"),
            "TTTC",
            "sequence coordinates 3..7 should select the stored bases TTTC"
        );

        // The shared diagram above shows full reverse retrieval of raw [3,7) as GAAA.
        let resolved = ResolvedGenRegion::from_annotation(&conn, &annotation, &accession, 0, 4)
            .expect("should resolve the cached reverse-strand feature");
        let locus = resolved
            .graph_locus(&conn, workspace)
            .expect("should build the full reverse-strand locus");
        assert_eq!(
            locus.sequence(&conn, workspace),
            b"GAAA",
            "reverse retrieval should reverse-complement the raw node slice TTTC"
        );
    }

    #[test]
    fn test_resolved_reverse_feature_interior_bounds_map_before_fetch() {
        let conn = crate::test_helpers::get_connection();
        crate::test_helpers::setup_test_data(&conn);
        let (annotation, accession, node_id) = create_cached_reverse_test_feature(&conn);
        let workspace = crate::test_helpers::test_workspace();
        let stored_sequences = Node::get_sequences_by_node_ids(&conn, workspace, &[node_id], None);
        assert_eq!(
            stored_sequences[&node_id]
                .get_sequence(Some(5), Some(7))
                .expect("should fetch the reverse-oriented interior source slice"),
            "TC",
            "stored coordinates [5,7) should retain their original node orientation"
        );

        // The shared single-node diagram above shows feature [0,2) mapping to raw [5,7).
        let resolved = ResolvedGenRegion::from_annotation(&conn, &annotation, &accession, 0, 2)
            .expect("should resolve the reverse-strand feature prefix");
        let locus = resolved
            .graph_locus(&conn, workspace)
            .expect("should map the reverse-strand feature prefix to node coordinates");
        assert_eq!(
            (locus.slices[0].start, locus.slices[0].end),
            (2, 4),
            "feature-relative [0,2) should map to the stored block's final two bases"
        );
        assert_eq!(
            locus.sequence(&conn, workspace),
            b"GA",
            "the resolved reverse-strand interior should fetch GA, not the same raw bounds"
        );
    }

    #[test]
    fn test_reverse_graph_locus_fetches_multiple_node_slices_in_feature_order() {
        let conn = crate::test_helpers::get_connection();
        crate::test_helpers::setup_test_data(&conn);
        let sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("AAAACCCCTTTTGGGG")
            .save(&conn)
            .expect("should store multi-node DNA in original orientation");
        let first_node_id = HashId::convert_str("reverse-multi-fetch-first-node");
        let second_node_id = HashId::convert_str("reverse-multi-fetch-second-node");
        Node::create(&conn, &sequence.hash, &first_node_id).expect("should create first test node");
        Node::create(&conn, &sequence.hash, &second_node_id)
            .expect("should create second test node");
        let workspace = crate::test_helpers::test_workspace();
        let stored_sequences = Node::get_sequences_by_node_ids(
            &conn,
            workspace,
            &[first_node_id, second_node_id],
            None,
        );
        assert_eq!(
            stored_sequences[&first_node_id]
                .get_sequence(Some(2), Some(6))
                .expect("should fetch first raw node slice"),
            "AACC",
            "the first stored node slice should remain AACC"
        );
        assert_eq!(
            stored_sequences[&second_node_id]
                .get_sequence(Some(9), Some(14))
                .expect("should fetch second raw node slice"),
            "TTTGG",
            "the second stored node slice should remain TTTGG"
        );

        // This tests reverse complementing a feature spanning 2 nodes:
        //   index tens: 0000000000111111
        //   index ones: 0123456789012345
        //   stored DNA: AAAACCCCTTTTGGGG
        //   node A:       ^---^ [2,6) = AACC
        //   node B:              ^----^ [9,14) = TTTGG
        //   GraphNodeSlice order (both strand -): node B [9,14), then node A [2,6).
        //
        // The crucial bit here is not reverse complementing as the sequence is streamed.
        // For example, if we RC(AACC)+RC(TTTGG), we end up with GGTTCCAAA instead of
        // CCAAAGGTT.

        let locus = GraphLocus {
            slices: vec![
                GraphNodeSlice {
                    block: GraphNode {
                        node_id: second_node_id,
                        sequence_start: 9,
                        sequence_end: 14,
                    },
                    start: 0,
                    end: 5,
                    strand: Strand::Reverse,
                },
                GraphNodeSlice {
                    block: GraphNode {
                        node_id: first_node_id,
                        sequence_start: 2,
                        sequence_end: 6,
                    },
                    start: 0,
                    end: 4,
                    strand: Strand::Reverse,
                },
            ],
        };
        assert_eq!(
            locus.sequence(&conn, workspace),
            b"CCAAAGGTT",
            "reverse-complemented node slices should concatenate in feature order"
        );
    }

    fn translate_gff_input(conn: &gen_models::db::GraphConnection, input: &str) -> String {
        let mut translated = Vec::new();
        translate_gff(
            conn,
            crate::test_helpers::test_workspace(),
            "test",
            Sample::DEFAULT_NAME,
            None,
            Cursor::new(input.as_bytes()),
            &mut translated,
        )
        .expect("should translate GFF input");
        String::from_utf8(translated).expect("should write UTF-8 GFF output")
    }

    fn translate_bed_input(conn: &gen_models::db::GraphConnection, input: &str) -> String {
        let mut translated = Vec::new();
        translate_bed(
            conn,
            crate::test_helpers::test_workspace(),
            "test",
            Sample::DEFAULT_NAME,
            None,
            Cursor::new(input.as_bytes()),
            &mut translated,
        )
        .expect("should translate BED input");
        String::from_utf8(translated).expect("should write UTF-8 BED output")
    }

    #[test]
    fn test_gene_name_only_gff_record_parses_and_translates_case_insensitively() {
        let conn = crate::test_helpers::get_connection();
        crate::test_helpers::setup_test_data(&conn);
        let block_group = Sample::get_block_groups(&conn, "test", Sample::DEFAULT_NAME, None)
            .into_iter()
            .find(|block_group| block_group.name == "m123")
            .expect("should find test block group");
        let context = AnnotationTranslationContext {
            conn: &conn,
            workspace: crate::test_helpers::test_workspace(),
            collection_name: "test",
            sample_name: Sample::DEFAULT_NAME,
            block_group_id: block_group.id,
            history_ref: None,
        };
        let input = "m123\tsource\tgene\t5\t6\t.\t+\t.\tID=private-gene-name-id;gene_name=BRCA1\n";
        let mut annotation =
            parse_gff_annotation("brca1", BufReader::new(Cursor::new(input.as_bytes())))
                .expect("should match a gene_name-only GFF record ignoring case");
        assert_eq!(annotation.name, "brca1");
        let metadata = annotation
            .extra
            .as_ref()
            .and_then(|extra| extra.gff.as_ref())
            .expect("should preserve the gene_name-only GFF metadata");
        assert!(
            metadata
                .attributes
                .iter()
                .any(|attribute| { attribute.key == "gene_name" && attribute.values == ["BRCA1"] })
        );

        let accession = translate_gff_annotation(
            &context,
            &mut annotation,
            BufReader::new(Cursor::new(input.as_bytes())),
        )
        .expect("should translate the gene_name-only GFF record");
        assert!(annotation.length(&conn).expect("should measure annotation") > 0);
        assert!(accession.length(&conn).expect("should measure accession") > 0);
    }

    #[test]
    fn test_translates_reverse_split_gff_to_exact_node_local_spans() {
        // This test takes a block group with 2 nodes connected by a reverse strand and takes a slice at the boundary.
        // This translates to the beginning on one node and the end of the other
        let conn = crate::test_helpers::get_connection();
        crate::test_helpers::setup_test_data(&conn);
        let (block_group, node_a_id, node_b_id) = create_reverse_test_block_group(&conn);
        // Path edges traverse B then A; edge endpoint strands at B/A are -. Feature strand is +.
        //   path index:  01234567890123456
        //   path nodes:  BBBBBBBBAAAAAAAAA
        //   path DNA:    TGCAACGGTAACCGGTT    (negative strand)
        //   feature:            ^^ [7,9) = GT
        //
        // Stored node B (these are positive strand sequences, the stored block group has a reverse edge):
        //   index:     012345678
        //   DNA:       CCGTTGCAT
        //   span:      ^-------^ [0,8)
        //   selected:  ^ [0,1) = C
        //
        // Stored node A:
        //   index:     012345678
        //   DNA:       AACCGGTTA
        //   span:      ^--------^ [0,9)
        //   selected:          ^ [8,9) = A
        // Edge strand - maps stored C to G and A to T; feature strand + emits fragments B then A.
        // GFF 8..9 maps to path [7,9), producing B 1..1 (-) then A 9..9 (-).
        // This explains orientation; the assertion checks emitted records, not fetched DNA.
        let input = format!(
            "##gff-version 3\n{}\tsource\tgene\t8\t9\t.\t+\t.\tID=cross-boundary;Name=cross-boundary\n",
            block_group.name
        );
        let translated = translate_gff_input(&conn, &input);
        assert_eq!(
            translated,
            format!(
                "{node_b_id}\tsource\tgene\t1\t1\t.\t-\t.\tID=cross-boundary;Name=cross-boundary\n{node_a_id}\tsource\tgene\t9\t9\t.\t-\t.\tID=cross-boundary;Name=cross-boundary\n"
            ),
            "positive-coordinate GFF fragments should clip to each reverse-oriented node in path order"
        );
    }

    #[test]
    fn test_translates_reverse_split_bed_to_exact_node_local_spans() {
        let conn = crate::test_helpers::get_connection();
        crate::test_helpers::setup_test_data(&conn);
        let (block_group, node_a_id, node_b_id) = create_reverse_test_block_group(&conn);
        // Path edges traverse B then A; edge endpoint strands at B/A are -. Feature strand is +.
        //   path index:  01234567890123456
        //   path nodes:  BBBBBBBBAAAAAAAAA
        //   path DNA:    TGCAACGGTAACCGGTT
        //   feature:            ^^ [7,9) = GT
        //
        // Stored node B:
        //   index:     012345678
        //   DNA:       CCGTTGCAT
        //   span:      ^-------^ [0,8)
        //   selected:  ^ [0,1) = C
        //
        // Stored node A:
        //   index:     012345678
        //   DNA:       AACCGGTTA
        //   span:      ^--------^ [0,9)
        //   selected:          ^ [8,9) = A
        // Edge strand - maps stored C to G and A to T; feature strand + emits fragments B then A.
        // BED [7,9) maps to path [7,9), producing B [0,1) (-) then A [8,9) (-).
        // This explains orientation; the assertion checks emitted records, not fetched DNA.
        let input = format!("{}\t7\t9\tcross-boundary\t0\t+\n", block_group.name);
        let translated = translate_bed_input(&conn, &input);
        assert_eq!(
            translated,
            format!(
                "{node_b_id}\t0\t1\tcross-boundary\t0\t-\n{node_a_id}\t8\t9\tcross-boundary\t0\t-\n"
            ),
            "positive-coordinate BED fragments should clip to each reverse-oriented node in path order"
        );
    }

    #[test]
    fn test_translates_single_base_gff_before_node_boundary() {
        let conn = crate::test_helpers::get_connection();
        crate::test_helpers::setup_test_data(&conn);
        let (block_group, _, node_b_id) = create_reverse_test_block_group(&conn);
        // Fixture edge endpoints at B and A have strand -; this GFF feature has strand +.
        // GFF coordinate 8 selects path position 7 and maps to stored B [0,1), slice strand -.
        let input = format!(
            "{}\tsource\tgene\t8\t8\t.\t+\t.\tID=before-boundary;Name=before-boundary\n",
            block_group.name
        );
        let translated = translate_gff_input(&conn, &input);
        assert_eq!(
            translated,
            format!(
                "{node_b_id}\tsource\tgene\t1\t1\t.\t-\t.\tID=before-boundary;Name=before-boundary\n"
            ),
            "GFF coordinate 8 should map to the final base of the first path node"
        );
    }

    #[test]
    fn test_translates_single_base_gff_after_node_boundary() {
        let conn = crate::test_helpers::get_connection();
        crate::test_helpers::setup_test_data(&conn);
        let (block_group, node_a_id, _) = create_reverse_test_block_group(&conn);
        // Fixture edge endpoints at B and A have strand -; this GFF feature has strand +.
        // GFF coordinate 9 selects path position 8 and maps to stored A [8,9), slice strand -.
        let input = format!(
            "{}\tsource\tgene\t9\t9\t.\t+\t.\tID=after-boundary;Name=after-boundary\n",
            block_group.name
        );
        let translated = translate_gff_input(&conn, &input);
        assert_eq!(
            translated,
            format!(
                "{node_a_id}\tsource\tgene\t9\t9\t.\t-\t.\tID=after-boundary;Name=after-boundary\n"
            ),
            "GFF coordinate 9 should map to the first base of the second path node"
        );
    }

    #[test]
    fn test_resolves_minus_strand_gff_from_forward_and_reverse_path_dna() {
        let conn = crate::test_helpers::get_connection();
        crate::test_helpers::setup_test_data(&conn);
        let forward = create_forward_test_block_group(&conn);
        let reverse = create_reverse_test_block_group(&conn);

        // The same minus-strand feature is fetched through both fixture edge traversals.
        for ((block_group, node_a_id, node_b_id), name, expected_sequence) in [
            (forward, "forward", b"GGTTTGCAA".as_slice()),
            (reverse, "reverse", b"GTTACCGTT".as_slice()),
        ] {
            let context = AnnotationTranslationContext {
                conn: &conn,
                workspace: crate::test_helpers::test_workspace(),
                collection_name: "test",
                sample_name: Sample::DEFAULT_NAME,
                block_group_id: block_group.id,
                history_ref: None,
            };
            let identifier = format!("minus-{name}");
            let input = format!(
                "{name}\tsource\tgene\t4\t12\t.\t-\t.\tID={identifier};Name={identifier}\n"
            );
            let mut annotation = parse_gff_annotation(
                identifier.as_str(),
                BufReader::new(Cursor::new(input.as_bytes())),
            )
            .expect("should parse minus-strand GFF feature");
            let accession = translate_gff_annotation(
                &context,
                &mut annotation,
                BufReader::new(Cursor::new(input.as_bytes())),
            )
            .expect("should translate minus-strand GFF feature");
            let resolved = ResolvedGenRegion::from_annotation(&conn, &annotation, &accession, 0, 9)
                .expect("should resolve minus-strand GFF feature");
            let graph_locus = resolved
                .graph_locus(&conn, crate::test_helpers::test_workspace())
                .expect("should build minus-strand GFF graph locus");
            assert_eq!(
                graph_locus.sequence(&conn, crate::test_helpers::test_workspace()),
                expected_sequence,
                "minus-strand GFF sequence should be fetched in feature orientation on the {name} path"
            );
            assert_eq!(
                graph_locus
                    .slices
                    .iter()
                    .map(|slice| slice.block.node_id)
                    .collect::<Vec<_>>(),
                [node_a_id, node_b_id],
                "minus-strand GFF slices should follow feature order on the {name} path"
            );

            for (start, end, expected_base) in [
                (0, 1, &expected_sequence[..1]),
                (8, 9, &expected_sequence[8..9]),
            ] {
                let resolved =
                    ResolvedGenRegion::from_annotation(&conn, &annotation, &accession, start, end)
                        .expect("should resolve one minus-strand GFF base");
                let graph_locus = resolved
                    .graph_locus(&conn, crate::test_helpers::test_workspace())
                    .expect("should build one-base minus-strand GFF graph locus");
                assert_eq!(
                    graph_locus.sequence(&conn, crate::test_helpers::test_workspace()),
                    expected_base,
                    "one-base GFF slices should retain feature orientation on the {name} path"
                );
            }

            // These are two separate path inputs; both features have strand -.
            // Case 1: edge endpoints at B/A +.
            //   feature nodes:   AAAABBBBB
            //   feature offset:  012345678
            //   feature DNA:     GGTTTGCAA
            //   query [1,3):      ^-^ = GT
            //   query [2,6):       ^---^ = TTTG
            // Case 2: edge endpoints at B/A -.
            //   feature nodes:   AAAABBBBB
            //   feature offset:  012345678
            //   feature DNA:     GTTACCGTT
            //   query [1,3):      ^-^ = TT
            //   query [2,6):       ^---^ = TACC
            // [1,3) checks an interior slice of A; [2,6) trims across A/B. These independent
            // queries overlap at offset 2 (T), included in both results.
            for (start, end, expected_forward, expected_reverse) in [
                (1, 3, b"GT".as_slice(), b"TT".as_slice()),
                (2, 6, b"TTTG".as_slice(), b"TACC".as_slice()),
            ] {
                let expected_subsequence = if name == "forward" {
                    expected_forward
                } else {
                    expected_reverse
                };
                let resolved =
                    ResolvedGenRegion::from_annotation(&conn, &annotation, &accession, start, end)
                        .expect("should resolve a minus-strand GFF subrange");
                let graph_locus = resolved
                    .graph_locus(&conn, crate::test_helpers::test_workspace())
                    .expect("should build the trimmed minus-strand GFF graph locus");
                assert_eq!(
                    graph_locus.sequence(&conn, crate::test_helpers::test_workspace()),
                    expected_subsequence,
                    "minus-strand GFF subrange {start}..{end} should trim in feature orientation on the {name} path"
                );
                if (start, end) == (2, 6) {
                    let actual_slices = graph_locus
                        .slices
                        .iter()
                        .map(|slice| {
                            (
                                slice.block.node_id,
                                slice.block.sequence_start,
                                slice.block.sequence_end,
                                slice.start,
                                slice.end,
                                slice.strand,
                            )
                        })
                        .collect::<Vec<_>>();
                    let expected_slices = if name == "forward" {
                        vec![
                            (node_a_id, 0, 4, 0, 2, Strand::Reverse),
                            (node_b_id, 3, 8, 3, 5, Strand::Reverse),
                        ]
                    } else {
                        vec![
                            (node_a_id, 5, 9, 2, 4, Strand::Forward),
                            (node_b_id, 0, 5, 0, 2, Strand::Forward),
                        ]
                    };
                    assert_eq!(
                        actual_slices, expected_slices,
                        "boundary-crossing GFF subrange should trim node-local slices in feature order"
                    );
                }
            }
        }
    }

    #[test]
    fn test_resolves_minus_strand_bed_from_forward_and_reverse_path_dna() {
        let conn = crate::test_helpers::get_connection();
        crate::test_helpers::setup_test_data(&conn);
        let forward = create_forward_test_block_group(&conn);
        let reverse = create_reverse_test_block_group(&conn);

        // The same minus-strand feature is fetched through both fixture edge traversals.
        for ((block_group, node_a_id, node_b_id), name, expected_sequence) in [
            (forward, "forward", b"GGTTTGCAA".as_slice()),
            (reverse, "reverse", b"GTTACCGTT".as_slice()),
        ] {
            let context = AnnotationTranslationContext {
                conn: &conn,
                workspace: crate::test_helpers::test_workspace(),
                collection_name: "test",
                sample_name: Sample::DEFAULT_NAME,
                block_group_id: block_group.id,
                history_ref: None,
            };
            let identifier = format!("minus-{name}");
            let input = format!("{name}\t3\t12\t{identifier}\t0\t-\n");
            let mut annotation =
                parse_bed_annotation(identifier.as_str(), Cursor::new(input.as_bytes()))
                    .expect("should parse minus-strand BED feature");
            let accession =
                translate_bed_annotation(&context, &mut annotation, Cursor::new(input.as_bytes()))
                    .expect("should translate minus-strand BED feature");
            let resolved = ResolvedGenRegion::from_annotation(&conn, &annotation, &accession, 0, 9)
                .expect("should resolve minus-strand BED feature");
            let graph_locus = resolved
                .graph_locus(&conn, crate::test_helpers::test_workspace())
                .expect("should build minus-strand BED graph locus");
            assert_eq!(
                graph_locus.sequence(&conn, crate::test_helpers::test_workspace()),
                expected_sequence,
                "minus-strand BED sequence should be fetched in feature orientation on the {name} path"
            );
            assert_eq!(
                graph_locus
                    .slices
                    .iter()
                    .map(|slice| slice.block.node_id)
                    .collect::<Vec<_>>(),
                [node_a_id, node_b_id],
                "minus-strand BED slices should follow feature order on the {name} path"
            );

            for (start, end, expected_base) in [
                (0, 1, &expected_sequence[..1]),
                (8, 9, &expected_sequence[8..9]),
            ] {
                let resolved =
                    ResolvedGenRegion::from_annotation(&conn, &annotation, &accession, start, end)
                        .expect("should resolve one minus-strand BED base");
                let graph_locus = resolved
                    .graph_locus(&conn, crate::test_helpers::test_workspace())
                    .expect("should build one-base minus-strand BED graph locus");
                assert_eq!(
                    graph_locus.sequence(&conn, crate::test_helpers::test_workspace()),
                    expected_base,
                    "one-base BED slices should retain feature orientation on the {name} path"
                );
            }

            // These are two separate path inputs; both features have strand -.
            // Case 1: edge endpoints at B/A +.
            //   feature nodes:   AAAABBBBB
            //   feature offset:  012345678
            //   feature DNA:     GGTTTGCAA
            //   query [1,3):      ^-^ = GT
            //   query [2,6):       ^---^ = TTTG
            // Case 2: edge endpoints at B/A -.
            //   feature nodes:   AAAABBBBB
            //   feature offset:  012345678
            //   feature DNA:     GTTACCGTT
            //   query [1,3):      ^-^ = TT
            //   query [2,6):       ^---^ = TACC
            // [1,3) checks an interior slice of A; [2,6) trims across A/B. These independent
            // queries overlap at offset 2 (T), included in both results.
            for (start, end, expected_forward, expected_reverse) in [
                (1, 3, b"GT".as_slice(), b"TT".as_slice()),
                (2, 6, b"TTTG".as_slice(), b"TACC".as_slice()),
            ] {
                let expected_subsequence = if name == "forward" {
                    expected_forward
                } else {
                    expected_reverse
                };
                let resolved =
                    ResolvedGenRegion::from_annotation(&conn, &annotation, &accession, start, end)
                        .expect("should resolve a minus-strand BED subrange");
                let graph_locus = resolved
                    .graph_locus(&conn, crate::test_helpers::test_workspace())
                    .expect("should build the trimmed minus-strand BED graph locus");
                assert_eq!(
                    graph_locus.sequence(&conn, crate::test_helpers::test_workspace()),
                    expected_subsequence,
                    "minus-strand BED subrange {start}..{end} should trim in feature orientation on the {name} path"
                );
                if (start, end) == (2, 6) {
                    let actual_slices = graph_locus
                        .slices
                        .iter()
                        .map(|slice| {
                            (
                                slice.block.node_id,
                                slice.block.sequence_start,
                                slice.block.sequence_end,
                                slice.start,
                                slice.end,
                                slice.strand,
                            )
                        })
                        .collect::<Vec<_>>();
                    let expected_slices = if name == "forward" {
                        vec![
                            (node_a_id, 0, 4, 0, 2, Strand::Reverse),
                            (node_b_id, 3, 8, 3, 5, Strand::Reverse),
                        ]
                    } else {
                        vec![
                            (node_a_id, 5, 9, 2, 4, Strand::Forward),
                            (node_b_id, 0, 5, 0, 2, Strand::Forward),
                        ]
                    };
                    assert_eq!(
                        actual_slices, expected_slices,
                        "boundary-crossing BED subrange should trim node-local slices in feature order"
                    );
                }
            }
        }
    }
}
