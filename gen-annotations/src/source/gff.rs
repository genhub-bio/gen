//! GFF parsing and translation adapters.
//!
//! This serves as a general mechanism for parsing GFF records from any source implementing the
//! Read trait and converting them into Annotation objects. This lets us provide GFF files from
//! a variety of sources.

use std::{
    collections::HashMap,
    io::{BufRead, Read},
    sync::Arc,
};

use gen_core::{HashId, NodeIntervalBlock, Strand};
use gen_models::{
    ForeignKeyCache,
    accession::Accession,
    annotations::{Annotation, AnnotationExtra, GffAttribute, GffExtra},
    block_group::BlockGroup,
    interval_tree::IntervalTreeSource as _,
    reference_alias::ReferenceAlias,
    region::ResolvedGenRegion,
    sample::Sample,
};
use intervaltree::IntervalTree;
use noodles::gff;

use super::{AnnotationTranslationContext, FileAnnotationError};
use crate::{gff::gff_attribute_value_to_string, translate::compose_strand};

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
        cached_accession: ForeignKeyCache::default(),
        cached_interval_tree: None,
        extra: Some(AnnotationExtra {
            gff: Some(gff),
            ..AnnotationExtra::default()
        }),
    })
}

/// Translate matching GFF records and return an annotation with cached graph intervals and accession.
pub fn translate_gff_annotation<R>(
    context: &AnnotationTranslationContext<'_>,
    annotation: &Annotation,
    reader: R,
) -> Result<Annotation, FileAnnotationError>
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

/// Translate selected GFF records in supplied order, orienting each record's reverse-strand
/// fragments from its 5′ end, and return an annotation with cached graph intervals and accession.
pub fn translate_gff_annotation_records<I>(
    context: &AnnotationTranslationContext<'_>,
    annotation: &Annotation,
    records: I,
) -> Result<Annotation, FileAnnotationError>
where
    I: IntoIterator<Item = gff::feature::RecordBuf>,
{
    let mut translated_annotation = annotation.clone();
    let mut position = 0;
    let mut blocks = Vec::new();
    let sample_block_groups = Sample::get_block_groups(
        context.conn,
        context.collection_name,
        context.sample_name,
        context.history_ref,
    )
    .into_iter()
    .map(|block_group| (block_group.name, block_group.id))
    .collect::<HashMap<_, _>>();
    let references = sample_block_groups.keys().cloned().collect();
    let references_by_alias =
        ReferenceAlias::get_references_by_alias(context.conn, references, context.history_ref)
            .map_err(|error| FileAnnotationError::Translation(error.to_string()))?;
    let mut regions_by_reference = HashMap::<String, ResolvedGenRegion>::new();
    for record in records {
        let reference_name = record.reference_sequence_name().to_string();
        let start = record.start().get() as i64 - 1;
        let end = record.end().get() as i64;
        let feature_strand: Strand = record.strand().into();
        let resolved_reference = references_by_alias
            .get(&reference_name)
            .map_or(reference_name.as_str(), String::as_str);
        let Some(block_group_id) = sample_block_groups.get(resolved_reference) else {
            continue;
        };
        let mut region = match regions_by_reference.get(resolved_reference) {
            Some(region) => region.clone(),
            None => {
                let path =
                    BlockGroup::get_current_path(context.conn, block_group_id, context.history_ref)
                        .map_err(|error| FileAnnotationError::Translation(error.to_string()))?;
                let region = ResolvedGenRegion::from_path(context.conn, &path, 0, 0)
                    .map_err(|error| FileAnnotationError::Translation(error.to_string()))?;
                regions_by_reference.insert(resolved_reference.to_string(), region.clone());
                region
            }
        };
        region.start = start;
        region.end = end;
        let mut node_slices = region
            .graph_locus(context.conn, context.workspace)
            .map_err(|error| FileAnnotationError::Translation(error.to_string()))?
            .slices;
        if feature_strand == Strand::Reverse {
            node_slices.reverse();
        }
        for node_slice in node_slices {
            let sequence_start = node_slice.block.sequence_start + node_slice.start as i64;
            let sequence_end = node_slice.block.sequence_start + node_slice.end as i64;
            if sequence_end <= sequence_start {
                continue;
            }
            let block = NodeIntervalBlock {
                node_id: node_slice.block.node_id,
                start: position,
                end: position + sequence_end - sequence_start,
                sequence_start,
                sequence_end,
                strand: compose_strand(feature_strand, node_slice.strand),
            };
            position = block.end;
            blocks.push(block);
        }
    }
    if blocks.is_empty() {
        return Err(FileAnnotationError::Empty);
    }
    let interval_tree = Arc::new(
        blocks
            .into_iter()
            .map(|block| (block.start..block.end, block))
            .collect::<IntervalTree<_, _>>(),
    );
    translated_annotation.set_cached_interval_tree(Some(Arc::clone(&interval_tree)));
    let mut accession = Accession {
        id: translated_annotation.accession_id,
        name: translated_annotation.name.clone(),
        block_group_id: context.block_group_id,
        parent_accession_id: None,
        cached_interval_tree: None,
    };
    accession.set_cached_interval_tree(Some(interval_tree));
    translated_annotation.set_accession(Arc::new(accession));
    Ok(translated_annotation)
}

fn matching_gff_record(record: &gff::feature::RecordBuf, identifier: &str) -> bool {
    ["Name", "ID", "gene", "gene_name", "db_xref"]
        .iter()
        .any(|key| {
            gff_attribute_value_to_string(record.attributes(), key)
                .is_some_and(|value| value.eq_ignore_ascii_case(identifier))
        })
}

#[cfg(test)]
mod tests {
    use std::{
        fs::File,
        io::{BufReader, Cursor},
        sync::Arc,
    };

    use gen_core::{HashId, NodeIntervalBlock, Strand};
    use gen_graph::{GraphNode, GraphNodeSlice};
    use gen_models::{
        ForeignKeyCache,
        accession::Accession,
        annotations::Annotation,
        block_group::BlockGroup,
        history::dolt::{active_branch, commit_all},
        interval_tree::IntervalTreeSource as _,
        locus::GraphLocus,
        node::Node,
        region::ResolvedGenRegion,
        sample::Sample,
        sequence::Sequence,
    };
    use intervaltree::IntervalTree;

    use super::{AnnotationTranslationContext, parse_gff_annotation, translate_gff_annotation};
    use crate::{
        test_helpers::{create_forward_test_block_group, create_reverse_test_block_group},
        translate::gff::translate_gff,
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
        let annotation = parse_gff_annotation(
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
        let translated_annotation = translate_gff_annotation(
            &context,
            &annotation,
            BufReader::new(File::open(path).expect("should reopen simple GFF fixture")),
        )
        .expect("should translate the matching GFF record");
        assert!(annotation.cached_interval_tree().is_none());
        assert!(annotation.cached_accession.snapshot().is_none());
        let accession = translated_annotation
            .get_accession(&conn)
            .expect("should return translated GFF accession");
        assert_eq!(accession.id, translated_annotation.accession_id);
        let annotation_tree = translated_annotation
            .intervaltree(&conn)
            .expect("should get the cached GFF annotation tree");
        let accession_tree = accession
            .intervaltree(&conn)
            .expect("should get the cached GFF accession tree");
        assert!(Arc::ptr_eq(&annotation_tree, &accession_tree));
        assert!(Arc::ptr_eq(
            &annotation_tree,
            translated_annotation
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
        assert_eq!(
            translated_annotation
                .length(&conn)
                .expect("should get GFF length"),
            16
        );
        assert!(
            Accession::select(&conn)
                .id(accession.id)
                .load()
                .expect("should query absent file accession")
                .is_empty()
        );

        let mut accession_without_cache = accession.as_ref().clone();
        accession_without_cache.set_cached_interval_tree(None);
        let mut annotation_with_uncached_accession = translated_annotation.clone();
        annotation_with_uncached_accession.set_accession(Arc::new(accession_without_cache));
        let resolved =
            ResolvedGenRegion::from_annotation(&conn, &annotation_with_uncached_accession, 12, 15)
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
    ) -> (Annotation, HashId) {
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
            cached_accession: ForeignKeyCache::default(),
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
        annotation.set_accession(Arc::new(accession));

        (annotation, node_id)
    }

    #[test]
    fn test_reverse_graph_locus_fetches_reverse_complement_from_original_node_dna() {
        let conn = crate::test_helpers::get_connection();
        crate::test_helpers::setup_test_data(&conn);
        let (annotation, node_id) = create_cached_reverse_test_feature(&conn);
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
        let resolved = ResolvedGenRegion::from_annotation(&conn, &annotation, 0, 4)
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
        let (annotation, node_id) = create_cached_reverse_test_feature(&conn);
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
        let resolved = ResolvedGenRegion::from_annotation(&conn, &annotation, 0, 2)
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
        let annotation =
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

        let translated_annotation = translate_gff_annotation(
            &context,
            &annotation,
            BufReader::new(Cursor::new(input.as_bytes())),
        )
        .expect("should translate the gene_name-only GFF record");
        assert!(
            translated_annotation
                .length(&conn)
                .expect("should measure annotation")
                > 0
        );
        let accession = translated_annotation
            .get_accession(&conn)
            .expect("should get translated accession");
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
        let mut translated = Vec::new();
        translate_gff(
            &conn,
            crate::test_helpers::test_workspace(),
            "test",
            Sample::DEFAULT_NAME,
            None,
            Cursor::new(input.as_bytes()),
            &mut translated,
        )
        .expect("should translate GFF input");
        let translated = String::from_utf8(translated).expect("should write UTF-8 GFF output");
        assert_eq!(
            translated,
            format!(
                "{node_b_id}\tsource\tgene\t1\t1\t.\t-\t.\tID=cross-boundary;Name=cross-boundary\n{node_a_id}\tsource\tgene\t9\t9\t.\t-\t.\tID=cross-boundary;Name=cross-boundary\n"
            ),
            "positive-coordinate GFF fragments should clip to each reverse-oriented node in path order"
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
        let mut translated = Vec::new();
        translate_gff(
            &conn,
            crate::test_helpers::test_workspace(),
            "test",
            Sample::DEFAULT_NAME,
            None,
            Cursor::new(input.as_bytes()),
            &mut translated,
        )
        .expect("should translate GFF input");
        let translated = String::from_utf8(translated).expect("should write UTF-8 GFF output");
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
        let mut translated = Vec::new();
        translate_gff(
            &conn,
            crate::test_helpers::test_workspace(),
            "test",
            Sample::DEFAULT_NAME,
            None,
            Cursor::new(input.as_bytes()),
            &mut translated,
        )
        .expect("should translate GFF input");
        let translated = String::from_utf8(translated).expect("should write UTF-8 GFF output");
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
            let annotation = parse_gff_annotation(
                identifier.as_str(),
                BufReader::new(Cursor::new(input.as_bytes())),
            )
            .expect("should parse minus-strand GFF feature");
            let annotation = translate_gff_annotation(
                &context,
                &annotation,
                BufReader::new(Cursor::new(input.as_bytes())),
            )
            .expect("should translate minus-strand GFF feature");
            let resolved = ResolvedGenRegion::from_annotation(&conn, &annotation, 0, 9)
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
                let resolved = ResolvedGenRegion::from_annotation(&conn, &annotation, start, end)
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
                let resolved = ResolvedGenRegion::from_annotation(&conn, &annotation, start, end)
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
    fn test_multipart_minus_gff_cache_preserves_supplied_record_order() {
        let conn = crate::test_helpers::get_connection();
        crate::test_helpers::setup_test_data(&conn);
        let (block_group, node_a_id, node_b_id) = create_reverse_test_block_group(&conn);
        // The two records are supplied low then high on the reverse-traversed path.
        //   path index: 01234567890123456
        //   path nodes: BBBBBBBBAAAAAAAAA
        //   path DNA:   TGCAACGGTAACCGGTT
        //   GFF 3..4 - selects CA and yields feature DNA TG.
        //   GFF 11..12 - selects AC and yields feature DNA GT.
        // The adapter assembles those records in supplied order: TG + GT = TGGT.
        let input = format!(
            concat!(
                "{}\tsource\texon\t3\t4\t.\t-\t.\tID=low;Name=minus-multipart\n",
                "{}\tsource\texon\t11\t12\t.\t-\t.\tID=high;Name=minus-multipart\n"
            ),
            block_group.name, block_group.name
        );
        let mut translated = Vec::new();
        translate_gff(
            &conn,
            crate::test_helpers::test_workspace(),
            "test",
            Sample::DEFAULT_NAME,
            None,
            BufReader::new(Cursor::new(input.as_bytes())),
            &mut translated,
        )
        .expect("should translate multipart GFF records");
        assert_eq!(
            String::from_utf8(translated).expect("should write UTF-8 GFF output"),
            format!(
                "{node_b_id}\tsource\texon\t5\t6\t.\t+\t.\tID=low;Name=minus-multipart\n{node_a_id}\tsource\texon\t6\t7\t.\t+\t.\tID=high;Name=minus-multipart\n"
            ),
            "translated GFF fragments should stay in positive path order"
        );

        let context = AnnotationTranslationContext {
            conn: &conn,
            workspace: crate::test_helpers::test_workspace(),
            collection_name: "test",
            sample_name: Sample::DEFAULT_NAME,
            block_group_id: block_group.id,
            history_ref: None,
        };
        let annotation = parse_gff_annotation(
            "minus-multipart",
            BufReader::new(Cursor::new(input.as_bytes())),
        )
        .expect("should parse multipart minus-strand GFF records");
        let annotation = translate_gff_annotation(
            &context,
            &annotation,
            BufReader::new(Cursor::new(input.as_bytes())),
        )
        .expect("should cache multipart minus-strand GFF records");
        let tree = annotation
            .intervaltree(&conn)
            .expect("should read multipart GFF cache");
        let mut spans = tree
            .iter()
            .map(|entry| (entry.range.clone(), entry.value))
            .collect::<Vec<_>>();
        spans.sort_by_key(|(range, _)| range.start);
        assert_eq!(
            spans
                .iter()
                .map(|(range, block)| (
                    range.clone(),
                    block.node_id,
                    block.sequence_start,
                    block.sequence_end,
                    block.strand,
                ))
                .collect::<Vec<_>>(),
            [
                (0..2, node_b_id, 4, 6, Strand::Forward),
                (2..4, node_a_id, 5, 7, Strand::Forward),
            ],
            "minus-strand fragments should preserve supplied record order with positive stored coordinates"
        );
        let resolved = ResolvedGenRegion::from_annotation(&conn, &annotation, 0, 4)
            .expect("should resolve multipart minus-strand GFF annotation");
        let graph_locus = resolved
            .graph_locus(&conn, crate::test_helpers::test_workspace())
            .expect("should build multipart minus-strand GFF graph locus");
        assert_eq!(
            graph_locus.sequence(&conn, crate::test_helpers::test_workspace()),
            b"TGGT",
            "multipart minus-strand GFF cache should assemble records in supplied order"
        );
    }

    #[test]
    fn test_multipart_gff_cache_preserves_input_record_order() {
        let conn = crate::test_helpers::get_connection();
        crate::test_helpers::setup_test_data(&conn);

        // Each input has three records in ascending or descending coordinate order. The adapter
        // preserves that order and reverses fragments only within an individual minus record.
        // Stored node B:
        //   index:   012345678
        //   DNA:     CCGTTGCAT
        //   used:    ^-------^ [0,8)
        // Stored node A:
        //   index:   012345678
        //   DNA:     AACCGGTTA
        //   used:    ^--------^ [0,9)
        // Edge endpoint strands at B/A +:
        //   offset:  01234567890123456
        //   nodes:   BBBBBBBBAAAAAAAAA
        //   path:    CCGTTGCAAACCGGTTA
        //   parts:   ..LL.MM...HH.....
        //   path:    L [2,4), M [5,7), H [10,12)
        //   GFF:     L 3..4=GT, M 6..7=GC, H 11..12=CC (inclusive coordinates)
        //   feature +: L+M+H = GTGCCC; H+M+L = CCGCGT.
        //   feature -: L+M+H = ACGCGG; H+M+L = GGGCAC.
        // Edge endpoint strands at B/A -:
        //   offset:  01234567890123456
        //   nodes:   BBBBBBBBAAAAAAAAA
        //   path:    TGCAACGGTAACCGGTT
        //   parts:   ..LL.MM...HH.....
        //   path:    L [2,4), M [5,7), H [10,12)
        //   GFF:     L 3..4=CA, M 6..7=CG, H 11..12=AC (inclusive coordinates)
        //   feature +: L+M+H = CACGAC; H+M+L = ACCGCA.
        //   feature -: L+M+H = TGCGGT; H+M+L = GTCGTG.
        let forward = create_forward_test_block_group(&conn);
        let reverse = create_reverse_test_block_group(&conn);
        for (name, feature_strand, ascending_sequence, descending_sequence, fixture) in [
            (
                "forward-plus",
                "+",
                b"GTGCCC".as_slice(),
                b"CCGCGT".as_slice(),
                &forward,
            ),
            (
                "forward-minus",
                "-",
                b"ACGCGG".as_slice(),
                b"GGGCAC".as_slice(),
                &forward,
            ),
            (
                "reverse-plus",
                "+",
                b"CACGAC".as_slice(),
                b"ACCGCA".as_slice(),
                &reverse,
            ),
            (
                "reverse-minus",
                "-",
                b"TGCGGT".as_slice(),
                b"GTCGTG".as_slice(),
                &reverse,
            ),
        ] {
            let (block_group, node_a_id, node_b_id) = fixture;
            let identifier = format!("multipart-{name}");
            let parts = [("low", 3, 4), ("middle", 6, 7), ("high", 11, 12)];

            for (record_order, part_indices) in
                [("ascending", [0, 1, 2]), ("descending", [2, 1, 0])]
            {
                let expected_sequence = if record_order == "ascending" {
                    ascending_sequence
                } else {
                    descending_sequence
                };
                let input = part_indices
                    .into_iter()
                    .map(|index| {
                        let (part, start, end) = parts[index];
                        format!(
                            "{}\tsource\texon\t{start}\t{end}\t.\t{feature_strand}\t.\tID={part};Name={identifier}\n",
                            block_group.name
                        )
                    })
                    .collect::<String>();
                let context = AnnotationTranslationContext {
                    conn: &conn,
                    workspace: crate::test_helpers::test_workspace(),
                    collection_name: "test",
                    sample_name: Sample::DEFAULT_NAME,
                    block_group_id: block_group.id,
                    history_ref: None,
                };
                let annotation = parse_gff_annotation(
                    identifier.as_str(),
                    BufReader::new(Cursor::new(input.as_bytes())),
                )
                .expect("should parse multipart GFF feature");
                let annotation = translate_gff_annotation(
                    &context,
                    &annotation,
                    BufReader::new(Cursor::new(input.as_bytes())),
                )
                .expect("should translate multipart GFF feature");
                let resolved = ResolvedGenRegion::from_annotation(&conn, &annotation, 0, 6)
                    .expect("should resolve multipart GFF feature");
                assert_eq!(
                    resolved.feature_length, 6,
                    "should measure all three GFF parts"
                );
                let graph_locus = resolved
                    .graph_locus(&conn, crate::test_helpers::test_workspace())
                    .expect("should build multipart GFF graph locus");
                assert_eq!(
                    graph_locus.sequence(&conn, crate::test_helpers::test_workspace()),
                    expected_sequence,
                    "{record_order} GFF records should preserve feature DNA for {name}"
                );
                let expected_node_order = if record_order == "ascending" {
                    [*node_b_id, *node_b_id, *node_a_id]
                } else {
                    [*node_a_id, *node_b_id, *node_b_id]
                };
                assert_eq!(
                    graph_locus
                        .slices
                        .iter()
                        .map(|slice| slice.block.node_id)
                        .collect::<Vec<_>>(),
                    expected_node_order,
                    "{record_order} GFF records should keep the parts in feature order for {name}"
                );

                // This independent query trims the first and last bases across the three parts.
                //   offset   012345
                //   [1,5)    .^^^^.
                let resolved = ResolvedGenRegion::from_annotation(&conn, &annotation, 1, 5)
                    .expect("should resolve trimmed multipart GFF feature");
                let graph_locus = resolved
                    .graph_locus(&conn, crate::test_helpers::test_workspace())
                    .expect("should build trimmed multipart GFF graph locus");
                assert_eq!(
                    graph_locus.sequence(&conn, crate::test_helpers::test_workspace()),
                    &expected_sequence[1..5],
                    "{record_order} GFF records should trim across feature parts for {name}"
                );
            }
        }
    }

    #[test]
    fn test_mixed_strand_gff_cache_preserves_record_order_and_local_orientation() {
        let conn = crate::test_helpers::get_connection();
        crate::test_helpers::setup_test_data(&conn);
        let forward = create_forward_test_block_group(&conn);
        let reverse = create_reverse_test_block_group(&conn);

        // Stored node B:
        //   index: 012345678
        //   DNA:   CCGTTGCAT
        //   slice: ^-------^ [0,8)
        // Stored node A:
        //   index: 012345678
        //   DNA:   AACCGGTTA
        //   slice: ^--------^ [0,9)
        // Edge endpoints at B/A +:
        //   path index: 01234567890123456
        //   path nodes: BBBBBBBBAAAAAAAAA
        //   path DNA:   CCGTTGCAAACCGGTTA
        //   GFF 4..12 maps to [3,12)=TTGCAAACC; GFF 15..16 maps to [14,16)=TT.
        //   parts:      ...CCCCCCCCC..TT.
        //   case -,+:   ...---------..++.
        //   case +,-:   ...+++++++++..--.
        //   4..12 - then 15..16 + => GGTTTGCAATT; reverse record order => TTGGTTTGCAA.
        //   4..12 + then 15..16 - => TTGCAAACCAA; reverse record order => AATTGCAAACC.
        //   Inclusive GFF spans 1..2, 4..12, 15..16 map to [0,2), [3,12), [14,16); middle crosses B/A.
        //   parts:      HH.MMMMMMMMM..TT.
        //   case +,-,+: ++.---------..++.
        //   case -,+,-: --.+++++++++..--.
        //   +,-,+ => CCGGTTTGCAATT; -,+,- => GGTTGCAAACCAA.
        // Edge endpoints at B/A -:
        //   path index: 01234567890123456
        //   path nodes: BBBBBBBBAAAAAAAAA
        //   path DNA:   TGCAACGGTAACCGGTT
        //   GFF 4..12 maps to [3,12)=AACGGTAAC; GFF 15..16 maps to [14,16)=GT.
        //   parts:      ...CCCCCCCCC..TT.
        //   case -,+:   ...---------..++.
        //   case +,-:   ...+++++++++..--.
        //   4..12 - then 15..16 + => GTTACCGTTGT; reverse record order => GTGTTACCGTT.
        //   4..12 + then 15..16 - => AACGGTAACAC; reverse record order => ACAACGGTAAC.
        //   GFF 1..2 maps to [0,2)=TG, then 4..12 - then 15..16 +; middle crosses B/A.
        //   parts:      HH.MMMMMMMMM..TT.
        //   case +,-,+: ++.---------..++.
        //   case -,+,-: --.+++++++++..--.
        //   +,-,+ => TGGTTACCGTTGT; -,+,- => CAAACGGTAACAC.
        // These are independent source records. Each record's strand orders only its own
        // projected fragments; the two-record query [1,10) and three-record query [1,12)
        // below independently trim the assembled feature.
        type ExpectedBlock = (HashId, i64, i64, Strand);
        type SourceRecord = (&'static str, usize, usize);
        type MixedCase = (
            &'static str,
            Vec<SourceRecord>,
            &'static [u8],
            Vec<ExpectedBlock>,
        );
        for ((block_group, node_a_id, node_b_id), path_name) in
            [(forward, "forward"), (reverse, "reverse")]
        {
            let cases: Vec<MixedCase> = if path_name == "forward" {
                vec![
                    (
                        "cross-minus-then-tail-plus",
                        vec![("-", 4, 12), ("+", 15, 16)],
                        b"GGTTTGCAATT",
                        vec![
                            (node_a_id, 0, 4, Strand::Reverse),
                            (node_b_id, 3, 8, Strand::Reverse),
                            (node_a_id, 6, 8, Strand::Forward),
                        ],
                    ),
                    (
                        "cross-plus-then-tail-minus",
                        vec![("+", 4, 12), ("-", 15, 16)],
                        b"TTGCAAACCAA",
                        vec![
                            (node_b_id, 3, 8, Strand::Forward),
                            (node_a_id, 0, 4, Strand::Forward),
                            (node_a_id, 6, 8, Strand::Reverse),
                        ],
                    ),
                    (
                        "tail-plus-then-cross-minus",
                        vec![("+", 15, 16), ("-", 4, 12)],
                        b"TTGGTTTGCAA",
                        vec![
                            (node_a_id, 6, 8, Strand::Forward),
                            (node_a_id, 0, 4, Strand::Reverse),
                            (node_b_id, 3, 8, Strand::Reverse),
                        ],
                    ),
                    (
                        "tail-minus-then-cross-plus",
                        vec![("-", 15, 16), ("+", 4, 12)],
                        b"AATTGCAAACC",
                        vec![
                            (node_a_id, 6, 8, Strand::Reverse),
                            (node_b_id, 3, 8, Strand::Forward),
                            (node_a_id, 0, 4, Strand::Forward),
                        ],
                    ),
                    (
                        "alternating-plus-minus-plus",
                        vec![("+", 1, 2), ("-", 4, 12), ("+", 15, 16)],
                        b"CCGGTTTGCAATT",
                        vec![
                            (node_b_id, 0, 2, Strand::Forward),
                            (node_a_id, 0, 4, Strand::Reverse),
                            (node_b_id, 3, 8, Strand::Reverse),
                            (node_a_id, 6, 8, Strand::Forward),
                        ],
                    ),
                    (
                        "alternating-minus-plus-minus",
                        vec![("-", 1, 2), ("+", 4, 12), ("-", 15, 16)],
                        b"GGTTGCAAACCAA",
                        vec![
                            (node_b_id, 0, 2, Strand::Reverse),
                            (node_b_id, 3, 8, Strand::Forward),
                            (node_a_id, 0, 4, Strand::Forward),
                            (node_a_id, 6, 8, Strand::Reverse),
                        ],
                    ),
                ]
            } else {
                vec![
                    (
                        "cross-minus-then-tail-plus",
                        vec![("-", 4, 12), ("+", 15, 16)],
                        b"GTTACCGTTGT",
                        vec![
                            (node_a_id, 5, 9, Strand::Forward),
                            (node_b_id, 0, 5, Strand::Forward),
                            (node_a_id, 1, 3, Strand::Reverse),
                        ],
                    ),
                    (
                        "cross-plus-then-tail-minus",
                        vec![("+", 4, 12), ("-", 15, 16)],
                        b"AACGGTAACAC",
                        vec![
                            (node_b_id, 0, 5, Strand::Reverse),
                            (node_a_id, 5, 9, Strand::Reverse),
                            (node_a_id, 1, 3, Strand::Forward),
                        ],
                    ),
                    (
                        "tail-plus-then-cross-minus",
                        vec![("+", 15, 16), ("-", 4, 12)],
                        b"GTGTTACCGTT",
                        vec![
                            (node_a_id, 1, 3, Strand::Reverse),
                            (node_a_id, 5, 9, Strand::Forward),
                            (node_b_id, 0, 5, Strand::Forward),
                        ],
                    ),
                    (
                        "tail-minus-then-cross-plus",
                        vec![("-", 15, 16), ("+", 4, 12)],
                        b"ACAACGGTAAC",
                        vec![
                            (node_a_id, 1, 3, Strand::Forward),
                            (node_b_id, 0, 5, Strand::Reverse),
                            (node_a_id, 5, 9, Strand::Reverse),
                        ],
                    ),
                    (
                        "alternating-plus-minus-plus",
                        vec![("+", 1, 2), ("-", 4, 12), ("+", 15, 16)],
                        b"TGGTTACCGTTGT",
                        vec![
                            (node_b_id, 6, 8, Strand::Reverse),
                            (node_a_id, 5, 9, Strand::Forward),
                            (node_b_id, 0, 5, Strand::Forward),
                            (node_a_id, 1, 3, Strand::Reverse),
                        ],
                    ),
                    (
                        "alternating-minus-plus-minus",
                        vec![("-", 1, 2), ("+", 4, 12), ("-", 15, 16)],
                        b"CAAACGGTAACAC",
                        vec![
                            (node_b_id, 6, 8, Strand::Forward),
                            (node_b_id, 0, 5, Strand::Reverse),
                            (node_a_id, 5, 9, Strand::Reverse),
                            (node_a_id, 1, 3, Strand::Forward),
                        ],
                    ),
                ]
            };

            for (case_name, records, expected_sequence, expected_blocks) in cases {
                let identifier = format!("mixed-{path_name}-{case_name}");
                let input = records
                    .into_iter()
                    .enumerate()
                    .map(|(index, (strand, start, end))| {
                        format!(
                            "{}\tsource\texon\t{start}\t{end}\t.\t{strand}\t.\tID=part-{index};Name={identifier}\n",
                            block_group.name
                        )
                    })
                    .collect::<String>();
                let context = AnnotationTranslationContext {
                    conn: &conn,
                    workspace: crate::test_helpers::test_workspace(),
                    collection_name: "test",
                    sample_name: Sample::DEFAULT_NAME,
                    block_group_id: block_group.id,
                    history_ref: None,
                };
                let annotation = parse_gff_annotation(
                    identifier.as_str(),
                    BufReader::new(Cursor::new(input.as_bytes())),
                )
                .expect("should parse mixed-strand GFF records");
                let annotation = translate_gff_annotation(
                    &context,
                    &annotation,
                    BufReader::new(Cursor::new(input.as_bytes())),
                )
                .expect("should translate mixed-strand GFF records");
                let tree = annotation
                    .intervaltree(&conn)
                    .expect("should read mixed-strand GFF intervals");
                let mut actual_blocks = tree
                    .iter()
                    .map(|entry| {
                        (
                            entry.range.start,
                            entry.range.end,
                            entry.value.node_id,
                            entry.value.sequence_start,
                            entry.value.sequence_end,
                            entry.value.strand,
                        )
                    })
                    .collect::<Vec<_>>();
                actual_blocks.sort_by_key(|block| block.0);
                let mut feature_start = 0;
                let expected_blocks = expected_blocks
                    .into_iter()
                    .map(|(node_id, sequence_start, sequence_end, strand)| {
                        let feature_end = feature_start + sequence_end - sequence_start;
                        let block = (
                            feature_start,
                            feature_end,
                            node_id,
                            sequence_start,
                            sequence_end,
                            strand,
                        );
                        feature_start = feature_end;
                        block
                    })
                    .collect::<Vec<_>>();
                assert_eq!(
                    actual_blocks, expected_blocks,
                    "{case_name} GFF records should retain exact node coordinates and strands"
                );

                let feature_end = expected_sequence.len() as i64;
                let resolved =
                    ResolvedGenRegion::from_annotation(&conn, &annotation, 0, feature_end)
                        .expect("should resolve mixed-strand GFF feature");
                assert_eq!(
                    resolved.feature_length, feature_end,
                    "should measure all supplied GFF records"
                );
                let graph_locus = resolved
                    .graph_locus(&conn, crate::test_helpers::test_workspace())
                    .expect("should build mixed-strand GFF graph locus");
                assert_eq!(
                    graph_locus.sequence(&conn, crate::test_helpers::test_workspace()),
                    expected_sequence,
                    "{case_name} GFF records should retrieve their supplied feature sequence"
                );
                let trim_end = feature_end - 1;
                let resolved = ResolvedGenRegion::from_annotation(&conn, &annotation, 1, trim_end)
                    .expect("should resolve an interior mixed-strand GFF range");
                let graph_locus = resolved
                    .graph_locus(&conn, crate::test_helpers::test_workspace())
                    .expect("should build the trimmed mixed-strand GFF graph locus");
                assert_eq!(
                    graph_locus.sequence(&conn, crate::test_helpers::test_workspace()),
                    &expected_sequence[1..trim_end as usize],
                    "{case_name} GFF range should trim across the source-record and node boundaries"
                );
            }
        }
    }

    #[test]
    fn test_gff_annotation_uses_historical_path_after_current_path_deletion() {
        let conn = crate::test_helpers::get_connection();
        crate::test_helpers::setup_test_data(&conn);
        let block_group = Sample::get_block_groups(&conn, "test", Sample::DEFAULT_NAME, None)
            .into_iter()
            .find(|block_group| block_group.name == "m123")
            .expect("should find the test block group");
        let initial_branch = active_branch(&conn).expect("should read the active branch");
        let history_ref = commit_all(&conn, "store GFF annotation path")
            .expect("should commit the GFF annotation path")
            .to_string();

        // GFF interval 6..8 is the negative-strand feature on the first of two path nodes.
        //   tens:  00000000001111111
        //   ones:  01234567890123456
        //   node1: ATCGATCGATCGATCGA [0,17)
        //   edge:  +++++++++++++++++
        //   GFF:   .....^^^......... 6..8, raw TCG fetches as CGA.
        // The following node covers path [17,34).
        // The historical path must supply this interval after the current row is deleted.
        let input = "m123\tsource\tgene\t6\t8\t.\t-\t.\tID=history-feature\n";
        let annotation = parse_gff_annotation(
            "history-feature",
            BufReader::new(Cursor::new(input.as_bytes())),
        )
        .expect("should parse the negative-strand GFF feature");

        conn.execute(
            "DELETE FROM paths WHERE block_group_id = ?1",
            [block_group.id],
        )
        .expect("should delete the current path row");
        assert!(
            BlockGroup::get_current_path(&conn, &block_group.id, None).is_err(),
            "should not find a current path after deletion"
        );
        let historical_block_group =
            BlockGroup::get_by_id(&conn, &block_group.id, Some(&history_ref))
                .expect("should load the block group from history");
        assert_eq!(historical_block_group, block_group);

        let context = AnnotationTranslationContext {
            conn: &conn,
            workspace: crate::test_helpers::test_workspace(),
            collection_name: "test",
            sample_name: Sample::DEFAULT_NAME,
            block_group_id: historical_block_group.id,
            history_ref: Some(&history_ref),
        };
        let translated = translate_gff_annotation(
            &context,
            &annotation,
            BufReader::new(Cursor::new(input.as_bytes())),
        )
        .expect("should translate the GFF feature from its historical path");
        let tree = translated
            .intervaltree(&conn)
            .expect("should read the translated GFF intervals");
        let intervals: Vec<_> = tree.iter().collect();
        assert_eq!(intervals.len(), 1, "should emit one historical interval");
        assert_eq!(intervals[0].range, 0..3);
        assert_eq!(
            intervals[0].value.node_id,
            HashId::try_from("b59698a422128d20462c44537b2d23ef")
                .expect("should parse the historical node id")
        );
        assert_eq!(intervals[0].value.sequence_start, 5);
        assert_eq!(intervals[0].value.sequence_end, 8);
        assert_eq!(
            intervals[0].value.strand,
            Strand::Reverse,
            "should preserve the feature's reverse strand"
        );

        let resolved = ResolvedGenRegion::from_annotation(&conn, &translated, 0, 3)
            .expect("should resolve the historical GFF feature");
        let locus = resolved
            .graph_locus(&conn, crate::test_helpers::test_workspace())
            .expect("should build the historical GFF graph locus");
        assert_eq!(
            locus.sequence(&conn, crate::test_helpers::test_workspace()),
            b"CGA",
            "should fetch the complete negative-strand GFF feature"
        );
        assert!(
            BlockGroup::get_current_path(&conn, &block_group.id, None).is_err(),
            "historical translation should leave the current path deleted"
        );
        assert_eq!(
            active_branch(&conn).expect("should read the active branch"),
            initial_branch,
            "historical translation should not switch the active branch"
        );
    }
}
