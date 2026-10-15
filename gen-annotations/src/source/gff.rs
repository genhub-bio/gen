//! GFF parsing and translation adapters.
//!
//! This serves as a general mechanism for parsing GFF records from any source implementing the
//! Read trait and converting them into Annotation objects. This lets us provide GFF files from
//! a variety of sources.

use std::io::{BufRead, Read};

use gen_core::HashId;
use gen_models::{
    ForeignKeyCache,
    annotations::{Annotation, AnnotationExtra, AnnotationLocation, GffAttribute, GffExtra},
};
use noodles::gff;

use super::{AnnotationTranslationContext, FileAnnotationError};
use crate::{gff::gff_attribute_value_to_string, translate::translate_annotation};

/// Parse matching GFF records into an annotation with metadata and source coordinates.
///
/// The reader is consumed once, and the returned annotation can be translated after it is dropped.
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

/// Translate matching GFF records into an annotation with attached graph interval caches.
pub fn translate_gff_annotation<R>(
    context: &AnnotationTranslationContext<'_>,
    identifier: impl Into<String>,
    reader: R,
) -> Result<Annotation, FileAnnotationError>
where
    R: Read + BufRead,
{
    let annotation = parse_gff_annotation(identifier, reader)?;
    translate_annotation(context, &annotation)
}

/// Parse preselected GFF records into an annotation with metadata and source coordinates.
pub fn parse_gff_annotation_records<I>(
    identifier: impl Into<String>,
    records: I,
) -> Result<Annotation, FileAnnotationError>
where
    I: IntoIterator<Item = gff::feature::RecordBuf>,
{
    let identifier = identifier.into();
    let records = records.into_iter().collect::<Vec<_>>();
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
    let source_locations = records
        .iter()
        .map(|record| AnnotationLocation {
            reference_sequence_name: record.reference_sequence_name().to_string(),
            start: record.start().get() as i64 - 1,
            end: record.end().get() as i64,
            strand: record.strand().into(),
        })
        .collect();
    let id = HashId::convert_str(&identifier);
    Ok(Annotation {
        id,
        name: identifier,
        group: "gff".to_string(),
        accession_id: id,
        cached_accession: ForeignKeyCache::default(),
        cached_interval_tree: None,
        history_ref: None,
        source_locations,
        extra: Some(AnnotationExtra {
            gff: Some(gff),
            ..AnnotationExtra::default()
        }),
    })
}

/// Parse preselected GFF records and translate their source coordinates onto the sample graph.
pub fn translate_gff_annotation_records<I>(
    context: &AnnotationTranslationContext<'_>,
    identifier: impl Into<String>,
    records: I,
) -> Result<Annotation, FileAnnotationError>
where
    I: IntoIterator<Item = gff::feature::RecordBuf>,
{
    let annotation = parse_gff_annotation_records(identifier, records)?;
    translate_annotation(context, &annotation)
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
        io::{BufReader, Cursor},
        sync::Arc,
    };

    use gen_core::{HashId, Strand};
    use gen_models::{
        accession::Accession, annotations::AnnotationLocation, db::GraphConnection,
        interval_tree::IntervalTreeSource as _, region::ResolvedGenRegion, sample::Sample,
    };
    use noodles::gff;

    use super::{
        AnnotationTranslationContext, FileAnnotationError, parse_gff_annotation,
        parse_gff_annotation_records, translate_gff_annotation, translate_gff_annotation_records,
    };
    use crate::{
        test_helpers::create_reverse_test_block_group,
        translate::{gff::translate_gff, translate_annotation},
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
        let input = std::fs::read(path).expect("should read simple GFF fixture");
        // GFF 5..20 (one-based inclusive) is retained as [4,20):
        //   source offset: 01234567890123456789
        //   feature:       ....^^^^^^^^^^^^^^^^ [4,20), strand +
        let annotation =
            translate_gff_annotation(&context, "GENE-A0001", BufReader::new(input.as_slice()))
                .expect("should translate the matching GFF record from one reader");
        assert!(matches!(
            translate_gff_annotation(&context, "missing", BufReader::new(input.as_slice())),
            Err(FileAnnotationError::Empty)
        ));
        assert!(matches!(
            translate_gff_annotation(&context, "missing", BufReader::new(&b""[..])),
            Err(FileAnnotationError::Empty)
        ));
        assert!(matches!(
            translate_gff_annotation(
                &context,
                "malformed",
                BufReader::new(&b"m123\tsource\tgene\tbad\t6\t.\t+\t.\tID=bad\n"[..]),
            ),
            Err(FileAnnotationError::Io(_))
        ));
        assert_eq!(annotation.name, "GENE-A0001");
        assert_eq!(annotation.id, HashId::convert_str("GENE-A0001"));
        assert_eq!(annotation.accession_id, annotation.id);
        assert_eq!(annotation.group, "gff");
        assert_eq!(
            annotation.source_locations,
            vec![AnnotationLocation {
                reference_sequence_name: "m123".to_string(),
                start: 4,
                end: 20,
                strand: gen_core::Strand::Forward,
            }]
        );
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
        let empty_conn = GraphConnection(
            rusqlite::Connection::open_in_memory().expect("should open empty database"),
        );
        let accession = annotation
            .get_accession(&empty_conn)
            .expect("should return translated GFF accession");
        assert_eq!(accession.id, annotation.accession_id);
        let annotation_tree = annotation
            .intervaltree(&empty_conn)
            .expect("should get the cached GFF annotation tree without database queries");
        let accession_tree = accession
            .intervaltree(&empty_conn)
            .expect("should get the cached GFF accession tree without database queries");
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
        assert_eq!(
            annotation
                .length(&empty_conn)
                .expect("should get GFF length"),
            16
        );
        assert_eq!(
            accession
                .length(&empty_conn)
                .expect("should get GFF accession length"),
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
        let mut annotation_with_uncached_accession = annotation.clone();
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
    }

    #[test]
    fn test_gff_annotation_records_preserve_metadata_across_matching_rows() {
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
        let input = concat!(
            "##gff-version 3\n",
            "m123\tsource\tgene\t5\t6\t.\t+\t.\tID=private-gene-name-id;gene_name=BRCA1\n",
            "m123\tsource\tmRNA\t7\t9\t.\t-\t.\tID=transcript-id;Name=BRCA1;note=second-matching-row\n",
        );
        // GFF rows are converted to zero-based half-open offsets in supplied order:
        //   source offset: 012345678
        //   row A (+):     ....^^... [4,6)
        //   row B (-):     ......^^^ [6,9)
        let parsed_reader_annotation =
            parse_gff_annotation("brca1", BufReader::new(input.as_bytes()))
                .expect("should parse matching gene_name records ignoring case");
        assert_eq!(
            parsed_reader_annotation.source_locations,
            vec![
                AnnotationLocation {
                    reference_sequence_name: "m123".to_string(),
                    start: 4,
                    end: 6,
                    strand: Strand::Forward,
                },
                AnnotationLocation {
                    reference_sequence_name: "m123".to_string(),
                    start: 6,
                    end: 9,
                    strand: Strand::Reverse,
                },
            ]
        );
        let reader_annotation = translate_annotation(&context, &parsed_reader_annotation)
            .expect("should translate parsed GFF after the reader is dropped");
        let reader_metadata = reader_annotation
            .extra
            .as_ref()
            .and_then(|extra| extra.gff.as_ref())
            .expect("should retain metadata from all matching GFF rows");
        let records = gff::io::Reader::new(Cursor::new(input.as_bytes()))
            .record_bufs()
            .collect::<Result<Vec<_>, _>>()
            .expect("should parse GFF records");
        let parsed_records = parse_gff_annotation_records("selected-name", records.clone())
            .expect("should parse preselected one-shot GFF records without graph context");
        assert_eq!(parsed_records.name, "selected-name");
        assert_eq!(
            parsed_records.source_locations,
            parsed_reader_annotation.source_locations
        );
        let annotation = translate_gff_annotation_records(&context, "brca1", records)
            .expect("should translate one-shot matching GFF records ignoring case");
        assert_eq!(annotation.name, "brca1");
        let metadata = annotation
            .extra
            .as_ref()
            .and_then(|extra| extra.gff.as_ref())
            .expect("should preserve the gene_name-only GFF metadata");
        assert_eq!(reader_metadata, metadata);
        assert!(
            metadata
                .attributes
                .iter()
                .any(|attribute| { attribute.key == "gene_name" && attribute.values == ["BRCA1"] })
        );
        assert!(metadata.attributes.iter().any(|attribute| {
            attribute.key == "note" && attribute.values == ["second-matching-row"]
        }));
        let empty_conn = GraphConnection(
            rusqlite::Connection::open_in_memory().expect("should open empty database"),
        );
        let reader_accession = reader_annotation
            .get_accession(&empty_conn)
            .expect("should get the accession from general GFF translation");
        let reader_tree = reader_annotation
            .intervaltree(&empty_conn)
            .expect("should get the tree from general GFF translation");
        let reader_accession_tree = reader_accession
            .intervaltree(&empty_conn)
            .expect("should get the accession tree from general GFF translation");
        assert!(Arc::ptr_eq(&reader_tree, &reader_accession_tree));
        assert_eq!(
            reader_annotation
                .length(&empty_conn)
                .expect("should measure general GFF translation"),
            5
        );
        let accession = annotation
            .get_accession(&empty_conn)
            .expect("should get translated accession");
        let annotation_tree = annotation
            .intervaltree(&empty_conn)
            .expect("should get translated interval tree without database queries");
        let accession_tree = accession
            .intervaltree(&empty_conn)
            .expect("should get accession interval tree without database queries");
        assert!(Arc::ptr_eq(&annotation_tree, &accession_tree));
        let mut blocks = annotation_tree
            .iter()
            .map(|interval| {
                (
                    interval.value.start,
                    interval.value.end,
                    interval.value.sequence_start,
                    interval.value.sequence_end,
                    interval.value.strand,
                )
            })
            .collect::<Vec<_>>();
        blocks.sort_by_key(|block| block.0);
        assert_eq!(
            blocks,
            vec![(0, 2, 4, 6, Strand::Forward), (2, 5, 6, 9, Strand::Reverse),]
        );
        assert!(
            annotation
                .length(&empty_conn)
                .expect("should measure annotation")
                > 0
        );
        assert!(
            accession
                .length(&empty_conn)
                .expect("should measure accession")
                > 0
        );
    }

    #[test]
    fn test_translates_reverse_split_gff_to_exact_node_local_spans() {
        let conn = crate::test_helpers::get_connection();
        crate::test_helpers::setup_test_data(&conn);
        let (block_group, node_a_id, node_b_id) = create_reverse_test_block_group(&conn);
        // Path edges traverse B then A with negative edge strands; input feature strand is +.
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
                "{node_b_id}\tsource\tgene\t1\t1\t.\t+\t.\tID=cross-boundary;Name=cross-boundary\n{node_a_id}\tsource\tgene\t9\t9\t.\t+\t.\tID=cross-boundary;Name=cross-boundary\n"
            ),
            "GFF fragments should map to stored-node spans in path order and preserve the input feature strand"
        );
    }

    #[test]
    fn test_translates_single_base_gff_before_node_boundary() {
        let conn = crate::test_helpers::get_connection();
        crate::test_helpers::setup_test_data(&conn);
        let (block_group, _, node_b_id) = create_reverse_test_block_group(&conn);
        // The reverse edge maps path position 7 to stored B [0,1); feature strand stays +.
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
                "{node_b_id}\tsource\tgene\t1\t1\t.\t+\t.\tID=before-boundary;Name=before-boundary\n"
            ),
            "GFF coordinate 8 should map to the final base of the first path node"
        );
    }

    #[test]
    fn test_translates_single_base_gff_after_node_boundary() {
        let conn = crate::test_helpers::get_connection();
        crate::test_helpers::setup_test_data(&conn);
        let (block_group, node_a_id, _) = create_reverse_test_block_group(&conn);
        // The reverse edge maps path position 8 to stored A [8,9); feature strand stays +.
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
                "{node_a_id}\tsource\tgene\t9\t9\t.\t+\t.\tID=after-boundary;Name=after-boundary\n"
            ),
            "GFF coordinate 9 should map to the first base of the second path node"
        );
    }
}
