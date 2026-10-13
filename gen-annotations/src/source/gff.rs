//! GFF parsing and translation adapters.
//!
//! This serves as a general mechanism for parsing GFF records from any source implementing the
//! Read trait and converting them into Annotation objects. This lets us provide GFF files from
//! a variety of sources.

use std::{
    io::{BufRead, Cursor, Read},
    sync::Arc,
};

use gen_core::{HashId, NodeIntervalBlock};
use gen_models::{
    ForeignKeyCache,
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

/// Translate selected GFF records and return an annotation with cached graph intervals and accession.
pub fn translate_gff_annotation_records<I>(
    context: &AnnotationTranslationContext<'_>,
    annotation: &Annotation,
    records: I,
) -> Result<Annotation, FileAnnotationError>
where
    I: IntoIterator<Item = gff::feature::RecordBuf>,
{
    let mut translated_annotation = annotation.clone();
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

    use gen_core::HashId;
    use gen_models::{
        accession::Accession, interval_tree::IntervalTreeSource as _, region::ResolvedGenRegion,
        sample::Sample,
    };

    use super::{AnnotationTranslationContext, parse_gff_annotation, translate_gff_annotation};
    use crate::{test_helpers::create_reverse_test_block_group, translate::gff::translate_gff};

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
}
