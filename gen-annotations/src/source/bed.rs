//! BED parsing and translation adapters.
//!
//! This serves as a general mechanism for parsing bed records from any source implementing the
//! Read trait and converting them into Annotation objects. This lets us provide BED files from
//! a variety of sources.

use std::io::Read;

use gen_core::{HashId, Strand};
use gen_models::{
    ForeignKeyCache,
    annotations::{Annotation, AnnotationExtra, AnnotationLocation, BedExtra},
};
use noodles::bed;

use super::{AnnotationTranslationContext, FileAnnotationError};
use crate::translate::translate_annotation;

/// Parse matching BED records into an annotation with metadata and source coordinates.
///
/// The reader is consumed once, and the returned annotation can be translated after it is dropped.
pub fn parse_bed_annotation<R>(
    identifier: impl Into<String>,
    reader: R,
) -> Result<Annotation, FileAnnotationError>
where
    R: Read,
{
    let identifier = identifier.into();
    let mut bed_reader = bed::io::reader::Builder::<6>.build_from_reader(reader);
    let mut record = bed::Record::<6>::default();
    let mut records = Vec::new();
    while bed_reader.read_record(&mut record)? != 0 {
        if matching_bed_record(&record, &identifier) {
            records.push(record.clone());
        }
    }
    parse_bed_annotation_records(identifier, records)
}

/// Translate matching BED records into an annotation with attached graph interval caches.
pub fn translate_bed_annotation<R>(
    context: &AnnotationTranslationContext<'_>,
    identifier: impl Into<String>,
    reader: R,
) -> Result<Annotation, FileAnnotationError>
where
    R: Read,
{
    let annotation = parse_bed_annotation(identifier, reader)?;
    translate_annotation(context, &annotation)
}

/// Parse preselected BED records into an annotation with metadata and source coordinates.
pub fn parse_bed_annotation_records<I>(
    identifier: impl Into<String>,
    records: I,
) -> Result<Annotation, FileAnnotationError>
where
    I: IntoIterator<Item = bed::Record<6>>,
{
    let identifier = identifier.into();
    let records = records.into_iter().collect::<Vec<_>>();
    let first = records.first().ok_or(FileAnnotationError::Empty)?;
    let fields = bed_other_fields(first);
    let bed = BedExtra {
        score: Some(first.score()?.to_string()),
        thick_start: fields.first().and_then(|value| value.parse().ok()),
        thick_end: fields.get(1).and_then(|value| value.parse().ok()),
        item_rgb: fields.get(2).cloned(),
        block_count: fields.get(3).and_then(|value| value.parse().ok()),
        block_sizes: fields
            .get(4)
            .map(|value| parse_bed_list(value))
            .filter(|values| !values.is_empty()),
        block_starts: fields
            .get(5)
            .map(|value| parse_bed_list(value))
            .filter(|values| !values.is_empty()),
        other_fields: fields.get(6..).unwrap_or_default().to_vec(),
    };
    let source_locations = records
        .iter()
        .map(|record| {
            let start = record.feature_start()?.get() as i64 - 1;
            let end = record
                .feature_end()
                .ok_or_else(|| {
                    FileAnnotationError::Translation("BED record has no end".to_string())
                })??
                .get() as i64;
            let strand = match record.strand() {
                Ok(Some(bed::feature::record::Strand::Forward)) => Strand::Forward,
                Ok(Some(bed::feature::record::Strand::Reverse)) => Strand::Reverse,
                Ok(None) | Err(_) => Strand::Unknown,
            };
            Ok(AnnotationLocation {
                reference_sequence_name: record.reference_sequence_name().to_string(),
                start,
                end,
                strand,
            })
        })
        .collect::<Result<Vec<_>, FileAnnotationError>>()?;

    let id = HashId::convert_str(&identifier);
    Ok(Annotation {
        id,
        name: identifier,
        group: "bed".to_string(),
        accession_id: id,
        cached_accession: ForeignKeyCache::default(),
        cached_interval_tree: None,
        history_ref: None,
        source_locations,
        extra: Some(AnnotationExtra {
            bed: Some(bed),
            ..AnnotationExtra::default()
        }),
    })
}

/// Parse preselected BED records and translate their source coordinates onto the sample graph.
pub fn translate_bed_annotation_records<I>(
    context: &AnnotationTranslationContext<'_>,
    identifier: impl Into<String>,
    records: I,
) -> Result<Annotation, FileAnnotationError>
where
    I: IntoIterator<Item = bed::Record<6>>,
{
    let annotation = parse_bed_annotation_records(identifier, records)?;
    translate_annotation(context, &annotation)
}

fn matching_bed_record(record: &bed::Record<6>, identifier: &str) -> bool {
    record
        .name()
        .and_then(|name| std::str::from_utf8(name.as_ref()).ok())
        .is_some_and(|name| name.eq_ignore_ascii_case(identifier))
}

fn bed_other_fields(record: &bed::Record<6>) -> Vec<String> {
    record
        .other_fields()
        .iter()
        .map(|value| String::from_utf8_lossy(value.as_ref()).into_owned())
        .collect()
}

fn parse_bed_list(value: &str) -> Vec<i64> {
    value
        .split(',')
        .filter(|item| !item.is_empty())
        .filter_map(|item| item.parse().ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use std::{
        io::{BufReader, Cursor},
        sync::Arc,
    };

    use gen_core::{HashId, Strand};
    use gen_models::{
        annotations::AnnotationLocation, db::GraphConnection,
        interval_tree::IntervalTreeSource as _, sample::Sample,
    };
    use noodles::bed;

    use super::{
        AnnotationTranslationContext, FileAnnotationError, parse_bed_annotation,
        parse_bed_annotation_records, translate_bed_annotation, translate_bed_annotation_records,
    };
    use crate::{
        test_helpers::create_reverse_test_block_group,
        translate::{bed::translate_bed, translate_annotation},
    };

    #[test]
    fn test_bed_annotation_matches_identifier_and_preserves_metadata() {
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
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/simple.bed");
        let input = std::fs::read(path).expect("should read simple BED fixture");
        // BED source offsets are zero-based and half-open:
        //   source offset: 01234567890
        //   record:         ^--------^ [1,10), strand -
        let parsed = parse_bed_annotation("abc123.1", BufReader::new(input.as_slice()))
            .expect("should parse the matching BED record from one reader");
        assert_eq!(
            parsed.source_locations,
            vec![AnnotationLocation {
                reference_sequence_name: "m123".to_string(),
                start: 1,
                end: 10,
                strand: Strand::Reverse,
            }]
        );
        assert!(parsed.cached_interval_tree.is_none());
        assert!(parsed.cached_accession.snapshot().is_none());
        let annotation = translate_annotation(&context, &parsed)
            .expect("should translate parsed BED coordinates after the reader is dropped");
        let via_reader_wrapper =
            translate_bed_annotation(&context, "abc123.1", BufReader::new(input.as_slice()))
                .expect("should translate BED through its reader convenience API");
        assert_eq!(via_reader_wrapper.source_locations, parsed.source_locations);
        assert!(matches!(
            translate_bed_annotation(&context, "missing", BufReader::new(input.as_slice())),
            Err(FileAnnotationError::Empty)
        ));
        assert!(matches!(
            translate_bed_annotation(&context, "missing", BufReader::new(&b""[..])),
            Err(FileAnnotationError::Empty)
        ));
        assert!(matches!(
            translate_bed_annotation(
                &context,
                "malformed",
                BufReader::new(&b"m123\tbad\t5\tmalformed\t0\t+\n"[..]),
            ),
            Err(FileAnnotationError::Io(_))
        ));
        assert_eq!(annotation.name, "abc123.1");
        assert_eq!(annotation.id, HashId::convert_str("abc123.1"));
        assert_eq!(annotation.accession_id, annotation.id);
        assert_eq!(annotation.group, "bed");
        let metadata = annotation
            .extra
            .as_ref()
            .and_then(|extra| extra.bed.as_ref())
            .expect("should preserve BED metadata");
        assert_eq!(metadata.score.as_deref(), Some("0"));
        assert_eq!(metadata.block_count, Some(3));
        assert_eq!(
            metadata.block_sizes.as_deref(),
            Some([102, 188, 129].as_slice())
        );
        assert_eq!(
            metadata.block_starts.as_deref(),
            Some([0, 3508, 4691].as_slice())
        );
        let empty_conn = GraphConnection(
            rusqlite::Connection::open_in_memory().expect("should open empty database"),
        );
        let accession = annotation
            .get_accession(&empty_conn)
            .expect("should return translated BED accession");
        assert_eq!(accession.id, annotation.accession_id);
        let annotation_tree = annotation
            .intervaltree(&empty_conn)
            .expect("should get translated BED tree without database queries");
        let accession_tree = accession
            .intervaltree(&empty_conn)
            .expect("should get translated BED accession tree without database queries");
        assert!(Arc::ptr_eq(&annotation_tree, &accession_tree));
        assert_eq!(annotation_tree.iter().count(), 1);
        assert_eq!(
            annotation
                .length(&empty_conn)
                .expect("should measure BED annotation"),
            accession
                .length(&empty_conn)
                .expect("should measure BED accession")
        );
        assert!(
            annotation
                .length(&empty_conn)
                .expect("should measure BED annotation")
                > 0
        );
    }

    #[test]
    fn test_bed_annotation_preserves_negative_strand_coordinates() {
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
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/simple.bed");
        let input = std::fs::read(path).expect("should read simple BED fixture");
        let mut reader = bed::io::reader::Builder::<6>.build_from_reader(Cursor::new(input));
        let mut record = bed::Record::<6>::default();
        let mut records = Vec::new();
        while reader
            .read_record(&mut record)
            .expect("should read BED record")
            != 0
        {
            if super::matching_bed_record(&record, "xyz.1") {
                records.push(record.clone());
            }
        }
        assert_eq!(records.len(), 1, "should select the matching BED record");
        let parsed = parse_bed_annotation_records("selected-name", records.clone())
            .expect("should parse selected one-shot BED records without a graph context");
        assert_eq!(parsed.name, "selected-name");
        assert_eq!(parsed.source_locations[0].reference_sequence_name, "m123");
        assert_eq!(parsed.source_locations[0].start, 5);
        assert_eq!(parsed.source_locations[0].end, 8);
        assert_eq!(parsed.source_locations[0].strand, Strand::Reverse);
        let translated_annotation =
            translate_annotation(&context, &parsed).expect("should translate parsed BED records");
        let via_records_wrapper = translate_bed_annotation_records(&context, "xyz.1", records)
            .expect("should translate BED through its one-shot records API");
        assert_eq!(
            via_records_wrapper.source_locations,
            parsed.source_locations
        );
        let empty_conn = GraphConnection(
            rusqlite::Connection::open_in_memory().expect("should open empty database"),
        );
        let accession = translated_annotation
            .get_accession(&empty_conn)
            .expect("should return translated BED accession");
        assert_eq!(accession.name, "selected-name");
        let annotation_tree = translated_annotation
            .intervaltree(&empty_conn)
            .expect("should get the cached BED annotation tree without database queries");
        let accession_tree = accession
            .intervaltree(&empty_conn)
            .expect("should get the cached BED accession tree without database queries");
        assert!(Arc::ptr_eq(&annotation_tree, &accession_tree));
        assert!(Arc::ptr_eq(
            &annotation_tree,
            translated_annotation
                .cached_interval_tree()
                .expect("should find the cached BED annotation tree")
        ));
        assert!(Arc::ptr_eq(
            &accession_tree,
            accession
                .cached_interval_tree()
                .expect("should find the cached BED accession tree")
        ));
        assert_eq!(
            accession
                .length(&empty_conn)
                .expect("should get BED length"),
            3
        );
        let cloned_annotation = translated_annotation.clone();
        assert!(Arc::ptr_eq(
            &annotation_tree,
            cloned_annotation
                .cached_interval_tree()
                .expect("should clone the cached BED annotation tree")
        ));
        let cloned_accession = accession.clone();
        assert!(Arc::ptr_eq(
            &accession_tree,
            cloned_accession
                .cached_interval_tree()
                .expect("should clone the cached BED accession tree")
        ));

        let intervals: Vec<_> = annotation_tree.iter().collect();
        assert_eq!(intervals.len(), 1, "should emit one node interval");

        let interval = intervals[0];
        assert_eq!(interval.range, 0..3);
        assert_eq!(
            interval.value.node_id,
            HashId::try_from("b59698a422128d20462c44537b2d23ef")
                .expect("should parse the translated node id")
        );
        assert_eq!(interval.value.start, 0);
        assert_eq!(interval.value.end, 3);
        assert_eq!(interval.value.sequence_start, 5);
        assert_eq!(interval.value.sequence_end, 8);
        assert_eq!(interval.value.strand, Strand::Reverse);
    }

    #[test]
    fn test_translates_reverse_split_bed_to_exact_node_local_spans() {
        // This test has 2 nodes connected by a negative edge strand while the feature's annotated strand is +.
        // So this is a translation with a graph of the negative strand with a feature annotated on the + strand.
        // Path edges traverse B then A
        //   path index:  01234567890123456
        //   path nodes:  BBBBBBBBAAAAAAAAA
        //   path DNA:    TGCAACGGTAACCGGTT
        //   feature:            ^^ [7,9) = GT
        //
        // Stored node B:
        //   index:     012345678
        //   DNA:       CCGTTGCAT
        //   span:      ^-------^ [0,8)
        //   feature:   ^ [0,1) = C
        //
        // Stored node A:
        //   index:     012345678
        //   DNA:       AACCGGTTA
        //   span:      ^--------^ [0,9)
        //   feature:           ^ [8,9) = A

        // BED [7,9) maps to path [7,9), producing B [0,1) then A [8,9) with feature strand +.
        let conn = crate::test_helpers::get_connection();
        crate::test_helpers::setup_test_data(&conn);
        let (block_group, node_a_id, node_b_id) = create_reverse_test_block_group(&conn);
        let input = format!("{}\t7\t9\tcross-boundary\t0\t+\n", block_group.name);
        let mut translated = Vec::new();
        translate_bed(
            &conn,
            crate::test_helpers::test_workspace(),
            "test",
            Sample::DEFAULT_NAME,
            None,
            Cursor::new(input.as_bytes()),
            &mut translated,
        )
        .expect("should translate BED input");
        let translated = String::from_utf8(translated).expect("should write UTF-8 BED output");
        assert_eq!(
            translated,
            format!(
                "{node_b_id}\t0\t1\tcross-boundary\t0\t+\n{node_a_id}\t8\t9\tcross-boundary\t0\t+\n"
            ),
            "BED fragments should map to stored-node spans in path order and preserve the input feature strand"
        );
    }
}
