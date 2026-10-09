//! BED parsing and translation adapters.
//!
//! The source of the BED file comes from the caller, this file works off a readable input.
//! The readers here carry out parsing that input, matching identifiers, and returning an
//! Annotation object for use downstream.

use std::{
    io::{Cursor, Read},
    sync::Arc,
};

use gen_core::{HashId, NodeIntervalBlock, Strand, is_terminal};
use gen_models::{
    accession::Accession,
    annotations::{Annotation, AnnotationExtra, BedExtra},
    interval_tree::IntervalTreeSource as _,
};
use intervaltree::IntervalTree;
use noodles::bed;

use super::{AnnotationTranslationContext, FileAnnotationError};
use crate::translate::bed::translate_bed;

/// Parse one BED name's matching records into the existing annotation model.
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

/// Build a BED annotation from records already matched by an upstream lookup provider.
pub fn parse_bed_annotation_records<I>(
    identifier: impl Into<String>,
    records: I,
) -> Result<Annotation, FileAnnotationError>
where
    I: IntoIterator<Item = bed::Record<6>>,
{
    let identifier = identifier.into();
    let records: Vec<_> = records.into_iter().collect();
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
    let id = HashId::convert_str(&identifier);
    Ok(Annotation {
        id,
        name: identifier,
        group: "bed".to_string(),
        accession_id: id,
        cached_interval_tree: None,
        extra: Some(AnnotationExtra {
            bed: Some(bed),
            ..AnnotationExtra::default()
        }),
    })
}

/// Translate matching BED records and cache their graph intervals on the annotation.
///
/// Returns the associated accession.
pub fn translate_bed_annotation<R>(
    context: &AnnotationTranslationContext<'_>,
    annotation: &mut Annotation,
    reader: R,
) -> Result<Accession, FileAnnotationError>
where
    R: Read,
{
    let mut bed_reader = bed::io::reader::Builder::<6>.build_from_reader(reader);
    let mut record = bed::Record::<6>::default();
    let mut records = Vec::new();
    while bed_reader.read_record(&mut record)? != 0 {
        if matching_bed_record(&record, &annotation.name) {
            records.push(record.clone());
        }
    }
    translate_bed_annotation_records(context, annotation, records)
}

/// Translate selected BED records and cache their graph intervals on the annotation.
///
/// Returns the associated accession.
pub fn translate_bed_annotation_records<I>(
    context: &AnnotationTranslationContext<'_>,
    annotation: &mut Annotation,
    records: I,
) -> Result<Accession, FileAnnotationError>
where
    I: IntoIterator<Item = bed::Record<6>>,
{
    let mut input = Vec::new();
    {
        let mut writer = bed::io::Writer::<6, _>::new(&mut input);
        for record in records {
            writer.write_record(&record)?;
        }
    }
    let mut translated = Vec::new();
    translate_bed(
        context.conn,
        context.workspace,
        context.collection_name,
        context.sample_name,
        context.history_ref,
        Cursor::new(input),
        &mut translated,
    )
    .map_err(|error| FileAnnotationError::Translation(error.to_string()))?;
    let interval_tree = Arc::new(translated_bed_interval_tree(&translated)?);
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

fn translated_bed_interval_tree(
    translated: &[u8],
) -> Result<IntervalTree<i64, NodeIntervalBlock>, FileAnnotationError> {
    let mut reader = bed::io::reader::Builder::<6>.build_from_reader(Cursor::new(translated));
    let mut record = bed::Record::<6>::default();
    let mut position = 0;
    let mut blocks = Vec::new();
    while reader.read_record(&mut record)? != 0 {
        let node_name = String::from_utf8_lossy(record.reference_sequence_name().as_ref());
        let node_id = HashId::try_from(node_name.as_ref())
            .map_err(|_| FileAnnotationError::InvalidNode(node_name.to_string()))?;
        if is_terminal(node_id) {
            continue;
        }
        let start = record
            .feature_start()
            .map_err(|error| FileAnnotationError::Translation(error.to_string()))?
            .get() as i64
            - 1;
        let end = record
            .feature_end()
            .ok_or_else(|| FileAnnotationError::Translation("BED record has no end".to_string()))?
            .map_err(|error| FileAnnotationError::Translation(error.to_string()))?
            .get() as i64;
        if end <= start {
            continue;
        }
        let strand = match record.strand() {
            Ok(Some(bed::feature::record::Strand::Forward)) => Strand::Forward,
            Ok(Some(bed::feature::record::Strand::Reverse)) => Strand::Reverse,
            Ok(None) | Err(_) => Strand::Unknown,
        };
        let block = NodeIntervalBlock {
            node_id,
            start: position,
            end: position + end - start,
            sequence_start: start,
            sequence_end: end,
            strand,
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
    use std::{fs::File, sync::Arc};

    use gen_core::{HashId, Strand};
    use gen_models::{interval_tree::IntervalTreeSource as _, sample::Sample};

    use super::{AnnotationTranslationContext, parse_bed_annotation, translate_bed_annotation};

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
        let mut annotation = parse_bed_annotation(
            "abc123.1",
            File::open(path).expect("should open simple BED fixture"),
        )
        .expect("should parse the matching BED record");
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
        let accession = translate_bed_annotation(
            &context,
            &mut annotation,
            File::open(path).expect("should reopen simple BED fixture"),
        )
        .expect("should translate the matching BED record");
        assert_eq!(accession.id, annotation.accession_id);
        assert_eq!(
            annotation
                .intervaltree(&conn)
                .expect("should get translated BED tree")
                .iter()
                .count(),
            1
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
        let mut annotation = parse_bed_annotation(
            "xyz.1",
            File::open(path).expect("should open simple BED fixture"),
        )
        .expect("should parse the negative-strand BED record");
        assert_eq!(annotation.name, "xyz.1");

        let accession = translate_bed_annotation(
            &context,
            &mut annotation,
            File::open(path).expect("should reopen simple BED fixture"),
        )
        .expect("should translate the negative-strand BED record");
        assert_eq!(accession.name, "xyz.1");
        let annotation_tree = annotation
            .intervaltree(&conn)
            .expect("should get the cached BED annotation tree");
        let accession_tree = accession
            .intervaltree(&conn)
            .expect("should get the cached BED accession tree");
        assert!(Arc::ptr_eq(&annotation_tree, &accession_tree));
        assert!(Arc::ptr_eq(
            &annotation_tree,
            annotation
                .cached_interval_tree()
                .expect("should find the cached BED annotation tree")
        ));
        assert!(Arc::ptr_eq(
            &accession_tree,
            accession
                .cached_interval_tree()
                .expect("should find the cached BED accession tree")
        ));
        assert_eq!(accession.length(&conn).expect("should get BED length"), 3);
        let cloned_annotation = annotation.clone();
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
}
