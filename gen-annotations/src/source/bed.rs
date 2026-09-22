//! BED parsing and translation adapters.
//!
//! Source selection remains with the caller. Reader functions perform only identifier matching;
//! record-based functions accept records already selected by an upstream provider so an indexed
//! lookup can replace a scan without changing annotation construction or translation.

use std::io::{Cursor, Read};

use gen_core::{HashId, NodeIntervalBlock, Strand, is_terminal};
use gen_models::annotations::{Annotation, AnnotationExtra, BedExtra};
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
        extra: Some(AnnotationExtra {
            bed: Some(bed),
            ..AnnotationExtra::default()
        }),
    })
}

/// Translate matching BED records from a selected reader into cumulative node intervals.
pub fn translate_bed_annotation<R>(
    context: &AnnotationTranslationContext<'_>,
    identifier: impl Into<String>,
    reader: R,
) -> Result<IntervalTree<i64, NodeIntervalBlock>, FileAnnotationError>
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
    translate_bed_annotation_records(context, records)
}

/// Translate BED records already selected by an upstream lookup provider.
pub fn translate_bed_annotation_records<I>(
    context: &AnnotationTranslationContext<'_>,
    records: I,
) -> Result<IntervalTree<i64, NodeIntervalBlock>, FileAnnotationError>
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
    translated_bed_interval_tree(&translated)
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
    use std::fs::File;

    use gen_core::HashId;
    use gen_models::sample::Sample;

    use super::{AnnotationTranslationContext, parse_bed_annotation, translate_bed_annotation};

    #[test]
    fn test_bed_annotation_matches_identifier_and_preserves_metadata() {
        let conn = crate::test_helpers::get_connection();
        crate::test_helpers::setup_test_data(&conn);
        let context = AnnotationTranslationContext {
            conn: &conn,
            workspace: crate::test_helpers::test_workspace(),
            collection_name: "test",
            sample_name: Sample::DEFAULT_NAME,
            history_ref: None,
        };
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/simple.bed");
        let annotation = parse_bed_annotation(
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
        let tree = translate_bed_annotation(
            &context,
            "abc123.1",
            File::open(path).expect("should reopen simple BED fixture"),
        )
        .expect("should translate the matching BED record");
        assert_eq!(tree.iter().count(), 1);
    }
}
