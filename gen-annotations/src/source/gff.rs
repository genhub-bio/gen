//! GFF parsing and translation adapters.
//!
//! The caller selects the source and opens it.  These functions only match records from that
//! already-selected source, build the existing annotation value, and hand matched records to the
//! established graph translator.  The record-based entry points are the seam for a future indexed
//! lookup that can supply matches without scanning the source here.

use std::io::{BufRead, Cursor, Read};

use gen_core::{HashId, NodeIntervalBlock};
use gen_models::annotations::{Annotation, AnnotationExtra, GffAttribute, GffExtra};
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
        extra: Some(AnnotationExtra {
            gff: Some(gff),
            ..AnnotationExtra::default()
        }),
    })
}

/// Translate matching GFF records from a selected reader into cumulative node intervals.
pub fn translate_gff_annotation<R>(
    context: &AnnotationTranslationContext<'_>,
    identifier: impl Into<String>,
    reader: R,
) -> Result<IntervalTree<i64, NodeIntervalBlock>, FileAnnotationError>
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
    translate_gff_annotation_records(context, records)
}

/// Translate GFF records already selected by an upstream lookup provider.
pub fn translate_gff_annotation_records<I>(
    context: &AnnotationTranslationContext<'_>,
    records: I,
) -> Result<IntervalTree<i64, NodeIntervalBlock>, FileAnnotationError>
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
    translated_gff_interval_tree(&translated)
}

fn matching_gff_record(record: &gff::feature::RecordBuf, identifier: &str) -> bool {
    ["Name", "ID", "gene", "db_xref"].iter().any(|key| {
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
    use std::{fs::File, io::BufReader};

    use gen_core::{
        HashId,
        region::{Region, normalize_user_search_region},
    };
    use gen_models::{region::ResolvedGenRegion, sample::Sample};

    use super::{AnnotationTranslationContext, parse_gff_annotation, translate_gff_annotation};

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
        let tree = translate_gff_annotation(
            &context,
            "gene-a0001",
            BufReader::new(File::open(path).expect("should reopen simple GFF fixture")),
        )
        .expect("should translate the matching GFF record");
        assert_eq!(tree.iter().count(), 2);

        let normalized_region = normalize_user_search_region(
            &Region::parse("gene-a0001:13-16").expect("should parse a cross-slice range"),
        );
        let resolved = ResolvedGenRegion::from_annotation_intervals(
            &conn,
            &annotation,
            block_group.id,
            tree,
            normalized_region.start.expect("should have a range start"),
            normalized_region.end.expect("should have a range end"),
        )
        .expect("should build a resolved GFF annotation region")
        .find_graph_positions(&conn, crate::test_helpers::test_workspace(), 0, 0)
        .expect("should resolve graph positions across translated slices");
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
}
