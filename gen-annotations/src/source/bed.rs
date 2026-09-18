//! BED-backed annotation lookup and translation.

use std::io::{Cursor, Read};

use gen_core::{HashId, NodeIntervalBlock, Strand, is_terminal};
use gen_models::annotations::MaterializedAnnotation;
use intervaltree::IntervalTree;
use noodles::bed;

use super::{AnnotationTranslationContext, FileAnnotationError};
use crate::translate::bed::translate_bed;

/// A BED record accepted by [`BedAnnotation::from_records`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BedRecord {
    /// Reference sequence name used by the selected graph.
    pub reference_sequence_name: String,
    /// Zero-based inclusive feature start from the matched source record.
    pub start: i64,
    /// Zero-based exclusive feature end from the matched source record.
    pub end: i64,
    /// Optional identifier retained by the source record.
    pub name: Option<String>,
    /// Source record strand.
    pub strand: Strand,
}

/// A BED annotation selected by name from a caller-provided source.
#[derive(Clone, Debug)]
pub struct BedAnnotation {
    materialized: MaterializedAnnotation,
}

impl BedAnnotation {
    /// Scan a selected BED reader for one name and translate only matching records.
    pub fn from_reader<R>(
        context: &AnnotationTranslationContext<'_>,
        identifier: impl Into<String>,
        reader: R,
    ) -> Result<Self, FileAnnotationError>
    where
        R: Read,
    {
        let identifier = identifier.into();
        let mut bed_reader = bed::io::reader::Builder::<6>.build_from_reader(reader);
        let mut record = bed::Record::<6>::default();
        let mut records = Vec::new();
        while bed_reader.read_record(&mut record)? != 0 {
            let name = record
                .name()
                .and_then(|value| std::str::from_utf8(value.as_ref()).ok())
                .map(str::to_string);
            if name
                .as_deref()
                .is_some_and(|name| name.eq_ignore_ascii_case(&identifier))
            {
                let start = record
                    .feature_start()
                    .map_err(|error| FileAnnotationError::Translation(error.to_string()))?
                    .get() as i64
                    - 1;
                let end = record
                    .feature_end()
                    .ok_or_else(|| {
                        FileAnnotationError::Translation(
                            "BED record has no feature end".to_string(),
                        )
                    })?
                    .map_err(|error| FileAnnotationError::Translation(error.to_string()))?
                    .get() as i64;
                let strand = match record.strand() {
                    Ok(Some(bed::feature::record::Strand::Forward)) => Strand::Forward,
                    Ok(Some(bed::feature::record::Strand::Reverse)) => Strand::Reverse,
                    Ok(None) | Err(_) => Strand::Unknown,
                };
                records.push(BedRecord {
                    reference_sequence_name: String::from_utf8_lossy(
                        record.reference_sequence_name().as_ref(),
                    )
                    .to_string(),
                    start,
                    end,
                    name,
                    strand,
                });
            }
        }
        Self::from_records(context, identifier, records)
    }

    /// Build an annotation from records already matched by an upstream lookup provider.
    pub fn from_records<I>(
        context: &AnnotationTranslationContext<'_>,
        identifier: impl Into<String>,
        records: I,
    ) -> Result<Self, FileAnnotationError>
    where
        I: IntoIterator<Item = BedRecord>,
    {
        let identifier = identifier.into();
        let mut input = Vec::new();
        {
            let mut writer = bed::io::Writer::<6, _>::new(&mut input);
            for record in records {
                let start = record.start.max(0) as usize + 1;
                let end = record.end.max(record.start) as usize;
                let mut builder = bed::feature::RecordBuf::<6>::builder()
                    .set_reference_sequence_name(record.reference_sequence_name)
                    .set_feature_start(
                        noodles::core::Position::try_from(start)
                            .map_err(|error| FileAnnotationError::Translation(error.to_string()))?,
                    )
                    .set_feature_end(
                        noodles::core::Position::try_from(end)
                            .map_err(|error| FileAnnotationError::Translation(error.to_string()))?,
                    );
                if let Some(name) = record.name {
                    builder = builder.set_name(name);
                }
                builder = match record.strand {
                    Strand::Forward => builder.set_strand(bed::feature::record::Strand::Forward),
                    Strand::Reverse => builder.set_strand(bed::feature::record::Strand::Reverse),
                    Strand::Unknown | Strand::ImportantButUnknown => builder,
                };
                writer.write_feature_record(&builder.build())?;
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
        let interval_tree = translated_bed_interval_tree(&translated)?;
        let materialized =
            MaterializedAnnotation::new(identifier, context.block_group_id, interval_tree)?;
        Ok(Self { materialized })
    }

    /// Identifier selected from the source records.
    pub fn identifier(&self) -> &str {
        self.materialized.name()
    }

    /// Return the source-neutral translated annotation used by model resolution.
    pub fn annotation(&self) -> &MaterializedAnnotation {
        &self.materialized
    }

    /// Consume the source wrapper and return its translated annotation.
    pub fn into_annotation(self) -> MaterializedAnnotation {
        self.materialized
    }
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

    use gen_models::sample::Sample;

    use super::{AnnotationTranslationContext, BedAnnotation};

    #[test]
    fn test_bed_annotation_matches_identifier_and_materializes_tree() {
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
            block_group_id: block_group.id,
        };
        let annotation = BedAnnotation::from_reader(
            &context,
            "abc123.1",
            File::open(concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/simple.bed"))
                .expect("should open simple BED fixture"),
        )
        .expect("should translate the matching BED record");

        assert_eq!(annotation.identifier(), "abc123.1");
        assert_eq!(annotation.annotation().name(), "abc123.1");
        assert_eq!(annotation.annotation().interval_tree().iter().count(), 1);
    }
}
