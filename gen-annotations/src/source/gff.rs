//! GFF-backed annotation lookup and translation.

use std::io::{BufRead, Cursor, Read};

use gen_core::{HashId, NodeIntervalBlock};
use gen_models::{annotations::Annotation, db::GraphConnection, region::AnnotationSource};
use intervaltree::IntervalTree;
use noodles::gff;

use super::{AnnotationTranslationContext, FileAnnotationError, source_annotation};
use crate::{gff::gff_attribute_value_to_string, translate::gff::translate_gff};

/// A GFF annotation selected by identifier from a caller-provided source.
#[derive(Clone, Debug)]
pub struct GffAnnotation {
    annotation: Annotation,
    interval_tree: IntervalTree<i64, NodeIntervalBlock>,
    block_group_id: HashId,
}

impl GffAnnotation {
    /// Scan a selected GFF reader for one identifier and translate only matching records.
    pub fn from_reader<R>(
        context: &AnnotationTranslationContext<'_>,
        identifier: impl Into<String>,
        reader: R,
    ) -> Result<Self, FileAnnotationError>
    where
        R: Read + BufRead,
    {
        let identifier = identifier.into();
        let mut gff_reader = gff::io::Reader::new(reader);
        let records = gff_reader
            .record_bufs()
            .filter_map(|result| match result {
                Ok(record) => matching_gff_record(&record, &identifier).then_some(Ok(record)),
                Err(error) => Some(Err(error)),
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_records(context, identifier, records)
    }

    /// Build an annotation from records already matched by an upstream lookup provider.
    pub fn from_records<I>(
        context: &AnnotationTranslationContext<'_>,
        identifier: impl Into<String>,
        records: I,
    ) -> Result<Self, FileAnnotationError>
    where
        I: IntoIterator<Item = gff::feature::RecordBuf>,
    {
        let identifier = identifier.into();
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
        let interval_tree = translated_gff_interval_tree(&translated)?;
        let annotation = source_annotation(&identifier, "gff");
        Ok(Self {
            annotation,
            interval_tree,
            block_group_id: context.block_group_id,
        })
    }

    /// Identifier selected from the source records.
    pub fn identifier(&self) -> &str {
        &self.annotation.name
    }

    /// Return the selected source annotation identity.
    pub fn annotation(&self) -> &Annotation {
        &self.annotation
    }

    /// Consume the source wrapper and return its annotation identity.
    pub fn into_annotation(self) -> Annotation {
        self.annotation
    }
}

impl AnnotationSource for GffAnnotation {
    type Error = std::convert::Infallible;

    fn annotation(&self) -> &Annotation {
        &self.annotation
    }

    fn annotation_intervals(
        &self,
        _conn: &GraphConnection,
    ) -> Result<(IntervalTree<i64, NodeIntervalBlock>, HashId), Self::Error> {
        Ok((self.interval_tree.clone(), self.block_group_id))
    }
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

    use gen_core::HashId;
    use gen_models::{region::AnnotationSource, sample::Sample};

    use super::{AnnotationTranslationContext, GffAnnotation};

    #[test]
    fn test_gff_annotation_matches_identifier_and_builds_tree() {
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
        let annotation = GffAnnotation::from_reader(
            &context,
            "gene-a0001",
            BufReader::new(
                File::open(concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/simple.gff"))
                    .expect("should open simple GFF fixture"),
            ),
        )
        .expect("should translate the matching GFF record");

        assert_eq!(annotation.identifier(), "gene-a0001");
        assert_eq!(annotation.annotation().name, "gene-a0001");
        assert_eq!(
            annotation.annotation().id,
            HashId::convert_str("gene-a0001")
        );
        let (interval_tree, _) = annotation
            .annotation_intervals(&conn)
            .expect("should expose translated annotation intervals");
        assert_eq!(interval_tree.iter().count(), 2);
    }
}
