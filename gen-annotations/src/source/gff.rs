//! GFF-backed annotation lookup and translation.

use std::io::{BufRead, Cursor, Read};

use gen_core::{HashId, NodeIntervalBlock};
use intervaltree::IntervalTree;
use noodles::gff;

use super::{AnnotationTranslationContext, FileAnnotationError};
use crate::{
    gff::gff_attribute_value_to_string,
    region::{AnnotationRegionData, AnnotationRegionSource},
    translate::gff::translate_gff,
};

/// A GFF annotation selected by identifier from a caller-provided source.
#[derive(Clone, Debug)]
pub struct GffAnnotation {
    identifier: String,
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
        Ok(Self {
            identifier,
            interval_tree,
            block_group_id: context.block_group_id,
        })
    }

    /// Identifier selected from the source records.
    pub fn identifier(&self) -> &str {
        &self.identifier
    }
}

impl AnnotationRegionSource for GffAnnotation {
    type Context = ();
    type Error = std::convert::Infallible;

    fn annotation_region(
        &self,
        _context: &Self::Context,
    ) -> Result<AnnotationRegionData, Self::Error> {
        Ok(AnnotationRegionData {
            interval_tree: self.interval_tree.clone(),
            block_group_id: self.block_group_id,
        })
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
