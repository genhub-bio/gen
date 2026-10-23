//! Source-specific annotation records and their translated graph intervals.
//!
//! The application chooses and opens an annotation asset. These adapters receive a reader for the
//! selected asset and translate each record independently. A future attribute index can provide the
//! same selected records without changing the downstream interval-tree or region-resolution layers.

mod bed;
mod gff;

use std::{collections::HashMap, sync::Arc};

pub use bed::translate_bed_annotation;
use gen_core::{HashId, NodeIntervalBlock, Strand, Workspace};
use gen_models::{
    block_group::BlockGroup, db::GraphConnection, reference_alias::ReferenceAlias,
    region::ResolvedGenRegion, sample::Sample,
};
pub use gff::{import_gff_annotations, translate_gff_annotation};
use intervaltree::IntervalTree;
use thiserror::Error;

/// Context used while translating records selected by the application.
pub struct AnnotationTranslationContext<'a> {
    /// Database connection used by the existing translators.
    pub conn: &'a GraphConnection,
    /// Workspace used by graph/path loading.
    pub workspace: &'a Workspace,
    /// Optional history reference for graph lookup.
    pub history_ref: Option<&'a str>,
    block_groups_by_name: HashMap<String, HashId>,
    references_by_alias: HashMap<String, String>,
    regions_by_reference: HashMap<String, ResolvedGenRegion>,
}

/// Errors raised while building a file-backed annotation from matched records.
#[derive(Debug, Error)]
pub enum FileAnnotationError {
    #[error("annotation record I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// A BED6 record has an empty or missing name field.
    #[error("BED annotation record has no name")]
    MissingBedName,
    /// A GFF record has no nonempty supported name attribute.
    #[error("GFF annotation record has no name attribute")]
    MissingGffName,
    /// The BED6 score field could not be parsed by noodles.
    #[error("invalid BED score: {0}")]
    InvalidBedScore(#[source] std::io::Error),
    #[error("annotation translation error: {0}")]
    Translation(String),
    #[error("annotation has no translated records")]
    Empty,
    #[error("failed to persist translated annotation: {0}")]
    Persistence(String),
}

impl<'a> AnnotationTranslationContext<'a> {
    /// Creates translation state and loads the selected sample's graph references and aliases.
    pub fn new(
        conn: &'a GraphConnection,
        workspace: &'a Workspace,
        collection_name: &str,
        sample_name: &str,
        history_ref: Option<&'a str>,
    ) -> Result<Self, FileAnnotationError> {
        let block_groups_by_name =
            Sample::get_block_groups(conn, collection_name, sample_name, history_ref)
                .into_iter()
                .map(|block_group| (block_group.name, block_group.id))
                .collect::<HashMap<_, _>>();
        let references = block_groups_by_name.keys().cloned().collect();
        let references_by_alias =
            ReferenceAlias::get_references_by_alias(conn, references, history_ref)
                .map_err(|error| FileAnnotationError::Translation(error.to_string()))?;

        Ok(Self {
            conn,
            workspace,
            history_ref,
            block_groups_by_name,
            references_by_alias,
            regions_by_reference: HashMap::new(),
        })
    }

    pub(super) fn intervaltree(
        &mut self,
        reference_name: &str,
        start: i64,
        end: i64,
        strand: Strand,
    ) -> Result<(HashId, Arc<IntervalTree<i64, NodeIntervalBlock>>), FileAnnotationError> {
        let resolved_reference = self
            .references_by_alias
            .get(reference_name)
            .map_or(reference_name, String::as_str);
        let Some(block_group_id) = self.block_groups_by_name.get(resolved_reference) else {
            return Err(FileAnnotationError::Empty);
        };

        let region = match self.regions_by_reference.get(resolved_reference) {
            Some(region) => region.clone(),
            None => {
                let path =
                    BlockGroup::get_current_path(self.conn, block_group_id, self.history_ref)
                        .map_err(|error| FileAnnotationError::Translation(error.to_string()))?;
                let region = ResolvedGenRegion::from_path(self.conn, &path, 0, 0)
                    .map_err(|error| FileAnnotationError::Translation(error.to_string()))?;
                self.regions_by_reference
                    .insert(resolved_reference.to_string(), region.clone());
                region
            }
        };
        let mut region = region;
        region.start = start;
        region.end = end;
        let slices = region
            .graph_locus(self.conn, self.workspace)
            .map(|locus| locus.slices)
            .map_err(|error| FileAnnotationError::Translation(error.to_string()))?;

        let mut position = 0;
        let mut blocks = Vec::new();
        for slice in slices {
            let start_offset = i64::try_from(slice.start)
                .map_err(|error| FileAnnotationError::Translation(error.to_string()))?;
            let end_offset = i64::try_from(slice.end)
                .map_err(|error| FileAnnotationError::Translation(error.to_string()))?;
            let sequence_start = slice.block.sequence_start + start_offset;
            let sequence_end = slice.block.sequence_start + end_offset;
            let block = NodeIntervalBlock {
                node_id: slice.block.node_id,
                start: position,
                end: position + sequence_end - sequence_start,
                sequence_start,
                sequence_end,
                strand,
            };
            position = block.end;
            blocks.push(block);
        }
        if blocks.is_empty() {
            return Err(FileAnnotationError::Empty);
        }

        let interval_tree = Arc::new(
            blocks
                .into_iter()
                .map(|block| (block.start..block.end, block))
                .collect::<IntervalTree<i64, NodeIntervalBlock>>(),
        );
        Ok((region.block_group.id, interval_tree))
    }
}

#[cfg(test)]
mod tests {
    use gen_core::Strand;
    use gen_models::sample::Sample;

    use super::AnnotationTranslationContext;
    use crate::test_helpers::{
        create_reverse_test_block_group, create_sliced_test_block_group, get_connection,
        setup_test_data, test_workspace,
    };

    #[test]
    fn test_intervaltree_maps_sliced_nodes_to_stored_coordinates() {
        // A region is a GraphNodeSlice -- this ensures that we convert it to blocks correctly.
        let conn = get_connection();
        setup_test_data(&conn);
        let (forward_block_group, forward_node_id) =
            create_sliced_test_block_group(&conn, "sliced-forward", Strand::Forward);
        let (reverse_block_group, reverse_node_id) =
            create_sliced_test_block_group(&conn, "sliced-reverse", Strand::Reverse);
        let mut context = AnnotationTranslationContext::new(
            &conn,
            test_workspace(),
            "test",
            Sample::DEFAULT_NAME,
            None,
        )
        .expect("should create annotation translation context");

        // The GraphNodeSlices are from position 3 and position 5. So here we're just taking the first 2 bases for an ITree and making
        // sure we get the correct values for sequence_start/end in the interval
        for (reference_name, expected_block_group_id, expected_node_id, sequence_start, strand) in [
            (
                "sliced-forward",
                forward_block_group.id,
                forward_node_id,
                3,
                Strand::Reverse,
            ),
            (
                "sliced-reverse",
                reverse_block_group.id,
                reverse_node_id,
                5,
                Strand::Forward,
            ),
        ] {
            let (block_group_id, tree) = context
                .intervaltree(reference_name, 0, 2, strand)
                .expect("should map the sliced path interval");
            assert_eq!(block_group_id, expected_block_group_id);

            let intervals = tree.iter().collect::<Vec<_>>();
            assert_eq!(intervals.len(), 1);
            assert_eq!(intervals[0].range, 0..2);
            assert_eq!(intervals[0].value.node_id, expected_node_id);
            assert_eq!(intervals[0].value.start, 0);
            assert_eq!(intervals[0].value.end, 2);
            assert_eq!(intervals[0].value.sequence_start, sequence_start);
            assert_eq!(intervals[0].value.sequence_end, sequence_start + 2);
            assert_eq!(intervals[0].value.strand, strand);
        }
    }
}
