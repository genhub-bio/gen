use std::{collections::HashMap, sync::Arc};

use gen_core::NodeIntervalBlock;
use gen_models::{
    accession::Accession, annotations::Annotation, block_group::BlockGroup,
    interval_tree::IntervalTreeSource as _, reference_alias::ReferenceAlias,
    region::ResolvedGenRegion, sample::Sample,
};
use intervaltree::IntervalTree;

use crate::source::{AnnotationTranslationContext, FileAnnotationError};

pub mod bed;
pub mod gff;

/// Translate an annotation's retained source coordinates onto the selected sample graph.
///
/// Parsed annotations keep these coordinates so the source reader can be dropped before graph
/// access begins and the same annotation can be translated against another context later.
pub fn translate_annotation(
    context: &AnnotationTranslationContext<'_>,
    annotation: &Annotation,
) -> Result<Annotation, FileAnnotationError> {
    if annotation.source_locations.is_empty() {
        return Err(FileAnnotationError::MissingSourceLocations);
    }

    let sample_block_groups = Sample::get_block_groups(
        context.conn,
        context.collection_name,
        context.sample_name,
        context.history_ref,
    )
    .into_iter()
    .map(|block_group| (block_group.name, block_group.id))
    .collect::<HashMap<_, _>>();
    let references_by_alias = ReferenceAlias::get_references_by_alias(
        context.conn,
        sample_block_groups.keys().cloned().collect(),
        context.history_ref,
    )
    .map_err(|error| FileAnnotationError::Translation(error.to_string()))?;
    let mut regions_by_reference = HashMap::<String, ResolvedGenRegion>::new();
    let mut position = 0;
    let mut blocks = Vec::new();

    for location in &annotation.source_locations {
        let reference = references_by_alias
            .get(&location.reference_sequence_name)
            .unwrap_or(&location.reference_sequence_name);
        let Some(block_group_id) = sample_block_groups.get(reference) else {
            continue;
        };
        if !regions_by_reference.contains_key(reference) {
            let path =
                BlockGroup::get_current_path(context.conn, block_group_id, context.history_ref)
                    .map_err(|error| FileAnnotationError::Translation(error.to_string()))?;
            let region = ResolvedGenRegion::from_path(context.conn, &path, 0, 0)
                .map_err(|error| FileAnnotationError::Translation(error.to_string()))?;
            regions_by_reference.insert(reference.clone(), region);
        }
        let mut region = regions_by_reference
            .get(reference)
            .expect("should cache the selected reference region")
            .clone();
        region.start = location.start;
        region.end = location.end;
        let slices = region
            .graph_locus(context.conn, context.workspace)
            .map_err(|error| FileAnnotationError::Translation(error.to_string()))?
            .slices;

        for slice in slices {
            let node_id = slice.block.node_id;
            let start = slice.block.sequence_start + slice.start as i64;
            let end = slice.block.sequence_start + slice.end as i64;
            let block = NodeIntervalBlock {
                node_id,
                start: position,
                end: position + end - start,
                sequence_start: start,
                sequence_end: end,
                strand: location.strand,
            };
            position = block.end;
            blocks.push(block);
        }
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
    let mut translated_annotation = annotation.clone();
    translated_annotation.history_ref = context.history_ref.map(str::to_owned);
    translated_annotation.set_cached_interval_tree(Some(Arc::clone(&interval_tree)));

    let mut accession = Accession {
        id: translated_annotation.accession_id,
        name: translated_annotation.name.clone(),
        block_group_id: context.block_group_id,
        parent_accession_id: None,
        cached_interval_tree: None,
        history_ref: context.history_ref.map(str::to_owned),
    };
    accession.set_cached_interval_tree(Some(interval_tree));
    translated_annotation.set_accession(Arc::new(accession));
    Ok(translated_annotation)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use gen_core::{HashId, Strand};
    use gen_models::{
        ForeignKeyCache,
        annotations::{Annotation, AnnotationLocation},
        db::GraphConnection,
        reference_alias::ReferenceAlias,
        sample::Sample,
    };

    use super::translate_annotation;
    use crate::{
        source::AnnotationTranslationContext,
        test_helpers::{self, create_reverse_test_block_group},
    };

    #[test]
    fn test_translates_format_neutral_locations_in_order_through_reference_alias() {
        let conn = test_helpers::get_connection();
        test_helpers::setup_test_data(&conn);
        let block_group = Sample::get_block_groups(&conn, "test", Sample::DEFAULT_NAME, None)
            .into_iter()
            .find(|block_group| block_group.name == "m123")
            .expect("should find test block group");
        ReferenceAlias::create(
            &conn,
            &block_group.name,
            None,
            Some(block_group.name.clone()),
            None,
            None,
            Some("source-alias".to_string()),
            None,
        )
        .expect("should create source alias for test reference");
        let context = AnnotationTranslationContext {
            conn: &conn,
            workspace: test_helpers::test_workspace(),
            collection_name: "test",
            sample_name: Sample::DEFAULT_NAME,
            block_group_id: block_group.id,
            history_ref: None,
        };
        let id = HashId::convert_str("manual-annotation");
        let annotation = Annotation {
            id,
            name: "manual-annotation".to_string(),
            group: "source".to_string(),
            accession_id: id,
            cached_accession: ForeignKeyCache::default(),
            cached_interval_tree: None,
            history_ref: None,
            source_locations: vec![
                AnnotationLocation {
                    reference_sequence_name: "source-alias".to_string(),
                    start: 6,
                    end: 8,
                    strand: Strand::Reverse,
                },
                AnnotationLocation {
                    reference_sequence_name: "source-alias".to_string(),
                    start: 2,
                    end: 4,
                    strand: Strand::Forward,
                },
            ],
            extra: None,
        };

        // Source intervals retain supplied order, even when coordinates descend:
        //   source offset: 01234567
        //   path node:     AAAAAAAA
        //   stored DNA:    ATCGATCG
        //   row A (-):     ......^^ [6,8)
        //   row B (+):     ..^^.... [2,4)
        //   output order:  A then B at annotation offsets [0,2), [2,4)
        let translated = translate_annotation(&context, &annotation)
            .expect("should translate locations without format-specific metadata");
        assert!(annotation.cached_interval_tree.is_none());
        assert!(annotation.cached_accession.snapshot().is_none());
        assert_eq!(translated.source_locations, annotation.source_locations);
        assert_eq!(translated.history_ref, None);
        let empty_conn = GraphConnection(
            rusqlite::Connection::open_in_memory().expect("should open schema-less database"),
        );
        let annotation_tree = translated
            .intervaltree(&empty_conn)
            .expect("should return the translated annotation tree from its cache");
        let accession = translated
            .get_accession(&empty_conn)
            .expect("should return the translated accession from its cache");
        let accession_tree = accession
            .intervaltree(&empty_conn)
            .expect("should return the translated accession tree from its cache");
        assert!(Arc::ptr_eq(&annotation_tree, &accession_tree));
        assert_eq!(
            translated
                .length(&empty_conn)
                .expect("should measure annotation"),
            4
        );
        assert_eq!(
            accession
                .length(&empty_conn)
                .expect("should measure accession"),
            4
        );

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
            vec![(0, 2, 6, 8, Strand::Reverse), (2, 4, 2, 4, Strand::Forward),]
        );

        let mut without_source_locations = annotation.clone();
        without_source_locations.source_locations.clear();
        assert!(matches!(
            translate_annotation(&context, &without_source_locations),
            Err(crate::source::FileAnnotationError::MissingSourceLocations)
        ));
    }

    #[test]
    fn test_translation_keeps_reverse_graph_slices_in_positive_node_coordinates() {
        let conn = test_helpers::get_connection();
        test_helpers::setup_test_data(&conn);
        let (block_group, node_a_id, node_b_id) = create_reverse_test_block_group(&conn);
        let context = AnnotationTranslationContext {
            conn: &conn,
            workspace: test_helpers::test_workspace(),
            collection_name: "test",
            sample_name: Sample::DEFAULT_NAME,
            block_group_id: block_group.id,
            history_ref: None,
        };
        let id = HashId::convert_str("reverse-manual-annotation");
        let annotation = Annotation {
            id,
            name: "reverse-manual-annotation".to_string(),
            group: "source".to_string(),
            accession_id: id,
            cached_accession: ForeignKeyCache::default(),
            cached_interval_tree: None,
            history_ref: None,
            source_locations: vec![AnnotationLocation {
                reference_sequence_name: block_group.name,
                start: 7,
                end: 9,
                strand: Strand::Forward,
            }],
            extra: None,
        };

        // Negative graph edges reverse traversal while emitted feature strand stays +:
        //   path index:    01234567890123456
        //   path nodes:    BBBBBBBBAAAAAAAAA
        //   path DNA:      TGCAACGGTAACCGGTT
        //   feature(+):            ^^ [7,9)
        //
        // Stored node B:
        //   index:    012345678
        //   DNA:      CCGTTGCAT
        //   selected: ^ [0,1)
        //
        // Stored node A:
        //   index:    012345678
        //   DNA:      AACCGGTTA
        //   selected:         ^ [8,9)
        let translated = translate_annotation(&context, &annotation)
            .expect("should translate a reverse-edge source interval");
        let tree = translated
            .cached_interval_tree
            .expect("should cache translated reverse-edge intervals");
        let mut blocks = tree
            .iter()
            .map(|interval| {
                (
                    interval.value.start,
                    interval.value.end,
                    interval.value.node_id,
                    interval.value.sequence_start,
                    interval.value.sequence_end,
                    interval.value.strand,
                )
            })
            .collect::<Vec<_>>();
        blocks.sort_by_key(|block| block.0);
        assert_eq!(
            blocks,
            vec![
                (0, 1, node_b_id, 0, 1, Strand::Forward),
                (1, 2, node_a_id, 8, 9, Strand::Forward),
            ]
        );
    }
}
