use std::str;

use gen_core::{HashId, NO_CHROMOSOME_INDEX, PathBlock, Strand, region::RegionResolver};
use gen_models::{
    annotations::Annotation,
    block_group::{AccessionChange, AnnotationChange, BlockGroup, PathChange},
    db::DbContext,
    edge::Edge,
    node::Node,
    operations::{Operation, OperationInfo},
    region::{GenRegionError, Region, ResolvedGenRegion, ResolvedRegionKind, resolve},
    sample::Sample,
    sequence::Sequence,
    traits::*,
};
use rusqlite::{self, params};

use crate::errors::SequenceUpdateError;

#[allow(clippy::too_many_arguments)]
pub fn update_with_sequence(
    context: &DbContext,
    collection_name: &str,
    parent_sample_name: &str,
    new_sample_name: &str,
    region_name: &str,
    sequence: &str,
    disable_reference_path_update: bool,
) -> Result<Operation, SequenceUpdateError> {
    let conn = context.graph().conn();
    let mut session = gen_models::session_operations::start_operation(conn);
    let parsed_region = Region::parse(region_name).map_err(GenRegionError::from)?;
    let resolved_region = match resolve(&parsed_region, conn, collection_name, parent_sample_name) {
        Ok(region) => region,
        Err(GenRegionError::NotFound(_)) => {
            return Err(GenRegionError::NotFound(region_name.to_string()).into());
        }
        Err(err) => return Err(err.into()),
    };
    let source_annotation = if resolved_region.kind == ResolvedRegionKind::Annotation {
        Some(
            match Annotation::resolve(&parsed_region, conn, collection_name, parent_sample_name) {
                Ok(annotation) => annotation,
                Err(gen_core::region::RegionResolutionError::NotFound(_)) => {
                    return Err(GenRegionError::NotFound(region_name.to_string()).into());
                }
                Err(gen_core::region::RegionResolutionError::Ambiguous(name)) => {
                    return Err(GenRegionError::Ambiguous(name).into());
                }
                Err(gen_core::region::RegionResolutionError::Lookup(err)) => {
                    return Err(GenRegionError::from(err).into());
                }
            },
        )
    } else {
        None
    };
    let has_coordinates = parsed_region.start.is_some() || parsed_region.end.is_some();
    if !has_coordinates
        && matches!(
            resolved_region.kind,
            ResolvedRegionKind::Path | ResolvedRegionKind::BlockGroup
        )
    {
        return Err(SequenceUpdateError::MissingCoordinates(
            region_name.to_string(),
        ));
    }
    let _new_sample = Sample::get_or_create_child(
        conn,
        collection_name,
        new_sample_name,
        vec![parent_sample_name.to_string()],
    )?;
    let block_groups = Sample::get_block_groups(conn, collection_name, parent_sample_name);

    let mut target_block_groups = vec![];
    for block_group in block_groups {
        let new_block_groups = BlockGroup::get_or_create_sample_block_groups(
            conn,
            collection_name,
            new_sample_name,
            &block_group.name,
            vec![parent_sample_name.to_string()],
        )?;

        if block_group.name == resolved_region.block_group.name {
            target_block_groups = new_block_groups;
        }
    }

    if target_block_groups.is_empty() {
        return Err(GenRegionError::NotFound(region_name.to_string()).into());
    }

    for target_block_group in &target_block_groups {
        let path = if matches!(
            resolved_region.kind,
            ResolvedRegionKind::Path | ResolvedRegionKind::BlockGroup
        ) || !disable_reference_path_update
        {
            Some(BlockGroup::get_current_path(conn, &target_block_group.id))
        } else {
            None
        };
        let node_id = if sequence.is_empty() {
            HashId::convert_str("")
        } else {
            let seq = Sequence::new()
                .sequence_type("DNA")
                .sequence(sequence)
                .save(conn)?;
            Node::create(
                conn,
                &seq.hash,
                &HashId::convert_str(&format!(
                    "{target_id}:{ref_start}-{ref_end}->{sequence_hash}",
                    target_id = target_block_group.id,
                    ref_start = 0,
                    ref_end = seq.length,
                    sequence_hash = seq.hash
                )),
            )?
        };
        let path_block = PathBlock {
            node_id,
            block_sequence: sequence.to_string(),
            sequence_start: 0,
            sequence_end: if sequence.is_empty() {
                0
            } else {
                sequence.len() as i64
            },
            path_start: resolved_region.start,
            path_end: resolved_region.end,
            strand: Strand::Forward,
        };

        match resolved_region.kind {
            ResolvedRegionKind::Path | ResolvedRegionKind::BlockGroup => {
                let path = path.as_ref().ok_or_else(|| {
                    GenRegionError::Unmappable(resolved_region.block_group.name.clone())
                })?;
                let path_change = PathChange {
                    block_group_id: target_block_group.id,
                    intervaltree_source: path.clone(),
                    path_accession: None,
                    start: resolved_region.start,
                    end: resolved_region.end,
                    block: path_block,
                    chromosome_index: NO_CHROMOSOME_INDEX,
                    phased: 0,
                    preserve_edge: true,
                };
                BlockGroup::insert_change(conn, &path_change)?;
            }
            ResolvedRegionKind::Annotation => {
                let annotation = source_annotation.as_ref().ok_or_else(|| {
                    GenRegionError::Unmappable(resolved_region.block_group.name.clone())
                })?;
                let change = AnnotationChange {
                    block_group_id: target_block_group.id,
                    intervaltree_source: annotation.clone(),
                    path_accession: None,
                    start: resolved_region.start,
                    end: resolved_region.end,
                    block: path_block,
                    chromosome_index: NO_CHROMOSOME_INDEX,
                    phased: 0,
                    preserve_edge: true,
                };
                BlockGroup::insert_change(conn, &change)?;
            }
            ResolvedRegionKind::Accession => {
                let resolved_accession = resolved_region.accession.as_ref().ok_or_else(|| {
                    GenRegionError::Unmappable(resolved_region.block_group.name.clone())
                })?;
                let change = AccessionChange {
                    block_group_id: target_block_group.id,
                    intervaltree_source: resolved_accession.clone(),
                    path_accession: None,
                    start: resolved_region.start,
                    end: resolved_region.end,
                    block: path_block,
                    chromosome_index: NO_CHROMOSOME_INDEX,
                    phased: 0,
                    preserve_edge: true,
                };
                BlockGroup::insert_change(conn, &change)?;
            }
        }

        if !disable_reference_path_update
            && matches!(
                resolved_region.kind,
                ResolvedRegionKind::Path | ResolvedRegionKind::BlockGroup
            )
        {
            let (start_coordinate, end_coordinate) =
                resolved_path_update_coordinates(&resolved_region);
            let path = path
                .as_ref()
                .expect("path updates require an existing current path");
            if node_id == HashId::convert_str("") {
                let _ = path.new_path_with_deletion(conn, start_coordinate, end_coordinate);
            } else {
                let edge_to_new_node = Edge::query(
                    conn,
                    "select * from edges where target_node_id = ?1",
                    params![node_id],
                )[0]
                .clone();
                let edge_from_new_node = Edge::query(
                    conn,
                    "select * from edges where source_node_id = ?1",
                    params![node_id],
                )[0]
                .clone();
                path.new_path_with(
                    conn,
                    start_coordinate,
                    end_coordinate,
                    &edge_to_new_node,
                    &edge_from_new_node,
                )?;
            }
        }
    }

    let summary_str =
        format!("Sequences {mod}", mod=if sequence.is_empty() { "deleted" } else { "inserted" });
    let op = gen_models::session_operations::end_operation(
        context,
        &mut session,
        &OperationInfo {
            files: vec![],
            description: "fasta_update".to_string(),
        },
        &summary_str,
        None,
    )
    .unwrap();

    println!("Updated with sequence.");

    Ok(op)
}

fn resolved_path_update_coordinates(resolved_region: &ResolvedGenRegion) -> (i64, i64) {
    (resolved_region.start, resolved_region.end)
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, path::PathBuf};

    use gen_core::{PATH_END_NODE_ID, PATH_START_NODE_ID};
    use gen_models::{
        accession::AccessionError,
        annotations::Annotation,
        block_group::{NewBlockGroup, PathCache},
        block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
        collection::Collection,
        path::Path,
        sample::NewSample,
    };

    use super::*;
    use crate::{
        imports::fasta::import_fasta,
        test_helpers::{get_sample_bg, setup_gen},
        track_database,
    };

    fn import_simple_fasta(context: &DbContext, collection: &str) {
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        import_fasta(
            context,
            &fasta_path.to_str().unwrap().to_string(),
            collection,
            Sample::DEFAULT_NAME,
            false,
        )
        .unwrap();
    }

    fn add_multinode_annotation_fixture(context: &DbContext, collection: &str) {
        let conn = context.graph().conn();
        Collection::get_or_create(conn, collection).unwrap();
        Sample::get_or_create(
            conn,
            NewSample {
                name: Sample::DEFAULT_NAME,
                ..Default::default()
            },
        )
        .unwrap();

        let block_group = BlockGroup::create(
            conn,
            NewBlockGroup {
                collection_name: collection,
                sample_name: Sample::DEFAULT_NAME,
                name: "chr1",
                ..Default::default()
            },
        )
        .unwrap();
        let a_seq = Sequence::new()
            .sequence_type("DNA")
            .sequence("AAAAA")
            .save(conn)
            .unwrap();
        let a_node_id =
            Node::create(conn, &a_seq.hash, &HashId::convert_str("anno-a-node")).unwrap();
        let t_seq = Sequence::new()
            .sequence_type("DNA")
            .sequence("TTTTT")
            .save(conn)
            .unwrap();
        let t_node_id =
            Node::create(conn, &t_seq.hash, &HashId::convert_str("anno-t-node")).unwrap();
        let c_seq = Sequence::new()
            .sequence_type("DNA")
            .sequence("CCCCCCCC")
            .save(conn)
            .unwrap();
        let c_node_id =
            Node::create(conn, &c_seq.hash, &HashId::convert_str("anno-c-node")).unwrap();
        let g_seq = Sequence::new()
            .sequence_type("DNA")
            .sequence("GGGGGGGG")
            .save(conn)
            .unwrap();
        let g_node_id =
            Node::create(conn, &g_seq.hash, &HashId::convert_str("anno-g-node")).unwrap();
        let atcg_seq = Sequence::new()
            .sequence_type("DNA")
            .sequence("ATCGATCG")
            .save(conn)
            .unwrap();
        let atcg_node_id =
            Node::create(conn, &atcg_seq.hash, &HashId::convert_str("anno-atcg-node")).unwrap();

        let start_to_a = Edge::create(
            conn,
            PATH_START_NODE_ID,
            0,
            Strand::Forward,
            a_node_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let start_to_t = Edge::create(
            conn,
            PATH_START_NODE_ID,
            0,
            Strand::Forward,
            t_node_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let a_to_c = Edge::create(
            conn,
            a_node_id,
            5,
            Strand::Forward,
            c_node_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let t_to_c = Edge::create(
            conn,
            t_node_id,
            5,
            Strand::Forward,
            c_node_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let c_to_g = Edge::create(
            conn,
            c_node_id,
            8,
            Strand::Forward,
            g_node_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let g_to_atcg = Edge::create(
            conn,
            g_node_id,
            8,
            Strand::Forward,
            atcg_node_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let atcg_to_end = Edge::create(
            conn,
            atcg_node_id,
            8,
            Strand::Forward,
            PATH_END_NODE_ID,
            0,
            Strand::Forward,
        )
        .unwrap();

        BlockGroupEdge::bulk_create(
            conn,
            &[
                BlockGroupEdgeData {
                    block_group_id: block_group.id,
                    edge_id: start_to_a.id,
                    chromosome_index: 0,
                    phased: 0,
                },
                BlockGroupEdgeData {
                    block_group_id: block_group.id,
                    edge_id: start_to_t.id,
                    chromosome_index: 1,
                    phased: 0,
                },
                BlockGroupEdgeData {
                    block_group_id: block_group.id,
                    edge_id: a_to_c.id,
                    chromosome_index: 0,
                    phased: 0,
                },
                BlockGroupEdgeData {
                    block_group_id: block_group.id,
                    edge_id: t_to_c.id,
                    chromosome_index: 1,
                    phased: 0,
                },
                BlockGroupEdgeData {
                    block_group_id: block_group.id,
                    edge_id: c_to_g.id,
                    chromosome_index: 0,
                    phased: 0,
                },
                BlockGroupEdgeData {
                    block_group_id: block_group.id,
                    edge_id: g_to_atcg.id,
                    chromosome_index: 0,
                    phased: 0,
                },
                BlockGroupEdgeData {
                    block_group_id: block_group.id,
                    edge_id: atcg_to_end.id,
                    chromosome_index: 0,
                    phased: 0,
                },
            ],
        );

        let path = Path::create(
            conn,
            "chr1",
            &block_group.id,
            &[
                start_to_a.id,
                a_to_c.id,
                c_to_g.id,
                g_to_atcg.id,
                atcg_to_end.id,
            ],
        )
        .unwrap();
        let mut path_cache = PathCache::new(conn);
        let accession =
            BlockGroup::add_accession(conn, &path, "span-annotation", 10, 24, &mut path_cache)
                .unwrap();
        Annotation::get_or_create(conn, "gene-span", "genes", &accession.id, None).unwrap();
    }

    fn add_mreb_accession_and_annotation(context: &DbContext, collection: &str) {
        let conn = context.graph().conn();
        let block_group = get_sample_bg(conn, collection, Sample::DEFAULT_NAME);
        let path = BlockGroup::get_current_path(conn, &block_group.id);
        let mut path_cache = PathCache::new(conn);
        let accession =
            BlockGroup::add_accession(conn, &path, "mreB", 5, 20, &mut path_cache).unwrap();
        Annotation::get_or_create(conn, "gene-mreB", "genes", &accession.id, None).unwrap();
    }

    #[test]
    fn test_update_with_sequence() {
        /*
        Graph after sequence update:
        AT ----> CGA ------> TCGATCGATCGATCGGGAACACACAGAGA
           \-> AAAAAAAA --/
        */
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();
        track_database(conn, op_conn).unwrap();

        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
        )
        .unwrap();
        let _ = update_with_sequence(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "m123:2-5",
            "AAAAAAAA",
            false,
        );

        let expected_sequences = vec![
            "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
            "ATAAAAAAAATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
        ];
        let block_groups = BlockGroup::query(
            conn,
            "select * from block_groups where collection_name = ?1 AND sample_name = ?2;",
            params![collection, "child sample"],
        );
        assert_eq!(block_groups.len(), 1);
        assert_eq!(
            Sample::get_parent_names(conn, "child sample"),
            vec![Sample::DEFAULT_NAME.to_string()]
        );
        assert_eq!(
            BlockGroup::get_all_sequences(conn, &block_groups[0].id, false),
            HashSet::from_iter(expected_sequences),
        );
    }

    #[test]
    fn test_update_with_accession_region() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();
        track_database(conn, op_conn).unwrap();

        let collection = "test".to_string();
        import_simple_fasta(&context, &collection);
        add_mreb_accession_and_annotation(&context, &collection);

        update_with_sequence(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "mreB",
            "TTTT",
            false,
        )
        .unwrap();

        assert_eq!(
            BlockGroup::get_all_sequences(
                conn,
                &get_sample_bg(conn, &collection, "child sample").id,
                false
            ),
            HashSet::from_iter(vec![
                "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
                "ATCGATTTTGGAACACACAGAGA".to_string(),
            ])
        );
    }

    #[test]
    fn test_update_with_annotation_region() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();
        track_database(conn, op_conn).unwrap();

        let collection = "test".to_string();
        import_simple_fasta(&context, &collection);
        add_mreb_accession_and_annotation(&context, &collection);

        update_with_sequence(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "gene-mreB",
            "GGGG",
            false,
        )
        .unwrap();

        assert_eq!(
            BlockGroup::get_all_sequences(
                conn,
                &get_sample_bg(conn, &collection, "child sample").id,
                false
            ),
            HashSet::from_iter(vec![
                "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
                "ATCGAGGGGGGAACACACAGAGA".to_string(),
            ])
        );
    }

    #[test]
    fn test_update_with_accession_negative_index() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();
        track_database(conn, op_conn).unwrap();

        let collection = "test".to_string();
        import_simple_fasta(&context, &collection);
        add_mreb_accession_and_annotation(&context, &collection);

        update_with_sequence(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "mreb:-5",
            "NNNN",
            false,
        )
        .unwrap();

        assert_eq!(
            BlockGroup::get_all_sequences(
                conn,
                &get_sample_bg(conn, &collection, "child sample").id,
                false
            ),
            HashSet::from_iter(vec![
                "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
                "NNNNATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
            ])
        );
    }

    #[test]
    fn test_update_with_accession_region_after_path_diverges() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();
        track_database(conn, op_conn).unwrap();

        let collection = "test".to_string();
        import_simple_fasta(&context, &collection);
        add_mreb_accession_and_annotation(&context, &collection);

        update_with_sequence(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "mreb:-5",
            "NNNN",
            false,
        )
        .unwrap();

        let err = update_with_sequence(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "mreB",
            "TTTT",
            false,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            SequenceUpdateError::BlockGroupError(gen_models::errors::BlockGroupError::AccessionError(
                AccessionError::NotFound(ref name)
            )) if name == "mreB"
        ));
        assert_eq!(
            BlockGroup::get_all_sequences(
                conn,
                &get_sample_bg(conn, &collection, "child sample").id,
                false
            ),
            HashSet::from_iter(vec![
                "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
                "NNNNATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
            ])
        );
    }

    #[test]
    fn test_update_with_annotation_region_on_blockgroup_with_no_paths() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();
        track_database(conn, op_conn).unwrap();

        let collection = "test".to_string();
        import_simple_fasta(&context, &collection);
        add_mreb_accession_and_annotation(&context, &collection);

        Sample::get_or_create_child(
            conn,
            &collection,
            "child sample",
            vec![Sample::DEFAULT_NAME.to_string()],
        )
        .unwrap();

        let parent_block_group = get_sample_bg(conn, &collection, Sample::DEFAULT_NAME);
        let child_block_groups = BlockGroup::get_or_create_sample_block_groups(
            conn,
            &collection,
            "child sample",
            &parent_block_group.name,
            vec![Sample::DEFAULT_NAME.to_string()],
        )
        .unwrap();
        assert_eq!(child_block_groups.len(), 1);

        let child_paths = Path::query(
            conn,
            "select * from paths where block_group_id = ?1",
            params![child_block_groups[0].id],
        );
        for child_path in &child_paths {
            conn.execute(
                "delete from path_edges where path_id = ?1",
                params![child_path.id],
            )
            .unwrap();
        }
        conn.execute(
            "delete from paths where block_group_id = ?1",
            params![child_block_groups[0].id],
        )
        .unwrap();

        assert!(
            Path::query(
                conn,
                "select * from paths where block_group_id = ?1",
                params![child_block_groups[0].id],
            )
            .is_empty()
        );

        update_with_sequence(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "gene-mreB",
            "GGGG",
            true,
        )
        .unwrap();

        assert!(
            Path::query(
                conn,
                "select * from paths where block_group_id = ?1",
                params![child_block_groups[0].id],
            )
            .is_empty()
        );
        assert_eq!(
            BlockGroup::get_all_sequences(conn, &child_block_groups[0].id, false),
            HashSet::from_iter(vec![
                "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
                "ATCGAGGGGGGAACACACAGAGA".to_string(),
            ])
        );
    }

    #[test]
    fn test_update_with_annotation_negative_index_spanning_multiple_nodes() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();
        track_database(conn, op_conn).unwrap();

        let collection = "test".to_string();
        add_multinode_annotation_fixture(&context, &collection);

        Sample::get_or_create_child(
            conn,
            &collection,
            "child sample",
            vec![Sample::DEFAULT_NAME.to_string()],
        )
        .unwrap();

        let parent_block_group = get_sample_bg(conn, &collection, Sample::DEFAULT_NAME);
        let child_block_groups = BlockGroup::get_or_create_sample_block_groups(
            conn,
            &collection,
            "child sample",
            &parent_block_group.name,
            vec![Sample::DEFAULT_NAME.to_string()],
        )
        .unwrap();
        assert_eq!(child_block_groups.len(), 1);

        let child_paths = Path::query(
            conn,
            "select * from paths where block_group_id = ?1",
            params![child_block_groups[0].id],
        );
        for child_path in &child_paths {
            conn.execute(
                "delete from path_edges where path_id = ?1",
                params![child_path.id],
            )
            .unwrap();
        }
        conn.execute(
            "delete from paths where block_group_id = ?1",
            params![child_block_groups[0].id],
        )
        .unwrap();

        update_with_sequence(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "gene-span:-10",
            "NNNN",
            true,
        )
        .unwrap();

        assert!(
            Path::query(
                conn,
                "select * from paths where block_group_id = ?1",
                params![child_block_groups[0].id],
            )
            .is_empty()
        );
        assert_eq!(
            BlockGroup::get_all_sequences(conn, &child_block_groups[0].id, false),
            HashSet::from_iter(vec![
                "AAAAACCCCCCCCGGGGGGGGATCGATCG".to_string(),
                "TTTTTCCCCCCCCGGGGGGGGATCGATCG".to_string(),
                "NNNNAAAAACCCCCCCCGGGGGGGGATCGATCG".to_string(),
                "NNNNTTTTTCCCCCCCCGGGGGGGGATCGATCG".to_string(),
            ])
        );
    }

    #[test]
    fn test_disable_reference_path_update() {
        // This tests if we stop updating the reference path if explicitly asked for when there
        // is a single insert occurring
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();
        track_database(conn, op_conn).unwrap();

        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
        )
        .unwrap();
        let _ = update_with_sequence(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "m123:2-5",
            "AAAAAAAA",
            false,
        );
        let _ = update_with_sequence(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "other sample",
            "m123:2-5",
            "AAAAAAAA",
            true,
        );

        let child_blockgroup = get_sample_bg(conn, &collection, "child sample").id;
        let other_blockgroup = get_sample_bg(conn, &collection, "other sample").id;
        let child_path = BlockGroup::get_current_path(conn, &child_blockgroup);
        let other_path = BlockGroup::get_current_path(conn, &other_blockgroup);
        assert_eq!(
            child_path.sequence(conn).unwrap(),
            "ATAAAAAAAATCGATCGATCGATCGGGAACACACAGAGA"
        );
        assert_eq!(
            other_path.sequence(conn).unwrap(),
            "ATCGATCGATCGATCGATCGGGAACACACAGAGA"
        );
    }

    #[test]
    fn test_update_within_update() {
        /*
        Graph after sequence updates:
        AT --------------> CGA ----------------> TCGATCGATCGATCGGGAACACACAGAGA
            \-> AA -----> AA -------> AAAA --/
                   \--> TTTTTTTT --/
        */
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();
        track_database(conn, op_conn).unwrap();

        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let collection = "test".to_string();

        let _ = import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
        )
        .unwrap();
        let _ = update_with_sequence(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "m123:2-5",
            "AAAAAAAA",
            false,
        );
        // Second sequence update replacing part of the first update sequence
        let _ = update_with_sequence(
            &context,
            &collection,
            "child sample",
            "grandchild sample",
            "m123:4-6",
            "TTTTTTTT",
            false,
        );
        let expected_sequences = vec![
            "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
            "ATAAAAAAAATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
            "ATAATTTTTTTTAAAATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
        ];
        let block_groups = BlockGroup::query(
            conn,
            "select * from block_groups where collection_name = ?1 AND sample_name = ?2;",
            params![collection, "grandchild sample"],
        );
        assert_eq!(block_groups.len(), 1);
        assert_eq!(
            BlockGroup::get_all_sequences(conn, &block_groups[0].id, false),
            HashSet::from_iter(expected_sequences),
        );
    }

    #[test]
    fn test_update_with_two_sequences_partial_leading_overlap() {
        /*
        Graph after sequence updates:
        A --> T --------------> CGA ----------------> TCGATCGATCGATCGGGAACACACAGAGA
         \       \-> AAAA -------> AAAA --/
          \--> TTTTTTTT --/
        */
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();
        track_database(conn, op_conn).unwrap();

        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
        )
        .unwrap();
        let _ = update_with_sequence(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "m123:2-5",
            "AAAAAAAA",
            false,
        );
        // Second sequence update replacing parts of both the original and first update sequences
        let _ = update_with_sequence(
            &context,
            &collection,
            "child sample",
            "grandchild sample",
            "m123:1-6",
            "TTTTTTTT",
            false,
        );
        let expected_sequences = vec![
            "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
            "ATAAAAAAAATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
            "ATTTTTTTTAAAATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
        ];
        let block_groups = BlockGroup::query(
            conn,
            "select * from block_groups where collection_name = ?1 AND sample_name = ?2;",
            params![collection, "grandchild sample"],
        );
        assert_eq!(block_groups.len(), 1);
        assert_eq!(
            BlockGroup::get_all_sequences(conn, &block_groups[0].id, false),
            HashSet::from_iter(expected_sequences),
        );
    }

    #[test]
    fn test_update_with_two_sequences_partial_trailing_overlap() {
        /*
        Graph after sequence updates:
        A --> T --------------> CGA ----------------> TC --> GATCGATCGATCGGGAACACACAGAGA
         \       \-----> AAAAAAAA ---------/             /
          \-------------> TTTTTTTT ---------------------/
        */
        /*
        Graph after sequence updates:
        AT --------------> CGA ------------> TC --> GATCGATCGATCGGGAACACACAGAGA
              \-> AAAA -------> AAAA ----/        /
                           \--> TTTTTTTT --------/
        */
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();
        track_database(conn, op_conn).unwrap();

        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
        )
        .unwrap();
        let _ = update_with_sequence(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "m123:2-5",
            "AAAAAAAA",
            false,
        );
        // Second sequence update replacing parts of both the original and first update sequences
        let _ = update_with_sequence(
            &context,
            &collection,
            "child sample",
            "grandchild sample",
            "m123:1-12",
            "TTTTTTTT",
            false,
        );
        let expected_sequences = vec![
            "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
            "ATAAAAAAAATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
            "ATTTTTTTTGATCGATCGATCGGGAACACACAGAGA".to_string(),
        ];
        let block_groups = BlockGroup::query(
            conn,
            "select * from block_groups where collection_name = ?1 AND sample_name = ?2;",
            params![collection, "grandchild sample"],
        );
        assert_eq!(block_groups.len(), 1);
        assert_eq!(
            BlockGroup::get_all_sequences(conn, &block_groups[0].id, false),
            HashSet::from_iter(expected_sequences),
        );
    }

    #[test]
    fn test_update_with_two_sequences_second_over_first() {
        /*
        Graph after sequence updates:
        AT --------------> CGA ------------> TC --> GATCGATCGATCGGGAACACACAGAGA
              \-> AAAA -------> AAAA ----/        /
                           \--> TTTTTTTT --------/
        */
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();
        track_database(conn, op_conn).unwrap();

        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
        )
        .unwrap();
        let _ = update_with_sequence(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "m123:2-5",
            "AAAAAAAA",
            false,
        );
        // Second sequence update replacing parts of both the original and first update sequences
        let _ = update_with_sequence(
            &context,
            &collection,
            "child sample",
            "grandchild sample",
            "m123:6-12",
            "TTTTTTTT",
            false,
        );
        let expected_sequences = vec![
            "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
            "ATAAAAAAAATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
            "ATAAAATTTTTTTTGATCGATCGATCGGGAACACACAGAGA".to_string(),
        ];
        let block_groups = BlockGroup::query(
            conn,
            "select * from block_groups where collection_name = ?1 AND sample_name = ?2;",
            params![collection, "grandchild sample"],
        );
        assert_eq!(block_groups.len(), 1);
        assert_eq!(
            BlockGroup::get_all_sequences(conn, &block_groups[0].id, false),
            HashSet::from_iter(expected_sequences),
        );
    }

    #[test]
    fn test_update_with_same_sequence_twice() {
        /*
        Graph after sequence updates:
        AT --------------> CGA ----------------> TCGATCGATCGATCGGGAACACACAGAGA
            \-> AA -----> AA -------> AAAA --/
                   \--> AAAAAAAA --/
        */
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();
        track_database(conn, op_conn).unwrap();

        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
        )
        .unwrap();
        let _ = update_with_sequence(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "m123:2-5",
            "AAAAAAAA",
            false,
        );
        // Same sequence second time
        let _ = update_with_sequence(
            &context,
            &collection,
            "child sample",
            "grandchild sample",
            "m123:4-6",
            "AAAAAAAA",
            false,
        );
        let expected_sequences = vec![
            "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
            "ATAAAAAAAATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
            "ATAAAAAAAAAAAAAATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
        ];
        let block_groups = BlockGroup::query(
            conn,
            "select * from block_groups where collection_name = ?1 AND sample_name = ?2;",
            params![collection, "grandchild sample"],
        );
        assert_eq!(block_groups.len(), 1);
        assert_eq!(
            BlockGroup::get_all_sequences(conn, &block_groups[0].id, false),
            HashSet::from_iter(expected_sequences),
        );
    }

    #[test]
    fn test_deletion() {
        /*
        Graph after sequence update:
        AT ----> CGA ------> TCGATCGATCGATCGGGAACACACAGAGA
           \-> -------- --/
        */
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();
        track_database(conn, op_conn).unwrap();

        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
        )
        .unwrap();
        let _ = update_with_sequence(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "m123:2-5",
            "",
            false,
        );

        let expected_sequences = vec![
            "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
            "ATTCGATCGATCGATCGGGAACACACAGAGA".to_string(),
        ];
        let block_groups = BlockGroup::query(
            conn,
            "select * from block_groups where collection_name = ?1 AND sample_name = ?2;",
            params![collection, "child sample"],
        );
        assert_eq!(block_groups.len(), 1);
        assert_eq!(
            BlockGroup::get_all_sequences(conn, &block_groups[0].id, false),
            HashSet::from_iter(expected_sequences),
        );

        let latest_path = BlockGroup::get_current_path(conn, &block_groups[0].id);
        assert_eq!(
            latest_path.sequence(conn).unwrap(),
            "ATTCGATCGATCGATCGGGAACACACAGAGA"
        );
    }
}
