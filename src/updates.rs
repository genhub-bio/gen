use gen_core::{HashId, NO_CHROMOSOME_INDEX, PathBlock, Strand};
use gen_graph::{GraphNode, GraphNodePosition};
use gen_models::{
    block_group::{BlockGroup, BlockGroupChange},
    db::GraphConnection,
    errors::BlockGroupError,
    path::Path,
    region::{
        GenRegionError, Region, ResolvedGenRegion, ResolvedRegionKind, resolve_accession,
        resolve_annotation, resolve_path,
    },
    sample::Sample,
    sequence::Sequence,
    traits::Query,
};

pub mod fasta;
pub mod gaf;
pub mod genbank;
pub mod gfa;
pub mod library;
pub mod sequence;
pub mod vcf;

pub(crate) fn resolve_update_region(
    region: &Region,
    conn: &GraphConnection,
    collection_name: &str,
    sample_name: &str,
) -> Result<ResolvedGenRegion, GenRegionError> {
    match resolve_path(region, conn, collection_name, sample_name) {
        Ok(region) => return Ok(region),
        Err(GenRegionError::NotFound(_)) => {}
        Err(err) => return Err(err),
    }

    match resolve_annotation(region, conn, collection_name, sample_name) {
        Ok(region) => return Ok(region),
        Err(GenRegionError::NotFound(_)) => {}
        Err(err) => return Err(err),
    }

    resolve_accession(region, conn, collection_name, sample_name)
}

pub(crate) fn target_update_region(
    conn: &GraphConnection,
    region: &ResolvedGenRegion,
    target_block_group_id: HashId,
    target_path: Option<&Path>,
) -> Result<ResolvedGenRegion, GenRegionError> {
    let mut target_region = region.clone();
    target_region.block_group = BlockGroup::get_by_id(conn, &target_block_group_id, None)?;
    target_region.path = if region.kind == ResolvedRegionKind::Path {
        Some(
            target_path
                .ok_or_else(|| GenRegionError::NotFound("missing target path".to_string()))?
                .clone(),
        )
    } else {
        None
    };
    Ok(target_region)
}

pub(crate) struct InsertChangeData {
    pub block: PathBlock,
    pub path_accession: Option<String>,
    pub chromosome_index: i64,
    pub phased: i64,
    pub preserve_edge: bool,
}

impl InsertChangeData {
    pub(crate) fn new(block: PathBlock) -> Self {
        Self {
            block,
            path_accession: None,
            chromosome_index: NO_CHROMOSOME_INDEX,
            phased: 0,
            preserve_edge: true,
        }
    }
}

pub(crate) fn insert_update_change(
    conn: &GraphConnection,
    region: ResolvedGenRegion,
    data: InsertChangeData,
) -> Result<(), BlockGroupError> {
    let change = BlockGroupChange {
        region,
        path_accession: data.path_accession,
        block: data.block,
        chromosome_index: data.chromosome_index,
        phased: data.phased,
        preserve_edge: data.preserve_edge,
    };
    BlockGroup::insert_change(conn, &change)
}

pub(crate) struct SequenceUpdate<'a> {
    pub collection_name: &'a str,
    pub parent_sample_name: &'a str,
    pub new_sample_name: &'a str,
    pub region: &'a ResolvedGenRegion,
    pub disable_reference_path_update: bool,
}

pub(crate) fn prepare_path_update_region(
    conn: &GraphConnection,
    region: &mut ResolvedGenRegion,
) -> Result<(), BlockGroupError> {
    if region.kind != ResolvedRegionKind::Path
        || region.start <= 0
        || region.end >= region.feature_length
    {
        return Ok(());
    }
    let path = region
        .path
        .as_ref()
        .expect("should have a path for a path region");
    let interval_tree = path.intervaltree(conn)?;
    let start_block = interval_tree
        .query_point(region.start)
        .next()
        .ok_or_else(|| BlockGroupError::ChangeOutOfBounds("missing path start block".to_string()))?
        .value;
    let end_coordinate = if region.end == region.start {
        region.end
    } else {
        region.end - 1
    };
    let end_block = interval_tree
        .query_point(end_coordinate)
        .next()
        .ok_or_else(|| BlockGroupError::ChangeOutOfBounds("missing path end block".to_string()))?
        .value;
    let starts_at_node_boundary =
        region.start - start_block.start + start_block.sequence_start == start_block.sequence_start;
    let ends_at_node_boundary =
        region.end - end_block.start + end_block.sequence_start == end_block.sequence_end;
    if !starts_at_node_boundary && !ends_at_node_boundary {
        return Ok(());
    }
    region.start_anchors = Some(vec![GraphNodePosition {
        graph_node: GraphNode {
            node_id: start_block.node_id,
            sequence_start: start_block.sequence_start,
            sequence_end: start_block.sequence_end,
        },
        offset: region.start - start_block.start,
    }]);
    region.end_anchors = Some(vec![GraphNodePosition {
        graph_node: GraphNode {
            node_id: end_block.node_id,
            sequence_start: end_block.sequence_start,
            sequence_end: end_block.sequence_end,
        },
        offset: region.end - end_block.start,
    }]);
    Ok(())
}

pub(crate) fn apply_sequence_updates<E>(
    conn: &GraphConnection,
    update: &SequenceUpdate<'_>,
    sequences: impl IntoIterator<Item = String>,
) -> Result<usize, E>
where
    E: From<gen_models::errors::BlockGroupError>
        + From<gen_models::errors::NodeError>
        + From<gen_models::errors::PathError>
        + From<gen_models::errors::QueryError>
        + From<gen_models::errors::SampleError>
        + From<gen_models::errors::SequenceError>
        + From<GenRegionError>,
{
    Sample::get_or_create_child(
        conn,
        update.collection_name,
        update.new_sample_name,
        vec![update.parent_sample_name.to_string()],
    )?;
    let block_groups = Sample::get_block_groups(
        conn,
        update.collection_name,
        update.parent_sample_name,
        None,
    );
    let mut target_block_groups = Vec::new();
    for block_group in block_groups {
        let new_block_groups = BlockGroup::get_or_create_sample_block_groups(
            conn,
            update.collection_name,
            update.new_sample_name,
            &block_group.name,
            vec![update.parent_sample_name.to_string()],
        )?;
        if block_group.name == update.region.block_group.name {
            target_block_groups = new_block_groups;
        }
    }
    if target_block_groups.is_empty() {
        return Err(E::from(GenRegionError::NotFound(
            update.region.block_group.name.clone(),
        )));
    }

    let sequences = sequences.into_iter().collect::<Vec<_>>();
    let change_count = target_block_groups.len() * sequences.len();
    let update_reference_path = !update.disable_reference_path_update && sequences.len() == 1;
    for target_block_group in target_block_groups {
        let path = if update.region.kind == ResolvedRegionKind::Path {
            Some(BlockGroup::get_current_path(
                conn,
                &target_block_group.id,
                None,
            )?)
        } else {
            None
        };
        for sequence in &sequences {
            let (node_id, sequence_end) = if sequence.is_empty() {
                (HashId::convert_str(""), 0)
            } else {
                let saved_sequence = Sequence::new()
                    .sequence_type("DNA")
                    .sequence(sequence)
                    .save(conn)?;
                let node_id = gen_models::node::Node::create(
                    conn,
                    &saved_sequence.hash,
                    &HashId::convert_str(&format!(
                        "{block_group_id}:0-{sequence_end}->{sequence_hash}",
                        block_group_id = target_block_group.id,
                        sequence_end = saved_sequence.length,
                        sequence_hash = saved_sequence.hash,
                    )),
                )?;
                (node_id, saved_sequence.length)
            };
            let block = PathBlock {
                node_id,
                block_sequence: sequence.clone(),
                sequence_start: 0,
                sequence_end,
                path_start: update.region.start,
                path_end: update.region.end,
                strand: Strand::Forward,
            };
            let mut target_region =
                target_update_region(conn, update.region, target_block_group.id, path.as_ref())?;
            prepare_path_update_region(conn, &mut target_region)?;
            insert_update_change(conn, target_region, InsertChangeData::new(block))?;

            if update_reference_path && let Some(path) = &path {
                if sequence.is_empty() {
                    let _ =
                        path.new_path_with_deletion(conn, update.region.start, update.region.end);
                } else {
                    let edge_to_new_node = gen_models::edge::Edge::query(
                        conn,
                        "select * from edges where target_node_id = ?1",
                        rusqlite::params![node_id],
                    )[0]
                    .clone();
                    let edge_from_new_node = gen_models::edge::Edge::query(
                        conn,
                        "select * from edges where source_node_id = ?1",
                        rusqlite::params![node_id],
                    )[0]
                    .clone();
                    path.new_path_with(
                        conn,
                        update.region.start,
                        update.region.end,
                        &edge_to_new_node,
                        &edge_from_new_node,
                    )?;
                }
            }
        }
    }
    Ok(change_count)
}
