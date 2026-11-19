use gen_core::{HashId, NO_CHROMOSOME_INDEX, PathBlock, Workspace};
use gen_models::{
    block_group::{BlockGroup, BlockGroupChange},
    db::GraphConnection,
    edge::Edge,
    errors::BlockGroupError,
    path::Path,
    region::{
        GenRegionError, Region, ResolvedGenRegion, ResolvedRegionKind, resolve_accession,
        resolve_annotation, resolve_block_group, resolve_path,
    },
};

pub mod fasta;
pub mod gaf;
pub mod genbank;
pub mod gfa;
pub mod library;
pub mod sequence;
pub mod vcf;

pub(crate) fn resolve_sequence_update_region(
    region: &Region,
    conn: &GraphConnection,
    collection_name: &str,
    sample_name: &str,
) -> Result<ResolvedGenRegion, GenRegionError> {
    match resolve_block_group(region, conn, collection_name, sample_name) {
        Ok(mut resolved) => {
            // Sequence edits use the current path's coordinates for a block-group target.
            resolved.kind = ResolvedRegionKind::Path;
            Ok(resolved)
        }
        Err(GenRegionError::NotFound(_)) => {
            resolve_update_region(region, conn, collection_name, sample_name)
        }
        Err(error) => Err(error),
    }
}

pub(crate) fn child_update_path(
    conn: &GraphConnection,
    region: &ResolvedGenRegion,
    target_block_group_id: HashId,
) -> Result<Option<Path>, GenRegionError> {
    if region.kind != ResolvedRegionKind::Path {
        return Ok(None);
    }
    let parent_path = region
        .path
        .as_ref()
        .ok_or_else(|| GenRegionError::NotFound("missing resolved path".to_string()))?;
    let child_path = BlockGroup::get_path_by_name(conn, &target_block_group_id, &parent_path.name)?
        .ok_or_else(|| GenRegionError::NotFound(parent_path.name.clone()))?;
    Ok(Some(child_path))
}

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

pub(crate) struct InsertedPathEdges {
    pub incoming: Edge,
    pub outgoing: Edge,
}

pub(crate) fn insert_update_change(
    conn: &GraphConnection,
    workspace: &Workspace,
    region: ResolvedGenRegion,
    data: InsertChangeData,
) -> Result<Option<InsertedPathEdges>, BlockGroupError> {
    let change = BlockGroupChange {
        region,
        path_accession: data.path_accession,
        block: data.block,
        chromosome_index: data.chromosome_index,
        phased: data.phased,
        preserve_edge: data.preserve_edge,
    };
    let planned_edges = BlockGroup::insert_change_with_edges(conn, workspace, &change)?;
    if change.region.kind != ResolvedRegionKind::Path
        || change.block.sequence_start == change.block.sequence_end
    {
        return Ok(None);
    }

    // A reused node can have edges for several edit locations. Only use this
    // change's planned splice edges, excluding edges that preserve the old route.
    let load_edge = |incoming: bool| -> Result<Edge, BlockGroupError> {
        let mut edge_ids = planned_edges
            .iter()
            .filter(|edge| edge.chromosome_index == change.chromosome_index)
            .filter(|edge| {
                let edge = &edge.edge_data;
                if incoming {
                    edge.target_node_id == change.block.node_id
                        && edge.target_coordinate == change.block.sequence_start
                } else {
                    edge.source_node_id == change.block.node_id
                        && edge.source_coordinate == change.block.sequence_end
                }
            })
            .map(|edge| edge.edge_data.id_hash())
            .collect::<Vec<_>>();
        edge_ids.sort_unstable();
        edge_ids.dedup();
        if edge_ids.len() != 1 {
            return Err(BlockGroupError::ChangeOutOfBounds(
                "expected one splice edge for the selected path".to_string(),
            ));
        }
        Edge::select(conn)
            .query_by_ids(edge_ids)?
            .pop()
            .ok_or_else(|| {
                BlockGroupError::ChangeOutOfBounds("missing persisted splice edge".to_string())
            })
    };
    Ok(Some(InsertedPathEdges {
        incoming: load_edge(true)?,
        outgoing: load_edge(false)?,
    }))
}
