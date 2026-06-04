use gen_core::{HashId, NO_CHROMOSOME_INDEX, PathBlock};
use gen_models::{
    block_group::{BlockGroup, BlockGroupChange},
    db::GraphConnection,
    errors::BlockGroupError,
    path::Path,
    region::{
        GenRegionError, Region, ResolvedGenRegion, ResolvedRegionKind, resolve_accession,
        resolve_annotation, resolve_path,
    },
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
    target_region.block_group = BlockGroup::get_by_id(conn, &target_block_group_id)?;
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
    pub block_group_id: HashId,
    pub start: i64,
    pub end: i64,
    pub block: PathBlock,
    pub path_accession: Option<String>,
    pub chromosome_index: i64,
    pub phased: i64,
    pub preserve_edge: bool,
}

impl InsertChangeData {
    pub(crate) fn new(block_group_id: HashId, start: i64, end: i64, block: PathBlock) -> Self {
        Self {
            block_group_id,
            start,
            end,
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
        block_group_id: data.block_group_id,
        intervaltree_source: region,
        path_accession: data.path_accession,
        start: data.start,
        end: data.end,
        block: data.block,
        chromosome_index: data.chromosome_index,
        phased: data.phased,
        preserve_edge: data.preserve_edge,
    };
    BlockGroup::insert_change(conn, &change)
}
