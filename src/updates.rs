use gen_core::{HashId, NO_CHROMOSOME_INDEX, NodeIntervalBlock, PathBlock};
use gen_models::{
    accession::Accession,
    annotations::Annotation,
    block_group::{BlockGroup, BlockGroupChange, ChangeSource, IntervalTreeSource},
    db::GraphConnection,
    errors::BlockGroupError,
    path::Path,
    region::{
        GenRegionError, Region, ResolvedGenRegion, ResolvedRegionKind, resolve_accession,
        resolve_annotation, resolve_path,
    },
};
use intervaltree::IntervalTree;

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

pub(crate) enum UpdateChangeSource {
    Path(Path),
    Accession(Accession),
    Annotation(Annotation),
}

impl UpdateChangeSource {
    pub(crate) fn from_region(
        region: &ResolvedGenRegion,
        target_path: Option<&Path>,
    ) -> Result<Self, GenRegionError> {
        match region.kind {
            ResolvedRegionKind::Path => {
                let path = target_path
                    .ok_or_else(|| GenRegionError::NotFound("missing target path".to_string()))?;
                Ok(Self::Path(path.clone()))
            }
            ResolvedRegionKind::Accession => {
                let accession = region
                    .accession
                    .as_ref()
                    .ok_or_else(|| GenRegionError::NotFound(region.block_group.name.clone()))?;
                Ok(Self::Accession(accession.clone()))
            }
            ResolvedRegionKind::Annotation => {
                let annotation = region
                    .annotation
                    .as_ref()
                    .ok_or_else(|| GenRegionError::NotFound(region.block_group.name.clone()))?;
                Ok(Self::Annotation(annotation.clone()))
            }
            ResolvedRegionKind::BlockGroup => {
                Err(GenRegionError::NotFound(region.block_group.name.clone()))
            }
        }
    }
}

impl IntervalTreeSource for UpdateChangeSource {
    fn intervaltree(
        &self,
        conn: &GraphConnection,
    ) -> Result<IntervalTree<i64, NodeIntervalBlock>, BlockGroupError> {
        match self {
            UpdateChangeSource::Path(path) => IntervalTreeSource::intervaltree(path, conn),
            UpdateChangeSource::Accession(accession) => {
                IntervalTreeSource::intervaltree(accession, conn)
            }
            UpdateChangeSource::Annotation(annotation) => {
                IntervalTreeSource::intervaltree(annotation, conn)
            }
        }
    }
}

impl ChangeSource for UpdateChangeSource {
    fn plan_edges(
        conn: &GraphConnection,
        change: &BlockGroupChange<Self>,
        tree: Option<&IntervalTree<i64, NodeIntervalBlock>>,
    ) -> Result<Vec<gen_models::block_group_edge::AugmentedEdgeData>, BlockGroupError> {
        match &change.intervaltree_source {
            UpdateChangeSource::Path(path) => {
                let change = BlockGroupChange {
                    block_group_id: change.block_group_id,
                    intervaltree_source: path.clone(),
                    path_accession: change.path_accession.clone(),
                    start: change.start,
                    end: change.end,
                    block: change.block.clone(),
                    chromosome_index: change.chromosome_index,
                    phased: change.phased,
                    preserve_edge: change.preserve_edge,
                };
                Path::plan_edges(conn, &change, tree)
            }
            UpdateChangeSource::Accession(accession) => {
                let change = BlockGroupChange {
                    block_group_id: change.block_group_id,
                    intervaltree_source: accession.clone(),
                    path_accession: change.path_accession.clone(),
                    start: change.start,
                    end: change.end,
                    block: change.block.clone(),
                    chromosome_index: change.chromosome_index,
                    phased: change.phased,
                    preserve_edge: change.preserve_edge,
                };
                Accession::plan_edges(conn, &change, tree)
            }
            UpdateChangeSource::Annotation(annotation) => {
                let change = BlockGroupChange {
                    block_group_id: change.block_group_id,
                    intervaltree_source: annotation.clone(),
                    path_accession: change.path_accession.clone(),
                    start: change.start,
                    end: change.end,
                    block: change.block.clone(),
                    chromosome_index: change.chromosome_index,
                    phased: change.phased,
                    preserve_edge: change.preserve_edge,
                };
                Annotation::plan_edges(conn, &change, tree)
            }
        }
    }
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

pub(crate) fn insert_update_change<T: ChangeSource>(
    conn: &GraphConnection,
    intervaltree_source: T,
    data: InsertChangeData,
) -> Result<(), BlockGroupError> {
    let change = BlockGroupChange {
        block_group_id: data.block_group_id,
        intervaltree_source,
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
