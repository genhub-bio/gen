use gen_core::{HashId, NO_CHROMOSOME_INDEX, PathBlock, Workspace};
use gen_models::{
    block_group::{BlockGroup, BlockGroupChange},
    db::GraphConnection,
    edge::Edge,
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

#[cfg(test)]
mod tests {
    use gen_core::{HashId, PATH_START_NODE_ID, PathBlock, Strand};
    use gen_models::{node::Node, path::Path, region::ResolvedGenRegion, sequence::Sequence};

    use super::{InsertChangeData, insert_update_change};
    use crate::test_helpers::{setup_block_group, setup_gen};

    // Both edits use the GG slice at 1..3 of the same node.
    // Earlier  ... --> [TT] --> [GG] --> [T x6] --> ...
    // Prefix   [START:0] -----> [GG] --> [A x8] --> ...
    #[test]
    fn test_insert_update_change_returns_splice_edges_for_reused_node() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let (_, path) = setup_block_group(conn);
        let path_edges = Path::edges_for_path(conn, &path.id, None);
        let first_node = path_edges[0].target_node_id;
        let second_node = path_edges[1].target_node_id;
        let sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("CGGA")
            .save(conn)
            .unwrap();
        let node_id = Node::create(conn, &sequence.hash, &HashId::convert_str("reused")).unwrap();

        for (start, end, source_node, source_offset, target_node, target_offset) in [
            (12, 14, second_node, 2, second_node, 4),
            (0, 2, PATH_START_NODE_ID, 0, first_node, 2),
        ] {
            let region = ResolvedGenRegion::from_path(conn, &path, start, end).unwrap();
            let data = InsertChangeData::new(PathBlock {
                node_id,
                block_sequence: "GG".to_string(),
                sequence_start: 1,
                sequence_end: 3,
                path_start: start,
                path_end: end,
                strand: Strand::Forward,
            });
            let edges = insert_update_change(conn, context.workspace(), region, data)
                .unwrap()
                .unwrap();

            assert_eq!(
                (
                    edges.incoming.source_node_id,
                    edges.incoming.source_coordinate,
                    edges.incoming.target_node_id,
                    edges.incoming.target_coordinate,
                ),
                (source_node, source_offset, node_id, 1),
                "incoming edge for {start}..{end}"
            );
            assert_eq!(
                (
                    edges.outgoing.source_node_id,
                    edges.outgoing.source_coordinate,
                    edges.outgoing.target_node_id,
                    edges.outgoing.target_coordinate,
                ),
                (node_id, 3, target_node, target_offset),
                "outgoing edge for {start}..{end}"
            );
        }
    }
}
