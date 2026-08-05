pub use gen_core::region::Region;
use gen_core::{
    GraphNodePosition, HashId, NodeIntervalBlock,
    region::{RegionParseError, RegionResolutionError, RegionResolver},
};
use gen_graph::graph_loader::{self, RegionPositionQuery};
use intervaltree::IntervalTree;
use thiserror::Error;

use crate::{
    accession::{Accession, AccessionError},
    annotations::{Annotation, AnnotationError},
    block_group::{BlockGroup, BlockGroupChange, BlockGroupError, IntervalTreeSource},
    block_group_edge::AugmentedEdgeData,
    db::GraphConnection,
    edge::EdgeData,
    errors::PathError,
    path::Path,
    traits::Query,
};

#[derive(Clone, Debug)]
pub struct ResolvedGenRegion {
    pub block_group: BlockGroup,
    pub path: Option<Path>,
    pub accession: Option<Accession>,
    pub annotation: Option<Annotation>,
    pub kind: ResolvedRegionKind,
    pub anchor_start: i64,
    pub anchor_end: i64,
    pub feature_length: i64,
    pub start: i64,
    pub end: i64,
    pub start_anchors: Option<Vec<GraphNodePosition>>,
    pub end_anchors: Option<Vec<GraphNodePosition>>,
    pub remove_ambiguous_positions: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ResolvedRegionKind {
    Path,
    Annotation,
    Accession,
    BlockGroup,
}

#[derive(Debug, Error)]
pub enum GenRegionError {
    #[error(transparent)]
    Parse(#[from] RegionParseError),
    #[error("Block group error: {0}")]
    BlockGroup(#[from] crate::block_group::BlockGroupError),
    #[error("Path error: {0}")]
    Path(#[from] PathError),
    #[error("Accession error: {0}")]
    Accession(#[from] AccessionError),
    #[error("Annotation error: {0}")]
    Annotation(#[from] AnnotationError),
    #[error("Region not found: {0}")]
    NotFound(String),
    #[error("Region is ambiguous: {0}")]
    Ambiguous(String),
    #[error(
        "Region {region} resolves to coordinates ({start}, {end}) outside path bounds (0-{path_length})"
    )]
    OutOfBounds {
        region: String,
        start: i64,
        end: i64,
        path_length: i64,
    },
    #[error("Failed to resolve {0} onto a path")]
    Unmappable(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RegionTargetKind {
    Path,
    Annotation,
    Accession,
    BlockGroup,
}

#[derive(Debug)]
struct RegionTarget {
    kind: RegionTargetKind,
    block_group: BlockGroup,
    path: Option<Path>,
    accession: Option<Accession>,
    annotation: Option<Annotation>,
    anchor_start: i64,
    anchor_end: i64,
    feature_length: i64,
}

pub fn resolve(
    region: &Region,
    conn: &GraphConnection,
    collection_name: &str,
    sample_name: &str,
) -> Result<ResolvedGenRegion, GenRegionError> {
    match resolve_block_group(region, conn, collection_name, sample_name) {
        Ok(region) => return Ok(region),
        Err(GenRegionError::NotFound(_)) => {}
        Err(err) => return Err(err),
    }

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

pub fn resolve_path(
    region: &Region,
    conn: &GraphConnection,
    collection_name: &str,
    sample_name: &str,
) -> Result<ResolvedGenRegion, GenRegionError> {
    let path = match Path::resolve(region, conn, collection_name, sample_name) {
        Ok(path) => path,
        Err(RegionResolutionError::NotFound(_)) => {
            return Err(GenRegionError::NotFound(region.name.clone()));
        }
        Err(RegionResolutionError::Ambiguous(name)) => return Err(GenRegionError::Ambiguous(name)),
        Err(RegionResolutionError::Lookup(err)) => return Err(err.into()),
    };
    let block_group = BlockGroup::get_by_id(conn, &path.block_group_id, None)?;
    let path_length = path.length(conn, None)?;
    resolve_target(
        region,
        RegionTarget {
            kind: RegionTargetKind::Path,
            block_group,
            path: Some(path),
            accession: None,
            annotation: None,
            anchor_start: 0,
            anchor_end: path_length,
            feature_length: path_length,
        },
    )
}

pub fn resolve_block_group(
    region: &Region,
    conn: &GraphConnection,
    collection_name: &str,
    sample_name: &str,
) -> Result<ResolvedGenRegion, GenRegionError> {
    let block_group = match BlockGroup::resolve(region, conn, collection_name, sample_name) {
        Ok(block_group) => block_group,
        Err(RegionResolutionError::NotFound(_)) => {
            return Err(GenRegionError::NotFound(region.name.clone()));
        }
        Err(RegionResolutionError::Ambiguous(name)) => return Err(GenRegionError::Ambiguous(name)),
        Err(RegionResolutionError::Lookup(err)) => return Err(err.into()),
    };
    let path = BlockGroup::get_current_path(conn, &block_group.id, None)?;
    let path_length = path.length(conn, None)?;
    resolve_target(
        region,
        RegionTarget {
            kind: RegionTargetKind::BlockGroup,
            block_group,
            path: Some(path),
            accession: None,
            annotation: None,
            anchor_start: 0,
            anchor_end: path_length,
            feature_length: path_length,
        },
    )
}

pub fn resolve_accession(
    region: &Region,
    conn: &GraphConnection,
    collection_name: &str,
    sample_name: &str,
) -> Result<ResolvedGenRegion, GenRegionError> {
    let accession = match Accession::resolve(region, conn, collection_name, sample_name) {
        Ok(accession) => accession,
        Err(RegionResolutionError::NotFound(_)) => {
            return Err(GenRegionError::NotFound(region.name.clone()));
        }
        Err(RegionResolutionError::Ambiguous(name)) => return Err(GenRegionError::Ambiguous(name)),
        Err(RegionResolutionError::Lookup(err)) => return Err(err.into()),
    };
    let target = target_from_accession(
        region,
        conn,
        collection_name,
        sample_name,
        RegionTargetKind::Accession,
        accession,
        None,
        true,
    )?;
    resolve_target(region, target)
}

pub fn resolve_annotation(
    region: &Region,
    conn: &GraphConnection,
    collection_name: &str,
    sample_name: &str,
) -> Result<ResolvedGenRegion, GenRegionError> {
    let annotation = match Annotation::resolve(region, conn, collection_name, sample_name) {
        Ok(annotation) => annotation,
        Err(RegionResolutionError::NotFound(_)) => {
            return Err(GenRegionError::NotFound(region.name.clone()));
        }
        Err(RegionResolutionError::Ambiguous(name)) => return Err(GenRegionError::Ambiguous(name)),
        Err(RegionResolutionError::Lookup(err)) => return Err(err.into()),
    };
    let accession = Accession::get_by_id(conn, &annotation.accession_id, None)
        .ok_or_else(|| GenRegionError::Unmappable(region.name.clone()))?;
    let target = target_from_accession(
        region,
        conn,
        collection_name,
        sample_name,
        RegionTargetKind::Annotation,
        accession,
        Some(annotation),
        false,
    )?;
    resolve_target(region, target)
}

fn resolve_target(
    region: &Region,
    target: RegionTarget,
) -> Result<ResolvedGenRegion, GenRegionError> {
    let (start, end) = match (region.start, region.end) {
        (None, None) => (target.anchor_start, target.anchor_end),
        (Some(start), None) => {
            if target.kind == RegionTargetKind::Path || target.kind == RegionTargetKind::BlockGroup
            {
                (start, target.feature_length)
            } else {
                (target.anchor_start + start, target.anchor_end)
            }
        }
        (Some(start), Some(end)) => {
            if target.kind == RegionTargetKind::Path || target.kind == RegionTargetKind::BlockGroup
            {
                (start, end)
            } else {
                (target.anchor_start + start, target.anchor_start + end)
            }
        }
        (None, Some(_)) => return Err(RegionParseError::InvalidSyntax.into()),
    };

    let out_of_bounds = match target.kind {
        RegionTargetKind::Path | RegionTargetKind::BlockGroup => {
            start < 0 || end > target.feature_length
        }
        RegionTargetKind::Annotation | RegionTargetKind::Accession => false,
    };

    if out_of_bounds {
        return Err(GenRegionError::OutOfBounds {
            region: region.to_string(),
            start,
            end,
            path_length: target.feature_length,
        });
    }

    Ok(ResolvedGenRegion {
        block_group: target.block_group,
        path: target.path,
        accession: target.accession,
        annotation: target.annotation,
        kind: match target.kind {
            RegionTargetKind::Path => ResolvedRegionKind::Path,
            RegionTargetKind::Annotation => ResolvedRegionKind::Annotation,
            RegionTargetKind::Accession => ResolvedRegionKind::Accession,
            RegionTargetKind::BlockGroup => ResolvedRegionKind::BlockGroup,
        },
        anchor_start: target.anchor_start,
        anchor_end: target.anchor_end,
        feature_length: target.feature_length,
        start,
        end,
        start_anchors: None,
        end_anchors: None,
        remove_ambiguous_positions: false,
    })
}

#[allow(clippy::too_many_arguments)]
fn target_from_accession(
    region: &Region,
    conn: &GraphConnection,
    collection_name: &str,
    sample_name: &str,
    kind: RegionTargetKind,
    accession: Accession,
    annotation: Option<Annotation>,
    require_current_sample: bool,
) -> Result<RegionTarget, GenRegionError> {
    let block_group = BlockGroup::get_by_id(conn, &accession.block_group_id, None)?;
    if block_group.collection_name != collection_name
        || (require_current_sample && block_group.sample_name != sample_name)
    {
        return Err(GenRegionError::NotFound(region.name.clone()));
    }
    let path_length = accession.length(conn)?;
    Ok(RegionTarget {
        kind,
        block_group,
        path: None,
        accession: Some(accession),
        annotation,
        anchor_start: 0,
        anchor_end: path_length,
        feature_length: path_length,
    })
}

impl ResolvedGenRegion {
    pub fn from_path(
        conn: &GraphConnection,
        block_group_id: HashId,
        path: &Path,
        start: i64,
        end: i64,
    ) -> Result<Self, BlockGroupError> {
        let block_group = BlockGroup::get_by_id(conn, &block_group_id, None)?;
        let path_length = path.length(conn, None)?;
        Ok(ResolvedGenRegion {
            block_group,
            path: Some(path.clone()),
            accession: None,
            annotation: None,
            kind: ResolvedRegionKind::Path,
            anchor_start: 0,
            anchor_end: path_length,
            feature_length: path_length,
            start,
            end,
            start_anchors: None,
            end_anchors: None,
            remove_ambiguous_positions: false,
        })
    }

    pub fn from_accession(
        conn: &GraphConnection,
        accession: &Accession,
        start: i64,
        end: i64,
    ) -> Result<Self, BlockGroupError> {
        let block_group = BlockGroup::get_by_id(conn, &accession.block_group_id, None)?;
        let accession_length = accession.length(conn)?;
        Ok(ResolvedGenRegion {
            block_group,
            path: None,
            accession: Some(accession.clone()),
            annotation: None,
            kind: ResolvedRegionKind::Accession,
            anchor_start: 0,
            anchor_end: accession_length,
            feature_length: accession_length,
            start,
            end,
            start_anchors: None,
            end_anchors: None,
            remove_ambiguous_positions: false,
        })
    }

    pub fn from_annotation(
        conn: &GraphConnection,
        annotation: &Annotation,
        accession: &Accession,
        start: i64,
        end: i64,
    ) -> Result<Self, BlockGroupError> {
        let block_group = BlockGroup::get_by_id(conn, &accession.block_group_id, None)?;
        let accession_length = accession.length(conn)?;
        Ok(ResolvedGenRegion {
            block_group,
            path: None,
            accession: Some(accession.clone()),
            annotation: Some(annotation.clone()),
            kind: ResolvedRegionKind::Annotation,
            anchor_start: 0,
            anchor_end: accession_length,
            feature_length: accession_length,
            start,
            end,
            start_anchors: None,
            end_anchors: None,
            remove_ambiguous_positions: false,
        })
    }

    pub fn intervaltree_cache_key(&self) -> (HashId, ResolvedRegionKind) {
        let id = match self.kind {
            ResolvedRegionKind::Path => {
                self.path
                    .as_ref()
                    .expect("should have path for Path region")
                    .id
            }
            ResolvedRegionKind::Accession | ResolvedRegionKind::Annotation => {
                self.accession
                    .as_ref()
                    .expect("should have accession for region")
                    .id
            }
            ResolvedRegionKind::BlockGroup => self.block_group.id,
        };
        (id, self.kind)
    }

    pub fn offset_range(
        &self,
        start_offset: i64,
        end_offset: i64,
    ) -> Result<(i64, i64), GenRegionError> {
        let (start, end) = match self.kind {
            ResolvedRegionKind::Annotation | ResolvedRegionKind::Accession => (
                self.anchor_start + start_offset,
                self.anchor_start + end_offset,
            ),
            ResolvedRegionKind::Path | ResolvedRegionKind::BlockGroup => (start_offset, end_offset),
        };

        let out_of_bounds = match self.kind {
            ResolvedRegionKind::Path | ResolvedRegionKind::BlockGroup => {
                start < 0 || end > self.feature_length
            }
            ResolvedRegionKind::Annotation | ResolvedRegionKind::Accession => false,
        };

        if out_of_bounds {
            return Err(GenRegionError::OutOfBounds {
                region: self
                    .path
                    .as_ref()
                    .map(|path| path.name.clone())
                    .unwrap_or_else(|| self.block_group.name.clone()),
                start,
                end,
                path_length: self.feature_length,
            });
        }

        Ok((start, end))
    }

    pub fn intervaltree(
        &self,
        conn: &GraphConnection,
    ) -> Result<IntervalTree<i64, NodeIntervalBlock>, GenRegionError> {
        match self.kind {
            ResolvedRegionKind::Path => {
                let path = self
                    .path
                    .as_ref()
                    .ok_or_else(|| GenRegionError::NotFound("No path for region".to_string()))?;
                Ok(path.intervaltree(conn)?)
            }
            ResolvedRegionKind::Annotation => {
                let accession = self.accession.as_ref().ok_or_else(|| {
                    GenRegionError::NotFound("No accession for annotation".to_string())
                })?;
                Ok(accession.intervaltree(conn)?)
            }
            ResolvedRegionKind::Accession => {
                let accession = self.accession.as_ref().ok_or_else(|| {
                    GenRegionError::NotFound("No accession for region".to_string())
                })?;
                Ok(accession.intervaltree(conn)?)
            }
            ResolvedRegionKind::BlockGroup => Ok(BlockGroup::intervaltree_for(
                conn,
                &self.block_group.id,
                self.remove_ambiguous_positions,
            )?),
        }
    }

    pub fn find_graph_positions(
        &self,
        conn: &GraphConnection,
        start_offset: i64,
        end_offset: i64,
    ) -> Result<Self, gen_graph::GraphError> {
        let (start_positions, end_positions) =
            self.compute_graph_positions(conn, start_offset, end_offset)?;
        let mut region = self.clone();
        region.start_anchors = Some(start_positions);
        region.end_anchors = Some(end_positions);
        Ok(region)
    }

    fn compute_graph_positions(
        &self,
        conn: &GraphConnection,
        start_offset: i64,
        end_offset: i64,
    ) -> Result<(Vec<GraphNodePosition>, Vec<GraphNodePosition>), gen_graph::GraphError> {
        let interval_tree = self
            .intervaltree(conn)
            .map_err(|_| gen_graph::GraphError::NoPath)?;
        graph_loader::resolve_region_positions(
            &interval_tree,
            RegionPositionQuery {
                start: self.start,
                end: self.end,
                start_offset,
                end_offset,
            },
            |node_id| {
                crate::node::Node::query_nodes_length(conn, &[node_id])
                    .ok()
                    .and_then(|lengths| lengths.get(&node_id).copied())
            },
            |graph, node_id| crate::graph::expand(conn, graph, &self.block_group.id, node_id),
        )
    }

    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip(self, conn, change, tree))
    )]
    pub fn plan_edges(
        &self,
        conn: &GraphConnection,
        change: &BlockGroupChange,
        tree: Option<&IntervalTree<i64, NodeIntervalBlock>>,
    ) -> Result<Vec<AugmentedEdgeData>, BlockGroupError> {
        match self.kind {
            ResolvedRegionKind::Path | ResolvedRegionKind::BlockGroup => {
                let local_tree;
                let tree = match tree {
                    Some(tree) => tree,
                    None => {
                        local_tree = IntervalTreeSource::intervaltree(self, conn)?;
                        &local_tree
                    }
                };
                return BlockGroup::set_up_new_edges(change, tree);
            }
            ResolvedRegionKind::Annotation | ResolvedRegionKind::Accession => {}
        };

        let graph_positions_from_tree = |coordinate| {
            let positions = graph_loader::positions_at_coordinate(tree?, coordinate);
            (!positions.is_empty()).then_some(positions)
        };

        let (start_positions, end_positions) =
            if let (Some(start), Some(end)) = (&self.start_anchors, &self.end_anchors) {
                (start.clone(), end.clone())
            } else if let Some(start_positions) = graph_positions_from_tree(self.start)
                && let Some(end_positions) = graph_positions_from_tree(self.end)
            {
                (start_positions, end_positions)
            } else {
                let resolved = self
                    .find_graph_positions(conn, 0, 0)
                    .map_err(|err| BlockGroupError::ChangeOutOfBounds(err.to_string()))?;
                (
                    resolved.start_anchors.expect("should have start anchors"),
                    resolved.end_anchors.expect("should have end anchors"),
                )
            };
        Ok(graph_loader::plan_region_edges(
            &start_positions,
            &end_positions,
            &change.block,
            change.preserve_edge,
            change.chromosome_index,
            change.phased,
        )
        .into_iter()
        .map(|edge| AugmentedEdgeData {
            edge_data: EdgeData {
                source_node_id: edge.source_node_id,
                source_coordinate: edge.source_coordinate,
                source_strand: edge.source_strand,
                target_node_id: edge.target_node_id,
                target_coordinate: edge.target_coordinate,
                target_strand: edge.target_strand,
            },
            chromosome_index: edge.chromosome_index,
            phased: edge.phased,
        })
        .collect())
    }
}

impl IntervalTreeSource for ResolvedGenRegion {
    fn intervaltree(
        &self,
        conn: &GraphConnection,
    ) -> Result<IntervalTree<i64, NodeIntervalBlock>, BlockGroupError> {
        ResolvedGenRegion::intervaltree(self, conn)
            .map_err(|err| BlockGroupError::ChangeOutOfBounds(err.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        annotations::Annotation,
        block_group::{BlockGroup, PathCache},
        test_helpers::{get_connection, setup_block_group},
    };

    fn setup_targets() -> (
        crate::db::GraphConnection,
        BlockGroup,
        Path,
        Accession,
        Annotation,
    ) {
        let conn = get_connection(None).unwrap();
        let (block_group_id, path) = setup_block_group(&conn);
        let block_group = BlockGroup::get_by_id(&conn, &block_group_id, None).unwrap();
        let mut path_cache = PathCache::new(&conn);
        let accession =
            BlockGroup::add_accession(&conn, &path, "mreB", 10, 30, &mut path_cache).unwrap();
        let annotation =
            Annotation::get_or_create(&conn, "gene-mreB", "genes", &accession.id, None).unwrap();
        (conn, block_group, path, accession, annotation)
    }

    mod region_resolver {
        use super::*;

        #[test]
        fn test_resolves_path_ranges() {
            let (conn, _block_group, path, _accession, _annotation) = setup_targets();

            let point =
                resolve_path(&Region::parse("chr1:5").unwrap(), &conn, "test", "test").unwrap();
            assert_eq!(point.kind, ResolvedRegionKind::Path);
            assert_eq!(point.path.as_ref().unwrap().id, path.id);
            assert_eq!((point.start, point.end), (5, 5));

            let to_end =
                resolve_path(&Region::parse("chr1:5..").unwrap(), &conn, "test", "test").unwrap();
            assert_eq!((to_end.start, to_end.end), (5, 40));

            let just_end = resolve_path(
                &Region {
                    name: "chr1".to_string(),
                    start: None,
                    end: Some(5),
                },
                &conn,
                "test",
                "test",
            )
            .unwrap_err();
            assert!(matches!(
                just_end,
                GenRegionError::Parse(RegionParseError::InvalidSyntax)
            ));

            let negative =
                resolve_path(&Region::parse("chr1:-5-5").unwrap(), &conn, "test", "test")
                    .unwrap_err();
            assert!(matches!(
                negative,
                GenRegionError::OutOfBounds {
                    start: -5,
                    end: 5,
                    ..
                }
            ));

            let wrap =
                resolve_path(&Region::parse("chr1:30-10").unwrap(), &conn, "test", "test").unwrap();
            assert_eq!((wrap.start, wrap.end), (30, 10));
        }

        #[test]
        fn test_resolves_block_group_ranges() {
            let (conn, block_group, _path, _accession, _annotation) = setup_targets();

            let point =
                resolve_block_group(&Region::parse("chr1:5").unwrap(), &conn, "test", "test")
                    .unwrap();
            assert_eq!(point.kind, ResolvedRegionKind::BlockGroup);
            assert_eq!(point.block_group.id, block_group.id);
            assert_eq!((point.start, point.end), (5, 5));

            let to_end =
                resolve_block_group(&Region::parse("chr1:5..").unwrap(), &conn, "test", "test")
                    .unwrap();
            assert_eq!((to_end.start, to_end.end), (5, 40));

            let just_end = resolve_block_group(
                &Region {
                    name: "chr1".to_string(),
                    start: None,
                    end: Some(5),
                },
                &conn,
                "test",
                "test",
            )
            .unwrap_err();
            assert!(matches!(
                just_end,
                GenRegionError::Parse(RegionParseError::InvalidSyntax)
            ));

            let negative =
                resolve_block_group(&Region::parse("chr1:-5-5").unwrap(), &conn, "test", "test")
                    .unwrap_err();
            assert!(matches!(
                negative,
                GenRegionError::OutOfBounds {
                    start: -5,
                    end: 5,
                    ..
                }
            ));

            let wrap =
                resolve_block_group(&Region::parse("chr1:30-10").unwrap(), &conn, "test", "test")
                    .unwrap();
            assert_eq!((wrap.start, wrap.end), (30, 10));
        }

        #[test]
        fn test_resolves_accession_ranges() {
            let (conn, _block_group, _path, accession, _annotation) = setup_targets();

            let point = resolve_accession(&Region::parse("mreB:5").unwrap(), &conn, "test", "test")
                .unwrap();
            assert_eq!(point.kind, ResolvedRegionKind::Accession);
            assert_eq!(point.accession.as_ref().unwrap().id, accession.id);
            assert_eq!((point.start, point.end), (5, 5));

            let to_end =
                resolve_accession(&Region::parse("mreB:5..").unwrap(), &conn, "test", "test")
                    .unwrap();
            assert_eq!((to_end.start, to_end.end), (5, 20));

            let just_end = resolve_accession(
                &Region {
                    name: "mreB".to_string(),
                    start: None,
                    end: Some(5),
                },
                &conn,
                "test",
                "test",
            )
            .unwrap_err();
            assert!(matches!(
                just_end,
                GenRegionError::Parse(RegionParseError::InvalidSyntax)
            ));

            let negative =
                resolve_accession(&Region::parse("mreB:-5-5").unwrap(), &conn, "test", "test")
                    .unwrap();
            assert_eq!((negative.start, negative.end), (-5, 5));

            let wrap =
                resolve_accession(&Region::parse("mreB:15-5").unwrap(), &conn, "test", "test")
                    .unwrap();
            assert_eq!((wrap.start, wrap.end), (15, 5));
        }

        #[test]
        fn test_resolves_annotation_ranges() {
            let (conn, _block_group, _path, accession, annotation) = setup_targets();

            let point = resolve_annotation(
                &Region::parse("gene-mreB:5").unwrap(),
                &conn,
                "test",
                "test",
            )
            .unwrap();
            assert_eq!(point.kind, ResolvedRegionKind::Annotation);
            assert_eq!(point.accession.as_ref().unwrap().id, accession.id);
            assert_eq!(point.start, 5);
            assert_eq!(point.end, 5);

            let to_end = resolve_annotation(
                &Region::parse("gene-mreB:5..").unwrap(),
                &conn,
                "test",
                "test",
            )
            .unwrap();
            assert_eq!(to_end.start, 5);
            assert_eq!(to_end.end, 20);

            let just_end = resolve_annotation(
                &Region {
                    name: annotation.name.clone(),
                    start: None,
                    end: Some(5),
                },
                &conn,
                "test",
                "test",
            )
            .unwrap_err();
            assert!(matches!(
                just_end,
                GenRegionError::Parse(RegionParseError::InvalidSyntax)
            ));

            let negative = resolve_annotation(
                &Region::parse("gene-mreB:-5-5").unwrap(),
                &conn,
                "test",
                "test",
            )
            .unwrap();
            assert_eq!((negative.start, negative.end), (-5, 5));

            let wrap = resolve_annotation(
                &Region::parse("gene-mreB:15-5").unwrap(),
                &conn,
                "test",
                "test",
            )
            .unwrap();
            assert_eq!((wrap.start, wrap.end), (15, 5));
        }
    }
}
