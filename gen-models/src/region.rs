pub use gen_core::region::Region;
use gen_core::{
    NodeIntervalBlock, is_terminal,
    region::{RegionParseError, RegionResolutionError, RegionResolver},
};
use gen_graph::{FromNodeIntervalTree, GenGraph};
use intervaltree::IntervalTree;
use thiserror::Error;

use crate::{
    accession::{Accession, AccessionError},
    annotations::{Annotation, AnnotationError},
    block_group::BlockGroup,
    db::GraphConnection,
    errors::PathError,
    graph::{GraphPositionAnchor, GraphTraversalError, SqlGraphTraversal},
    node::Node,
    path::Path,
    traits::Query,
};

#[derive(Clone, Debug)]
pub struct ResolvedGenRegion {
    pub block_group: BlockGroup,
    pub path: Option<Path>,
    pub accession: Option<Accession>,
    pub kind: ResolvedRegionKind,
    pub anchor_start: i64,
    pub anchor_end: i64,
    pub feature_length: i64,
    pub start: i64,
    pub end: i64,
    pub start_anchors: Vec<GraphPositionAnchor>,
    pub end_anchors: Vec<GraphPositionAnchor>,
    pub allow_out_of_bounds: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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
    #[error("Failed to resolve graph position: {0}")]
    GraphTraversal(#[from] GraphTraversalError),
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
    anchor_start: i64,
    anchor_end: i64,
    feature_length: i64,
    allow_out_of_bounds: bool,
    intervaltree: IntervalTree<i64, NodeIntervalBlock>,
}

#[derive(Clone, Copy)]
enum AnchorBias {
    Current,
    Previous,
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
    let block_group = BlockGroup::get_by_id(conn, &path.block_group_id)?;
    let path_length = path.length(conn)?;
    let intervaltree = path.intervaltree(conn)?;
    resolve_target(
        region,
        conn,
        RegionTarget {
            kind: RegionTargetKind::Path,
            block_group,
            path: Some(path),
            accession: None,
            anchor_start: 0,
            anchor_end: path_length,
            feature_length: path_length,
            allow_out_of_bounds: false,
            intervaltree,
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
    let path = BlockGroup::get_current_path(conn, &block_group.id);
    let path_length = path.length(conn)?;
    let intervaltree = path.intervaltree(conn)?;
    resolve_target(
        region,
        conn,
        RegionTarget {
            kind: RegionTargetKind::BlockGroup,
            block_group,
            path: Some(path),
            accession: None,
            anchor_start: 0,
            anchor_end: path_length,
            feature_length: path_length,
            allow_out_of_bounds: false,
            intervaltree,
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
    )?;
    resolve_target(region, conn, target)
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
    let accession = Accession::get_by_id(conn, &annotation.accession_id)
        .ok_or_else(|| GenRegionError::Unmappable(region.name.clone()))?;
    let target = target_from_accession(
        region,
        conn,
        collection_name,
        sample_name,
        RegionTargetKind::Annotation,
        accession,
    )?;
    resolve_target(region, conn, target)
}

fn resolve_target(
    region: &Region,
    conn: &GraphConnection,
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

    let out_of_bounds = start < 0 || end > target.feature_length;

    if out_of_bounds && !target.allow_out_of_bounds {
        return Err(GenRegionError::OutOfBounds {
            region: region.to_string(),
            start,
            end,
            path_length: target.feature_length,
        });
    }

    let start_anchors = resolve_graph_anchors(
        conn,
        &target.intervaltree,
        start,
        target.feature_length,
        AnchorBias::Current,
        target.allow_out_of_bounds,
        region,
    )?;
    let end_anchors = resolve_graph_anchors(
        conn,
        &target.intervaltree,
        end,
        target.feature_length,
        AnchorBias::Current,
        target.allow_out_of_bounds,
        region,
    )?;

    Ok(ResolvedGenRegion {
        block_group: target.block_group,
        path: target.path,
        accession: target.accession,
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
        start_anchors,
        end_anchors,
        allow_out_of_bounds: target.allow_out_of_bounds,
    })
}

fn target_from_accession(
    region: &Region,
    conn: &GraphConnection,
    collection_name: &str,
    sample_name: &str,
    kind: RegionTargetKind,
    accession: Accession,
) -> Result<RegionTarget, GenRegionError> {
    let block_group = BlockGroup::get_by_id(conn, &accession.block_group_id)?;
    if block_group.collection_name != collection_name || block_group.sample_name != sample_name {
        return Err(GenRegionError::NotFound(region.name.clone()));
    }
    let path_length = accession.length(conn)?;
    let intervaltree = accession.intervaltree(conn)?;
    Ok(RegionTarget {
        kind,
        block_group,
        path: None,
        accession: Some(accession),
        anchor_start: 0,
        anchor_end: path_length,
        feature_length: path_length,
        allow_out_of_bounds: kind == RegionTargetKind::Annotation,
        intervaltree,
    })
}

fn resolve_graph_anchors(
    conn: &GraphConnection,
    tree: &IntervalTree<i64, NodeIntervalBlock>,
    coordinate: i64,
    feature_length: i64,
    bias: AnchorBias,
    allow_out_of_bounds: bool,
    region: &Region,
) -> Result<Vec<GraphPositionAnchor>, GenRegionError> {
    if (0..=feature_length).contains(&coordinate) {
        return Ok(resolve_intervaltree_anchors(tree, coordinate, bias));
    }

    if !allow_out_of_bounds {
        return Err(GenRegionError::OutOfBounds {
            region: region.to_string(),
            start: coordinate,
            end: coordinate,
            path_length: feature_length,
        });
    }

    let (boundary_coordinate, boundary_bias, distance) = if coordinate < 0 {
        (0, AnchorBias::Current, coordinate)
    } else {
        (
            feature_length,
            AnchorBias::Previous,
            coordinate - feature_length,
        )
    };
    let boundary_anchors = resolve_intervaltree_anchors(tree, boundary_coordinate, boundary_bias);
    if boundary_anchors.is_empty() {
        return Err(GenRegionError::OutOfBounds {
            region: region.to_string(),
            start: coordinate,
            end: coordinate,
            path_length: feature_length,
        });
    }

    if let Some(anchors) =
        resolve_same_node_out_of_bounds(conn, tree, coordinate, boundary_coordinate, boundary_bias)?
    {
        return Ok(anchors);
    }

    let graph = GenGraph::from_node_interval_tree(tree);
    let mut traversal = SqlGraphTraversal::new(conn, graph);
    let mut anchors = Vec::new();
    for anchor in boundary_anchors {
        let positions = traversal.find_offset(anchor, distance)?;
        anchors.extend(positions.into_iter().map(|position| GraphPositionAnchor {
            coordinate: position.graph_node.sequence_start + position.offset,
            node_id: Some(position.graph_node.node_id),
        }));
    }
    anchors.sort_by_key(|anchor| (anchor.node_id, anchor.coordinate));
    anchors.dedup();

    if anchors.is_empty() {
        Err(GenRegionError::OutOfBounds {
            region: region.to_string(),
            start: coordinate,
            end: coordinate,
            path_length: feature_length,
        })
    } else {
        Ok(anchors)
    }
}

fn resolve_same_node_out_of_bounds(
    conn: &GraphConnection,
    tree: &IntervalTree<i64, NodeIntervalBlock>,
    coordinate: i64,
    boundary_coordinate: i64,
    boundary_bias: AnchorBias,
) -> Result<Option<Vec<GraphPositionAnchor>>, GenRegionError> {
    let boundary_blocks = blocks_for_coordinate(tree, boundary_coordinate, boundary_bias)
        .into_iter()
        .filter(|block| !is_terminal(block.node_id))
        .collect::<Vec<_>>();
    if boundary_blocks.is_empty() {
        return Ok(None);
    }

    let node_ids = boundary_blocks
        .iter()
        .map(|block| block.node_id)
        .collect::<Vec<_>>();
    let node_lengths = Node::query_nodes_length(conn, &node_ids).map_err(|error| {
        GenRegionError::GraphTraversal(GraphTraversalError::DatabaseError(error.to_string()))
    })?;
    let mut anchors = Vec::new();
    for block in boundary_blocks {
        let projected_coordinate = coordinate - block.start + block.sequence_start;
        if (0..=*node_lengths.get(&block.node_id).unwrap_or(&-1)).contains(&projected_coordinate) {
            anchors.push(GraphPositionAnchor {
                coordinate: projected_coordinate,
                node_id: Some(block.node_id),
            });
        }
    }
    anchors.sort_by_key(|anchor| (anchor.node_id, anchor.coordinate));
    anchors.dedup();

    if anchors.is_empty() {
        Ok(None)
    } else {
        Ok(Some(anchors))
    }
}

fn resolve_intervaltree_anchors(
    tree: &IntervalTree<i64, NodeIntervalBlock>,
    coordinate: i64,
    bias: AnchorBias,
) -> Vec<GraphPositionAnchor> {
    let blocks = blocks_for_coordinate(tree, coordinate, bias);
    blocks
        .into_iter()
        .map(|block| GraphPositionAnchor {
            coordinate: if is_terminal(block.node_id) {
                0
            } else {
                coordinate - block.start + block.sequence_start
            },
            node_id: Some(block.node_id),
        })
        .collect()
}

fn blocks_for_coordinate(
    tree: &IntervalTree<i64, NodeIntervalBlock>,
    coordinate: i64,
    bias: AnchorBias,
) -> Vec<NodeIntervalBlock> {
    let blocks = tree
        .query_point(coordinate)
        .map(|element| element.value)
        .collect::<Vec<_>>();

    if matches!(bias, AnchorBias::Previous)
        && coordinate > i64::MIN
        && blocks.iter().any(|block| block.start == coordinate)
    {
        tree.query_point(coordinate - 1)
            .map(|element| element.value)
            .collect()
    } else {
        blocks
    }
}

impl ResolvedGenRegion {
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

        let out_of_bounds = start < 0 || end > self.feature_length;

        if out_of_bounds && !self.allow_out_of_bounds {
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
}

#[cfg(test)]
mod tests {
    use gen_core::HashId;

    use super::*;
    use crate::{
        annotations::Annotation,
        block_group::{BlockGroup, PathCache},
        graph::GraphPositionAnchor,
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
        let block_group = BlockGroup::get_by_id(&conn, &block_group_id).unwrap();
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
        fn path_range_finding() {
            let (conn, _block_group, path, _accession, _annotation) = setup_targets();

            let point =
                resolve_path(&Region::parse("chr1:5").unwrap(), &conn, "test", "test").unwrap();
            assert_eq!(point.kind, ResolvedRegionKind::Path);
            assert_eq!(point.path.as_ref().unwrap().id, path.id);
            assert_eq!((point.start, point.end), (5, 5));
            assert_eq!(
                point.start_anchors,
                vec![GraphPositionAnchor {
                    coordinate: 5,
                    node_id: Some(HashId::convert_str("test-a-node")),
                }]
            );
            assert_eq!(point.end_anchors, point.start_anchors);

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
        fn block_group_range_finding() {
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
        fn accession_range_finding() {
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
                resolve_accession(&Region::parse("mreB:15-5").unwrap(), &conn, "test", "test")
                    .unwrap();
            assert_eq!((wrap.start, wrap.end), (15, 5));
        }

        #[test]
        fn annotation_range_finding() {
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
            assert_eq!(
                negative.start_anchors,
                vec![GraphPositionAnchor {
                    coordinate: 5,
                    node_id: Some(HashId::convert_str("test-a-node")),
                }]
            );
            assert_eq!(
                negative.end_anchors,
                vec![GraphPositionAnchor {
                    coordinate: 5,
                    node_id: Some(HashId::convert_str("test-t-node")),
                }]
            );

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
