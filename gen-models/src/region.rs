pub use gen_core::region::Region;
use gen_core::{
    NodeIntervalBlock,
    region::{RegionParseError, RegionResolutionError, RegionResolver},
};
use intervaltree::IntervalTree;
use thiserror::Error;

use crate::{
    accession::{Accession, AccessionError},
    annotations::{Annotation, AnnotationError},
    block_group::BlockGroup,
    db::GraphConnection,
    errors::PathError,
    path::Path,
    traits::Query,
};

#[derive(Debug)]
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
    resolve_target(
        region,
        RegionTarget {
            kind: RegionTargetKind::Path,
            block_group,
            path: Some(path),
            accession: None,
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
    let path = BlockGroup::get_current_path(conn, &block_group.id)?;
    let path_length = path.length(conn)?;
    resolve_target(
        region,
        RegionTarget {
            kind: RegionTargetKind::BlockGroup,
            block_group,
            path: Some(path),
            accession: None,
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
    Ok(RegionTarget {
        kind,
        block_group,
        path: None,
        accession: Some(accession),
        anchor_start: 0,
        anchor_end: path_length,
        feature_length: path_length,
    })
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
                false,
            )?),
        }
    }

    pub fn find_graph_positions(
        &self,
        conn: &GraphConnection,
        start_offset: i64,
        end_offset: i64,
    ) -> Result<
        (
            Vec<gen_graph::GraphNodePosition>,
            Vec<gen_graph::GraphNodePosition>,
        ),
        gen_graph::GraphError,
    > {
        let interval_tree = self
            .intervaltree(conn)
            .map_err(|_| gen_graph::GraphError::NoPath)?;

        let filtered: Vec<(std::ops::Range<i64>, gen_core::NodeIntervalBlock)> = interval_tree
            .iter()
            .filter(|item| !gen_core::is_terminal(item.value.node_id))
            .map(|item| (item.range.clone(), item.value))
            .collect();
        let tree: intervaltree::IntervalTree<i64, gen_core::NodeIntervalBlock> =
            filtered.into_iter().collect();

        let mut graph = gen_graph::graph_from_interval_tree(&tree);

        let resolved = crate::graph::ResolvedGraph {
            graph: graph.clone(),
            interval_tree: tree,
            block_group_id: self.block_group.id,
        };
        let start_anchor = resolved.resolve_anchor(self.start, conn)?;
        let end_anchor = resolved.resolve_anchor(self.end, conn)?;

        let start_positions =
            crate::graph::find_offset(&mut graph, &start_anchor, start_offset, |g, nid| {
                crate::graph::expand(conn, g, &self.block_group.id, nid)
            })?;
        let end_positions =
            crate::graph::find_offset(&mut graph, &end_anchor, end_offset, |g, nid| {
                crate::graph::expand(conn, g, &self.block_group.id, nid)
            })?;

        Ok((start_positions, end_positions))
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

    mod find_graph_positions {
        use std::collections::HashSet;

        use gen_core::{HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand};

        use super::*;
        use crate::{
            block_group::{BlockGroup, NewBlockGroup, PathCache},
            block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
            collection::Collection,
            edge::Edge,
            node::Node,
            path::Path,
            sample::{NewSample, Sample},
            sequence::Sequence,
        };

        fn setup_graph() -> (crate::db::GraphConnection, HashId) {
            let conn = get_connection(None).unwrap();
            Collection::get_or_create(&conn, "test").unwrap();
            Sample::get_or_create(
                &conn,
                NewSample {
                    name: "test",
                    ..Default::default()
                },
            )
            .unwrap();
            let block_group = BlockGroup::create(
                &conn,
                NewBlockGroup {
                    collection_name: "test",
                    sample_name: "test",
                    name: "chr1",
                    ..Default::default()
                },
            )
            .unwrap();

            let seq_x = Sequence::new()
                .sequence_type("DNA")
                .sequence("XXXXX")
                .save(&conn)
                .unwrap();
            let seq_y = Sequence::new()
                .sequence_type("DNA")
                .sequence("YYYYY")
                .save(&conn)
                .unwrap();
            let seq_z = Sequence::new()
                .sequence_type("DNA")
                .sequence("ZZZZZ")
                .save(&conn)
                .unwrap();

            let node_x = Node::create(&conn, &seq_x.hash, &HashId::convert_str("node-x")).unwrap();
            let node_y = Node::create(&conn, &seq_y.hash, &HashId::convert_str("node-y")).unwrap();
            let node_z = Node::create(&conn, &seq_z.hash, &HashId::convert_str("node-z")).unwrap();

            let e_start = Edge::create(
                &conn,
                PATH_START_NODE_ID,
                -1,
                Strand::Forward,
                node_x,
                0,
                Strand::Forward,
            )
            .unwrap();
            let e_xy = Edge::create(
                &conn,
                node_x,
                5,
                Strand::Forward,
                node_y,
                0,
                Strand::Forward,
            )
            .unwrap();
            let e_yz = Edge::create(
                &conn,
                node_y,
                5,
                Strand::Forward,
                node_z,
                0,
                Strand::Forward,
            )
            .unwrap();
            let e_end = Edge::create(
                &conn,
                node_z,
                5,
                Strand::Forward,
                PATH_END_NODE_ID,
                0,
                Strand::Forward,
            )
            .unwrap();

            BlockGroupEdge::bulk_create(
                &conn,
                &[
                    BlockGroupEdgeData {
                        block_group_id: block_group.id,
                        edge_id: e_start.id,
                        chromosome_index: 0,
                        phased: 0,
                    },
                    BlockGroupEdgeData {
                        block_group_id: block_group.id,
                        edge_id: e_xy.id,
                        chromosome_index: 0,
                        phased: 0,
                    },
                    BlockGroupEdgeData {
                        block_group_id: block_group.id,
                        edge_id: e_yz.id,
                        chromosome_index: 0,
                        phased: 0,
                    },
                    BlockGroupEdgeData {
                        block_group_id: block_group.id,
                        edge_id: e_end.id,
                        chromosome_index: 0,
                        phased: 0,
                    },
                ],
            );

            (conn, block_group.id)
        }

        fn create_accession(
            conn: &crate::db::GraphConnection,
            block_group_id: HashId,
            name: &str,
            start: i64,
            end: i64,
        ) -> Accession {
            let edges = BlockGroupEdge::edges_for_block_group(conn, &block_group_id);
            let mut by_source: std::collections::HashMap<
                HashId,
                &crate::block_group_edge::AugmentedEdge,
            > = std::collections::HashMap::new();
            for ae in &edges {
                by_source.insert(ae.edge.source_node_id, ae);
            }
            let mut ordered = vec![];
            let mut current = Some(PATH_START_NODE_ID);
            while let Some(src) = current {
                if let Some(ae) = by_source.get(&src) {
                    ordered.push(ae.edge.id);
                    current = if ae.edge.target_node_id == PATH_END_NODE_ID {
                        None
                    } else {
                        Some(ae.edge.target_node_id)
                    };
                } else {
                    break;
                }
            }
            let path = Path::create(conn, name, &block_group_id, &ordered).unwrap();
            let mut path_cache = PathCache::new(conn);
            let accession =
                BlockGroup::add_accession(conn, &path, name, start, end, &mut path_cache).unwrap();
            Path::delete(conn, name, &block_group_id);
            accession
        }

        fn create_accession_from_edges(
            conn: &crate::db::GraphConnection,
            block_group_id: HashId,
            name: &str,
            edge_ids: &[HashId],
            start: i64,
            end: i64,
        ) -> Accession {
            let path = Path::create(conn, name, &block_group_id, edge_ids).unwrap();
            let mut path_cache = PathCache::new(conn);
            let accession =
                BlockGroup::add_accession(conn, &path, name, start, end, &mut path_cache).unwrap();
            Path::delete(conn, name, &block_group_id);
            accession
        }

        fn make_region(
            bg: BlockGroup,
            accession: Accession,
            anchor_start: i64,
            anchor_end: i64,
            feature_length: i64,
            start: i64,
            end: i64,
        ) -> ResolvedGenRegion {
            ResolvedGenRegion {
                block_group: bg,
                path: None,
                accession: Some(accession),
                kind: ResolvedRegionKind::Accession,
                anchor_start,
                anchor_end,
                feature_length,
                start,
                end,
            }
        }

        /// Creates a branched graph: {AAA,GGG} → TTT → {CCC,ATC}
        /// Path: AAA→TTT→CCC (positions 0..9)
        fn setup_branched_graph() -> (crate::db::GraphConnection, HashId) {
            let conn = get_connection(None).unwrap();
            Collection::get_or_create(&conn, "test").unwrap();
            Sample::get_or_create(
                &conn,
                NewSample {
                    name: "test",
                    ..Default::default()
                },
            )
            .unwrap();
            let bg = BlockGroup::create(
                &conn,
                NewBlockGroup {
                    collection_name: "test",
                    sample_name: "test",
                    name: "branched",
                    ..Default::default()
                },
            )
            .unwrap();

            let seq_aaa = Sequence::new()
                .sequence_type("DNA")
                .sequence("AAA")
                .save(&conn)
                .unwrap();
            let seq_ggg = Sequence::new()
                .sequence_type("DNA")
                .sequence("GGG")
                .save(&conn)
                .unwrap();
            let seq_ttt = Sequence::new()
                .sequence_type("DNA")
                .sequence("TTT")
                .save(&conn)
                .unwrap();
            let seq_ccc = Sequence::new()
                .sequence_type("DNA")
                .sequence("CCC")
                .save(&conn)
                .unwrap();
            let seq_atc = Sequence::new()
                .sequence_type("DNA")
                .sequence("ATC")
                .save(&conn)
                .unwrap();

            let n_aaa =
                Node::create(&conn, &seq_aaa.hash, &HashId::convert_str("node-aaa")).unwrap();
            let n_ggg =
                Node::create(&conn, &seq_ggg.hash, &HashId::convert_str("node-ggg")).unwrap();
            let n_ttt =
                Node::create(&conn, &seq_ttt.hash, &HashId::convert_str("node-ttt")).unwrap();
            let n_ccc =
                Node::create(&conn, &seq_ccc.hash, &HashId::convert_str("node-ccc")).unwrap();
            let n_atc =
                Node::create(&conn, &seq_atc.hash, &HashId::convert_str("node-atc")).unwrap();

            let e_start = Edge::create(
                &conn,
                PATH_START_NODE_ID,
                -1,
                Strand::Forward,
                n_aaa,
                0,
                Strand::Forward,
            )
            .unwrap();
            let e_ggg_start = Edge::create(
                &conn,
                PATH_START_NODE_ID,
                -1,
                Strand::Forward,
                n_ggg,
                0,
                Strand::Forward,
            )
            .unwrap();
            let e_aaa_ttt =
                Edge::create(&conn, n_aaa, 3, Strand::Forward, n_ttt, 0, Strand::Forward).unwrap();
            let e_ttt_ccc =
                Edge::create(&conn, n_ttt, 3, Strand::Forward, n_ccc, 0, Strand::Forward).unwrap();
            let e_ccc_end = Edge::create(
                &conn,
                n_ccc,
                3,
                Strand::Forward,
                PATH_END_NODE_ID,
                0,
                Strand::Forward,
            )
            .unwrap();
            let e_atc_end = Edge::create(
                &conn,
                n_atc,
                3,
                Strand::Forward,
                PATH_END_NODE_ID,
                0,
                Strand::Forward,
            )
            .unwrap();
            let e_ggg_ttt =
                Edge::create(&conn, n_ggg, 3, Strand::Forward, n_ttt, 0, Strand::Forward).unwrap();
            let e_ttt_atc =
                Edge::create(&conn, n_ttt, 3, Strand::Forward, n_atc, 0, Strand::Forward).unwrap();

            BlockGroupEdge::bulk_create(
                &conn,
                &[
                    BlockGroupEdgeData {
                        block_group_id: bg.id,
                        edge_id: e_start.id,
                        chromosome_index: 0,
                        phased: 0,
                    },
                    BlockGroupEdgeData {
                        block_group_id: bg.id,
                        edge_id: e_ggg_start.id,
                        chromosome_index: 0,
                        phased: 0,
                    },
                    BlockGroupEdgeData {
                        block_group_id: bg.id,
                        edge_id: e_aaa_ttt.id,
                        chromosome_index: 0,
                        phased: 0,
                    },
                    BlockGroupEdgeData {
                        block_group_id: bg.id,
                        edge_id: e_ttt_ccc.id,
                        chromosome_index: 0,
                        phased: 0,
                    },
                    BlockGroupEdgeData {
                        block_group_id: bg.id,
                        edge_id: e_ccc_end.id,
                        chromosome_index: 0,
                        phased: 0,
                    },
                    BlockGroupEdgeData {
                        block_group_id: bg.id,
                        edge_id: e_atc_end.id,
                        chromosome_index: 0,
                        phased: 0,
                    },
                    BlockGroupEdgeData {
                        block_group_id: bg.id,
                        edge_id: e_ggg_ttt.id,
                        chromosome_index: 0,
                        phased: 0,
                    },
                    BlockGroupEdgeData {
                        block_group_id: bg.id,
                        edge_id: e_ttt_atc.id,
                        chromosome_index: 0,
                        phased: 0,
                    },
                ],
            );

            (conn, bg.id)
        }

        struct GraphFixture {
            conn: crate::db::GraphConnection,
            block_group_id: HashId,
            path: Vec<HashId>,
        }

        /// Creates a branched graph: AAA -> {CC, GGGG} -> TTT.
        fn setup_variable_length_branched_graph() -> GraphFixture {
            let conn = get_connection(None).unwrap();
            Collection::get_or_create(&conn, "test").unwrap();
            Sample::get_or_create(
                &conn,
                NewSample {
                    name: "test",
                    ..Default::default()
                },
            )
            .unwrap();
            let bg = BlockGroup::create(
                &conn,
                NewBlockGroup {
                    collection_name: "test",
                    sample_name: "test",
                    name: "variable-length-branched",
                    ..Default::default()
                },
            )
            .unwrap();

            let seq_aaa = Sequence::new()
                .sequence_type("DNA")
                .sequence("AAA")
                .save(&conn)
                .unwrap();
            let seq_cc = Sequence::new()
                .sequence_type("DNA")
                .sequence("CC")
                .save(&conn)
                .unwrap();
            let seq_gggg = Sequence::new()
                .sequence_type("DNA")
                .sequence("GGGG")
                .save(&conn)
                .unwrap();
            let seq_ttt = Sequence::new()
                .sequence_type("DNA")
                .sequence("TTT")
                .save(&conn)
                .unwrap();

            let n_aaa =
                Node::create(&conn, &seq_aaa.hash, &HashId::convert_str("node-aaa")).unwrap();
            let n_cc = Node::create(&conn, &seq_cc.hash, &HashId::convert_str("node-cc")).unwrap();
            let n_gggg =
                Node::create(&conn, &seq_gggg.hash, &HashId::convert_str("node-gggg")).unwrap();
            let n_ttt =
                Node::create(&conn, &seq_ttt.hash, &HashId::convert_str("node-ttt")).unwrap();

            let e_start = Edge::create(
                &conn,
                PATH_START_NODE_ID,
                -1,
                Strand::Forward,
                n_aaa,
                0,
                Strand::Forward,
            )
            .unwrap();
            let e_aaa_cc =
                Edge::create(&conn, n_aaa, 3, Strand::Forward, n_cc, 0, Strand::Forward).unwrap();
            let e_aaa_gggg =
                Edge::create(&conn, n_aaa, 3, Strand::Forward, n_gggg, 0, Strand::Forward).unwrap();
            let e_cc_ttt =
                Edge::create(&conn, n_cc, 2, Strand::Forward, n_ttt, 0, Strand::Forward).unwrap();
            let e_gggg_ttt =
                Edge::create(&conn, n_gggg, 4, Strand::Forward, n_ttt, 0, Strand::Forward).unwrap();
            let e_end = Edge::create(
                &conn,
                n_ttt,
                3,
                Strand::Forward,
                PATH_END_NODE_ID,
                0,
                Strand::Forward,
            )
            .unwrap();

            BlockGroupEdge::bulk_create(
                &conn,
                &[
                    BlockGroupEdgeData {
                        block_group_id: bg.id,
                        edge_id: e_start.id,
                        chromosome_index: 0,
                        phased: 0,
                    },
                    BlockGroupEdgeData {
                        block_group_id: bg.id,
                        edge_id: e_aaa_cc.id,
                        chromosome_index: 0,
                        phased: 0,
                    },
                    BlockGroupEdgeData {
                        block_group_id: bg.id,
                        edge_id: e_aaa_gggg.id,
                        chromosome_index: 0,
                        phased: 0,
                    },
                    BlockGroupEdgeData {
                        block_group_id: bg.id,
                        edge_id: e_cc_ttt.id,
                        chromosome_index: 0,
                        phased: 0,
                    },
                    BlockGroupEdgeData {
                        block_group_id: bg.id,
                        edge_id: e_gggg_ttt.id,
                        chromosome_index: 0,
                        phased: 0,
                    },
                    BlockGroupEdgeData {
                        block_group_id: bg.id,
                        edge_id: e_end.id,
                        chromosome_index: 0,
                        phased: 0,
                    },
                ],
            );

            GraphFixture {
                conn,
                block_group_id: bg.id,
                path: vec![e_start.id, e_aaa_cc.id, e_cc_ttt.id, e_end.id],
            }
        }

        fn position_set(positions: &[gen_graph::GraphNodePosition]) -> HashSet<(HashId, i64)> {
            positions
                .iter()
                .map(|pos| (pos.graph_node.node_id, pos.offset))
                .collect()
        }

        #[test]
        fn test_finds_graph_positions_within_node() {
            let (conn, bg_id) = setup_graph();
            let bg = BlockGroup::get_by_id(&conn, &bg_id).unwrap();
            let acc = create_accession(&conn, bg_id, "within", 0, 15);
            let region = make_region(bg, acc, 0, 15, 15, 7, 7);

            let (start_pos, end_pos) = region.find_graph_positions(&conn, 2, 2).unwrap();
            assert_eq!(start_pos.len(), 1);
            assert_eq!(
                start_pos[0].graph_node.node_id,
                HashId::convert_str("node-y")
            );
            assert_eq!(start_pos[0].offset, 4);
            assert_eq!(end_pos.len(), 1);
            assert_eq!(end_pos[0].graph_node.node_id, HashId::convert_str("node-y"));
            assert_eq!(end_pos[0].offset, 4);
        }

        #[test]
        fn test_finds_graph_positions_forward_across_nodes() {
            let (conn, bg_id) = setup_graph();
            let bg = BlockGroup::get_by_id(&conn, &bg_id).unwrap();
            let acc = create_accession(&conn, bg_id, "fwd", 0, 15);
            let region = make_region(bg, acc, 0, 15, 15, 7, 7);

            let (start_pos, end_pos) = region.find_graph_positions(&conn, 5, 5).unwrap();
            assert_eq!(start_pos.len(), 1);
            assert_eq!(
                start_pos[0].graph_node.node_id,
                HashId::convert_str("node-z")
            );
            assert_eq!(start_pos[0].offset, 2);
            assert_eq!(end_pos.len(), 1);
            assert_eq!(end_pos[0].graph_node.node_id, HashId::convert_str("node-z"));
            assert_eq!(end_pos[0].offset, 2);
        }

        #[test]
        fn test_finds_graph_positions_backwards_across_nodes() {
            let (conn, bg_id) = setup_graph();
            let bg = BlockGroup::get_by_id(&conn, &bg_id).unwrap();
            let acc = create_accession(&conn, bg_id, "bwd", 0, 15);
            let region = make_region(bg, acc, 0, 15, 15, 7, 7);

            let (start_pos, end_pos) = region.find_graph_positions(&conn, -5, -5).unwrap();
            assert_eq!(start_pos.len(), 1);
            assert_eq!(
                start_pos[0].graph_node.node_id,
                HashId::convert_str("node-x")
            );
            assert_eq!(start_pos[0].offset, 2);
            assert_eq!(end_pos.len(), 1);
            assert_eq!(end_pos[0].graph_node.node_id, HashId::convert_str("node-x"));
            assert_eq!(end_pos[0].offset, 2);
        }

        #[test]
        fn test_reports_out_of_bounds() {
            let (conn, bg_id) = setup_graph();
            let bg = BlockGroup::get_by_id(&conn, &bg_id).unwrap();
            let acc = create_accession(&conn, bg_id, "oob", 0, 15);
            let region = make_region(bg, acc, 0, 15, 15, 7, 7);

            assert!(region.find_graph_positions(&conn, 100, 100).is_err());
        }

        #[test]
        fn test_finds_graph_positions_from_start() {
            let (conn, bg_id) = setup_graph();
            let bg = BlockGroup::get_by_id(&conn, &bg_id).unwrap();
            let acc = create_accession(&conn, bg_id, "start", 0, 15);
            let region = make_region(bg, acc, 0, 15, 15, 0, 0);

            let (start_pos, end_pos) = region.find_graph_positions(&conn, 12, 12).unwrap();
            assert_eq!(start_pos.len(), 1);
            assert_eq!(
                start_pos[0].graph_node.node_id,
                HashId::convert_str("node-z")
            );
            assert_eq!(start_pos[0].offset, 2);
            assert_eq!(end_pos.len(), 1);
            assert_eq!(end_pos[0].graph_node.node_id, HashId::convert_str("node-z"));
            assert_eq!(end_pos[0].offset, 2);
        }

        #[test]
        fn test_finds_graph_positions_from_end() {
            let (conn, bg_id) = setup_graph();
            let bg = BlockGroup::get_by_id(&conn, &bg_id).unwrap();
            let acc = create_accession(&conn, bg_id, "end", 0, 15);
            let region = make_region(bg, acc, 0, 15, 15, 14, 14);

            let (start_pos, end_pos) = region.find_graph_positions(&conn, -14, -14).unwrap();
            assert_eq!(start_pos.len(), 1);
            assert_eq!(
                start_pos[0].graph_node.node_id,
                HashId::convert_str("node-x")
            );
            assert_eq!(start_pos[0].offset, 0);
            assert_eq!(end_pos.len(), 1);
            assert_eq!(end_pos[0].graph_node.node_id, HashId::convert_str("node-x"));
            assert_eq!(end_pos[0].offset, 0);
        }

        #[test]
        fn test_finds_graph_positions_within_accessions() {
            let (conn, bg_id) = setup_graph();
            let bg = BlockGroup::get_by_id(&conn, &bg_id).unwrap();
            let acc = create_accession(&conn, bg_id, "acc-within", 5, 10);
            let region = make_region(bg, acc, 5, 10, 5, 2, 2);

            let (start_pos, end_pos) = region.find_graph_positions(&conn, 1, 1).unwrap();
            assert_eq!(start_pos.len(), 1);
            assert_eq!(
                start_pos[0].graph_node.node_id,
                HashId::convert_str("node-y")
            );
            assert_eq!(start_pos[0].offset, 3);
            assert_eq!(end_pos.len(), 1);
            assert_eq!(end_pos[0].graph_node.node_id, HashId::convert_str("node-y"));
            assert_eq!(end_pos[0].offset, 3);
        }

        #[test]
        fn test_finds_graph_positions_expands_accession_forward() {
            let (conn, bg_id) = setup_graph();
            let bg = BlockGroup::get_by_id(&conn, &bg_id).unwrap();
            let acc = create_accession(&conn, bg_id, "acc-fwd", 5, 10);
            let region = make_region(bg, acc, 5, 10, 5, 3, 3);

            let (start_pos, end_pos) = region.find_graph_positions(&conn, 5, 5).unwrap();
            assert_eq!(start_pos.len(), 1);
            assert_eq!(
                start_pos[0].graph_node.node_id,
                HashId::convert_str("node-z")
            );
            assert_eq!(start_pos[0].offset, 3);
            assert_eq!(end_pos.len(), 1);
            assert_eq!(end_pos[0].graph_node.node_id, HashId::convert_str("node-z"));
            assert_eq!(end_pos[0].offset, 3);
        }

        #[test]
        fn test_finds_graph_positions_expands_accession_backward() {
            let (conn, bg_id) = setup_graph();
            let bg = BlockGroup::get_by_id(&conn, &bg_id).unwrap();
            let acc = create_accession(&conn, bg_id, "acc-bwd", 5, 10);
            let region = make_region(bg, acc, 5, 10, 5, 1, 1);

            let (start_pos, end_pos) = region.find_graph_positions(&conn, -4, -4).unwrap();
            assert_eq!(start_pos.len(), 1);
            assert_eq!(
                start_pos[0].graph_node.node_id,
                HashId::convert_str("node-x")
            );
            assert_eq!(start_pos[0].offset, 2);
            assert_eq!(end_pos.len(), 1);
            assert_eq!(end_pos[0].graph_node.node_id, HashId::convert_str("node-x"));
            assert_eq!(end_pos[0].offset, 2);
        }

        #[test]
        fn test_finds_graph_positions_reports_accession_out_of_bounds() {
            let (conn, bg_id) = setup_graph();
            let bg = BlockGroup::get_by_id(&conn, &bg_id).unwrap();
            let acc = create_accession(&conn, bg_id, "acc-oob", 5, 10);
            let region = make_region(bg, acc, 5, 10, 5, 2, 2);

            assert!(region.find_graph_positions(&conn, 100, 100).is_err());
        }

        #[test]
        fn test_finds_graph_positions_in_branched_graph_backwards_returns_all_positions() {
            let (conn, bg_id) = setup_branched_graph();
            let bg = BlockGroup::get_by_id(&conn, &bg_id).unwrap();
            // Accession on TTT: path positions 3..6, accession-relative 0..3
            let acc = create_accession(&conn, bg_id, "branched-bwd", 3, 6);
            let region = make_region(bg, acc, 3, 6, 3, 0, 0);

            // Backward 3 from TTT offset 0 → should find AAA and GGG at offset 0
            let (start_pos, end_pos) = region.find_graph_positions(&conn, -3, -3).unwrap();
            assert_eq!(start_pos.len(), 2);
            let start_ids: Vec<HashId> = start_pos.iter().map(|p| p.graph_node.node_id).collect();
            assert!(start_ids.contains(&HashId::convert_str("node-aaa")));
            assert!(start_ids.contains(&HashId::convert_str("node-ggg")));
            for pos in &start_pos {
                assert_eq!(pos.offset, 0);
            }
            assert_eq!(end_pos.len(), 2);
            let end_ids: Vec<HashId> = end_pos.iter().map(|p| p.graph_node.node_id).collect();
            assert!(end_ids.contains(&HashId::convert_str("node-aaa")));
            assert!(end_ids.contains(&HashId::convert_str("node-ggg")));
        }

        #[test]
        fn test_finds_graph_positions_in_branched_graph_forwardsgr_returns_all_positions() {
            let (conn, bg_id) = setup_branched_graph();
            let bg = BlockGroup::get_by_id(&conn, &bg_id).unwrap();
            // Accession on TTT: path positions 3..6, accession-relative 0..3
            let acc = create_accession(&conn, bg_id, "branched-fwd", 3, 6);
            let region = make_region(bg, acc, 3, 6, 3, 2, 2);

            // Forward 3 from TTT offset 2 → should find CCC and ATC at offset 2
            let (start_pos, end_pos) = region.find_graph_positions(&conn, 3, 3).unwrap();
            assert_eq!(start_pos.len(), 2);
            let start_ids: Vec<HashId> = start_pos.iter().map(|p| p.graph_node.node_id).collect();
            assert!(start_ids.contains(&HashId::convert_str("node-ccc")));
            assert!(start_ids.contains(&HashId::convert_str("node-atc")));
            for pos in &start_pos {
                assert_eq!(pos.offset, 2);
            }
            assert_eq!(end_pos.len(), 2);
            let end_ids: Vec<HashId> = end_pos.iter().map(|p| p.graph_node.node_id).collect();
            assert!(end_ids.contains(&HashId::convert_str("node-ccc")));
            assert!(end_ids.contains(&HashId::convert_str("node-atc")));
        }

        #[test]
        fn test_finds_graph_positions_in_variable_length_branch_finds_middle_nodes() {
            let fixture = setup_variable_length_branched_graph();
            let bg = BlockGroup::get_by_id(&fixture.conn, &fixture.block_group_id).unwrap();
            let aaa_acc = create_accession_from_edges(
                &fixture.conn,
                fixture.block_group_id,
                "variable-aaa",
                &fixture.path,
                0,
                3,
            );
            let aaa_region = make_region(bg.clone(), aaa_acc, 0, 3, 3, 2, 2);

            let (from_aaa, _) = aaa_region
                .find_graph_positions(&fixture.conn, 2, 2)
                .unwrap();
            assert_eq!(
                position_set(&from_aaa),
                HashSet::from([
                    (HashId::convert_str("node-cc"), 1),
                    (HashId::convert_str("node-gggg"), 1)
                ])
            );

            let ttt_acc = create_accession_from_edges(
                &fixture.conn,
                fixture.block_group_id,
                "variable-ttt",
                &fixture.path,
                5,
                8,
            );
            let ttt_region = make_region(bg, ttt_acc, 5, 8, 3, 0, 0);

            let (from_ttt, _) = ttt_region
                .find_graph_positions(&fixture.conn, -2, -2)
                .unwrap();
            assert_eq!(
                position_set(&from_ttt),
                HashSet::from([
                    (HashId::convert_str("node-cc"), 0),
                    (HashId::convert_str("node-gggg"), 2)
                ])
            );
        }

        #[test]
        fn test_finds_graph_positions_in_variable_length_branch_returns_single_position() {
            let fixture = setup_variable_length_branched_graph();
            let bg = BlockGroup::get_by_id(&fixture.conn, &fixture.block_group_id).unwrap();
            let acc = create_accession_from_edges(
                &fixture.conn,
                fixture.block_group_id,
                "variable-single",
                &fixture.path,
                0,
                3,
            );
            let region = make_region(bg, acc, 0, 3, 3, 1, 1);

            let (positions, _) = region.find_graph_positions(&fixture.conn, 1, 1).unwrap();
            assert_eq!(
                position_set(&positions),
                HashSet::from([(HashId::convert_str("node-aaa"), 2)])
            );
        }

        #[test]
        fn test_finds_graph_positions_in_variable_length_branch_finds_different_ttt_offsets() {
            let fixture = setup_variable_length_branched_graph();
            let bg = BlockGroup::get_by_id(&fixture.conn, &fixture.block_group_id).unwrap();
            let acc = create_accession_from_edges(
                &fixture.conn,
                fixture.block_group_id,
                "variable-ttt-offsets",
                &fixture.path,
                0,
                3,
            );
            let region = make_region(bg, acc, 0, 3, 3, 2, 2);

            let (positions, _) = region.find_graph_positions(&fixture.conn, 6, 6).unwrap();
            assert_eq!(
                position_set(&positions),
                HashSet::from([
                    (HashId::convert_str("node-ttt"), 1),
                    (HashId::convert_str("node-ttt"), 3)
                ])
            );
        }
    }
}
