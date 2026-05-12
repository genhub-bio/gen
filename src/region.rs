use gen_annotations::projection::{accession_edges_to_segments, project_annotation_segments};
pub use gen_core::region::Region;
use gen_core::region::{RegionParseError, RegionResolutionError, RegionResolver};
use gen_models::{
    accession::{Accession, AccessionError},
    annotations::{Annotation, AnnotationError},
    block_group::BlockGroup,
    db::GraphConnection,
    errors::PathError,
    path::Path,
    traits::Query,
};
use thiserror::Error;

#[derive(Debug)]
pub struct ResolvedGenRegion {
    pub block_group: BlockGroup,
    pub path: Path,
    pub accession: Option<Accession>,
    pub kind: ResolvedRegionKind,
    pub anchor_start: i64,
    pub anchor_end: i64,
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
    BlockGroup(#[from] gen_models::block_group::BlockGroupError),
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
    #[error("Invalid region range for {region}: start {start} is greater than end {end}")]
    InvalidRange {
        region: String,
        start: i64,
        end: i64,
    },
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
    path: Path,
    accession: Option<Accession>,
    anchor_start: i64,
    anchor_end: i64,
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
    let anchor_end = path.length(conn)?;
    resolve_target(
        region,
        conn,
        RegionTarget {
            kind: RegionTargetKind::Path,
            block_group,
            path,
            accession: None,
            anchor_start: 0,
            anchor_end,
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
    let anchor_end = path.length(conn)?;
    resolve_target(
        region,
        conn,
        RegionTarget {
            kind: RegionTargetKind::BlockGroup,
            block_group,
            path,
            accession: None,
            anchor_start: 0,
            anchor_end,
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
    let path_length = target.path.length(conn)?;

    let (start, end) = match (region.start, region.end) {
        (None, None) => (target.anchor_start, target.anchor_end),
        (Some(start), None) => {
            if target.kind == RegionTargetKind::Path || target.kind == RegionTargetKind::BlockGroup
            {
                (start, path_length)
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

    if start < 0 || end > path_length {
        return Err(GenRegionError::OutOfBounds {
            region: region.to_string(),
            start,
            end,
            path_length,
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
    let path = Path::get_by_id(conn, &accession.path_id);
    let block_group = BlockGroup::get_by_id(conn, &path.block_group_id)?;
    if block_group.collection_name != collection_name || block_group.sample_name != sample_name {
        return Err(GenRegionError::NotFound(region.name.clone()));
    }

    let (anchor_start, anchor_end) = accession_bounds(conn, &path, &accession)?;
    Ok(RegionTarget {
        kind,
        block_group,
        path,
        accession: Some(accession),
        anchor_start,
        anchor_end,
    })
}

impl ResolvedGenRegion {
    pub fn offset_range(
        &self,
        conn: &GraphConnection,
        start_offset: i64,
        end_offset: i64,
    ) -> Result<(i64, i64), GenRegionError> {
        let path_length = self.path.length(conn)?;
        let (start, end) = match self.kind {
            ResolvedRegionKind::Annotation | ResolvedRegionKind::Accession => (
                self.anchor_start + start_offset,
                self.anchor_start + end_offset,
            ),
            ResolvedRegionKind::Path | ResolvedRegionKind::BlockGroup => (start_offset, end_offset),
        };

        if start < 0 || end > path_length {
            return Err(GenRegionError::OutOfBounds {
                region: self.path.name.clone(),
                start,
                end,
                path_length,
            });
        }

        Ok((start, end))
    }
}

fn accession_bounds(
    conn: &GraphConnection,
    path: &Path,
    accession: &Accession,
) -> Result<(i64, i64), GenRegionError> {
    let projected = project_annotation_segments(
        &accession_edges_to_segments(&Accession::get_edges_by_id(conn, &accession.id)),
        &path.blocks(conn)?,
        false,
    );

    let start = projected.iter().map(|segment| segment.range.start).min();
    let end = projected.iter().map(|segment| segment.range.end).max();

    match (start, end) {
        (Some(start), Some(end)) if start <= end => Ok((start, end)),
        _ => Err(GenRegionError::Unmappable(accession.name.clone())),
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use gen_models::{annotations::add_annotation, db::DbContext, sample::Sample};

    use super::*;
    use crate::{imports::fasta::import_fasta, test_helpers::setup_gen, track_database};

    fn setup_context() -> DbContext {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();
        track_database(conn, op_conn).unwrap();

        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            "test",
            Sample::DEFAULT_NAME,
            false,
        )
        .unwrap();

        add_annotation(
            &context,
            "test",
            "mreB",
            Some("genes"),
            Sample::DEFAULT_NAME,
            "m123:5-15",
        )
        .unwrap();

        context
    }

    #[test]
    fn parses_region_name_only() {
        assert_eq!(
            Region::parse("m123").unwrap(),
            Region {
                name: "m123".to_string(),
                start: None,
                end: None,
            }
        );
    }

    #[test]
    fn parses_region_with_start_only() {
        assert_eq!(
            Region::parse("m123:34").unwrap(),
            Region {
                name: "m123".to_string(),
                start: Some(34),
                end: None,
            }
        );
    }

    #[test]
    fn parses_region_with_negative_range() {
        assert_eq!(
            Region::parse("mreB:-35--10").unwrap(),
            Region {
                name: "mreB".to_string(),
                start: Some(-35),
                end: Some(-10),
            }
        );
    }

    #[test]
    fn resolves_annotation_relative_coordinates() {
        let context = setup_context();
        let region = Region::parse("mreB:-2-3").unwrap();
        let resolved = resolve(
            &region,
            context.graph().conn(),
            "test",
            Sample::DEFAULT_NAME,
        )
        .unwrap();

        assert_eq!(resolved.block_group.name, "m123");
        assert_eq!(resolved.start, 3);
        assert_eq!(resolved.end, 8);
    }

    #[test]
    fn resolves_annotation_without_explicit_coordinates() {
        let context = setup_context();
        let region = Region::parse("mreB").unwrap();
        let resolved = resolve(
            &region,
            context.graph().conn(),
            "test",
            Sample::DEFAULT_NAME,
        )
        .unwrap();

        assert_eq!(resolved.block_group.name, "m123");
        assert_eq!(resolved.start, 5);
        assert_eq!(resolved.end, 15);
    }

    #[test]
    fn resolves_annotation_case_insensitively() {
        let context = setup_context();
        let region = Region::parse("MREB").unwrap();
        let resolved = resolve(
            &region,
            context.graph().conn(),
            "test",
            Sample::DEFAULT_NAME,
        )
        .unwrap();

        assert_eq!(resolved.block_group.name, "m123");
        assert_eq!(resolved.start, 5);
        assert_eq!(resolved.end, 15);
    }

    #[test]
    fn resolves_typed_path() {
        let context = setup_context();
        let region = Region::parse("m123").unwrap();
        let resolved = resolve_path(
            &region,
            context.graph().conn(),
            "test",
            Sample::DEFAULT_NAME,
        )
        .unwrap();

        assert_eq!(resolved.kind, ResolvedRegionKind::Path);
        assert_eq!(resolved.path.name, "m123");
    }

    #[test]
    fn resolves_typed_block_group() {
        let context = setup_context();
        let region = Region::parse("m123").unwrap();
        let resolved = resolve_block_group(
            &region,
            context.graph().conn(),
            "test",
            Sample::DEFAULT_NAME,
        )
        .unwrap();

        assert_eq!(resolved.kind, ResolvedRegionKind::BlockGroup);
        assert_eq!(resolved.block_group.name, "m123");
    }

    #[test]
    fn resolves_typed_annotation() {
        let context = setup_context();
        let region = Region::parse("mreB").unwrap();
        let resolved = resolve_annotation(
            &region,
            context.graph().conn(),
            "test",
            Sample::DEFAULT_NAME,
        )
        .unwrap();

        assert_eq!(resolved.kind, ResolvedRegionKind::Annotation);
        assert_eq!(resolved.start, 5);
        assert_eq!(resolved.end, 15);
    }

    #[test]
    fn typed_resolver_does_not_cross_kinds() {
        let context = setup_context();
        let region = Region::parse("mreB").unwrap();
        let err = resolve_path(
            &region,
            context.graph().conn(),
            "test",
            Sample::DEFAULT_NAME,
        )
        .unwrap_err();

        assert!(matches!(err, GenRegionError::NotFound(name) if name == "mreB"));
    }
}
