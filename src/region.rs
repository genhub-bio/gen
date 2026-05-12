pub use gen_core::region::Region;
use gen_core::{HashId, PathBlock, Strand, is_end_node, is_start_node, region::RegionParseError};
use gen_models::{
    accession::{Accession, AccessionEdge},
    annotations::Annotation,
    block_group::BlockGroup,
    db::GraphConnection,
    errors::PathError,
    path::Path,
    sample::Sample,
    traits::Query,
};
use rusqlite::params;
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

#[derive(Clone, Copy, Debug)]
struct AccessionSegment {
    node_id: HashId,
    start: i64,
    end: i64,
}

pub trait RegionResolverExt {
    fn resolve(
        &self,
        conn: &GraphConnection,
        collection_name: &str,
        sample_name: &str,
    ) -> Result<ResolvedGenRegion, GenRegionError>;

    fn resolve_path(
        &self,
        conn: &GraphConnection,
        collection_name: &str,
        sample_name: &str,
    ) -> Result<ResolvedGenRegion, GenRegionError>;

    fn resolve_block_group(
        &self,
        conn: &GraphConnection,
        collection_name: &str,
        sample_name: &str,
    ) -> Result<ResolvedGenRegion, GenRegionError>;

    fn resolve_accession(
        &self,
        conn: &GraphConnection,
        collection_name: &str,
        sample_name: &str,
    ) -> Result<ResolvedGenRegion, GenRegionError>;

    fn resolve_annotation(
        &self,
        conn: &GraphConnection,
        collection_name: &str,
        sample_name: &str,
    ) -> Result<ResolvedGenRegion, GenRegionError>;
}

impl RegionResolverExt for Region {
    fn resolve(
        &self,
        conn: &GraphConnection,
        collection_name: &str,
        sample_name: &str,
    ) -> Result<ResolvedGenRegion, GenRegionError> {
        let target = find_target(self, conn, collection_name, sample_name)?;
        resolve_target(self, conn, target)
    }

    fn resolve_path(
        &self,
        conn: &GraphConnection,
        collection_name: &str,
        sample_name: &str,
    ) -> Result<ResolvedGenRegion, GenRegionError> {
        let target = find_path_target(self, conn, collection_name, sample_name)?
            .ok_or_else(|| GenRegionError::NotFound(self.name.clone()))?;
        resolve_target(self, conn, target)
    }

    fn resolve_block_group(
        &self,
        conn: &GraphConnection,
        collection_name: &str,
        sample_name: &str,
    ) -> Result<ResolvedGenRegion, GenRegionError> {
        let target = find_block_group_target(self, conn, collection_name, sample_name)?
            .ok_or_else(|| GenRegionError::NotFound(self.name.clone()))?;
        resolve_target(self, conn, target)
    }

    fn resolve_accession(
        &self,
        conn: &GraphConnection,
        collection_name: &str,
        sample_name: &str,
    ) -> Result<ResolvedGenRegion, GenRegionError> {
        let target = find_accession_target(self, conn, collection_name, sample_name)?
            .ok_or_else(|| GenRegionError::NotFound(self.name.clone()))?;
        resolve_target(self, conn, target)
    }

    fn resolve_annotation(
        &self,
        conn: &GraphConnection,
        collection_name: &str,
        sample_name: &str,
    ) -> Result<ResolvedGenRegion, GenRegionError> {
        let target = find_annotation_target(self, conn, collection_name, sample_name)?
            .ok_or_else(|| GenRegionError::NotFound(self.name.clone()))?;
        resolve_target(self, conn, target)
    }
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

fn find_target(
    region: &Region,
    conn: &GraphConnection,
    collection_name: &str,
    sample_name: &str,
) -> Result<RegionTarget, GenRegionError> {
    if let Some(target) = find_block_group_target(region, conn, collection_name, sample_name)? {
        return Ok(target);
    }

    if let Some(target) = find_path_target(region, conn, collection_name, sample_name)? {
        return Ok(target);
    }

    if let Some(target) = find_annotation_target(region, conn, collection_name, sample_name)? {
        return Ok(target);
    }

    if let Some(target) = find_accession_target(region, conn, collection_name, sample_name)? {
        return Ok(target);
    }

    Err(GenRegionError::NotFound(region.name.clone()))
}

fn find_path_target(
    region: &Region,
    conn: &GraphConnection,
    collection_name: &str,
    sample_name: &str,
) -> Result<Option<RegionTarget>, GenRegionError> {
    let region_name = region.name.to_lowercase();
    let matches = Path::query_for_collection_and_sample(conn, collection_name, sample_name)
        .into_iter()
        .filter(|path| path.name.to_lowercase() == region_name)
        .collect::<Vec<_>>();

    if matches.is_empty() {
        return Ok(None);
    }

    if matches.len() > 1 {
        return Err(GenRegionError::Ambiguous(format!(
            "multiple paths named {}",
            region.name
        )));
    }

    let path = matches.into_iter().next().unwrap();
    let block_group = BlockGroup::get_by_id(conn, &path.block_group_id)?;
    let anchor_end = path.length(conn)?;
    Ok(Some(RegionTarget {
        kind: RegionTargetKind::Path,
        block_group,
        path,
        accession: None,
        anchor_start: 0,
        anchor_end,
    }))
}

fn find_annotation_target(
    region: &Region,
    conn: &GraphConnection,
    collection_name: &str,
    sample_name: &str,
) -> Result<Option<RegionTarget>, GenRegionError> {
    let region_name = region.name.to_lowercase();
    let matches = Annotation::query(
        conn,
        "select a.* from annotations a \
         join accessions acc on a.accession_id = acc.id \
         join paths p on acc.path_id = p.id \
         join block_groups bg on p.block_group_id = bg.id \
         where bg.collection_name = ?1 and bg.sample_name = ?2 and lower(a.name) = lower(?3)",
        params![collection_name, sample_name, region_name],
    );

    if matches.is_empty() {
        return Ok(None);
    }

    if matches.len() > 1 {
        return Err(GenRegionError::Ambiguous(format!(
            "multiple annotations named {}",
            region.name
        )));
    }

    let annotation = matches.into_iter().next().unwrap();
    let accession = Accession::query(
        conn,
        "select * from accessions where id = ?1",
        params![annotation.accession_id],
    )
    .into_iter()
    .next()
    .ok_or_else(|| GenRegionError::Unmappable(region.name.clone()))?;
    target_from_accession(
        region,
        conn,
        collection_name,
        sample_name,
        RegionTargetKind::Annotation,
        accession,
    )
    .map(Some)
}

fn find_accession_target(
    region: &Region,
    conn: &GraphConnection,
    collection_name: &str,
    sample_name: &str,
) -> Result<Option<RegionTarget>, GenRegionError> {
    let region_name = region.name.to_lowercase();
    let matches = Accession::query(
        conn,
        "select a.* from accessions a \
         join paths p on a.path_id = p.id \
         join block_groups bg on p.block_group_id = bg.id \
         where bg.collection_name = ?1 and bg.sample_name = ?2 and lower(a.name) = lower(?3)",
        params![collection_name, sample_name, region_name],
    );

    if matches.is_empty() {
        return Ok(None);
    }

    if matches.len() > 1 {
        return Err(GenRegionError::Ambiguous(format!(
            "multiple accessions named {}",
            region.name
        )));
    }

    let accession = matches.into_iter().next().unwrap();
    target_from_accession(
        region,
        conn,
        collection_name,
        sample_name,
        RegionTargetKind::Accession,
        accession,
    )
    .map(Some)
}

fn find_block_group_target(
    region: &Region,
    conn: &GraphConnection,
    collection_name: &str,
    sample_name: &str,
) -> Result<Option<RegionTarget>, GenRegionError> {
    let region_name = region.name.to_lowercase();
    let matches = Sample::get_block_groups(conn, collection_name, sample_name)
        .into_iter()
        .filter(|block_group| block_group.name.to_lowercase() == region_name)
        .collect::<Vec<_>>();

    if matches.is_empty() {
        return Ok(None);
    }

    if matches.len() > 1 {
        return Err(GenRegionError::Ambiguous(format!(
            "multiple block groups named {}",
            region.name
        )));
    }

    let block_group = matches.into_iter().next().unwrap();
    let path = BlockGroup::get_current_path(conn, &block_group.id);
    let anchor_end = path.length(conn)?;
    Ok(Some(RegionTarget {
        kind: RegionTargetKind::BlockGroup,
        block_group,
        path,
        accession: None,
        anchor_start: 0,
        anchor_end,
    }))
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
    let blocks = path.blocks(conn)?;
    let segments = accession_segments(conn, &accession.id);

    let mut min_coordinate = None;
    let mut max_coordinate = None;

    for segment in segments {
        let block = blocks
            .iter()
            .find(|block| {
                block.node_id == segment.node_id
                    && segment.start >= block.sequence_start
                    && segment.end <= block.sequence_end
            })
            .ok_or_else(|| GenRegionError::Unmappable(accession.name.clone()))?;
        let start = translate_coordinate(block, segment.start);
        let end = translate_coordinate(block, segment.end);
        let segment_start = start.min(end);
        let segment_end = start.max(end);
        min_coordinate = Some(
            min_coordinate
                .map(|current: i64| current.min(segment_start))
                .unwrap_or(segment_start),
        );
        max_coordinate = Some(
            max_coordinate
                .map(|current: i64| current.max(segment_end))
                .unwrap_or(segment_end),
        );
    }

    match (min_coordinate, max_coordinate) {
        (Some(start), Some(end)) => Ok((start, end)),
        _ => Err(GenRegionError::Unmappable(accession.name.clone())),
    }
}

fn accession_segments(conn: &GraphConnection, accession_id: &HashId) -> Vec<AccessionSegment> {
    let edges = Accession::get_edges_by_id(conn, accession_id);
    edges_to_segments(&edges)
}

fn edges_to_segments(edges: &[AccessionEdge]) -> Vec<AccessionSegment> {
    let mut segments = Vec::new();
    let mut current_node = None;
    let mut current_start = None;

    for edge in edges {
        if is_start_node(edge.source_node_id) {
            current_node = Some(edge.target_node_id);
            current_start = Some(edge.target_coordinate);
            continue;
        }

        if is_end_node(edge.target_node_id) {
            if let (Some(node_id), Some(start)) = (current_node, current_start) {
                let (segment_start, segment_end) = if start <= edge.source_coordinate {
                    (start, edge.source_coordinate)
                } else {
                    (edge.source_coordinate, start)
                };
                segments.push(AccessionSegment {
                    node_id,
                    start: segment_start,
                    end: segment_end,
                });
            }
            break;
        }

        if let (Some(node_id), Some(start)) = (current_node, current_start) {
            let (segment_start, segment_end) = if start <= edge.source_coordinate {
                (start, edge.source_coordinate)
            } else {
                (edge.source_coordinate, start)
            };
            segments.push(AccessionSegment {
                node_id,
                start: segment_start,
                end: segment_end,
            });
        }

        current_node = Some(edge.target_node_id);
        current_start = Some(edge.target_coordinate);
    }

    segments
}

fn translate_coordinate(block: &PathBlock, coordinate: i64) -> i64 {
    match block.strand {
        Strand::Forward => block.path_start + coordinate - block.sequence_start,
        Strand::Reverse => block.path_start + block.sequence_end - coordinate,
        Strand::Unknown | Strand::ImportantButUnknown => {
            block.path_start + coordinate - block.sequence_start
        }
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
        let resolved = Region::parse("mreB:-2-3")
            .unwrap()
            .resolve(context.graph().conn(), "test", Sample::DEFAULT_NAME)
            .unwrap();

        assert_eq!(resolved.block_group.name, "m123");
        assert_eq!(resolved.start, 3);
        assert_eq!(resolved.end, 8);
    }

    #[test]
    fn resolves_annotation_without_explicit_coordinates() {
        let context = setup_context();
        let resolved = Region::parse("mreB")
            .unwrap()
            .resolve(context.graph().conn(), "test", Sample::DEFAULT_NAME)
            .unwrap();

        assert_eq!(resolved.block_group.name, "m123");
        assert_eq!(resolved.start, 5);
        assert_eq!(resolved.end, 15);
    }

    #[test]
    fn resolves_annotation_case_insensitively() {
        let context = setup_context();
        let resolved = Region::parse("MREB")
            .unwrap()
            .resolve(context.graph().conn(), "test", Sample::DEFAULT_NAME)
            .unwrap();

        assert_eq!(resolved.block_group.name, "m123");
        assert_eq!(resolved.start, 5);
        assert_eq!(resolved.end, 15);
    }

    #[test]
    fn resolves_typed_path() {
        let context = setup_context();
        let resolved = Region::parse("m123")
            .unwrap()
            .resolve_path(context.graph().conn(), "test", Sample::DEFAULT_NAME)
            .unwrap();

        assert_eq!(resolved.kind, ResolvedRegionKind::Path);
        assert_eq!(resolved.path.name, "m123");
    }

    #[test]
    fn resolves_typed_block_group() {
        let context = setup_context();
        let resolved = Region::parse("m123")
            .unwrap()
            .resolve_block_group(context.graph().conn(), "test", Sample::DEFAULT_NAME)
            .unwrap();

        assert_eq!(resolved.kind, ResolvedRegionKind::BlockGroup);
        assert_eq!(resolved.block_group.name, "m123");
    }

    #[test]
    fn resolves_typed_annotation() {
        let context = setup_context();
        let resolved = Region::parse("mreB")
            .unwrap()
            .resolve_annotation(context.graph().conn(), "test", Sample::DEFAULT_NAME)
            .unwrap();

        assert_eq!(resolved.kind, ResolvedRegionKind::Annotation);
        assert_eq!(resolved.start, 5);
        assert_eq!(resolved.end, 15);
    }

    #[test]
    fn typed_resolver_does_not_cross_kinds() {
        let context = setup_context();
        let err = Region::parse("mreB")
            .unwrap()
            .resolve_path(context.graph().conn(), "test", Sample::DEFAULT_NAME)
            .unwrap_err();

        assert!(matches!(err, GenRegionError::NotFound(name) if name == "mreB"));
    }
}
