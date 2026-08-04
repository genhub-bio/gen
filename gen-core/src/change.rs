use crate::PathBlock;

/// A requested sequence-graph change paired with its resolved region representation.
///
/// The region is generic so persistence and graph layers can use the same change payload without
/// introducing a dependency from core onto either layer.
#[derive(Clone, Debug)]
pub struct BlockGroupChange<Region> {
    pub region: Region,
    pub path_accession: Option<String>,
    pub block: PathBlock,
    pub chromosome_index: i64,
    pub phased: i64,
    pub preserve_edge: bool,
}
