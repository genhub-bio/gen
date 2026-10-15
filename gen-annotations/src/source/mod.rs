//! Source-specific annotation parsing and graph translation.
//!
//! Parsers retain format-neutral source locations on `Annotation`; translation then resolves
//! those locations against the selected sample's graph.

mod bed;
mod gff;

pub use bed::{
    parse_bed_annotation, parse_bed_annotation_records, translate_bed_annotation,
    translate_bed_annotation_records,
};
use gen_core::{HashId, Workspace};
use gen_models::db::GraphConnection;
pub use gff::{
    parse_gff_annotation, parse_gff_annotation_records, translate_gff_annotation,
    translate_gff_annotation_records,
};
use thiserror::Error;

/// Graph context used to translate a parsed annotation's source locations.
pub struct AnnotationTranslationContext<'a> {
    /// Database connection used to load the selected sample graph.
    pub conn: &'a GraphConnection,
    /// Workspace used by graph/path loading.
    pub workspace: &'a Workspace,
    /// Collection containing the selected graph.
    pub collection_name: &'a str,
    /// Sample whose graph receives the translated records.
    pub sample_name: &'a str,
    /// Block group assigned to the translated annotation's synthetic accession.
    pub block_group_id: HashId,
    /// Optional history reference for graph lookup.
    pub history_ref: Option<&'a str>,
}

/// Errors raised while parsing or translating source annotations.
#[derive(Debug, Error)]
pub enum FileAnnotationError {
    #[error("annotation record I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("annotation translation error: {0}")]
    Translation(String),
    #[error("annotation has no records or translated graph intervals")]
    Empty,
    #[error("annotation has no source locations to translate")]
    MissingSourceLocations,
}
