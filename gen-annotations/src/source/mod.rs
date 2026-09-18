//! Source-specific annotation records and their translated graph intervals.
//!
//! The application chooses and opens an annotation asset. These adapters only receive the
//! selected records (or a reader for the selected asset), match one identifier, and hand the
//! matching records to the existing graph translators. A future attribute index can provide the
//! same matched records without changing the downstream interval-tree or region-resolution layers.

mod bed;
mod gff;

pub use bed::{
    parse_bed_annotation, parse_bed_annotation_records, translate_bed_annotation,
    translate_bed_annotation_records,
};
use gen_core::Workspace;
use gen_models::db::GraphConnection;
pub use gff::{
    parse_gff_annotation, parse_gff_annotation_records, translate_gff_annotation,
    translate_gff_annotation_records,
};
use thiserror::Error;

/// Context used while translating records selected by the application.
pub struct AnnotationTranslationContext<'a> {
    /// Database connection used by the existing translators.
    pub conn: &'a GraphConnection,
    /// Workspace used by graph/path loading.
    pub workspace: &'a Workspace,
    /// Collection containing the selected graph.
    pub collection_name: &'a str,
    /// Sample whose graph receives the translated records.
    pub sample_name: &'a str,
    /// Optional history reference for graph lookup.
    pub history_ref: Option<&'a str>,
}

/// Errors raised while building a file-backed annotation from matched records.
#[derive(Debug, Error)]
pub enum FileAnnotationError {
    #[error("annotation record I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("annotation translation error: {0}")]
    Translation(String),
    #[error("annotation has no translated records")]
    Empty,
    #[error("translated annotation reference is not a node id: {0}")]
    InvalidNode(String),
}
