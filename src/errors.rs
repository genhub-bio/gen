use std::io::Error as IOError;

use gen_models::{
    errors::{BlockGroupError, NodeError, OperationError, PathError, QueryError, SequenceError},
    region::GenRegionError,
    sample::SampleError,
};
use thiserror::Error;

pub use crate::{
    diffs::gfa::GfaDiffError,
    exports::{fasta::FastaExportError, genbank::GenbankExportError, gfa::GfaExportError},
    patch::CreatePatchError,
    updates::gaf::GafUpdateError,
};

#[derive(Debug, Error)]
pub enum SequenceUpdateError {
    #[error("Operation Error: {0}")]
    OperationError(#[from] OperationError),
    #[error("IO Error: {0}")]
    IOError(#[from] IOError),
    #[error("SQL query Error: {0}")]
    SQLQueryError(#[from] QueryError),
    #[error("Node creation error: {0}")]
    NodeError(#[from] NodeError),
    #[error("Path creation error: {0}")]
    PathError(#[from] PathError),
    #[error("Block group creation error: {0}")]
    BlockGroupError(#[from] BlockGroupError),
    #[error("Sequence save error: {0}")]
    SequenceError(#[from] SequenceError),
    #[error("Sample error: {0}")]
    SampleError(#[from] SampleError),
    #[error("Missing segment '{0}' in GFA input")]
    MissingSegment(String),
    #[error("Missing strand for path '{path_name}' at segment index {index}")]
    MissingPathStrand { path_name: String, index: usize },
    #[error("Path '{0}' has no segments")]
    EmptyPath(String),
    #[error("No path block found for path id {path_id} at coordinate {coordinate}")]
    MissingPathBlock { path_id: String, coordinate: i64 },
    #[error("Region Error: {0}")]
    RegionError(#[from] GenRegionError),
    #[error("Missing coordinates for region '{0}'. Use region syntax like 'name:start-end'.")]
    MissingCoordinates(String),
    #[error("Unsupported region type for sequence update: {0}")]
    UnsupportedRegionType(String),
    #[error("Resolved path '{path_name}' was not found in target block group '{block_group_name}'")]
    MissingResolvedPath {
        path_name: String,
        block_group_name: String,
    },
}
