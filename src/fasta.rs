use std::io::Error as IOError;

use gen_models::{
    errors::{
        BlockGroupError, CollectionError, EdgeError, FileAdditionError, NodeError, OperationError,
        PathError, QueryError, SampleError, SequenceError,
    },
    region::GenRegionError,
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum FastaError {
    #[error("Operation Error: {0}")]
    OperationError(#[from] OperationError),
    #[error("Asset Error: {0}")]
    FileAdditionError(#[from] FileAdditionError),
    #[error("IO Error: {0}")]
    IOError(#[from] IOError),
    #[error("SQL query Error: {0}")]
    SQLQueryError(#[from] QueryError),
    #[error("Collection creation error: {0}")]
    CollectionError(#[from] CollectionError),
    #[error("Sample creation error: {0}")]
    SampleError(#[from] SampleError),
    #[error("Block group write error: {0}")]
    BlockGroupError(#[from] BlockGroupError),
    #[error("Edge write error: {0}")]
    EdgeError(#[from] EdgeError),
    #[error("Node creation error: {0}")]
    NodeError(#[from] NodeError),
    #[error("Path creation error: {0}")]
    PathError(#[from] PathError),
    #[error("Sequence save error: {0}")]
    SequenceError(#[from] SequenceError),
    #[error("Region Error: {0}")]
    RegionError(#[from] GenRegionError),
    #[error("Missing coordinates for region '{0}'. Use region syntax like 'name:start-end'.")]
    MissingCoordinates(String),
    #[error("Unsupported region type for FASTA update: {0}")]
    UnsupportedRegionType(String),
    #[error("Resolved path '{path_name}' was not found in target block group '{block_group_name}'")]
    MissingResolvedPath {
        path_name: String,
        block_group_name: String,
    },
}
