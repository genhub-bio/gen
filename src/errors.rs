use std::io::Error as IOError;

use gen_models::{
    block_group::BlockGroupError,
    errors::{OperationError, QueryError},
    node::NodeError,
    path::PathError,
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
}
