use std::io::Error as IOError;

use gen_models::errors::{OperationError, QueryError};
use thiserror::Error;

pub use crate::{
    diffs::gfa::GfaDiffError,
    exports::{fasta::FastaExportError, genbank::GenbankExportError, gfa::GfaExportError},
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
}
