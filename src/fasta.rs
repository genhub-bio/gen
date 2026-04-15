use std::io::Error as IOError;

use gen_models::{
    block_group::BlockGroupError,
    edge::EdgeError,
    errors::{OperationError, QueryError},
    node::NodeError,
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum FastaError {
    #[error("Operation Error: {0}")]
    OperationError(#[from] OperationError),
    #[error("IO Error: {0}")]
    IOError(#[from] IOError),
    #[error("SQL query Error: {0}")]
    SQLQueryError(#[from] QueryError),
    #[error("Block group write error: {0}")]
    BlockGroupError(#[from] BlockGroupError),
    #[error("Edge write error: {0}")]
    EdgeError(#[from] EdgeError),
    #[error("Node creation error: {0}")]
    NodeError(#[from] NodeError),
}
