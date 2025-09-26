use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum ConnectionError {
    #[error("Failed to open database connection: {0}")]
    OpenFailed(#[from] rusqlite::Error),
    #[error("Database tracking error: {0}")]
    DatabaseTracking(String),
}

#[derive(Debug, Error, PartialEq)]
pub enum StrandError {
    #[error("Invalid Strand: {0}")]
    InvalidStrand(String),
    #[error("Rusqlite Error: {0}")]
    SqliteError(#[from] rusqlite::Error),
}
