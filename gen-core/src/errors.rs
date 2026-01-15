use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum ConfigError {
    #[error("Failed to find Gen directory")]
    GenDirectoryNotFound,
    #[error("Unable to determine repository root")]
    RepoRootNotFound,
}

#[derive(Debug, Error, PartialEq)]
pub enum ConnectionError {
    #[error("Failed to open database connection: {0}")]
    OpenFailed(#[from] rusqlite::Error),
    #[error("Database tracking error: {0}")]
    DatabaseTracking(String),
    #[error("Config Error: {0}")]
    ConfigError(#[from] ConfigError),
}

#[derive(Debug, Error, PartialEq)]
pub enum StrandError {
    #[error("Invalid Strand: {0}")]
    InvalidStrand(String),
    #[error("Rusqlite Error: {0}")]
    SqliteError(#[from] rusqlite::Error),
}
