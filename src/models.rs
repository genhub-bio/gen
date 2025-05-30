use thiserror::Error;

pub mod accession;
pub mod block_group;
pub mod block_group_edge;
pub mod collection;
pub mod edge;
pub mod file_types;
pub mod metadata;
pub mod node;
pub mod operations;
pub mod path;
pub mod path_edge;
pub mod sample;
pub mod sequence;
pub mod strand;
pub mod traits;

#[derive(Clone, Debug, Eq, Error, Hash, PartialEq)]
pub enum QueryError {
    #[error("Results not found: {0}")]
    ResultsNotFound(String),
}
