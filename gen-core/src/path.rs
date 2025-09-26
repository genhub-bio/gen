use crate::{HashId, Strand};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct PathBlock {
    pub id: i64,
    pub node_id: HashId,
    pub block_sequence: String,
    pub sequence_start: i64,
    pub sequence_end: i64,
    pub path_start: i64,
    pub path_end: i64,
    pub strand: Strand,
}
