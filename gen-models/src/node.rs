use std::collections::HashMap;

use gen_core::{HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, calculate_hash, traits::Capnp};
use rusqlite::{Row, params, types::Value};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    db::GraphConnection,
    gen_models_capnp::node,
    sequence::Sequence,
    traits::{self, *},
};

#[derive(Clone, Debug, Eq, Deserialize, Hash, Serialize, PartialEq)]
pub struct Node {
    pub id: HashId,
    pub sequence_hash: HashId,
}

impl<'a> Capnp<'a> for Node {
    type Builder = node::Builder<'a>;
    type Reader = node::Reader<'a>;

    fn write_capnp(&self, builder: &mut Self::Builder) {
        builder.set_id(&self.id.0).unwrap();
        builder.set_sequence_hash(&self.sequence_hash.0).unwrap();
    }

    fn read_capnp(reader: Self::Reader) -> Self {
        let id = reader
            .get_id()
            .unwrap()
            .as_slice()
            .unwrap()
            .try_into()
            .unwrap();
        let sequence_hash = reader
            .get_sequence_hash()
            .unwrap()
            .as_slice()
            .unwrap()
            .try_into()
            .unwrap();

        Node { id, sequence_hash }
    }
}

impl Query for Node {
    type Model = Node;

    const TABLE_NAME: &'static str = "nodes";

    fn process_row(row: &Row) -> Self::Model {
        Node {
            id: row.get(0).unwrap(),
            sequence_hash: row.get(1).unwrap(),
        }
    }
}

#[derive(Debug, Error, PartialEq)]
pub enum NodeError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] rusqlite::Error),
}

impl Node {
    #[cfg_attr(
        all(debug_assertions, feature = "profiling"),
        tracing::instrument(skip(conn, sequence_hash, node_hash))
    )]
    pub fn create(
        conn: &GraphConnection,
        sequence_hash: &HashId,
        node_hash: &HashId,
    ) -> Result<HashId, NodeError> {
        let insert_statement = "INSERT INTO nodes (id, sequence_hash) VALUES (?1, ?2);";
        let mut stmt = match conn.prepare_cached(insert_statement) {
            Ok(s) => s,
            Err(e) => return Err(NodeError::DatabaseError(e)),
        };
        match stmt.execute(params![node_hash, sequence_hash]) {
            Ok(_) => Ok(*node_hash),
            Err(rusqlite::Error::SqliteFailure(e, _))
                if e.code == rusqlite::ErrorCode::ConstraintViolation =>
            {
                // Node already exists, return the existing node hash
                Ok(*node_hash)
            }
            Err(e) => Err(NodeError::DatabaseError(e)),
        }
    }

    pub fn get_sequences_by_node_ids(
        conn: &GraphConnection,
        node_ids: &[HashId],
        history_ref: Option<&str>,
    ) -> HashMap<HashId, Sequence> {
        let nodes = Node::query_by_ids(conn, node_ids, history_ref);
        let sequence_hashes_by_node_id = nodes
            .iter()
            .map(|node| (node.id, node.sequence_hash))
            .collect::<HashMap<HashId, HashId>>();
        let sequences_by_hash: HashMap<HashId, Sequence> = HashMap::from_iter(
            Sequence::query_by_ids(
                conn,
                &sequence_hashes_by_node_id
                    .values()
                    .cloned()
                    .collect::<Vec<_>>(),
                history_ref,
            )
            .iter()
            .map(|seq| (seq.hash, seq.clone())),
        );
        sequence_hashes_by_node_id
            .clone()
            .into_iter()
            .map(|(node_id, sequence_hash)| {
                (
                    node_id,
                    sequences_by_hash.get(&sequence_hash).unwrap().clone(),
                )
            })
            .collect::<HashMap<HashId, Sequence>>()
    }

    pub fn query_nodes_length(
        conn: &GraphConnection,
        node_ids: &[HashId],
    ) -> Result<HashMap<HashId, i64>, NodeError> {
        if node_ids.is_empty() {
            return Ok(HashMap::new());
        }

        let mut lengths = HashMap::new();
        let batch_size = traits::max_rows_per_batch(conn, 1);
        let query = "
            WITH arr AS (
                SELECT value, rowid AS pos
                FROM rarray(?1)
            )
            SELECT n.id, s.length
            FROM nodes n
            JOIN sequences s ON s.hash = n.sequence_hash
            JOIN arr ON n.id = arr.value
            ORDER BY arr.pos;
        ";

        for chunk in node_ids.chunks(batch_size) {
            let values: Vec<Value> = chunk.iter().copied().map(Value::from).collect();
            let mut stmt = conn.prepare_cached(query)?;
            let rows = stmt.query_map(params![std::rc::Rc::new(values)], |row| {
                Ok((row.get::<_, HashId>(0)?, row.get::<_, i64>(1)?))
            })?;

            for row in rows {
                let (node_id, length) = row?;
                lengths.insert(node_id, length);
            }
        }

        Ok(lengths)
    }

    pub fn get_start_node() -> Node {
        Node {
            id: PATH_START_NODE_ID,
            sequence_hash: HashId(calculate_hash(
                "start-node-yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy",
            )),
        }
    }

    pub fn get_end_node() -> Node {
        Node {
            id: PATH_END_NODE_ID,
            sequence_hash: HashId(calculate_hash(
                "end-node-zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz",
            )),
        }
    }
}
#[cfg(test)]
mod tests {
    use capnp::message::TypedBuilder;

    use super::*;

    #[test]
    fn test_capnp_serialization() {
        let node = Node {
            id: HashId::convert_str("1"),
            sequence_hash: HashId::convert_str("test_sequence_hash"),
        };

        let mut message = TypedBuilder::<node::Owned>::new_default();
        let mut root = message.init_root();
        node.write_capnp(&mut root);

        let deserialized = Node::read_capnp(root.into_reader());
        assert_eq!(node, deserialized);
    }
}
