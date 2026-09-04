use std::collections::HashMap;

use gen_core::{
    HashId, PATH_END_NODE_ID, PATH_END_SEQUENCE_HASH, PATH_START_NODE_ID, PATH_START_SEQUENCE_HASH,
    Sha256Hash, Workspace, traits::Capnp,
};
use rusqlite::{Row, params, types::Value};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    ModelSelect,
    db::GraphConnection,
    gen_models_capnp::node,
    sequence::Sequence,
    traits::{self, *},
};

#[derive(Clone, Debug, Eq, Deserialize, Hash, Serialize, PartialEq, ModelSelect)]
pub struct Node {
    pub id: HashId,
    pub sequence_hash: Sha256Hash,
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
    /// Creates nodes in bounded batches, retaining any row that already has the same identifier.
    #[cfg_attr(feature = "profiling", tracing::instrument(skip(conn, nodes)))]
    pub fn bulk_create(conn: &GraphConnection, nodes: &[Node]) -> Result<(), NodeError> {
        let batch_size = traits::max_rows_per_batch(conn, 2);

        for chunk in nodes.chunks(batch_size) {
            let mut sql = String::from("INSERT OR IGNORE INTO nodes (id, sequence_hash) VALUES ");
            for row_index in 0..chunk.len() {
                if row_index > 0 {
                    sql.push(',');
                }
                sql.push_str("(?, ?)");
            }
            sql.push(';');

            let mut values = Vec::with_capacity(chunk.len() * 2);
            for node in chunk {
                values.push(Value::from(node.id));
                values.push(Value::from(node.sequence_hash));
            }
            let mut statement = conn.prepare_cached(&sql)?;
            statement.execute(rusqlite::params_from_iter(values))?;
        }

        Ok(())
    }

    #[cfg_attr(
        all(debug_assertions, feature = "profiling"),
        tracing::instrument(skip(conn, sequence_hash, node_hash))
    )]
    pub fn create(
        conn: &GraphConnection,
        sequence_hash: &Sha256Hash,
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
        workspace: &Workspace,
        node_ids: &[HashId],
        history_ref: Option<&str>,
    ) -> HashMap<HashId, Sequence> {
        let nodes = Node::query_by_ids(conn, node_ids, history_ref);
        let sequence_hashes_by_node_id = nodes
            .iter()
            .map(|node| (node.id, node.sequence_hash))
            .collect::<HashMap<HashId, Sha256Hash>>();
        let sequences_by_hash: HashMap<Sha256Hash, Sequence> = HashMap::from_iter(
            Sequence::query_by_ids(
                conn,
                workspace,
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
            .into_iter()
            .map(|(node_id, sequence_hash)| {
                (
                    node_id,
                    sequences_by_hash
                        .get(&sequence_hash)
                        .expect("should load sequence for node")
                        .clone(),
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
            sequence_hash: PATH_START_SEQUENCE_HASH,
        }
    }

    pub fn get_end_node() -> Node {
        Node {
            id: PATH_END_NODE_ID,
            sequence_hash: PATH_END_SEQUENCE_HASH,
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
            sequence_hash: Sha256Hash::convert_str("test_sequence_hash"),
        };

        let mut message = TypedBuilder::<node::Owned>::new_default();
        let mut root = message.init_root();
        node.write_capnp(&mut root);

        let deserialized = Node::read_capnp(root.into_reader());
        assert_eq!(node, deserialized);
    }
}
