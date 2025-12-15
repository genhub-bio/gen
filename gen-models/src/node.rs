use std::collections::HashMap;

use gen_core::{HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, calculate_hash, traits::Capnp};
use rusqlite::{Row, params};
use serde::{Deserialize, Serialize};

use crate::{db::GraphDb, gen_models_capnp::node, sequence::Sequence, traits::*};

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

impl Node {
    pub fn create(conn: &impl GraphDb, sequence_hash: &HashId, node_hash: &HashId) -> HashId {
        let conn = conn.graph_conn();
        let insert_statement = "INSERT INTO nodes (id, sequence_hash) VALUES (?1, ?2);";
        let mut stmt = conn.prepare_cached(insert_statement).unwrap();
        match stmt.execute(params![node_hash, sequence_hash]) {
            Ok(_) => *node_hash,
            Err(rusqlite::Error::SqliteFailure(err, _details)) => {
                if err.code == rusqlite::ErrorCode::ConstraintViolation {
                    *node_hash
                } else {
                    panic!("something bad happened querying the database")
                }
            }
            Err(_) => {
                panic!("something bad happened querying the database")
            }
        }
    }

    pub fn get_sequences_by_node_ids(
        conn: &impl GraphDb,
        node_ids: &[HashId],
    ) -> HashMap<HashId, Sequence> {
        let conn = conn.graph_conn();
        let nodes = Node::query_by_ids(conn, node_ids);
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
