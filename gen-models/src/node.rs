use std::{
    collections::HashMap,
    sync::{LazyLock, Mutex},
};

use cached::{Cached, SizedCache};
use gen_core::{
    HashId, PATH_END_NODE_ID, PATH_END_SEQUENCE_HASH, PATH_START_NODE_ID, PATH_START_SEQUENCE_HASH,
    Sha256Hash, Workspace, traits::Capnp,
};
use rusqlite::{OptionalExtension, params, types::Value};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    ModelSelect,
    db::{GraphConnection, max_rows_per_batch},
    gen_models_capnp::node,
    sequence::{Sequence, SequenceError, is_circular_sequence_type, stored_sequence_range},
};

#[derive(Clone, Debug, Eq, Deserialize, Hash, Serialize, PartialEq, ModelSelect)]
#[model_select(table = "nodes")]
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

#[derive(Debug, Error, PartialEq)]
pub enum NodeError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] rusqlite::Error),
}

impl Node {
    /// Creates nodes in bounded batches, retaining any row that already has the same identifier.
    #[cfg_attr(feature = "profiling", tracing::instrument(skip(conn, nodes)))]
    pub fn bulk_create(conn: &GraphConnection, nodes: &[Node]) -> Result<(), NodeError> {
        let batch_size = max_rows_per_batch(conn, 2);

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
        let node_select = Node::select(conn).with_ref(history_ref);
        let nodes = node_select
            .query_by_ids(node_ids.iter().copied())
            .expect("should load nodes by id");
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

    /// Bases `start..end` of the sequence backing `node_id`, or `None` if the node doesn't
    /// exist. A sequence stored in the database is read in cached chunks rather than whole, so a
    /// viewer can show short slices of chromosome-length nodes cheaply.
    pub fn get_sequence_range(
        conn: &GraphConnection,
        workspace: &Workspace,
        node_id: HashId,
        start: i64,
        end: i64,
    ) -> Result<Option<String>, SequenceError> {
        let Some(NodeSequenceInfo {
            hash,
            sequence_type,
            length,
            asset_ref_id,
        }) = node_sequence_info(conn, node_id)?
        else {
            return Ok(None);
        };
        if asset_ref_id.is_some() {
            // External sequences already read only the requested region from their asset.
            return Node::get_sequences_by_node_ids(conn, workspace, &[node_id], None)
                .get(&node_id)
                .map(|sequence| sequence.get_sequence(start, end))
                .transpose();
        }
        stored_sequence_range(
            conn,
            &hash,
            length,
            is_circular_sequence_type(&sequence_type),
            start,
            end,
        )
        .map(Some)
    }

    pub fn query_nodes_length(
        conn: &GraphConnection,
        node_ids: &[HashId],
    ) -> Result<HashMap<HashId, i64>, NodeError> {
        if node_ids.is_empty() {
            return Ok(HashMap::new());
        }

        let mut lengths = HashMap::new();
        let batch_size = max_rows_per_batch(conn, 1);
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
/// What [`Node::get_sequence_range`] needs to know about a node's sequence, short of its text.
#[derive(Clone, Debug)]
struct NodeSequenceInfo {
    hash: Sha256Hash,
    sequence_type: String,
    length: i64,
    asset_ref_id: Option<HashId>,
}

/// How many nodes' [`NodeSequenceInfo`] [`node_sequence_info`] keeps.
const NODE_SEQUENCE_INFO_CAPACITY: usize = 65_536;

/// [`NodeSequenceInfo`] by `(database path, node id)`. A node's sequence never changes once
/// stored, so an entry stays valid for as long as the database at that path exists.
static NODE_SEQUENCE_INFO: LazyLock<Mutex<SizedCache<(String, HashId), NodeSequenceInfo>>> =
    LazyLock::new(|| Mutex::new(SizedCache::with_size(NODE_SEQUENCE_INFO_CAPACITY)));

/// Look up `node_id`'s [`NodeSequenceInfo`], cached per database file. `length` and
/// `asset_ref_id` come after the sequence text in a `sequences` row, so reading them walks past
/// the whole stored sequence; for a chromosome-length node that is most of what a viewer spends
/// fetching a short slice. In-memory databases have no path to key on and are never cached.
fn node_sequence_info(
    conn: &GraphConnection,
    node_id: HashId,
) -> Result<Option<NodeSequenceInfo>, SequenceError> {
    let cache_key = conn
        .path()
        .filter(|path| !path.is_empty() && *path != ":memory:")
        .map(|path| (path.to_string(), node_id));
    if let Some(key) = &cache_key {
        let mut cache = NODE_SEQUENCE_INFO
            .lock()
            .map_err(|err| SequenceError::CachePoisoned(err.to_string()))?;
        if let Some(info) = cache.cache_get(key) {
            return Ok(Some(info.clone()));
        }
    }
    let info = conn
        .query_row(
            "SELECT sequences.hash, sequences.sequence_type, sequences.length,
                    sequences.asset_ref_id
             FROM nodes JOIN sequences ON sequences.hash = nodes.sequence_hash
             WHERE nodes.id = ?1",
            params![node_id],
            |row| {
                Ok(NodeSequenceInfo {
                    hash: row.get(0)?,
                    sequence_type: row.get(1)?,
                    length: row.get(2)?,
                    asset_ref_id: row.get(3)?,
                })
            },
        )
        .optional()?;
    if let (Some(key), Some(info)) = (cache_key, &info) {
        NODE_SEQUENCE_INFO
            .lock()
            .map_err(|err| SequenceError::CachePoisoned(err.to_string()))?
            .cache_set(key, info.clone());
    }
    Ok(info)
}

#[cfg(test)]
mod tests {
    use capnp::message::TypedBuilder;

    use super::*;
    use crate::test_helpers::{get_connection, test_workspace};

    /// A node over a stored sequence long enough to span several read chunks, with no two
    /// neighbouring chunks alike so a misplaced chunk shows up.
    fn long_stored_node(conn: &GraphConnection, sequence_type: &str) -> (HashId, Sequence) {
        let bases: String = (0..2_500)
            .map(|index| ['A', 'C', 'G', 'T'][(index * 7 + index / 13) % 4])
            .collect();
        let sequence = Sequence::new()
            .sequence_type(sequence_type)
            .sequence(&bases)
            .save(conn)
            .expect("should save the long sequence");
        let node_id = Node::create(conn, &sequence.hash, &HashId::convert_str("long-node"))
            .expect("should create the long node");
        (node_id, sequence)
    }

    #[test]
    fn test_get_sequence_range_matches_slicing_the_whole_sequence() {
        let conn = &get_connection(None).unwrap();
        let (node_id, sequence) = long_stored_node(conn, "DNA");
        for (start, end) in [
            (0, 0),
            (0, 1_000),
            (999, 1_001),
            (1_234, 1_235),
            (500, 2_500),
            (2_000, 2_500),
            (0, 2_500),
        ] {
            assert_eq!(
                Node::get_sequence_range(conn, test_workspace(), node_id, start, end).unwrap(),
                Some(sequence.get_sequence(start, end).unwrap()),
                "range {start}..{end}"
            );
        }
    }

    #[test]
    fn test_get_sequence_range_wraps_circular_sequences() {
        let conn = &get_connection(None).unwrap();
        let (node_id, sequence) = long_stored_node(conn, "circular DNA");
        assert_eq!(
            Node::get_sequence_range(conn, test_workspace(), node_id, 2_400, 100).unwrap(),
            Some(sequence.get_sequence(2_400, 100).unwrap())
        );
    }

    #[test]
    fn test_get_sequence_range_rejects_bad_ranges_and_unknown_nodes() {
        let conn = &get_connection(None).unwrap();
        let (node_id, _) = long_stored_node(conn, "DNA");
        assert_eq!(
            Node::get_sequence_range(conn, test_workspace(), node_id, 2_400, 100),
            Err(SequenceError::BoundsError {
                start: 2_400,
                end: 100,
                length: 2_500,
            })
        );
        assert_eq!(
            Node::get_sequence_range(conn, test_workspace(), node_id, 0, 2_501),
            Err(SequenceError::BoundsError {
                start: 0,
                end: 2_501,
                length: 2_500,
            })
        );
        assert_eq!(
            Node::get_sequence_range(
                conn,
                test_workspace(),
                HashId::convert_str("missing-node"),
                0,
                1
            ),
            Ok(None)
        );
    }

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
