use std::{collections::HashMap, rc::Rc};

use gen_core::{HashId, traits::Capnp};
use indexmap::IndexSet;
use rusqlite::{self, Row, ToSql, params, types::Value};
use serde::{Deserialize, Serialize};
use xxhash_rust::xxh3::xxh3_128;

use crate::{
    db::GraphConnection,
    edge::{Edge, EdgeData},
    gen_models_capnp::block_group_edge,
    traits::*,
};

#[derive(Clone, Debug, Deserialize, Serialize, Eq, Hash, PartialEq, Ord, PartialOrd)]
pub struct BlockGroupEdge {
    pub id: HashId,
    pub block_group_id: HashId,
    pub edge_id: HashId,
    pub chromosome_index: i64,
    pub phased: i64,
    pub created_on: i64,
}

impl<'a> Capnp<'a> for BlockGroupEdge {
    type Builder = block_group_edge::Builder<'a>;
    type Reader = block_group_edge::Reader<'a>;

    fn write_capnp(&self, builder: &mut Self::Builder) {
        builder.set_id(&self.id.0).unwrap();
        builder.set_block_group_id(&self.block_group_id.0).unwrap();
        builder.set_edge_id(&self.edge_id.0).unwrap();
        builder.set_chromosome_index(self.chromosome_index);
        builder.set_phased(self.phased);
        builder.set_created_on(self.created_on);
    }

    fn read_capnp(reader: Self::Reader) -> Self {
        let id = reader
            .get_id()
            .unwrap()
            .as_slice()
            .unwrap()
            .try_into()
            .unwrap();
        let block_group_id = reader
            .get_block_group_id()
            .unwrap()
            .as_slice()
            .unwrap()
            .try_into()
            .unwrap();
        let edge_id = reader
            .get_edge_id()
            .unwrap()
            .as_slice()
            .unwrap()
            .try_into()
            .unwrap();
        let chromosome_index = reader.get_chromosome_index();
        let phased = reader.get_phased();
        let created_on = reader.get_created_on();

        BlockGroupEdge {
            id,
            block_group_id,
            edge_id,
            chromosome_index,
            phased,
            created_on,
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, Ord, PartialOrd)]
pub struct BlockGroupEdgeData {
    pub block_group_id: HashId,
    pub edge_id: HashId,
    pub chromosome_index: i64,
    pub phased: i64,
}

impl BlockGroupEdgeData {
    pub fn id_hash(&self) -> HashId {
        let half = xxh3_128(
            format!(
                "{}:{}:{}:{}",
                self.block_group_id, self.edge_id, self.chromosome_index, self.phased
            )
            .as_bytes(),
        )
        .to_le_bytes();
        let mut hash = [0_u8; 32];
        hash[..16].copy_from_slice(&half);
        hash[16..].copy_from_slice(&half);
        HashId(hash)
    }
}

impl From<&BlockGroupEdge> for BlockGroupEdgeData {
    fn from(item: &BlockGroupEdge) -> Self {
        BlockGroupEdgeData {
            block_group_id: item.block_group_id,
            edge_id: item.edge_id,
            chromosome_index: item.chromosome_index,
            phased: item.phased,
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, Ord, PartialOrd)]
pub struct AugmentedEdge {
    pub edge: Edge,
    pub chromosome_index: i64,
    pub phased: i64,
    pub created_on: i64,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, Ord, PartialOrd)]
pub struct AugmentedEdgeData {
    pub edge_data: EdgeData,
    pub chromosome_index: i64,
    pub phased: i64,
}

impl Query for BlockGroupEdge {
    type Model = BlockGroupEdge;

    const TABLE_NAME: &'static str = "block_group_edges";

    fn process_row(row: &Row) -> Self::Model {
        BlockGroupEdge {
            id: row.get(0).unwrap(),
            block_group_id: row.get(1).unwrap(),
            edge_id: row.get(2).unwrap(),
            chromosome_index: row.get(3).unwrap(),
            phased: row.get(4).unwrap(),
            created_on: row.get(5).unwrap(),
        }
    }
}

impl BlockGroupEdge {
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip(conn, block_group_edges))
    )]
    pub fn bulk_create(conn: &GraphConnection, block_group_edges: &[BlockGroupEdgeData]) {
        let unique_block_group_edges = block_group_edges
            .iter()
            .collect::<IndexSet<_>>()
            .into_iter()
            .cloned()
            .collect::<Vec<_>>();
        if unique_block_group_edges.is_empty() {
            return;
        }
        let batch_size = max_rows_per_batch(conn, 6);

        for chunk in unique_block_group_edges.chunks(batch_size) {
            let timestamp = chrono::Utc::now().timestamp_nanos_opt().unwrap();
            let mut sql = String::from(
                "INSERT OR IGNORE INTO block_group_edges
                 (id, block_group_id, edge_id, chromosome_index, phased, created_on) VALUES ",
            );
            for row_index in 0..chunk.len() {
                if row_index > 0 {
                    sql.push(',');
                }
                sql.push_str("(?, ?, ?, ?, ?, ?)");
            }
            sql.push(';');
            let mut params = Vec::with_capacity(chunk.len() * 6);
            for block_group_edge in chunk {
                params.push(Value::from(block_group_edge.id_hash()));
                params.push(Value::from(block_group_edge.block_group_id));
                params.push(Value::from(block_group_edge.edge_id));
                params.push(Value::from(block_group_edge.chromosome_index));
                params.push(Value::from(block_group_edge.phased));
                params.push(Value::from(timestamp));
            }
            let mut stmt = conn.prepare_cached(&sql).unwrap();
            stmt.execute(rusqlite::params_from_iter(params)).unwrap();
        }
    }

    pub fn bulk_delete(conn: &GraphConnection, block_group_edges: &[BlockGroupEdgeData]) {
        let hashes = block_group_edges
            .iter()
            .map(|bge| bge.id_hash())
            .collect::<Vec<_>>();
        BlockGroupEdge::delete_by_ids(conn, &hashes);
    }

    #[cfg_attr(
        all(debug_assertions, feature = "profiling"),
        tracing::instrument(skip(conn, block_group_id))
    )]
    pub fn edges_for_block_group(
        conn: &GraphConnection,
        block_group_id: &HashId,
        history_ref: Option<&str>,
    ) -> Vec<AugmentedEdge> {
        let query = format!(
            "select * from {} where block_group_id = :block_group_id ORDER BY created_on DESC;",
            Self::table_name_with_history_ref(history_ref),
        );
        let mut params: Vec<(&str, &dyn ToSql)> = vec![(":block_group_id", block_group_id)];
        if let Some(history_ref) = history_ref.as_ref() {
            params.push((":history_ref", history_ref));
        }
        let block_group_edges = BlockGroupEdge::query(conn, &query, &params[..]);
        let edge_ids = block_group_edges
            .iter()
            .map(|block_group_edge| block_group_edge.edge_id)
            .collect::<Vec<_>>();
        let edges = Edge::query_by_ids(conn, &edge_ids, history_ref);
        let edge_map = edges
            .iter()
            .map(|edge| (&edge.id, edge))
            .collect::<HashMap<_, &Edge>>();
        block_group_edges
            .into_iter()
            .map(|bge| {
                let edge_info = *edge_map.get(&bge.edge_id).unwrap();
                AugmentedEdge {
                    edge: edge_info.clone(),
                    chromosome_index: bge.chromosome_index,
                    phased: bge.phased,
                    created_on: bge.created_on,
                }
            })
            .collect()
    }

    pub fn specific_edges_for_block_group(
        conn: &GraphConnection,
        block_group_id: &HashId,
        edge_ids: &[HashId],
    ) -> Vec<AugmentedEdge> {
        let block_group_edges = BlockGroupEdge::query(
            conn,
            "SELECT * FROM block_group_edges WHERE block_group_id = ?1 AND edge_id in rarray(?2);",
            params![
                block_group_id,
                Rc::new(
                    edge_ids
                        .iter()
                        .map(|x| Value::from(*x))
                        .collect::<Vec<Value>>()
                )
            ],
        );
        let edge_ids = block_group_edges
            .iter()
            .map(|block_group_edge| block_group_edge.edge_id)
            .collect::<Vec<_>>();
        let edges = Edge::query_by_ids(conn, &edge_ids, None);

        let edge_map = edges
            .iter()
            .map(|edge| (&edge.id, edge))
            .collect::<HashMap<_, &Edge>>();
        block_group_edges
            .into_iter()
            .map(|bge| {
                let edge_info = *edge_map.get(&bge.edge_id).unwrap();
                AugmentedEdge {
                    edge: edge_info.clone(),
                    chromosome_index: bge.chromosome_index,
                    phased: bge.phased,
                    created_on: bge.created_on,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use capnp::message::TypedBuilder;
    use chrono::Utc;

    use super::*;
    use crate::gen_models_capnp::block_group_edge;

    #[test]
    fn test_block_group_edge_capnp_serialization() {
        let block_group_edge = BlockGroupEdge {
            id: "0000000000000000000000000000030000000000000000000000000000000000"
                .try_into()
                .unwrap(),
            block_group_id: "0000000000000000000000000000030000000020000000000000000000000000"
                .try_into()
                .unwrap(),
            edge_id: "0000000000000000000000000000030000000000000000000000000000000000"
                .try_into()
                .unwrap(),
            chromosome_index: 1,
            phased: 0,
            created_on: Utc::now().timestamp_nanos_opt().unwrap(),
        };

        let mut message = TypedBuilder::<block_group_edge::Owned>::new_default();
        let mut root = message.init_root();
        block_group_edge.write_capnp(&mut root);

        let deserialized = BlockGroupEdge::read_capnp(root.into_reader());
        assert_eq!(block_group_edge, deserialized);
    }

    #[test]
    fn test_block_group_edge_data_id_hash_is_deterministic() {
        let block_group_edge = BlockGroupEdgeData {
            block_group_id: HashId::convert_str("block-group"),
            edge_id: HashId::convert_str("edge"),
            chromosome_index: -7,
            phased: 11,
        };
        let changed_block_group_edge = BlockGroupEdgeData {
            phased: 12,
            ..block_group_edge.clone()
        };

        assert_eq!(block_group_edge.id_hash(), block_group_edge.id_hash());
        assert_ne!(
            block_group_edge.id_hash(),
            changed_block_group_edge.id_hash()
        );
    }
}
