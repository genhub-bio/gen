use std::{collections::HashMap, rc::Rc};

use gen_core::{HashId, calculate_hash, traits::Capnp};
use indexmap::IndexSet;
use rusqlite::{self, Row, ToSql, types::Value};
use serde::{Deserialize, Serialize};

use crate::{
    Direction, ModelSelect,
    db::GraphConnection,
    edge::{Edge, EdgeData},
    gen_models_capnp::block_group_edge,
    traits::*,
};

#[derive(
    Clone, Debug, Deserialize, Serialize, Eq, Hash, PartialEq, Ord, PartialOrd, ModelSelect,
)]
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
        HashId(calculate_hash(&format!(
            "{}:{}:{}:{}",
            self.block_group_id, self.edge_id, self.chromosome_index, self.phased
        )))
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
        let mut select = BlockGroupEdge::select(conn)
            .block_group_id(*block_group_id)
            .order_by(BlockGroupEdgeSelect::CreatedOn, Direction::Desc);
        if let Some(history_ref) = history_ref {
            select = select.with_ref(history_ref);
        }
        let block_group_edges = select.load().expect("should load block group edges");
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
        history_ref: Option<&str>,
    ) -> Vec<AugmentedEdge> {
        let query = format!(
            "SELECT * FROM {} WHERE block_group_id = :block_group_id AND edge_id in rarray(:edge_ids);",
            Self::table_name_with_history_ref(history_ref),
        );
        let edge_id_values = Rc::new(
            edge_ids
                .iter()
                .map(|edge_id| Value::from(*edge_id))
                .collect::<Vec<_>>(),
        );
        let mut params: Vec<(&str, &dyn ToSql)> = vec![
            (":block_group_id", block_group_id),
            (":edge_ids", &edge_id_values),
        ];
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
}

#[cfg(test)]
mod tests {
    use capnp::message::TypedBuilder;
    use chrono::Utc;
    use gen_core::{PATH_END_NODE_ID, PATH_START_NODE_ID, Strand};

    use super::*;
    use crate::{
        collection::Collection,
        gen_models_capnp::block_group_edge,
        history::dolt::commit_all,
        test_helpers::{create_bg, get_connection},
    };

    #[test]
    fn test_block_group_edge_capnp_serialization() {
        let block_group_edge = BlockGroupEdge {
            id: HashId::pad_str(300),
            block_group_id: HashId::pad_str(301),
            edge_id: HashId::pad_str(302),
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

    #[test]
    fn test_specific_edges_for_block_group_respects_history_ref() {
        let conn = &get_connection(None).expect("should create graph connection");
        Collection::get_or_create(conn, "test").expect("should create collection");
        let block_group = create_bg(conn, "test", "test", "chr1");
        let edge = Edge::create(
            conn,
            PATH_START_NODE_ID,
            0,
            Strand::Forward,
            PATH_END_NODE_ID,
            0,
            Strand::Forward,
        )
        .expect("should create edge");
        let block_group_edge = BlockGroupEdgeData {
            block_group_id: block_group.id,
            edge_id: edge.id,
            chromosome_index: 0,
            phased: 0,
        };
        BlockGroupEdge::bulk_create(conn, core::slice::from_ref(&block_group_edge));
        let historical_edges = BlockGroupEdge::edges_for_block_group(conn, &block_group.id, None);
        let historical_edge = historical_edges
            .first()
            .expect("should create a block group edge")
            .clone();
        let historical_commit =
            commit_all(conn, "create block group edges").expect("should commit block group edges");
        BlockGroupEdge::bulk_delete(conn, &[block_group_edge]);
        Edge::delete_by_ids(conn, &[historical_edge.edge.id]);

        assert!(
            BlockGroupEdge::specific_edges_for_block_group(
                conn,
                &block_group.id,
                &[historical_edge.edge.id],
                None,
            )
            .is_empty()
        );
        assert_eq!(
            BlockGroupEdge::specific_edges_for_block_group(
                conn,
                &block_group.id,
                &[historical_edge.edge.id],
                Some(&historical_commit.to_string()),
            ),
            vec![historical_edge]
        );
    }
}
