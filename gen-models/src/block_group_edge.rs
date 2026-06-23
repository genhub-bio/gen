use std::{collections::HashMap, rc::Rc};

use gen_core::{HashId, calculate_hash, traits::Capnp};
use rusqlite::{self, Row, params, types::Value};
use serde::{Deserialize, Serialize};

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
        all(debug_assertions, feature = "profiling"),
        tracing::instrument(skip(conn, block_group_edges))
    )]
    pub fn bulk_create(conn: &GraphConnection, block_group_edges: &[BlockGroupEdgeData]) {
        let batch_size = max_rows_per_batch(conn, 6);

        for chunk in block_group_edges.chunks(batch_size) {
            let mut sql = String::from(
                "INSERT OR IGNORE INTO block_group_edges
                 (id, block_group_id, edge_id, chromosome_index, phased, created_on) VALUES ",
            );
            let mut rows_to_insert = vec![];
            let mut params: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
            let mut inserted_block_group_edges_by_id = HashMap::new();
            let timestamp = chrono::Utc::now().timestamp_nanos_opt().unwrap();
            for block_group_edge in chunk {
                let hash = block_group_edge.id_hash();
                rows_to_insert.push("(?, ?, ?, ?, ?, ?)".to_string());
                params.push(Box::new(hash));
                params.push(Box::new(block_group_edge.block_group_id));
                params.push(Box::new(block_group_edge.edge_id));
                params.push(Box::new(block_group_edge.chromosome_index));
                params.push(Box::new(block_group_edge.phased));
                params.push(Box::new(timestamp));
                inserted_block_group_edges_by_id.insert(
                    hash,
                    BlockGroupEdge {
                        id: hash,
                        block_group_id: block_group_edge.block_group_id,
                        edge_id: block_group_edge.edge_id,
                        chromosome_index: block_group_edge.chromosome_index,
                        phased: block_group_edge.phased,
                        created_on: timestamp,
                    },
                );
            }
            if rows_to_insert.is_empty() {
                continue;
            }

            sql.push_str(&rows_to_insert.join(", "));
            sql.push_str(" RETURNING id");

            let mut stmt = conn.prepare(&sql).unwrap();
            let inserted_block_group_edge_ids = stmt
                .query_map(rusqlite::params_from_iter(params), |row| {
                    row.get::<_, HashId>(0)
                })
                .unwrap();
            for block_group_edge_id in inserted_block_group_edge_ids {
                let block_group_edge_id = block_group_edge_id.unwrap();
                let block_group_edge = inserted_block_group_edges_by_id
                    .remove(&block_group_edge_id)
                    .expect("should have a block group edge for a returned inserted ID");
                crate::operation_recorder::record_block_group_edge(block_group_edge);
            }
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
    ) -> Vec<AugmentedEdge> {
        let block_group_edges = BlockGroupEdge::query(
            conn,
            "select * from block_group_edges where block_group_id = ?1 ORDER BY created_on DESC;",
            params![block_group_id],
        );
        let edge_ids = block_group_edges
            .iter()
            .map(|block_group_edge| block_group_edge.edge_id)
            .collect::<Vec<_>>();
        let edges = Edge::query_by_ids(conn, &edge_ids);
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
        let edges = Edge::query_by_ids(conn, &edge_ids);

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
}
