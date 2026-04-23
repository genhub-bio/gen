use std::{collections::HashMap, rc::Rc};

use gen_core::{HashId, calculate_hash, traits::Capnp};
use itertools::Itertools;
use rusqlite::{self, Row, params, types::Value};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    db::GraphConnection, edge::Edge, gen_models_capnp::path_edge as PathEdgeCapnp, traits::*,
};

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct PathEdge {
    pub id: HashId,
    pub path_id: HashId,
    pub edge_id: HashId,
    pub index_in_path: i64,
}

#[derive(Debug, Error)]
pub enum PathEdgeError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] rusqlite::Error),
    #[error("Duplicate entry with uuid: {0}")]
    Duplicate(String),
}

impl<'a> Capnp<'a> for PathEdge {
    type Builder = PathEdgeCapnp::Builder<'a>;
    type Reader = PathEdgeCapnp::Reader<'a>;

    fn write_capnp(&self, builder: &mut Self::Builder) {
        builder.set_id(&self.id.0).unwrap();
        builder.set_path_id(&self.path_id.0).unwrap();
        builder.set_index_in_path(self.index_in_path);
        builder.set_edge_id(&self.edge_id.0).unwrap();
    }

    fn read_capnp(reader: Self::Reader) -> Self {
        let id = reader
            .get_id()
            .unwrap()
            .as_slice()
            .unwrap()
            .try_into()
            .unwrap();
        let path_id = reader
            .get_path_id()
            .unwrap()
            .as_slice()
            .unwrap()
            .try_into()
            .unwrap();
        let index_in_path = reader.get_index_in_path();
        let edge_id = reader
            .get_edge_id()
            .unwrap()
            .as_slice()
            .unwrap()
            .try_into()
            .unwrap();

        PathEdge {
            id,
            path_id,
            index_in_path,
            edge_id,
        }
    }
}

impl Query for PathEdge {
    type Model = PathEdge;

    const TABLE_NAME: &'static str = "path_edges";

    fn process_row(row: &Row) -> Self::Model {
        PathEdge {
            id: row.get(0).unwrap(),
            path_id: row.get(1).unwrap(),
            edge_id: row.get(2).unwrap(),
            index_in_path: row.get(3).unwrap(),
        }
    }
}

impl PathEdge {
    pub fn create(
        conn: &GraphConnection,
        path_id: &HashId,
        index_in_path: i64,
        edge_id: HashId,
    ) -> Result<PathEdge, PathEdgeError> {
        let query =
            "INSERT INTO path_edges (id, path_id, edge_id, index_in_path) VALUES (?1, ?2, ?3, ?4);";
        let mut stmt = match conn.prepare(query) {
            Ok(stmt) => stmt,
            Err(e) => return Err(PathEdgeError::DatabaseError(e)),
        };
        let hash = HashId(calculate_hash(&format!(
            "{path_id}:{edge_id}:{index_in_path}"
        )));
        match stmt.execute(params![hash, path_id, edge_id, index_in_path]) {
            Ok(_) => Ok(PathEdge {
                id: hash,
                path_id: *path_id,
                index_in_path,
                edge_id,
            }),
            Err(rusqlite::Error::SqliteFailure(err, _details))
                if err.code == rusqlite::ErrorCode::ConstraintViolation =>
            {
                Ok(PathEdge {
                    id: hash,
                    path_id: *path_id,
                    index_in_path,
                    edge_id,
                })
            }
            Err(e) => Err(PathEdgeError::DatabaseError(e)),
        }
    }

    pub fn bulk_create(
        conn: &GraphConnection,
        path_id: &HashId,
        edge_ids: &[HashId],
    ) -> Result<(), PathEdgeError> {
        let batch_size = max_rows_per_batch(conn, 4);

        for (index1, chunk) in edge_ids.chunks(batch_size).enumerate() {
            let mut rows_to_insert = vec![];
            let mut params: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
            for (index2, edge_id) in chunk.iter().enumerate() {
                rows_to_insert.push("(?, ?, ?, ?)".to_string());
                let index_in = index1 * 100000 + index2;
                let hash = HashId(calculate_hash(&format!("{path_id}:{edge_id}:{index_in}")));
                params.push(Box::new(hash));
                params.push(Box::new(path_id));
                params.push(Box::new(edge_id));
                params.push(Box::new(index_in));
            }

            let sql = format!(
                "INSERT OR IGNORE INTO path_edges (id, path_id, edge_id, index_in_path) VALUES {};",
                rows_to_insert.join(", ")
            );

            let mut stmt = match conn.prepare(&sql) {
                Ok(stmt) => stmt,
                Err(e) => return Err(PathEdgeError::DatabaseError(e)),
            };
            match stmt.execute(rusqlite::params_from_iter(params)) {
                Ok(_) => (),
                Err(e) => return Err(PathEdgeError::DatabaseError(e)),
            }
        }

        Ok(())
    }

    pub fn delete(conn: &GraphConnection, path_id: &HashId) {
        let statement = "DELETE from path_edges WHERE path_id = ?1;";
        conn.execute(statement, params![path_id]).unwrap();
    }

    pub fn edges_for_path(conn: &GraphConnection, path_id: &HashId) -> Vec<Edge> {
        let path_edges = PathEdge::query(
            conn,
            "select * from path_edges where path_id = ?1 order by index_in_path ASC",
            params![path_id],
        );
        let edge_ids = path_edges
            .into_iter()
            .map(|path_edge| path_edge.edge_id)
            .collect::<Vec<HashId>>();
        let edges = Edge::query_by_ids(conn, &edge_ids);
        let edges_by_id = edges
            .into_iter()
            .map(|edge| (edge.id, edge))
            .collect::<HashMap<HashId, Edge>>();
        edge_ids
            .into_iter()
            .map(|edge_id| edges_by_id[&edge_id].clone())
            .collect::<Vec<Edge>>()
    }

    pub fn edges_for_paths(
        conn: &GraphConnection,
        path_ids: Vec<HashId>,
    ) -> HashMap<HashId, Vec<Edge>> {
        let query_path_ids = path_ids
            .iter()
            .map(|path_id| Value::from(*path_id))
            .collect::<Vec<Value>>();
        let path_edges = PathEdge::query(
            conn,
            "select * from path_edges where path_id in rarray(?1) ORDER BY path_id, index_in_path",
            params![Rc::new(query_path_ids)],
        );
        let edge_ids = path_edges
            .iter()
            .map(|path_edge| path_edge.edge_id)
            .collect::<Vec<_>>();
        let edges = Edge::query_by_ids(conn, &edge_ids);
        let edges_by_id = edges
            .into_iter()
            .map(|edge| (edge.id, edge))
            .collect::<HashMap<_, Edge>>();
        let path_edges_by_path_id = path_edges
            .iter()
            .map(|path_edge| (path_edge.path_id, path_edge.edge_id))
            .into_group_map();
        path_edges_by_path_id
            .into_iter()
            .map(|(path_id, edge_ids)| {
                (
                    path_id,
                    edge_ids
                        .into_iter()
                        .map(|edge_id| edges_by_id[&edge_id].clone())
                        .collect::<Vec<Edge>>(),
                )
            })
            .collect::<HashMap<HashId, Vec<Edge>>>()
    }
}

#[cfg(test)]
mod tests {
    use capnp::message::TypedBuilder;

    use super::*;
    use crate::gen_models_capnp::path_edge;

    #[test]
    fn test_path_edge_capnp_serialization() {
        let path_edge = PathEdge {
            id: HashId::pad_str(100),
            path_id: HashId::pad_str(200),
            index_in_path: 5,
            edge_id: HashId::pad_str(300),
        };

        let mut message = TypedBuilder::<path_edge::Owned>::new_default();
        let mut root = message.init_root();
        path_edge.write_capnp(&mut root);

        let deserialized = PathEdge::read_capnp(root.into_reader());
        assert_eq!(path_edge, deserialized);
    }
}
