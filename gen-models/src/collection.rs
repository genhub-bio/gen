use gen_core::traits::Capnp;
use rusqlite::{Row, params_from_iter};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    block_group::BlockGroup, db::GraphConnection, gen_models_capnp::collection, traits::*,
};

#[derive(Debug, Error, PartialEq)]
pub enum CollectionError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] rusqlite::Error),
    #[error("Collection already exists")]
    Duplicate(Collection),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Collection {
    pub name: String,
}

impl<'a> Capnp<'a> for Collection {
    type Builder = collection::Builder<'a>;
    type Reader = collection::Reader<'a>;

    fn write_capnp(&self, builder: &mut Self::Builder) {
        builder.set_name(self.name.clone());
    }

    fn read_capnp(reader: Self::Reader) -> Self {
        let name = reader.get_name().unwrap().to_string().unwrap();

        Collection { name }
    }
}

impl Query for Collection {
    type Model = Collection;

    const PRIMARY_KEY: &'static str = "name";
    const TABLE_NAME: &'static str = "collections";

    fn process_row(row: &Row) -> Self::Model {
        Collection {
            name: row.get(0).unwrap(),
        }
    }
}

impl Collection {
    pub fn exists(conn: &GraphConnection, name: &str) -> bool {
        let mut stmt = conn
            .prepare("select name from collections where name = ?1")
            .unwrap();
        stmt.exists([name]).unwrap()
    }

    pub fn create(conn: &GraphConnection, name: &str) -> Result<Collection, CollectionError> {
        let mut stmt = match conn.prepare("INSERT INTO collections (name) VALUES (?1) RETURNING *;")
        {
            Ok(stmt) => stmt,
            Err(e) => return Err(CollectionError::DatabaseError(e)),
        };

        match stmt.query_row((name,), |_row| {
            Ok(Collection {
                name: name.to_string(),
            })
        }) {
            Ok(res) => Ok(res),
            Err(rusqlite::Error::SqliteFailure(e, _))
                if e.code == rusqlite::ErrorCode::ConstraintViolation =>
            {
                Err(CollectionError::Duplicate(Collection {
                    name: name.to_string(),
                }))
            }
            Err(e) => Err(CollectionError::DatabaseError(e)),
        }
    }

    pub fn get_or_create(
        conn: &GraphConnection,
        name: &str,
    ) -> Result<Collection, CollectionError> {
        match Collection::create(conn, name) {
            Ok(collection) => Ok(collection),
            Err(CollectionError::Duplicate(_)) => {
                let collections =
                    Self::query(conn, "SELECT * FROM collections WHERE name = ?1", [name]);
                if let Some(collection) = collections.into_iter().next() {
                    Ok(collection)
                } else {
                    Err(CollectionError::DatabaseError(
                        rusqlite::Error::QueryReturnedNoRows,
                    ))
                }
            }
            Err(e) => Err(e),
        }
    }

    pub fn bulk_create(conn: &GraphConnection, names: &Vec<String>) -> Vec<Collection> {
        let placeholders = names.iter().map(|_| "(?)").collect::<Vec<_>>().join(", ");
        let q = format!("INSERT INTO collections (name) VALUES {placeholders} RETURNING *",);
        let mut stmt = conn.prepare(&q).unwrap();
        let rows = stmt
            .query_map(params_from_iter(names), |row| {
                Ok(Collection { name: row.get(0)? })
            })
            .unwrap();
        rows.map(|row| row.unwrap()).collect()
    }

    pub fn get_block_groups(conn: &GraphConnection, collection_name: &str) -> Vec<BlockGroup> {
        // Load all block groups that have the given collection_name
        let mut stmt = conn
            .prepare("SELECT * FROM block_groups WHERE collection_name = ?1 order by created_on;")
            .unwrap();
        let block_group_iter = stmt
            .query_map([collection_name], |row| Ok(BlockGroup::process_row(row)))
            .unwrap();
        block_group_iter.map(|bg| bg.unwrap()).collect()
    }

    pub fn delete_by_name(conn: &GraphConnection, name: &str) {
        let mut stmt = conn
            .prepare("delete from collections where name = ?1")
            .unwrap();
        stmt.execute([name]).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use capnp::message::TypedBuilder;

    use super::*;
    use crate::test_helpers::get_connection;

    #[test]
    pub fn test_serialization() {
        let c = Collection {
            name: "test".to_string(),
        };
        let mut message = TypedBuilder::<collection::Owned>::new_default();
        let mut root = message.init_root();
        c.write_capnp(&mut root);

        let d = Collection::read_capnp(root.into_reader());
        assert_eq!(c, d);
    }

    #[test]
    pub fn test_delete_by_name() {
        let conn = &get_connection(None).unwrap();
        let collection1 = Collection::create(conn, "test1").unwrap();
        let collection2 = Collection::create(conn, "test2").unwrap();

        Collection::delete_by_name(conn, &collection1.name);

        // Verify only the correct one was deleted
        assert!(!Collection::exists(conn, &collection1.name));
        assert!(Collection::exists(conn, &collection2.name));
    }
}
