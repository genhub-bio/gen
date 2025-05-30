use rusqlite::{params_from_iter, Connection, Row};

use crate::models::block_group::BlockGroup;
use crate::models::traits::*;

#[derive(Clone, Debug)]
pub struct Collection {
    pub name: String,
}

impl Query for Collection {
    type Model = Collection;
    fn process_row(row: &Row) -> Self::Model {
        Collection {
            name: row.get(0).unwrap(),
        }
    }
}

impl Collection {
    pub fn exists(conn: &Connection, name: &str) -> bool {
        let mut stmt = conn
            .prepare("select name from collections where name = ?1")
            .unwrap();
        stmt.exists([name]).unwrap()
    }

    pub fn create(conn: &Connection, name: &str) -> Collection {
        let mut stmt = conn
            .prepare("INSERT INTO collections (name) VALUES (?1) RETURNING *;")
            .unwrap();

        match stmt.query_row((name,), |_row| {
            Ok(Collection {
                name: name.to_string(),
            })
        }) {
            Ok(res) => res,
            Err(rusqlite::Error::SqliteFailure(err, _details)) => {
                if err.code == rusqlite::ErrorCode::ConstraintViolation {
                    Collection {
                        name: name.to_string(),
                    }
                } else {
                    panic!("something bad happened querying the database")
                }
            }
            Err(err) => {
                println!("{err:?}");
                panic!("something bad happened querying the database")
            }
        }
    }

    pub fn bulk_create(conn: &Connection, names: &Vec<String>) -> Vec<Collection> {
        let placeholders = names.iter().map(|_| "(?)").collect::<Vec<_>>().join(", ");
        let q = format!(
            "INSERT INTO collections (name) VALUES {} RETURNING *",
            placeholders
        );
        let mut stmt = conn.prepare(&q).unwrap();
        let rows = stmt
            .query_map(params_from_iter(names), |row| {
                Ok(Collection { name: row.get(0)? })
            })
            .unwrap();
        rows.map(|row| row.unwrap()).collect()
    }

    pub fn get_block_groups(conn: &Connection, collection_name: &str) -> Vec<BlockGroup> {
        // Load all block groups that have the given collection_name
        let mut stmt = conn
            .prepare("SELECT * FROM block_groups WHERE collection_name = ?1")
            .unwrap();
        let block_group_iter = stmt
            .query_map([collection_name], |row| {
                Ok(BlockGroup {
                    id: row.get(0)?,
                    collection_name: row.get(1)?,
                    sample_name: row.get(2)?,
                    name: row.get(3)?,
                })
            })
            .unwrap();
        block_group_iter.map(|bg| bg.unwrap()).collect()
    }
}
