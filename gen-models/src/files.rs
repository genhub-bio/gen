use std::{collections::HashMap, rc::Rc};

use gen_core::{HashId, traits::Capnp};
use rusqlite::{Connection, Result as SQLResult, Row, params, types::Value};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{gen_models_capnp::gen_database, traits::*};

#[derive(Debug, Error)]
pub enum GenDatabaseError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] rusqlite::Error),
}

#[derive(Clone, Debug, Eq, Hash, Serialize, Deserialize, PartialEq)]
pub struct GenDatabase {
    pub db_uuid: String,
    pub name: String,
    pub path: String,
}

impl<'a> Capnp<'a> for GenDatabase {
    type Builder = gen_database::Builder<'a>;
    type Reader = gen_database::Reader<'a>;

    fn write_capnp(&self, builder: &mut Self::Builder) {
        builder.set_db_uuid(&self.db_uuid);
        builder.set_name(&self.name);
        builder.set_path(&self.path);
    }

    fn read_capnp(reader: Self::Reader) -> Self {
        let db_uuid = reader.get_db_uuid().unwrap().to_string().unwrap();
        let name = reader.get_name().unwrap().to_string().unwrap();
        let path = reader.get_path().unwrap().to_string().unwrap();

        GenDatabase {
            db_uuid,
            name,
            path,
        }
    }
}

impl Query for GenDatabase {
    type Model = GenDatabase;

    const PRIMARY_KEY: &'static str = "db_uuid";
    const TABLE_NAME: &'static str = "gen_databases";

    fn process_row(row: &Row) -> Self::Model {
        GenDatabase {
            db_uuid: row.get(0).unwrap(),
            name: row.get(1).unwrap(),
            path: row.get(2).unwrap(),
        }
    }
}

impl GenDatabase {
    pub fn create(
        conn: &Connection,
        db_uuid: &str,
        name: &str,
        path: &str,
    ) -> SQLResult<GenDatabase> {
        let query = "INSERT INTO gen_databases (db_uuid, name, path) VALUES (?1, ?2, ?3);";
        let mut stmt = conn.prepare(query)?;
        stmt.execute(params![db_uuid, name, path])?;
        Ok(GenDatabase {
            db_uuid: db_uuid.to_string(),
            name: name.to_string(),
            path: path.to_string(),
        })
    }

    pub fn delete_by_uuid(conn: &Connection, db_uuid: &str) -> SQLResult<GenDatabase> {
        GenDatabase::get(
            conn,
            "DELETE FROM gen_databases WHERE db_uuid = ?1",
            params![db_uuid],
        )
    }

    pub fn get_by_uuid(conn: &Connection, db_uuid: &str) -> SQLResult<GenDatabase> {
        GenDatabase::get(
            conn,
            "SELECT * FROM gen_databases WHERE db_uuid = ?1",
            params![db_uuid],
        )
    }

    pub fn get_by_path(conn: &Connection, path: &str) -> SQLResult<GenDatabase> {
        GenDatabase::get(
            conn,
            "SELECT * FROM gen_databases WHERE path = ?1",
            params![path],
        )
    }

    pub fn get_or_create(
        conn: &Connection,
        db_uuid: &str,
        name: &str,
        path: &str,
    ) -> SQLResult<GenDatabase> {
        match GenDatabase::create(conn, db_uuid, name, path) {
            Ok(new) => Ok(new),
            Err(rusqlite::Error::SqliteFailure(err, _details)) => {
                if err.code == rusqlite::ErrorCode::ConstraintViolation {
                    match GenDatabase::get(
                        conn,
                        "select * from gen_databases where db_uuid = ?1 AND name = ?2 AND path = ?3",
                        params![db_uuid, name, path],
                    ) {
                        Ok(result) => Ok(result),
                        Err(e) => Err(e),
                    }
                } else {
                    panic!("something bad happened querying the database")
                }
            }
            Err(_) => {
                panic!("something bad happened.")
            }
        }
    }

    pub fn query_by_operations(
        conn: &Connection,
        operations: &[HashId],
    ) -> Result<HashMap<HashId, Vec<GenDatabase>>, GenDatabaseError> {
        let query = "select gd.*, od.operation_hash from gen_databases gd left join operation_databases od on (gd.db_uuid = od.database_uuid) where od.operation_hash in rarray(?1)";
        let mut stmt = conn.prepare(query).unwrap();
        let rows = stmt
            .query_map(
                params![Rc::new(
                    operations
                        .iter()
                        .map(|h| Value::from(*h))
                        .collect::<Vec<Value>>()
                )],
                |row| Ok((GenDatabase::process_row(row), row.get::<_, HashId>(3)?)),
            )
            .unwrap();
        rows.into_iter()
            .try_fold(HashMap::new(), |mut acc: HashMap<_, Vec<_>>, row| {
                let (item, hash) = row?;
                acc.entry(hash).or_default().push(item);
                Ok(acc)
            })
    }
}

#[cfg(test)]
mod tests {
    use capnp::message::TypedBuilder;

    use super::*;
    use crate::test_helpers::get_operation_connection;

    #[test]
    fn test_gen_database_capnp_serialization() {
        let gen_database = GenDatabase {
            db_uuid: "test-uuid-123".to_string(),
            name: "test_database".to_string(),
            path: "/path/to/test.db".to_string(),
        };

        let mut message = TypedBuilder::<gen_database::Owned>::new_default();
        let mut root = message.init_root();
        gen_database.write_capnp(&mut root);

        let deserialized = GenDatabase::read_capnp(root.into_reader());
        assert_eq!(gen_database, deserialized);
    }

    #[test]
    fn test_create_gen_database() {
        let conn = get_operation_connection(None).unwrap();

        let db = GenDatabase::create(&conn, "test-uuid-123", "test_db", "path/to/db.db").unwrap();

        assert_eq!(db.db_uuid, "test-uuid-123");
        assert_eq!(db.name, "test_db");
        assert_eq!(db.path, "path/to/db.db");
    }

    #[test]
    fn test_get_by_uuid() {
        let conn = get_operation_connection(None).unwrap();

        let created_db =
            GenDatabase::create(&conn, "test-uuid-456", "test_db2", "path/to/db2.db").unwrap();

        let retrieved_db = GenDatabase::get_by_uuid(&conn, "test-uuid-456").unwrap();

        assert_eq!(retrieved_db.db_uuid, "test-uuid-456");
        assert_eq!(retrieved_db.name, "test_db2");
        assert_eq!(retrieved_db.path, "path/to/db2.db");
    }

    #[test]
    fn test_get_by_path() {
        let conn = get_operation_connection(None).unwrap();

        let created_db =
            GenDatabase::create(&conn, "test-uuid-789", "test_db3", "path/to/db3.db").unwrap();

        let retrieved_db = GenDatabase::get_by_path(&conn, "path/to/db3.db").unwrap();

        assert_eq!(retrieved_db.db_uuid, "test-uuid-789");
        assert_eq!(retrieved_db.name, "test_db3");
        assert_eq!(retrieved_db.path, "path/to/db3.db");
    }

    #[test]
    fn test_get_or_create_existing() {
        let conn = get_operation_connection(None).unwrap();

        let created_db = GenDatabase::create(
            &conn,
            "test-uuid-existing",
            "existing_db",
            "path/to/existing.db",
        )
        .unwrap();

        // Try to get_or_create with same UUID - should return existing
        let retrieved_db = GenDatabase::get_or_create(
            &conn,
            "test-uuid-existing",
            "existing_db",
            "path/to/existing.db",
        )
        .unwrap();

        assert_eq!(retrieved_db.db_uuid, "test-uuid-existing");
        assert_eq!(retrieved_db.name, "existing_db"); // Should keep original name
        assert_eq!(retrieved_db.path, "path/to/existing.db"); // Should keep original path
    }

    #[test]
    fn test_get_or_create_conflict() {
        let conn = get_operation_connection(None).unwrap();

        // Create a database entry
        let _ = GenDatabase::create(
            &conn,
            "test-uuid-existing",
            "existing_db",
            "path/to/existing.db",
        )
        .unwrap();

        // Try to get_or_create with same UUID - should return existing
        let retrieved_db = GenDatabase::get_or_create(
            &conn,
            "test-uuid-existing",
            "something_else",
            "path/to/something_else.db",
        );
        assert!(retrieved_db.is_err())
    }

    #[test]
    fn test_get_or_create_new() {
        let conn = get_operation_connection(None).unwrap();

        // Try to get_or_create with non-existing UUID - should create new
        let new_db =
            GenDatabase::get_or_create(&conn, "test-uuid-new", "new_db", "path/to/new.db").unwrap();

        assert_eq!(new_db.db_uuid, "test-uuid-new");
        assert_eq!(new_db.name, "new_db");
        assert_eq!(new_db.path, "path/to/new.db");
    }

    #[test]
    fn test_get_by_uuid_not_found() {
        let conn = get_operation_connection(None).unwrap();

        let result = GenDatabase::get_by_uuid(&conn, "non-existing-uuid");

        assert!(result.is_err());
    }

    #[test]
    fn test_get_by_path_not_found() {
        let conn = get_operation_connection(None).unwrap();

        let result = GenDatabase::get_by_path(&conn, "non/existing/path.db");

        assert!(result.is_err());
    }
}
