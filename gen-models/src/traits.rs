use std::rc::Rc;

use itertools::Itertools;
use rusqlite::{Connection, Params, Result as SQLResult, Row, limits::Limit, params, types::Value};

use crate::errors::QueryError;

/// Returns the SQLite variable parameter limit for the provided connection.
pub fn sqlite_parameter_limit(conn: &Connection) -> usize {
    let limit = conn
        .limit(Limit::SQLITE_LIMIT_VARIABLE_NUMBER)
        .expect("SQLite parameter limit should be readable");
    usize::try_from(limit).expect("SQLite parameter limit should be positive")
}

/// Computes how many rows can be inserted per batch given a parameter count.
pub fn max_rows_per_batch(conn: &Connection, params_per_row: usize) -> usize {
    let params_per_row = params_per_row.max(1);
    let max_params = sqlite_parameter_limit(conn);
    (max_params / params_per_row).max(1)
}

pub trait Query {
    type Model;
    const PRIMARY_KEY: &'static str = "id";
    const TABLE_NAME: &'static str;

    fn query(conn: &Connection, query: &str, params: impl Params) -> Vec<Self::Model> {
        let mut stmt = conn.prepare(query).unwrap();
        let rows = stmt
            .query_map(params, |row| Ok(Self::process_row(row)))
            .unwrap();
        let mut objs = vec![];
        for row in rows {
            objs.push(row.unwrap());
        }
        objs
    }

    fn try_query(
        conn: &Connection,
        query: &str,
        params: impl Params,
    ) -> Result<Vec<Self::Model>, QueryError> {
        let mut stmt = conn.prepare(query)?;
        let rows = stmt.query_map(params, |row| Ok(Self::process_row(row)))?;
        let mut objs = vec![];
        for row in rows {
            objs.push(row?);
        }
        Ok(objs)
    }

    fn get(conn: &Connection, query: &str, params: impl Params) -> SQLResult<Self::Model> {
        let mut stmt = conn.prepare(query).unwrap();
        stmt.query_row(params, |row| Ok(Self::process_row(row)))
    }

    fn get_by_id<'a, T>(conn: &Connection, id: &'a T) -> Option<Self::Model>
    where
        T: Clone + 'a,
        Value: From<T>,
    {
        Self::query(
            conn,
            &format!(
                "select * from {} where {} = ?1",
                Self::TABLE_NAME,
                Self::PRIMARY_KEY
            ),
            params![Value::from(id.clone())],
        )
        .pop()
    }

    fn query_by_ids<'a, I: ?Sized, T>(conn: &Connection, ids: &'a I) -> Vec<Self::Model>
    where
        &'a I: IntoIterator<Item = &'a T>,
        T: Clone + 'a,
        Value: From<T>,
    {
        let mut results = vec![];
        let batch_size = max_rows_per_batch(conn, 1);
        for chunk in &ids.into_iter().chunks(batch_size) {
            let values: Vec<Value> = chunk
                .map(|value: &'a T| Value::from(value.clone()))
                .collect();
            results.append(&mut Self::query(
                conn,
                // The use of rarray/rowid is to preserve the order of the input IDs. If it becomes a performance hit,
                // we can consider seeing if the input is an ordered preserving structure or not.
                &format!(
                    "
                    WITH arr AS (
                    SELECT value, rowid AS pos
                    FROM rarray(?1)
                    )
                    SELECT {}.*
                    FROM {}
                    JOIN arr ON {}.{} = arr.value
                    ORDER BY arr.pos;
                    ",
                    Self::TABLE_NAME,
                    Self::TABLE_NAME,
                    Self::TABLE_NAME,
                    Self::PRIMARY_KEY
                ),
                params!(Rc::new(values)),
            ))
        }
        results
    }

    fn delete_by_ids<'a, I: ?Sized, T>(conn: &Connection, ids: &'a I) -> Vec<Self::Model>
    where
        &'a I: IntoIterator<Item = &'a T>,
        T: Clone + 'a,
        Value: From<T>,
    {
        let mut results = vec![];
        let batch_size = max_rows_per_batch(conn, 1);
        for chunk in &ids.into_iter().chunks(batch_size) {
            let values: Vec<Value> = chunk
                .map(|value: &'a T| Value::from(value.clone()))
                .collect();
            results.append(&mut Self::query(
                conn,
                &format!(
                    "delete from {} where {} in rarray(?1)",
                    Self::TABLE_NAME,
                    Self::PRIMARY_KEY
                ),
                params!(Rc::new(values)),
            ))
        }
        results
    }

    fn process_row(row: &Row) -> Self::Model;

    fn table_name() -> &'static str {
        Self::TABLE_NAME
    }

    fn all(conn: &Connection) -> Vec<Self::Model> {
        let query = format!("SELECT * FROM {}", Self::TABLE_NAME);
        Self::query(conn, &query, [])
    }

    fn all_with_limit(conn: &Connection, limit: usize) -> Vec<Self::Model> {
        let query = format!("SELECT * FROM {} LIMIT {}", Self::TABLE_NAME, limit);
        Self::query(conn, &query, [])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{collection::Collection, test_helpers::get_connection};

    #[test]
    fn test_all_method() {
        let conn = get_connection(None).expect("Failed to get connection");

        // Test with empty table
        let empty_results = Collection::all(&conn);
        assert!(empty_results.is_empty());

        // Create test collections
        let collection_names = vec![
            "test_collection_1",
            "test_collection_2",
            "test_collection_3",
        ];
        for name in &collection_names {
            Collection::create(&conn, name).unwrap();
        }

        let all_results = Collection::all(&conn);
        assert_eq!(all_results.len(), collection_names.len());

        let returned_names: Vec<String> = all_results.iter().map(|c| c.name.clone()).collect();
        for name in &collection_names {
            assert!(returned_names.contains(&name.to_string()));
        }
    }

    #[test]
    fn test_all_with_limit_method() {
        let conn = get_connection(None).expect("Failed to get connection");

        for i in 0..10 {
            Collection::create(&conn, &format!("test_collection_{}", i)).unwrap();
        }

        let limited_results = Collection::all_with_limit(&conn, 5);
        assert_eq!(limited_results.len(), 5);

        // Test limit larger than available records
        let all_results = Collection::all(&conn);
        let large_limit_results = Collection::all_with_limit(&conn, all_results.len() + 20);
        assert_eq!(large_limit_results.len(), all_results.len());

        // Test limit of 0
        let zero_limit_results = Collection::all_with_limit(&conn, 0);
        assert!(zero_limit_results.is_empty());

        // Test limit of 1
        let one_limit_results = Collection::all_with_limit(&conn, 1);
        assert_eq!(one_limit_results.len(), 1);
    }
}
