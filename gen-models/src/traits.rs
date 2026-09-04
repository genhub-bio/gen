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
    const HISTORY_TABLE_NAME: Option<&'static str> = Some(Self::TABLE_NAME);

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

    fn table_name_with_history_ref(history_ref: Option<&str>) -> String {
        if history_ref.is_some() {
            let history_table_name =
                Self::HISTORY_TABLE_NAME.expect("should support history ref queries");
            format!("dolt_at_{history_table_name}(:history_ref)")
        } else {
            Self::TABLE_NAME.to_string()
        }
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
}
