use rusqlite::{Connection, Params, Result as SQLResult, Row, limits::Limit};

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
    const TABLE_NAME: &'static str;
    const HISTORY_TABLE_NAME: Option<&'static str> = Some(Self::TABLE_NAME);

    fn query(conn: &Connection, query: &str, params: impl Params) -> Vec<Self::Model> {
        let mut stmt = conn.prepare(query).unwrap();
        let rows = stmt.query_map(params, Self::process_row).unwrap();
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
        let rows = stmt.query_map(params, Self::process_row)?;
        let mut objs = vec![];
        for row in rows {
            objs.push(row?);
        }
        Ok(objs)
    }

    fn get(conn: &Connection, query: &str, params: impl Params) -> SQLResult<Self::Model> {
        let mut stmt = conn.prepare(query)?;
        stmt.query_row(params, Self::process_row)
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

    fn process_row(row: &Row) -> SQLResult<Self::Model>;
}
