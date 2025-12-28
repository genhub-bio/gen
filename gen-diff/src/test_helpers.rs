use std::fs;

use gen_core::{Workspace, errors::ConnectionError};
use gen_models::{
    db::{DbContext, GraphConnection, OperationsConnection},
    migrations::{run_migrations, run_operation_migrations},
};
use rusqlite::Connection;
use tempfile::tempdir;

pub fn get_connection<'a>(
    db_path: impl Into<Option<&'a str>>,
) -> Result<GraphConnection, ConnectionError> {
    let path: Option<&str> = db_path.into();
    let mut conn;
    if let Some(v) = path {
        if fs::metadata(v).is_ok() {
            fs::remove_file(v).expect("Unable to remove database entry.");
        }
        conn = Connection::open(v).map_err(ConnectionError::OpenFailed)?;
    } else {
        conn = Connection::open_in_memory().map_err(ConnectionError::OpenFailed)?;
    }
    rusqlite::vtab::array::load_module(&conn)?;
    run_migrations(&mut conn);
    Ok(GraphConnection(conn))
}

pub fn get_operation_connection<'a>(
    db_path: impl Into<Option<&'a str>>,
) -> Result<OperationsConnection, ConnectionError> {
    let path: Option<&str> = db_path.into();
    let mut conn;
    if let Some(v) = path {
        if fs::metadata(v).is_ok() {
            fs::remove_file(v).expect("Unable to remove database entry.");
        }
        conn = Connection::open(v).map_err(ConnectionError::OpenFailed)?;
    } else {
        conn = Connection::open_in_memory().map_err(ConnectionError::OpenFailed)?;
    }
    rusqlite::vtab::array::load_module(&conn)?;
    run_operation_migrations(&mut conn);
    Ok(OperationsConnection(conn))
}

pub fn setup_gen() -> DbContext {
    let tmp_dir = tempdir().unwrap().keep();
    let workspace = Workspace::new(tmp_dir);
    workspace.ensure_gen_dir();
    let graph_conn = get_connection(None).unwrap();
    let operation_conn = get_operation_connection(None).unwrap();
    DbContext::new(workspace, graph_conn, operation_conn)
}
