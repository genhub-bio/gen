use std::{fs, path::PathBuf};

use gen_core::{
    config::{get_or_create_gen_dir, set_base_dir},
    errors::ConnectionError,
};
use gen_models::migrations::run_operation_migrations;
use rusqlite::Connection;
use tempfile::tempdir;

pub fn get_operation_connection<'a>(
    db_path: impl Into<Option<&'a str>>,
) -> Result<Connection, ConnectionError> {
    let path: Option<&str> = db_path.into();
    let mut conn;
    if let Some(v) = path {
        if fs::metadata(v).is_ok() {
            fs::remove_file(v).expect("Unable to delete existing file");
        }
        conn = Connection::open(v).map_err(ConnectionError::OpenFailed)?;
    } else {
        conn = Connection::open_in_memory().map_err(ConnectionError::OpenFailed)?;
    }
    rusqlite::vtab::array::load_module(&conn)?;
    run_operation_migrations(&mut conn);
    Ok(conn)
}

pub fn setup_gen_dir() -> PathBuf {
    let tmp_dir = tempdir().unwrap().keep();
    set_base_dir(&tmp_dir);
    get_or_create_gen_dir()
}
