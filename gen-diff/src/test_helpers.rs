use std::{
    fs,
    ops::{Deref, DerefMut},
    rc::Rc,
};

use gen_core::errors::ConnectionError;
use gen_models::{
    db::{ConfigConnection, DbContext, GraphConnection},
    migrations::{run_config_migrations, run_migrations},
};
use rusqlite::Connection;
use tempfile::TempDir;

#[derive(Clone)]
pub(crate) struct TestContext {
    context: DbContext,
    _temp_dir: Rc<TempDir>,
}

impl TestContext {
    pub(crate) fn new(context: DbContext, temp_dir: TempDir) -> Self {
        Self {
            context,
            _temp_dir: Rc::new(temp_dir),
        }
    }
}

impl Deref for TestContext {
    type Target = DbContext;

    fn deref(&self) -> &Self::Target {
        &self.context
    }
}

impl DerefMut for TestContext {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.context
    }
}

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

pub fn get_config_connection<'a>(
    db_path: impl Into<Option<&'a str>>,
) -> Result<ConfigConnection, ConnectionError> {
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
    run_config_migrations(&mut conn);
    Ok(ConfigConnection(conn))
}
