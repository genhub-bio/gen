use std::path::PathBuf;

use gen_core::config::Workspace;
use gen_models::db::DbContext;
use pyo3::{exceptions::PyRuntimeError, prelude::*};

pub mod exports;
pub mod imports;
pub mod python_api;
pub mod updates;

#[pyclass(name = "DbContext", unsendable)]
pub struct PyDbContext(DbContext);

#[pymethods]
impl PyDbContext {
    #[new]
    #[pyo3(signature = (workspace_path=None, db_path=None))]
    fn new(workspace_path: Option<String>, db_path: Option<String>) -> PyResult<Self> {
        let workspace = match workspace_path {
            Some(path) => Workspace::new(path),
            None => Workspace::from_current_dir(),
        };

        let gen_dir = workspace.ensure_gen_dir();
        let operations_path = gen_dir.join("gen.db");
        let operations_conn =
            r#gen::get_operation_connection(Some(operations_path)).map_err(|err| {
                PyRuntimeError::new_err(format!("Failed to open operations database: {err}"))
            })?;

        let db_path = match db_path {
            Some(path) => path,
            None => {
                let mut stmt = operations_conn
                    .prepare("select db_name from defaults where id = 1;")
                    .map_err(|err| {
                        PyRuntimeError::new_err(format!("Failed to load defaults: {err}"))
                    })?;
                let row: Option<String> = stmt.query_row([], |row| row.get(0)).ok();

                row.unwrap_or_else(|| gen_dir.join("default.db").to_string_lossy().into_owned())
            }
        };

        let graph_conn = r#gen::get_connection(PathBuf::from(&db_path)).map_err(|err| {
            PyRuntimeError::new_err(format!("Failed to open database '{db_path}': {err}"))
        })?;

        Ok(Self(DbContext::new(workspace, graph_conn, operations_conn)))
    }
}

#[pyfunction]
pub fn init() -> PyResult<String> {
    Workspace::from_current_dir().ensure_gen_dir();
    Ok("Gen repository initialized.".to_string())
}
