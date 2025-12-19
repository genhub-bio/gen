use std::path::PathBuf;

use r#gen::{
    commands::get_default_collection, exports::fasta::export_fasta as gen_export_fasta,
    fasta::FastaError, get_connection, get_operation_connection,
    imports::fasta::import_fasta as gen_import_fasta, track_database,
    updates::fasta::update_with_fasta as gen_update_with_fasta,
};
use gen_core::config::Workspace;
use gen_models::{db::DbContext, errors::OperationError};
use pyo3::{exceptions::PyRuntimeError, prelude::*};

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
        let operations_conn = get_operation_connection(Some(operations_path)).map_err(|err| {
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

        let graph_conn = get_connection(PathBuf::from(&db_path)).map_err(|err| {
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

#[pyfunction]
pub fn import_fasta(
    context: PyRef<'_, PyDbContext>,
    filename: String,
    name: Option<String>,
    sample: Option<String>,
    shallow: bool,
) -> PyResult<String> {
    println!("Fasta import called");

    let context = &context.0;
    let operation_conn = context.operations().conn();
    let conn = context.graph().conn();

    match track_database(conn, operation_conn) {
        Ok(_) => {}
        Err(err) => {
            panic!("Error tracking database: {err}");
        }
    };

    // initialize the selected database if needed.
    conn.execute("BEGIN TRANSACTION", []).unwrap();
    operation_conn.execute("BEGIN TRANSACTION", []).unwrap();

    let name = name.unwrap_or_else(|| get_default_collection(operation_conn));

    match gen_import_fasta(
        context,
        &filename,
        &name,
        sample.clone().as_deref(),
        shallow,
    ) {
        Ok(_) => {
            conn.execute("END TRANSACTION;", []).unwrap();
            operation_conn.execute("END TRANSACTION;", []).unwrap();
            Ok("Fasta imported.".to_string())
        }
        Err(FastaError::OperationError(OperationError::NoChanges)) => {
            conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            operation_conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            Err(PyRuntimeError::new_err("Fasta contents already exist."))
        }
        Err(_) => {
            conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            operation_conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            Err(PyRuntimeError::new_err("Import failed."))
        }
    }
}

#[pyfunction]
#[expect(
    clippy::too_many_arguments,
    reason = "Python API mirrors the CLI signature to avoid breaking changes"
)]
pub fn update_with_fasta(
    context: PyRef<'_, PyDbContext>,
    filename: String,
    name: Option<String>,
    sample: Option<String>,
    new_sample: String,
    region_name: String,
    start: i64,
    end: i64,
) -> PyResult<String> {
    println!("Update with fasta called");

    let context = &context.0;
    let conn = context.graph().conn();
    let operation_conn = context.operations().conn();

    match track_database(conn, operation_conn) {
        Ok(_) => {}
        Err(err) => {
            panic!("Error tracking database: {err}");
        }
    };

    // initialize the selected database if needed.

    conn.execute("BEGIN TRANSACTION", []).unwrap();
    operation_conn.execute("BEGIN TRANSACTION", []).unwrap();

    let name = name.unwrap_or_else(|| get_default_collection(operation_conn));

    // TODO: Take as parameter
    let no_reference_path_update = false;

    match gen_update_with_fasta(
        context,
        name.as_str(),
        sample.clone().as_deref(),
        &new_sample,
        &region_name,
        start,
        end,
        &filename,
        no_reference_path_update,
    ) {
        Ok(_) => {
            conn.execute("END TRANSACTION;", []).unwrap();
            operation_conn.execute("END TRANSACTION;", []).unwrap();
            Ok("Updated with fasta.".to_string())
        }
        Err(FastaError::OperationError(OperationError::NoChanges)) => {
            conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            operation_conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            Err(PyRuntimeError::new_err("Fasta contents already exist."))
        }
        Err(_) => {
            conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            operation_conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            Err(PyRuntimeError::new_err("Import failed."))
        }
    }
}

#[pyfunction]
pub fn export_fasta(
    context: PyRef<'_, PyDbContext>,
    filename: String,
    name: Option<String>,
    sample: Option<String>,
) -> PyResult<()> {
    println!("GFA export called");
    let context = &context.0;
    let conn = context.graph().conn();
    let operation_conn = context.operations().conn();

    match track_database(conn, operation_conn) {
        Ok(_) => {}
        Err(err) => {
            panic!("Error tracking database: {err}");
        }
    };

    // initialize the selected database if needed.

    let name = name.unwrap_or_else(|| get_default_collection(operation_conn));

    if let Err(err) = gen_export_fasta(
        conn,
        &name,
        sample.clone().as_deref(),
        &PathBuf::from(filename),
    ) {
        return Err(PyRuntimeError::new_err(format!(
            "FASTA export failed: {err}"
        )));
    }
    Ok(())
}

/// A Python module implemented in Rust.
#[pymodule]
fn gen_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDbContext>()?;
    m.add_function(wrap_pyfunction!(init, m)?)?;
    m.add_function(wrap_pyfunction!(import_fasta, m)?)?;
    m.add_function(wrap_pyfunction!(update_with_fasta, m)?)?;
    m.add_function(wrap_pyfunction!(export_fasta, m)?)?;

    Ok(())
}
