use std::path::PathBuf;

use r#gen::{
    commands::{get_db_for_command, get_default_collection},
    exports::fasta::export_fasta as gen_export_fasta,
    fasta::FastaError,
    get_connection, get_operation_connection,
    imports::fasta::import_fasta as gen_import_fasta,
    track_database,
    updates::fasta::update_with_fasta as gen_update_with_fasta,
};
use gen_core::config::Workspace;
use gen_models::{db::DbContext, errors::OperationError};
use pyo3::{exceptions::PyRuntimeError, prelude::*};

#[pyfunction]
fn init() -> PyResult<String> {
    Workspace::from_current_dir().ensure_gen_dir();
    Ok("Gen repository initialized.".to_string())
}

#[pyfunction]
fn import_fasta(
    filename: String,
    db_name: Option<String>,
    name: Option<String>,
    sample: Option<String>,
    shallow: bool,
) -> PyResult<String> {
    println!("Fasta import called");

    let workspace = Workspace::from_current_dir();
    let operation_conn = get_operation_connection(Some(
        workspace.gen_db_path().expect("No .gen directory found."),
    ))
    .unwrap();
    let db = get_db_for_command(db_name, &operation_conn);
    let conn = get_connection(&db).unwrap();
    let context = DbContext::new(workspace, conn, operation_conn);

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

    match gen_import_fasta(
        &context,
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
#[allow(clippy::too_many_arguments)]
pub fn update_with_fasta(
    filename: String,
    db_name: Option<String>,
    name: Option<String>,
    sample: Option<String>,
    new_sample: String,
    region_name: String,
    start: i64,
    end: i64,
) -> PyResult<String> {
    println!("Update with fasta called");

    let workspace = Workspace::from_current_dir();
    let operation_conn = get_operation_connection(Some(
        workspace.gen_db_path().expect("No .gen directory found."),
    ))
    .unwrap();
    let db = get_db_for_command(db_name, &operation_conn);
    let conn = get_connection(&db).unwrap();
    let context = DbContext::new(workspace, conn, operation_conn);

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
        &context,
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
    filename: String,
    name: Option<String>,
    db_name: Option<String>,
    sample: Option<String>,
) -> PyResult<()> {
    println!("GFA export called");
    let operation_conn = get_operation_connection(None).unwrap();
    let db = get_db_for_command(db_name.clone(), &operation_conn);
    let conn = get_connection(&db).unwrap();

    match track_database(&conn, &operation_conn) {
        Ok(_) => {}
        Err(err) => {
            panic!("Error tracking database: {err}");
        }
    };

    // initialize the selected database if needed.

    conn.execute("BEGIN TRANSACTION", []).unwrap();
    operation_conn.execute("BEGIN TRANSACTION", []).unwrap();

    let name = name.unwrap_or_else(|| get_default_collection(&operation_conn));

    if let Err(err) = gen_export_fasta(
        &conn,
        &name,
        sample.clone().as_deref(),
        &PathBuf::from(filename),
    ) {
        conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
        operation_conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
        return Err(PyRuntimeError::new_err(format!(
            "FASTA export failed: {err}"
        )));
    }

    conn.execute("END TRANSACTION", []).unwrap();
    operation_conn.execute("END TRANSACTION", []).unwrap();

    Ok(())
}

/// A Python module implemented in Rust.
#[pymodule]
fn gen_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(init, m)?)?;
    m.add_function(wrap_pyfunction!(import_fasta, m)?)?;
    m.add_function(wrap_pyfunction!(update_with_fasta, m)?)?;
    m.add_function(wrap_pyfunction!(export_fasta, m)?)?;

    Ok(())
}
