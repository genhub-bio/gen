use std::path::PathBuf;

use pyo3::{exceptions::PyRuntimeError, prelude::*};

use crate::PyDbContext;

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

    if let Err(err) = r#gen::track_database(conn, operation_conn) {
        panic!("Error tracking database: {err}");
    }

    let name = name.unwrap_or_else(|| r#gen::commands::get_default_collection(operation_conn));

    if let Err(err) = r#gen::exports::fasta::export_fasta(
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
