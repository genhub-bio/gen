use gen_models::errors::OperationError;
use pyo3::{exceptions::PyRuntimeError, prelude::*};

use crate::PyDbContext;

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

    if let Err(err) = r#gen::track_database(conn, operation_conn) {
        panic!("Error tracking database: {err}");
    }

    conn.execute("BEGIN TRANSACTION", []).unwrap();
    operation_conn.execute("BEGIN TRANSACTION", []).unwrap();

    let name = name.unwrap_or_else(|| r#gen::commands::get_default_collection(operation_conn));

    let no_reference_path_update = false;

    match r#gen::updates::fasta::update_with_fasta(
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
        Err(r#gen::fasta::FastaError::OperationError(OperationError::NoChanges)) => {
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
