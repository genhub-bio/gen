use std::{fs::File, path::PathBuf};

use gen_models::operations::Defaults;
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

    let defaults = Defaults::get(operation_conn).unwrap();
    let collection_name = name.unwrap_or_else(|| defaults.collection_name.unwrap());

    if let Err(err) = r#gen::exports::fasta::export_fasta(
        conn,
        &collection_name,
        sample.clone().as_deref(),
        &PathBuf::from(filename),
    ) {
        return Err(PyRuntimeError::new_err(format!(
            "FASTA export failed: {err}"
        )));
    }
    Ok(())
}

#[pyfunction]
pub fn export_gfa(
    context: PyRef<'_, PyDbContext>,
    filename: String,
    name: Option<String>,
    sample: String,
    node_max: Option<i64>,
) -> PyResult<()> {
    println!("GFA export called");
    let context = &context.0;
    let conn = context.graph().conn();
    let operation_conn = context.operations().conn();

    if let Err(err) = r#gen::track_database(conn, operation_conn) {
        panic!("Error tracking database: {err}");
    }

    let defaults = Defaults::get(operation_conn).unwrap();
    let collection_name = name.unwrap_or_else(|| defaults.collection_name.unwrap());

    if let Err(err) = r#gen::exports::gfa::export_gfa(
        conn,
        &collection_name,
        &PathBuf::from(filename),
        &sample,
        node_max,
    ) {
        return Err(PyRuntimeError::new_err(format!("GFA export failed: {err}")));
    }
    Ok(())
}

#[pyfunction]
pub fn export_genbank(
    context: PyRef<'_, PyDbContext>,
    filename: String,
    name: Option<String>,
    sample: String,
) -> PyResult<()> {
    println!("GenBank export called");
    let context = &context.0;
    let conn = context.graph().conn();
    let operation_conn = context.operations().conn();

    if let Err(err) = r#gen::track_database(conn, operation_conn) {
        panic!("Error tracking database: {err}");
    }

    let defaults = Defaults::get(operation_conn).unwrap();
    let collection_name = name.unwrap_or_else(|| defaults.collection_name.unwrap());

    if let Err(err) = r#gen::exports::genbank::export_genbank(
        conn,
        &collection_name,
        &sample,
        File::create(PathBuf::from(filename)).map_err(|io_err| {
            PyRuntimeError::new_err(format!("GenBank export failed: {io_err}"))
        })?,
    ) {
        return Err(PyRuntimeError::new_err(format!(
            "GenBank export failed: {err}"
        )));
    }
    Ok(())
}
