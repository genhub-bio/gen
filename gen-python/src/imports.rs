use std::{fs::File, path::PathBuf};

use gen_models::{
    errors::OperationError,
    file_types::FileTypes,
    operations::{OperationFile, OperationInfo},
};
use pyo3::{exceptions::PyRuntimeError, prelude::*};

use crate::PyDbContext;

#[pyfunction]
#[pyo3(signature = (context, filename, name=None, sample=None, shallow=false))]
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

    if let Err(err) = r#gen::track_database(conn, operation_conn) {
        panic!("Error tracking database: {err}");
    }

    conn.execute("BEGIN TRANSACTION", []).unwrap();
    operation_conn.execute("BEGIN TRANSACTION", []).unwrap();

    let name = name.unwrap_or_else(|| r#gen::commands::get_default_collection(operation_conn));

    match r#gen::imports::fasta::import_fasta(
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

#[pyfunction]
#[pyo3(signature = (context, filename, name=None, sample=None))]
pub fn import_gfa(
    context: PyRef<'_, PyDbContext>,
    filename: String,
    name: Option<String>,
    sample: Option<String>,
) -> PyResult<String> {
    println!("GFA import called");

    let context = &context.0;
    let operation_conn = context.operations().conn();
    let conn = context.graph().conn();

    if let Err(err) = r#gen::track_database(conn, operation_conn) {
        panic!("Error tracking database: {err}");
    }

    conn.execute("BEGIN TRANSACTION", []).unwrap();
    operation_conn.execute("BEGIN TRANSACTION", []).unwrap();

    let name = name.unwrap_or_else(|| r#gen::commands::get_default_collection(operation_conn));

    match r#gen::imports::gfa::import_gfa(
        context,
        &PathBuf::from(filename),
        &name,
        sample.as_deref(),
    ) {
        Ok(_) => {
            conn.execute("END TRANSACTION;", []).unwrap();
            operation_conn.execute("END TRANSACTION;", []).unwrap();
            Ok("GFA imported.".to_string())
        }
        Err(r#gen::imports::gfa::GFAImportError::OperationError(OperationError::NoChanges)) => {
            conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            operation_conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            Err(PyRuntimeError::new_err("GFA already exists."))
        }
        Err(_) => {
            conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            operation_conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            Err(PyRuntimeError::new_err("Import failed."))
        }
    }
}

#[pyfunction]
#[pyo3(signature = (context, filename, name=None, sample=None))]
pub fn import_genbank(
    context: PyRef<'_, PyDbContext>,
    filename: String,
    name: Option<String>,
    sample: Option<String>,
) -> PyResult<String> {
    println!("GenBank import called");

    let context = &context.0;
    let operation_conn = context.operations().conn();
    let conn = context.graph().conn();

    if let Err(err) = r#gen::track_database(conn, operation_conn) {
        panic!("Error tracking database: {err}");
    }

    conn.execute("BEGIN TRANSACTION", []).unwrap();
    operation_conn.execute("BEGIN TRANSACTION", []).unwrap();

    let name = name.unwrap_or_else(|| r#gen::commands::get_default_collection(operation_conn));
    let mut reader: Box<dyn std::io::Read> = if filename.ends_with(".gz") {
        let file = File::open(&filename).map_err(|err| {
            PyRuntimeError::new_err(format!("Failed to open GenBank file: {err}"))
        })?;
        Box::new(flate2::read::GzDecoder::new(file))
    } else {
        Box::new(File::open(&filename).map_err(|err| {
            PyRuntimeError::new_err(format!("Failed to open GenBank file: {err}"))
        })?)
    };

    match r#gen::imports::genbank::import_genbank(
        context,
        &mut reader,
        name.as_ref(),
        sample.as_deref(),
        OperationInfo {
            files: vec![OperationFile {
                file_path: filename.clone(),
                file_type: FileTypes::GenBank,
            }],
            description: "GenBank Import".to_string(),
        },
    ) {
        Ok(_) => {
            conn.execute("END TRANSACTION;", []).unwrap();
            operation_conn.execute("END TRANSACTION;", []).unwrap();
            Ok("GenBank imported.".to_string())
        }
        Err(err) => {
            conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            operation_conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            Err(PyRuntimeError::new_err(format!("Import failed: {err}")))
        }
    }
}

#[pyfunction]
#[pyo3(signature = (context, library_name, parts, library, name=None, sample=None))]
pub fn import_library(
    context: PyRef<'_, PyDbContext>,
    library_name: String,
    parts: String,
    library: String,
    name: Option<String>,
    sample: Option<String>,
) -> PyResult<String> {
    println!("Library import called");

    let context = &context.0;
    let operation_conn = context.operations().conn();
    let conn = context.graph().conn();

    if let Err(err) = r#gen::track_database(conn, operation_conn) {
        panic!("Error tracking database: {err}");
    }

    conn.execute("BEGIN TRANSACTION", []).unwrap();
    operation_conn.execute("BEGIN TRANSACTION", []).unwrap();

    let name = name.unwrap_or_else(|| r#gen::commands::get_default_collection(operation_conn));

    match r#gen::imports::library::import_library(
        context,
        &name,
        sample.as_deref(),
        &parts,
        &library,
        &library_name,
    ) {
        Ok(_) => {
            conn.execute("END TRANSACTION;", []).unwrap();
            operation_conn.execute("END TRANSACTION;", []).unwrap();
            Ok("Library imported.".to_string())
        }
        Err(r#gen::imports::library::LibraryImportError::OperationError(
            OperationError::NoChanges,
        )) => {
            conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            operation_conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            Err(PyRuntimeError::new_err("Library already exists."))
        }
        Err(err) => {
            conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            operation_conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            Err(PyRuntimeError::new_err(format!(
                "Library import failed: {err}"
            )))
        }
    }
}
