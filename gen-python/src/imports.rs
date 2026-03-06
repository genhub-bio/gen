use std::{fs::File, path::PathBuf};

use r#gen::graphs::combinatorial_library::{SequencePart, parse_library};
use gen_models::{
    errors::OperationError,
    file_types::FileTypes,
    operations::{Defaults, OperationFile, OperationInfo},
    sample::Sample,
};
use pyo3::{exceptions::PyRuntimeError, prelude::*};

use crate::{PyDbContext, python_api::sequence_part::PySequencePart};

#[pyfunction]
pub fn import_fasta(
    context: PyRef<'_, PyDbContext>,
    filename: String,
    name: Option<String>,
    sample: String,
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

    let defaults = Defaults::get(operation_conn).unwrap();
    let name = name.unwrap_or_else(|| defaults.collection_name.unwrap());

    match r#gen::imports::fasta::import_fasta(context, &filename, &name, &sample, shallow) {
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
pub fn import_gfa(
    context: PyRef<'_, PyDbContext>,
    filename: String,
    name: Option<String>,
    sample: String,
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

    let defaults = Defaults::get(operation_conn).unwrap();
    let name = name.unwrap_or_else(|| defaults.collection_name.unwrap());

    match r#gen::imports::gfa::import_gfa(context, &PathBuf::from(filename), &name, &sample) {
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
pub fn import_genbank(
    context: PyRef<'_, PyDbContext>,
    filename: String,
    name: Option<String>,
    sample: String,
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

    let defaults = Defaults::get(operation_conn).unwrap();
    let name = name.unwrap_or_else(|| defaults.collection_name.unwrap());

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
        &sample,
        OperationInfo {
            files: vec![OperationFile {
                file_path: filename.clone(),
                file_type: FileTypes::GenBank,
            }],
            description: "GenBank Import".to_string(),
        },
        r#gen::imports::genbank::GenBankImportOptions::default(),
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
pub fn import_library_files(
    context: PyRef<'_, PyDbContext>,
    library_name: String,
    parts: String,
    library: String,
    name: Option<String>,
    sample: String,
) -> PyResult<String> {
    println!("Import library with files called");

    let parts_list = match parse_library(&parts, &library) {
        Ok(result) => result,
        Err(err) => {
            return Err(PyRuntimeError::new_err(format!(
                "Problem parsing library files: {err}"
            )));
        }
    };

    let context = &context.0;
    let operation_conn = context.operations().conn();
    let conn = context.graph().conn();

    if let Err(err) = r#gen::track_database(conn, operation_conn) {
        panic!("Error tracking database: {err}");
    }

    conn.execute("BEGIN TRANSACTION", []).unwrap();
    operation_conn.execute("BEGIN TRANSACTION", []).unwrap();

    let defaults = Defaults::get(operation_conn).unwrap();
    let name = name.unwrap_or_else(|| defaults.collection_name.unwrap());

    match r#gen::imports::library::import_library(
        context,
        &name,
        &sample,
        &library_name,
        parts_list,
        Some(&parts),
        Some(&library),
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

#[pyfunction]
pub fn import_library(
    context: PyRef<'_, PyDbContext>,
    library_name: String,
    parts_list: Vec<Vec<PySequencePart>>,
    name: Option<String>,
    sample: Option<String>,
) -> PyResult<String> {
    println!("Import library called");

    let context = &context.0;
    let operation_conn = context.operations().conn();
    let conn = context.graph().conn();

    if let Err(err) = r#gen::track_database(conn, operation_conn) {
        panic!("Error tracking database: {err}");
    }

    conn.execute("BEGIN TRANSACTION", []).unwrap();
    operation_conn.execute("BEGIN TRANSACTION", []).unwrap();

    let defaults = Defaults::get(operation_conn).unwrap();
    let name = name.unwrap_or_else(|| defaults.collection_name.unwrap());

    let rust_parts_list = parts_list
        .iter()
        .map(|parts| {
            parts
                .iter()
                .map(|part| SequencePart {
                    name: part.name.clone(),
                    sequence: part.sequence.clone(),
                    sequence_length: part.sequence_length,
                })
                .collect()
        })
        .collect();

    let sample_name = if let Some(sample) = sample {
        &sample.clone()
    } else {
        Sample::DEFAULT_NAME
    };

    match r#gen::imports::library::import_library(
        context,
        &name,
        sample_name,
        &library_name,
        rust_parts_list,
        None,
        None,
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
