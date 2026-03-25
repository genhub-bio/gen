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
    sample: String,
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
        &sample,
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

#[pyfunction]
pub fn update_with_gfa(
    context: PyRef<'_, PyDbContext>,
    filename: String,
    name: Option<String>,
    sample: String,
    new_sample: String,
) -> PyResult<String> {
    println!("Update with GFA called");

    let context = &context.0;
    let conn = context.graph().conn();
    let operation_conn = context.operations().conn();

    if let Err(err) = r#gen::track_database(conn, operation_conn) {
        panic!("Error tracking database: {err}");
    }

    conn.execute("BEGIN TRANSACTION", []).unwrap();
    operation_conn.execute("BEGIN TRANSACTION", []).unwrap();

    let name = name.unwrap_or_else(|| r#gen::commands::get_default_collection(operation_conn));

    match r#gen::updates::gfa::update_with_gfa(context, &name, &sample, &new_sample, &filename) {
        Ok(_) => {
            conn.execute("END TRANSACTION;", []).unwrap();
            operation_conn.execute("END TRANSACTION;", []).unwrap();
            Ok("Updated with GFA.".to_string())
        }
        Err(err) => {
            conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            operation_conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            Err(PyRuntimeError::new_err(format!("Update failed: {err}")))
        }
    }
}

#[pyfunction]
pub fn update_with_gaf(
    context: PyRef<'_, PyDbContext>,
    filename: String,
    csv: String,
    name: Option<String>,
    sample: String,
    parent_sample: Option<String>,
) -> PyResult<String> {
    println!("Update with GAF called");

    let context = &context.0;
    let conn = context.graph().conn();
    let operation_conn = context.operations().conn();

    if let Err(err) = r#gen::track_database(conn, operation_conn) {
        panic!("Error tracking database: {err}");
    }

    conn.execute("BEGIN TRANSACTION", []).unwrap();
    operation_conn.execute("BEGIN TRANSACTION", []).unwrap();

    let name = name.unwrap_or_else(|| r#gen::commands::get_default_collection(operation_conn));

    if let Err(err) = r#gen::updates::gaf::update_with_gaf(
        context,
        &filename,
        &csv,
        &name,
        &sample,
        parent_sample.as_deref(),
    ) {
        conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
        operation_conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
        return Err(PyRuntimeError::new_err(format!("Update failed: {err}")));
    }

    conn.execute("END TRANSACTION;", []).unwrap();
    operation_conn.execute("END TRANSACTION;", []).unwrap();

    Ok("Updated with GAF.".to_string())
}

#[pyfunction]
pub fn update_with_vcf(
    context: PyRef<'_, PyDbContext>,
    filename: String,
    name: Option<String>,
    genotype: Option<String>,
    sample: Option<String>,
    parent_sample: Option<String>,
    in_place: bool,
) -> PyResult<String> {
    println!("Update with VCF called");

    let context = &context.0;
    let conn = context.graph().conn();
    let operation_conn = context.operations().conn();

    if let Err(err) = r#gen::track_database(conn, operation_conn) {
        panic!("Error tracking database: {err}");
    }

    conn.execute("BEGIN TRANSACTION", []).unwrap();
    operation_conn.execute("BEGIN TRANSACTION", []).unwrap();

    let name = name.unwrap_or_else(|| r#gen::commands::get_default_collection(operation_conn));

    match r#gen::updates::vcf::update_with_vcf(
        context,
        &filename,
        &name,
        genotype.unwrap_or_default(),
        sample.as_deref(),
        parent_sample.as_deref(),
        in_place,
    ) {
        Ok(_) => {
            conn.execute("END TRANSACTION;", []).unwrap();
            operation_conn.execute("END TRANSACTION;", []).unwrap();
            Ok("Updated with VCF.".to_string())
        }
        Err(r#gen::updates::vcf::VcfError::OperationError(OperationError::NoChanges)) => {
            conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            operation_conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            Err(PyRuntimeError::new_err(
                "No changes made. Provide sample and genotype if missing from VCF.",
            ))
        }
        Err(err) => {
            conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            operation_conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            Err(PyRuntimeError::new_err(format!("Update failed: {err}")))
        }
    }
}

#[pyfunction]
pub fn update_with_genbank(
    context: PyRef<'_, PyDbContext>,
    filename: String,
    name: Option<String>,
    sample: String,
    create_missing: bool,
) -> PyResult<String> {
    println!("Update with GenBank called");

    let context = &context.0;
    let conn = context.graph().conn();
    let operation_conn = context.operations().conn();

    if let Err(err) = r#gen::track_database(conn, operation_conn) {
        panic!("Error tracking database: {err}");
    }

    conn.execute("BEGIN TRANSACTION", []).unwrap();
    operation_conn.execute("BEGIN TRANSACTION", []).unwrap();

    let name = name.unwrap_or_else(|| r#gen::commands::get_default_collection(operation_conn));
    let file = std::fs::File::open(&filename)
        .map_err(|err| PyRuntimeError::new_err(format!("Failed to open GenBank file: {err}")))?;

    match r#gen::updates::genbank::update_with_genbank(
        context,
        &file,
        name.as_ref(),
        &sample,
        create_missing,
        &gen_models::operations::OperationInfo {
            files: vec![gen_models::operations::OperationFile {
                file_path: filename.clone(),
                file_type: gen_models::file_types::FileTypes::GenBank,
            }],
            description: "Update from GenBank".to_string(),
        },
    ) {
        Ok(_) => {
            conn.execute("END TRANSACTION;", []).unwrap();
            operation_conn.execute("END TRANSACTION;", []).unwrap();
            Ok("Updated with GenBank.".to_string())
        }
        Err(err) => {
            conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            operation_conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            Err(PyRuntimeError::new_err(format!("Update failed: {err}")))
        }
    }
}

#[pyfunction]
#[expect(
    clippy::too_many_arguments,
    reason = "Python API mirrors the CLI signature to avoid breaking changes"
)]
pub fn update_with_library(
    context: PyRef<'_, PyDbContext>,
    name: Option<String>,
    sample: String,
    new_sample: String,
    path_name: String,
    start: i64,
    end: i64,
    library: String,
    parts: String,
) -> PyResult<String> {
    println!("Update with library called");

    let context = &context.0;
    let conn = context.graph().conn();
    let operation_conn = context.operations().conn();

    if let Err(err) = r#gen::track_database(conn, operation_conn) {
        panic!("Error tracking database: {err}");
    }

    conn.execute("BEGIN TRANSACTION", []).unwrap();
    operation_conn.execute("BEGIN TRANSACTION", []).unwrap();

    let name = name.unwrap_or_else(|| r#gen::commands::get_default_collection(operation_conn));

    if let Err(err) = r#gen::updates::library::update_with_library(
        context,
        &name,
        &sample,
        &new_sample,
        &path_name,
        start,
        end,
        &parts,
        &library,
    ) {
        conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
        operation_conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
        return Err(PyRuntimeError::new_err(format!("Update failed: {err}")));
    }

    conn.execute("END TRANSACTION;", []).unwrap();
    operation_conn.execute("END TRANSACTION;", []).unwrap();

    Ok("Updated with library.".to_string())
}

#[pyfunction]
#[expect(
    clippy::too_many_arguments,
    reason = "Python API mirrors the CLI signature to avoid breaking changes"
)]
pub fn update_with_sequence(
    context: PyRef<'_, PyDbContext>,
    sequence: String,
    name: Option<String>,
    sample: String,
    new_sample: String,
    region_name: String,
    start: i64,
    end: i64,
    no_reference_path_update: bool,
) -> PyResult<String> {
    println!("Update with sequence called");

    let context = &context.0;
    let conn = context.graph().conn();
    let operation_conn = context.operations().conn();

    if let Err(err) = r#gen::track_database(conn, operation_conn) {
        panic!("Error tracking database: {err}");
    }

    conn.execute("BEGIN TRANSACTION", []).unwrap();
    operation_conn.execute("BEGIN TRANSACTION", []).unwrap();

    let name = name.unwrap_or_else(|| r#gen::commands::get_default_collection(operation_conn));

    if let Err(err) = r#gen::updates::sequence::update_with_sequence(
        context,
        &name,
        &sample,
        &new_sample,
        &region_name,
        start,
        end,
        &sequence,
        no_reference_path_update,
    ) {
        conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
        operation_conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
        return Err(PyRuntimeError::new_err(format!("Update failed: {err}")));
    }

    conn.execute("END TRANSACTION;", []).unwrap();
    operation_conn.execute("END TRANSACTION;", []).unwrap();

    Ok("Updated with sequence.".to_string())
}
