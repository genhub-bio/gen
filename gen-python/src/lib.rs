use core::ops::Range;
use std::{collections::HashMap, fs::File, path::PathBuf};

use r#gen::{
    commands::{get_db_for_command, get_default_collection},
    exports::fasta::export_fasta as gen_export_fasta,
    fasta::FastaError,
    get_connection, get_operation_connection,
    graph_operators::{
        GraphOperationError, derive_chunks, get_path, make_stitch as gen_make_stitch,
    },
    imports::{
        fasta::import_fasta as gen_import_fasta,
        genbank::import_genbank as gen_import_genbank,
        gfa::{GFAImportError, import_gfa as gen_import_gfa},
        library::import_design,
    },
    track_database,
    updates::fasta::{
        update_with_fasta as gen_update_with_fasta,
        update_with_sequences as gen_update_with_sequences,
    },
};
use gen_core::config::get_or_create_gen_dir;
use gen_models::{
    errors::OperationError,
    file_types::FileTypes,
    operations::{OperationFile, OperationInfo, setup_db},
    sample::Sample,
    session_operations,
};
use pyo3::{exceptions::PyRuntimeError, prelude::*};

#[pyfunction]
fn init() -> PyResult<String> {
    get_or_create_gen_dir();
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

    let operation_conn = get_operation_connection(None).unwrap();

    let db = get_db_for_command(db_name, &operation_conn);
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

    match gen_import_fasta(
        &filename,
        &name,
        sample.clone().as_deref(),
        shallow,
        &conn,
        &operation_conn,
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
fn import_genbank(
    filename: String,
    db_name: Option<String>,
    name: Option<String>,
    sample: Option<String>,
) -> PyResult<String> {
    let operation_conn = get_operation_connection(None).unwrap();
    setup_db(&operation_conn);

    let db = get_db_for_command(db_name, &operation_conn);
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

    let name = name
        .clone()
        .unwrap_or_else(|| get_default_collection(&operation_conn));
    let mut reader: Box<dyn std::io::Read> = if filename.ends_with(".gz") {
        let file = File::open(filename.clone()).unwrap();
        Box::new(flate2::read::GzDecoder::new(file))
    } else {
        Box::new(File::open(filename.clone()).unwrap())
    };
    match gen_import_genbank(
        &conn,
        &operation_conn,
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
            Err(PyRuntimeError::new_err(format!("Import failed: {:?}", err)))
        }
    }
}

#[pyfunction]
fn import_gfa(
    filename: String,
    db_name: Option<String>,
    name: Option<String>,
    sample: Option<String>,
) -> PyResult<String> {
    let operation_conn = get_operation_connection(None).unwrap();
    setup_db(&operation_conn);

    let db = get_db_for_command(db_name, &operation_conn);
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

    let name = name
        .clone()
        .unwrap_or_else(|| get_default_collection(&operation_conn));
    match gen_import_gfa(
        &PathBuf::from(filename.clone()),
        &name,
        sample.as_deref(),
        &conn,
        &operation_conn,
    ) {
        Ok(_) => {
            conn.execute("END TRANSACTION;", []).unwrap();
            operation_conn.execute("END TRANSACTION;", []).unwrap();
            Ok("GFA imported.".to_string())
        }
        Err(GFAImportError::OperationError(OperationError::NoChanges)) => {
            conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            operation_conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            Ok("GFA already exists.".to_string())
        }
        Err(err) => {
            conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            operation_conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            Err(PyRuntimeError::new_err(format!("Import failed: {:?}", err)))
        }
    }
}

#[pyfunction]
fn import_library(
    db_name: Option<String>,
    name: Option<String>,
    sample: Option<String>,
    part_sequences_by_name: HashMap<String, String>,
    library: HashMap<i64, Vec<String>>,
    region_name: &str,
) -> PyResult<String> {
    let operation_conn = get_operation_connection(None).unwrap();
    setup_db(&operation_conn);

    let db = get_db_for_command(db_name, &operation_conn);
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

    let name = name
        .clone()
        .unwrap_or_else(|| get_default_collection(&operation_conn));

    let mut session = session_operations::start_operation(&conn);

    let result = import_design(
        &conn,
        &name,
        sample.as_deref(),
        &part_sequences_by_name,
        &library,
        region_name,
    );

    match result {
        Ok(path_changes_count) => {
            // TODO: Handle error cases.  Do we need a way to roll back an operation that was started?

            let summary_str = format!("{region_name}: {path_changes_count} changes.\n");
            session_operations::end_operation(
                &conn,
                &operation_conn,
                &mut session,
                &OperationInfo {
                    files: vec![],
                    description: "library_csv_import".to_string(),
                },
                &summary_str,
                None,
            )
            .unwrap();

            conn.execute("END TRANSACTION;", []).unwrap();
            operation_conn.execute("END TRANSACTION;", []).unwrap();
            Ok("Library imported.".to_string())
        }
        Err(_) => {
            conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            operation_conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            Err(PyRuntimeError::new_err("Import failed."))
        }
    }
}

#[pyfunction]
fn update_with_fasta(
    filename: String,
    db_name: Option<String>,
    name: Option<String>,
    sample: Option<String>,
    new_sample: String,
    region_name: String,
    start: i64,
    end: i64,
    no_reference_path_update: bool,
) -> PyResult<String> {
    println!("Update with fasta called");

    let operation_conn = get_operation_connection(None).unwrap();
    let db = get_db_for_command(db_name, &operation_conn);
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

    match gen_update_with_fasta(
        &conn,
        &operation_conn,
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
            Err(PyRuntimeError::new_err("Update failed."))
        }
    }
}

#[pyfunction]
fn update_with_sequence(
    sequence: String,
    db_name: Option<String>,
    name: Option<String>,
    sample: Option<String>,
    new_sample: String,
    region_name: String,
    start: i64,
    end: i64,
    no_reference_path_update: bool,
) -> PyResult<String> {
    println!("Update with sequence called");

    let operation_conn = get_operation_connection(None).unwrap();
    let db = get_db_for_command(db_name, &operation_conn);
    let conn = get_connection(&db).unwrap();

    match track_database(&conn, &operation_conn) {
        Ok(_) => {}
        Err(err) => {
            panic!("Error tracking database: {err}");
        }
    };

    // initialize the selected database if needed.
    setup_db(&operation_conn);
    conn.execute("BEGIN TRANSACTION", []).unwrap();
    operation_conn.execute("BEGIN TRANSACTION", []).unwrap();

    let name = name.unwrap_or_else(|| get_default_collection(&operation_conn));

    match gen_update_with_sequences(
        &conn,
        name.as_str(),
        sample.clone().as_deref(),
        &new_sample,
        &region_name,
        start,
        end,
        vec![sequence],
        no_reference_path_update,
    ) {
        Ok(_) => {
            conn.execute("END TRANSACTION;", []).unwrap();
            operation_conn.execute("END TRANSACTION;", []).unwrap();
            Ok("Updated with sequence.".to_string())
        }
        Err(_) => {
            conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            operation_conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            Err(PyRuntimeError::new_err("Update failed."))
        }
    }
}

#[pyfunction]
fn export_fasta(
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

    gen_export_fasta(
        &conn,
        &name,
        sample.clone().as_deref(),
        &PathBuf::from(filename),
    );

    conn.execute("END TRANSACTION", []).unwrap();
    operation_conn.execute("END TRANSACTION", []).unwrap();

    Ok(())
}

#[pyfunction]
fn derive_subgraph(
    db_name: Option<String>,
    name: &str,
    sample_name: Option<&str>,
    new_sample_name: &str,
    region_name: &str,
    start_coordinate: i64,
    end_coordinate: i64,
    backbone: Option<&str>,
) -> PyResult<String> {
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
    setup_db(&operation_conn);

    conn.execute("BEGIN TRANSACTION", []).unwrap();
    operation_conn.execute("BEGIN TRANSACTION", []).unwrap();

    match derive_chunks(
        &conn,
        &operation_conn,
        name,
        sample_name,
        new_sample_name,
        region_name,
        backbone.as_deref(),
        vec![Range {
            start: start_coordinate,
            end: end_coordinate,
        }],
    ) {
        Ok(_) => {
            conn.execute("END TRANSACTION;", []).unwrap();
            operation_conn.execute("END TRANSACTION;", []).unwrap();
            Ok("Derived subgraph successfully.".to_string())
        }
        Err(e) => {
            conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            operation_conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            Err(PyRuntimeError::new_err(format!(
                "Error deriving subgraph: {}",
                e
            )))
        }
    }
}

#[pyfunction]
fn derive_breakpoint_chunks(
    db_name: Option<String>,
    name: &str,
    sample_name: Option<&str>,
    new_sample_name: &str,
    region_name: &str,
    breakpoints: Vec<i64>,
    backbone: Option<&str>,
) -> PyResult<String> {
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
    setup_db(&operation_conn);

    conn.execute("BEGIN TRANSACTION", []).unwrap();
    operation_conn.execute("BEGIN TRANSACTION", []).unwrap();

    let path_length = match get_path(
        &conn,
        name,
        sample_name.as_deref(),
        region_name,
        backbone.as_deref(),
    ) {
        Ok(path) => path.length(&conn),
        Err(e) => panic!("Error deriving subgraph(s): {e}"),
    };

    if breakpoints.is_empty() {
        panic!("No chunk coordinates provided.");
    }

    if breakpoints[breakpoints.len() - 1] > path_length {
        panic!("At least one chunk coordinate exceeds path length.");
    }

    let mut range_start = 0;
    let mut chunk_ranges = vec![];
    for chunk_point in breakpoints {
        chunk_ranges.push(Range {
            start: range_start,
            end: chunk_point,
        });
        range_start = chunk_point;
    }
    chunk_ranges.push(Range {
        start: range_start,
        end: path_length,
    });

    match derive_chunks(
        &conn,
        &operation_conn,
        name,
        sample_name,
        new_sample_name,
        region_name,
        backbone.as_deref(),
        chunk_ranges,
    ) {
        Ok(_) => {
            conn.execute("END TRANSACTION;", []).unwrap();
            operation_conn.execute("END TRANSACTION;", []).unwrap();
            Ok("Derived chunks successfully.".to_string())
        }
        Err(e) => {
            conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            operation_conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            Err(PyRuntimeError::new_err(format!(
                "Error deriving chunks: {}",
                e
            )))
        }
    }
}

#[pyfunction]
fn derive_sized_chunks(
    db_name: Option<String>,
    name: &str,
    sample_name: Option<&str>,
    new_sample_name: &str,
    region_name: &str,
    chunk_size: i64,
    backbone: Option<&str>,
) -> PyResult<String> {
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
    setup_db(&operation_conn);

    conn.execute("BEGIN TRANSACTION", []).unwrap();
    operation_conn.execute("BEGIN TRANSACTION", []).unwrap();

    let path_length = match get_path(
        &conn,
        name,
        sample_name.as_deref(),
        region_name,
        backbone.as_deref(),
    ) {
        Ok(path) => path.length(&conn),
        Err(e) => panic!("Error deriving subgraph(s): {e}"),
    };

    let chunk_count = path_length / chunk_size;
    let chunk_points = (0..chunk_count)
        .map(|i| i * chunk_size)
        .collect::<Vec<i64>>();

    let mut range_start = 0;
    let mut chunk_ranges = vec![];
    for chunk_point in chunk_points {
        chunk_ranges.push(Range {
            start: range_start,
            end: chunk_point,
        });
        range_start = chunk_point;
    }
    chunk_ranges.push(Range {
        start: range_start,
        end: path_length,
    });

    match derive_chunks(
        &conn,
        &operation_conn,
        name,
        sample_name,
        new_sample_name,
        region_name,
        backbone.as_deref(),
        chunk_ranges,
    ) {
        Ok(_) => {
            conn.execute("END TRANSACTION;", []).unwrap();
            operation_conn.execute("END TRANSACTION;", []).unwrap();
            Ok("Derived chunks successfully.".to_string())
        }
        Err(e) => {
            conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            operation_conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            Err(PyRuntimeError::new_err(format!(
                "Error deriving chunks: {}",
                e
            )))
        }
    }
}

#[pyfunction]
fn make_stitch(
    db_name: Option<String>,
    name: &str,
    sample_name: Option<&str>,
    new_sample_name: &str,
    region_names: Vec<String>,
    new_region_name: &str,
) -> PyResult<String> {
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
    setup_db(&operation_conn);

    conn.execute("BEGIN TRANSACTION", []).unwrap();
    operation_conn.execute("BEGIN TRANSACTION", []).unwrap();

    let referenced_region_names = region_names
        .iter()
        .map(|region_name| region_name.as_str())
        .collect();

    match gen_make_stitch(
        &conn,
        &operation_conn,
        name,
        sample_name.as_deref(),
        new_sample_name,
        &referenced_region_names,
        new_region_name,
    ) {
        Ok(_) => {
            conn.execute("END TRANSACTION;", []).unwrap();
            operation_conn.execute("END TRANSACTION;", []).unwrap();
            Ok("Stitched regions successfully.".to_string())
        }
        Err(GraphOperationError::RegionNotFound(_)) => {
            conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            operation_conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            Err(PyRuntimeError::new_err(
                "One or more input regions do not exist.",
            ))
        }
        Err(e) => {
            conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            operation_conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            Err(PyRuntimeError::new_err(format!(
                "Error stitching regions: {}",
                e
            )))
        }
    }
}

#[pyfunction]
fn list_samples(db_name: Option<String>) -> Vec<String> {
    let operation_conn = get_operation_connection(None).unwrap();
    let db = get_db_for_command(db_name.clone(), &operation_conn);
    let conn = get_connection(&db).unwrap();

    let mut samples = vec!["".to_string()]; // Null sample

    let sample_names = Sample::get_all_names(&conn);
    for sample_name in &sample_names {
        samples.push(sample_name.to_string());
    }

    samples
}

#[pyfunction]
fn list_graphs(
    db_name: Option<String>,
    name: Option<String>,
    sample: Option<String>,
) -> Vec<String> {
    let operation_conn = get_operation_connection(None).unwrap();
    let db = get_db_for_command(db_name.clone(), &operation_conn);
    let conn = get_connection(&db).unwrap();

    let name = &name
        .clone()
        .unwrap_or_else(|| get_default_collection(&operation_conn));
    let block_groups = Sample::get_block_groups(&conn, name, sample.as_deref());

    let mut graphs = vec![];
    for block_group in block_groups {
        graphs.push(block_group.name);
    }

    graphs
}

/// A Python module implemented in Rust.
#[pymodule]
fn gen_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(init, m)?)?;
    m.add_function(wrap_pyfunction!(import_fasta, m)?)?;
    m.add_function(wrap_pyfunction!(import_genbank, m)?)?;
    m.add_function(wrap_pyfunction!(import_gfa, m)?)?;
    m.add_function(wrap_pyfunction!(import_library, m)?)?;
    m.add_function(wrap_pyfunction!(update_with_fasta, m)?)?;
    m.add_function(wrap_pyfunction!(update_with_sequence, m)?)?;
    m.add_function(wrap_pyfunction!(export_fasta, m)?)?;
    m.add_function(wrap_pyfunction!(derive_subgraph, m)?)?;
    m.add_function(wrap_pyfunction!(derive_breakpoint_chunks, m)?)?;
    m.add_function(wrap_pyfunction!(derive_sized_chunks, m)?)?;
    m.add_function(wrap_pyfunction!(make_stitch, m)?)?;
    m.add_function(wrap_pyfunction!(list_samples, m)?)?;
    m.add_function(wrap_pyfunction!(list_graphs, m)?)?;

    Ok(())
}
