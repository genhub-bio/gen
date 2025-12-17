use std::path::Path;

use gen_core::config::get_gen_dir;
use pyo3::{
    prelude::*,
    types::{PyBytes, PyModule},
};
use rusqlite::{Connection, types::ValueRef};

/// Helper function to convert SQLite errors to Python exceptions
pub fn sqlite_err_to_pyerr(err: rusqlite::Error) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!("SQLite error: {err}"))
}

/// Helper function to convert a Rust path to a Python pathlib.Path object
pub fn path_to_py_path(py: Python, path: &Path) -> PyResult<PyObject> {
    let pathlib = PyModule::import(py, "pathlib")?;
    let path_class = pathlib.getattr("Path")?;
    let py_path = path_class.call1((path.to_str().unwrap(),))?;
    Ok(py_path.into_pyobject(py)?.into())
}

/// Helper function return sqlite query results as a list of lists of Python objects
pub fn py_query(conn: &GraphConnection, query: &str) -> PyResult<Vec<Vec<PyObject>>> {
    let mut stmt = conn.prepare(query).map_err(sqlite_err_to_pyerr)?;
    let column_count = stmt.column_count();
    let mut rows = Vec::new();
    let mut row_iter = stmt.query([]).map_err(sqlite_err_to_pyerr)?;

    Python::with_gil(|py| -> PyResult<_> {
        while let Some(row) = row_iter.next().map_err(sqlite_err_to_pyerr)? {
            let mut row_data = Vec::with_capacity(column_count);
            for i in 0..column_count {
                let value: PyObject = match row.get_ref(i).map_err(sqlite_err_to_pyerr)? {
                    ValueRef::Null => py.None(),
                    ValueRef::Integer(i) => i.into_pyobject(py)?.into(),
                    ValueRef::Real(f) => f.into_pyobject(py)?.into(),
                    ValueRef::Text(s) => std::str::from_utf8(s)
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyUnicodeDecodeError, _>(e.to_string())
                        })?
                        .into_pyobject(py)?
                        .into(),
                    ValueRef::Blob(b) => PyBytes::new(py, b).into_pyobject(py)?.into(),
                };
                row_data.push(value);
            }
            rows.push(row_data);
        }
        Ok(())
    })?;

    Ok(rows)
}

/// Returns the path to the .gen directory as a Python pathlib.Path object.
#[pyfunction(name = "get_gen_dir")]
pub fn get_gen_dir_py(py: Python) -> PyResult<PyObject> {
    match get_gen_dir() {
        Some(dir) => {
            let path = Path::new(&dir);
            path_to_py_path(py, path)
        }
        None => Err(pyo3::exceptions::PyFileNotFoundError::new_err(
            "No .gen directory found. Run 'gen init' in the project root directory to initialize gen.",
        )),
    }
}
