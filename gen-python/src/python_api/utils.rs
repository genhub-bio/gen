use std::{path::Path, str};

use gen_models::block_group::BlockGroupError;
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyBytes, PyModule},
};
use rusqlite::{Connection, types::ValueRef};

/// Helper function to convert SQLite errors to Python exceptions
pub fn sqlite_err_to_pyerr(err: rusqlite::Error) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!("SQLite error: {err}"))
}

/// Helper function to convert SQLite errors to Python exceptions
pub fn block_group_err_to_pyerr(err: BlockGroupError) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!("Block group error: {err}"))
}

/// Helper function to convert a Rust path to a Python pathlib.Path object
pub fn path_to_py_path(py: Python<'_>, path: &Path) -> PyResult<PyObject> {
    let pathlib = PyModule::import(py, "pathlib")?;
    let path_class = pathlib.getattr("Path")?;
    let py_path = path_class.call1((path.to_str().unwrap(),))?;
    Ok(py_path.into_pyobject(py)?.into())
}

/// Helper function return sqlite query results as a list of lists of Python objects
pub fn py_query(py: Python<'_>, conn: &Connection, query: &str) -> PyResult<Vec<Vec<PyObject>>> {
    let mut stmt = conn.prepare(query).map_err(sqlite_err_to_pyerr)?;
    let column_count = stmt.column_count();
    let mut rows = Vec::new();
    let mut row_iter = stmt.query([]).map_err(sqlite_err_to_pyerr)?;

    while let Some(row) = row_iter.next().map_err(sqlite_err_to_pyerr)? {
        let mut row_data = Vec::with_capacity(column_count);
        for i in 0..column_count {
            let value: PyObject = match row.get_ref(i).map_err(sqlite_err_to_pyerr)? {
                ValueRef::Null => py.None(),
                ValueRef::Integer(i) => i.into_pyobject(py)?.into(),
                ValueRef::Real(f) => f.into_pyobject(py)?.into(),
                ValueRef::Text(s) => str::from_utf8(s)
                    .map_err(|e| PyValueError::new_err(format!("UTF-8 decode error: {e}")))?
                    .into_pyobject(py)?
                    .into(),
                ValueRef::Blob(b) => PyBytes::new(py, b).into_pyobject(py)?.into(),
            };
            row_data.push(value);
        }
        rows.push(row_data);
    }

    Ok(rows)
}
