use std::{
    path::{Path, PathBuf, absolute},
    str,
};

use gen_models::block_group::BlockGroupError;
use pyo3::{
    exceptions::{PyOSError, PyValueError},
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

/// Resolves a file path given by the Python caller against the process's current working
/// directory.
///
/// Below the bindings, some loaders resolve a relative path against the workspace and others
/// against the current directory (several do both in turn, so a relative path only works if the
/// file exists in both places). Making every file argument absolute here gives all of them the
/// same meaning as Python's own `open()`.
pub fn absolute_path(path: impl AsRef<Path>) -> PyResult<PathBuf> {
    let path = path.as_ref();
    absolute(path).map_err(|error| {
        PyOSError::new_err(format!("Cannot resolve '{}': {error}", path.display()))
    })
}

/// [`absolute_path`] for the APIs that take file names as strings.
pub fn absolute_path_string(path: &str) -> PyResult<String> {
    absolute_path(path)?
        .into_os_string()
        .into_string()
        .map_err(|_| PyValueError::new_err(format!("'{path}' must be valid UTF-8")))
}

/// Helper function to convert a Rust path to a Python pathlib.Path object
pub fn path_to_py_path(py: Python<'_>, path: &Path) -> PyResult<Py<PyAny>> {
    let pathlib = PyModule::import(py, "pathlib")?;
    let path_class = pathlib.getattr("Path")?;
    let py_path = path_class.call1((path.to_str().unwrap(),))?;
    Ok(py_path.into_pyobject(py)?.into())
}

/// Helper function return sqlite query results as a list of lists of Python objects
pub fn py_query(py: Python<'_>, conn: &Connection, query: &str) -> PyResult<Vec<Vec<Py<PyAny>>>> {
    let mut stmt = conn.prepare(query).map_err(sqlite_err_to_pyerr)?;
    let column_count = stmt.column_count();
    let mut rows = Vec::new();
    let mut row_iter = stmt.query([]).map_err(sqlite_err_to_pyerr)?;

    while let Some(row) = row_iter.next().map_err(sqlite_err_to_pyerr)? {
        let mut row_data = Vec::with_capacity(column_count);
        for i in 0..column_count {
            let value: Py<PyAny> = match row.get_ref(i).map_err(sqlite_err_to_pyerr)? {
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

#[cfg(test)]
mod tests {
    use std::{env::current_dir, path::Path};

    use super::{absolute_path, absolute_path_string};

    #[test]
    fn test_relative_paths_resolve_against_the_current_directory() {
        let current = current_dir().expect("should have a current directory");
        assert_eq!(
            absolute_path("data/parts.fa").expect("should resolve"),
            current.join("data/parts.fa")
        );
        assert_eq!(
            absolute_path_string("parts.fa").expect("should resolve"),
            current.join("parts.fa").to_string_lossy()
        );
    }

    #[test]
    fn test_absolute_paths_are_unchanged() {
        let absolute = current_dir()
            .expect("should have a current directory")
            .join("parts.fa");
        assert_eq!(
            absolute_path(&absolute).expect("should resolve"),
            Path::new(&absolute)
        );
    }
}
