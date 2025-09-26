use pyo3::{Bound, prelude::*, types::PyModule};

// Define modules for Python API components
pub mod block_group;
pub mod factory;
pub mod layouts;
pub mod node_key;
pub mod repository;
pub mod utils;

// Re-export components for use in the main module
use crate::python_api::{
    block_group::PyBlockGroup,
    layouts::{PyBaseLayout, PyScaledLayout},
    node_key::PyNodeKey,
    repository::PyRepository,
    utils::get_gen_dir_py,
};

/// Adds functions and classes to the Python module.
/// Remember to also add them to the __init__.py file
/// to expose them to the user.
#[pymodule]
pub fn r#gen(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_gen_dir_py, m)?)?;

    m.add_class::<PyRepository>()?;
    m.add_class::<PyBlockGroup>()?;
    m.add_class::<PyBaseLayout>()?;
    m.add_class::<PyScaledLayout>()?;
    m.add_class::<PyNodeKey>()?;

    Ok(())
}
