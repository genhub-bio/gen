use pyo3::{Bound, prelude::*, types::PyModule};

pub mod block;
pub mod block_group;
pub mod graph_search;
pub mod hash_id;
pub mod jupyter_widget;
pub mod repository;
pub mod sequence_part;
pub mod utils;

use crate::python_api::{
    block::{PyGraphNode, PyGraphNodeSlice},
    block_group::PyBlockGroup,
    graph_search::{PyAnnotation, PyAnnotationOffset, PyGraphLocus, PyGraphPos},
    hash_id::PyHashId,
    jupyter_widget::PyGraphController,
    repository::PyRepository,
    sequence_part::PySequencePart,
};

/// Adds functions and classes to the Python module.
/// Remember to also add them to the __init__.py file
/// to expose them to the user.
#[pymodule]
pub fn r#gen(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRepository>()?;
    m.add_class::<PyBlockGroup>()?;
    m.add_class::<PyHashId>()?;
    m.add_class::<PyGraphNode>()?;
    m.add_class::<PyGraphNodeSlice>()?;
    m.add_class::<PyGraphPos>()?;
    m.add_class::<PyGraphLocus>()?;
    m.add_class::<PyAnnotation>()?;
    m.add_class::<PyAnnotationOffset>()?;
    m.add_class::<PySequencePart>()?;
    m.add_class::<PyGraphController>()?;

    Ok(())
}
