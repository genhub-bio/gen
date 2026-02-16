use pyo3::{exceptions::PyRuntimeError, prelude::*};

use crate::PyDbContext;

#[pyfunction]
pub fn derive_chunks(
    context: PyRef<'_, PyDbContext>,
    name: Option<String>,
    sample: Option<String>,
    new_sample: String,
    region: String,
    backbone: Option<String>,
    breakpoints: Option<String>,
    chunk_size: Option<i64>,
) -> PyResult<()> {
    let db_context = &context.0;

    match r#gen::commands::graph_operations::derive_chunks::derive_chunks_operation(
        &db_context,
        name,
        sample,
        new_sample,
        region,
        backbone,
        breakpoints,
        chunk_size,
    ) {
        Ok(_) => Ok(()),
        Err(e) => Err(PyRuntimeError::new_err(format!(
            "Error deriving chunks: {e}"
        ))),
    }
}

#[pyfunction]
pub fn derive_subgraph(
    context: PyRef<'_, PyDbContext>,
    name: Option<String>,
    sample: Option<String>,
    new_sample: String,
    region: String,
    backbone: Option<String>,
) -> PyResult<()> {
    let db_context = &context.0;

    match r#gen::commands::graph_operations::derive_subgraph::derive_subgraph_operation(
        &db_context,
        name,
        sample,
        new_sample,
        region,
        backbone,
    ) {
        Ok(_) => Ok(()),
        Err(e) => Err(PyRuntimeError::new_err(format!(
            "Error deriving subgraph: {e}"
        ))),
    }
}

#[pyfunction]
pub fn make_stitch(
    context: PyRef<'_, PyDbContext>,
    name: Option<String>,
    sample: Option<String>,
    new_sample: String,
    regions: String,
    new_region: String,
) -> PyResult<()> {
    let db_context = &context.0;

    match r#gen::commands::graph_operations::make_stitch::make_stitch_operation(
        &db_context,
        name,
        sample,
        new_sample,
        regions,
        new_region,
    ) {
        Ok(_) => Ok(()),
        Err(e) => Err(PyRuntimeError::new_err(format!("Error making stitch: {e}"))),
    }
}
