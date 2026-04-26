use pyo3::{Bound, prelude::*, types::PyModule};

// Define modules for Python API components
pub mod block_group;
pub mod graph_search;
pub mod hash_id;
pub mod jupyter_widget;
pub mod block;
pub mod repository;
pub mod sequence_part;
pub mod utils;

// Re-export components for use in the main module
use crate::{
    PyDbContext,
    exports::{export_fasta, export_genbank, export_gfa},
    graph_operations::{derive_chunks, derive_subgraph, make_stitch},
    imports::{import_fasta, import_genbank, import_gfa, import_library, import_library_files},
    init,
    python_api::{
        block_group::PyBlockGroup,
        graph_search::{PyAnnotation, PyGraphLocus, PyGraphPos},
        hash_id::PyHashId,
        jupyter_widget::PyGraphController,
        block::PyBlock,
        repository::PyRepository,
        sequence_part::PySequencePart,
        utils::get_gen_dir_py,
    },
    updates::{
        update_with_fasta, update_with_gaf, update_with_genbank, update_with_gfa,
        update_with_library, update_with_library_files, update_with_sequence, update_with_vcf,
    },
};

/// Adds functions and classes to the Python module.
/// Remember to also add them to the __init__.py file
/// to expose them to the user.
#[pymodule]
pub fn r#gen(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDbContext>()?;
    m.add_function(wrap_pyfunction!(init, m)?)?;
    m.add_function(wrap_pyfunction!(import_fasta, m)?)?;
    m.add_function(wrap_pyfunction!(import_gfa, m)?)?;
    m.add_function(wrap_pyfunction!(import_genbank, m)?)?;
    m.add_function(wrap_pyfunction!(import_library, m)?)?;
    m.add_function(wrap_pyfunction!(import_library_files, m)?)?;
    m.add_function(wrap_pyfunction!(update_with_fasta, m)?)?;
    m.add_function(wrap_pyfunction!(update_with_gfa, m)?)?;
    m.add_function(wrap_pyfunction!(update_with_gaf, m)?)?;
    m.add_function(wrap_pyfunction!(update_with_vcf, m)?)?;
    m.add_function(wrap_pyfunction!(update_with_genbank, m)?)?;
    m.add_function(wrap_pyfunction!(update_with_library, m)?)?;
    m.add_function(wrap_pyfunction!(update_with_library_files, m)?)?;
    m.add_function(wrap_pyfunction!(update_with_sequence, m)?)?;
    m.add_function(wrap_pyfunction!(export_fasta, m)?)?;
    m.add_function(wrap_pyfunction!(export_gfa, m)?)?;
    m.add_function(wrap_pyfunction!(export_genbank, m)?)?;
    m.add_function(wrap_pyfunction!(get_gen_dir_py, m)?)?;
    m.add_function(wrap_pyfunction!(derive_chunks, m)?)?;
    m.add_function(wrap_pyfunction!(derive_subgraph, m)?)?;
    m.add_function(wrap_pyfunction!(make_stitch, m)?)?;

    m.add_class::<PyRepository>()?;
    m.add_class::<PyBlockGroup>()?;
    m.add_class::<PyHashId>()?;
    m.add_class::<PyBlock>()?;
    m.add_class::<PyGraphPos>()?;
    m.add_class::<PyGraphLocus>()?;
    m.add_class::<PyAnnotation>()?;
    m.add_class::<PySequencePart>()?;
    m.add_class::<PyGraphController>()?;

    Ok(())
}
