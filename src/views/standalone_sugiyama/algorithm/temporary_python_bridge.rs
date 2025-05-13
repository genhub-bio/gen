#![cfg(feature = "python-bindings")] // Entire module is conditional

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule, PyIterator, PyTuple};
use petgraph::stable_graph::{StableDiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::visit::IntoEdgeReferences;
use std::path::Path;
use std::collections::HashMap;


use super::LayoutNode;

/// Converts a Rust StableDiGraph<LayoutNode, ()> to a Python NetworkX DiGraph.
fn rust_graph_to_py_graph<'py>(
    py: Python<'py>,
    layout_graph: &StableDiGraph<LayoutNode, ()>,
) -> PyResult<Bound<'py, PyAny>> {
    let nx = py.import_bound("networkx")?;
    let nx_graph = nx.call_method0("DiGraph")?;

    for node_idx in layout_graph.node_indices() {
        let node_data = &layout_graph[node_idx];
        let node_attrs = PyDict::new_bound(py);
        // Create 'pos' tuple instead of separate 'x', 'y'
        let pos_tuple = PyTuple::new_bound(py, &[node_data.x.into_py(py), node_data.y.into_py(py)]);
        node_attrs.set_item("pos", pos_tuple)?;
        match node_data.original_id {
            Some(id) => node_attrs.set_item("original_id", id)?,
            None => node_attrs.set_item("original_id", py.None())?,
        };
        nx_graph.call_method("add_node", (node_idx.index(),), Some(&node_attrs))?;
    }
    for edge_ref in layout_graph.edge_references() {
        let source_idx = edge_ref.source().index();
        let target_idx = edge_ref.target().index();
        nx_graph.call_method1("add_edge", (source_idx, target_idx))?;
    }
    Ok(nx_graph)
}

/// Converts a Python NetworkX DiGraph (represented as PyAny) to a Rust StableDiGraph<LayoutNode, ()>.
fn py_graph_to_rust_graph(
    py: Python<'_>,
    py_graph: &Bound<'_, PyAny>,
) -> PyResult<StableDiGraph<LayoutNode, ()>> {
    let mut new_rust_graph = StableDiGraph::<LayoutNode, ()>::new();
    let mut py_to_rust_node_map: HashMap<usize, NodeIndex> = HashMap::new();

    let kwargs_nodes = PyDict::new_bound(py);
    kwargs_nodes.set_item("data", true)?;
    
    let py_nodes_view = py_graph.call_method("nodes", (), Some(&kwargs_nodes))?;
    let py_nodes_iter_obj = py.import_bound("builtins")?.call_method1("iter", (py_nodes_view,))?;
    let py_nodes_iter: &Bound<'_, PyIterator> = py_nodes_iter_obj.downcast::<PyIterator>()?;

    for item_py_any_result in py_nodes_iter {
        let py_node_and_attrs_tuple_any: Bound<'_, PyAny> = item_py_any_result?;
        let py_node_and_attrs_tuple: &Bound<'_, PyTuple> = py_node_and_attrs_tuple_any.downcast::<PyTuple>()?;

        let py_node_id_any: Bound<'_, PyAny> = py_node_and_attrs_tuple.get_item(0)?;
        let py_node_id = py_node_id_any.extract::<usize>()?;

        let py_attrs_any: Bound<'_, PyAny> = py_node_and_attrs_tuple.get_item(1)?;
        let py_attrs: &Bound<'_, PyDict> = py_attrs_any.downcast::<PyDict>()?;

        let py_pos_attr_any = py_attrs.get_item("pos")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Node missing 'pos' attribute"))?;
        let py_pos_tuple: &Bound<'_, PyTuple> = py_pos_attr_any.downcast::<PyTuple>()?;
        
        let x_any: Bound<'_, PyAny> = py_pos_tuple.get_item(0)?;
        let x = x_any.extract::<i32>()?;

        let y_any: Bound<'_, PyAny> = py_pos_tuple.get_item(1)?;
        let y = y_any.extract::<i32>()?;
        
        let original_id_item_any = py_attrs.get_item("original_id")?; 
        let original_id = match original_id_item_any {
            Some(val_any) if !val_any.is_none() => Some(val_any.extract::<usize>()?),
            _ => None, 
        };

        let layout_node = LayoutNode {
            x,
            y,
            original_id,
        };
        let rust_node_idx = new_rust_graph.add_node(layout_node);
        py_to_rust_node_map.insert(py_node_id, rust_node_idx);
    }

    let builtins = PyModule::import_bound(py, "builtins")?;
    let py_edges_view = py_graph.call_method0("edges")?;
    let py_edges_iter_obj = builtins.call_method1("iter", (py_edges_view,))?;
    let py_edges_iter: &Bound<'_, PyIterator> = py_edges_iter_obj.downcast::<PyIterator>()?;

    for item_py_any_result in py_edges_iter {
        let py_edge_tuple_any: Bound<'_, PyAny> = item_py_any_result?;
        let py_edge_tuple: &Bound<'_, PyTuple> = py_edge_tuple_any.downcast::<PyTuple>()?;

        let py_source_id_any: Bound<'_, PyAny> = py_edge_tuple.get_item(0)?;
        let py_source_id = py_source_id_any.extract::<usize>()?;

        let py_target_id_any: Bound<'_, PyAny> = py_edge_tuple.get_item(1)?;
        let py_target_id = py_target_id_any.extract::<usize>()?;

        let rust_source_idx = py_to_rust_node_map.get(&py_source_id)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Source node ID {} not found in mapped nodes", py_source_id)))?;
        let rust_target_idx = py_to_rust_node_map.get(&py_target_id)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Target node ID {} not found in mapped nodes", py_target_id)))?;
        
        new_rust_graph.add_edge(*rust_source_idx, *rust_target_idx, ());
    }

    Ok(new_rust_graph)
}

/// Calls a Python function with the layout graph, converts the Python graph it returns
/// back to a PetGraph StableDiGraph.
pub fn call_python_with_layout_graph(
    layout_graph: &StableDiGraph<LayoutNode, ()>,
    python_module_name: &str,
    python_function_name: &str,
) -> PyResult<StableDiGraph<LayoutNode, ()>> {
    Python::with_gil(|py| {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let module_path = Path::new(manifest_dir).join("python").join("prototyping");
        let module_dir_str = module_path.to_str().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Failed to convert module path to string (is it non-UTF8?).",
            )
        })?;
        let sys = py.import_bound("sys")?;
        let path = sys.getattr("path")?;
        path.call_method1("append", (module_dir_str,))?;

        // Convert Rust graph to Python graph
        let py_graph_obj = rust_graph_to_py_graph(py, layout_graph)?;

        // Import and call the target Python function
        let target_module = py.import_bound(python_module_name)?;
        let target_function = target_module.getattr(python_function_name)?;
        let returned_py_graph_obj = target_function.call1((py_graph_obj,))?;

        // Convert the returned Python graph back to Rust graph
        py_graph_to_rust_graph(py, &returned_py_graph_obj)
    })
} 