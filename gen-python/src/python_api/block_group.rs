use std::{collections::HashMap, fs, path::PathBuf};

use r#gen::{
    core::HashId,
    get_connection,
    graphs::graph_search::{GenGraphMatcher, SeedIndex, SequenceKind},
};
use gen_graph::GraphNode;
use gen_core::is_terminal;
use gen_models::{block_group::BlockGroup, node::Node};
use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyDict};

use super::{
    hash_id::PyHashId,
    jupyter_widget::{PyGraphController, build_and_display_widget},
    block::{PyBlock, PyEndBlock, PyStartBlock},
};

fn parse_sequence_kind(s: &str) -> PyResult<SequenceKind> {
    match s {
        "dna" => Ok(SequenceKind::Dna),
        "ssdna" => Ok(SequenceKind::SsDna),
        "protein" => Ok(SequenceKind::Protein),
        _ => Err(PyRuntimeError::new_err(format!(
            "Unknown sequence_kind '{s}'; use 'dna', 'ssdna', or 'protein'"
        ))),
    }
}

/// Exposes a BlockGroup to Python.
#[pyclass]
#[derive(Clone)]
pub struct PyBlockGroup {
    pub id: HashId,
    #[pyo3(get)]
    pub collection_name: String,
    #[pyo3(get)]
    pub sample_name: String,
    #[pyo3(get)]
    pub name: String,
    /// Path to the SQLite database for the Repository that created this block group.
    pub db_path: Option<PathBuf>,
}

impl PyBlockGroup {
    fn open_conn(&self, method: &str) -> PyResult<gen_models::db::GraphConnection> {
        let db_path = self.db_path.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err(format!(
                "{method}() requires a db_path; obtain BlockGroup via Repository"
            ))
        })?;
        get_connection(db_path).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn fetch_sequences(
        conn: &gen_models::db::GraphConnection,
        nodes: &[GraphNode],
    ) -> HashMap<HashId, String> {
        let ids: Vec<HashId> = nodes.iter().map(|n| n.node_id).collect();
        let seqs = Node::get_sequences_by_node_ids(conn, &ids);
        nodes
            .iter()
            .filter_map(|n| {
                // Export full underlying sequence so sequence_start/sequence_end
                // remain valid indices on reimport.
                seqs.get(&n.node_id)
                    .map(|s| (n.node_id, s.get_sequence(None::<i64>, None::<i64>)))
            })
            .collect()
    }
}

#[pymethods]
impl PyBlockGroup {
    #[new]
    pub fn new(id: HashId, collection_name: String, name: String, sample_name: String) -> Self {
        PyBlockGroup {
            id,
            collection_name,
            sample_name,
            name,
            db_path: None,
        }
    }

    #[getter]
    fn id(&self) -> PyHashId {
        PyHashId::new(self.id)
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "BlockGroup({}, {}, {}, {})",
            self.id, self.collection_name, self.sample_name, self.name
        ))
    }

    fn __hash__(&self) -> PyResult<isize> {
        // Fold the raw bytes of the HashId into a single isize using a
        // polynomial rolling hash (same approach as PyHashId).
        let mut hash: isize = 0;
        for &b in &self.id.0 {
            hash = hash.wrapping_mul(31).wrapping_add(b as isize);
        }
        Ok(hash)
    }

    fn __eq__(&self, py: Python<'_>, other: PyObject) -> PyResult<bool> {
        // Try to extract PyBlockGroup from the PyObject
        if let Ok(other_bg) = other.extract::<PyRef<PyBlockGroup>>(py) {
            Ok(self.id == other_bg.id
                && self.collection_name == other_bg.collection_name
                && self.sample_name == other_bg.sample_name
                && self.name == other_bg.name)
        } else {
            // If other is not a PyBlockGroup, they're not equal
            Ok(false)
        }
    }

    /// Plot this block group's graph as an interactive Jupyter widget.
    ///
    /// Displays the widget immediately and returns it for further use.
    /// Outside of an IPython/Jupyter environment the display call is silently
    /// skipped and only the widget is returned.
    ///
    /// Raises ``RuntimeError`` if this block group was not created via a
    /// ``Repository`` (i.e. ``db_path`` is unset).
    ///
    /// Parameters
    /// ----------
    /// rows : int, optional
    ///     Initial viewport height in terminal rows.
    /// cols : int, optional
    ///     Initial viewport width in terminal columns.
    /// detail : {"normal", "full", "minimal"}, optional
    ///     Initial level of node detail.  ``"normal"`` (default) shows
    ///     truncated labels; ``"full"`` shows complete labels; ``"minimal"``
    ///     shows the smallest representation.
    #[pyo3(signature = (rows=None, cols=None, detail=None))]
    fn plot(
        slf: &Bound<'_, PyBlockGroup>,
        rows: Option<u32>,
        cols: Option<u32>,
        detail: Option<&str>,
    ) -> PyResult<PyObject> {
        let py = slf.py();

        let (db_path, bg_id) = {
            let bg = slf.borrow();
            match bg.db_path.clone() {
                Some(p) => (p, bg.id),
                None => {
                    return Err(PyRuntimeError::new_err(
                        "plot() requires a db_path; obtain BlockGroup via Repository",
                    ));
                }
            }
        };

        let conn = get_connection(&db_path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let graph = BlockGroup::get_graph(&conn, &bg_id);
        let mut ctrl = PyGraphController::new(db_path, graph);
        ctrl.block_group_id = Some(bg_id);
        if let Some(node_detail) = detail {
            ctrl.set_detail(node_detail)?;
        }
        let ctrl = Py::new(py, ctrl)?;
        build_and_display_widget(py, ctrl, rows, cols)
    }

    /// Search for exact occurrences of `query` in this block group.
    ///
    /// Returns a list of `GraphLocus` objects. Each locus exposes:
    ///   - `.start()` / `.end()` → `GraphPos` (node + byte offset)
    ///   - `.blocks` → `list[Block]`
    ///
    /// Parameters
    /// ----------
    /// query : str
    ///     The sequence to search for.
    /// sequence_kind : {"dna", "ssdna", "protein"}, optional
    ///     Biological interpretation of the query (default: ``"dna"``).
    #[pyo3(signature = (query, sequence_kind="dna"))]
    pub fn search(
        &self,
        query: &str,
        sequence_kind: &str,
    ) -> PyResult<Vec<crate::python_api::graph_search::PyGraphLocus>> {
        let kind = parse_sequence_kind(sequence_kind)?;
        let db_path = self.db_path.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("search() requires a db_path; obtain BlockGroup via Repository")
        })?;
        let conn = get_connection(db_path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let graph = BlockGroup::get_graph(&conn, &self.id);
        let matcher = GenGraphMatcher::new_with_sequence_kind(&conn, graph, kind);

        let gen_dir = db_path.parent().ok_or_else(|| {
            PyRuntimeError::new_err("Cannot determine .gen directory from db_path")
        })?;
        let index_path = gen_dir
            .join("search_index")
            .join(format!("{}.bin", self.id));
        let case_sensitive = kind == SequenceKind::Protein;
        let index = fs::read(&index_path)
            .ok()
            .and_then(|bytes| SeedIndex::from_bytes_with_header(&bytes, 16, case_sensitive).ok());

        let query_bytes = query.as_bytes();
        let matches = match index {
            Some(idx) => matcher
                .find_all_with_seed_index(&idx, query_bytes)
                .unwrap_or_else(|_| matcher.find_all(query_bytes)),
            None => matcher.find_all(query_bytes),
        };

        Ok(matches
            .into_iter()
            .map(crate::python_api::graph_search::PyGraphLocus::from_locus)
            .collect())
    }

    /// IPython display hook — called by `display(block_group)` in Jupyter.
    fn _ipython_display_(slf: &Bound<'_, PyBlockGroup>) -> PyResult<()> {
        // plot() already calls IPython.display.display() internally; just
        // delegate and ignore errors (e.g. db_path unset, anywidget missing).
        let _ = slf.call_method0("plot");
        Ok(())
    }

    /// Build a junction-aware k-mer seed index for this block group.
    ///
    /// Saves the index to `.gen/search_index/{id}.bin` so that subsequent
    /// calls to `Repository.search()` load it automatically.
    ///
    /// Raises ``RuntimeError`` if this block group was not created via a
    /// ``Repository`` (i.e. ``db_path`` is unset).
    ///
    /// Parameters
    /// ----------
    /// sequence_kind : {"dna", "ssdna", "protein"}, optional
    ///     Biological interpretation of the sequences (default: ``"dna"``).
    /// k : int, optional
    ///     k-mer size. Defaults to 16.
    #[pyo3(signature = (sequence_kind="dna", k=16))]
    pub fn build_index(&self, sequence_kind: &str, k: usize) -> PyResult<()> {
        let kind = parse_sequence_kind(sequence_kind)?;
        let db_path = self.db_path.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err(
                "build_index() requires a db_path; obtain BlockGroup via Repository",
            )
        })?;
        let gen_dir = db_path.parent().ok_or_else(|| {
            PyRuntimeError::new_err("Cannot determine .gen directory from db_path")
        })?;
        let index_dir = gen_dir.join("search_index");
        fs::create_dir_all(&index_dir)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create index dir: {e}")))?;
        let conn = get_connection(db_path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let graph = BlockGroup::get_graph(&conn, &self.id);
        let matcher = GenGraphMatcher::new_with_sequence_kind(&conn, graph, kind);
        let index = SeedIndex::build(&matcher, k);
        let case_sensitive = kind == SequenceKind::Protein;
        let path = index_dir.join(format!("{}.bin", self.id));
        let bytes = index
            .to_bytes_with_header(case_sensitive)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to serialize index: {e}")))?;
        fs::write(&path, bytes)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to write index: {e}")))?;
        Ok(())
    }

    /// Clear the search index for this block group.
    ///
    /// Removes `.gen/search_index/{id}.bin` if it exists.
    ///
    /// Raises ``RuntimeError`` if this block group was not created via a
    /// ``Repository`` (i.e. ``db_path`` is unset).
    pub fn clear_index(&self) -> PyResult<()> {
        let db_path = self.db_path.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err(
                "clear_index() requires a db_path; obtain BlockGroup via Repository",
            )
        })?;
        let gen_dir = db_path.parent().ok_or_else(|| {
            PyRuntimeError::new_err("Cannot determine .gen directory from db_path")
        })?;
        let path = gen_dir
            .join("search_index")
            .join(format!("{}.bin", self.id));
        if path.exists() {
            fs::remove_file(&path)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to delete index: {e}")))?;
        }
        Ok(())
    }

    /// Convert this block group's graph to a plain Python dictionary.
    ///
    /// Returns ``{"nodes": {Block: {...}}, "edges": {(Block, Block): [...]}}``.
    /// Each node dict includes ``node_id`` (hex string), ``sequence_start``,
    /// ``sequence_end``, and ``sequence``.
    pub fn to_dict(&self) -> PyResult<PyObject> {
        let conn = self.open_conn("to_dict")?;
        let graph = BlockGroup::get_graph(&conn, &self.id);
        let nodes: Vec<GraphNode> = graph.nodes().filter(|n| !is_terminal(n.node_id)).collect();
        let seqs = Self::fetch_sequences(&conn, &nodes);

        Python::with_gil(|py| {
            let dict = PyDict::new(py);

            let nodes_dict = PyDict::new(py);
            for node in &nodes {
                let node_dict = PyDict::new(py);
                node_dict.set_item("node_id", node.node_id.to_string())?;
                node_dict.set_item("sequence_start", node.sequence_start)?;
                node_dict.set_item("sequence_end", node.sequence_end)?;
                if let Some(seq) = seqs.get(&node.node_id) {
                    node_dict.set_item("sequence", seq)?;
                }
                let node_key = PyBlock {
                    node_id: node.node_id,
                    sequence_start: node.sequence_start,
                    sequence_end: node.sequence_end,
                    node_seq: seqs.get(&node.node_id).cloned(),
                    block_seq: None,
                };
                nodes_dict.set_item(node_key, node_dict)?;
            }
            dict.set_item("nodes", nodes_dict)?;

            let edges_dict = PyDict::new(py);
            for (src, dst, edge_weights) in graph.all_edges() {
                let mut weights: Vec<_> = vec![];
                for weight in edge_weights {
                    let w = PyDict::new(py);
                    w.set_item("edge_id", weight.edge_id)?;
                    w.set_item("source_strand", weight.source_strand.to_string())?;
                    w.set_item("target_strand", weight.target_strand.to_string())?;
                    w.set_item("chromosome_index", weight.chromosome_index)?;
                    w.set_item("phased", weight.phased)?;
                    weights.push(w);
                }
                let src_key = PyBlock {
                    node_id: src.node_id,
                    sequence_start: src.sequence_start,
                    sequence_end: src.sequence_end,
                    node_seq: seqs.get(&src.node_id).cloned(),
                    block_seq: None,
                };
                let dst_key = PyBlock {
                    node_id: dst.node_id,
                    sequence_start: dst.sequence_start,
                    sequence_end: dst.sequence_end,
                    node_seq: seqs.get(&dst.node_id).cloned(),
                    block_seq: None,
                };
                edges_dict.set_item((src_key, dst_key), weights)?;
            }
            dict.set_item("edges", edges_dict)?;

            Ok(dict.into_pyobject(py)?.into_any().unbind())
        })
    }

    /// Convert this block group's graph to a NetworkX ``DiGraph``.
    ///
    /// Node keys are ``Block`` objects carrying all identity and sequence data
    /// as properties (``node_id``, ``sequence_start``, ``sequence_end``,
    /// ``sequence``, ``_node_sequence``).  No redundant node attributes are set.
    /// Edge attributes: ``source_strand``, ``target_strand``, ``weights``
    /// (list of per-chromosome weight dicts for full fidelity).
    ///
    /// ``StartBlock``/``EndBlock`` sentinel nodes and their edges are always
    /// included.  To work with content nodes only, filter them out with
    /// ``isinstance(node, (StartBlock, EndBlock))``.
    ///
    /// The result is compatible with
    /// ``Repository.create_block_group_from_graph()`` for a round-trip.
    pub fn to_networkx(&self) -> PyResult<PyObject> {
        let conn = self.open_conn("to_networkx")?;
        let graph = BlockGroup::get_graph(&conn, &self.id);
        let nodes: Vec<GraphNode> = graph.nodes().filter(|n| !is_terminal(n.node_id)).collect();
        let seqs = Self::fetch_sequences(&conn, &nodes);

        Python::with_gil(|py| {
            let networkx = PyModule::import(py, "networkx").map_err(|_| {
                pyo3::exceptions::PyModuleNotFoundError::new_err(
                    "The 'networkx' module is not installed. \
                     Please install it using 'pip install networkx'.",
                )
            })?;
            let nx_graph = networkx.getattr("DiGraph")?.call0()?;

            for node in &nodes {
                let node_key = PyBlock {
                    node_id: node.node_id,
                    sequence_start: node.sequence_start,
                    sequence_end: node.sequence_end,
                    node_seq: seqs.get(&node.node_id).cloned(),
                    block_seq: None,
                };
                nx_graph.call_method1("add_node", (node_key,))?;
            }

            let start_obj = Py::new(py, PyStartBlock::new())?;
            let end_obj = Py::new(py, PyEndBlock::new())?;
            nx_graph.call_method1("add_node", (start_obj,))?;
            nx_graph.call_method1("add_node", (end_obj,))?;

            for (src, dst, edge_weights) in graph.all_edges() {

                let src_key: PyObject = PyBlock {
                    node_id: src.node_id,
                    sequence_start: src.sequence_start,
                    sequence_end: src.sequence_end,
                    node_seq: if is_terminal(src.node_id) {
                        None
                    } else {
                        seqs.get(&src.node_id).cloned()
                    },
                    block_seq: None,
                }
                .into_pyobject(py)?
                .into_any()
                .unbind();

                let dst_key: PyObject = PyBlock {
                    node_id: dst.node_id,
                    sequence_start: dst.sequence_start,
                    sequence_end: dst.sequence_end,
                    node_seq: if is_terminal(dst.node_id) {
                        None
                    } else {
                        seqs.get(&dst.node_id).cloned()
                    },
                    block_seq: None,
                }
                .into_pyobject(py)?
                .into_any()
                .unbind();

                let edge_attrs = PyDict::new(py);
                let mut weights: Vec<_> = vec![];
                let mut first = true;
                for weight in edge_weights {
                    if first {
                        edge_attrs.set_item("source_strand", weight.source_strand.to_string())?;
                        edge_attrs.set_item("target_strand", weight.target_strand.to_string())?;
                        first = false;
                    }
                    let w = PyDict::new(py);
                    w.set_item("edge_id", weight.edge_id)?;
                    w.set_item("source_strand", weight.source_strand.to_string())?;
                    w.set_item("target_strand", weight.target_strand.to_string())?;
                    w.set_item("chromosome_index", weight.chromosome_index)?;
                    w.set_item("phased", weight.phased)?;
                    weights.push(w);
                }
                edge_attrs.set_item("weights", weights)?;
                nx_graph.call_method("add_edge", (src_key, dst_key), Some(&edge_attrs))?;
            }

            Ok(nx_graph.into_pyobject(py)?.into_any().unbind())
        })
    }

    /// Convert this block group's graph to a rustworkx ``PyDiGraph``.
    ///
    /// Each node payload is a dict with ``node_id`` (hex string),
    /// ``sequence_start``, ``sequence_end``, ``sequence``, and ``key``
    /// (the ``Block`` object).
    /// Each edge payload is a list of weight dicts (same fields as
    /// ``to_dict``).
    pub fn to_rustworkx(&self) -> PyResult<PyObject> {
        let conn = self.open_conn("to_rustworkx")?;
        let graph = BlockGroup::get_graph(&conn, &self.id);
        let nodes: Vec<GraphNode> = graph.nodes().filter(|n| !is_terminal(n.node_id)).collect();
        let seqs = Self::fetch_sequences(&conn, &nodes);

        Python::with_gil(|py| {
            let rustworkx = PyModule::import(py, "rustworkx").map_err(|_| {
                pyo3::exceptions::PyModuleNotFoundError::new_err(
                    "The 'rustworkx' module is not installed. \
                     Please install it using 'pip install rustworkx'.",
                )
            })?;
            let rx_graph = rustworkx.getattr("PyDiGraph")?.call0()?;

            let mut node_map: HashMap<GraphNode, usize> = HashMap::new();
            for node in &nodes {
                let node_data = PyDict::new(py);
                node_data.set_item("node_id", node.node_id.to_string())?;
                node_data.set_item("sequence_start", node.sequence_start)?;
                node_data.set_item("sequence_end", node.sequence_end)?;
                if let Some(seq) = seqs.get(&node.node_id) {
                    node_data.set_item("sequence", seq)?;
                }
                let node_key = PyBlock {
                    node_id: node.node_id,
                    sequence_start: node.sequence_start,
                    sequence_end: node.sequence_end,
                    node_seq: seqs.get(&node.node_id).cloned(),
                    block_seq: None,
                };
                node_data.set_item("key", node_key)?;
                let idx: usize = rx_graph.call_method1("add_node", (node_data,))?.extract()?;
                node_map.insert(*node, idx);
            }

            for (src, dst, edge_weights) in graph.all_edges() {
                let src_idx = *node_map.get(&src).unwrap();
                let dst_idx = *node_map.get(&dst).unwrap();
                let mut weights: Vec<_> = vec![];
                for weight in edge_weights {
                    let w = PyDict::new(py);
                    w.set_item("edge_id", weight.edge_id)?;
                    w.set_item("source_strand", weight.source_strand.to_string())?;
                    w.set_item("target_strand", weight.target_strand.to_string())?;
                    w.set_item("chromosome_index", weight.chromosome_index)?;
                    w.set_item("phased", weight.phased)?;
                    weights.push(w);
                }
                rx_graph.call_method1("add_edge", (src_idx, dst_idx, weights))?;
            }

            Ok(rx_graph.into_pyobject(py)?.into_any().unbind())
        })
    }
}
