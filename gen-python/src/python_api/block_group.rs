use std::{collections::HashMap, fs, path::PathBuf};

use r#gen::{
    commands::graph_operations::{
        derive_chunks::derive_chunks_operation, derive_subgraph::derive_subgraph_operation,
    },
    core::HashId,
    exports::{fasta::export_fasta, genbank::export_genbank, gfa::export_gfa},
    graphs::graph_search::{GenGraphMatcher, SeedIndex, SequenceKind},
};
use gen_graph::GraphNode;
use gen_models::{block_group::BlockGroup, db::DbContext, sample::Sample};
use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyDict};

use super::{
    block::PyBlock,
    hash_id::PyHashId,
    jupyter_widget::{PyGraphController, build_widget},
};

pub(crate) fn parse_sequence_kind(s: &str) -> PyResult<SequenceKind> {
    match s {
        "exact" => Ok(SequenceKind::Exact),
        "dna" => Ok(SequenceKind::Dna),
        "ssdna" => Ok(SequenceKind::SsDna),
        "protein" => Ok(SequenceKind::Protein),
        _ => Err(PyRuntimeError::new_err(format!(
            "Unknown sequence_kind '{s}'; use 'exact', 'dna', 'ssdna', or 'protein'"
        ))),
    }
}

/// A block group returned by a ``Repository``.
///
/// ``BlockGroup`` objects cannot be shared across threads because they hold a
/// reference to the database connection of the ``Repository`` that created them.
///
/// To use a block group's identity in another thread, capture its ``id``,
/// open a fresh ``Repository`` in that thread, and look it up by id::
///
///     bg_id = bg.id
///     path  = str(repo.db_path)
///
///     def worker():
///         r = gen.Repository(path)
///         bg = r.get_block_group_by_id(bg_id)
///         ...
// unsendable because DbContext contains Rc (rusqlite::Connection is !Sync)
#[pyclass(name = "BlockGroup", unsendable)]
#[derive(Clone)]
pub struct PyBlockGroup {
    pub id: HashId,
    #[pyo3(get)]
    pub collection_name: String,
    #[pyo3(get)]
    pub sample_name: String,
    #[pyo3(get)]
    pub name: String,
    pub context: Option<DbContext>,
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
            context: None,
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
        let mut hash: isize = 0;
        for &b in &self.id.0 {
            hash = hash.wrapping_mul(31).wrapping_add(b as isize);
        }
        Ok(hash)
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        if let Ok(other_bg) = other.extract::<PyRef<PyBlockGroup>>() {
            Ok(self.id == other_bg.id
                && self.collection_name == other_bg.collection_name
                && self.sample_name == other_bg.sample_name
                && self.name == other_bg.name)
        } else {
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
    /// ``Repository``.
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

        let (context, bg_id) = {
            let bg = slf.borrow();
            match bg.context.clone() {
                Some(ctx) => (ctx, bg.id),
                None => {
                    return Err(PyRuntimeError::new_err(
                        "plot() requires a Repository context; obtain BlockGroups via Repository by query or id.",
                    ));
                }
            }
        };

        let graph_conn = context.graph().conn();
        let db_path = graph_conn
            .path()
            .map(std::path::PathBuf::from)
            .ok_or_else(|| PyRuntimeError::new_err("graph DB has no file path"))?;
        let graph = BlockGroup::get_graph(graph_conn, &bg_id);
        let mut ctrl = PyGraphController::new(db_path, graph);
        ctrl.block_group_id = Some(bg_id);
        ctrl.auto_load_annotation_groups(graph_conn);
        if let Some(node_detail) = detail {
            ctrl.set_detail(node_detail)?;
        }
        let ctrl = Py::new(py, ctrl)?;
        build_widget(py, ctrl, rows, cols)
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
    /// sequence_kind : {"exact", "dna", "ssdna", "protein"}, optional
    ///     Biological interpretation of the query (default: ``"dna"``).
    ///     ``"exact"`` performs case-sensitive raw-byte matching with no IUPAC
    ///     expansion and no reverse complement.
    #[pyo3(signature = (query, sequence_kind="dna"))]
    pub fn search(
        &self,
        query: &str,
        sequence_kind: &str,
    ) -> PyResult<Vec<crate::python_api::graph_search::PyGraphLocus>> {
        let kind = parse_sequence_kind(sequence_kind)?;
        let context = self.context.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err(
                "search() requires a Repository context; obtain BlockGroup via Repository",
            )
        })?;
        let conn = context.graph().conn();
        let graph = BlockGroup::get_graph(conn, &self.id);
        let matcher = GenGraphMatcher::new_with_sequence_kind(conn, graph, kind);

        let gen_dir = context.workspace().ensure_gen_dir();
        let index_path = gen_dir
            .join("search_index")
            .join(format!("{}.bin", self.id));
        let index = fs::read(&index_path)
            .ok()
            .and_then(|bytes| SeedIndex::from_bytes_with_header(&bytes, 16).ok());

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

    /// IPython display hook — called when a cell ends with a BlockGroup.
    fn _ipython_display_(slf: &Bound<'_, PyBlockGroup>) -> PyResult<()> {
        let py = slf.py();
        let widget = slf.call_method0("plot")?;
        PyModule::import(py, "IPython.display")?.call_method1("display", (widget,))?;
        Ok(())
    }

    /// Build a junction-aware k-mer seed index for this block group.
    ///
    /// Saves the index to `.gen/search_index/{id}.bin` so that subsequent
    /// calls to `Repository.search()` load it automatically.
    ///
    /// Raises ``RuntimeError`` if this block group was not created via a
    /// ``Repository``.
    ///
    /// Parameters
    /// ----------
    /// sequence_kind : {"exact", "dna", "ssdna", "protein"}, optional
    ///     Biological interpretation of the sequences (default: ``"dna"``).
    ///     ``"exact"`` builds a case-sensitive index; must match the value
    ///     used when calling ``search()``.
    /// k : int, optional
    ///     k-mer size. Defaults to 16.
    #[pyo3(signature = (sequence_kind="dna", k=16))]
    pub fn build_index(&self, sequence_kind: &str, k: usize) -> PyResult<()> {
        let kind = parse_sequence_kind(sequence_kind)?;
        let context = self.context.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err(
                "build_index() requires a Repository context; obtain BlockGroup via Repository",
            )
        })?;
        let gen_dir = context.workspace().ensure_gen_dir();
        let index_dir = gen_dir.join("search_index");
        fs::create_dir_all(&index_dir)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create index dir: {e}")))?;
        let conn = context.graph().conn();
        let graph = BlockGroup::get_graph(conn, &self.id);
        let matcher = GenGraphMatcher::new_with_sequence_kind(conn, graph, kind);
        let normalized = kind != SequenceKind::Exact;
        let index = SeedIndex::build(&matcher, k, normalized);
        let path = index_dir.join(format!("{}.bin", self.id));
        let bytes = index
            .to_bytes_with_header()
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
    /// ``Repository``.
    pub fn clear_index(&self) -> PyResult<()> {
        let context = self.context.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err(
                "clear_index() requires a Repository context; obtain BlockGroup via Repository",
            )
        })?;
        let gen_dir = context.workspace().ensure_gen_dir();
        let path = gen_dir
            .join("search_index")
            .join(format!("{}.bin", self.id));
        if path.exists() {
            fs::remove_file(&path)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to delete index: {e}")))?;
        }
        Ok(())
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let conn = self.require_context("to_dict()")?.graph().conn();
        let graph = BlockGroup::get_graph(conn, &self.id);
        let dict = PyDict::new(py);
        let nodes = PyDict::new(py);
        for node in graph.nodes() {
            let node_dict = PyDict::new(py);
            node_dict.set_item("node_id", node.node_id)?;
            node_dict.set_item("sequence_start", node.sequence_start)?;
            node_dict.set_item("sequence_end", node.sequence_end)?;
            nodes.set_item(
                PyBlock::new(node.node_id, node.sequence_start, node.sequence_end),
                node_dict,
            )?;
        }
        dict.set_item("nodes", nodes)?;
        let edges = PyDict::new(py);
        for (src, dst, edge_weights) in graph.all_edges() {
            let weights: PyResult<Vec<_>> = edge_weights
                .iter()
                .map(|w| {
                    let d = PyDict::new(py);
                    d.set_item("edge_id", w.edge_id)?;
                    d.set_item("source_strand", w.source_strand.to_string())?;
                    d.set_item("target_strand", w.target_strand.to_string())?;
                    d.set_item("chromosome_index", w.chromosome_index)?;
                    d.set_item("phased", w.phased)?;
                    Ok(d)
                })
                .collect();
            edges.set_item(
                (
                    PyBlock::new(src.node_id, src.sequence_start, src.sequence_end),
                    PyBlock::new(dst.node_id, dst.sequence_start, dst.sequence_end),
                ),
                weights?,
            )?;
        }
        dict.set_item("edges", edges)?;
        Ok(dict.into_pyobject(py)?.into_any().unbind())
    }

    fn to_rustworkx(&self, py: Python<'_>) -> PyResult<PyObject> {
        let conn = self.require_context("to_rustworkx()")?.graph().conn();
        let graph = BlockGroup::get_graph(conn, &self.id);
        {
            let rustworkx = PyModule::import(py, "rustworkx").map_err(|_| {
                pyo3::exceptions::PyModuleNotFoundError::new_err(
                    "rustworkx is not installed. Run: pip install rustworkx",
                )
            })?;
            let py_digraph = rustworkx.getattr("PyDiGraph")?.call0()?;
            let mut node_map: HashMap<GraphNode, usize> = HashMap::new();
            for node in graph.nodes() {
                let node_data = PyDict::new(py);
                node_data.set_item("node_id", node.node_id)?;
                node_data.set_item("sequence_start", node.sequence_start)?;
                node_data.set_item("sequence_end", node.sequence_end)?;
                node_data.set_item(
                    "key",
                    PyBlock::new(node.node_id, node.sequence_start, node.sequence_end),
                )?;
                let index: usize = py_digraph
                    .call_method1("add_node", (node_data,))?
                    .extract()?;
                node_map.insert(node, index);
            }
            for (src, dst, edge_weights) in graph.all_edges() {
                let weights: PyResult<Vec<_>> = edge_weights
                    .iter()
                    .map(|w| {
                        let d = PyDict::new(py);
                        d.set_item("edge_id", w.edge_id)?;
                        d.set_item("source_strand", w.source_strand.to_string())?;
                        d.set_item("target_strand", w.target_strand.to_string())?;
                        d.set_item("chromosome_index", w.chromosome_index)?;
                        d.set_item("phased", w.phased)?;
                        Ok(d)
                    })
                    .collect();
                py_digraph.call_method1("add_edge", (node_map[&src], node_map[&dst], weights?))?;
            }
            Ok(py_digraph.into_pyobject(py)?.into_any().unbind())
        }
    }

    fn to_networkx(&self, py: Python<'_>) -> PyResult<PyObject> {
        let conn = self.require_context("to_networkx()")?.graph().conn();
        let graph = BlockGroup::get_graph(conn, &self.id);
        {
            let networkx = PyModule::import(py, "networkx").map_err(|_| {
                pyo3::exceptions::PyModuleNotFoundError::new_err(
                    "networkx is not installed. Run: pip install networkx",
                )
            })?;
            let nx_digraph = networkx.getattr("DiGraph")?.call0()?;
            for node in graph.nodes() {
                let node_data = PyDict::new(py);
                node_data.set_item("node_id", node.node_id)?;
                node_data.set_item("sequence_start", node.sequence_start)?;
                node_data.set_item("sequence_end", node.sequence_end)?;
                let kwargs = PyDict::new(py);
                kwargs.set_item("attr_dict", node_data)?;
                nx_digraph.call_method(
                    "add_node",
                    (PyBlock::new(
                        node.node_id,
                        node.sequence_start,
                        node.sequence_end,
                    ),),
                    Some(&kwargs),
                )?;
            }
            for (src, dst, edge_weights) in graph.all_edges() {
                let weights: PyResult<Vec<_>> = edge_weights
                    .iter()
                    .map(|w| {
                        let d = PyDict::new(py);
                        d.set_item("edge_id", w.edge_id)?;
                        d.set_item("source_strand", w.source_strand.to_string())?;
                        d.set_item("target_strand", w.target_strand.to_string())?;
                        d.set_item("chromosome_index", w.chromosome_index)?;
                        d.set_item("phased", w.phased)?;
                        Ok(d)
                    })
                    .collect();
                let kwargs = PyDict::new(py);
                kwargs.set_item("attr_dict", weights?)?;
                nx_digraph.call_method(
                    "add_edge",
                    (
                        PyBlock::new(src.node_id, src.sequence_start, src.sequence_end),
                        PyBlock::new(dst.node_id, dst.sequence_start, dst.sequence_end),
                    ),
                    Some(&kwargs),
                )?;
            }
            Ok(nx_digraph.into_pyobject(py)?.into_any().unbind())
        }
    }
}

#[pymethods]
impl PyBlockGroup {
    /// Export all block groups in this block group's sample to FASTA.
    ///
    /// Parameters
    /// ----------
    /// filename : str
    ///     Output file path.
    fn export_fasta(&self, filename: String) -> PyResult<()> {
        let ctx = self.require_context("export_fasta()")?;
        let conn = ctx.graph().conn();
        export_fasta(
            conn,
            &self.collection_name,
            Some(&self.sample_name),
            &PathBuf::from(&filename),
        )
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to export FASTA '{}': {e}", filename)))
    }

    /// Export all block groups in this block group's sample to GFA.
    ///
    /// Parameters
    /// ----------
    /// filename : str
    ///     Output file path.
    /// node_max : int, optional
    ///     Maximum node sequence length before splitting.
    #[pyo3(signature = (filename, node_max=None))]
    fn export_gfa(&self, filename: String, node_max: Option<i64>) -> PyResult<()> {
        let ctx = self.require_context("export_gfa()")?;
        let conn = ctx.graph().conn();
        export_gfa(
            conn,
            &self.collection_name,
            &PathBuf::from(&filename),
            &self.sample_name,
            node_max,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to export GFA '{}': {e}", filename)))
    }

    /// Export all block groups in this block group's sample to GenBank.
    ///
    /// Parameters
    /// ----------
    /// filename : str
    ///     Output file path.
    fn export_genbank(&self, filename: String) -> PyResult<()> {
        let ctx = self.require_context("export_genbank()")?;
        let conn = ctx.graph().conn();
        let writer = fs::File::create(&filename).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to create '{}': {e}", filename))
        })?;
        export_genbank(conn, &self.collection_name, &self.sample_name, writer).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to export GenBank '{}': {e}", filename))
        })
    }
}

impl PyBlockGroup {
    fn require_context(&self, method: &str) -> PyResult<&DbContext> {
        self.context.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err(format!(
                "{method} requires a Repository context; obtain BlockGroup via Repository"
            ))
        })
    }

    /// Wraps a raw `BlockGroup` model with this block group's database context.
    fn into_py_block_group(&self, bg: BlockGroup) -> Self {
        PyBlockGroup {
            id: bg.id,
            collection_name: bg.collection_name,
            sample_name: bg.sample_name,
            name: bg.name,
            context: self.context.clone(),
        }
    }
}

#[pymethods]
impl PyBlockGroup {
    /// Derive a coordinate-bounded subgraph from this block group.
    ///
    /// Parameters
    /// ----------
    /// new_sample : str
    ///     Sample name for the derived block group.
    /// start : int
    ///     Start coordinate along the path (inclusive).
    /// end : int
    ///     End coordinate along the path (exclusive).
    /// backbone : str, optional
    ///     Named path to use as the coordinate backbone.
    #[pyo3(signature = (new_sample, start, end, backbone=None))]
    fn subgraph(
        &self,
        new_sample: String,
        start: i64,
        end: i64,
        backbone: Option<String>,
    ) -> PyResult<PyBlockGroup> {
        let ctx = self.require_context("subgraph()")?;
        let region = format!("{}:{}-{}", self.name, start, end);
        derive_subgraph_operation(
            ctx,
            Some(self.collection_name.clone()),
            self.sample_name.clone(),
            new_sample.clone(),
            region,
            backbone,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("Error deriving subgraph: {e}")))?;
        let conn = ctx.graph().conn();
        let child_id = BlockGroup::get_id(
            &self.collection_name,
            &new_sample,
            &self.name,
            Some(&self.id),
        );
        let found = BlockGroup::get_by_id(conn, &child_id)
            .map_err(|e| PyRuntimeError::new_err(format!("Subgraph created but not found: {e}")))?;
        Ok(self.into_py_block_group(found))
    }

    /// Split this block group into coordinate-bounded subgraphs.
    ///
    /// Parameters
    /// ----------
    /// new_sample : str
    ///     Sample name for the derived block groups.
    /// breakpoints : str, optional
    ///     Comma-separated coordinate values at which to split.
    /// chunk_size : int, optional
    ///     Split into equal-length pieces of this many bases.
    /// backbone : str, optional
    ///     Named path to use as the coordinate backbone.
    #[pyo3(signature = (new_sample, breakpoints=None, chunk_size=None, backbone=None))]
    fn chunks(
        &self,
        new_sample: String,
        breakpoints: Option<Vec<i64>>,
        chunk_size: Option<i64>,
        backbone: Option<String>,
    ) -> PyResult<Vec<PyBlockGroup>> {
        let ctx = self.require_context("chunks()")?;
        derive_chunks_operation(
            ctx,
            Some(self.collection_name.clone()),
            self.sample_name.clone(),
            new_sample.clone(),
            self.name.clone(),
            backbone,
            breakpoints,
            chunk_size,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("Error deriving chunks: {e}")))?;
        let conn = ctx.graph().conn();
        let prefix = format!("{}.", self.name);
        Ok(
            Sample::get_block_groups(conn, &self.collection_name, &new_sample)
                .into_iter()
                .filter(|bg| bg.name == self.name || bg.name.starts_with(&prefix))
                .map(|bg| self.into_py_block_group(bg))
                .collect(),
        )
    }
}
