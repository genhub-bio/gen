use std::{collections::HashMap, fs, path::PathBuf};

use r#gen::{
    commands::graph_operations::{
        derive_chunks::derive_chunks_operation, derive_subgraph::derive_subgraph_operation,
    },
    core::HashId,
    exports::{fasta::export_fasta, genbank::export_genbank, gfa::export_gfa},
    graphs::{
        graph_search::{GenGraphMatcher, SeedIndex, SequenceKind},
        translation::{
            translate_annotation, translate_block_group, translate_from_path,
            with_translation_operation,
        },
    },
};
use gen_annotations::projection::annotation_segments;
use gen_graph::GraphNode;
use gen_models::{
    annotations::Annotation, block_group::BlockGroup, db::DbContext, node::Node, sample::Sample,
};
use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyDict};

use super::{
    annotation::PyAnnotation,
    graph_node::PyGraphNode,
    hash_id::PyHashId,
    jupyter_widget::{PyGraphController, build_widget},
    translation::build_translation_params,
    utils::block_group_err_to_pyerr,
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

/// A sequence graph returned by a ``Repository``.
///
/// ``SequenceGraph`` objects cannot be shared across threads because they hold a
/// reference to the database connection of the ``Repository`` that created them.
///
/// To use a sequence graph's identity in another thread, capture its ``id``,
/// open a fresh ``Repository`` in that thread, and look it up by id::
///
///     sg_id = sg.id
///     path  = str(repo.db_path)
///
///     def worker():
///         r = gen.Repository(path)
///         sg = r.get_sequence_graph_by_id(sg_id)
///         ...
// unsendable because DbContext contains Rc (rusqlite::Connection is !Sync)
#[pyclass(name = "SequenceGraph", unsendable)]
#[derive(Clone)]
pub struct PySequenceGraph {
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
impl PySequenceGraph {
    #[new]
    pub fn new(id: HashId, collection_name: String, name: String, sample_name: String) -> Self {
        PySequenceGraph {
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
            "SequenceGraph({}, collection={:?}, sample={:?}, name={:?})",
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
        if let Ok(other_bg) = other.extract::<PyRef<PySequenceGraph>>() {
            Ok(self.id == other_bg.id
                && self.collection_name == other_bg.collection_name
                && self.sample_name == other_bg.sample_name
                && self.name == other_bg.name)
        } else {
            Ok(false)
        }
    }

    /// Plot this sequence graph as an interactive Jupyter widget.
    ///
    /// Displays the widget immediately and returns it for further use.
    /// Outside of an IPython/Jupyter environment the display call is silently
    /// skipped and only the widget is returned.
    ///
    /// Raises ``RuntimeError`` if this sequence graph was not created via a
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
        slf: &Bound<'_, PySequenceGraph>,
        rows: Option<u32>,
        cols: Option<u32>,
        detail: Option<&str>,
    ) -> PyResult<PyObject> {
        let py = slf.py();
        let mut ctrl = PyGraphController::for_sequence_graph(&slf.borrow())?;
        if let Some(node_detail) = detail {
            ctrl.set_detail(node_detail)?;
        }
        let ctrl = Py::new(py, ctrl)?;
        build_widget(py, ctrl, rows, cols)
    }

    /// Search for exact occurrences of `query` in this sequence graph.
    ///
    /// Returns a list of `Locus` objects. Each locus exposes:
    ///   - `.start()` / `.end()` → `Position` (node + byte offset)
    ///   - `.slices` → `list[NodeSlice]`
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
                "search() requires a Repository context; obtain SequenceGraph via Repository",
            )
        })?;
        let conn = context.graph().conn();
        let graph = BlockGroup::get_graph(conn, &self.id).map_err(block_group_err_to_pyerr)?;
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

    /// IPython display hook — called when a cell ends with a SequenceGraph.
    fn _ipython_display_(slf: &Bound<'_, PySequenceGraph>) -> PyResult<()> {
        let py = slf.py();
        let widget = slf.call_method0("plot")?;
        PyModule::import(py, "IPython.display")?.call_method1("display", (widget,))?;
        Ok(())
    }

    /// Build a junction-aware k-mer seed index for this sequence graph.
    ///
    /// Saves the index to `.gen/search_index/{id}.bin` so that subsequent
    /// calls to `Repository.search()` load it automatically.
    ///
    /// Raises ``RuntimeError`` if this sequence graph was not created via a
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
                "build_index() requires a Repository context; obtain SequenceGraph via Repository",
            )
        })?;
        let gen_dir = context.workspace().ensure_gen_dir();
        let index_dir = gen_dir.join("search_index");
        fs::create_dir_all(&index_dir)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create index dir: {e}")))?;
        let conn = context.graph().conn();
        let graph = BlockGroup::get_graph(conn, &self.id).map_err(block_group_err_to_pyerr)?;
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

    /// Clear the search index for this sequence graph.
    ///
    /// Removes `.gen/search_index/{id}.bin` if it exists.
    ///
    /// Raises ``RuntimeError`` if this sequence graph was not created via a
    /// ``Repository``.
    pub fn clear_index(&self) -> PyResult<()> {
        let context = self.context.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err(
                "clear_index() requires a Repository context; obtain SequenceGraph via Repository",
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

    /// Return the sequence for a graph node.
    ///
    /// Parameters
    /// ----------
    /// node : Node
    ///     A ``Node`` obtained from ``to_dict()["nodes"]``, ``search()`` results,
    ///     or any other API that returns graph nodes.
    ///
    /// Raises ``RuntimeError`` if this sequence graph was not created via a
    /// ``Repository``.
    fn get_node_sequence(&self, node: &PyGraphNode) -> PyResult<String> {
        let conn = self.require_context("get_node_sequence()")?.graph().conn();
        let sequences = Node::get_sequences_by_node_ids(conn, &[node.node_id]);
        let sequence = sequences.get(&node.node_id).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Node with id {:?} not found",
                node.node_id
            ))
        })?;
        sequence
            .get_sequence(node.sequence_start, node.sequence_end)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let conn = self.require_context("to_dict()")?.graph().conn();
        let graph = BlockGroup::get_graph(conn, &self.id).map_err(block_group_err_to_pyerr)?;
        let dict = PyDict::new(py);
        let nodes: Vec<PyGraphNode> = graph
            .nodes()
            .map(|node| PyGraphNode::new(node.node_id, node.sequence_start, node.sequence_end))
            .collect();
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
                    PyGraphNode::new(src.node_id, src.sequence_start, src.sequence_end),
                    PyGraphNode::new(dst.node_id, dst.sequence_start, dst.sequence_end),
                ),
                weights?,
            )?;
        }
        dict.set_item("edges", edges)?;
        Ok(dict.into_pyobject(py)?.into_any().unbind())
    }

    fn to_rustworkx(&self, py: Python<'_>) -> PyResult<PyObject> {
        let conn = self.require_context("to_rustworkx()")?.graph().conn();
        let graph = BlockGroup::get_graph(conn, &self.id).map_err(block_group_err_to_pyerr)?;
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
                node_data.set_item(
                    "key",
                    PyGraphNode::new(node.node_id, node.sequence_start, node.sequence_end),
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
        let graph = BlockGroup::get_graph(conn, &self.id).map_err(block_group_err_to_pyerr)?;
        {
            let networkx = PyModule::import(py, "networkx").map_err(|_| {
                pyo3::exceptions::PyModuleNotFoundError::new_err(
                    "networkx is not installed. Run: pip install networkx",
                )
            })?;
            let nx_digraph = networkx.getattr("DiGraph")?.call0()?;
            for node in graph.nodes() {
                nx_digraph.call_method(
                    "add_node",
                    (PyGraphNode::new(
                        node.node_id,
                        node.sequence_start,
                        node.sequence_end,
                    ),),
                    None,
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
                        PyGraphNode::new(src.node_id, src.sequence_start, src.sequence_end),
                        PyGraphNode::new(dst.node_id, dst.sequence_start, dst.sequence_end),
                    ),
                    Some(&kwargs),
                )?;
            }
            Ok(nx_digraph.into_pyobject(py)?.into_any().unbind())
        }
    }
}

#[pymethods]
impl PySequenceGraph {
    /// Export all sequence graphs in this sequence graph's sample to FASTA.
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

    /// Export all sequence graphs in this sequence graph's sample to GFA.
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

    /// Export all sequence graphs in this sequence graph's sample to GenBank.
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

    /// Return all gene annotations associated with this sequence graph.
    ///
    /// Returns
    /// -------
    /// list[GeneAnnotation]
    fn list_annotations(&self) -> PyResult<Vec<PyAnnotation>> {
        let ctx = self.require_context("list_annotations()")?;
        let conn = ctx.graph().conn();
        let bg_id = self.id;
        let annotations = Annotation::query_with_lineage(
            conn,
            &self.collection_name,
            &self.sample_name,
            &self.name,
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(annotations
            .into_iter()
            .map(|a| PyAnnotation {
                ann_segments: annotation_segments(conn, &a),
                inner: a,
                context: Some(ctx.clone()),
                source_block_group_id: Some(bg_id),
                locus: None,
            })
            .collect())
    }

    /// Translate a sequence graph or annotation into a protein ``SequenceGraph``.
    ///
    /// When ``region`` is a string it is resolved against this sequence graph
    /// only, in priority order: named path within this graph first, then
    /// annotation in this graph's lineage. No other sequence graphs are
    /// searched. The protein sequence graph is created in this graph's sample.
    ///
    /// Parameters
    /// ----------
    /// region : str or Annotation, optional
    ///     - ``str``: a path name or annotation name scoped to this sequence
    ///       graph. Path names take priority over annotation names.
    ///     - ``Annotation``: an object returned by ``list_annotations()``.
    ///       Identified by database id, so unambiguous.
    ///     - omitted: translates the entire sequence graph.
    /// start : int, optional
    ///     0-based path-space coordinate to translate from. Defaults to 0 (the
    ///     start of the path) when omitted, and is ignored when ``region`` names
    ///     an annotation (the annotation's own entry point is used instead).
    ///     Translation reads forward from this coordinate to its own first
    ///     in-frame stop codon; it is not bounded by any end coordinate.
    /// output_collection : str, optional
    ///     Collection for the protein sequence graph. Defaults to this graph's collection.
    /// name : str, optional
    ///     Name for the protein sequence graph. Defaults to "{region} (protein)".
    /// strand : str, optional
    ///     ``"forward"`` or ``"reverse"``. Inferred from the annotation when omitted.
    /// frame : int
    ///     Initial reading frame offset (0, 1, or 2). Default: 0.
    /// codon_table : int
    ///     NCBI codon table ID. Default: 1 (Standard).
    #[pyo3(signature = (region=None, output_collection=None, name=None, strand=None, frame=0, codon_table=1, start=None))]
    #[expect(clippy::too_many_arguments, reason = "mirrors underlying API")]
    fn translate_annotation(
        &self,
        region: Option<Bound<'_, PyAny>>,
        output_collection: Option<&str>,
        name: Option<&str>,
        strand: Option<&str>,
        frame: u8,
        codon_table: u8,
        start: Option<i64>,
    ) -> PyResult<PySequenceGraph> {
        let ctx = self.require_context("translate_annotation()")?;
        let conn = ctx.graph().conn();

        let out_collection = output_collection.unwrap_or(&self.collection_name);

        let params = build_translation_params(out_collection, name, strand, frame, codon_table)?;

        let protein_bg = match region {
            None => {
                let label = self.name.clone();
                let bg_id = self.id;
                if let Some(start) = start {
                    with_translation_operation(ctx, &label, || {
                        translate_from_path(conn, &bg_id, start, params)
                    })
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                } else {
                    with_translation_operation(ctx, &label, || {
                        translate_block_group(conn, &bg_id, params)
                    })
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                }
            }
            Some(region) => {
                if let Ok(ann) = region.extract::<PyRef<PyAnnotation>>() {
                    let annotation = ann.inner.clone();
                    with_translation_operation(ctx, &annotation.name, || {
                        translate_annotation(conn, &annotation, Some(&self.id), params)
                    })
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                } else if let Ok(region_str) = region.extract::<&str>() {
                    // Resolution scoped to self: named path first, then annotation in lineage.
                    let path = BlockGroup::get_path_by_name(conn, &self.id, region_str)
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

                    if path.is_some() {
                        let coordinate = start.unwrap_or(0);
                        with_translation_operation(ctx, region_str, || {
                            translate_from_path(conn, &self.id, coordinate, params)
                        })
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    } else {
                        let annotation = Annotation::query_with_lineage(
                            conn,
                            &self.collection_name,
                            &self.sample_name,
                            &self.name,
                        )
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                        .into_iter()
                        .find(|a| a.name.eq_ignore_ascii_case(region_str))
                        .ok_or_else(|| {
                            PyRuntimeError::new_err(format!(
                                "no path or annotation named '{region_str}' in sequence graph '{}'",
                                self.name
                            ))
                        })?;

                        with_translation_operation(ctx, region_str, || {
                            translate_annotation(conn, &annotation, Some(&self.id), params)
                        })
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    }
                } else {
                    return Err(PyRuntimeError::new_err(
                        "region must be a string or Annotation object",
                    ));
                }
            }
        };

        Ok(self.to_py_block_group(protein_bg))
    }
}

impl PySequenceGraph {
    fn require_context(&self, method: &str) -> PyResult<&DbContext> {
        self.context.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err(format!(
                "{method} requires a Repository context; obtain SequenceGraph via Repository"
            ))
        })
    }

    /// Wraps a raw `BlockGroup` model with this sequence graph's database context.
    fn to_py_block_group(&self, bg: BlockGroup) -> Self {
        PySequenceGraph {
            id: bg.id,
            collection_name: bg.collection_name,
            sample_name: bg.sample_name,
            name: bg.name,
            context: self.context.clone(),
        }
    }
}

#[pymethods]
impl PySequenceGraph {
    /// Derive a coordinate-bounded subgraph from this sequence graph.
    ///
    /// Parameters
    /// ----------
    /// new_sample : str
    ///     Sample name for the derived sequence graph.
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
    ) -> PyResult<PySequenceGraph> {
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
        Ok(self.to_py_block_group(found))
    }

    /// Split this sequence graph into coordinate-bounded subgraphs.
    ///
    /// Parameters
    /// ----------
    /// new_sample : str
    ///     Sample name for the derived sequence graphs.
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
    ) -> PyResult<Vec<PySequenceGraph>> {
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
                .map(|bg| self.to_py_block_group(bg))
                .collect(),
        )
    }
}
