use std::{
    collections::{HashMap, HashSet},
    fs,
    path::PathBuf,
};

use r#gen::{
    commands::graph_operations::{
        derive_chunks::derive_chunks_operation, derive_subgraph::derive_subgraph_operation,
    },
    core::HashId,
    exports::{fasta::export_fasta, genbank::export_genbank, gfa::export_gfa},
    graphs::{
        NodePoint,
        graph_search::{GenGraphMatcher, SeedIndex, SequenceKind},
        translation::{
            TranslationError, translate_annotation, translate_block_group, translate_from_path,
        },
    },
    views::{
        annotation_files::load_annotation_file_entries,
        annotation_track::annotation_segments_from_graph_locus,
        annotations::{AnnotationFileTrackRequest, load_annotation_file_track},
    },
};
use gen_annotations::projection::annotation_segments;
use gen_core::{Strand, is_end_node, is_start_node};
use gen_graph::GraphNode;
use gen_models::{
    accession::{Accession, AccessionSpan, NewAccession},
    annotations::{Annotation, AnnotationGroupSample},
    block_group::{BlockGroup, RandomSequences},
    db::DbContext,
    locus::GraphLocus,
    operations::{OperationInfo, OperationSummary, commit_operation_summary},
    sample::Sample,
};
use pyo3::{
    exceptions::{PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
    types::PyDict,
};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use super::{
    annotation::PyAnnotation,
    editing::{
        Breakpoint, EditKind, EditMode, EditRequest, ExchangeSites, InsertSite, connect_positions, cut_position,
        edit_sequence_graph, exchange_positions, insert_at_positions,
    },
    gaf_reads::{PyGafReads, load_gaf_reads},
    graph_node::PyGraphNode,
    graph_read::{current_graph, locus_from_region, shortest_route},
    graph_search::PyGraphLocus,
    hash_id::PyHashId,
    jupyter_widget::{PlotOptions, PyGraphController, WidgetOptions, build_widget},
    locus::GraphLocusExt as _,
    position::{PyPosition, SuperpositionError, graph_of, positions_of},
    repository::run_context_operation_write,
    sample::PySample,
    sequence::PySequence,
    subgraph::derive_subgraph_between,
    translation::build_translation_params,
    utils::{block_group_err_to_pyerr, distinct_sequences, export_all_sequences_fasta},
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

fn run_translation_operation<F>(
    context: &DbContext,
    label: &str,
    translate: F,
) -> PyResult<BlockGroup>
where
    F: FnOnce() -> Result<BlockGroup, TranslationError>,
{
    let graph_conn = context.graph().conn();
    graph_conn
        .execute("BEGIN TRANSACTION", [])
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

    let protein_bg = match translate() {
        Ok(block_group) => block_group,
        Err(err) => {
            graph_conn.execute("ROLLBACK", []).ok();
            return Err(PyRuntimeError::new_err(err.to_string()));
        }
    };

    graph_conn
        .execute("END TRANSACTION", [])
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

    let operation_summary = OperationSummary::new(
        OperationInfo {
            files: vec![],
            description: "translate annotation".to_string(),
        },
        format!(
            " {}: protein sequence graph derived from {label}",
            protein_bg.name
        ),
    );
    commit_operation_summary(context, &operation_summary)
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

    Ok(protein_bg)
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
///     path  = str(repo.gen_dir)
///
///     def worker():
///         r = gen.Repository(path)
///         sg = r.get_sequence_graph(sg_id)
///         ...
// unsendable because DbContext contains Rc (rusqlite::Connection is !Sync)
#[gen_stub_pyclass]
#[pyclass(name = "SequenceGraph", unsendable, from_py_object)]
#[derive(Clone)]
pub struct PySequenceGraph {
    pub id: HashId,
    /// Collection this sequence graph belongs to.
    #[pyo3(get, name = "collection")]
    pub collection_name: String,
    pub sample_name: String,
    /// Name of the graph, such as the FASTA record or chromosome name. Region strings start with it
    /// (`"name:start-end"`).
    #[pyo3(get)]
    pub name: String,
    pub context: Option<DbContext>,
}

impl PySequenceGraph {
    /// Features of every annotation file in the repository that land on this graph.
    ///
    /// Unlike the widget, which reads indexed files a viewport at a time, this reads each file
    /// across the whole graph so callers can filter by name. A file that can't be read is
    /// skipped so one missing file doesn't hide the database annotations.
    fn file_annotations(&self, context: &DbContext) -> PyResult<Vec<PyAnnotation>> {
        let conn = context.graph().conn();
        let graph = BlockGroup::get_graph(conn, context.workspace(), &self.id, None)
            .map_err(block_group_err_to_pyerr)?;
        let node_filter: HashSet<HashId> = graph
            .nodes()
            .filter(|node| !is_start_node(node.node_id) && !is_end_node(node.node_id))
            .map(|node| node.node_id)
            .collect();
        let Some(start) = graph
            .nodes()
            .filter(|node| node_filter.contains(&node.node_id))
            .map(|node| node.sequence_start)
            .min()
        else {
            return Ok(Vec::new());
        };
        let end = graph
            .nodes()
            .filter(|node| node_filter.contains(&node.node_id))
            .map(|node| node.sequence_end)
            .max()
            .unwrap_or(start);
        let mut annotations = Vec::new();
        for entry in load_annotation_file_entries(conn, None) {
            let Ok(loaded) = load_annotation_file_track(&AnnotationFileTrackRequest {
                conn,
                history_ref: None,
                workspace: context.workspace(),
                collection_name: &self.collection_name,
                sample_name: &self.sample_name,
                block_group_name: Some(&self.name),
                query_window: Some((start, end)),
                node_filter: &node_filter,
                entry: &entry,
            }) else {
                continue;
            };
            annotations.extend(
                loaded
                    .track
                    .annotations
                    .iter()
                    .map(|span| PyAnnotation::from_file_span(span, &entry.display_name, self)),
            );
        }
        Ok(annotations)
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PySequenceGraph {
    #[new]
    #[gen_stub(skip)]
    pub fn new(id: HashId, collection_name: String, name: String, sample_name: String) -> Self {
        PySequenceGraph {
            id,
            collection_name,
            sample_name,
            name,
            context: None,
        }
    }

    /// Hash ID of this sequence graph.
    #[getter]
    fn id(&self) -> PyHashId {
        PyHashId::new(self.id)
    }

    /// The sample that owns this graph.
    #[getter]
    fn sample(&self) -> PyResult<PySample> {
        let context = self.require_context("sample")?;
        Ok(PySample::new(
            self.collection_name.clone(),
            self.sample_name.clone(),
            context.clone(),
        ))
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

    /// Plot this sequence graph as an interactive widget or a static PNG.
    ///
    /// By default, returns a widget for navigation and display. Outside a
    /// Jupyter kernel, this is a text widget that needs no browser.
    ///
    /// Call ``save_png(path)`` on the returned widget to write the current view
    /// to a PNG file instead; this needs no Jupyter dependencies.
    /// Set ``interactive=False`` to return a displayable PNG without writing a file.
    ///
    /// Raises ``RuntimeError`` if this sequence graph was not created via a
    /// ``Repository``.
    ///
    /// Parameters
    /// rows : int, optional
    ///     Initial viewport height in terminal rows. Defaults to 12 for an
    ///     interactive widget and 24 for a static PNG.
    /// cols : int, optional
    ///     Initial viewport width in terminal columns. Defaults to 60 for an
    ///     interactive widget and 400 for a static PNG. Because a static PNG is
    ///     cropped to its content, the wider canvas captures more of the graph
    ///     without adding blank margins.
    /// detail : {"minimal", "normal", "labels", "full"}, optional
    ///     Initial level of node detail.  ``"minimal"`` shows the smallest
    ///     representation; ``"normal"`` shows truncated nodes with annotation names
    ///     hidden; ``"labels"`` shows truncated nodes with annotation names;
    ///     ``"full"`` shows complete sequences.  When omitted, the graph opens at
    ///     ``"normal"``.
    /// colors : callable | dict | list, optional
    ///     Controls how annotation group entries are coloured when they are
    ///     auto-loaded from the repository.
    ///
    ///     - **callable** ``(ann: Annotation) -> str | None`` — called once per
    ///       annotation; return a CSS hex colour (``"#rrggbb"``) to paint it,
    ///       or ``None`` to hide it.
    ///     - **dict** ``{name: color}`` — maps ``ann.name`` to a colour; annotations
    ///       absent from the dict are hidden.
    ///     - **list** ``[color, ...]`` — assigns colours from the list cyclically.
    ///
    ///     When omitted the theme accent palette is used automatically.
    /// show_history : bool, optional
    ///     Keep retired edit-site and pruned edges in the graph, dimmed,
    ///     instead of removing them along with the nodes only they reach.
    ///     Defaults to ``False``.
    /// align_path : bool, optional
    ///     Lay the current path out as one straight row, with variation branching
    ///     above and below it and annotation rows growing downward from the sequence.
    ///     The current path is the one the viewer is following; it is not necessarily
    ///     the reference path. Defaults to ``True``; without a path, no nodes
    ///     are pinned to a row.
    /// interactive : bool, optional
    ///     Return a static PNG when ``False``; defaults to ``True``.
    /// go_to : Position | SuperPosition | Locus | Annotation, optional
    ///     Navigate to this target before rendering, including for static PNGs.
    /// scale : int, optional
    ///     Image pixels drawn per logical pixel, from 1 to 8. A scale of 2 renders the
    ///     widget's PNGs on twice as many pixels and shows them at the same size, which
    ///     is sharper on high-density screens. The PNG is labeled 96 dpi times the scale:
    ///     1 is 96 dpi, 2 is 192 dpi, 3 is 288 dpi (close to a 300 dpi print) and 4 is
    ///     384 dpi. Defaults to 2.
    #[pyo3(signature = (rows=None, cols=None, detail=None, colors=None, show_history=false, align_path=true, interactive=true, *, go_to=None, scale=2))]
    fn plot(
        slf: &Bound<'_, PySequenceGraph>,
        rows: Option<u32>,
        cols: Option<u32>,
        detail: Option<&str>,
        colors: Option<Py<PyAny>>,
        show_history: bool,
        align_path: bool,
        interactive: bool,
        #[gen_stub(override_type(type_repr = "Position | SuperPosition | Locus | Annotation | None", imports = ()))]
        go_to: Option<Py<PyAny>>,
        scale: usize,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let mut ctrl = PyGraphController::for_sequence_graph(
            &slf.borrow(),
            PlotOptions {
                show_history,
                align_path,
            },
        )?;
        if let Some(node_detail) = detail {
            ctrl.set_detail(node_detail)?;
        }
        let ctrl = Py::new(py, ctrl)?;
        build_widget(
            py,
            ctrl,
            WidgetOptions {
                rows,
                cols,
                colors,
                interactive,
                go_to,
                scale,
            },
            &[slf.borrow().clone()],
        )
    }

    /// Search for exact occurrences of `query` in this sequence graph.
    ///
    /// Returns a list of `Locus` objects. Each locus exposes:
    ///   - `.start()` / `.end()` → `Position` (first and last position)
    ///   - `.sequence` → `str` (the bases it covers)
    ///
    /// Parameters
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
        // Search the routes the sequence graph reads, so every hit is a target an edit accepts.
        let graph = current_graph(context, &self.id)?;
        let matcher =
            GenGraphMatcher::new_with_sequence_kind(conn, context.workspace(), graph, kind);

        let gen_dir = context.workspace().ensure_gen_dir();
        let index_path = gen_dir
            .join("search_index")
            .join(format!("{}.bin", self.id));
        let index = fs::read(&index_path)
            .ok()
            .and_then(|bytes| SeedIndex::from_bytes_with_header(&bytes, 16).ok())
            .filter(|index| index.is_valid_for(&matcher));

        let query_bytes = query.as_bytes();
        let matches = match index {
            Some(idx) => matcher
                .find_all_with_seed_index(&idx, query_bytes)
                .unwrap_or_else(|_| matcher.find_all(query_bytes)),
            None => matcher.find_all(query_bytes),
        };

        // A hit that crosses a zero-width routing block lists an empty slice for it, and routes
        // that differ only in the routing blocks they cross find the same hit again. Neither
        // changes what the hit reads, so each hit is reported once, without empty slices. An
        // insertion inside a node meets the node's zero-width junction on both sides, so a hit
        // can leave the inserted block into the junction and re-enter the same block, reading
        // it twice, which no route of the graph does. Walks that repeat a block through real
        // sequence, such as around a cycle, are kept.
        let mut hits: Vec<GraphLocus> = vec![];
        for mut locus in matches {
            let mut readable = locus
                .slices
                .iter()
                .enumerate()
                .filter(|(_, slice)| slice.end > slice.start);
            let mut loops_through_junction = false;
            let mut previous = readable.next();
            for (index, slice) in readable {
                if let Some((previous_index, previous_slice)) = previous {
                    loops_through_junction |= index > previous_index + 1
                        && previous_slice.block == slice.block
                        && previous_slice.strand == slice.strand;
                }
                previous = Some((index, slice));
            }
            locus.slices.retain(|slice| slice.end > slice.start);
            if !loops_through_junction && !hits.contains(&locus) {
                hits.push(locus);
            }
        }
        Ok(hits
            .into_iter()
            .map(|locus| {
                PyGraphLocus::with_context(locus, Some(context.clone()))
                    .attached_to(Some(self.clone()))
            })
            .collect())
    }

    /// Load the reads of a minigraph GAF file as one `Locus` per read.
    ///
    /// The GAF must have been aligned to a GFA exported from this repository, so each path step
    /// names a segment as `<node hash>.<start>.<end>`. Each `Locus` spells the aligned part of its
    /// read's walk, clipped to the alignment interval and oriented as the read travels, so
    /// `locus.sequence` is the graph sequence under the read, `locus.strand` is `"mixed"` when the
    /// walk crosses an inversion, and `widget.show(locus)` lights exactly the edges the read takes.
    ///
    /// A read with several rows keeps the mapped row with the longest aligned block. With
    /// `primary_only` (the default) only primary rows (`tp:A:P`, or no `tp` tag) are considered.
    /// Reads that do not lie in this sequence graph are counted in the result's `skipped`, and a
    /// read whose walk takes a step this graph stores no edge for is never loaded and is listed
    /// in `untraceable`. Returns a `GafReads`, a read-only mapping from read name to `Locus`.
    #[pyo3(signature = (path, *, primary_only=true))]
    pub fn load_gaf(&self, path: &str, primary_only: bool) -> PyResult<PyGafReads> {
        let context = self.require_context("load_gaf()")?;
        load_gaf_reads(self, context, path, primary_only)
    }

    /// The `Locus` covering path coordinates `start` to `end` of this sequence graph: 0-based,
    /// half-open, counted along the graph's current path. `graph.locus(100, 110)` is the same
    /// as `graph.region("<name>:100-110")` without building the string.
    ///
    /// Pass it to editing methods, `graph.subgraph()`, `repo.stitch()` or a widget's `go_to()`. Raises
    /// `ValueError` if the span is empty, outside the path, or maps to more than one route.
    // A `backbone` argument naming another path to count along belongs here once named paths are
    // supported; until then coordinates always follow the current path.
    pub fn locus(&self, start: i64, end: i64) -> PyResult<PyGraphLocus> {
        if start < 0 || end <= start {
            return Err(PyValueError::new_err(format!(
                "locus needs 0 <= start < end, got start={start}, end={end}"
            )));
        }
        self.region(&format!("{}:{start}-{end}", self.name))
    }

    /// Resolve a region string (e.g. ``"chr1:100-110"``) to a ``Locus`` in this sequence graph.
    ///
    /// Unlike passing a region string to ``replace()``, ``delete()``, or ``insert()``, this
    /// performs no edit — it only looks up the coordinates, for inspection (``.sequence``) or as
    /// an argument to a widget's ``go_to()``. Raises ``ValueError`` if the region cannot be
    /// resolved in this sequence graph.
    pub fn region(&self, region: &str) -> PyResult<PyGraphLocus> {
        let context = self.require_context("region()")?;
        let target = locus_from_region(context, region, &self.collection_name, &self.sample_name)?;
        if let Some(block_group_id) = target.block_group_id
            && block_group_id != self.id
        {
            return Err(PyValueError::new_err(format!(
                "region resolves outside sequence graph '{}'",
                self.name
            )));
        }
        Ok(
            PyGraphLocus::with_context(target.locus, Some(context.clone()))
                .attached_to(Some(self.clone())),
        )
    }

    /// Every distinct sequence a path through the graph spells, as a list of `Sequence` sorted by
    /// their bases; `str(sequence)` is its bases. Paths that spell the same bases appear once. The
    /// list is built in full, so a graph with very many paths takes time and memory.
    fn all_sequences(&self) -> PyResult<Vec<PySequence>> {
        let context = self.require_context("all_sequences()")?;
        Ok(distinct_sequences(context, &self.id)?
            .into_iter()
            .map(PySequence::unnamed)
            .collect())
    }

    /// An endless iterator of sequences, each read by a random walk from the start of the graph to
    /// its end that takes one outgoing edge, chosen uniformly, at every step. Nothing is
    /// enumerated, so it suits a graph with very many paths or with cycles, and a Monte Carlo
    /// simulation takes as many as it needs, for example with ``itertools.islice``. A cycle can be
    /// walked any number of times. Pass ``seed`` to repeat the same sequences.
    #[pyo3(signature = (seed=None))]
    fn random_sequences(&self, seed: Option<u64>) -> PyResult<PyRandomSequences> {
        let context = self.require_context("random_sequences()")?;
        BlockGroup::random_sequences(context.graph().conn(), context.workspace(), &self.id, seed)
            .map(|sequences| PyRandomSequences { sequences })
            .map_err(block_group_err_to_pyerr)
    }

    /// IPython display hook — called when a cell ends with a SequenceGraph.
    #[gen_stub(skip)]
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
        let graph = BlockGroup::get_graph(conn, context.workspace(), &self.id, None)
            .map_err(block_group_err_to_pyerr)?;
        let matcher =
            GenGraphMatcher::new_with_sequence_kind(conn, context.workspace(), graph, kind);
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

    /// Return the graph as a dict with `nodes` (a list of `Node`) and `edges` (edge keys are
    /// `(source, target)` node pairs mapped to lists of edge-weight dicts).
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let context = self.require_context("to_dict()")?;
        let conn = context.graph().conn();
        let graph = BlockGroup::get_graph(conn, context.workspace(), &self.id, None)
            .map_err(block_group_err_to_pyerr)?;
        let dict = PyDict::new(py);
        let nodes: Vec<PyGraphNode> = graph
            .nodes()
            .map(|node| {
                PyGraphNode::new(node.node_id, node.sequence_start, node.sequence_end)
                    .with_context(self.context.clone())
            })
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
                    PyGraphNode::new(src.node_id, src.sequence_start, src.sequence_end)
                        .with_context(self.context.clone()),
                    PyGraphNode::new(dst.node_id, dst.sequence_start, dst.sequence_end)
                        .with_context(self.context.clone()),
                ),
                weights?,
            )?;
        }
        dict.set_item("edges", edges)?;
        Ok(dict.into_pyobject(py)?.into_any().unbind())
    }

    /// Return the graph as a `rustworkx.PyDiGraph`. Requires the optional `rustworkx` package.
    fn to_rustworkx(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let context = self.require_context("to_rustworkx()")?;
        let conn = context.graph().conn();
        let graph = BlockGroup::get_graph(conn, context.workspace(), &self.id, None)
            .map_err(block_group_err_to_pyerr)?;
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
                    PyGraphNode::new(node.node_id, node.sequence_start, node.sequence_end)
                        .with_context(self.context.clone()),
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

    /// Return the graph as a `networkx.DiGraph`. Requires the optional `networkx` package.
    fn to_networkx(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let context = self.require_context("to_networkx()")?;
        let conn = context.graph().conn();
        let graph = BlockGroup::get_graph(conn, context.workspace(), &self.id, None)
            .map_err(block_group_err_to_pyerr)?;
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
                    (
                        PyGraphNode::new(node.node_id, node.sequence_start, node.sequence_end)
                            .with_context(self.context.clone()),
                    ),
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
                        PyGraphNode::new(src.node_id, src.sequence_start, src.sequence_end)
                            .with_context(self.context.clone()),
                        PyGraphNode::new(dst.node_id, dst.sequence_start, dst.sequence_end)
                            .with_context(self.context.clone()),
                    ),
                    Some(&kwargs),
                )?;
            }
            Ok(nx_digraph.into_pyobject(py)?.into_any().unbind())
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PySequenceGraph {
    /// Export all sequence graphs in this sequence graph's sample to FASTA.
    ///
    /// Parameters
    /// filename : str
    ///     Output file path.
    /// all_sequences : bool, optional
    ///     Export all graph paths as ``"{sequence_graph_name}.{index}"`` (1-based).
    ///     Defaults to ``False`` (current paths only).
    #[pyo3(signature = (filename, all_sequences=false))]
    fn export_fasta(&self, filename: String, all_sequences: bool) -> PyResult<()> {
        let ctx = self.require_context("export_fasta()")?;
        if all_sequences {
            return export_all_sequences_fasta(
                ctx,
                &self.collection_name,
                Some(&self.sample_name),
                &PathBuf::from(&filename),
            );
        }
        let conn = ctx.graph().conn();
        export_fasta(
            conn,
            ctx.workspace(),
            &self.collection_name,
            Some(&self.sample_name),
            &PathBuf::from(&filename),
            None,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to export FASTA '{}': {e}", filename)))
    }

    /// Export all sequence graphs in this sequence graph's sample to GFA.
    ///
    /// Parameters
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
            ctx.workspace(),
            &self.collection_name,
            &PathBuf::from(&filename),
            &self.sample_name,
            node_max,
            None,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to export GFA '{}': {e}", filename)))
    }

    /// Export all sequence graphs in this sequence graph's sample to GenBank.
    ///
    /// Parameters
    /// filename : str
    ///     Output file path.
    fn export_genbank(&self, filename: String) -> PyResult<()> {
        let ctx = self.require_context("export_genbank()")?;
        let conn = ctx.graph().conn();
        let writer = fs::File::create(&filename).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to create '{}': {e}", filename))
        })?;
        export_genbank(
            conn,
            ctx.workspace(),
            &self.collection_name,
            &self.sample_name,
            writer,
            None,
        )
        .map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to export GenBank '{}': {e}", filename))
        })
    }

    /// All annotations associated with this sequence graph.
    ///
    /// Annotations imported from files (``Repository.import_annotations()``) are read in full
    /// and listed after the database ones, so they can be searched by name alike. Their
    /// ``track`` is the file's display name.
    ///
    /// Returns
    /// list[Annotation]
    #[getter]
    fn annotations(&self) -> PyResult<Vec<PyAnnotation>> {
        let ctx = self.require_context("annotations")?;
        let conn = ctx.graph().conn();
        let bg_id = self.id;
        let annotations = Annotation::query_with_lineage(
            conn,
            &self.collection_name,
            &self.sample_name,
            &self.name,
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let mut result: Vec<PyAnnotation> = annotations
            .into_iter()
            .map(|a| PyAnnotation {
                ann_segments: annotation_segments(conn, &a, None),
                inner: a,
                context: Some(ctx.clone()),
                source_block_group_id: Some(bg_id),
                locus: None,
                sequence_graph: Some(self.clone()),
            })
            .collect();
        result.extend(self.file_annotations(ctx)?);
        Ok(result)
    }

    /// Persist an annotation over a locus in this sequence graph.
    ///
    /// Parameters
    /// target : str, Locus or Annotation
    ///     Graph-space span to annotate: a region string, a ``Locus`` or an existing
    ///     ``Annotation``. It must cover at least one base and belong to this sequence graph.
    /// name : str
    ///     Human-readable annotation name.
    /// track : str, optional
    ///     Annotation group in the database. Defaults to ``"default"``.
    ///
    /// Returns the database-backed ``Annotation`` that was created.
    #[pyo3(signature = (target, name, track="default"))]
    fn add_annotation(
        &self,
        #[gen_stub(override_type(type_repr = "str | Locus | Annotation", imports = ()))]
        target: &Bound<'_, PyAny>,
        name: &str,
        track: &str,
    ) -> PyResult<PyAnnotation> {
        let target = target
            .extract::<PyRef<PyGraphLocus>>()
            .map_err(|_| PyTypeError::new_err("annotation target must be a Locus"))?;
        let target_locus = target.graph_locus();
        if target_locus.slices.is_empty() || target_locus.length() == 0 {
            return Err(PyValueError::new_err(
                "annotation target must cover at least one base",
            ));
        }
        let context = self.require_context("add_annotation()")?;
        let annotation_segments = annotation_segments_from_graph_locus(&target_locus);

        run_context_operation_write(
            context,
            |context| {
                let conn = context.graph().conn();
                let spans: Vec<AccessionSpan> = annotation_segments
                    .iter()
                    .map(|segment| AccessionSpan {
                        node_id: segment.node_id,
                        range: segment.range,
                        strand: segment.strand,
                    })
                    .collect();
                let accession = Accession::get_or_create(
                    conn,
                    &NewAccession {
                        name: name.to_string(),
                        block_group_id: self.id,
                        parent_accession_id: None,
                        spans,
                    },
                )
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
                let annotation = Annotation::get_or_create(conn, name, track, &accession.id, None)
                    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
                AnnotationGroupSample::create(conn, track, &self.sample_name)
                    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
                let result = PyAnnotation {
                    inner: annotation,
                    context: Some(context.clone()),
                    ann_segments: annotation_segments.clone(),
                    source_block_group_id: Some(self.id),
                    locus: Some(target_locus),
                    sequence_graph: Some(self.clone()),
                };
                Ok((
                    result,
                    OperationSummary::new(
                        OperationInfo {
                            files: vec![],
                            description: "add_annotation".to_string(),
                        },
                        format!("add annotation '{name}' to track '{track}'"),
                    ),
                ))
            },
            |error| PyRuntimeError::new_err(format!("failed to add annotation: {error}")),
        )
    }

    /// Translate a sequence graph or annotation into a protein ``SequenceGraph``.
    ///
    /// When ``region`` is a string it is resolved against this sequence graph
    /// only, in priority order: named path within this graph first, then
    /// annotation in this graph's lineage. No other sequence graphs are
    /// searched. The protein sequence graph is created in this graph's sample.
    ///
    /// Parameters
    /// region : str or Annotation, optional
    ///     - ``str``: a path name or annotation name scoped to this sequence
    ///       graph. Path names take priority over annotation names.
    ///     - ``Annotation``: an object returned by ``annotations``.
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
        #[gen_stub(override_type(type_repr = "str | Annotation | None", imports = ()))]
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
                    run_translation_operation(ctx, &label, || {
                        translate_from_path(conn, ctx.workspace(), &bg_id, start, params)
                    })
                } else {
                    run_translation_operation(ctx, &label, || {
                        translate_block_group(conn, ctx.workspace(), &bg_id, params)
                    })
                }
            }
            Some(region) => {
                if let Ok(ann) = region.extract::<PyRef<PyAnnotation>>() {
                    let annotation = ann.inner.clone();
                    run_translation_operation(ctx, &annotation.name, || {
                        translate_annotation(
                            conn,
                            ctx.workspace(),
                            &annotation,
                            Some(&self.id),
                            params,
                        )
                    })
                } else if let Ok(region_str) = region.extract::<&str>() {
                    // Resolution scoped to self: named path first, then annotation in lineage.
                    let path = BlockGroup::get_path_by_name(conn, &self.id, region_str)
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

                    if path.is_some() {
                        let coordinate = start.unwrap_or(0);
                        run_translation_operation(ctx, region_str, || {
                            translate_from_path(conn, ctx.workspace(), &self.id, coordinate, params)
                        })
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

                        run_translation_operation(ctx, region_str, || {
                            translate_annotation(
                                conn,
                                ctx.workspace(),
                                &annotation,
                                Some(&self.id),
                                params,
                            )
                        })
                    }
                } else {
                    return Err(PyRuntimeError::new_err(
                        "region must be a string or Annotation object",
                    ));
                }
            }
        }?;

        Ok(self.to_py_block_group(protein_bg))
    }
}

impl PySequenceGraph {
    /// `subgraph()` for integer path coordinates.
    fn subgraph_by_coordinates(
        &self,
        ctx: &DbContext,
        new_sample: String,
        start: i64,
        end: i64,
    ) -> PyResult<PySequenceGraph> {
        let region = format!("{}:{}-{}", self.name, start, end);
        derive_subgraph_operation(
            ctx,
            Some(self.collection_name.clone()),
            self.sample_name.clone(),
            new_sample.clone(),
            region,
            None,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("Error deriving subgraph: {e}")))?;
        let conn = ctx.graph().conn();
        let child_id = BlockGroup::get_id(
            &self.collection_name,
            &new_sample,
            &self.name,
            Some(&self.id),
        );
        let found = BlockGroup::get_by_id(conn, &child_id, None)
            .map_err(|e| PyRuntimeError::new_err(format!("Subgraph created but not found: {e}")))?;
        Ok(self.to_py_block_group(found))
    }

    /// The node points bounding a first and last `Position` in this graph. The end point is just
    /// past the last base, as the last position names an included base.
    fn subgraph_points(
        &self,
        ctx: &DbContext,
        start: &Bound<'_, PyAny>,
        end: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(NodePoint, NodePoint)> {
        let (first, last) = if let (Ok(first), Some(Ok(last))) = (
            start.extract::<PyRef<PyPosition>>(),
            end.map(|end| end.extract::<PyRef<PyPosition>>()),
        ) {
            if first.strand() != "+" || last.strand() != "+" {
                return Err(PyValueError::new_err(
                    "subgraph() takes forward-strand positions",
                ));
            }
            (first.clone(), last.clone())
        } else {
            return Err(PyTypeError::new_err(
                "subgraph() takes two Positions or two integer coordinates",
            ));
        };

        let graph = current_graph(ctx, &self.id)?;
        let locate = |position: &PyPosition, which: &str| {
            PyPosition::located(&graph, position.position).map_err(|_| {
                PyValueError::new_err(format!(
                    "the {which} position is not in this sequence graph"
                ))
            })
        };
        let first = locate(&first, "start")?;
        let last = locate(&last, "end")?;
        let reaches = if first.block == last.block {
            first.position.coordinate <= last.position.coordinate
        } else {
            shortest_route(&graph, first.block, last.block).is_some()
        };
        if !reaches {
            return Err(PyValueError::new_err(
                "the end position cannot be reached from the start position in this sequence graph",
            ));
        }
        Ok((
            NodePoint {
                id: first.position.node_id,
                coordinate: first.position.coordinate,
                strand: Strand::Forward,
            },
            NodePoint {
                id: last.position.node_id,
                coordinate: last.position.coordinate + 1,
                strand: Strand::Forward,
            },
        ))
    }

    pub(crate) fn require_context(&self, method: &str) -> PyResult<&DbContext> {
        self.context.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err(format!(
                "{method} requires a Repository context; obtain SequenceGraph via Repository"
            ))
        })
    }

    /// Wraps a sequence graph model with this sequence graph's database context.
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

#[gen_stub_pymethods]
#[pymethods]
impl PySequenceGraph {
    /// Derive a subgraph of this sequence graph into a new sample, holding every variant route
    /// between two points.
    ///
    /// The span can be given two ways:
    ///
    /// - two `Position`s, the first and last base to include. For a `Locus`, pass
    ///   `locus.start()` and `locus.end()`;
    /// - two integers, path coordinates along the current path (0-based, end exclusive):
    ///   `graph.subgraph("mcs", 390, 460)`.
    ///
    /// Positions must be forward-strand. Raises
    /// `ValueError` if a position is not in this graph, the span is empty or reversed, or the end
    /// cannot be reached from the start. The new graph gets a current path when this
    /// graph's current path runs from start to end.
    ///
    /// Parameters
    /// new_sample : str
    ///     Sample name for the derived sequence graph.
    /// start : Position or int
    ///     The first position (or path coordinate) of the span.
    /// end : Position or int
    ///     The last position (or exclusive path coordinate) of the span.
    // Integer coordinates follow the current path. A `backbone` argument naming another path to
    // count along can be added once named paths are supported.
    #[pyo3(signature = (new_sample, start, end))]
    fn subgraph(
        &self,
        new_sample: String,
        #[gen_stub(override_type(type_repr = "Position | int", imports = ()))] start: &Bound<
            '_,
            PyAny,
        >,
        #[gen_stub(override_type(type_repr = "Position | int", imports = ()))] end: &Bound<
            '_,
            PyAny,
        >,
    ) -> PyResult<PySequenceGraph> {
        let ctx = self.require_context("subgraph()")?;
        if let (Ok(start), Ok(end)) = (start.extract::<i64>(), end.extract::<i64>()) {
            return self.subgraph_by_coordinates(ctx, new_sample, start, end);
        }
        let (start_point, end_point) = self.subgraph_points(ctx, start, Some(end))?;
        let child = run_context_operation_write(
            ctx,
            |context| {
                let block_group =
                    derive_subgraph_between(context, &self.id, &new_sample, start_point, end_point)
                        .map_err(|e| {
                            PyRuntimeError::new_err(format!("Error deriving subgraph: {e}"))
                        })?;
                let summary = OperationSummary::new(
                    OperationInfo {
                        files: vec![],
                        description: "subgraph".to_string(),
                    },
                    format!(" {new_sample}: subgraph of {}", self.name),
                );
                Ok((block_group, summary))
            },
            |error| PyRuntimeError::new_err(format!("Error deriving subgraph: {error}")),
        )?;
        Ok(self.to_py_block_group(child))
    }

    /// Split this sequence graph into coordinate-bounded subgraphs.
    ///
    /// Parameters
    /// new_sample : str
    ///     Sample name for the derived sequence graphs.
    /// breakpoints : list[int], optional
    ///     Coordinates at which to split.
    /// chunk_size : int, optional
    ///     Split into equal-length pieces of this many bases.
    ///
    /// Returns the chunks in order along the graph, so ``repo.stitch(chunks, ...)`` joins them back
    /// in the order they were cut.
    // Coordinates follow the current path. A `backbone` argument naming another path to count
    // along can be added once named paths are supported.
    #[pyo3(signature = (new_sample, breakpoints=None, chunk_size=None))]
    fn chunks(
        &self,
        new_sample: String,
        breakpoints: Option<Vec<i64>>,
        chunk_size: Option<i64>,
    ) -> PyResult<Vec<PySequenceGraph>> {
        let ctx = self.require_context("chunks()")?;
        derive_chunks_operation(
            ctx,
            Some(self.collection_name.clone()),
            self.sample_name.clone(),
            new_sample.clone(),
            self.name.clone(),
            None,
            breakpoints,
            chunk_size,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("Error deriving chunks: {e}")))?;
        let conn = ctx.graph().conn();
        let prefix = format!("{}.", self.name);
        let mut chunks = Sample::get_block_groups(conn, &self.collection_name, &new_sample, None)
            .into_iter()
            .filter(|bg| bg.name == self.name || bg.name.starts_with(&prefix))
            .collect::<Vec<_>>();
        // Chunks are numbered along the graph, so list them in that order rather than by name,
        // which puts `.10` before `.2`. Stitching the list then rebuilds the original order.
        chunks.sort_by_key(|bg| {
            let number = bg
                .name
                .strip_prefix(&prefix)
                .and_then(|suffix| suffix.parse::<u64>().ok());
            (number, bg.name.clone())
        });
        Ok(chunks
            .into_iter()
            .map(|bg| self.to_py_block_group(bg))
            .collect())
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PySequenceGraph {
    /// Replace the sequence covered by ``target`` with ``sequence``.
    ///
    /// The target may be a region string, a ``Locus`` from ``search()``, or an
    /// ``Annotation`` from ``annotations``. On a reverse-strand target the
    /// sequence is read on that strand. The edit is recorded as its own operation,
    /// using ``message`` as the operation's commit message when given, or a
    /// generated description otherwise. Returns the ``Locus`` of the replacement
    /// sequence.
    ///
    /// Parameters
    /// stack : bool, optional
    ///     Add this replacement as a sibling option at the target instead of
    ///     superseding what's there, so the original sequence is left standing
    ///     alongside it (default ``False``). A stacked replacement never
    ///     becomes the reference route: the sequence graph's current Path is
    ///     left exactly as it was, even when the target lies on it.
    #[pyo3(signature = (target, sequence, message=None, stack=false))]
    fn replace(
        &self,
        #[gen_stub(override_type(type_repr = "str | Locus | Annotation", imports = ()))]
        target: &Bound<'_, PyAny>,
        sequence: &str,
        message: Option<&str>,
        stack: bool,
    ) -> PyResult<PyGraphLocus> {
        let request = EditRequest {
            kind: EditKind::Replace,
            sequence,
            message,
            mode: EditMode { stack },
        };
        edit_sequence_graph(self, target, &request).map(|locus| {
            PyGraphLocus::with_context(
                locus.expect("should return the replacement locus"),
                self.context.clone(),
            )
            .attached_to(Some(self.clone()))
        })
    }

    /// Delete the sequence covered by ``target``.
    ///
    /// The target may be a region string, a ``Locus`` from ``search()``, or an
    /// ``Annotation`` from ``annotations``. The edit is recorded as its own
    /// operation, using ``message`` as the operation's commit message when given,
    /// or a generated description otherwise::
    ///
    ///     for annotation in sg.annotations:
    ///         sg.delete(annotation.locus)
    ///
    /// Parameters
    /// stack : bool, optional
    ///     Add the deletion as a sibling option at the target instead of
    ///     removing the only route through it, so the original sequence stays
    ///     reachable alongside the shortcut around them (default ``False``).
    ///     A stacked deletion never becomes the reference route: the
    ///     sequence graph's current Path is left exactly as it was, even
    ///     when the target lies on it.
    #[pyo3(signature = (target, message=None, stack=false))]
    fn delete(
        &self,
        #[gen_stub(override_type(type_repr = "str | Locus | Annotation", imports = ()))]
        target: &Bound<'_, PyAny>,
        message: Option<&str>,
        stack: bool,
    ) -> PyResult<()> {
        let request = EditRequest {
            kind: EditKind::Delete,
            sequence: "",
            message,
            mode: EditMode { stack },
        };
        edit_sequence_graph(self, target, &request).map(|_| ())
    }

    /// Insert ``sequence`` next to a position.
    ///
    /// Give either ``after`` or ``before``, each a ``Position`` or a ``SuperPosition``:
    /// ``after=a`` reads ``a`` then the new sequence, and ``before=b`` reads the new sequence then
    /// ``b``::
    ///
    ///     sg.insert("ACGT", after=locus.end())
    ///     sg.insert("ACGT", before=locus.start())
    ///
    /// The other side is whatever reads next to the positions given. Where ``a`` forks into ``b``
    /// and ``c``, ``after=a.end()`` wires ``a`` to the new sequence and the new sequence to both
    /// ``b`` and ``c``, while ``before=b.start()`` goes only between ``a`` and ``b`` and leaves the
    /// connection from ``a`` into ``c`` as it was.
    ///
    /// The positions a ``SuperPosition`` names choose which routes the sequence lands on. With
    /// ``a1``, ``a2`` and ``a3`` each reading into ``b1``, ``b2`` and ``b3`` in a combinatorial
    /// layer::
    ///
    ///     sg.insert("ACGT", after=a1.end())                  # a1 -> new -> b1, b2, b3
    ///     sg.insert("ACGT", before=b1.start())               # a1, a2, a3 -> new -> b1
    ///     sg.insert("ACGT", after=a1.end() | a2.end())       # a1, a2 -> new -> b1, b2, b3
    ///     sg.insert("ACGT", before=b1.start() | b2.start())  # a1, a2, a3 -> new -> b1, b2
    ///
    /// Every route that reaches a point reads every route that leaves it. A stacked ``replace()``
    /// leaves at the first base of the sequence it sits beside and rejoins at its last, so an
    /// insertion before that first base or after that last base would land on the alternative
    /// too, and raises ``ValueError``. Insert inside that sequence, or at the alternative's ends.
    ///
    /// One call inserts one piece of sequence, connected to everything its positions reach. To
    /// keep routes apart, insert once per route::
    ///
    ///     for left in (a.end(), c.end()):
    ///         sg.insert("ACGT", after=left)
    ///
    /// On the reverse strand, "after" is toward the start of the node and the sequence is read on
    /// that strand. Returns the ``Locus`` of the inserted sequence.
    ///
    /// Parameters
    /// stack : bool, optional
    ///     Add the insertion alongside the existing routes instead of retiring the connections
    ///     it lands on (default ``False``). A stacked insertion leaves the current Path as it was.
    /// message : str, optional
    ///     Commit message for the recorded operation.
    #[pyo3(signature = (sequence, *, before=None, after=None, message=None, stack=false))]
    fn insert(
        &self,
        sequence: &str,
        #[gen_stub(override_type(type_repr = "Position | SuperPosition | None", imports = ()))]
        before: Option<&Bound<'_, PyAny>>,
        #[gen_stub(override_type(type_repr = "Position | SuperPosition | None", imports = ()))]
        after: Option<&Bound<'_, PyAny>>,
        message: Option<&str>,
        stack: bool,
    ) -> PyResult<PyGraphLocus> {
        let mode = EditMode { stack };
        let before = before.map(positions_of).transpose()?;
        let after = after.map(positions_of).transpose()?;
        let site = match (&after, &before) {
            (Some(after), None) => InsertSite::After(after),
            (None, Some(before)) => InsertSite::Before(before),
            (Some(_), Some(_)) => {
                return Err(PyTypeError::new_err(
                    "insert() takes after= or before=, not both; to land on some routes only, \
                     name their positions with a SuperPosition, such as before=b1.start() | \
                     b2.start()",
                ));
            }
            (None, None) => {
                return Err(PyTypeError::new_err(
                    "insert() needs a position: pass after= or before=",
                ));
            }
        };
        insert_at_positions(self, &site, sequence, message, mode).map(|locus| {
            PyGraphLocus::with_context(locus, self.context.clone()).attached_to(Some(self.clone()))
        })
    }

    /// Mark a cut in the sequence, just ``after`` one ``Position`` or just ``before`` it.
    ///
    /// A cut is a block boundary for later joins to attach to, such as the end a restriction
    /// enzyme leaves in a Golden Gate digest. It does not break the graph: the sequence reads
    /// across it as before, so the graph stays a supergraph of the molecules before and after the
    /// cut. It is recorded as one operation, read on the top strand and taken as blunt on both
    /// strands, so the staggered overhang of an enzyme is not modelled yet. The cut must lie inside
    /// a node, and cutting the same point twice raises ``ValueError``.
    ///
    /// Parameters
    /// message : str, optional
    ///     Commit message for the recorded operation.
    #[pyo3(signature = (*, after=None, before=None, message=None))]
    fn cut(
        &self,
        #[gen_stub(override_type(type_repr = "Position | None", imports = ()))] after: Option<
            &Bound<'_, PyAny>,
        >,
        #[gen_stub(override_type(type_repr = "Position | None", imports = ()))] before: Option<
            &Bound<'_, PyAny>,
        >,
        message: Option<&str>,
    ) -> PyResult<()> {
        let single = |name: &str, value: &Bound<'_, PyAny>| {
            let positions = positions_of(value)?;
            match positions.as_slice() {
                [position] => Ok(*position),
                _ => Err(PyValueError::new_err(format!(
                    "cut() needs one Position for {name}=, not {}",
                    positions.len()
                ))),
            }
        };
        match (after, before) {
            (Some(after), None) => cut_position(self, &single("after", after)?, true, message),
            (None, Some(before)) => cut_position(self, &single("before", before)?, false, message),
            _ => Err(PyTypeError::new_err(
                "cut() takes exactly one of after= or before=",
            )),
        }
    }

    /// Add a raw edge that reads on from ``after`` into ``before``, without writing new sequence.
    ///
    /// ``after`` and ``before`` are each one ``Position``, read on its own strand. The edge leaves
    /// the point just after ``after`` and arrives at the point just before ``before``. On the
    /// reverse strand "after" is toward the start of the node, so the edge between a forward and
    /// a reverse position joins the forward strand to the reverse complement of the sequence
    /// beyond it. This is the primitive for rearrangements that reconnect existing sequence, such
    /// as Cre-loxP recombination. Within one molecule, an excision is two forward connections that
    /// close the flanks and the excised segment separately, and an inversion is two connections
    /// between a forward and a reverse position::
    ///
    ///     # excision of the segment between two direct-repeat sites, leaving a circle
    ///     sg.connect(after=left_cut - 1, before=right_cut)
    ///     sg.connect(after=right_cut - 1, before=left_cut)
    ///
    ///     # inversion of the segment between two inverted-repeat sites
    ///     sg.connect(after=left_cut - 1, before=(right_cut - 1).reverse_complement())
    ///     sg.connect(after=left_cut.reverse_complement(), before=right_cut)
    ///
    /// A position attached to another sequence graph of the same sample, with ``position.on(other)``,
    /// joins two molecules. ``other`` is folded into this sequence graph, which then holds both
    /// molecules as components, and ``other`` no longer exists in the sample. Reciprocal crossover
    /// between two linear molecules is two connections, each running a left flank into the other
    /// molecule's right flank::
    ///
    ///     a.connect(after=a_left_last, before=b_right_first.on(b))
    ///     a.connect(after=b_left_last.on(b), before=a_right_first)
    ///
    /// Both molecules attach to this graph's one start and end. Paths of ``other`` are not moved.
    /// Annotations of ``other`` are copied onto this graph, except one already here over exactly the
    /// same nodes and coordinates, which is the same feature.
    ///
    /// An annotation name that this graph already uses for different spans takes primes, so joining
    /// graphs that both name a different feature ``P`` keeps one ``P`` and moves the other as
    /// ``P'``, then ``P''``, and so on until the name is free. Pass ``strict=True`` to raise
    /// ``ValueError`` on a clash instead.
    ///
    /// Everything stays in this one sequence graph, so a circle split off by an excision is a
    /// component of the same graph. A node cut at either point stays cut there. Joining a point
    /// back to the position that originally followed it restores the original continuity, which
    /// reverses an earlier connection.
    ///
    /// The current path is left as it was.
    ///
    /// Parameters
    /// stack : bool, optional
    ///     Keep the original continuity at both points as an alternative instead of retiring it
    ///     (default ``False``).
    /// strict : bool, optional
    ///     Raise ``ValueError`` instead of renaming an annotation whose name is already
    ///     taken here (default ``False``).
    /// message : str, optional
    ///     Commit message for the recorded operation.
    #[pyo3(signature = (*, after, before, message=None, stack=false, strict=false))]
    fn connect(
        &self,
        #[gen_stub(override_type(type_repr = "Position", imports = ()))] after: &Bound<'_, PyAny>,
        #[gen_stub(override_type(type_repr = "Position", imports = ()))] before: &Bound<'_, PyAny>,
        message: Option<&str>,
        stack: bool,
        strict: bool,
    ) -> PyResult<()> {
        let single = |name: &str, value: &Bound<'_, PyAny>| {
            let positions = positions_of(value)?;
            match positions.as_slice() {
                [position] => Ok(*position),
                _ => Err(PyValueError::new_err(format!(
                    "connect() needs one Position for {name}=, not {}",
                    positions.len()
                ))),
            }
        };
        let mut joined = vec![];
        for graph in [graph_of(after), graph_of(before)].into_iter().flatten() {
            if graph.id != self.id
                && !joined
                    .iter()
                    .any(|other: &PySequenceGraph| other.id == graph.id)
            {
                joined.push(graph);
            }
        }
        let (after, before) = (single("after", after)?, single("before", before)?);
        connect_positions(
            self,
            &joined,
            &after,
            &before,
            message,
            EditMode { stack },
            strict,
        )
    }

    /// Exchange the sequence across two breakpoints, as one operation.
    ///
    /// Each breakpoint is a pair of nucleotides, read upstream to downstream on one strand, that the
    /// graph joins: ``exchange((A, B), (C, D))``. The exchange joins ``A`` to ``D`` and ``C`` to ``B``,
    /// the two products of a recombinase crossing over, and makes no other pairing. Naming both sides of
    /// each cut pins it to one junction when a position has several continuations, such as the two
    /// alleles of a variant, or a stacked library of events. A pair the graph does not join raises
    /// ``ValueError``.
    ///
    /// A ``SuperPosition`` in a pair raises ``SuperpositionError``, which is what a step next to a
    /// fork gives. A step is fine anywhere else, so ``(pos, pos + 1)`` names a breakpoint, and so does
    /// ``(hit.start(), hit.start() + 1)`` for a search hit. Whether two nucleotides are joined is
    /// checked in the graph: inside a node on one strand they always are, and elsewhere it takes an
    /// edge, which may cross strands.
    ///
    /// The strands the pairs are written on are the orientation of their sites, and the sequence is
    /// never read to decide it. Both pairs on the same strand line the sites up as direct repeats, and
    /// one pair on each strand as inverted repeats.
    /// Within one molecule direct repeats excise the segment between them as a
    /// circle and inverted repeats turn it around. Between two linear molecules direct repeats cross the
    /// arms over, and inverted repeats join each arm to the other's reverse complement. Between a circle
    /// and a line direct repeats integrate the circle, and inverted repeats integrate it reverse
    /// complemented. Writing both pairs on the other strand, ``(B.reverse_complement(),
    /// A.reverse_complement())`` and ``(D.reverse_complement(), C.reverse_complement())``, describes the
    /// same exchange.
    ///
    /// A position attached to another sequence graph of the same sample, with
    /// ``position.on(other)``, folds ``other`` into this sequence graph first, as in
    /// ``connect``, and ``other`` no longer exists in the sample.
    ///
    /// Unless stacked, the exchange is final: the original continuity at both breakpoints is
    /// removed from the graph. By default every molecule it leaves stays in this one graph, so an
    /// excised circle is a component beside the linear molecule, and ``plot(show_history=True,
    /// align_path=False)`` draws the link between them as retired edges. A plot without
    /// ``show_history`` and ``all_sequences()`` read only what is reachable from the start. A stacked
    /// exchange keeps the originals as an alternative, so everything also stays in this graph.
    ///
    /// With ``split=True`` an unstacked exchange gives each molecule it leaves its own sequence
    /// graph. Two linear molecules give two graphs, an excision gives the molecule and a circle, and
    /// an inversion or an integration gives one. The graph continuing this graph's start keeps this
    /// graph, the one continuing the other graph's start gets that graph back under its name, and
    /// any other, such as an excised circle, is named after this graph with a number. Paths are
    /// dropped, and an annotation that spans a breakpoint is kept by every graph it touches.
    ///
    /// Returns the resulting sequence graphs, this one first. Unless ``split`` gives it more, that
    /// is this graph alone.
    ///
    /// Parameters
    /// first : tuple of Position
    ///     The first breakpoint, ``(A, B)``.
    /// second : tuple of Position
    ///     The second breakpoint, ``(C, D)``.
    /// stack : bool, optional
    ///     Keep the original continuity at both points as an alternative instead of removing it
    ///     (default ``False``).
    /// strict : bool, optional
    ///     Raise ``ValueError`` instead of renaming an annotation whose name is already
    ///     taken here (default ``False``).
    /// split : bool, optional
    ///     Give each molecule an unstacked exchange leaves its own sequence graph, instead of
    ///     keeping them all in this one (default ``False``).
    /// message : str, optional
    ///     Commit message for the recorded operation.
    #[pyo3(signature = (first, second, /, *, message=None, stack=false, strict=false, split=false))]
    fn exchange(
        &self,
        #[gen_stub(override_type(type_repr = "tuple[Position, Position]", imports = ()))]
        first: &Bound<'_, PyAny>,
        #[gen_stub(override_type(type_repr = "tuple[Position, Position]", imports = ()))]
        second: &Bound<'_, PyAny>,
        message: Option<&str>,
        stack: bool,
        strict: bool,
        split: bool,
    ) -> PyResult<Vec<PySequenceGraph>> {
        let pair = |name: &str, value: &Bound<'_, PyAny>| {
            let (upstream, downstream) = value.extract::<(Bound<'_, PyAny>, Bound<'_, PyAny>)>().map_err(|_| {
                PyValueError::new_err(format!(
                    "exchange() needs a pair of Positions for the {name} breakpoint, such as (a, a + 1)"
                ))
            })?;
            let single = |value: &Bound<'_, PyAny>| match positions_of(value)?.as_slice() {
                [position] => Ok(*position),
                positions => Err(SuperpositionError::new_err(format!(
                    "exchange() needs one Position at each end of a breakpoint, but one in the {name} \
                     breakpoint holds {}. A step next to a fork is a SuperPosition: name the \
                     position on the branch you mean",
                    positions.len()
                ))),
            };
            Ok::<_, PyErr>((
                [graph_of(&upstream), graph_of(&downstream)],
                Breakpoint {
                    upstream: single(&upstream)?,
                    downstream: single(&downstream)?,
                },
            ))
        };
        let ((first_graphs, first), (second_graphs, second)) =
            (pair("first", first)?, pair("second", second)?);
        let mut foreign = first_graphs
            .into_iter()
            .chain(second_graphs)
            .flatten()
            .filter(|graph| graph.id != self.id)
            .collect::<Vec<_>>();
        foreign.dedup_by(|a, b| a.id == b.id);
        if foreign.len() > 1 {
            return Err(PyValueError::new_err(
                "exchange() joins at most one other sequence graph, and one position must be in this one",
            ));
        }
        let sites = ExchangeSites { first, second };
        let context = self.require_context("exchange")?;
        let products = exchange_positions(
            self,
            foreign.first(),
            &sites,
            message,
            EditMode { stack },
            strict,
            split,
        )?;
        Ok(products
            .into_iter()
            .map(|product| PySequenceGraph {
                id: product.id,
                collection_name: product.collection_name,
                sample_name: product.sample_name,
                name: product.name,
                context: Some(context.clone()),
            })
            .collect())
    }

    /// Recombine two sites, as one strand exchange.
    ///
    /// A site is a stretch of sequence a recombinase acts on, with the place each strand is cut, as a
    /// ``Site`` gives it. ``recombine(site_a, site_b)`` exchanges the strands across the two sites, so
    /// it makes the same junctions as ``exchange`` does at their top-strand cuts, and what the call does
    /// follows from where the sites are: direct repeats in one molecule excise the segment between them
    /// as a circle, inverted repeats turn it around, two linear molecules cross their arms over, and a
    /// circle with a line integrates the circle.
    ///
    /// The sites are aligned by the strings they are, so each locus is taken in its reading direction.
    /// A site with a direction of its own, found by searching for its sequence, is aligned the way it was
    /// found. A palindromic one, which reads the same on both strands, has no direction, so aligning it
    /// the other way round takes ``site.reverse_complement()``.
    ///
    /// Staggered sites are checked first: their overhangs must be the same length and the same bases,
    /// because each product carries one strand's overhang from each parent, and different bases would
    /// leave a bubble in the double strand. Raises ``ValueError`` when they differ. Blunt sites need no
    /// check. The cut itself is made at the top strand, which the matching overhangs make equivalent to
    /// the bottom one.
    ///
    /// Inverted sites in two linear molecules are not supported yet, and raise ``ValueError``: one
    /// product would run from the end of a molecule backwards and so have no start, and the other no
    /// end. A circle with a line is fine, and so is one molecule.
    ///
    /// Everything else is as in ``exchange``: ``stack`` keeps the originals as an alternative, a site
    /// attached to another sequence graph with ``position.on(other)`` folds that graph in first, and the
    /// resulting sequence graphs are returned, this one first.
    ///
    /// Parameters
    /// first : Site
    ///     The first site.
    /// second : Site
    ///     The second site.
    /// stack : bool, optional
    ///     Keep the original continuity at both sites as an alternative instead of removing it
    ///     (default ``False``).
    /// strict : bool, optional
    ///     Raise ``ValueError`` instead of renaming an annotation whose name is already
    ///     taken here (default ``False``).
    /// split : bool, optional
    ///     Give each molecule an unstacked recombination leaves its own sequence graph, instead of
    ///     keeping them all in this one (default ``False``).
    /// message : str, optional
    ///     Commit message for the recorded operation.
    #[pyo3(signature = (first, second, /, *, message=None, stack=false, strict=false, split=false))]
    fn recombine(
        &self,
        #[gen_stub(override_type(type_repr = "gen.sites.Site", imports = ("gen.sites",)))] first: &Bound<'_, PyAny>,
        #[gen_stub(override_type(type_repr = "gen.sites.Site", imports = ("gen.sites",)))] second: &Bound<'_, PyAny>,
        message: Option<&str>,
        stack: bool,
        strict: bool,
        split: bool,
    ) -> PyResult<Vec<PySequenceGraph>> {
        let (first_top, second_top) = (top_breakpoint(first)?, top_breakpoint(second)?);
        first.call_method1("check_overhang_matches", (second,))?;
        self.exchange(&first_top, &second_top, message, stack, strict, split)
    }
}

/// The top-strand breakpoint of a `Site`, which is where `recombine` makes its junction.
fn top_breakpoint<'py>(site: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if !site.hasattr("breakpoints")? {
        return Err(PyTypeError::new_err(
            "recombine() needs two Sites, such as gen.Site(locus, cut)",
        ));
    }
    site.call_method0("breakpoints")?.get_item(0)
}

#[cfg(test)]
mod tests {
    use r#gen::test_helpers::{create_bg, setup_block_group, setup_gen_on_disk};
    use gen_core::{HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand};
    use gen_models::{
        block_group::BlockGroup,
        block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
        db::DbContext,
        edge::Edge,
        node::Node,
        path::Path,
        sequence::Sequence,
    };

    use super::PySequenceGraph;
    use crate::python_api::locus::GraphLocusExt as _;

    /// Wraps `sequence_graph_id` as the `PySequenceGraph` a `Repository` would hand back.
    fn sequence_graph(context: &DbContext, block_group_id: HashId, name: &str) -> PySequenceGraph {
        PySequenceGraph {
            id: block_group_id,
            collection_name: "test".to_string(),
            sample_name: "test".to_string(),
            name: name.to_string(),
            context: Some(context.clone()),
        }
    }

    /// A second, unrelated sequence graph named "chr2" with a single-node path.
    fn second_sequence_graph(context: &DbContext) -> HashId {
        let conn = context.graph().conn();
        let block_group = create_bg(conn, "test", "test", "chr2");
        let sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("GGGGG")
            .save(conn)
            .expect("should save sequence");
        let node_id = Node::create(conn, &sequence.hash, &HashId::convert_str("chr2-node"))
            .expect("should create node");
        let edge0 = Edge::create(
            conn,
            PATH_START_NODE_ID,
            0,
            Strand::Forward,
            node_id,
            0,
            Strand::Forward,
        )
        .expect("should create edge");
        let edge1 = Edge::create(
            conn,
            node_id,
            5,
            Strand::Forward,
            PATH_END_NODE_ID,
            0,
            Strand::Forward,
        )
        .expect("should create edge");
        BlockGroupEdge::bulk_create(
            conn,
            &[
                BlockGroupEdgeData {
                    block_group_id: block_group.id,
                    edge_id: edge0.id,
                    chromosome_index: 0,
                    phased: 0,
                },
                BlockGroupEdgeData {
                    block_group_id: block_group.id,
                    edge_id: edge1.id,
                    chromosome_index: 0,
                    phased: 0,
                },
            ],
        );
        Path::create(conn, "chr2", &block_group.id, &[edge0.id, edge1.id])
            .expect("should create current path");
        block_group.id
    }

    #[test]
    fn test_region_resolves_locus_read_only() {
        let context = setup_gen_on_disk();
        let conn = context.graph().conn();
        let (block_group_id, _path) = setup_block_group(conn);
        let graph = sequence_graph(&context, block_group_id, "chr1");

        // The fixture path reads "AAAAAAAAAA" then "TTTTTTTTTT", so 8-13 covers the
        // last two bases of the first node and the first three of the second.
        let locus = graph.region("chr1:8-13").expect("should resolve region");
        let graph_locus = locus.graph_locus();
        assert_eq!(graph_locus.length(), 5);
        assert_eq!(graph_locus.sequence(conn, context.workspace()), b"AATTT");

        // Resolving is read-only: the current path is untouched.
        assert_eq!(
            BlockGroup::get_current_path(conn, &block_group_id, None)
                .expect("should still have a current path")
                .sequence(conn, context.workspace(), None)
                .expect("should read the current path"),
            "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG"
        );
    }

    #[test]
    fn test_region_outside_sequence_graph_is_a_value_error() {
        let context = setup_gen_on_disk();
        let conn = context.graph().conn();
        let (block_group_id, _path) = setup_block_group(conn);
        second_sequence_graph(&context);
        let graph = sequence_graph(&context, block_group_id, "chr1");

        let error = match graph.region("chr2:0-5") {
            Ok(_) => panic!("region in a different sequence graph should fail"),
            Err(error) => error,
        };
        assert!(
            error
                .to_string()
                .contains("resolves outside sequence graph 'chr1'"),
            "unexpected error message: {error}"
        );
    }
}

/// The endless iterator returned by ``SequenceGraph.random_sequences()``.
#[gen_stub_pyclass]
#[pyclass(name = "RandomSequences", unsendable)]
pub struct PyRandomSequences {
    sequences: RandomSequences,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyRandomSequences {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<PySequence> {
        slf.sequences.next().map(PySequence::unnamed)
    }
}
