use r#gen::{
    graphs::graph_search::GraphPos,
    views::annotation_track::{AnnotationSpan, annotation_span_from_graph_locus},
};
use gen_graph::GraphNode;
use gen_models::locus::GraphLocus;
use pyo3::prelude::*;

use super::block::{PyGraphNode, PyGraphNodeSlice};

/// A position in the graph: a specific node plus a byte offset within that
/// node's local text (`0..node.length()`).
///
/// Returned by `PyGraphLocus.start()` and `PyGraphLocus.end()`.
/// Pass directly to `GenGraphWidget.go_to()`.
#[pyclass(name = "GraphPos")]
#[derive(Clone)]
pub struct PyGraphPos {
    pub inner: GraphPos,
}

impl PyGraphPos {
    pub fn new(block: gen_graph::GraphNode, offset: usize) -> Self {
        Self {
            inner: GraphPos { block, offset },
        }
    }
}

#[pymethods]
impl PyGraphPos {
    /// The graph node this position is inside.
    #[getter]
    fn block(&self) -> PyGraphNode {
        let n = self.inner.block;
        PyGraphNode::new(n.node_id, n.sequence_start, n.sequence_end)
    }

    /// Byte offset within the node's local text (`0..node.length()`).
    #[getter]
    fn offset(&self) -> usize {
        self.inner.offset
    }

    fn __repr__(&self) -> String {
        let n = self.inner.block;
        let hash = format!("{}", n.node_id);
        format!(
            "GraphPos({}[{}..{}] +{})",
            &hash[..8],
            n.sequence_start,
            n.sequence_end,
            self.inner.offset
        )
    }
}

/// A linear sequence of nodes in graph space, anchored by byte offsets.
///
/// Obtain via `repo.search(block_group, query)`.
/// Pass `.start()` or `.end()` to `widget.go_to()`.
#[pyclass(name = "GraphLocus")]
pub struct PyGraphLocus {
    pub inner: GraphLocus,
}

impl PyGraphLocus {
    pub fn from_locus(l: GraphLocus) -> Self {
        Self { inner: l }
    }

    /// Return the raw node sequence for highlight operations.
    pub fn locus_nodes(&self) -> Vec<GraphNode> {
        self.inner.slices.iter().map(|s| s.block).collect()
    }
}

#[pymethods]
impl PyGraphLocus {
    /// Position of the first matched byte (start of the locus).
    fn start(&self) -> PyGraphPos {
        let s = &self.inner.slices[0];
        PyGraphPos::new(s.block, s.start)
    }

    /// Position one past the last matched byte (exclusive end of the locus).
    fn end(&self) -> PyGraphPos {
        let s = self.inner.slices.last().unwrap();
        PyGraphPos::new(s.block, s.end)
    }

    /// Ordered sequence of block slices that span this locus.
    ///
    /// Each `NodeSlice` carries the underlying block, local byte offsets
    /// within that block, and the strand for that slice.
    #[getter]
    fn slices(&self) -> Vec<PyGraphNodeSlice> {
        self.inner
            .slices
            .iter()
            .map(|s| PyGraphNodeSlice::from_slice(*s))
            .collect()
    }

    fn __repr__(&self) -> String {
        let first = &self.inner.slices[0];
        let last = self.inner.slices.last().unwrap();
        let sh = format!("{}", first.block.node_id);
        let eh = format!("{}", last.block.node_id);
        format!(
            "GraphLocus({}[{}..{}]+{} → {}[{}..{}]+{}, {} blocks)",
            &sh[..8],
            first.block.sequence_start,
            first.block.sequence_end,
            first.start,
            &eh[..8],
            last.block.sequence_start,
            last.block.sequence_end,
            last.end,
            self.inner.slices.len(),
        )
    }
}

/// A named annotation in graph space, built from one or more `GraphLocus` objects.
///
/// Create with ``Annotation(locus, name)`` for a single hit or
/// ``Annotation([locus1, locus2], name)`` to combine multiple hits into one
/// named span.  Pass a list of ``Annotation`` objects to
/// ``widget.add_annotation_track()``.
#[pyclass(name = "Annotation")]
#[derive(Clone)]
pub struct PyAnnotation {
    pub inner: AnnotationSpan,
    pub(crate) locus: GraphLocus,
}

#[pymethods]
impl PyAnnotation {
    #[new]
    fn new(locus: PyRef<PyGraphLocus>, name: &str) -> Self {
        let span = annotation_span_from_graph_locus(&locus.inner, name);
        PyAnnotation {
            inner: span,
            locus: locus.inner.clone(),
        }
    }

    #[getter]
    fn locus(&self) -> PyGraphLocus {
        PyGraphLocus::from_locus(self.locus.clone())
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    fn __repr__(&self) -> String {
        format!(
            "Annotation(name={:?}, segments={})",
            self.inner.name,
            self.inner.segments.len()
        )
    }
}
