use r#gen::{
    graphs::graph_search::GraphPos,
    views::annotation_track::{
        AnnotationSegment, AnnotationSpan, annotation_span_from_graph_locus,
    },
};
use gen_core::{HashId, Strand};
use gen_graph::GraphNode;
use gen_models::locus::GraphLocus;
use pyo3::prelude::*;

use super::block::PyBlock;

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
    fn block(&self) -> PyBlock {
        let n = self.inner.block;
        PyBlock::new(n.node_id, n.sequence_start, n.sequence_end)
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

    /// Ordered sequence of blocks that span this locus.
    ///
    /// Each `PyBlock` carries `(node_id, sequence_start, sequence_end)`.
    /// Note that `node_id` alone is **not** unique — multiple blocks can be
    /// carved from the same graph node.  Use all three fields together to
    /// uniquely identify a block.
    #[getter]
    fn blocks(&self) -> Vec<PyBlock> {
        self.inner
            .slices
            .iter()
            .map(|s| {
                PyBlock::new(
                    s.block.node_id,
                    s.block.sequence_start,
                    s.block.sequence_end,
                )
            })
            .collect()
    }

    /// Strand of the matched sequence: `"forward"`, `"reverse"`, or `"unknown"`.
    #[getter]
    fn strand(&self) -> &str {
        match self.inner.strand {
            Strand::Forward => "forward",
            Strand::Reverse => "reverse",
            _ => "unknown",
        }
    }

    fn __repr__(&self) -> String {
        let first = &self.inner.slices[0];
        let last = self.inner.slices.last().unwrap();
        let sh = format!("{}", first.block.node_id);
        let eh = format!("{}", last.block.node_id);
        let strand = match self.inner.strand {
            Strand::Forward => "+",
            Strand::Reverse => "-",
            _ => ".",
        };
        format!(
            "GraphLocus({}[{}..{}]+{} → {}[{}..{}]+{}, {} blocks, strand={})",
            &sh[..8],
            first.block.sequence_start,
            first.block.sequence_end,
            first.start,
            &eh[..8],
            last.block.sequence_start,
            last.block.sequence_end,
            last.end,
            self.inner.slices.len(),
            strand,
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
}

#[pymethods]
impl PyAnnotation {
    #[new]
    fn new(locus_or_loci: &Bound<'_, PyAny>, name: &str) -> PyResult<Self> {
        if let Ok(single) = locus_or_loci.extract::<PyRef<PyGraphLocus>>() {
            let span = annotation_span_from_graph_locus(&single.inner, name);
            Ok(PyAnnotation { inner: span })
        } else if let Ok(list) = locus_or_loci.extract::<Vec<PyRef<PyGraphLocus>>>() {
            let segments: Vec<AnnotationSegment> = list
                .iter()
                .flat_map(|l| annotation_span_from_graph_locus(&l.inner, name).segments)
                .collect();
            let span = AnnotationSpan {
                id: HashId::convert_str(name),
                name: name.to_string(),
                segments,
            };
            Ok(PyAnnotation { inner: span })
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "first argument must be a GraphLocus or list[GraphLocus]",
            ))
        }
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
