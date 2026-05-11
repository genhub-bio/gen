use r#gen::{
    graphs::graph_search::{GraphLocus, GraphPos},
    views::annotation_track::{AnnotationSegment, AnnotationSpan, graphlocus_to_annotation_span},
};
use gen_core::{HashId, Strand};
use gen_graph::GraphNode;
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
        self.inner.blocks.clone()
    }
}

#[pymethods]
impl PyGraphLocus {
    /// Position of the first matched byte (start of the locus).
    fn start(&self) -> PyGraphPos {
        PyGraphPos::new(self.inner.blocks[0], self.inner.start_offset)
    }

    /// Position one past the last matched byte (exclusive end of the locus).
    fn end(&self) -> PyGraphPos {
        PyGraphPos::new(*self.inner.blocks.last().unwrap(), self.inner.end_offset)
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
            .blocks
            .iter()
            .map(|n| PyBlock::new(n.node_id, n.sequence_start, n.sequence_end))
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
        let sn = self.inner.blocks[0];
        let en = *self.inner.blocks.last().unwrap();
        let sh = format!("{}", sn.node_id);
        let eh = format!("{}", en.node_id);
        let strand = match self.inner.strand {
            Strand::Forward => "+",
            Strand::Reverse => "-",
            _ => ".",
        };
        format!(
            "GraphLocus({}[{}..{}]+{} → {}[{}..{}]+{}, {} blocks, strand={})",
            &sh[..8],
            sn.sequence_start,
            sn.sequence_end,
            self.inner.start_offset,
            &eh[..8],
            en.sequence_start,
            en.sequence_end,
            self.inner.end_offset,
            self.inner.blocks.len(),
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
    /// Source loci retained for viewport-aware midpoint computation at render time.
    pub loci: Vec<GraphLocus>,
}

#[pymethods]
impl PyAnnotation {
    #[new]
    fn new(locus_or_loci: &Bound<'_, PyAny>, name: &str) -> PyResult<Self> {
        if let Ok(single) = locus_or_loci.extract::<PyRef<PyGraphLocus>>() {
            let span = graphlocus_to_annotation_span(&single.inner, name);
            let loci = vec![single.inner.clone()];
            Ok(PyAnnotation { inner: span, loci })
        } else if let Ok(list) = locus_or_loci.extract::<Vec<PyRef<PyGraphLocus>>>() {
            let loci: Vec<GraphLocus> = list.iter().map(|l| l.inner.clone()).collect();
            let segments: Vec<AnnotationSegment> = loci
                .iter()
                .flat_map(|l| graphlocus_to_annotation_span(l, name).segments)
                .collect();
            let span = AnnotationSpan {
                id: HashId::convert_str(name),
                name: name.to_string(),
                segments,
            };
            Ok(PyAnnotation { inner: span, loci })
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
