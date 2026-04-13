use r#gen::graphs::graph_search::{GraphLocus, GraphPos};
use gen_graph::GraphNode;
use pyo3::prelude::*;

use super::node_key::PyBlock;

/// A position in the graph: a specific node plus a byte offset within that
/// node's local text (`0..node.length()`).
///
/// Returned by `PyGraphLocus.start()` and `PyGraphLocus.end()`.
/// Pass directly to `GenGraphWidget.go_to()`.
#[pyclass]
#[derive(Clone)]
pub struct PyGraphPos {
    pub inner: GraphPos,
}

impl PyGraphPos {
    pub fn new(node: gen_graph::GraphNode, offset: usize) -> Self {
        Self {
            inner: GraphPos { node, offset },
        }
    }
}

#[pymethods]
impl PyGraphPos {
    /// The graph node this position is inside.
    #[getter]
    fn block(&self) -> PyBlock {
        let n = self.inner.node;
        PyBlock::new(n.node_id, n.sequence_start, n.sequence_end)
    }

    /// Byte offset within the node's local text (`0..node.length()`).
    #[getter]
    fn offset(&self) -> usize {
        self.inner.offset
    }

    fn __repr__(&self) -> String {
        let n = self.inner.node;
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
#[pyclass]
pub struct PyGraphLocus {
    pub inner: GraphLocus,
}

impl PyGraphLocus {
    pub fn from_locus(l: GraphLocus) -> Self {
        Self { inner: l }
    }

    /// Return the raw node sequence for highlight operations.
    pub fn locus_nodes(&self) -> Vec<GraphNode> {
        self.inner.nodes.clone()
    }
}

#[pymethods]
impl PyGraphLocus {
    /// Position of the first matched byte (start of the locus).
    fn start(&self) -> PyGraphPos {
        PyGraphPos::new(self.inner.nodes[0], self.inner.start_offset)
    }

    /// Position one past the last matched byte (exclusive end of the locus).
    fn end(&self) -> PyGraphPos {
        PyGraphPos::new(*self.inner.nodes.last().unwrap(), self.inner.end_offset)
    }

    /// Ordered sequence of nodes that span this locus.
    #[getter]
    fn nodes(&self) -> Vec<PyBlock> {
        self.inner
            .nodes
            .iter()
            .map(|n| PyBlock::new(n.node_id, n.sequence_start, n.sequence_end))
            .collect()
    }

    fn __repr__(&self) -> String {
        let sn = self.inner.nodes[0];
        let en = *self.inner.nodes.last().unwrap();
        let sh = format!("{}", sn.node_id);
        let eh = format!("{}", en.node_id);
        format!(
            "GraphLocus({}[{}..{}]+{} → {}[{}..{}]+{}, {} nodes)",
            &sh[..8],
            sn.sequence_start,
            sn.sequence_end,
            self.inner.start_offset,
            &eh[..8],
            en.sequence_start,
            en.sequence_end,
            self.inner.end_offset,
            self.inner.nodes.len(),
        )
    }
}
