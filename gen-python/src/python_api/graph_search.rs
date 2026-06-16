use r#gen::graphs::graph_search::GraphPos;
use gen_core::Strand;
use gen_models::locus::GraphLocus;
use pyo3::prelude::*;

use super::graph_node::{PyGraphNode, PyGraphNodeSlice};

/// A position in the graph: a specific node plus a byte offset within that
/// node's local text (`0..node.length()`).
///
/// Returned by `Locus.start()` and `Locus.end()`.
/// Pass directly to `GraphWidget.go_to()`.
#[pyclass(name = "Position")]
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
    fn node(&self) -> PyGraphNode {
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
        let h = format!("{}", n.node_id);
        let hash8 = &h[..8.min(h.len())];
        format!(
            "GraphPos({}[{}:{}][{}])",
            hash8, n.sequence_start, n.sequence_end, self.inner.offset
        )
    }

    fn __str__(&self) -> String {
        let n = self.inner.block;
        let h = format!("{}", n.node_id);
        let hash8 = &h[..8.min(h.len())];
        format!(
            "{}:{}-{}+{}",
            hash8, n.sequence_start, n.sequence_end, self.inner.offset
        )
    }

    fn __hash__(&self) -> isize {
        let n = self.inner.block;
        let mut hash: isize = 0;
        for &b in &n.node_id.0 {
            hash = hash.wrapping_mul(31).wrapping_add(b as isize);
        }
        hash = hash
            .wrapping_mul(31)
            .wrapping_add(n.sequence_start as isize);
        hash = hash.wrapping_mul(31).wrapping_add(n.sequence_end as isize);
        hash = hash
            .wrapping_mul(31)
            .wrapping_add(self.inner.offset as isize);
        hash
    }
}

/// A linear sequence of nodes in graph space, anchored by byte offsets.
///
/// Obtain via `sg.search(query)` or `repo.search(query)`.
/// Pass `.start()` or `.end()` to `widget.go_to()`.
#[pyclass(name = "Locus")]
pub struct PyGraphLocus {
    pub inner: GraphLocus,
}

impl PyGraphLocus {
    pub fn from_locus(l: GraphLocus) -> Self {
        Self { inner: l }
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

    /// Strand of this locus: ``"+"`` forward, ``"-"`` reverse, ``"mixed"`` if slices differ, ``"."`` if empty.
    #[getter]
    fn strand(&self) -> &str {
        let mut iter = self.inner.slices.iter().map(|s| s.strand);
        match iter.next() {
            None => ".",
            Some(first) => {
                if iter.all(|s| s == first) {
                    match first {
                        Strand::Forward => "+",
                        Strand::Reverse => "-",
                        _ => ".",
                    }
                } else {
                    "mixed"
                }
            }
        }
    }

    fn __repr__(&self) -> String {
        let strand = self.strand();
        let segs: Vec<String> = self
            .inner
            .slices
            .iter()
            .map(|s| {
                let h = format!("{}", s.block.node_id);
                let hash8 = &h[..8.min(h.len())];
                let block_len = (s.block.sequence_end - s.block.sequence_start) as usize;
                let full_width = s.start == 0 && s.end == block_len;
                if full_width {
                    format!(
                        "{}[{}:{}][:]",
                        hash8, s.block.sequence_start, s.block.sequence_end
                    )
                } else {
                    format!(
                        "{}[{}:{}][{}:{}]",
                        hash8, s.block.sequence_start, s.block.sequence_end, s.start, s.end
                    )
                }
            })
            .collect();
        format!("Locus([{}], strand='{}')", segs.join(", "), strand)
    }

    fn __str__(&self) -> String {
        let segs: Vec<String> = self
            .inner
            .slices
            .iter()
            .map(|s| {
                let h = format!("{}", s.block.node_id);
                let hash8 = &h[..8.min(h.len())];
                format!(
                    "{}:{}-{}",
                    hash8, s.block.sequence_start, s.block.sequence_end
                )
            })
            .collect();
        let coords = segs.join(",");
        match self.strand() {
            "+" => format!("{}(+)", coords),
            "-" => format!("{}(-)", coords),
            s => format!("{}({})", coords, s),
        }
    }

    fn __hash__(&self) -> isize {
        let mut hash: isize = 0;
        for s in &self.inner.slices {
            for &b in &s.block.node_id.0 {
                hash = hash.wrapping_mul(31).wrapping_add(b as isize);
            }
            hash = hash
                .wrapping_mul(31)
                .wrapping_add(s.block.sequence_start as isize);
            hash = hash
                .wrapping_mul(31)
                .wrapping_add(s.block.sequence_end as isize);
            hash = hash.wrapping_mul(31).wrapping_add(s.start as isize);
            hash = hash.wrapping_mul(31).wrapping_add(s.end as isize);
            hash = hash.wrapping_mul(31).wrapping_add(s.strand as isize);
        }
        hash
    }
}
