use gen_core::{HashId, Strand};
use gen_graph::{GraphNode, GraphNodeSlice};
use gen_models::{db::DbContext, locus::GraphLocus};
use pyo3::{
    Bound, Py, PyAny, PyRef, PyResult,
    exceptions::{PyIndexError, PyOverflowError, PyRuntimeError, PyValueError},
    pyclass, pymethods,
    types::{PyAnyMethods as _, PySlice, PySliceMethods as _},
};

use super::{
    block_group::PySequenceGraph, graph_node::PyGraphNodeSlice, locus::GraphLocusExt as _,
    position::PyPosition,
};

/// An ordered span in graph space, independent of how nodes are split for display.
///
/// Obtain via `sg.search(query)` or `repo.search(query)`.
/// Pass it, or its `.start()` or `.end()`, to `widget.go_to()`.
/// `.sequence` reads the bases it covers fresh from the database on every access.
#[pyclass(name = "Locus", unsendable)]
#[derive(Clone)]
pub struct PyGraphLocus {
    ranges: Vec<NodeRange>,
    // Presentation snapshot only: preserves the nodes and local offsets exposed by slices
    // and Positions. Editing, annotations, equality, and hashing use absolute ranges.
    presentation: GraphLocus,
    // Database connection used to read `.sequence` fresh on every access. `None` only for
    // loci built without a real Repository, such as internal Rust-side tests.
    context: Option<DbContext>,
    // The sequence graph this locus came from, which the positions it hands out step through.
    // `None` for loci built without one, such as an `Annotation` made from a locus.
    sequence_graph: Option<PySequenceGraph>,
}

#[derive(Clone, PartialEq, Eq)]
struct NodeRange {
    node_id: HashId,
    start: i64,
    end: i64,
    strand: Strand,
}

impl PyGraphLocus {
    /// Build a locus without a database connection to read its sequence from. Only
    /// internal Rust-side tests should use this; real Python-facing loci should carry
    /// a context via [`Self::with_context`] so that `.sequence` works.
    pub fn from_locus(locus: GraphLocus) -> Self {
        Self::with_context(locus, None)
    }

    pub fn with_context(locus: GraphLocus, context: Option<DbContext>) -> Self {
        let ranges = locus
            .canonical()
            .slices
            .iter()
            .map(|slice| NodeRange {
                node_id: slice.block.node_id,
                start: slice.block.sequence_start,
                end: slice.block.sequence_end,
                strand: slice.strand,
            })
            .collect();
        Self {
            ranges,
            presentation: locus,
            context,
            sequence_graph: None,
        }
    }

    /// The database connection `.sequence` reads through, if this locus has one.
    pub(crate) fn context(&self) -> Option<DbContext> {
        self.context.clone()
    }

    /// The sequence graph this locus came from, if it has one.
    pub(crate) fn sequence_graph(&self) -> Option<PySequenceGraph> {
        self.sequence_graph.clone()
    }

    /// This locus attached to `sequence_graph`, so its positions can step through it.
    pub(crate) fn attached_to(mut self, sequence_graph: Option<PySequenceGraph>) -> Self {
        self.sequence_graph = sequence_graph;
        self
    }

    /// Adapt the absolute address to the graph algorithms' slice representation.
    pub(crate) fn graph_locus(&self) -> GraphLocus {
        GraphLocus {
            slices: self
                .ranges
                .iter()
                .map(|range| {
                    GraphNodeSlice::full(
                        GraphNode {
                            node_id: range.node_id,
                            sequence_start: range.start,
                            sequence_end: range.end,
                        },
                        range.strand,
                    )
                })
                .collect(),
        }
    }

    /// Resolve a reading-order offset while retaining the block used to display the position.
    pub(crate) fn position_at(&self, mut offset: usize) -> PyResult<PyPosition> {
        for slice in &self.presentation.slices {
            let length = slice.end - slice.start;
            if offset < length {
                let local_offset = if slice.strand == Strand::Reverse {
                    slice.end - 1 - offset
                } else {
                    slice.start + offset
                };
                // in_block adds sequence_start to obtain the stable node coordinate.
                return Ok(
                    PyPosition::in_block(slice.block, local_offset, slice.strand)
                        .attached_to(self.sequence_graph.clone()),
                );
            }
            offset -= length;
        }
        Err(PyIndexError::new_err("Locus index out of range"))
    }
}

#[pymethods]
impl PyGraphLocus {
    /// The first position of the locus in reading order.
    fn start(&self) -> PyResult<PyPosition> {
        let slice = self
            .presentation
            .slices
            .iter()
            .find(|slice| slice.start < slice.end)
            .ok_or_else(|| PyValueError::new_err("Locus is empty"))?;
        // A reverse-strand slice reads toward lower offsets, so it starts at its highest one.
        let offset = if slice.strand == Strand::Reverse {
            slice.end - 1
        } else {
            slice.start
        };
        Ok(PyPosition::in_block(slice.block, offset, slice.strand)
            .attached_to(self.sequence_graph.clone()))
    }

    /// The last position of the locus in reading order.
    fn end(&self) -> PyResult<PyPosition> {
        let slice = self
            .presentation
            .slices
            .iter()
            .rev()
            .find(|slice| slice.start < slice.end)
            .ok_or_else(|| PyValueError::new_err("Locus is empty"))?;
        let offset = if slice.strand == Strand::Reverse {
            slice.start
        } else {
            slice.end - 1
        };
        Ok(PyPosition::in_block(slice.block, offset, slice.strand)
            .attached_to(self.sequence_graph.clone()))
    }

    /// Ordered node slices as displayed when this locus was obtained.
    ///
    /// Each `NodeSlice` carries a node, local offsets, and a strand. Later edits
    /// can split nodes without changing this locus's identity.
    #[getter]
    fn slices(&self) -> Vec<PyGraphNodeSlice> {
        self.presentation
            .slices
            .iter()
            .map(|s| PyGraphNodeSlice::from_slice(*s))
            .collect()
    }

    /// Length of the sequence this locus covers.
    fn __len__(&self) -> usize {
        self.ranges
            .iter()
            .map(|range| (range.end - range.start) as usize)
            .sum()
    }

    /// Index a Position or slice a contiguous Locus in reading order.
    ///
    /// Negative indices and omitted bounds follow Python conventions. Booleans act as
    /// integers. Empty slices raise ``IndexError``; steps other than 1 raise ``ValueError``.
    fn __getitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let python = key.py();
        let length = self.__len__() as isize;
        if let Ok(slice) = key.downcast::<PySlice>() {
            let indices = slice.indices(length)?;
            if indices.step != 1 {
                return Err(PyValueError::new_err("Locus slices require a step of 1"));
            }
            return Py::new(
                python,
                self.slice(indices.start as usize, indices.stop as usize)?,
            )
            .map(Py::into_any);
        }
        let index = key.extract::<isize>().map_err(|error| {
            if error.is_instance_of::<PyOverflowError>(python) {
                PyIndexError::new_err("Locus index out of range")
            } else {
                error
            }
        })?;
        let offset = if index < 0 { length + index } else { index };
        if offset < 0 || offset >= length {
            return Err(PyIndexError::new_err("Locus index out of range"));
        }
        Py::new(python, self.position_at(offset as usize)?).map(Py::into_any)
    }

    /// The same positions read from the opposite strand.
    fn reverse_complement(&self) -> PyGraphLocus {
        PyGraphLocus::with_context(self.presentation.reverse_complement(), self.context.clone())
            .attached_to(self.sequence_graph.clone())
    }

    /// Sub-locus covering positions ``start:end`` in reading order.
    ///
    /// Raises ``IndexError`` when the slice is empty or runs past the end of the locus.
    fn slice(&self, start: usize, end: usize) -> PyResult<PyGraphLocus> {
        self.presentation
            .slice(start, end)
            .map(|locus| {
                PyGraphLocus::with_context(locus, self.context.clone())
                    .attached_to(self.sequence_graph.clone())
            })
            .ok_or_else(|| {
                PyIndexError::new_err(format!(
                    "slice {start}:{end} is empty or outside a locus of length {}",
                    self.__len__()
                ))
            })
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<PyGraphLocus>>()
            .is_ok_and(|other| other.ranges == self.ranges)
    }

    /// Strand of this locus: ``"+"`` forward, ``"-"`` reverse, ``"mixed"`` if slices differ, ``"."`` if empty.
    #[getter]
    fn strand(&self) -> &str {
        let mut iter = self.ranges.iter().map(|range| range.strand);
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
            .presentation
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

    /// The sequence text this locus covers, read fresh from the database on every call.
    #[getter]
    fn sequence(&self) -> PyResult<String> {
        let context = self.context.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("locus has no database connection to read its sequence from")
        })?;
        let bytes = self
            .presentation
            .sequence(context.graph().conn(), context.workspace());
        Ok(String::from_utf8(bytes).expect("sequence bytes should be valid UTF-8"))
    }

    fn __str__(&self) -> PyResult<String> {
        self.sequence()
    }

    fn __hash__(&self) -> isize {
        let mut hash: isize = 0;
        for range in &self.ranges {
            for &b in &range.node_id.0 {
                hash = hash.wrapping_mul(31).wrapping_add(b as isize);
            }
            hash = hash.wrapping_mul(31).wrapping_add(range.start as isize);
            hash = hash.wrapping_mul(31).wrapping_add(range.end as isize);
            hash = hash.wrapping_mul(31).wrapping_add(range.strand as isize);
        }
        hash
    }
}

#[cfg(test)]
mod tests {
    use gen_core::{HashId, Strand};
    use gen_graph::{GenGraph, GraphNode, GraphNodeSlice};
    use gen_models::locus::GraphLocus;
    use pyo3::{
        Py, Python,
        ffi::c_str,
        prepare_freethreaded_python,
        types::{PyDict, PyDictMethods as _},
    };

    use super::PyGraphLocus;
    use crate::python_api::locus::GraphLocusExt as _;

    #[test]
    fn test_locus_identity_ignores_display_node_boundaries() {
        let node_id = HashId::convert_str("same-sequence");
        let whole = GraphLocus {
            slices: vec![GraphNodeSlice {
                block: GraphNode {
                    node_id,
                    sequence_start: 100,
                    sequence_end: 120,
                },
                start: 5,
                end: 18,
                strand: Strand::Forward,
            }],
        };
        let carved = GraphLocus {
            slices: vec![
                GraphNodeSlice {
                    block: GraphNode {
                        node_id,
                        sequence_start: 100,
                        sequence_end: 110,
                    },
                    start: 5,
                    end: 10,
                    strand: Strand::Forward,
                },
                GraphNodeSlice {
                    block: GraphNode {
                        node_id,
                        sequence_start: 110,
                        sequence_end: 120,
                    },
                    start: 0,
                    end: 8,
                    strand: Strand::Forward,
                },
            ],
        };
        // Merging carved blocks back into their whole node is covered by
        // GraphLocusExt::canonical's own tests; here we only check that
        // PyGraphLocus's identity, hashing, and offset resolution agree
        // regardless of which display form produced it.
        for (left, right) in [
            (whole.clone(), carved.clone()),
            (whole.reverse_complement(), carved.reverse_complement()),
        ] {
            let left = PyGraphLocus::from_locus(left);
            let right = PyGraphLocus::from_locus(right);
            assert!(left.ranges == right.ranges);
            assert_eq!(left.__hash__(), right.__hash__());
            assert_eq!(
                right.slices().len(),
                2,
                "presentation retains the original nodes"
            );
            for offset in 0..13 {
                assert_eq!(
                    left.position_at(offset)
                        .expect("should resolve the offset")
                        .position,
                    right
                        .position_at(offset)
                        .expect("should resolve the offset")
                        .position,
                );
            }
        }
    }

    #[test]
    fn test_position_at_resolves_absolute_coordinates_on_current_nodes() {
        let node_id = HashId::convert_str("split");
        let left_block = GraphNode {
            node_id,
            sequence_start: 100,
            sequence_end: 110,
        };
        let right_block = GraphNode {
            node_id,
            sequence_start: 110,
            sequence_end: 120,
        };
        let mut graph = GenGraph::new();
        graph.add_node(left_block);
        graph.add_node(right_block);
        let locus = GraphLocus {
            slices: vec![GraphNodeSlice::full(
                GraphNode {
                    node_id,
                    sequence_start: 105,
                    sequence_end: 117,
                },
                Strand::Forward,
            )],
        };
        for (locus, first, middle) in [
            (locus.clone(), (left_block, 5), (right_block, 1)),
            (
                locus.reverse_complement(),
                (right_block, 6),
                (right_block, 0),
            ),
        ] {
            let target = PyGraphLocus::from_locus(locus);
            for (offset, expected) in [(0, first), (target.__len__() / 2, middle)] {
                let position = target
                    .position_at(offset)
                    .expect("should resolve a base in reading order");
                assert_eq!(position.locate(&graph), Some(expected));
            }
        }
    }

    #[test]
    fn test_python_indexing_empty_loci_and_zero_length_slices() {
        prepare_freethreaded_python();
        Python::with_gil(|python| {
            let block = GraphNode {
                node_id: HashId::convert_str("indexing"),
                sequence_start: 100,
                sequence_end: 110,
            };
            let empty_slice = GraphNodeSlice {
                block,
                start: 3,
                end: 3,
                strand: Strand::Forward,
            };
            let locals = PyDict::new(python);
            for (name, slices) in [
                ("empty", vec![]),
                ("zero_length", vec![empty_slice]),
                (
                    "mixed",
                    vec![
                        empty_slice,
                        GraphNodeSlice {
                            start: 2,
                            end: 4,
                            ..empty_slice
                        },
                        empty_slice,
                        GraphNodeSlice {
                            start: 6,
                            end: 9,
                            strand: Strand::Reverse,
                            ..empty_slice
                        },
                        empty_slice,
                    ],
                ),
            ] {
                let locus = Py::new(python, PyGraphLocus::from_locus(GraphLocus { slices }))
                    .expect("should create an internal locus");
                locals
                    .set_item(name, locus)
                    .expect("should expose the locus to Python");
            }
            python
                .run(
                    c_str!(
                        "
for locus in (empty, zero_length):
    assert len(locus) == 0
    for key in (0, -1, slice(None)):
        try:
            locus[key]
        except IndexError:
            pass
        else:
            raise AssertionError('empty loci must reject access')
    for endpoint in (locus.start, locus.end):
        try:
            endpoint()
        except ValueError:
            pass
        else:
            raise AssertionError('existing endpoint errors must be preserved')
assert [mixed[i].offset for i in range(len(mixed))] == [2, 3, 8, 7, 6]
assert [mixed[i].strand for i in range(len(mixed))] == ['+', '+', '-', '-', '-']
assert mixed[0] == mixed.start()
assert mixed[-1] == mixed.end()
assert mixed[2].node.sequence_start + mixed[2].offset == 108
assert [mixed[1:4][i].offset for i in range(3)] == [3, 8, 7]
"
                    ),
                    Some(&locals),
                    Some(&locals),
                )
                .expect("should index empty and mixed-strand loci through Python");
        });
    }
}
