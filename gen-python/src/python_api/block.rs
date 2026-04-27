use gen_core::{PATH_END_NODE_ID, PATH_START_NODE_ID};
use r#gen::core::HashId;
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::PySlice,
};

use super::hash_id::PyHashId;

#[pyclass(name = "Block", subclass)]
#[derive(Clone)]
pub struct PyBlock {
    pub node_id: HashId,
    #[pyo3(get)]
    pub sequence_start: i64,
    #[pyo3(get)]
    pub sequence_end: i64,
    /// Full underlying node sequence (optional payload, not part of identity).
    pub node_seq: Option<String>,
    /// Already-sliced block sequence (optional payload, not part of identity).
    pub block_seq: Option<String>,
}

impl PyBlock {
    /// Priority: _node_sequence[start:end] > _block_sequence > None.
    pub fn display_sequence(&self) -> Option<String> {
        if let Some(ns) = &self.node_seq {
            let s = self.sequence_start as usize;
            let e = self.sequence_end as usize;
            Some(ns[s..e].to_string())
        } else {
            self.block_seq.clone()
        }
    }

    /// Returns the sequence string to use for DB insertion, or None if unavailable.
    /// For _node_sequence: always returns it.
    /// For _block_sequence: only if coordinates are normalized to 0..len.
    pub fn node_sequence_for_db(&self) -> Option<&str> {
        if let Some(ns) = &self.node_seq {
            Some(ns.as_str())
        } else if let Some(bs) = &self.block_seq {
            if self.sequence_start == 0 && self.sequence_end == bs.len() as i64 {
                Some(bs.as_str())
            } else {
                None
            }
        } else {
            None
        }
    }
}

#[pymethods]
impl PyBlock {
    /// Block(sequence, node_id=None)
    ///
    /// Create a staged block. sequence_start=0, sequence_end=len(sequence).
    /// node_id defaults to a fresh UUID-7 if not provided.
    #[new]
    #[pyo3(signature = (sequence, node_id=None))]
    pub fn new(sequence: String, node_id: Option<PyHashId>) -> Self {
        let seq_len = sequence.len() as i64;
        PyBlock {
            node_id: node_id.map(|h| h.hash_id).unwrap_or_else(HashId::uuid7),
            sequence_start: 0,
            sequence_end: seq_len,
            node_seq: Some(sequence),
            block_seq: None,
        }
    }

    /// Block.from_node_sequence(sequence, start=None, end=None, node_id=None)
    ///
    /// sequence is the full underlying node sequence. start/end select the slice.
    #[staticmethod]
    #[pyo3(signature = (sequence, start=None, end=None, node_id=None))]
    pub fn from_node_sequence(
        sequence: String,
        start: Option<i64>,
        end: Option<i64>,
        node_id: Option<PyHashId>,
    ) -> PyResult<Self> {
        let seq_len = sequence.len() as i64;
        let s = start.unwrap_or(0);
        let e = end.unwrap_or(seq_len);
        if s < 0 {
            return Err(PyValueError::new_err("sequence_start must be >= 0"));
        }
        if e < s {
            return Err(PyValueError::new_err(
                "sequence_end must be >= sequence_start",
            ));
        }
        if e > seq_len {
            return Err(PyValueError::new_err(
                "sequence_end exceeds len(_node_sequence)",
            ));
        }
        Ok(PyBlock {
            node_id: node_id.map(|h| h.hash_id).unwrap_or_else(HashId::uuid7),
            sequence_start: s,
            sequence_end: e,
            node_seq: Some(sequence),
            block_seq: None,
        })
    }

    /// Block.from_block_sequence(sequence, start=None, end=None, node_id=None)
    ///
    /// sequence is already the represented block slice.
    /// start/end are absolute coordinates on the underlying node.
    /// len(sequence) must equal sequence_end - sequence_start.
    #[staticmethod]
    #[pyo3(signature = (sequence, start=None, end=None, node_id=None))]
    pub fn from_block_sequence(
        sequence: String,
        start: Option<i64>,
        end: Option<i64>,
        node_id: Option<PyHashId>,
    ) -> PyResult<Self> {
        let seq_len = sequence.len() as i64;
        let s = start.unwrap_or(0);
        let e = end.unwrap_or(s + seq_len);
        if s < 0 {
            return Err(PyValueError::new_err("sequence_start must be >= 0"));
        }
        if e < s {
            return Err(PyValueError::new_err(
                "sequence_end must be >= sequence_start",
            ));
        }
        if e - s != seq_len {
            return Err(PyValueError::new_err(
                "len(_block_sequence) must equal sequence_end - sequence_start",
            ));
        }
        Ok(PyBlock {
            node_id: node_id.map(|h| h.hash_id).unwrap_or_else(HashId::uuid7),
            sequence_start: s,
            sequence_end: e,
            node_seq: None,
            block_seq: Some(sequence),
        })
    }

    /// Block.ref(node_id, start, end) — coordinate-only reference, no sequence payload.
    #[staticmethod]
    #[pyo3(name = "ref")]
    pub fn from_ref(node_id: PyHashId, start: i64, end: i64) -> PyResult<Self> {
        if start < 0 {
            return Err(PyValueError::new_err("sequence_start must be >= 0"));
        }
        if end < start {
            return Err(PyValueError::new_err(
                "sequence_end must be >= sequence_start",
            ));
        }
        Ok(PyBlock {
            node_id: node_id.hash_id,
            sequence_start: start,
            sequence_end: end,
            node_seq: None,
            block_seq: None,
        })
    }

    #[getter]
    pub fn node_id(&self) -> PyHashId {
        PyHashId::new(self.node_id)
    }

    /// Optional display sequence: _node_sequence[start:end] or _block_sequence or None.
    #[getter]
    pub fn sequence(&self) -> Option<String> {
        self.display_sequence()
    }

    /// Full underlying node sequence, if present.
    #[getter]
    pub fn _node_sequence(&self) -> Option<&str> {
        self.node_seq.as_deref()
    }

    /// Already-sliced block sequence, if present.
    #[getter]
    pub fn _block_sequence(&self) -> Option<&str> {
        self.block_seq.as_deref()
    }

    fn __str__(&self) -> String {
        match self.display_sequence() {
            Some(s) => s,
            None => format!(
                "Block({}[{}:{}])",
                self.node_id, self.sequence_start, self.sequence_end
            ),
        }
    }

    fn __repr__(&self) -> String {
        let seq_preview = match self.display_sequence() {
            None => "None".to_string(),
            Some(s) if s.len() <= 20 => format!("'{s}'"),
            Some(s) => format!("'{}...'", &s[..20]),
        };
        format!(
            "Block(node_id={}, sequence_start={}, sequence_end={}, sequence={})",
            self.node_id, self.sequence_start, self.sequence_end, seq_preview
        )
    }

    fn __hash__(&self) -> isize {
        let mut hash: isize = 0;
        for &b in &self.node_id.0 {
            hash = hash.wrapping_mul(31).wrapping_add(b as isize);
        }
        hash = hash.wrapping_mul(31).wrapping_add(self.sequence_start as isize);
        hash = hash.wrapping_mul(31).wrapping_add(self.sequence_end as isize);
        hash
    }

    fn __eq__(&self, py: Python<'_>, other: PyObject) -> bool {
        other
            .extract::<PyRef<PyBlock>>(py)
            .map(|o| {
                self.node_id == o.node_id
                    && self.sequence_start == o.sequence_start
                    && self.sequence_end == o.sequence_end
            })
            .unwrap_or(false)
    }

    /// Support Python slicing: block[start:end].
    ///
    /// Slice indices are relative to the current block window.
    /// Returns a new Block with the same node_id and adjusted absolute coordinates.
    /// Rejects integer indexing (TypeError) and non-unit steps (ValueError).
    fn __getitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<PyBlock> {
        let slice = key.downcast::<PySlice>().map_err(|_| {
            PyTypeError::new_err(
                "Block indices must be slices; integer indexing is not supported",
            )
        })?;

        let block_len = self.sequence_end - self.sequence_start;
        let indices = slice.indices(block_len as isize)?;

        if indices.step != 1 {
            return Err(PyValueError::new_err(
                "Block slices must have step 1; step slicing is not supported",
            ));
        }

        let rel_start = indices.start as i64;
        let rel_stop = indices.stop as i64;
        let new_start = self.sequence_start + rel_start;
        let new_end = self.sequence_start + rel_stop;

        let new_block_seq = if self.node_seq.is_none() {
            self.block_seq.as_ref().map(|bs| {
                let s = rel_start as usize;
                let e = rel_stop as usize;
                bs[s..e].to_string()
            })
        } else {
            None
        };

        Ok(PyBlock {
            node_id: self.node_id,
            sequence_start: new_start,
            sequence_end: new_end,
            node_seq: self.node_seq.clone(),
            block_seq: new_block_seq,
        })
    }
}

/// Sentinel representing the upstream boundary of a path.
#[pyclass(name = "StartBlock", extends = PyBlock)]
pub struct PyStartBlock;

#[pymethods]
impl PyStartBlock {
    #[new]
    pub fn new() -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyBlock {
            node_id: PATH_START_NODE_ID,
            sequence_start: 0,
            sequence_end: 0,
            node_seq: None,
            block_seq: None,
        })
        .add_subclass(PyStartBlock)
    }

    fn __repr__(&self) -> &str {
        "StartBlock()"
    }

    fn __str__(&self) -> &str {
        "StartBlock()"
    }
}

/// Sentinel representing the downstream boundary of a path.
#[pyclass(name = "EndBlock", extends = PyBlock)]
pub struct PyEndBlock;

#[pymethods]
impl PyEndBlock {
    #[new]
    pub fn new() -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyBlock {
            node_id: PATH_END_NODE_ID,
            sequence_start: 0,
            sequence_end: 0,
            node_seq: None,
            block_seq: None,
        })
        .add_subclass(PyEndBlock)
    }

    fn __repr__(&self) -> &str {
        "EndBlock()"
    }

    fn __str__(&self) -> &str {
        "EndBlock()"
    }
}

#[cfg(test)]
mod block_tests {
    use pyo3::{prelude::*, py_run};

    use super::{PyBlock, PyEndBlock, PyStartBlock};

    #[test]
    fn test_basic_construction() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let block_type = py.get_type::<PyBlock>();
            py_run!(
                py,
                block_type,
                r#"
b = block_type("ACTG")
assert str(b) == "ACTG", f"expected 'ACTG', got {str(b)!r}"
assert b.sequence_start == 0
assert b.sequence_end == 4
"#
            );
        });
    }

    #[test]
    fn test_from_node_sequence() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let block_type = py.get_type::<PyBlock>();
            py_run!(
                py,
                block_type,
                r#"
b = block_type.from_node_sequence("ACTG", start=1, end=3)
assert str(b) == "CT", f"expected 'CT', got {str(b)!r}"
assert b.sequence_start == 1
assert b.sequence_end == 3
"#
            );
        });
    }

    #[test]
    fn test_from_block_sequence() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let block_type = py.get_type::<PyBlock>();
            py_run!(
                py,
                block_type,
                r#"
b = block_type.from_block_sequence("CT", start=1, end=3)
assert str(b) == "CT", f"expected 'CT', got {str(b)!r}"
assert b.sequence_start == 1
assert b.sequence_end == 3
"#
            );
        });
    }

    #[test]
    fn test_ref_constructor() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let block_type = py.get_type::<PyBlock>();
            py_run!(
                py,
                block_type,
                r#"
b = block_type("ACTG")
nid = b.node_id
r = block_type.ref(nid, 1, 3)
assert r.sequence_start == 1
assert r.sequence_end == 3
assert "Block(" in str(r)
assert r._node_sequence is None
assert r._block_sequence is None
"#
            );
        });
    }

    #[test]
    fn test_equality_ignores_sequence_payload() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let block_type = py.get_type::<PyBlock>();
            py_run!(
                py,
                block_type,
                r#"
b = block_type("ACTG")
nid = b.node_id
a = block_type.ref(nid, 0, 4)
c = block_type.from_node_sequence("ACTG", node_id=nid)
assert a == c, "ref and from_node_sequence with same id/coords must be equal"
assert hash(a) == hash(c), "hash must match"
"#
            );
        });
    }

    #[test]
    fn test_slice_from_node_sequence() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let block_type = py.get_type::<PyBlock>();
            py_run!(
                py,
                block_type,
                r#"
a = block_type("ACTG")
b = a[:2]
assert str(b) == "AC", f"expected 'AC', got {str(b)!r}"
assert b.node_id == a.node_id
assert b.sequence_start == 0
assert b.sequence_end == 2
"#
            );
        });
    }

    #[test]
    fn test_slice_nonzero_coords() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let block_type = py.get_type::<PyBlock>();
            py_run!(
                py,
                block_type,
                r#"
a = block_type.from_node_sequence("AAACCCGGG", start=3, end=6)
assert str(a) == "CCC"
b = a[:2]
assert str(b) == "CC", f"expected 'CC', got {str(b)!r}"
assert b.node_id == a.node_id
assert b.sequence_start == 3
assert b.sequence_end == 5
"#
            );
        });
    }

    #[test]
    fn test_slice_from_block_sequence() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let block_type = py.get_type::<PyBlock>();
            py_run!(
                py,
                block_type,
                r#"
a = block_type.from_block_sequence("ACTG", start=100, end=104)
b = a[1:3]
assert str(b) == "CT", f"expected 'CT', got {str(b)!r}"
assert b.sequence_start == 101
assert b.sequence_end == 103
"#
            );
        });
    }

    #[test]
    fn test_slice_coordinate_only() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let block_type = py.get_type::<PyBlock>();
            py_run!(
                py,
                block_type,
                r#"
b = block_type("ACTG")
nid = b.node_id
a = block_type.ref(nid, 100, 104)
s = a[1:3]
assert s.node_id == a.node_id
assert s.sequence_start == 101
assert s.sequence_end == 103
assert s._node_sequence is None
assert s._block_sequence is None
"#
            );
        });
    }

    #[test]
    fn test_invalid_indexing() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let block_type = py.get_type::<PyBlock>();
            py_run!(
                py,
                block_type,
                r#"
a = block_type("ACTG")
try:
    a[1]
    assert False, "expected TypeError for integer index"
except TypeError:
    pass
try:
    a[::2]
    assert False, "expected ValueError for step slicing"
except (ValueError, NotImplementedError):
    pass
"#
            );
        });
    }

    #[test]
    fn test_start_end_block_repr() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let start_type = py.get_type::<PyStartBlock>();
            let end_type = py.get_type::<PyEndBlock>();
            py_run!(
                py,
                start_type end_type,
                r#"
s = start_type()
e = end_type()
assert repr(s) == "StartBlock()"
assert repr(e) == "EndBlock()"
assert str(s) == "StartBlock()"
assert str(e) == "EndBlock()"
assert s == start_type()
assert e == end_type()
"#
            );
        });
    }

    #[test]
    fn test_two_blocks_same_seq_distinct() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let block_type = py.get_type::<PyBlock>();
            // Two Block("ACTG") calls → different node_ids → not equal
            let b1 = Py::new(py, PyBlock::new("ACTG".to_string(), None)).unwrap();
            let b2 = Py::new(py, PyBlock::new("ACTG".to_string(), None)).unwrap();
            py_run!(
                py,
                b1 b2,
                r#"
assert b1 != b2, "independent Blocks must not be equal"
assert hash(b1) != hash(b2) or True  # hash collision possible, but ids differ
"#
            );
        });
    }
}
