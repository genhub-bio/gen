use r#gen::core::HashId;
use gen_core::DoltHashId;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyBytes};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

/// A content hash identifying a stored object, such as a sequence graph, annotation or operation.
///
/// Use it as a dict key or compare it with `==`. `str(hash_id)` gives the hex digest.
#[gen_stub_pyclass]
#[pyclass(name = "HashId")]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct PyHashId {
    bytes: HashBytes,
}

/// Gen domain ids and Dolt commit hashes have different widths but surface as one Python type.
#[derive(Clone, Copy, PartialEq, Eq)]
enum HashBytes {
    Gen(HashId),
    Dolt(DoltHashId),
}

impl PyHashId {
    pub fn new(hash_id: HashId) -> Self {
        PyHashId {
            bytes: HashBytes::Gen(hash_id),
        }
    }

    /// Wrap the commit hash of an operation or branch head.
    pub fn from_dolt(hash: DoltHashId) -> Self {
        PyHashId {
            bytes: HashBytes::Dolt(hash),
        }
    }

    /// The Gen domain id, for lookups; fails for commit hashes, which name operations instead.
    pub fn gen_hash_id(&self) -> PyResult<HashId> {
        match self.bytes {
            HashBytes::Gen(hash_id) => Ok(hash_id),
            HashBytes::Dolt(_) => Err(PyValueError::new_err(
                "this HashId names an operation, not a stored object",
            )),
        }
    }

    fn as_bytes(&self) -> &[u8] {
        match &self.bytes {
            HashBytes::Gen(hash_id) => &hash_id.0,
            HashBytes::Dolt(hash) => &hash.0,
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyHashId {
    pub fn __str__(&self) -> String {
        match self.bytes {
            HashBytes::Gen(hash_id) => hash_id.to_string(),
            HashBytes::Dolt(hash) => hash.to_string(),
        }
    }

    fn __repr__(&self) -> String {
        format!("HashId(\"{}\")", self.__str__())
    }

    pub fn __hash__(&self) -> isize {
        // Combine the bytes of the hash until it fits
        let mut hash: isize = 0;
        for &b in self.as_bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(b as isize);
        }
        hash
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<PyHashId>>()
            .is_ok_and(|other_hash_id| *self == *other_hash_id)
    }

    /// Returns the raw bytes of the hash.
    #[expect(
        clippy::wrong_self_convention,
        reason = "exposed to Python as to_bytes(); pyo3 pyclass methods require &self"
    )]
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, self.as_bytes())
    }
}
