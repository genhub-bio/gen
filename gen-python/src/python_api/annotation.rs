use gen_annotations::projection::AnnotationSegment;
use gen_core::{HashId, range::Range};
use gen_models::{annotations::Annotation, db::DbContext, locus::GraphLocus};
use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyAny};
use serde_json::{Map, Value, to_string as json_to_string, to_value as json_to_value};

use super::graph_search::PyGraphLocus;

/// A named genomic annotation.
///
/// **From a database** — obtain via ``SequenceGraph.list_annotations()``
/// or ``GraphWidget.list_annotations()``.
///
/// **From a search result** — create with ``Annotation(locus, name)``
/// where *locus* is a ``Locus`` returned by ``SequenceGraph.search()``.
#[pyclass(name = "Annotation", unsendable)]
#[derive(Clone)]
pub struct PyAnnotation {
    pub inner: Annotation,
    pub context: Option<DbContext>,
    pub ann_segments: Vec<AnnotationSegment>,
    pub source_block_group_id: Option<HashId>,
    /// Pre-computed locus; present only for annotations built from a ``Locus``
    /// via the Python constructor.
    pub locus: Option<GraphLocus>,
}

#[pymethods]
impl PyAnnotation {
    /// Create an ephemeral annotation from a search-result locus.
    ///
    /// Parameters
    /// locus : Locus
    ///     A ``Locus`` returned by ``SequenceGraph.search()``.
    /// name : str
    ///     Human-readable label for the annotation.
    #[new]
    pub fn from_locus(locus: PyRef<PyGraphLocus>, name: &str) -> Self {
        let ann_segments = locus
            .inner
            .slices
            .iter()
            .map(|s| AnnotationSegment {
                node_id: s.block.node_id,
                range: Range {
                    start: s.block.sequence_start + s.start as i64,
                    end: s.block.sequence_start + s.end as i64,
                },
                strand: s.strand,
            })
            .collect();
        PyAnnotation {
            inner: Annotation {
                id: HashId::convert_str(name),
                name: name.to_string(),
                group: String::new(),
                accession_id: HashId([0u8; 32]),
                extra: None,
            },
            context: None,
            ann_segments,
            source_block_group_id: None,
            locus: Some(locus.inner.clone()),
        }
    }

    /// The graph-space locus covered by this annotation.
    ///
    /// Only available for annotations created with ``Annotation(locus, name)``.
    /// Returns ``None`` for database annotations from ``list_annotations()``.
    #[getter]
    fn locus(&self) -> Option<PyGraphLocus> {
        self.locus
            .as_ref()
            .map(|l| PyGraphLocus::from_locus(l.clone()))
    }

    /// Hash ID of this annotation.
    #[getter]
    fn id(&self) -> String {
        self.inner.id.to_string()
    }

    /// Human-readable annotation name.
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    /// Annotation group this annotation belongs to.
    #[getter]
    fn group(&self) -> &str {
        &self.inner.group
    }

    /// Track (annotation group) this annotation was loaded from, or ``None`` for
    /// ephemeral annotations created with ``Annotation(locus, name)``.
    #[getter]
    fn track(&self) -> Option<&str> {
        if self.inner.group.is_empty() {
            None
        } else {
            Some(&self.inner.group)
        }
    }

    /// All source-agnostic annotation metadata as a flat dict, or ``None`` if no
    /// extra data is stored.
    ///
    /// All source-specific sub-dicts (``genbank``, ``gff``, ``bed``) are merged
    /// into a single dict so callers do not need to know which file format the
    /// annotation originated from.  In practice only one source is populated per
    /// annotation, so there are no key collisions.
    ///
    /// Example
    /// ::
    ///
    ///     for ann in widget.list_annotations():
    ///         print(ann.metadata)
    ///         # GenBank CDS:  {"kind": "CDS", "qualifiers": [...]}
    #[getter]
    fn metadata<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let Some(ref extra) = self.inner.extra else {
            return Ok(None);
        };

        // GenBank / GFF / BED: flat-merge the non-null source sub-dicts.
        let top = json_to_value(extra).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let mut merged = Map::new();
        if let Value::Object(map) = top {
            for (_, child) in map {
                if let Value::Object(child_map) = child {
                    merged.extend(child_map);
                }
            }
        }
        if merged.is_empty() {
            return Ok(None);
        }
        let json_str =
            json_to_string(&merged).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Some(py.import("json")?.call_method1("loads", (json_str,))?))
    }

    /// Genomic segments covered by this annotation.
    ///
    /// Each segment is a dict with keys ``node_id`` (str), ``start`` (int),
    /// ``end`` (int), and ``strand`` (``"+"`` or ``"-"``).
    #[getter]
    fn segments<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, pyo3::types::PyDict>> {
        self.ann_segments
            .iter()
            .map(|s| {
                let d = pyo3::types::PyDict::new(py);
                d.set_item("node_id", s.node_id.to_string()).unwrap();
                d.set_item("start", s.range.start).unwrap();
                d.set_item("end", s.range.end).unwrap();
                d.set_item("strand", s.strand.to_string()).unwrap();
                d
            })
            .collect()
    }

    /// Total length of the annotation in base pairs (sum across all segments).
    fn __len__(&self) -> usize {
        self.ann_segments
            .iter()
            .map(|s| (s.range.end - s.range.start) as usize)
            .sum()
    }

    fn __repr__(&self) -> String {
        let len: usize = self
            .ann_segments
            .iter()
            .map(|s| (s.range.end - s.range.start) as usize)
            .sum();
        let track = if self.inner.group.is_empty() {
            "None".to_string()
        } else {
            format!("{:?}", self.inner.group)
        };
        format!(
            "Annotation(name={:?}, track={track}, len={len})",
            self.inner.name
        )
    }
}
