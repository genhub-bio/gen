use gen_annotations::projection::AnnotationSegment;
use gen_core::{HashId, range::Range};
use gen_models::{
    accession::Accession,
    annotations::Annotation,
    db::{DbContext, GraphConnection},
    locus::GraphLocus,
};
use pyo3::prelude::*;

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
    /// ----------
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

    /// GenBank feature kind (e.g. ``"gene"``, ``"CDS"``, ``"sig_peptide"``), if available.
    #[getter]
    fn kind(&self) -> Option<&str> {
        self.inner
            .extra
            .as_ref()?
            .genbank
            .as_ref()
            .map(|g| g.kind.as_str())
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
        format!(
            "Annotation(name={:?}, group={:?}, len={len})",
            self.inner.name, self.inner.group
        )
    }
}

/// Compute the annotation segments from accession nodes.
pub(super) fn annotation_segments(
    conn: &GraphConnection,
    annotation: &Annotation,
) -> Vec<AnnotationSegment> {
    Accession::get_nodes_by_id(conn, &annotation.accession_id)
        .iter()
        .map(AnnotationSegment::from)
        .collect()
}
