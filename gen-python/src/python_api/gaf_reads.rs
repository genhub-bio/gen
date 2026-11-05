use std::{collections::HashMap, fs};

use r#gen::{
    updates::gaf::{GafAlignment, GafUpdateError, clip_gaf_walk, parse_gaf_line},
    views::gen_graph_widget::walk_hop_edge_key,
};
use gen_graph::{GenGraph, GraphNodeSlice};
use gen_models::{db::DbContext, locus::GraphLocus};
use pyo3::{
    Bound, PyAny, PyResult, Python,
    exceptions::{PyKeyError, PyValueError},
    pyclass, pymethods,
    types::{PyAnyMethods as _, PyDict, PyDictMethods as _, PyList},
};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use super::{block_group::PySequenceGraph, graph_read::current_graph, graph_search::PyGraphLocus};

/// The reads of a GAF file that trace through one sequence graph, one `Locus` per read.
///
/// Returned by `SequenceGraph.load_gaf()`. It behaves like a read-only mapping from read name to
/// `Locus`: `len()`, `reads["name"]`, `"name" in reads` and iteration over names all work.
/// The reads that did not load are counted by why:
///
/// - `skipped`: reads aligned to segments this sequence graph does not contain, such as a read
///   from another locus when the GAF was made against a whole sample.
/// - `unmapped`: reads with no mapped alignment.
/// - `untraceable`: reads whose walk takes a step the graph has no stored edge for, with the
///   first such step described. These point at a disagreement between the GAF's graph and this
///   one, so they are never loaded.
#[gen_stub_pyclass]
#[pyclass(name = "GafReads", unsendable)]
pub struct PyGafReads {
    loci: Vec<(String, PyGraphLocus)>,
    #[pyo3(get)]
    skipped: usize,
    #[pyo3(get)]
    unmapped: usize,
    untraceable: Vec<(String, String)>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyGafReads {
    /// Read names mapped to the sequence graph's `Locus` for that read.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "dict[str, Locus]", imports = ()))]
    fn loci<'py>(&self, python: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let loci = PyDict::new(python);
        for (name, locus) in &self.loci {
            loci.set_item(name, locus.clone())?;
        }
        Ok(loci)
    }

    /// Untraceable read names mapped to a description of their first step without a stored edge.
    #[getter]
    fn untraceable(&self) -> HashMap<String, String> {
        self.untraceable.iter().cloned().collect()
    }

    fn __len__(&self) -> usize {
        self.loci.len()
    }

    fn __contains__(&self, name: &str) -> bool {
        self.loci.iter().any(|(read, _)| read == name)
    }

    fn __getitem__(&self, name: &str) -> PyResult<PyGraphLocus> {
        self.loci
            .iter()
            .find(|(read, _)| read == name)
            .map(|(_, locus)| locus.clone())
            .ok_or_else(|| PyKeyError::new_err(name.to_string()))
    }

    #[gen_stub(override_return_type(type_repr = "typing.Iterator[str]", imports = ("typing",)))]
    fn __iter__<'py>(&self, python: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        PyList::new(python, self.loci.iter().map(|(name, _)| name))?.call_method0("__iter__")
    }

    fn __repr__(&self) -> String {
        format!(
            "GafReads(loaded={}, skipped={}, unmapped={}, untraceable={})",
            self.loci.len(),
            self.skipped,
            self.unmapped,
            self.untraceable.len()
        )
    }
}

/// The alignment to keep for each read, in file order of first appearance: the mapped row with
/// the longest aligned block among the rows allowed by `primary_only`.
fn best_alignment_per_read(text: &str, primary_only: bool) -> PyResult<Vec<GafAlignment>> {
    let mut best: Vec<GafAlignment> = vec![];
    let mut position_of: HashMap<String, usize> = HashMap::new();
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        let alignment = parse_gaf_line(line).map_err(gaf_error)?;
        if primary_only && !alignment.is_primary() {
            continue;
        }
        match position_of.get(&alignment.query_name) {
            Some(&position) if best[position].align_block_len >= alignment.align_block_len => {}
            Some(&position) => best[position] = alignment,
            None => {
                position_of.insert(alignment.query_name.clone(), best.len());
                best.push(alignment);
            }
        }
    }
    Ok(best)
}

fn gaf_error(error: GafUpdateError) -> pyo3::PyErr {
    PyValueError::new_err(error.to_string())
}

/// The first step of a walk the graph stores no edge for, described by the slices it joins.
fn first_missing_hop(graph: &GenGraph, walk: &[GraphNodeSlice]) -> Option<String> {
    let describe = |slice: &GraphNodeSlice| {
        format!(
            "{}.{}.{}{}",
            slice.block.node_id, slice.block.sequence_start, slice.block.sequence_end, slice.strand
        )
    };
    walk.windows(2)
        .find(|hop| walk_hop_edge_key(graph, &hop[0], &hop[1]).is_none())
        .map(|hop| format!("{} -> {}", describe(&hop[0]), describe(&hop[1])))
}

/// Load every read of a GAF file that lies in `sequence_graph`, as one locus per read.
pub(crate) fn load_gaf_reads(
    sequence_graph: &PySequenceGraph,
    context: &DbContext,
    path: &str,
    primary_only: bool,
) -> PyResult<PyGafReads> {
    let text = fs::read_to_string(path)?;
    let graph = current_graph(context, &sequence_graph.id)?;
    let mut reads = PyGafReads {
        loci: vec![],
        skipped: 0,
        unmapped: 0,
        untraceable: vec![],
    };
    for alignment in best_alignment_per_read(&text, primary_only)? {
        let walk = clip_gaf_walk(&alignment).map_err(gaf_error)?;
        if walk.is_empty() {
            reads.unmapped += 1;
        } else if walk.iter().any(|slice| !graph.contains_node(slice.block)) {
            reads.skipped += 1;
        } else if let Some(hop) = first_missing_hop(&graph, &walk) {
            reads.untraceable.push((alignment.query_name, hop));
        } else {
            let locus =
                PyGraphLocus::with_context(GraphLocus { slices: walk }, Some(context.clone()))
                    .attached_to(Some(sequence_graph.clone()));
            reads.loci.push((alignment.query_name, locus));
        }
    }
    Ok(reads)
}
