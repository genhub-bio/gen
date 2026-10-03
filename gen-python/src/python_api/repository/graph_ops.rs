use r#gen::commands::graph_operations::{
    derive_chunks::derive_chunks_operation, derive_subgraph::derive_subgraph_operation,
};
use gen_core::{Strand, region::Region};
use gen_models::operations::{OperationInfo, OperationSummary};
use pyo3::{
    exceptions::{PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
};
use pyo3_stub_gen::derive::gen_stub_pymethods;

use super::{
    PyRepository, run_context_operation_write,
    stitch::{StitchSource, stitch_sources},
};
use crate::python_api::{
    block_group::PySequenceGraph, graph_search::PyGraphLocus, sample::PySample,
};

#[gen_stub_pymethods]
#[pymethods]
impl PyRepository {
    /// Split the region of `sample` into chunks, either at `breakpoints` or every `chunk_size`
    /// bases, and store them in `new_sample`. Returns the new `Sample`. See also `graph.chunks()`.
    #[pyo3(name = "_derive_chunks")]
    #[gen_stub(skip)]
    #[pyo3(signature = (sample, new_sample, region, backbone=None, breakpoints=None, chunk_size=None, collection=None))]
    #[expect(clippy::too_many_arguments, reason = "mirrors underlying API")]
    fn derive_chunks(
        &self,
        sample: String,
        new_sample: String,
        region: String,
        backbone: Option<String>,
        breakpoints: Option<Vec<i64>>,
        chunk_size: Option<i64>,
        collection: Option<String>,
    ) -> PyResult<PySample> {
        let collection = collection.unwrap_or_else(|| self.get_default_collection());
        derive_chunks_operation(
            &self.context,
            Some(collection.clone()),
            sample,
            new_sample.clone(),
            region,
            backbone,
            breakpoints,
            chunk_size,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("Error deriving chunks: {e}")))?;
        Ok(self.block_groups_in_sample(&collection, &new_sample))
    }

    /// Copy the region of `sample` into `new_sample` as a smaller sequence graph and return the new
    /// `Sample`. See also `graph.subgraph()`.
    #[pyo3(name = "_derive_subgraph")]
    #[gen_stub(skip)]
    #[pyo3(signature = (sample, new_sample, region, backbone=None, collection=None))]
    fn derive_subgraph(
        &self,
        sample: String,
        new_sample: String,
        region: String,
        backbone: Option<String>,
        collection: Option<String>,
    ) -> PyResult<PySequenceGraph> {
        let collection = collection.unwrap_or_else(|| self.get_default_collection());
        let parsed_region = Region::parse(&region).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to parse region '{region}': {e}"))
        })?;
        derive_subgraph_operation(
            &self.context,
            Some(collection.clone()),
            sample,
            new_sample.clone(),
            region,
            backbone,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("Error deriving subgraph: {e}")))?;
        self.get_block_group(&collection, &new_sample, &parsed_region.name.to_string())
    }

    /// Join `parts` end to end into a new sequence graph named `new_region` in `new_sample`.
    ///
    /// Each part is a `SequenceGraph`, or a `Locus` such as `graph.region("chr1:100-200")`. The end
    /// of each part is connected to the start of the next, so a graph can be assembled from
    /// pieces of several others without first materializing a subgraph of each.
    ///
    /// A `Locus` is a linear span, but what is stitched is the subgraph of every variant route
    /// between its first and last positions: stitching `graph.region("chr1:100-200")` keeps all
    /// the alternatives that lie inside that region, not just the sequence that region reads
    /// along the current path. A whole `SequenceGraph` part contributes all of its routes. The
    /// new graph's current path reads the parts' own routes one after another.
    ///
    /// Every `SequenceGraph` part needs a current path, which a subgraph taken between positions
    /// off the current path may lack; stitch a `Locus` of such a graph instead.
    ///
    /// Parts must come from one collection, be forward-strand (reverse loci are rejected) and not
    /// overlap, since that would make the result cyclic.
    ///
    /// Example::
    ///
    ///     construct = repo.stitch(
    ///         [vector.region("vector:0-100"), insert, vector.region("vector:150-400")],
    ///         new_sample="assembly",
    ///         new_region="construct",
    ///     )
    fn stitch(
        &self,
        #[gen_stub(override_type(type_repr = "typing.Sequence[SequenceGraph | Locus]", imports = ("typing")))]
        parts: Vec<Bound<'_, PyAny>>,
        new_sample: String,
        new_region: String,
    ) -> PyResult<PySequenceGraph> {
        let mut collection: Option<String> = None;
        let mut sources = Vec::with_capacity(parts.len());
        for part in &parts {
            let (part_collection, source) = if let Ok(graph) =
                part.extract::<PyRef<PySequenceGraph>>()
            {
                (
                    graph.collection_name.clone(),
                    StitchSource::BlockGroup(graph.id),
                )
            } else if let Ok(locus) = part.extract::<PyRef<PyGraphLocus>>() {
                let graph = locus.sequence_graph().ok_or_else(|| {
                    PyValueError::new_err(
                        "a locus to stitch must come from a sequence graph, such as graph.region(...)",
                    )
                })?;
                let ranges = locus
                    .graph_locus()
                    .slices
                    .iter()
                    .filter(|slice| slice.start < slice.end)
                    .map(|slice| {
                        if slice.strand == Strand::Reverse {
                            return Err(PyValueError::new_err(
                                "stitch() takes forward-strand loci; reverse loci are not supported",
                            ));
                        }
                        Ok((
                            slice.block.node_id,
                            slice.block.sequence_start + slice.start as i64
                                ..slice.block.sequence_start + slice.end as i64,
                        ))
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                (
                    graph.collection_name.clone(),
                    StitchSource::Locus {
                        block_group_id: graph.id,
                        ranges,
                    },
                )
            } else {
                return Err(PyTypeError::new_err(
                    "stitch() parts must be SequenceGraph or Locus objects",
                ));
            };
            match &collection {
                Some(collection) if *collection != part_collection => {
                    return Err(PyValueError::new_err(format!(
                        "all parts must be in the same collection ('{collection}' vs '{part_collection}')"
                    )));
                }
                _ => collection = Some(part_collection),
            }
            sources.push(source);
        }
        let collection = collection
            .ok_or_else(|| PyValueError::new_err("stitch() requires at least one part"))?;

        let stitched = run_context_operation_write(
            &self.context,
            |context| {
                let block_group =
                    stitch_sources(context, &collection, &new_sample, &new_region, &sources)?;
                let summary = OperationSummary::new(
                    OperationInfo {
                        files: vec![],
                        description: "stitch".to_string(),
                    },
                    format!(
                        " {new_sample}: stitched {} parts into {new_region}",
                        sources.len()
                    ),
                );
                Ok((block_group, summary))
            },
            |error| PyRuntimeError::new_err(format!("Error stitching parts: {error}")),
        )?;
        Ok(self.to_py_block_group(stitched))
    }
}
