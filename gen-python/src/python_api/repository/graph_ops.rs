use r#gen::commands::graph_operations::{
    derive_chunks::derive_chunks_operation, derive_subgraph::derive_subgraph_operation,
    make_stitch::make_stitch_operation,
};
use gen_models::block_group::BlockGroup;
use pyo3::{exceptions::PyRuntimeError, prelude::*};

use super::PyRepository;
use crate::python_api::block_group::PySequenceGraph;

#[pymethods]
impl PyRepository {
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
    ) -> PyResult<()> {
        if self.in_transaction {
            return Err(PyRuntimeError::new_err(
                "derive_chunks cannot be called inside a transaction block",
            ));
        }
        derive_chunks_operation(
            &self.context,
            collection,
            sample,
            new_sample,
            region,
            backbone,
            breakpoints,
            chunk_size,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("Error deriving chunks: {e}")))
    }

    #[pyo3(signature = (sample, new_sample, region, backbone=None, collection=None))]
    fn derive_subgraph(
        &self,
        sample: String,
        new_sample: String,
        region: String,
        backbone: Option<String>,
        collection: Option<String>,
    ) -> PyResult<()> {
        if self.in_transaction {
            return Err(PyRuntimeError::new_err(
                "derive_subgraph cannot be called inside a transaction block",
            ));
        }
        derive_subgraph_operation(
            &self.context,
            collection,
            sample,
            new_sample,
            region,
            backbone,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("Error deriving subgraph: {e}")))
    }

    #[pyo3(signature = (sample, new_sample, regions, new_region, collection=None))]
    fn make_stitch(
        &self,
        sample: String,
        new_sample: String,
        regions: String,
        new_region: String,
        collection: Option<String>,
    ) -> PyResult<()> {
        if self.in_transaction {
            return Err(PyRuntimeError::new_err(
                "make_stitch cannot be called inside a transaction block",
            ));
        }
        make_stitch_operation(
            &self.context,
            collection,
            sample,
            new_sample,
            regions,
            new_region,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("Error making stitch: {e}")))
    }

    /// Stitch multiple block groups into a single new block group.
    ///
    /// All block groups must be in the same collection and sample.  The end
    /// nodes of each preceding block group are connected to the start nodes of
    /// the following one, producing a single concatenated graph.
    ///
    /// Parameters
    /// ----------
    /// bgs : list[BlockGroup]
    ///     Block groups to concatenate, in order.
    /// new_sample : str
    ///     Sample name for the result.
    /// new_region : str
    ///     Name for the result block group.
    #[pyo3(signature = (bgs, new_sample, new_region))]
    fn stitch(
        &self,
        bgs: Vec<PySequenceGraph>,
        new_sample: String,
        new_region: String,
    ) -> PyResult<PySequenceGraph> {
        if bgs.is_empty() {
            return Err(PyRuntimeError::new_err(
                "stitch() requires at least one block group",
            ));
        }
        let first = &bgs[0];
        for bg in &bgs[1..] {
            if bg.collection_name != first.collection_name {
                return Err(PyRuntimeError::new_err(format!(
                    "All block groups must be in the same collection ('{}' vs '{}')",
                    first.collection_name, bg.collection_name
                )));
            }
            if bg.sample_name != first.sample_name {
                return Err(PyRuntimeError::new_err(format!(
                    "All block groups must be in the same sample ('{}' vs '{}')",
                    first.sample_name, bg.sample_name
                )));
            }
        }
        let regions = bgs
            .iter()
            .map(|bg| bg.name.as_str())
            .collect::<Vec<_>>()
            .join(",");
        make_stitch_operation(
            &self.context,
            Some(first.collection_name.clone()),
            first.sample_name.clone(),
            new_sample.clone(),
            regions,
            new_region.clone(),
        )
        .map_err(|e| PyRuntimeError::new_err(format!("Error stitching block groups: {e}")))?;

        let conn = self.context.graph().conn();
        let child_id = BlockGroup::get_id(&first.collection_name, &new_sample, &new_region, None);
        let found = BlockGroup::get_by_id(conn, &child_id).map_err(|e| {
            PyRuntimeError::new_err(format!("Stitched BG created but not found: {e}"))
        })?;
        Ok(self.to_py_block_group(found))
    }
}
