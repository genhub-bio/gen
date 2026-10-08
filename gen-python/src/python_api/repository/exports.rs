use std::{fs, path::PathBuf};

use r#gen::exports::{fasta::export_fasta, genbank::export_genbank, gfa::export_gfa};
use gen_models::sample::Sample;
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use pyo3_stub_gen::derive::gen_stub_pymethods;

use super::PyRepository;
use crate::python_api::utils::export_all_sequences_fasta;

#[gen_stub_pymethods]
#[pymethods]
impl PyRepository {
    /// Export current paths as FASTA, optionally restricted by sample and collection.
    /// Set `all_sequences=True` to export all graph paths with 1-based
    /// ``"{sequence_graph_name}.{index}"`` record names.
    #[pyo3(name = "_export_fasta")]
    #[gen_stub(skip)]
    #[pyo3(signature = (filename, sample=None, collection=None, all_sequences=false))]
    fn export_fasta(
        &self,
        filename: String,
        sample: Option<String>,
        collection: Option<String>,
        all_sequences: bool,
    ) -> PyResult<()> {
        let conn = self.context.graph().conn();
        let collection = collection.unwrap_or_else(|| self.get_default_collection());
        if all_sequences {
            return export_all_sequences_fasta(
                &self.context,
                &collection,
                sample.as_deref(),
                &PathBuf::from(&filename),
            );
        }
        export_fasta(
            conn,
            self.context.workspace(),
            &collection,
            sample.as_deref(),
            &PathBuf::from(&filename),
            None,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to export '{}': {e}", filename)))
    }

    /// Write a sample's graph structure to a GFA file. `node_max` splits nodes longer than that
    /// many bases.
    #[pyo3(name = "_export_gfa")]
    #[gen_stub(skip)]
    #[pyo3(signature = (filename, sample=None, node_max=None, collection=None))]
    fn export_gfa(
        &self,
        filename: String,
        sample: Option<String>,
        node_max: Option<i64>,
        collection: Option<String>,
    ) -> PyResult<()> {
        let conn = self.context.graph().conn();
        let collection = collection.unwrap_or_else(|| self.get_default_collection());
        let sample = sample.unwrap_or_else(|| Sample::DEFAULT_NAME.to_string());
        export_gfa(
            conn,
            self.context.workspace(),
            &collection,
            &PathBuf::from(&filename),
            &sample,
            node_max,
            None,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to export '{}': {e}", filename)))
    }

    /// Write a sample's sequences and annotations to a GenBank file.
    #[pyo3(name = "_export_genbank")]
    #[gen_stub(skip)]
    #[pyo3(signature = (filename, sample=None, collection=None))]
    fn export_genbank(
        &self,
        filename: String,
        sample: Option<String>,
        collection: Option<String>,
    ) -> PyResult<()> {
        let conn = self.context.graph().conn();
        let collection = collection.unwrap_or_else(|| self.get_default_collection());
        let sample = sample.unwrap_or_else(|| Sample::DEFAULT_NAME.to_string());
        let writer = fs::File::create(&filename).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to create '{}': {e}", filename))
        })?;
        export_genbank(
            conn,
            self.context.workspace(),
            &collection,
            &sample,
            writer,
            None,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to export '{}': {e}", filename)))
    }
}
