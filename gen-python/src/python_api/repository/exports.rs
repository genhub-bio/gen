use std::{fs, path::PathBuf};

use r#gen::{
    exports::{fasta::export_fasta, genbank::export_genbank, gfa::export_gfa},
    track_database,
};
use gen_models::sample::Sample;
use pyo3::{exceptions::PyRuntimeError, prelude::*};

use super::PyRepository;

#[pymethods]
impl PyRepository {
    #[pyo3(signature = (filename, sample=None, collection=None))]
    fn export_fasta(
        &self,
        filename: String,
        sample: Option<String>,
        collection: Option<String>,
    ) -> PyResult<()> {
        let conn = self.context.graph().conn();
        let op_conn = self.context.operations().conn();
        if !self.in_transaction {
            track_database(conn, op_conn)
                .map_err(|e| PyRuntimeError::new_err(format!("Error tracking database: {e}")))?;
        }
        let collection = collection.unwrap_or_else(|| self.get_default_collection());
        export_fasta(
            conn,
            &collection,
            sample.as_deref(),
            &PathBuf::from(&filename),
        )
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to export '{}': {e}", filename)))
    }

    #[pyo3(signature = (filename, sample=None, node_max=None, collection=None))]
    fn export_gfa(
        &self,
        filename: String,
        sample: Option<String>,
        node_max: Option<i64>,
        collection: Option<String>,
    ) -> PyResult<()> {
        let conn = self.context.graph().conn();
        let op_conn = self.context.operations().conn();
        if !self.in_transaction {
            track_database(conn, op_conn)
                .map_err(|e| PyRuntimeError::new_err(format!("Error tracking database: {e}")))?;
        }
        let collection = collection.unwrap_or_else(|| self.get_default_collection());
        let sample = sample.unwrap_or_else(|| Sample::DEFAULT_NAME.to_string());
        export_gfa(
            conn,
            &collection,
            &PathBuf::from(&filename),
            &sample,
            node_max,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to export '{}': {e}", filename)))
    }

    #[pyo3(signature = (filename, sample=None, collection=None))]
    fn export_genbank(
        &self,
        filename: String,
        sample: Option<String>,
        collection: Option<String>,
    ) -> PyResult<()> {
        let conn = self.context.graph().conn();
        let op_conn = self.context.operations().conn();
        if !self.in_transaction {
            track_database(conn, op_conn)
                .map_err(|e| PyRuntimeError::new_err(format!("Error tracking database: {e}")))?;
        }
        let collection = collection.unwrap_or_else(|| self.get_default_collection());
        let sample = sample.unwrap_or_else(|| Sample::DEFAULT_NAME.to_string());
        let writer = fs::File::create(&filename).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to create '{}': {e}", filename))
        })?;
        export_genbank(conn, &collection, &sample, writer)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to export '{}': {e}", filename)))
    }
}
