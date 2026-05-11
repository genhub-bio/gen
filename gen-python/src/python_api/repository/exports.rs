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
    #[pyo3(signature = (filename, name=None, sample=None))]
    fn export_fasta(
        &self,
        filename: String,
        name: Option<String>,
        sample: Option<String>,
    ) -> PyResult<()> {
        let conn = self.context.graph().conn();
        let op_conn = self.context.operations().conn();
        if !self.in_transaction {
            track_database(conn, op_conn)
                .map_err(|e| PyRuntimeError::new_err(format!("Error tracking database: {e}")))?;
        }
        let name = name.unwrap_or_else(|| self.get_default_collection());
        export_fasta(conn, &name, sample.as_deref(), &PathBuf::from(&filename))
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to export '{}': {e}", filename)))
    }

    #[pyo3(signature = (filename, name=None, sample=None, node_max=None))]
    fn export_gfa(
        &self,
        filename: String,
        name: Option<String>,
        sample: Option<String>,
        node_max: Option<i64>,
    ) -> PyResult<()> {
        let conn = self.context.graph().conn();
        let op_conn = self.context.operations().conn();
        if !self.in_transaction {
            track_database(conn, op_conn)
                .map_err(|e| PyRuntimeError::new_err(format!("Error tracking database: {e}")))?;
        }
        let name = name.unwrap_or_else(|| self.get_default_collection());
        let sample = sample.unwrap_or_else(|| Sample::DEFAULT_NAME.to_string());
        export_gfa(conn, &name, &PathBuf::from(&filename), &sample, node_max)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to export '{}': {e}", filename)))
    }

    #[pyo3(signature = (filename, name=None, sample=None))]
    fn export_genbank(
        &self,
        filename: String,
        name: Option<String>,
        sample: Option<String>,
    ) -> PyResult<()> {
        let conn = self.context.graph().conn();
        let op_conn = self.context.operations().conn();
        if !self.in_transaction {
            track_database(conn, op_conn)
                .map_err(|e| PyRuntimeError::new_err(format!("Error tracking database: {e}")))?;
        }
        let name = name.unwrap_or_else(|| self.get_default_collection());
        let sample = sample.unwrap_or_else(|| Sample::DEFAULT_NAME.to_string());
        let writer = fs::File::create(&filename).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to create '{}': {e}", filename))
        })?;
        export_genbank(conn, &name, &sample, writer)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to export '{}': {e}", filename)))
    }
}
