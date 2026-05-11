use std::path::PathBuf;

use r#gen::{
    fasta::FastaError,
    graphs::combinatorial_library::{SequencePart, parse_library},
    imports::{
        fasta::import_fasta,
        genbank::{GenBankImportOptions, import_genbank},
        gfa::{GFAImportError, import_gfa},
        library::{LibraryImportError, import_library},
    },
};
use gen_models::{errors::OperationError, sample::Sample};
use pyo3::{exceptions::PyRuntimeError, prelude::*};

use super::{PyRepository, run_write};
use crate::python_api::sequence_part::PySequencePart;

#[pymethods]
impl PyRepository {
    #[pyo3(signature = (filename, name=None, sample=None, shallow=false))]
    pub fn import_fasta(
        &self,
        filename: String,
        name: Option<String>,
        sample: Option<String>,
        shallow: bool,
    ) -> PyResult<String> {
        let name = name.unwrap_or_else(|| self.get_default_collection());
        let sample = sample.unwrap_or_else(|| Sample::DEFAULT_NAME.to_string());
        run_write(&self.context, !self.in_transaction, |ctx| {
            import_fasta(ctx, &filename, &name, &sample, shallow)
                .map(|_| format!("'{}' imported.", filename))
                .map_err(|e| match e {
                    FastaError::OperationError(OperationError::NoChanges) => {
                        PyRuntimeError::new_err(format!("'{}': contents already exist", filename))
                    }
                    _ => PyRuntimeError::new_err(format!("Failed to import '{}': {e}", filename)),
                })
        })
    }

    #[pyo3(signature = (filename, name=None, sample=None))]
    fn import_gfa(
        &self,
        filename: String,
        name: Option<String>,
        sample: Option<String>,
    ) -> PyResult<String> {
        let name = name.unwrap_or_else(|| self.get_default_collection());
        let sample = sample.unwrap_or_else(|| Sample::DEFAULT_NAME.to_string());
        run_write(&self.context, !self.in_transaction, |ctx| {
            import_gfa(ctx, &PathBuf::from(&filename), &name, &sample)
                .map(|_| format!("'{}' imported.", filename))
                .map_err(|e| match e {
                    GFAImportError::OperationError(OperationError::NoChanges) => {
                        PyRuntimeError::new_err(format!("'{}': already exists", filename))
                    }
                    _ => PyRuntimeError::new_err(format!("Failed to import '{}': {e}", filename)),
                })
        })
    }

    #[pyo3(signature = (filename, name=None, sample=None))]
    fn import_genbank(
        &self,
        filename: String,
        name: Option<String>,
        sample: Option<String>,
    ) -> PyResult<String> {
        use std::fs::File;
        let name = name.unwrap_or_else(|| self.get_default_collection());
        let sample = sample.unwrap_or_else(|| Sample::DEFAULT_NAME.to_string());
        run_write(&self.context, !self.in_transaction, |ctx| {
            let mut reader: Box<dyn std::io::Read> = if filename.ends_with(".gz") {
                let file = File::open(&filename).map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to open '{}': {e}", filename))
                })?;
                Box::new(flate2::read::GzDecoder::new(file))
            } else {
                Box::new(File::open(&filename).map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to open '{}': {e}", filename))
                })?)
            };
            import_genbank(
                ctx,
                &mut reader,
                name.as_ref(),
                &sample,
                gen_models::operations::OperationInfo {
                    files: vec![{
                        let mut f = gen_models::operations::OperationFile::new(filename.clone());
                        f.file_type = gen_models::file_types::FileTypes::GenBank;
                        f
                    }],
                    description: "GenBank Import".to_string(),
                },
                GenBankImportOptions::default(),
            )
            .map(|_| format!("'{}' imported.", filename))
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to import '{}': {e}", filename)))
        })
    }

    #[pyo3(signature = (library_name, parts_list, name=None, sample=None))]
    fn import_library(
        &self,
        library_name: String,
        parts_list: Vec<Vec<PySequencePart>>,
        name: Option<String>,
        sample: Option<String>,
    ) -> PyResult<String> {
        let name = name.unwrap_or_else(|| self.get_default_collection());
        let sample = sample.unwrap_or_else(|| Sample::DEFAULT_NAME.to_string());
        let rust_parts_list: Vec<Vec<SequencePart>> = parts_list
            .iter()
            .map(|parts| {
                parts
                    .iter()
                    .map(|p| SequencePart {
                        name: p.name.clone(),
                        sequence: p.sequence.clone(),
                        sequence_length: p.sequence_length,
                    })
                    .collect()
            })
            .collect();
        run_write(&self.context, !self.in_transaction, |ctx| {
            import_library(
                ctx,
                &name,
                &sample,
                &library_name,
                rust_parts_list.clone(),
                None,
                None,
            )
            .map(|_| format!("Library '{}' imported.", library_name))
            .map_err(|e| match e {
                LibraryImportError::OperationError(OperationError::NoChanges) => {
                    PyRuntimeError::new_err(format!("Library '{}': already exists", library_name))
                }
                _ => PyRuntimeError::new_err(format!(
                    "Failed to import library '{}': {e}",
                    library_name
                )),
            })
        })
    }

    #[pyo3(signature = (library_name, parts, library, name=None, sample=None))]
    fn import_library_files(
        &self,
        library_name: String,
        parts: String,
        library: String,
        name: Option<String>,
        sample: Option<String>,
    ) -> PyResult<String> {
        let parts_list = parse_library(&parts, &library)
            .map_err(|e| PyRuntimeError::new_err(format!("Problem parsing library files: {e}")))?;
        let name = name.unwrap_or_else(|| self.get_default_collection());
        let sample = sample.unwrap_or_else(|| Sample::DEFAULT_NAME.to_string());
        run_write(&self.context, !self.in_transaction, |ctx| {
            import_library(
                ctx,
                &name,
                &sample,
                &library_name,
                parts_list.clone(),
                Some(&parts),
                Some(&library),
            )
            .map(|_| format!("Library '{}' imported.", library_name))
            .map_err(|e| match e {
                LibraryImportError::OperationError(OperationError::NoChanges) => {
                    PyRuntimeError::new_err(format!("Library '{}': already exists", library_name))
                }
                _ => PyRuntimeError::new_err(format!(
                    "Failed to import library '{}': {e}",
                    library_name
                )),
            })
        })
    }
}
