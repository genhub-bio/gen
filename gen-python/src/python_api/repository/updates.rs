use r#gen::{
    fasta::FastaError,
    graphs::combinatorial_library::{SequencePart, parse_library},
    updates::{
        fasta::update_with_fasta,
        gaf::update_with_gaf,
        genbank::update_with_genbank,
        gfa::update_with_gfa,
        library::update_with_library,
        sequence::update_with_sequence,
        vcf::{VcfError, update_with_vcf},
    },
};
use gen_models::{errors::OperationError, sample::Sample};
use pyo3::{exceptions::PyRuntimeError, prelude::*};

use super::{PyRepository, run_write};
use crate::python_api::sequence_part::PySequencePart;

#[pymethods]
impl PyRepository {
    #[pyo3(signature = (filename, sample, new_sample, region_name, start, end, name=None))]
    #[expect(clippy::too_many_arguments, reason = "mirrors underlying API")]
    fn update_with_fasta(
        &self,
        filename: String,
        sample: String,
        new_sample: String,
        region_name: String,
        start: i64,
        end: i64,
        name: Option<String>,
    ) -> PyResult<String> {
        let name = name.unwrap_or_else(|| self.get_default_collection());
        run_write(&self.context, !self.in_transaction, |ctx| {
            update_with_fasta(
                ctx,
                &name,
                &sample,
                &new_sample,
                &region_name,
                start,
                end,
                &filename,
                false,
            )
            .map(|_| format!("Updated from '{}'.", filename))
            .map_err(|e| match e {
                FastaError::OperationError(OperationError::NoChanges) => {
                    PyRuntimeError::new_err(format!("'{}': contents already exist", filename))
                }
                _ => PyRuntimeError::new_err(format!("Failed to update from '{}': {e}", filename)),
            })
        })
    }

    #[pyo3(signature = (filename, sample, new_sample, name=None))]
    fn update_with_gfa(
        &self,
        filename: String,
        sample: String,
        new_sample: String,
        name: Option<String>,
    ) -> PyResult<String> {
        let name = name.unwrap_or_else(|| self.get_default_collection());
        run_write(&self.context, !self.in_transaction, |ctx| {
            update_with_gfa(ctx, &name, &sample, &new_sample, &filename)
                .map(|_| format!("Updated from '{}'.", filename))
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to update from '{}': {e}", filename))
                })
        })
    }

    #[pyo3(signature = (filename, csv, sample, name=None, parent_sample=None))]
    fn update_with_gaf(
        &self,
        filename: String,
        csv: String,
        sample: String,
        name: Option<String>,
        parent_sample: Option<String>,
    ) -> PyResult<String> {
        let name = name.unwrap_or_else(|| self.get_default_collection());
        run_write(&self.context, !self.in_transaction, |ctx| {
            update_with_gaf(
                ctx,
                &filename,
                &csv,
                &name,
                &sample,
                parent_sample.as_deref(),
            )
            .map(|_| format!("Updated from '{}'.", filename))
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to update from '{}': {e}", filename))
            })
        })
    }

    #[pyo3(signature = (filename, name=None, genotype=None, sample=None, parent_samples=None, in_place=false))]
    fn update_with_vcf(
        &self,
        filename: String,
        name: Option<String>,
        genotype: Option<String>,
        sample: Option<String>,
        parent_samples: Option<Vec<String>>,
        in_place: bool,
    ) -> PyResult<String> {
        let name = name.unwrap_or_else(|| self.get_default_collection());
        run_write(&self.context, !self.in_transaction, |ctx| {
            update_with_vcf(
                ctx,
                &filename,
                &name,
                genotype.clone().unwrap_or_default(),
                sample.as_deref(),
                parent_samples.unwrap_or_default(),
                in_place,
            )
            .map(|_| format!("Updated from '{}'.", filename))
            .map_err(|e| match e {
                VcfError::OperationError(OperationError::NoChanges) => PyRuntimeError::new_err(
                    "No changes made. Provide sample and genotype if missing from VCF.",
                ),
                _ => PyRuntimeError::new_err(format!("Failed to update from '{}': {e}", filename)),
            })
        })
    }

    #[pyo3(signature = (filename, sample, name=None, create_missing=false))]
    fn update_with_genbank(
        &self,
        filename: String,
        sample: String,
        name: Option<String>,
        create_missing: bool,
    ) -> PyResult<String> {
        use std::fs::File;
        let name = name.unwrap_or_else(|| self.get_default_collection());
        run_write(&self.context, !self.in_transaction, |ctx| {
            let file = File::open(&filename).map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to open '{}': {e}", filename))
            })?;
            update_with_genbank(
                ctx,
                &file,
                name.as_ref(),
                &sample,
                create_missing,
                &gen_models::operations::OperationInfo {
                    files: vec![gen_models::operations::OperationFile {
                        file_path: filename.clone(),
                        file_type: gen_models::file_types::FileTypes::GenBank,
                    }],
                    description: "Update from GenBank".to_string(),
                },
            )
            .map(|_| format!("Updated from '{}'.", filename))
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to update from '{}': {e}", filename))
            })
        })
    }

    #[pyo3(signature = (sequence, sample, new_sample, region_name, start, end, name=None, no_reference_path_update=false))]
    #[expect(clippy::too_many_arguments, reason = "mirrors underlying API")]
    fn update_with_sequence(
        &self,
        sequence: String,
        sample: String,
        new_sample: String,
        region_name: String,
        start: i64,
        end: i64,
        name: Option<String>,
        no_reference_path_update: bool,
    ) -> PyResult<String> {
        let name = name.unwrap_or_else(|| self.get_default_collection());
        run_write(&self.context, !self.in_transaction, |ctx| {
            update_with_sequence(
                ctx,
                &name,
                &sample,
                &new_sample,
                &region_name,
                start,
                end,
                &sequence,
                no_reference_path_update,
            )
            .map(|_| "Updated with sequence.".to_string())
            .map_err(|e| PyRuntimeError::new_err(format!("Update failed: {e}")))
        })
    }

    #[pyo3(signature = (name, sample, new_sample_name, path_name, start, end, parts_list))]
    #[expect(clippy::too_many_arguments, reason = "mirrors underlying API")]
    fn update_with_library(
        &self,
        name: Option<String>,
        sample: Option<String>,
        new_sample_name: String,
        path_name: String,
        start: i64,
        end: i64,
        parts_list: Vec<Vec<PySequencePart>>,
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
            update_with_library(
                ctx,
                &name,
                &sample,
                &new_sample_name,
                &path_name,
                start,
                end,
                rust_parts_list.clone(),
                None,
                None,
            )
            .map(|_| "Updated with library.".to_string())
            .map_err(|e| PyRuntimeError::new_err(format!("Update failed: {e}")))
        })
    }

    #[pyo3(signature = (name, sample, new_sample, path_name, start, end, library, parts))]
    #[expect(clippy::too_many_arguments, reason = "mirrors underlying API")]
    fn update_with_library_files(
        &self,
        name: Option<String>,
        sample: String,
        new_sample: String,
        path_name: String,
        start: i64,
        end: i64,
        library: String,
        parts: String,
    ) -> PyResult<String> {
        let parts_list = parse_library(&parts, &library)
            .map_err(|_| PyRuntimeError::new_err("Couldn't parse library files."))?;
        let name = name.unwrap_or_else(|| self.get_default_collection());
        run_write(&self.context, !self.in_transaction, |ctx| {
            update_with_library(
                ctx,
                &name,
                &sample,
                &new_sample,
                &path_name,
                start,
                end,
                parts_list.clone(),
                Some(&parts),
                Some(&library),
            )
            .map(|_| "Updated with library.".to_string())
            .map_err(|e| PyRuntimeError::new_err(format!("Update failed: {e}")))
        })
    }
}
