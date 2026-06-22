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
use crate::python_api::{sample::PySample, sequence_part::PySequencePart};

#[pymethods]
impl PyRepository {
    #[pyo3(signature = (filename, sample, new_sample, region_name, collection=None))]
    fn update_with_fasta(
        &self,
        filename: String,
        sample: String,
        new_sample: String,
        region_name: String,
        collection: Option<String>,
    ) -> PyResult<PySample> {
        let collection = collection.unwrap_or_else(|| self.get_default_collection());
        run_write(&self.context, !self.in_transaction, |ctx| {
            update_with_fasta(
                ctx,
                &collection,
                &sample,
                &new_sample,
                &region_name,
                &filename,
                false,
            )
            .map_err(|e| match e {
                FastaError::OperationError(OperationError::NoChanges) => {
                    PyRuntimeError::new_err(format!("'{}': contents already exist", filename))
                }
                _ => PyRuntimeError::new_err(format!("Failed to update from '{}': {e}", filename)),
            })?;
            Ok(self.block_groups_in_sample(&collection, &new_sample))
        })
    }

    #[pyo3(signature = (filename, sample, new_sample, collection=None))]
    fn update_with_gfa(
        &self,
        filename: String,
        sample: String,
        new_sample: String,
        collection: Option<String>,
    ) -> PyResult<PySample> {
        let collection = collection.unwrap_or_else(|| self.get_default_collection());
        run_write(&self.context, !self.in_transaction, |ctx| {
            update_with_gfa(ctx, &collection, &sample, &new_sample, &filename).map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to update from '{}': {e}", filename))
            })?;
            Ok(self.block_groups_in_sample(&collection, &new_sample))
        })
    }

    #[pyo3(signature = (filename, csv, sample, parent_sample=None, collection=None))]
    fn update_with_gaf(
        &self,
        filename: String,
        csv: String,
        sample: String,
        parent_sample: Option<String>,
        collection: Option<String>,
    ) -> PyResult<PySample> {
        let collection = collection.unwrap_or_else(|| self.get_default_collection());
        run_write(&self.context, !self.in_transaction, |ctx| {
            update_with_gaf(
                ctx,
                &filename,
                &csv,
                &collection,
                &sample,
                parent_sample.as_deref(),
            )
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to update from '{}': {e}", filename))
            })?;
            Ok(self.block_groups_in_sample(&collection, &sample))
        })
    }

    #[pyo3(signature = (filename, reference=None, genotype=None, sample=None, in_place=false, collection=None))]
    fn update_with_vcf(
        &self,
        filename: String,
        reference: Option<Bound<'_, PyAny>>,
        genotype: Option<String>,
        sample: Option<String>,
        in_place: bool,
        collection: Option<String>,
    ) -> PyResult<Vec<PySample>> {
        let parent_samples = match reference {
            None => vec![],
            Some(ref obj) => {
                if let Ok(s) = obj.extract::<String>() {
                    vec![s]
                } else {
                    obj.extract::<Vec<String>>().map_err(|_| {
                        PyRuntimeError::new_err("reference must be a string or list of strings")
                    })?
                }
            }
        };
        let collection = collection.unwrap_or_else(|| self.get_default_collection());
        run_write(&self.context, !self.in_transaction, |ctx| {
            let (_, output_samples) = update_with_vcf(
                ctx,
                &filename,
                &collection,
                genotype.clone().unwrap_or_default(),
                sample.as_deref(),
                parent_samples.clone(),
                in_place,
            )
            .map_err(|e| match e {
                VcfError::OperationError(OperationError::NoChanges) => PyRuntimeError::new_err(
                    "No changes made. Provide sample and genotype if missing from VCF.",
                ),
                _ => PyRuntimeError::new_err(format!("Failed to update from '{}': {e}", filename)),
            })?;
            Ok(output_samples
                .into_iter()
                .map(|sample_name| self.block_groups_in_sample(&collection, &sample_name))
                .collect())
        })
    }

    #[pyo3(signature = (filename, sample, create_missing=false, collection=None))]
    fn update_with_genbank(
        &self,
        filename: String,
        sample: String,
        create_missing: bool,
        collection: Option<String>,
    ) -> PyResult<PySample> {
        use std::fs::File;
        let collection = collection.unwrap_or_else(|| self.get_default_collection());
        run_write(&self.context, !self.in_transaction, |ctx| {
            let file = File::open(&filename).map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to open '{}': {e}", filename))
            })?;
            update_with_genbank(
                ctx,
                &file,
                collection.as_ref(),
                &sample,
                create_missing,
                &gen_models::operations::OperationInfo {
                    files: vec![{
                        let mut f = gen_models::operations::OperationFile::new(filename.clone());
                        f.file_type = gen_models::file_types::FileTypes::GenBank;
                        f
                    }],
                    description: "Update from GenBank".to_string(),
                },
            )
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to update from '{}': {e}", filename))
            })?;
            Ok(self.block_groups_in_sample(&collection, &sample))
        })
    }

    #[pyo3(signature = (sequence, sample, new_sample, region_name, no_reference_path_update=false, collection=None))]
    fn update_with_sequence(
        &self,
        sequence: String,
        sample: String,
        new_sample: String,
        region_name: String,
        no_reference_path_update: bool,
        collection: Option<String>,
    ) -> PyResult<PySample> {
        let collection = collection.unwrap_or_else(|| self.get_default_collection());
        run_write(&self.context, !self.in_transaction, |ctx| {
            update_with_sequence(
                ctx,
                &collection,
                &sample,
                &new_sample,
                &region_name,
                &sequence,
                no_reference_path_update,
            )
            .map_err(|e| PyRuntimeError::new_err(format!("Update failed: {e}")))?;
            Ok(self.block_groups_in_sample(&collection, &new_sample))
        })
    }

    #[pyo3(signature = (sample, new_sample_name, path_name, parts_list, collection=None))]
    fn update_with_library(
        &self,
        sample: Option<String>,
        new_sample_name: String,
        path_name: String,
        parts_list: Vec<Vec<PySequencePart>>,
        collection: Option<String>,
    ) -> PyResult<PySample> {
        let collection = collection.unwrap_or_else(|| self.get_default_collection());
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
                &collection,
                &sample,
                &new_sample_name,
                &path_name,
                rust_parts_list.clone(),
                None,
                None,
            )
            .map_err(|e| PyRuntimeError::new_err(format!("Update failed: {e}")))?;
            Ok(self.block_groups_in_sample(&collection, &new_sample_name))
        })
    }

    #[pyo3(signature = (sample, new_sample, path_name, library, parts, collection=None))]
    fn update_with_library_files(
        &self,
        sample: String,
        new_sample: String,
        path_name: String,
        library: String,
        parts: String,
        collection: Option<String>,
    ) -> PyResult<PySample> {
        let parts_list = parse_library(&parts, &library)
            .map_err(|_| PyRuntimeError::new_err("Couldn't parse library files."))?;
        let collection = collection.unwrap_or_else(|| self.get_default_collection());
        run_write(&self.context, !self.in_transaction, |ctx| {
            update_with_library(
                ctx,
                &collection,
                &sample,
                &new_sample,
                &path_name,
                parts_list.clone(),
                Some(&parts),
                Some(&library),
            )
            .map_err(|e| PyRuntimeError::new_err(format!("Update failed: {e}")))?;
            Ok(self.block_groups_in_sample(&collection, &new_sample))
        })
    }
}
