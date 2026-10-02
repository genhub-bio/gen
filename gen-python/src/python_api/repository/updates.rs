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
use pyo3_stub_gen::derive::gen_stub_pymethods;

use super::{PyRepository, run_context_operation_write};
use crate::python_api::{sample::PySample, sequence_part::PySequencePart};

#[gen_stub_pymethods]
#[pymethods]
impl PyRepository {
    /// Replace the region `region_name` of `sample` with the sequence in a FASTA file, storing the
    /// result as `new_sample`. Returns the new `Sample`.
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
        run_context_operation_write(
            &self.context,
            |ctx| {
                let operation_summary = update_with_fasta(
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
                    _ => PyRuntimeError::new_err(format!(
                        "Failed to update from '{}': {e}",
                        filename
                    )),
                })?;
                Ok((
                    self.block_groups_in_sample(&collection, &new_sample),
                    operation_summary,
                ))
            },
            |err| match err {
                OperationError::NoChanges => {
                    PyRuntimeError::new_err(format!("'{}': contents already exist", filename))
                }
                _ => {
                    PyRuntimeError::new_err(format!("Failed to update from '{}': {err}", filename))
                }
            },
        )
    }

    /// Apply the graph in a GFA file to `sample`, storing the result as `new_sample`. Returns the
    /// new `Sample`.
    #[pyo3(signature = (filename, sample, new_sample, collection=None))]
    fn update_with_gfa(
        &self,
        filename: String,
        sample: String,
        new_sample: String,
        collection: Option<String>,
    ) -> PyResult<PySample> {
        let collection = collection.unwrap_or_else(|| self.get_default_collection());
        run_context_operation_write(
            &self.context,
            |ctx| {
                let operation_summary =
                    update_with_gfa(ctx, &collection, &sample, &new_sample, &filename).map_err(
                        |e| {
                            PyRuntimeError::new_err(format!(
                                "Failed to update from '{}': {e}",
                                filename
                            ))
                        },
                    )?;
                Ok((
                    self.block_groups_in_sample(&collection, &new_sample),
                    operation_summary,
                ))
            },
            |err| PyRuntimeError::new_err(format!("Failed to update from '{}': {err}", filename)),
        )
    }

    /// Apply a GAF alignment file with its CSV of replacement sequences, writing the result to
    /// `sample` (derived from `parent_sample` when given). Returns that `Sample`.
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
        run_context_operation_write(
            &self.context,
            |ctx| {
                let operation_summary = update_with_gaf(
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
                Ok((
                    self.block_groups_in_sample(&collection, &sample),
                    operation_summary,
                ))
            },
            |err| PyRuntimeError::new_err(format!("Failed to update from '{}': {err}", filename)),
        )
    }

    /// Apply variants from a VCF file to the `reference` sample (a name or list of names), creating
    /// one
    /// `Sample` per VCF sample column (or only `sample`). With `in_place=True` the reference is
    /// edited
    /// instead. Returns the list of `Sample` objects.
    #[pyo3(signature = (filename, reference=None, genotype=None, sample=None, in_place=false, collection=None))]
    fn update_with_vcf(
        &self,
        filename: String,
        #[gen_stub(override_type(type_repr = "str | list[str] | None", imports = ()))]
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
        run_context_operation_write(
            &self.context,
            |ctx| {
                let (operation_summary, output_samples) = update_with_vcf(
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
                    _ => PyRuntimeError::new_err(format!(
                        "Failed to update from '{}': {e}",
                        filename
                    )),
                })?;
                let samples = output_samples
                    .into_iter()
                    .map(|sample_name| self.block_groups_in_sample(&collection, &sample_name))
                    .collect();
                Ok((samples, operation_summary))
            },
            |err| match err {
                OperationError::NoChanges => PyRuntimeError::new_err(
                    "No changes made. Provide sample and genotype if missing from VCF.",
                ),
                _ => {
                    PyRuntimeError::new_err(format!("Failed to update from '{}': {err}", filename))
                }
            },
        )
    }

    /// Update `sample` with the sequences and features of a GenBank file. `create_missing=True`
    /// allows graphs not yet in the sample. Returns the updated `Sample`.
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
        run_context_operation_write(
            &self.context,
            |ctx| {
                let file = File::open(&filename).map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to open '{}': {e}", filename))
                })?;
                let operation_summary = update_with_genbank(
                    ctx,
                    &file,
                    collection.as_ref(),
                    &sample,
                    create_missing,
                    &gen_models::operations::OperationInfo {
                        files: vec![{
                            let mut f =
                                gen_models::operations::OperationFile::new(filename.clone());
                            f.file_type = gen_models::file_types::FileTypes::GenBank;
                            f
                        }],
                        description: "Update from GenBank".to_string(),
                    },
                )
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to update from '{}': {e}", filename))
                })?;
                Ok((
                    self.block_groups_in_sample(&collection, &sample),
                    operation_summary,
                ))
            },
            |err| PyRuntimeError::new_err(format!("Failed to update from '{}': {err}", filename)),
        )
    }

    /// Replace the region `region_name` of `sample` with a literal sequence string, storing the
    /// result as `new_sample`. Returns the new `Sample`. For editing a copied sample in place,
    /// `graph.replace()` is usually simpler.
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
        run_context_operation_write(
            &self.context,
            |ctx| {
                let operation_summary = update_with_sequence(
                    ctx,
                    &collection,
                    &sample,
                    &new_sample,
                    &region_name,
                    &sequence,
                    no_reference_path_update,
                )
                .map_err(|e| PyRuntimeError::new_err(format!("Update failed: {e}")))?;
                Ok((
                    self.block_groups_in_sample(&collection, &new_sample),
                    operation_summary,
                ))
            },
            |err| PyRuntimeError::new_err(format!("Update failed: {err}")),
        )
    }

    /// Replace the region `path_name` of `sample` with a combinatorial library built from
    /// `parts_list` (columns of `SequencePart` alternatives), storing the result as
    /// `new_sample_name`. Returns the new `Sample`.
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
        run_context_operation_write(
            &self.context,
            |ctx| {
                let operation_summary = update_with_library(
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
                Ok((
                    self.block_groups_in_sample(&collection, &new_sample_name),
                    operation_summary,
                ))
            },
            |err| PyRuntimeError::new_err(format!("Update failed: {err}")),
        )
    }

    /// Like `update_with_library`, with the parts given as a named-parts FASTA (`parts`) and a
    /// headerless CSV (`library`). Returns the new `Sample`.
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
        run_context_operation_write(
            &self.context,
            |ctx| {
                let operation_summary = update_with_library(
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
                Ok((
                    self.block_groups_in_sample(&collection, &new_sample),
                    operation_summary,
                ))
            },
            |err| PyRuntimeError::new_err(format!("Update failed: {err}")),
        )
    }
}
