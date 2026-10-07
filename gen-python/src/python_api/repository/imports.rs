use std::{path::PathBuf, slice::from_ref};

use r#gen::{
    fasta::FastaError,
    graphs::combinatorial_library::parse_library,
    imports::{
        fasta::import_fasta,
        genbank::{GenBankImportOptions, import_genbank},
        gfa::{GFAImportError, import_gfa},
        library::{LibraryImportError, import_library},
        sequences::import_sequences,
    },
};
use gen_core::{HashId, NO_CHROMOSOME_INDEX, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand};
use gen_models::{
    annotations::{AnnotationFileChecksumOverrides, add_annotation_file},
    block_group::BlockGroup,
    block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
    db::GraphConnection,
    edge::Edge,
    errors::OperationError,
    path::Path,
    sample::{NewSample, Sample},
};
use pyo3::{
    exceptions::{PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
};
use pyo3_stub_gen::derive::gen_stub_pymethods;

use super::{PyRepository, run_context_operation_write};
use crate::python_api::{
    block_group::PySequenceGraph,
    hash_id::PyHashId,
    sample::PySample,
    sequence::{PySequence, library_parts},
    utils::{absolute_path_string, block_group_err_to_pyerr},
};

/// Biopython `SeqRecord`s are recognised by shape (`id` and `seq`) so gen never imports Biopython.
fn is_seq_record(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(value.hasattr("id")? && value.hasattr("seq")?)
}

/// Plain strings and Biopython `Seq`/`MutableSeq` (recognised by `reverse_complement`) are
/// sequences; anything else is rejected rather than stringified.
fn sequence_text(value: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(text) = value.extract::<String>() {
        return Ok(text);
    }
    if value.hasattr("reverse_complement")? {
        return value.str()?.extract();
    }
    Err(PyValueError::new_err(
        "sequence must be a string, a Biopython Seq, or a Biopython SeqRecord",
    ))
}

/// A sample is given by name or as a `Sample`. Returns the sample name and, for a `Sample`, its
/// collection. A `SequenceGraph` is rejected because it is unclear whether importing "into" one
/// would add to its sample or replace it.
fn sample_reference(sample: &Bound<'_, PyAny>) -> PyResult<(String, Option<String>)> {
    if let Ok(name) = sample.extract::<String>() {
        return Ok((name, None));
    }
    match sample.extract::<PySample>() {
        Ok(sample) => Ok((sample.sample_name, Some(sample.collection_name))),
        Err(_) => Err(PyTypeError::new_err(
            "sample must be a sample name or a Sample",
        )),
    }
}

/// Resolves the `(name, sequence)` pair for one import. A `SeqRecord` supplies its own name
/// (overridable with `name`); plain strings and `Seq`s need one.
fn parse_sequence_entry(
    sequence: &Bound<'_, PyAny>,
    name: Option<String>,
) -> PyResult<(String, String)> {
    let (name, text) = if is_seq_record(sequence)? {
        let name = match name {
            Some(name) => name,
            None => sequence.getattr("id")?.extract()?,
        };
        (name, sequence_text(&sequence.getattr("seq")?)?)
    } else {
        let name = name.ok_or_else(|| {
            PyValueError::new_err("name is required unless sequence is a Biopython SeqRecord")
        })?;
        (name, sequence_text(sequence)?)
    };
    if name.is_empty() {
        return Err(PyValueError::new_err("sequence name must not be empty"));
    }
    if text.is_empty() {
        return Err(PyValueError::new_err(format!("sequence '{name}' is empty")));
    }
    Ok((name, text))
}

fn sequence_import_error(error: FastaError) -> PyErr {
    match error {
        FastaError::OperationError(OperationError::NoChanges) => {
            PyRuntimeError::new_err("sequences: contents already exist")
        }
        _ => PyRuntimeError::new_err(format!("Failed to import sequences: {error}")),
    }
}

/// Close a newly imported path while retaining its linear path for sequence extraction.
fn circularize(conn: &GraphConnection, block_group_id: &HashId) -> PyResult<()> {
    let path = BlockGroup::get_current_path(conn, block_group_id, None)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let edges = Path::edges_for_path(conn, &path.id, None);
    let first_edge = edges
        .first()
        .ok_or_else(|| PyRuntimeError::new_err("Cannot circularize an empty path"))?;
    let last_edge = edges
        .last()
        .ok_or_else(|| PyRuntimeError::new_err("Cannot circularize an empty path"))?;
    let cycle_edge = Edge::create(
        conn,
        last_edge.source_node_id,
        last_edge.source_coordinate,
        last_edge.source_strand,
        first_edge.target_node_id,
        first_edge.target_coordinate,
        first_edge.target_strand,
    )
    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let sentinel_edge = Edge::create(
        conn,
        PATH_END_NODE_ID,
        0,
        Strand::Forward,
        PATH_START_NODE_ID,
        0,
        Strand::Forward,
    )
    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let closing_edges = [cycle_edge.id, sentinel_edge.id].map(|edge_id| BlockGroupEdgeData {
        block_group_id: *block_group_id,
        edge_id,
        chromosome_index: NO_CHROMOSOME_INDEX,
        phased: 0,
    });
    BlockGroupEdge::bulk_create(conn, &closing_edges);
    Ok(())
}

impl PyRepository {
    /// Importing a name already present in the sample would silently keep the old graph, so
    /// callers get an error instead. With `exist_ok`, a graph holding exactly the requested
    /// sequence is returned as is, which makes repeated notebook cells harmless; a graph holding
    /// anything else is still an error.
    fn existing_graph(
        &self,
        entry: &(String, String),
        collection: &str,
        sample: &str,
        exist_ok: bool,
    ) -> PyResult<Option<PySequenceGraph>> {
        let (name, text) = entry;
        let exists =
            Sample::get_block_groups(self.context.graph().conn(), collection, sample, None)
                .iter()
                .any(|block_group| block_group.name == *name);
        if !exists {
            return Ok(None);
        }
        let graph = self.get_block_group(collection, sample, name)?;
        if exist_ok {
            let mut sequences = BlockGroup::sequences_iter(
                self.context.graph().conn(),
                self.context.workspace(),
                &graph.id,
                None,
            )
            .map_err(block_group_err_to_pyerr)?;
            let holds_requested_sequence =
                sequences.next().as_ref() == Some(text) && sequences.next().is_none();
            if holds_requested_sequence {
                return Ok(Some(graph));
            }
            return Err(PyRuntimeError::new_err(format!(
                "sequence graph '{name}' already exists in sample '{sample}' with a different sequence; \
                 edit it with graph.replace(), or choose another name"
            )));
        }
        Err(PyRuntimeError::new_err(format!(
            "sequence graph '{name}' already exists in sample '{sample}'; fetch it from repo.samples, \
             choose another name, or pass exist_ok=True to reuse it when the sequence is identical"
        )))
    }

    fn import_sequence_entry(
        &self,
        entry: &(String, String),
        collection: &str,
        sample: &str,
        circular: bool,
        exist_ok: bool,
    ) -> PyResult<PySequenceGraph> {
        if let Some(graph) = self.existing_graph(entry, collection, sample, exist_ok)? {
            return Ok(graph);
        }
        run_context_operation_write(
            &self.context,
            |ctx| {
                let operation_summary = import_sequences(ctx, from_ref(entry), collection, sample)
                    .map_err(sequence_import_error)?;
                let graph = self.get_block_group(collection, sample, &entry.0)?;
                if circular {
                    circularize(ctx.graph().conn(), &graph.id)?;
                }
                Ok((graph, operation_summary))
            },
            |err| sequence_import_error(FastaError::OperationError(err)),
        )
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyRepository {
    /// Record an annotation file in the repository and return the `HashId` of the operation that
    /// recorded it.
    ///
    /// The format is inferred from the filename unless provided. A neighboring tabix index is
    /// discovered unless an index path is provided. The file's features then appear in
    /// `graph.annotations` and in plots of every sequence graph they land on. `name` sets the
    /// display and track name; `message` is the operation's commit message. `filename` and `index`
    /// are path strings; a relative path is resolved against the current working directory.
    #[pyo3(signature = (filename, format=None, index=None, name=None, message=None))]
    fn import_annotations(
        &self,
        filename: &str,
        format: Option<&str>,
        index: Option<&str>,
        name: Option<&str>,
        message: Option<&str>,
    ) -> PyResult<PyHashId> {
        let filename = absolute_path_string(filename)?;
        let index = index.map(absolute_path_string).transpose()?;
        add_annotation_file(
            &self.context,
            &filename,
            format,
            index.as_deref(),
            name,
            message,
            AnnotationFileChecksumOverrides::default(),
        )
        .map(PyHashId::from_dolt)
        .map_err(|error| PyRuntimeError::new_err(format!("Failed to import '{filename}': {error}")))
    }

    /// Import every record of a FASTA file into `sample` (default sample if omitted) and return the
    /// `Sample` holding one sequence graph per record. Fails if the same contents were already
    /// imported.
    ///
    /// `filename` is a path string; a relative path is resolved against the current working
    /// directory. `collection` defaults to the default collection.
    ///
    /// For a remote indexed BGZF FASTA, pass its `.fai` and `.gzi` files explicitly with `fai`
    /// and `gzi`.
    #[pyo3(signature = (filename, sample=None, collection=None, fai=None, gzi=None))]
    pub fn import_fasta(
        &self,
        filename: String,
        sample: Option<String>,
        collection: Option<String>,
        fai: Option<String>,
        gzi: Option<String>,
    ) -> PyResult<PySample> {
        let filename = absolute_path_string(&filename)?;
        let collection = collection.unwrap_or_else(|| self.get_default_collection());
        let sample = sample.unwrap_or_else(|| Sample::DEFAULT_NAME.to_string());
        run_context_operation_write(
            &self.context,
            |ctx| {
                let operation_summary = import_fasta(
                    ctx,
                    &filename,
                    &collection,
                    &sample,
                    fai.as_deref(),
                    gzi.as_deref(),
                )
                .map_err(|e| match e {
                    FastaError::OperationError(OperationError::NoChanges) => {
                        PyRuntimeError::new_err(format!("'{}': contents already exist", filename))
                    }
                    _ => PyRuntimeError::new_err(format!("Failed to import '{}': {e}", filename)),
                })?;
                Ok((
                    self.block_groups_in_sample(&collection, &sample),
                    operation_summary,
                ))
            },
            |err| match err {
                OperationError::NoChanges => {
                    PyRuntimeError::new_err(format!("'{}': contents already exist", filename))
                }
                _ => PyRuntimeError::new_err(format!("Failed to import '{}': {err}", filename)),
            },
        )
    }

    /// Import a FASTA file as the reference sample `reference`, which other samples (for example
    /// VCF variants) are derived against. Returns the `Sample`.
    ///
    /// `filename` is a path string; a relative path is resolved against the current working
    /// directory. `collection` defaults to the default collection.
    ///
    /// For a remote indexed BGZF FASTA, pass its `.fai` and `.gzi` files explicitly with `fai`
    /// and `gzi`.
    #[pyo3(signature = (filename, reference, collection=None, fai=None, gzi=None))]
    pub fn import_reference_fasta(
        &self,
        filename: String,
        reference: String,
        collection: Option<String>,
        fai: Option<String>,
        gzi: Option<String>,
    ) -> PyResult<PySample> {
        let filename = absolute_path_string(&filename)?;
        let collection = collection.unwrap_or_else(|| self.get_default_collection());
        run_context_operation_write(
            &self.context,
            |ctx| {
                Sample::get_or_create(
                    ctx.graph().conn(),
                    NewSample {
                        name: &reference,
                        is_reference: true,
                    },
                )
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to create reference sample: {e}"))
                })?;
                let operation_summary = import_fasta(
                    ctx,
                    &filename,
                    &collection,
                    &reference,
                    fai.as_deref(),
                    gzi.as_deref(),
                )
                .map_err(|e| match e {
                    FastaError::OperationError(OperationError::NoChanges) => {
                        PyRuntimeError::new_err(format!("'{}': contents already exist", filename))
                    }
                    _ => PyRuntimeError::new_err(format!("Failed to import '{}': {e}", filename)),
                })?;
                Ok((
                    self.block_groups_in_sample(&collection, &reference),
                    operation_summary,
                ))
            },
            |err| match err {
                OperationError::NoChanges => {
                    PyRuntimeError::new_err(format!("'{}': contents already exist", filename))
                }
                _ => PyRuntimeError::new_err(format!("Failed to import '{}': {err}", filename)),
            },
        )
    }

    /// Add one in-memory sequence to a sample as a new sequence graph, without a FASTA file, and
    /// return that graph.
    ///
    /// `sequence` is a string, a Biopython `Seq`, or a Biopython `SeqRecord`; a `SeqRecord`
    /// supplies its own `name` (its id) unless one is given. `sample` is a name or a `Sample`, and
    /// defaults to the default sample, "reference"; call it repeatedly with the same `sample` to
    /// build up a sample from several sequences. With `circular=True` the sequence is stored as a
    /// circular graph. Each call is its own operation, so use `import_fasta` for large files.
    ///
    /// `collection` defaults to the default collection. A graph with this name must not already
    /// exist in the sample; with `exist_ok=True`, one holding exactly this sequence is returned
    /// as it is instead of raising.
    #[pyo3(signature = (sequence, name=None, sample=None, circular=false, collection=None, *, exist_ok=false))]
    pub fn import_sequence(
        &self,
        sequence: &Bound<'_, PyAny>,
        name: Option<String>,
        #[gen_stub(override_type(type_repr = "str | Sample | None", imports = ()))] sample: Option<
            &Bound<'_, PyAny>,
        >,
        circular: bool,
        collection: Option<String>,
        exist_ok: bool,
    ) -> PyResult<PySequenceGraph> {
        let entry = parse_sequence_entry(sequence, name)?;
        let (sample, sample_collection) = match sample {
            Some(sample) => sample_reference(sample)?,
            None => (Sample::DEFAULT_NAME.to_string(), None),
        };
        let collection = collection
            .or(sample_collection)
            .unwrap_or_else(|| self.get_default_collection());
        self.import_sequence_entry(&entry, &collection, &sample, circular, exist_ok)
    }

    /// Like `import_sequence`, but adds to the reference sample `reference` (a name or a
    /// `Sample`); `sequence`, `name`, `circular`, `collection` and `exist_ok` behave the same.
    #[pyo3(signature = (sequence, reference, name=None, circular=false, collection=None, *, exist_ok=false))]
    pub fn import_reference_sequence(
        &self,
        sequence: &Bound<'_, PyAny>,
        #[gen_stub(override_type(type_repr = "str | Sample", imports = ()))] reference: &Bound<
            '_,
            PyAny,
        >,
        name: Option<String>,
        circular: bool,
        collection: Option<String>,
        exist_ok: bool,
    ) -> PyResult<PySequenceGraph> {
        let entry = parse_sequence_entry(sequence, name)?;
        let (reference, reference_collection) = sample_reference(reference)?;
        let collection = collection
            .or(reference_collection)
            .unwrap_or_else(|| self.get_default_collection());
        if let Some(graph) = self.existing_graph(&entry, &collection, &reference, exist_ok)? {
            return Ok(graph);
        }
        run_context_operation_write(
            &self.context,
            |ctx| {
                Sample::get_or_create(
                    ctx.graph().conn(),
                    NewSample {
                        name: &reference,
                        is_reference: true,
                    },
                )
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to create reference sample: {e}"))
                })?;
                let operation_summary =
                    import_sequences(ctx, from_ref(&entry), &collection, &reference)
                        .map_err(sequence_import_error)?;
                let graph = self.get_block_group(&collection, &reference, &entry.0)?;
                if circular {
                    circularize(ctx.graph().conn(), &graph.id)?;
                }
                Ok((graph, operation_summary))
            },
            |err| sequence_import_error(FastaError::OperationError(err)),
        )
    }

    /// Import a GFA file as one sequence graph, preserving its nodes and edges, and return that
    /// `SequenceGraph`.
    ///
    /// `sample` defaults to the default sample. `filename` is a path string; a relative path is
    /// resolved against the current working directory. `collection` defaults to the default
    /// collection.
    #[pyo3(signature = (filename, sample=None, collection=None))]
    fn import_gfa(
        &self,
        filename: String,
        sample: Option<String>,
        collection: Option<String>,
    ) -> PyResult<PySequenceGraph> {
        let filename = absolute_path_string(&filename)?;
        let collection = collection.unwrap_or_else(|| self.get_default_collection());
        let sample = sample.unwrap_or_else(|| Sample::DEFAULT_NAME.to_string());
        run_context_operation_write(
            &self.context,
            |ctx| {
                let operation_summary = import_gfa(
                    ctx,
                    &PathBuf::from(&filename),
                    &collection,
                    &sample,
                )
                .map_err(|e| match e {
                    GFAImportError::OperationError(OperationError::NoChanges) => {
                        PyRuntimeError::new_err(format!("'{}': already exists", filename))
                    }
                    _ => PyRuntimeError::new_err(format!("Failed to import '{}': {e}", filename)),
                })?;
                Ok((
                    self.get_block_group(&collection, &sample, "")?,
                    operation_summary,
                ))
            },
            |err| match err {
                OperationError::NoChanges => {
                    PyRuntimeError::new_err(format!("'{}': already exists", filename))
                }
                _ => PyRuntimeError::new_err(format!("Failed to import '{}': {err}", filename)),
            },
        )
    }

    /// Import a GenBank file, including its features (readable through `graph.annotations`), and
    /// return the `Sample`.
    ///
    /// `sample` defaults to the default sample. `filename` is a path string; a relative path is
    /// resolved against the current working directory. `collection` defaults to the default
    /// collection.
    #[pyo3(signature = (filename, sample=None, collection=None))]
    fn import_genbank(
        &self,
        filename: String,
        sample: Option<String>,
        collection: Option<String>,
    ) -> PyResult<PySample> {
        let filename = absolute_path_string(&filename)?;
        use std::fs::File;
        let collection = collection.unwrap_or_else(|| self.get_default_collection());
        let sample = sample.unwrap_or_else(|| Sample::DEFAULT_NAME.to_string());
        run_context_operation_write(
            &self.context,
            |ctx| {
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
                let operation_summary = import_genbank(
                    ctx,
                    &mut reader,
                    collection.as_ref(),
                    &sample,
                    gen_models::operations::OperationInfo {
                        files: vec![{
                            let mut f =
                                gen_models::operations::OperationFile::new(filename.clone());
                            f.file_type = gen_models::file_types::FileTypes::GenBank;
                            f
                        }],
                        description: "GenBank Import".to_string(),
                    },
                    GenBankImportOptions::default(),
                )
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to import '{}': {e}", filename))
                })?;
                Ok((
                    self.block_groups_in_sample(&collection, &sample),
                    operation_summary,
                ))
            },
            |err| PyRuntimeError::new_err(format!("Failed to import '{}': {err}", filename)),
        )
    }

    /// Build a combinatorial library from `parts_list`, a list of columns each holding alternative
    /// named `Sequence` objects, and return the resulting `SequenceGraph`. Every path through the
    /// graph is one assembled design; read them with `graph.all_sequences()`.
    ///
    /// `library_name` names the new graph; `sample` and `collection` default to the default sample
    /// and collection.
    #[pyo3(signature = (library_name, parts_list, sample=None, collection=None))]
    fn import_library(
        &self,
        library_name: String,
        parts_list: Vec<Vec<PySequence>>,
        sample: Option<String>,
        collection: Option<String>,
    ) -> PyResult<PySequenceGraph> {
        let collection = collection.unwrap_or_else(|| self.get_default_collection());
        let sample = sample.unwrap_or_else(|| Sample::DEFAULT_NAME.to_string());
        let rust_parts_list = library_parts(&parts_list)?;
        run_context_operation_write(
            &self.context,
            |ctx| {
                let operation_summary = import_library(
                    ctx,
                    &collection,
                    &sample,
                    &library_name,
                    rust_parts_list.clone(),
                    None,
                    None,
                )
                .map_err(|e| match e {
                    LibraryImportError::OperationError(OperationError::NoChanges) => {
                        PyRuntimeError::new_err(format!(
                            "Library '{}': already exists",
                            library_name
                        ))
                    }
                    _ => PyRuntimeError::new_err(format!(
                        "Failed to import library '{}': {e}",
                        library_name
                    )),
                })?;
                Ok((
                    self.get_block_group(&collection, &sample, &library_name)?,
                    operation_summary,
                ))
            },
            |err| match err {
                OperationError::NoChanges => {
                    PyRuntimeError::new_err(format!("Library '{}': already exists", library_name))
                }
                _ => PyRuntimeError::new_err(format!(
                    "Failed to import library '{}': {err}",
                    library_name
                )),
            },
        )
    }

    /// Build a combinatorial library from files: `parts` is a FASTA of named parts and `library` a
    /// headerless CSV with one column per slot and alternatives in rows. Returns the
    /// `SequenceGraph`.
    ///
    /// `library_name` names the new graph; `sample` and `collection` default to the default sample
    /// and collection. `parts` and `library` are path strings; a relative path is resolved against
    /// the current working directory.
    #[pyo3(signature = (library_name, parts, library, sample=None, collection=None))]
    fn import_library_files(
        &self,
        library_name: String,
        parts: String,
        library: String,
        sample: Option<String>,
        collection: Option<String>,
    ) -> PyResult<PySequenceGraph> {
        let parts = absolute_path_string(&parts)?;
        let library = absolute_path_string(&library)?;
        let parts_list = parse_library(&parts, &library)
            .map_err(|e| PyRuntimeError::new_err(format!("Problem parsing library files: {e}")))?;
        let collection = collection.unwrap_or_else(|| self.get_default_collection());
        let sample = sample.unwrap_or_else(|| Sample::DEFAULT_NAME.to_string());
        run_context_operation_write(
            &self.context,
            |ctx| {
                let operation_summary = import_library(
                    ctx,
                    &collection,
                    &sample,
                    &library_name,
                    parts_list.clone(),
                    Some(&parts),
                    Some(&library),
                )
                .map_err(|e| match e {
                    LibraryImportError::OperationError(OperationError::NoChanges) => {
                        PyRuntimeError::new_err(format!(
                            "Library '{}': already exists",
                            library_name
                        ))
                    }
                    _ => PyRuntimeError::new_err(format!(
                        "Failed to import library '{}': {e}",
                        library_name
                    )),
                })?;
                Ok((
                    self.get_block_group(&collection, &sample, &library_name)?,
                    operation_summary,
                ))
            },
            |err| match err {
                OperationError::NoChanges => {
                    PyRuntimeError::new_err(format!("Library '{}': already exists", library_name))
                }
                _ => PyRuntimeError::new_err(format!(
                    "Failed to import library '{}': {err}",
                    library_name
                )),
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use r#gen::test_helpers::setup_gen_on_disk;
    use gen_core::{NO_CHROMOSOME_INDEX, is_end_node, is_start_node};
    use gen_models::{block_group::BlockGroup, block_group_edge::BlockGroupEdge, path::Path};
    use pyo3::{Python, types::PyString};

    use super::PyRepository;

    #[test]
    fn test_import_sequence_circular_closes_graph_and_preserves_path() {
        Python::initialize();
        Python::attach(|python| {
            let repository = PyRepository {
                context: setup_gen_on_disk(),
            };
            let sequence = PyString::new(python, "ATCG");
            for circular in [false, true] {
                let name = if circular { "circular" } else { "linear" };
                let graph = repository
                    .import_sequence(
                        sequence.as_any(),
                        Some(name.to_string()),
                        None,
                        circular,
                        None,
                        false,
                    )
                    .unwrap();
                let conn = repository.context.graph().conn();
                let edges = BlockGroupEdge::edges_for_block_group(conn, &graph.id, None);
                let closing_edges = edges
                    .iter()
                    .filter(|edge| edge.chromosome_index == NO_CHROMOSOME_INDEX)
                    .collect::<Vec<_>>();
                assert_eq!(closing_edges.len(), if circular { 2 } else { 0 });
                if circular {
                    assert!(closing_edges.iter().any(|edge| {
                        edge.edge.source_node_id == edge.edge.target_node_id
                            && edge.edge.source_coordinate == 4
                            && edge.edge.target_coordinate == 0
                    }));
                    assert!(closing_edges.iter().any(|edge| {
                        is_end_node(edge.edge.source_node_id)
                            && is_start_node(edge.edge.target_node_id)
                    }));
                }
                let path = BlockGroup::get_current_path(conn, &graph.id, None).unwrap();
                assert_eq!(Path::edges_for_path(conn, &path.id, None).len(), 2);
            }
        });
    }
}
