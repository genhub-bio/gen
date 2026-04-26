use std::{collections::{HashMap, HashSet}, fs, path::PathBuf};

use r#gen::{
    self,
    core::HashId,
    fasta::FastaError,
    graphs::{
        combinatorial_library::{SequencePart, parse_library},
        graph_search::{GenGraphMatcher, SeedIndex, SequenceKind},
    },
    imports::{gfa::GFAImportError, library::LibraryImportError},
    updates::vcf::VcfError,
};
use gen_core::{
    NO_CHROMOSOME_INDEX, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand, config::Workspace,
};
use gen_models::{
    block_group::BlockGroup,
    block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
    collection::Collection,
    db::DbContext,
    edge::{Edge, EdgeData},
    errors::OperationError,
    node::Node,
    operations::{Defaults, OperationInfo},
    path::Path,
    sample::Sample,
    sequence::Sequence,
    session_operations::{end_operation, start_operation},
    traits::Query,
};
use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyDict};

use super::{
    block_group::PyBlockGroup,
    graph_search::PyGraphLocus,
    jupyter_widget::{PyGraphController, build_and_display_widget},
    block::PyBlock,
    sequence_part::PySequencePart,
    utils::{path_to_py_path, py_query, sqlite_err_to_pyerr},
};

fn parse_strand(s: &str) -> Strand {
    match s {
        "+" | "forward" | "Forward" => Strand::Forward,
        "-" | "reverse" | "Reverse" => Strand::Reverse,
        _ => Strand::Forward,
    }
}

fn parse_sequence_kind(s: &str) -> PyResult<SequenceKind> {
    match s {
        "dna" => Ok(SequenceKind::Dna),
        "ssdna" => Ok(SequenceKind::SsDna),
        "protein" => Ok(SequenceKind::Protein),
        _ => Err(PyRuntimeError::new_err(format!(
            "Unknown sequence_kind '{s}'; use 'dna', 'ssdna', or 'protein'"
        ))),
    }
}

/// Wraps all transaction boilerplate for write operations.
///
/// When `managed` is true (no user transaction is open), this function calls
/// `track_database`, opens a transaction, calls `op`, then commits or rolls
/// back. When `managed` is false the caller is inside a `with repo.transaction()`
/// block: the transaction is already open and `__exit__` owns the final
/// commit/rollback, so we just call `op` and propagate any error.
fn run_write<F, T>(context: &DbContext, managed: bool, op: F) -> PyResult<T>
where
    F: FnOnce(&DbContext) -> PyResult<T>,
{
    if managed {
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();
        r#gen::track_database(conn, op_conn)
            .map_err(|e| PyRuntimeError::new_err(format!("Error tracking database: {e}")))?;
        conn.execute("BEGIN TRANSACTION", [])
            .map_err(sqlite_err_to_pyerr)?;
        op_conn
            .execute("BEGIN TRANSACTION", [])
            .map_err(sqlite_err_to_pyerr)?;
    }
    match op(context) {
        Ok(val) => {
            if managed {
                context
                    .graph()
                    .conn()
                    .execute("END TRANSACTION", [])
                    .map_err(sqlite_err_to_pyerr)?;
                context
                    .operations()
                    .conn()
                    .execute("END TRANSACTION", [])
                    .map_err(sqlite_err_to_pyerr)?;
            }
            Ok(val)
        }
        Err(err) => {
            if managed {
                context.graph().conn().execute("ROLLBACK", []).ok();
                context.operations().conn().execute("ROLLBACK", []).ok();
            }
            Err(err)
        }
    }
}

/// The main entry point for the gen Python module.
///
/// This class manages the database connection and provides methods for
/// querying and manipulating the database.
#[pyclass(name = "Repository", unsendable)]
pub struct PyRepository {
    pub context: DbContext,
    in_transaction: bool,
}

impl PyRepository {
    fn get_default_collection(&self) -> String {
        Defaults::get(self.context.operations().conn())
            .and_then(|d| d.collection_name)
            .unwrap_or_else(|| "default".to_string())
    }
}

#[pymethods]
impl PyRepository {
    #[new]
    #[pyo3(signature = (path = Option::<String>::None))]
    fn new(path: Option<String>) -> PyResult<Self> {
        let workspace = match path {
            Some(path_str) => Workspace::new(path_str),
            None => Workspace::from_current_dir(),
        };

        let gen_dir = workspace.ensure_gen_dir();
        let operations_path = gen_dir.join("gen.db");
        let operations_conn =
            r#gen::get_operation_connection(Some(operations_path)).map_err(|err| {
                PyRuntimeError::new_err(format!("Failed to open operations database: {err}"))
            })?;

        let db_path = {
            let mut stmt = operations_conn
                .prepare("select db_name from defaults where id = 1;")
                .map_err(|err| {
                    PyRuntimeError::new_err(format!("Failed to load defaults: {err}"))
                })?;
            let row: Option<String> = stmt.query_row([], |row| row.get(0)).ok();
            row.unwrap_or_else(|| gen_dir.join("default.db").to_string_lossy().into_owned())
        };

        let graph_conn = r#gen::get_connection(PathBuf::from(&db_path)).map_err(|err| {
            PyRuntimeError::new_err(format!("Failed to open database '{db_path}': {err}"))
        })?;

        Ok(PyRepository {
            context: DbContext::new(workspace, graph_conn, operations_conn),
            in_transaction: false,
        })
    }

    #[getter]
    fn get_gen_dir(&self, py: Python) -> PyResult<PyObject> {
        path_to_py_path(py, &self.context.workspace().ensure_gen_dir())
    }

    #[getter]
    fn get_db_path(&self, py: Python) -> PyResult<PyObject> {
        let path = self
            .context
            .graph()
            .path()
            .map(|p| p.to_path_buf())
            .unwrap_or_default();
        path_to_py_path(py, &path)
    }

    // -------------------------------------------------------------------------
    // Transaction context manager
    // -------------------------------------------------------------------------

    /// Returns self so that Python's `with` statement calls `__enter__`/`__exit__`
    /// on this repository, batching multiple operations into one transaction.
    ///
    /// Example:
    ///     with repo.transaction():
    ///         repo.import_fasta("reference.fasta")
    ///         repo.import_gfa("graph.gfa")
    fn transaction(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __enter__(mut slf: PyRefMut<'_, Self>) -> PyResult<()> {
        {
            let conn = slf.context.graph().conn();
            let op_conn = slf.context.operations().conn();
            r#gen::track_database(conn, op_conn)
                .map_err(|e| PyRuntimeError::new_err(format!("Error tracking database: {e}")))?;
            conn.execute("BEGIN TRANSACTION", [])
                .map_err(sqlite_err_to_pyerr)?;
            op_conn
                .execute("BEGIN TRANSACTION", [])
                .map_err(sqlite_err_to_pyerr)?;
        }
        slf.in_transaction = true;
        Ok(())
    }

    fn __exit__(
        mut slf: PyRefMut<'_, Self>,
        exc_type: Option<&Bound<'_, PyAny>>,
        _exc_val: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        slf.in_transaction = false;
        if exc_type.is_some() {
            slf.context.graph().conn().execute("ROLLBACK", []).ok();
            slf.context.operations().conn().execute("ROLLBACK", []).ok();
        } else {
            slf.context
                .graph()
                .conn()
                .execute("END TRANSACTION", [])
                .map_err(sqlite_err_to_pyerr)?;
            slf.context
                .operations()
                .conn()
                .execute("END TRANSACTION", [])
                .map_err(sqlite_err_to_pyerr)?;
        }
        Ok(false)
    }

    // -------------------------------------------------------------------------
    // Raw database access
    // -------------------------------------------------------------------------

    fn execute(&self, query: &str) -> PyResult<()> {
        self.context
            .graph()
            .conn()
            .execute(query, [])
            .map_err(sqlite_err_to_pyerr)?;
        Ok(())
    }

    fn query(&self, query: &str) -> PyResult<Vec<Vec<PyObject>>> {
        py_query(self.context.graph().conn(), query)
    }

    // -------------------------------------------------------------------------
    // BlockGroup queries
    // -------------------------------------------------------------------------

    fn get_block_group_by_id(&self, id: &HashId) -> PyResult<PyBlockGroup> {
        let conn = self.context.graph().conn();
        let block_group = BlockGroup::get_by_id(conn, id);
        Ok(PyBlockGroup {
            id: block_group.id,
            collection_name: block_group.collection_name,
            sample_name: block_group.sample_name,
            name: block_group.name,
            db_path: self.context.graph().path().map(|p| p.to_path_buf()),
        })
    }

    fn get_block_groups(&self) -> PyResult<Vec<PyBlockGroup>> {
        let conn = self.context.graph().conn();
        let db_path = self.context.graph().path().map(|p| p.to_path_buf());
        Ok(BlockGroup::all(conn)
            .into_iter()
            .map(|bg| PyBlockGroup {
                id: bg.id,
                collection_name: bg.collection_name,
                sample_name: bg.sample_name,
                name: bg.name,
                db_path: db_path.clone(),
            })
            .collect())
    }

    fn get_block_groups_by_collection(&self, collection_name: &str) -> PyResult<Vec<PyBlockGroup>> {
        let conn = self.context.graph().conn();
        let db_path = self.context.graph().path().map(|p| p.to_path_buf());
        Ok(BlockGroup::query(
            conn,
            "SELECT * FROM block_groups WHERE collection_name = ?1",
            rusqlite::params![collection_name],
        )
        .into_iter()
        .map(|bg| PyBlockGroup {
            id: bg.id,
            collection_name: bg.collection_name,
            sample_name: bg.sample_name,
            name: bg.name,
            db_path: db_path.clone(),
        })
        .collect())
    }

    fn create_block_group(
        &self,
        name: String,
        collection_name: String,
        sample_name: String,
    ) -> PyResult<PyBlockGroup> {
        let conn = self.context.graph().conn();
        let block_group = BlockGroup::create(
            conn,
            gen_models::block_group::NewBlockGroup {
                collection_name: &collection_name,
                sample_name: &sample_name,
                name: &name,
                ..Default::default()
            },
        );
        Ok(PyBlockGroup {
            id: block_group.id,
            collection_name: block_group.collection_name,
            sample_name: block_group.sample_name,
            name: block_group.name,
            db_path: self.context.graph().path().map(|p| p.to_path_buf()),
        })
    }

    /// Create a block group from a sequence string.
    ///
    /// Creates a single node holding the full sequence, wired between
    /// start/end sentinel nodes.  A path named `name` is also created so the
    /// sequence is immediately traversable.  The operation is recorded in the
    /// operations database.
    ///
    /// `sequence_start` and `sequence_end` let you expose only a window of the
    /// stored sequence (half-open, 0-based).  Defaults: 0 and `len(sequence)`.
    #[pyo3(signature = (name, sequence, collection_name=None, sample_name=None, sequence_start=None, sequence_end=None))]
    fn create_block_group_from_sequence(
        &self,
        name: String,
        sequence: String,
        collection_name: Option<String>,
        sample_name: Option<String>,
        sequence_start: Option<i64>,
        sequence_end: Option<i64>,
    ) -> PyResult<PyBlockGroup> {
        let collection_name = collection_name.unwrap_or_else(|| self.get_default_collection());
        let sample_name = sample_name.unwrap_or_else(|| Sample::DEFAULT_NAME.to_string());

        run_write(&self.context, !self.in_transaction, |ctx| {
            let conn = ctx.graph().conn();
            let mut session = start_operation(conn);

            Collection::create(conn, &collection_name);
            Sample::get_or_create(conn, &sample_name);

            let block_group = BlockGroup::create(
                conn,
                gen_models::block_group::NewBlockGroup {
                    collection_name: &collection_name,
                    sample_name: &sample_name,
                    name: &name,
                    ..Default::default()
                },
            );

            let seq = Sequence::new()
                .sequence_type("DNA")
                .sequence(sequence.as_str())
                .save(conn);
            let seq_len = seq.length;
            let node_id = HashId::uuid7();
            Node::create(conn, &seq.hash, &node_id);

            let start = sequence_start.unwrap_or(0);
            let end = sequence_end.unwrap_or(seq_len);

            let edges = vec![
                EdgeData {
                    source_node_id: PATH_START_NODE_ID,
                    source_coordinate: 0,
                    source_strand: Strand::Forward,
                    target_node_id: node_id,
                    target_coordinate: start,
                    target_strand: Strand::Forward,
                },
                EdgeData {
                    source_node_id: node_id,
                    source_coordinate: end,
                    source_strand: Strand::Forward,
                    target_node_id: PATH_END_NODE_ID,
                    target_coordinate: 0,
                    target_strand: Strand::Forward,
                },
            ];
            let edge_ids = Edge::bulk_create(conn, &edges);
            BlockGroupEdge::bulk_create(
                conn,
                &edge_ids
                    .iter()
                    .map(|id| BlockGroupEdgeData {
                        block_group_id: block_group.id,
                        edge_id: *id,
                        chromosome_index: NO_CHROMOSOME_INDEX,
                        phased: 0,
                    })
                    .collect::<Vec<_>>(),
            );
            Path::create(conn, &name, &block_group.id, &edge_ids);

            end_operation(
                ctx,
                &mut session,
                &OperationInfo {
                    files: vec![],
                    description: "python_block_group_create".to_string(),
                },
                &format!("Created block group '{name}' from sequence"),
                None,
            )
            .map_err(|e| PyRuntimeError::new_err(format!("Operation error: {e}")))?;

            Ok(PyBlockGroup {
                id: block_group.id,
                collection_name: block_group.collection_name,
                sample_name: block_group.sample_name,
                name: block_group.name,
                db_path: ctx.graph().path().map(|p| p.to_path_buf()),
            })
        })
    }

    /// Create a block group from a NetworkX `DiGraph`.
    ///
    /// Each node must have a `"sequence"` attribute.  An optional `"node_id"`
    /// attribute (64-char hex string) re-uses an existing node; otherwise a
    /// fresh UUID-7 node hash is generated.
    ///
    /// Edge attributes `"source_strand"` and `"target_strand"` accept `"+"`/
    /// `"-"` or `"forward"`/`"reverse"` (case-insensitive); both default to
    /// forward.
    ///
    /// Source nodes (no in-edges) get a `start → node` sentinel edge; sink
    /// nodes (no out-edges) get a `node → end` sentinel edge.
    ///
    /// If `path_name` is given the graph must be a simple linear chain (exactly
    /// one source and one sink, no branching) — a `Path` is then recorded.
    #[pyo3(signature = (graph, name, collection_name=None, sample_name=None, path_name=None))]
    fn create_block_group_from_graph(
        &self,
        py: Python<'_>,
        graph: Bound<'_, PyAny>,
        name: String,
        collection_name: Option<String>,
        sample_name: Option<String>,
        path_name: Option<String>,
    ) -> PyResult<PyBlockGroup> {
        let collection_name = collection_name.unwrap_or_else(|| self.get_default_collection());
        let sample_name = sample_name.unwrap_or_else(|| Sample::DEFAULT_NAME.to_string());

        run_write(&self.context, !self.in_transaction, |ctx| {
            let conn = ctx.graph().conn();
            let mut session = start_operation(conn);

            Collection::create(conn, &collection_name);
            Sample::get_or_create(conn, &sample_name);

            let block_group = BlockGroup::create(
                conn,
                gen_models::block_group::NewBlockGroup {
                    collection_name: &collection_name,
                    sample_name: &sample_name,
                    name: &name,
                    ..Default::default()
                },
            );

            // Each node key is a PyBlock carrying sequence, node_id, start, end.
            // "start" and "end" string nodes are sentinel placeholders — skip them.
            let mut all_blocks: Vec<PyBlock> = Vec::new();
            for node_obj in graph.getattr("nodes")?.call0()?.try_iter()? {
                let node_obj = node_obj?;
                if let Ok(s) = node_obj.extract::<String>() {
                    if s == "start" || s == "end" {
                        continue;
                    }
                    return Err(PyRuntimeError::new_err(format!(
                        "Unexpected string node {s:?}; only \"start\" and \"end\" are reserved"
                    )));
                }
                let block: PyBlock = node_obj.extract()?;
                if block.sequence.is_empty() {
                    return Err(PyRuntimeError::new_err(
                        "Block has empty sequence; construct with Block(\"ACGT\", ...)",
                    ));
                }
                let seq = Sequence::new()
                    .sequence_type("DNA")
                    .sequence(&block.sequence)
                    .save(conn);
                Node::create(conn, &seq.hash, &block.node_id);
                all_blocks.push(block);
            }

            // Process edges.  Edges from/to the "start"/"end" sentinel string
            // nodes are explicit sentinel connections; all others are user edges.
            let mut user_edge_data: Vec<EdgeData> = Vec::new();
            let mut explicit_sentinel_edges: Vec<EdgeData> = Vec::new();
            let mut explicit_start_targets: HashSet<HashId> = HashSet::new();
            let mut explicit_end_sources: HashSet<HashId> = HashSet::new();
            let mut in_degrees: HashMap<HashId, usize> = HashMap::new();
            let mut out_degrees: HashMap<HashId, usize> = HashMap::new();

            let kw_data = PyDict::new(py);
            kw_data.set_item("data", true)?;
            for item in graph.getattr("edges")?.call((), Some(&kw_data))?.try_iter()? {
                let item = item?;
                let src_obj = item.get_item(0)?;
                let dst_obj = item.get_item(1)?;
                let attrs = item.get_item(2)?;

                let src_sentinel = src_obj.extract::<String>().ok();
                let dst_sentinel = dst_obj.extract::<String>().ok();

                let target_strand = attrs
                    .get_item("target_strand")
                    .ok()
                    .and_then(|s| s.extract::<String>().ok())
                    .map(|s| parse_strand(&s))
                    .unwrap_or(Strand::Forward);
                let source_strand = attrs
                    .get_item("source_strand")
                    .ok()
                    .and_then(|s| s.extract::<String>().ok())
                    .map(|s| parse_strand(&s))
                    .unwrap_or(Strand::Forward);

                match (src_sentinel.as_deref(), dst_sentinel.as_deref()) {
                    (Some("start"), _) => {
                        let dst: PyBlock = dst_obj.extract()?;
                        explicit_start_targets.insert(dst.node_id);
                        explicit_sentinel_edges.push(EdgeData {
                            source_node_id: PATH_START_NODE_ID,
                            source_coordinate: 0,
                            source_strand: Strand::Forward,
                            target_node_id: dst.node_id,
                            target_coordinate: dst.sequence_start,
                            target_strand,
                        });
                    }
                    (_, Some("end")) => {
                        let src: PyBlock = src_obj.extract()?;
                        explicit_end_sources.insert(src.node_id);
                        explicit_sentinel_edges.push(EdgeData {
                            source_node_id: src.node_id,
                            source_coordinate: src.sequence_end,
                            source_strand,
                            target_node_id: PATH_END_NODE_ID,
                            target_coordinate: 0,
                            target_strand: Strand::Forward,
                        });
                    }
                    _ => {
                        let src: PyBlock = src_obj.extract()?;
                        let dst: PyBlock = dst_obj.extract()?;
                        user_edge_data.push(EdgeData {
                            source_node_id: src.node_id,
                            source_coordinate: src.sequence_end,
                            source_strand,
                            target_node_id: dst.node_id,
                            target_coordinate: dst.sequence_start,
                            target_strand,
                        });
                        *out_degrees.entry(src.node_id).or_insert(0) += 1;
                        *in_degrees.entry(dst.node_id).or_insert(0) += 1;
                    }
                }
            }

            // Build full edge list: user edges + explicit sentinel edges + auto-
            // generated sentinel edges for any block not already connected.
            let mut all_edge_data = user_edge_data.clone();
            all_edge_data.extend(explicit_sentinel_edges);
            for block in &all_blocks {
                if !explicit_start_targets.contains(&block.node_id)
                    && in_degrees.get(&block.node_id).copied().unwrap_or(0) == 0
                {
                    all_edge_data.push(EdgeData {
                        source_node_id: PATH_START_NODE_ID,
                        source_coordinate: 0,
                        source_strand: Strand::Forward,
                        target_node_id: block.node_id,
                        target_coordinate: block.sequence_start,
                        target_strand: Strand::Forward,
                    });
                }
                if !explicit_end_sources.contains(&block.node_id)
                    && out_degrees.get(&block.node_id).copied().unwrap_or(0) == 0
                {
                    all_edge_data.push(EdgeData {
                        source_node_id: block.node_id,
                        source_coordinate: block.sequence_end,
                        source_strand: Strand::Forward,
                        target_node_id: PATH_END_NODE_ID,
                        target_coordinate: 0,
                        target_strand: Strand::Forward,
                    });
                }
            }

            let edge_ids = Edge::bulk_create(conn, &all_edge_data);
            BlockGroupEdge::bulk_create(
                conn,
                &edge_ids
                    .iter()
                    .map(|id| BlockGroupEdgeData {
                        block_group_id: block_group.id,
                        edge_id: *id,
                        chromosome_index: NO_CHROMOSOME_INDEX,
                        phased: 0,
                    })
                    .collect::<Vec<_>>(),
            );

            // Optional path creation for linear chains.
            if let Some(pname) = &path_name {
                let source_blocks: Vec<&PyBlock> = all_blocks
                    .iter()
                    .filter(|b| in_degrees.get(&b.node_id).copied().unwrap_or(0) == 0)
                    .collect();
                let sink_blocks: Vec<&PyBlock> = all_blocks
                    .iter()
                    .filter(|b| out_degrees.get(&b.node_id).copied().unwrap_or(0) == 0)
                    .collect();

                if source_blocks.len() != 1 || sink_blocks.len() != 1 {
                    return Err(PyRuntimeError::new_err(
                        "path_name requires a linear chain: exactly one source and one sink node",
                    ));
                }

                let src_id = source_blocks[0].node_id;
                let src_start = source_blocks[0].sequence_start;
                let sink_id = sink_blocks[0].node_id;
                let sink_end = sink_blocks[0].sequence_end;

                let mut path_edge_data: Vec<EdgeData> = vec![EdgeData {
                    source_node_id: PATH_START_NODE_ID,
                    source_coordinate: 0,
                    source_strand: Strand::Forward,
                    target_node_id: src_id,
                    target_coordinate: src_start,
                    target_strand: Strand::Forward,
                }];

                let mut current = src_id;
                while current != sink_id {
                    let next = user_edge_data
                        .iter()
                        .find(|e| e.source_node_id == current)
                        .ok_or_else(|| {
                            PyRuntimeError::new_err(
                                "path_name: graph is not a linear chain (dead end encountered)",
                            )
                        })?;
                    path_edge_data.push(*next);
                    current = next.target_node_id;
                }

                path_edge_data.push(EdgeData {
                    source_node_id: sink_id,
                    source_coordinate: sink_end,
                    source_strand: Strand::Forward,
                    target_node_id: PATH_END_NODE_ID,
                    target_coordinate: 0,
                    target_strand: Strand::Forward,
                });

                let saved_edges = Edge::query_by_ids(conn, &edge_ids);
                let edge_id_by_data: HashMap<EdgeData, HashId> = saved_edges
                    .iter()
                    .map(|e| (EdgeData::from(e), e.id))
                    .collect();

                let path_edge_ids: Vec<HashId> = path_edge_data
                    .iter()
                    .map(|ed| {
                        edge_id_by_data.get(ed).copied().ok_or_else(|| {
                            PyRuntimeError::new_err("Path edge not found after bulk create")
                        })
                    })
                    .collect::<PyResult<_>>()?;

                Path::create(conn, pname, &block_group.id, &path_edge_ids);
            }

            end_operation(
                ctx,
                &mut session,
                &OperationInfo {
                    files: vec![],
                    description: "python_graph_import".to_string(),
                },
                &format!("Imported block group '{name}' from Python graph"),
                None,
            )
            .map_err(|e| PyRuntimeError::new_err(format!("Operation error: {e}")))?;

            Ok(PyBlockGroup {
                id: block_group.id,
                collection_name: block_group.collection_name,
                sample_name: block_group.sample_name,
                name: block_group.name,
                db_path: ctx.graph().path().map(|p| p.to_path_buf()),
            })
        })
    }

    // -------------------------------------------------------------------------
    // Plot
    // -------------------------------------------------------------------------

    #[pyo3(signature = (block_group, rows=None, cols=None, detail=None))]
    fn plot(
        &self,
        py: Python<'_>,
        block_group: &PyBlockGroup,
        rows: Option<u32>,
        cols: Option<u32>,
        detail: Option<&str>,
    ) -> PyResult<PyObject> {
        let db_path = self
            .context
            .graph()
            .path()
            .map(|p| p.to_path_buf())
            .unwrap_or_default();
        let graph = BlockGroup::get_graph(self.context.graph().conn(), &block_group.id);
        let mut ctrl = PyGraphController::new(db_path, graph);
        if let Some(node_detail) = detail {
            ctrl.set_detail(node_detail)?;
        }
        let ctrl = Py::new(py, ctrl)?;
        build_and_display_widget(py, ctrl, rows, cols)
    }

    fn get_block_sequence(&self, node_key: &PyBlock) -> PyResult<String> {
        let sequences_by_node_id =
            Node::get_sequences_by_node_ids(self.context.graph().conn(), &[node_key.node_id]);
        let sequence = sequences_by_node_id.get(&node_key.node_id).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Node with id {:?} not found",
                node_key.node_id
            ))
        })?;
        Ok(sequence.get_sequence(node_key.sequence_start, node_key.sequence_end))
    }

    // -------------------------------------------------------------------------
    // Imports
    // -------------------------------------------------------------------------

    #[pyo3(signature = (filename, name=None, sample=None, shallow=false))]
    fn import_fasta(
        &self,
        filename: String,
        name: Option<String>,
        sample: Option<String>,
        shallow: bool,
    ) -> PyResult<String> {
        let name = name.unwrap_or_else(|| self.get_default_collection());
        let sample = sample.unwrap_or_else(|| Sample::DEFAULT_NAME.to_string());
        run_write(&self.context, !self.in_transaction, |ctx| {
            r#gen::imports::fasta::import_fasta(ctx, &filename, &name, &sample, shallow)
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
            r#gen::imports::gfa::import_gfa(ctx, &PathBuf::from(&filename), &name, &sample)
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
            r#gen::imports::genbank::import_genbank(
                ctx,
                &mut reader,
                name.as_ref(),
                &sample,
                gen_models::operations::OperationInfo {
                    files: vec![gen_models::operations::OperationFile {
                        file_path: filename.clone(),
                        file_type: gen_models::file_types::FileTypes::GenBank,
                    }],
                    description: "GenBank Import".to_string(),
                },
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
            r#gen::imports::library::import_library(
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
            r#gen::imports::library::import_library(
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

    // -------------------------------------------------------------------------
    // Exports (read-only — no transaction management)
    // -------------------------------------------------------------------------

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
            r#gen::track_database(conn, op_conn)
                .map_err(|e| PyRuntimeError::new_err(format!("Error tracking database: {e}")))?;
        }
        let name = name.unwrap_or_else(|| self.get_default_collection());
        r#gen::exports::fasta::export_fasta(
            conn,
            &name,
            sample.as_deref(),
            &PathBuf::from(&filename),
        )
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
            r#gen::track_database(conn, op_conn)
                .map_err(|e| PyRuntimeError::new_err(format!("Error tracking database: {e}")))?;
        }
        let name = name.unwrap_or_else(|| self.get_default_collection());
        let sample = sample.unwrap_or_else(|| Sample::DEFAULT_NAME.to_string());
        r#gen::exports::gfa::export_gfa(conn, &name, &PathBuf::from(&filename), &sample, node_max)
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
            r#gen::track_database(conn, op_conn)
                .map_err(|e| PyRuntimeError::new_err(format!("Error tracking database: {e}")))?;
        }
        let name = name.unwrap_or_else(|| self.get_default_collection());
        let sample = sample.unwrap_or_else(|| Sample::DEFAULT_NAME.to_string());
        r#gen::exports::genbank::export_genbank(conn, &name, &sample, &PathBuf::from(&filename))
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to export '{}': {e}", filename)))
    }

    // -------------------------------------------------------------------------
    // Updates
    // -------------------------------------------------------------------------

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
            r#gen::updates::fasta::update_with_fasta(
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
            r#gen::updates::gfa::update_with_gfa(ctx, &name, &sample, &new_sample, &filename)
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
            r#gen::updates::gaf::update_with_gaf(
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
            r#gen::updates::vcf::update_with_vcf(
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
            r#gen::updates::genbank::update_with_genbank(
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
            r#gen::updates::sequence::update_with_sequence(
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
            r#gen::updates::library::update_with_library(
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
            r#gen::updates::library::update_with_library(
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

    // -------------------------------------------------------------------------
    // Graph operations (manage their own transactions internally)
    // -------------------------------------------------------------------------

    #[pyo3(signature = (sample, new_sample, region, name=None, backbone=None, breakpoints=None, chunk_size=None))]
    #[expect(clippy::too_many_arguments, reason = "mirrors underlying API")]
    fn derive_chunks(
        &self,
        sample: String,
        new_sample: String,
        region: String,
        name: Option<String>,
        backbone: Option<String>,
        breakpoints: Option<String>,
        chunk_size: Option<i64>,
    ) -> PyResult<()> {
        r#gen::commands::graph_operations::derive_chunks::derive_chunks_operation(
            &self.context,
            name,
            sample,
            new_sample,
            region,
            backbone,
            breakpoints,
            chunk_size,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("Error deriving chunks: {e}")))
    }

    #[pyo3(signature = (sample, new_sample, region, name=None, backbone=None))]
    fn derive_subgraph(
        &self,
        sample: String,
        new_sample: String,
        region: String,
        name: Option<String>,
        backbone: Option<String>,
    ) -> PyResult<()> {
        r#gen::commands::graph_operations::derive_subgraph::derive_subgraph_operation(
            &self.context,
            name,
            sample,
            new_sample,
            region,
            backbone,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("Error deriving subgraph: {e}")))
    }

    #[pyo3(signature = (sample, new_sample, regions, new_region, name=None))]
    fn make_stitch(
        &self,
        sample: String,
        new_sample: String,
        regions: String,
        new_region: String,
        name: Option<String>,
    ) -> PyResult<()> {
        r#gen::commands::graph_operations::make_stitch::make_stitch_operation(
            &self.context,
            name,
            sample,
            new_sample,
            regions,
            new_region,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("Error making stitch: {e}")))
    }

    // -------------------------------------------------------------------------
    // Search
    // -------------------------------------------------------------------------

    /// Build a junction-aware k-mer seed index for `block_group` and save it
    /// to `.gen/search_index/{block_group_id}.bin`.
    ///
    /// If `block_groups` is None or empty, indexes all block groups.
    /// Subsequent calls to `search()` will load this index automatically.
    #[pyo3(signature = (sequence_kind="dna", k=16, bgs=None))]
    fn build_index(
        &self,
        sequence_kind: &str,
        k: usize,
        bgs: Option<Vec<PyBlockGroup>>,
    ) -> PyResult<()> {
        let kind = parse_sequence_kind(sequence_kind)?;
        let case_sensitive = kind == SequenceKind::Protein;
        let conn = self.context.graph().conn();
        let index_dir = self
            .context
            .workspace()
            .ensure_gen_dir()
            .join("search_index");
        fs::create_dir_all(&index_dir)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create index dir: {e}")))?;

        let bgs: Vec<_> = match bgs {
            Some(bgs) if !bgs.is_empty() => bgs,
            _ => BlockGroup::all(conn)
                .into_iter()
                .map(|bg| PyBlockGroup {
                    id: bg.id,
                    collection_name: bg.collection_name,
                    sample_name: bg.sample_name,
                    name: bg.name,
                    db_path: self.context.graph().path().map(|p| p.to_path_buf()),
                })
                .collect(),
        };

        for bg in bgs {
            let graph = BlockGroup::get_graph(conn, &bg.id);
            let matcher = GenGraphMatcher::new_with_sequence_kind(conn, graph, kind);
            let index = SeedIndex::build(&matcher, k);
            let path = index_dir.join(format!("{}.bin", bg.id));
            let bytes = index
                .to_bytes_with_header(case_sensitive)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to serialize index: {e}")))?;
            fs::write(&path, bytes)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to write index: {e}")))?;
        }
        Ok(())
    }

    /// Search for exact occurrences of `query`.
    ///
    /// Returns a list of tuples `(block_group, matches)` where:
    ///   - `block_group` is a `PyBlockGroup` object
    ///   - `matches` is a list of `GraphLocus` objects. Each locus exposes:
    ///       - `.start()` / `.end()` → `GraphPos` (node + byte offset) — pass
    ///         directly to `widget.go_to()`
    ///       - `.blocks` → `list[Block]`
    ///
    /// If `bgs` is None or empty, searches all block groups.
    /// If a seed index was previously built with `build_index()`, it is loaded
    /// automatically to accelerate the search. Falls back to a full scan
    /// when no index is found.
    #[pyo3(signature = (query, bgs=None, sequence_kind="dna"))]
    fn search(
        &self,
        query: &str,
        bgs: Option<Vec<PyBlockGroup>>,
        sequence_kind: &str,
    ) -> PyResult<Vec<(PyBlockGroup, Vec<PyGraphLocus>)>> {
        let kind = parse_sequence_kind(sequence_kind)?;
        let case_sensitive = kind == SequenceKind::Protein;
        let conn = self.context.graph().conn();

        let bgs: Vec<_> = match bgs {
            Some(bgs) if !bgs.is_empty() => bgs,
            _ => BlockGroup::all(conn)
                .into_iter()
                .map(|bg| PyBlockGroup {
                    id: bg.id,
                    collection_name: bg.collection_name,
                    sample_name: bg.sample_name,
                    name: bg.name,
                    db_path: self.context.graph().path().map(|p| p.to_path_buf()),
                })
                .collect(),
        };

        let query_bytes = query.as_bytes();
        let mut results = Vec::new();
        for bg in bgs {
            let graph = BlockGroup::get_graph(conn, &bg.id);
            let matcher = GenGraphMatcher::new_with_sequence_kind(conn, graph, kind);

            let index_path = self
                .context
                .workspace()
                .find_gen_dir()
                .map(|d| d.join("search_index").join(format!("{}.bin", bg.id)));
            let index = index_path.and_then(|p| fs::read(p).ok()).and_then(|bytes| {
                SeedIndex::from_bytes_with_header(&bytes, 16, case_sensitive).ok()
            });

            let matches = match index {
                Some(idx) => matcher
                    .find_all_with_seed_index(&idx, query_bytes)
                    .unwrap_or_else(|_| matcher.find_all(query_bytes)),
                None => matcher.find_all(query_bytes),
            };

            if !matches.is_empty() {
                results.push((
                    bg,
                    matches.into_iter().map(PyGraphLocus::from_locus).collect(),
                ));
            }
        }

        Ok(results)
    }

    /// Clear the search index cache.
    ///
    /// If `block_groups` is None or empty, clears all indices in `.gen/search_index/`.
    /// Otherwise, clears only the specified block group indices.
    #[pyo3(signature = (bgs=None))]
    fn clear_index(&self, bgs: Option<Vec<PyBlockGroup>>) -> PyResult<()> {
        let index_dir = self
            .context
            .workspace()
            .find_gen_dir()
            .ok_or_else(|| PyRuntimeError::new_err("No .gen directory found".to_string()))?
            .join("search_index");

        if !index_dir.exists() {
            return Ok(());
        }

        match bgs {
            Some(bgs) if !bgs.is_empty() => {
                for bg in bgs {
                    let path = index_dir.join(format!("{}.bin", bg.id));
                    if path.exists() {
                        fs::remove_file(&path).map_err(|e| {
                            PyRuntimeError::new_err(format!("Failed to delete index: {e}"))
                        })?;
                    }
                }
            }
            _ => {
                for entry in fs::read_dir(&index_dir).map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to read index dir: {e}"))
                })? {
                    let entry = entry.map_err(|e| {
                        PyRuntimeError::new_err(format!("Failed to read entry: {e}"))
                    })?;
                    let path = entry.path();
                    if path.extension().and_then(|s| s.to_str()) == Some("bin") {
                        let _ = fs::remove_file(&path);
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod python_tests {
    use std::fs;

    use r#gen::test_helpers::{setup_gen, setup_gen_on_disk};
    use pyo3::{prelude::*, py_run};
    use tempfile::tempdir;

    use crate::python_api::repository::PyRepository;

    fn make_repo(py: Python<'_>) -> Py<PyRepository> {
        let ctx = setup_gen_on_disk();
        Py::new(
            py,
            PyRepository {
                context: ctx,
                in_transaction: false,
            },
        )
        .unwrap()
    }

    fn write_fasta(
        dir: &tempfile::TempDir,
        name: &str,
        seq_name: &str,
        sequence: &str,
    ) -> std::path::PathBuf {
        let path = dir.path().join(name);
        fs::write(&path, format!(">{seq_name}\n{sequence}\n")).unwrap();
        path
    }

    #[test]
    fn test_repository_creation() {
        setup_gen();
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let repository = py.get_type::<PyRepository>();
            py_run!(
                py,
                repository,
                r#"
                repo = repository()
                assert hasattr(repo, "gen_dir")
                assert hasattr(repo, "db_path")
            "#
            );
        });
    }

    #[test]
    fn test_import_fasta_creates_block_group() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let py_repo = make_repo(py);
            let dir = tempdir().unwrap();
            let fasta = write_fasta(&dir, "test.fa", "chr1", "ACGTACGT");

            py_repo
                .borrow(py)
                .import_fasta(
                    fasta.to_str().unwrap().to_string(),
                    Some("test".to_string()),
                    None,
                    false,
                )
                .unwrap();

            let block_groups = py_repo.borrow(py).get_block_groups().unwrap();
            assert_eq!(block_groups.len(), 1);
            assert_eq!(block_groups[0].name, "chr1");
        });
    }

    #[test]
    fn test_import_fasta_duplicate_gives_specific_error() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let py_repo = make_repo(py);
            let dir = tempdir().unwrap();
            let fasta = write_fasta(&dir, "test.fa", "chr1", "ACGTACGT");
            let path = fasta.to_str().unwrap().to_string();

            py_repo
                .borrow(py)
                .import_fasta(path.clone(), Some("test".to_string()), None, false)
                .unwrap();

            let err = py_repo
                .borrow(py)
                .import_fasta(path, Some("test".to_string()), None, false)
                .unwrap_err()
                .to_string();
            assert!(
                err.contains("already exist"),
                "Expected 'already exist' in error: {err}"
            );
        });
    }

    #[test]
    fn test_transaction_commits_both_imports() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let py_repo = make_repo(py);
            let dir = tempdir().unwrap();
            let fasta1 = write_fasta(&dir, "one.fa", "chr1", "ACGTACGT");
            let fasta2 = write_fasta(&dir, "two.fa", "chr2", "TTTTGGGG");

            PyRepository::__enter__(py_repo.borrow_mut(py)).unwrap();

            {
                let borrow = py_repo.borrow(py);
                borrow
                    .import_fasta(
                        fasta1.to_str().unwrap().to_string(),
                        Some("test".to_string()),
                        None,
                        false,
                    )
                    .unwrap();
                borrow
                    .import_fasta(
                        fasta2.to_str().unwrap().to_string(),
                        Some("test".to_string()),
                        None,
                        false,
                    )
                    .unwrap();
            }

            PyRepository::__exit__(py_repo.borrow_mut(py), None, None, None).unwrap();

            let block_groups = py_repo.borrow(py).get_block_groups().unwrap();
            assert_eq!(
                block_groups.len(),
                2,
                "Both imports should have been committed"
            );
        });
    }

    #[test]
    fn test_search_finds_exact_match() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let py_repo = make_repo(py);
            let dir = tempdir().unwrap();
            let fasta = write_fasta(&dir, "test.fa", "chr1", "ACGTACGTACGT");

            py_repo
                .borrow(py)
                .import_fasta(
                    fasta.to_str().unwrap().to_string(),
                    Some("test".to_string()),
                    None,
                    false,
                )
                .unwrap();

            let hits = py_repo.borrow(py).search("ACGT", None, "dna").unwrap();
            assert!(!hits.is_empty(), "Expected at least one match for 'ACGT'");
            assert_eq!(hits.len(), 1);
            let (_, loci) = &hits[0];
            assert!(!loci.is_empty(), "Expected at least one locus for 'ACGT'");
        });
    }

    #[test]
    fn test_search_no_match() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let py_repo = make_repo(py);
            let dir = tempdir().unwrap();
            let fasta = write_fasta(&dir, "test.fa", "chr1", "ACGTACGTACGT");

            py_repo
                .borrow(py)
                .import_fasta(
                    fasta.to_str().unwrap().to_string(),
                    Some("test".to_string()),
                    None,
                    false,
                )
                .unwrap();

            let hits = py_repo.borrow(py).search("ZZZZ", None, "dna").unwrap();
            assert!(hits.is_empty(), "Expected no matches for 'ZZZZ'");
        });
    }

    #[test]
    fn test_build_index_creates_file() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let py_repo = make_repo(py);
            let dir = tempdir().unwrap();
            let fasta = write_fasta(&dir, "test.fa", "chr1", "ACGTACGTACGT");

            py_repo
                .borrow(py)
                .import_fasta(
                    fasta.to_str().unwrap().to_string(),
                    Some("test".to_string()),
                    None,
                    false,
                )
                .unwrap();

            let block_groups = py_repo.borrow(py).get_block_groups().unwrap();
            let bg = &block_groups[0];

            py_repo.borrow(py).build_index("dna", 4, None).unwrap();

            let index_dir = py_repo
                .borrow(py)
                .context
                .workspace()
                .ensure_gen_dir()
                .join("search_index");
            let index_file = index_dir.join(format!("{}.bin", bg.id));
            assert!(
                index_file.exists(),
                "Index file should exist after build_index"
            );
        });
    }

    #[test]
    fn test_search_with_index_finds_match() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let py_repo = make_repo(py);
            let dir = tempdir().unwrap();
            let fasta = write_fasta(&dir, "test.fa", "chr1", "ACGTACGTACGT");

            py_repo
                .borrow(py)
                .import_fasta(
                    fasta.to_str().unwrap().to_string(),
                    Some("test".to_string()),
                    None,
                    false,
                )
                .unwrap();

            py_repo.borrow(py).build_index("dna", 4, None).unwrap();
            let hits = py_repo.borrow(py).search("ACGT", None, "dna").unwrap();
            assert!(!hits.is_empty(), "Expected match when searching with index");
        });
    }

    #[test]
    fn test_clear_index_removes_file() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let py_repo = make_repo(py);
            let dir = tempdir().unwrap();
            let fasta = write_fasta(&dir, "test.fa", "chr1", "ACGTACGTACGT");

            py_repo
                .borrow(py)
                .import_fasta(
                    fasta.to_str().unwrap().to_string(),
                    Some("test".to_string()),
                    None,
                    false,
                )
                .unwrap();

            let block_groups = py_repo.borrow(py).get_block_groups().unwrap();
            let bg = &block_groups[0];

            py_repo.borrow(py).build_index("dna", 4, None).unwrap();
            let index_dir = py_repo
                .borrow(py)
                .context
                .workspace()
                .ensure_gen_dir()
                .join("search_index");
            let index_file = index_dir.join(format!("{}.bin", bg.id));
            assert!(index_file.exists(), "Index should exist before clear");

            py_repo.borrow(py).clear_index(None).unwrap();
            assert!(
                !index_file.exists(),
                "Index should be gone after clear_index"
            );
        });
    }

    #[test]
    fn test_blockgroup_build_and_clear_index() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let py_repo = make_repo(py);
            let dir = tempdir().unwrap();
            let fasta = write_fasta(&dir, "test.fa", "chr1", "ACGTACGTACGT");

            py_repo
                .borrow(py)
                .import_fasta(
                    fasta.to_str().unwrap().to_string(),
                    Some("test".to_string()),
                    None,
                    false,
                )
                .unwrap();

            let block_groups = py_repo.borrow(py).get_block_groups().unwrap();
            let bg = &block_groups[0];
            let bg_id = bg.id;

            let index_dir = py_repo
                .borrow(py)
                .context
                .workspace()
                .ensure_gen_dir()
                .join("search_index");
            let index_file = index_dir.join(format!("{}.bin", bg_id));

            bg.build_index("protein", 4).unwrap();
            assert!(
                index_file.exists(),
                "Index should exist after PyBlockGroup::build_index"
            );

            bg.clear_index().unwrap();
            assert!(
                !index_file.exists(),
                "Index should be gone after PyBlockGroup::clear_index"
            );
        });
    }

    #[test]
    fn test_transaction_rolls_back_on_error() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let py_repo = make_repo(py);
            let dir = tempdir().unwrap();
            let fasta = write_fasta(&dir, "test.fa", "chr1", "ACGTACGT");

            PyRepository::__enter__(py_repo.borrow_mut(py)).unwrap();

            py_repo
                .borrow(py)
                .import_fasta(
                    fasta.to_str().unwrap().to_string(),
                    Some("test".to_string()),
                    None,
                    false,
                )
                .unwrap();

            // Simulate an exception reaching __exit__ — passes a non-None exc_type
            let fake_exc = py.None().into_bound(py);
            PyRepository::__exit__(py_repo.borrow_mut(py), Some(&fake_exc), None, None).unwrap();

            let block_groups = py_repo.borrow(py).get_block_groups().unwrap();
            assert!(
                block_groups.is_empty(),
                "Import should have been rolled back, but found {} block group(s)",
                block_groups.len()
            );
        });
    }

    // -------------------------------------------------------------------------
    // Round-trip tests (to_dict / to_networkx / create_block_group_from_graph)
    // -------------------------------------------------------------------------

    #[test]
    fn test_to_dict_has_expected_fields() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let py_repo = make_repo(py);
            let bg = py_repo
                .borrow(py)
                .create_block_group_from_sequence(
                    "test".to_string(),
                    "ACGTACGT".to_string(),
                    None,
                    None,
                    None,
                    None,
                )
                .unwrap();

            let d = bg.to_dict().unwrap();
            py_run!(
                py,
                d,
                r#"
nodes = d["nodes"]
edges = d["edges"]
assert len(nodes) > 0, "expected at least one node"
node = next(iter(nodes.values()))
assert "node_id"        in node, "node missing node_id"
assert "sequence_start" in node, "node missing sequence_start"
assert "sequence_end"   in node, "node missing sequence_end"
assert "sequence"       in node, "node missing sequence"
assert len(node["sequence"]) > 0, "sequence is empty"
# node_id must be a hex string parseable back to a HashId
assert isinstance(node["node_id"], str), "node_id should be a string"
assert len(node["node_id"]) == 64, "node_id should be 64-char hex"
"#
            );
        });
    }

    #[test]
    fn test_to_dict_sequence_window_correct() {
        // sequence_start / sequence_end exported are valid indices into
        // the exported full `sequence` string.
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let py_repo = make_repo(py);
            let bg = py_repo
                .borrow(py)
                .create_block_group_from_sequence(
                    "test".to_string(),
                    "ACGTACGT".to_string(),
                    None,
                    None,
                    Some(2),
                    Some(6),
                )
                .unwrap();

            let d = bg.to_dict().unwrap();
            py_run!(
                py,
                d,
                r#"
nodes = d["nodes"]
node = next(iter(nodes.values()))
seq   = node["sequence"]
start = node["sequence_start"]
end   = node["sequence_end"]
# The full sequence is exported; start/end are valid slice indices.
assert end <= len(seq), f"sequence_end={end} > len(sequence)={len(seq)}"
window = seq[start:end]
assert window == "GTAC", f"expected window 'GTAC', got '{window}'"
"#
            );
        });
    }

    #[test]
    fn test_to_networkx_round_trip() {
        // DB ──► to_networkx() ──► create_block_group_from_graph() ──► DB
        // PyBlock nodes carry node_id, so the round-trip preserves node_ids.
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            if PyModule::import(py, "networkx").is_err() {
                eprintln!("networkx not available – skipping test_to_networkx_round_trip");
                return;
            }
            let py_repo = make_repo(py);
            let bg = py_repo
                .borrow(py)
                .create_block_group_from_sequence(
                    "original".to_string(),
                    "ACGTACGT".to_string(),
                    None,
                    None,
                    None,
                    None,
                )
                .unwrap();

            let nx_graph = bg.to_networkx(true).unwrap();

            let bg2 = py_repo
                .borrow(py)
                .create_block_group_from_graph(
                    py,
                    nx_graph.into_bound(py),
                    "round_trip".to_string(),
                    None,
                    None,
                    None,
                )
                .unwrap();

            let d1 = bg.to_dict().unwrap();
            let d2 = bg2.to_dict().unwrap();

            py_run!(
                py,
                d1 d2,
                r#"
nodes1 = {v["node_id"] for v in d1["nodes"].values()}
nodes2 = {v["node_id"] for v in d2["nodes"].values()}
assert nodes1 == nodes2, f"node_id sets differ: {nodes1} vs {nodes2}"

seqs1 = {v["sequence"] for v in d1["nodes"].values()}
seqs2 = {v["sequence"] for v in d2["nodes"].values()}
assert seqs1 == seqs2, f"sequence sets differ: {seqs1} vs {seqs2}"
"#
            );
        });
    }

    #[test]
    fn test_to_networkx_round_trip_with_modification() {
        // Build a fresh graph with a new Block("TTTTTTTT") — gets new UUID-7.
        // Re-imported block group must have different sequence than original.
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            if PyModule::import(py, "networkx").is_err() {
                eprintln!(
                    "networkx not available – skipping test_to_networkx_round_trip_with_modification"
                );
                return;
            }
            let py_repo = make_repo(py);
            let bg = py_repo
                .borrow(py)
                .create_block_group_from_sequence(
                    "original".to_string(),
                    "ACGTACGT".to_string(),
                    None,
                    None,
                    None,
                    None,
                )
                .unwrap();

            // Build a new networkx graph with a fresh Block (new UUID-7, new sequence).
            use crate::python_api::block::PyBlock;
            let networkx = PyModule::import(py, "networkx").unwrap();
            let new_graph = networkx.getattr("DiGraph").unwrap().call0().unwrap();
            let new_block = PyBlock::new("TTTTTTTT".to_string(), None, None, None);
            new_graph.call_method1("add_node", (new_block,)).unwrap();

            let bg2 = py_repo
                .borrow(py)
                .create_block_group_from_graph(
                    py,
                    new_graph,
                    "modified".to_string(),
                    None,
                    None,
                    None,
                )
                .unwrap();

            let d_orig = bg.to_dict().unwrap();
            let d_mod = bg2.to_dict().unwrap();

            py_run!(
                py,
                d_orig d_mod,
                r#"
orig_seqs = {v["sequence"] for v in d_orig["nodes"].values()}
mod_seqs  = {v["sequence"] for v in d_mod["nodes"].values()}
assert orig_seqs != mod_seqs, "sequences should differ after modification"
assert "TTTTTTTT" in mod_seqs, f"expected mutant seq 'TTTTTTTT' in {mod_seqs}"
"#
            );
        });
    }

    #[test]
    fn test_new_block_gets_fresh_node_id() {
        // Two Block("ACGTACGT") calls produce different UUID-7 node_ids.
        // Two block groups built from independent Blocks must have disjoint node_ids
        // even though sequences are identical.
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            if PyModule::import(py, "networkx").is_err() {
                eprintln!(
                    "networkx not available – skipping test_new_block_gets_fresh_node_id"
                );
                return;
            }
            use crate::python_api::block::PyBlock;
            let py_repo = make_repo(py);
            let networkx = PyModule::import(py, "networkx").unwrap();

            let make_single_node_graph = |seq: &str| {
                let g = networkx.getattr("DiGraph").unwrap().call0().unwrap();
                let b = PyBlock::new(seq.to_string(), None, None, None);
                g.call_method1("add_node", (b,)).unwrap();
                g
            };

            let g1 = make_single_node_graph("ACGTACGT");
            let g2 = make_single_node_graph("ACGTACGT");

            let bg1 = py_repo
                .borrow(py)
                .create_block_group_from_graph(py, g1, "first".to_string(), None, None, None)
                .unwrap();
            let bg2 = py_repo
                .borrow(py)
                .create_block_group_from_graph(py, g2, "second".to_string(), None, None, None)
                .unwrap();

            let d1 = bg1.to_dict().unwrap();
            let d2 = bg2.to_dict().unwrap();

            py_run!(
                py,
                d1 d2,
                r#"
ids1 = {v["node_id"] for v in d1["nodes"].values()}
ids2 = {v["node_id"] for v in d2["nodes"].values()}
print(f"ids1 ({len(ids1)}): {ids1}")
print(f"ids2 ({len(ids2)}): {ids2}")
assert ids1.isdisjoint(ids2), \
    f"Expected disjoint node_ids for independent Blocks, got overlap: {ids1 & ids2}"

seqs1 = {v["sequence"] for v in d1["nodes"].values()}
seqs2 = {v["sequence"] for v in d2["nodes"].values()}
assert seqs1 == seqs2, f"sequences should match: {seqs1} vs {seqs2}"
"#
            );
        });
    }
}
