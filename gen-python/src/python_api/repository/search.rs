use std::fs;

use r#gen::graphs::graph_search::{GenGraphMatcher, SeedIndex, SequenceKind};
use gen_models::{block_group::BlockGroup, traits::Query};
use pyo3::{exceptions::PyRuntimeError, prelude::*};

use super::PyRepository;
use crate::python_api::{
    block_group::{PySequenceGraph, parse_sequence_kind},
    graph_search::PyGraphLocus,
};

#[pymethods]
impl PyRepository {
    /// Build a junction-aware k-mer seed index for `block_group` and save it
    /// to `.gen/search_index/{block_group_id}.bin`.
    ///
    /// If `block_groups` is None or empty, indexes all sequence graphs.
    /// Subsequent calls to `search()` will load this index automatically.
    #[pyo3(signature = (sequence_kind="dna", k=16, bgs=None))]
    pub fn build_index(
        &self,
        sequence_kind: &str,
        k: usize,
        bgs: Option<Vec<PySequenceGraph>>,
    ) -> PyResult<()> {
        let kind = parse_sequence_kind(sequence_kind)?;
        let normalized = kind != SequenceKind::Exact;
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
                .map(|bg| self.into_py_block_group(bg))
                .collect(),
        };

        for bg in bgs {
            let graph = BlockGroup::get_graph(conn, &bg.id);
            let matcher = GenGraphMatcher::new_with_sequence_kind(conn, graph, kind);
            let index = SeedIndex::build(&matcher, k, normalized);
            let path = index_dir.join(format!("{}.bin", bg.id));
            let bytes = index
                .to_bytes_with_header()
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to serialize index: {e}")))?;
            fs::write(&path, bytes)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to write index: {e}")))?;
        }
        Ok(())
    }

    /// Search for exact occurrences of `query`.
    ///
    /// Returns a list of tuples `(block_group, matches)` where:
    ///   - `block_group` is a `PySequenceGraph` object
    ///   - `matches` is a list of `GraphLocus` objects. Each locus exposes:
    ///       - `.start()` / `.end()` → `GraphPos` (node + byte offset) — pass
    ///         directly to `widget.go_to()`
    ///       - `.slices` → `list[NodeSlice]`
    ///
    /// If `bgs` is None or empty, searches all sequence graphs.
    /// If a seed index was previously built with `build_index()`, it is loaded
    /// automatically to accelerate the search. Falls back to a full scan
    /// when no index is found.
    #[pyo3(signature = (query, bgs=None, sequence_kind="dna"))]
    pub fn search(
        &self,
        query: &str,
        bgs: Option<Vec<PySequenceGraph>>,
        sequence_kind: &str,
    ) -> PyResult<Vec<(PySequenceGraph, Vec<PyGraphLocus>)>> {
        let kind = parse_sequence_kind(sequence_kind)?;
        let conn = self.context.graph().conn();

        let bgs: Vec<_> = match bgs {
            Some(bgs) if !bgs.is_empty() => bgs,
            _ => BlockGroup::all(conn)
                .into_iter()
                .map(|bg| self.into_py_block_group(bg))
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
            let index = index_path
                .and_then(|p| fs::read(p).ok())
                .and_then(|bytes| SeedIndex::from_bytes_with_header(&bytes, 16).ok());

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
    /// Otherwise, clears only the specified sequence graph indices.
    #[pyo3(signature = (bgs=None))]
    pub fn clear_index(&self, bgs: Option<Vec<PySequenceGraph>>) -> PyResult<()> {
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
