use std::path::Path;

use pyo3_stub_gen::{Result, StubInfo};

pub mod python_api;

/// Gathers the annotated bindings for `src/bin/stub_gen.rs`. The maturin configuration lives in
/// the repository-root `pyproject.toml`, one directory above this crate's manifest.
pub fn stub_info() -> Result<StubInfo> {
    let manifest_dir: &Path = env!("CARGO_MANIFEST_DIR").as_ref();
    StubInfo::from_pyproject_toml(manifest_dir.join("../pyproject.toml"))
}
