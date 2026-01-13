use pyo3::{prelude::*, py_run};

use crate::python_api::repository::PyRepository;

#[test]
fn test_block_group_creation_with_sequence() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let repository = py.get_type::<PyRepository>();
        py_run!(
            py,
            repository,
            r#"
import tempfile
import os

with tempfile.TemporaryDirectory() as temp_dir:
    repo = repository(temp_dir)
    
    bg = repo.create_block_group(
        name="my_sequence",
        collection_name="test_collection",
        sequence="ATCGATCGATCGATCGATCGGGAACACACAGAGA"
    )
    
    assert bg.name == "my_sequence"
    assert bg.collection_name == "test_collection"
    
    # Retrieve all block groups to verify
    all_block_groups = repo.get_block_groups()
    assert len(all_block_groups) == 1
    assert all_block_groups[0].name == "my_sequence"
    
    sequences = repo.query("SELECT sequence FROM sequences WHERE sequence_type = 'DNA'")
    assert len(sequences) == 1
    assert sequences[0][0] == "ATCGATCGATCGATCGATCGGGAACACACAGAGA"
        "#
        );
    });
}

#[test]
fn test_multiple_block_groups() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let repository = py.get_type::<PyRepository>();
        py_run!(
            py,
            repository,
            r#"
import tempfile

with tempfile.TemporaryDirectory() as temp_dir:
    repo = repository(temp_dir)
    
    bg1 = repo.create_block_group(
        name="sequence_1",
        collection_name="collection_a",
        sequence="AAAAAAAAAA"
    )
    
    bg2 = repo.create_block_group(
        name="sequence_2",
        collection_name="collection_b",
        sequence="TTTTTTTTTT"
    )
    
    all_bgs = repo.get_block_groups()
    assert len(all_bgs) == 2
    
    names = {bg.name for bg in all_bgs}
    assert "sequence_1" in names
    assert "sequence_2" in names
        "#
        );
    });
}

