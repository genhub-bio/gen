use gen_models::{
    operations::{OperationInfo, OperationSummary},
    sample::Sample,
    sample_metadata::{MetadataValue, SampleMetadata},
};
use pyo3::{
    exceptions::{PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyBool, PyDict, PyFloat, PyInt, PyString},
};

use super::PySample;
use crate::python_api::repository::run_context_operation_write;

fn metadata_value(value: &Bound<'_, PyAny>) -> PyResult<MetadataValue> {
    if value.is_instance_of::<PyString>() {
        Ok(MetadataValue::Text(value.extract()?))
    } else if value.is_instance_of::<PyBool>() {
        Ok(MetadataValue::Boolean(value.extract()?))
    } else if value.is_instance_of::<PyInt>() {
        Ok(MetadataValue::Integer(value.extract()?))
    } else if value.is_instance_of::<PyFloat>() {
        let float: f64 = value.extract()?;
        if !float.is_finite() {
            return Err(PyValueError::new_err(
                "sample metadata floats must be finite",
            ));
        }
        Ok(MetadataValue::Float(float))
    } else {
        Err(PyTypeError::new_err(
            "sample metadata values must be strings, signed 64-bit integers, finite floats, or booleans",
        ))
    }
}

impl PySample {
    fn sample_metadata(&self, sample_name: &str) -> PyResult<Vec<SampleMetadata>> {
        let connection = self.context.graph().conn();
        if Sample::select(connection)
            .name(sample_name)
            .load()
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
            .is_empty()
        {
            return Err(PyValueError::new_err(format!(
                "Sample not found: {sample_name}"
            )));
        }
        SampleMetadata::select(connection)
            .sample_name(sample_name)
            .load()
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }
}

#[pymethods]
impl PySample {
    /// Add metadata to this sample.
    ///
    /// Keys must be strings; values must be strings, signed 64-bit integers,
    /// finite floats, or booleans. Existing keys are updated and other keys are preserved.
    /// The entire dictionary is validated before any values are written.
    fn add_metadata(&self, metadata: &Bound<'_, PyDict>) -> PyResult<()> {
        let sample_name = &self.sample_name;
        let existing = self.sample_metadata(sample_name)?;
        let values = metadata
            .iter()
            .map(|(key, value)| Ok((key.extract::<String>()?, metadata_value(&value)?)))
            .collect::<PyResult<Vec<_>>>()?;
        if values.iter().all(|(key, value)| {
            existing
                .iter()
                .any(|entry| entry.key == *key && entry.value == *value)
        }) {
            return Ok(());
        }

        // Use the same transaction and history workflow as sequence updates so
        // metadata follows the sample through branch checkout and reset.
        run_context_operation_write(
            &self.context,
            |context| {
                for (key, value) in &values {
                    SampleMetadata::upsert(context.graph().conn(), sample_name, key, value)
                        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
                }
                Ok((
                    (),
                    OperationSummary::new(
                        OperationInfo {
                            files: vec![],
                            description: "add sample metadata".to_string(),
                        },
                        format!("Updated metadata for sample '{sample_name}'"),
                    ),
                ))
            },
            |error| PyRuntimeError::new_err(error.to_string()),
        )
    }

    /// Return a fresh metadata dictionary for this sample.
    ///
    /// Values retain their string, integer, float, or boolean types. A sample without
    /// metadata returns an empty dictionary; an unknown sample raises ValueError.
    #[getter]
    fn metadata<'py>(&self, python: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let metadata = PyDict::new(python);
        for entry in self.sample_metadata(&self.sample_name)? {
            match entry.value {
                MetadataValue::Text(value) => metadata.set_item(entry.key, value)?,
                MetadataValue::Integer(value) => metadata.set_item(entry.key, value)?,
                MetadataValue::Float(value) => metadata.set_item(entry.key, value)?,
                MetadataValue::Boolean(value) => metadata.set_item(entry.key, value)?,
            }
        }
        Ok(metadata)
    }
}

#[cfg(test)]
mod tests {
    use r#gen::test_helpers::setup_gen_on_disk;
    use gen_models::sample::{NewSample, Sample};
    use pyo3::{prelude::*, py_run};

    use crate::python_api::{repository::PyRepository, sample::PySample};

    #[test]
    fn test_sample_metadata_python_round_trip() {
        Python::initialize();
        Python::attach(|python| {
            let context = setup_gen_on_disk();
            for name in ["sample", "other"] {
                Sample::create(
                    context.graph().conn(),
                    NewSample {
                        name,
                        ..Default::default()
                    },
                )
                .expect("should create sample");
            }
            let repository = Py::new(
                python,
                PyRepository {
                    context: context.clone(),
                },
            )
            .expect("should create Python repository");
            let sample = Py::new(
                python,
                PySample::new("default".to_string(), "sample".to_string(), context.clone()),
            )
            .expect("should create Python sample");
            let other = Py::new(
                python,
                PySample::new("default".to_string(), "other".to_string(), context.clone()),
            )
            .expect("should create other Python sample");
            let missing = Py::new(
                python,
                PySample::new("default".to_string(), "missing".to_string(), context),
            )
            .expect("should create missing Python sample");
            py_run!(python, repository sample other missing, r#"
assert sample.metadata == {}
values = {"label": "case", "count": 9223372036854775807, "score": 1.25, "enabled": True, "disabled": False}
sample.add_metadata(values)
assert sample.metadata == values
assert other.metadata == {}
result = sample.metadata
assert type(result["count"]) is int
assert type(result["score"]) is float
assert type(result["enabled"]) is bool
assert type(result["disabled"]) is bool
result["label"] = "local change"
assert sample.metadata == values
sample.add_metadata({"count": "updated", "minimum": -9223372036854775808})
values.update(count="updated", minimum=-9223372036854775808)
assert sample.metadata == values
sample.add_metadata(values)
sample.add_metadata({})
assert sample.metadata == values
repository.checkout("metadata-branch", create=True)
sample.add_metadata({"label": "branch"})
assert sample.metadata["label"] == "branch"
repository.checkout("main")
assert sample.metadata == values
for invalid in [None, [], {}, float("nan"), float("inf"), float("-inf"), 2**63, -2**63 - 1]:
    try:
        sample.add_metadata({"first": "must not persist", "invalid": invalid})
    except (TypeError, ValueError, OverflowError):
        pass
    else:
        raise AssertionError("invalid metadata should be rejected")
    assert sample.metadata == values
for invalid in [{1: "invalid key"}, [("key", "value")]]:
    try:
        sample.add_metadata(invalid)
    except TypeError:
        pass
    else:
        raise AssertionError("invalid metadata dictionary should be rejected")
assert not hasattr(repository, "get_sample_metadata")
assert not hasattr(repository, "add_sample_metadata")
for method in [lambda missing=missing: missing.metadata, lambda missing=missing: missing.add_metadata({})]:
    try:
        method()
    except ValueError:
        pass
    else:
        raise AssertionError("missing sample should be rejected")
sample.add_metadata({"after_errors": 2.0})
assert sample.metadata["after_errors"] == 2.0
"#);
        });
    }
}
