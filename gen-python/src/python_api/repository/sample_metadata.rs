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

use super::{PyRepository, run_operation_write};
use crate::python_api::sample::PySample;

fn sample_name(sample: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(sample) = sample.extract::<PyRef<'_, PySample>>() {
        Ok(sample.sample_name.clone())
    } else {
        sample.extract::<String>()
    }
}

fn metadata_value(value: &Bound<'_, PyAny>) -> PyResult<MetadataValue> {
    if value.is_instance_of::<PyString>() {
        Ok(MetadataValue::Text(value.extract()?))
    } else if value.is_instance_of::<PyInt>() && !value.is_instance_of::<PyBool>() {
        Ok(MetadataValue::Integer(value.extract()?))
    } else if value.is_instance_of::<PyFloat>() {
        let real: f64 = value.extract()?;
        if !real.is_finite() {
            return Err(PyValueError::new_err(
                "sample metadata floats must be finite",
            ));
        }
        Ok(MetadataValue::Real(real))
    } else {
        Err(PyTypeError::new_err(
            "sample metadata values must be strings, signed 64-bit integers, or finite floats",
        ))
    }
}

impl PyRepository {
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
impl PyRepository {
    /// Add metadata to a Sample or its string ID (sample_name).
    ///
    /// Keys must be strings; values must be strings, signed 64-bit integers,
    /// or finite floats. Existing keys are updated and other keys are preserved.
    /// The entire dictionary is validated before any values are written.
    fn add_sample_metadata(
        &self,
        sample: &Bound<'_, PyAny>,
        metadata: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        let sample_name = sample_name(sample)?;
        let existing = self.sample_metadata(&sample_name)?;
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
        run_operation_write(
            self,
            |context| {
                for (key, value) in &values {
                    SampleMetadata::upsert(context.graph().conn(), &sample_name, key, value)
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

    /// Return a metadata dictionary for a Sample or its string ID (sample_name).
    ///
    /// Values retain their string, integer, or float types. A sample without
    /// metadata returns an empty dictionary; an unknown sample raises ValueError.
    fn get_sample_metadata<'py>(&self, sample: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
        let metadata = PyDict::new(sample.py());
        for entry in self.sample_metadata(&sample_name(sample)?)? {
            match entry.value {
                MetadataValue::Text(value) => metadata.set_item(entry.key, value)?,
                MetadataValue::Integer(value) => metadata.set_item(entry.key, value)?,
                MetadataValue::Real(value) => metadata.set_item(entry.key, value)?,
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
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|python| {
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
            let repository =
                Py::new(python, PyRepository { context }).expect("should create Python repository");
            let sample = Py::new(
                python,
                PySample::new("default".to_string(), "sample".to_string(), vec![]),
            )
            .expect("should create Python sample");
            py_run!(python, repository sample, r#"
assert repository.get_sample_metadata(sample) == {}
values = {"label": "case", "count": 9223372036854775807, "score": 1.25}
repository.add_sample_metadata(sample, values)
assert repository.get_sample_metadata("sample") == values
assert repository.get_sample_metadata("other") == {}
result = repository.get_sample_metadata(sample)
assert type(result["count"]) is int
assert type(result["score"]) is float
result["label"] = "local change"
assert repository.get_sample_metadata(sample) == values
repository.add_sample_metadata("sample", {"count": "updated", "minimum": -9223372036854775808})
values.update(count="updated", minimum=-9223372036854775808)
assert repository.get_sample_metadata(sample) == values
repository.add_sample_metadata(sample, values)
repository.add_sample_metadata(sample, {})
assert repository.get_sample_metadata(sample) == values
repository.checkout("metadata-branch", create=True)
repository.add_sample_metadata(sample, {"label": "branch"})
assert repository.get_sample_metadata(sample)["label"] == "branch"
repository.checkout("main")
assert repository.get_sample_metadata(sample) == values
for invalid in [None, True, [], {}, float("nan"), float("inf"), float("-inf"), 2**63, -2**63 - 1]:
    try:
        repository.add_sample_metadata(sample, {"first": "must not persist", "invalid": invalid})
    except (TypeError, ValueError, OverflowError):
        pass
    else:
        raise AssertionError("invalid metadata should be rejected")
    assert repository.get_sample_metadata(sample) == values
for invalid in [{1: "invalid key"}, [("key", "value")]]:
    try:
        repository.add_sample_metadata(sample, invalid)
    except TypeError:
        pass
    else:
        raise AssertionError("invalid metadata dictionary should be rejected")
for method in [repository.get_sample_metadata, lambda sample, repository=repository: repository.add_sample_metadata(sample, {})]:
    for invalid, error_type in [(42, TypeError), ("missing", ValueError)]:
        try:
            method(invalid)
        except error_type:
            pass
        else:
            raise AssertionError("invalid sample should be rejected")
repository.add_sample_metadata(sample, {"after_errors": 2.0})
assert repository.get_sample_metadata(sample)["after_errors"] == 2.0
"#);
        });
    }
}
