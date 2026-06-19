use r#gen::graphs::translation::{CodonTable, TranslationParams};
use gen_core::Strand;
use pyo3::{exceptions::PyRuntimeError, prelude::*};

pub(super) fn build_translation_params<'a>(
    output_collection: &'a str,
    name: Option<&'a str>,
    strand: Option<&str>,
    frame: u8,
    codon_table: u8,
) -> PyResult<TranslationParams<'a>> {
    let resolved_strand = match strand {
        None => None,
        Some("forward") => Some(Strand::Forward),
        Some("reverse") => Some(Strand::Reverse),
        Some(s) => {
            return Err(PyRuntimeError::new_err(format!(
                "Unknown strand '{s}'; use 'forward' or 'reverse'"
            )));
        }
    };
    let table = CodonTable::ncbi(codon_table).ok_or_else(|| {
        PyRuntimeError::new_err(format!("Unknown NCBI codon table id {codon_table}"))
    })?;

    let mut params = TranslationParams::new(output_collection)
        .initial_frame(frame)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        .codon_table(table);
    if let Some(s) = resolved_strand {
        params = params.strand(s);
    }
    if let Some(name) = name {
        params = params.name(name);
    }
    Ok(params)
}
