use pyo3_stub_gen::Result;

// Regenerates gen-python/python/gen/gen.pyi from the annotated Rust bindings.
fn main() -> Result<()> {
    let stub = gen_python::stub_info()?;
    stub.generate()?;
    Ok(())
}
