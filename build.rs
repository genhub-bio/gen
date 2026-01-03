fn main() {
    println!("cargo:rerun-if-changed=gen-schema.capnp");
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let gen_dir = std::path::Path::new(&manifest_dir).to_path_buf();
    let output_dir = std::path::Path::new(&manifest_dir).join("src/generated");
    println!("cargo:warning=gen capnp output: {}", output_dir.display());
    capnpc::CompilerCommand::new()
        .file("./gen-schema.capnp")
        .import_path(gen_dir)
        .output_path(&output_dir)
        .run()
        .expect("Failed to compile Cap'n Proto schema");
}
