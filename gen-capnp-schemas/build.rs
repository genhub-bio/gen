fn main() {
    println!("cargo:rerun-if-changed=gen-core.capnp");
    println!("cargo:rerun-if-changed=gen-models.capnp");
    println!("cargo:rerun-if-changed=gen-schema.capnp");

    let manifest_dir = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let output_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());

    capnpc::CompilerCommand::new()
        .file("./gen-core.capnp")
        .file("./gen-models.capnp")
        .file("./gen-schema.capnp")
        .import_path(manifest_dir)
        .output_path(&output_dir)
        .run()
        .expect("Failed to compile Cap'n Proto schemas");
}
