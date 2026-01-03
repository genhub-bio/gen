fn main() {
    println!("cargo:rerun-if-changed=gen-core.capnp");
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let output_dir = std::path::Path::new(&manifest_dir).join("src/generated");
    println!(
        "cargo:warning=gen-core capnp output: {}",
        output_dir.display()
    );
    capnpc::CompilerCommand::new()
        .file("./gen-core.capnp")
        .output_path(&output_dir)
        .run()
        .expect("Failed to compile Cap'n Proto schema");
}
