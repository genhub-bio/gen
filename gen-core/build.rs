fn main() {
    println!("cargo:rerun-if-changed=gen-core.capnp");
    let output_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
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
