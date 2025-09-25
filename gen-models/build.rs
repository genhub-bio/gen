fn main() {
    println!("cargo:rerun-if-changed=gen-models.capnp");
    let gen_dir = std::env::current_dir()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    capnpc::CompilerCommand::new()
        .file("./gen-models.capnp")
        .import_path(gen_dir)
        .output_path("src/generated")
        .run()
        .expect("Failed to compile Cap'n Proto schema");
}
