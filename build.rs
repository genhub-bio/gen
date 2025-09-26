fn main() {
    println!("cargo:rerun-if-changed=gen-schema.capnp");
    let gen_dir = std::env::current_dir().unwrap().to_path_buf();
    capnpc::CompilerCommand::new()
        .file("./gen-schema.capnp")
        .import_path(gen_dir)
        .output_path("src/generated")
        .run()
        .expect("Failed to compile Cap'n Proto schema");
}
