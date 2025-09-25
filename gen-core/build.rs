fn main() {
    println!("cargo:rerun-if-changed=gen-core.capnp");
    capnpc::CompilerCommand::new()
        .file("./gen-core.capnp")
        .output_path("src/generated")
        .run()
        .expect("Failed to compile Cap'n Proto schema");
}
