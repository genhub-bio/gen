use std::env;

fn main() {
    let source = "src/commands/remote/http/browser_http.c";
    println!("cargo:rerun-if-changed={source}");
    println!("cargo:rerun-if-changed=src/commands/remote/http/browser_http.h");
    if env::var("CARGO_CFG_TARGET_OS").is_ok_and(|target| target == "emscripten") {
        cc::Build::new().file(source).compile("gen_browser_http");
    }
}
