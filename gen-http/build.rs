use std::env;

fn main() {
    println!("cargo:rerun-if-changed=src/browser_http.c");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("emscripten") {
        cc::Build::new()
            .file("src/browser_http.c")
            .warnings(true)
            .compile("gen_http_browser");
    }
}
