//! A small, target-gated blocking HTTP transport shared by native and `target_os = "emscripten"`
//! builds.
//!
//! Native builds keep using `reqwest::blocking` (see [`native_http`]). The `emscripten` build has
//! no sockets and no `reqwest`; it uses the browser's own networking stack through Emscripten's
//! official synchronous Fetch API (`emscripten/fetch.h`) instead (see [`browser_http`]). All
//! `unsafe` code for that binding lives in [`browser_http`] and [`emscripten_bindings`]; callers
//! of [`request`] see only safe, fully Rust-owned types.
//!
//! Both the request body and the response body are fully buffered in memory (matching
//! `EMSCRIPTEN_FETCH_LOAD_TO_MEMORY` on the browser side); there is no streaming request or
//! response path. That's fine for the small JSON API calls this boundary was built for, and
//! acceptable so far for the largest thing routed through it today (asset upload/download in
//! `remote::operations`), but revisit if a real asset transfer exposes a memory problem.

pub mod browser_http;
#[cfg(target_os = "emscripten")]
pub mod emscripten_bindings;
#[cfg(not(target_os = "emscripten"))]
pub mod native_http;

#[cfg(target_os = "emscripten")]
pub use browser_http::request;
#[cfg(not(target_os = "emscripten"))]
pub use native_http::request;

/// A blocking HTTP request. Borrows all of its data; callers keep ownership.
pub struct HttpRequest<'a> {
    pub method: &'a str,
    pub url: &'a str,
    pub headers: &'a [(&'a str, &'a str)],
    pub body: Option<&'a [u8]>,
}

/// A completed HTTP response. Always fully owned; never borrows from the transport.
#[derive(Debug)]
pub struct HttpResponse {
    pub status: u16,
    pub body: Vec<u8>,
}

#[derive(Debug, thiserror::Error)]
pub enum BrowserHttpError {
    #[error("{field} contains an embedded null byte")]
    EmbeddedNullByte { field: &'static str },
    #[error("HTTP method {0:?} is not valid for this transport")]
    InvalidMethod(String),
    #[error("browser fetch failed to start (invalid URL or attributes)")]
    FetchStartFailed,
    #[error("response body is too large to address on this platform")]
    ResponseTooLarge,
    #[error("network error: {0}")]
    Network(String),
    #[cfg(not(target_os = "emscripten"))]
    #[error(transparent)]
    Native(#[from] reqwest::Error),
}
