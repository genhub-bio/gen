//! A small, target-gated blocking HTTP transport shared by native and `target_os = "emscripten"`
//! builds.
//!
//! Native builds use `reqwest::blocking`; Emscripten builds use the shared
//! C/XHR shim, including when linked into the Python extension. Calls must run
//! in a browser worker. Requests and responses are fully buffered binary data.

pub mod browser_http;
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
    #[error("browser request failed to start (invalid URL or attributes)")]
    FetchStartFailed,
    #[error("response body is too large to address on this platform")]
    ResponseTooLarge,
    #[error("browser HTTP buffer allocation failed")]
    AllocationFailed,
    #[error("synchronous browser HTTP requires execution in a browser worker")]
    WorkerRequired,
    #[error("network error: {0}")]
    Network(String),
    #[cfg(not(target_os = "emscripten"))]
    #[error(transparent)]
    Native(#[from] reqwest::Error),
}
