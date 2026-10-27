//! Small synchronous HTTP transport shared by GenHub and remote asset clients.
//!
//! Native requests use reqwest's streaming response and request bodies. On Emscripten,
//! requests use synchronous `XMLHttpRequest`, which is available to the Pyodide worker and
//! carries ordinary browser CORS behavior into Rust callers.

#[cfg(target_os = "emscripten")]
use std::io::Cursor;
use std::{fmt, io, io::Read, time::Duration};

pub use http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode, header};
use thiserror::Error;

pub struct Request {
    pub method: Method,
    pub url: String,
    pub headers: HeaderMap,
    body: Option<RequestBody>,
    timeout: Option<Duration>,
    max_response_bytes: Option<usize>,
}

enum RequestBody {
    Bytes(Vec<u8>),
    Reader(Box<dyn Read + Send>),
}

impl Request {
    /// Creates a request with no body or custom headers.
    pub fn new(method: Method, url: impl Into<String>) -> Self {
        Self {
            method,
            url: url.into(),
            headers: HeaderMap::new(),
            body: None,
            timeout: None,
            max_response_bytes: None,
        }
    }

    /// Adds one header, retaining any earlier values with the same name.
    pub fn header(mut self, name: HeaderName, value: HeaderValue) -> Self {
        self.headers.append(name, value);
        self
    }

    /// Adds a byte body.
    pub fn body_bytes(mut self, body: impl Into<Vec<u8>>) -> Self {
        self.body = Some(RequestBody::Bytes(body.into()));
        self
    }

    /// Adds a streaming body. Native requests stream from this reader; browser requests buffer it
    /// because synchronous XHR accepts a complete body before the request begins.
    pub fn body_reader(mut self, body: impl Read + Send + 'static) -> Self {
        self.body = Some(RequestBody::Reader(Box::new(body)));
        self
    }

    /// Applies a transport timeout where the platform HTTP implementation supports it.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Rejects a browser response after XHR has received more than this many bytes.
    ///
    /// Browser XHR buffers a response before returning to Rust, so this bounds what callers can
    /// consume but cannot prevent the browser from allocating the response. Use it with HTTP Range
    /// requests when downloading large objects.
    pub fn max_response_bytes(mut self, maximum: usize) -> Self {
        self.max_response_bytes = Some(maximum);
        self
    }
}

impl fmt::Debug for Request {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Request")
            .field("method", &self.method)
            .field("url", &"[REDACTED]")
            .field("header_count", &self.headers.len())
            .field("has_body", &self.body.is_some())
            .field("timeout", &self.timeout)
            .field("max_response_bytes", &self.max_response_bytes)
            .finish()
    }
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("invalid HTTP request URL: {0}")]
    InvalidUrl(String),
    #[error("HTTP request header value is not valid text: {0}")]
    InvalidRequestHeader(String),
    #[error("HTTP response contains an invalid header: {0}")]
    InvalidResponseHeader(String),
    #[error("HTTP request failed: {0}")]
    RequestBody(#[from] io::Error),
    #[error("HTTP transport failed: {0}")]
    Transport(String),
    #[error(
        "browser response contained {actual} bytes, exceeding the configured {maximum}-byte limit"
    )]
    ResponseTooLarge { actual: usize, maximum: usize },
    #[cfg(not(target_os = "emscripten"))]
    #[error("native HTTP request failed: {0}")]
    NativeRequest(#[from] reqwest::Error),
}

/// Reusable HTTP client with platform-selected native or browser-worker transport.
#[derive(Clone)]
pub struct Client {
    #[cfg(not(target_os = "emscripten"))]
    native: reqwest::blocking::Client,
}

impl Client {
    /// Creates a client using the platform's default connection and redirect behavior.
    pub fn new() -> Self {
        Self {
            #[cfg(not(target_os = "emscripten"))]
            native: reqwest::blocking::Client::new(),
        }
    }

    /// Creates a client that does not follow redirects on native targets.
    ///
    /// Browser XHR follows redirects according to browser policy and cannot disable that behavior.
    pub fn without_redirects() -> Result<Self, Error> {
        #[cfg(not(target_os = "emscripten"))]
        let native = reqwest::blocking::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()?;
        Ok(Self {
            #[cfg(not(target_os = "emscripten"))]
            native,
        })
    }

    /// Sends an HTTP request and returns the response body as a streaming reader on native targets.
    pub fn execute(&self, request: Request) -> Result<Response, Error> {
        #[cfg(target_os = "emscripten")]
        {
            execute_browser(request)
        }
        #[cfg(not(target_os = "emscripten"))]
        {
            self.execute_native(request)
        }
    }

    #[cfg(not(target_os = "emscripten"))]
    fn execute_native(&self, mut request: Request) -> Result<Response, Error> {
        let mut builder = self
            .native
            .request(request.method, &request.url)
            .headers(request.headers);
        if let Some(timeout) = request.timeout {
            builder = builder.timeout(timeout);
        }
        if let Some(body) = request.body.take() {
            let body = match body {
                RequestBody::Bytes(body) => reqwest::blocking::Body::from(body),
                RequestBody::Reader(body) => reqwest::blocking::Body::new(body),
            };
            builder = builder.body(body);
        }
        let response = builder
            .send()
            .map_err(|error| Error::NativeRequest(error.without_url()))?;
        let status = StatusCode::from_u16(response.status().as_u16())
            .map_err(|error| Error::Transport(error.to_string()))?;
        let headers = response.headers().clone();
        let final_url = response.url().to_string();
        Ok(Response {
            status,
            headers,
            final_url,
            body: Box::new(NativeResponseReader(response)),
        })
    }
}

#[cfg(not(target_os = "emscripten"))]
struct NativeResponseReader(reqwest::blocking::Response);

#[cfg(not(target_os = "emscripten"))]
impl Read for NativeResponseReader {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        self.0
            .read(buffer)
            .map_err(|error| std::io::Error::new(error.kind(), "HTTP response body read failed"))
    }
}

impl Default for Client {
    fn default() -> Self {
        Self::new()
    }
}

/// HTTP response metadata plus a reader for its body.
pub struct Response {
    status: StatusCode,
    headers: HeaderMap,
    final_url: String,
    body: Box<dyn Read + Send>,
}

impl Response {
    pub fn status(&self) -> StatusCode {
        self.status
    }

    pub fn headers(&self) -> &HeaderMap {
        &self.headers
    }

    /// URL after redirects, as reported by the underlying HTTP implementation.
    pub fn final_url(&self) -> &str {
        &self.final_url
    }

    pub fn body_mut(&mut self) -> &mut (dyn Read + Send) {
        &mut *self.body
    }

    pub fn into_body(self) -> Box<dyn Read + Send> {
        self.body
    }
}

impl fmt::Debug for Response {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Response")
            .field("status", &self.status)
            .field("header_count", &self.headers.len())
            .field("final_url", &"[REDACTED]")
            .finish_non_exhaustive()
    }
}

#[cfg(target_os = "emscripten")]
fn execute_browser(mut request: Request) -> Result<Response, Error> {
    use std::{
        ffi::{CStr, CString},
        ptr,
    };

    let method = CString::new(request.method.as_str())
        .map_err(|error| Error::Transport(error.to_string()))?;
    let url =
        CString::new(request.url.as_str()).map_err(|error| Error::InvalidUrl(error.to_string()))?;
    let headers = request
        .headers
        .iter()
        .map(|(name, value)| {
            value
                .to_str()
                .map(|value| (name.as_str(), value))
                .map_err(|_| Error::InvalidRequestHeader(name.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let headers_json =
        serde_json::to_string(&headers).map_err(|error| Error::Transport(error.to_string()))?;
    let headers_json =
        CString::new(headers_json).map_err(|error| Error::Transport(error.to_string()))?;
    let request_body = match request.body.take() {
        Some(RequestBody::Bytes(body)) => body,
        Some(RequestBody::Reader(mut body)) => {
            let mut bytes = Vec::new();
            body.read_to_end(&mut bytes)?;
            bytes
        }
        None => Vec::new(),
    };

    let mut status = 0_u32;
    let mut response_length = 0_usize;
    let mut response_headers_length = 0_usize;
    let mut response_url_length = 0_usize;
    let timeout_millis = request.timeout.map_or(0, |timeout| {
        u32::try_from(timeout.as_millis()).unwrap_or(u32::MAX)
    });
    let mut error_message = [0_i8; 1024];
    let body_pointer = if request_body.is_empty() {
        ptr::null()
    } else {
        request_body.as_ptr()
    };
    // SAFETY: All request pointers reference live, NUL-terminated strings or byte slices for the
    // duration of the synchronous XHR call. Output pointers refer to initialized local storage.
    // The JavaScript response slots are safe for this call because the worker executes the
    // synchronous request serially and the response guard clears them before returning to Python.
    let result = unsafe {
        gen_http_xhr_request(
            method.as_ptr(),
            url.as_ptr(),
            headers_json.as_ptr(),
            body_pointer,
            request_body.len(),
            &mut status,
            &mut response_length,
            &mut response_headers_length,
            &mut response_url_length,
            timeout_millis,
            error_message.as_mut_ptr(),
            error_message.len(),
        )
    };
    let _response_guard = BrowserResponseGuard;
    if result != 0 {
        // SAFETY: The C shim writes a terminating NUL into the fixed-size error buffer.
        let message = unsafe { CStr::from_ptr(error_message.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        return Err(Error::Transport(message));
    }
    if request
        .max_response_bytes
        .is_some_and(|maximum| response_length > maximum)
    {
        return Err(Error::ResponseTooLarge {
            actual: response_length,
            maximum: request
                .max_response_bytes
                .expect("should have a response limit"),
        });
    }

    let mut response_body = vec![0_u8; response_length];
    if response_length > 0 {
        // SAFETY: The C shim copies exactly the previously reported number of response bytes.
        unsafe { gen_http_xhr_copy_body(response_body.as_mut_ptr()) };
    }
    let mut response_headers = vec![0_i8; response_headers_length + 1];
    // SAFETY: The destination is sized for the reported UTF-8 header bytes and terminating NUL.
    unsafe {
        gen_http_xhr_copy_headers(response_headers.as_mut_ptr(), response_headers.len());
    }
    // SAFETY: The C shim guarantees a NUL-terminated response header string.
    let response_headers = unsafe { CStr::from_ptr(response_headers.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    let headers = parse_browser_headers(&response_headers)?;
    let mut response_url = vec![0_i8; response_url_length + 1];
    // SAFETY: The destination is sized for the reported URL bytes and terminating NUL.
    unsafe { gen_http_xhr_copy_url(response_url.as_mut_ptr(), response_url.len()) };
    // SAFETY: The C shim guarantees a NUL-terminated response URL string.
    let final_url = unsafe { CStr::from_ptr(response_url.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    let status =
        StatusCode::from_u16(status as u16).map_err(|error| Error::Transport(error.to_string()))?;

    Ok(Response {
        status,
        headers,
        final_url,
        body: Box::new(Cursor::new(response_body)),
    })
}

#[cfg(target_os = "emscripten")]
fn parse_browser_headers(raw_headers: &str) -> Result<HeaderMap, Error> {
    let mut headers = HeaderMap::new();
    for line in raw_headers.split("\r\n").filter(|line| !line.is_empty()) {
        let Some((name, value)) = line.split_once(':') else {
            return Err(Error::InvalidResponseHeader(line.to_string()));
        };
        let name = HeaderName::from_bytes(name.trim().as_bytes())
            .map_err(|error| Error::InvalidResponseHeader(error.to_string()))?;
        let value = HeaderValue::from_bytes(value.trim().as_bytes())
            .map_err(|error| Error::InvalidResponseHeader(error.to_string()))?;
        headers.append(name, value);
    }
    Ok(headers)
}

#[cfg(target_os = "emscripten")]
unsafe extern "C" {
    fn gen_http_xhr_request(
        method: *const std::ffi::c_char,
        url: *const std::ffi::c_char,
        headers_json: *const std::ffi::c_char,
        body: *const u8,
        body_length: usize,
        status: *mut u32,
        response_length: *mut usize,
        response_headers_length: *mut usize,
        response_url_length: *mut usize,
        timeout_millis: u32,
        error_message: *mut std::ffi::c_char,
        error_capacity: usize,
    ) -> i32;
    fn gen_http_xhr_copy_body(destination: *mut u8);
    fn gen_http_xhr_copy_headers(destination: *mut std::ffi::c_char, capacity: usize);
    fn gen_http_xhr_copy_url(destination: *mut std::ffi::c_char, capacity: usize);
    fn gen_http_xhr_clear_response();
}

#[cfg(target_os = "emscripten")]
struct BrowserResponseGuard;

#[cfg(target_os = "emscripten")]
impl Drop for BrowserResponseGuard {
    fn drop(&mut self) {
        // SAFETY: The synchronous XHR shim owns this response state in the current Emscripten
        // module and clearing it is safe after every request outcome.
        unsafe { gen_http_xhr_clear_response() };
    }
}

#[cfg(test)]
mod tests {
    use std::{
        io::{Read as _, Write as _},
        net::{TcpListener, TcpStream},
        thread::{self, JoinHandle},
    };

    use http::{HeaderValue, Method, header::CONTENT_TYPE};

    use crate::{Client, Request};

    fn read_request(stream: &mut TcpStream) -> Vec<u8> {
        let mut request = Vec::new();
        let mut buffer = [0_u8; 1024];
        loop {
            let count = stream.read(&mut buffer).expect("should read request bytes");
            assert!(
                count > 0,
                "request should not end before its header terminator"
            );
            request.extend_from_slice(&buffer[..count]);
            let Some(header_end) = request.windows(4).position(|window| window == b"\r\n\r\n")
            else {
                continue;
            };
            let headers = String::from_utf8_lossy(&request[..header_end]);
            let content_length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())
                        .flatten()
                })
                .unwrap_or(0);
            if request.len() >= header_end + 4 + content_length {
                return request;
            }
        }
    }

    fn response_server(status: u16, body: &'static [u8]) -> (String, JoinHandle<Vec<u8>>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("should bind fixture server");
        let address = listener
            .local_addr()
            .expect("should read fixture server address");
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("should accept request");
            let request = read_request(&mut stream);
            write!(
                stream,
                "HTTP/1.1 {status} Fixture\r\nContent-Length: {}\r\nX-Remote-Version: 7\r\nConnection: close\r\n\r\n",
                body.len()
            )
            .expect("should write response headers");
            stream.write_all(body).expect("should write response body");
            request
        });
        (format!("http://{address}/transfer?signature=a%2Bb"), server)
    }

    #[test]
    fn test_native_request_preserves_headers_signed_query_and_streams_response() {
        let (url, server) = response_server(201, b"accepted");
        let request = Request::new(Method::POST, url)
            .header(
                http::HeaderName::from_static("x-api-key"),
                HeaderValue::from_static("secret-value"),
            )
            .header(CONTENT_TYPE, HeaderValue::from_static("application/json"))
            .body_bytes(br#"{"operation":"clone"}"#.to_vec());
        let mut response = Client::new()
            .execute(request)
            .expect("should send request through the native backend");
        assert_eq!(response.status().as_u16(), 201);
        assert_eq!(response.headers()["x-remote-version"], "7");
        assert!(response.final_url().contains("signature=a%2Bb"));
        let mut body = Vec::new();
        response
            .body_mut()
            .read_to_end(&mut body)
            .expect("should stream the response body");
        assert_eq!(body, b"accepted");
        let request = server.join().expect("should join fixture server");
        let request = String::from_utf8(request).expect("should decode request");
        assert!(
            request
                .to_ascii_lowercase()
                .contains("x-api-key: secret-value")
        );
        assert!(request.contains("signature=a%2Bb"));
        assert!(request.contains("{\"operation\":\"clone\"}"));
    }

    #[test]
    fn test_native_response_keeps_http_error_status_and_body() {
        let (url, server) = response_server(403, b"forbidden");
        let mut response = Client::new()
            .execute(Request::new(Method::GET, url))
            .expect("HTTP error statuses are responses, not transport failures");
        assert_eq!(response.status().as_u16(), 403);
        let mut body = String::new();
        response
            .body_mut()
            .read_to_string(&mut body)
            .expect("should read error response body");
        assert_eq!(body, "forbidden");
        server.join().expect("should join fixture server");
    }
}
