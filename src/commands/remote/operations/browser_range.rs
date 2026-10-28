//! Range-backed reads for files exposed to the browser through a read capability.
//!
//! A notebook can query the remote graph through its cloud VFS while opening a local
//! SQLite connection, but file assets also need a file-like interface. This reader
//! fetches bounded ranges on demand and retains the most recent range as a seekable cache, so
//! large assets are not cloned into a browser-side file before they are used.

use std::{
    fmt,
    io::{self, Read, Seek, SeekFrom},
    sync::Arc,
};

use gen_http::{
    Client, HeaderValue, Method, Request, Response, StatusCode,
    header::{CONTENT_LENGTH, CONTENT_RANGE, RANGE},
};
use url::Url;

/// Maximum number of asset bytes requested or retained in the seek cache per range.
pub(super) const MAX_RANGE_BYTES: usize = 256 * 1024;

const MAX_RANGE_BYTES_U64: u64 = MAX_RANGE_BYTES as u64;

/// Resolves a capability URL, with `true` requesting a fresh capability for the same asset.
pub(super) type UrlProvider = dyn Fn(bool) -> io::Result<String> + Send + Sync;

/// Reads an HTTP asset on demand while keeping a renewed capability bound to one asset path.
///
/// The callback owns capability issuance and refresh. It is invoked before each range request,
/// then once more with `true` after an authorization failure. The URL's query may change during
/// refresh, while its scheme, host, port, and path must remain fixed for this reader's lifetime.
pub(super) struct RangeAssetReader {
    client: Client,
    url_provider: Arc<UrlProvider>,
    transport_identity: Option<TransportIdentity>,
    position: u64,
    length: Option<u64>,
    buffer_start: u64,
    buffer: Vec<u8>,
}

impl RangeAssetReader {
    /// Creates a lazy range reader without issuing a capability or making a request.
    ///
    /// # Errors
    ///
    /// Returns an error if the bounded HTTP client cannot be initialized.
    pub(super) fn new(url_provider: Arc<UrlProvider>) -> io::Result<Self> {
        let client = Client::without_redirects()
            .map_err(|_| io::Error::other("could not create the asset HTTP client"))?;

        Ok(Self {
            client,
            url_provider,
            transport_identity: None,
            position: 0,
            length: None,
            buffer_start: 0,
            buffer: Vec::new(),
        })
    }

    /// Obtains a capability URL and rejects refreshes that change the source asset identity.
    fn capability_url(&mut self, force_renew: bool) -> io::Result<Url> {
        let raw_url = (self.url_provider)(force_renew)
            .map_err(|error| io::Error::new(error.kind(), "asset capability is unavailable"))?;
        let url = Url::parse(&raw_url)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "invalid asset URL"))?;
        let identity = TransportIdentity::from_url(&url)?;

        match &self.transport_identity {
            Some(expected) if expected != &identity => {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "renewed capability points to a different asset",
                ));
            }
            Some(_) => {}
            None => self.transport_identity = Some(identity),
        }

        Ok(url)
    }

    /// Gets one bounded byte range, refreshing the capability once after an authorization error.
    fn fetch_range(&mut self, start: u64) -> io::Result<()> {
        let end = start.saturating_add(MAX_RANGE_BYTES_U64 - 1);
        let url = self.capability_url(false)?;
        let mut response = self.request_range(&url, start, end)?;

        if is_unauthorized(response.status()) {
            let refreshed_url = self.capability_url(true)?;
            response = self.request_range(&refreshed_url, start, end)?;
        }

        self.accept_response(response, start, end)
    }

    /// Sends an unauthenticated range GET; the capability query is the only request credential.
    fn request_range(&self, url: &Url, start: u64, end: u64) -> io::Result<Response> {
        let range_value = HeaderValue::from_str(&format!("bytes={start}-{end}"))
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "invalid byte range"))?;
        let request = Request::new(Method::GET, url.as_str())
            .header(RANGE, range_value)
            .max_response_bytes(MAX_RANGE_BYTES);
        let response = self
            .client
            .execute(request)
            .map_err(|_| io::Error::other("asset range request failed"))?;

        // Native HTTP disables redirects. Browser XHR may follow a redirect before returning, so
        // reject a response whose reported final URL differs from the requested capability URL.
        if response.final_url() != url.as_str() {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "asset request changed URL",
            ));
        }

        Ok(response)
    }

    /// Accepts only exact, bounded partial responses, or a validated end-of-file response.
    ///
    /// # Errors
    ///
    /// Returns an error for an unexpected status, changed object size, invalid range metadata,
    /// or a body whose size does not match the declared range.
    fn accept_response(
        &mut self,
        mut response: Response,
        requested_start: u64,
        requested_end: u64,
    ) -> io::Result<()> {
        // Do not expose body bytes until the server confirms the exact requested range and the
        // stable object size. Ignored ranges or object replacement must not yield corrupted data.
        match response.status() {
            StatusCode::PARTIAL_CONTENT => {
                let (actual_start, actual_end, total) = parse_content_range(&response)?;
                if total == 0 || actual_start != requested_start || actual_start > actual_end {
                    return Err(invalid_response());
                }

                let expected_end = requested_end.min(total - 1);
                if actual_end != expected_end || actual_end >= total {
                    return Err(invalid_response());
                }

                let expected_length = actual_end
                    .checked_sub(actual_start)
                    .and_then(|length| length.checked_add(1))
                    .ok_or_else(invalid_response)?;
                if expected_length > MAX_RANGE_BYTES_U64 {
                    return Err(invalid_response());
                }
                let content_length = declared_content_length(&response)?;
                if content_length > MAX_RANGE_BYTES_U64 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "asset response exceeded the range size limit",
                    ));
                }
                if content_length != expected_length {
                    return Err(invalid_response());
                }

                let bytes = read_bounded_body(&mut response)?;
                if bytes.len() as u64 != expected_length {
                    return Err(invalid_response());
                }

                self.remember_length(total)?;
                self.buffer_start = actual_start;
                self.buffer = bytes;
                Ok(())
            }
            StatusCode::RANGE_NOT_SATISFIABLE => {
                let total = parse_unsatisfied_content_range(&response)?;
                if requested_start < total {
                    return Err(invalid_response());
                }

                // 416 bodies are server error details, not asset bytes. The validated unsatisfied
                // range is sufficient to establish EOF, so drop the response without consuming it.
                self.remember_length(total)?;
                self.buffer_start = requested_start;
                self.buffer.clear();
                Ok(())
            }
            status if is_unauthorized(status) => Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "asset capability was rejected after refresh",
            )),
            _ => Err(io::Error::other(
                "asset server returned an unexpected response",
            )),
        }
    }

    /// Keeps all fetched ranges tied to one immutable object size.
    fn remember_length(&mut self, total: u64) -> io::Result<()> {
        if self.length.is_some_and(|known| known != total) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "asset size changed during range reads",
            ));
        }
        self.length = Some(total);
        Ok(())
    }

    /// Resolves file size on demand for `SeekFrom::End` without downloading the whole asset.
    fn ensure_length(&mut self) -> io::Result<u64> {
        if let Some(length) = self.length {
            return Ok(length);
        }

        self.fetch_range(0)?;
        self.length.ok_or_else(invalid_response)
    }
}

impl Read for RangeAssetReader {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        if output.is_empty() || self.length.is_some_and(|length| self.position >= length) {
            return Ok(0);
        }

        let buffered_end = self.buffer_start.saturating_add(self.buffer.len() as u64);
        if self.position < self.buffer_start || self.position >= buffered_end {
            self.fetch_range(self.position)?;
        }

        if self.buffer.is_empty() {
            return Ok(0);
        }

        let offset =
            usize::try_from(self.position - self.buffer_start).map_err(|_| invalid_response())?;
        let available = self.buffer.len().saturating_sub(offset);
        let count = available.min(output.len());
        output[..count].copy_from_slice(&self.buffer[offset..offset + count]);
        self.position = self
            .position
            .checked_add(count as u64)
            .ok_or_else(invalid_response)?;
        Ok(count)
    }
}

impl Seek for RangeAssetReader {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        let target = match position {
            SeekFrom::Start(position) => position,
            SeekFrom::Current(offset) => checked_seek(self.position, offset)?,
            SeekFrom::End(offset) => checked_seek(self.ensure_length()?, offset)?,
        };
        self.position = target;
        Ok(target)
    }
}

impl fmt::Debug for RangeAssetReader {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RangeAssetReader")
            .field("position", &self.position)
            .field("length", &self.length)
            .field("buffer_start", &self.buffer_start)
            .field("buffer_len", &self.buffer.len())
            .finish_non_exhaustive()
    }
}

#[derive(PartialEq, Eq)]
struct TransportIdentity {
    scheme: String,
    host: String,
    port: u16,
    path: String,
}

impl TransportIdentity {
    /// Extracts the URL components that must stay stable when a signed query is renewed.
    ///
    /// # Errors
    ///
    /// Returns an error for unsupported schemes, embedded credentials, or URL fragments.
    fn from_url(url: &Url) -> io::Result<Self> {
        let scheme = url.scheme();
        if !matches!(scheme, "http" | "https")
            || !url.username().is_empty()
            || url.password().is_some()
            || url.fragment().is_some()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "asset URL has an unsupported form",
            ));
        }

        let host = url
            .host_str()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "asset URL has no host"))?;
        let port = url
            .port_or_known_default()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "asset URL has no port"))?;

        Ok(Self {
            scheme: scheme.to_owned(),
            host: host.to_owned(),
            port,
            path: url.path().to_owned(),
        })
    }
}

/// Treats expired or denied capabilities as the only condition that may trigger one renewal.
fn is_unauthorized(status: StatusCode) -> bool {
    status == StatusCode::UNAUTHORIZED || status == StatusCode::FORBIDDEN
}

/// Parses a satisfiable range so callers can validate its exact bounds before using the bytes.
fn parse_content_range(response: &Response) -> io::Result<(u64, u64, u64)> {
    let value = response
        .headers()
        .get(CONTENT_RANGE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("bytes "))
        .ok_or_else(invalid_response)?;
    let (range, total) = value.split_once('/').ok_or_else(invalid_response)?;
    let (start, end) = range.split_once('-').ok_or_else(invalid_response)?;
    let start = start.parse().map_err(|_| invalid_response())?;
    let end = end.parse().map_err(|_| invalid_response())?;
    let total = total.parse().map_err(|_| invalid_response())?;
    Ok((start, end, total))
}

/// Parses the object size reported for a request starting at or beyond end-of-file.
fn parse_unsatisfied_content_range(response: &Response) -> io::Result<u64> {
    response
        .headers()
        .get(CONTENT_RANGE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("bytes */"))
        .and_then(|total| total.parse().ok())
        .ok_or_else(invalid_response)
}

/// Requires an explicit body length so partial or malformed responses cannot be accepted.
fn declared_content_length(response: &Response) -> io::Result<u64> {
    response
        .headers()
        .get(CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|length| length.parse().ok())
        .ok_or_else(invalid_response)
}

/// Reads one response body with a hard cap independent of transport-provided size metadata.
fn read_bounded_body(response: &mut Response) -> io::Result<Vec<u8>> {
    let mut body = Vec::new();
    let mut limited_reader = response
        .body_mut()
        .take(MAX_RANGE_BYTES_U64.saturating_add(1));
    limited_reader
        .read_to_end(&mut body)
        .map_err(|_| io::Error::other("asset response body could not be read"))?;
    if body.len() > MAX_RANGE_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "asset response exceeded the range size limit",
        ));
    }
    Ok(body)
}

/// Applies a signed seek offset without wrapping before the beginning or beyond `u64`.
fn checked_seek(base: u64, offset: i64) -> io::Result<u64> {
    let target = i128::from(base) + i128::from(offset);
    if target < 0 || target > i128::from(u64::MAX) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "seek position is outside the supported range",
        ));
    }
    Ok(target as u64)
}

/// Produces a generic response error without exposing server bodies or capability data.
fn invalid_response() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, "invalid asset range response")
}

#[cfg(all(test, not(target_os = "emscripten")))]
mod tests {
    use std::{
        io::{Read as _, Seek as _, SeekFrom, Write as _},
        net::{SocketAddr, TcpListener, TcpStream},
        sync::{Arc, Mutex},
        thread::{self, JoinHandle},
        time::Duration,
    };

    use super::{MAX_RANGE_BYTES, RangeAssetReader};

    #[derive(Clone, Debug)]
    struct CapturedRequest {
        method: String,
        target: String,
        range: String,
        has_authorization: bool,
        body_bytes: usize,
        chunked_body: bool,
    }

    fn spawn_server(
        count: usize,
        handler: impl Fn(&CapturedRequest, usize) -> (String, Vec<u8>) + Send + 'static,
    ) -> (SocketAddr, JoinHandle<Vec<CapturedRequest>>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("should bind fake HTTP server");
        let address = listener
            .local_addr()
            .expect("should read fake HTTP server address");
        let thread = thread::spawn(move || {
            let mut requests = Vec::with_capacity(count);
            for index in 0..count {
                let (mut stream, _) = listener.accept().expect("should accept fake HTTP request");
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .expect("should set request timeout");
                let request = read_request(&mut stream);
                // The signed URL is the only credential; asset data is fetched with bodyless GETs.
                assert_eq!(request.method, "GET");
                assert!(!request.has_authorization);
                assert_eq!(request.body_bytes, 0);
                assert!(!request.chunked_body);
                let (headers, body) = handler(&request, index);
                requests.push(request);
                stream
                    .write_all(headers.as_bytes())
                    .expect("should write fake HTTP headers");
                stream
                    .write_all(&body)
                    .expect("should write fake HTTP body");
            }
            requests
        });
        (address, thread)
    }

    fn read_request(stream: &mut TcpStream) -> CapturedRequest {
        let mut request = Vec::new();
        let mut chunk = [0_u8; 1024];
        while !request.windows(4).any(|window| window == b"\r\n\r\n") {
            let count = stream
                .read(&mut chunk)
                .expect("should read fake HTTP request");
            assert_ne!(count, 0, "request should contain HTTP headers");
            request.extend_from_slice(&chunk[..count]);
            assert!(
                request.len() <= 16 * 1024,
                "request headers should be bounded"
            );
        }

        let header_end = request
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .expect("request should include a header terminator")
            + 4;
        let headers = String::from_utf8(request[..header_end].to_vec())
            .expect("should parse HTTP request headers as text");
        let mut lines = headers.lines();
        let request_line = lines.next().expect("should have an HTTP request line");
        let method = request_line
            .split_whitespace()
            .next()
            .expect("request line should include a method")
            .to_owned();
        let target = request_line
            .split_whitespace()
            .nth(1)
            .expect("request line should include a target")
            .to_owned();
        let range = lines
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("range")
                    .then(|| value.trim().to_owned())
            })
            .expect("request should include a byte range");
        let has_authorization = headers.lines().any(|line| {
            line.split_once(':')
                .is_some_and(|(name, _)| name.eq_ignore_ascii_case("authorization"))
        });
        let content_length = headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length").then(|| {
                    value
                        .trim()
                        .parse::<usize>()
                        .expect("content length is numeric")
                })
            })
            .unwrap_or_default();
        let chunked_body = headers.lines().any(|line| {
            line.split_once(':').is_some_and(|(name, value)| {
                name.eq_ignore_ascii_case("transfer-encoding")
                    && value.trim().eq_ignore_ascii_case("chunked")
            })
        });
        let body_bytes = content_length.max(request.len().saturating_sub(header_end));
        CapturedRequest {
            method,
            target,
            range,
            has_authorization,
            body_bytes,
            chunked_body,
        }
    }

    fn response(status: &str, headers: &[(&str, String)], body: Vec<u8>) -> (String, Vec<u8>) {
        let mut response = format!("HTTP/1.1 {status}\r\n");
        for (name, value) in headers {
            response.push_str(name);
            response.push_str(": ");
            response.push_str(value);
            response.push_str("\r\n");
        }
        response.push_str("Connection: close\r\n\r\n");
        (response, body)
    }

    fn partial_response(data: &[u8], request: &CapturedRequest) -> (String, Vec<u8>) {
        let (start, end) = parse_requested_range(&request.range);
        if start >= data.len() as u64 {
            return response(
                "416 Range Not Satisfiable",
                &[
                    ("Content-Range", format!("bytes */{}", data.len())),
                    ("Content-Length", "0".to_owned()),
                ],
                Vec::new(),
            );
        }

        let actual_end = end.min(data.len() as u64 - 1);
        let body = data[start as usize..=actual_end as usize].to_vec();
        response(
            "206 Partial Content",
            &[
                (
                    "Content-Range",
                    format!("bytes {start}-{actual_end}/{}", data.len()),
                ),
                ("Content-Length", body.len().to_string()),
            ],
            body,
        )
    }

    fn parse_requested_range(range: &str) -> (u64, u64) {
        let value = range
            .strip_prefix("bytes=")
            .expect("range should use byte units");
        let (start, end) = value.split_once('-').expect("range should have bounds");
        (
            start.parse().expect("range start should be numeric"),
            end.parse().expect("range end should be numeric"),
        )
    }

    fn capability(address: SocketAddr, path: &str, token: &str) -> String {
        format!("http://{address}{path}?token={token}")
    }

    #[test]
    fn test_reads_chunks_seeks_and_accepts_clipped_eof() {
        let mut data = vec![0_u8; MAX_RANGE_BYTES + 17];
        for (index, byte) in data.iter_mut().enumerate() {
            *byte = (index % 251) as u8;
        }
        let server_data = data.clone();
        let (address, server) =
            spawn_server(2, move |request, _| partial_response(&server_data, request));
        let url = capability(address, "/asset", "read");
        let mut reader = RangeAssetReader::new(Arc::new(move |_| Ok(url.clone())))
            .expect("should create range reader");

        let mut downloaded = Vec::new();
        reader
            .read_to_end(&mut downloaded)
            .expect("should read remote asset");
        assert_eq!(downloaded, data);
        assert_eq!(
            reader.seek(SeekFrom::End(-8)).unwrap(),
            (MAX_RANGE_BYTES + 9) as u64
        );
        let mut tail = [0_u8; 8];
        reader
            .read_exact(&mut tail)
            .expect("should reread buffered tail");
        assert_eq!(&tail, &data[data.len() - 8..]);

        let requests = server.join().expect("should finish fake HTTP server");
        assert_eq!(requests.len(), 2);
        assert_eq!(parse_requested_range(&requests[0].range).0, 0);
        assert_eq!(
            parse_requested_range(&requests[0].range).1,
            MAX_RANGE_BYTES as u64 - 1
        );
        assert_eq!(
            parse_requested_range(&requests[1].range).0,
            MAX_RANGE_BYTES as u64
        );
        assert_eq!(requests[0].target, "/asset?token=read");
    }

    #[test]
    fn test_renews_expired_capability_once_for_same_asset() {
        let data = Arc::new(b"notebook asset".to_vec());
        let handler_data = Arc::clone(&data);
        let (address, server) = spawn_server(2, move |request, index| {
            if index == 0 {
                response(
                    "403 Forbidden",
                    &[("Content-Length", "0".to_owned())],
                    Vec::new(),
                )
            } else {
                partial_response(&handler_data, request)
            }
        });
        let callback_address = address;
        let callback_count = Arc::new(Mutex::new(Vec::new()));
        let observed_calls = Arc::clone(&callback_count);
        let mut reader = RangeAssetReader::new(Arc::new(move |force_renew| {
            observed_calls.lock().unwrap().push(force_renew);
            let token = if force_renew { "fresh" } else { "expired" };
            Ok(capability(callback_address, "/asset", token))
        }))
        .expect("should create range reader");

        let mut contents = Vec::new();
        reader
            .read_to_end(&mut contents)
            .expect("should retry with a fresh capability");
        assert_eq!(contents, *data);
        assert_eq!(*callback_count.lock().unwrap(), [false, true]);

        let requests = server.join().expect("should finish fake HTTP server");
        assert_eq!(requests[0].target, "/asset?token=expired");
        assert_eq!(requests[1].target, "/asset?token=fresh");
    }

    #[test]
    fn test_rejects_capability_refresh_for_different_path_before_request() {
        let (address, server) = spawn_server(1, move |_, _| {
            response(
                "403 Forbidden",
                &[("Content-Length", "0".to_owned())],
                Vec::new(),
            )
        });
        let callback_count = Arc::new(Mutex::new(Vec::new()));
        let observed_calls = Arc::clone(&callback_count);
        let mut reader = RangeAssetReader::new(Arc::new(move |force_renew| {
            observed_calls.lock().unwrap().push(force_renew);
            let path = if force_renew { "/different" } else { "/asset" };
            Ok(capability(address, path, "token"))
        }))
        .expect("should create range reader");

        let mut byte = [0_u8; 1];
        let error = reader
            .read(&mut byte)
            .expect_err("should reject a path-changing refresh");
        assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
        assert_eq!(*callback_count.lock().unwrap(), [false, true]);
        let requests = server.join().expect("should finish fake HTTP server");
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].target, "/asset?token=token");
    }

    #[test]
    fn test_rejects_invalid_range_metadata_and_oversized_body() {
        let (address, server) = spawn_server(1, |_, _| {
            response(
                "206 Partial Content",
                &[
                    ("Content-Range", "bytes 1-1/2".to_owned()),
                    ("Content-Length", "1".to_owned()),
                ],
                vec![1],
            )
        });
        let url = capability(address, "/invalid", "read");
        let mut reader = RangeAssetReader::new(Arc::new(move |_| Ok(url.clone())))
            .expect("should create range reader");
        assert_eq!(
            reader.read(&mut [0_u8; 1]).unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
        server
            .join()
            .expect("should finish invalid metadata server");

        let (address, server) = spawn_server(1, |_, _| {
            let body = vec![7; MAX_RANGE_BYTES + 1];
            response(
                "206 Partial Content",
                &[
                    (
                        "Content-Range",
                        format!("bytes 0-{}/{}", MAX_RANGE_BYTES - 1, MAX_RANGE_BYTES + 1),
                    ),
                    ("Content-Length", body.len().to_string()),
                ],
                body,
            )
        });
        let url = capability(address, "/oversized", "read");
        let mut reader = RangeAssetReader::new(Arc::new(move |_| Ok(url.clone())))
            .expect("should create range reader");
        assert_eq!(
            reader.read(&mut [0_u8; 1]).unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
        server.join().expect("should finish oversized body server");
    }

    #[test]
    fn test_rejects_object_size_changes_between_ranges() {
        let (address, server) = spawn_server(2, |request, index| {
            let (start, end) = parse_requested_range(&request.range);
            let total = MAX_RANGE_BYTES as u64 + if index == 0 { 2 } else { 3 };
            let actual_end = end.min(total - 1);
            let body_length = (actual_end - start + 1) as usize;
            response(
                "206 Partial Content",
                &[
                    (
                        "Content-Range",
                        format!("bytes {start}-{actual_end}/{total}"),
                    ),
                    ("Content-Length", body_length.to_string()),
                ],
                vec![3; body_length],
            )
        });
        let url = capability(address, "/changing", "read");
        let mut reader = RangeAssetReader::new(Arc::new(move |_| Ok(url.clone())))
            .expect("should create range reader");
        let mut output = vec![0_u8; MAX_RANGE_BYTES + 1];
        assert_eq!(
            reader.read_exact(&mut output).unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
        server.join().expect("should finish changing-size server");
    }

    #[test]
    fn test_empty_asset_uses_unsatisfied_range_body_and_seek_checks_bounds() {
        let body = b"<Error>range not satisfiable</Error>".to_vec();
        let (address, server) = spawn_server(1, move |_, _| {
            let response_body = body.clone();
            response(
                "416 Range Not Satisfiable",
                &[
                    ("Content-Range", "bytes */0".to_owned()),
                    ("Content-Length", response_body.len().to_string()),
                ],
                response_body,
            )
        });
        let url = capability(address, "/empty", "read");
        let mut reader = RangeAssetReader::new(Arc::new(move |_| Ok(url.clone())))
            .expect("should create range reader");
        assert_eq!(reader.read(&mut [0_u8; 1]).unwrap(), 0);
        assert_eq!(reader.seek(SeekFrom::End(0)).unwrap(), 0);
        assert_eq!(
            reader.seek(SeekFrom::Current(-1)).unwrap_err().kind(),
            std::io::ErrorKind::InvalidInput
        );
        server.join().expect("should finish empty-asset server");
    }

    #[test]
    fn test_closed_capability_callback_fails_without_network_request() {
        let mut reader = RangeAssetReader::new(Arc::new(|_| {
            Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "closed source token must not escape",
            ))
        }))
        .expect("should create lazy range reader");
        let error = reader
            .read(&mut [0_u8; 1])
            .expect_err("should report a closed source callback");
        assert_eq!(error.kind(), std::io::ErrorKind::BrokenPipe);
        assert!(!error.to_string().contains("token"));
    }
}
