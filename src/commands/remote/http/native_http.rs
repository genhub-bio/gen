//! Native implementation of the shared HTTP boundary, backed by `reqwest::blocking`. Exists so
//! the same [`super::HttpRequest`]/[`super::HttpResponse`] shape can eventually front both the
//! native and `emscripten` transports; existing call sites (e.g. `remote::client`) are unchanged
//! and continue to use `reqwest` directly.

use std::str::FromStr;

use reqwest::{Method, blocking::Client};

use super::{BrowserHttpError, HttpRequest, HttpResponse};

pub fn request(http_request: HttpRequest<'_>) -> Result<HttpResponse, BrowserHttpError> {
    let method = Method::from_str(http_request.method)
        .map_err(|_| BrowserHttpError::InvalidMethod(http_request.method.to_string()))?;
    let client = Client::new();
    let mut builder = client.request(method, http_request.url);
    for (name, value) in http_request.headers {
        builder = builder.header(*name, *value);
    }
    if let Some(body) = http_request.body {
        builder = builder.body(body.to_vec());
    }
    let response = builder.send()?;
    let status = response.status().as_u16();
    let body = response.bytes()?.to_vec();
    Ok(HttpResponse { status, body })
}

#[cfg(test)]
mod tests {
    use std::{
        io::{Read as _, Write as _},
        net::TcpListener,
        thread,
    };

    use super::{BrowserHttpError, HttpRequest, request};

    #[test]
    fn test_request_get_returns_status_and_body() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("should bind");
        let address = listener.local_addr().expect("should read address");
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("should accept");
            let mut buffer = [0_u8; 1024];
            let _ = stream.read(&mut buffer);
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\nConnection: close\r\n\r\nhello",
                )
                .expect("should write response");
        });
        let response = request(HttpRequest {
            method: "GET",
            url: &format!("http://{address}"),
            headers: &[],
            body: None,
        })
        .expect("request should succeed");
        handle.join().expect("server thread should finish");

        assert_eq!(response.status, 200);
        assert_eq!(response.body, b"hello");
    }

    #[test]
    fn test_request_rejects_invalid_method() {
        let error = request(HttpRequest {
            method: "NOT A METHOD",
            url: "http://127.0.0.1:1",
            headers: &[],
            body: None,
        })
        .expect_err("invalid method should be rejected before connecting");
        assert!(matches!(error, BrowserHttpError::InvalidMethod(_)));
    }
}
