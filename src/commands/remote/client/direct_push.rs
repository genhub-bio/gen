//! Signed manifest publication for direct-GCS graph pushes.
//!
//! The graph VFS stages immutable blocks under a downscoped GCS token. The remote operations
//! layer calls this module only after Dolt finishes staging, so GenHub can atomically publish the
//! manifest. Its recovery probe asks GenHub to confirm that an accepted session is stale before
//! the client resets that session; signed URLs and intermediary error bodies remain private.

use std::io;

use reqwest::{
    StatusCode, Version,
    blocking::Client,
    header::{CONTENT_LENGTH, CONTENT_TYPE, HeaderMap},
    redirect::Policy,
};

use super::{DirectPushCapability, RemoteClientError};

const MAX_DIRECT_PUSH_ERROR_BODY_BYTES: u64 = 4096;
const STALE_SESSION_CHECK_HEADER: &str = "x-gen-session-stale-check";

/// Direct-GCS blocks stay unpublished until the remote operations layer calls this commit boundary.
///
/// After Dolt finishes staging, GenHub atomically publishes the manifest through the signed URL.
pub fn publish_direct_push(capability: &DirectPushCapability) -> Result<(), RemoteClientError> {
    send_direct_push_request(capability, false)
}

/// A saved push can outlive the remote head it was accepted against, so recovery asks GenHub before
/// discarding that session and lease.
///
/// A successful probe leaves a valid session untouched; only GenHub's exact stale-session response
/// permits the operations layer to start a fresh session.
pub fn confirm_direct_push_session_stale(
    capability: &DirectPushCapability,
) -> Result<bool, RemoteClientError> {
    match send_direct_push_request(capability, true) {
        Ok(()) => Ok(false),
        Err(RemoteClientError::StaleGraphSession) => Ok(true),
        Err(error) => Err(error),
    }
}

/// A signed URL authorizes only one endpoint, and strict frontends require `Content-Length: 0` for
/// the empty manifest commit POST.
fn send_direct_push_request(
    capability: &DirectPushCapability,
    stale_check: bool,
) -> Result<(), RemoteClientError> {
    let client = Client::builder().redirect(Policy::none()).build()?;
    let mut request = client
        .post(&capability.publish_url)
        .header(CONTENT_LENGTH, "0");
    if stale_check {
        request = request.header(STALE_SESSION_CHECK_HEADER, "1");
    }
    let mut response = request
        .send()
        .map_err(|error| RemoteClientError::Request(error.without_url()))?;
    if !response.status().is_success() {
        let status = response.status();
        if status == StatusCode::CONFLICT && is_stale_session_response(&mut response) {
            return Err(RemoteClientError::StaleGraphSession);
        }
        let content_type = response_content_type_label(response.headers());
        let protocol = http_version_label(response.version());
        log::debug!(
            "Direct GCS manifest publication returned HTTP {status}; response protocol {protocol}, content type {content_type}"
        );
        return Err(RemoteClientError::Http {
            status,
            message: "direct GCS manifest publication failed".to_string(),
        });
    }
    Ok(())
}

/// Recognizes only GenHub’s bounded stale-session response before recovery resets a lease.
/// Proxy and provider error pages must not authorize that state change.
fn is_stale_session_response(response: &mut reqwest::blocking::Response) -> bool {
    let is_json = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| {
            value.split(';').next().is_some_and(|media_type| {
                media_type.trim().eq_ignore_ascii_case("application/json")
            })
        });
    if !is_json {
        return false;
    }
    let mut bounded_response = io::Read::take(&mut *response, MAX_DIRECT_PUSH_ERROR_BODY_BYTES + 1);
    let mut body = Vec::new();
    if io::Read::read_to_end(&mut bounded_response, &mut body).is_err() {
        return false;
    }
    if body.len() as u64 > MAX_DIRECT_PUSH_ERROR_BODY_BYTES {
        return false;
    }
    serde_json::from_slice::<serde_json::Value>(&body)
        .ok()
        .and_then(|value| {
            value
                .get("reason")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned)
        })
        .is_some_and(|reason| reason == "stale_graph_session")
}

fn http_version_label(version: Version) -> &'static str {
    match version {
        Version::HTTP_09 => "HTTP/0.9",
        Version::HTTP_10 => "HTTP/1.0",
        Version::HTTP_11 => "HTTP/1.1",
        Version::HTTP_2 => "HTTP/2",
        Version::HTTP_3 => "HTTP/3",
        _ => "unknown HTTP version",
    }
}

fn response_content_type_label(headers: &HeaderMap) -> &'static str {
    let Some(value) = headers.get(CONTENT_TYPE) else {
        return "absent";
    };
    let Ok(value) = value.to_str() else {
        return "other";
    };
    let media_type = value
        .split_once(';')
        .map_or(value, |(media_type, _)| media_type)
        .trim()
        .to_ascii_lowercase();
    if media_type == "application/json"
        || (media_type.starts_with("application/") && media_type.ends_with("+json"))
    {
        "JSON"
    } else if media_type == "text/html" || media_type == "application/xhtml+xml" {
        "HTML"
    } else {
        "other"
    }
}

#[cfg(test)]
mod tests {
    use std::{
        io::{Read as _, Write as _},
        net::{TcpListener, TcpStream},
        thread::{self, JoinHandle},
    };

    use super::{
        MAX_DIRECT_PUSH_ERROR_BODY_BYTES, confirm_direct_push_session_stale, http_version_label,
        publish_direct_push, response_content_type_label,
    };
    use crate::commands::remote::client::{
        DirectPushCapability, RemoteClientError, SessionScope,
        tests::{TEST_TRANSFER_ID, mock_server},
    };

    fn read_request_head(stream: &mut TcpStream) -> Vec<u8> {
        let mut request = Vec::new();
        let mut buffer = [0_u8; 4096];
        loop {
            let read = stream
                .read(&mut buffer)
                .expect("should read request headers");
            assert!(read > 0, "request should include its headers");
            request.extend_from_slice(&buffer[..read]);
            if request.windows(4).any(|window| window == b"\r\n\r\n") {
                return request;
            }
        }
    }

    fn strict_empty_post_server(request_count: usize) -> (String, JoinHandle<Vec<(String, u16)>>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("should bind strict HTTP fixture");
        let address = listener
            .local_addr()
            .expect("should read strict HTTP fixture address");
        let handle = thread::spawn(move || {
            let mut requests = Vec::with_capacity(request_count);
            for _ in 0..request_count {
                let (mut stream, _) = listener
                    .accept()
                    .expect("should accept strict publication request");
                let request = String::from_utf8(read_request_head(&mut stream))
                    .expect("should decode request headers as UTF-8");
                let forced_response = if request.starts_with("POST /reject-html?") {
                    Some((411, "Length Required", "Content-Type: text/html\r\n"))
                } else if request.starts_with("POST /reject-json?") {
                    Some((
                        502,
                        "Bad Gateway",
                        "Content-Type: application/problem+json\r\n",
                    ))
                } else if request.starts_with("POST /reject-other?") {
                    Some((
                        502,
                        "Bad Gateway",
                        "Content-Type: application/x-private; name=private-canary\r\n",
                    ))
                } else if request.starts_with("POST /reject-absent?") {
                    Some((502, "Bad Gateway", ""))
                } else {
                    None
                };
                let header_end = request
                    .as_bytes()
                    .windows(4)
                    .position(|window| window == b"\r\n\r\n")
                    .expect("should find the end of request headers");
                let headers = request[..header_end].to_ascii_lowercase();
                let content_lengths = headers
                    .lines()
                    .filter_map(|line| line.split_once(':'))
                    .filter(|(name, _)| name.trim() == "content-length")
                    .map(|(_, value)| value.trim())
                    .collect::<Vec<_>>();
                let body = &request.as_bytes()[header_end + 4..];
                let (status, reason, content_type, response_body) =
                    if let Some((status, reason, content_type)) = forced_response {
                        (status, reason, content_type, "response-body-canary")
                    } else if content_lengths.as_slice() == ["0"] && body.is_empty() {
                        (204, "No Content", "", "")
                    } else {
                        (
                            411,
                            "Length Required",
                            "Content-Type: text/html\r\n",
                            "length required",
                        )
                    };
                write!(
                    stream,
                    "HTTP/1.1 {status} {reason}\r\n{content_type}Content-Length: {}\r\nConnection: close\r\n\r\n{response_body}",
                    response_body.len()
                )
                .expect("should write strict publication response");
                requests.push((request, status));
            }
            requests
        });
        (format!("http://{address}"), handle)
    }

    #[test]
    fn test_publish_direct_push_requires_explicit_zero_content_length() {
        let (origin, server) = strict_empty_post_server(7);
        let address = origin
            .strip_prefix("http://")
            .expect("should use HTTP/1.1 in the strict fixture");
        let mut missing_length_request =
            TcpStream::connect(address).expect("should connect to strict fixture");
        write!(
            missing_length_request,
            "POST /publish HTTP/1.1\r\nHost: {address}\r\nConnection: close\r\n\r\n"
        )
        .expect("should send request without Content-Length");
        let mut missing_length_rejection = String::new();
        missing_length_request
            .read_to_string(&mut missing_length_rejection)
            .expect("should read missing Content-Length rejection");
        assert!(
            missing_length_rejection.starts_with("HTTP/1.1 411 Length Required\r\n"),
            "strict fixture should reject missing Content-Length: {missing_length_rejection}"
        );

        let mut nonempty_length_request =
            TcpStream::connect(address).expect("should connect to strict fixture");
        write!(
            nonempty_length_request,
            "POST /publish HTTP/1.1\r\nHost: {address}\r\nContent-Length: 1\r\nConnection: close\r\n\r\n"
        )
        .expect("should send nonzero-length request headers");
        let mut rejection = String::new();
        nonempty_length_request
            .read_to_string(&mut rejection)
            .expect("should read strict fixture rejection");
        assert!(
            rejection.starts_with("HTTP/1.1 411 Length Required\r\n"),
            "strict fixture should reject nonzero Content-Length: {rejection}"
        );

        let capability = DirectPushCapability {
            database_uri: "gcs://bucket/prefix?access_token=not-used".to_string(),
            token_expires_at: "2030-01-01T00:00:00Z"
                .parse()
                .expect("should parse test token expiration"),
            session_id: TEST_TRANSFER_ID,
            session_scope: SessionScope {
                principal: "alice".to_string(),
                target_database: "example".to_string(),
                operations: "commit".to_string(),
            },
            publish_url: format!("{origin}/publish?signature=publish-secret"),
        };
        let publish_result = publish_direct_push(&capability);

        let rejection_cases = [
            ("html", 411, "HTML"),
            ("json", 502, "JSON"),
            ("other", 502, "other"),
            ("absent", 502, "absent"),
        ];
        let rejected_results = rejection_cases.map(|(body_type, _, _)| {
            let rejected_capability = DirectPushCapability {
                publish_url: format!("{origin}/reject-{body_type}?signature=publish-secret"),
                ..capability.clone()
            };
            publish_direct_push(&rejected_capability)
                .expect_err("should report forced publication rejection")
                .to_string()
        });
        let requests = server.join().expect("should finish strict fixture");

        assert_eq!(requests.len(), 7);
        assert_eq!(
            requests[0].1, 411,
            "missing Content-Length should be rejected"
        );
        assert_eq!(
            requests[1].1, 411,
            "nonzero Content-Length should be rejected"
        );
        assert_eq!(
            requests[2].1, 204,
            "publisher should send Content-Length: 0 and no request body; captured request: {}",
            requests[2].0
        );
        assert_eq!(requests[3].1, 411, "should return the HTML fixture status");
        assert_eq!(requests[4].1, 502, "should return the JSON fixture status");
        assert_eq!(requests[5].1, 502, "should return the other fixture status");
        assert_eq!(
            requests[6].1, 502,
            "should return the absent fixture status"
        );
        assert!(
            publish_result.is_ok(),
            "empty manifest publication should be accepted: {publish_result:?}"
        );
        let request = &requests[2].0;
        let (headers, body) = request
            .split_once("\r\n\r\n")
            .expect("should find captured publication request headers");
        assert!(
            headers.lines().any(|line| {
                line.split_once(':').is_some_and(|(name, value)| {
                    name.eq_ignore_ascii_case("content-length") && value.trim() == "0"
                })
            }),
            "wire request should explicitly contain Content-Length: 0: {headers}"
        );
        assert!(
            body.is_empty(),
            "manifest publication request body should be empty"
        );

        for ((body_type, status, content_type), rendered_error) in
            rejection_cases.into_iter().zip(rejected_results)
        {
            assert!(
                rendered_error.contains(&format!("Remote endpoint returned HTTP {status}")),
                "should identify the remote HTTP response for {body_type}: {rendered_error}"
            );
            assert!(
                rendered_error.contains("direct GCS manifest publication failed"),
                "should identify the failing publication stage: {rendered_error}"
            );
            assert!(
                !rendered_error.contains("publish-secret")
                    && !rendered_error.contains("response-body-canary")
                    && !rendered_error.contains("private-canary")
                    && !rendered_error.contains("HTTP/1.1")
                    && !rendered_error.contains(content_type),
                "default errors should hide signed URLs, response bodies, and transport diagnostics: {rendered_error}"
            );
        }
    }

    #[test]
    fn test_stale_session_response_requires_bounded_exact_reason() {
        let oversized_body = format!(
            "{{\"reason\":\"stale_graph_session\",\"detail\":\"{}\"}}",
            "x".repeat(MAX_DIRECT_PUSH_ERROR_BODY_BYTES as usize + 64)
        );
        let (origin, server) = mock_server(vec![
            (
                409,
                r#"{"reason":"stale_graph_session","message":"safe"}"#.to_string(),
            ),
            (
                409,
                r#"{"reason":"another_conflict","message":"private-canary"}"#.to_string(),
            ),
            (
                409,
                r#"{"reason":"stale_graph_session","message":"safe"}"#.to_string(),
            ),
            (409, oversized_body),
        ]);
        let capability = DirectPushCapability {
            database_uri: "gcs://bucket/prefix?access_token=not-used".to_string(),
            token_expires_at: "2030-01-01T00:00:00Z"
                .parse()
                .expect("should parse test token expiration"),
            session_id: TEST_TRANSFER_ID,
            session_scope: SessionScope {
                principal: "alice".to_string(),
                target_database: "default.db".to_string(),
                operations: "push".to_string(),
            },
            publish_url: format!("{origin}/publish?signature=publish-secret"),
        };

        assert!(
            confirm_direct_push_session_stale(&capability)
                .expect("should parse exact stale-session response")
        );
        let unrelated_conflict = confirm_direct_push_session_stale(&capability)
            .expect_err("should reject an unrelated conflict as stale");
        assert!(!unrelated_conflict.to_string().contains("private-canary"));
        assert!(matches!(
            publish_direct_push(&capability),
            Err(RemoteClientError::StaleGraphSession)
        ));
        let oversized_conflict = confirm_direct_push_session_stale(&capability)
            .expect_err("should reject an oversized stale response");
        assert!(
            !oversized_conflict
                .to_string()
                .contains("stale_graph_session")
        );

        let requests = server
            .join()
            .expect("should finish the mock GenHub request");
        assert_eq!(requests.len(), 4);
        for request in requests.iter().take(2).chain(requests.iter().skip(3)) {
            assert!(
                request
                    .lines()
                    .any(|line| line.eq_ignore_ascii_case("x-gen-session-stale-check: 1")),
                "stale verification should use the dedicated request header"
            );
            assert!(
                request
                    .lines()
                    .any(|line| line.eq_ignore_ascii_case("content-length: 0")),
                "stale verification should have an empty request body"
            );
        }
        assert!(
            !requests[2]
                .lines()
                .any(|line| line.eq_ignore_ascii_case("x-gen-session-stale-check: 1")),
            "normal manifest publication should not be sent as a stale probe"
        );
    }

    #[test]
    fn test_publish_response_diagnostic_labels_are_fixed() {
        let cases = [
            (Some("application/json; charset=utf-8"), "JSON"),
            (Some("application/problem+json"), "JSON"),
            (Some("text/html"), "HTML"),
            (Some("application/xhtml+xml"), "HTML"),
            (Some("application/octet-stream"), "other"),
            (Some("not-a-media-type"), "other"),
            (None, "absent"),
        ];
        for (value, expected) in cases {
            let mut headers = reqwest::header::HeaderMap::new();
            if let Some(value) = value {
                headers.insert(
                    reqwest::header::CONTENT_TYPE,
                    value.parse().expect("should parse test Content-Type"),
                );
            }
            assert_eq!(response_content_type_label(&headers), expected);
        }
        assert_eq!(http_version_label(reqwest::Version::HTTP_11), "HTTP/1.1");
    }
}
