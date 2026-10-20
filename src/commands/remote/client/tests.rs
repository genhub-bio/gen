//! Endpoint facade tests shared with the remote transfer operations.

use std::{
    collections::HashMap,
    io::{self, Read as _, Write as _},
    net::{TcpListener, TcpStream},
    sync::Mutex,
    thread::{self, JoinHandle},
};

use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use reqwest::blocking::Client;
use uuid::Uuid;

use super::{
    AssetTransferRequest, CapabilityRequest, CapabilityResponse, DirectPushCapability,
    RemoteClientError, RemoteOperation, RepositoryRemote, SessionScope,
    acquire_capability_with_store, acquire_capability_with_store_and_token,
    authentication::{RequestAuthorization, TokenStore},
    normalized_origin, send_asset_transfers,
};
use crate::commands::remote::server::AuthTokens;

pub(super) const TEST_TRANSFER_ID: Uuid = Uuid::from_u128(1);

pub(super) fn test_jwt_with_expiration(expiration: i64) -> String {
    let header = serde_json::json!({"alg": "HS256", "typ": "JWT"}).to_string();
    let claims = serde_json::json!({"exp": expiration}).to_string();
    format!(
        "{}.{}.{}",
        URL_SAFE_NO_PAD.encode(header),
        URL_SAFE_NO_PAD.encode(claims),
        URL_SAFE_NO_PAD.encode("test-signature")
    )
}

pub(super) struct MemoryTokenStore {
    tokens: Mutex<HashMap<String, AuthTokens>>,
}

impl MemoryTokenStore {
    pub(super) fn empty() -> Self {
        Self {
            tokens: Mutex::new(HashMap::new()),
        }
    }

    pub(super) fn with_tokens(identity: &str, tokens: AuthTokens) -> Self {
        let store = Self::empty();
        store
            .tokens
            .lock()
            .expect("should lock token store")
            .insert(identity.to_string(), tokens);
        store
    }

    pub(super) fn current(&self, identity: &str) -> Option<AuthTokens> {
        self.tokens
            .lock()
            .expect("should lock token store")
            .get(identity)
            .cloned()
    }
}

impl TokenStore for MemoryTokenStore {
    fn load(&self, identity: &str) -> io::Result<AuthTokens> {
        self.current(identity)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no in-memory credentials"))
    }

    fn save(&self, identity: &str, tokens: &AuthTokens) -> io::Result<()> {
        self.tokens
            .lock()
            .expect("should lock token store")
            .insert(identity.to_string(), tokens.clone());
        Ok(())
    }
}

fn read_request(stream: &mut TcpStream) -> String {
    let mut request = Vec::new();
    let mut buffer = [0_u8; 4096];
    let mut expected_length = None;
    loop {
        let read = stream.read(&mut buffer).expect("should read mock request");
        assert!(
            read > 0,
            "mock request should not close before its body is complete"
        );
        request.extend_from_slice(&buffer[..read]);
        if expected_length.is_none()
            && let Some(header_end) = request.windows(4).position(|window| window == b"\r\n\r\n")
        {
            let headers = String::from_utf8_lossy(&request[..header_end]).to_ascii_lowercase();
            let content_length = headers
                .lines()
                .find_map(|line| line.strip_prefix("content-length:"))
                .map(str::trim)
                .map(|value| value.parse::<usize>().expect("should parse content length"))
                .unwrap_or(0);
            expected_length = Some(header_end + 4 + content_length);
        }
        if expected_length.is_some_and(|length| request.len() >= length) {
            break;
        }
    }
    String::from_utf8(request).expect("mock request should be UTF-8")
}

fn status_reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        _ => "Test Response",
    }
}

pub(super) fn mock_server(responses: Vec<(u16, String)>) -> (String, JoinHandle<Vec<String>>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("should bind mock GenHub");
    let address = listener
        .local_addr()
        .expect("should read mock GenHub address");
    let handle = thread::spawn(move || {
        let mut requests = Vec::with_capacity(responses.len());
        for (status, body) in responses {
            let (mut stream, _) = listener
                .accept()
                .expect("should accept mock GenHub request");
            requests.push(read_request(&mut stream));
            write!(
                stream,
                "HTTP/1.1 {status} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                status_reason(status),
                body.len()
            )
            .expect("should write mock GenHub response");
        }
        requests
    });
    (format!("http://{address}"), handle)
}

pub(super) fn mock_server_disconnects_during_refresh() -> (String, JoinHandle<Vec<String>>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("should bind mock GenHub");
    let address = listener
        .local_addr()
        .expect("should read mock GenHub address");
    let handle = thread::spawn(move || {
        let (mut capability_stream, _) =
            listener.accept().expect("should accept capability request");
        let capability_request = read_request(&mut capability_stream);
        let body = "{\"message\":\"expired\"}";
        write!(
            capability_stream,
            "HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
        .expect("should reject expired access token");

        let (mut refresh_stream, _) = listener.accept().expect("should accept refresh request");
        let refresh_request = read_request(&mut refresh_stream);
        drop(refresh_stream);
        vec![capability_request, refresh_request]
    });
    (format!("http://{address}"), handle)
}

pub(super) fn capability_body(remote_url: &str) -> String {
    serde_json::json!({
        "remote_url": remote_url,
        "expires_at": "2030-01-01T00:00:00Z",
        "default_branch": "main",
        "transfer_id": TEST_TRANSFER_ID
    })
    .to_string()
}

pub(super) fn repository(origin: &str) -> RepositoryRemote {
    RepositoryRemote::parse(&format!("{origin}/api/repos/alice/example"))
        .expect("should parse mock repository")
}

pub(super) fn assert_response_decode_context(
    error: &RemoteClientError,
    endpoint: &str,
    response_body: &str,
) {
    let rendered = error.to_string();
    assert!(
        rendered.contains(endpoint),
        "error should identify {endpoint}: {rendered}"
    );
    assert!(
        rendered.contains("HTTP 200"),
        "error should include status: {rendered}"
    );
    assert!(
        rendered.contains(&format!(
            "declared Content-Length Some({})",
            response_body.len()
        )),
        "error should include only the declared response length: {rendered}"
    );
    assert!(
        !format!("{error:?}").contains("sensitive-canary"),
        "error must not expose response-body contents"
    );
    assert!(
        std::error::Error::source(error)
            .and_then(|source| source.downcast_ref::<reqwest::Error>())
            .is_some(),
        "response decoder error should retain its reqwest source"
    );
}

pub(super) fn no_interactive_login(
    _origin: &str,
) -> Result<AuthTokens, Box<dyn std::error::Error>> {
    Err("interactive login should not run".into())
}

#[test]
fn test_normalized_origin_preserves_non_default_port() {
    assert_eq!(
        normalized_origin("http://localhost:5800/api/repos/alice/example").unwrap(),
        "http://localhost:5800"
    );
}

#[test]
fn test_normalized_origin_omits_default_ports_and_formats_ipv6() {
    assert_eq!(
        normalized_origin("https://GenHub.Bio:443/api/repos/alice/example").unwrap(),
        "https://genhub.bio"
    );
    assert_eq!(
        normalized_origin("http://[::1]:5800/api/repos/alice/example").unwrap(),
        "http://[::1]:5800"
    );
}

#[test]
fn test_repository_remote_normalizes_page_and_api_urls() {
    let page = RepositoryRemote::parse("https://genhub.bio/repos/alice/example").unwrap();
    let api = RepositoryRemote::parse("https://genhub.bio/api/repos/alice/example").unwrap();

    assert_eq!(page, api);
    assert_eq!(
        page.canonical_url(),
        "https://genhub.bio/api/repos/alice/example"
    );
    assert_eq!(page.slug(), "example");
}

#[test]
fn test_repository_remote_rejects_non_repository_paths() {
    for invalid in [
        "https://genhub.bio/api/repos/alice/example/settings",
        "https://genhub.bio/api/users/alice/example",
        "https://genhub.bio/api/repos/alice/example?token=secret",
        "file:///tmp/example",
    ] {
        assert!(
            RepositoryRemote::parse(invalid).is_err(),
            "{invalid} should not parse as a canonical repository"
        );
    }
}

#[test]
fn test_public_read_uses_anonymous_capability_and_parses_response() {
    let expected_url = "http://127.0.0.1:9000/dolt/capability/default.db";
    let (origin, server) = mock_server(vec![(200, capability_body(expected_url))]);
    let repository = repository(&origin);
    let store = MemoryTokenStore::empty();
    let response = acquire_capability_with_store(
        &Client::new(),
        &repository,
        &CapabilityRequest {
            operation: RemoteOperation::Clone,
            branch: None,
            force: false,
        },
        Some("api-key-that-should-not-be-used"),
        &store,
        no_interactive_login,
    )
    .expect("public clone should mint anonymously");
    let requests = server.join().expect("mock GenHub should finish");

    assert_eq!(
        response,
        CapabilityResponse {
            remote_url: Some(expected_url.to_string()),
            expires_at: "2030-01-01T00:00:00Z"
                .parse()
                .expect("should parse capability expiry"),
            default_branch: "main".to_string(),
            transfer_id: TEST_TRANSFER_ID,
            direct_push: None,
        }
    );
    assert!(!requests[0].to_ascii_lowercase().contains("x-api-key:"));
    assert!(!requests[0].to_ascii_lowercase().contains("authorization:"));
}

#[test]
fn test_push_capability_sends_idempotency_token_and_parses_direct_gcs_grant() {
    let session_scope = SessionScope {
        principal: "repository:repo-uuid".to_string(),
        target_database: "default.db".to_string(),
        operations: r#"["write","push","main",false]"#.to_string(),
    };
    let body = serde_json::json!({
            "remote_url": null,
            "expires_at": "2030-01-01T00:00:00Z",
            "default_branch": "main",
            "transfer_id": TEST_TRANSFER_ID,
            "direct_push": {
                "database_uri": "gcs://bucket/repos/alice/example/.gen/graph_db/?vfs=blockcachevfs&access_token=secret",
                "token_expires_at": "2030-01-01T00:15:00Z",
                "session_id": TEST_TRANSFER_ID,
                "session_scope": {
                    "principal": session_scope.principal.clone(),
                    "target_database": session_scope.target_database.clone(),
                    "operations": session_scope.operations.clone()
                },
                "publish_url": "https://genhub.bio/api/publish?token=publish-secret",
                "abort_url": "https://genhub.bio/api/retire?token=abort-secret"
            }
        })
        .to_string();
    let (origin, server) = mock_server(vec![(200, body)]);
    let repository = repository(&origin);
    let capability = acquire_capability_with_store_and_token(
        &Client::new(),
        &repository,
        &CapabilityRequest {
            operation: RemoteOperation::Push,
            branch: Some("main"),
            force: false,
        },
        Some(TEST_TRANSFER_ID),
        Some("test-api-key"),
        &MemoryTokenStore::empty(),
        no_interactive_login,
    )
    .expect("push should receive direct GCS capability");
    let requests = server.join().expect("mock GenHub should finish");

    assert!(
        requests[0]
            .to_ascii_lowercase()
            .contains(&format!("idempotency-token: {TEST_TRANSFER_ID}"))
    );
    assert_eq!(capability.transfer_id, TEST_TRANSFER_ID);
    assert_eq!(
            capability.direct_push,
            Some(DirectPushCapability {
                database_uri: "gcs://bucket/repos/alice/example/.gen/graph_db/?vfs=blockcachevfs&access_token=secret".to_string(),
                token_expires_at: "2030-01-01T00:15:00Z"
                    .parse()
                    .expect("should parse CAB token expiry"),
                session_id: TEST_TRANSFER_ID,
                session_scope,
                publish_url: "https://genhub.bio/api/publish?token=publish-secret".to_string(),
                abort_url: "https://genhub.bio/api/retire?token=abort-secret".to_string(),
            })
        );
    let debug = format!("{capability:?}");
    assert!(!debug.contains("legacy-transfer"));
    assert!(!debug.contains("access_token=secret"));
    assert!(!debug.contains("token=publish-secret"));
    assert!(!debug.contains("token=abort-secret"));
    assert!(debug.contains("abort_url: \"[REDACTED]\""));
}

#[test]
fn test_direct_push_abort_url_is_required() {
    let capability = serde_json::json!({
        "database_uri": "gcs://bucket/prefix?access_token=secret",
        "token_expires_at": "2050-01-01T00:15:00Z",
        "session_id": TEST_TRANSFER_ID,
        "session_scope": {
            "principal": "repository:repo-uuid",
            "target_database": "default.db",
            "operations": "push",
        },
        "publish_url": "https://genhub.bio/api/publish?token=publish-secret",
    });

    assert!(
        serde_json::from_value::<DirectPushCapability>(capability).is_err(),
        "clients must not synthesize a missing server-supplied abort URL"
    );
}

#[test]
fn test_capability_decode_error_has_safe_endpoint_context() {
    let response_body =
        r#"{"direct_push":{"database_uri":"gcs://bucket/?access_token=sensitive-canary""#;
    let (origin, server) = mock_server(vec![(200, response_body.to_string())]);
    let repository = repository(&origin);
    let error = acquire_capability_with_store_and_token(
        &Client::new(),
        &repository,
        &CapabilityRequest {
            operation: RemoteOperation::Push,
            branch: Some("main"),
            force: false,
        },
        Some(TEST_TRANSFER_ID),
        Some("test-api-key"),
        &MemoryTokenStore::empty(),
        no_interactive_login,
    )
    .expect_err("malformed capability JSON should fail");
    server.join().expect("mock GenHub should finish");

    assert_response_decode_context(&error, "remote capability", response_body);
}

#[test]
fn test_asset_transfer_decode_error_has_safe_endpoint_context() {
    let response_body = r#"{"assets":[{"url":"https://storage.example/?token=sensitive-canary""#;
    let (origin, server) = mock_server(vec![(200, response_body.to_string())]);
    let repository = repository(&origin);
    let request = AssetTransferRequest {
        operation: RemoteOperation::Push,
        branch: "main",
        from_commit: None,
        to_commit: None,
    };
    let error = send_asset_transfers(
        &Client::new(),
        &repository,
        &request,
        RequestAuthorization::ApiKey("test-api-key"),
    )
    .expect_err("malformed asset transfer JSON should fail");
    server.join().expect("mock GenHub should finish");

    assert_response_decode_context(&error, "asset transfers", response_body);
}
