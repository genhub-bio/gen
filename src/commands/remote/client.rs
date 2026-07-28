//! GenHub client for authorizing graph and asset transfers.
//!
//! Gen stores an HTTP(S) remote in the config database using its canonical GenHub
//! repository URL. [`RepositoryRemote`] parses either a repository page URL or API URL
//! into that canonical identity and the GenHub endpoints associated with it. Local
//! `file://` remotes bypass this client and are handled directly by the remote operations
//! module.
//!
//! The canonical URL is a control-plane address, not the URL used by Dolt to transfer
//! commits. Before a clone, pull, or push, [`acquire_capability`] sends the operation,
//! branch, and force setting to GenHub. GenHub returns a short-lived Dolt-compatible
//! transfer URL. The remote operations module installs that URL in the graph database,
//! runs the native Dolt operation, and then restores the canonical GenHub URL. If Dolt
//! rejects an expired capability, the operation requests a fresh capability and retries.
//! After the graph transfer, [`acquire_asset_transfers`] obtains the per-asset upload or
//! download URLs used to transfer files referenced by the selected branch.
//!
//! Both acquisition functions use the same authentication sequence. Public clone and
//! pull requests may first be attempted anonymously; otherwise the client tries
//! `GENHUB_API_KEY`, tokens stored for the normalized GenHub origin, and finally the
//! interactive login callback supplied by the command. Rejected access tokens are
//! refreshed through GenHub's CLI refresh endpoint when possible, and refreshed or
//! newly issued tokens are saved for later requests.

use std::{env, io};

use gen_core::HashId;
use reqwest::{
    StatusCode, Url,
    blocking::{Client, RequestBuilder},
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::commands::remote::{
    server::AuthTokens,
    utils::{load_tokens, save_tokens},
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RepositoryRemote {
    origin: String,
    namespace: String,
    slug: String,
    canonical_url: String,
}

impl RepositoryRemote {
    pub fn parse(remote_url: &str) -> Result<Self, RemoteClientError> {
        let parsed = Url::parse(remote_url)
            .map_err(|_| RemoteClientError::InvalidRepositoryUrl(remote_url.to_string()))?;
        if parsed.scheme() != "http" && parsed.scheme() != "https" {
            return Err(RemoteClientError::InvalidRepositoryUrl(
                remote_url.to_string(),
            ));
        }
        let origin = normalized_origin(remote_url)?;
        if parsed.query().is_some() || parsed.fragment().is_some() {
            return Err(RemoteClientError::InvalidRepositoryUrl(
                remote_url.to_string(),
            ));
        }
        let segments = parsed
            .path_segments()
            .ok_or_else(|| RemoteClientError::InvalidRepositoryUrl(remote_url.to_string()))?
            .filter(|segment| !segment.is_empty())
            .collect::<Vec<_>>();
        let repository_segments = match segments.as_slice() {
            ["repos", namespace, slug] | ["api", "repos", namespace, slug] => (*namespace, *slug),
            _ => {
                return Err(RemoteClientError::InvalidRepositoryUrl(
                    remote_url.to_string(),
                ));
            }
        };
        let namespace = repository_segments.0.to_string();
        let slug = repository_segments.1.to_string();
        if namespace.is_empty() || slug.is_empty() {
            return Err(RemoteClientError::InvalidRepositoryUrl(
                remote_url.to_string(),
            ));
        }
        let canonical_url = format!("{origin}/api/repos/{namespace}/{slug}");
        Ok(Self {
            origin,
            namespace,
            slug,
            canonical_url,
        })
    }

    pub fn origin(&self) -> &str {
        &self.origin
    }

    pub fn slug(&self) -> &str {
        &self.slug
    }

    pub fn canonical_url(&self) -> &str {
        &self.canonical_url
    }

    fn capability_url(&self) -> String {
        format!(
            "{}/api/repos/{}/{}/remote-capability",
            self.origin, self.namespace, self.slug
        )
    }

    fn asset_transfers_url(&self) -> String {
        format!(
            "{}/api/repos/{}/{}/asset-transfers",
            self.origin, self.namespace, self.slug
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum RemoteOperation {
    Clone,
    Pull,
    Push,
}

#[derive(Clone, Debug, Serialize)]
pub struct CapabilityRequest<'branch> {
    pub operation: RemoteOperation,
    pub branch: Option<&'branch str>,
    pub force: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct CapabilityResponse {
    pub remote_url: String,
    pub expires_at: String,
    pub default_branch: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct AssetTransferRequest<'branch> {
    pub operation: RemoteOperation,
    pub branch: &'branch str,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct AssetTransfer {
    pub id: HashId,
    pub url: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct AssetTransferResponse {
    pub assets: Vec<AssetTransfer>,
}

#[derive(Debug, Deserialize)]
struct RefreshResponse {
    access_token: String,
    refresh_token: String,
}

#[derive(Debug, Error)]
pub enum RemoteClientError {
    #[error("Invalid GenHub repository URL: {0}")]
    InvalidRepositoryUrl(String),
    #[error("Authentication is required; run `gen remote login` or set GENHUB_API_KEY")]
    AuthenticationRequired,
    #[error("GenHub request failed with HTTP {status}: {message}")]
    Http { status: StatusCode, message: String },
    #[error("HTTP client error: {0}")]
    Request(#[from] reqwest::Error),
    #[error("Token storage error: {0}")]
    TokenStorage(#[from] std::io::Error),
}

pub fn normalized_origin(remote_url: &str) -> Result<String, RemoteClientError> {
    let parsed = Url::parse(remote_url)
        .map_err(|_| RemoteClientError::InvalidRepositoryUrl(remote_url.to_string()))?;
    if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
        return Err(RemoteClientError::InvalidRepositoryUrl(
            remote_url.to_string(),
        ));
    }
    Ok(parsed.origin().ascii_serialization())
}

fn response_error(response: reqwest::blocking::Response) -> RemoteClientError {
    let status = response.status();
    let message = response
        .text()
        .unwrap_or_else(|_| "Unable to read response".to_string());
    RemoteClientError::Http { status, message }
}

#[derive(Clone, Copy)]
enum RequestAuthorization<'credential> {
    Anonymous,
    ApiKey(&'credential str),
    Bearer(&'credential str),
}

fn authorize_request(
    builder: RequestBuilder,
    authorization: RequestAuthorization<'_>,
) -> RequestBuilder {
    match authorization {
        RequestAuthorization::Anonymous => builder,
        RequestAuthorization::ApiKey(api_key) => builder.header("x-api-key", api_key),
        RequestAuthorization::Bearer(token) => builder.bearer_auth(token),
    }
}

fn send_capability(
    client: &Client,
    repository: &RepositoryRemote,
    request: &CapabilityRequest<'_>,
    authorization: RequestAuthorization<'_>,
) -> Result<CapabilityResponse, RemoteClientError> {
    let response = authorize_request(
        client.post(repository.capability_url()).json(request),
        authorization,
    )
    .send()?;
    if !response.status().is_success() {
        return Err(response_error(response));
    }
    Ok(response.json()?)
}

fn send_asset_transfers(
    client: &Client,
    repository: &RepositoryRemote,
    request: &AssetTransferRequest<'_>,
    authorization: RequestAuthorization<'_>,
) -> Result<AssetTransferResponse, RemoteClientError> {
    let response = authorize_request(
        client.post(repository.asset_transfers_url()).json(request),
        authorization,
    )
    .send()?;
    if !response.status().is_success() {
        return Err(response_error(response));
    }
    Ok(response.json()?)
}

fn refresh_tokens(
    client: &Client,
    repository: &RepositoryRemote,
    tokens: &AuthTokens,
) -> Result<AuthTokens, RemoteClientError> {
    let response = client
        .post(format!(
            "{}/api/auth/cli/token-refresh",
            repository.origin()
        ))
        .json(&serde_json::json!({
            "refresh_token": tokens.refresh_token,
            "client_id": "cli"
        }))
        .send()?;
    if !response.status().is_success() {
        return Err(response_error(response));
    }
    let refreshed: RefreshResponse = response.json()?;
    Ok(AuthTokens {
        jwt: refreshed.access_token,
        refresh_token: refreshed.refresh_token,
    })
}

trait TokenStore {
    fn load(&self, identity: &str) -> io::Result<AuthTokens>;
    fn save(&self, identity: &str, tokens: &AuthTokens) -> io::Result<()>;
}

struct FileTokenStore;

impl TokenStore for FileTokenStore {
    fn load(&self, identity: &str) -> io::Result<AuthTokens> {
        load_tokens(identity)
    }

    fn save(&self, identity: &str, tokens: &AuthTokens) -> io::Result<()> {
        save_tokens(identity, tokens)
    }
}

struct AuthenticationOptions<'credential, Store> {
    api_key: Option<&'credential str>,
    allow_anonymous: bool,
    token_store: &'credential Store,
}

fn acquire_request_with_store<T, Store: TokenStore>(
    client: &Client,
    repository: &RepositoryRemote,
    options: AuthenticationOptions<'_, Store>,
    interactive_login: impl FnOnce(&str) -> Result<AuthTokens, Box<dyn std::error::Error>>,
    mut send: impl for<'credential> FnMut(
        RequestAuthorization<'credential>,
    ) -> Result<T, RemoteClientError>,
) -> Result<T, RemoteClientError> {
    if options.allow_anonymous {
        match send(RequestAuthorization::Anonymous) {
            Ok(response) => return Ok(response),
            Err(RemoteClientError::Http {
                status: StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN | StatusCode::NOT_FOUND,
                ..
            }) => {}
            Err(error) => return Err(error),
        }
    }
    if let Some(api_key) = options.api_key.filter(|api_key| !api_key.is_empty()) {
        match send(RequestAuthorization::ApiKey(api_key)) {
            Ok(response) => return Ok(response),
            Err(RemoteClientError::Http {
                status: StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN | StatusCode::NOT_FOUND,
                ..
            }) => {}
            Err(error) => return Err(error),
        }
    }
    if let Ok(tokens) = options.token_store.load(repository.origin()) {
        match send(RequestAuthorization::Bearer(&tokens.jwt)) {
            Ok(response) => return Ok(response),
            Err(RemoteClientError::Http {
                status: StatusCode::FORBIDDEN,
                ..
            }) => {}
            Err(RemoteClientError::Http {
                status: StatusCode::UNAUTHORIZED | StatusCode::NOT_FOUND,
                ..
            }) => {
                let refreshed = refresh_tokens(client, repository, &tokens)?;
                options.token_store.save(repository.origin(), &refreshed)?;
                match send(RequestAuthorization::Bearer(&refreshed.jwt)) {
                    Ok(response) => return Ok(response),
                    Err(RemoteClientError::Http {
                        status: StatusCode::FORBIDDEN,
                        ..
                    }) => {}
                    Err(error) => return Err(error),
                }
            }
            Err(error) => return Err(error),
        }
    }
    let tokens = interactive_login(repository.origin())
        .map_err(|_| RemoteClientError::AuthenticationRequired)?;
    options.token_store.save(repository.origin(), &tokens)?;
    send(RequestAuthorization::Bearer(&tokens.jwt))
}

fn acquire_capability_with_store(
    client: &Client,
    repository: &RepositoryRemote,
    request: &CapabilityRequest<'_>,
    api_key: Option<&str>,
    token_store: &impl TokenStore,
    interactive_login: impl FnOnce(&str) -> Result<AuthTokens, Box<dyn std::error::Error>>,
) -> Result<CapabilityResponse, RemoteClientError> {
    let allow_anonymous = matches!(
        request.operation,
        RemoteOperation::Clone | RemoteOperation::Pull
    );
    acquire_request_with_store(
        client,
        repository,
        AuthenticationOptions {
            api_key,
            allow_anonymous,
            token_store,
        },
        interactive_login,
        |authorization| send_capability(client, repository, request, authorization),
    )
}

pub fn acquire_capability(
    repository: &RepositoryRemote,
    request: &CapabilityRequest<'_>,
    interactive_login: impl FnOnce(&str) -> Result<AuthTokens, Box<dyn std::error::Error>>,
) -> Result<CapabilityResponse, RemoteClientError> {
    let client = Client::new();
    let api_key = env::var("GENHUB_API_KEY").ok();
    acquire_capability_with_store(
        &client,
        repository,
        request,
        api_key.as_deref(),
        &FileTokenStore,
        interactive_login,
    )
}

pub fn acquire_asset_transfers(
    repository: &RepositoryRemote,
    request: &AssetTransferRequest<'_>,
    interactive_login: impl FnOnce(&str) -> Result<AuthTokens, Box<dyn std::error::Error>>,
) -> Result<AssetTransferResponse, RemoteClientError> {
    let client = Client::new();
    let api_key = env::var("GENHUB_API_KEY").ok();
    let allow_anonymous = matches!(
        request.operation,
        RemoteOperation::Clone | RemoteOperation::Pull
    );
    acquire_request_with_store(
        &client,
        repository,
        AuthenticationOptions {
            api_key: api_key.as_deref(),
            allow_anonymous,
            token_store: &FileTokenStore,
        },
        interactive_login,
        |authorization| send_asset_transfers(&client, repository, request, authorization),
    )
}

#[cfg(test)]
mod tests {
    use std::{
        io::{self, Read as _, Write as _},
        net::{TcpListener, TcpStream},
        sync::Mutex,
        thread::{self, JoinHandle},
    };

    use reqwest::{StatusCode, blocking::Client};

    use super::{
        AuthTokens, CapabilityRequest, CapabilityResponse, RemoteClientError, RemoteOperation,
        RepositoryRemote, TokenStore, acquire_capability_with_store, normalized_origin,
    };

    struct MemoryTokenStore {
        tokens: Mutex<Option<AuthTokens>>,
    }

    impl MemoryTokenStore {
        fn empty() -> Self {
            Self {
                tokens: Mutex::new(None),
            }
        }

        fn with_tokens(tokens: AuthTokens) -> Self {
            Self {
                tokens: Mutex::new(Some(tokens)),
            }
        }

        fn current(&self) -> Option<AuthTokens> {
            self.tokens.lock().expect("should lock token store").clone()
        }
    }

    impl TokenStore for MemoryTokenStore {
        fn load(&self, _identity: &str) -> io::Result<AuthTokens> {
            self.current()
                .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no in-memory credentials"))
        }

        fn save(&self, _identity: &str, tokens: &AuthTokens) -> io::Result<()> {
            *self.tokens.lock().expect("should lock token store") = Some(tokens.clone());
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
                && let Some(header_end) =
                    request.windows(4).position(|window| window == b"\r\n\r\n")
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

    fn mock_server(responses: Vec<(u16, String)>) -> (String, JoinHandle<Vec<String>>) {
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

    fn capability_body(remote_url: &str) -> String {
        serde_json::json!({
            "remote_url": remote_url,
            "expires_at": "2030-01-01T00:00:00Z",
            "default_branch": "main"
        })
        .to_string()
    }

    fn repository(origin: &str) -> RepositoryRemote {
        RepositoryRemote::parse(&format!("{origin}/api/repos/alice/example"))
            .expect("should parse mock repository")
    }

    fn no_interactive_login(_origin: &str) -> Result<AuthTokens, Box<dyn std::error::Error>> {
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
                remote_url: expected_url.to_string(),
                expires_at: "2030-01-01T00:00:00Z".to_string(),
                default_branch: "main".to_string(),
            }
        );
        assert!(!requests[0].to_ascii_lowercase().contains("x-api-key:"));
        assert!(!requests[0].to_ascii_lowercase().contains("authorization:"));
    }

    #[test]
    fn test_push_uses_genhub_api_key_header() {
        let (origin, server) = mock_server(vec![(
            200,
            capability_body("http://127.0.0.1:9000/dolt/write/default.db"),
        )]);
        let repository = repository(&origin);
        acquire_capability_with_store(
            &Client::new(),
            &repository,
            &CapabilityRequest {
                operation: RemoteOperation::Push,
                branch: Some("main"),
                force: false,
            },
            Some("test-api-key"),
            &MemoryTokenStore::empty(),
            no_interactive_login,
        )
        .expect("API key should authorize push capability");
        let requests = server.join().expect("mock GenHub should finish");

        assert!(
            requests[0]
                .to_ascii_lowercase()
                .contains("x-api-key: test-api-key")
        );
    }

    #[test]
    fn test_push_logs_in_when_login_token_is_missing() {
        let (origin, server) = mock_server(vec![(
            200,
            capability_body("http://127.0.0.1:9000/dolt/write/default.db"),
        )]);
        let repository = repository(&origin);
        let store = MemoryTokenStore::empty();
        let mut login_attempted = false;

        acquire_capability_with_store(
            &Client::new(),
            &repository,
            &CapabilityRequest {
                operation: RemoteOperation::Push,
                branch: Some("main"),
                force: false,
            },
            None,
            &store,
            |login_origin| {
                login_attempted = true;
                assert_eq!(login_origin, origin);
                Ok(AuthTokens {
                    jwt: "login-access".to_string(),
                    refresh_token: "login-refresh".to_string(),
                })
            },
        )
        .expect("missing token should trigger login");
        let requests = server.join().expect("mock GenHub should finish");
        let stored = store.current().expect("login tokens should be stored");

        assert!(login_attempted);
        assert!(requests[0].contains("authorization: Bearer login-access"));
        assert_eq!(stored.jwt, "login-access");
        assert_eq!(stored.refresh_token, "login-refresh");
    }

    #[test]
    fn test_forbidden_token_triggers_login() {
        let (origin, server) = mock_server(vec![
            (403, "{\"message\":\"permission denied\"}".to_string()),
            (
                200,
                capability_body("http://127.0.0.1:9000/dolt/write/default.db"),
            ),
        ]);
        let repository = repository(&origin);
        let store = MemoryTokenStore::with_tokens(AuthTokens {
            jwt: "forbidden-access".to_string(),
            refresh_token: "old-refresh".to_string(),
        });
        let mut login_attempted = false;

        acquire_capability_with_store(
            &Client::new(),
            &repository,
            &CapabilityRequest {
                operation: RemoteOperation::Push,
                branch: Some("main"),
                force: false,
            },
            None,
            &store,
            |_| {
                login_attempted = true;
                Ok(AuthTokens {
                    jwt: "login-access".to_string(),
                    refresh_token: "login-refresh".to_string(),
                })
            },
        )
        .expect("forbidden token should trigger login");
        let requests = server.join().expect("mock GenHub should finish");
        let stored = store.current().expect("login tokens should be stored");

        assert!(login_attempted);
        assert!(requests[0].contains("authorization: Bearer forbidden-access"));
        assert!(requests[1].contains("authorization: Bearer login-access"));
        assert_eq!(stored.jwt, "login-access");
        assert_eq!(stored.refresh_token, "login-refresh");
    }

    #[test]
    fn test_private_clone_and_pull_log_in_after_anonymous_request_is_rejected() {
        for operation in [RemoteOperation::Clone, RemoteOperation::Pull] {
            let (origin, server) = mock_server(vec![
                (404, "{\"message\":\"repository not found\"}".to_string()),
                (
                    200,
                    capability_body("http://127.0.0.1:9000/dolt/read/default.db"),
                ),
            ]);
            let repository = repository(&origin);
            let store = MemoryTokenStore::empty();
            let mut login_attempted = false;

            acquire_capability_with_store(
                &Client::new(),
                &repository,
                &CapabilityRequest {
                    operation,
                    branch: (operation == RemoteOperation::Pull).then_some("main"),
                    force: false,
                },
                None,
                &store,
                |_| {
                    login_attempted = true;
                    Ok(AuthTokens {
                        jwt: "login-access".to_string(),
                        refresh_token: "login-refresh".to_string(),
                    })
                },
            )
            .expect("private read should trigger login");
            let requests = server.join().expect("mock GenHub should finish");

            assert!(login_attempted);
            assert!(!requests[0].to_ascii_lowercase().contains("authorization:"));
            assert!(requests[1].contains("authorization: Bearer login-access"));
        }
    }

    #[test]
    fn test_expired_access_token_refreshes_and_persists_rotated_tokens() {
        let (origin, server) = mock_server(vec![
            (404, "{\"message\":\"not found\"}".to_string()),
            (
                200,
                serde_json::json!({
                    "access_token": "new-access",
                    "refresh_token": "new-refresh"
                })
                .to_string(),
            ),
            (
                200,
                capability_body("http://127.0.0.1:9000/dolt/refreshed/default.db"),
            ),
        ]);
        let repository = repository(&origin);
        let store = MemoryTokenStore::with_tokens(AuthTokens {
            jwt: "old-access".to_string(),
            refresh_token: "old-refresh".to_string(),
        });
        acquire_capability_with_store(
            &Client::new(),
            &repository,
            &CapabilityRequest {
                operation: RemoteOperation::Push,
                branch: Some("main"),
                force: false,
            },
            None,
            &store,
            no_interactive_login,
        )
        .expect("expired access token should refresh");
        let requests = server.join().expect("mock GenHub should finish");
        let stored = store.current().expect("rotated tokens should be stored");

        assert!(requests[0].contains("authorization: Bearer old-access"));
        assert!(requests[1].starts_with("POST /api/auth/cli/token-refresh "));
        assert!(requests[1].contains("\"refresh_token\":\"old-refresh\""));
        assert!(requests[2].contains("authorization: Bearer new-access"));
        assert_eq!(stored.jwt, "new-access");
        assert_eq!(stored.refresh_token, "new-refresh");
    }

    #[test]
    fn test_invalid_refresh_is_reported_without_overwriting_tokens() {
        let (origin, server) = mock_server(vec![
            (401, "{\"message\":\"expired\"}".to_string()),
            (400, "{\"error\":\"invalid_grant\"}".to_string()),
        ]);
        let repository = repository(&origin);
        let store = MemoryTokenStore::with_tokens(AuthTokens {
            jwt: "expired-access".to_string(),
            refresh_token: "invalid-refresh".to_string(),
        });
        let error = acquire_capability_with_store(
            &Client::new(),
            &repository,
            &CapabilityRequest {
                operation: RemoteOperation::Push,
                branch: Some("main"),
                force: false,
            },
            None,
            &store,
            no_interactive_login,
        )
        .expect_err("invalid refresh should fail");
        server.join().expect("mock GenHub should finish");
        let stored = store.current().expect("old tokens should remain stored");

        assert!(matches!(
            error,
            RemoteClientError::Http {
                status: StatusCode::BAD_REQUEST,
                ..
            }
        ));
        assert_eq!(stored.jwt, "expired-access");
        assert_eq!(stored.refresh_token, "invalid-refresh");
    }
}
