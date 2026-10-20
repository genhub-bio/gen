//! Origin-scoped credentials shared by GenHub repository remotes.
//!
//! Capability and asset endpoint helpers call this module so anonymous public reads, API keys,
//! saved CLI credentials, refresh-token rotation, and interactive login follow one sequence for
//! every repository hosted at the same GenHub origin.

use std::io;

use base64::{
    Engine as _,
    engine::general_purpose::{URL_SAFE, URL_SAFE_NO_PAD},
};
use chrono::{DateTime, Utc};
use reqwest::{
    StatusCode,
    blocking::{Client, RequestBuilder},
};
use serde::Deserialize;

use super::{RemoteClientError, RepositoryRemote, response_error};
use crate::commands::remote::{
    server::AuthTokens,
    utils::{load_tokens, save_tokens},
};

#[derive(Debug, Deserialize)]
struct RefreshResponse {
    access_token: String,
    refresh_token: String,
}

/// Distinguishes an explicit OAuth `invalid_grant` response, such as an expired or revoked refresh
/// token, from transient or unrelated refresh failures that should remain visible to the caller.
fn is_invalid_refresh_grant(error: &RemoteClientError) -> bool {
    let RemoteClientError::Http { status, message } = error else {
        return false;
    };
    if *status != StatusCode::BAD_REQUEST {
        return false;
    }
    serde_json::from_str::<serde_json::Value>(message)
        .ok()
        .is_some_and(|body| {
            body.get("error").and_then(serde_json::Value::as_str) == Some("invalid_grant")
        })
}

#[derive(Clone, Copy)]
pub(super) enum RequestAuthorization<'credential> {
    Anonymous,
    ApiKey(&'credential str),
    Bearer(&'credential str),
}

pub(super) fn authorize_request(
    builder: RequestBuilder,
    authorization: RequestAuthorization<'_>,
) -> RequestBuilder {
    match authorization {
        RequestAuthorization::Anonymous => builder,
        RequestAuthorization::ApiKey(api_key) => builder.header("x-api-key", api_key),
        RequestAuthorization::Bearer(token) => builder.bearer_auth(token),
    }
}
/// Request authentication calls this when the JWT expiry hint has elapsed or GenHub rejects the
/// saved access token; the rotated credentials are then stored under the shared origin identity.
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

/// Avoids sending an obviously expired JWT; GenHub remains responsible for validating every token.
fn access_token_expired_hint(token: &str, now: DateTime<Utc>) -> bool {
    let mut segments = token.split('.');
    let (Some(header_segment), Some(payload_segment), Some(signature_segment), None) = (
        segments.next(),
        segments.next(),
        segments.next(),
        segments.next(),
    ) else {
        return false;
    };
    if header_segment.is_empty() || payload_segment.is_empty() || signature_segment.is_empty() {
        return false;
    }

    let decode_segment = |segment: &str| {
        URL_SAFE_NO_PAD
            .decode(segment)
            .or_else(|_| URL_SAFE.decode(segment))
            .ok()
    };
    let Some(header_bytes) = decode_segment(header_segment) else {
        return false;
    };
    let Ok(header) = serde_json::from_slice::<serde_json::Value>(&header_bytes) else {
        return false;
    };
    if !header
        .get("alg")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|algorithm| !algorithm.is_empty() && algorithm != "none")
    {
        return false;
    }
    if decode_segment(signature_segment).is_none() {
        return false;
    }

    let Some(payload_bytes) = decode_segment(payload_segment) else {
        return false;
    };
    let Ok(claims) = serde_json::from_slice::<serde_json::Value>(&payload_bytes) else {
        return false;
    };
    let Some(expiration) = claims.get("exp").and_then(serde_json::Value::as_f64) else {
        return false;
    };

    expiration <= now.timestamp() as f64
}

/// Persists credentials by normalized GenHub origin so repository remotes share token rotation.
pub(super) trait TokenStore {
    fn load(&self, identity: &str) -> io::Result<AuthTokens>;
    fn save(&self, identity: &str, tokens: &AuthTokens) -> io::Result<()>;
}

/// Connects request authentication to the CLI's existing origin-keyed credential files.
pub(super) struct FileTokenStore;

impl TokenStore for FileTokenStore {
    fn load(&self, identity: &str) -> io::Result<AuthTokens> {
        load_tokens(identity)
    }

    fn save(&self, identity: &str, tokens: &AuthTokens) -> io::Result<()> {
        save_tokens(identity, tokens)
    }
}

/// Request authentication uses this when credentials are missing, rejected, or no longer
/// refreshable, then saves the new session for other remotes on the same origin.
fn interactive_login_tokens<Store, Login>(
    repository: &RepositoryRemote,
    token_store: &Store,
    interactive_login: Login,
) -> Result<AuthTokens, RemoteClientError>
where
    Store: TokenStore,
    Login: FnOnce(&str) -> Result<AuthTokens, Box<dyn std::error::Error>>,
{
    let tokens = interactive_login(repository.origin())
        .map_err(|_| RemoteClientError::AuthenticationRequired)?;
    token_store.save(repository.origin(), &tokens)?;
    Ok(tokens)
}

pub(super) struct AuthenticationOptions<'credential, Store> {
    pub(super) api_key: Option<&'credential str>,
    pub(super) allow_anonymous: bool,
    pub(super) token_store: &'credential Store,
}

/// Gives each capability or asset request one consistent anonymous, API-key, refresh, and login path.
///
/// Same-origin remotes reuse the latest stored credentials. An explicit `invalid_grant` starts a
/// new login, while transport and unrelated refresh failures remain visible to the caller.
pub(super) fn acquire_request_with_store<T, Store: TokenStore>(
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
        let tokens = if access_token_expired_hint(&tokens.jwt, Utc::now()) {
            match refresh_tokens(client, repository, &tokens) {
                Ok(refreshed) => {
                    options.token_store.save(repository.origin(), &refreshed)?;
                    refreshed
                }
                Err(error) if is_invalid_refresh_grant(&error) => {
                    let login_tokens = interactive_login_tokens(
                        repository,
                        options.token_store,
                        interactive_login,
                    )?;
                    return send(RequestAuthorization::Bearer(&login_tokens.jwt));
                }
                Err(error) => return Err(error),
            }
        } else {
            tokens
        };
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
                let refreshed = match refresh_tokens(client, repository, &tokens) {
                    Ok(refreshed) => {
                        options.token_store.save(repository.origin(), &refreshed)?;
                        refreshed
                    }
                    Err(error) if is_invalid_refresh_grant(&error) => {
                        let login_tokens = interactive_login_tokens(
                            repository,
                            options.token_store,
                            interactive_login,
                        )?;
                        return send(RequestAuthorization::Bearer(&login_tokens.jwt));
                    }
                    Err(error) => return Err(error),
                };
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
    let tokens = interactive_login_tokens(repository, options.token_store, interactive_login)?;
    send(RequestAuthorization::Bearer(&tokens.jwt))
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use reqwest::{StatusCode, blocking::Client};

    use super::access_token_expired_hint;
    use crate::commands::remote::{
        client::{
            CapabilityRequest, RemoteClientError, RemoteOperation, RepositoryRemote,
            acquire_capability_with_store, acquire_capability_with_store_and_token,
            tests::{
                MemoryTokenStore, TEST_TRANSFER_ID, capability_body, mock_server,
                mock_server_disconnects_during_refresh, no_interactive_login, repository,
                test_jwt_with_expiration,
            },
        },
        server::AuthTokens,
    };

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
        let stored = store
            .current(repository.origin())
            .expect("login tokens should be stored");

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
        let store = MemoryTokenStore::with_tokens(
            repository.origin(),
            AuthTokens {
                jwt: "forbidden-access".to_string(),
                refresh_token: "old-refresh".to_string(),
            },
        );
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
        let stored = store
            .current(repository.origin())
            .expect("login tokens should be stored");

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
    fn test_same_origin_remotes_share_rotated_tokens() {
        let (origin, server) = mock_server(vec![
            (401, "{\"message\":\"expired\"}".to_string()),
            (
                200,
                serde_json::json!({
                    "access_token": "rotated-access",
                    "refresh_token": "rotated-refresh"
                })
                .to_string(),
            ),
            (
                200,
                capability_body("http://127.0.0.1:9000/dolt/first/default.db"),
            ),
            (
                200,
                capability_body("http://127.0.0.1:9000/dolt/second/default.db"),
            ),
        ]);
        let first_remote = repository(&origin);
        let second_remote = RepositoryRemote::parse(&format!("{origin}/api/repos/bob/another"))
            .expect("should parse second repository on the same GenHub origin");
        let store = MemoryTokenStore::with_tokens(
            first_remote.origin(),
            AuthTokens {
                jwt: "expired-access".to_string(),
                refresh_token: "old-refresh".to_string(),
            },
        );
        let request = CapabilityRequest {
            operation: RemoteOperation::Push,
            branch: Some("main"),
            force: false,
        };

        acquire_capability_with_store(
            &Client::new(),
            &first_remote,
            &request,
            None,
            &store,
            no_interactive_login,
        )
        .expect("first remote should rotate its shared origin tokens");
        acquire_capability_with_store(
            &Client::new(),
            &second_remote,
            &request,
            None,
            &store,
            no_interactive_login,
        )
        .expect("second remote should reuse the rotated origin tokens");
        let requests = server.join().expect("mock GenHub should finish");
        let stored = store
            .current(first_remote.origin())
            .expect("rotated tokens should be stored for the GenHub origin");

        assert_eq!(requests.len(), 4);
        assert!(requests[0].contains("authorization: Bearer expired-access"));
        assert!(requests[1].starts_with("POST /api/auth/cli/token-refresh "));
        assert!(requests[2].contains("authorization: Bearer rotated-access"));
        assert!(requests[3].starts_with("POST /api/repos/bob/another/remote-capability "));
        assert!(requests[3].contains("authorization: Bearer rotated-access"));
        assert_eq!(stored.jwt, "rotated-access");
        assert_eq!(stored.refresh_token, "rotated-refresh");
    }

    #[test]
    fn test_tokens_for_another_origin_are_not_reused() {
        let first_origin = "https://other.genhub.test";
        let (second_origin, server) = mock_server(vec![(
            200,
            capability_body("http://127.0.0.1:9000/dolt/second-origin/default.db"),
        )]);
        assert_ne!(first_origin, second_origin);
        let second_remote = repository(&second_origin);
        let store = MemoryTokenStore::with_tokens(
            first_origin,
            AuthTokens {
                jwt: "first-origin-access".to_string(),
                refresh_token: "first-origin-refresh".to_string(),
            },
        );
        let mut login_attempts = 0;

        acquire_capability_with_store(
            &Client::new(),
            &second_remote,
            &CapabilityRequest {
                operation: RemoteOperation::Push,
                branch: Some("main"),
                force: false,
            },
            None,
            &store,
            |login_origin| {
                login_attempts += 1;
                assert_eq!(login_origin, second_remote.origin());
                Ok(AuthTokens {
                    jwt: "second-origin-access".to_string(),
                    refresh_token: "second-origin-refresh".to_string(),
                })
            },
        )
        .expect("a different origin should use its own login");
        let requests = server.join().expect("mock GenHub should finish");

        assert_eq!(login_attempts, 1);
        assert_eq!(requests.len(), 1);
        assert!(requests[0].contains("authorization: Bearer second-origin-access"));
        assert!(!requests[0].contains("first-origin-access"));
        assert_eq!(
            store
                .current(first_origin)
                .expect("first origin tokens should remain stored")
                .jwt,
            "first-origin-access"
        );
        assert_eq!(
            store
                .current(second_remote.origin())
                .expect("second origin tokens should be stored")
                .jwt,
            "second-origin-access"
        );
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
        let store = MemoryTokenStore::with_tokens(
            repository.origin(),
            AuthTokens {
                jwt: "old-access".to_string(),
                refresh_token: "old-refresh".to_string(),
            },
        );
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
        let stored = store
            .current(repository.origin())
            .expect("rotated tokens should be stored");

        assert!(requests[0].contains("authorization: Bearer old-access"));
        assert!(requests[1].starts_with("POST /api/auth/cli/token-refresh "));
        assert!(requests[1].contains("\"refresh_token\":\"old-refresh\""));
        assert!(requests[2].contains("authorization: Bearer new-access"));
        assert_eq!(stored.jwt, "new-access");
        assert_eq!(stored.refresh_token, "new-refresh");
    }

    #[test]
    fn test_expired_jwt_refreshes_before_remote_capability_request() {
        let (origin, server) = mock_server(vec![
            (
                200,
                serde_json::json!({
                    "access_token": "refreshed-access",
                    "refresh_token": "refreshed-token"
                })
                .to_string(),
            ),
            (
                200,
                capability_body("http://127.0.0.1:9000/dolt/write/default.db"),
            ),
        ]);
        let repository = repository(&origin);
        let store = MemoryTokenStore::with_tokens(
            repository.origin(),
            AuthTokens {
                jwt: test_jwt_with_expiration(Utc::now().timestamp() - 60),
                refresh_token: "old-refresh".to_string(),
            },
        );

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
        .expect("expired JWT should refresh before requesting a capability");
        let requests = server.join().expect("mock GenHub should finish");
        let stored = store
            .current(repository.origin())
            .expect("refreshed tokens should be stored");

        assert_eq!(requests.len(), 2);
        assert!(requests[0].starts_with("POST /api/auth/cli/token-refresh "));
        assert!(requests[0].contains("\"refresh_token\":\"old-refresh\""));
        assert!(requests[1].contains("authorization: Bearer refreshed-access"));
        assert_eq!(stored.jwt, "refreshed-access");
        assert_eq!(stored.refresh_token, "refreshed-token");
    }

    #[test]
    fn test_access_token_expiry_hint_ignores_malformed_and_opaque_tokens() {
        let now = Utc::now();
        for token in [
            "opaque-access-token",
            "header.payload.signature",
            "e30.bm90LWpzb24.c2lnbmF0dXJl",
            "e30.eyJleHAiOjF9.c2lnbmF0dXJl",
        ] {
            assert!(
                !access_token_expired_hint(token, now),
                "malformed or opaque access token should not trigger proactive refresh"
            );
        }
        assert!(access_token_expired_hint(
            &test_jwt_with_expiration(now.timestamp() - 1),
            now
        ));
        assert!(!access_token_expired_hint(
            &test_jwt_with_expiration(now.timestamp() + 60),
            now
        ));
    }

    #[test]
    fn test_unexpired_jwt_forbidden_response_keeps_interactive_login_behavior() {
        let (origin, server) = mock_server(vec![
            (403, "{\"message\":\"permission denied\"}".to_string()),
            (
                200,
                capability_body("http://127.0.0.1:9000/dolt/write/default.db"),
            ),
        ]);
        let repository = repository(&origin);
        let token = test_jwt_with_expiration(Utc::now().timestamp() + 3600);
        let store = MemoryTokenStore::with_tokens(
            repository.origin(),
            AuthTokens {
                jwt: token.clone(),
                refresh_token: "old-refresh".to_string(),
            },
        );
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
        .expect("valid-token authorization rejection should keep existing login behavior");
        let requests = server.join().expect("mock GenHub should finish");

        assert!(login_attempted);
        assert_eq!(requests.len(), 2);
        assert!(requests[0].contains(&format!("authorization: Bearer {token}")));
        assert!(!requests[0].contains("token-refresh"));
        assert!(requests[1].contains("authorization: Bearer login-access"));
    }

    #[test]
    fn test_invalid_refresh_grant_logs_in_and_retries_with_same_idempotency_token() {
        let (origin, server) = mock_server(vec![
            (401, "{\"message\":\"expired\"}".to_string()),
            (
                400,
                "{\"error\":\"invalid_grant\",\"error_description\":\"Invalid or expired refresh token\",\"error_uri\":null}".to_string(),
            ),
            (
                200,
                capability_body("http://127.0.0.1:9000/dolt/login/default.db"),
            ),
        ]);
        let repository = repository(&origin);
        let store = MemoryTokenStore::with_tokens(
            repository.origin(),
            AuthTokens {
                jwt: "expired-access".to_string(),
                refresh_token: "invalid-refresh".to_string(),
            },
        );
        let request = CapabilityRequest {
            operation: RemoteOperation::Push,
            branch: Some("main"),
            force: false,
        };
        let mut login_attempts = 0;

        acquire_capability_with_store_and_token(
            &Client::new(),
            &repository,
            &request,
            Some(TEST_TRANSFER_ID),
            None,
            &store,
            |login_origin| {
                login_attempts += 1;
                assert_eq!(login_origin, repository.origin());
                Ok(AuthTokens {
                    jwt: "login-access".to_string(),
                    refresh_token: "login-refresh".to_string(),
                })
            },
        )
        .expect("invalid refresh grant should recover through interactive login");
        let requests = server.join().expect("mock GenHub should finish");
        let stored = store
            .current(repository.origin())
            .expect("login tokens should be stored for the origin");
        let expected_idempotency_header = format!("idempotency-token: {TEST_TRANSFER_ID}");

        assert_eq!(login_attempts, 1);
        assert_eq!(requests.len(), 3);
        assert!(requests[0].contains("authorization: Bearer expired-access"));
        assert!(
            requests[0]
                .to_ascii_lowercase()
                .contains(&expected_idempotency_header)
        );
        assert!(requests[1].starts_with("POST /api/auth/cli/token-refresh "));
        assert!(requests[2].contains("authorization: Bearer login-access"));
        assert!(
            requests[2]
                .to_ascii_lowercase()
                .contains(&expected_idempotency_header)
        );
        assert_eq!(stored.jwt, "login-access");
        assert_eq!(stored.refresh_token, "login-refresh");
    }

    #[test]
    fn test_expired_jwt_with_invalid_refresh_grant_logs_in_before_capability_request() {
        let (origin, server) = mock_server(vec![
            (
                400,
                "{\"error\":\"invalid_grant\",\"error_description\":\"Invalid or expired refresh token\",\"error_uri\":null}".to_string(),
            ),
            (
                200,
                capability_body("http://127.0.0.1:9000/dolt/login/default.db"),
            ),
        ]);
        let repository = repository(&origin);
        let store = MemoryTokenStore::with_tokens(
            repository.origin(),
            AuthTokens {
                jwt: test_jwt_with_expiration(Utc::now().timestamp() - 60),
                refresh_token: "invalid-refresh".to_string(),
            },
        );
        let mut login_attempts = 0;

        acquire_capability_with_store_and_token(
            &Client::new(),
            &repository,
            &CapabilityRequest {
                operation: RemoteOperation::Push,
                branch: Some("main"),
                force: false,
            },
            Some(TEST_TRANSFER_ID),
            None,
            &store,
            |login_origin| {
                login_attempts += 1;
                assert_eq!(login_origin, repository.origin());
                Ok(AuthTokens {
                    jwt: "login-access".to_string(),
                    refresh_token: "login-refresh".to_string(),
                })
            },
        )
        .expect("expired JWT and invalid refresh grant should recover through login");
        let requests = server.join().expect("mock GenHub should finish");

        assert_eq!(login_attempts, 1);
        assert_eq!(requests.len(), 2);
        assert!(requests[0].starts_with("POST /api/auth/cli/token-refresh "));
        assert!(requests[1].contains("authorization: Bearer login-access"));
        assert!(
            requests[1]
                .to_ascii_lowercase()
                .contains(&format!("idempotency-token: {TEST_TRANSFER_ID}"))
        );
    }

    #[test]
    fn test_invalid_refresh_grant_with_failed_login_preserves_existing_tokens() {
        let (origin, server) = mock_server(vec![
            (401, "{\"message\":\"expired\"}".to_string()),
            (400, "{\"error\":\"invalid_grant\"}".to_string()),
        ]);
        let repository = repository(&origin);
        let original_tokens = AuthTokens {
            jwt: "expired-access".to_string(),
            refresh_token: "invalid-refresh".to_string(),
        };
        let store = MemoryTokenStore::with_tokens(repository.origin(), original_tokens.clone());
        let mut login_attempts = 0;
        let error = acquire_capability_with_store_and_token(
            &Client::new(),
            &repository,
            &CapabilityRequest {
                operation: RemoteOperation::Push,
                branch: Some("main"),
                force: false,
            },
            Some(TEST_TRANSFER_ID),
            None,
            &store,
            |_| {
                login_attempts += 1;
                Err("login did not complete".into())
            },
        )
        .expect_err("failed interactive login should report authentication required");
        let requests = server.join().expect("mock GenHub should finish");

        assert!(matches!(error, RemoteClientError::AuthenticationRequired));
        assert_eq!(requests.len(), 2);
        assert_eq!(login_attempts, 1);
        let stored = store
            .current(repository.origin())
            .expect("old tokens should remain stored");
        assert_eq!(stored.jwt, original_tokens.jwt);
        assert_eq!(stored.refresh_token, original_tokens.refresh_token);
    }

    #[test]
    fn test_failed_request_after_invalid_refresh_login_is_returned_once() {
        let (origin, server) = mock_server(vec![
            (401, "{\"message\":\"expired\"}".to_string()),
            (400, "{\"error\":\"invalid_grant\"}".to_string()),
            (401, "{\"message\":\"still unauthorized\"}".to_string()),
        ]);
        let repository = repository(&origin);
        let store = MemoryTokenStore::with_tokens(
            repository.origin(),
            AuthTokens {
                jwt: "expired-access".to_string(),
                refresh_token: "invalid-refresh".to_string(),
            },
        );
        let mut login_attempts = 0;
        let error = acquire_capability_with_store_and_token(
            &Client::new(),
            &repository,
            &CapabilityRequest {
                operation: RemoteOperation::Push,
                branch: Some("main"),
                force: false,
            },
            Some(TEST_TRANSFER_ID),
            None,
            &store,
            |_| {
                login_attempts += 1;
                Ok(AuthTokens {
                    jwt: "login-access".to_string(),
                    refresh_token: "login-refresh".to_string(),
                })
            },
        )
        .expect_err("the single post-login retry error should be returned");
        let requests = server.join().expect("mock GenHub should finish");

        assert!(
            matches!(error, RemoteClientError::Http { status, .. } if status == StatusCode::UNAUTHORIZED)
        );
        assert_eq!(requests.len(), 3);
        assert_eq!(login_attempts, 1);
        assert!(requests[2].contains("authorization: Bearer login-access"));
        assert!(
            requests[2]
                .to_ascii_lowercase()
                .contains(&format!("idempotency-token: {TEST_TRANSFER_ID}"))
        );
    }

    #[test]
    fn test_refresh_transport_failure_propagates_without_logging_in() {
        let (origin, server) = mock_server_disconnects_during_refresh();
        let repository = repository(&origin);
        let original_tokens = AuthTokens {
            jwt: "expired-access".to_string(),
            refresh_token: "refresh-token".to_string(),
        };
        let store = MemoryTokenStore::with_tokens(repository.origin(), original_tokens.clone());
        let mut login_attempts = 0;
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
            |_| {
                login_attempts += 1;
                Ok(AuthTokens {
                    jwt: "unexpected-login".to_string(),
                    refresh_token: "unexpected-refresh".to_string(),
                })
            },
        )
        .expect_err("refresh transport failure should be returned");
        let requests = server.join().expect("mock GenHub should finish");
        let stored = store
            .current(repository.origin())
            .expect("original tokens should remain stored");

        assert!(matches!(error, RemoteClientError::Request(_)));
        assert_eq!(requests.len(), 2);
        assert_eq!(login_attempts, 0);
        assert_eq!(stored.jwt, original_tokens.jwt);
        assert_eq!(stored.refresh_token, original_tokens.refresh_token);
    }

    #[test]
    fn test_other_refresh_errors_propagate_without_logging_in() {
        for (status, body) in [
            (400, "{\"error\":\"invalid_request\"}"),
            (400, "not an OAuth error response"),
            (500, "{\"error\":\"invalid_grant\"}"),
            (500, "{\"error\":\"temporarily_unavailable\"}"),
        ] {
            let (origin, server) = mock_server(vec![
                (401, "{\"message\":\"expired\"}".to_string()),
                (status, body.to_string()),
            ]);
            let repository = repository(&origin);
            let store = MemoryTokenStore::with_tokens(
                repository.origin(),
                AuthTokens {
                    jwt: "expired-access".to_string(),
                    refresh_token: "refresh-token".to_string(),
                },
            );
            let mut login_attempts = 0;
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
                |_| {
                    login_attempts += 1;
                    Ok(AuthTokens {
                        jwt: "unexpected-login".to_string(),
                        refresh_token: "unexpected-refresh".to_string(),
                    })
                },
            )
            .expect_err("only invalid_grant should start interactive login");
            let requests = server.join().expect("mock GenHub should finish");
            let stored = store
                .current(repository.origin())
                .expect("original tokens should remain stored");

            assert!(
                matches!(error, RemoteClientError::Http { status: actual, .. } if actual.as_u16() == status)
            );
            assert_eq!(requests.len(), 2);
            assert_eq!(login_attempts, 0);
            assert_eq!(stored.jwt, "expired-access");
            assert_eq!(stored.refresh_token, "refresh-token");
        }
    }
}
