//! GenHub capability coordination for graph transfers.
//!
//! Clone, pull, and fetch use short-lived GCS read capabilities, while push uses a persisted
//! idempotency token to identify a direct-GCS staging session. This module keeps push token renewal
//! tied to that same session: long block uploads may outlive one CAB token, but they must never
//! switch repositories, prefixes, operation scope, or publication origin while refreshing it.

use std::{
    error::Error,
    io,
    sync::{Arc, Mutex},
};

use chrono::{DateTime, Duration as ChronoDuration, Utc};
use gen_models::operations::Remote;
use rusqlite::blockcachevfs::{
    AuthError, AuthRefreshReason, CloudConnectionIdentity, CloudConnectionUri,
};
use url::Url;
use uuid::Uuid;

use super::{direct_gcs::PushGraphTransferError, file_graph_url};
use crate::commands::remote::{
    client::{
        CapabilityRequest, CapabilityResponse, DirectPushCapability, RemoteClientError,
        RemoteOperation, RepositoryRemote, SessionScope as ClientSessionScope, abort_direct_push,
        acquire_capability, acquire_push_capability, confirm_direct_push_session_stale,
        publish_direct_push,
    },
    login_origin,
    server::AuthTokens,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct PushTransferLease {
    pub(super) transfer_id: Uuid,
    pub(super) expires_at: i64,
}

pub(super) struct GraphTransferAuthorization {
    pub(super) remote_url: Option<String>,
    pub(super) push_lease: Option<PushTransferLease>,
    pub(super) direct_push: Option<DirectPushCapability>,
    pub(super) capability_response: Option<CapabilityResponse>,
}

const PUSH_AUTHORIZATION_REFRESH_MARGIN_SECONDS: i64 = 60;

type PushCapabilityFetcher =
    dyn Fn() -> Result<CapabilityResponse, RemoteClientError> + Send + Sync + 'static;

struct PushCapabilityState {
    response: CapabilityResponse,
    issued_at: DateTime<Utc>,
}

enum CapabilityRefreshFailure {
    Fetch(RemoteClientError),
    Expired,
    InvalidScope,
}

#[derive(Clone)]
pub(super) struct PushCapabilityRenewal {
    expected_repository_origin: String,
    expected_scope: ClientSessionScope,
    pub(super) expected_session_id: Uuid,
    expected_transfer_id: Uuid,
    expected_uri_identity: CloudConnectionIdentity,
    fetch: Arc<PushCapabilityFetcher>,
    state: Arc<Mutex<PushCapabilityState>>,
}

impl PushCapabilityRenewal {
    fn new(
        response: CapabilityResponse,
        expected_transfer_id: Uuid,
        expected_session_id: Uuid,
        repository_origin: &str,
        fetch: Arc<PushCapabilityFetcher>,
    ) -> Result<Self, PushGraphTransferError> {
        if response.transfer_id != expected_transfer_id || response.remote_url.is_some() {
            return Err(PushGraphTransferError::Protocol(
                "GenHub returned a direct-push capability for a different transfer".to_string(),
            ));
        }
        let direct_push = response.direct_push.as_ref().ok_or_else(|| {
            PushGraphTransferError::Protocol(
                "GenHub push capability did not include a direct GCS session".to_string(),
            )
        })?;
        if direct_push.session_id != expected_session_id {
            return Err(PushGraphTransferError::Protocol(
                "GenHub returned a direct-push session that does not match the idempotency token"
                    .to_string(),
            ));
        }
        let connection_uri =
            CloudConnectionUri::parse(&direct_push.database_uri).map_err(|_| {
                PushGraphTransferError::Protocol(
                    "GenHub returned an invalid direct GCS database URI".to_string(),
                )
            })?;
        if connection_uri.gcs_access_token().is_none() {
            return Err(PushGraphTransferError::Protocol(
                "GenHub returned an invalid direct GCS database URI".to_string(),
            ));
        }
        let expected_uri_identity = connection_uri.identity();
        if !same_http_origin(repository_origin, &direct_push.publish_url) {
            return Err(PushGraphTransferError::Protocol(
                "GenHub returned a direct-push publication URL outside the repository origin"
                    .to_string(),
            ));
        }
        if !same_http_origin(repository_origin, &direct_push.abort_url) {
            return Err(PushGraphTransferError::Protocol(
                "GenHub returned a direct-push abort URL outside the repository origin".to_string(),
            ));
        }
        let expected_scope = direct_push.session_scope.clone();
        let renewal = Self {
            expected_repository_origin: repository_origin.to_string(),
            expected_scope,
            expected_session_id,
            expected_transfer_id,
            expected_uri_identity,
            fetch,
            state: Arc::new(Mutex::new(PushCapabilityState {
                response,
                issued_at: Utc::now(),
            })),
        };
        renewal
            .auth_token_internal(AuthRefreshReason::Request)
            .map_err(PushCapabilityRenewal::transfer_error)?;
        Ok(renewal)
    }

    pub(super) fn auth_token(&self, reason: AuthRefreshReason) -> Result<String, AuthError> {
        self.auth_token_internal(reason)
            .map_err(|_| AuthError("GenHub rejected direct-push credential renewal".to_string()))
    }

    /// BlockCacheVFS can request a new CAB after a token expires during a long upload.
    ///
    /// Its storage callback uses this path to rotate credentials while keeping the push bound to
    /// the persisted session and its already accepted checkpoints.
    fn auth_token_internal(
        &self,
        reason: AuthRefreshReason,
    ) -> Result<String, CapabilityRefreshFailure> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| CapabilityRefreshFailure::InvalidScope)?;
        let should_refresh = matches!(reason, AuthRefreshReason::Unauthorized)
            || capability_needs_refresh(&state.response, state.issued_at, Utc::now());
        if should_refresh {
            let response = (self.fetch)().map_err(CapabilityRefreshFailure::Fetch)?;
            self.validate_renewal(&response)?;
            state.response = response;
            state.issued_at = Utc::now();
        }
        direct_push_access_token(&state.response)
            .map_err(|_| CapabilityRefreshFailure::InvalidScope)
    }

    fn validate_renewal(
        &self,
        response: &CapabilityResponse,
    ) -> Result<(), CapabilityRefreshFailure> {
        self.validate_session_identity_internal(response)?;
        let direct_push = response
            .direct_push
            .as_ref()
            .ok_or(CapabilityRefreshFailure::InvalidScope)?;
        if response.expires_at <= Utc::now() || direct_push.token_expires_at <= Utc::now() {
            return Err(CapabilityRefreshFailure::Expired);
        }
        Ok(())
    }

    /// A resumed push must retain the repository and prefix GenHub authorized originally.
    ///
    /// The retry coordinator checks refreshed capabilities here before allowing the same session to
    /// continue after an authorization failure.
    pub(super) fn validate_session_identity(
        &self,
        response: &CapabilityResponse,
    ) -> Result<(), PushGraphTransferError> {
        self.validate_session_identity_internal(response)
            .map_err(Self::transfer_error)
    }

    fn validate_session_identity_internal(
        &self,
        response: &CapabilityResponse,
    ) -> Result<(), CapabilityRefreshFailure> {
        let direct_push = response
            .direct_push
            .as_ref()
            .ok_or(CapabilityRefreshFailure::InvalidScope)?;
        let uri_identity_matches =
            CloudConnectionUri::parse(&direct_push.database_uri).is_ok_and(|uri| {
                uri.gcs_access_token().is_some() && uri.identity() == self.expected_uri_identity
            });
        if response.remote_url.is_some()
            || response.transfer_id != self.expected_transfer_id
            || direct_push.session_id != self.expected_session_id
            || direct_push.session_scope != self.expected_scope
            || !uri_identity_matches
            || (!same_http_origin(&self.expected_repository_origin, &direct_push.publish_url)
                || !same_http_origin(&self.expected_repository_origin, &direct_push.abort_url))
        {
            return Err(CapabilityRefreshFailure::InvalidScope);
        }
        Ok(())
    }

    pub(super) fn capability_response(&self) -> Result<CapabilityResponse, PushGraphTransferError> {
        let state = self.state.lock().map_err(|_| {
            PushGraphTransferError::Protocol(
                "direct-push credential state is unavailable".to_string(),
            )
        })?;
        Ok(state.response.clone())
    }

    pub(super) fn push_lease(&self) -> Result<PushTransferLease, PushGraphTransferError> {
        let response = self.capability_response()?;
        Ok(PushTransferLease {
            transfer_id: response.transfer_id,
            expires_at: response.expires_at.timestamp(),
        })
    }

    pub(super) fn ensure_fresh(&self) -> Result<PushTransferLease, PushGraphTransferError> {
        self.auth_token_internal(AuthRefreshReason::Request)
            .map_err(PushCapabilityRenewal::transfer_error)?;
        self.push_lease()
    }

    pub(super) fn publish(&self) -> Result<(), PushGraphTransferError> {
        self.ensure_fresh()?;
        let response = self.capability_response()?;
        let direct_push = response.direct_push.as_ref().ok_or_else(|| {
            PushGraphTransferError::Protocol(
                "GenHub push capability did not include a direct GCS session".to_string(),
            )
        })?;
        publish_direct_push(direct_push).map_err(PushGraphTransferError::Client)
    }

    /// A rejected branch must release this session's claim so an immediate forced retry can start.
    ///
    /// The refresh check is cheap while the signed capability remains valid and renews the same
    /// client-owned lease only when the abort endpoint would otherwise receive an expired grant.
    pub(super) fn abort(&self) -> Result<(), PushGraphTransferError> {
        self.ensure_fresh()?;
        let response = self.capability_response()?;
        let direct_push = response.direct_push.as_ref().ok_or_else(|| {
            PushGraphTransferError::Protocol(
                "GenHub push capability did not include a direct GCS session".to_string(),
            )
        })?;
        abort_direct_push(direct_push).map_err(PushGraphTransferError::Client)
    }

    pub(super) fn confirm_stale_session(&self) -> Result<bool, PushGraphTransferError> {
        let response = self.capability_response()?;
        let direct_push = response.direct_push.as_ref().ok_or_else(|| {
            PushGraphTransferError::Protocol(
                "GenHub push capability did not include a direct GCS session".to_string(),
            )
        })?;
        confirm_direct_push_session_stale(direct_push).map_err(PushGraphTransferError::Client)
    }
}

impl PushCapabilityRenewal {
    fn transfer_error(error: CapabilityRefreshFailure) -> PushGraphTransferError {
        match error {
            CapabilityRefreshFailure::Fetch(error) => push_capability_error(error),
            CapabilityRefreshFailure::Expired => {
                PushGraphTransferError::Client(RemoteClientError::AuthenticationRequired)
            }
            CapabilityRefreshFailure::InvalidScope => PushGraphTransferError::Protocol(
                "GenHub changed the scope of the direct-push capability".to_string(),
            ),
        }
    }
}

/// Maps a capability conflict to a retryable state only when GenHub identifies active contention.
///
/// Capability responses include a structured message; parsing that before sanitization prevents a
/// transient claim collision from retiring the client's resumable session UUID.
pub(super) fn push_capability_error(error: RemoteClientError) -> PushGraphTransferError {
    match error {
        RemoteClientError::Http { status, message } if status == reqwest::StatusCode::CONFLICT => {
            if is_push_contention_response(&message) {
                PushGraphTransferError::CapabilityContention
            } else if is_inactive_push_response(&message) {
                PushGraphTransferError::CapabilityLeaseInactive
            } else {
                PushGraphTransferError::Client(RemoteClientError::Http {
                    status,
                    message: "direct-push capability was rejected".to_string(),
                })
            }
        }
        error => PushGraphTransferError::Client(sanitize_capability_fetch_error(error)),
    }
}

/// Preserves typed capability failures while the generic transfer helper still returns a box.
pub(super) fn wrap_push_authorization_error(error: Box<dyn Error>) -> Box<dyn Error> {
    match error.downcast::<RemoteClientError>() {
        Ok(error) => Box::new(push_capability_error(*error)),
        Err(error) => error,
    }
}

fn is_push_contention_response(message: &str) -> bool {
    structured_conflict_message(message).as_deref() == Some("Another push is in progress")
}

fn is_inactive_push_response(message: &str) -> bool {
    structured_conflict_message(message).as_deref() == Some("Remote transfer ID cannot be reused")
}

fn structured_conflict_message(body: &str) -> Option<String> {
    let body = serde_json::from_str::<serde_json::Value>(body).ok()?;
    if body.get("code").and_then(serde_json::Value::as_u64) != Some(409) {
        return None;
    }
    body.get("message")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
}

/// Renewal failures can carry signed URLs or provider details that must not reach CLI errors.
///
/// `PushCapabilityRenewal::transfer_error` uses this so retry policy retains the failure class
/// while diagnostics do not expose credentials from a long-running push's refreshed grant.
fn sanitize_capability_fetch_error(error: RemoteClientError) -> RemoteClientError {
    match error {
        RemoteClientError::InvalidRepositoryUrl(_) => {
            RemoteClientError::InvalidRepositoryUrl("configured GenHub repository".to_string())
        }
        RemoteClientError::AuthenticationRequired => RemoteClientError::AuthenticationRequired,
        RemoteClientError::StaleGraphSession => RemoteClientError::StaleGraphSession,
        RemoteClientError::Http { status, .. } => RemoteClientError::Http {
            status,
            message: "direct-push credential renewal failed".to_string(),
        },
        RemoteClientError::ResponseDecode {
            endpoint,
            status,
            declared_content_length,
            source,
        } => RemoteClientError::ResponseDecode {
            endpoint,
            status,
            declared_content_length,
            source: source.without_url(),
        },
        RemoteClientError::Request(source) => RemoteClientError::Request(source.without_url()),
        RemoteClientError::TokenStorage(_) => RemoteClientError::TokenStorage(io::Error::other(
            "direct-push credential storage failed",
        )),
    }
}

/// Short-lived CABs need a proportional refresh margin to avoid immediate repeated renewals.
///
/// The storage callback and publication path use this before a capability is reused.
fn capability_needs_refresh(
    response: &CapabilityResponse,
    issued_at: DateTime<Utc>,
    now: DateTime<Utc>,
) -> bool {
    let Some(direct_push) = response.direct_push.as_ref() else {
        return true;
    };
    let expires_at = if response.expires_at <= direct_push.token_expires_at {
        response.expires_at
    } else {
        direct_push.token_expires_at
    };
    let lifetime_milliseconds = expires_at
        .signed_duration_since(issued_at)
        .num_milliseconds()
        .max(0);
    let margin_milliseconds =
        (lifetime_milliseconds / 10).min(PUSH_AUTHORIZATION_REFRESH_MARGIN_SECONDS * 1_000);
    now >= expires_at - ChronoDuration::milliseconds(margin_milliseconds)
}

fn direct_push_access_token(response: &CapabilityResponse) -> Result<String, AuthError> {
    let direct_push = response
        .direct_push
        .as_ref()
        .ok_or_else(|| AuthError("GenHub did not return a direct-push capability".to_string()))?;
    let connection_uri = CloudConnectionUri::parse(&direct_push.database_uri)
        .map_err(|_| AuthError("GenHub returned an invalid direct GCS database URI".to_string()))?;
    connection_uri
        .gcs_access_token()
        .map(str::to_string)
        .ok_or_else(|| AuthError("GenHub returned an invalid direct GCS database URI".to_string()))
}

fn same_http_origin(expected_origin: &str, candidate_url: &str) -> bool {
    let Ok(expected) = Url::parse(expected_origin) else {
        return false;
    };
    let Ok(candidate) = Url::parse(candidate_url) else {
        return false;
    };
    matches!(candidate.scheme(), "http" | "https")
        && candidate.username().is_empty()
        && candidate.password().is_none()
        && candidate.fragment().is_none()
        && expected.origin() == candidate.origin()
}

/// GenHub graph operations need scoped credentials, while configured local file remotes need none.
///
/// Read operations receive a GCS BlockCacheVFS URI in `remote_url`. The push coordinator supplies
/// its persisted idempotency token here to reopen the same direct-GCS staging session.
pub(super) fn transfer_authorization(
    remote: &Remote,
    operation: RemoteOperation,
    branch: Option<&str>,
    force: bool,
    idempotency_token: Option<Uuid>,
    interactive_login: impl FnOnce(&str) -> Result<AuthTokens, Box<dyn Error>>,
) -> Result<GraphTransferAuthorization, Box<dyn Error>> {
    if remote.url.starts_with("file://") {
        return Ok(GraphTransferAuthorization {
            remote_url: Some(file_graph_url(&remote.url)?),
            push_lease: None,
            direct_push: None,
            capability_response: None,
        });
    }
    let repository = RepositoryRemote::parse(&remote.url)?;
    let request = CapabilityRequest {
        operation,
        branch,
        force,
    };
    let capability = if operation == RemoteOperation::Push {
        let idempotency_token = idempotency_token
            .ok_or("GenHub push capabilities require a persisted idempotency token")?;
        acquire_push_capability(&repository, &request, idempotency_token, interactive_login)?
    } else {
        acquire_capability(&repository, &request, interactive_login)?
    };
    let push_lease = (operation == RemoteOperation::Push).then_some(PushTransferLease {
        transfer_id: capability.transfer_id,
        expires_at: capability.expires_at.timestamp(),
    });
    let direct_push = capability.direct_push.clone();
    Ok(GraphTransferAuthorization {
        remote_url: capability.remote_url.clone(),
        push_lease,
        direct_push,
        capability_response: Some(capability),
    })
}

/// A direct-GCS push may outlive its initial CAB while its accepted session must stay unchanged.
///
/// `run_push_graph_transfer` creates this coordinator before opening the loopback server so the
/// native VFS callback can renew credentials against the persisted session UUID.
pub(super) fn push_capability_renewal(
    remote: &Remote,
    branch: &str,
    force: bool,
    session_id: Uuid,
    expected_transfer_id: Uuid,
    response: CapabilityResponse,
) -> Result<PushCapabilityRenewal, PushGraphTransferError> {
    let repository =
        RepositoryRemote::parse(&remote.url).map_err(PushGraphTransferError::Client)?;
    let refresh_repository = repository.clone();
    let refresh_branch = branch.to_string();
    let fetch: Arc<PushCapabilityFetcher> = Arc::new(move || {
        let request = CapabilityRequest {
            operation: RemoteOperation::Push,
            branch: Some(&refresh_branch),
            force,
        };
        acquire_push_capability(&refresh_repository, &request, session_id, |_| {
            Err("interactive login is unavailable during a push".into())
        })
    });
    PushCapabilityRenewal::new(
        response,
        expected_transfer_id,
        session_id,
        repository.origin(),
        fetch,
    )
}

/// Reopens one client's prior push scope for cleanup or a published-lease resume.
///
/// Unpublished sessions have no stored transfer ID, so the capability response supplies it; the
/// renewal still validates the session UUID, repository origin, and any known published lease ID.
pub(super) fn acquire_push_session_renewal(
    remote: &Remote,
    branch: &str,
    force: bool,
    session_id: Uuid,
    expected_transfer_id: Option<Uuid>,
) -> Result<PushCapabilityRenewal, PushGraphTransferError> {
    let repository =
        RepositoryRemote::parse(&remote.url).map_err(PushGraphTransferError::Client)?;
    let request = CapabilityRequest {
        operation: RemoteOperation::Push,
        branch: Some(branch),
        force,
    };
    let response = acquire_push_capability(&repository, &request, session_id, login_origin)
        .map_err(push_capability_error)?;
    let expected_transfer_id = expected_transfer_id.unwrap_or(response.transfer_id);
    push_capability_renewal(
        remote,
        branch,
        force,
        session_id,
        expected_transfer_id,
        response,
    )
}

#[cfg(test)]
mod tests {
    use std::{
        io::{ErrorKind, Read as _, Write as _},
        net::TcpListener,
        sync::{Arc, Mutex},
        thread,
        time::{Duration, Instant},
    };

    use chrono::{DateTime, Duration as ChronoDuration, Utc};
    use reqwest::StatusCode;
    use rusqlite::blockcachevfs::AuthRefreshReason;
    use uuid::Uuid;

    use super::{PushCapabilityFetcher, PushCapabilityRenewal, push_capability_error};
    use crate::commands::remote::client::{
        CapabilityResponse, DirectPushCapability, RemoteClientError,
        SessionScope as ClientSessionScope,
    };

    const TEST_TRANSFER_ID: Uuid = Uuid::from_u128(1);

    fn push_capability_response(
        token: &str,
        token_expires_at: DateTime<Utc>,
        expires_at: DateTime<Utc>,
        database_uri: &str,
        publish_url: &str,
        abort_url: &str,
        session_scope: ClientSessionScope,
    ) -> CapabilityResponse {
        CapabilityResponse {
            remote_url: None,
            expires_at,
            default_branch: "main".to_string(),
            transfer_id: TEST_TRANSFER_ID,
            direct_push: Some(DirectPushCapability {
                database_uri: database_uri.replace("TOKEN", token),
                token_expires_at,
                session_id: TEST_TRANSFER_ID,
                session_scope,
                publish_url: publish_url.to_string(),
                abort_url: abort_url.to_string(),
            }),
        }
    }

    fn push_session_scope() -> ClientSessionScope {
        ClientSessionScope {
            principal: "repository:repo-id".to_string(),
            target_database: "default.db".to_string(),
            operations: r#"["write","push","main",false]"#.to_string(),
        }
    }

    fn push_database_uri() -> &'static str {
        "gcs://bucket/.gen/graph_db/?database=default.db&project=test-project&vfs=blockcachevfs&endpoint=https%3A%2F%2Fstorage.googleapis.com&access_token=TOKEN"
    }

    fn push_publish_url(token: &str) -> String {
        format!("https://genhub.bio/api/publish?capability={token}")
    }

    fn push_abort_url(token: &str) -> String {
        format!("https://genhub.bio/api/abort?capability={token}")
    }

    #[test]
    fn test_push_capability_conflicts_keep_contention_retryable_and_retire_rejections() {
        let contention = push_capability_error(RemoteClientError::Http {
            status: StatusCode::CONFLICT,
            message: r#"{"code":409,"message":"Another push is in progress"}"#.to_string(),
        });
        assert!(matches!(
            &contention,
            super::super::direct_gcs::PushGraphTransferError::CapabilityContention
        ));
        assert!(
            !contention.is_terminal(),
            "active lease contention must preserve a resumable session"
        );

        let permanent = push_capability_error(RemoteClientError::Http {
            status: StatusCode::CONFLICT,
            message: r#"{"code":409,"message":"Remote transfer ID cannot be reused"}"#.to_string(),
        });
        assert!(
            permanent.is_terminal(),
            "a rejected capability UUID must be retired"
        );
        assert!(
            !permanent
                .to_string()
                .contains("Remote transfer ID cannot be reused"),
            "capability error details should be sanitized"
        );
        assert!(matches!(
            &permanent,
            super::super::direct_gcs::PushGraphTransferError::CapabilityLeaseInactive
        ));
    }

    #[test]
    fn test_capability_conflict_classification_requires_structured_error_code() {
        for body in [
            r#" { "message" : "Another push is in progress", "code" : 409 } "#,
            r#"{"code":409,"message":"Another push is in progress"}"#,
        ] {
            assert!(matches!(
                push_capability_error(RemoteClientError::Http {
                    status: StatusCode::CONFLICT,
                    message: body.to_string(),
                }),
                super::super::direct_gcs::PushGraphTransferError::CapabilityContention
            ));
        }

        for body in [
            r#"{"code":500,"message":"Another push is in progress"}"#,
            "proxy says another push is in progress",
        ] {
            let error = push_capability_error(RemoteClientError::Http {
                status: StatusCode::CONFLICT,
                message: body.to_string(),
            });
            assert!(error.is_terminal());
            assert!(!matches!(
                error,
                super::super::direct_gcs::PushGraphTransferError::CapabilityContention
            ));
        }

        for body in [
            r#"{"code":409,"message":"Remote transfer ID cannot be reused"}"#,
            r#" { "message":"Remote transfer ID cannot be reused", "code":409 } "#,
        ] {
            assert!(matches!(
                push_capability_error(RemoteClientError::Http {
                    status: StatusCode::CONFLICT,
                    message: body.to_string(),
                }),
                super::super::direct_gcs::PushGraphTransferError::CapabilityLeaseInactive
            ));
        }

        for body in [
            r#"{"code":400,"message":"Remote transfer ID cannot be reused"}"#,
            "proxy says transfer cannot be reused",
        ] {
            let error = push_capability_error(RemoteClientError::Http {
                status: StatusCode::CONFLICT,
                message: body.to_string(),
            });
            assert!(error.is_terminal());
            assert!(!matches!(
                error,
                super::super::direct_gcs::PushGraphTransferError::CapabilityLeaseInactive
            ));
        }
    }

    #[test]
    fn test_initial_push_capability_rejects_off_origin_abort_url() {
        let now = Utc::now();
        let response = push_capability_response(
            "initial-token",
            now + ChronoDuration::hours(1),
            now + ChronoDuration::minutes(15),
            push_database_uri(),
            &push_publish_url("initial-signed-value"),
            "https://attacker.example/api/abort?signature=private-abort-secret",
            push_session_scope(),
        );
        let fetch: Arc<PushCapabilityFetcher> =
            Arc::new(|| Err(RemoteClientError::AuthenticationRequired));

        let error = match PushCapabilityRenewal::new(
            response,
            TEST_TRANSFER_ID,
            TEST_TRANSFER_ID,
            "https://genhub.bio",
            fetch,
        ) {
            Ok(_) => panic!("should reject an abort endpoint outside the repository origin"),
            Err(error) => error,
        };

        assert!(
            error
                .to_string()
                .contains("abort URL outside the repository origin")
        );
        assert!(!error.to_string().contains("private-abort-secret"));
    }

    #[test]
    fn test_expired_initial_push_capability_renews_before_open() {
        let now = Utc::now();
        let initial = push_capability_response(
            "expired-token",
            now - ChronoDuration::seconds(1),
            now + ChronoDuration::minutes(15),
            push_database_uri(),
            &push_publish_url("initial-signed-value"),
            &push_abort_url("initial-signed-value"),
            push_session_scope(),
        );
        let refreshed = push_capability_response(
            "rotated-token",
            now + ChronoDuration::hours(1),
            now + ChronoDuration::minutes(30),
            push_database_uri(),
            &push_publish_url("refreshed-signed-value"),
            &push_abort_url("refreshed-signed-value"),
            push_session_scope(),
        );
        let refresh_count = Arc::new(Mutex::new(0));
        let fetch_count = Arc::clone(&refresh_count);
        let fetch_response = refreshed.clone();
        let fetch: Arc<PushCapabilityFetcher> = Arc::new(move || {
            *fetch_count
                .lock()
                .expect("should lock capability renewal counter") += 1;
            Ok(fetch_response.clone())
        });

        let renewal = PushCapabilityRenewal::new(
            initial,
            TEST_TRANSFER_ID,
            TEST_TRANSFER_ID,
            "https://genhub.bio",
            fetch,
        )
        .expect("should renew an expired capability before opening the database");

        assert_eq!(
            renewal
                .auth_token(AuthRefreshReason::Request)
                .expect("should use the renewed access token"),
            "rotated-token"
        );
        assert_eq!(
            *refresh_count
                .lock()
                .expect("should lock capability renewal counter"),
            1,
            "a valid renewed capability should be reused"
        );
        assert_eq!(
            renewal
                .capability_response()
                .expect("should read the renewed capability"),
            refreshed
        );
    }

    #[test]
    fn test_refreshed_abort_url_is_retained_and_used() {
        let now = Utc::now();
        let listener = TcpListener::bind("127.0.0.1:0").expect("should bind abort request server");
        listener
            .set_nonblocking(true)
            .expect("should bound the abort request accept wait");
        let address = listener
            .local_addr()
            .expect("should read abort request server address");
        let repository_origin = format!("http://{address}");
        let refreshed_abort_url = format!(
            "{repository_origin}/custom/session/retire/current?signature=refreshed-abort-secret&mode=force"
        );
        let initial = push_capability_response(
            "expired-token",
            now - ChronoDuration::seconds(1),
            now - ChronoDuration::seconds(1),
            push_database_uri(),
            &format!("{repository_origin}/custom/session/publish?signature=initial-publish-secret"),
            &format!("{repository_origin}/custom/session/retire?signature=initial-abort-secret"),
            push_session_scope(),
        );
        let refreshed = push_capability_response(
            "rotated-token",
            now + ChronoDuration::hours(1),
            now + ChronoDuration::minutes(30),
            push_database_uri(),
            &format!(
                "{repository_origin}/custom/session/commit?signature=refreshed-publish-secret"
            ),
            &refreshed_abort_url,
            push_session_scope(),
        );
        let fetch_response = refreshed.clone();
        let fetch: Arc<PushCapabilityFetcher> = Arc::new(move || Ok(fetch_response.clone()));
        let renewal = PushCapabilityRenewal::new(
            initial,
            TEST_TRANSFER_ID,
            TEST_TRANSFER_ID,
            &repository_origin,
            fetch,
        )
        .expect("should refresh a valid capability before opening the database");
        let server = thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(5);
            let (mut stream, _) = loop {
                match listener.accept() {
                    Ok(connection) => break connection,
                    Err(error)
                        if error.kind() == ErrorKind::WouldBlock && Instant::now() < deadline =>
                    {
                        thread::sleep(Duration::from_millis(10));
                    }
                    Err(error) => panic!("should accept the session abort request: {error}"),
                }
            };
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .expect("should bound the abort request read");
            let mut request = Vec::new();
            let mut buffer = [0_u8; 4096];
            loop {
                let read = stream.read(&mut buffer).expect("should read abort request");
                assert!(read > 0, "abort request should include HTTP headers");
                request.extend_from_slice(&buffer[..read]);
                if request.windows(4).any(|window| window == b"\r\n\r\n") {
                    break;
                }
            }
            stream
                .write_all(
                    b"HTTP/1.1 204 No Content\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                )
                .expect("should accept the session abort request");
            String::from_utf8(request).expect("should decode abort request as UTF-8")
        });

        renewal
            .abort()
            .expect("should abort with the refreshed endpoint");
        let request = server.join().expect("abort request server should finish");
        assert!(
            request.starts_with(
                "POST /custom/session/retire/current?signature=refreshed-abort-secret&mode=force "
            ),
            "abort should use the refreshed server-supplied URL: {request}"
        );
        assert!(request.to_ascii_lowercase().contains("content-length: 0"));
    }

    #[test]
    fn test_push_capability_request_uses_fractional_margin_for_short_lifetimes() {
        let now = Utc::now();
        let response = push_capability_response(
            "short-lived-token",
            now + ChronoDuration::seconds(10),
            now + ChronoDuration::minutes(15),
            push_database_uri(),
            &push_publish_url("short-lived-value"),
            &push_abort_url("short-lived-value"),
            push_session_scope(),
        );

        assert!(
            !super::capability_needs_refresh(
                &response,
                now,
                now + ChronoDuration::milliseconds(500)
            ),
            "a fresh short-lived token should not trigger an immediate renewal"
        );
        assert!(super::capability_needs_refresh(
            &response,
            now,
            now + ChronoDuration::seconds(9)
        ));
    }

    #[test]
    fn test_push_capability_renewal_rejects_changed_uri_or_scope() {
        let now = Utc::now();
        let cases = [
            push_capability_response(
                "rotated-token",
                now + ChronoDuration::hours(2),
                now + ChronoDuration::minutes(30),
                "gcs://other-bucket/.gen/graph_db/?database=default.db&project=test-project&vfs=blockcachevfs&endpoint=https%3A%2F%2Fstorage.googleapis.com&access_token=TOKEN",
                &push_publish_url("refreshed-signed-value"),
                &push_abort_url("refreshed-signed-value"),
                push_session_scope(),
            ),
            push_capability_response(
                "rotated-token",
                now + ChronoDuration::hours(2),
                now + ChronoDuration::minutes(30),
                push_database_uri(),
                &push_publish_url("refreshed-signed-value"),
                &push_abort_url("refreshed-signed-value"),
                ClientSessionScope {
                    operations: r#"["write","push","feature",false]"#.to_string(),
                    ..push_session_scope()
                },
            ),
            push_capability_response(
                "rotated-token",
                now + ChronoDuration::hours(2),
                now + ChronoDuration::minutes(30),
                push_database_uri(),
                &push_publish_url("refreshed-signed-value"),
                "https://attacker.example/api/abort?signature=private-abort-secret",
                push_session_scope(),
            ),
        ];

        for refreshed in cases {
            let initial = push_capability_response(
                "initial-token",
                now + ChronoDuration::hours(1),
                now + ChronoDuration::minutes(15),
                push_database_uri(),
                &push_publish_url("initial-signed-value"),
                &push_abort_url("initial-signed-value"),
                push_session_scope(),
            );
            let response = refreshed.clone();
            let fetch: Arc<PushCapabilityFetcher> = Arc::new(move || Ok(response.clone()));
            let renewal = PushCapabilityRenewal::new(
                initial.clone(),
                TEST_TRANSFER_ID,
                TEST_TRANSFER_ID,
                "https://genhub.bio",
                fetch,
            )
            .expect("should create a valid initial renewal coordinator");

            let error = renewal
                .auth_token(AuthRefreshReason::Unauthorized)
                .expect_err("renewal must reject changes outside access_token");

            assert!(!format!("{error:?}").contains("initial-token"));
            assert!(!format!("{error:?}").contains("private-abort-secret"));
            assert_eq!(
                renewal
                    .capability_response()
                    .expect("should retain the previous capability"),
                initial
            );
        }
    }
}
