//! GenHub capability coordination for graph transfers.
//!
//! Clone and pull need short-lived Dolt URLs, while push uses a persisted idempotency token to
//! identify a direct-GCS staging session. This module keeps token renewal tied to that same
//! session: long block uploads may outlive one CAB token, but they must never switch repositories,
//! prefixes, operation scope, or publication origin while refreshing it.

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
        RemoteOperation, RepositoryRemote, SessionScope as ClientSessionScope, acquire_capability,
        acquire_push_capability, confirm_direct_push_session_stale, publish_direct_push,
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
    expected_publish_origin: String,
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
        publish_origin: &str,
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
        if !same_http_origin(publish_origin, &direct_push.publish_url) {
            return Err(PushGraphTransferError::Protocol(
                "GenHub returned a direct-push publication URL outside the repository origin"
                    .to_string(),
            ));
        }
        let expected_scope = direct_push.session_scope.clone();
        let renewal = Self {
            expected_publish_origin: publish_origin.to_string(),
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
            || !same_http_origin(&self.expected_publish_origin, &direct_push.publish_url)
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
            CapabilityRefreshFailure::Fetch(error) => {
                PushGraphTransferError::Client(sanitize_capability_fetch_error(error))
            }
            CapabilityRefreshFailure::Expired => {
                PushGraphTransferError::Client(RemoteClientError::AuthenticationRequired)
            }
            CapabilityRefreshFailure::InvalidScope => PushGraphTransferError::Protocol(
                "GenHub changed the scope of the direct-push capability".to_string(),
            ),
        }
    }
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

/// GenHub graph operations need scoped credentials, while local file remotes need none.
///
/// `run_graph_transfer` uses temporary Dolt URLs for clone/pull; the push coordinator supplies its
/// persisted idempotency token here to reopen the same direct-GCS staging session.
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

pub(super) fn acquire_existing_push_renewal(
    remote: &Remote,
    branch: &str,
    force: bool,
    session_id: Uuid,
    expected_transfer_id: Uuid,
) -> Result<PushCapabilityRenewal, PushGraphTransferError> {
    let repository =
        RepositoryRemote::parse(&remote.url).map_err(PushGraphTransferError::Client)?;
    let request = CapabilityRequest {
        operation: RemoteOperation::Push,
        branch: Some(branch),
        force,
    };
    let response = acquire_push_capability(&repository, &request, session_id, login_origin)
        .map_err(PushGraphTransferError::Client)?;
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
    use std::sync::{Arc, Mutex};

    use chrono::{DateTime, Duration as ChronoDuration, Utc};
    use rusqlite::blockcachevfs::AuthRefreshReason;
    use uuid::Uuid;

    use super::{PushCapabilityFetcher, PushCapabilityRenewal};
    use crate::commands::remote::client::{
        CapabilityResponse, DirectPushCapability, SessionScope as ClientSessionScope,
    };

    const TEST_TRANSFER_ID: Uuid = Uuid::from_u128(1);

    fn push_capability_response(
        token: &str,
        token_expires_at: DateTime<Utc>,
        expires_at: DateTime<Utc>,
        database_uri: &str,
        publish_url: &str,
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

    #[test]
    fn test_expired_initial_push_capability_renews_before_open() {
        let now = Utc::now();
        let initial = push_capability_response(
            "expired-token",
            now - ChronoDuration::seconds(1),
            now + ChronoDuration::minutes(15),
            push_database_uri(),
            &push_publish_url("initial-signed-value"),
            push_session_scope(),
        );
        let refreshed = push_capability_response(
            "rotated-token",
            now + ChronoDuration::hours(1),
            now + ChronoDuration::minutes(30),
            push_database_uri(),
            &push_publish_url("refreshed-signed-value"),
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
    fn test_push_capability_request_uses_fractional_margin_for_short_lifetimes() {
        let now = Utc::now();
        let response = push_capability_response(
            "short-lived-token",
            now + ChronoDuration::seconds(10),
            now + ChronoDuration::minutes(15),
            push_database_uri(),
            &push_publish_url("short-lived-value"),
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
                push_session_scope(),
            ),
            push_capability_response(
                "rotated-token",
                now + ChronoDuration::hours(2),
                now + ChronoDuration::minutes(30),
                push_database_uri(),
                &push_publish_url("refreshed-signed-value"),
                ClientSessionScope {
                    operations: r#"["write","push","feature",false]"#.to_string(),
                    ..push_session_scope()
                },
            ),
        ];

        for refreshed in cases {
            let initial = push_capability_response(
                "initial-token",
                now + ChronoDuration::hours(1),
                now + ChronoDuration::minutes(15),
                push_database_uri(),
                &push_publish_url("initial-signed-value"),
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
            assert_eq!(
                renewal
                    .capability_response()
                    .expect("should retain the previous capability"),
                initial
            );
        }
    }
}
