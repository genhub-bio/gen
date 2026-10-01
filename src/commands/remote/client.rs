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
//! download URLs used to transfer files referenced by the selected branch. A push then
//! calls [`complete_asset_transfers`] with the successful capability's transfer ID and
//! GCS-validated upload receipts so GenHub can verify the stored objects before releasing
//! the push lease.
//!
//! Both acquisition functions use the same authentication sequence. Public clone and
//! pull requests may first be attempted anonymously; otherwise the client tries
//! `GENHUB_API_KEY`, tokens stored for the normalized GenHub origin, and finally the
//! interactive login callback supplied by the command. Rejected access tokens are
//! refreshed through GenHub's CLI refresh endpoint when possible, and refreshed or
//! newly issued tokens are saved for later requests.

use std::{env, fmt, io};

use base64::{
    Engine as _,
    engine::general_purpose::{URL_SAFE, URL_SAFE_NO_PAD},
};
use chrono::{DateTime, Utc};
use gen_core::{DoltHashId, HashId};
use reqwest::{
    StatusCode, Url, Version,
    blocking::{Client, RequestBuilder},
    header::{CONTENT_LENGTH, CONTENT_TYPE, HeaderMap},
    redirect::Policy,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

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

    fn asset_transfer_completion_url(&self) -> String {
        format!(
            "{}/api/repos/{}/{}/asset-transfers/complete",
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

#[derive(Clone, Deserialize, Eq, PartialEq)]
pub struct CapabilityResponse {
    pub remote_url: Option<String>,
    pub expires_at: DateTime<Utc>,
    pub default_branch: String,
    /// Identifies the capability's transfer-scoped push lease.
    pub transfer_id: Uuid,
    /// Direct GCS upload details returned for GenHub pushes.
    #[serde(default)]
    pub direct_push: Option<DirectPushCapability>,
}

impl fmt::Debug for CapabilityResponse {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CapabilityResponse")
            .field("remote_url", &"[REDACTED]")
            .field("expires_at", &self.expires_at)
            .field("default_branch", &self.default_branch)
            .field("transfer_id", &self.transfer_id)
            .field("direct_push", &self.direct_push)
            .finish()
    }
}

/// A GenHub capability for staging graph blocks directly in cloud storage.
#[derive(Clone, Deserialize, Eq, PartialEq)]
pub struct DirectPushCapability {
    /// GCS database URI containing the short-lived, downscoped token.
    pub database_uri: String,
    /// Expiration time of the downscoped token carried by `database_uri`.
    pub token_expires_at: DateTime<Utc>,
    /// Stable client session UUID shared with the GenHub lease.
    pub session_id: Uuid,
    /// Context used to bind the local and server-side session attachments.
    pub session_scope: SessionScope,
    /// Signed GenHub endpoint that publishes the accepted session manifest.
    pub publish_url: String,
}

impl fmt::Debug for DirectPushCapability {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DirectPushCapability")
            .field("database_uri", &"[REDACTED]")
            .field("session_id", &self.session_id)
            .field("session_scope", &self.session_scope)
            .field("publish_url", &"[REDACTED]")
            .finish()
    }
}

/// Scope fields shared by the direct uploader and GenHub's manifest publisher.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct SessionScope {
    pub principal: String,
    pub target_database: String,
    pub operations: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct AssetTransferRequest<'request> {
    /// The complete remote operation whose asset phase is being requested.
    pub operation: RemoteOperation,
    /// The branch whose reachable assets should be transferred.
    pub branch: &'request str,
    /// The lower commit boundary, whose reachable assets are excluded. This is
    /// the commit we are currently at.
    pub from_commit: Option<&'request DoltHashId>,
    /// The inclusive upper commit boundary for the operation.
    pub to_commit: Option<&'request DoltHashId>,
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

/// Identifies one pushed asset and the CRC32C expected from object storage.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct AssetUploadReceipt {
    /// The uploaded branch asset's stable ID.
    pub id: HashId,
    /// The base64-encoded CRC32C calculated alongside the asset's SHA-256.
    pub crc32c: String,
}

/// Signals that all uploads for a pushed branch have completed.
#[derive(Clone, Debug, Serialize)]
pub struct AssetTransferCompletionRequest<'request> {
    /// The capability-scoped owner of the push lease being completed.
    pub transfer_id: Uuid,
    /// The branch whose expected asset set should be verified.
    pub branch: &'request str,
    /// Provider-checksum receipts for every uploaded or reused object.
    pub assets: &'request [AssetUploadReceipt],
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
    #[error("Remote endpoint returned HTTP {status}: {message}")]
    Http { status: StatusCode, message: String },
    #[error("GenHub confirmed that the direct GCS graph session is stale")]
    StaleGraphSession,
    #[error(
        "Failed to decode {endpoint} response (HTTP {status}, declared Content-Length {declared_content_length:?}): {source}"
    )]
    ResponseDecode {
        endpoint: &'static str,
        status: StatusCode,
        declared_content_length: Option<u64>,
        #[source]
        source: reqwest::Error,
    },
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
    idempotency_token: Option<Uuid>,
    authorization: RequestAuthorization<'_>,
) -> Result<CapabilityResponse, RemoteClientError> {
    let mut builder = client.post(repository.capability_url()).json(request);
    if let Some(idempotency_token) = idempotency_token {
        builder = builder.header("Idempotency-Token", idempotency_token.to_string());
    }
    let response = authorize_request(builder, authorization).send()?;
    if !response.status().is_success() {
        return Err(response_error(response));
    }
    let status = response.status();
    let declared_content_length = response.content_length();
    response
        .json()
        .map_err(|source| RemoteClientError::ResponseDecode {
            endpoint: "remote capability",
            status,
            declared_content_length,
            source,
        })
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
    let status = response.status();
    let declared_content_length = response.content_length();
    response
        .json()
        .map_err(|source| RemoteClientError::ResponseDecode {
            endpoint: "asset transfers",
            status,
            declared_content_length,
            source,
        })
}

fn send_asset_transfer_completion(
    client: &Client,
    repository: &RepositoryRemote,
    request: &AssetTransferCompletionRequest<'_>,
    authorization: RequestAuthorization<'_>,
) -> Result<(), RemoteClientError> {
    let response = authorize_request(
        client
            .post(repository.asset_transfer_completion_url())
            .json(request),
        authorization,
    )
    .send()?;
    if !response.status().is_success() {
        return Err(response_error(response));
    }
    Ok(())
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
        let tokens = if access_token_expired_hint(&tokens.jwt, Utc::now()) {
            let refreshed = refresh_tokens(client, repository, &tokens)?;
            options.token_store.save(repository.origin(), &refreshed)?;
            refreshed
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
    acquire_capability_with_store_and_token(
        client,
        repository,
        request,
        None,
        api_key,
        token_store,
        interactive_login,
    )
}

fn acquire_capability_with_store_and_token(
    client: &Client,
    repository: &RepositoryRemote,
    request: &CapabilityRequest<'_>,
    idempotency_token: Option<Uuid>,
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
        |authorization| {
            send_capability(
                client,
                repository,
                request,
                idempotency_token,
                authorization,
            )
        },
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

/// Requests a push capability with a caller-persisted idempotency token.
pub fn acquire_push_capability(
    repository: &RepositoryRemote,
    request: &CapabilityRequest<'_>,
    idempotency_token: Uuid,
    interactive_login: impl FnOnce(&str) -> Result<AuthTokens, Box<dyn std::error::Error>>,
) -> Result<CapabilityResponse, RemoteClientError> {
    let client = Client::new();
    let api_key = env::var("GENHUB_API_KEY").ok();
    acquire_capability_with_store_and_token(
        &client,
        repository,
        request,
        Some(idempotency_token),
        api_key.as_deref(),
        &FileTokenStore,
        interactive_login,
    )
}

const MAX_DIRECT_PUSH_ERROR_BODY_BYTES: u64 = 4096;
const STALE_SESSION_CHECK_HEADER: &str = "x-gen-session-stale-check";

/// Publishes a staged direct-GCS session through its signed GenHub endpoint.
pub fn publish_direct_push(capability: &DirectPushCapability) -> Result<(), RemoteClientError> {
    send_direct_push_request(capability, false)
}

/// Asks GenHub to reopen a session and report whether its accepted base is stale.
///
/// A successful response means the session is not stale. The server leaves a valid session and
/// its transfer lease untouched; only the exact stale-session response returns `true`.
pub fn confirm_direct_push_session_stale(
    capability: &DirectPushCapability,
) -> Result<bool, RemoteClientError> {
    match send_direct_push_request(capability, true) {
        Ok(()) => Ok(false),
        Err(RemoteClientError::StaleGraphSession) => Ok(true),
        Err(error) => Err(error),
    }
}

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
        return Err(RemoteClientError::Http {
            status,
            message: format!(
                "manifest publication request received {protocol}; response Content-Type: {content_type}"
            ),
        });
    }
    Ok(())
}

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

pub fn complete_asset_transfers(
    repository: &RepositoryRemote,
    request: &AssetTransferCompletionRequest<'_>,
    interactive_login: impl FnOnce(&str) -> Result<AuthTokens, Box<dyn std::error::Error>>,
) -> Result<(), RemoteClientError> {
    let client = Client::new();
    let api_key = env::var("GENHUB_API_KEY").ok();
    acquire_request_with_store(
        &client,
        repository,
        AuthenticationOptions {
            api_key: api_key.as_deref(),
            allow_anonymous: false,
            token_store: &FileTokenStore,
        },
        interactive_login,
        |authorization| send_asset_transfer_completion(&client, repository, request, authorization),
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

    use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
    use chrono::Utc;
    use reqwest::{StatusCode, blocking::Client};
    use uuid::Uuid;

    use super::{
        AssetTransferRequest, AuthTokens, CapabilityRequest, CapabilityResponse,
        DirectPushCapability, MAX_DIRECT_PUSH_ERROR_BODY_BYTES, RemoteClientError, RemoteOperation,
        RepositoryRemote, RequestAuthorization, SessionScope, TokenStore,
        access_token_expired_hint, acquire_capability_with_store,
        acquire_capability_with_store_and_token, confirm_direct_push_session_stale,
        http_version_label, normalized_origin, publish_direct_push, response_content_type_label,
        send_asset_transfers,
    };

    const TEST_TRANSFER_ID: Uuid = Uuid::from_u128(1);

    fn test_jwt_with_expiration(expiration: i64) -> String {
        let header = serde_json::json!({"alg": "HS256", "typ": "JWT"}).to_string();
        let claims = serde_json::json!({"exp": expiration}).to_string();
        format!(
            "{}.{}.{}",
            URL_SAFE_NO_PAD.encode(header),
            URL_SAFE_NO_PAD.encode(claims),
            URL_SAFE_NO_PAD.encode("test-signature")
        )
    }

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

    fn capability_body(remote_url: &str) -> String {
        serde_json::json!({
            "remote_url": remote_url,
            "expires_at": "2030-01-01T00:00:00Z",
            "default_branch": "main",
            "transfer_id": TEST_TRANSFER_ID
        })
        .to_string()
    }

    fn repository(origin: &str) -> RepositoryRemote {
        RepositoryRemote::parse(&format!("{origin}/api/repos/alice/example"))
            .expect("should parse mock repository")
    }

    fn assert_response_decode_context(
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
                "publish_url": "https://genhub.bio/api/publish?token=secret"
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
                publish_url: "https://genhub.bio/api/publish?token=secret".to_string(),
            })
        );
        let debug = format!("{capability:?}");
        assert!(!debug.contains("legacy-transfer"));
        assert!(!debug.contains("access_token=secret"));
        assert!(!debug.contains("token=secret"));
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
        let response_body =
            r#"{"assets":[{"url":"https://storage.example/?token=sensitive-canary""#;
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
        let store = MemoryTokenStore::with_tokens(AuthTokens {
            jwt: test_jwt_with_expiration(Utc::now().timestamp() - 60),
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
        .expect("expired JWT should refresh before requesting a capability");
        let requests = server.join().expect("mock GenHub should finish");
        let stored = store.current().expect("refreshed tokens should be stored");

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
        let store = MemoryTokenStore::with_tokens(AuthTokens {
            jwt: token.clone(),
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
        .expect("valid-token authorization rejection should keep existing login behavior");
        let requests = server.join().expect("mock GenHub should finish");

        assert!(login_attempted);
        assert_eq!(requests.len(), 2);
        assert!(requests[0].contains(&format!("authorization: Bearer {token}")));
        assert!(!requests[0].contains("token-refresh"));
        assert!(requests[1].contains("authorization: Bearer login-access"));
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
                rendered_error.contains("manifest publication request received HTTP/1.1"),
                "should identify the publication request protocol: {rendered_error}"
            );
            assert!(
                rendered_error.contains(&format!("response Content-Type: {content_type}")),
                "should classify the response content type: {rendered_error}"
            );
            assert!(
                !rendered_error.contains("publish-secret")
                    && !rendered_error.contains("response-body-canary")
                    && !rendered_error.contains("private-canary"),
                "should hide signed URLs, response bodies, and raw Content-Type values: {rendered_error}"
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
