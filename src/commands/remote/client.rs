//! GenHub client for authorizing graph and asset transfers.
//!
//! Gen stores an HTTP(S) remote as its canonical GenHub repository URL. [`RepositoryRemote`]
//! accepts either the repository page or API URL and derives the control-plane endpoints from
//! that identity. Local `file://` remotes bypass this client and are handled by remote operations.
//!
//! Remote operations request a capability before contacting the graph store. Clone and pull use
//! the short-lived Dolt URL returned by GenHub. Push uses a direct-GCS session instead: the VFS
//! writes blocks under a scoped token, then the operations layer asks GenHub to publish the
//! staged manifest. Asset transfer endpoints handle files referenced by the selected branch, and
//! push completion lets GenHub verify the object-store receipts before releasing its lease.
//!
//! Capability and asset requests share one origin-scoped authentication flow. Public reads can
//! start anonymously; authenticated requests use an API key, saved credentials, or interactive
//! login. Token refresh and re-login live in `authentication` so remotes on one GenHub host share the
//! same rotated credentials. Signed manifest publication and stale-session checks live in
//! `direct_push` because they are the commit boundary for direct-GCS pushes.

mod authentication;
mod direct_push;
pub use direct_push::{confirm_direct_push_session_stale, publish_direct_push};
#[cfg(test)]
mod tests;

use std::{env, fmt};

use chrono::{DateTime, Utc};
use gen_core::{DoltHashId, HashId};
use reqwest::{StatusCode, Url, blocking::Client};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

use self::authentication::{
    AuthenticationOptions, FileTokenStore, RequestAuthorization, TokenStore,
    acquire_request_with_store, authorize_request,
};
use crate::commands::remote::server::AuthTokens;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RepositoryRemote {
    origin: String,
    namespace: String,
    slug: String,
    canonical_url: String,
}

impl RepositoryRemote {
    /// Remote configuration accepts both page and API URLs, while endpoint calls need one
    /// canonical repository identity from which GenHub routes can be derived.
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

/// Keys credentials by GenHub origin so multiple repository remotes on that host share one login.
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

/// Clone and pull use this to obtain GenHub’s temporary Dolt transfer URL.
///
/// Pushes use [`acquire_push_capability`] so the session idempotency token stays attached to the
/// capability request.
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

/// Requests a push grant using the idempotency token persisted by the remote operation.
///
/// Reusing the token lets a retried CLI push recover the GenHub lease associated with that
/// logical transfer.
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

/// Push operations call this after graph and asset staging so GenHub can verify receipts before
/// releasing the transfer lease.
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
