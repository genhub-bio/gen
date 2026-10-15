//! Session-owned direct-GCS graph staging for GenHub pushes.
//!
//! A push first stages Dolt blocks through a loopback RemoteServer and validates the requested
//! branch against the local destination. GenHub publishes the manifest only after staging succeeds.
//! Accepted checkpoints can be reused after interruption; a server-confirmed stale checkpoint is
//! surfaced to the operation coordinator so it can begin a fresh session. Cleanup keeps the original
//! Dolt/storage failure visible when closing the temporary server also fails.

use std::{error::Error, path::Path};

use gen_core::DoltHashId;
use gen_models::{db::GraphConnection, operations::Remote};
use reqwest::StatusCode;
use rusqlite::{
    BlockCacheSessionOptions, Error as SqlError, RemoteFailureKind, RemoteServer,
    RemoteServerOptions, SessionOperationId, SessionOperationStatus, SessionScope,
    blockcachevfs::{CloudConnectionUri, StorageFailure},
};
use uuid::Uuid;

use super::{
    capability::{PushCapabilityRenewal, push_capability_renewal, transfer_authorization},
    ensure_graph_remote, is_authorization_error, push_graph_branch_with_progress,
    restore_canonical_url,
};
use crate::commands::remote::{
    client::{RemoteClientError, RemoteOperation},
    login_origin,
    progress::{GraphUploadProgressReporter, TransferProgress},
};

const DIRECT_GCS_UPLOAD_CONCURRENCY: u32 = 10;

fn is_stale_accepted_session_error(error: &SqlError) -> bool {
    error.remote_failure_kind() == Some(RemoteFailureKind::StaleSession)
}

pub(super) fn is_non_fast_forward_push_error(error: &SqlError) -> bool {
    error.remote_failure_kind() == Some(RemoteFailureKind::NonFastForward)
}

#[derive(Debug, thiserror::Error)]
pub(super) enum PushGraphTransferError {
    #[error("{phase} failed: {source}")]
    Database {
        phase: &'static str,
        #[source]
        source: SqlError,
    },
    #[error(transparent)]
    Client(#[from] RemoteClientError),
    #[error("GenHub confirmed the direct GCS graph session is stale")]
    StaleSessionConfirmed,
    #[error("The branch is not a fast-forward of the remote. Use --force to overwrite the remote.")]
    NonFastForward,
    #[error("{primary}; GenHub could not verify the stale session: {confirmation}")]
    StaleSessionCheckFailed {
        primary: Box<PushGraphTransferError>,
        confirmation: Box<PushGraphTransferError>,
    },
    #[error("object storage reported {0}")]
    StorageFailure(StorageFailure),
    #[error("{primary}; object storage also reported {failure}")]
    Storage {
        #[source]
        primary: Box<PushGraphTransferError>,
        failure: StorageFailure,
    },
    #[error("{primary}; local server cleanup also failed: {cleanup}")]
    Cleanup {
        #[source]
        primary: Box<PushGraphTransferError>,
        cleanup: String,
    },
    #[error("{0}")]
    Protocol(String),
}

/// Keeps signed URLs and provider response details out of cleanup diagnostics.
///
/// Direct-session close paths use this when reporting a secondary failure beside the original error.
fn safe_graph_error_context(error: &PushGraphTransferError) -> String {
    match error {
        PushGraphTransferError::Database {
            phase,
            source: SqlError::SqliteFailure(code, _),
        } => format!("{phase} (SQLite code {})", code.extended_code),
        PushGraphTransferError::Database { phase, .. } => format!("{phase} (database error)"),
        PushGraphTransferError::StorageFailure(failure) => {
            format!("object storage: {failure}")
        }
        PushGraphTransferError::Storage { primary, failure } => {
            format!(
                "{}; object storage: {failure}",
                safe_graph_error_context(primary)
            )
        }
        PushGraphTransferError::Cleanup { primary, cleanup } => {
            format!(
                "{}; additionally {cleanup}",
                safe_graph_error_context(primary)
            )
        }
        PushGraphTransferError::Client(RemoteClientError::Http { status, .. }) => {
            format!("remote request HTTP {status}")
        }
        PushGraphTransferError::Client(_) => "remote client request".to_string(),
        PushGraphTransferError::StaleSessionConfirmed
        | PushGraphTransferError::StaleSessionCheckFailed { .. } => {
            "stale direct GCS session recovery".to_string()
        }
        PushGraphTransferError::NonFastForward => {
            "Dolt rejected a non-fast-forward branch update".to_string()
        }
        PushGraphTransferError::Protocol(_) => "remote protocol response".to_string(),
    }
}

impl PushGraphTransferError {
    pub(super) fn database(phase: &'static str, source: SqlError) -> Self {
        Self::Database { phase, source }
    }

    pub(super) fn is_authorization_error(&self) -> bool {
        match self {
            Self::Database { source, .. } => is_authorization_error(source),
            Self::Storage { primary, .. } | Self::Cleanup { primary, .. } => {
                primary.is_authorization_error()
            }
            Self::StorageFailure(_) => false,
            Self::Client(RemoteClientError::Http { status, .. }) => {
                matches!(*status, StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN)
            }
            Self::Client(_)
            | Self::StaleSessionConfirmed
            | Self::NonFastForward
            | Self::StaleSessionCheckFailed { .. }
            | Self::Protocol(_) => false,
        }
    }

    pub(super) fn is_terminal(&self) -> bool {
        match self {
            Self::Client(RemoteClientError::Http { status, .. }) => *status == StatusCode::CONFLICT,
            Self::Client(RemoteClientError::StaleGraphSession)
            | Self::StaleSessionConfirmed
            | Self::StaleSessionCheckFailed { .. } => false,
            Self::StorageFailure(_) => true,
            Self::Storage { primary, .. } | Self::Cleanup { primary, .. } => primary.is_terminal(),
            Self::Protocol(_) => true,
            Self::Database { .. } | Self::Client(_) | Self::NonFastForward => false,
        }
    }

    pub(super) fn is_stale_session(&self) -> bool {
        match self {
            Self::Database { source, .. } => is_stale_accepted_session_error(source),
            Self::Storage { primary, .. } | Self::Cleanup { primary, .. } => {
                primary.is_stale_session()
            }
            Self::StorageFailure(_) => false,
            Self::Client(RemoteClientError::StaleGraphSession) => true,
            Self::Client(_)
            | Self::StaleSessionConfirmed
            | Self::NonFastForward
            | Self::StaleSessionCheckFailed { .. }
            | Self::Protocol(_) => false,
        }
    }

    pub(super) fn is_non_fast_forward(&self) -> bool {
        match self {
            Self::NonFastForward => true,
            Self::Storage { primary, .. } | Self::Cleanup { primary, .. } => {
                primary.is_non_fast_forward()
            }
            _ => false,
        }
    }
}

/// RemoteServer owns background block uploads, so push completion must quiesce them before reuse.
///
/// `push_graph_through_direct_session_inner` calls this after staging, validation, and error paths
/// so GenHub publishes only after uploads settle and the server has released its session resources.
fn close_direct_push_server(
    mut server: RemoteServer,
    progress: &GraphUploadProgressReporter,
    preserve_phase: bool,
) -> Result<(), PushGraphTransferError> {
    let _heartbeat = if preserve_phase {
        progress.heartbeat_preserving_phase("closing the direct GCS graph session")
    } else {
        progress.heartbeat("closing the direct GCS graph session")
    };
    let quiesce_result = server
        .quiesce()
        .map_err(|error| PushGraphTransferError::database("quiescing direct GCS server", error));
    let storage_failure = server.first_storage_error();
    let close_result = server
        .close()
        .map_err(|error| PushGraphTransferError::database("closing direct GCS server", error));
    let cleanup_result = match (quiesce_result, close_result) {
        (Err(quiesce_error), Err(close_error)) => Err(PushGraphTransferError::Cleanup {
            primary: Box::new(quiesce_error),
            cleanup: safe_graph_error_context(&close_error),
        }),
        (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
        (Ok(()), Ok(())) => Ok(()),
    };
    match (cleanup_result, storage_failure) {
        (Err(error), Some(failure)) => Err(PushGraphTransferError::Storage {
            primary: Box::new(error),
            failure,
        }),
        (Err(error), None) => Err(error),
        (Ok(()), Some(failure)) => Err(PushGraphTransferError::StorageFailure(failure)),
        (Ok(()), None) => Ok(()),
    }
}

/// Keeps the push failure primary when temporary-server cleanup fails afterward.
///
/// The push coordinator uses this on graph-operation errors because cleanup is secondary and must
/// not erase a typed Dolt or storage cause needed for retry and user-facing error handling.
fn preserve_graph_error_after_close(
    primary_error: PushGraphTransferError,
    close_result: Result<(), PushGraphTransferError>,
) -> PushGraphTransferError {
    match close_result {
        Err(PushGraphTransferError::StorageFailure(failure)) => PushGraphTransferError::Storage {
            primary: Box::new(primary_error),
            failure,
        },
        Err(cleanup) => PushGraphTransferError::Cleanup {
            primary: Box::new(primary_error),
            cleanup: safe_graph_error_context(&cleanup),
        },
        Ok(()) => primary_error,
    }
}

fn close_server_after_graph_failure(
    server: RemoteServer,
    progress: &GraphUploadProgressReporter,
    primary_error: PushGraphTransferError,
) -> PushGraphTransferError {
    progress.waiting_for_local_server_cleanup();
    preserve_graph_error_after_close(
        primary_error,
        close_direct_push_server(server, progress, true),
    )
}

struct DirectGraphPushRequest<'a> {
    graph: &'a GraphConnection,
    remote: &'a Remote,
    branch: &'a str,
    force: bool,
    destination_hash: &'a DoltHashId,
    renewal: &'a PushCapabilityRenewal,
    attempt: usize,
    progress: &'a TransferProgress,
}

fn push_graph_through_direct_session(
    request: DirectGraphPushRequest<'_>,
) -> Result<(), PushGraphTransferError> {
    let graph_progress = request.progress.graph_reporter(request.attempt, 2);
    let result = push_graph_through_direct_session_inner(request, &graph_progress);
    if let Err(error) = &result {
        if error.is_non_fast_forward() {
            graph_progress.failed_silently();
        } else {
            graph_progress.failed();
        }
    }
    result
}

/// GenHub must keep uploaded blocks private until the requested branch is validated and staged.
///
/// `push_graph_through_direct_session` calls this for each authorized attempt; its checkpoint is
/// still unpublished until the outer coordinator receives success.
fn push_graph_through_direct_session_inner(
    request: DirectGraphPushRequest<'_>,
    graph_progress: &GraphUploadProgressReporter,
) -> Result<(), PushGraphTransferError> {
    let DirectGraphPushRequest {
        graph,
        remote,
        branch,
        force,
        destination_hash,
        renewal,
        attempt,
        progress,
    } = request;
    let response = {
        let _heartbeat = progress.heartbeat("fetching direct-push session capability");
        renewal.capability_response()?
    };
    let capability = response.direct_push.as_ref().ok_or_else(|| {
        PushGraphTransferError::Protocol(
            "GenHub push capability did not include a direct GCS session".to_string(),
        )
    })?;
    if capability.session_id != renewal.expected_session_id {
        return Err(PushGraphTransferError::Protocol(
            "GenHub returned a direct-push session that does not match the idempotency token"
                .to_string(),
        ));
    }
    let scope = SessionScope::new(
        &capability.session_scope.principal,
        &capability.session_scope.target_database,
        &capability.session_scope.operations,
    )
    .map_err(|error| PushGraphTransferError::database("creating session scope", error))?;
    let operation_id = SessionOperationId::from_request("POST", "/default.db/commit", b"")
        .map_err(|error| {
            PushGraphTransferError::database("creating session operation ID", error)
        })?;
    progress.println(&format!(
        "Opening direct GCS graph session (attempt {attempt}/2)..."
    ));
    let renewal_callback = renewal.clone();
    let graph_progress_callback = graph_progress.clone();
    let session =
        BlockCacheSessionOptions::for_uri(capability.session_id.to_string(), scope, operation_id)
            .map_err(|error| PushGraphTransferError::database("attaching GCS session", error))?
            .upload_concurrency(DIRECT_GCS_UPLOAD_CONCURRENCY)
            .auth_callback(move |_storage, _account, _container, reason| {
                renewal_callback.auth_token(reason)
            })
            .upload_progress_callback(move |progress| graph_progress_callback.report(progress));
    let options = RemoteServerOptions::new().blockcache_session(session);
    let connection_uri = CloudConnectionUri::parse(&capability.database_uri).map_err(|_| {
        PushGraphTransferError::Protocol(
            "GenHub returned an invalid direct GCS database URI".to_string(),
        )
    })?;
    if connection_uri.gcs_access_token().is_none() {
        return Err(PushGraphTransferError::Protocol(
            "GenHub returned an invalid direct GCS database URI".to_string(),
        ));
    }
    let mut local_server = {
        let _heartbeat = progress.heartbeat("opening the direct GCS graph database");
        RemoteServer::start_with_options(Path::new(connection_uri.as_str()), &options)
            .map_err(|error| PushGraphTransferError::database("opening direct GCS server", error))?
    };
    let operation_status_result = {
        let _heartbeat = progress.heartbeat("checking direct-push session status");
        local_server.operation_status()
    };
    let operation_status = match operation_status_result {
        Ok(operation_status) => operation_status,
        Err(error) => {
            return Err(close_server_after_graph_failure(
                local_server,
                graph_progress,
                PushGraphTransferError::database("reading direct-push session status", error),
            ));
        }
    };
    match operation_status {
        SessionOperationStatus::Accepted | SessionOperationStatus::Committed => {
            progress.println("Reusing accepted graph checkpoint; verifying the staged branch...");
            let validation = {
                let _heartbeat = progress.heartbeat("validating the staged graph branch");
                validate_direct_session_branch(&local_server, branch, destination_hash)
            };
            if let Err(error) = validation {
                graph_progress.finish();
                return Err(close_server_after_graph_failure(
                    local_server,
                    graph_progress,
                    error,
                ));
            }
            graph_progress.finish();
            close_direct_push_server(local_server, graph_progress, false)?;
            progress.println("Accepted graph checkpoint is ready for publication.");
        }
        SessionOperationStatus::New => {
            progress.println(&format!(
                "Staging graph database to GCS (attempt {attempt}/2)..."
            ));
            let local_remote_url =
                local_server.database_url(&capability.session_scope.target_database);
            let configure_remote = {
                let _heartbeat = progress.heartbeat("configuring the loopback Dolt remote");
                ensure_graph_remote(graph, &remote.name, &local_remote_url)
            };
            if let Err(error) = configure_remote {
                graph_progress.finish();
                return Err(close_server_after_graph_failure(
                    local_server,
                    graph_progress,
                    PushGraphTransferError::database("configuring loopback Dolt remote", error),
                ));
            }
            log::debug!(
                "Planning destination-missing Dolt chunks before the first logical chunk upload...",
            );
            let push_result = {
                let _heartbeat =
                    progress.heartbeat("running Dolt push through loopback RemoteServer");
                push_graph_branch_with_progress(graph, &remote.name, branch, force, graph_progress)
            };
            if let Err(error) = push_result {
                graph_progress.finish();
                let push_error = if is_non_fast_forward_push_error(&error) {
                    PushGraphTransferError::NonFastForward
                } else {
                    PushGraphTransferError::database(
                        "running Dolt push through loopback RemoteServer",
                        error,
                    )
                };
                return Err(close_server_after_graph_failure(
                    local_server,
                    graph_progress,
                    push_error,
                ));
            }
            let validation = {
                let _heartbeat = progress.heartbeat("validating the staged graph branch");
                validate_direct_session_branch(&local_server, branch, destination_hash)
            };
            if let Err(error) = validation {
                graph_progress.finish();
                return Err(close_server_after_graph_failure(
                    local_server,
                    graph_progress,
                    error,
                ));
            }
            let stage_result = {
                let _heartbeat =
                    progress.heartbeat("staging graph blocks and checkpointing the session");
                local_server.stage_request().map_err(|error| {
                    PushGraphTransferError::database("staging direct-push operation", error)
                })
            };
            if let Err(primary_error) = stage_result {
                return Err(close_server_after_graph_failure(
                    local_server,
                    graph_progress,
                    primary_error,
                ));
            }
            let close_result = close_direct_push_server(local_server, graph_progress, false);
            graph_progress.finish();
            close_result?;
            progress.println("Graph staging complete.");
        }
        SessionOperationStatus::Failed => {
            graph_progress.finish();
            return Err(close_server_after_graph_failure(
                local_server,
                graph_progress,
                PushGraphTransferError::Protocol(
                    "GenHub direct-push session has already failed".to_string(),
                ),
            ));
        }
        SessionOperationStatus::Conflict => {
            graph_progress.finish();
            return Err(close_server_after_graph_failure(
                local_server,
                graph_progress,
                PushGraphTransferError::Protocol(
                    "GenHub direct-push session conflicts with another operation".to_string(),
                ),
            ));
        }
    }
    progress.println("Publishing graph manifest...");
    {
        let _heartbeat = progress.heartbeat("publishing the graph manifest to GenHub");
        renewal.publish()?;
    }
    progress.println("Graph manifest published.");
    Ok(())
}

/// A resumed push must not publish an accepted checkpoint for a different local branch head.
///
/// `push_graph_through_direct_session_inner` calls this after reopening accepted work and after a
/// new Dolt staging pass before GenHub is allowed to publish its manifest.
fn validate_direct_session_branch(
    local_server: &RemoteServer,
    branch: &str,
    destination_hash: &DoltHashId,
) -> Result<(), PushGraphTransferError> {
    let Some(connection) = local_server.database_connection() else {
        return Err(PushGraphTransferError::Protocol(
            "GenHub direct-push server has no inspectable database connection".to_string(),
        ));
    };
    let staged_hash: DoltHashId = connection
        .query_row("SELECT dolt_hashof(?1)", [branch], |row| row.get(0))
        .map_err(|error| {
            PushGraphTransferError::Protocol(format!(
                "GenHub direct-push session has no valid target branch: {error:?}"
            ))
        })?;
    if staged_hash != *destination_hash {
        return Err(PushGraphTransferError::Protocol(format!(
            "GenHub direct-push session branch '{branch}' does not match the local destination"
        )));
    }
    Ok(())
}

pub(super) struct PushGraphTransferRequest<'a> {
    pub(super) graph: &'a GraphConnection,
    pub(super) remote: &'a Remote,
    pub(super) branch: &'a str,
    pub(super) force: bool,
    pub(super) idempotency_token: Uuid,
    pub(super) expected_transfer_id: Option<Uuid>,
    pub(super) destination_hash: &'a DoltHashId,
    pub(super) progress: &'a TransferProgress,
}

/// The persisted session UUID lets an interrupted push resume without changing its accepted GCS session.
///
/// `execute_push` calls this after recording the operation's session token; it retries authorization
/// and publishes only after the session server validates staging.
pub(super) fn run_push_graph_transfer(
    request: PushGraphTransferRequest<'_>,
) -> Result<PushCapabilityRenewal, Box<dyn Error>> {
    let PushGraphTransferRequest {
        graph,
        remote,
        branch,
        force,
        idempotency_token,
        expected_transfer_id,
        destination_hash,
        progress,
    } = request;
    let mut last_error = None;
    let mut expected_session_identity: Option<PushCapabilityRenewal> = None;
    for attempt in 0..2 {
        if attempt > 0 {
            progress.println("Retrying direct GCS graph transfer (attempt 2/2)...");
        }
        let authorization = {
            let _heartbeat = progress.heartbeat("requesting GenHub direct-push capability");
            if attempt == 0 {
                transfer_authorization(
                    remote,
                    RemoteOperation::Push,
                    Some(branch),
                    force,
                    Some(idempotency_token),
                    login_origin,
                )
            } else {
                transfer_authorization(
                    remote,
                    RemoteOperation::Push,
                    Some(branch),
                    force,
                    Some(idempotency_token),
                    |_| Err("interactive login is unavailable during a push retry".into()),
                )
            }
        }?;
        let Some(capability) = authorization.direct_push.as_ref() else {
            restore_canonical_url(graph, remote);
            return Err(Box::new(PushGraphTransferError::Protocol(
                "GenHub push capability did not include a direct GCS session".to_string(),
            )));
        };
        let response = authorization
            .capability_response
            .clone()
            .ok_or_else(|| "GenHub push capability response was missing".to_string())?;
        if let Some(expected_session_identity) = expected_session_identity.as_ref() {
            expected_session_identity.validate_session_identity(&response)?;
        }
        let expected_transfer_id = expected_transfer_id.unwrap_or(
            authorization
                .push_lease
                .ok_or_else(|| "GenHub push capability did not include a lease".to_string())?
                .transfer_id,
        );
        let renewal = push_capability_renewal(
            remote,
            branch,
            force,
            idempotency_token,
            expected_transfer_id,
            response,
        )?;
        if expected_session_identity.is_none() {
            expected_session_identity = Some(renewal.clone());
        }
        if capability.session_id != idempotency_token {
            restore_canonical_url(graph, remote);
            return Err(Box::new(PushGraphTransferError::Protocol(
                "GenHub returned a direct-push session that does not match the idempotency token"
                    .to_string(),
            )));
        }
        let transfer_result = push_graph_through_direct_session(DirectGraphPushRequest {
            graph,
            remote,
            branch,
            force,
            destination_hash,
            renewal: &renewal,
            attempt: attempt + 1,
            progress,
        });
        match transfer_result {
            Ok(()) => {
                restore_canonical_url(graph, remote);
                return Ok(renewal);
            }
            Err(error) if attempt == 0 && error.is_authorization_error() => {
                restore_canonical_url(graph, remote);
                last_error = Some(error);
            }
            Err(error) if error.is_stale_session() => {
                restore_canonical_url(graph, remote);
                if matches!(
                    &error,
                    PushGraphTransferError::Client(RemoteClientError::StaleGraphSession)
                ) {
                    return Err(Box::new(PushGraphTransferError::StaleSessionConfirmed));
                }
                let confirmation = {
                    let _heartbeat =
                        progress.heartbeat("confirming the stale graph session with GenHub");
                    renewal.confirm_stale_session()
                };
                return match confirmation {
                    Ok(true) => Err(Box::new(PushGraphTransferError::StaleSessionConfirmed)),
                    Ok(false) => Err(Box::new(error)),
                    Err(confirmation) => {
                        Err(Box::new(PushGraphTransferError::StaleSessionCheckFailed {
                            primary: Box::new(error),
                            confirmation: Box::new(confirmation),
                        }))
                    }
                };
            }
            Err(error) => {
                restore_canonical_url(graph, remote);
                return Err(Box::new(error));
            }
        }
    }
    restore_canonical_url(graph, remote);
    Err(Box::new(
        last_error.expect("should retain authorization error"),
    ))
}

#[cfg(test)]
mod tests {
    use rusqlite::{
        Error as SqlError,
        blockcachevfs::{StorageFailure, StorageFailureCause, StorageFailurePhase},
    };

    use super::{preserve_graph_error_after_close, safe_graph_error_context};

    #[test]
    fn test_direct_push_cleanup_error_does_not_replace_primary_sql_error() {
        let primary = super::PushGraphTransferError::database(
            "running Dolt push through loopback RemoteServer",
            SqlError::InvalidQuery,
        );
        let close = Err(super::PushGraphTransferError::database(
            "closing direct GCS server",
            SqlError::SqliteFailure(
                rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_IOERR),
                Some("signed-url=secret access_token=secret".to_string()),
            ),
        ));

        let cleanup_context = safe_graph_error_context(close.as_ref().expect_err("should fail"));
        let error = preserve_graph_error_after_close(primary, close);

        let super::PushGraphTransferError::Cleanup { primary, cleanup } = &error else {
            panic!("should retain the primary and cleanup errors together: {error}");
        };
        assert!(matches!(
            primary.as_ref(),
            super::PushGraphTransferError::Database {
                phase: "running Dolt push through loopback RemoteServer",
                source: SqlError::InvalidQuery,
            }
        ));
        assert_eq!(cleanup, &cleanup_context);
        assert!(cleanup.contains("closing direct GCS server"));
        assert!(cleanup_context.contains("SQLite code"));
        assert!(!cleanup_context.contains("secret"));
        assert!(!error.to_string().contains("access_token"));
        assert!(!error.to_string().contains("signed"));
    }

    #[test]
    fn test_direct_push_cleanup_preserves_typed_storage_failure_without_provider_details() {
        let primary = super::PushGraphTransferError::database(
            "running Dolt push through loopback RemoteServer",
            SqlError::InvalidQuery,
        );
        let failure = StorageFailure {
            phase: StorageFailurePhase::StageBlockPut,
            cause: StorageFailureCause::HttpStatus(403),
        };
        let error = preserve_graph_error_after_close(
            primary,
            Err(super::PushGraphTransferError::StorageFailure(failure)),
        );

        let super::PushGraphTransferError::Storage {
            primary,
            failure: actual_failure,
        } = &error
        else {
            panic!("should retain the primary and typed storage failure: {error}");
        };
        assert!(matches!(
            primary.as_ref(),
            super::PushGraphTransferError::Database {
                phase: "running Dolt push through loopback RemoteServer",
                source: SqlError::InvalidQuery,
            }
        ));
        assert_eq!(*actual_failure, failure);
        assert!(
            error
                .to_string()
                .contains("staged block upload failed with HTTP 403")
        );
    }

    #[test]
    fn test_stale_session_classifier_requires_exact_sqlite_busy_error() {
        let exact = SqlError::SqliteFailure(
            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_BUSY),
            Some("published manifest changed since the accepted session head".to_string()),
        );
        let other_busy = SqlError::SqliteFailure(
            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_BUSY),
            Some("database is locked".to_string()),
        );
        let other_code = SqlError::SqliteFailure(
            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_IOERR),
            Some("published manifest changed since the accepted session head".to_string()),
        );

        assert!(super::is_stale_accepted_session_error(&exact));
        assert!(!super::is_stale_accepted_session_error(&other_busy));
        assert!(!super::is_stale_accepted_session_error(&other_code));
    }
}
