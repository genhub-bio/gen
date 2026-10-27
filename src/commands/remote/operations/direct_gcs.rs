//! Session-owned direct-GCS graph staging for GenHub pushes.
//!
//! A push stages Dolt blocks through an owned CloudSession and validates the requested branch
//! against the local destination. GenHub publishes the manifest only after staging succeeds.
//! Accepted checkpoints can be reused after interruption; a server-confirmed stale checkpoint is
//! surfaced to the operation coordinator so it can begin a fresh session. Cleanup keeps the
//! original Dolt/storage failure visible when closing the cloud session also fails.

use std::error::Error;

use gen_core::DoltHashId;
use gen_models::{db::GraphConnection, operations::Remote};
use rusqlite::{
    BlockCacheSessionOptions, CloudSession, SessionOperationId, SessionOperationStatus,
    SessionScope, blockcachevfs::CloudConnectionUri,
};

#[cfg(not(all(feature = "remote-native", not(target_os = "emscripten"))))]
use super::push_graph_branch;
#[cfg(all(feature = "remote-native", not(target_os = "emscripten")))]
use super::push_graph_branch_with_progress;
use super::{
    PushGraphTransferRequest,
    capability::{
        PushCapabilityRenewal, push_capability_renewal, transfer_authorization,
        wrap_push_authorization_error,
    },
    ensure_graph_remote, is_non_fast_forward_push_error,
    push_session::{PushGraphTransferError, safe_graph_error_context},
    restore_canonical_url,
};
#[cfg(all(feature = "remote-native", not(target_os = "emscripten")))]
use crate::commands::remote::progress::GraphUploadProgressReporter;
use crate::commands::remote::{
    client::{RemoteClientError, RemoteOperation},
    login_origin,
    progress::TransferProgress,
};

const DIRECT_GCS_UPLOAD_CONCURRENCY: u32 = 10;

#[cfg(test)]
fn is_stale_accepted_session_error(error: &rusqlite::Error) -> bool {
    error.remote_failure_kind() == Some(rusqlite::RemoteFailureKind::StaleSession)
}

/// Closes a cloud session after Dolt has finished using its ordinary file remote.
///
/// `push_graph_through_direct_session_inner` calls this after staging, validation, and error paths
/// so GenHub publishes only after the session has released its database and storage resources.
fn close_direct_push_session(
    mut session: CloudSession,
    progress: &TransferProgress,
    preserve_phase: bool,
) -> Result<(), PushGraphTransferError> {
    let _heartbeat = progress.heartbeat(if preserve_phase {
        "closing the direct GCS graph session after failure"
    } else {
        "closing the direct GCS graph session"
    });
    let close_result = match session.close() {
        Ok(()) => Ok(()),
        Err(first_error) => match session.close() {
            Ok(()) => Ok(()),
            Err(retry_error) => {
                let primary =
                    PushGraphTransferError::database("closing direct GCS session", first_error);
                let retry = PushGraphTransferError::database(
                    "retrying direct GCS session close",
                    retry_error,
                );
                Err(PushGraphTransferError::Cleanup {
                    primary: Box::new(primary),
                    cleanup: safe_graph_error_context(&retry),
                })
            }
        },
    };
    let storage_failure = session.first_storage_error();
    match (close_result, storage_failure) {
        (Err(error), Some(failure)) => Err(PushGraphTransferError::Storage {
            primary: Box::new(error),
            failure,
        }),
        (Err(error), None) => Err(error),
        (Ok(()), Some(failure)) => Err(PushGraphTransferError::StorageFailure(failure)),
        (Ok(()), None) => Ok(()),
    }
}

/// Keeps the push failure primary when cloud-session cleanup fails afterward.
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

fn close_session_after_graph_failure(
    session: CloudSession,
    progress: &TransferProgress,
    primary_error: PushGraphTransferError,
) -> PushGraphTransferError {
    preserve_graph_error_after_close(
        primary_error,
        close_direct_push_session(session, progress, true),
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

#[cfg(all(feature = "remote-native", not(target_os = "emscripten")))]
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

#[cfg(not(all(feature = "remote-native", not(target_os = "emscripten"))))]
fn push_graph_through_direct_session(
    request: DirectGraphPushRequest<'_>,
) -> Result<(), PushGraphTransferError> {
    push_graph_through_direct_session_inner(request)
}

/// GenHub must keep uploaded blocks private until the requested branch is validated and staged.
///
/// `push_graph_through_direct_session` calls this for each authorized attempt; its checkpoint is
/// still unpublished until the outer coordinator receives success.
fn push_graph_through_direct_session_inner(
    request: DirectGraphPushRequest<'_>,
    #[cfg(all(feature = "remote-native", not(target_os = "emscripten")))]
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
    let session_options =
        BlockCacheSessionOptions::for_uri(capability.session_id.to_string(), scope, operation_id)
            .map_err(|error| PushGraphTransferError::database("attaching GCS session", error))?
            .upload_concurrency(DIRECT_GCS_UPLOAD_CONCURRENCY)
            .auth_callback(move |_storage, _project, _container, reason| {
                renewal_callback.auth_token(reason)
            });
    #[cfg(all(feature = "remote-native", not(target_os = "emscripten")))]
    let session_options = session_options.upload_progress_callback({
        let graph_progress_callback = graph_progress.clone();
        move |progress| graph_progress_callback.report(progress)
    });
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
    let mut session = {
        let _heartbeat = progress.heartbeat("opening the direct GCS graph database");
        session_options
            .open_cloud_uri(connection_uri.as_str())
            .map_err(|error| {
                PushGraphTransferError::database("opening direct GCS session", error)
            })?
    };
    let operation_status_result = {
        let _heartbeat = progress.heartbeat("checking direct-push session status");
        session.operation_status()
    };
    let operation_status = match operation_status_result {
        Ok(operation_status) => operation_status,
        Err(error) => {
            return Err(close_session_after_graph_failure(
                session,
                progress,
                PushGraphTransferError::database("reading direct-push session status", error),
            ));
        }
    };
    match operation_status {
        SessionOperationStatus::Accepted | SessionOperationStatus::Committed => {
            progress.println("Reusing accepted graph checkpoint; verifying the staged branch...");
            let validation = {
                let _heartbeat = progress.heartbeat("validating the staged graph branch");
                validate_direct_session_branch(&session, branch, destination_hash)
            };
            if let Err(error) = validation {
                #[cfg(all(feature = "remote-native", not(target_os = "emscripten")))]
                graph_progress.finish();
                return Err(close_session_after_graph_failure(session, progress, error));
            }
            #[cfg(all(feature = "remote-native", not(target_os = "emscripten")))]
            graph_progress.finish();
            close_direct_push_session(session, progress, false)?;
            progress.println("Accepted graph checkpoint is ready for publication.");
        }
        SessionOperationStatus::New => {
            progress.println(&format!(
                "Staging graph database to GCS (attempt {attempt}/2)..."
            ));
            let local_remote_url = session.database_url().to_string();
            let configure_remote = {
                let _heartbeat = progress.heartbeat("configuring the direct cloud Dolt remote");
                ensure_graph_remote(graph, &remote.name, &local_remote_url)
            };
            if let Err(error) = configure_remote {
                #[cfg(all(feature = "remote-native", not(target_os = "emscripten")))]
                graph_progress.finish();
                return Err(close_session_after_graph_failure(
                    session,
                    progress,
                    PushGraphTransferError::database("configuring direct cloud Dolt remote", error),
                ));
            }
            log::debug!(
                "Planning destination-missing Dolt chunks before the first logical chunk upload...",
            );
            let push_result = {
                #[cfg(all(feature = "remote-native", not(target_os = "emscripten")))]
                {
                    let _heartbeat = graph_progress.heartbeat("Dolt push to the direct GCS remote");
                    push_graph_branch_with_progress(
                        graph,
                        &remote.name,
                        branch,
                        force,
                        graph_progress,
                    )
                }
                #[cfg(not(all(feature = "remote-native", not(target_os = "emscripten"))))]
                {
                    let _heartbeat = progress.heartbeat("Dolt push to the direct GCS remote");
                    push_graph_branch(graph, &remote.name, branch, force)
                }
            };
            if let Err(error) = push_result {
                #[cfg(all(feature = "remote-native", not(target_os = "emscripten")))]
                graph_progress.finish();
                let push_error = if is_non_fast_forward_push_error(&error) {
                    PushGraphTransferError::NonFastForward
                } else {
                    PushGraphTransferError::database(
                        "running Dolt push to direct cloud remote",
                        error,
                    )
                };
                return Err(close_session_after_graph_failure(
                    session, progress, push_error,
                ));
            }
            let validation = {
                let _heartbeat = progress.heartbeat("validating the staged graph branch");
                validate_direct_session_branch(&session, branch, destination_hash)
            };
            if let Err(error) = validation {
                #[cfg(all(feature = "remote-native", not(target_os = "emscripten")))]
                graph_progress.finish();
                return Err(close_session_after_graph_failure(session, progress, error));
            }
            let stage_result = {
                let _heartbeat =
                    progress.heartbeat("staging graph blocks and accepting the session");
                session.stage_request().map_err(|error| {
                    PushGraphTransferError::database("staging direct-push operation", error)
                })
            };
            if let Err(primary_error) = stage_result {
                #[cfg(all(feature = "remote-native", not(target_os = "emscripten")))]
                graph_progress.finish();
                return Err(close_session_after_graph_failure(
                    session,
                    progress,
                    primary_error,
                ));
            }
            let close_result = close_direct_push_session(session, progress, false);
            #[cfg(all(feature = "remote-native", not(target_os = "emscripten")))]
            graph_progress.finish();
            close_result?;
            progress.println("Graph staging complete.");
        }
        SessionOperationStatus::Failed => {
            return Err(close_session_after_graph_failure(
                session,
                progress,
                PushGraphTransferError::Protocol(
                    "GenHub direct-push session has already failed".to_string(),
                ),
            ));
        }
        SessionOperationStatus::Conflict => {
            return Err(close_session_after_graph_failure(
                session,
                progress,
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
/// `push_graph_through_direct_session_inner` calls this after reopening accepted work and after
/// a new Dolt staging pass before GenHub is allowed to publish its manifest.
fn validate_direct_session_branch(
    session: &CloudSession,
    branch: &str,
    destination_hash: &DoltHashId,
) -> Result<(), PushGraphTransferError> {
    let connection = session
        .database_connection()
        .map_err(|error| PushGraphTransferError::database("opening direct GCS database", error))?;
    let staged_hash: DoltHashId = connection
        .query_row("SELECT dolt_hashof(?1)", [branch], |row| row.get(0))
        .map_err(|_| {
            PushGraphTransferError::Protocol(
                "GenHub direct-push session has no valid target branch".to_string(),
            )
        })?;
    if staged_hash != *destination_hash {
        return Err(PushGraphTransferError::Protocol(format!(
            "GenHub direct-push session branch '{branch}' does not match the local destination"
        )));
    }
    Ok(())
}

/// The persisted session UUID lets an interrupted push resume without changing its accepted GCS session.
///
/// `execute_push` calls this after recording the operation's session token; it retries authorization
/// and publishes only after the session validates staging.
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
        let authorization_result = {
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
        };
        let authorization = authorization_result.map_err(wrap_push_authorization_error)?;
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
                if error.is_non_fast_forward()
                    && let Err(abort_error) = {
                        let _heartbeat =
                            progress.heartbeat("releasing the rejected GenHub push lease");
                        renewal.abort()
                    }
                {
                    return Err(Box::new(PushGraphTransferError::Abort {
                        primary: Box::new(error),
                        abort: Box::new(abort_error),
                    }));
                }
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
    use gen_http::StatusCode;
    use rusqlite::{
        Error as SqlError,
        blockcachevfs::{StorageFailure, StorageFailureCause, StorageFailurePhase},
    };

    use super::{
        PushGraphTransferError, is_stale_accepted_session_error, preserve_graph_error_after_close,
        safe_graph_error_context,
    };

    #[test]
    fn test_direct_push_cleanup_error_does_not_replace_primary_sql_error() {
        let primary = super::PushGraphTransferError::database(
            "running Dolt push to direct cloud remote",
            SqlError::InvalidQuery,
        );
        let close = Err(super::PushGraphTransferError::database(
            "closing direct GCS session",
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
                phase: "running Dolt push to direct cloud remote",
                source: SqlError::InvalidQuery,
            }
        ));
        assert_eq!(cleanup, &cleanup_context);
        assert!(cleanup.contains("closing direct GCS session"));
        assert!(cleanup_context.contains("SQLite code"));
        assert!(!cleanup_context.contains("secret"));
        assert!(!error.to_string().contains("access_token"));
        assert!(!error.to_string().contains("signed"));
    }

    #[test]
    fn test_direct_push_cleanup_preserves_typed_storage_failure_without_provider_details() {
        let primary = super::PushGraphTransferError::database(
            "running Dolt push to direct cloud remote",
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
                phase: "running Dolt push to direct cloud remote",
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
    fn test_non_fast_forward_remains_terminal_when_lease_abort_fails() {
        let error = PushGraphTransferError::Abort {
            primary: Box::new(PushGraphTransferError::NonFastForward),
            abort: Box::new(PushGraphTransferError::Client(
                crate::commands::remote::client::RemoteClientError::Http {
                    status: StatusCode::SERVICE_UNAVAILABLE,
                    message: "direct GCS push lease abort failed".to_string(),
                },
            )),
        };

        assert!(error.is_non_fast_forward());
        assert!(error.is_terminal());
        assert!(
            error
                .non_fast_forward_cleanup_warning()
                .contains("GenHub could not release the rejected push lease"),
            "a failed abort should be shown beside the original non-fast-forward result"
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

        assert!(is_stale_accepted_session_error(&exact));
        assert!(!is_stale_accepted_session_error(&other_busy));
        assert!(!is_stale_accepted_session_error(&other_code));
    }
}
