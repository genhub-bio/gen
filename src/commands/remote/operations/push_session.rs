//! Errors and retry state shared by native and browser push sessions.

use gen_http::StatusCode;
#[cfg(any(
    all(feature = "remote-native", not(target_os = "emscripten")),
    feature = "browser-cloud"
))]
use rusqlite::blockcachevfs::StorageFailure;
use rusqlite::{Error as SqlError, RemoteFailureKind};
use thiserror::Error;

use crate::commands::remote::client::RemoteClientError;

#[derive(Debug, Error)]
pub(super) enum PushGraphTransferError {
    #[error("{phase} failed: {source}")]
    Database {
        phase: &'static str,
        #[source]
        source: SqlError,
    },
    #[error(transparent)]
    Client(#[from] RemoteClientError),
    #[error("GenHub confirmed the accepted graph session is stale")]
    StaleSessionConfirmed,
    #[error("GenHub reports another push is in progress; retry after it completes")]
    CapabilityContention,
    #[error("GenHub no longer recognizes the push transfer lease")]
    CapabilityLeaseInactive,
    #[error("The branch is not a fast-forward of the remote. Use --force to overwrite the remote.")]
    NonFastForward,
    #[error("{primary}; GenHub could not verify the stale session: {confirmation}")]
    StaleSessionCheckFailed {
        primary: Box<PushGraphTransferError>,
        confirmation: Box<PushGraphTransferError>,
    },
    #[cfg(any(
        all(feature = "remote-native", not(target_os = "emscripten")),
        feature = "browser-cloud"
    ))]
    #[error("object storage reported {0}")]
    StorageFailure(StorageFailure),
    #[cfg(any(
        all(feature = "remote-native", not(target_os = "emscripten")),
        feature = "browser-cloud"
    ))]
    #[error("{primary}; object storage also reported {failure}")]
    Storage {
        #[source]
        primary: Box<PushGraphTransferError>,
        failure: StorageFailure,
    },
    #[cfg(any(
        all(feature = "remote-native", not(target_os = "emscripten")),
        feature = "browser-cloud"
    ))]
    #[error("{primary}; cloud session cleanup also failed: {cleanup}")]
    Cleanup {
        #[source]
        primary: Box<PushGraphTransferError>,
        cleanup: String,
    },
    #[error("{primary}; GenHub could not release the rejected push lease: {abort}")]
    Abort {
        #[source]
        primary: Box<PushGraphTransferError>,
        abort: Box<PushGraphTransferError>,
    },
    #[error("{0}")]
    Protocol(String),
}

impl PushGraphTransferError {
    pub(super) fn database(phase: &'static str, source: SqlError) -> Self {
        Self::Database { phase, source }
    }

    pub(super) fn is_authorization_error(&self) -> bool {
        match self {
            Self::Database { source, .. } => {
                source.remote_failure_kind() == Some(RemoteFailureKind::Authorization)
            }
            #[cfg(any(
                all(feature = "remote-native", not(target_os = "emscripten")),
                feature = "browser-cloud"
            ))]
            Self::Storage { primary, .. } | Self::Cleanup { primary, .. } => {
                primary.is_authorization_error()
            }
            Self::Abort { primary, .. } => primary.is_authorization_error(),
            #[cfg(any(
                all(feature = "remote-native", not(target_os = "emscripten")),
                feature = "browser-cloud"
            ))]
            Self::StorageFailure(_) => false,
            Self::Client(RemoteClientError::Http { status, .. }) => {
                matches!(*status, StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN)
            }
            Self::Client(_)
            | Self::StaleSessionConfirmed
            | Self::CapabilityContention
            | Self::CapabilityLeaseInactive
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
            | Self::CapabilityContention
            | Self::StaleSessionCheckFailed { .. } => false,
            Self::CapabilityLeaseInactive | Self::Protocol(_) | Self::NonFastForward => true,
            #[cfg(any(
                all(feature = "remote-native", not(target_os = "emscripten")),
                feature = "browser-cloud"
            ))]
            Self::StorageFailure(_) => true,
            #[cfg(any(
                all(feature = "remote-native", not(target_os = "emscripten")),
                feature = "browser-cloud"
            ))]
            Self::Storage { primary, .. } | Self::Cleanup { primary, .. } => primary.is_terminal(),
            Self::Abort { primary, .. } => primary.is_terminal(),
            Self::Database { .. } | Self::Client(_) => false,
        }
    }

    pub(super) fn is_stale_session(&self) -> bool {
        match self {
            Self::Database { source, .. } => {
                source.remote_failure_kind() == Some(RemoteFailureKind::StaleSession)
            }
            #[cfg(any(
                all(feature = "remote-native", not(target_os = "emscripten")),
                feature = "browser-cloud"
            ))]
            Self::Storage { primary, .. } | Self::Cleanup { primary, .. } => {
                primary.is_stale_session()
            }
            Self::Abort { primary, .. } => primary.is_stale_session(),
            #[cfg(any(
                all(feature = "remote-native", not(target_os = "emscripten")),
                feature = "browser-cloud"
            ))]
            Self::StorageFailure(_) => false,
            Self::Client(RemoteClientError::StaleGraphSession) => true,
            Self::Client(_)
            | Self::StaleSessionConfirmed
            | Self::CapabilityContention
            | Self::CapabilityLeaseInactive
            | Self::NonFastForward
            | Self::StaleSessionCheckFailed { .. }
            | Self::Protocol(_) => false,
        }
    }

    pub(super) fn is_non_fast_forward(&self) -> bool {
        match self {
            Self::NonFastForward => true,
            #[cfg(any(
                all(feature = "remote-native", not(target_os = "emscripten")),
                feature = "browser-cloud"
            ))]
            Self::Storage { primary, .. } | Self::Cleanup { primary, .. } => {
                primary.is_non_fast_forward()
            }
            Self::Abort { primary, .. } => primary.is_non_fast_forward(),
            _ => false,
        }
    }

    pub(super) fn non_fast_forward_cleanup_warning(&self) -> String {
        {
            match self {
                Self::Abort { primary, abort } => {
                    let mut warnings = primary.non_fast_forward_cleanup_warning();
                    let abort_warning = format!(
                        "GenHub could not release the rejected push lease: {}",
                        safe_graph_error_context(abort)
                    );
                    append_cleanup_warning(&mut warnings, &abort_warning);
                    warnings
                }
                #[cfg(any(
                    all(feature = "remote-native", not(target_os = "emscripten")),
                    feature = "browser-cloud"
                ))]
                Self::Cleanup { primary, cleanup } if primary.is_non_fast_forward() => {
                    let mut warnings = primary.non_fast_forward_cleanup_warning();
                    append_cleanup_warning(
                        &mut warnings,
                        &format!("cloud session cleanup also failed: {cleanup}"),
                    );
                    warnings
                }
                #[cfg(any(
                    all(feature = "remote-native", not(target_os = "emscripten")),
                    feature = "browser-cloud"
                ))]
                Self::Storage { primary, failure } if primary.is_non_fast_forward() => {
                    let mut warnings = primary.non_fast_forward_cleanup_warning();
                    append_cleanup_warning(
                        &mut warnings,
                        &format!("object storage cleanup also failed: {failure}"),
                    );
                    warnings
                }
                _ => String::new(),
            }
        }
    }
}

/// Hides signed URLs and provider response details from secondary cleanup diagnostics.
pub(super) fn safe_graph_error_context(error: &PushGraphTransferError) -> String {
    match error {
        PushGraphTransferError::Database {
            phase,
            source: SqlError::SqliteFailure(code, _),
        } => format!("{phase} (SQLite code {})", code.extended_code),
        PushGraphTransferError::Database { phase, .. } => {
            format!("{phase} (database error)")
        }
        #[cfg(any(
            all(feature = "remote-native", not(target_os = "emscripten")),
            feature = "browser-cloud"
        ))]
        PushGraphTransferError::StorageFailure(failure) => format!("object storage: {failure}"),
        #[cfg(any(
            all(feature = "remote-native", not(target_os = "emscripten")),
            feature = "browser-cloud"
        ))]
        PushGraphTransferError::Storage { primary, failure } => format!(
            "{}; object storage: {failure}",
            safe_graph_error_context(primary)
        ),
        #[cfg(any(
            all(feature = "remote-native", not(target_os = "emscripten")),
            feature = "browser-cloud"
        ))]
        PushGraphTransferError::Cleanup { primary, cleanup } => format!(
            "{}; additionally {cleanup}",
            safe_graph_error_context(primary)
        ),
        PushGraphTransferError::Abort { primary, abort } => format!(
            "{}; additionally GenHub could not release the rejected push lease: {}",
            safe_graph_error_context(primary),
            safe_graph_error_context(abort),
        ),
        PushGraphTransferError::Client(RemoteClientError::Http { status, .. }) => {
            format!("remote request HTTP {status}")
        }
        PushGraphTransferError::Client(_) => "remote client request".to_string(),
        PushGraphTransferError::StaleSessionConfirmed
        | PushGraphTransferError::StaleSessionCheckFailed { .. } => {
            "stale graph session recovery".to_string()
        }
        PushGraphTransferError::CapabilityContention => {
            "temporary push lease contention".to_string()
        }
        PushGraphTransferError::CapabilityLeaseInactive => {
            "inactive push transfer lease".to_string()
        }
        PushGraphTransferError::NonFastForward => {
            "Dolt rejected a non-fast-forward branch update".to_string()
        }
        PushGraphTransferError::Protocol(_) => "remote protocol response".to_string(),
    }
}

fn append_cleanup_warning(warnings: &mut String, warning: &str) {
    if !warnings.is_empty() {
        warnings.push_str("; ");
    }
    warnings.push_str(warning);
}
