//! Browser-based login for `target_os = "emscripten"`.
//!
//! The native `login_origin` (see `super::login_origin`) starts a localhost callback server and
//! opens the system browser; neither is available here (no TCP sockets, no system browser
//! process to open), so this module hands the interactive part off to the terminal's host page
//! instead:
//!
//! 1. This process writes a sentinel-framed "begin" message to stdout. The host page's
//!    main-thread JS (wasm-cli's `index.ts`, which already intercepts every byte of this
//!    process's stdout via cockle's `outputCallback` before it reaches the visible terminal)
//!    recognizes the sentinel, strips it from what's displayed, and takes over: it attempts to
//!    open the GenHub login URL in a popup, falling back to a clickable link if the popup is
//!    blocked, then waits for the dedicated callback page to deliver a result via `postMessage`.
//! 2. Once the host page has a result, it writes a sentinel-framed "result" message into this
//!    process's stdin -- the same transport `views::emscripten_input` already uses for keyboard
//!    and mouse input in `gen view`, which blocks (via cockle's `SharedArrayBufferMainIO` and
//!    `Atomics.wait`) without a busy-loop. This process is already blocked reading stdin waiting
//!    for that message, so no new suspension mechanism -- and no Asyncify -- is needed here.
//! 3. This process parses the result, validates the state nonce against the value it generated
//!    for this attempt, and saves tokens through the existing `utils::save_tokens` path exactly
//!    as the native flow does.
//!
//! The sentinel framing (`BEGIN_SENTINEL`/`RESULT_SENTINEL`) exists only so the host page's JS
//! can find these out-of-band messages inside the same byte stream as ordinary terminal I/O; it
//! carries no security meaning of its own. The state nonce comparison in
//! `parse_bridge_message` is what actually authenticates the callback -- exactly as it does in
//! the native flow's `server::start_callback_server`.

#[cfg(target_os = "emscripten")]
use core::error::Error;
#[cfg(target_os = "emscripten")]
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::from_str as json_from_str;
use url::Url;

use crate::commands::remote::server::AuthTokens;
#[cfg(target_os = "emscripten")]
use crate::{
    commands::remote::utils,
    views::emscripten_input::{read_available, stdin_ready},
};

/// Marks the start of a login "begin" message written to stdout for the host page to intercept.
pub const BEGIN_SENTINEL: &str = "\u{0}GEN_LOGIN_BEGIN\u{0}";
/// Marks the start of a login "result" message the host page writes back to stdin.
pub const RESULT_SENTINEL: &str = "\u{0}GEN_LOGIN_RESULT\u{0}";
/// Terminates a sentinel-framed message on either stream.
pub const MESSAGE_TERMINATOR: char = '\u{0}';

#[cfg(target_os = "emscripten")]
const STDIN_POLL_INTERVAL: Duration = Duration::from_millis(200);
#[cfg(target_os = "emscripten")]
const LOGIN_TIMEOUT: Duration = Duration::from_secs(300);

/// The JupyterLite/Cockle base path the terminal is deployed under (e.g. `/lab`, or empty for a
/// root deployment). Read from an environment variable the host page's JS is expected to set
/// (via Emscripten's `ENV`, matching the `-sEXPORTED_RUNTIME_METHODS=ENV` link flag the wasm
/// build already uses) before running a command that may need to log in, so this never assumes
/// the deployment lives at `/`.
#[cfg(target_os = "emscripten")]
fn jupyterlite_base_path() -> String {
    std::env::var("GEN_JUPYTERLITE_BASE_PATH").unwrap_or_default()
}

/// The terminal's own origin, set by the host page for the same reason as
/// `jupyterlite_base_path`. Used both to build the callback URL and as the exact origin the host
/// page must validate incoming `postMessage` results against.
#[cfg(target_os = "emscripten")]
fn terminal_origin() -> Result<String, Box<dyn Error>> {
    std::env::var("GEN_TERMINAL_ORIGIN").map_err(|_| {
        "GEN_TERMINAL_ORIGIN is not set; the host page must set it before gen runs".into()
    })
}

/// Builds the dedicated browser callback page URL from the terminal's origin and JupyterLite
/// base path, e.g. `https://example.com/lab/gen-login-callback.html`. Never assumes the
/// deployment lives at `/`.
pub fn build_callback_url(terminal_origin: &str, base_path: &str) -> String {
    let trimmed_origin = terminal_origin.trim_end_matches('/');
    let trimmed_base = base_path.trim_matches('/');
    if trimmed_base.is_empty() {
        format!("{trimmed_origin}/gen-login-callback.html")
    } else {
        format!("{trimmed_origin}/{trimmed_base}/gen-login-callback.html")
    }
}

/// Builds the GenHub CLI login URL. Identical in shape to the native flow's request (same
/// `redirect_uri` pointing at the existing `/api/auth/cli/callback` endpoint, same `state`),
/// except `redirect_to` points at the browser callback page instead of a localhost address.
pub fn build_login_url(
    origin: &str,
    state: &str,
    callback_url: &str,
) -> Result<String, url::ParseError> {
    let mut url = Url::parse(&format!("{origin}/api/auth/cli/login/"))?;
    url.query_pairs_mut()
        .append_pair("redirect_uri", &format!("{origin}/api/auth/cli/callback"))
        .append_pair("state", state)
        .append_pair("redirect_to", callback_url);
    Ok(url.to_string())
}

/// The result the host page's JS bridge reports back over stdin, one message per login attempt.
///
/// Only decoded by `receive_login_result` (`target_os = "emscripten"`) outside of tests, so a
/// native, non-test build sees no caller and would otherwise warn this whole parse chain as dead
/// code.
#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(tag = "status", rename_all = "snake_case")]
#[cfg_attr(not(target_os = "emscripten"), allow(dead_code))]
enum BridgeOutcome {
    Success {
        state: String,
        jwt: String,
        refresh_token: String,
    },
    Cancelled,
    Timeout,
    BridgeError {
        message: String,
    },
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum BrowserLoginError {
    #[error("Login was cancelled.")]
    Cancelled,
    #[error("Timed out waiting for the browser login callback.")]
    Timeout,
    #[error("Browser login bridge error: {0}")]
    Bridge(String),
    #[error("State mismatch: the browser callback did not match the pending login attempt.")]
    StateMismatch,
    #[error("Malformed login result from the browser bridge: {0}")]
    MalformedResult(String),
}

/// Parses and validates a "result" message from the host page's JS bridge. This is the sole
/// place that authenticates the callback: it compares the returned state against the nonce
/// generated for the active attempt, exactly as `server::start_callback_server` does for the
/// native flow. A callback is never accepted on the strength of a valid-looking JWT alone.
#[cfg_attr(not(target_os = "emscripten"), allow(dead_code))]
fn parse_bridge_message(json: &str, expected_state: &str) -> Result<AuthTokens, BrowserLoginError> {
    let outcome: BridgeOutcome = json_from_str(json)
        .map_err(|error| BrowserLoginError::MalformedResult(error.to_string()))?;
    match outcome {
        BridgeOutcome::Success {
            state,
            jwt,
            refresh_token,
        } => {
            if state != expected_state {
                return Err(BrowserLoginError::StateMismatch);
            }
            if jwt.is_empty() {
                return Err(BrowserLoginError::MalformedResult(
                    "missing jwt".to_string(),
                ));
            }
            if refresh_token.is_empty() {
                return Err(BrowserLoginError::MalformedResult(
                    "missing refresh_token".to_string(),
                ));
            }
            Ok(AuthTokens { jwt, refresh_token })
        }
        BridgeOutcome::Cancelled => Err(BrowserLoginError::Cancelled),
        BridgeOutcome::Timeout => Err(BrowserLoginError::Timeout),
        BridgeOutcome::BridgeError { message } => Err(BrowserLoginError::Bridge(message)),
    }
}

/// Finds the first complete `sentinel`-prefixed, terminator-delimited message in `buffer`, if
/// any. Pure byte scanning; used both by the real stdin-polling loop below and directly by unit
/// tests, so the framing logic itself needs no emscripten target or real stdin to exercise.
#[cfg_attr(not(target_os = "emscripten"), allow(dead_code))]
fn extract_sentinel_message(buffer: &[u8], sentinel: &str) -> Option<String> {
    let text = String::from_utf8_lossy(buffer);
    let start = text.find(sentinel)? + sentinel.len();
    let rest = &text[start..];
    let end = rest.find(MESSAGE_TERMINATOR)?;
    Some(rest[..end].to_string())
}

/// Blocks (via the same non-busy-waiting stdin mechanism `emscripten_input` uses) until a
/// complete `sentinel`-framed message arrives or `timeout` elapses.
#[cfg(target_os = "emscripten")]
fn read_sentinel_message(sentinel: &str, timeout: Duration) -> Option<String> {
    let deadline = Instant::now() + timeout;
    let mut buffer = Vec::new();
    loop {
        if let Some(message) = extract_sentinel_message(&buffer, sentinel) {
            return Some(message);
        }
        if Instant::now() >= deadline {
            return None;
        }
        if stdin_ready(STDIN_POLL_INTERVAL) {
            buffer.extend(read_available());
        }
    }
}

/// The exact callback page URL, if the host page's JS already computed it (it knows its actual
/// deployed path, including any JupyterLite base path or routing, better than this process
/// could reconstruct from parts). Falls back to joining `terminal_origin()` with
/// `jupyterlite_base_path()` when the host page has not set it.
#[cfg(target_os = "emscripten")]
fn callback_url(terminal_origin: &str) -> String {
    std::env::var("GEN_LOGIN_CALLBACK_URL")
        .unwrap_or_else(|_| build_callback_url(terminal_origin, &jupyterlite_base_path()))
}

/// Logs in via the browser flow described at the top of this module.
#[cfg(target_os = "emscripten")]
pub fn login_origin(origin: &str) -> Result<AuthTokens, Box<dyn Error>> {
    let state = utils::generate_state().expect("should generate random nonce");
    let terminal_origin = terminal_origin()?;
    let callback_url = callback_url(&terminal_origin);
    let login_url = build_login_url(origin, &state, &callback_url)?;

    println!("Logging in to remote: {origin}");
    let begin_payload = serde_json::json!({
        "login_url": login_url,
        "callback_url": callback_url,
        "expected_state": state,
        "expected_origin": terminal_origin,
    });
    println!("{BEGIN_SENTINEL}{begin_payload}{MESSAGE_TERMINATOR}");

    let message =
        read_sentinel_message(RESULT_SENTINEL, LOGIN_TIMEOUT).ok_or(BrowserLoginError::Timeout)?;
    Ok(parse_bridge_message(&message, &state)?)
}

#[cfg(test)]
mod tests {
    mod callback_url {
        use super::super::build_callback_url;

        #[test]
        fn test_build_callback_url_with_base_path() {
            assert_eq!(
                build_callback_url("https://terminal.example.com", "/lab/"),
                "https://terminal.example.com/lab/gen-login-callback.html"
            );
        }

        #[test]
        fn test_build_callback_url_at_root() {
            assert_eq!(
                build_callback_url("https://terminal.example.com", ""),
                "https://terminal.example.com/gen-login-callback.html"
            );
        }

        #[test]
        fn test_build_callback_url_trims_trailing_slash_on_origin() {
            assert_eq!(
                build_callback_url("https://terminal.example.com/", "base"),
                "https://terminal.example.com/base/gen-login-callback.html"
            );
        }
    }

    mod login_url {
        use super::super::build_login_url;

        #[test]
        fn test_build_login_url_preserves_callback_endpoint_and_params() {
            let url = build_login_url(
                "https://genhub.bio",
                "state-123",
                "https://terminal.example.com/lab/gen-login-callback.html",
            )
            .unwrap();
            assert!(url.starts_with("https://genhub.bio/api/auth/cli/login/?"));
            assert!(
                url.contains("redirect_uri=https%3A%2F%2Fgenhub.bio%2Fapi%2Fauth%2Fcli%2Fcallback")
            );
            assert!(url.contains("state=state-123"));
            assert!(url.contains(
                "redirect_to=https%3A%2F%2Fterminal.example.com%2Flab%2Fgen-login-callback.html"
            ));
        }

        #[test]
        fn test_build_login_url_rejects_invalid_origin() {
            assert!(build_login_url("not a url", "state", "https://example.com/callback").is_err());
        }
    }

    mod sentinel_framing {
        use super::super::{MESSAGE_TERMINATOR, RESULT_SENTINEL, extract_sentinel_message};

        #[test]
        fn test_extract_sentinel_message_finds_complete_message() {
            let buffer =
                format!("prefix{RESULT_SENTINEL}hello{MESSAGE_TERMINATOR}suffix").into_bytes();
            assert_eq!(
                extract_sentinel_message(&buffer, RESULT_SENTINEL),
                Some("hello".to_string())
            );
        }

        #[test]
        fn test_extract_sentinel_message_returns_none_without_terminator() {
            let buffer = format!("{RESULT_SENTINEL}hello").into_bytes();
            assert_eq!(extract_sentinel_message(&buffer, RESULT_SENTINEL), None);
        }

        #[test]
        fn test_extract_sentinel_message_returns_none_without_sentinel() {
            let buffer = b"just some ordinary typed input".to_vec();
            assert_eq!(extract_sentinel_message(&buffer, RESULT_SENTINEL), None);
        }

        #[test]
        fn test_extract_sentinel_message_ignores_stray_bytes_before_sentinel() {
            let buffer =
                format!("garbage{RESULT_SENTINEL}payload{MESSAGE_TERMINATOR}").into_bytes();
            assert_eq!(
                extract_sentinel_message(&buffer, RESULT_SENTINEL),
                Some("payload".to_string())
            );
        }
    }

    mod bridge_message {
        use super::super::{BrowserLoginError, parse_bridge_message};

        fn success_json(state: &str) -> String {
            format!(
                r#"{{"status":"success","state":"{state}","jwt":"jwt-value","refresh_token":"refresh-value"}}"#
            )
        }

        #[test]
        fn test_parse_bridge_message_accepts_matching_state() {
            let tokens =
                parse_bridge_message(&success_json("expected-state"), "expected-state").unwrap();
            assert_eq!(tokens.jwt, "jwt-value");
            assert_eq!(tokens.refresh_token, "refresh-value");
        }

        #[test]
        fn test_parse_bridge_message_rejects_mismatched_state() {
            assert_eq!(
                parse_bridge_message(&success_json("wrong-state"), "expected-state"),
                Err(BrowserLoginError::StateMismatch)
            );
        }

        #[test]
        fn test_parse_bridge_message_rejects_missing_state_field() {
            let json = r#"{"status":"success","jwt":"jwt-value","refresh_token":"refresh-value"}"#;
            assert!(matches!(
                parse_bridge_message(json, "expected-state"),
                Err(BrowserLoginError::MalformedResult(_))
            ));
        }

        #[test]
        fn test_parse_bridge_message_rejects_missing_jwt() {
            let json = r#"{"status":"success","state":"expected-state","jwt":"","refresh_token":"refresh-value"}"#;
            assert_eq!(
                parse_bridge_message(json, "expected-state"),
                Err(BrowserLoginError::MalformedResult(
                    "missing jwt".to_string()
                ))
            );
        }

        #[test]
        fn test_parse_bridge_message_rejects_missing_refresh_token() {
            let json = r#"{"status":"success","state":"expected-state","jwt":"jwt-value","refresh_token":""}"#;
            assert_eq!(
                parse_bridge_message(json, "expected-state"),
                Err(BrowserLoginError::MalformedResult(
                    "missing refresh_token".to_string()
                ))
            );
        }

        #[test]
        fn test_parse_bridge_message_rejects_malformed_json() {
            assert!(matches!(
                parse_bridge_message("not json at all", "expected-state"),
                Err(BrowserLoginError::MalformedResult(_))
            ));
        }

        #[test]
        fn test_parse_bridge_message_maps_cancelled() {
            let json = r#"{"status":"cancelled"}"#;
            assert_eq!(
                parse_bridge_message(json, "expected-state"),
                Err(BrowserLoginError::Cancelled)
            );
        }

        #[test]
        fn test_parse_bridge_message_maps_timeout() {
            let json = r#"{"status":"timeout"}"#;
            assert_eq!(
                parse_bridge_message(json, "expected-state"),
                Err(BrowserLoginError::Timeout)
            );
        }

        #[test]
        fn test_parse_bridge_message_maps_bridge_error() {
            let json = r#"{"status":"bridge_error","message":"popup closed unexpectedly"}"#;
            assert_eq!(
                parse_bridge_message(json, "expected-state"),
                Err(BrowserLoginError::Bridge(
                    "popup closed unexpectedly".to_string()
                ))
            );
        }
    }
}
