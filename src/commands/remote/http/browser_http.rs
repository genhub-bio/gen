//! Safe wrapper around Emscripten's synchronous Fetch API (see [`super::emscripten_bindings`]).
//!
//! [`request`] (the `target_os = "emscripten"` half, at the bottom of this file) is the only
//! place in this module -- and in the whole `http` boundary -- that touches raw pointers or
//! Emscripten types. Everything above it is pure, target-independent request preparation, kept
//! separate specifically so it can be unit-tested on a native `cargo test` run without an
//! Emscripten toolchain or a fake runtime.

#[cfg(any(test, target_os = "emscripten"))]
use core::ptr;
#[cfg(any(test, target_os = "emscripten"))]
use std::{ffi::CString, os::raw::c_char};

#[cfg(any(test, target_os = "emscripten"))]
use super::BrowserHttpError;
#[cfg(target_os = "emscripten")]
use super::HttpRequest;

/// Converts an HTTP method into the fixed 32-byte buffer `emscripten_fetch_attr_t::requestMethod`
/// expects (it is an inline `char[32]`, not a pointer -- see `emscripten_bindings.rs`). Rejects
/// methods with an embedded null byte or ones too long to fit with a null terminator.
#[cfg(any(test, target_os = "emscripten"))]
pub(super) fn method_buffer(method: &str) -> Result<[c_char; 32], BrowserHttpError> {
    if method.contains('\0') {
        return Err(BrowserHttpError::EmbeddedNullByte { field: "method" });
    }
    let bytes = method.as_bytes();
    // Reserve one byte for the null terminator.
    if bytes.len() >= 32 {
        return Err(BrowserHttpError::InvalidMethod(method.to_string()));
    }
    let mut buffer = [0 as c_char; 32];
    for (index, byte) in bytes.iter().enumerate() {
        buffer[index] = *byte as c_char;
    }
    Ok(buffer)
}

/// Converts request headers into owned, null-terminated C strings. Kept as owned `CString`s
/// (rather than raw pointers) so the caller controls their lifetime explicitly and this function
/// stays safe and independently testable.
#[cfg(any(test, target_os = "emscripten"))]
pub(super) fn header_c_strings(headers: &[(&str, &str)]) -> Result<Vec<CString>, BrowserHttpError> {
    let mut strings = Vec::with_capacity(headers.len() * 2);
    for (name, value) in headers {
        strings.push(
            CString::new(*name).map_err(|_| BrowserHttpError::EmbeddedNullByte {
                field: "header name",
            })?,
        );
        strings.push(
            CString::new(*value).map_err(|_| BrowserHttpError::EmbeddedNullByte {
                field: "header value",
            })?,
        );
    }
    Ok(strings)
}

/// Builds the alternating `{key, value, ..., NULL}` pointer array `requestHeaders` expects,
/// borrowing from (and only valid as long as) `strings`. Returns `None` when there are no
/// headers, since `emscripten_fetch_attr_t::requestHeaders` should stay null in that case rather
/// than point at a lone terminator.
#[cfg(any(test, target_os = "emscripten"))]
pub(super) fn header_pointer_array(strings: &[CString]) -> Option<Vec<*const c_char>> {
    if strings.is_empty() {
        return None;
    }
    let mut pointers: Vec<*const c_char> = strings.iter().map(|value| value.as_ptr()).collect();
    pointers.push(ptr::null());
    Some(pointers)
}

/// Reads a nul-terminated, fixed-size C char buffer (e.g. `emscripten_fetch_t::statusText`) as a
/// lossy UTF-8 `String`, stopping at the first null byte or the end of the buffer.
#[cfg(any(test, target_os = "emscripten"))]
pub(super) fn c_buffer_to_string(buffer: &[c_char]) -> String {
    let length = buffer
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(buffer.len());
    let bytes: Vec<u8> = buffer[..length].iter().map(|byte| *byte as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

#[cfg(target_os = "emscripten")]
mod fetch {
    use core::{mem, ptr, slice};
    use std::ffi::CString;

    use super::{
        BrowserHttpError, HttpRequest, c_buffer_to_string, header_c_strings, header_pointer_array,
        method_buffer,
    };
    use crate::commands::remote::http::{
        HttpResponse,
        emscripten_bindings::{
            EMSCRIPTEN_FETCH_LOAD_TO_MEMORY, EMSCRIPTEN_FETCH_REPLACE,
            EMSCRIPTEN_FETCH_SYNCHRONOUS, EmscriptenFetch, EmscriptenFetchAttr, emscripten_fetch,
            emscripten_fetch_attr_init, emscripten_fetch_close,
        },
    };

    /// Ensures `emscripten_fetch_close` runs exactly once, on every return path (including the
    /// early returns below), for every non-null `emscripten_fetch` result.
    struct FetchGuard(*mut EmscriptenFetch);

    impl Drop for FetchGuard {
        fn drop(&mut self) {
            // SAFETY: `self.0` is non-null (only constructed from a checked, non-null
            // `emscripten_fetch` return value) and this `Drop` impl is the sole place that calls
            // `emscripten_fetch_close` on it, so it is closed exactly once.
            unsafe {
                emscripten_fetch_close(self.0);
            }
        }
    }

    pub fn request(http_request: HttpRequest<'_>) -> Result<HttpResponse, BrowserHttpError> {
        let method = method_buffer(http_request.method)?;
        let url = CString::new(http_request.url)
            .map_err(|_| BrowserHttpError::EmbeddedNullByte { field: "url" })?;
        let header_strings = header_c_strings(http_request.headers)?;
        let header_pointers = header_pointer_array(&header_strings);

        // SAFETY: `emscripten_fetch_attr_init`'s documented contract is that it only writes
        // defaults into every field of `attr`; handing it zeroed memory first is valid because
        // every field of `EmscriptenFetchAttr` (fixed-size char buffers, raw pointers, integers,
        // `bool`, `Option<fn>`) accepts an all-zero bit pattern.
        let mut attr: EmscriptenFetchAttr = unsafe {
            let mut attr = mem::zeroed();
            emscripten_fetch_attr_init(&mut attr);
            attr
        };
        attr.request_method = method;
        attr.attributes = EMSCRIPTEN_FETCH_LOAD_TO_MEMORY
            | EMSCRIPTEN_FETCH_SYNCHRONOUS
            | EMSCRIPTEN_FETCH_REPLACE;
        // `header_pointers` (and the `header_strings` it borrows from) outlive this whole
        // function, which outlives the synchronous `emscripten_fetch` call below.
        attr.request_headers = header_pointers
            .as_ref()
            .map_or(ptr::null(), |pointers| pointers.as_ptr());
        if let Some(body) = http_request.body {
            attr.request_data = body.as_ptr().cast::<i8>();
            attr.request_data_size = body.len();
        }

        let fetch = {
            // SAFETY: `url` is a live `CString` for the duration of this call. `attr`'s pointer
            // fields point at `header_strings`/`http_request.body`, both of which are still alive
            // and unmoved for the whole call because `EMSCRIPTEN_FETCH_SYNCHRONOUS` makes
            // `emscripten_fetch` run the request to completion before returning -- it never
            // retains those pointers past this call.
            unsafe { emscripten_fetch(&mut attr, url.as_ptr()) }
        };
        if fetch.is_null() {
            return Err(BrowserHttpError::FetchStartFailed);
        }
        // From here on `guard` owns `fetch` and will close it on every return path, including
        // the early returns below.
        let guard = FetchGuard(fetch);

        // SAFETY: `guard.0` is non-null (checked above) and was returned by `emscripten_fetch`,
        // whose contract guarantees `status`, `data`, and `numBytes` are populated once the
        // (synchronous) call has returned; `guard` keeps the pointee alive for this borrow.
        let (status, data_pointer, num_bytes, status_text) = unsafe {
            let handle = &*guard.0;
            (
                handle.status,
                handle.data,
                handle.num_bytes,
                handle.status_text,
            )
        };

        let body = if data_pointer.is_null() || num_bytes == 0 {
            // A zero-length response may legally report a null `data` pointer even with
            // `EMSCRIPTEN_FETCH_LOAD_TO_MEMORY` set; never dereference it in that case.
            Vec::new()
        } else {
            let length =
                usize::try_from(num_bytes).map_err(|_| BrowserHttpError::ResponseTooLarge)?;
            // SAFETY: `data_pointer` is non-null and, per `emscripten_fetch_t`'s documented
            // contract with `EMSCRIPTEN_FETCH_LOAD_TO_MEMORY` set, points at exactly `num_bytes`
            // initialized bytes that stay valid until `emscripten_fetch_close` runs; `guard` is
            // still alive (not yet dropped) for the duration of this slice's use.
            unsafe { slice::from_raw_parts(data_pointer.cast::<u8>(), length) }.to_vec()
        };

        // The browser fetch stack (and Emscripten's wrapper around it) reports transport-level
        // failures -- unreachable host, blocked CORS request, malformed URL -- as `status == 0`;
        // no real HTTP server returns that. Any other status, including 4xx/5xx, is a completed
        // HTTP exchange and is returned to the caller rather than turned into an error here.
        if status == 0 {
            return Err(BrowserHttpError::Network(c_buffer_to_string(&status_text)));
        }

        Ok(HttpResponse { status, body })
    }
}

#[cfg(target_os = "emscripten")]
pub use fetch::request;

#[cfg(test)]
mod tests {
    mod method_buffer_tests {
        use super::super::{BrowserHttpError, method_buffer};

        #[test]
        fn test_method_buffer_encodes_short_method() {
            let buffer = method_buffer("POST").expect("POST should fit");
            let text: String = buffer
                .iter()
                .take_while(|byte| **byte != 0)
                .map(|byte| *byte as u8 as char)
                .collect();
            assert_eq!(text, "POST");
        }

        #[test]
        fn test_method_buffer_rejects_embedded_null() {
            let error = method_buffer("GE\0T").expect_err("embedded null should be rejected");
            assert!(matches!(
                error,
                BrowserHttpError::EmbeddedNullByte { field: "method" }
            ));
        }

        #[test]
        fn test_method_buffer_rejects_method_too_long_for_buffer() {
            let too_long = "A".repeat(32);
            let error = method_buffer(&too_long).expect_err("32-byte method should not fit");
            assert!(matches!(error, BrowserHttpError::InvalidMethod(_)));
        }

        #[test]
        fn test_method_buffer_accepts_method_at_exact_capacity() {
            let exactly_31 = "A".repeat(31);
            assert!(method_buffer(&exactly_31).is_ok());
        }
    }

    mod header_encoding_tests {
        use std::ffi::CStr;

        use super::super::{BrowserHttpError, c_char, header_c_strings, header_pointer_array};

        #[test]
        fn test_header_c_strings_rejects_embedded_null_in_name() {
            let error = header_c_strings(&[("bad\0name", "value")])
                .expect_err("embedded null in header name should be rejected");
            assert!(matches!(
                error,
                BrowserHttpError::EmbeddedNullByte {
                    field: "header name"
                }
            ));
        }

        #[test]
        fn test_header_c_strings_rejects_embedded_null_in_value() {
            let error = header_c_strings(&[("name", "bad\0value")])
                .expect_err("embedded null in header value should be rejected");
            assert!(matches!(
                error,
                BrowserHttpError::EmbeddedNullByte {
                    field: "header value"
                }
            ));
        }

        #[test]
        fn test_header_pointer_array_is_none_for_no_headers() {
            let strings = header_c_strings(&[]).unwrap();
            assert!(header_pointer_array(&strings).is_none());
        }

        #[test]
        fn test_header_pointer_array_alternates_and_terminates_with_null() {
            let strings =
                header_c_strings(&[("Authorization", "Bearer token"), ("X-Test", "1")]).unwrap();
            let pointers = header_pointer_array(&strings).expect("headers should be present");

            // 2 headers -> 4 key/value pointers + 1 null terminator.
            assert_eq!(pointers.len(), 5);
            assert!(pointers.last().unwrap().is_null());
            assert!(pointers[..4].iter().all(|pointer| !pointer.is_null()));

            // SAFETY: test-only read-back through the exact `CString`s that produced these
            // pointers, all of which are still alive (owned by `strings`) at this point.
            let read =
                |pointer: *const c_char| unsafe { CStr::from_ptr(pointer).to_str().unwrap() };
            assert_eq!(read(pointers[0]), "Authorization");
            assert_eq!(read(pointers[1]), "Bearer token");
            assert_eq!(read(pointers[2]), "X-Test");
            assert_eq!(read(pointers[3]), "1");
        }
    }

    mod c_buffer_to_string_tests {
        use super::super::{c_buffer_to_string, c_char};

        #[test]
        fn test_c_buffer_to_string_stops_at_null_terminator() {
            let mut buffer = [0 as c_char; 8];
            for (index, byte) in b"OK\0junk".iter().enumerate() {
                buffer[index] = *byte as c_char;
            }
            assert_eq!(c_buffer_to_string(&buffer), "OK");
        }

        #[test]
        fn test_c_buffer_to_string_handles_full_buffer_without_terminator() {
            let buffer = [b'A' as c_char; 4];
            assert_eq!(c_buffer_to_string(&buffer), "AAAA");
        }
    }
}
