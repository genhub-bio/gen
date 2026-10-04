//! Blocking browser HTTP via the shared C/XHR shim. Request preparation is
//! target independent so it can be tested without an Emscripten runtime.

#[cfg(any(test, target_os = "emscripten"))]
use core::ptr;
#[cfg(any(test, target_os = "emscripten"))]
use std::{ffi::CString, os::raw::c_char};

#[cfg(any(test, target_os = "emscripten"))]
use super::BrowserHttpError;

/// Encodes the method while preserving the transport's existing 31-byte limit.
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

/// Builds alternating name/value pointers terminated by NULL, borrowing `strings`.
/// An absent header list is represented by a null pointer at the C boundary.
#[cfg(any(test, target_os = "emscripten"))]
pub(super) fn header_pointer_array(strings: &[CString]) -> Option<Vec<*const c_char>> {
    if strings.is_empty() {
        return None;
    }
    let mut pointers: Vec<*const c_char> = strings.iter().map(|value| value.as_ptr()).collect();
    pointers.push(ptr::null());
    Some(pointers)
}

#[cfg(target_os = "emscripten")]
mod bridge {
    use core::{ffi::c_char, ptr, slice};
    use std::ffi::CString;

    use super::{header_c_strings, header_pointer_array, method_buffer};
    use crate::commands::remote::http::{BrowserHttpError, HttpRequest, HttpResponse};

    #[repr(C)]
    struct Request {
        method: *const c_char,
        url: *const c_char,
        headers: *const *const c_char,
        body: *const u8,
        body_length: usize,
    }

    #[repr(C)]
    struct Response {
        body: *mut u8,
        body_length: usize,
        status: u32,
    }

    unsafe extern "C" {
        fn gen_browser_http_request(request: *const Request, response: *mut Response) -> i32;
        fn gen_browser_http_response_free(response: *mut Response);
    }

    impl Drop for Response {
        fn drop(&mut self) {
            // SAFETY: the shim initializes the response on every return path;
            // this guard uniquely owns its buffer and releases it exactly once.
            unsafe { gen_browser_http_response_free(self) };
        }
    }

    pub fn request(http_request: HttpRequest<'_>) -> Result<HttpResponse, BrowserHttpError> {
        let method = method_buffer(http_request.method)?;
        let url = CString::new(http_request.url)
            .map_err(|_| BrowserHttpError::EmbeddedNullByte { field: "url" })?;
        let header_strings = header_c_strings(http_request.headers)?;
        let header_pointers = header_pointer_array(&header_strings);
        let request = Request {
            method: method.as_ptr(),
            url: url.as_ptr(),
            headers: header_pointers
                .as_ref()
                .map_or(ptr::null(), |headers| headers.as_ptr()),
            body: http_request.body.map_or(ptr::null(), |body| body.as_ptr()),
            body_length: http_request.body.map_or(0, |body| body.len()),
        };
        let mut response = Response {
            body: ptr::null_mut(),
            body_length: 0,
            status: 0,
        };
        // SAFETY: all borrowed request buffers outlive this synchronous call.
        // The shim's wasm32 C structs match these repr(C) layouts, initializes
        // response, and transfers ownership of its malloc buffer to our guard.
        let result = unsafe { gen_browser_http_request(&request, &mut response) };
        match result {
            0 => {}
            2 => return Err(BrowserHttpError::AllocationFailed),
            3 => return Err(BrowserHttpError::ResponseTooLarge),
            4 => return Err(BrowserHttpError::WorkerRequired),
            _ => {
                return Err(BrowserHttpError::Network(
                    "browser request failed (network, CORS, or invalid request)".to_owned(),
                ));
            }
        }
        let mut body = Vec::new();
        body.try_reserve_exact(response.body_length)
            .map_err(|_| BrowserHttpError::AllocationFailed)?;
        if response.body_length != 0 {
            // SAFETY: a successful nonempty response owns body_length initialized
            // bytes. The guard keeps that buffer alive through this copy.
            body.extend_from_slice(unsafe {
                slice::from_raw_parts(response.body, response.body_length)
            });
        }
        Ok(HttpResponse {
            status: response.status as u16,
            body,
        })
    }
}

#[cfg(target_os = "emscripten")]
pub use bridge::request;

#[cfg(test)]
mod tests {
    mod method_buffer_tests {
        use super::super::{BrowserHttpError, method_buffer};

        #[test]
        fn test_method_buffer_encodes_short_method() {
            let buffer = method_buffer("POST").expect("should fit POST in the method buffer");
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
            let pointers = header_pointer_array(&strings).expect("should have headers present");

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
}
