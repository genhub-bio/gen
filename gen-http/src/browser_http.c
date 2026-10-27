#include <emscripten.h>
#include <stddef.h>
#include <stdint.h>

EM_JS(int, gen_http_xhr_request_js,
      (const char *method, const char *url, const char *headers_json,
       const uint8_t *body, size_t body_length, uint32_t *status,
       size_t *response_length, size_t *response_headers_length,
       size_t *response_url_length, uint32_t timeout_millis,
       char *error_message, size_t error_capacity), {
  Module.__gen_http_response_body = null;
  Module.__gen_http_response_headers = null;
  Module.__gen_http_response_url = null;
  try {
    const xhr = new XMLHttpRequest();
    xhr.open(UTF8ToString(method), UTF8ToString(url), false);
    xhr.responseType = "arraybuffer";
    if (timeout_millis > 0) {
      xhr.timeout = timeout_millis;
    }

    const request_headers = JSON.parse(UTF8ToString(headers_json));
    for (const [name, value] of request_headers) {
      xhr.setRequestHeader(name, value);
    }

    const request_body = body_length === 0
        ? null
        : HEAPU8.slice(body, body + body_length);
    xhr.send(request_body);

    const response_body = xhr.response == null
        ? new Uint8Array(0)
        : new Uint8Array(xhr.response);
    const response_headers = xhr.getAllResponseHeaders() || "";
    const response_url = xhr.responseURL || UTF8ToString(url);
    Module.__gen_http_response_body = response_body;
    Module.__gen_http_response_headers = response_headers;
    Module.__gen_http_response_url = response_url;
    HEAPU32[status >> 2] = xhr.status;
    HEAPU32[response_length >> 2] = response_body.length;
    HEAPU32[response_headers_length >> 2] = lengthBytesUTF8(response_headers);
    HEAPU32[response_url_length >> 2] = lengthBytesUTF8(response_url);

    if (xhr.status === 0) {
      Module.__gen_http_response_body = null;
      Module.__gen_http_response_headers = null;
      Module.__gen_http_response_url = null;
      stringToUTF8("XMLHttpRequest completed without an HTTP status", error_message, error_capacity);
      return -1;
    }
    return 0;
  } catch (error) {
    Module.__gen_http_response_body = null;
    Module.__gen_http_response_headers = null;
    Module.__gen_http_response_url = null;
    const message = error && error.name === "TimeoutError"
        ? "XMLHttpRequest timed out"
        : "XMLHttpRequest failed (invalid request or network/CORS error)";
    stringToUTF8(message, error_message, error_capacity);
    return -1;
  }
});

EM_JS(void, gen_http_xhr_clear_response_js, (), {
  Module.__gen_http_response_body = null;
  Module.__gen_http_response_headers = null;
  Module.__gen_http_response_url = null;
});

EM_JS(void, gen_http_xhr_copy_body_js, (uint8_t *destination), {
  const response_body = Module.__gen_http_response_body;
  if (response_body && response_body.length > 0) {
    HEAPU8.set(response_body, destination);
  }
});

EM_JS(void, gen_http_xhr_copy_headers_js,
      (char *destination, size_t capacity), {
  stringToUTF8(Module.__gen_http_response_headers || "", destination, capacity);
});

EM_JS(void, gen_http_xhr_copy_url_js, (char *destination, size_t capacity), {
  stringToUTF8(Module.__gen_http_response_url || "", destination, capacity);
});

/*
 * These real C entry points force the archive member carrying the EM_JS bodies
 * into the Rust side module.
 */
int gen_http_xhr_request(const char *method, const char *url,
                         const char *headers_json, const uint8_t *body,
                         size_t body_length, uint32_t *status,
                         size_t *response_length,
                         size_t *response_headers_length,
                         size_t *response_url_length, uint32_t timeout_millis,
                         char *error_message, size_t error_capacity) {
  return gen_http_xhr_request_js(
      method, url, headers_json, body, body_length, status, response_length,
      response_headers_length, response_url_length, timeout_millis,
      error_message, error_capacity);
}

void gen_http_xhr_clear_response(void) {
  gen_http_xhr_clear_response_js();
}

void gen_http_xhr_copy_body(uint8_t *destination) {
  gen_http_xhr_copy_body_js(destination);
}

void gen_http_xhr_copy_headers(char *destination, size_t capacity) {
  gen_http_xhr_copy_headers_js(destination, capacity);
}

void gen_http_xhr_copy_url(char *destination, size_t capacity) {
  gen_http_xhr_copy_url_js(destination, capacity);
}
