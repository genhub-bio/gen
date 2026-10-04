#include "browser_http.h"

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <stdlib.h>

/* The FFI targets wasm32 only. Explicit offsets keep JS independent of runtime
 * struct helpers; fail compilation rather than silently misread a new ABI. */
_Static_assert(sizeof(void *) == 4, "browser HTTP requires wasm32");
_Static_assert(offsetof(GenHttpRequest, body_length) == 16, "request ABI");
_Static_assert(offsetof(GenHttpResponse, body_length) == 4, "response ABI");
_Static_assert(offsetof(GenHttpResponse, status) == 8, "response ABI");

EM_JS(int, gen_http_send, (const GenHttpRequest *request, GenHttpResponse *response), {
    if (typeof WorkerGlobalScope === 'undefined' ||
        !(globalThis instanceof WorkerGlobalScope)) return 4;
    try {
        // Decode locally so dynamic modules need only the standard heap and
        // Module globals, with no separately exported UTF8ToString helper.
        var readString = function(pointer) {
            var end = pointer;
            while (HEAPU8[end]) ++end;
            return new TextDecoder('utf-8').decode(HEAPU8.slice(pointer, end));
        };
        var method = readString(HEAPU32[request >>> 2]);
        var url = readString(HEAPU32[(request + 4) >>> 2]);
        var headers = HEAPU32[(request + 8) >>> 2];
        var bodyPointer = HEAPU32[(request + 12) >>> 2];
        var bodyLength = HEAPU32[(request + 16) >>> 2];
        // XHR must never receive a view backed by shared Wasm memory. This
        // also keeps the request valid if memory grows during the exchange.
        var body = null;
        if (bodyPointer) {
            body = new Uint8Array(bodyLength);
            body.set(HEAPU8.subarray(bodyPointer, bodyPointer + bodyLength));
        }
        var xhr = new XMLHttpRequest();
        xhr.open(method, url, false);
        xhr.responseType = 'arraybuffer';
        if (headers) {
            for (var pointer = headers; HEAPU32[pointer >>> 2]; pointer += 8) {
                xhr.setRequestHeader(readString(HEAPU32[pointer >>> 2]),
                    readString(HEAPU32[(pointer + 4) >>> 2]));
            }
        }
        xhr.send(body);
        if (!xhr.status) return 1;
        var bytes = xhr.response ? new Uint8Array(xhr.response) : new Uint8Array(0);
        if (bytes.length > 0x7fffffff) return 3;
        // No global response slot: each synchronous call has its own key.
        var responses = Module.__genHttpResponses ||
            (Module.__genHttpResponses = new Map());
        responses.set(response, bytes);
        HEAPU32[(response + 4) >>> 2] = bytes.length;
        HEAPU32[(response + 8) >>> 2] = xhr.status;
        return 0;
    } catch (error) {
        // Browser exception text may contain URLs, query tokens or headers.
        return error instanceof RangeError ? 2 : 1;
    }
});

EM_JS(void, gen_http_take, (GenHttpResponse *response, unsigned char *destination), {
    var responses = Module.__genHttpResponses;
    var bytes = responses.get(response);
    try {
        // C allocation may grow memory. Resolve HEAPU8 here, after malloc,
        // instead of retaining the view used by gen_http_send.
        if (destination) HEAPU8.set(bytes, destination);
    } finally {
        responses.delete(response);
    }
});

int gen_browser_http_request(const GenHttpRequest *request, GenHttpResponse *response) {
    *response = (GenHttpResponse){0};
    int result = gen_http_send(request, response);
    if (result != 0) return result;
    if (response->body_length != 0) {
        response->body = malloc(response->body_length);
        if (!response->body) {
            gen_http_take(response, NULL);
            return 2;
        }
    }
    gen_http_take(response, response->body);
    return 0;
}

void gen_browser_http_response_free(GenHttpResponse *response) {
    free(response->body);
    *response = (GenHttpResponse){0};
}
#endif
