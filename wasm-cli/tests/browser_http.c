#include "browser_http.h"
#include <stdlib.h>
#include <string.h>

static int fail_allocation;
void *__real_malloc(size_t length);
void *__wrap_malloc(size_t length) {
    return fail_allocation ? NULL : __real_malloc(length);
}

/* Called from JS in an actual worker, for both static and dynamic linking. */
int run_test(const char *url, const char *method, int status, int length, int result) {
    const unsigned char body[] = {0, 128, 255, 13};
    const char *headers[] = {"X-Gen-Test", "binary", "Content-Type", "application/octet-stream", NULL};
    GenHttpRequest request = {method, url, headers, body, sizeof(body)};
    if (strcmp(method, "GET") == 0 || strcmp(method, "HEAD") == 0) {
        request.headers = NULL;
        request.body = NULL;
        request.body_length = 0;
    }
    GenHttpResponse response;
    fail_allocation = result == 2;
    int actual = gen_browser_http_request(&request, &response);
    fail_allocation = 0;
    int valid = actual == result;
    if (actual == 0) {
        valid = valid && response.status == (unsigned)status && response.body_length == (unsigned)length;
        for (size_t index = 0; valid && index < response.body_length; ++index) {
            unsigned char expected = length == 4 ? body[index] : (unsigned char)index;
            valid = response.body[index] == expected;
        }
    }
    gen_browser_http_response_free(&response);
    return valid && !response.body && !response.body_length && !response.status;
}
