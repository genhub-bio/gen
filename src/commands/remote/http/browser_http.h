#ifndef GEN_BROWSER_HTTP_H
#define GEN_BROWSER_HTTP_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    const char *method;
    const char *url;
    const char *const *headers;
    const unsigned char *body;
    size_t body_length;
} GenHttpRequest;

typedef struct {
    unsigned char *body;
    size_t body_length;
    uint32_t status;
} GenHttpResponse;

/* Input pointers are borrowed for the synchronous call only. The initialized
 * response owns a malloc buffer, including on error, released exactly once by
 * gen_browser_http_response_free. HTTP error statuses are successful exchanges.
 * Return codes: 0 success, 1 network/browser failure, 2 allocation failure,
 * 3 response too large, 4 execution outside a browser worker. */
int gen_browser_http_request(const GenHttpRequest *request, GenHttpResponse *response);
void gen_browser_http_response_free(GenHttpResponse *response);

#endif
