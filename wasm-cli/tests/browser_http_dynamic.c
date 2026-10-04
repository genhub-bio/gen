#include <dlfcn.h>

typedef int (*TestFunction)(const char *, const char *, int, int, int);

int run_test(const char *url, const char *method, int status, int length, int result) {
    static TestFunction test;
    if (!test) {
        void *module = dlopen("http-side.wasm", RTLD_NOW);
        if (!module) return 0;
        test = (TestFunction)dlsym(module, "run_test");
        if (!test) return 0;
    }
    return test(url, method, status, length, result);
}
