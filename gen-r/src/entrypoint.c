#include <R_ext/Rdynload.h>

void R_init_genr_extendr(DllInfo *dll);

void R_init_genr(DllInfo *dll) {
  R_init_genr_extendr(dll);
}
