#include <R.h>
#include <Rinternals.h>

// declaration of our routine
SEXP _llamacppR_llama_simple(SEXP model_pathSEXP, SEXP promptSEXP, SEXP n_predictSEXP);

static const R_CallMethodDef CallEntries[] = {
    {"_llamacppR_llama_simple", (DL_FUNC) &_llamacppR_llama_simple, 3},
    {NULL, NULL, 0}
};

void R_init_llamacppR(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}

