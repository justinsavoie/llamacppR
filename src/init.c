#include <R.h>
#include <Rinternals.h>

// declarations of our routines
SEXP _llamacppR_llama_simple(SEXP model_pathSEXP, SEXP promptSEXP, SEXP n_predictSEXP);
SEXP _llamacppR_llama_sampled(SEXP model_pathSEXP, SEXP promptSEXP, SEXP n_predictSEXP,
                              SEXP temperatureSEXP, SEXP top_pSEXP, SEXP top_kSEXP, SEXP min_pSEXP,
                              SEXP repeat_penaltySEXP, SEXP repeat_last_nSEXP, SEXP presence_penaltySEXP,
                              SEXP frequency_penaltySEXP, SEXP seedSEXP);

SEXP _llamacppR_load(SEXP model_pathSEXP, SEXP n_ctxSEXP);
SEXP _llamacppR_unload(SEXP handleSEXP);
SEXP _llamacppR_generate(SEXP handleSEXP, SEXP promptSEXP, SEXP n_predictSEXP);
SEXP _llamacppR_generate_sampled(SEXP handleSEXP, SEXP promptSEXP, SEXP n_predictSEXP,
                                 SEXP temperatureSEXP, SEXP top_pSEXP, SEXP top_kSEXP, SEXP min_pSEXP,
                                 SEXP repeat_penaltySEXP, SEXP repeat_last_nSEXP, SEXP presence_penaltySEXP,
                                 SEXP frequency_penaltySEXP, SEXP seedSEXP);

static const R_CallMethodDef CallEntries[] = {
    {"_llamacppR_llama_simple", (DL_FUNC) &_llamacppR_llama_simple, 3},
    {"_llamacppR_llama_sampled", (DL_FUNC) &_llamacppR_llama_sampled, 12},
    {"_llamacppR_load", (DL_FUNC) &_llamacppR_load, 2},
    {"_llamacppR_unload", (DL_FUNC) &_llamacppR_unload, 1},
    {"_llamacppR_generate", (DL_FUNC) &_llamacppR_generate, 3},
    {"_llamacppR_generate_sampled", (DL_FUNC) &_llamacppR_generate_sampled, 12},
    {NULL, NULL, 0}
};

void R_init_llamacppR(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
