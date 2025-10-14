// Avoid macro remapping like `length` from R headers colliding with C++
#define R_NO_REMAP 1
#include <R.h>
#include <Rinternals.h>

#include "third_party/llama.cpp/include/llama.h"

#include <string>
#include <vector>

extern "C" SEXP _llamacppR_llama_simple(SEXP model_pathSEXP, SEXP promptSEXP, SEXP n_predictSEXP) {
    if (!Rf_isString(model_pathSEXP) || Rf_length(model_pathSEXP) != 1) {
        Rf_error("model must be a length-1 string");
    }
    if (!Rf_isString(promptSEXP) || Rf_length(promptSEXP) != 1) {
        Rf_error("prompt must be a length-1 string");
    }
    if (!Rf_isInteger(n_predictSEXP) && !Rf_isReal(n_predictSEXP)) {
        Rf_error("n_predict must be integer-like");
    }

    const char * model_path_c = CHAR(STRING_ELT(model_pathSEXP, 0));
    const char * prompt_c     = CHAR(STRING_ELT(promptSEXP, 0));
    int n_predict             = Rf_asInteger(n_predictSEXP);
    if (n_predict <= 0) n_predict = 1;

    std::string model_path(model_path_c);
    std::string prompt(prompt_c);

    // load dynamic backends (CPU only acceptable)
    ggml_backend_load_all();

    // model params (CPU-only: n_gpu_layers = 0)
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (model == NULL) {
        Rf_error("unable to load model");
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // tokenize prompt
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(), NULL, 0, /*special=*/true, /*parse_special=*/true);
    if (n_prompt <= 0) {
        llama_model_free(model);
        Rf_error("failed to tokenize prompt");
    }
    std::vector<llama_token> prompt_tokens((size_t)n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(), prompt_tokens.data(), (int)prompt_tokens.size(), /*special=*/true, /*parse_special=*/true) < 0) {
        llama_model_free(model);
        Rf_error("failed to tokenize prompt");
    }

    // init context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx   = n_prompt + n_predict - 1;
    ctx_params.n_batch = n_prompt;
    ctx_params.no_perf = true;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == NULL) {
        llama_model_free(model);
        Rf_error("failed to create llama_context");
    }

    // sampler (greedy, no perf)
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = true;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    if (!smpl) {
        llama_free(ctx);
        llama_model_free(model);
        Rf_error("failed to create sampler");
    }
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // build output by echoing prompt pieces then generated tokens
    std::string out;
    out.reserve(prompt.size() + (size_t)n_predict * 4);

    // echo prompt tokens
    for (auto id : prompt_tokens) {
        char buf[256];
        int n = llama_token_to_piece(vocab, id, buf, (int)sizeof(buf), 0, /*special=*/true);
        if (n < 0) {
            llama_sampler_free(smpl);
            llama_free(ctx);
            llama_model_free(model);
            Rf_error("failed to convert token to piece");
        }
        out.append(buf, (size_t)n);
    }

    // prepare initial batch
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), (int)prompt_tokens.size());

    if (llama_model_has_encoder(model)) {
        if (llama_encode(ctx, batch)) {
            llama_sampler_free(smpl);
            llama_free(ctx);
            llama_model_free(model);
            Rf_error("failed to eval (encode)");
        }
        llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
        if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
            decoder_start_token_id = llama_vocab_bos(vocab);
        }
        batch = llama_batch_get_one(&decoder_start_token_id, 1);
    }

    int n_decode = 0;
    llama_token new_token_id;

    for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict;) {
        if (llama_decode(ctx, batch)) {
            llama_sampler_free(smpl);
            llama_free(ctx);
            llama_model_free(model);
            Rf_error("failed to eval (decode)");
        }

        n_pos += batch.n_tokens;

        new_token_id = llama_sampler_sample(smpl, ctx, -1);

        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }

        char buf[256];
        int n = llama_token_to_piece(vocab, new_token_id, buf, (int)sizeof(buf), 0, /*special=*/true);
        if (n < 0) {
            llama_sampler_free(smpl);
            llama_free(ctx);
            llama_model_free(model);
            Rf_error("failed to convert token to piece");
        }
        out.append(buf, (size_t)n);

        batch = llama_batch_get_one(&new_token_id, 1);
        n_decode += 1;
    }

    // cleanup
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    SEXP ans = PROTECT(Rf_allocVector(STRSXP, 1));
    SET_STRING_ELT(ans, 0, Rf_mkCharLenCE(out.c_str(), (int)out.size(), CE_UTF8));
    UNPROTECT(1);
    return ans;
}
