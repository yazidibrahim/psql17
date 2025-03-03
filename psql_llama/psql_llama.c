#include "psql_llama.h"
#include <string.h>

PG_MODULE_MAGIC;

// Global model state
static llama_model* model = NULL;
static llama_context* ctx = NULL;

// Initialize model
static void init_model() {
    if (model != NULL) return;
    
    const char* model_path = "/home/cybrosys/Downloads/mistral-7b-v0.1.Q2_K.gguf";
    
    struct llama_context_params params = llama_context_default_params();
    params.n_ctx = 2048;
    params.n_threads = 4;
    
    model = llama_load_model_from_file(model_path, params);
    if (model == NULL) {
        ereport(ERROR,
                (errcode(ERRCODE_INTERNAL_ERROR),
                 errmsg("Failed to load GGUF model")));
    }
    
    ctx = llama_new_context_with_model(model, params);
    if (ctx == NULL) {
        llama_free_model(model);
        model = NULL;
        ereport(ERROR,
                (errcode(ERRCODE_INTERNAL_ERROR),
                 errmsg("Failed to create context")));
    }
}

void _PG_init(void) {
    // Initialize when extension is loaded
    init_model();
}

void _PG_fini(void) {
    // Cleanup when extension is unloaded
    if (ctx != NULL) {
        llama_free(ctx);
        ctx = NULL;
    }
    if (model != NULL) {
        llama_free_model(model);
        model = NULL;
    }
}

PG_FUNCTION_INFO_V1(llama_generate);

Datum llama_generate(PG_FUNCTION_ARGS) {
    text* input_text = PG_GETARG_TEXT_PP(0);
    char* input = text_to_cstring(input_text);
    
    if (model == NULL || ctx == NULL) {
        init_model();
    }
    
    struct llama_sampling_params sparams = llama_sampling_default_params();
    sparams.temp = 0.7f;
    sparams.top_p = 0.9f;
    sparams.top_k = 40;
    sparams.n_tokens_predict = 128;
    
    std::vector<llama_token> tokens;
    tokens = llama_tokenize(model, input, true);
    
    std::string result;
    llama_batch batch = llama_batch_init(512, 0, 1);
    
    for (int i = 0; i < tokens.size(); i++) {
        llama_batch_add(batch, tokens[i], i, { 0 }, false);
    }
    
    if (llama_decode(ctx, batch) != 0) {
        llama_batch_free(batch);
        PG_RETURN_NULL();
    }
    
    for (int i = 0; i < sparams.n_tokens_predict; i++) {
        llama_token token = llama_sample_token(ctx, NULL, sparams);
        if (token == llama_token_eos(model)) break;
        
        char* token_str = llama_token_to_str(model, token);
        result += token_str;
        free(token_str);
        
        llama_batch_clear(batch);
        llama_batch_add(batch, token, tokens.size() + i, { 0 }, false);
        if (llama_decode(ctx, batch) != 0) break;
    }
    
    llama_batch_free(batch);
    text* result_text = cstring_to_text(result.c_str());
    PG_RETURN_TEXT_P(result_text);
}
