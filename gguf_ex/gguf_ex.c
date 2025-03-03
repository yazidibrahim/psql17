#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "llama.h" 

PG_MODULE_MAGIC;

// Structure to hold model state
typedef struct {
    llama_context* ctx;
    llama_model* model;
} LlamaState;

static LlamaState* llm_state = NULL;

// Initialize model function
static void init_model(void) {
    if (llm_state != NULL) {
        return;  // Already initialized
    }

    const char* model_path = "/home/cybrosys/Downloads/mistral-7b-v0.1.Q2_K.gguf";
    llm_state = (LlamaState*)palloc(sizeof(LlamaState));
    
    // Initialize parameters
    struct llama_context_params params = llama_context_default_params();
    params.n_ctx = 2048;
    params.n_threads = 4;
    
    // Load model
    llm_state->model = llama_load_model_from_file(model_path, params);
    if (!llm_state->model) {
        ereport(ERROR,
                (errcode(ERRCODE_EXTERNAL_ROUTINE_EXCEPTION),
                 errmsg("Failed to load GGUF model")));
    }
    
    // Create context
    llm_state->ctx = llama_new_context_with_model(llm_state->model, params);
    if (!llm_state->ctx) {
        llama_free_model(llm_state->model);
        ereport(ERROR,
                (errcode(ERRCODE_EXTERNAL_ROUTINE_EXCEPTION),
                 errmsg("Failed to create context")));
    }
}

// Function to generate text
PG_FUNCTION_INFO_V1(generate_text);
Datum
generate_text(PG_FUNCTION_ARGS)
{
    // Get input prompt
    text* prompt = PG_GETARG_TEXT_PP(0);
    char* prompt_str = text_to_cstring(prompt);
    
    // Initialize model if needed
    if (llm_state == NULL) {
        init_model();
    }
    
    // Tokenize input
    const int max_tokens = 1024;
    llama_token* tokens = (llama_token*)palloc(max_tokens * sizeof(llama_token));
    int n_tokens = llama_tokenize(llm_state->model, prompt_str, strlen(prompt_str), 
                                tokens, max_tokens, true);
    
    if (n_tokens < 0) {
        pfree(tokens);
        ereport(ERROR,
                (errcode(ERRCODE_EXTERNAL_ROUTINE_EXCEPTION),
                 errmsg("Tokenization failed")));
    }
    
    // Evaluate tokens
    if (llama_eval(llm_state->ctx, tokens, n_tokens, 0, 4) != 0) {
        pfree(tokens);
        ereport(ERROR,
                (errcode(ERRCODE_EXTERNAL_ROUTINE_EXCEPTION),
                 errmsg("Evaluation failed")));
    }
    
    // Generate response
    StringInfoData response;
    initStringInfo(&response);
    
    const int max_response_tokens = 256;
    for (int i = 0; i < max_response_tokens; i++) {
        llama_token token = llama_sample_token(llm_state->ctx);
        
        if (token == llama_token_eos()) {
            break;
        }
        
        const char* piece = llama_token_to_str(llm_state->model, token);
        appendStringInfoString(&response, piece);
    }
    
    // Cleanup
    pfree(tokens);
    
    // Return result
    text* result = cstring_to_text(response.data);
    pfree(response.data);
    
    PG_RETURN_TEXT_P(result);
}

// Cleanup function
PG_FUNCTION_INFO_V1(cleanup_model);
Datum
cleanup_model(PG_FUNCTION_ARGS)
{
    if (llm_state) {
        if (llm_state->ctx) {
            llama_free(llm_state->ctx);
        }
        if (llm_state->model) {
            llama_free_model(llm_state->model);
        }
        pfree(llm_state);
        llm_state = NULL;
    }
    PG_RETURN_VOID();
}