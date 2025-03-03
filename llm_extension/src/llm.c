#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "storage/spin.h"
#include "common/string.h"
#include "llama.h"
#include "utils/memutils.h"


PG_MODULE_MAGIC;

/* Global variables for model state */
static struct llama_model *model = NULL;
static struct llama_context *ctx = NULL;
static slock_t llm_lock; /* Spinlock for concurrency handling */

/* Function declarations */
PG_FUNCTION_INFO_V1(init_llm);
PG_FUNCTION_INFO_V1(llm_generate);

/* Initialize the LLM model */
Datum init_llm(PG_FUNCTION_ARGS) {
    char *model_path;
    struct llama_model_params model_params;
    struct llama_context_params ctx_params;
    text *model_path_text = PG_GETARG_TEXT_PP(0);
    
    model_path = text_to_cstring(model_path_text);

    SpinLockAcquire(&llm_lock);
    /* Clean up any existing model */
    if (ctx) {
        llama_free(ctx);
        ctx = NULL;
    }
    if (model) {
        llama_model_free(model);
        model = NULL;
    }

    /* Initialize llama backend */
    llama_backend_init();

    /* Set up model parameters */
    model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;  /* CPU only for simplicity */
    model_params.use_mmap = true;
    model_params.use_mlock = false;

    /* Load the model */
    model = llama_model_load_from_file(model_path, model_params);
    if (!model) {
        SpinLockRelease(&llm_lock);
        ereport(ERROR,
                (errcode(ERRCODE_INTERNAL_ERROR),
                 errmsg("Failed to load LLM model from %s", model_path)));
    }

    /* Set up context parameters */
    ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;    /* Context window size */
    ctx_params.n_batch = 512;   /* Batch size */
    ctx_params.n_threads = 4;   /* Number of CPU threads */

    /* Create context */
    ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        llama_model_free(model);
        model = NULL;
        SpinLockRelease(&llm_lock);
        ereport(ERROR,
                (errcode(ERRCODE_INTERNAL_ERROR),
                 errmsg("Failed to create LLM context")));
    }

    SpinLockRelease(&llm_lock);
    PG_RETURN_BOOL(true);
}

/* Generate text using the initialized model */
Datum llm_generate(PG_FUNCTION_ARGS) {
    char *prompt;
    llama_token *tokens;
    int n_tokens;
    struct llama_batch batch;
    char *response;
    size_t response_size, response_pos;
    text *result;
    const size_t max_tokens = 1024;
    const size_t max_response_tokens = 512;
    char token_str[8];
    text *prompt_text;
    struct llama_sampler *smpl;
    llama_token eos_token;
    MemoryContext oldcontext, llm_context;

    elog(NOTICE, "Starting llm_generate");

    SpinLockAcquire(&llm_lock);
    if (!model || !ctx) {
        SpinLockRelease(&llm_lock);
        ereport(ERROR,
                (errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
                 errmsg("LLM model not initialized. Call init_llm first.")));
    }
    SpinLockRelease(&llm_lock);

    prompt_text = PG_GETARG_TEXT_PP(0);
    prompt = text_to_cstring(prompt_text);
    elog(NOTICE, "Received prompt: %s", prompt);

    /* Memory Context to manage allocations */
    llm_context = AllocSetContextCreate(CurrentMemoryContext,
                                        "LLM Memory Context",
                                        ALLOCSET_DEFAULT_SIZES);
    oldcontext = MemoryContextSwitchTo(llm_context);

    /* Allocate token buffer */
    tokens = (llama_token *)palloc(max_tokens * sizeof(llama_token));

    /* Tokenize the prompt */
    n_tokens = llama_tokenize(llama_model_get_vocab(model), prompt, strlen(prompt), tokens, max_tokens, true, false);
    if (n_tokens < 0) {
        MemoryContextSwitchTo(oldcontext);
        MemoryContextDelete(llm_context);
        ereport(ERROR,
                (errcode(ERRCODE_INTERNAL_ERROR),
                 errmsg("Failed to tokenize prompt")));
    }

    /* Initialize sampler */
    smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_temp_ext(0.7f, 0.0f, 1.0f));

    /* Process input batch */
    batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = false;
    }
    batch.n_tokens = n_tokens;

    if (llama_decode(ctx, batch) != 0) {
        MemoryContextSwitchTo(oldcontext);
        MemoryContextDelete(llm_context);
        ereport(ERROR,
                (errcode(ERRCODE_INTERNAL_ERROR),
                 errmsg("Failed to process prompt")));
    }

    /* Prepare response buffer */
    response_size = 4096;
    response = (char *)palloc(response_size);
    response_pos = 0;

    /* Get special tokens */
    eos_token = llama_vocab_eos(llama_model_get_vocab(model));

    /* Generate response */
    for (int i = 0; i < max_response_tokens; i++) {
        float *logits;
        llama_token new_token;
        int token_str_len;

        logits = llama_get_logits(ctx);
        new_token = llama_sampler_sample(smpl, logits, llama_vocab_n_tokens(llama_model_get_vocab(model)));

        if (new_token == eos_token) {
            break;
        }

        /* Convert token to string */
        token_str_len = llama_token_to_piece(llama_model_get_vocab(model), new_token, token_str, sizeof(token_str), 0, false);
        if (token_str_len < 0) continue;

        /* Ensure buffer has enough space */
        if (response_pos + token_str_len >= response_size) {
            response_size *= 2;
            response = repalloc(response, response_size);
        }

        /* Append token string */
        memcpy(response + response_pos, token_str, token_str_len);
        response_pos += token_str_len;

        /* Process the new token */
        batch = llama_batch_init(1, 0, 1);
        batch.token[0] = new_token;
        batch.pos[0] = n_tokens + i;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = true;
        batch.n_tokens = 1;

        if (llama_decode(ctx, batch) != 0) {
            elog(WARNING, "Failed to decode token %d", i);
            break;
        }
    }

    /* Null termination */
    response[response_pos] = '\0';

    elog(NOTICE, "Generated response: %s", response);

    /* Convert to PostgreSQL text type */
    result = cstring_to_text(response);

    /* Cleanup */
    MemoryContextSwitchTo(oldcontext);
    MemoryContextDelete(llm_context);
    llama_sampler_free(smpl);

    PG_RETURN_TEXT_P(result);
}
