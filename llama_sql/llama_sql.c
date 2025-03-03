/*
 * llama_sql.c
 * PostgreSQL extension for integrating GGUF models to generate SQL queries
 */


#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "miscadmin.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/memutils.h"
#include "llama.cpp/include/ggml.hjn"      // Include ggml.h first
         // Then include llama.h
#include <pthread.h>

PG_MODULE_MAGIC;

/* Declarations */
void _PG_init(void);
void _PG_fini(void);

/* Global variables */
static llama_context* model_ctx = NULL;
static llama_model* model = NULL;
static char* model_path = NULL;
static int max_tokens = 2048;
static int context_size = 2048;
static float temperature = 0.7f;
static pthread_mutex_t model_mutex = PTHREAD_MUTEX_INITIALIZER;

/* GUC variables */
static void model_path_assign_hook(const char *newval, void *extra);
static void max_tokens_assign_hook(int newval, void *extra);
static void context_size_assign_hook(int newval, void *extra);
static void temperature_assign_hook(double newval, void *extra);

/* Function declarations */
PG_FUNCTION_INFO_V1(generate_sql);
PG_FUNCTION_INFO_V1(generate_sql_with_context);

/* Initialization function */
void _PG_init(void) {
    DefineCustomStringVariable(
        "llama_sql.model_path",
        "Path to the GGUF model file",
        NULL,
        &model_path,
        "",
        PGC_USERSET,
        0,
        NULL,
        model_path_assign_hook,
        NULL
    );

    DefineCustomIntVariable(
        "llama_sql.max_tokens",
        "Maximum number of tokens to generate",
        NULL,
        &max_tokens,
        2048,
        1,
        8192,
        PGC_USERSET,
        0,
        NULL,
        max_tokens_assign_hook,
        NULL
    );

    DefineCustomIntVariable(
        "llama_sql.context_size",
        "Size of the context window",
        NULL,
        &context_size,
        2048,
        1,
        8192,
        PGC_USERSET,
        0,
        NULL,
        context_size_assign_hook,
        NULL
    );

    DefineCustomRealVariable(
        "llama_sql.temperature",
        "Temperature for token generation",
        NULL,
        &temperature,
        0.7,
        0.0,
        2.0,
        PGC_USERSET,
        0,
        NULL,
        temperature_assign_hook,
        NULL
    );
}

/* Cleanup function */
void _PG_fini(void) {
    pthread_mutex_lock(&model_mutex);
    if (model_ctx != NULL) {
        llama_free(model_ctx);
        model_ctx = NULL;
    }
    if (model != NULL) {
        llama_free_model(model);
        model = NULL;
    }
    pthread_mutex_unlock(&model_mutex);
    pthread_mutex_destroy(&model_mutex);
}

/* GUC assign hooks */
static void model_path_assign_hook(const char *newval, void *extra) {
    pthread_mutex_lock(&model_mutex);
    if (model_ctx != NULL) {
        llama_free(model_ctx);
        model_ctx = NULL;
    }
    if (model != NULL) {
        llama_free_model(model);
        model = NULL;
    }
    pthread_mutex_unlock(&model_mutex);
}

static void max_tokens_assign_hook(int newval, void *extra) {
    /* No action needed */
}

static void context_size_assign_hook(int newval, void *extra) {
    pthread_mutex_lock(&model_mutex);
    if (model_ctx != NULL) {
        llama_free(model_ctx);
        model_ctx = NULL;
    }
    pthread_mutex_unlock(&model_mutex);
}

static void temperature_assign_hook(double newval, void *extra) {
    /* No action needed */
}

/* Model initialization */
static void init_llama_model(void) {
    if (model_ctx != NULL) return;

    if (strlen(model_path) == 0) {
        ereport(ERROR,
                (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                 errmsg("Model path not set"),
                 errhint("Set llama_sql.model_path configuration parameter")));
    }

    pthread_mutex_lock(&model_mutex);

    if (model_ctx == NULL) {
        struct llama_context_params params = llama_context_default_params();
        params.n_ctx = context_size;
        params.seed = -1;
        params.n_threads = 4;
        params.n_threads_batch = 4;
        
        model = llama_load_model_from_file(model_path, llama_model_default_params());
        
        if (model == NULL) {
            pthread_mutex_unlock(&model_mutex);
            ereport(ERROR,
                    (errcode(ERRCODE_INTERNAL_ERROR),
                     errmsg("Failed to load GGUF model from %s", model_path)));
        }
        
        model_ctx = llama_new_context_with_model(model, params);
        
        if (model_ctx == NULL) {
            llama_free_model(model);
            model = NULL;
            pthread_mutex_unlock(&model_mutex);
            ereport(ERROR,
                    (errcode(ERRCODE_INTERNAL_ERROR),
                     errmsg("Failed to create llama context")));
        }
    }

    pthread_mutex_unlock(&model_mutex);
}

/* Helper function to generate SQL */
static char* generate_sql_internal(const char* prompt, const char* context) {
    init_llama_model();

    /* Prepare the full prompt */
    StringInfoData full_prompt;
    initStringInfo(&full_prompt);

    appendStringInfo(&full_prompt, "You are a PostgreSQL expert. Generate a SQL query for the following request.\n");
    
    if (context != NULL && strlen(context) > 0) {
        appendStringInfo(&full_prompt, "Context about the database:\n%s\n\n", context);
    }
    
    appendStringInfo(&full_prompt, "Request: %s\n\nSQL Query:", prompt);

    /* Generate tokens */
    pthread_mutex_lock(&model_mutex);

    llama_batch batch = llama_batch_init(max_tokens, 0, 1);
    if (!llama_tokenize(model_ctx, full_prompt.data, full_prompt.len, &batch, true)) {
        pthread_mutex_unlock(&model_mutex);
        pfree(full_prompt.data);
        ereport(ERROR,
                (errcode(ERRCODE_INTERNAL_ERROR),
                 errmsg("Failed to tokenize input")));
    }

    /* Process tokens and generate response */
    StringInfoData result;
    initStringInfo(&result);
    bool in_sql = false;

    if (llama_decode(model_ctx, batch)) {
        llama_token token;
        int generated_tokens = 0;

        while (generated_tokens < max_tokens) {
            token = llama_sample_token_greedy(model_ctx);
            
            if (token == llama_token_eos(model) || token < 0) {
                break;
            }

            char token_str[8];
            int str_len = llama_token_to_str(model_ctx, token, token_str, sizeof(token_str));
            
            if (str_len > 0) {
                appendBinaryStringInfo(&result, token_str, str_len);
            }

            batch.n_tokens = 1;
            batch.token[0] = token;
            
            if (!llama_decode(model_ctx, batch)) {
                break;
            }

            generated_tokens++;
        }
    }

    llama_batch_free(batch);
    pthread_mutex_unlock(&model_mutex);
    pfree(full_prompt.data);

    /* Extract just the SQL query from the response */
    char* sql_start = strstr(result.data, "SELECT");
    if (sql_start == NULL) sql_start = strstr(result.data, "WITH");
    if (sql_start == NULL) sql_start = strstr(result.data, "INSERT");
    if (sql_start == NULL) sql_start = strstr(result.data, "UPDATE");
    if (sql_start == NULL) sql_start = strstr(result.data, "DELETE");
    if (sql_start == NULL) sql_start = strstr(result.data, "CREATE");
    if (sql_start == NULL) sql_start = strstr(result.data, "ALTER");
    if (sql_start == NULL) sql_start = strstr(result.data, "DROP");

    char* final_result;
    if (sql_start != NULL) {
        final_result = pstrdup(sql_start);
        /* Remove any trailing explanation */
        char* explanation_start = strstr(final_result, "\n\n");
        if (explanation_start != NULL) {
            *explanation_start = '\0';
        }
    } else {
        final_result = pstrdup(result.data);
    }

    pfree(result.data);
    return final_result;
}

/* Main function to generate SQL from natural language */
Datum
generate_sql(PG_FUNCTION_ARGS) {
    text* input_text = PG_GETARG_TEXT_PP(0);
    char* prompt = text_to_cstring(input_text);
    
    char* result = generate_sql_internal(prompt, NULL);
    text* result_text = cstring_to_text(result);
    
    pfree(prompt);
    pfree(result);
    
    PG_RETURN_TEXT_P(result_text);
}

/* Function to generate SQL with additional context */
Datum
generate_sql_with_context(PG_FUNCTION_ARGS) {
    text* input_text = PG_GETARG_TEXT_PP(0);
    text* context_text = PG_GETARG_TEXT_PP(1);
    
    char* prompt = text_to_cstring(input_text);
    char* context = text_to_cstring(context_text);
    
    char* result = generate_sql_internal(prompt, context);
    text* result_text = cstring_to_text(result);
    
    pfree(prompt);
    pfree(context);
    pfree(result);
    
    PG_RETURN_TEXT_P(result_text);
}