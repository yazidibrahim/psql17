#ifndef LLM_H
#define LLM_H

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"

// Forward declarations to avoid include conflicts
struct llama_model;
struct llama_context;

extern struct llama_model* g_model;
extern struct llama_context* g_ctx;

void _PG_init(void);
void _PG_fini(void);

Datum init_llm(PG_FUNCTION_ARGS);
Datum llm_generate(PG_FUNCTION_ARGS);

#endif