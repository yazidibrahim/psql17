
# File: psql_llama.h
#ifndef PSQL_LLAMA_H
#define PSQL_LLAMA_H

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "../llama.cpp/llama.h"

// Function declarations
void _PG_init(void);
void _PG_fini(void);
Datum llama_generate(PG_FUNCTION_ARGS);

#endif