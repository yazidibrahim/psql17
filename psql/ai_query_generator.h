// ai_query_generator.h
#ifndef AI_QUERY_GENERATOR_H
#define AI_QUERY_GENERATOR_H

#define MODEL_PATH "/home/cybrosys/gemma2"
#define MAX_QUERY_LENGTH 1024

// Function to generate SQL query from natural language
char* generate_ai_query(const char* natural_language_query);

#endif