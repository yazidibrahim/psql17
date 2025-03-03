// 1. src/bin/psql/ai_interface.h
#ifndef AI_INTERFACE_H
#define AI_INTERFACE_H

#include <stdbool.h>
#include <python3.10/Python.h>
// AI Mode Configuration Structure
typedef struct {
    bool ai_mode_enabled;
    PyObject* ai_model;
    PyObject* tokenizer;
} AIConfiguration;

// Function Prototypes for AI Integration
int initialize_ai_model(AIConfiguration* ai_config);
char* convert_natural_language_to_sql(
    AIConfiguration* ai_config, 
    const char* natural_language_query
);
void cleanup_ai_model(AIConfiguration* ai_config);
bool validate_sql_query(const char* sql_query);

#endif // AI_INTERFACE_H
