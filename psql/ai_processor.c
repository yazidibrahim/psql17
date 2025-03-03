#include "ai_interface.h"
#include <stdio.h>
#include <stdlib.h>
#include <python3.10/Python.h>

// Initialize AI Model with Python Embedding
int initialize_ai_model(AIConfiguration* ai_config) {
    Py_Initialize();
    
    // Import required Python modules
    PyObject* pName = PyUnicode_DecodeFSDefault("transformers");
    PyObject* pModule = PyImport_Import(pName);
    
    if (!pModule) {
        PyErr_Print();
        return -1;
    }
    
    // Load pre-trained model for SQL generation
    PyObject* pFunc = PyObject_GetAttrString(pModule, "AutoModelForSeq2SeqLM");
    PyObject* pArgs = PyTuple_Pack(1, PyUnicode_FromString("/home/cybrosys/gemma2"));
    
    ai_config->ai_model = PyObject_CallObject(pFunc, pArgs);
    ai_config->tokenizer = PyObject_CallMethod(
        pModule, 
        "AutoTokenizer", 
        "O", 
        PyUnicode_FromString("/home/cybrosys/gemma2")
    );
    
    ai_config->ai_mode_enabled = true;
    return 0;
}

// Convert Natural Language to SQL
char* convert_natural_language_to_sql(
    AIConfiguration* ai_config, 
    const char* natural_language_query
) {
    if (!ai_config->ai_mode_enabled) return NULL;
    
    // Tokenize input
    PyObject* pTokens = PyObject_CallMethod(
        ai_config->tokenizer, 
        "encode", 
        "s", 
        natural_language_query
    );
    
    // Generate SQL Query
    PyObject* pSQLQuery = PyObject_CallMethod(
        ai_config->ai_model, 
        "generate", 
        "O", 
        pTokens
    );
    
    // Convert Python string to C string
    char* sql_query = PyUnicode_AsUTF8(pSQLQuery);
    return strdup(sql_query);
}

// Validate Generated SQL Query
bool validate_sql_query(const char* sql_query) {
    // Implement basic SQL injection prevention
    const char* dangerous_keywords[] = {
        "DROP", "DELETE", "TRUNCATE", "--", ";"
    };
    
    for (int i = 0; i < sizeof(dangerous_keywords)/sizeof(char*); i++) {
        if (strstr(sql_query, dangerous_keywords[i]) != NULL) {
            return false;
        }
    }
    
    return true;
}

// Cleanup AI Resources
void cleanup_ai_model(AIConfiguration* ai_config) {
    Py_DECREF(ai_config->ai_model);
    Py_DECREF(ai_config->tokenizer);
    Py_Finalize();
}
