// ai_query_generator.c
#include "ai_query_generator.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char* generate_ai_query(const char* natural_language_query) {
    char command[2048];
    FILE* pipe;
    char* result = malloc(MAX_QUERY_LENGTH * sizeof(char));
    
    // Construct command to interact with Gemma2b model
    snprintf(command, sizeof(command), 
        "python3 -c \"from transformers import AutoTokenizer, AutoModelForCausalLM; "
        "tokenizer = AutoTokenizer.from_pretrained('%s'); "
        "model = AutoModelForCausalLM.from_pretrained('%s'); "
        "prompt = 'Convert to SQL query: %s\\nSQL Query:'; "
        "inputs = tokenizer(prompt, return_tensors=\\\"pt\\\"); "    
        "outputs = model.generate(**inputs, max_length=200, num_return_sequences=1); "
        "print(tokenizer.decode(outputs[0], skip_special_tokens=True))\"", 
        MODEL_PATH, MODEL_PATH, natural_language_query);
    
    // Execute Python command and capture output
    pipe = popen(command, "r");
    if (!pipe) {
        fprintf(stderr, "Failed to execute AI query generation\n");
        return NULL;
    }
    
    // Read generated query
    if (fgets(result, MAX_QUERY_LENGTH, pipe) == NULL) {
        fprintf(stderr, "Failed to read AI query\n");
        pclose(pipe);
        free(result);
        return NULL;
    }
    
    pclose(pipe);
    
    // Remove newline and trim
    result[strcspn(result, "\n")] = 0;
    return result;
}