#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include <Python.h>
#include "utils/elog.h"

PG_MODULE_MAGIC;

/* Function declarations */
PG_FUNCTION_INFO_V1(ai_chat_response);
void _PG_init(void);
void _PG_fini(void);

/* Initialize Python */
void _PG_init(void)
{
    if (!Py_IsInitialized()) {
        Py_Initialize();
        
        // Add the path where llama_chat.py is located
        PyRun_SimpleString("import sys");
        PyRun_SimpleString("sys.path.append('/home/cybrosys/PSQL/postgresql/contrib/ai_chat')");  // Add the directory containing your Python file
        
        // Log initialization
        elog(NOTICE, "Python interpreter initialized");
        elog(NOTICE, "Python path: %s", Py_GetPath());
        
        // Import required modules early to catch any initialization errors
        PyObject *pModule = PyImport_ImportModule("llama_cpp");
        if (pModule == NULL) {
            elog(ERROR, "Failed to import llama_cpp module. Please ensure llama-cpp-python is installed.");
        }
        Py_XDECREF(pModule);
    }
}

/* Cleanup Python */
void _PG_fini(void)
{
    if (Py_IsInitialized()) {
        Py_Finalize();
    }
}

/* Main chat function */
Datum
ai_chat_response(PG_FUNCTION_ARGS)
{
    // Get input text
    text* input_text = PG_GETARG_TEXT_PP(0);
    char* input_str = text_to_cstring(input_text);
    
    elog(NOTICE, "Attempting to import llama_chat module");
    
    // Import the Python module
    PyObject *pModule = PyImport_ImportModule("llama_chat");
    if (pModule == NULL) {
        PyErr_Print();
        ereport(ERROR,
                (errcode(ERRCODE_EXTERNAL_ROUTINE_EXCEPTION),
                 errmsg("Could not import the Python module 'llama_chat'"),
                 errhint("Ensure llama_chat.py is in the Python path")));
    }
    
    // Get the get_response function
    PyObject *pFunc = PyObject_GetAttrString(pModule, "get_response");
    if (!(pFunc && PyCallable_Check(pFunc))) {
        Py_DECREF(pModule);
        ereport(ERROR,
                (errcode(ERRCODE_EXTERNAL_ROUTINE_EXCEPTION),
                 errmsg("Could not find function 'get_response' in 'llama_chat'")));
    }
    
    // Create arguments tuple
    PyObject *pArgs = PyTuple_New(1);
    PyObject *pValue = PyUnicode_FromString(input_str);
    PyTuple_SetItem(pArgs, 0, pValue);  // This steals the reference to pValue
    
    // Call the get_response function
    pValue = PyObject_CallObject(pFunc, pArgs);
    if (pValue == NULL) {
        Py_DECREF(pArgs);
        Py_DECREF(pFunc);
        Py_DECREF(pModule);
        PyErr_Print();
        ereport(ERROR,
                (errcode(ERRCODE_EXTERNAL_ROUTINE_EXCEPTION),
                 errmsg("Failed to execute get_response function")));
    }
    
    // Convert the result to string
    const char* result = PyUnicode_AsUTF8(pValue);
    if (result == NULL) {
        result = "Error: No response from model";
    }
    
    // Clean up Python objects
    Py_DECREF(pValue);
    Py_DECREF(pArgs);
    Py_DECREF(pFunc);
    Py_DECREF(pModule);
    
    // Return the result
    PG_RETURN_TEXT_P(cstring_to_text(result));
}