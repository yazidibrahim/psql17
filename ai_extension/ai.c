#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/memutils.h"
#include "utils/elog.h"
#include <python3.10/Python.h>

PG_MODULE_MAGIC;

// Global variables for Python module
static PyObject *py_module = NULL;
static PyObject *py_generate_func = NULL;

// Function to get Python error details
static char* get_python_error(void) {
    PyObject *ptype, *pvalue, *ptraceback;
    PyErr_Fetch(&ptype, &pvalue, &ptraceback);
    
    if (!pvalue) {
        return pstrdup("No error details available");
    }
    
    PyObject *str_obj = PyObject_Str(pvalue);
    if (!str_obj) {
        Py_XDECREF(ptype);
        Py_XDECREF(pvalue);
        Py_XDECREF(ptraceback);
        return pstrdup("Could not convert error to string");
    }
    
    const char *str = PyUnicode_AsUTF8(str_obj);
    char *result = str ? pstrdup(str) : pstrdup("Could not get error message");
    
    Py_DECREF(str_obj);
    Py_XDECREF(ptype);
    Py_XDECREF(pvalue);
    Py_XDECREF(ptraceback);
    
    return result;
}

// Function to initialize Python and load Gemma model
static void initialize_python(void) {
    if (!Py_IsInitialized()) {
        // Initialize Python with full error checking
        Py_Initialize();
        if (!Py_IsInitialized()) {
            elog(ERROR, "Failed to initialize Python interpreter");
            return;
        }
        
        // Add module path to Python path
        PyObject *sys_path = PySys_GetObject("path");
        if (sys_path == NULL) {
            elog(ERROR, "Could not get sys.path");
            return;
        }
        
        PyObject *path = PyUnicode_FromString("/home/cybrosys/PSQL/postgresql/contrib/ai_extension");
        if (path == NULL) {
            elog(ERROR, "Could not create path string: %s", get_python_error());
            return;
        }
        
        if (PyList_Append(sys_path, path) < 0) {
            elog(ERROR, "Could not append to sys.path: %s", get_python_error());
            Py_DECREF(path);
            return;
        }
        Py_DECREF(path);
        
        // Import our Python module with detailed error reporting
        PyObject *module_name = PyUnicode_FromString("gemma_query_generator");
        if (!module_name) {
            elog(ERROR, "Could not create module name string: %s", get_python_error());
            return;
        }
        
        py_module = PyImport_Import(module_name);
        Py_DECREF(module_name);
        
        if (py_module == NULL) {
            char *error = get_python_error();
            elog(ERROR, "Failed to import gemma_query_generator module: %s", error);
            pfree(error);
            return;
        }
        
        // Get the generate_sql_query function with error details
        py_generate_func = PyObject_GetAttrString(py_module, "generate_sql_query");
        if (!py_generate_func || !PyCallable_Check(py_generate_func)) {
            char *error = get_python_error();
            elog(ERROR, "Cannot find or call generate_sql_query function: %s", error);
            pfree(error);
            Py_XDECREF(py_generate_func);
            Py_DECREF(py_module);
            return;
        }
    }
}

// Function to cleanup Python
static void cleanup_python(void) {
    if (Py_IsInitialized()) {
        Py_XDECREF(py_generate_func);
        Py_XDECREF(py_module);
        Py_Finalize();
    }
}

// Main function to generate SQL query
PG_FUNCTION_INFO_V1(generate_ai_query);

Datum
generate_ai_query(PG_FUNCTION_ARGS) {
    text *nl_query = PG_GETARG_TEXT_P(0);
    text *schema_context = PG_GETARG_TEXT_P(1);
    char *nl_query_str = text_to_cstring(nl_query);
    char *schema_context_str = text_to_cstring(schema_context);
    
    // Initialize Python with error handling
    PG_TRY();
    {
        initialize_python();
    }
    PG_CATCH();
    {
        PG_RE_THROW();
    }
    PG_END_TRY();
    
    // Create Python arguments
    PyObject *args = PyTuple_New(2);
    if (!args) {
        char *error = get_python_error();
        elog(ERROR, "Failed to create argument tuple: %s", error);
        pfree(error);
        PG_RETURN_NULL();
    }
    
    PyTuple_SetItem(args, 0, PyUnicode_FromString(nl_query_str));
    PyTuple_SetItem(args, 1, PyUnicode_FromString(schema_context_str));
    
    // Call Python function with error handling
    PyObject *result = PyObject_CallObject(py_generate_func, args);
    Py_DECREF(args);
    
    if (result == NULL) {
        char *error = get_python_error();
        elog(ERROR, "Failed to generate SQL query: %s", error);
        pfree(error);
        PG_RETURN_NULL();
    }
    
    // Convert Python result to C string with error handling
    const char *sql_query = PyUnicode_AsUTF8(result);
    if (!sql_query) {
        char *error = get_python_error();
        elog(ERROR, "Failed to convert result to string: %s", error);
        pfree(error);
        Py_DECREF(result);
        PG_RETURN_NULL();
    }
    
    text *query_text = cstring_to_text(sql_query);
    Py_DECREF(result);
    pfree(nl_query_str);
    pfree(schema_context_str);
    
    PG_RETURN_TEXT_P(query_text);
}

// Module callbacks
void
_PG_init(void) {
    initialize_python();
}

void
_PG_fini(void) {
    cleanup_python();
}