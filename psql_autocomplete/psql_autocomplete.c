#include "postgres.h"
#include "fmgr.h"  /* For PG_MODULE_MAGIC and PG_FUNCTION_INFO_V1 */
#include "/home/cybrosys/PSQL/postgresql/include/libpq-fe.h"
#include <readline/readline.h>
#include <readline/history.h>
#include <string.h>
#include <stdlib.h>

#define MAX_SUGGESTIONS 100

PG_MODULE_MAGIC;

static char *suggestions[MAX_SUGGESTIONS];
static int num_suggestions = 0;
static PGconn *conn = NULL;

void _PG_init(void);
static char *completion_generator(const char *text, int state);
static char **psql_completion(const char *text, int start, int end);
static void fetch_completions(const char *text);

PG_FUNCTION_INFO_V1(psql_autocomplete_enable);
Datum psql_autocomplete_enable(PG_FUNCTION_ARGS);

/* Initialize the module (client-side for psql) */
void
_PG_init(void)
{
    if (getenv("PSQL_AUTOCOMPLETE"))
    {
        rl_attempted_completion_function = psql_completion;
        printf("Autocomplete loaded. Start typing and press Tab for suggestions.\n");
    }
}

/* Server-side function to signal activation */
Datum
psql_autocomplete_enable(PG_FUNCTION_ARGS)
{
    elog(NOTICE, "psql_autocomplete enabled. Run psql with PSQL_AUTOCOMPLETE=1 to activate autocompletion.");
    PG_RETURN_VOID();
}

/* Generate completion matches */
static char *
completion_generator(const char *text, int state)
{
    static int list_index;

    if (!state)
        list_index = 0;

    while (list_index < num_suggestions)
    {
        char *suggestion = suggestions[list_index++];
        if (strncmp(suggestion, text, strlen(text)) == 0)
            return strdup(suggestion);
    }

    return NULL;
}

/* Fetch schema-based completions */
static void
fetch_completions(const char *text)
{
    PGresult *res;
    char query[512];
    const char *dbname;
    char conninfo[256];
    int i;

    for (i = 0; i < num_suggestions; i++)
        free(suggestions[i]);
    num_suggestions = 0;

    if (!conn || PQstatus(conn) != CONNECTION_OK)
    {
        if (conn)
            PQfinish(conn);
        dbname = getenv("PGDATABASE") ? getenv("PGDATABASE") : "postgres";
        snprintf(conninfo, sizeof(conninfo), "dbname=%s", dbname);
        conn = PQconnectdb(conninfo);
        if (PQstatus(conn) != CONNECTION_OK)
        {
            fprintf(stderr, "Autocomplete connection failed: %s\n", PQerrorMessage(conn));
            PQfinish(conn);
            conn = NULL;
            return;
        }
    }

    snprintf(query, sizeof(query),
             "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' "
             "UNION SELECT column_name FROM information_schema.columns WHERE table_schema = 'public' "
             "UNION SELECT unnest(ARRAY['SELECT', 'FROM', 'WHERE', 'INSERT', 'UPDATE', 'DELETE']) "
             "WHERE upper(unnest) LIKE upper('%s%%')", text);
    res = PQexec(conn, query);

    if (PQresultStatus(res) == PGRES_TUPLES_OK)
    {
        for (i = 0; i < PQntuples(res) && i < MAX_SUGGESTIONS; i++)
            suggestions[i] = strdup(PQgetvalue(res, i, 0));
        num_suggestions = PQntuples(res) < MAX_SUGGESTIONS ? PQntuples(res) : MAX_SUGGESTIONS;
    }

    PQclear(res);
}

/* Custom completion function */
static char **
psql_completion(const char *text, int start, int end)
{
    rl_completion_append_character = ' ';
    rl_completion_suppress_append = (strlen(text) == 0);

    fetch_completions(text);

    if (num_suggestions == 0)
        return NULL;

    return rl_completion_matches(text, completion_generator);
}