CREATE FUNCTION psql_autocomplete_enable() RETURNS void
AS '$libdir/psql_autocomplete', 'psql_autocomplete_enable'
LANGUAGE C VOLATILE;

\echo Use "PSQL_AUTOCOMPLETE=1 psql -d <dbname>" to enable autocompletion after running CREATE EXTENSION