-- complain if script is sourced in psql, rather than via CREATE EXTENSION
\echo Use "CREATE EXTENSION llm" to load this file. \quit

CREATE FUNCTION init_llm(text)
    RETURNS boolean
    AS 'MODULE_PATHNAME', 'init_llm'
    LANGUAGE C STRICT;

CREATE FUNCTION llm_generate(text)
    RETURNS text
    AS 'MODULE_PATHNAME', 'llm_generate'
    LANGUAGE C STRICT;
    