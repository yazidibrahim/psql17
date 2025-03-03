
# File: psql_llama--1.0.sql
-- complain if script is sourced in psql, rather than via CREATE EXTENSION
\echo Use "CREATE EXTENSION psql_llama" to load this file. \quit

CREATE FUNCTION llama_generate(text)
RETURNS text
AS 'psql_llama', 'llama_generate'
LANGUAGE C STRICT;
