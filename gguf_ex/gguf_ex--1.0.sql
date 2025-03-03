-- complain if script is sourced in psql, rather than via CREATE EXTENSION
\echo Use "CREATE EXTENSION gguf_ex" to load this file. \quit

-- Function to generate text from the model
CREATE FUNCTION generate_text(text)
RETURNS text
AS '$libdir/gguf_ex'
LANGUAGE C STRICT;

-- Function to cleanup model resources
CREATE FUNCTION cleanup_model()
RETURNS void
AS '$libdir/gguf_ex'
LANGUAGE C STRICT;