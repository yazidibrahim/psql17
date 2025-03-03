CREATE OR REPLACE FUNCTION ai_chat_response(text)
RETURNS text
AS 'ai_chat', 'ai_chat_response'
LANGUAGE C STRICT;

COMMENT ON FUNCTION ai_chat_response(text) IS 'Returns AI-generated response using GGUF model';

