-- Completes the PostgreSQL extension
\echo Use "CREATE EXTENSION ai" to load this file. \quit

-- Version constants
CREATE OR REPLACE FUNCTION _ai_version()
RETURNS text AS $$
  SELECT '1.2.0'::text;
$$ LANGUAGE SQL IMMUTABLE;

-- Utility functions
CREATE OR REPLACE FUNCTION clean_sql_query(query_text text)
RETURNS text AS $$
DECLARE
    cleaned text;
BEGIN
    -- Basic cleaning
    cleaned := regexp_replace(query_text, '/\*.*?\*/', '', 'g');  -- Remove multi-line comments
    cleaned := regexp_replace(cleaned, '--.*$', '', 'g');         -- Remove single-line comments
    cleaned := regexp_replace(cleaned, '\s+', ' ', 'g');          -- Normalize whitespace
    cleaned := regexp_replace(cleaned, ';\s*$', '');              -- Remove trailing semicolon
    cleaned := btrim(cleaned);                                    -- Remove leading/trailing whitespace
    
    -- Validate basic SQL structure
    IF cleaned !~ '^\s*(SELECT|INSERT|UPDATE|DELETE)\s+.*$' THEN
        RETURN NULL;
    END IF;
    
    RETURN cleaned;
END;
$$ LANGUAGE plpgsql STRICT;

-- Create extension function
CREATE FUNCTION generate_ai_query(text, text)
RETURNS text
AS 'ai', 'generate_ai_query'
LANGUAGE C STRICT;

-- Enhanced schema information function
CREATE OR REPLACE FUNCTION get_schema_info()
RETURNS text AS $$
DECLARE
    schema_info text;
BEGIN
    WITH RECURSIVE 
    table_info AS (
        SELECT 
            c.table_name,
            string_agg(
                format(
                    '%I %s%s%s',
                    c.column_name,
                    c.data_type,
                    CASE WHEN c.character_maximum_length IS NOT NULL 
                         THEN format('(%s)', c.character_maximum_length)
                         ELSE '' 
                    END,
                    CASE WHEN c.is_nullable = 'NO' THEN ' NOT NULL' ELSE '' END
                ),
                E',\n    ' ORDER BY c.ordinal_position
            ) as columns,
            string_agg(
                CASE WHEN tc.constraint_type = 'PRIMARY KEY' 
                     THEN c.column_name 
                END,
                ', '
            ) as primary_keys,
            string_agg(
                CASE WHEN tc.constraint_type = 'FOREIGN KEY' THEN
                    format(
                        'FOREIGN KEY (%I) REFERENCES %I(%I)',
                        kcu.column_name,
                        ccu.table_name,
                        ccu.column_name
                    )
                END,
                E',\n    '
            ) as foreign_keys,
            obj_description(to_regclass(quote_ident(c.table_name)::text), 'pg_class') as table_description
        FROM 
            information_schema.columns c
            LEFT JOIN information_schema.key_column_usage kcu
                ON c.table_name = kcu.table_name 
                AND c.column_name = kcu.column_name
            LEFT JOIN information_schema.table_constraints tc
                ON kcu.constraint_name = tc.constraint_name
            LEFT JOIN information_schema.constraint_column_usage ccu
                ON tc.constraint_name = ccu.constraint_name
        WHERE 
            c.table_schema = 'public'
            AND c.table_name NOT IN ('ai_version', 'ai_config', 'ai_query_log')
        GROUP BY 
            c.table_name
    )
    SELECT 
        string_agg(
            format(
                '-- %s\nCREATE TABLE %I (\n    %s%s%s\n);',
                COALESCE(table_description, 'No description available'),
                table_name,
                columns,
                CASE WHEN primary_keys IS NOT NULL 
                     THEN E',\n    PRIMARY KEY(' || primary_keys || ')'
                     ELSE '' 
                END,
                CASE WHEN foreign_keys IS NOT NULL 
                     THEN E',\n    ' || foreign_keys
                     ELSE '' 
                END
            ),
            E'\n\n'
        )
    INTO schema_info
    FROM table_info;
    
    RETURN COALESCE(schema_info, 'No tables found in public schema');
END;
$$ LANGUAGE plpgsql;

-- Query execution with safety checks
CREATE OR REPLACE FUNCTION execute_generated_query(query_text text) 
RETURNS TABLE (result json) AS $$
DECLARE
    cleaned_query text;
    query_start_time timestamptz;
    timeout_seconds int;
    max_rows int;
BEGIN
    -- Get configuration
    SELECT setting_value::int INTO timeout_seconds 
    FROM ai_config 
    WHERE setting_name = 'timeout_seconds';
    
    SELECT setting_value::int INTO max_rows 
    FROM ai_config 
    WHERE setting_name = 'max_rows';
    
    -- Clean and validate query
    cleaned_query := clean_sql_query(query_text);
    
    IF cleaned_query IS NULL THEN
        RAISE EXCEPTION 'Invalid SQL query structure';
    END IF;
    
    -- Set statement timeout
    EXECUTE format('SET LOCAL statement_timeout = %s', timeout_seconds * 1000);
    
    -- Log query execution
    INSERT INTO ai_query_log (query_text, cleaned_query)
    VALUES (query_text, cleaned_query)
    RETURNING execution_start INTO query_start_time;
    
    -- Execute query with row limit
    RETURN QUERY EXECUTE 
        format(
            'SELECT row_to_json(t) as result FROM (%s LIMIT %s) t',
            cleaned_query,
            COALESCE(max_rows, 1000)
        );
    
    -- Update log with success
    UPDATE ai_query_log 
    SET 
        execution_end = clock_timestamp(),
        status = 'SUCCESS'
    WHERE execution_start = query_start_time;
    
EXCEPTION WHEN OTHERS THEN
    -- Log error
    IF query_start_time IS NOT NULL THEN
        UPDATE ai_query_log 
        SET 
            execution_end = clock_timestamp(),
            status = 'ERROR',
            error_message = SQLERRM
        WHERE execution_start = query_start_time;
    END IF;
    
    -- Return error as JSON
    RETURN QUERY SELECT row_to_json(r) FROM (
        SELECT 
            query_text as original_query,
            cleaned_query as cleaned_query,
            SQLERRM as error
    ) r;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Query logging table
CREATE TABLE ai_query_log (
    id bigserial PRIMARY KEY,
    query_text text NOT NULL,
    cleaned_query text,
    execution_start timestamptz DEFAULT clock_timestamp(),
    execution_end timestamptz,
    status text DEFAULT 'PENDING',
    error_message text,
    CONSTRAINT valid_status CHECK (status IN ('PENDING', 'SUCCESS', 'ERROR'))
);

-- Main AI query function
CREATE OR REPLACE FUNCTION ai_query(query_text text)
RETURNS TABLE (result json) AS $$
DECLARE
    generated_sql text;
    cleaned_sql text;
    schema_info text;
    max_retries int;
    current_try int := 1;
BEGIN
    -- Get configuration
    SELECT setting_value::int INTO max_retries 
    FROM ai_config 
    WHERE setting_name = 'max_retries';
    
    -- Get schema information
    schema_info := get_schema_info();
    
    -- Retry loop for query generation
    LOOP
        -- Generate SQL query
        generated_sql := generate_ai_query(query_text, schema_info);
        cleaned_sql := clean_sql_query(generated_sql);
        
        -- Log attempt
        RAISE NOTICE 'Attempt %: Generated SQL: %', current_try, generated_sql;
        
        -- Check if we got a valid query
        IF cleaned_sql IS NOT NULL AND cleaned_sql !~ '^SELECT ''(Error|Invalid)' THEN
            -- Execute and return results
            RETURN QUERY SELECT * FROM execute_generated_query(cleaned_sql);
            RETURN;
        END IF;
        
        -- Check if we should retry
        IF current_try >= COALESCE(max_retries, 3) THEN
            RAISE NOTICE 'Max retries reached. Using last generated query.';
            RETURN QUERY SELECT * FROM execute_generated_query(generated_sql);
            RETURN;
        END IF;
        
        current_try := current_try + 1;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Version tracking and configuration
CREATE TABLE ai_version (
    version text NOT NULL,
    installed_on timestamptz DEFAULT now()
);

CREATE TABLE ai_config (
    setting_name text PRIMARY KEY,
    setting_value text NOT NULL,
    description text,
    last_modified timestamptz DEFAULT now(),
    modified_by text DEFAULT current_user,
    CONSTRAINT valid_setting CHECK (
        setting_name IN (
            'max_retries', 
            'log_queries', 
            'timeout_seconds',
            'max_rows'
        )
    )
);

-- Configuration management functions
CREATE OR REPLACE FUNCTION update_ai_config(
    p_setting_name text,
    p_setting_value text
) RETURNS void AS $$
BEGIN
    INSERT INTO ai_config (setting_name, setting_value)
    VALUES (p_setting_name, p_setting_value)
    ON CONFLICT (setting_name) DO UPDATE
    SET 
        setting_value = p_setting_value,
        last_modified = now(),
        modified_by = current_user;
END;
$$ LANGUAGE plpgsql;

-- Insert default configurations
INSERT INTO ai_config (setting_name, setting_value, description) VALUES
    ('max_retries', '3', 'Maximum number of query generation retries'),
    ('log_queries', 'true', 'Whether to log generated queries'),
    ('timeout_seconds', '30', 'Query execution timeout in seconds'),
    ('max_rows', '1000', 'Maximum number of rows to return')
ON CONFLICT (setting_name) DO NOTHING;

-- Insert version
INSERT INTO ai_version (version) 
VALUES (_ai_version());

-- Permissions
GRANT SELECT ON ai_config TO public;
GRANT SELECT ON ai_version TO public;
GRANT SELECT ON ai_query_log TO public;
GRANT EXECUTE ON FUNCTION clean_sql_query(text) TO public;
GRANT EXECUTE ON FUNCTION generate_ai_query(text, text) TO public;
GRANT EXECUTE ON FUNCTION execute_generated_query(text) TO public;
GRANT EXECUTE ON FUNCTION get_schema_info() TO public;
GRANT EXECUTE ON FUNCTION ai_query(text) TO public;
GRANT EXECUTE ON FUNCTION update_ai_config(text, text) TO public;

-- Create indexes
CREATE INDEX idx_query_log_status ON ai_query_log(status);
CREATE INDEX idx_query_log_execution_start ON ai_query_log(execution_start);