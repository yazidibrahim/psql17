\echo Use "CREATE EXTENSION pai" to load this file. \quit

-- Create schema
CREATE SCHEMA pai;

-- Create functions to manage Python path
CREATE OR REPLACE FUNCTION pai.get_extension_path()
RETURNS text AS $$
    import os
    from urllib.parse import urlparse
    
    # Get the path to the extension directory
    sql_file = __file__
    extension_dir = os.path.dirname(os.path.dirname(sql_file))
    return extension_dir
$$ LANGUAGE plpython3u;

CREATE OR REPLACE FUNCTION pai.setup_python_path()
RETURNS void AS $$
    import sys
    import os
    
    # Use fixed path where the extension is installed
    extension_path = '/home/cybrosys/PSQL/postgresql/contrib/pai_extension'
    python_path = os.path.join(extension_path, 'python')
    
    if python_path not in sys.path:
        sys.path.append(python_path)
$$ LANGUAGE plpython3u;


-- Function to get schema information
CREATE OR REPLACE FUNCTION pai.get_schema_info()
RETURNS TABLE (
    table_name text,
    column_name text,
    data_type text,
    constraints text,
    foreign_keys text
) AS $$
BEGIN
    RETURN QUERY
    WITH fk_info AS (
        SELECT
            conrelid::regclass::text AS table_name,
            a.attname AS column_name,
            confrelid::regclass::text AS foreign_table,
            af.attname AS foreign_column
        FROM pg_constraint c
        JOIN pg_attribute a ON a.attnum = ANY(c.conkey) AND a.attrelid = c.conrelid
        JOIN pg_attribute af ON af.attnum = ANY(c.confkey) AND af.attrelid = c.confrelid
        WHERE c.contype = 'f'
    )
    SELECT 
        t.relname::text as table_name,
        a.attname::text as column_name,
        pg_catalog.format_type(a.atttypid, a.atttypmod)::text as data_type,
        COALESCE(
            string_agg(
                DISTINCT con.conname || ' (' || 
                CASE con.contype
                    WHEN 'p' THEN 'PRIMARY KEY'
                    WHEN 'u' THEN 'UNIQUE'
                    WHEN 'c' THEN 'CHECK'
                    WHEN 'f' THEN 'FOREIGN KEY'
                    ELSE con.contype::text
                END || ')',
                ', '
            ),
            'none'
        ) as constraints,
        COALESCE(
            string_agg(
                DISTINCT 'REFERENCES ' || fk.foreign_table || '(' || fk.foreign_column || ')',
                ', '
            ),
            'none'
        ) as foreign_keys
    FROM pg_catalog.pg_class t
    JOIN pg_catalog.pg_attribute a ON t.oid = a.attrelid
    LEFT JOIN pg_catalog.pg_constraint con ON (
        t.oid = con.conrelid AND 
        a.attnum = ANY(con.conkey)
    )
    LEFT JOIN fk_info fk ON (
        t.relname::text = fk.table_name AND
        a.attname::text = fk.column_name
    )
    WHERE t.relkind = 'r'
    AND NOT a.attisdropped
    AND a.attnum > 0
    AND t.relnamespace = (
        SELECT oid 
        FROM pg_catalog.pg_namespace 
        WHERE nspname = 'public'
    )
    GROUP BY 
        t.relname,
        a.attname,
        a.atttypid,
        a.atttypmod;
END;
$$ LANGUAGE plpgsql STABLE;

-- Function to generate SQL queries using Gemma
-- Update the generate_query function
-- Update the generate_query function
CREATE OR REPLACE FUNCTION pai.generate_query(
    query_description text,
    OUT generated_sql text
) AS $$
    import sys
    import os
    import plpy
    from datetime import datetime
    
    def log_query(query_text, status, error_msg=None, generated_sql=None):
        plpy.execute(
            """
            INSERT INTO ai_query_log 
                (query_text, status, error_message, cleaned_query, execution_start, execution_end)
            VALUES
                ($1, $2, $3, $4, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """,
            [query_text, status, error_msg, generated_sql]
        )
    
    try:
        # Setup Python path
        plpy.execute("SELECT pai.setup_python_path()")
        
        # Import GemmaSQL
        try:
            from gemma_integration import GemmaSQL
        except ImportError as e:
            error_msg = f"Failed to import GemmaSQL: {str(e)}"
            log_query(query_description, 'ERROR', error_msg)
            return error_msg
        
        # Get schema information
        schema_result = plpy.execute("""
            SELECT string_agg(
                format(
                    'Table: %s, Column: %s (%s), Constraints: %s',
                    table_name, column_name, data_type, constraints
                ),
                E'\n'
            ) as info
            FROM pai.get_schema_info()
        """)
        
        if len(schema_result) == 0:
            error_msg = "Failed to retrieve schema information"
            log_query(query_description, 'ERROR', error_msg)
            return error_msg
            
        schema_info = schema_result[0]['info']
        
        # Generate query
        try:
            gemma = GemmaSQL.get_instance()
            generated_sql = gemma.generate_sql(schema_info, query_description)
            if generated_sql:
                log_query(query_description, 'SUCCESS', None, generated_sql)
                return generated_sql
            else:
                error_msg = "Failed to generate SQL query"
                log_query(query_description, 'ERROR', error_msg)
                return error_msg
        except Exception as e:
            error_msg = f"Error generating SQL query: {str(e)}"
            log_query(query_description, 'ERROR', error_msg)
            return error_msg
            
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        log_query(query_description, 'ERROR', error_msg)
        return error_msg
$$ LANGUAGE plpython3u;

-- Function to validate and execute generated query
CREATE OR REPLACE FUNCTION pai.execute_generated_query(
    query_description text,
    OUT query text,
    OUT results json
) AS $$
BEGIN
    -- Generate the query
    query := pai.generate_query(query_description);
    
    -- Execute the query and capture results
    EXECUTE format('
        WITH query_results AS (%s)
        SELECT json_agg(row_to_json(query_results))
        FROM query_results
    ', query) INTO results;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION pai.check_installation()
RETURNS TABLE (
    component text,
    status text,
    details text
) AS $$
BEGIN
    -- Check extension directory
    RETURN QUERY
    SELECT 
        'Extension Directory' as component,
        CASE WHEN EXISTS (
            SELECT 1 FROM pg_ls_dir('/home/cybrosys/PSQL/postgresql/contrib/pai_extension')
        ) THEN 'OK' ELSE 'Missing' END as status,
        '/home/cybrosys/PSQL/postgresql/contrib/pai_extension' as details;
        
    -- Check Python module
    RETURN QUERY
    SELECT 
        'Python Module' as component,
        CASE WHEN EXISTS (
            SELECT 1 FROM pg_ls_dir('/home/cybrosys/PSQL/postgresql/contrib/pai_extension/python')
            WHERE pg_ls_dir = 'gemma_integration.py'
        ) THEN 'OK' ELSE 'Missing' END as status,
        '/home/cybrosys/PSQL/postgresql/contrib/pai_extension/python/gemma_integration.py' as details;
END;
$$ LANGUAGE plpgsql;