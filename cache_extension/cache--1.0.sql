-- CREATE FUNCTION cache_init()
-- RETURNS boolean
-- AS '/home/cybrosys/PSQL/postgresql/lib/cache.so', 'cache_init'
-- LANGUAGE C;

-- CREATE FUNCTION cache_set(key text, value text)
-- RETURNS boolean
-- AS '/home/cybrosys/PSQL/postgresql/lib/cache.so', 'cache_set'
-- LANGUAGE C;

-- CREATE FUNCTION cache_get(key text)
-- RETURNS text
-- AS '/home/cybrosys/PSQL/postgresql/lib/cache.so', 'cache_get'
-- LANGUAGE C;

-- CREATE FUNCTION cache_vacuum()
-- RETURNS integer
-- AS '/home/cybrosys/PSQL/postgresql/lib/cache.so', 'cache_vacuum'
-- LANGUAGE C;


-- CREATE FUNCTION cache_cleanup() RETURNS BOOLEAN AS '/home/cybrosys/PSQL/postgresql/lib/cache.so', 'cache_cleanup' LANGUAGE C;







-- Create the functions
-- CREATE FUNCTION kv_init(capacity integer) RETURNS void
--     AS '$libdir/cache.so', 'kv_init'
--     LANGUAGE c;

CREATE FUNCTION kv_set(key text, value text) RETURNS void
    AS '$libdir/cache.so', 'kv_set'
    LANGUAGE c;


CREATE FUNCTION kv_get(key text) RETURNS text
    AS '$libdir/cache.so', 'kv_get'
    LANGUAGE c;

-- CREATE FUNCTION kv_clear() RETURNS void
--     AS '$libdir/cache.so', 'kv_clear'
--     LANGUAGE c;


-- CREATE OR REPLACE FUNCTION kv_stats()
-- RETURNS text
-- AS '$libdir/cache.so', 'kv_stats'
-- LANGUAGE C STRICT;

























-- CREATE FUNCTION ultra_cache_set(key TEXT, value TEXT) RETURNS BOOLEAN
-- AS '/home/cybrosys/PSQL/postgresql/lib/cache.so', 'ultra_cache_set' LANGUAGE C STRICT;

-- CREATE FUNCTION ultra_cache_get(key TEXT) RETURNS TEXT
-- AS '/home/cybrosys/PSQL/postgresql/lib/cache.so', 'ultra_cache_get' LANGUAGE C STRICT;

-- CREATE FUNCTION ultra_cache_delete(key TEXT) RETURNS BOOLEAN
-- AS '/home/cybrosys/PSQL/postgresql/lib/cache.so', 'ultra_cache_delete' LANGUAGE C STRICT;

-- CREATE FUNCTION ultra_cache_clear() RETURNS void
-- AS '/home/cybrosys/PSQL/postgresql/lib/cache.so', 'ultra_cache_clear' LANGUAGE C STRICT;

-- CREATE FUNCTION ultra_cache_stats() 
-- RETURNS TABLE(
--     gets bigint,
--     sets bigint,
--     hits bigint,
--     misses bigint,
--     evictions bigint,
--     total_compressed bigint,
--     total_uncompressed bigint
-- )
-- AS '/home/cybrosys/PSQL/postgresql/lib/cache.so', 'ultra_cache_stats' LANGUAGE C STRICT;



















-- CREATE OR REPLACE FUNCTION large_micro_cache_init(cache_size integer DEFAULT NULL, max_entry_size bigint DEFAULT NULL, entry_lifetime integer DEFAULT NULL)
-- RETURNS boolean
-- AS '/home/cybrosys/PSQL/postgresql/lib/cache.so', 'large_micro_cache_init'
-- LANGUAGE C STRICT;

-- CREATE OR REPLACE FUNCTION large_micro_cache_set(key text, value text)
-- RETURNS boolean
-- AS '/home/cybrosys/PSQL/postgresql/lib/cache.so', 'large_micro_cache_set'
-- LANGUAGE C STRICT;

-- CREATE OR REPLACE FUNCTION large_micro_cache_get(key text)
-- RETURNS text
-- AS '/home/cybrosys/PSQL/postgresql/lib/cache.so', 'large_micro_cache_get'
-- LANGUAGE C STRICT;

-- CREATE OR REPLACE FUNCTION large_micro_cache_get_config()
-- RETURNS TABLE(
--     cache_size integer, 
--     max_entry_size bigint, 
--     entry_lifetime integer
-- )
-- AS '/home/cybrosys/PSQL/postgresql/lib/cache.so', 'large_micro_cache_get_config'
-- LANGUAGE C STRICT;

-- CREATE OR REPLACE FUNCTION large_micro_cache_cleanup()
-- RETURNS boolean
-- AS '/home/cybrosys/PSQL/postgresql/lib/cache.so', 'large_micro_cache_cleanup'
-- LANGUAGE C STRICT;








-- Create extension functions
-- Initialize the Ultra Micro Cache

-- Create extension functions
-- Initialize the Ultra Simple Cache (note the function name change)
-- CREATE FUNCTION ultra_cache_config_init()
-- RETURNS boolean
-- AS '/home/cybrosys/PSQL/postgresql/lib/cache.so', 'ultra_cache_config_init'
-- LANGUAGE C STRICT;

-- Set a key-value pair in the cache (note the function name change)
-- CREATE FUNCTION ultra_simple_cache_init()
-- RETURNS boolean
-- AS '/home/cybrosys/PSQL/postgresql/lib/cache.so', 'ultra_simple_cache_init'
-- LANGUAGE C STRICT;

--    CREATE OR REPLACE FUNCTION ultra_micro_cache_init()
-- RETURNS boolean
-- AS '/home/cybrosys/PSQL/postgresql/lib/cache.so', 'ultra_micro_cache_init'
-- LANGUAGE C STRICT;

-- -- Set a key-value pair in the RAM Cache
-- CREATE OR REPLACE FUNCTION ultra_micro_cache_set(key text, value text)
-- RETURNS boolean
-- AS '/home/cybrosys/PSQL/postgresql/lib/cache.so', 'ultra_micro_cache_set'
-- LANGUAGE C STRICT;

-- -- Get a value by key from the ultra_micro Cache
-- CREATE OR REPLACE FUNCTION ultra_micro_cache_get(key text)
-- RETURNS text
-- AS '/home/cybrosys/PSQL/postgresql/lib/cache.so', 'ultra_micro_cache_get'
-- LANGUAGE C STRICT;

-- -- Delete a key from the RAM Cache
-- CREATE OR REPLACE FUNCTION ultra_micro_cache_cleanup(key text)
-- RETURNS boolean
-- AS '/home/cybrosys/PSQL/postgresql/lib/cache.so', 'ultra_micro_cache_cleanup'
-- LANGUAGE C STRICT;

-- Clear all entries in the ultra_micro Cache
-- CREATE OR REPLACE FUNCTION ultra_micro_cache_shutdown()
-- RETURNS boolean
-- AS '/home/cybrosys/PSQL/postgresql/lib/cache.so', 'ultra_micro_cache_shutdown'
-- LANGUAGE C STRICT;


-- Get the current size of the RAM Cache
-- CREATE OR REPLACE FUNCTION ram_cache_size()
-- RETURNS int
-- AS '/home/cybrosys/PSQL/postgresql/lib/cache.so', 'ram_cache_size'
-- LANGUAGE C STRICT;
    
    -- CREATE OR REPLACE FUNCTION ultra_cache_shutdown()
    -- RETURNS boolean
    -- AS '/home/cybrosys/PSQL/postgresql/lib/cache.so', 'ultra_cache_shutdown'
    -- LANGUAGE C STRICT;

   
-- -- Delete a specific key from the cache
-- CREATE FUNCTION ultra_micro_cache_delete(key text)
-- RETURNS boolean
-- AS '/home/cybrosys/PSQL/postgresql/lib/cache.so', 'ultra_micro_cache_delete'
-- LANGUAGE C STRICT;

-- Cleanup expired entries from the cache
-- CREATE FUNCTION ultra_micro_cache_cleanup()
-- RETURNS boolean
-- AS '/home/cybrosys/PSQL/postgresql/lib/cache.so', 'ultra_micro_cache_cleanup'
-- LANGUAGE C STRICT;

-- -- Completely shutdown and free cache resources
-- CREATE FUNCTION ultra_micro_cache_shutdown()
-- RETURNS boolean
-- AS $$
-- BEGIN
--     -- Perform cleanup of all entries
--     PERFORM ultra_micro_cache_cleanup();
    
--     -- Reset the cache (similar to _PG_fini in the C implementation)
--     UPDATE pg_proc 
--     SET proname = 'ultra_micro_cache_init' 
--     WHERE proname = 'ultra_micro_cache_shutdown';
    
--     RETURN true;
-- END;
-- $$ LANGUAGE plpgsql;


















-- CREATE FUNCTION ultra_micro_cache_get_config()
-- RETURNS record
-- AS '/home/cybrosys/PSQL/postgresql/lib/cache.so', 'simple_micro_cache_get_config'
-- LANGUAGE C STRICT;


















-- CREATE OR REPLACE FUNCTION cache_init()
-- RETURNS boolean AS 'cache', 'cache_init'
-- LANGUAGE C STRICT;

-- CREATE OR REPLACE FUNCTION cache_set(key text, value text)
-- RETURNS boolean AS 'cache', 'cache_set'
-- LANGUAGE C STRICT;

-- CREATE OR REPLACE FUNCTION cache_get(key text)
-- RETURNS text AS 'cache', 'cache_get'
-- LANGUAGE C STRICT;

-- CREATE OR REPLACE FUNCTION cache_cleanup()
-- RETURNS boolean AS 'cache', 'cache_cleanup'
-- LANGUAGE C STRICT;