CREATE OR REPLACE FUNCTION micro_newcache_init()
RETURNS boolean
AS '/home/cybrosys/PSQL/postgresql/lib/new.so', 'micro_newcache_init'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION micro_newcache_set(key text, value text)
RETURNS boolean
AS '/home/cybrosys/PSQL/postgresql/lib/new.so', 'micro_newcache_set'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION micro_newcache_get(key text)
RETURNS text
AS '/home/cybrosys/PSQL/postgresql/lib/new.so', 'micro_newcache_get'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION micro_newcache_vacuum()
RETURNS integer
AS '/home/cybrosys/PSQL/postgresql/lib/new.so', 'micro_newcache_vacuum'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION micro_newcache_cleanup()
RETURNS boolean
AS '/home/cybrosys/PSQL/postgresql/lib/new.so', 'micro_newcache_cleanup'
LANGUAGE C STRICT;
