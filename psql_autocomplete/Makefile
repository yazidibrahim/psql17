EXTENSION = psql_autocomplete
DATA = psql_autocomplete--1.0.sql
MODULE_big = psql_autocomplete
OBJS = psql_autocomplete.o
PG_CONFIG = /home/cybrosys/PSQL/postgresql/bin/pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
CFLAGS += -I$(shell $(PG_CONFIG) --includedir)
SHLIB_LINK += -lreadline  # Link against libreadline
include $(PGXS)