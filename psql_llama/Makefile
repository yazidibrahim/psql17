MODULES = gguf_ex
EXTENSION = gguf_ex
DATA = gguf_ex--1.0.sql

# Add llama.cpp include and library paths
# Adjust these paths according to where you cloned llama.cpp
PG_CONFIG = /home/cybrosys/PSQL/postgresql/bin/pg_config
SHLIB_LINK = -L$(HOME)/llama.cpp/build -lllama

PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)