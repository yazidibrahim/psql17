MODULES = gguf_ex
EXTENSION = gguf_ex
DATA = gguf_ex--1.0.sql

# Update paths to use local deps directory
PG_CPPFLAGS = -I$(PWD)/deps/llama.cpp
SHLIB_LINK = -L$(PWD)/deps/llama.cpp/build -lllama

PG_CONFIG = /home/cybrosys/PSQL/postgresql/bin/pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)