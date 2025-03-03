LLAMA_PATH = postgresql/contrib/llm_extension/include/llama.cpp

MODULE_big = llm
OBJS = src/llm.o
EXTENSION = llm
DATA = llm--1.0.sql

PG_CONFIG = /home/cybrosys/PSQL/postgresql/bin/pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)

# Add include paths
PG_CPPFLAGS = -I$(LLAMA_PATH)/include -I./include

# Add any necessary library paths and libraries
SHLIB_LINK = -L$(LLAMA_PATH)/build/bin -lllama


include $(PGXS)