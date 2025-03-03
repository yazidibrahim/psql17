MODULE_big = ai
OBJS = ai.o
EXTENSION = ai
DATA = ai--1.0.0.sql

PG_CONFIG = /home/cybrosys/PSQL/postgresql/bin/pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)

# Python specific paths for Ubuntu
PYTHON_VERSION = 3.10
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)
PYTHON_CONFIG = /usr/lib/python$(PYTHON_VERSION)/config-$(PYTHON_VERSION)-x86_64-linux-gnu

# Compiler and linker flags
override CPPFLAGS += -I$(PYTHON_INCLUDE)
override SHLIB_LINK += -L$(PYTHON_CONFIG) -L/usr/lib/x86_64-linux-gnu -lpython$(PYTHON_VERSION) -lcrypt -ldl -lm

include $(PGXS)

# Debug target
debug:
	@echo "Python include path: $(PYTHON_INCLUDE)"
	@echo "Python config path: $(PYTHON_CONFIG)"
	@echo "CPPFLAGS: $(CPPFLAGS)"
	@echo "SHLIB_LINK: $(SHLIB_LINK)"