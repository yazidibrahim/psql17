# Path to pg_config
PG_CONFIG = /home/cybrosys/PSQL/postgresql/bin/pg_config
# Debugging flags
MAKEFLAGS += --warn-undefined-variables
SHELL = /bin/bash
# PostgreSQL build system
PGXS := $(shell $(PG_CONFIG) --pgxs)
$(info PGXS path: $(PGXS))
# Extension details
EXTENSION = new
EXTVERSION = 1.0
# Compilation settings
CC = gcc
CFLAGS = -Wall -std=gnu99 -Wmissing-prototypes -Wpointer-arith -Wdeclaration-after-statement -Wendif-labels \
         -Wmissing-format-attribute -Wimplicit-fallthrough=3 -Wcast-function-type -Wshadow=compatible-local \
         -Wformat-security -fno-strict-aliasing -fwrapv -fexcess-precision=standard \
         -g -O2 -fPIC -march=native -flto -I/home/cybrosys/PSQL/postgresql/include/server
include $(shell $(PG_CONFIG) --pgxs)



PG_CPPFLAGS = -I/home/cybrosys/PSQL/postgresql/include/server

# Source and object files
SRCS = new.c
OBJS = new.o
# Targets
all: new.so
# Explicit compilation rules
new.o: new.c
	@echo "Compiling $<"
	$(CC) $(CFLAGS) $(PG_CPPFLAGS) -c $< -o $@
new.so: new.o
	@echo "Linking $@"
	$(CC) -shared -o $@ $< -L$(shell $(PG_CONFIG) --libdir) -lc
# Install target
install: new.so
	@echo "Installing extension"
	mkdir -p $(shell $(PG_CONFIG) --pkglibdir)
	cp new.so $(shell $(PG_CONFIG) --pkglibdir)
	mkdir -p $(shell $(PG_CONFIG) --sharedir)/extension
	cp new--$(EXTVERSION).sql $(shell $(PG_CONFIG) --sharedir)/extension
	cp new.control $(shell $(PG_CONFIG) --sharedir)/extension
# Clean target
clean:
	@echo "Cleaning build artifacts"
	rm -f new.o new.so
.PHONY: all install clean




