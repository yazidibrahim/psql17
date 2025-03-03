# Path to pg_config
PG_CONFIG = /home/cybrosys/PSQL/postgresql/bin/pg_config

# Debugging flags
MAKEFLAGS += --warn-undefined-variables
SHELL = /bin/bash

# PostgreSQL build system
PGXS := $(shell $(PG_CONFIG) --pgxs)
$(info PGXS path: $(PGXS))

# Extension details
EXTENSION = cache
EXTVERSION = 1.0

# Compilation settings
CC = gcc
CFLAGS = -Wall -std=gnu99 -Wmissing-prototypes -Wpointer-arith -Wdeclaration-after-statement -Wendif-labels \
         -Wmissing-format-attribute -Wimplicit-fallthrough=3 -Wcast-function-type -Wshadow=compatible-local \
         -Wformat-security -fno-strict-aliasing -fwrapv -fexcess-precision=standard \
          -O3 -fPIC   -march=native -flto -I/home/cybrosys/PSQL/postgresql/include/server 
PG_CPPFLAGS = -I/home/cybrosys/PSQL/postgresql/include/server -I/usr/include

# Source and object files
SRCS = cache.c
OBJS = cache.o

# Targets
all: cache.so

# Explicit compilation rules
cache.o: cache.c
	@echo "Compiling $<"
	$(CC) $(CFLAGS) $(PG_CPPFLAGS) -c $< -o $@

cache.so: cache.o
	@echo "Linking $@"
	$(CC) -shared -o $@ $< -L$(shell $(PG_CONFIG) --libdir) -llz4 -lxxhash -lc

# Install target
install: cache.so
	@echo "Installing extension"
	mkdir -p $(shell $(PG_CONFIG) --pkglibdir)
	cp cache.so $(shell $(PG_CONFIG) --pkglibdir)
	mkdir -p $(shell $(PG_CONFIG) --sharedir)/extension
	cp cache--$(EXTVERSION).sql $(shell $(PG_CONFIG) --sharedir)/extension
	cp cache.control $(shell $(PG_CONFIG) --sharedir)/extension

# Clean target
clean:
	@echo "Cleaning build artifacts"
	rm -f cache.o cache.so
.PHONY: all install clean


# # Path to pg_config
# PG_CONFIG = /home/cybrosys/PSQL/postgresql/bin/pg_config

# # Debugging flags
# MAKEFLAGS += --warn-undefined-variables
# SHELL = /bin/bash

# # PostgreSQL build system
# PGXS := $(shell $(PG_CONFIG) --pgxs)
# $(info PGXS path: $(PGXS))

# # Extension details
# EXTENSION = cache
# EXTVERSION = 1.0

# # Compilation settings
# CC = gcc

# # Compiler flags
# CFLAGS = -Wall -std=gnu11 \
#          -Wmissing-prototypes -Wpointer-arith \
#          -Wdeclaration-after-statement -Wendif-labels \
#          -Wmissing-format-attribute -Wimplicit-fallthrough=3 \
#          -Wcast-function-type -Wshadow=compatible-local \
#          -Wformat-security -fno-strict-aliasing \
#          -fwrapv -fexcess-precision=standard \
#          -O3 -fPIC -march=native -flto

# # Include paths
# PG_CPPFLAGS = -I$(shell $(PG_CONFIG) --includedir)/server \
#               -I$(shell $(PG_CONFIG) --includedir) \
#               -I/usr/include/lz4

# # Libraries
# LIBS = -L$(shell $(PG_CONFIG) --libdir) \
#        -lpq \
#        -llz4 \
#        -lpgcommon \
#        -lpgport

# # Source and object files
# SRCS = cache.c
# OBJS = cache.o

# # Default target
# all: cache.so

# # Compilation rules
# %.o: %.c
# 	@echo "Compiling $<"
# 	$(CC) $(CFLAGS) $(PG_CPPFLAGS) -c $< -o $@

# # Linking rule
# cache.so: $(OBJS)
# 	@echo "Linking $@"
# 	$(CC) -shared -o $@ $^ $(LIBS) -lc

# # Include PGXS
# include $(PGXS)

# # Install target
# install: cache.so
# 	@echo "Installing extension"
# 	mkdir -p $(shell $(PG_CONFIG) --pkglibdir)
# 	cp cache.so $(shell $(PG_CONFIG) --pkglibdir)
# 	mkdir -p $(shell $(PG_CONFIG) --sharedir)/extension
# 	cp cache--$(EXTVERSION).sql $(shell $(PG_CONFIG) --sharedir)/extension
# 	cp cache.control $(shell $(PG_CONFIG) --sharedir)/extension

# # Clean target
# clean:
# 	@echo "Cleaning build artifacts"
# 	rm -f cache.o cache.so

# # Phony targets
# .PHONY: all install clean
























# # Path to pg_config
# PG_CONFIG = /home/cybrosys/PSQL/postgresql/bin/pg_config
# # Debugging flags
# MAKEFLAGS += --warn-undefined-variables
# SHELL = /bin/bash
# # PostgreSQL build system
# PGXS := $(shell $(PG_CONFIG) --pgxs)
# $(info PGXS path: $(PGXS))
# # Extension details
# EXTENSION = cache
# EXTVERSION = 1.0
# # Compilation settings
# CC = gcc
# CFLAGS = -Wall -std=gnu99 -Wmissing-prototypes -Wpointer-arith -Wdeclaration-after-statement -Wendif-labels \
#          -Wmissing-format-attribute -Wimplicit-fallthrough=3 -Wcast-function-type -Wshadow=compatible-local \
#          -Wformat-security -fno-strict-aliasing -fwrapv -fexcess-precision=standard \
#          -g -O2 -fPIC -march=native -flto -I/home/cybrosys/PSQL/postgresql/include/server
# include $(shell $(PG_CONFIG) --pgxs)



# PG_CPPFLAGS = -I/home/cybrosys/PSQL/postgresql/include/server

# # Source and object files
# SRCS = cache.c
# OBJS = cache.o
# # Targets
# all: cache.so
# # Explicit compilation rules
# cache.o: cache.c
# 	@echo "Compiling $<"
# 	$(CC) $(CFLAGS) $(PG_CPPFLAGS) -c $< -o $@
# cache.so: cache.o
# 	@echo "Linking $@"
# 	$(CC) -shared -o $@ $< -L$(shell $(PG_CONFIG) --libdir) -lc
# # Install target
# install: cache.so
# 	@echo "Installing extension"
# 	mkdir -p $(shell $(PG_CONFIG) --pkglibdir)
# 	cp cache.so $(shell $(PG_CONFIG) --pkglibdir)
# 	mkdir -p $(shell $(PG_CONFIG) --sharedir)/extension
# 	cp cache--$(EXTVERSION).sql $(shell $(PG_CONFIG) --sharedir)/extension
# 	cp cache.control $(shell $(PG_CONFIG) --sharedir)/extension
# # Clean target
# clean:
# 	@echo "Cleaning build artifacts"
# 	rm -f cache.o cache.so
# .PHONY: all install clean


















# # Verbose Makefile for PostgreSQL cache extension

# # Path to pg_config
# PG_CONFIG = ../../bin/pg_config

# # Debugging flags
# MAKEFLAGS += --warn-undefined-variables
# SHELL = /bin/bash

# # PostgreSQL build system
# PGXS := $(shell $(PG_CONFIG) --pgxs)
# $(info PGXS path: $(PGXS))

# # Extension details
# EXTENSION = cache
# EXTVERSION = 1.0

# # Compilation settings
# CC = gcc
# CFLAGS = -Wall -Wmissing-prototypes -Wpointer-arith -fPIC
# PG_CPPFLAGS = -I$(shell $(PG_CONFIG) --includedir-server)
# $(info Include path: $(PG_CPPFLAGS))

# # Source and object files
# SRCS = cache.c
# OBJS = cache.o

# # Targets
# all: cache.so

# # Explicit compilation rules
# cache.o: cache.c
# 	@echo "Compiling $<"
# 	$(CC) $(CFLAGS) $(PG_CPPFLAGS) -c $< -o $@

# cache.so: cache.o
# 	@echo "Linking $@"
# 	$(CC) -shared -o $@ $< -L$(shell $(PG_CONFIG) --libdir) -lc

# # Install target
# install: cache.so
# 	@echo "Installing extension"
# 	mkdir -p $(shell $(PG_CONFIG) --pkglibdir)
# 	cp cache.so $(shell $(PG_CONFIG) --pkglibdir)
# 	cp cache--$(EXTVERSION).sql $(shell $(PG_CONFIG) --sharedir)/extension
# 	cp cache.control $(shell $(PG_CONFIG) --sharedir)/extension

# # Clean target
# clean:
# 	@echo "Cleaning build artifacts"
# 	rm -f cache.o cache.so

# .PHONY: all install clean


# # Verbose Makefile for PostgreSQL cache extension
# # Path to pg_config
# PG_CONFIG = ../../bin/pg_config

# # Debugging flags
# MAKEFLAGS += --warn-undefined-variables
# SHELL = /bin/bash

# # PostgreSQL build system
# PGXS := $(shell $(PG_CONFIG) --pgxs)
# $(info PGXS path: $(PGXS))

# # Extension details
# EXTENSION = cache
# EXTVERSION = 1.0

# # Compilation settings
# CC = gcc
# # Aggressive optimization flags
# CFLAGS = -O3 -march=native -mtune=native \
#          -Wall -Wmissing-prototypes -Wpointer-arith -fPIC \
#          -Wdeclaration-after-statement -Werror=vla \
#          -Wmissing-format-attribute -Wformat-security \
#          -fno-strict-aliasing -fwrapv -fexcess-precision=standard

# PG_CPPFLAGS = -I$(shell $(PG_CONFIG) --includedir-server)
# $(info Include path: $(PG_CPPFLAGS))

# # Source and object files
# SRCS = cache.c
# OBJS = cache.o

# # Targets
# all: cache.so

# # Explicit compilation rules
# cache.o: cache.c
# 	@echo "Compiling $< with optimizations"
# 	$(CC) $(CFLAGS) $(PG_CPPFLAGS) -c $< -o $@

# cache.so: cache.o
# 	@echo "Linking $@"
# 	$(CC) -shared -o $@ $< -L$(shell $(PG_CONFIG) --libdir) -lc

# # Install target
# install: cache.so
# 	@echo "Installing extension"
# 	mkdir -p $(shell $(PG_CONFIG) --pkglibdir)
# 	cp cache.so $(shell $(PG_CONFIG) --pkglibdir)
# 	cp cache--$(EXTVERSION).sql $(shell $(PG_CONFIG) --sharedir)/extension
# 	cp cache.control $(shell $(PG_CONFIG) --sharedir)/extension

# # Clean target
# clean:
# 	@echo "Cleaning build artifacts"
# 	rm -f cache.o cache.so

# # Include PostgreSQL's build rules
# include $(PGXS)

# .PHONY: all install clean