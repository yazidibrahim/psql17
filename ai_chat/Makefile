MODULE_big = ai_chat
OBJS = ai_chat.o
EXTENSION = ai_chat
DATA = ai_chat--1.0.sql

PG_CONFIG = pg_config

# Python configuration - using simple, direct paths
PG_CPPFLAGS = -I/usr/include/python3.10
SHLIB_LINK = -L/usr/lib/x86_64-linux-gnu -lpython3.10 -lcrypt -ldl -lm

ifdef USE_PGXS
PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
else
subdir = contrib/ai_chat
top_builddir = ../..
include $(top_builddir)/src/Makefile.global
include $(top_srcdir)/contrib/contrib-global.mk
endif


# Install llama-cpp-python if not installed
check_llama:
	@python3.10 -c "import llama_cpp" 2>/dev/null || (echo 'Installing llama-cpp-python...'; python3.10 -m pip install llama-cpp-python)

# # Override the `all` target to include the package check before building
# all: check_llama
#   $(MAKE) -C $(subdir)

