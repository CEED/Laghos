# Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
# the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
# reserved. See files LICENSE and NOTICE for details.
#
# This file is part of CEED, a collection of benchmarks, miniapps, software
# libraries and APIs for efficient high-order finite element and spectral
# element discretizations for exascale applications. For more information and
# source code availability see http://github.com/ceed.
#
# The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
# a collaborative effort of two U.S. Department of Energy organizations (Office
# of Science and the National Nuclear Security Administration) responsible for
# the planning and preparation of a capable exascale ecosystem, including
# software, applications, hardware, advanced system engineering and early
# testbed platforms, in support of the nation's exascale computing imperative.

define LAGHOS_HELP_MSG

Laghos dFEM makefile targets:

   make
   make status/info
   make test
   make tests
   make install
   make clean
   make distclean
   make style

Examples:

make -j 4
   Build laghos_dfem and laghos_tests using the current configuration options
   from MFEM.
   (Laghos requires the MFEM finite element library, and uses its compiler and
    linker options in its build process.)
make status
   Display information about the current configuration.
make test
   Quick run of laghos_dfem on 4 ranks.
make tests
   Run the laghos_tests on 1, 2 and 3 ranks, on the cpu and on the devices,
   with the local and the global q-functions, printing each run with its time
   and a summary.
   (TESTS selects the tests with a Catch2 test spec, e.g. TESTS=[chk], RANKS,
    DEVICES (cpu, dbg, gpu) and QFS (loc, glb) restrict the runs, e.g. RANKS=1
    DEVICES=cpu QFS=loc. Nothing is written to disk: the output of a failed run
    is printed after its KO line. KEEP_LOGS=1 also keeps the logs of the failed
    runs in results/tests.)
make install PREFIX=<dir>
   Install the laghos_dfem and laghos_tests executables in <dir>.
make clean
   Clean the executables, object files and test logs.
make distclean
   In addition to "make clean", remove the local installation directory.
make style
   Format the C++ source files using the Artistic Style (astyle) settings from
   MFEM.

endef

NPROC = $(shell getconf _NPROCESSORS_ONLN)
GOALS = help clean distclean

# Default installation location
PREFIX ?= ./bin
INSTALL = /usr/bin/install

# Use the MFEM source, build, or install directory
MFEM_DIR ?= ../../mfem
CONFIG_MK := $(MFEM_DIR)/config/config.mk
ifeq ($(wildcard $(CONFIG_MK)),)
   CONFIG_MK := $(MFEM_DIR)/share/mfem/config.mk
endif
TEST_MK = $(MFEM_TEST_MK)

# Use the compiler used by MFEM. Get the compiler and the options for compiling
# and linking from MFEM's config.mk. (Skip this if the target does not require
# building.)
MFEM_LIB_FILE = mfem_is_not_built
ifeq (,$(filter $(GOALS),$(MAKECMDGOALS)))
   -include $(CONFIG_MK)
   ifneq ($(realpath $(MFEM_DIR)),$(realpath $(MFEM_SOURCE_DIR)))
      ifneq ($(realpath $(MFEM_DIR)),$(realpath $(MFEM_INSTALL_DIR)))
         MFEM_BUILD_DIR := $(MFEM_DIR)
         override MFEM_DIR := $(MFEM_SOURCE_DIR)
      endif
   endif
endif

CXX = $(MFEM_CXX)
CPPFLAGS = $(MFEM_CPPFLAGS)
CXXFLAGS = $(MFEM_CXXFLAGS)
LAGHOS_FLAGS = $(CPPFLAGS) $(CXXFLAGS) $(MFEM_INCFLAGS)
# Extra include dir, needed for now to include headers like "general/forall.hpp"
EXTRA_INC_DIR = $(or $(wildcard $(MFEM_DIR)/include/mfem),$(MFEM_DIR))
# With an MFEM build directory, the source headers above find the generated
# _config.hpp in the config directory of the build, e.g. for <fem/...> includes
ifneq ($(MFEM_BUILD_DIR),)
   EXTRA_INC_DIR += $(wildcard $(MFEM_BUILD_DIR)/config)
endif
CCC = $(strip $(CXX) $(LAGHOS_FLAGS) $(addprefix -I,$(EXTRA_INC_DIR)))

LAGHOS_LIBS = $(MFEM_LIBS)
LIBS = $(strip $(LAGHOS_LIBS) $(LDFLAGS))

# Catch2 single header of the MFEM source tree, for laghos_tests
CATCH_DIR ?= $(MFEM_SOURCE_DIR)/tests/unit

# Assembly and solver, compiled once for laghos_dfem and laghos_tests
LIB_SOURCE_FILES = laghos_assembly.cpp laghos_solver.cpp laghos_qf_hydro.cpp \
	laghos_qf_velocity.cpp laghos_qf_energy.cpp
LIB_OBJECT_FILES = $(LIB_SOURCE_FILES:.cpp=.o)
SOURCE_FILES = $(sort $(wildcard *.cpp))
HEADER_FILES = $(sort $(wildcard *.hpp))
OBJECT_FILES = $(SOURCE_FILES:.cpp=.o) laghos_main.o

# Targets

.PHONY: all clean distclean install status info test tests tests-run \
	tests-header style clean-build clean-exec

.SUFFIXES: .cpp .o
.cpp.o:
	cd $(<D); $(CCC) -c $(<F)

all: laghos_dfem laghos_tests

laghos_dfem: laghos.o $(LIB_OBJECT_FILES) $(CONFIG_MK) $(MFEM_LIB_FILE)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o $@ laghos.o $(LIB_OBJECT_FILES) $(LIBS)

# laghos.cpp again, with its main as laghos(), called by the tests
laghos_main.o: laghos.cpp
	$(CCC) -Dmain=laghos -c laghos.cpp -o $@

laghos_tests.o: laghos_tests.cpp
	$(CCC) -isystem $(CATCH_DIR) -c laghos_tests.cpp

laghos_tests: laghos_tests.o laghos_main.o $(LIB_OBJECT_FILES) $(CONFIG_MK) \
	$(MFEM_LIB_FILE)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o $@ laghos_tests.o laghos_main.o \
	$(LIB_OBJECT_FILES) $(LIBS)

$(OBJECT_FILES): $(HEADER_FILES) $(CONFIG_MK)

# Quick test with specific execution options
MFEM_TESTS = laghos_dfem
RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP)
test: laghos_dfem
	@$(call mfem-test,$<, $(RUN_MPI) 4, Laghos dFEM miniapp,\
	-p 0 -m ../data/square01_quad.mesh -rs 3 -tf 0.1,SKIP-NO-VIS)
# Testing: "test" target and mfem-test* variables are defined in MFEM's
# config/test.mk
ifeq (,$(filter $(GOALS),$(MAKECMDGOALS)))
include $(TEST_MK)
endif

# Non-regression tests: the laghos_tests of TESTS, as listed by laghos_tests
# itself, on the RANKS, on the DEVICES, with the q-function backends of QFS,
# with the ctest names
TESTS =
RANKS = 1 2 3
QFS = loc glb
DEVICES = cpu $(if $(DEVICE_gpu),gpu) dbg
# MFEM devices and q-function backends of the DEVICES and QFS name tags
DEVICE_cpu = cpu
DEVICE_dbg = debug
DEVICE_gpu = $(if $(filter YES,$(MFEM_USE_CUDA)),cuda,$(if \
	$(filter YES,$(MFEM_USE_HIP)),hip))
QF_loc = local
QF_glb = global
# Debug device fault: hypre host reads of aliases protect valid device pages
SKIP_RUNS = dfem_dbg_loc_imp_run_sdv_3d_q21_mid_av7_dtn_np% \
	dfem_dbg_glb_imp_run_sdv_3d_q21_mid_av7_dtn_np%
# KEEP_LOGS=1 keeps the logs of the failed runs in TESTS_DIR
KEEP_LOGS =
TESTS_KEEP = $(filter-out 0 no NO,$(KEEP_LOGS))
TESTS_DIR = results/tests
TESTS_KEPT = , logs in $(TESTS_DIR)

# Wall clock time in seconds, since the epoch or since the time given
CLOCK = perl -MTime::HiRes=time -e 'printf "%.2f", time - ($$ARGV[0] || 0)'
# Use the -j of the parent when it has a jobserver, -j $(NPROC) otherwise
TESTS_JOBS = $(if $(findstring jobserver,$(MAKEFLAGS)),,-j $(NPROC))

# Each run keeps its output in memory, and prints it when it fails.
# The failed runs are listed in the TESTS_FAILED temporary file,
# removed at the end. With KEEP_LOGS, TESTS_DIR is cleared
# (with results if empty) and holds the logs of the failed runs.
tests: laghos_tests
	@$(if $(TESTS_KEEP),rm -rf $(TESTS_DIR); \
	   rmdir $(patsubst %/,%,$(dir $(TESTS_DIR))) 2> /dev/null;) \
	 fl=`mktemp /tmp/laghos_tests.XXXXXX` || exit 1; \
	 trap 'rm -f $$fl' 0; trap 'exit 130' 1 2 15; \
	 $(MAKE) --no-print-directory $(TESTS_JOBS) tests-run \
	   CONFIG_MK=$(CONFIG_MK) TESTS_T0=`$(CLOCK)` TESTS_FAILED=$$fl

tests-header:
	@echo "laghos_tests: $(words $(RUNS)) runs$(if $(TESTS), of '$(TESTS)')," \
	  "ranks $(RANKS), devices $(DEVICES), q-functions $(QFS)"
	@echo "MFEM_DIR = $(patsubst %/config/config.mk,%,$(CONFIG_MK))"

# 1: test, 2: ranks, 3: device, 4: q-function backend
laghos_run = dfem_$(3)_$(4)_$(1)_np$(2)
# Lines of a failed output: its ✗ checks, or its last lines outside the MPI
# boxes
TESTS_LOG_LINES = awk 'length($$0) == 74 && /^-+$$/ { skip = !skip; next } \
	!skip && NF { l[++n] = $$0; if (index($$0, "│ ✗")) x[++m] = $$0 } \
	END { if (m) for (i = 1; i <= m; i++) print "    " x[i]; \
	else for (i = (n > 5 ? n - 4 : 1); i <= n; i++) print "      | " l[i] }'
# 1: test, 2: ranks, 3: device, 4: q-function backend, printing OK or KO, the
# run and its time, and on failure its log (with KEEP_LOGS) and output lines,
# with a single printf
laghos_run_cmd = t0=`$(CLOCK)`; \
	if out=$$($(RUN_MPI) $(2) ./laghos_tests --device $(DEVICE_$(3)) \
	   --qf $(QF_$(4)) $(1) \
	   2>&1); \
	then s=OK c='\033[0;32m'; else s=KO c='\033[0;31m'; fi; \
	t=`$(CLOCK) $$t0`; [ -t 1 ] || c=; \
	f="   $${c}%s$${c:+\033[0m}  %s  (%s s)\n"; \
	if [ $$s = OK ]; then printf "$$f" $$s $@ $$t; \
	else echo $@ >> $(TESTS_FAILED); $(if $(TESTS_KEEP),mkdir -p $(TESTS_DIR) \
	   && printf '%s\n' "$$out" > $(TESTS_DIR)/$@.log;) \
	   l=$$(printf '%s\n' "$$out" | $(TESTS_LOG_LINES)); \
	   printf "$$f$(if $(TESTS_KEEP),      %s\n)%s\n" $$s $@ $$t \
	   $(if $(TESTS_KEEP),$(TESTS_DIR)/$@.log) "$$l"; fi
# 1: run, 2: test, 3: ranks, 4: device, 5: q-function backend
define laghos_run_template
RUNS += $(1)
$(1): laghos_tests | tests-header
	@$$(call laghos_run_cmd,$(2),$(3),$(4),$(5))
endef
# 1: tests, 2: q-function backend
laghos_runs = $(foreach t,$(1),$(foreach r,$(RANKS),$(foreach d,$(DEVICES),\
	$(if $(filter $(SKIP_RUNS),$(call laghos_run,$(t),$(r),$(d),$(2))),,\
	$(eval $(call laghos_run_template,$(call laghos_run,$(t),$(r),$(d),$(2)),\
	$(t),$(r),$(d),$(2)))))))
ifneq (,$(filter tests-run,$(MAKECMDGOALS)))
   ALL_TESTS := $(shell $(RUN_MPI) 1 ./laghos_tests --list-test-names-only \
	$(if $(TESTS),'$(TESTS)') 2> /dev/null)
   $(foreach q,$(QFS),$(call laghos_runs,$(ALL_TESTS),$(q)))
endif
tests-run: $(RUNS)
	@failed=`sort $(TESTS_FAILED)`; \
	 nf=`echo $$failed | wc -w`; total=$(words $(RUNS)); logs=; \
	 if [ $$nf -gt 0 ]; then echo "Failed runs:"; \
	    $(if $(TESTS_KEEP),logs="$(TESTS_KEPT)";) \
	    for r in $$failed; do \
	       echo "   $$r$(if $(TESTS_KEEP),  ($(TESTS_DIR)/$$r.log))"; done; fi; \
	 echo "laghos_tests: `expr $$total - $$nf`/$$total runs passed in" \
	   "`$(CLOCK) $(TESTS_T0)` s$$logs"; \
	 test $$nf -eq 0 && test $$total -gt 0

# Generate an error message if the MFEM library is not built and exit
$(CONFIG_MK) $(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

cln clean: clean-build clean-exec

clean-build:
	rm -rf laghos_dfem laghos_tests *.o *~ *.dSYM
clean-exec:
	rm -rf ./results tgreen.mesh tgreen.gf
distclean: clean
	rm -rf bin/

install: laghos_dfem laghos_tests
	mkdir -p $(PREFIX)
	$(INSTALL) -m 750 laghos_dfem laghos_tests $(PREFIX)

help:
	$(info $(value LAGHOS_HELP_MSG))
	@true

status info:
	$(info MFEM_DIR     = $(MFEM_DIR))
	$(info LAGHOS_FLAGS = $(LAGHOS_FLAGS))
	$(info LAGHOS_LIBS  = $(value LAGHOS_LIBS))
	$(info DEVICES      = $(DEVICES))
	$(info QFS          = $(QFS))
	$(info PREFIX       = $(PREFIX))
	@true

ASTYLE = astyle --options=$(MFEM_DIR)/config/mfem.astylerc
FORMAT_FILES := $(SOURCE_FILES) $(HEADER_FILES)
style:
	@if ! $(ASTYLE) $(FORMAT_FILES) | grep Formatted; then\
	   echo "No source files were changed.";\
	fi
