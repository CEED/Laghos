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

# number of proc to use for compilation stage **********************************
NPROC = $(shell echo $(shell getconf _NPROCESSORS_ONLN)*2|bc -l)

define LAGHOS_HELP_MSG

Laghos makefile targets:

   make
   make status/info
   make install
   make clean
   make distclean
   make style

Examples:

make -j 4
   Build Laghos using the current configuration options from MFEM.
   (Laghos requires the MFEM finite element library, and uses its compiler and
    linker options in its build process.)
make status
   Display information about the current configuration.
make install PREFIX=<dir>
   Install the Laghos executable in <dir>.
make clean
   Clean the Laghos executable, library and object files.
make distclean
   In addition to "make clean", remove the local installation directory and some
   run-time generated files.
make style
   Format the Laghos C++ source files using the Artistic Style (astyle) settings
   from MFEM.

endef

# Default installation location
PREFIX = ./bin
INSTALL = /usr/bin/install

# Use the MFEM build directory
MFEM_DIR = ../mfem
CONFIG_MK = $(MFEM_DIR)/config/config.mk
TEST_MK = $(MFEM_DIR)/config/test.mk
# Use the MFEM install directory
# MFEM_DIR = ../mfem/mfem
# CONFIG_MK = $(MFEM_DIR)/config.mk
# TEST_MK = $(MFEM_DIR)/test.mk

# Use two relative paths to MFEM: first one for compilation in '.' and second
# one for compilation in 'lib'.
MFEM_DIR1 := $(MFEM_DIR)
MFEM_DIR2 := $(realpath $(MFEM_DIR))

# Use the compiler used by MFEM. Get the compiler and the options for compiling
# and linking from MFEM's config.mk. (Skip this if the target does not require
# building.)
MFEM_LIB_FILE = mfem_is_not_built
ifeq (,$(filter help clean distclean style,$(MAKECMDGOALS)))
   -include $(CONFIG_MK)
endif

CXX = $(MFEM_CXX)
CPPFLAGS = $(MFEM_CPPFLAGS)
CXXFLAGS = $(MFEM_CXXFLAGS)

# MFEM config does not define C compiler
CC     = gcc
CFLAGS = -O3

# Optional link flags
LDFLAGS =

OPTIM_OPTS = -O3
DEBUG_OPTS = -g -Wall
LAGHOS_DEBUG = $(MFEM_DEBUG)
ifneq ($(LAGHOS_DEBUG),$(MFEM_DEBUG))
   ifeq ($(LAGHOS_DEBUG),YES)
      CXXFLAGS = $(DEBUG_OPTS)
   else
      CXXFLAGS = $(OPTIM_OPTS)
   endif
endif

LAGHOS_FLAGS = $(CPPFLAGS) $(CXXFLAGS) $(MFEM_INCFLAGS)
LAGHOS_LIBS = $(MFEM_LIBS) $(MFEM_EXT_LIBS)

ifeq ($(LAGHOS_DEBUG),YES)
   LAGHOS_FLAGS += -DLAGHOS_DEBUG
endif

LIBS = $(strip $(LAGHOS_LIBS) $(LDFLAGS))
CCC  = $(strip $(CXX) $(LAGHOS_FLAGS))
Ccc  = $(strip $(CC) $(CFLAGS) $(GL_OPTS))

SOURCE_FILES = laghos.cpp laghos_solver.cpp laghos_assembly.cpp\
 laghos_timeinteg.cpp
OBJECT_FILES1 = $(SOURCE_FILES:.cpp=.o)
OBJECT_FILES = $(OBJECT_FILES1:.c=.o)
HEADER_FILES = laghos_solver.hpp laghos_assembly.hpp laghos_timeinteg.hpp

# Targets

.PHONY: all clean distclean install status info opt debug test style\
	clean-build clean-exec

.SUFFIXES: .c .cpp .o
.cpp.o:
	cd $(<D); $(CCC) -c $(<F)
.c.o:
	cd $(<D); $(Ccc) -c $(<F)

laghos: override MFEM_DIR = $(MFEM_DIR1)
laghos:	$(OBJECT_FILES) $(CONFIG_MK) $(MFEM_LIB_FILE)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o laghos $(OBJECT_FILES) $(LIBS)

all:;@$(MAKE) -j $(NPROC) laghos

opt:
	$(MAKE) "LAGHOS_DEBUG=NO"

debug:
	$(MAKE) "LAGHOS_DEBUG=YES"

$(OBJECT_FILES): override MFEM_DIR = $(MFEM_DIR2)
$(OBJECT_FILES): $(HEADER_FILES) $(CONFIG_MK)

MFEM_TESTS = laghos
include $(TEST_MK)
# Testing: Specific execution options
RUN_MPI_4 = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) 4
test: laghos
	@$(call mfem-test,$<, $(RUN_MPI_4), Laghos miniapp,\
	-p 0 -m data/square01_quad.mesh -rs 3 -tf 0.1)
# Testing: "test" target and mfem-test* variables are defined in MFEM's
# config/test.mk

# Generate an error message if the MFEM library is not built and exit
$(CONFIG_MK) $(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

cln clean: clean-build clean-exec

clean-build:
	rm -rf laghos *.o *~ *.dSYM
clean-exec:
	rm -rf ./results

distclean: clean
	rm -rf bin/

install: laghos
	mkdir -p $(PREFIX)
	$(INSTALL) -m 750 laghos $(PREFIX)

help:
	$(info $(value LAGHOS_HELP_MSG))
	@true

status info:
	$(info MFEM_DIR    = $(MFEM_DIR))
	$(info LAGHOS_FLAGS = $(LAGHOS_FLAGS))
	$(info LAGHOS_LIBS  = $(value LAGHOS_LIBS))
	$(info PREFIX      = $(PREFIX))
	@true

ASTYLE = astyle --options=$(MFEM_DIR1)/config/mfem.astylerc
FORMAT_FILES := $(SOURCE_FILES) $(HEADER_FILES)

style:
	@if ! $(ASTYLE) $(FORMAT_FILES) | grep Formatted; then\
	   echo "No source files were changed.";\
	fi

# ******************************************************************************
problems=0 1 2 3 4 5 6
meshs=square01_quad cube01_hex
cuda=$(if $(MFEM_CXX:nvcc=),,-o-q-d_cuda)
options=-fa -pa -o -o-q $(cuda)
optioni = $(shell for i in {1..$(words $(options))}; do echo $$i; done)
ranks=1 3
ECHO=/bin/echo
SED=/usr/bin/sed -e
OPTS=-cgt 1.e-14 -rs 0 --checks

# problem:1 mesh:2 option:3 mpi:4
define mfem_test_template
.PHONY: laghos_$(1)_$(2)_$(3)_$(4)
laghos_$(1)_$(2)_$(3)_$(4): laghos
	$(eval name=laghos$(4)-p$(1)-$(2)$(word $(3),$(options)))
	$(eval command=$(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(4) ./laghos $(OPTS) -p $(1) -m data/$(2).mesh $$(shell echo $(word $(3),$(options))|$(SED) "s/-/ -/g"|$(SED) "s/_/ /g"))
#	@echo name: $(name)
#	@echo command: $(command)
	@$(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(4) ./$$< $(OPTS) -p $(1) -m data/$(2).mesh $(shell echo $(word $(3),$(options))|$(SED) "s/-/ -/g"|$(SED) "s/_/ /g") > /dev/null 2>&1 && \
		$(call COLOR_PRINT,'\033[0;32m',OK,': $(name)\n') || $(call COLOR_PRINT,'\033[1;31m',KO,': $(command)\n');
endef

# Generate all targets
$(foreach p, $(problems), $(foreach m, $(meshs), $(foreach o, $(optioni), $(foreach r, $(ranks),\
	$(eval $(call mfem_test_template,$(p),$(m),$(o),$(r)))))))

#$(foreach p, $(problems), $(foreach m, $(meshs), $(foreach o, $(optioni), $(foreach r, $(ranks), $(info $(call _test_template,$(p),$(m),$(o),$(r)))))))

checks check: laghos|$(foreach p,$(problems), $(foreach m,$(meshs), $(foreach o,$(optioni), $(foreach r,$(ranks), laghos_$(p)_$(m)_$(o)_$(r)))))
c chk: ;@$(MAKE) -j $(NPROC) check

go one: ;@$(MAKE) -j 8 check ranks=1
