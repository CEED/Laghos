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

Laghos makefile targets:

   make
   make status/info
   make install
   make clean
   make clean-exec
   make clean-regtest
   make distclean
   make style
   make regtest

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
make clean-exec
   Clean previous Laghos simulation data files.
make clean-regtest
   Clean the regression test output.
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
LIBS_DIR ?= ..
MFEM_DIR = $(LIBS_DIR)/mfem
CONFIG_MK = $(MFEM_DIR)/config/config.mk
TEST_MK = $(MFEM_DIR)/config/test.mk

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

LAGHOS_FLAGS = $(CPPFLAGS) $(CXXFLAGS) $(MFEM_INCFLAGS) -I$(LIBS_DIR)/libROM
LAGHOS_LIBS = $(MFEM_LIBS)

ifeq ($(LAGHOS_DEBUG),YES)
   LAGHOS_FLAGS += -DLAGHOS_DEBUG
endif

LIBS = $(strip $(LAGHOS_LIBS) $(LDFLAGS))
CCC  = $(strip $(CXX) $(LAGHOS_FLAGS))
Ccc  = $(strip $(CC) $(CFLAGS) $(GL_OPTS))

SOURCE_FILES = laghos.cpp laghos_solver.cpp laghos_assembly.cpp laghos_timeinteg.cpp laghos_rom.cpp laghos_utils.cpp
OBJECT_FILES1 = $(SOURCE_FILES:.cpp=.o)
OBJECT_FILES = $(OBJECT_FILES1:.c=.o)
HEADER_FILES = laghos_solver.hpp laghos_assembly.hpp laghos_timeinteg.hpp laghos_rom.hpp SampleMesh.hpp laghos_utils.hpp
TEST_FILES = tests/basisComparator.cpp tests/fileComparator.cpp

include $(CURDIR)/user.mk

# Targets

.PHONY: all clean distclean install status info opt debug test style clean-build clean-exec clean-regtest regtest

.SUFFIXES: .c .cpp .o
.cpp.o:
	cd $(<D); $(CCC) -c $(<F)
.c.o:
	cd $(<D); $(Ccc) -c $(<F)

laghos: $(OBJECT_FILES) $(CONFIG_MK) $(MFEM_LIB_FILE)
				$(CCC) -o laghos $(OBJECT_FILES) $(LIBS) -Wl,-rpath,$(LIBS_DIR)/libROM/build -L$(LIBS_DIR)/libROM/build -lROM $(SCALAPACK_FLAGS)

merge: merge.o $(CONFIG_MK) $(MFEM_LIB_FILE)
				$(CCC) -o merge merge.o laghos_utils.o $(LIBS) -Wl,-rpath,$(LIBS_DIR)/libROM/build -L$(LIBS_DIR)/libROM/build -lROM $(SCALAPACK_FLAGS)

all: laghos merge

opt:
	$(MAKE) "LAGHOS_DEBUG=NO"

debug:
	$(MAKE) "LAGHOS_DEBUG=YES"

$(OBJECT_FILES): override MFEM_DIR = $(MFEM_DIR2)
$(OBJECT_FILES): $(HEADER_FILES) $(CONFIG_MK)

MFEM_TESTS = laghos
include $(TEST_MK)
# Testing: Specific execution options
RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) 4
test: laghos
	@$(call mfem-test,$<, $(RUN_MPI), Laghos miniapp,\
	-p 0 -m data/square01_quad.mesh -rs 3 -tf 0.1)
# Testing: "test" target and mfem-test* variables are defined in MFEM's
# config/test.mk

# Generate an error message if the MFEM library is not built and exit
$(CONFIG_MK) $(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

regtest: tests/fileComparator.cpp tests/basisComparator.cpp tests/solutionComparator.cpp
	$(CXX) $(CXXFLAGS) -o tests/fileComparator tests/fileComparator.cpp
	$(CXX) $(CXXFLAGS) -I$(LIBS_DIR)/libROM -o tests/basisComparator tests/basisComparator.cpp -Wl,-rpath,$(LIBS_DIR)/libROM/build -L$(LIBS_DIR)/libROM/build -lROM
	$(CXX) $(CXXFLAGS) $(MFEM_INCFLAGS) -o tests/solutionComparator tests/solutionComparator.cpp $(LIBS)

clean: clean-regtest clean-build

clean-build:
	rm -rf laghos *.o *~ *.dSYM run merge

clean-exec:
	rm -rf run/*

clean-regtest: clean-exec
	rm -rf tests/Laghos tests/fileComparator tests/basisComparator tests/solutionComparator tests/results

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

#ASTYLE = astyle --options=$(MFEM_DIR1)/config/mfem.astylerc
FORMAT_FILES := $(SOURCE_FILES) $(HEADER_FILES) $(TEST_FILES) merge.cpp

style:
	@if ! $(ASTYLE) $(FORMAT_FILES) | grep Formatted; then\
	   echo "No source files were changed.";\
	fi
$(shell mkdir -p run)
