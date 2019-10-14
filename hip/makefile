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

AMD_ARCH ?= --amdgpu-target=gfx906
ROCM_DIR ?= /opt/rocm

MPI_HOME ?= /usr/lib/openmpi
MPI_INCFLAGS = -I$(MPI_HOME)/include

MPI_LIBS := $(shell mpicxx --showme:link)

HIPCC_CXXFLAGS = -std=c++11 -Wall $(AMD_ARCH) -I$(ROCM_DIR)/include $(shell $(ROCM_DIR)/bin/hipconfig --cpp_config)
HIPCC_LIBS =  -L$(ROCM_DIR)/lib/ -lhip_hcc $(MPI_INCFLAGS) $(MPI_LIBS)


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
MFEM_DIR = ../../mfem
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

CXX = $(ROCM_DIR)/bin/hipcc
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

LAGHOS_FLAGS = $(CPPFLAGS) $(CXXFLAGS) $(MFEM_INCFLAGS) \
      $(MPI_INCFLAGS) $(HIPCC_CXXFLAGS) $(HIPCC_INCFLAGS)
LAGHOS_LIBS = $(MFEM_LIBS) $(HIPCC_LIBS)

ifeq ($(LAGHOS_DEBUG),YES)
   LAGHOS_FLAGS += -DLAGHOS_DEBUG
endif

LIBS = $(strip $(LAGHOS_LIBS) $(LDFLAGS))
CCC  = $(strip $(CXX) $(LAGHOS_FLAGS))
Ccc  = $(strip $(CC) $(CFLAGS) $(GL_OPTS))

SOURCE_FILES = laghos.cpp laghos_assembly.cpp laghos_solver.cpp \
	hip/linalg/solvers.cpp \
	hip/linalg/vector.cpp \
	hip/kernels/share/qDataUpdateS.cpp \
	hip/kernels/share/gridFuncToQuadS.cpp \
	hip/kernels/share/forceS.cpp \
	hip/kernels/share/massMultAddS.cpp \
	hip/kernels/share/massAssembleS.cpp \
	hip/kernels/force/force.cpp \
	hip/kernels/geom/initGeom.cpp \
	hip/kernels/quad/gridFuncToQuad.cpp \
	hip/kernels/quad/qDataUpdate.cpp \
	hip/kernels/quad/qDataInit.cpp \
	hip/kernels/maps/globalToLocal.cpp \
	hip/kernels/maps/mapping.cpp \
	hip/kernels/maps/localToGlobal.cpp \
	hip/kernels/mass/multAdd.cpp \
	hip/kernels/mass/assemble.cpp \
	hip/kernels/blas/vector_map_dofs.cpp \
	hip/kernels/blas/vector_vec_sub.cpp \
	hip/kernels/blas/vector_dot.cpp \
	hip/kernels/blas/vector_clear_dofs.cpp \
	hip/kernels/blas/vector_xsy.cpp \
	hip/kernels/blas/vector_xpay.cpp \
	hip/kernels/blas/vector_axpy.cpp \
	hip/kernels/blas/vector_op_eq.cpp \
	hip/kernels/blas/vector_get_subvector.cpp \
	hip/kernels/blas/vector_vec_add.cpp \
	hip/kernels/blas/vector_min.cpp \
	hip/kernels/blas/vector_set_subvector.cpp \
	hip/kernels/blas/vector_set_subvector_const.cpp \
	hip/kernels/blas/vector_vec_mul.cpp \
	hip/kernels/blas/vector_neg.cpp \
	hip/fem/bilinearform.cpp \
	hip/fem/hipGridfunc.cpp \
	hip/fem/restrict.cpp \
	hip/fem/fespace.cpp \
	hip/fem/bilininteg.cpp \
	hip/fem/conform.cpp \
	hip/fem/prolong.cpp \
	hip/general/memcpy.cpp \
	hip/general/commd.cpp \
	hip/general/table.cpp \
	hip/config/config.cpp

OBJECT_FILES1 = $(SOURCE_FILES:.cpp=.o)
OBJECT_FILES = $(OBJECT_FILES1:.c=.o)
HEADER_FILES = laghos_solver.hpp laghos_assembly.hpp \
	hip/linalg/solvers.hpp \
	hip/linalg/operator.hpp \
	hip/linalg/ode.hpp \
	hip/linalg/vector.hpp \
	hip/kernels/hip.hpp \
	hip/kernels/include/forall.hpp \
	hip/kernels/include/kernels.hpp \
	hip/kernels/include/offsets.hpp \
	hip/fem/prolong.hpp \
	hip/fem/hipGridfunc.hpp \
	hip/fem/conform.hpp \
	hip/fem/restrict.hpp \
	hip/fem/bilininteg.hpp \
	hip/fem/fespace.hpp \
	hip/fem/bilinearform.hpp \
	hip/general/commd.hpp \
	hip/general/table.hpp \
	hip/general/malloc.hpp \
	hip/general/memcpy.hpp \
	hip/general/array.hpp \
	hip/config/config.hpp \
	hip/hip.hpp \

# Targets

.PHONY: all clean distclean install status info opt debug test style clean-build clean-exec

.SUFFIXES: .c .cpp .o
.cpp.o:
	cd $(<D); $(CCC) -c $(<F)
.c.o:
	cd $(<D); $(Ccc) -c $(<F)

laghos: override MFEM_DIR = $(MFEM_DIR1)
laghos:	$(OBJECT_FILES) $(CONFIG_MK) $(MFEM_LIB_FILE)
	$(CXX) -o laghos $(OBJECT_FILES) $(LIBS)

all: laghos

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

clean: clean-build clean-exec

clean-build:
	rm -rf laghos $(OBJECT_FILES) *~ *.dSYM
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
