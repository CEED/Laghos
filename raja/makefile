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

enableCuda= 0
enableHip = 1

RAJA_DIR ?= /usr/local/raja
RAJA_INCFLAGS = -I$(RAJA_DIR)/include -I$(RAJA_DIR)/tpl/cub -I$(RAJA_DIR)/tpl/rocPRIM/rocprim/include  
RAJA_LIBS = -fopenmp $(RAJA_DIR)/lib/libRAJA.a
RAJA_CXXFLAGS = 

NV_ARCH ?= -arch=sm_61
CUDA_DIR ?= /usr/local/cuda

HIP_ARCH ?= --amdgpu-target=gfx906
ROCM_DIR ?= /opt/rocm

MPI_HOME ?= /usr/local/mpi
MPI_INCFLAGS = -I$(MPI_HOME)/include
MPI_LIBS := $(shell mpicxx --showme:link)


NVCC_CXXFLAGS = --expt-extended-lambda -Xcompiler -fopenmp -x=cu -std=c++11 -m64 --restrict -Xcompiler -Wall $(NV_ARCH)
NVCC_LIBS = -Wl,-rpath -Wl,$(CUDA_DIR)/lib64 -L$(CUDA_DIR)/lib64 \
            -lcuda -lcudart -lcudadevrt -lnvToolsExt

HIPCC_CXXFLAGS = -fopenmp -std=c++11 -Wall -I$(ROCM_DIR) $(shell $(ROCM_DIR)/bin/hipconfig --cpp_config)
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
# MFEM_DIR = ../../mfem
MFEM_DIR = ./mfem
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

ifeq (1,$(enableCuda))
CXX = nvcc
else ifeq (1,$(enableHip))
CXX = ${ROCM_DIR}/bin/hipcc
endif

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
 $(MPI_INCFLAGS) $(RAJA_INCFLAGS) $(RAJA_CXXFLAGS)
LAGHOS_LIBS = $(MFEM_LIBS) $(RAJA_LIBS)

ifeq (1,$(enableCuda))
LD = $(MFEM_CXX)
LAGHOS_FLAGS += $(NVCC_CXXFLAGS) $(NVCC_INCFLAGS)
LAGHOS_LIBS += $(NVCC_LIBS)
else ifeq (1,$(enableHip))
LD = ${ROCM_DIR}/bin/hipcc
LAGHOS_FLAGS += $(HIPCC_CXXFLAGS) $(HIPCC_INCFLAGS)
LAGHOS_LIBS += $(HIPCC_LIBS)
endif


ifeq ($(LAGHOS_DEBUG),YES)
   LAGHOS_FLAGS += -DLAGHOS_DEBUG
endif

LIBS = $(strip $(LAGHOS_LIBS) $(LDFLAGS))
CCC  = $(strip $(CXX) $(LAGHOS_FLAGS))
Ccc  = $(strip $(CC) $(CFLAGS) $(GL_OPTS))

SOURCE_FILES = ./laghos.cpp \
./laghos_assembly.cpp \
./raja/linalg/rsolvers.cpp \
./raja/linalg/rvector.cpp \
./raja/kernels/force/rForce.cpp \
./raja/kernels/geom/rInitGeom.cpp \
./raja/kernels/quad/rQDataInit.cpp \
./raja/kernels/quad/rQDataUpdate.cpp \
./raja/kernels/quad/rGridFuncToQuad.cpp \
./raja/kernels/maps/rLocalToGlobal.cpp \
./raja/kernels/maps/rMapping.cpp \
./raja/kernels/maps/rGlobalToLocal.cpp \
./raja/kernels/mass/rMassAssemble.cpp \
./raja/kernels/mass/rMassMultAdd.cpp \
./raja/kernels/blas/vector_map_dofs.cpp \
./raja/kernels/blas/vector_vec_sub.cpp \
./raja/kernels/blas/vector_dot.cpp \
./raja/kernels/blas/vector_clear_dofs.cpp \
./raja/kernels/blas/vector_xsy.cpp \
./raja/kernels/blas/vector_xpay.cpp \
./raja/kernels/blas/vector_axpy.cpp \
./raja/kernels/blas/vector_op_eq.cpp \
./raja/kernels/blas/vector_get_subvector.cpp \
./raja/kernels/blas/vector_vec_add.cpp \
./raja/kernels/blas/vector_min.cpp \
./raja/kernels/blas/vector_set_subvector.cpp \
./raja/kernels/blas/vector_set_subvector_const.cpp \
./raja/kernels/blas/vector_vec_mul.cpp \
./raja/kernels/blas/vector_neg.cpp \
./raja/fem/rconform.cpp \
./raja/fem/rrestrict.cpp \
./raja/fem/rbilinearform.cpp \
./raja/fem/rbilininteg.cpp \
./raja/fem/rprolong.cpp \
./raja/fem/rgridfunc.cpp \
./raja/fem/rfespace.cpp \
./raja/general/rmemcpy.cpp \
./raja/general/rtable.cpp \
./raja/general/rcommd.cpp \
./raja/config/rconfig.cpp \
./laghos_solver.cpp

OBJECT_FILES1 = $(SOURCE_FILES:.cpp=.o)
OBJECT_FILES = $(OBJECT_FILES1:.c=.o)
HEADER_FILES = ./raja/raja.hpp \
./raja/linalg/rode.hpp \
./raja/linalg/rsolvers.hpp \
./raja/linalg/rvector.hpp \
./raja/linalg/roperator.hpp \
./raja/kernels/raja.hpp \
./raja/kernels/include/forall.hpp \
./raja/kernels/include/kernels.hpp \
./raja/kernels/include/offsets.hpp \
./raja/fem/rbilinearform.hpp \
./raja/fem/rrestrict.hpp \
./raja/fem/rfespace.hpp \
./raja/fem/rbilininteg.hpp \
./raja/fem/rgridfunc.hpp \
./raja/fem/rprolong.hpp \
./raja/fem/rconform.hpp \
./raja/general/rarray.hpp \
./raja/general/rtable.hpp \
./raja/general/rmalloc.hpp \
./raja/general/rmemcpy.hpp \
./raja/general/rcommd.hpp \
./raja/config/rconfig.hpp \
./laghos_solver.hpp \
./laghos_assembly.hpp

# Targets

.PHONY: all clean distclean install status info opt debug test style clean-build clean-exec

.SUFFIXES: .c .cpp .o
.cpp.o:
	cd $(<D); $(CCC) -c $(<F)
.c.o:
	cd $(<D); $(Ccc) -c $(<F)

laghos: override MFEM_DIR = $(MFEM_DIR1)
laghos:	$(OBJECT_FILES) $(CONFIG_MK) $(MFEM_LIB_FILE)
	 $(LD) -o laghos $(OBJECT_FILES) $(LIBS)

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
