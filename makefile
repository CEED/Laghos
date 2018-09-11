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

# SETUP ************************************************************************
CUB_DIR  ?= ./cub
CUDA_DIR ?= /usr/local/cuda
MFEM_DIR ?= $(HOME)/home/mfem/kernels-gpu-cpu
RAJA_DIR ?= $(HOME)/usr/local/raja/last
MPI_HOME ?= $(HOME)/usr/local/openmpi/3.0.0
#NV_ARCH ?= -arch=sm_60 #-gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60
#CXXEXTRA = -std=c++11 -m64 #-DNDEBUG=1 #-D__NVVP__ #-D__NVVP__ # -DLAGHOS_DEBUG -D__NVVP__

# number of proc to use for compilation stage
CPU = $(shell echo $(shell getconf _NPROCESSORS_ONLN)*2|bc -l)

# fetch current/working directory
pwd = $(patsubst %/,%,$(dir $(abspath $(firstword $(MAKEFILE_LIST)))))

kernels = $(pwd)/kernels

# OKRTC ************************************************************************
#OKRTC_DIR ?= ~/usr/local/okrtc
ifneq ($(wildcard $(OKRTC_DIR)/bin/okrtc),)
	OKRTC ?= dbg=1 $(OKRTC_DIR)/bin/okrtc
endif

# ******************************************************************************
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
DEBUG_OPTS = -g #-Wall
LAGHOS_DEBUG = $(MFEM_DEBUG)
ifneq ($(LAGHOS_DEBUG),$(MFEM_DEBUG))
   ifeq ($(LAGHOS_DEBUG),YES)
      CXXFLAGS = $(DEBUG_OPTS)
   else
      CXXFLAGS = $(OPTIM_OPTS)
   endif
endif

# CXXFLAGS ADDONS **************************************************************
CXXFLAGS += $(CXXEXTRA)

# NVCC *************************************************************************
ifneq (,$(nvcc))
#	CXX = nvcc
#	CUFLAGS = -std=c++11 -m64 --restrict $(NV_ARCH) #-rdc=true
#	CXXFLAGS += --restrict $(NV_ARCH) -x=cu
#	CXXFLAGS += $(if $(templates),-D__TEMPLATES__)
#	CUDA_INC = -I$(CUDA_DIR)/samples/common/inc
#	CXXFLAGS += --expt-extended-lambda
	CUDA_LIBS = -lcuda -lcudart -lcudadevrt -lnvToolsExt
endif

# all, targets & laghos ********************************************************
all:;@$(MAKE) -j $(CPU) laghos
nv nvcc cuda:;$(MAKE) nvcc=1 templates=1 all

# MPI **************************************************************************
MPI_INC = -I$(MPI_HOME)/include 
MPI_LIB = -L$(MPI_HOME)/lib -lmpi

# LAGHOS FLAGS *****************************************************************
LAGHOS_FLAGS = $(CPPFLAGS) $(CXXFLAGS) $(MFEM_INCFLAGS) \
					$(CUB_INC) $(MPI_INC) $(RAJA_INC)
LAGHOS_LIBS = $(MFEM_LIBS) $(MPI_LIB) $(RAJA_LIBS) $(CUDA_LIBS) -ldl 

ifeq ($(LAGHOS_DEBUG),YES)
   LAGHOS_FLAGS += -DLAGHOS_DEBUG
endif

LIBS = $(strip $(LAGHOS_LIBS) $(LDFLAGS))
CCC  = $(strip $(CXX) $(LAGHOS_FLAGS))
Ccc  = $(strip $(CC) $(CFLAGS) $(GL_OPTS))

MAKEFILE_DIR = $(dir $(abspath $(firstword $(MAKEFILE_LIST))))
KERNELS_DIR = $(MAKEFILE_DIR)/kernels

# SOURCE FILES SETUP ***********************************************************
SOURCE_FILES = laghos.cpp laghos_solver.cpp laghos_assembly.cpp \
	qupdate/d2q.cpp \
	qupdate/dof2quad.cpp \
	qupdate/geom.cpp \
	qupdate/maps.cpp \
	qupdate/qupdate.cpp \
	qupdate/densemat.cpp \
	qupdate/eigen.cpp \
	qupdate/global2local.cpp \
	qupdate/memcpy.cpp\
	$(KERNELS_DIR)/kForcePAOperator.cpp \
	$(KERNELS_DIR)/kMassPAOperator.cpp
# Kernel files setup
KERNELS_RTC_DIRS = $(KERNELS_DIR)/force
KERNELS_RTC_SRC_FILES = $(foreach dir,$(KERNELS_RTC_DIRS),$(wildcard $(dir)/*.cpp))

# OBJECT FILES *****************************************************************
OBJECT_FILES1 = $(SOURCE_FILES:.cpp=.o)
OBJECT_FILES = $(OBJECT_FILES1:.c=.o)
OBJECT_KERNELS = $(KERNELS_RTC_SRC_FILES:.cpp=.o)

# HEADER FILES *****************************************************************
HEADER_FILES = laghos_solver.hpp laghos_assembly.hpp

# Targets **********************************************************************
.PHONY: all clean distclean install status info opt debug test style clean-build clean-exec

.SUFFIXES: .c .cpp .o
.cpp.o:
	cd $(<D); $(CCC) -c $(abspath $<)
.c.o:
	cd $(<D); $(Ccc) -c $(<F)

# ******************************************************************************
laghos: override MFEM_DIR = $(MFEM_DIR1)
laghos:	$(OBJECT_FILES) $(OBJECT_KERNELS) $(CONFIG_MK) $(MFEM_LIB_FILE)
	$(MFEM_CXX) -o laghos $(OBJECT_FILES) $(OBJECT_KERNELS) $(LIBS)

# go ***************************************************************************
go:;@./laghos -cfl 0.1 -rs 0
pgo:;@mpirun -n 2 ./laghos -cfl 0.1 -rs 0
pgo2:;@DBG=1 mpirun -xterm -1! -n 2 ./laghos -cfl 0.1 -rs 0 -ng -ms 1 -cgt 0 -cgm 1
#pgo:;@mpirun -n 2 -xterm -1! --tag-output --merge-stderr-to-stdout ./laghos -cfl 0.1 -rs 0

ng:;@./laghos -cfl 0.1 -rs 0 -ng
png:;@mpirun -n 1 ./laghos -cfl 0.1 -ng
png2:;mpirun -n 2 ./laghos -cfl 0.1 -ng
png3:;mpirun -n 3 ./laghos -cfl 0.1 -ng

dng:;@cuda-gdb --args ./laghos -cfl 0.1 -rs 0 -ng #-cgt 0 -cgm 2
mng:;cuda-memcheck ./laghos -cfl 0.1 -rs 0 -ng -ms 1
ddng:;@DBG=1 cuda-gdb --args ./laghos -cfl 0.1 -rs 0 -ng -cgt 0 -cgm 2

ng1:;DBG=1 ./laghos -cfl 0.1 -rs 0 -ng -ms 1 -cgt 0 -cgm 1
mng1:;DBG=1 cuda-memcheck ./laghos -cfl 0.1 -rs 0 -ng -ms 1 -cgt 0 -cgm 1
mng2:;DBG=1 cuda-memcheck ./laghos -cfl 0.1 -rs 0 -ng -ms 1 -cgt 0 -cgm 2
dng1:;DBG=1 cuda-gdb --args ./laghos -cfl 0.1 -rs 0 -ng -ms 1 -cgt 0 -cgm 1

vng:;@valgrind ./laghos -cfl 0.1 -rs 0 -ms 1 -ng
mpng2:;@DBG=1 mpirun -xterm -1! -n 2 cuda-memcheck ./laghos -cfl 0.1 -rs 0 -ng -ms 1 -cgt 0 -cgm 1
vpng2:;@DBG=1 mpirun -xterm -1! -n 2 valgrind --leak-check=full --track-origins=yes ./laghos -cfl 0.1 -rs 0 -ng -ms 1 -cgt 0 -cgm 1

png2d:;@DBG=1 mpirun -xterm -1! --merge-stderr-to-stdout -n 2 ./laghos -cfl 0.1 -ng
vpng2d:;@DBG=1 mpirun -xterm -1! --merge-stderr-to-stdout -n 2 valgrind --leak-check=full --track-origins=yes ./laghos -cfl 0.1 -rs 2 -ng -ms 1
#png:;@mpirun -n 2 --tag-output --merge-stderr-to-stdout ./laghos -cfl 0.1 -rs 0 -ng
#png:;@mpirun -n 2 -xterm 1 ./laghos -cfl 0.1 -rs 0 -ng
pngd:;DBG=1 mpirun -xterm -1! --merge-stderr-to-stdout -n 2 ./laghos -cfl 0.1 -rs 1 -ms 2 -ng
#vpng:;@mpirun -n 2 valgrind ./laghos -cfl 0.1 -rs 2 -ms 1 -ng
vpngd:;DBG=1 mpirun -xterm -1! --merge-stderr-to-stdout -n 3 valgrind --leak-check=full --track-origins=yes ./laghos -cfl 0.1 -rs 1 -ms 1 -ng

opt:
	$(MAKE) "LAGHOS_DEBUG=NO"

debug:
	$(MAKE) "LAGHOS_DEBUG=YES"

$(OBJECT_FILES): override MFEM_DIR = $(MFEM_DIR2)
$(OBJECT_FILES): $(HEADER_FILES) $(CONFIG_MK)
$(OBJECT_KERNELS): override MFEM_DIR = $(MFEM_DIR2)

#rtc:;@echo OBJECT_KERNELS=$(OBJECT_KERNELS)
$(OBJECT_KERNELS): %.o: %.cpp
	$(OKRTC) $(CCC) -o $(@) -c $(CUB_INC) $(MPI_INC) $(RAJA_INC) -I$(realpath $(dir $(<))) $(<)

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

cln clean: clean-build clean-exec

clean-build:
	rm -rf laghos *.o *.so *~ *.dSYM kernels/*.o $(OBJECT_KERNELS)
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
