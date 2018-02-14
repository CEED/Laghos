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

# fetch current/working directory
pwd = $(patsubst %/,%,$(dir $(abspath $(firstword $(MAKEFILE_LIST)))))
home = $(HOME)
raja = $(pwd)/raja
kernels = $(raja)/kernels

CPU = $(shell echo $(shell getconf _NPROCESSORS_ONLN)*2|bc -l)

# Default installation location
PREFIX = ./bin
INSTALL = /usr/bin/install

# Use the MFEM build directory
MFEM_DIR = /home/camier1/home/mfem/mfem-raja
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
#MFEM_DIR1 := $(MFEM_DIR)/build/mfem/lib
#MFEM_DIR2 := $(MFEM_DIR)/build/mfem/include/mfem

# Use the compiler used by MFEM. Get the compiler and the options for compiling
# and linking from MFEM's config.mk. (Skip this if the target does not require
# building.)
MFEM_LIB_FILE = mfem_is_not_built
ifeq (,$(filter help clean distclean style,$(MAKECMDGOALS)))
   -include $(CONFIG_MK)
endif

#################
# MFEM compiler #
#################
CXX = $(MFEM_CXX)
CPPFLAGS = $(MFEM_CPPFLAGS)
CXXFLAGS = $(MFEM_CXXFLAGS)

# MFEM config does not define C compiler
CC     = gcc
CFLAGS = -O3 -Wall

#############
# CUDA ARCH #
# No arch for Quadro M2000, sm_5.2 & GeForce GTX 1070, sm_6.1
# sm_61, sm_5.2
#############
NV_ARCH = -arch=sm_60
#NV_ARCH = -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60

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

###################
# CXXFLAGS ADDONS #
# -fno-omit-frame-pointer
###################
CXXFLAGS += -std=c++11 -m64

#############
# SANITIZER #
# -fsanitize=undefined 
# -fsanitize=address
#############
#CXXFLAGS += -g -O1 -fno-inline -fsanitize=address -fsanitize=undefined
#ASAN_LIB = -lasan -lubsan
#-fsanitize=undefined -fsanitize=address -fno-omit-frame-pointer

###############################
# RAJA compilation: make rj=1 #
###############################
ifneq (,$(rj))
	CXX = nvcc
	CXXFLAGS += -D__RAJA__ -DUSE_CUDA
	CXXFLAGS += -D__LAMBDA__ --expt-extended-lambda
	CXXFLAGS += --restrict -Xcompiler -fopenmp 
	CXXFLAGS += -x=cu $(NV_ARCH) -Xptxas -dlcm=cg
endif

############################
# CUDA compiler: make nv=1 #
############################
ifneq (,$(nv))
	CXX = nvcc
	CXXFLAGS += -D__NVVP__
	CXXFLAGS += --restrict $(NV_ARCH) -x=cu -Xptxas -dlcm=cg
ifneq (,$(l))	
	CXXFLAGS += --expt-extended-lambda
endif
endif

############################
# LAMBDAS launch: make l=1 #
############################
ifneq (,$(l))
	CXXFLAGS += -D__LAMBDA__
#	CXXFLAGS += -DLAGHOS_DEBUG
endif

#############################
# TEMPLATE launch: make t=1 #
#############################
ifneq (,$(t))
	CXXFLAGS += -D__TEMPLATES__
endif

#################################################
# CPU,LAMBDA: make v=1 l=1
# CPU,LAMBDA,TEMPLATE: make v=1 l=1 t=1
# GPU,LAMBDA,TEMPLATES: make v=1 nv=1 l=1 t=1
# GPU,CUDA Kernel, TEMPLATES: make v=1 nv=1 t=1
# RAJA: make v=1 rj=1 t=1
##################################################
.PHONY: cpuL cpuLT rj raja cpuRJ nvl gpuLT nvk gpuKT
cpuL:;$(MAKE) l=1 all
cpuLT:;$(MAKE) l=1 t=1 all

rj raja cpuRJ:;$(MAKE) rj=1 t=1 all

nv nvl gpuLT:;$(MAKE) nv=1 l=1 t=1 all
nvk gpuKT:;$(MAKE) nv=1 t=1 all

#######################
# TPL INCLUDES & LIBS #
#######################
MPI_INC = -I$(home)/usr/local/openmpi/3.0.0/include 

#DBG_INC = -I/home/camier1/home/dbg
#DBG_LIB = -Wl,-rpath -Wl,/home/camier1/home/dbg -L/home/camier1/home/dbg -ldbg 
#BKT_LIB = -Wl,-rpath -Wl,$(HOME)/lib -L$(HOME)/lib -lbacktrace

CUDA_HOME = /usr/local/cuda
CUDA_INC = -I$(CUDA_HOME)/samples/common/inc
CUDA_LIBS = -Wl,-rpath -Wl,/usr/local/cuda/lib64/lib -L/usr/local/cuda/lib64 \
				-lcuda -lcudart -lcudadevrt -lnvToolsExt

CUB_DIR = $(home)/usr/src/cub
RAJA_INC = -I$(CUB_DIR) -I$(home)/usr/local/raja/last/include
RAJA_LIBS = $(home)/usr/local/raja/last/lib/libRAJA.a

LAGHOS_FLAGS = $(CPPFLAGS) $(CXXFLAGS) $(MFEM_INCFLAGS) $(RAJA_INC) $(CUDA_INC) $(MPI_INC) $(DBG_INC)
LAGHOS_LIBS = $(ASAN_LIB) $(MFEM_LIBS) -fopenmp $(RAJA_LIBS) $(CUDA_LIBS) -ldl $(DBG_LIB) $(BKT_LIB)

ifeq ($(LAGHOS_DEBUG),YES)
   LAGHOS_FLAGS += -DLAGHOS_DEBUG
endif

#########################
# FINAL LIBS, CCC & Ccc #
#########################
LIBS = $(strip $(LAGHOS_LIBS) $(LDFLAGS))
CCC  = $(strip $(CXX) $(LAGHOS_FLAGS))
Ccc  = $(strip $(CC) $(CFLAGS) $(GL_OPTS))

################
# SOURCE FILES #
################
SOURCE_FILES = $(wildcard $(pwd)/*.cpp)
KERNEL_FILES = $(wildcard $(kernels)/*.cpp)
KERNEL_FILES += $(wildcard $(kernels)/blas/*.cpp)
KERNEL_FILES += $(wildcard $(kernels)/force/*.cpp)
KERNEL_FILES += $(wildcard $(kernels)/geom/*.cpp)
KERNEL_FILES += $(wildcard $(kernels)/maps/*.cpp)
KERNEL_FILES += $(wildcard $(kernels)/mass/*.cpp)
KERNEL_FILES += $(wildcard $(kernels)/quad/*.cpp)
KERNEL_FILES += $(wildcard $(kernels)/share/*.cpp)
RAJA_FILES = $(wildcard $(raja)/config/*.cpp)
RAJA_FILES += $(wildcard $(raja)/fem/*.cpp)
RAJA_FILES += $(wildcard $(raja)/general/*.cpp)
RAJA_FILES += $(wildcard $(raja)/linalg/*.cpp)

################
# OBJECT FILES #
################
OBJECT_FILES  = $(SOURCE_FILES:.cpp=.o)
OBJECT_FILES += $(KERNEL_FILES:.cpp=.o)
OBJECT_FILES += $(RAJA_FILES:.cpp=.o)
HEADER_FILES = laghos_solver.hpp laghos_assembly.hpp

################
# OUTPUT rules #
################
rule_file = $(notdir $(1))
rule_path = $(patsubst %/,%,$(dir $(1)))
last_path = $(notdir $(patsubst %/,%,$(dir $(1))))
ansicolor = $(shell echo $(call last_path,$(1)) | cksum | cut -b1-2 | xargs -IS expr 2 \* S + 17)
emacs_out = @echo -e $(call last_path,$(2))/$(call rule_file,$(2))
color_out = @if [ -t 1 ]; then \
				printf "%s \033[38;5;%d;1m%s\033[m/%s\n" \
					$(shell echo $(firstword $($(1)))) $(call ansicolor,$(2)) \
					$(call last_path,$(2)) $(call rule_file,$(2)); else \
				printf "%s %s\n" $(1) $(2); fi
# if TERM=dumb, use it, otherwise switch to the term one
output = $(if $(TERM:dumb=),$(call color_out,$1,$2),$(call emacs_out,$1,$2))

# if V is set to non-nil, turn the verbose mode
quiet = $(if $(v),$($(1)),$(call output,$1,$@);$($(1)))

###########
# Targets #
###########
.PHONY: all clean distclean install status info opt debug test style clean-build clean-exec

.SUFFIXES: .c .cpp .o

$(pwd)/%.o:$(pwd)/%.cpp
	$(call quiet,CCC) -c -o $@ $(abspath $<)

$(raja)/%.o: $(raja)/%.cpp $(raja)/%.hpp $(raja)/raja.hpp $(raja)/rmanaged.hpp
	$(call quiet,CCC) -c -o $@ $(abspath $<)

$(kernels)/%.o: $(kernels)/%.cpp $(kernels)/kernels.hpp $(kernels)/defines.hpp
	$(call quiet,CCC) -c -o $@ $(abspath $<)

all: 
	@$(MAKE) -j $(CPU) laghos

laghos: override MFEM_DIR = $(MFEM_DIR1)
laghos:	$(OBJECT_FILES) $(CONFIG_MK) $(MFEM_LIB_FILE)
	$(MFEM_CXX) -o laghos $(OBJECT_FILES) $(LIBS)

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

clean cln: clean-build clean-exec

clean-build:
	rm -rf laghos laghos-nvcc laghos-mpicxx *.o *~ *.dSYM \
	raja/*/*.o \
	raja/kernels/*.o raja/kernels/*/*.o 
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

print :
	@echo $(VAR)=$($(VAR))

print-%:
	$(info [ variable name]: $*)
	$(info [        origin]: $(origin $*))
	$(info [         value]: $(value $*))
	$(info [expanded value]: $($*))
	$(info )
	@true
