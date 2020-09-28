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
   make setup
   make setup MFEM_BUILD=pcuda
   make status/info
   make test
   make tests
   make checks
   make install
   make clean
   make distclean
   make style

Examples:

make setup
   Build Laghos third party libraries: HYPRE, METIS and MFEM
   (By default MFEM will be compiled in parallel mode, but MFEM_BUILD=pcuda
    will allow a parallel CUDA build.)
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

NPROC = $(shell getconf _NPROCESSORS_ONLN)
GOALS = help clean distclean style setup mfem metis hypre

# Default installation location
PREFIX ?= ./bin
INSTALL = /usr/bin/install

# Use the MFEM source, build, or install directory
MFEM_DIR ?= ../mfem
CONFIG_MK = $(MFEM_DIR)/config/config.mk
ifeq ($(wildcard $(CONFIG_MK)),)
   CONFIG_MK = $(MFEM_DIR)/share/mfem/config.mk
endif
TEST_MK = $(MFEM_TEST_MK)

# Use the compiler used by MFEM. Get the compiler and the options for compiling
# and linking from MFEM's config.mk. (Skip this if the target does not require
# building.)
MFEM_LIB_FILE = mfem_is_not_built
ifeq (,$(filter $(GOALS),$(MAKECMDGOALS)))
   -include $(CONFIG_MK)
   ifneq ($(realpath $(MFEM_DIR)),$(MFEM_SOURCE_DIR))
      ifneq ($(realpath $(MFEM_DIR)),$(MFEM_INSTALL_DIR))
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
CCC = $(strip $(CXX) $(LAGHOS_FLAGS) $(if $(EXTRA_INC_DIR),-I$(EXTRA_INC_DIR)))

LAGHOS_LIBS = $(MFEM_LIBS) $(MFEM_EXT_LIBS)
LIBS = $(strip $(LAGHOS_LIBS) $(LDFLAGS))

SOURCE_FILES = $(sort $(wildcard *.cpp))
HEADER_FILES = $(sort $(wildcard *.hpp))
OBJECT_FILES = $(SOURCE_FILES:.cpp=.o)

# Targets

.PHONY: all clean distclean install status info opt debug test tests style \
	clean-build clean-exec clean-tests setup mfem hypre metis

.SUFFIXES: .cpp .o
.cpp.o:
	cd $(<D); $(CCC) -c $(<F)

laghos: $(OBJECT_FILES) $(CONFIG_MK) $(MFEM_LIB_FILE)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o laghos $(OBJECT_FILES) $(LIBS)

all:;@$(MAKE) -j $(NPROC) laghos

$(OBJECT_FILES): $(HEADER_FILES) $(CONFIG_MK)

# Quick test with specific execution options
MFEM_TESTS = laghos
RUN_MPI_4 = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) 4
test: laghos
	@$(call mfem-test,$<, $(RUN_MPI_4), Laghos miniapp,\
	-p 0 -m data/square01_quad.mesh -rs 3 -tf 0.1)
# Testing: "test" target and mfem-test* variables are defined in MFEM's
# config/test.mk
ifeq (,$(filter $(GOALS),$(MAKECMDGOALS)))
include $(TEST_MK)
endif

# Generate an error message if the MFEM library is not built and exit
$(CONFIG_MK) $(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

cln clean: clean-build clean-exec clean-tests

clean-build:
	rm -rf laghos *.o *~ *.dSYM
clean-exec:
	rm -rf ./results
clean-tests:
	rm -rf BASELINE.dat RUN.dat RESULTS.dat
distclean: clean
	rm -rf bin/

install: laghos
	mkdir -p $(PREFIX)
	$(INSTALL) -m 750 laghos $(PREFIX)

help:
	$(info $(value LAGHOS_HELP_MSG))
	@true

status info:
	$(info MFEM_DIR     = $(MFEM_DIR))
	$(info LAGHOS_FLAGS = $(LAGHOS_FLAGS))
	$(info LAGHOS_LIBS  = $(value LAGHOS_LIBS))
	$(info PREFIX       = $(PREFIX))
	@true

ASTYLE = astyle --options=$(MFEM_DIR)/config/mfem.astylerc
FORMAT_FILES := $(SOURCE_FILES) $(HEADER_FILES)
style:
	@if ! $(ASTYLE) $(FORMAT_FILES) | grep Formatted; then\
	   echo "No source files were changed.";\
	fi

# Laghos checks template - Default arguments
ECHO=echo
SED=sed -e
ranks=1
dims=2 3
problems=0 1 2 3 4 5 6
OPTS=-cgt 1.e-14 -rs 0 --checks
USE_CUDA := $(MFEM_USE_CUDA:NO=)
optioni=1 2$(if $(USE_CUDA), 3)
options=-fa -pa $(if $(USE_CUDA),-d_cuda) #-d_debug
#optioni = $(shell for i in {1..$(words $(options))}; do echo $$i; done)

# Laghos checks template - Targets
define laghos_checks_template
.PHONY: laghos_$(1)_$(2)_$(3)_$(4)
laghos_$(1)_$(2)_$(3)_$(4): laghos
	$(eval name=laghos-x$(4)-p$(1)-$(2)D$(word $(3),$(options)))
	$(eval command=$(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(4) ./laghos $(OPTS) -p $(1) -dim $(2) $(shell echo $(word $(3),$(options))|$(SED) "s/-/ -/g"|$(SED) "s/_/ /g"))
	@$(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(4) ./$$< $(OPTS) -p $(1) -dim $(2) $(shell echo $(word $(3),$(options))|$(SED) "s/-/ -/g"|$(SED) "s/_/ /g") > /dev/null 2>&1 && \
	$(call COLOR_PRINT,'\033[0;32m',OK,': $(name)\n') || $(call COLOR_PRINT,'\033[1;31m',KO,': $(command)\n');
endef
# Generate all Laghos checks template targets
$(foreach p, $(problems), $(foreach d, $(dims), $(foreach o, $(optioni), $(foreach r, $(ranks),\
	$(eval $(call laghos_checks_template,$(p),$(d),$(o),$(r)))))))
# Output info on all Laghos checks template targets
#$(foreach p, $(problems), $(foreach d, $(dims), $(foreach o, $(optioni), $(foreach r, $(ranks),\
#   $(info $(call laghos_checks_template,$(p),$(d),$(o),$(r)))))))
checks: laghos
checks: |$(foreach p,$(problems), $(foreach d,$(dims), $(foreach o,$(optioni), $(foreach r,$(ranks), laghos_$(p)_$(d)_$(o)_$(r)))))

1:;@$(MAKE) -j $(NPROC) checks ranks=1
2:;@$(MAKE) -j 8 checks ranks=2
3:;@$(MAKE) -j 4 checks ranks=3

# Laghos run tests
tests:
	cat << EOF > RESULTS.dat
	$(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP) \
	./laghos -p 0 -dim 2 -rs 3 -tf 0.75 -pa -vs 100 | tee RUN.dat
	cat RUN.dat | tail -n 20 | head -n 1 | \
	awk '{ printf("step = %04d, dt = %s |e| = %.10e\n", $$2, $$8, $$11); }' >> RESULTS.dat
	$(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP) \
	./laghos -p 0 -dim 3 -rs 1 -tf 0.75 -pa -vs 100 | tee RUN.dat
	cat RUN.dat | tail -n 20 | head -n 1 | \
	awk '{ printf("step = %04d, dt = %s |e| = %.10e\n", $$2, $$8, $$11); }' >> RESULTS.dat
	$(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP) \
	./laghos -p 1 -dim 2 -rs 3 -tf 0.8 -pa -vs 100 | tee RUN.dat
	cat RUN.dat | tail -n 17 | head -n 1 | \
	awk '{ printf("step = %04d, dt = %s |e| = %.10e\n", $$2, $$8, $$11); }' >> RESULTS.dat
	$(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP) \
	./laghos -p 1 -dim 3 -rs 2 -tf 0.6 -pa -vs 100 | tee RUN.dat
	cat RUN.dat | tail -n 17 | head -n 1 | \
	awk '{ printf("step = %04d, dt = %s |e| = %.10e\n", $$2, $$8, $$11); }' >> RESULTS.dat
	$(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP) \
	./laghos -p 2 -dim 1 -rs 5 -tf 0.2 -fa -vs 100 | tee RUN.dat
	cat RUN.dat | tail -n 18 | head -n 1 | \
	awk '{ printf("step = %04d, dt = %s |e| = %.10e\n", $$2, $$8, $$11); }' >> RESULTS.dat
	$(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP) \
	./laghos -p 3 -m data/rectangle01_quad.mesh -rs 2 -tf 3.0 -pa -vs 100 | tee RUN.dat
	cat RUN.dat | tail -n 17 | head -n 1 | \
	awk '{ printf("step = %04d, dt = %s |e| = %.10e\n", $$2, $$8, $$11); }' >> RESULTS.dat
	$(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP) \
	./laghos -p 3 -m data/box01_hex.mesh -rs 1 -tf 3.0 -pa -vs 100 | tee RUN.dat
	cat RUN.dat | tail -n 17 | head -n 1 | \
	awk '{ printf("step = %04d, dt = %s |e| = %.10e\n", $$2, $$8, $$11); }' >> RESULTS.dat
	$(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP) \
	./laghos -p 4 -m data/square_gresho.mesh -rs 3 -ok 3 \
	         -ot 2 -tf 0.62831853 -s 7 -pa -vs 100 | tee RUN.dat
	cat RUN.dat | tail -n 20 | head -n 1 | \
	awk '{ printf("step = %04d, dt = %s |e| = %.10e\n", $$2, $$8, $$11); }' >> RESULTS.dat
	$(shell cat << EOF > BASELINE.dat)
	$(shell echo 'step = 0339, dt = 0.000702, |e| = 4.9695537349e+01' >> BASELINE.dat)
	$(shell echo 'step = 1041, dt = 0.000121, |e| = 3.3909635545e+03' >> BASELINE.dat)
	$(shell echo 'step = 1154, dt = 0.001655, |e| = 4.6303396053e+01' >> BASELINE.dat)
	$(shell echo 'step = 0560, dt = 0.002449, |e| = 1.3408616722e+02' >> BASELINE.dat)
	$(shell echo 'step = 0413, dt = 0.000470, |e| = 3.2012077410e+01' >> BASELINE.dat)
	$(shell echo 'step = 2872, dt = 0.000064, |e| = 5.6547039096e+01' >> BASELINE.dat)
	$(shell echo 'step = 0528, dt = 0.000180, |e| = 5.6505348812e+01' >> BASELINE.dat)
	$(shell echo 'step = 0776, dt = 0.000045, |e| = 4.0982431726e+02' >> BASELINE.dat)
	diff --report-identical-files RESULTS.dat BASELINE.dat

# Setup: download & install third party libraries: HYPRE, METIS & MFEM

HYPRE_URL = https://computation.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods
HYPRE_VER = 2.11.2
HYPRE_DIR = hypre
hypre:
	@(if [[ ! -e ../$(HYPRE_DIR) ]]; then cd ..; \
		wget -nc $(HYPRE_URL)/download/hypre-$(HYPRE_VER).tar.gz &&\
		tar xzvf hypre-$(HYPRE_VER).tar.gz &&\
		ln -s hypre-$(HYPRE_VER) $(HYPRE_DIR) &&\
		cd $(HYPRE_DIR)/src &&\
		./configure --disable-fortran --without-fei CC=mpicc CXX=mpic++ &&\
		make -j $(NPROC);	else echo "Using existing ../$(HYPRE_DIR)"; fi)

METIS_URL = http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis
METIS_VER = 4.0.3
METIS_DIR = metis-4.0
metis:
	@(if [[ ! -e ../$(METIS_DIR) ]]; then cd ..; \
		wget -nc $(METIS_URL)/OLD/metis-$(METIS_VER).tar.gz &&\
		tar zxvf metis-$(METIS_VER).tar.gz &&\
		ln -s metis-$(METIS_VER) $(METIS_DIR) &&\
		cd $(METIS_DIR) &&\
		make -j $(NPROC) OPTFLAGS="-O2";\
		else echo "Using existing ../$(METIS_DIR)"; fi)

MFEM_GIT = https://github.com/mfem/mfem.git
MFEM_BUILD ?= parallel
#MFEM_BUILD ?= pcuda -j CUDA_ARCH=sm_70
mfem: hypre metis
	@(if [[ ! -e ../mfem ]]; then cd ..; \
		git clone --single-branch --branch master --depth 1 $(MFEM_GIT) &&\
		cd mfem &&\
		make $(MFEM_BUILD) -j $(NPROC); else echo "Using existing ../mfem"; fi)

setup: mfem
