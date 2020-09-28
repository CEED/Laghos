#!/bin/bash

# Replace with your LIB_DIR
LIB_DIR=$PWD/dependencies
mkdir -p $LIB_DIR

# Install HYPRE
cd $LIB_DIR
if [ ! -d "hypre" ]; then
  wget https://computing.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods/download/hypre-2.11.2.tar.gz
  tar -zxvf hypre-2.11.2.tar.gz
  cd hypre-2.11.2/src
  ./configure --disable-fortran
  make -j
  cd $LIB_DIR
  mv hypre-2.11.2 hypre
fi

# Install PARMETIS 4.0.3
cd $LIB_DIR
if [ ! -d "parmetis-4.0.3" ]; then

  wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/parmetis/parmetis-4.0.3.tar.gz
  tar -zxvf parmetis-4.0.3.tar.gz
  cd parmetis-4.0.3
  make config
  make
fi

METIS_DIR=$LIB_DIR/parmetis-4.0.3
METIS_OPT=-I${METIS_DIR}/metis/include
MACHINE_ARCH=$(ls ${METIS_DIR}/build)
METIS_LIB="-L${METIS_DIR}/build/${MACHINE_ARCH}/libparmetis -lparmetis -L${METIS_DIR}/build/${MACHINE_ARCH}/libmetis -lmetis"

# Install MFEM
cd $LIB_DIR
if [ ! -d "mfem" ]; then
  git clone https://github.com/mfem/mfem.git
  cd mfem
  make parallel -j MFEM_USE_MPI=YES MFEM_USE_METIS=YES MFEM_USE_METIS_5=YES METIS_DIR="$METIS_DIR" METIS_OPT="$METIS_OPT" METIS_LIB="$METIS_LIB"
fi

# Install libROM
cd $LIB_DIR
if [ ! -d "libROM" ]; then
  git clone https://github.com/LLNL/libROM.git
  cd libROM
  ./scripts/laghos_compile.sh
fi
