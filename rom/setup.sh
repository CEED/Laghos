#!/bin/bash

# Check whether Homebrew or wget is installed
if [ "$(uname)" == "Darwin" ]; then
  which -s brew > /dev/null
  if [[ $? != 0 ]] ; then
      # Install Homebrew
      echo "Homebrew installation is required."
      exit 1
  fi

  which -s wget > /dev/null
  # Install wget
  if [[ $? != 0 ]] ; then
      brew install wget
  fi
fi

# Replace with your LIB_DIR
LIB_DIR=$PWD/dependencies
mkdir -p $LIB_DIR

# Install HYPRE
cd $LIB_DIR
if [ ! -d "hypre" ]; then
  git clone https://github.com/hypre-space/hypre.git
  cd hypre/src
  ./configure --disable-fortran
  make -j
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
fi
cd mfem
git pull
if [[ "debug" == $1 ]]; then
  make pdebug -j MFEM_USE_MPI=YES MFEM_USE_METIS=YES MFEM_USE_METIS_5=YES METIS_DIR="$METIS_DIR" METIS_OPT="$METIS_OPT" METIS_LIB="$METIS_LIB"
else
  make parallel -j MFEM_USE_MPI=YES MFEM_USE_METIS=YES MFEM_USE_METIS_5=YES METIS_DIR="$METIS_DIR" METIS_OPT="$METIS_OPT" METIS_LIB="$METIS_LIB"
fi

# Install libROM
cd $LIB_DIR
if [ ! -d "libROM" ]; then
  git clone https://github.com/LLNL/libROM.git
fi
cd libROM
git pull
if [[ "debug" == $1 ]]; then
  ./scripts/laghos_compile.sh -DCMAKE_BUILD_TYPE=Debug
else
  ./scripts/laghos_compile.sh
fi

#Install astyle
cd $LIB_DIR
if [ ! -d "astyle" ]; then
  # Check machine
  case "$(uname -s)" in
      Linux*)
        wget -O astyle_2.05.1.tar.gz https://sourceforge.net/projects/astyle/files/astyle/astyle%202.05.1/astyle_2.05.1_linux.tar.gz/download
        ;;
      Darwin*)
        wget -O astyle_2.05.1.tar.gz https://sourceforge.net/projects/astyle/files/astyle/astyle%202.05.1/astyle_2.05.1_macosx.tar.gz/download
        ;;
  esac
  tar -zxvf astyle_2.05.1.tar.gz
  cd astyle/build
  if [ -d "gcc" ]; then
    cd gcc
  elif [ -d "mac" ]; then
    cd mac
  fi
  make
fi
