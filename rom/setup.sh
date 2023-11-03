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
LAGHOS_LIB_DIR=$PWD/dependencies
mkdir -p $LAGHOS_LIB_DIR

# Install libROM
cd $LAGHOS_LIB_DIR
if [ ! -d "libROM" ]; then
  git clone https://github.com/LLNL/libROM.git
fi
cd libROM
git pull
if [[ $1 == "YES" ]]; then
    DEBUG="-d"
fi
if [[ $2  == "YES" ]]; then
    UPDATE="-u"
fi
./scripts/compile.sh -m $DEBUG $UPDATE

#Install astyle
cd $LAGHOS_LIB_DIR
if [ ! -d "astyle" ]; then
  # Check machine
  case "$(uname -s)" in
      Linux*)
        # wget -O astyle_2.05.1.tar.gz https://sourceforge.net/projects/astyle/files/astyle/astyle%202.05.1/astyle_2.05.1_linux.tar.gz/download
        wget -O astyle_3.1.tar.gz https://sourceforge.net/projects/astyle/files/astyle/astyle%203.1/astyle_3.1_linux.tar.gz/download
        ;;
      Darwin*)
        # wget -O astyle_2.05.1.tar.gz https://sourceforge.net/projects/astyle/files/astyle/astyle%202.05.1/astyle_2.05.1_macosx.tar.gz/download
        wget -O astyle_3.1.tar.gz https://sourceforge.net/projects/astyle/files/astyle/astyle%203.1/astyle_3.1_macosx.tar.gz/download
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
