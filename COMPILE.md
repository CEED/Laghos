# Let's compile Laghos and libROM!

This document explains how to compile Laghos/libROM together for several different machines:

## First, on macbook!

1. Run the following commands to get the required dependencies to compile Laghos, libROM, Hypre, MFEM, and Metis.
   * xcode-select --install
   * brew install open-mpi
   * brew install openblas
   * brew install lapack
   * brew install scalapack
   * brew install hdf5
2. Choose a base directory, call it \<PATH\>.
3. From \<PATH\>, clone the rom-dev branch of Laghos.
4. From \<PATH\>, download and build HYPRE and Metis following the instructions from the README.md of the master branch of Laghos.
5. From \<PATH\>, download MFEM and run the following commands from \<PATH\>/mfem/config:
   * cp defaults.mk user.mk
   * Alter HYPRE_DIR to user.mk to point to your HYPRE folder. Mine looks like the following:

      HYPRE_DIR = @MFEM_DIR@/../hypre-2.11.2/src/hypre
6. Build MFEM following the instructions from the README.md of the master branch of Laghos.
7. From \<PATH\>, clone the master branch of the libROM repository.
8. Run the following commands from \<PATH\>/libROM/build:
   * cmake ../
   * make
9. From \<PATH\>/Laghos, create a user.mk file that includes your ScaLAPACK path and flags. Here is an example:
   * SCALAPACK_FLAGS=-L/usr/local/opt/scalapack -lscalapack       
10. From \<PATH\>/Laghos, run the following command:
      * make
