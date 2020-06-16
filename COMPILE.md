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
4. From \<PATH\>, follow the instructions from the master branch of Laghos to build HYPRE, Metis, and MFEM, but don’t compile Laghos yet. The instructions in the rom-dev branch are out-of-date and will lead to errors as the instructions say to use an older MFEM version that won’t compile with rom-dev.
5. Before building MFEM by running (make parallel -j), you will need to run the following commands from the \<PATH\>/mfem/config directory:
   * cp defaults.mk user.mk
   * Alter HYPRE_DIR to user.mk to point to your HYPRE folder. Mine looks like the following: 
   
      HYPRE_DIR = @MFEM_DIR@/../hypre-2.11.2/src/hypre
      
6. From \<PATH\>, clone the master branch of the libROM repository. 
7. Alter the CMakeLists.txt by making the libROM library shared so you can avoid having to relink its dependencies when compiling Laghos. If you want to relink the dependencies in the Laghos makefile instead of making the library shared, you can reinclude (MPI, HDF5, Fortran, etc.) in the Laghos makefile. You will also need to include MPI_Fortran so Scalapack can be linked correctly. Here is a git diff of the changes I made from the master version of CMakeLists.txt:
   
         diff --git a/CMakeLists.txt b/CMakeLists.txt
         index b81fae5..4f19ce6 100644
         --- a/CMakeLists.txt
         +++ b/CMakeLists.txt
         @@ -144,7 +144,7 @@ list(APPEND source_files
            scalapack_c_wrapper.c
            scalapack_f_wrapper.f90)
 
         -add_library(ROM ${source_files})
         +add_library(ROM SHARED ${source_files} )
 
         # List minimum version requirements for dependencies where possible to make
         # packaging easier later.
         @@ -194,7 +194,7 @@ find_package(GTest 1.6.0)
         # but is done here to ease a potential rollback to CMake 2.8 or CMake 3.0
         target_link_libraries(ROM
         -  PUBLIC ${MPI_C_LINK_FLAGS} ${MPI_C_LIBRARIES} MPI::MPI_C ${HDF5_LIBRARIES}
         +  PUBLIC ${MPI_C_LINK_FLAGS} ${MPI_C_LIBRARIES} MPI::MPI_C ${MPI_Fortran_LINK_FLAGS} ${MPI_Fortran_LIBRARIES} MPI::MPI_Fortran ${HDF5_LIBRARIES}
            ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES}
            PRIVATE ${ZLIB_LIBRARIES} ZLIB::ZLIB)
 
         @@ -218,7 +218,7 @@ foreach(name IN LISTS regression_test_names)
            add_executable(${name} ${name}.C)
 
            target_link_libraries(${name}
         -    PRIVATE ROM ${MPI_C_LINK_FLAGS} ${MPI_C_LIBRARIES} MPI::MPI_C)
         +    PRIVATE ROM ${MPI_C_LINK_FLAGS} ${MPI_C_LIBRARIES} MPI::MPI_C ${MPI_Fortran_LINK_FLAGS} ${MPI_Fortran_LIBRARIES} MPI::MPI_Fortran)
            target_include_directories(${name}
              PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
              ${MPI_C_INCLUDE_DIRS})
         @@ -234,7 +234,7 @@ if(GTEST_FOUND)
            foreach(stem IN LISTS unit_test_stems)
              add_executable(test_${stem} tests/test_${stem}.C)
              target_link_libraries(test_${stem} PRIVATE ROM
         -      ${MPI_C_LINK_FLAGS} ${MPI_C_LIBRARIES} MPI::MPI_C GTest::GTest)
         +      ${MPI_C_LINK_FLAGS} ${MPI_C_LIBRARIES} MPI::MPI_C GTest::GTest ${MPI_Fortran_LIBRARIES})
              target_include_directories(test_${stem}
                PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                ${MPI_C_INCLUDE_DIRS})
8. Run the following commands from \<PATH\>/libROM/build:
   * cmake ../
   * make
9. In \<PATH\>/Laghos, create a user.mk file that includes your ScaLAPACK path and flags. Here is an example:
   * SCALAPACKLIB = -Wl,-rpath,/usr/local/opt/scalapack/lib
   * SCALAPACK_FLAGS = -lscalapack -lpthread -lm -ldl
10. We need to include the correct source files and libraries in the Laghos executable in the makefile for it to build correctly. Here is the git diff of the changes I made from the rom-dev branch version of the makefile:
   
         --- a/makefile
         +++ b/makefile
         @@ -122,8 +122,10 @@ include user.mk
         .c.o:
            cd $(<D); $(Ccc) -c $(<F)
      
         -laghos: override MFEM_DIR = $(MFEM_DIR1)
         -include user2.mk
         +laghos: $(OBJECT_FILES) $(CONFIG_MK) $(MFEM_LIB_FILE)
         +      $(CCC) -o laghos $(OBJECT_FILES) $(LIBS) -L../libROM/build -lROM -Wl,-rpath,../libROM/build
         +      $(SCALAPACKLIB) $(SCALAPACK_FLAGS)
         +#include user2.mk
         
11. From \<PATH\>/Laghos, run the following command:
      * make
   
Now Laghos and libROM have compiled together and we are finished.
