# dFEM kernels are differentiated by Enzyme at compile time (ClangEnzyme plugin)
find_package(Enzyme REQUIRED HINTS ${ENZYME_DIR})

add_executable(laghos_dfem laghos.cpp)

target_include_directories(laghos_dfem PUBLIC "${MFEM_DIR}/../../../include/mfem")

target_link_libraries(laghos_dfem
  PUBLIC mfem
  PUBLIC MPI::MPI_CXX
  PUBLIC HYPRE::HYPRE
  PUBLIC ClangEnzymeFlags
)

# Non-regression tests, see laghos_tests.cpp, with the Catch2 single header
# of the MFEM source tree (tests/unit/catch.hpp), as in quest/mfem/dft
get_target_property(mfemIncludeDirs mfem INTERFACE_INCLUDE_DIRECTORIES)
find_path(CATCH_INCLUDE_DIR catch.hpp
  HINTS ${MFEM_SOURCE_DIR} ${MFEM_DIR} ${MFEM_DIR}/.. ${mfemIncludeDirs}
  PATH_SUFFIXES tests/unit
  NO_DEFAULT_PATH)
if (NOT CATCH_INCLUDE_DIR)
  message(WARNING "MFEM's tests/unit/catch.hpp not found, no laghos_tests: "
                  "set MFEM_SOURCE_DIR to the MFEM source tree")
  return()
endif()
message(STATUS "Catch2 for laghos_tests: ${CATCH_INCLUDE_DIR}/catch.hpp")

add_executable(laghos_tests laghos_tests.cpp laghos.cpp)
target_include_directories(laghos_tests SYSTEM PRIVATE ${CATCH_INCLUDE_DIR})
target_compile_definitions(laghos_tests PRIVATE
  LAGHOS_TESTS
  LAGHOS_DATA_DIR="${PROJECT_SOURCE_DIR}/data/")
target_include_directories(laghos_tests PUBLIC "${MFEM_DIR}/../../../include/mfem")
target_link_libraries(laghos_tests
  PUBLIC mfem
  PUBLIC MPI::MPI_CXX
  PUBLIC HYPRE::HYPRE
  PUBLIC ClangEnzymeFlags
)

foreach(test sedov_2d sedov_2d_rk2 sedov_3d
             taylor_green_2d taylor_green_3d taylor_green_3d_av)
  add_test(NAME dfem_${test}
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
            $<TARGET_FILE:laghos_tests> ${test})
endforeach()

# Same references in parallel
foreach(test sedov_2d sedov_3d taylor_green_2d)
  add_test(NAME dfem_${test}_np2
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:laghos_tests> ${test})
endforeach()
