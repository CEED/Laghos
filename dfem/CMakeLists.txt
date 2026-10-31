# dFEM kernels are differentiated by Enzyme at compile time (ClangEnzyme plugin)
find_package(Enzyme REQUIRED HINTS ${ENZYME_DIR})

# Assembly and solver, compiled once for laghos_dfem and laghos_tests
add_library(laghos_dfem_lib OBJECT laghos_assembly.cpp laghos_solver.cpp)

target_include_directories(laghos_dfem_lib PUBLIC "${MFEM_DIR}/../../../include/mfem")

target_link_libraries(laghos_dfem_lib
  PUBLIC mfem
  PUBLIC MPI::MPI_CXX
  PUBLIC HYPRE::HYPRE
  PUBLIC ClangEnzymeFlags
)

add_executable(laghos_dfem laghos.cpp)
target_link_libraries(laghos_dfem PUBLIC laghos_dfem_lib)

# Non-regression tests, see laghos_tests.cpp, with the Catch2 single header
# of the MFEM source tree (tests/unit/catch.hpp)
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

# laghos.cpp again, with its main as laghos(), called by the tests
add_library(laghos_tests_lib OBJECT laghos.cpp)
target_compile_definitions(laghos_tests_lib PRIVATE main=laghos)
target_link_libraries(laghos_tests_lib PUBLIC laghos_dfem_lib)

add_executable(laghos_tests laghos_tests.cpp)
target_include_directories(laghos_tests SYSTEM PRIVATE ${CATCH_INCLUDE_DIR})
target_compile_definitions(laghos_tests PRIVATE
  LAGHOS_DATA_DIR="${PROJECT_SOURCE_DIR}/data/")
target_link_libraries(laghos_tests PRIVATE laghos_tests_lib laghos_dfem_lib)

# No colors in the ctest logs: mpiexec may give the ranks a pseudo-terminal
foreach(test sedov_2d sedov_2d_rk2 sedov_3d sedov_2d_stock sedov_3d_stock
             taylor_green_2d taylor_green_3d taylor_green_3d_av
             readme_taylor_green_2d readme_taylor_green_3d
             readme_sedov_2d readme_sedov_3d
             taylor_green_3d_q1q0 sedov_3d_q1q0_stock taylor_green_3d_stock_av
             triple_point_3d triple_point_3d_av)
  add_test(NAME dfem_${test}
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
            $<TARGET_FILE:laghos_tests> ${test})
  set_tests_properties(dfem_${test} PROPERTIES ENVIRONMENT NO_COLOR=1
    LABELS explicit)
endforeach()

# Same references in parallel
foreach(test sedov_2d sedov_3d sedov_2d_stock sedov_3d_stock taylor_green_2d
             taylor_green_3d taylor_green_3d_av
             readme_taylor_green_2d readme_taylor_green_3d readme_sedov_2d
             taylor_green_3d_q1q0 sedov_3d_q1q0_stock taylor_green_3d_stock_av
             triple_point_3d triple_point_3d_av)
  add_test(NAME dfem_${test}_np2
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:laghos_tests> ${test})
  set_tests_properties(dfem_${test}_np2 PROPERTIES ENVIRONMENT NO_COLOR=1
    LABELS explicit)
endforeach()

# Same references on an odd number of ranks, readme_sedov_3d drifts by 1.5e-10
foreach(test sedov_2d sedov_2d_rk2 sedov_3d sedov_2d_stock sedov_3d_stock
             taylor_green_2d taylor_green_3d taylor_green_3d_av
             readme_taylor_green_2d readme_taylor_green_3d readme_sedov_2d
             taylor_green_3d_q1q0 sedov_3d_q1q0_stock taylor_green_3d_stock_av
             triple_point_3d triple_point_3d_av)
  add_test(NAME dfem_${test}_np3
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 3
            $<TARGET_FILE:laghos_tests> ${test})
  set_tests_properties(dfem_${test}_np3 PROPERTIES ENVIRONMENT NO_COLOR=1
    LABELS explicit)
endforeach()

# Implicit Newton and AMG runs with the references of the pre-dfem-dev MFEM,
# ctest -L implicit, or ctest -LE implicit for the explicit tests only
foreach(test implicit_taylor_green_2d implicit_taylor_green_2d_be
             implicit_sedov_2d implicit_sedov_2d_av
             implicit_taylor_green_3d implicit_sedov_3d_av)
  foreach(np 1 2 3)
    add_test(NAME dfem_${test}_np${np}
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
              $<TARGET_FILE:laghos_tests> ${test})
    set_tests_properties(dfem_${test}_np${np} PROPERTIES ENVIRONMENT NO_COLOR=1
      LABELS implicit)
  endforeach()
endforeach()
