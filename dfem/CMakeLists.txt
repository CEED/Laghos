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

# Non-regression tests, see laghos_tests.cpp
add_executable(laghos_tests laghos_tests.cpp laghos.cpp)

target_compile_definitions(laghos_tests PRIVATE
  LAGHOS_TESTS
  LAGHOS_DATA_DIR="${PROJECT_SOURCE_DIR}/data/"
)

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
            $<TARGET_FILE:laghos_tests> -t ${test})
endforeach()

# Same references in parallel
foreach(test sedov_2d sedov_3d taylor_green_2d)
  add_test(NAME dfem_${test}_np2
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:laghos_tests> -t ${test})
endforeach()
