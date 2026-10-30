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
