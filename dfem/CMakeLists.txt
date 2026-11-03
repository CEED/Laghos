# Assembly and solver, compiled once for laghos_dfem and laghos_tests
add_library(laghos_dfem_lib OBJECT laghos_assembly.cpp laghos_solver.cpp
            laghos_qf_hydro.cpp laghos_qf_velocity.cpp laghos_qf_energy.cpp)

target_include_directories(laghos_dfem_lib PUBLIC "${MFEM_DIR}/../../../include/mfem")

target_link_libraries(laghos_dfem_lib
  PUBLIC mfem
  PUBLIC MPI::MPI_CXX
  PUBLIC HYPRE::HYPRE
)

if (MFEM_USE_ENZYME)
  find_package(Enzyme REQUIRED HINTS ${ENZYME_DIR})
  target_link_libraries(laghos_dfem_lib PUBLIC ClangEnzymeFlags)
endif()

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

# Devices of the ctest variants: the GPU of MFEM, if any, and the debug device,
# ctest -L device, or ctest -LE device for the cpu tests only
set(devices debug)
if (MFEM_USE_CUDA)
  set(devices cuda debug)
elseif (MFEM_USE_HIP)
  set(devices hip debug)
endif()

# One ctest per device of a test on np ranks with the qf q-function backend,
# dfem_<name> on the cpu and dfem_<device>_<name> otherwise, with a global_
# name prefix and label for the global q-functions, no colors: mpiexec may give
# a pseudo-terminal
function(laghos_test name test np label qf)
  set(qf_labels)
  if (qf STREQUAL global)
    set(name global_${name})
    set(qf_labels global)
  endif()
  foreach(device ${ARGN})
    if (device STREQUAL cpu)
      set(ctest dfem_${name})
      set(labels ${label} ${qf_labels})
    else()
      set(ctest dfem_${device}_${name})
      set(labels ${label} ${qf_labels} device ${device})
    endif()
    add_test(NAME ${ctest}
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
              $<TARGET_FILE:laghos_tests> --device ${device} --qf ${qf} ${test})
    set_tests_properties(${ctest} PROPERTIES ENVIRONMENT NO_COLOR=1
      LABELS "${labels}")
  endforeach()
endfunction()

# Runs of each q-function backend with the same references, ctest -L global,
# or ctest -LE global for the local q-functions only
foreach(qf local global)
  # Explicit runs with the same references on 1, 2 and 3 ranks
  foreach(test sedov_2d sedov_2d_rk2 sedov_3d sedov_2d_stock sedov_3d_stock
               taylor_green_2d taylor_green_3d taylor_green_3d_av
               readme_taylor_green_2d readme_taylor_green_3d
               readme_sedov_2d readme_sedov_3d
               taylor_green_3d_q1q0 sedov_3d_q1q0_stock taylor_green_3d_stock_av
               triple_point_3d triple_point_3d_av)
    laghos_test(${test} ${test} 1 explicit ${qf} cpu ${devices})
    foreach(np 2 3)
      laghos_test(${test}_np${np} ${test} ${np} explicit ${qf} cpu ${devices})
    endforeach()
  endforeach()

  # Implicit Newton and AMG runs with the references of the pre-dfem-dev MFEM,
  # ctest -L implicit, or ctest -LE implicit for the explicit tests only
  foreach(test implicit_taylor_green_2d implicit_taylor_green_2d_be
               implicit_sedov_2d implicit_sedov_2d_av implicit_sedov_2d_av1
               implicit_taylor_green_3d implicit_sedov_3d_av)
    set(test_devices ${devices})
    # Debug device fault: hypre host reads of aliases protect valid device pages
    if (test STREQUAL implicit_sedov_3d_av)
      list(REMOVE_ITEM test_devices debug)
    endif()
    foreach(np 1 2 3)
      laghos_test(${test}_np${np} ${test} ${np} implicit ${qf} cpu
                  ${test_devices})
    endforeach()
  endforeach()
endforeach()
