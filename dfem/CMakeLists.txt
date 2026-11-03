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

# Name tags of the MFEM devices and of the q-function backends: cpu, dbg for
# debug, gpu for cuda or hip, loc for local and glb for global
set(device_cpu cpu)
set(device_dbg debug)
set(devices cpu)
if (MFEM_USE_CUDA)
  set(device_gpu cuda)
  list(APPEND devices gpu)
elseif (MFEM_USE_HIP)
  set(device_gpu hip)
  list(APPEND devices gpu)
endif()
list(APPEND devices dbg)
set(qf_loc local)
set(qf_glb global)

# Tests of the TEST_CASE names of laghos_tests.cpp, exp_ or imp_ prefixed
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS laghos_tests.cpp)
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/laghos_tests.cpp tests
  REGEX "^TEST_CASE\\(\"")
list(TRANSFORM tests REPLACE "^TEST_CASE\\(\"([^\"]+)\".*" "\\1")

# One ctest dfem_<device>_<qf>_<test>_np<np> of the Catch2 test spec <spec>,
# labeled <labels> with its device and its q-function backend, no colors:
# mpiexec may give a pseudo-terminal
function(laghos_test device qf test np spec labels)
  set(ctest dfem_${device}_${qf}_${test}_np${np})
  add_test(NAME ${ctest}
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:laghos_tests>
            ${MPIEXEC_POSTFLAGS} --device ${device_${device}}
            --qf ${qf_${qf}} ${spec})
  set_tests_properties(${ctest} PROPERTIES ENVIRONMENT NO_COLOR=1
    PROCESSORS ${np} LABELS "${device};${qf};${labels}")
endfunction()

# Debug device fault: hypre host reads of aliases protect valid pages
set(skip_dbg imp_run_sdv_3d_q21_mid_av7_dtn)

# Groups of all the tests in one process on cpu and dbg, not on GPUs where the
# results depend on the previous tests, or one ctest per test when OFF
option(LAGHOS_TESTS_GROUPS "One ctest per device, q-function and ranks" ON)
set(grouped)
if (LAGHOS_TESTS_GROUPS)
  set(grouped cpu dbg)
endif()

# Runs on 1, 2 and 3 ranks on each device with each q-function backend, with
# the same references: one dfem_<device>_<qf>_all_np<np> group, labeled group,
# or each test, labeled exp or imp, and chk for the README checks or run
foreach(qf loc glb)
  foreach(np 1 2 3)
    foreach(device ${grouped})
      set(spec "[dfem]")
      if (device STREQUAL dbg)
        string(APPEND spec "~${skip_dbg}")
      endif()
      laghos_test(${device} ${qf} all ${np} ${spec} group)
    endforeach()
  endforeach()
  foreach(test ${tests})
    string(SUBSTRING ${test} 0 3 run)
    string(SUBSTRING ${test} 4 3 kind)
    foreach(np 1 2 3)
      foreach(device ${devices})
        if (NOT (device IN_LIST grouped OR
                 (device STREQUAL dbg AND test STREQUAL "${skip_dbg}")))
          laghos_test(${device} ${qf} ${test} ${np} ${test} "${run};${kind}")
        endif()
      endforeach()
    endforeach()
  endforeach()
endforeach()
