cmake_minimum_required(VERSION 3.19)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

project(laghos CXX)

# build flags
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

if(NOT CMAKE_CONFIGURATION_TYPES)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

find_package(MFEM REQUIRED)
find_package(MPI REQUIRED)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to use.")
set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE BOOL
  "Force the use of the chosen C++ standard.")
set(CMAKE_CXX_EXTENSIONS OFF CACHE BOOL "Enable C++ standard extensions.")

if (MFEM_USE_CUDA)
  if (MFEM_USE_HIP)
    message(FATAL_ERROR " *** MFEM_USE_HIP cannot be combined with MFEM_USE_CUDA.")
  endif()
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70)
  endif()
  # Use ${CMAKE_CXX_COMPILER} as the cuda host compiler.
  if (NOT CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD} CACHE STRING
    "CUDA standard to use.")
  set(CMAKE_CUDA_STANDARD_REQUIRED ON CACHE BOOL
    "Force the use of the chosen CUDA standard.")
  set(CMAKE_CUDA_EXTENSIONS OFF CACHE BOOL "Enable CUDA standard extensions.")
  set(CUDA_FLAGS "--expt-extended-lambda")
  message(STATUS "Using CUDA architecture: ${CMAKE_CUDA_ARCHITECTURES}")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_FLAGS}")
endif()

if (MFEM_USE_HIP)
  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES gfx942)
  endif()
endif()


list(APPEND SOURCES
  laghos_assembly.cpp laghos.cpp laghos_solver.cpp
)

if (MFEM_USE_CUDA)
  set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CUDA)
endif()

add_executable(laghos ${SOURCES})

target_link_libraries(laghos
  PUBLIC mfem
  PUBLIC MPI::MPI_CXX
)
