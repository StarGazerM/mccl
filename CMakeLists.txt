cmake_minimum_required(VERSION 3.27)
project(mccl LANGUAGES C CXX)

if(NOT APPLE OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
  message(FATAL_ERROR "mccl requires Apple silicon and the macOS Metal SDK")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# Configure with -B build/native. Keep dylibs and their Metal resources together,
# including for multi-configuration generators.
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
foreach(config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${config} "${CMAKE_BINARY_DIR}")
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${config} "${CMAKE_BINARY_DIR}")
endforeach()
set(CMAKE_BUILD_RPATH "@loader_path")
set(CMAKE_INSTALL_RPATH "@loader_path")

set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared MLX libraries" FORCE)
set(MLX_C_BUILD_EXAMPLES OFF CACHE BOOL "Disable mlx-c examples" FORCE)
set(MLX_C_USE_SYSTEM_MLX OFF CACHE BOOL "Use the pinned MLX dependency" FORCE)
set(MLX_BUILD_TESTS OFF CACHE BOOL "Disable MLX tests" FORCE)
set(MLX_BUILD_EXAMPLES OFF CACHE BOOL "Disable MLX examples" FORCE)
set(MLX_BUILD_BENCHMARKS OFF CACHE BOOL "Disable MLX benchmarks" FORCE)
set(MLX_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "No Python runtime dependency" FORCE)
set(MLX_BUILD_METAL ON CACHE BOOL "Require the Metal backend" FORCE)
set(MLX_BUILD_CUDA OFF CACHE BOOL "No CUDA dependency" FORCE)
set(MLX_BUILD_GGUF OFF CACHE BOOL "No GGUF dependency" FORCE)
set(MLX_BUILD_SAFETENSORS OFF CACHE BOOL "No safetensors dependency" FORCE)
set(MLX_METAL_JIT OFF CACHE BOOL "Compile Metal libraries offline" FORCE)
set(MLX_METAL_PATH "${CMAKE_BINARY_DIR}" CACHE PATH "MLX metallib location" FORCE)

option(MCCL_PROFILE "Keep optimized native symbols and embedded Metal sources" OFF)
if(MCCL_PROFILE)
  add_compile_options(-g)
endif()

include(FetchContent)
# First declaration wins: pin the exact commit behind mlx-c's MLX v0.32.2.
FetchContent_Declare(mlx
  GIT_REPOSITORY https://github.com/ml-explore/mlx.git
  GIT_TAG 1f8e74e3f12f31365464a6867c6579f0e9b29d85)
FetchContent_Declare(mlx_c
  GIT_REPOSITORY https://github.com/ml-explore/mlx-c.git
  GIT_TAG a341b4925024b88b2c593468f16e12f5e17315da)
FetchContent_MakeAvailable(mlx_c)

find_program(XCRUN_EXECUTABLE xcrun REQUIRED)
set(MCCL_METAL_AIR "${CMAKE_BINARY_DIR}/mccl.air")
set(MCCL_METALLIB "${CMAKE_BINARY_DIR}/mccl.metallib")
set(MCCL_METAL_FLAGS -std=metal3.1 -O3)
if(MCCL_PROFILE)
  list(APPEND MCCL_METAL_FLAGS -gline-tables-only -frecord-sources)
endif()
if(CMAKE_OSX_DEPLOYMENT_TARGET)
  list(APPEND MCCL_METAL_FLAGS
    "-mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}")
endif()
add_custom_command(
  OUTPUT "${MCCL_METALLIB}"
  BYPRODUCTS "${MCCL_METAL_AIR}"
  COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metal ${MCCL_METAL_FLAGS}
    -c "${CMAKE_CURRENT_SOURCE_DIR}/native/merge.metal" -o "${MCCL_METAL_AIR}"
  COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metallib
    "${MCCL_METAL_AIR}" -o "${MCCL_METALLIB}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/native/merge.metal"
  VERBATIM)
add_custom_target(mccl_metallib DEPENDS "${MCCL_METALLIB}")

add_library(mccl_native SHARED native/mccl.cpp)
target_include_directories(mccl_native PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(mccl_native PRIVATE mlxc mlx)
set_target_properties(mccl_native PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "@loader_path"
  INSTALL_RPATH "@loader_path"
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)
add_dependencies(mccl_native mccl_metallib)
