cmake_minimum_required(VERSION 3.20)

# Custom TensorRT plugin kernels.
#
# This file owns only the shared toolchain setup (CUDA, TensorRT, CUTLASS) and
# then defers to each kernel's own recipe. A kernel declares its own sources and
# dependencies in <kernel>/kernel.cmake, next to the code they describe, so
# adding or changing one never means editing a global list here. The same facts
# are mirrored in Python by kernels/registry.py, and test_kernel_registry.py
# checks the two against the sources on disk.
#
# This directory is <repo>/foldquant/kernels/tensorrt; paths to the vendored
# third_party/ tree are three levels up.

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # The loader derives this from the running device; 87 (Orin) is the default
    # only for a bare invocation.
    set(CMAKE_CUDA_ARCHITECTURES 87)
endif()

project(foldquant_trt_plugins LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --use_fast_math")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -lineinfo")

# --- CUDA toolkit (portable: no hard-pinned JetPack paths) ---
find_package(CUDAToolkit REQUIRED)

# --- TensorRT (override via -DTENSORRT_ROOT or TENSORRT_ROOT env) ---
# Default HINTS cover the JetPack aarch64 layout and common x86 installs.
find_library(NVINFER_LIB nvinfer
    HINTS ${TENSORRT_ROOT}/lib $ENV{TENSORRT_ROOT}/lib
          /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu
    REQUIRED)
find_path(TRT_INCLUDE NvInferRuntime.h
    HINTS ${TENSORRT_ROOT}/include $ENV{TENSORRT_ROOT}/include
          /usr/include/aarch64-linux-gnu /usr/include/x86_64-linux-gnu /usr/include
    REQUIRED)

# --- CUTLASS include path ---
# Override via -DCUTLASS_INCLUDE_DIR=<path>. Default: the pinned git submodule
# at <repo>/third_party/cutlass (NVIDIA/cutlass v3.6.0) — header-only, run
# `git submodule update --init third_party/cutlass`. When absent, every kernel
# whose kernel.cmake sets NEEDS_CUTLASS is skipped, so a CUTLASS-free build
# still configures and produces the kernels it can.
if(NOT DEFINED CUTLASS_INCLUDE_DIR)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/cutlass/include)
        set(CUTLASS_INCLUDE_DIR
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/cutlass/include)
    else()
        message(WARNING
            "CUTLASS headers not found — kernels that need them will be skipped. Run "
            "`git submodule update --init third_party/cutlass` or set -DCUTLASS_INCLUDE_DIR=<path>.")
        set(CUTLASS_INCLUDE_DIR "")
    endif()
endif()
message(STATUS "CUTLASS_INCLUDE_DIR = ${CUTLASS_INCLUDE_DIR}")

# ---------------------------------------------------------------------------
# Per-kernel recipes.
#
# foldquant_add_kernel() is the one place a kernel target is defined, so every
# kernel gets the same include paths, link libraries and CUTLASS gating without
# restating them. Each kernel.cmake only declares what is specific to it.
#
# Both cuda/ and plugin/ land on the include path: the split is organisational
# (compute vs IPluginV3 wrapper) and same-dir `#include "X.h"` must keep
# resolving across it.
# ---------------------------------------------------------------------------
function(foldquant_add_kernel LIB_NAME)
    cmake_parse_arguments(K "NEEDS_CUTLASS;NEEDS_CUBLAS" "DIR" "SOURCES;EXTRA_INCLUDES" ${ARGN})

    if(K_NEEDS_CUTLASS AND NOT CUTLASS_INCLUDE_DIR)
        message(STATUS "skipping ${LIB_NAME} (no CUTLASS)")
        return()
    endif()

    set(_root ${CMAKE_CURRENT_SOURCE_DIR}/${K_DIR})
    set(_srcs "")
    foreach(s ${K_SOURCES})
        list(APPEND _srcs ${_root}/${s})
    endforeach()

    add_library(${LIB_NAME} SHARED ${_srcs})
    target_include_directories(${LIB_NAME} PRIVATE
        ${_root}/cuda ${_root}/plugin ${CMAKE_CURRENT_SOURCE_DIR} ${TRT_INCLUDE} ${CUTLASS_INCLUDE_DIR} ${K_EXTRA_INCLUDES})

    set(_libs ${NVINFER_LIB} CUDA::cudart)
    if(K_NEEDS_CUBLAS)
        list(APPEND _libs CUDA::cublas)
    endif()
    target_link_libraries(${LIB_NAME} PRIVATE ${_libs})
    set_target_properties(${LIB_NAME} PROPERTIES OUTPUT_NAME ${LIB_NAME})
endfunction()

include(${CMAKE_CURRENT_SOURCE_DIR}/int4_groupwise/kernel.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/int8_per_row/kernel.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/int4_per_row/kernel.cmake)
