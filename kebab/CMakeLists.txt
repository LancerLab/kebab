cmake_minimum_required(VERSION 3.18)

# Set CUDA compiler path from environment or default
# Prefer CUDA 12.x over 13.x for better sm_90a support
if(DEFINED CUDA_PATH)
    set(CMAKE_CUDA_COMPILER "${CUDA_PATH}/bin/nvcc")
else()
    # Try to find CUDA 12.x first (better sm_90a support)
    if(EXISTS "/usr/local/cuda-12.8/bin/nvcc")
        set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.8/bin/nvcc")
        message(STATUS "Using CUDA 12.8 for better sm_90a support")
    elseif(EXISTS "/usr/local/cuda-12/bin/nvcc")
        set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12/bin/nvcc")
        message(STATUS "Using CUDA 12.x for better sm_90a support")
    else()
        set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
    endif()
endif()

project(kebab LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDAToolkit REQUIRED)

# Include directories
include_directories(include)
include_directories(${CMAKE_SOURCE_DIR}/../third_party/cute/include)
include_directories(${CMAKE_SOURCE_DIR}/../third_party/yaml-cpp/include)

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -lineinfo")

# Set GPU architecture from environment or auto-detect
if(DEFINED CUDA_ARCH)
    # For sm_90a, we need to explicitly set -arch=sm_90a in CUDA_FLAGS
    # CMAKE_CUDA_ARCHITECTURES doesn't properly handle the 'a' suffix
    if(CUDA_ARCH MATCHES "sm_90a")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_90a")
        add_compile_definitions(__CUDA_ARCH_FEAT_SM90_ALL)
        # Disable CMAKE_CUDA_ARCHITECTURES to avoid conflicts
        set(CMAKE_CUDA_ARCHITECTURES OFF)
        message(STATUS "Enabling SM90a with WGMMA support: -arch=sm_90a, -D__CUDA_ARCH_FEAT_SM90_ALL")
    else()
        string(REGEX REPLACE "sm_([0-9]+)" "\\1" ARCH_NUM "${CUDA_ARCH}")
        set(CMAKE_CUDA_ARCHITECTURES "${ARCH_NUM}")
        message(STATUS "Setting CUDA architecture: ${ARCH_NUM}")
    endif()
elseif(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Default to SM90a with WGMMA support
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_90a")
    add_compile_definitions(__CUDA_ARCH_FEAT_SM90_ALL)
    set(CMAKE_CUDA_ARCHITECTURES OFF)
    message(STATUS "Defaulting to SM90a with WGMMA support: -arch=sm_90a, -D__CUDA_ARCH_FEAT_SM90_ALL")
endif()

# Add subdirectories
add_subdirectory(lib)