# Common library
add_library(kebab_common
    common/config_parser.cpp
)
target_include_directories(kebab_common PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(kebab_common yaml-cpp)

# CUDA backend
add_library(kebab_cuda
    cuda/cuda_elementwise_add.cu
    cuda/cuda_gemm.cu
    cuda/cuda_gemm_v1_warptiling.cu
    cuda/cuda_gemm_v2_wgmma_tma.cu
    cuda/cuda_gemm_v3_warpgroup.cu
    cuda/cuda_gemm_v4_warpspec.cu
    cuda/cuda_gemm_v5_persistent.cu
    cuda/cuda_gemm_v6_ptxbarrier.cu
    cuda/cuda_gemm_v7_cluster.cu
    cuda/cuda_gemm_v8_multicast.cu
    cuda/cuda_gemm_v9_streamstore.cu
    cuda/cuda_gemm_v10_tmastore.cu
    cuda/cuda_gemm_v11_hilbert.cu
    cuda/cuda_gemm_v12_stmatrix.cu
)
target_include_directories(kebab_cuda PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(kebab_cuda CUDA::cublas CUDA::cuda_driver)
set_target_properties(kebab_cuda PROPERTIES CUDA_ARCHITECTURES OFF)

# CuTe backend - all CuTe-based implementations
add_library(kebab_cute
    cute/cute_gemm_wgmma.cu
    cute/cute_gemm_wgmma_tma.cu
    cute/cute_gemm_wgmma_tma_v3.cu
    cute/cute_gemm.cu
    cute/cute_elementwise_add.cu
)
target_include_directories(kebab_cute PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(kebab_cute CUDA::cublas)
# Disable CMAKE_CUDA_ARCHITECTURES for this target to use -arch from CMAKE_CUDA_FLAGS
set_target_properties(kebab_cute PROPERTIES CUDA_ARCHITECTURES OFF)

# Main library
add_library(kebab
    $<TARGET_OBJECTS:kebab_common>
    $<TARGET_OBJECTS:kebab_cuda>
    $<TARGET_OBJECTS:kebab_cute>
)
target_include_directories(kebab PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(kebab CUDA::cublas CUDA::cuda_driver yaml-cpp)

# Benchmarks
add_subdirectory(benchmark)

# Microbenchmarks
add_subdirectory(microbench)

# Examples
add_subdirectory(examples)

# Tools
add_subdirectory(tools)