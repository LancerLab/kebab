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
)
target_include_directories(kebab_cuda PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(kebab_cuda CUDA::cublas)
set_target_properties(kebab_cuda PROPERTIES CUDA_ARCHITECTURES OFF)

# CuTe backend - all CuTe-based implementations
add_library(kebab_cute
    cute/gemm_wgmma.cu
    cute/gemm_wgmma_tma.cu
    cute/gemm.cu
    cute/elementwise_add.cu
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
target_link_libraries(kebab CUDA::cublas yaml-cpp)

# Benchmarks
add_subdirectory(benchmark)

# Examples
add_subdirectory(examples)

# Tools
add_subdirectory(tools)