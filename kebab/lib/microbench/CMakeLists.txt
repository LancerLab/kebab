# Microbenchmark executables

# Microbench common library
add_library(microbench_common
    microbench.cpp
)
target_include_directories(microbench_common PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(microbench_common CUDA::cudart)

# Copy GMEM to SMEM microbenchmark (1D linear copy)
add_executable(mbench_copy_gmem_to_smem
    copy_gmem_to_smem.cu
)
target_include_directories(mbench_copy_gmem_to_smem PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(mbench_copy_gmem_to_smem microbench_common CUDA::cudart)
set_target_properties(mbench_copy_gmem_to_smem PROPERTIES CUDA_ARCHITECTURES OFF)

# Copy GMEM to SMEM with 2D tiling microbenchmark
add_executable(mbench_copy_gmem_to_smem_2d_tiling
    copy_gmem_to_smem_2d_tiling.cu
)
target_include_directories(mbench_copy_gmem_to_smem_2d_tiling PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(mbench_copy_gmem_to_smem_2d_tiling microbench_common CUDA::cudart)
set_target_properties(mbench_copy_gmem_to_smem_2d_tiling PROPERTIES CUDA_ARCHITECTURES OFF)

# WGMMA MMA microbenchmark
add_executable(mbench_mma_wgmma
    mma_wgmma.cu
)
target_include_directories(mbench_mma_wgmma PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(mbench_mma_wgmma microbench_common CUDA::cudart)
set_target_properties(mbench_mma_wgmma PROPERTIES CUDA_ARCHITECTURES OFF)

# MMA.SYNC microbenchmark
add_executable(mbench_mma_mma_sync
    mma_mma_sync.cu
)
target_include_directories(mbench_mma_mma_sync PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(mbench_mma_mma_sync microbench_common CUDA::cudart)
set_target_properties(mbench_mma_mma_sync PROPERTIES CUDA_ARCHITECTURES 80)
target_compile_options(mbench_mma_mma_sync PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_80>)

# WMMA.SYNC microbenchmark
add_executable(mbench_mma_wmma_sync
    mma_wmma_sync.cu
)
target_include_directories(mbench_mma_wmma_sync PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(mbench_mma_wmma_sync microbench_common CUDA::cudart)
set_target_properties(mbench_mma_wmma_sync PROPERTIES CUDA_ARCHITECTURES 80)
target_compile_options(mbench_mma_wmma_sync PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_80>)
