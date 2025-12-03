# Microbenchmark executables

# Microbench common library
add_library(microbench_common
    microbench.cpp
)
target_include_directories(microbench_common PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(microbench_common CUDA::cudart)

# Copy GMEM to SMEM microbenchmark
add_executable(mbench_copy_gmem_to_smem
    copy_gmem_to_smem.cu
)
target_include_directories(mbench_copy_gmem_to_smem PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(mbench_copy_gmem_to_smem microbench_common CUDA::cudart)
set_target_properties(mbench_copy_gmem_to_smem PROPERTIES CUDA_ARCHITECTURES OFF)

# WGMMA MMA microbenchmark
add_executable(mbench_mma_wgmma
    mma_wgmma.cu
)
target_include_directories(mbench_mma_wgmma PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(mbench_mma_wgmma microbench_common CUDA::cudart)
set_target_properties(mbench_mma_wgmma PROPERTIES CUDA_ARCHITECTURES OFF)

