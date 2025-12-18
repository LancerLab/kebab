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

# Copy GMEM to SMEM with 2D tiling using cp.async microbenchmark (requires SM_80+)
add_executable(mbench_copy_gmem_to_smem_2d_cp_async
    copy_gmem_to_smem_2d_cp_async.cu
)
target_include_directories(mbench_copy_gmem_to_smem_2d_cp_async PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(mbench_copy_gmem_to_smem_2d_cp_async microbench_common CUDA::cudart)
set_target_properties(mbench_copy_gmem_to_smem_2d_cp_async PROPERTIES CUDA_ARCHITECTURES 90)
target_compile_options(mbench_copy_gmem_to_smem_2d_cp_async PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_90>)

# Copy GMEM to SMEM with 2D tiling using TMA with CuTe API microbenchmark
add_executable(mbench_copy_gmem_to_smem_2d_tma_cute
    copy_gmem_to_smem_2d_tma_cute.cu
)
target_include_directories(mbench_copy_gmem_to_smem_2d_tma_cute PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(mbench_copy_gmem_to_smem_2d_tma_cute microbench_common CUDA::cudart CUDA::cuda_driver)
set_target_properties(mbench_copy_gmem_to_smem_2d_tma_cute PROPERTIES CUDA_ARCHITECTURES 90)
target_compile_options(mbench_copy_gmem_to_smem_2d_tma_cute PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_90>)

# Copy GMEM to SMEM with 2D tiling using TMA with CDE API microbenchmark
add_executable(mbench_copy_gmem_to_smem_2d_tma_cde
    copy_gmem_to_smem_2d_tma_cde.cu
)
target_include_directories(mbench_copy_gmem_to_smem_2d_tma_cde PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(mbench_copy_gmem_to_smem_2d_tma_cde microbench_common CUDA::cudart CUDA::cuda_driver)
set_target_properties(mbench_copy_gmem_to_smem_2d_tma_cde PROPERTIES CUDA_ARCHITECTURES 90)
target_compile_options(mbench_copy_gmem_to_smem_2d_tma_cde PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_90>)

# Copy GMEM to SMEM with 2D tiling using TMA with PTX microbenchmark
add_executable(mbench_copy_gmem_to_smem_2d_tma_ptx
    copy_gmem_to_smem_2d_tma_ptx.cu
)
target_include_directories(mbench_copy_gmem_to_smem_2d_tma_ptx PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(mbench_copy_gmem_to_smem_2d_tma_ptx microbench_common CUDA::cudart CUDA::cuda_driver)
set_target_properties(mbench_copy_gmem_to_smem_2d_tma_ptx PROPERTIES CUDA_ARCHITECTURES 90)
target_compile_options(mbench_copy_gmem_to_smem_2d_tma_ptx PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_90>)

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

# GMMA Descriptor Info microbenchmark
add_executable(mbench_wgmma_descriptor_info
    wgmma_descriptor_info.cu
)
target_include_directories(mbench_wgmma_descriptor_info PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(mbench_wgmma_descriptor_info microbench_common CUDA::cudart)
set_target_properties(mbench_wgmma_descriptor_info PROPERTIES CUDA_ARCHITECTURES 90)
target_compile_options(mbench_wgmma_descriptor_info PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_90 --expt-relaxed-constexpr>)

# Simplified HGEMM kernel (no pipelining)
add_executable(mbench_hgemm
    hgemm_wgmma.cu
)
target_include_directories(mbench_hgemm PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(mbench_hgemm microbench_common kebab_cute CUDA::cudart)
set_target_properties(mbench_hgemm PROPERTIES CUDA_ARCHITECTURES 90a)
target_compile_options(mbench_hgemm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_90a --expt-relaxed-constexpr>)
