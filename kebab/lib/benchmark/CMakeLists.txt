# Benchmark executables

# Benchmark runner library
add_library(benchmark_runner
    benchmark_runner.cpp
)
target_include_directories(benchmark_runner PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(benchmark_runner kebab)

# GEMM benchmark
add_executable(bench_gemm
    bench_gemm.cu
)
target_include_directories(bench_gemm PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(bench_gemm kebab benchmark_runner CUDA::cublas)

# Single-run GEMM kernel for profiling (no loops, no cublas)
add_executable(runonce_gemm_cute
    runonce_gemm.cu
)
target_include_directories(runonce_gemm_cute PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(runonce_gemm_cute kebab benchmark_runner)

# Single-run cuBLAS reference kernel for profiling
add_executable(runonce_gemm_ref
    runonce_gemm_ref.cu
)
target_include_directories(runonce_gemm_ref PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(runonce_gemm_ref kebab benchmark_runner CUDA::cublas)

# Single-run CUDA baseline GEMM kernel for profiling
add_executable(runonce_gemm_cuda
    runonce_gemm_cuda.cu
)
target_include_directories(runonce_gemm_cuda PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(runonce_gemm_cuda kebab benchmark_runner CUDA::cublas)

# Element-wise add benchmark
add_executable(bench_elementwise_add
    bench_elementwise_add.cu
)
target_include_directories(bench_elementwise_add PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(bench_elementwise_add kebab benchmark_runner)

# Single-run elementwise_add kernel for profiling (no loops, no cublas)
add_executable(runonce_elementwise_add
    runonce_elementwise_add.cu
)
target_include_directories(runonce_elementwise_add PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(runonce_elementwise_add kebab benchmark_runner)

# Single-run CUDA baseline elementwise_add reference kernel for profiling
add_executable(runonce_elementwise_add_ref
    runonce_elementwise_add_ref.cu
)
target_include_directories(runonce_elementwise_add_ref PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(runonce_elementwise_add_ref kebab benchmark_runner)
