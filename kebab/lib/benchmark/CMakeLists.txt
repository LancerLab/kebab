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

# Element-wise add benchmark
add_executable(bench_elementwise_add
    bench_elementwise_add.cu
)
target_include_directories(bench_elementwise_add PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(bench_elementwise_add kebab benchmark_runner)