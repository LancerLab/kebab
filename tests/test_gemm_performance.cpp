#include "cutekernellib/operators/gemm.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <tuple>

// Simple performance test for GEMM
template<typename T>
void benchmark_gemm(int M, int N, int K, int num_runs = 100) {
    std::cout << "Benchmarking GEMM " << (sizeof(T) == 4 ? "float" : "half") 
              << " with M=" << M << ", N=" << N << ", K=" << K << "..." << std::endl;
    
    // Allocate host memory
    std::vector<T> h_A(M * K);
    std::vector<T> h_B(K * N);
    std::vector<T> h_C(M * N);
    
    // Initialize with random data
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < M * K; ++i) {
        if constexpr (std::is_same_v<T, float>) {
            h_A[i] = dist(gen);
        } else {
            h_A[i] = __float2half(dist(gen));
        }
    }
    for (int i = 0; i < K * N; ++i) {
        if constexpr (std::is_same_v<T, float>) {
            h_B[i] = dist(gen);
        } else {
            h_B[i] = __float2half(dist(gen));
        }
    }
    
    // Allocate device memory
    T *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(T)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(T), cudaMemcpyHostToDevice));
    
    // Warmup runs
    for (int i = 0; i < 10; ++i) {
        cutekernellib::gemm(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark runs
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_runs; ++i) {
        cutekernellib::gemm(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    // Calculate performance metrics
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg_time_ms = duration.count() / (1000.0 * num_runs);
    
    // Calculate FLOPS (2 * M * N * K operations per GEMM)
    double flops = 2.0 * M * N * K;
    double gflops = (flops / (avg_time_ms * 1e-3)) / 1e9;
    
    // Calculate memory bandwidth
    double bytes_transferred = (M * K + K * N + M * N) * sizeof(T);
    double bandwidth_gb_s = (bytes_transferred / (avg_time_ms * 1e-3)) / 1e9;
    
    std::cout << "  Average time: " << avg_time_ms << " ms" << std::endl;
    std::cout << "  Performance: " << gflops << " GFLOPS" << std::endl;
    std::cout << "  Bandwidth: " << bandwidth_gb_s << " GB/s" << std::endl;
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "GEMM Performance Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    // Test various matrix sizes
    std::vector<std::tuple<int, int, int>> sizes = {
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
        {2048, 2048, 2048}
    };
    
    std::cout << "Float precision benchmarks:" << std::endl;
    for (auto [M, N, K] : sizes) {
        benchmark_gemm<float>(M, N, K);
        std::cout << std::endl;
    }
    
    std::cout << "Half precision benchmarks:" << std::endl;
    for (auto [M, N, K] : sizes) {
        benchmark_gemm<__half>(M, N, K);
        std::cout << std::endl;
    }
    
    std::cout << "========================================" << std::endl;
    std::cout << "Benchmark completed!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}