#include "cuda_elementwise_add.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

// Helper function to check CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// CPU reference implementation
void cpu_elementwise_add(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }
}

// Verify correctness
bool verify_results(const float* gpu_result, const float* cpu_result, int N, float tolerance = 1e-5f) {
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        float diff = std::abs(gpu_result[i] - cpu_result[i]);
        if (diff > tolerance) {
            if (errors < 10) {  // Print first 10 errors
                std::cerr << "Mismatch at index " << i << ": GPU=" << gpu_result[i] 
                          << ", CPU=" << cpu_result[i] << ", diff=" << diff << std::endl;
            }
            errors++;
        }
    }
    
    if (errors > 0) {
        std::cerr << "Total errors: " << errors << " / " << N << std::endl;
        return false;
    }
    return true;
}

// Benchmark function
float benchmark_kernel(const float* d_A, const float* d_B, float* d_C, int N, int warmup_runs, int measurement_runs) {
    // Warmup
    for (int i = 0; i < warmup_runs; ++i) {
        baseline::elementwise_add(d_A, d_B, d_C, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Measurement
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < measurement_runs; ++i) {
        baseline::elementwise_add(d_A, d_B, d_C, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float total_ms;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return total_ms / measurement_runs;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "CUDA Baseline Element-wise Add Test" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Test parameters
    const int warmup_runs = 10;
    const int measurement_runs = 100;
    std::vector<int> test_sizes = {1024, 4096, 16384, 65536, 262144, 1048576};
    
    // Random number generator
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    bool all_tests_passed = true;
    
    for (int N : test_sizes) {
        std::cout << "\nTesting N = " << N << " elements..." << std::endl;
        
        // Allocate host memory
        std::vector<float> h_A(N), h_B(N), h_C_gpu(N), h_C_cpu(N);
        
        // Initialize with random data
        for (int i = 0; i < N; ++i) {
            h_A[i] = dist(gen);
            h_B[i] = dist(gen);
        }
        
        // Allocate device memory
        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C, N * sizeof(float)));
        
        // Copy data to device
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        
        // Run GPU kernel
        baseline::elementwise_add(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy result back
        CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, N * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Run CPU reference
        cpu_elementwise_add(h_A.data(), h_B.data(), h_C_cpu.data(), N);
        
        // Verify correctness
        std::cout << "  Correctness: ";
        if (verify_results(h_C_gpu.data(), h_C_cpu.data(), N)) {
            std::cout << "PASSED ✓" << std::endl;
        } else {
            std::cout << "FAILED ✗" << std::endl;
            all_tests_passed = false;
        }
        
        // Benchmark performance
        float latency_ms = benchmark_kernel(d_A, d_B, d_C, N, warmup_runs, measurement_runs);
        
        // Calculate bandwidth (3 arrays: 2 reads + 1 write)
        double bytes_transferred = 3.0 * N * sizeof(float);
        double bandwidth_gbps = (bytes_transferred / 1e9) / (latency_ms / 1e3);
        
        std::cout << "  Performance:" << std::endl;
        std::cout << "    Latency:    " << latency_ms << " ms" << std::endl;
        std::cout << "    Bandwidth:  " << bandwidth_gbps << " GB/s" << std::endl;
        
        // Cleanup
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
    }
    
    std::cout << "\n========================================" << std::endl;
    if (all_tests_passed) {
        std::cout << "All tests PASSED ✓" << std::endl;
        std::cout << "========================================" << std::endl;
        return 0;
    } else {
        std::cout << "Some tests FAILED ✗" << std::endl;
        std::cout << "========================================" << std::endl;
        return 1;
    }
}
