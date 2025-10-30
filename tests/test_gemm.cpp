#include "cutekernellib/operators/gemm.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <tuple>

// CPU reference implementation for GEMM
template<typename T>
void cpu_gemm(const T* A, const T* B, T* C, int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            T sum = T(0);
            for (int k = 0; k < K; ++k) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}

// Convert half to float for comparison
float half_to_float(__half h) {
    return __half2float(h);
}

// Test function for float
bool test_float_gemm(int M, int N, int K) {
    std::cout << "Testing float GEMM with M=" << M << ", N=" << N << ", K=" << K << "..." << std::endl;
    
    // Allocate host memory
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C_gpu(M * N);
    std::vector<float> h_C_cpu(M * N);
    
    // Initialize with random data
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = dist(gen);
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = dist(gen);
    }
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Run GPU kernel
    cutekernellib::gemm(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Run CPU reference
    cpu_gemm(h_A.data(), h_B.data(), h_C_cpu.data(), M, N, K);
    
    // Compare results
    bool passed = true;
    float max_error = 0.0f;
    float max_relative_error = 0.0f;
    int error_count = 0;
    
    for (int i = 0; i < M * N; ++i) {
        float error = std::abs(h_C_gpu[i] - h_C_cpu[i]);
        float relative_error = (h_C_cpu[i] != 0.0f) ? error / std::abs(h_C_cpu[i]) : error;
        
        max_error = std::max(max_error, error);
        max_relative_error = std::max(max_relative_error, relative_error);
        
        // Use relative tolerance for larger values, absolute for smaller ones
        float tolerance = std::max(1e-4f, std::abs(h_C_cpu[i]) * 1e-5f);
        
        if (error > tolerance) {
            if (error_count < 5) {  // Only print first few errors
                std::cout << "  Mismatch at index " << i << ": "
                          << "GPU=" << h_C_gpu[i] << ", CPU=" << h_C_cpu[i]
                          << ", error=" << error << ", rel_error=" << relative_error << std::endl;
            }
            error_count++;
            passed = false;
        }
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    if (passed) {
        std::cout << "  ✓ PASSED (max error: " << max_error 
                  << ", max rel error: " << max_relative_error << ")" << std::endl;
    } else {
        std::cout << "  ✗ FAILED (max error: " << max_error 
                  << ", max rel error: " << max_relative_error 
                  << ", " << error_count << " errors)" << std::endl;
    }
    
    return passed;
}

// Test function for half
bool test_half_gemm(int M, int N, int K) {
    std::cout << "Testing half GEMM with M=" << M << ", N=" << N << ", K=" << K << "..." << std::endl;
    
    // Allocate host memory
    std::vector<__half> h_A(M * K);
    std::vector<__half> h_B(K * N);
    std::vector<__half> h_C_gpu(M * N);
    std::vector<float> h_C_cpu(M * N);
    
    // Initialize with random data
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = __float2half(dist(gen));
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = __float2half(dist(gen));
    }
    
    // Allocate device memory
    __half *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(__half)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(__half), cudaMemcpyHostToDevice));
    
    // Run GPU kernel
    cutekernellib::gemm(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, M * N * sizeof(__half), cudaMemcpyDeviceToHost));
    
    // Run CPU reference (in float for accuracy)
    std::vector<float> h_A_float(M * K), h_B_float(K * N);
    for (int i = 0; i < M * K; ++i) {
        h_A_float[i] = half_to_float(h_A[i]);
    }
    for (int i = 0; i < K * N; ++i) {
        h_B_float[i] = half_to_float(h_B[i]);
    }
    cpu_gemm(h_A_float.data(), h_B_float.data(), h_C_cpu.data(), M, N, K);
    
    // Compare results (allow larger tolerance for half precision)
    bool passed = true;
    float max_error = 0.0f;
    float max_relative_error = 0.0f;
    int error_count = 0;
    
    for (int i = 0; i < M * N; ++i) {
        float gpu_val = half_to_float(h_C_gpu[i]);
        float error = std::abs(gpu_val - h_C_cpu[i]);
        float relative_error = (h_C_cpu[i] != 0.0f) ? error / std::abs(h_C_cpu[i]) : error;
        
        max_error = std::max(max_error, error);
        max_relative_error = std::max(max_relative_error, relative_error);
        
        // Use larger tolerance for half precision (more lenient due to accumulation errors)
        // Larger K means more accumulation, so scale tolerance with K
        float base_tolerance = 2e-2f + (K > 64 ? (K - 64) * 1e-4f : 0.0f);
        float tolerance = std::max(base_tolerance, std::abs(h_C_cpu[i]) * 1e-2f);
        
        if (error > tolerance) {
            if (error_count < 5) {  // Only print first few errors
                std::cout << "  Mismatch at index " << i << ": "
                          << "GPU=" << gpu_val << ", CPU=" << h_C_cpu[i]
                          << ", error=" << error << ", rel_error=" << relative_error << std::endl;
            }
            error_count++;
            passed = false;
        }
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    if (passed) {
        std::cout << "  ✓ PASSED (max error: " << max_error 
                  << ", max rel error: " << max_relative_error << ")" << std::endl;
    } else {
        std::cout << "  ✗ FAILED (max error: " << max_error 
                  << ", max rel error: " << max_relative_error 
                  << ", " << error_count << " errors)" << std::endl;
    }
    
    return passed;
}

// Test edge cases
bool test_edge_cases() {
    std::cout << "Testing edge cases..." << std::endl;
    
    bool all_passed = true;
    
    // Test 1x1x1 matrix
    all_passed &= test_float_gemm(1, 1, 1);
    
    // Test rectangular matrices
    all_passed &= test_float_gemm(1, 10, 5);
    all_passed &= test_float_gemm(10, 1, 5);
    all_passed &= test_float_gemm(5, 10, 1);
    
    return all_passed;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "GEMM Operator Test" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    bool all_passed = true;
    
    // Test various matrix sizes
    std::vector<std::tuple<int, int, int>> test_sizes = {
        {16, 16, 16},    // Small square
        {32, 32, 32},    // Medium square
        {64, 64, 64},    // Large square
        {128, 64, 32},   // Rectangular
        {64, 128, 32},   // Rectangular
        {32, 64, 128},   // Rectangular
        {100, 50, 75},   // Non-power-of-2
    };
    
    std::cout << "Testing float precision:" << std::endl;
    for (auto [M, N, K] : test_sizes) {
        all_passed &= test_float_gemm(M, N, K);
    }
    std::cout << std::endl;
    
    std::cout << "Testing half precision:" << std::endl;
    for (auto [M, N, K] : test_sizes) {
        all_passed &= test_half_gemm(M, N, K);
    }
    std::cout << std::endl;
    
    // Test edge cases
    all_passed &= test_edge_cases();
    std::cout << std::endl;
    
    std::cout << "========================================" << std::endl;
    if (all_passed) {
        std::cout << "All tests PASSED ✓" << std::endl;
    } else {
        std::cout << "Some tests FAILED ✗" << std::endl;
    }
    std::cout << "========================================" << std::endl;
    
    return all_passed ? 0 : 1;
}