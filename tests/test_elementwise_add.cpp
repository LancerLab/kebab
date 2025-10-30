#include "cutekernellib/operators/elementwise_add.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

// CPU reference implementation
template<typename T>
void cpu_elementwise_add(const T* A, const T* B, T* C, int N) {
    for (int i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }
}

// Convert half to float for comparison
float half_to_float(__half h) {
    return __half2float(h);
}

// Test function for float
bool test_float(int N) {
    std::cout << "Testing float with N=" << N << "..." << std::endl;
    
    // Allocate host memory
    std::vector<float> h_A(N);
    std::vector<float> h_B(N);
    std::vector<float> h_C_gpu(N);
    std::vector<float> h_C_cpu(N);
    
    // Initialize with random data
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
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
    cutekernellib::elementwise_add(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Run CPU reference
    cpu_elementwise_add(h_A.data(), h_B.data(), h_C_cpu.data(), N);
    
    // Compare results
    bool passed = true;
    float max_error = 0.0f;
    for (int i = 0; i < N; ++i) {
        float error = std::abs(h_C_gpu[i] - h_C_cpu[i]);
        max_error = std::max(max_error, error);
        if (error > 1e-5f) {
            if (passed) {
                std::cout << "  First mismatch at index " << i << ": "
                          << "GPU=" << h_C_gpu[i] << ", CPU=" << h_C_cpu[i]
                          << ", error=" << error << std::endl;
            }
            passed = false;
        }
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    if (passed) {
        std::cout << "  ✓ PASSED (max error: " << max_error << ")" << std::endl;
    } else {
        std::cout << "  ✗ FAILED (max error: " << max_error << ")" << std::endl;
    }
    
    return passed;
}

// Test function for half
bool test_half(int N) {
    std::cout << "Testing half with N=" << N << "..." << std::endl;
    
    // Allocate host memory
    std::vector<__half> h_A(N);
    std::vector<__half> h_B(N);
    std::vector<__half> h_C_gpu(N);
    std::vector<float> h_C_cpu(N);
    
    // Initialize with random data
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (int i = 0; i < N; ++i) {
        h_A[i] = __float2half(dist(gen));
        h_B[i] = __float2half(dist(gen));
    }
    
    // Allocate device memory
    __half *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_B, N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_C, N * sizeof(__half)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), N * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), N * sizeof(__half), cudaMemcpyHostToDevice));
    
    // Run GPU kernel
    cutekernellib::elementwise_add(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, N * sizeof(__half), cudaMemcpyDeviceToHost));
    
    // Run CPU reference (in float for accuracy)
    std::vector<float> h_A_float(N), h_B_float(N);
    for (int i = 0; i < N; ++i) {
        h_A_float[i] = half_to_float(h_A[i]);
        h_B_float[i] = half_to_float(h_B[i]);
    }
    cpu_elementwise_add(h_A_float.data(), h_B_float.data(), h_C_cpu.data(), N);
    
    // Compare results (allow larger tolerance for half precision)
    bool passed = true;
    float max_error = 0.0f;
    for (int i = 0; i < N; ++i) {
        float gpu_val = half_to_float(h_C_gpu[i]);
        float error = std::abs(gpu_val - h_C_cpu[i]);
        max_error = std::max(max_error, error);
        if (error > 1e-2f) {  // Larger tolerance for half precision
            if (passed) {
                std::cout << "  First mismatch at index " << i << ": "
                          << "GPU=" << gpu_val << ", CPU=" << h_C_cpu[i]
                          << ", error=" << error << std::endl;
            }
            passed = false;
        }
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    if (passed) {
        std::cout << "  ✓ PASSED (max error: " << max_error << ")" << std::endl;
    } else {
        std::cout << "  ✗ FAILED (max error: " << max_error << ")" << std::endl;
    }
    
    return passed;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Element-wise Add Operator Test" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    bool all_passed = true;
    
    // Test various sizes including edge cases
    std::vector<int> test_sizes = {1, 7, 16, 100, 256, 1000, 1024, 4096, 10000};
    
    std::cout << "Testing float precision:" << std::endl;
    for (int N : test_sizes) {
        all_passed &= test_float(N);
    }
    std::cout << std::endl;
    
    std::cout << "Testing half precision:" << std::endl;
    for (int N : test_sizes) {
        all_passed &= test_half(N);
    }
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
