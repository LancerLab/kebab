/**
 * @file mma_mma_sync.cu
 * @brief Microbenchmark for MMA.SYNC operations (m16n8k16, m16n8k8)
 *
 * Benchmarks MMA.SYNC tensor core operations with FP16 data type and FP32 accumulator.
 * Supports matrix sizes: m16n8k16 and m16n8k8.
 * Following the pattern of mma_wgmma.cu
 */

#include "microbench/microbench.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstring>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <type_traits>

using namespace kebab::microbench;

// ============================================================================
// MMA.SYNC Wrapper Functions for Inner Device Kernel
// ============================================================================

// Simple MMA m16n8k16 for FP16 (produces FP32 accumulator)
// This performs a single MMA operation per warp
__device__ void mma_m16n8k16_f16f32_kernel(float *d, const half *a,
                                           const half *b, const float *c) {
    int lane_id = threadIdx.x % 32;

    unsigned A[4];
    unsigned B[2];
    float C[4];
    float D[4];

    // Load A matrix using ldmatrix
    // For m16n8k16, we need 4 registers per thread (16x16 matrix)
    // ldmatrix loads 8x8 blocks, so we need x4 for 16x16
    uint32_t a_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(a));

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(A[0]), "=r"(A[1]), "=r"(A[2]), "=r"(A[3])
        : "r"(a_ptr));

    // Load B matrix (16x8 matrix)
    uint32_t b_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(b));

    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
                 : "=r"(B[0]), "=r"(B[1])
                 : "r"(b_ptr));

    // Load C matrix (16x8 matrix)
    int c_idx = (lane_id / 4) * 8 + (lane_id % 4) * 2;
    C[0] = c[c_idx];
    C[1] = c[c_idx + 1];
    C[2] = c[c_idx + 64];
    C[3] = c[c_idx + 65];

    // Perform MMA operation
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 " { %0, %1, %2, %3 }, "
                 " { %4, %5, %6, %7 }, "
                 " { %8, %9 }, "
                 " { %10, %11, %12, %13 };"
                 : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
                 : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]),
                   "r"(B[1]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));

    // Store result
    d[c_idx] = D[0];
    d[c_idx + 1] = D[1];
    d[c_idx + 64] = D[2];
    d[c_idx + 65] = D[3];
}

// Simple MMA m16n8k8 for FP16 (produces FP32 accumulator)
__device__ void mma_m16n8k8_f16f32_kernel(float *d, const half *a,
                                          const half *b, const float *c) {
    int lane_id = threadIdx.x % 32;

    unsigned A[2];
    unsigned B[1];
    float C[4];
    float D[4];

    // Load A matrix (16x8 matrix) - need to load two 8x8 blocks
    // First block at offset 0
    int lorr_id = lane_id % 16;
    const half *a_new = a + lorr_id * 8;
    uint32_t a_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(a_new));

    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                 : "=r"(A[0]), "=r"(A[1])
                 : "r"(a_ptr));

    // Load B matrix (8x8 matrix)
    int uord_id = lane_id % 8;
    const half *b_new = b + uord_id * 8;
    uint32_t b_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(b_new));

    asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];"
                 : "=r"(B[0])
                 : "r"(b_ptr));

    // Load C matrix (16x8 matrix)
    int c_idx = (lane_id / 4) * 8 + (lane_id % 4) * 2;
    C[0] = c[c_idx];
    C[1] = c[c_idx + 1];
    C[2] = c[c_idx + 64];
    C[3] = c[c_idx + 65];

    // Perform MMA operation
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                 " { %0, %1, %2, %3 }, "
                 " { %4, %5 }, "
                 " { %6 }, "
                 " { %7, %8, %9, %10 };"
                 : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
                 : "r"(A[0]), "r"(A[1]), "r"(B[0]), "f"(C[0]), "f"(C[1]),
                   "f"(C[2]), "f"(C[3]));

    // Store result
    d[c_idx] = D[0];
    d[c_idx + 1] = D[1];
    d[c_idx + 64] = D[2];
    d[c_idx + 65] = D[3];
}



// ============================================================================
// Device Kernels Outer
// ============================================================================

// For sake of convenience, we do not consider tiling and only use 1 warp
// In real case, we need to consider tiling by warps to fit mma contract
// M = 16, N = 8, K = 16
__global__ void mma_m16n8k16_f16f32(half* gA, half* gB, float* gC, float* gD, int iteration=1) {
    __shared__ alignas(128) half sA[16 * 16];
    __shared__ alignas(128) half sB[16 * 8];
    __shared__ alignas(128) float sC[16 * 8];
    __shared__ alignas(128) float sD[16 * 8];

    int tid = threadIdx.x;

    // Load data to shared memory
    // when copy data from global to shared, we do not use warpgroup
    // instead, we use thread, each thread copy total elements / blockDim.x
    for (int i = tid; i < 16 * 16; i += 16 * 16 / blockDim.x) sA[i] = gA[i];
    for (int i = tid; i < 16 * 8; i += 16 * 8 / blockDim.x) sB[i] = gB[i];
    for (int i = tid; i < 16 * 8; i += 16 * 8 / blockDim.x) sC[i] = gC[i];
    __syncthreads();

    // Initialize output
    for (int i = tid; i < 16 * 8; i += 16 * 8 / blockDim.x) sD[i] = sC[i];
    __syncthreads();

    // Perform MMA operation
    for (int i = 0; i < iteration; ++i) {
        mma_m16n8k16_f16f32_kernel(sD, sA, sB, sC);
    }

    // Copy result to output
    for (int i = tid; i < 16 * 8; i += 16 * 8 / blockDim.x) gD[i] = sD[i];
}

__global__ void mma_m16n8k8_f16f32(half* gA, half* gB, float* gC, float* gD, int iteration=1) {
    __shared__ alignas(128) half sA[16 * 8];
    __shared__ alignas(128) half sB[8 * 8];
    __shared__ alignas(128) float sC[16 * 8];
    __shared__ alignas(128) float sD[16 * 8];

    int tid = threadIdx.x;

    // Load data to shared memory
    for (int i = tid; i < 16 * 8; i += 16 * 8 / blockDim.x) sA[i] = gA[i];
    for (int i = tid; i < 8 * 8; i += 8 * 8 / blockDim.x) sB[i] = gB[i];
    for (int i = tid; i < 16 * 8; i += 16 * 8 / blockDim.x) sC[i] = gC[i];
    __syncthreads();

    // Initialize output
    for (int i = tid; i < 16 * 8; i += 16 * 8 / blockDim.x) sD[i] = sC[i];
    __syncthreads();

    // Perform MMA operation
    for (int i = 0; i < iteration; ++i) {
        mma_m16n8k8_f16f32_kernel(sD, sA, sB, sC);
    }

    // Copy result to output
    for (int i = tid; i < 16 * 8; i += 16 * 8 / blockDim.x) gD[i] = sD[i];
}

// ============================================================================
// Verification and Benchmarking
// ============================================================================

bool verifyOutput(const std::vector<float>& gpu_output, float expected, float tolerance = 0.5f) {
    std::cout << "Verification: checking " << gpu_output.size() << " outputs\n";
    int fail_count = 0;
    for (size_t i = 0; i < gpu_output.size(); ++i) {
        float diff = std::abs(gpu_output[i] - expected);
        if (diff > tolerance) {
            if (fail_count < 10) {  // Print first 10 failures
                std::cerr << "FAILED at [" << i << "]: expected " << expected
                          << ", got " << gpu_output[i] << "\n";
            }
            fail_count++;
        }
    }
    if (fail_count > 0) {
        std::cerr << "Total failures: " << fail_count << " out of " << gpu_output.size() << "\n";
        return false;
    }
    std::cout << "PASSED: all outputs match expected value " << expected << "\n";
    return true;
}

bool run_mma_m16n8k16_f16f32(bool bench=false) {
    constexpr int M = 16, N = 8, K = 16;
    constexpr int NUM_THREADS = 32;
    constexpr int NUM_BLOCKS = 1;
    constexpr int ITERATIONS = 100000;

    std::vector<half> h_A(M * K, __float2half(1.0f));
    std::vector<half> h_B(K * N, __float2half(1.0f));
    std::vector<float> h_C(M * N, 11.0f);
    std::vector<float> h_output(M * N);

    half *d_A, *d_B;
    float *d_C, *d_output;
    MBENCH_CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(half)));
    MBENCH_CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(half)));
    MBENCH_CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    MBENCH_CUDA_CHECK(cudaMalloc(&d_output, M * N * sizeof(float)));

    MBENCH_CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(half), cudaMemcpyHostToDevice));
    MBENCH_CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(half), cudaMemcpyHostToDevice));
    MBENCH_CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));

    if (bench) {
        mma_m16n8k16_f16f32<<<NUM_BLOCKS, NUM_THREADS>>>(d_A, d_B, d_C, d_output, 10);
        MBENCH_CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t start, stop;
        MBENCH_CUDA_CHECK(cudaEventCreate(&start));
        MBENCH_CUDA_CHECK(cudaEventCreate(&stop));

        MBENCH_CUDA_CHECK(cudaEventRecord(start));
        mma_m16n8k16_f16f32<<<NUM_BLOCKS, NUM_THREADS>>>(d_A, d_B, d_C, d_output, ITERATIONS);
        MBENCH_CUDA_CHECK(cudaEventRecord(stop));
        MBENCH_CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        MBENCH_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

        MBENCH_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost));

        double total_ops = 2.0 * M * N * K * ITERATIONS;
        double elapsed_us = elapsed_ms * 1000.0;
        double throughput_tflops = (total_ops / 1e6) / elapsed_us;
        double throughput_gflops = (total_ops / 1e3) / elapsed_us;
        std::cout << "\nBenchmark Results (mma_m16n8k16_f16f32_bench):\n";
        std::cout << "  Iterations: " << ITERATIONS << "\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "  Elapsed time: " << elapsed_ms << " ms\n";
        std::cout << "  Total operations: " << total_ops / 1e9 << " G ops\n";
        std::cout << "  Throughput: " << throughput_tflops << " TFLOPS";
        std::cout << " (" << throughput_gflops << " GFLOPS)\n";
        MBENCH_CUDA_CHECK(cudaEventDestroy(start));
        MBENCH_CUDA_CHECK(cudaEventDestroy(stop));
    } else {
        std::cout << "\nRunning mma_m16n8k16_f16f32_once...\n";
        mma_m16n8k16_f16f32<<<NUM_BLOCKS, NUM_THREADS>>>(d_A, d_B, d_C, d_output);
        MBENCH_CUDA_CHECK(cudaDeviceSynchronize());
        MBENCH_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost));
        bool verified = verifyOutput(h_output, 27.0f);
        return verified;
    }

    MBENCH_CUDA_CHECK(cudaFree(d_A));
    MBENCH_CUDA_CHECK(cudaFree(d_B));
    MBENCH_CUDA_CHECK(cudaFree(d_C));
    MBENCH_CUDA_CHECK(cudaFree(d_output));

    return true;
}

bool run_mma_m16n8k8_f16f32(bool bench=false) {
    constexpr int M = 16, N = 8, K = 8;
    constexpr int NUM_THREADS = 32;
    constexpr int NUM_BLOCKS = 1;
    constexpr int ITERATIONS = 100000;

    std::vector<__half> h_A(M * K, __float2half(1.0f));
    std::vector<__half> h_B(K * N, __float2half(1.0f));
    std::vector<float> h_C(M * N, 4.0f);
    std::vector<float> h_output(M * N);

    __half *d_A, *d_B;
    float *d_C, *d_output;
    MBENCH_CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(half)));
    MBENCH_CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(half)));
    MBENCH_CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    MBENCH_CUDA_CHECK(cudaMalloc(&d_output, M * N * sizeof(float)));

    MBENCH_CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(half), cudaMemcpyHostToDevice));
    MBENCH_CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(half), cudaMemcpyHostToDevice));
    MBENCH_CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));

    if (bench) {
        mma_m16n8k8_f16f32<<<NUM_BLOCKS, NUM_THREADS>>>(d_A, d_B, d_C, d_output, 10);
        MBENCH_CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t start, stop;
        MBENCH_CUDA_CHECK(cudaEventCreate(&start));
        MBENCH_CUDA_CHECK(cudaEventCreate(&stop));

        MBENCH_CUDA_CHECK(cudaEventRecord(start));
        mma_m16n8k8_f16f32<<<NUM_BLOCKS, NUM_THREADS>>>(d_A, d_B, d_C, d_output, ITERATIONS);
        MBENCH_CUDA_CHECK(cudaEventRecord(stop));
        MBENCH_CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        MBENCH_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

        MBENCH_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost));

        double total_ops = 2.0 * M * N * K * ITERATIONS;
        double elapsed_sec = elapsed_ms / 1000.0;
        double throughput_tflops = (total_ops / 1e12) / elapsed_sec;
        double throughput_gflops = (total_ops / 1e9) / elapsed_sec;
        std::cout << "\nBenchmark Results (mma_m16n8k8_f16f32_bench):\n";
        std::cout << "  Iterations: " << ITERATIONS << "\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "  Elapsed time: " << elapsed_ms << " ms\n";
        std::cout << "  Total operations: " << total_ops / 1e9 << " G ops\n";
        std::cout << "  Throughput: " << throughput_tflops << " TFLOPS";
        std::cout << " (" << throughput_gflops << " GFLOPS)\n";
        MBENCH_CUDA_CHECK(cudaEventDestroy(start));
        MBENCH_CUDA_CHECK(cudaEventDestroy(stop));
    } else {
        std::cout << "\nRunning mma_m16n8k8_f16f32_once...\n";
        mma_m16n8k8_f16f32<<<NUM_BLOCKS, NUM_THREADS>>>(d_A, d_B, d_C, d_output);
        MBENCH_CUDA_CHECK(cudaDeviceSynchronize());

        MBENCH_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost));

        bool verified = verifyOutput(h_output, 12.0f);
        return verified;
    }

    MBENCH_CUDA_CHECK(cudaFree(d_A));
    MBENCH_CUDA_CHECK(cudaFree(d_B));
    MBENCH_CUDA_CHECK(cudaFree(d_C));
    MBENCH_CUDA_CHECK(cudaFree(d_output));

    return true;
}

int main() {
    std::cout << "MMA.SYNC Microbenchmark\n";
    std::cout << "=======================\n";

    // Test m16n8k16
    if (!run_mma_m16n8k16_f16f32(false)) {
        std::cerr << "m16n8k16 verification failed!\n";
        return 1;
    }

    if (!run_mma_m16n8k16_f16f32(true)) {
        std::cerr << "m16n8k16 benchmark failed!\n";
        return 1;
    }

    // Test m16n8k8
    if (!run_mma_m16n8k8_f16f32(false)) {
        std::cerr << "m16n8k8 verification failed!\n";
        return 1;
    }

    if (!run_mma_m16n8k8_f16f32(true)) {
        std::cerr << "m16n8k8 benchmark failed!\n";
        return 1;
    }

    std::cout << "\nAll tests passed!\n";
    return 0;
}
