/**
 * @file wmma_wmma_sync.cu
 * @brief Microbenchmark for WMMA.SYNC operations (m16n16k16)
 *
 * Benchmarks WMMA tensor core operations with FP16 and BF16 data types.
 * Supports both nvcuda::wmma API and inline PTX implementations.
 * Matrix size: m16n16k16
 */

#include "microbench/microbench.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstring>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <type_traits>

using namespace kebab::microbench;
using namespace nvcuda::wmma;

// ============================================================================
// WMMA API Implementation - FP16
// ============================================================================

// WMMA m16n16k16 using nvcuda::wmma API with FP16
__device__ void wmma_m16n16k16_f16_api(float *d, const half *a,
                                       const half *b, const float *c) {
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, row_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;

    load_matrix_sync(a_frag, a, 16);
    load_matrix_sync(b_frag, b, 16);
    load_matrix_sync(c_frag, c, 16, mem_row_major);

    mma_sync(c_frag, a_frag, b_frag, c_frag);

    store_matrix_sync(d, c_frag, 16, mem_row_major);
}

// WMMA m16n16k16 using nvcuda::wmma API with BF16
// Note: BF16 WMMA requires SM80 or higher
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
__device__ void wmma_m16n16k16_bf16_api(float *d, const __nv_bfloat16 *a,
                                        const __nv_bfloat16 *b, const float *c) {
    fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, __nv_bfloat16, row_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;

    load_matrix_sync(a_frag, const_cast<__nv_bfloat16*>(a), 16);
    load_matrix_sync(b_frag, const_cast<__nv_bfloat16*>(b), 16);
    load_matrix_sync(c_frag, const_cast<float*>(c), 16, mem_row_major);

    mma_sync(c_frag, a_frag, b_frag, c_frag);

    store_matrix_sync(d, c_frag, 16, mem_row_major);
}
#else
__device__ void wmma_m16n16k16_bf16_api(float *d, const __nv_bfloat16 *a,
                                        const __nv_bfloat16 *b, const float *c) {
    // BF16 WMMA not supported on this architecture
}
#endif

// ============================================================================
// WMMA PTX Implementation - FP16
// ============================================================================

// WMMA m16n16k16 using inline PTX with FP16
__device__ void wmma_m16n16k16_f16_ptx(float *d, const half *a,
                                       const half *b, const float *c) {
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, row_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;

    load_matrix_sync(a_frag, const_cast<half*>(a), 16);
    load_matrix_sync(b_frag, const_cast<half*>(b), 16);
    load_matrix_sync(c_frag, const_cast<float*>(c), 16, mem_row_major);

    // Cast fragments to uint32_t pointers for inline PTX
    uint32_t const *A_frag = reinterpret_cast<uint32_t const *>(a_frag.x);
    uint32_t const *B_frag = reinterpret_cast<uint32_t const *>(b_frag.x);

    // Inline PTX for wmma.mma.sync.aligned.m16n16k16.row.col.f32.f32
    asm volatile("wmma.mma.sync.aligned.m16n16k16.row.col.f32.f32 \t"
                 "{%0, %1, %2, %3, %4, %5, %6, %7}, \t"
                 "{%8, %9, %10, %11, %12, %13, %14, %15}, \t"
                 "{%16, %17, %18, %19, %20, %21, %22, %23}, \t"
                 "{%24, %25, %26, %27, %28, %29, %30, %31};"
                 : "=f"(c_frag.x[0]), "=f"(c_frag.x[1]), "=f"(c_frag.x[2]),
                   "=f"(c_frag.x[3]), "=f"(c_frag.x[4]), "=f"(c_frag.x[5]),
                   "=f"(c_frag.x[6]), "=f"(c_frag.x[7])
                 : "r"(A_frag[0]), "r"(A_frag[1]), "r"(A_frag[2]), "r"(A_frag[3]),
                   "r"(A_frag[4]), "r"(A_frag[5]), "r"(A_frag[6]), "r"(A_frag[7]),
                   "r"(B_frag[0]), "r"(B_frag[1]), "r"(B_frag[2]), "r"(B_frag[3]),
                   "r"(B_frag[4]), "r"(B_frag[5]), "r"(B_frag[6]), "r"(B_frag[7]),
                   "f"(c_frag.x[0]), "f"(c_frag.x[1]), "f"(c_frag.x[2]),
                   "f"(c_frag.x[3]), "f"(c_frag.x[4]), "f"(c_frag.x[5]),
                   "f"(c_frag.x[6]), "f"(c_frag.x[7]));

    store_matrix_sync(d, c_frag, 16, mem_row_major);
}

// ============================================================================
// Device Kernels Outer
// ============================================================================

// WMMA m16n16k16 FP16 API kernel
__global__ void wmma_m16n16k16_f16_api_kernel(half* gA, half* gB, float* gC,
                                              float* gD, int iteration=1) {
    __shared__ alignas(128) half sA[16 * 16];
    __shared__ alignas(128) half sB[16 * 16];
    __shared__ alignas(128) float sC[16 * 16];
    __shared__ alignas(128) float sD[16 * 16];

    int tid = threadIdx.x;

    // Load data to shared memory
    for (int i = tid; i < 16 * 16; i += blockDim.x) {
        sA[i] = gA[i];
        sB[i] = gB[i];
        sC[i] = gC[i];
    }
    __syncthreads();

    // Initialize output
    for (int i = tid; i < 16 * 16; i += blockDim.x) sD[i] = sC[i];
    __syncthreads();

    // Perform WMMA operation
    for (int i = 0; i < iteration; ++i) {
        wmma_m16n16k16_f16_api(sD, sA, sB, sC);
    }

    // Copy result to output
    for (int i = tid; i < 16 * 16; i += blockDim.x) gD[i] = sD[i];
}

// WMMA m16n16k16 BF16 API kernel
__global__ void wmma_m16n16k16_bf16_api_kernel(__nv_bfloat16* gA, __nv_bfloat16* gB,
                                               float* gC, float* gD, int iteration=1) {
    __shared__ alignas(128) __nv_bfloat16 sA[16 * 16];
    __shared__ alignas(128) __nv_bfloat16 sB[16 * 16];
    __shared__ alignas(128) float sC[16 * 16];
    __shared__ alignas(128) float sD[16 * 16];

    int tid = threadIdx.x;

    // Load data to shared memory
    for (int i = tid; i < 16 * 16; i += blockDim.x) {
        sA[i] = gA[i];
        sB[i] = gB[i];
        sC[i] = gC[i];
    }
    __syncthreads();

    // Initialize output
    for (int i = tid; i < 16 * 16; i += blockDim.x) sD[i] = sC[i];
    __syncthreads();

    // Perform WMMA operation
    for (int i = 0; i < iteration; ++i) {
        wmma_m16n16k16_bf16_api(sD, sA, sB, sC);
    }

    // Copy result to output
    for (int i = tid; i < 16 * 16; i += blockDim.x) gD[i] = sD[i];
}

// WMMA m16n16k16 FP16 PTX kernel
__global__ void wmma_m16n16k16_f16_ptx_kernel(half* gA, half* gB, float* gC,
                                              float* gD, int iteration=1) {
    __shared__ alignas(128) half sA[16 * 16];
    __shared__ alignas(128) half sB[16 * 16];
    __shared__ alignas(128) float sC[16 * 16];
    __shared__ alignas(128) float sD[16 * 16];

    int tid = threadIdx.x;

    // Load data to shared memory
    for (int i = tid; i < 16 * 16; i += blockDim.x) {
        sA[i] = gA[i];
        sB[i] = gB[i];
        sC[i] = gC[i];
    }
    __syncthreads();

    // Initialize output
    for (int i = tid; i < 16 * 16; i += blockDim.x) sD[i] = sC[i];
    __syncthreads();

    // Perform WMMA operation
    for (int i = 0; i < iteration; ++i) {
        wmma_m16n16k16_f16_ptx(sD, sA, sB, sC);
    }

    // Copy result to output
    for (int i = tid; i < 16 * 16; i += blockDim.x) gD[i] = sD[i];
}

// ============================================================================
// Verification and Benchmarking
// ============================================================================

bool verifyOutput(const std::vector<float>& gpu_output, float expected,
                  float tolerance = 0.5f) {
    std::cout << "Verification: checking " << gpu_output.size() << " outputs\n";
    int fail_count = 0;
    for (size_t i = 0; i < gpu_output.size(); ++i) {
        float diff = std::abs(gpu_output[i] - expected);
        if (diff > tolerance) {
            if (fail_count < 10) {
                std::cerr << "FAILED at [" << i << "]: expected " << expected
                          << ", got " << gpu_output[i] << "\n";
            }
            fail_count++;
        }
    }
    if (fail_count > 0) {
        std::cerr << "Total failures: " << fail_count << " out of "
                  << gpu_output.size() << "\n";
        return false;
    }
    std::cout << "PASSED: all outputs match expected value " << expected << "\n";
    return true;
}

bool run_wmma_m16n16k16_f16_api(bool bench = false) {
    constexpr int M = 16, N = 16, K = 16;
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

    MBENCH_CUDA_CHECK(
        cudaMemcpy(d_A, h_A.data(), M * K * sizeof(half), cudaMemcpyHostToDevice));
    MBENCH_CUDA_CHECK(
        cudaMemcpy(d_B, h_B.data(), K * N * sizeof(half), cudaMemcpyHostToDevice));
    MBENCH_CUDA_CHECK(
        cudaMemcpy(d_C, h_C.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));

    if (bench) {
        wmma_m16n16k16_f16_api_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_A, d_B, d_C,
                                                                     d_output, 10);
        MBENCH_CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t start, stop;
        MBENCH_CUDA_CHECK(cudaEventCreate(&start));
        MBENCH_CUDA_CHECK(cudaEventCreate(&stop));

        MBENCH_CUDA_CHECK(cudaEventRecord(start));
        wmma_m16n16k16_f16_api_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(
            d_A, d_B, d_C, d_output, ITERATIONS);
        MBENCH_CUDA_CHECK(cudaEventRecord(stop));
        MBENCH_CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        MBENCH_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

        MBENCH_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, M * N * sizeof(float),
                                     cudaMemcpyDeviceToHost));

        double total_ops = 2.0 * M * N * K * ITERATIONS;
        double elapsed_us = elapsed_ms * 1000.0;
        double throughput_tflops = (total_ops / 1e6) / elapsed_us;
        double throughput_gflops = (total_ops / 1e3) / elapsed_us;
        std::cout << "\nBenchmark Results (wmma_m16n16k16_f16_api):\n";
        std::cout << "  Iterations: " << ITERATIONS << "\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "  Elapsed time: " << elapsed_ms << " ms\n";
        std::cout << "  Total operations: " << total_ops / 1e9 << " G ops\n";
        std::cout << "  Throughput: " << throughput_tflops << " TFLOPS";
        std::cout << " (" << throughput_gflops << " GFLOPS)\n";
        MBENCH_CUDA_CHECK(cudaEventDestroy(start));
        MBENCH_CUDA_CHECK(cudaEventDestroy(stop));
    } else {
        std::cout << "\nRunning wmma_m16n16k16_f16_api_once...\n";
        wmma_m16n16k16_f16_api_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_A, d_B, d_C,
                                                                     d_output);
        MBENCH_CUDA_CHECK(cudaDeviceSynchronize());
        MBENCH_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, M * N * sizeof(float),
                                     cudaMemcpyDeviceToHost));
        // Expected: C + A*B = 11 + 16*1 = 27
        bool verified = verifyOutput(h_output, 27.0f);
        return verified;
    }

    MBENCH_CUDA_CHECK(cudaFree(d_A));
    MBENCH_CUDA_CHECK(cudaFree(d_B));
    MBENCH_CUDA_CHECK(cudaFree(d_C));
    MBENCH_CUDA_CHECK(cudaFree(d_output));

    return true;
}

bool run_wmma_m16n16k16_bf16_api(bool bench = false) {
    constexpr int M = 16, N = 16, K = 16;
    constexpr int NUM_THREADS = 32;
    constexpr int NUM_BLOCKS = 1;
    constexpr int ITERATIONS = 100000;

    std::vector<__nv_bfloat16> h_A(M * K, __float2bfloat16(1.0f));
    std::vector<__nv_bfloat16> h_B(K * N, __float2bfloat16(1.0f));
    std::vector<float> h_C(M * N, 11.0f);
    std::vector<float> h_output(M * N);

    __nv_bfloat16 *d_A, *d_B;
    float *d_C, *d_output;
    MBENCH_CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(__nv_bfloat16)));
    MBENCH_CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(__nv_bfloat16)));
    MBENCH_CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    MBENCH_CUDA_CHECK(cudaMalloc(&d_output, M * N * sizeof(float)));

    MBENCH_CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(__nv_bfloat16),
                                 cudaMemcpyHostToDevice));
    MBENCH_CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(__nv_bfloat16),
                                 cudaMemcpyHostToDevice));
    MBENCH_CUDA_CHECK(
        cudaMemcpy(d_C, h_C.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));

    if (bench) {
        wmma_m16n16k16_bf16_api_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_A, d_B, d_C,
                                                                      d_output, 10);
        MBENCH_CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t start, stop;
        MBENCH_CUDA_CHECK(cudaEventCreate(&start));
        MBENCH_CUDA_CHECK(cudaEventCreate(&stop));

        MBENCH_CUDA_CHECK(cudaEventRecord(start));
        wmma_m16n16k16_bf16_api_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(
            d_A, d_B, d_C, d_output, ITERATIONS);
        MBENCH_CUDA_CHECK(cudaEventRecord(stop));
        MBENCH_CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        MBENCH_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

        MBENCH_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, M * N * sizeof(float),
                                     cudaMemcpyDeviceToHost));

        double total_ops = 2.0 * M * N * K * ITERATIONS;
        double elapsed_us = elapsed_ms * 1000.0;
        double throughput_tflops = (total_ops / 1e6) / elapsed_us;
        double throughput_gflops = (total_ops / 1e3) / elapsed_us;
        std::cout << "\nBenchmark Results (wmma_m16n16k16_bf16_api):\n";
        std::cout << "  Iterations: " << ITERATIONS << "\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "  Elapsed time: " << elapsed_ms << " ms\n";
        std::cout << "  Total operations: " << total_ops / 1e9 << " G ops\n";
        std::cout << "  Throughput: " << throughput_tflops << " TFLOPS";
        std::cout << " (" << throughput_gflops << " GFLOPS)\n";
        MBENCH_CUDA_CHECK(cudaEventDestroy(start));
        MBENCH_CUDA_CHECK(cudaEventDestroy(stop));
    } else {
        std::cout << "\nRunning wmma_m16n16k16_bf16_api_once...\n";
        wmma_m16n16k16_bf16_api_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_A, d_B, d_C,
                                                                      d_output);
        MBENCH_CUDA_CHECK(cudaDeviceSynchronize());
        MBENCH_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, M * N * sizeof(float),
                                     cudaMemcpyDeviceToHost));
        // Expected: C + A*B = 11 + 16*1 = 27
        bool verified = verifyOutput(h_output, 27.0f);
        return verified;
    }

    MBENCH_CUDA_CHECK(cudaFree(d_A));
    MBENCH_CUDA_CHECK(cudaFree(d_B));
    MBENCH_CUDA_CHECK(cudaFree(d_C));
    MBENCH_CUDA_CHECK(cudaFree(d_output));

    return true;
}

bool run_wmma_m16n16k16_f16_ptx(bool bench = false) {
    constexpr int M = 16, N = 16, K = 16;
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

    MBENCH_CUDA_CHECK(
        cudaMemcpy(d_A, h_A.data(), M * K * sizeof(half), cudaMemcpyHostToDevice));
    MBENCH_CUDA_CHECK(
        cudaMemcpy(d_B, h_B.data(), K * N * sizeof(half), cudaMemcpyHostToDevice));
    MBENCH_CUDA_CHECK(
        cudaMemcpy(d_C, h_C.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));

    if (bench) {
        wmma_m16n16k16_f16_ptx_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_A, d_B, d_C,
                                                                     d_output, 10);
        MBENCH_CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t start, stop;
        MBENCH_CUDA_CHECK(cudaEventCreate(&start));
        MBENCH_CUDA_CHECK(cudaEventCreate(&stop));

        MBENCH_CUDA_CHECK(cudaEventRecord(start));
        wmma_m16n16k16_f16_ptx_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(
            d_A, d_B, d_C, d_output, ITERATIONS);
        MBENCH_CUDA_CHECK(cudaEventRecord(stop));
        MBENCH_CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        MBENCH_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

        MBENCH_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, M * N * sizeof(float),
                                     cudaMemcpyDeviceToHost));

        double total_ops = 2.0 * M * N * K * ITERATIONS;
        double elapsed_us = elapsed_ms * 1000.0;
        double throughput_tflops = (total_ops / 1e6) / elapsed_us;
        double throughput_gflops = (total_ops / 1e3) / elapsed_us;
        std::cout << "\nBenchmark Results (wmma_m16n16k16_f16_ptx):\n";
        std::cout << "  Iterations: " << ITERATIONS << "\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "  Elapsed time: " << elapsed_ms << " ms\n";
        std::cout << "  Total operations: " << total_ops / 1e9 << " G ops\n";
        std::cout << "  Throughput: " << throughput_tflops << " TFLOPS";
        std::cout << " (" << throughput_gflops << " GFLOPS)\n";
        MBENCH_CUDA_CHECK(cudaEventDestroy(start));
        MBENCH_CUDA_CHECK(cudaEventDestroy(stop));
    } else {
        std::cout << "\nRunning wmma_m16n16k16_f16_ptx_once...\n";
        wmma_m16n16k16_f16_ptx_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_A, d_B, d_C,
                                                                     d_output);
        MBENCH_CUDA_CHECK(cudaDeviceSynchronize());
        MBENCH_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, M * N * sizeof(float),
                                     cudaMemcpyDeviceToHost));
        // Expected: C + A*B = 11 + 16*1 = 27
        bool verified = verifyOutput(h_output, 27.0f);
        return verified;
    }

    MBENCH_CUDA_CHECK(cudaFree(d_A));
    MBENCH_CUDA_CHECK(cudaFree(d_B));
    MBENCH_CUDA_CHECK(cudaFree(d_C));
    MBENCH_CUDA_CHECK(cudaFree(d_output));

    return true;
}

int main() {
    std::cout << "WMMA.SYNC Microbenchmark\n";
    std::cout << "========================\n";

    // Test FP16 API
    if (!run_wmma_m16n16k16_f16_api(false)) {
        std::cerr << "wmma_m16n16k16_f16_api verification failed!\n";
        return 1;
    }

    if (!run_wmma_m16n16k16_f16_api(true)) {
        std::cerr << "wmma_m16n16k16_f16_api benchmark failed!\n";
        return 1;
    }

    // Test BF16 API
    if (!run_wmma_m16n16k16_bf16_api(false)) {
        std::cerr << "wmma_m16n16k16_bf16_api verification failed!\n";
        return 1;
    }

    if (!run_wmma_m16n16k16_bf16_api(true)) {
        std::cerr << "wmma_m16n16k16_bf16_api benchmark failed!\n";
        return 1;
    }

    // Test FP16 PTX
    if (!run_wmma_m16n16k16_f16_ptx(false)) {
        std::cerr << "wmma_m16n16k16_f16_ptx verification failed!\n";
        return 1;
    }

    if (!run_wmma_m16n16k16_f16_ptx(true)) {
        std::cerr << "wmma_m16n16k16_f16_ptx benchmark failed!\n";
        return 1;
    }

    std::cout << "\nAll tests passed!\n";
    return 0;
}
