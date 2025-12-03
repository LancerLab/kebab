/**
 * @file mbench_wgmma_mma.cu
 * @brief Microbenchmark for WGMMA (Warp Group Matrix Multiply Accumulate) operations
 *
 * Simple single-operation WGMMA benchmark with verification.
 */

#include "microbench/microbench.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstring>
#include <vector>
#include <iostream>
#include <cmath>

using namespace kebab::microbench;

// ============================================================================
// WGMMA Wrapper Functions (from cuda_gemm_v2_wgmma_tma.cu)
// ============================================================================

__device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) {
    return (((x) & 0x3FFFF) >> 0x4);
}

// For 64x16 matrix: stride = 16 (in elements), leading_dim = 64 (in elements)
__device__ uint64_t make_smem_desc_wgmma(__half* ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0x0000000000000000;
    desc |= matrix_descriptor_encode(addr);
    desc |= matrix_descriptor_encode((uint64_t)16) << 16;   // stride in elements
    desc |= matrix_descriptor_encode((uint64_t)64) << 32;   // leading_dim in elements (was 1024 for 64x64)
    desc |= 1llu << 62;
    return desc;
}

__device__ void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ void warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N>
__device__ void warpgroup_wait() {
    static_assert(N >= 0 && N <= 7, "WGMMA wait: N must be in range [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

// WGMMA 64x64x16 for FP16 (produces FP32 accumulator)
template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ void wgmma64_fp16(float d[4][8], __half* sA, __half* sB) {
    uint64_t desc_a = make_smem_desc_wgmma(&sA[0]);
    uint64_t desc_b = make_smem_desc_wgmma(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31},"
        " %32,"
        " %33,"
        " %34, %35, %36, %37, %38;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
          "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
          "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
          "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
          "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
          "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
          "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
          "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7])
        : "l"(desc_a), "l"(desc_b),
          "n"(ScaleD), "n"(ScaleA), "n"(ScaleB), "n"(TransA), "n"(TransB)
        : "memory"
    );
}

// WGMMA 64x64x16 for BF16 (produces FP32 accumulator)
template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ void wgmma64_bf16(float d[4][8], __nv_bfloat16* sA, __nv_bfloat16* sB) {
    uint64_t desc_a = make_smem_desc_wgmma(reinterpret_cast<__half*>(sA));
    uint64_t desc_b = make_smem_desc_wgmma(reinterpret_cast<__half*>(sB));
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31},"
        " %32,"
        " %33,"
        " %34, %35, %36, %37, %38;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
          "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
          "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
          "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
          "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
          "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
          "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
          "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7])
        : "l"(desc_a), "l"(desc_b),
          "n"(ScaleD), "n"(ScaleA), "n"(ScaleB), "n"(TransA), "n"(TransB)
        : "memory"
    );
}

// template interface for call bf16 or fp16 wgmma and handle its fences
template<typename T>
__device__ void wgmma_m64n64k16(__shared__ __half* sA, __shared__ __half* sB, float out[4][8]) {
    // Single WGMMA operation
    warpgroup_arrive();
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        wgmma64_bf16<1, 1, 1, 0, 0>(out, reinterpret_cast<__nv_bfloat16*>(sA), reinterpret_cast<__nv_bfloat16*>(sB));
    } else if constexpr (std::is_same_v<T, __half>) {
        wgmma64_fp16<1, 1, 1, 0, 0>(out, sA, sB);
    }   
    warpgroup_commit_batch();
    warpgroup_wait<0>();
}

// ============================================================================
// Test Kernels
// ============================================================================

// Test 1: WGMMA 64x64x16 FP16 - single operation
__global__ void wgmma64_fp16_once(__half* gA, __half* gB, float* output) {
    __shared__ alignas(128) __half sA[64 * 16];  // 64x16
    __shared__ alignas(128) __half sB[16 * 64];  // 16x64

    int tid = threadIdx.x;
    // Load data to SMEM
    for (int i = tid; i < 64 * 16; i += blockDim.x) {
        sA[i] = gA[i];
        sB[i] = gB[i];
    }
    __syncthreads();

    // Accumulator: 64x64 output = 4 x 8 floats per thread (128 threads)
    float d[4][8];
    memset(d, 0, sizeof(d));

    // Call WGMMA
    wgmma_m64n64k16<__half>(sA, sB, d);

    // Store result
    if (tid < 32) {
        output[tid] = d[tid / 8][tid % 8];
    }
}



// ============================================================================
// Verification
// ============================================================================

bool verifyOutput(const std::vector<float>& gpu_output, float expected, float tolerance = 0.5f) {
    std::cout << "Verification: checking " << gpu_output.size() << " outputs\n";
    for (size_t i = 0; i < gpu_output.size(); ++i) {
        float diff = std::abs(gpu_output[i] - expected);
        if (diff > tolerance) {
            std::cerr << "FAILED at [" << i << "]: expected " << expected
                      << ", got " << gpu_output[i] << "\n";
            return false;
        }
    }
    std::cout << "PASSED: all outputs match expected value " << expected << "\n";
    return true;
}

bool run_wgmma64_fp16_once() {
    constexpr int M = 64, N = 64, K_TILE = 16;
    constexpr int NUM_THREADS = 128;
    constexpr int NUM_BLOCKS = 1;

    // Allocate host memory
    std::vector<__half> h_A(M * K_TILE, __float2half(1.0f));
    std::vector<__half> h_B(K_TILE * N, __float2half(1.0f));
    std::vector<float> h_output(32);

    // Allocate device memory
    __half *d_A, *d_B;
    float *d_output;
    MBENCH_CUDA_CHECK(cudaMalloc(&d_A, M * K_TILE * sizeof(__half)));
    MBENCH_CUDA_CHECK(cudaMalloc(&d_B, K_TILE * N * sizeof(__half)));
    MBENCH_CUDA_CHECK(cudaMalloc(&d_output, 32 * sizeof(float)));

    // Copy to device
    MBENCH_CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K_TILE * sizeof(__half), cudaMemcpyHostToDevice));
    MBENCH_CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K_TILE * N * sizeof(__half), cudaMemcpyHostToDevice));

    // Run kernel
    std::cout << "\nRunning kernel_wgmma64_fp16_single...\n";
    wgmma64_fp16_once<<<NUM_BLOCKS, NUM_THREADS>>>(d_A, d_B, d_output);
    MBENCH_CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    MBENCH_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, 32 * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify: A[64,16] * B[16,64] with all 1.0 => C[i,j] = 16
    bool verified = verifyOutput(h_output, 16.0f);

    // Cleanup
    MBENCH_CUDA_CHECK(cudaFree(d_A));
    MBENCH_CUDA_CHECK(cudaFree(d_B));
    MBENCH_CUDA_CHECK(cudaFree(d_output));

    return verified;
}

int main() {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "WGMMA Microbenchmark - Single Operation Verification\n";
    std::cout << std::string(80, '=') << "\n";
    bool verified = true;

    verified &= run_wgmma64_fp16_once();

    std::cout << std::string(80, '=') << "\n";
    return verified ? 0 : 1;
}

