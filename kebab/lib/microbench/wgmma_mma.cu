/**
 * @file mbench_wgmma_mma.cu
 * @brief Microbenchmark for WGMMA (Warp Group Matrix Multiply Accumulate) operations
 *
 * Simple single-operation WGMMA benchmark with verification.
 */

#include "microbench/microbench.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstring>
#include <vector>
#include <iostream>
#include <cmath>
#include <type_traits>

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
template<typename T, int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ void wgmma64_fp16(float d[4][8], T* sA, T* sB) {
    static_assert(std::is_same_v<T, __half>, "wgmma64_fp16 requires __half type");
    uint64_t desc_a = make_smem_desc_wgmma(reinterpret_cast<__half*>(sA));
    uint64_t desc_b = make_smem_desc_wgmma(reinterpret_cast<__half*>(sB));
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
template<typename T, int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ void wgmma64_bf16(float d[4][8], T* sA, T* sB) {
    static_assert(std::is_same_v<T, __nv_bfloat16>, "wgmma64_bf16 requires __nv_bfloat16 type");
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

// WGMMA 64x128x16 for FP16 (produces FP32 accumulator)
template<typename T, int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ void wgmma128_fp16(float d[8][8], T* sA, T* sB) {
    static_assert(std::is_same_v<T, __half>, "wgmma128_fp16 requires __half type");
    uint64_t desc_a = make_smem_desc_wgmma(sA);
    uint64_t desc_b = make_smem_desc_wgmma(sB);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
        " %64,"
        " %65,"
        " %66, %67, %68, %69, %70;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
          "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
          "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
          "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
          "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
          "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
          "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
          "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7]),
          "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
          "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]),
          "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
          "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
          "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]),
          "+f"(d[6][4]), "+f"(d[6][5]), "+f"(d[6][6]), "+f"(d[6][7]),
          "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
          "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7])
        : "l"(desc_a), "l"(desc_b),
          "n"(ScaleD), "n"(ScaleA), "n"(ScaleB), "n"(TransA), "n"(TransB)
        : "memory"
    );
}

// WGMMA 64x128x16 for BF16 (produces FP32 accumulator)
template<typename T, int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ void wgmma128_bf16(float d[8][8], T* sA, T* sB) {
    static_assert(std::is_same_v<T, __nv_bfloat16>, "wgmma128_bf16 requires __nv_bfloat16 type");
    uint64_t desc_a = make_smem_desc_wgmma(reinterpret_cast<__half*>(sA));
    uint64_t desc_b = make_smem_desc_wgmma(reinterpret_cast<__half*>(sB));
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
        " %64,"
        " %65,"
        " %66, %67, %68, %69, %70;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
          "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
          "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
          "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
          "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
          "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
          "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
          "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7]),
          "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
          "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]),
          "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
          "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
          "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]),
          "+f"(d[6][4]), "+f"(d[6][5]), "+f"(d[6][6]), "+f"(d[6][7]),
          "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
          "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7])
        : "l"(desc_a), "l"(desc_b),
          "n"(ScaleD), "n"(ScaleA), "n"(ScaleB), "n"(TransA), "n"(TransB)
        : "memory"
    );
}

// template interface for call bf16 or fp16 wgmma and handle its fences
template<typename T>
__device__ void wgmma_m64n64k16(T* sA, T* sB, float out[4][8]) {
    static_assert(std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>,
                  "wgmma_m64n64k16 requires __half or __nv_bfloat16 type");
    // Single WGMMA operation
    warpgroup_arrive();
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        wgmma64_bf16<T, 1, 1, 1, 0, 0>(out, sA, sB);
    } else if constexpr (std::is_same_v<T, __half>) {
        wgmma64_fp16<T, 1, 1, 1, 0, 0>(out, sA, sB);
    }
    warpgroup_commit_batch();
    warpgroup_wait<0>();
}

// template interface for m64n128k16 wgmma
template<typename T>
__device__ void wgmma_m64n128k16(T* sA, T* sB, float out[8][8]) {
    static_assert(std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>,
                  "wgmma_m64n128k16 requires __half or __nv_bfloat16 type");
    // Single WGMMA operation
    warpgroup_arrive();
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        wgmma128_bf16<T, 1, 1, 1, 0, 0>(out, sA, sB);
    } else if constexpr (std::is_same_v<T, __half>) {
        wgmma128_fp16<T, 1, 1, 1, 0, 0>(out, sA, sB);
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

// Test 2: WGMMA 64x64x16 FP16 - benchmark with multiple iterations
__global__ void wgmma64_fp16_bench(__half* gA, __half* gB, float* output, int iterations) {
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

    // Run WGMMA multiple times
    for (int iter = 0; iter < iterations; ++iter) {
        wgmma_m64n64k16<__half>(sA, sB, d);
    }

    // Store result
    if (tid < 32) {
        output[tid] = d[tid / 8][tid % 8];
    }
}

// Test 3: WGMMA 64x64x16 BF16 - single operation
__global__ void wgmma64_bf16_once(__nv_bfloat16* gA, __nv_bfloat16* gB, float* output) {
    __shared__ alignas(128) __nv_bfloat16 sA[64 * 16];  // 64x16
    __shared__ alignas(128) __nv_bfloat16 sB[16 * 64];  // 16x64

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
    wgmma_m64n64k16<__nv_bfloat16>(sA, sB, d);

    // Store result
    if (tid < 32) {
        output[tid] = d[tid / 8][tid % 8];
    }
}

// Test 4: WGMMA 64x64x16 BF16 - benchmark with multiple iterations
__global__ void wgmma64_bf16_bench(__nv_bfloat16* gA, __nv_bfloat16* gB, float* output, int iterations) {
    __shared__ alignas(128) __nv_bfloat16 sA[64 * 16];  // 64x16
    __shared__ alignas(128) __nv_bfloat16 sB[16 * 64];  // 16x64

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

    // Run WGMMA multiple times
    for (int iter = 0; iter < iterations; ++iter) {
        wgmma_m64n64k16<__nv_bfloat16>(sA, sB, d);
    }

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
    std::cout << "\nRunning wgmma64_fp16_once...\n";
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

bool bench_wgmma64_fp16() {
    constexpr int M = 64, N = 64, K_TILE = 16;
    constexpr int NUM_THREADS = 128;
    constexpr int NUM_BLOCKS = 1;
    constexpr int ITERATIONS = 10000;

    // Allocate host memory
    std::vector<__half> h_A(M * K_TILE, __float2half(1.0f));
    std::vector<__half> h_B(K_TILE * N, __float2half(1.0f));
    std::vector<float> h_output(32);

    // Allocate device memory
    __half *d_A, *d_B;
    float *d_output;
    int *d_iterations;
    MBENCH_CUDA_CHECK(cudaMalloc(&d_A, M * K_TILE * sizeof(__half)));
    MBENCH_CUDA_CHECK(cudaMalloc(&d_B, K_TILE * N * sizeof(__half)));
    MBENCH_CUDA_CHECK(cudaMalloc(&d_output, 32 * sizeof(float)));
    MBENCH_CUDA_CHECK(cudaMalloc(&d_iterations, sizeof(int)));

    // Copy to device
    MBENCH_CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K_TILE * sizeof(__half), cudaMemcpyHostToDevice));
    MBENCH_CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K_TILE * N * sizeof(__half), cudaMemcpyHostToDevice));
    MBENCH_CUDA_CHECK(cudaMemcpy(d_iterations, &ITERATIONS, sizeof(int), cudaMemcpyHostToDevice));

    // Warmup
    wgmma64_fp16_bench<<<NUM_BLOCKS, NUM_THREADS>>>(d_A, d_B, d_output, 10);
    MBENCH_CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark with timing
    cudaEvent_t start, stop;
    MBENCH_CUDA_CHECK(cudaEventCreate(&start));
    MBENCH_CUDA_CHECK(cudaEventCreate(&stop));

    MBENCH_CUDA_CHECK(cudaEventRecord(start));
    wgmma64_fp16_bench<<<NUM_BLOCKS, NUM_THREADS>>>(d_A, d_B, d_output, ITERATIONS);
    MBENCH_CUDA_CHECK(cudaEventRecord(stop));
    MBENCH_CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    MBENCH_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // Copy result back
    MBENCH_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, 32 * sizeof(float), cudaMemcpyDeviceToHost));

    // Calculate throughput
    // Each WGMMA: 64x64x16 = 65536 FP16 operations
    // Total operations = 2 * 65536 * ITERATIONS
    double total_ops = 2 * 65536.0 * ITERATIONS;
    double elapsed_sec = elapsed_ms / 1000.0;
    double throughput_tflops = (total_ops / 1e12) / elapsed_sec;

    std::cout << "\nBenchmark Results (wgmma64_fp16_bench):\n";
    std::cout << "  Iterations: " << ITERATIONS << "\n";
    std::cout << "  Elapsed time: " << elapsed_ms << " ms\n";
    std::cout << "  Total operations: " << total_ops / 1e9 << " G ops\n";
    std::cout << "  Throughput: " << throughput_tflops << " TFLOPS\n";

    // Cleanup
    MBENCH_CUDA_CHECK(cudaEventDestroy(start));
    MBENCH_CUDA_CHECK(cudaEventDestroy(stop));
    MBENCH_CUDA_CHECK(cudaFree(d_A));
    MBENCH_CUDA_CHECK(cudaFree(d_B));
    MBENCH_CUDA_CHECK(cudaFree(d_output));
    MBENCH_CUDA_CHECK(cudaFree(d_iterations));

    return true;
}

bool run_wgmma64_bf16_once() {
    constexpr int M = 64, N = 64, K_TILE = 16;
    constexpr int NUM_THREADS = 128;
    constexpr int NUM_BLOCKS = 1;

    // Allocate host memory
    std::vector<__nv_bfloat16> h_A(M * K_TILE, __float2bfloat16(1.0f));
    std::vector<__nv_bfloat16> h_B(K_TILE * N, __float2bfloat16(1.0f));
    std::vector<float> h_output(32);

    // Allocate device memory
    __nv_bfloat16 *d_A, *d_B;
    float *d_output;
    MBENCH_CUDA_CHECK(cudaMalloc(&d_A, M * K_TILE * sizeof(__nv_bfloat16)));
    MBENCH_CUDA_CHECK(cudaMalloc(&d_B, K_TILE * N * sizeof(__nv_bfloat16)));
    MBENCH_CUDA_CHECK(cudaMalloc(&d_output, 32 * sizeof(float)));

    // Copy to device
    MBENCH_CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K_TILE * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    MBENCH_CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K_TILE * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    // Run kernel
    std::cout << "\nRunning wgmma64_bf16_once...\n";
    wgmma64_bf16_once<<<NUM_BLOCKS, NUM_THREADS>>>(d_A, d_B, d_output);
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

bool bench_wgmma64_bf16() {
    constexpr int M = 64, N = 64, K_TILE = 16;
    constexpr int NUM_THREADS = 128;
    constexpr int NUM_BLOCKS = 1;
    constexpr int ITERATIONS = 10000;

    // Allocate host memory
    std::vector<__nv_bfloat16> h_A(M * K_TILE, __float2bfloat16(1.0f));
    std::vector<__nv_bfloat16> h_B(K_TILE * N, __float2bfloat16(1.0f));
    std::vector<float> h_output(32);

    // Allocate device memory
    __nv_bfloat16 *d_A, *d_B;
    float *d_output;
    int *d_iterations;
    MBENCH_CUDA_CHECK(cudaMalloc(&d_A, M * K_TILE * sizeof(__nv_bfloat16)));
    MBENCH_CUDA_CHECK(cudaMalloc(&d_B, K_TILE * N * sizeof(__nv_bfloat16)));
    MBENCH_CUDA_CHECK(cudaMalloc(&d_output, 32 * sizeof(float)));
    MBENCH_CUDA_CHECK(cudaMalloc(&d_iterations, sizeof(int)));

    // Copy to device
    MBENCH_CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K_TILE * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    MBENCH_CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K_TILE * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    MBENCH_CUDA_CHECK(cudaMemcpy(d_iterations, &ITERATIONS, sizeof(int), cudaMemcpyHostToDevice));

    // Warmup
    wgmma64_bf16_bench<<<NUM_BLOCKS, NUM_THREADS>>>(d_A, d_B, d_output, 10);
    MBENCH_CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark with timing
    cudaEvent_t start, stop;
    MBENCH_CUDA_CHECK(cudaEventCreate(&start));
    MBENCH_CUDA_CHECK(cudaEventCreate(&stop));

    MBENCH_CUDA_CHECK(cudaEventRecord(start));
    wgmma64_bf16_bench<<<NUM_BLOCKS, NUM_THREADS>>>(d_A, d_B, d_output, ITERATIONS);
    MBENCH_CUDA_CHECK(cudaEventRecord(stop));
    MBENCH_CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    MBENCH_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // Copy result back
    MBENCH_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, 32 * sizeof(float), cudaMemcpyDeviceToHost));

    // Calculate throughput
    // Each WGMMA: 64x64x16 = 65536 BF16 operations
    // Total operations = 2 * 65536 * ITERATIONS
    double total_ops = 2 * 65536.0 * ITERATIONS;
    double elapsed_sec = elapsed_ms / 1000.0;
    double throughput_tflops = (total_ops / 1e12) / elapsed_sec;

    std::cout << "\nBenchmark Results (wgmma64_bf16_bench):\n";
    std::cout << "  Iterations: " << ITERATIONS << "\n";
    std::cout << "  Elapsed time: " << elapsed_ms << " ms\n";
    std::cout << "  Total operations: " << total_ops / 1e9 << " G ops\n";
    std::cout << "  Throughput: " << throughput_tflops << " TFLOPS\n";

    // Cleanup
    MBENCH_CUDA_CHECK(cudaEventDestroy(start));
    MBENCH_CUDA_CHECK(cudaEventDestroy(stop));
    MBENCH_CUDA_CHECK(cudaFree(d_A));
    MBENCH_CUDA_CHECK(cudaFree(d_B));
    MBENCH_CUDA_CHECK(cudaFree(d_output));
    MBENCH_CUDA_CHECK(cudaFree(d_iterations));

    return true;
}

// ============================================================================
// m64n128k16 Tests
// ============================================================================

// Test 5: WGMMA 64x128x16 FP16 - single operation
__global__ void wgmma128_fp16_once(__half* gA, __half* gB, float* output) {
    __shared__ alignas(128) __half sA[64 * 16];   // 64x16
    __shared__ alignas(128) __half sB[16 * 128];  // 16x128

    int tid = threadIdx.x;
    // Load data to SMEM
    for (int i = tid; i < 64 * 16; i += blockDim.x) {
        sA[i] = gA[i];
    }
    for (int i = tid; i < 16 * 128; i += blockDim.x) {
        sB[i] = gB[i];
    }
    __syncthreads();

    // Accumulator: 64x128 output = 8 x 8 floats per thread (128 threads)
    float d[8][8];
    memset(d, 0, sizeof(d));

    // Call WGMMA
    wgmma_m64n128k16<__half>(sA, sB, d);

    // Store result
    if (tid < 64) {
        output[tid] = d[tid / 8][tid % 8];
    }
}

// Test 6: WGMMA 64x128x16 FP16 - benchmark with multiple iterations
__global__ void wgmma128_fp16_bench(__half* gA, __half* gB, float* output, int iterations) {
    __shared__ alignas(128) __half sA[64 * 16];   // 64x16
    __shared__ alignas(128) __half sB[16 * 128];  // 16x128

    int tid = threadIdx.x;
    // Load data to SMEM
    for (int i = tid; i < 64 * 16; i += blockDim.x) {
        sA[i] = gA[i];
    }
    for (int i = tid; i < 16 * 128; i += blockDim.x) {
        sB[i] = gB[i];
    }
    __syncthreads();

    // Accumulator: 64x128 output = 8 x 8 floats per thread (128 threads)
    float d[8][8];
    memset(d, 0, sizeof(d));

    // Run WGMMA multiple times
    for (int iter = 0; iter < iterations; ++iter) {
        wgmma_m64n128k16<__half>(sA, sB, d);
    }

    // Store result
    if (tid < 64) {
        output[tid] = d[tid / 8][tid % 8];
    }
}

// Test 7: WGMMA 64x128x16 BF16 - single operation
__global__ void wgmma128_bf16_once(__nv_bfloat16* gA, __nv_bfloat16* gB, float* output) {
    __shared__ alignas(128) __nv_bfloat16 sA[64 * 16];   // 64x16
    __shared__ alignas(128) __nv_bfloat16 sB[16 * 128];  // 16x128

    int tid = threadIdx.x;
    // Load data to SMEM
    for (int i = tid; i < 64 * 16; i += blockDim.x) {
        sA[i] = gA[i];
    }
    for (int i = tid; i < 16 * 128; i += blockDim.x) {
        sB[i] = gB[i];
    }
    __syncthreads();

    // Accumulator: 64x128 output = 8 x 8 floats per thread (128 threads)
    float d[8][8];
    memset(d, 0, sizeof(d));

    // Call WGMMA
    wgmma_m64n128k16<__nv_bfloat16>(sA, sB, d);

    // Store result
    if (tid < 64) {
        output[tid] = d[tid / 8][tid % 8];
    }
}

// Test 8: WGMMA 64x128x16 BF16 - benchmark with multiple iterations
__global__ void wgmma128_bf16_bench(__nv_bfloat16* gA, __nv_bfloat16* gB, float* output, int iterations) {
    __shared__ alignas(128) __nv_bfloat16 sA[64 * 16];   // 64x16
    __shared__ alignas(128) __nv_bfloat16 sB[16 * 128];  // 16x128

    int tid = threadIdx.x;
    // Load data to SMEM
    for (int i = tid; i < 64 * 16; i += blockDim.x) {
        sA[i] = gA[i];
    }
    for (int i = tid; i < 16 * 128; i += blockDim.x) {
        sB[i] = gB[i];
    }
    __syncthreads();

    // Accumulator: 64x128 output = 8 x 8 floats per thread (128 threads)
    float d[8][8];
    memset(d, 0, sizeof(d));

    // Run WGMMA multiple times
    for (int iter = 0; iter < iterations; ++iter) {
        wgmma_m64n128k16<__nv_bfloat16>(sA, sB, d);
    }

    // Store result
    if (tid < 64) {
        output[tid] = d[tid / 8][tid % 8];
    }
}

bool run_wgmma128_fp16_once() {
    constexpr int M = 64, N = 128, K_TILE = 16;
    constexpr int NUM_THREADS = 128;
    constexpr int NUM_BLOCKS = 1;

    // Allocate host memory
    std::vector<__half> h_A(M * K_TILE, __float2half(1.0f));
    std::vector<__half> h_B(K_TILE * N, __float2half(1.0f));
    std::vector<float> h_output(64);

    // Allocate device memory
    __half *d_A, *d_B;
    float *d_output;
    MBENCH_CUDA_CHECK(cudaMalloc(&d_A, M * K_TILE * sizeof(__half)));
    MBENCH_CUDA_CHECK(cudaMalloc(&d_B, K_TILE * N * sizeof(__half)));
    MBENCH_CUDA_CHECK(cudaMalloc(&d_output, 64 * sizeof(float)));

    // Copy to device
    MBENCH_CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K_TILE * sizeof(__half), cudaMemcpyHostToDevice));
    MBENCH_CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K_TILE * N * sizeof(__half), cudaMemcpyHostToDevice));

    // Run kernel
    std::cout << "\nRunning wgmma128_fp16_once...\n";
    wgmma128_fp16_once<<<NUM_BLOCKS, NUM_THREADS>>>(d_A, d_B, d_output);
    MBENCH_CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    MBENCH_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, 64 * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify: A[64,16] * B[16,128] with all 1.0 => C[i,j] = 16
    bool verified = verifyOutput(h_output, 16.0f);

    // Cleanup
    MBENCH_CUDA_CHECK(cudaFree(d_A));
    MBENCH_CUDA_CHECK(cudaFree(d_B));
    MBENCH_CUDA_CHECK(cudaFree(d_output));

    return verified;
}

bool bench_wgmma128_fp16() {
    constexpr int M = 64, N = 128, K_TILE = 16;
    constexpr int NUM_THREADS = 128;
    constexpr int NUM_BLOCKS = 1;
    constexpr int ITERATIONS = 10000;

    // Allocate host memory
    std::vector<__half> h_A(M * K_TILE, __float2half(1.0f));
    std::vector<__half> h_B(K_TILE * N, __float2half(1.0f));
    std::vector<float> h_output(64);

    // Allocate device memory
    __half *d_A, *d_B;
    float *d_output;
    int *d_iterations;
    MBENCH_CUDA_CHECK(cudaMalloc(&d_A, M * K_TILE * sizeof(__half)));
    MBENCH_CUDA_CHECK(cudaMalloc(&d_B, K_TILE * N * sizeof(__half)));
    MBENCH_CUDA_CHECK(cudaMalloc(&d_output, 64 * sizeof(float)));
    MBENCH_CUDA_CHECK(cudaMalloc(&d_iterations, sizeof(int)));

    // Copy to device
    MBENCH_CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K_TILE * sizeof(__half), cudaMemcpyHostToDevice));
    MBENCH_CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K_TILE * N * sizeof(__half), cudaMemcpyHostToDevice));
    MBENCH_CUDA_CHECK(cudaMemcpy(d_iterations, &ITERATIONS, sizeof(int), cudaMemcpyHostToDevice));

    // Warmup
    wgmma128_fp16_bench<<<NUM_BLOCKS, NUM_THREADS>>>(d_A, d_B, d_output, 10);
    MBENCH_CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark with timing
    cudaEvent_t start, stop;
    MBENCH_CUDA_CHECK(cudaEventCreate(&start));
    MBENCH_CUDA_CHECK(cudaEventCreate(&stop));

    MBENCH_CUDA_CHECK(cudaEventRecord(start));
    wgmma128_fp16_bench<<<NUM_BLOCKS, NUM_THREADS>>>(d_A, d_B, d_output, ITERATIONS);
    MBENCH_CUDA_CHECK(cudaEventRecord(stop));
    MBENCH_CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    MBENCH_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // Copy result back
    MBENCH_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, 64 * sizeof(float), cudaMemcpyDeviceToHost));

    // Calculate throughput
    // Each WGMMA: 64x128x16 = 131072 FP16 operations
    // Total operations = 2 * 131072 * ITERATIONS (2x for multiply-accumulate)
    double total_ops = 2 * 131072.0 * ITERATIONS;
    double elapsed_sec = elapsed_ms / 1000.0;
    double throughput_tflops = (total_ops / 1e12) / elapsed_sec;

    std::cout << "\nBenchmark Results (wgmma128_fp16_bench):\n";
    std::cout << "  Iterations: " << ITERATIONS << "\n";
    std::cout << "  Elapsed time: " << elapsed_ms << " ms\n";
    std::cout << "  Total operations: " << total_ops / 1e9 << " G ops\n";
    std::cout << "  Throughput: " << throughput_tflops << " TFLOPS\n";

    // Cleanup
    MBENCH_CUDA_CHECK(cudaEventDestroy(start));
    MBENCH_CUDA_CHECK(cudaEventDestroy(stop));
    MBENCH_CUDA_CHECK(cudaFree(d_A));
    MBENCH_CUDA_CHECK(cudaFree(d_B));
    MBENCH_CUDA_CHECK(cudaFree(d_output));
    MBENCH_CUDA_CHECK(cudaFree(d_iterations));

    return true;
}

bool run_wgmma128_bf16_once() {
    constexpr int M = 64, N = 128, K_TILE = 16;
    constexpr int NUM_THREADS = 128;
    constexpr int NUM_BLOCKS = 1;

    // Allocate host memory
    std::vector<__nv_bfloat16> h_A(M * K_TILE, __float2bfloat16(1.0f));
    std::vector<__nv_bfloat16> h_B(K_TILE * N, __float2bfloat16(1.0f));
    std::vector<float> h_output(64);

    // Allocate device memory
    __nv_bfloat16 *d_A, *d_B;
    float *d_output;
    MBENCH_CUDA_CHECK(cudaMalloc(&d_A, M * K_TILE * sizeof(__nv_bfloat16)));
    MBENCH_CUDA_CHECK(cudaMalloc(&d_B, K_TILE * N * sizeof(__nv_bfloat16)));
    MBENCH_CUDA_CHECK(cudaMalloc(&d_output, 64 * sizeof(float)));

    // Copy to device
    MBENCH_CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K_TILE * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    MBENCH_CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K_TILE * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    // Run kernel
    std::cout << "\nRunning wgmma128_bf16_once...\n";
    wgmma128_bf16_once<<<NUM_BLOCKS, NUM_THREADS>>>(d_A, d_B, d_output);
    MBENCH_CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    MBENCH_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, 64 * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify: A[64,16] * B[16,128] with all 1.0 => C[i,j] = 16
    bool verified = verifyOutput(h_output, 16.0f);

    // Cleanup
    MBENCH_CUDA_CHECK(cudaFree(d_A));
    MBENCH_CUDA_CHECK(cudaFree(d_B));
    MBENCH_CUDA_CHECK(cudaFree(d_output));

    return verified;
}

bool bench_wgmma128_bf16() {
    constexpr int M = 64, N = 128, K_TILE = 16;
    constexpr int NUM_THREADS = 128;
    constexpr int NUM_BLOCKS = 1;
    constexpr int ITERATIONS = 10000;

    // Allocate host memory
    std::vector<__nv_bfloat16> h_A(M * K_TILE, __float2bfloat16(1.0f));
    std::vector<__nv_bfloat16> h_B(K_TILE * N, __float2bfloat16(1.0f));
    std::vector<float> h_output(64);

    // Allocate device memory
    __nv_bfloat16 *d_A, *d_B;
    float *d_output;
    int *d_iterations;
    MBENCH_CUDA_CHECK(cudaMalloc(&d_A, M * K_TILE * sizeof(__nv_bfloat16)));
    MBENCH_CUDA_CHECK(cudaMalloc(&d_B, K_TILE * N * sizeof(__nv_bfloat16)));
    MBENCH_CUDA_CHECK(cudaMalloc(&d_output, 64 * sizeof(float)));
    MBENCH_CUDA_CHECK(cudaMalloc(&d_iterations, sizeof(int)));

    // Copy to device
    MBENCH_CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K_TILE * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    MBENCH_CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K_TILE * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    MBENCH_CUDA_CHECK(cudaMemcpy(d_iterations, &ITERATIONS, sizeof(int), cudaMemcpyHostToDevice));

    // Warmup
    wgmma128_bf16_bench<<<NUM_BLOCKS, NUM_THREADS>>>(d_A, d_B, d_output, 10);
    MBENCH_CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark with timing
    cudaEvent_t start, stop;
    MBENCH_CUDA_CHECK(cudaEventCreate(&start));
    MBENCH_CUDA_CHECK(cudaEventCreate(&stop));

    MBENCH_CUDA_CHECK(cudaEventRecord(start));
    wgmma128_bf16_bench<<<NUM_BLOCKS, NUM_THREADS>>>(d_A, d_B, d_output, ITERATIONS);
    MBENCH_CUDA_CHECK(cudaEventRecord(stop));
    MBENCH_CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    MBENCH_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // Copy result back
    MBENCH_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, 64 * sizeof(float), cudaMemcpyDeviceToHost));

    // Calculate throughput
    // Each WGMMA: 64x128x16 = 131072 BF16 operations
    // Total operations = 2 * 131072 * ITERATIONS (2x for multiply-accumulate)
    double total_ops = 2 * 131072.0 * ITERATIONS;
    double elapsed_sec = elapsed_ms / 1000.0;
    double throughput_tflops = (total_ops / 1e12) / elapsed_sec;

    std::cout << "\nBenchmark Results (wgmma128_bf16_bench):\n";
    std::cout << "  Iterations: " << ITERATIONS << "\n";
    std::cout << "  Elapsed time: " << elapsed_ms << " ms\n";
    std::cout << "  Total operations: " << total_ops / 1e9 << " G ops\n";
    std::cout << "  Throughput: " << throughput_tflops << " TFLOPS\n";

    // Cleanup
    MBENCH_CUDA_CHECK(cudaEventDestroy(start));
    MBENCH_CUDA_CHECK(cudaEventDestroy(stop));
    MBENCH_CUDA_CHECK(cudaFree(d_A));
    MBENCH_CUDA_CHECK(cudaFree(d_B));
    MBENCH_CUDA_CHECK(cudaFree(d_output));
    MBENCH_CUDA_CHECK(cudaFree(d_iterations));

    return true;
}

int main() {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "WGMMA Microbenchmark - Verification and Performance\n";
    std::cout << std::string(80, '=') << "\n";

    // ========== m64n64k16 Tests ==========
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "m64n64k16 Tests\n";
    std::cout << std::string(80, '=') << "\n";

    // Step 1: Verify FP16 correctness
    std::cout << "\n[STEP 1] Verification (FP16)\n";
    std::cout << std::string(80, '-') << "\n";
    bool verified_fp16_64 = run_wgmma64_fp16_once();

    if (!verified_fp16_64) {
        std::cerr << "m64n64k16 FP16 Verification failed!\n";
        return 1;
    }

    // Step 2: Benchmark FP16 performance
    std::cout << "\n[STEP 2] Benchmark (FP16)\n";
    std::cout << std::string(80, '-') << "\n";
    bench_wgmma64_fp16();

    // Step 3: Verify BF16 correctness
    std::cout << "\n[STEP 3] Verification (BF16)\n";
    std::cout << std::string(80, '-') << "\n";
    bool verified_bf16_64 = run_wgmma64_bf16_once();

    if (!verified_bf16_64) {
        std::cerr << "m64n64k16 BF16 Verification failed!\n";
        return 1;
    }

    // Step 4: Benchmark BF16 performance
    std::cout << "\n[STEP 4] Benchmark (BF16)\n";
    std::cout << std::string(80, '-') << "\n";
    bench_wgmma64_bf16();

    // ========== m64n128k16 Tests ==========
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "m64n128k16 Tests\n";
    std::cout << std::string(80, '=') << "\n";

    // Step 5: Verify FP16 correctness
    std::cout << "\n[STEP 5] Verification (FP16)\n";
    std::cout << std::string(80, '-') << "\n";
    bool verified_fp16_128 = run_wgmma128_fp16_once();

    if (!verified_fp16_128) {
        std::cerr << "m64n128k16 FP16 Verification failed!\n";
        return 1;
    }

    // Step 6: Benchmark FP16 performance
    std::cout << "\n[STEP 6] Benchmark (FP16)\n";
    std::cout << std::string(80, '-') << "\n";
    bench_wgmma128_fp16();

    // Step 7: Verify BF16 correctness
    std::cout << "\n[STEP 7] Verification (BF16)\n";
    std::cout << std::string(80, '-') << "\n";
    bool verified_bf16_128 = run_wgmma128_bf16_once();

    if (!verified_bf16_128) {
        std::cerr << "m64n128k16 BF16 Verification failed!\n";
        return 1;
    }

    // Step 8: Benchmark BF16 performance
    std::cout << "\n[STEP 8] Benchmark (BF16)\n";
    std::cout << std::string(80, '-') << "\n";
    bench_wgmma128_bf16();

    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "All tests completed successfully!\n";
    std::cout << std::string(80, '=') << "\n";
    return 0;
}

