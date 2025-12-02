/**
 * @file cuda_gemm_v3_warpgroup.cu
 * @brief CUDA V3 GEMM using Warp Groups + larger tiles (based on fast.cu kernel 3)
 *
 * This kernel uses:
 * - Warp group partitioning for compute
 * - Larger tiles: 128×128×64 (vs V2's 64×64×64)
 * - Dynamic shared memory
 * - WGMMA 64×N×16 where N can be 64/128/192/256
 *
 * RC mode layout:
 * - A: M×K row-major -> loaded as col-major via TMA
 * - B: K×N column-major (ldB=K)
 * - C: M×N column-major (ldC=M)
 *
 * Note: Requires SM90 (Hopper) architecture
 */

#include "kebab/cuda/cuda_gemm.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda/barrier>
#include <cuda/std/utility>
#include <cstdio>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>

namespace baseline {

namespace cde = cuda::device::experimental;
using barrier = cuda::barrier<cuda::thread_scope_block>;

// ============================================================================
// WGMMA Helper Functions (reused from V2 with extensions)
// ============================================================================

__device__ static inline uint64_t matrix_descriptor_encode_v3(uint64_t x) {
    return (((x) & 0x3FFFF) >> 0x4);
}

__device__ uint64_t make_smem_desc_v3(__half* ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0x0000000000000000;
    desc |= matrix_descriptor_encode_v3(addr);
    desc |= matrix_descriptor_encode_v3((uint64_t)16) << 16;
    desc |= matrix_descriptor_encode_v3((uint64_t)1024) << 32;
    desc |= 1llu << 62;  // 128B swizzle
    return desc;
}

__device__ void warpgroup_arrive_v3() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ void warpgroup_commit_batch_v3() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N>
__device__ void warpgroup_wait_v3() {
    static_assert(N >= 0 && N <= 7, "WGMMA wait: N must be in range [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

// WGMMA 64×64×16
template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ void wgmma64_v3(float d[4][8], __half* sA, __half* sB) {
    uint64_t desc_a = make_smem_desc_v3(&sA[0]);
    uint64_t desc_b = make_smem_desc_v3(&sB[0]);
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
          "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
          "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

// WGMMA 64×128×16
template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ void wgmma128_v3(float d[8][8], __half* sA, __half* sB) {
    uint64_t desc_a = make_smem_desc_v3(&sA[0]);
    uint64_t desc_b = make_smem_desc_v3(&sB[0]);
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
          "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
          "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

// Generic wgmma dispatcher
template<int WGMMA_N, int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ inline void wgmma_v3(float d[WGMMA_N/16][8], __half* sA, __half* sB) {
    static_assert(WGMMA_N == 64 || WGMMA_N == 128);
    if constexpr (WGMMA_N == 128)
        wgmma128_v3<ScaleD, ScaleA, ScaleB, TransA, TransB>(d, sA, sB);
    if constexpr (WGMMA_N == 64)
        wgmma64_v3<ScaleD, ScaleA, ScaleB, TransA, TransB>(d, sA, sB);
}

// ============================================================================
// TMA Functions
// ============================================================================

template <int BlockMajorSize, int BlockMinorSize>
void create_tensor_map_v3(CUtensorMap *tma_map, __half* gmem_ptr,
                          int blocks_height, int blocks_width) {
    void* gmem_address = (void*)gmem_ptr;
    uint64_t gmem_prob_shape[5] = {
        (uint64_t)BlockMinorSize * blocks_width,
        (uint64_t)BlockMajorSize * blocks_height,
        1, 1, 1
    };
    uint64_t gmem_prob_stride[5] = {
        sizeof(__half),
        sizeof(__half) * BlockMinorSize * blocks_width,
        0, 0, 0
    };
    uint32_t smem_box_shape[5] = {
        uint32_t(BlockMinorSize), uint32_t(BlockMajorSize), 1, 1, 1
    };
    uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        tma_map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, gmem_address, gmem_prob_shape,
        gmem_prob_stride + 1, smem_box_shape, smem_box_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "ERROR: cuTensorMapEncodeTiled failed: %d\n", result);
    }
}

template<int BlockMajorSize, int BlockMinorSize>
static inline CUtensorMap* allocate_and_create_tensor_map_v3(
    __half* src, int blocks_height, int blocks_width) {
    CUtensorMap *tma_map_d;
    cudaMalloc(&tma_map_d, sizeof(CUtensorMap));
    CUtensorMap tma_map_host;
    create_tensor_map_v3<BlockMajorSize, BlockMinorSize>(
        &tma_map_host, src, blocks_height, blocks_width);
    cudaMemcpy(tma_map_d, &tma_map_host, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    return tma_map_d;
}

// ============================================================================
// V3 Kernel Configuration
// ============================================================================

constexpr int V3_BM = 128;
constexpr int V3_BN = 128;
constexpr int V3_BK = 64;
constexpr int V3_WGMMA_M = 64;
constexpr int V3_WGMMA_N = V3_BN;  // Use full N dimension per WGMMA
constexpr int V3_WGMMA_K = 16;
constexpr int V3_NUM_THREADS = 128;  // 1 warp group

// Shared memory structure
template <int BM, int BN, int BK>
struct SMemV3 {
    alignas(128) __half A[BM * BK];
    alignas(128) __half B[BK * BN];
};

// ============================================================================
// V3 Kernel: Warp Group with larger tiles
// ============================================================================

template<int BM, int BN, int BK, int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
gemm_v3_warpgroup_kernel(int M, int N, int K, __half* C,
                          const CUtensorMap* tensorMapA,
                          const CUtensorMap* tensorMapB) {
    constexpr int WGMMA_M = 64, WGMMA_K = 16, WGMMA_N = BN;
    constexpr int B_WG_M = BM / (NUM_THREADS / 128);  // M per warp group

    extern __shared__ SMemV3<BM, BN, BK> smem[];
    __half *sA = smem->A;
    __half *sB = smem->B;

    // Accumulator: B_WG_M/WGMMA_M x WGMMA_N/16 x 8 floats per thread
    float d[B_WG_M / WGMMA_M][WGMMA_N / 16][8];
    static_assert(sizeof(d) * NUM_THREADS == BM * BN * sizeof(float));
    memset(d, 0, sizeof(d));

    const int num_blocks_k = K / BK;
    int num_block_n = blockIdx.x % (N / BN);
    int num_block_m = blockIdx.x / (N / BN);

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier barA, barB;

    if (threadIdx.x == 0) {
        init(&barA, blockDim.x);
        init(&barB, blockDim.x);
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    int wg_idx = threadIdx.x / 128;
    barrier::arrival_token tokenA, tokenB;

    // Main K-loop
    for (int block_k_iter = 0; block_k_iter < num_blocks_k; ++block_k_iter) {
        // TMA Load
        if (threadIdx.x == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(
                &sA[0], tensorMapA, block_k_iter * BK, num_block_m * BM, barA);
            tokenA = cuda::device::barrier_arrive_tx(barA, 1, BK * BM * sizeof(__half));

            cde::cp_async_bulk_tensor_2d_global_to_shared(
                &sB[0], tensorMapB, block_k_iter * BK, num_block_n * BN, barB);
            tokenB = cuda::device::barrier_arrive_tx(barB, 1, BK * BN * sizeof(__half));
        } else {
            tokenA = barA.arrive();
            tokenB = barB.arrive();
        }
        barA.wait(std::move(tokenA));
        barB.wait(std::move(tokenB));
        __syncthreads();

        // WGMMA Compute
        warpgroup_arrive_v3();
        #pragma unroll
        for (int m_it = 0; m_it < B_WG_M / WGMMA_M; ++m_it) {
            __half *wgmma_sA = sA + BK * (m_it + wg_idx * B_WG_M / WGMMA_M) * WGMMA_M;
            #pragma unroll
            for (int k_it = 0; k_it < BK / WGMMA_K; ++k_it) {
                wgmma_v3<WGMMA_N, 1, 1, 1, 0, 0>(
                    d[m_it], &wgmma_sA[k_it * WGMMA_K], &sB[k_it * WGMMA_K]);
            }
        }
        warpgroup_commit_batch_v3();
        warpgroup_wait_v3<0>();
    }

    // Store results to global memory (column-major C)
    {
        uint32_t tid = threadIdx.x % 128;
        uint32_t lane = tid & 31;
        uint32_t warp = tid / 32;
        uint32_t row = warp * 16 + lane / 4;

        __half *block_C = C + num_block_n * BN * M + num_block_m * BM;

        #pragma unroll
        for (uint32_t m_it = 0; m_it < B_WG_M / WGMMA_M; ++m_it) {
            int yo = m_it * WGMMA_M + wg_idx * B_WG_M;
            #pragma unroll
            for (uint32_t w = 0; w < WGMMA_N / 16; ++w) {
                int col = 16 * w + 2 * (tid % 4);
                #define IDX(i, j) ((j) * M + ((i) + yo))

                block_C[IDX(row, col)] = __float2half(d[m_it][w][0]);
                block_C[IDX(row, col + 1)] = __float2half(d[m_it][w][1]);
                block_C[IDX(row + 8, col)] = __float2half(d[m_it][w][2]);
                block_C[IDX(row + 8, col + 1)] = __float2half(d[m_it][w][3]);
                block_C[IDX(row, col + 8)] = __float2half(d[m_it][w][4]);
                block_C[IDX(row, col + 9)] = __float2half(d[m_it][w][5]);
                block_C[IDX(row + 8, col + 8)] = __float2half(d[m_it][w][6]);
                block_C[IDX(row + 8, col + 9)] = __float2half(d[m_it][w][7]);

                #undef IDX
            }
        }
    }
}

// ============================================================================
// Host Function
// ============================================================================

static CUtensorMap *v3_tma_map_A = nullptr;
static CUtensorMap *v3_tma_map_B = nullptr;
static int v3_prev_m = 0, v3_prev_n = 0, v3_prev_k = 0;

void gemm_v3_warpgroup_fp16(const __half* A, const __half* B, __half* C,
                            int M, int N, int K, char lhs_format, char rhs_format,
                            cudaStream_t stream) {
    if (lhs_format != 'R' || rhs_format != 'C') {
        fprintf(stderr, "ERROR: CUDA V3 (warp group) only supports RC mode\n");
        return;
    }

    // Check alignment requirements
    if (M % V3_BM != 0 || N % V3_BN != 0 || K % V3_BK != 0) {
        fprintf(stderr, "ERROR: CUDA V3 requires M,N to be multiples of 128, K multiple of 64\n");
        fprintf(stderr, "       Got: M=%d, N=%d, K=%d\n", M, N, K);
        return;
    }

    // Create or update TMA maps if dimensions changed
    if (!v3_tma_map_A || M != v3_prev_m || N != v3_prev_n || K != v3_prev_k) {
        if (v3_tma_map_A) cudaFree(v3_tma_map_A);
        if (v3_tma_map_B) cudaFree(v3_tma_map_B);

        v3_tma_map_A = allocate_and_create_tensor_map_v3<V3_BM, V3_BK>(
            const_cast<__half*>(A), M / V3_BM, K / V3_BK);
        v3_tma_map_B = allocate_and_create_tensor_map_v3<V3_BN, V3_BK>(
            const_cast<__half*>(B), N / V3_BN, K / V3_BK);

        v3_prev_m = M;
        v3_prev_n = N;
        v3_prev_k = K;
    }

    size_t sMemSize = sizeof(SMemV3<V3_BM, V3_BN, V3_BK>);
    auto kernel = gemm_v3_warpgroup_kernel<V3_BM, V3_BN, V3_BK, V3_NUM_THREADS>;

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize);

    dim3 grid((M / V3_BM) * (N / V3_BN));
    kernel<<<grid, V3_NUM_THREADS, sMemSize, stream>>>(
        M, N, K, const_cast<__half*>(C), v3_tma_map_A, v3_tma_map_B);
}

// ============================================================================
// BFloat16 Version
// ============================================================================

__device__ uint64_t make_smem_desc_v3_bf16(__nv_bfloat16* ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0x0000000000000000;
    desc |= matrix_descriptor_encode_v3(addr);
    desc |= matrix_descriptor_encode_v3((uint64_t)16) << 16;
    desc |= matrix_descriptor_encode_v3((uint64_t)1024) << 32;
    desc |= 1llu << 62;
    return desc;
}

template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ void wgmma64_v3_bf16(float d[4][8], __nv_bfloat16* sA, __nv_bfloat16* sB) {
    uint64_t desc_a = make_smem_desc_v3_bf16(&sA[0]);
    uint64_t desc_b = make_smem_desc_v3_bf16(&sB[0]);
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
          "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
          "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ void wgmma128_v3_bf16(float d[8][8], __nv_bfloat16* sA, __nv_bfloat16* sB) {
    uint64_t desc_a = make_smem_desc_v3_bf16(&sA[0]);
    uint64_t desc_b = make_smem_desc_v3_bf16(&sB[0]);
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
          "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
          "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

template<int WGMMA_N, int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ inline void wgmma_v3_bf16(float d[WGMMA_N/16][8], __nv_bfloat16* sA, __nv_bfloat16* sB) {
    static_assert(WGMMA_N == 64 || WGMMA_N == 128);
    if constexpr (WGMMA_N == 128)
        wgmma128_v3_bf16<ScaleD, ScaleA, ScaleB, TransA, TransB>(d, sA, sB);
    if constexpr (WGMMA_N == 64)
        wgmma64_v3_bf16<ScaleD, ScaleA, ScaleB, TransA, TransB>(d, sA, sB);
}

template <int BlockMajorSize, int BlockMinorSize>
void create_tensor_map_v3_bf16(CUtensorMap *tma_map, __nv_bfloat16* gmem_ptr,
                                int blocks_height, int blocks_width) {
    void* gmem_address = (void*)gmem_ptr;
    uint64_t gmem_prob_shape[5] = {
        (uint64_t)BlockMinorSize * blocks_width,
        (uint64_t)BlockMajorSize * blocks_height,
        1, 1, 1
    };
    uint64_t gmem_prob_stride[5] = {
        sizeof(__nv_bfloat16),
        sizeof(__nv_bfloat16) * BlockMinorSize * blocks_width,
        0, 0, 0
    };
    uint32_t smem_box_shape[5] = {
        uint32_t(BlockMinorSize), uint32_t(BlockMajorSize), 1, 1, 1
    };
    uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        tma_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, gmem_address, gmem_prob_shape,
        gmem_prob_stride + 1, smem_box_shape, smem_box_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "ERROR: cuTensorMapEncodeTiled (bf16) failed: %d\n", result);
    }
}

template<int BlockMajorSize, int BlockMinorSize>
static inline CUtensorMap* allocate_and_create_tensor_map_v3_bf16(
    __nv_bfloat16* src, int blocks_height, int blocks_width) {
    CUtensorMap *tma_map_d;
    cudaMalloc(&tma_map_d, sizeof(CUtensorMap));
    CUtensorMap tma_map_host;
    create_tensor_map_v3_bf16<BlockMajorSize, BlockMinorSize>(
        &tma_map_host, src, blocks_height, blocks_width);
    cudaMemcpy(tma_map_d, &tma_map_host, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    return tma_map_d;
}

template<int BM, int BN, int BK>
struct SMemV3_bf16 {
    alignas(128) __nv_bfloat16 A[BM * BK];
    alignas(128) __nv_bfloat16 B[BK * BN];
};

template<int BM, int BN, int BK, int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
gemm_v3_warpgroup_kernel_bf16(int M, int N, int K, __nv_bfloat16* C,
                               CUtensorMap* tensorMapA, CUtensorMap* tensorMapB) {
    constexpr int WGMMA_M = 64;
    constexpr int WGMMA_N = BN;
    constexpr int WGMMA_K = 16;

    extern __shared__ __align__(128) uint8_t smem_bf16[];
    SMemV3_bf16<BM, BN, BK>& s = *reinterpret_cast<SMemV3_bf16<BM, BN, BK>*>(smem_bf16);

    int num_block_n = N / BN;
    int num_block_k = K / BK;
    int block_id = blockIdx.x;
    int block_m = block_id / num_block_n;
    int block_n = block_id % num_block_n;

    __nv_bfloat16* sA = s.A;
    __nv_bfloat16* sB = s.B;

    barrier* bar = reinterpret_cast<barrier*>(smem_bf16 + sizeof(SMemV3_bf16<BM, BN, BK>));
    if (threadIdx.x == 0) {
        init(bar, blockDim.x);
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    float d[BM/WGMMA_M][WGMMA_N/16][8] = {0};

    for (int block_k = 0; block_k < num_block_k; ++block_k) {
        if (threadIdx.x == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(sA, tensorMapA, block_k, block_m, *bar);
            cde::cp_async_bulk_tensor_2d_global_to_shared(sB, tensorMapB, block_k, block_n, *bar);
            cde::cp_async_bulk_commit_group();
        }
        bar->arrive_and_wait();

        warpgroup_arrive_v3();
        for (int k_inner = 0; k_inner < BK / WGMMA_K; ++k_inner) {
            for (int m_inner = 0; m_inner < BM / WGMMA_M; ++m_inner) {
                __nv_bfloat16* wgmma_sA = sA + k_inner * WGMMA_K * BM + m_inner * WGMMA_M;
                __nv_bfloat16* wgmma_sB = sB + k_inner * WGMMA_K * BN;
                if (block_k == 0 && k_inner == 0) {
                    wgmma_v3_bf16<WGMMA_N, 0, 1, 1, 0, 0>(d[m_inner], wgmma_sA, wgmma_sB);
                } else {
                    wgmma_v3_bf16<WGMMA_N, 1, 1, 1, 0, 0>(d[m_inner], wgmma_sA, wgmma_sB);
                }
            }
        }
        warpgroup_commit_batch_v3();
        warpgroup_wait_v3<0>();

        __syncthreads();
    }

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    for (int m_tile = 0; m_tile < BM / WGMMA_M; ++m_tile) {
        int row_base = block_m * BM + m_tile * WGMMA_M + (warp_id / 2) * 16 + (lane_id / 4);
        int col_base = block_n * BN + (warp_id % 2) * 32 + (lane_id % 4) * 2;

        for (int n_tile = 0; n_tile < WGMMA_N / 16; ++n_tile) {
            for (int j = 0; j < 8; j += 2) {
                int row = row_base + (n_tile / 2) * 8;
                int col = col_base + (n_tile % 2) * 8 + j;
                if (row < M && col < N) {
                    C[col * M + row] = __float2bfloat16(d[m_tile][n_tile][j]);
                }
                if (row < M && col + 1 < N) {
                    C[(col + 1) * M + row] = __float2bfloat16(d[m_tile][n_tile][j + 1]);
                }
            }
        }
    }
}

static CUtensorMap *v3_tma_map_A_bf16 = nullptr;
static CUtensorMap *v3_tma_map_B_bf16 = nullptr;
static int v3_prev_m_bf16 = 0, v3_prev_n_bf16 = 0, v3_prev_k_bf16 = 0;

void gemm_v3_warpgroup_bf16(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
                            int M, int N, int K, char lhs_format, char rhs_format,
                            cudaStream_t stream) {
    if (lhs_format != 'R' || rhs_format != 'C') {
        fprintf(stderr, "ERROR: CUDA V3 BF16 (warpgroup) only supports RC mode\n");
        return;
    }

    if (M % V3_BM != 0 || N % V3_BN != 0 || K % V3_BK != 0) {
        fprintf(stderr, "ERROR: CUDA V3 BF16 requires M,N divisible by 128, K by 64\n");
        fprintf(stderr, "       Got: M=%d, N=%d, K=%d\n", M, N, K);
        return;
    }

    if (!v3_tma_map_A_bf16 || M != v3_prev_m_bf16 || N != v3_prev_n_bf16 || K != v3_prev_k_bf16) {
        if (v3_tma_map_A_bf16) cudaFree(v3_tma_map_A_bf16);
        if (v3_tma_map_B_bf16) cudaFree(v3_tma_map_B_bf16);

        v3_tma_map_A_bf16 = allocate_and_create_tensor_map_v3_bf16<V3_BM, V3_BK>(
            const_cast<__nv_bfloat16*>(A), M / V3_BM, K / V3_BK);
        v3_tma_map_B_bf16 = allocate_and_create_tensor_map_v3_bf16<V3_BN, V3_BK>(
            const_cast<__nv_bfloat16*>(B), N / V3_BN, K / V3_BK);

        v3_prev_m_bf16 = M;
        v3_prev_n_bf16 = N;
        v3_prev_k_bf16 = K;
    }

    size_t sMemSize = sizeof(SMemV3_bf16<V3_BM, V3_BN, V3_BK>);
    auto kernel = gemm_v3_warpgroup_kernel_bf16<V3_BM, V3_BN, V3_BK, V3_NUM_THREADS>;

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize);

    dim3 grid((M / V3_BM) * (N / V3_BN));
    kernel<<<grid, V3_NUM_THREADS, sMemSize, stream>>>(
        M, N, K, const_cast<__nv_bfloat16*>(C), v3_tma_map_A_bf16, v3_tma_map_B_bf16);
}

} // namespace baseline

