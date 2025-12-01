/**
 * @file cuda_gemm_v2_wgmma_tma.cu
 * @brief CUDA V2 GEMM using WGMMA + TMA (based on fast.cu kernel 2)
 *
 * This kernel uses:
 * - WGMMA (Warpgroup Matrix Multiply-Accumulate) for SM90
 * - TMA (Tensor Memory Accelerator) for efficient global->shared loads
 * - Block tile: 64×64 with BK=64
 * - WGMMA M=64, N=64, K=16
 *
 * RC mode layout:
 * - A: M×K row-major -> needs transpose for TMA (becomes col-major K×M)
 * - B: K×N column-major (ldB=K)
 * - C: M×N column-major (ldC=M)
 *
 * Note: Requires SM90 (Hopper) architecture
 */

#include "kebab/cuda/cuda_gemm.h"
#include <cuda_fp16.h>
#include <cuda/barrier>
#include <cuda/std/utility>
#include <cstdio>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>

namespace baseline {

// ============================================================================
// WGMMA and TMA Helper Functions
// ============================================================================

namespace cde = cuda::device::experimental;
using barrier = cuda::barrier<cuda::thread_scope_block>;

// Encode shared memory address for WGMMA descriptor
__device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) {
    return (((x) & 0x3FFFF) >> 0x4);
}

// Create shared memory descriptor for WGMMA
__device__ uint64_t make_smem_desc_v2(__half* ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0x0000000000000000;
    desc |= matrix_descriptor_encode(addr);
    desc |= matrix_descriptor_encode((uint64_t)16) << 16;     // stride in bytes
    desc |= matrix_descriptor_encode((uint64_t)1024) << 32;   // leading dim
    desc |= 1llu << 62;  // 128B swizzle
    return desc;
}

// WGMMA fence/sync primitives
__device__ void warpgroup_arrive_v2() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ void warpgroup_commit_batch_v2() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N>
__device__ void warpgroup_wait_v2() {
    static_assert(N >= 0 && N <= 7, "WGMMA wait: N must be in range [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

// WGMMA 64x64x16 for FP16 (produces FP32 accumulator)
template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ void wgmma64_fp16(float d[4][8], __half* sA, __half* sB) {
    uint64_t desc_a = make_smem_desc_v2(&sA[0]);
    uint64_t desc_b = make_smem_desc_v2(&sB[0]);
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

// ============================================================================
// TMA Tensor Map Creation
// ============================================================================

template <int BlockMajorSize, int BlockMinorSize>
void create_tensor_map_v2(CUtensorMap *tma_map, __half* gmem_ptr,
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
        fprintf(stderr, "ERROR: cuTensorMapEncodeTiled failed with error %d\n", result);
    }
}

template<int BlockMajorSize, int BlockMinorSize>
static inline CUtensorMap* allocate_and_create_tensor_map_v2(
    __half* src, int blocks_height, int blocks_width) {
    CUtensorMap *tma_map_d;
    cudaMalloc(&tma_map_d, sizeof(CUtensorMap));
    CUtensorMap tma_map_host;
    create_tensor_map_v2<BlockMajorSize, BlockMinorSize>(
        &tma_map_host, src, blocks_height, blocks_width);
    cudaMemcpy(tma_map_d, &tma_map_host, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    return tma_map_d;
}

// Kernel configuration
constexpr int V2_BM = 64;
constexpr int V2_BN = 64;
constexpr int V2_BK = 64;
constexpr int V2_WGMMA_M = 64;
constexpr int V2_WGMMA_N = 64;
constexpr int V2_WGMMA_K = 16;
constexpr int V2_NUM_THREADS = 128;

// ============================================================================
// WGMMA+TMA Kernel
// ============================================================================

__global__ void __launch_bounds__(V2_NUM_THREADS)
gemm_v2_wgmma_tma_kernel(int M, int N, int K, __half* C,
                          const CUtensorMap* tensorMapA,
                          const CUtensorMap* tensorMapB) {
    __shared__ alignas(128) __half sA[V2_BM * V2_BK];
    __shared__ alignas(128) __half sB[V2_BK * V2_BN];

    // Accumulator: 64x64 output = 4 x 8 floats per thread (128 threads)
    float d[V2_WGMMA_N / 16][8];
    static_assert(sizeof(d) * 128 == V2_BM * V2_BN * sizeof(float));
    memset(d, 0, sizeof(d));

    const int num_blocks_k = K / V2_BK;
    int num_block_n = blockIdx.x % (N / V2_BN);
    int num_block_m = blockIdx.x / (N / V2_BN);

    // Initialize barriers
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier barA;
    __shared__ barrier barB;

    if (threadIdx.x == 0) {
        init(&barA, blockDim.x);
        init(&barB, blockDim.x);
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    barrier::arrival_token tokenA, tokenB;

    // Main K-loop
    for (int block_k_iter = 0; block_k_iter < num_blocks_k; ++block_k_iter) {
        // TMA Load A and B tiles
        if (threadIdx.x == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(
                &sA[0], tensorMapA, block_k_iter * V2_BK, num_block_m * V2_BM, barA);
            tokenA = cuda::device::barrier_arrive_tx(barA, 1, sizeof(sA));

            cde::cp_async_bulk_tensor_2d_global_to_shared(
                &sB[0], tensorMapB, block_k_iter * V2_BK, num_block_n * V2_BN, barB);
            tokenB = cuda::device::barrier_arrive_tx(barB, 1, sizeof(sB));
        } else {
            tokenA = barA.arrive();
            tokenB = barB.arrive();
        }
        barA.wait(std::move(tokenA));
        barB.wait(std::move(tokenB));
        __syncthreads();

        // WGMMA Compute: 4 iterations of K=16 each (total BK=64)
        warpgroup_arrive_v2();
        wgmma64_fp16<1, 1, 1, 0, 0>(d, &sA[0], &sB[0]);
        wgmma64_fp16<1, 1, 1, 0, 0>(d, &sA[V2_WGMMA_K], &sB[V2_WGMMA_K]);
        wgmma64_fp16<1, 1, 1, 0, 0>(d, &sA[2 * V2_WGMMA_K], &sB[2 * V2_WGMMA_K]);
        wgmma64_fp16<1, 1, 1, 0, 0>(d, &sA[3 * V2_WGMMA_K], &sB[3 * V2_WGMMA_K]);
        warpgroup_commit_batch_v2();
        warpgroup_wait_v2<0>();
    }

    // Store results to global memory (column-major C)
    {
        int tid = threadIdx.x;
        int lane = tid % 32;
        int warp = tid / 32;
        uint32_t row = warp * 16 + lane / 4;
        __half *block_C = C + num_block_n * V2_BN * M + num_block_m * V2_BM;

        for (int m_it = 0; m_it < V2_BM / V2_WGMMA_M; ++m_it) {
            for (int n_it = 0; n_it < V2_BN / V2_WGMMA_N; ++n_it) {
                for (int w = 0; w < V2_WGMMA_N / 16; ++w) {
                    int col = 16 * w + 2 * (tid % 4);
                    #define IDX(i, j) ((j + n_it * V2_WGMMA_N) * M + ((i) + m_it * V2_WGMMA_M))

                    block_C[IDX(row, col)] = __float2half(d[w][0]);
                    block_C[IDX(row, col + 1)] = __float2half(d[w][1]);
                    block_C[IDX(row + 8, col)] = __float2half(d[w][2]);
                    block_C[IDX(row + 8, col + 1)] = __float2half(d[w][3]);
                    block_C[IDX(row, col + 8)] = __float2half(d[w][4]);
                    block_C[IDX(row, col + 9)] = __float2half(d[w][5]);
                    block_C[IDX(row + 8, col + 8)] = __float2half(d[w][6]);
                    block_C[IDX(row + 8, col + 9)] = __float2half(d[w][7]);

                    #undef IDX
                }
            }
        }
    }
}

// ============================================================================
// Host Function
// ============================================================================

// Cached TMA maps (for efficiency across calls with same dimensions)
static CUtensorMap *v2_tma_map_A = nullptr;
static CUtensorMap *v2_tma_map_B = nullptr;
static int v2_prev_m = 0, v2_prev_n = 0, v2_prev_k = 0;

void gemm_v2_wgmma_tma_fp16(const __half* A, const __half* B, __half* C,
                            int M, int N, int K, char lhs_format, char rhs_format,
                            cudaStream_t stream) {
    if (lhs_format != 'R' || rhs_format != 'C') {
        fprintf(stderr, "ERROR: CUDA V2 (WGMMA+TMA) only supports RC mode\n");
        return;
    }

    // Check alignment requirements
    if (M % V2_BM != 0 || N % V2_BN != 0 || K % V2_BK != 0) {
        fprintf(stderr, "ERROR: CUDA V2 requires M,N,K to be multiples of 64\n");
        fprintf(stderr, "       Got: M=%d, N=%d, K=%d\n", M, N, K);
        return;
    }

    // Create or update TMA maps if dimensions changed
    if (!v2_tma_map_A || M != v2_prev_m || N != v2_prev_n || K != v2_prev_k) {
        if (v2_tma_map_A) cudaFree(v2_tma_map_A);
        if (v2_tma_map_B) cudaFree(v2_tma_map_B);

        // Note: For RC mode with TMA, we need A in column-major (K×M)
        // The tensor map describes how to load tiles from column-major storage
        v2_tma_map_A = allocate_and_create_tensor_map_v2<V2_BM, V2_BK>(
            const_cast<__half*>(A), M / V2_BM, K / V2_BK);
        v2_tma_map_B = allocate_and_create_tensor_map_v2<V2_BN, V2_BK>(
            const_cast<__half*>(B), N / V2_BN, K / V2_BK);

        v2_prev_m = M;
        v2_prev_n = N;
        v2_prev_k = K;
    }

    dim3 grid((M / V2_BM) * (N / V2_BN));
    gemm_v2_wgmma_tma_kernel<<<grid, V2_NUM_THREADS, 0, stream>>>(
        M, N, K, const_cast<__half*>(C), v2_tma_map_A, v2_tma_map_B);
}

} // namespace baseline

