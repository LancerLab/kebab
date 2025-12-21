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
#include <cuda_bf16.h>
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

// --------------- WGMMA primitives (SM90+) ---------------
// refer to:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-leading-dimension-byte-offset
// 9.7.15.5.1.2.2. Matrix Descriptor Format
// SWIZZLE pattern enum
enum class WGMMA_Swizzle : uint64_t {
  NS = 0,  // No swizzle
  B32 = 3, // 32B swizzle
  B64 = 2, // 64B swizzle
  B128 = 1 // 128B swizzle
};

// Major order enum
enum class WGMMA_MajorOrder {
  K_MAJOR, // K dimension is major (leading)
  MN_MAJOR // M and N dimensions are major (leading)
};

__device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) {
    return (((x) & 0x3FFFF) >> 0x4);
}

template <WGMMA_MajorOrder MajorOrder, WGMMA_Swizzle Swizzle, typename T>
__device__ static inline uint64_t make_smem_desc(T* ptr) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
  uint64_t desc = 0x0000000000000000;
  desc |= matrix_descriptor_encode(addr);

  // Determine stride and leading dimension based on major order and swizzle
  uint64_t stride_bytes = 0;
  uint64_t leading_dim = 0;

  if constexpr (MajorOrder == WGMMA_MajorOrder::K_MAJOR) {
    // K-major layout: stride varies by swizzle pattern
    switch (Swizzle) {
    case WGMMA_Swizzle::NS:
      stride_bytes = 128;
      leading_dim = 64;
      break;
    case WGMMA_Swizzle::B32:
      stride_bytes = 16;
      leading_dim = 256;
      break;
    case WGMMA_Swizzle::B64:
      stride_bytes = 16;
      leading_dim = 512;
      break;
    case WGMMA_Swizzle::B128:
      stride_bytes = 16;
      leading_dim = 1024;
      break;
    }
  }
  // TODO: MN-major not handled
  
  desc |= matrix_descriptor_encode(stride_bytes) << 16;
  desc |= matrix_descriptor_encode(leading_dim) << 32;
  desc |= static_cast<uint64_t>(Swizzle) << 62;

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

// ============================================================================
// GMMA Descriptor Generation (based on CUTLASS make_gmma_desc logic)
// ============================================================================
//
// GmmaDescriptor bit layout (from cute/arch/mma_sm90_desc.hpp):
//   bits [0,14)   : start_address_ (smem addr >> 4)
//   bits [16,30)  : leading_byte_offset_ (in uint128_t units, i.e., bytes >> 4)
//   bits [32,46)  : stride_byte_offset_ (in uint128_t units, i.e., bytes >> 4)
//   bits [49,52)  : base_offset_ (always 0)
//   bits [62,64)  : layout_type_ (B128=1, B64=2, B32=3, INTERLEAVE=0)
//
// For SW128 (B128) layouts:
//   layout_type = 1
//   W = 8 (width factor)
//
// K-major B128 layout ((8,n),(T,2)) in uint128_t:
//   - T = 8 (128 bits / 16 bits per half_t = 8 half_t per uint128_t)
//   - For 64x64 tile with half_t, K=64 means 64/8=8 uint128_t per row
//   - For WGMMA k=16, that's 16/8=2 uint128_t
//   - stride_byte_offset = stride between rows = 8 uint128_t (for 64-wide K)
//   - leading_byte_offset = 1 (K stride is contiguous)
//
// MN-major B128 layout ((T,8,n),(8,k)) in uint128_t:
//   - For half_t: T = 8
//   - stride_byte_offset = k-stride (between K tiles)
//   - leading_byte_offset = 1 (MN stride within 8-element group)

// Create GMMA descriptor for MN-major layout
// MN-major means M/N dimension is contiguous in memory
// Values derived from CUTE kernel debug output:
//   - LBO = 1, SBO = 64, layout_type = 1 (B128)
__device__ uint64_t make_smem_desc_mn(__half* ptr) {
    uint64_t desc = 0;
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));

    // Encode fields (values from CUTE kernel debug output)
    uint64_t start_address = (addr >> 4) & 0x3FFF;         // bits [0,14)
    uint64_t leading_byte_offset = 1;                       // bits [16,30), LBO = 1
    uint64_t stride_byte_offset = 64;                       // bits [32,46), SBO = 64
    uint64_t base_offset = 0;                               // bits [49,52)
    uint64_t layout_type = 1;                               // bits [62,64), B128

    desc = start_address
         | (leading_byte_offset << 16)
         | (stride_byte_offset << 32)
         | (base_offset << 49)
         | (layout_type << 62);

    return desc;
}

// Create GMMA descriptor for K-major layout
// K-major means K dimension is contiguous in memory
// Values derived from CUTE kernel debug output:
//   - LBO = 0, SBO = 128, layout_type = 1 (B128)
__device__ uint64_t make_smem_desc_k(__half* ptr) {
    uint64_t desc = 0;
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));

    // Encode fields (values from CUTE kernel debug output)
    uint64_t start_address = (addr >> 4) & 0x3FFF;         // bits [0,14)
    uint64_t leading_byte_offset = 1;                       // bits [16,30), LBO = 0
    // uint64_t stride_byte_offset = 64;                      // bits [32,46), SBO = 128
    // uint64_t stride_byte_offset = 32;                      // bits [32,46), SBO = 128
    uint64_t stride_byte_offset = 16;                      // bits [32,46), SBO = 128
    uint64_t base_offset = 0;                               // bits [49,52)
    uint64_t swizzle_type = 3;                               // bits [62,64), B128

    desc = start_address
         | (leading_byte_offset << 16)
         | (stride_byte_offset << 32)
         | (base_offset << 49)
         | (swizzle_type << 62);

    return desc;
}

// WGMMA 64x64x16 for FP16 (produces FP32 accumulator)
// TransB=0: RC mode (A row-major, B col-major) -> both use MN-major descriptor
// TransB=1: RR mode (A row-major, B row-major) -> A uses MN-major, B uses K-major
template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ void wgmma64_fp16(float d[4][8], __half* sA, __half* sB) {
    if constexpr (TransB == 0) {
        // RC mode: both A and B use MN-major layout
        uint64_t desc_a = make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B32, __half>(&sA[0]);
        uint64_t desc_b = make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B32, __half>(&sB[0]);
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
    // } else if constexpr (TransB == 1) {
    //     assert(false);
    //     // RR mode: A uses MN-major, B uses K-major
    //     uint64_t desc_a = make_smem_desc_mn(&sA[0]);
    //     uint64_t desc_b = make_smem_desc_k(&sB[0]);

    //     // // Debug output for first warp
    //     // if (blockIdx.x == 0 && threadIdx.x == 0) {
    //     //     printf("RR WGMMA descriptors:\n");
    //     //     printf("  desc_a: 0x%016llx (LBO=%llu, SBO=%llu)\n",
    //     //            (unsigned long long)desc_a,
    //     //            (unsigned long long)((desc_a >> 16) & 0x3FFF),
    //     //            (unsigned long long)((desc_a >> 32) & 0x3FFF));
    //     //     printf("  desc_b: 0x%016llx (LBO=%llu, SBO=%llu)\n",
    //     //            (unsigned long long)desc_b,
    //     //            (unsigned long long)((desc_b >> 16) & 0x3FFF),
    //     //            (unsigned long long)((desc_b >> 32) & 0x3FFF));
    //     // }

    //     asm volatile(
    //         "{\n"
    //         "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
    //         "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
    //         " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
    //         " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
    //         " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31},"
    //         " %32,"
    //         " %33,"
    //         " %34, %35, %36, %37, %38;\n"
    //         "}\n"
    //         : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
    //         "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
    //         "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
    //         "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
    //         "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
    //         "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
    //         "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
    //         "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7])
    //         : "l"(desc_a), "l"(desc_b),
    //         "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
    //         "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
    }

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
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_32B,
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
constexpr int V2_BK = 16;
constexpr int V2_WGMMA_M = 64;
constexpr int V2_WGMMA_N = 64;
constexpr int V2_WGMMA_K = 16;
constexpr int V2_NUM_THREADS = 128;

// ============================================================================
// WGMMA+TMA Kernel
// ============================================================================

template<WGMMA_MajorOrder MajorOrderA, WGMMA_MajorOrder MajorOrderB>
__global__ void __launch_bounds__(V2_NUM_THREADS)
gemm_v2_wgmma_tma_kernel(int M, int N, int K, __half* C,
                          const CUtensorMap* tensorMapA,
                          const CUtensorMap* tensorMapB) {
    if constexpr (MajorOrderA == WGMMA_MajorOrder::K_MAJOR && MajorOrderB == WGMMA_MajorOrder::K_MAJOR) {
        // Shared memory for A and B tiles
        __shared__ alignas(128) __half sA[V2_BM * V2_BK];
        __shared__ alignas(128) __half sB[V2_BK * V2_BN];

        // Accumulator: 64x64 output = 4 x 8 floats per thread (128 threads)
        float d[V2_BM/V2_WGMMA_M][V2_BN/V2_WGMMA_N][V2_WGMMA_N / 16][8];
        // static_assert(sizeof(d) * 128 == V2_BM * V2_BN * sizeof(float));
        memset(d, 0, sizeof(d));

        const int num_blocks_k = K / V2_BK;
        int num_block_n = blockIdx.x % (N / V2_BN);
        int num_block_m = blockIdx.x / (N / V2_BN);

        // Initialize barriers
        #pragma nv_diag_suppress static_var_with_dynamic_init

        // TMA barriers
        __shared__ barrier barA;
        __shared__ barrier barB;

        if (threadIdx.x == 0) {
            init(&barA, blockDim.x);
            init(&barB, blockDim.x);
            cde::fence_proxy_async_shared_cta();
        }
        __syncthreads();
        // TMA barriers end

        barrier::arrival_token tokenA, tokenB;

        // Main K-loop
        // iterate over K dim, each time, one warpgroup load a tile of A and B from global to shared mem
        // A is 64x64 tile, B is 64x64 tile
        // then perform 4 unrolled WGMMA operation, each time, one warpgroup compute a 64x64 tile of C
        // NOTE: here we send different offset to wgmma, is equivalent to cute way without calculate offset
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
            for (int m_iter = 0; m_iter < V2_BM / V2_WGMMA_M; ++m_iter) {
            for (int n_iter = 0; n_iter < V2_BN / V2_WGMMA_N; ++n_iter) {
                wgmma64_fp16<1, 1, 1, 0, 0>(d[m_iter][n_iter], &sA[m_iter * V2_WGMMA_M * V2_BK + 0], &sB[n_iter * V2_WGMMA_N * V2_BK + 0]);
                // wgmma64_fp16<1, 1, 1, 0, 0>(d[m_iter][n_iter], &sA[m_iter * V2_WGMMA_M * V2_BK + V2_WGMMA_K], &sB[n_iter * V2_WGMMA_N * V2_BK + V2_WGMMA_K]);
            }
            }
            // wgmma64_fp16<1, 1, 1, 0, 0>(d, &sA[2 * V2_WGMMA_K], &sB[2 * V2_WGMMA_K]);
            // wgmma64_fp16<1, 1, 1, 0, 0>(d, &sA[3 * V2_WGMMA_K], &sB[3 * V2_WGMMA_K]);
            // wgmma64_fp16<1, 1, 1, 0, 0>(d, &sA[4 * V2_WGMMA_K], &sB[4 * V2_WGMMA_K]);
            // wgmma64_fp16<1, 1, 1, 0, 0>(d, &sA[5 * V2_WGMMA_K], &sB[5 * V2_WGMMA_K]);
            // wgmma64_fp16<1, 1, 1, 0, 0>(d, &sA[6 * V2_WGMMA_K], &sB[6 * V2_WGMMA_K]);
            // wgmma64_fp16<1, 1, 1, 0, 0>(d, &sA[7 * V2_WGMMA_K], &sB[7 * V2_WGMMA_K]);
            warpgroup_commit_batch_v2();
            warpgroup_wait_v2<0>();
        }

        // Store results to global memory (column-major C)
        // lane: 0-31, warp: 0-3
        // each warp handles 16 rows, thus row_base = warp * 16
        // every 4 lane (0-3) handles same row, row_base + lane / 4
        // each 4 lane team, handles 2 rows, (row, ...) and (row+8, ...)
        // this is easy to calcute: 32threads together handle 16rows, 4 threads per row, need 2 iterations
        // colwise:
        // group per 16 col, iterates 4 times, w = 0-3, col_base = w * 16, each time 4 lane handles 2 col
        // tid 0-3 handles 1 row 16 col, each handle (row, col), (row, col+1), (row, col+8), (row, col+9)
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

                        block_C[IDX(row, col)] = __float2half(d[m_it][n_it][w][0]);
                        block_C[IDX(row, col + 1)] = __float2half(d[m_it][n_it][w][1]);
                        block_C[IDX(row + 8, col)] = __float2half(d[m_it][n_it][w][2]);
                        block_C[IDX(row + 8, col + 1)] = __float2half(d[m_it][n_it][w][3]);
                        block_C[IDX(row, col + 8)] = __float2half(d[m_it][n_it][w][4]);
                        block_C[IDX(row, col + 9)] = __float2half(d[m_it][n_it][w][5]);
                        block_C[IDX(row + 8, col + 8)] = __float2half(d[m_it][n_it][w][6]);
                        block_C[IDX(row + 8, col + 9)] = __float2half(d[m_it][n_it][w][7]);

                        #undef IDX
                    }
                }
            }
        }
    } else {
        // Not implemented
        assert(false);
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
    if ((lhs_format != 'R' || rhs_format != 'C') &&
        (lhs_format != 'R' || rhs_format != 'R')) {
        fprintf(stderr, "ERROR: CUDA V2 (WGMMA+TMA) only supports RR, RC mode\n");
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

        if (lhs_format == 'R' && rhs_format == 'C') {
            v2_tma_map_A = allocate_and_create_tensor_map_v2<V2_BM, V2_BK>(
                const_cast<__half*>(A), M / V2_BM, K / V2_BK);
            v2_tma_map_B = allocate_and_create_tensor_map_v2<V2_BN, V2_BK>(
                const_cast<__half*>(B), N / V2_BN, K / V2_BK);
        } else if (lhs_format == 'R' && rhs_format == 'R') {
            v2_tma_map_A = allocate_and_create_tensor_map_v2<V2_BM, V2_BK>(
                const_cast<__half*>(A), M / V2_BM, K / V2_BK);
        }

        v2_prev_m = M;
        v2_prev_n = N;
        v2_prev_k = K;
    }

    dim3 grid((M / V2_BM) * (N / V2_BN));
    if (lhs_format == 'R' && rhs_format == 'R') {
        // TODO: 
        gemm_v2_wgmma_tma_kernel<WGMMA_MajorOrder::K_MAJOR, WGMMA_MajorOrder::K_MAJOR><<<grid, V2_NUM_THREADS, 0, stream>>>(
            M, N, K, const_cast<__half*>(C), v2_tma_map_A, v2_tma_map_B);
    } else if (lhs_format == 'R' && rhs_format == 'C') {
        gemm_v2_wgmma_tma_kernel<WGMMA_MajorOrder::K_MAJOR, WGMMA_MajorOrder::K_MAJOR><<<grid, V2_NUM_THREADS, 0, stream>>>(
            M, N, K, const_cast<__half*>(C), v2_tma_map_A, v2_tma_map_B);
    }
}

// ============================================================================
// BFloat16 Version
// ============================================================================

__device__ uint64_t make_smem_desc_v2_bf16(__nv_bfloat16* ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0x0000000000000000;
    desc |= matrix_descriptor_encode(addr);
    desc |= matrix_descriptor_encode((uint64_t)16) << 16;
    desc |= matrix_descriptor_encode((uint64_t)1024) << 32;
    desc |= 1llu << 62;
    return desc;
}

template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ void wgmma64_bf16(float d[4][8], __nv_bfloat16* sA, __nv_bfloat16* sB) {
    uint64_t desc_a = make_smem_desc_v2_bf16(&sA[0]);
    uint64_t desc_b = make_smem_desc_v2_bf16(&sB[0]);
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

template <int BlockMajorSize, int BlockMinorSize>
void create_tensor_map_v2_bf16(CUtensorMap *tma_map, __nv_bfloat16* gmem_ptr,
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
        tma_map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        gmem_address,
        gmem_prob_shape,
        gmem_prob_stride + 1,
        smem_box_shape,
        smem_box_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    assert(result == CUDA_SUCCESS);
}

template <int BlockMajorSize, int BlockMinorSize>
CUtensorMap* allocate_and_create_tensor_map_v2_bf16(__nv_bfloat16* gmem_ptr,
                                                     int blocks_height, int blocks_width) {
    CUtensorMap *tma_map_d;
    cudaMalloc(&tma_map_d, sizeof(CUtensorMap));
    CUtensorMap tma_map_host;
    create_tensor_map_v2_bf16<BlockMajorSize, BlockMinorSize>(&tma_map_host, gmem_ptr, blocks_height, blocks_width);
    cudaMemcpy(tma_map_d, &tma_map_host, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    return tma_map_d;
}

__global__ void __launch_bounds__(V2_NUM_THREADS)
gemm_v2_wgmma_tma_kernel_bf16(int M, int N, int K, __nv_bfloat16* C,
                               CUtensorMap* tensorMapA, CUtensorMap* tensorMapB) {
    extern __shared__ __align__(128) uint8_t smem[];
    __nv_bfloat16* sA = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sB = sA + V2_BM * V2_BK;
    barrier* bar = reinterpret_cast<barrier*>(sB + V2_BK * V2_BN);

    int num_block_m = M / V2_BM, num_block_n = N / V2_BN;
    int num_block_k = K / V2_BK;
    int block_id = blockIdx.x;

    int block_m = block_id / num_block_n;
    int block_n = block_id % num_block_n;

    if (threadIdx.x == 0) {
        init(bar, blockDim.x);
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    for (int block_k = 0; block_k < num_block_k; ++block_k) {
        if (threadIdx.x == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(
                sA, tensorMapA, block_k, block_m, *bar);
            cde::cp_async_bulk_tensor_2d_global_to_shared(
                sB, tensorMapB, block_k, block_n, *bar);
            cde::cp_async_bulk_commit_group();
        }
        bar->arrive_and_wait();

        float d[4][8] = {0};
        warpgroup_arrive_v2();
        for (int k_inner = 0; k_inner < V2_BK / 16; ++k_inner) {
            __nv_bfloat16* wgmma_sA = sA + k_inner * 16 * V2_BM;
            __nv_bfloat16* wgmma_sB = sB + k_inner * 16 * V2_BN;
            if (k_inner == 0) {
                wgmma64_bf16<0, 1, 1, 0, 0>(d, wgmma_sA, wgmma_sB);
            } else {
                wgmma64_bf16<1, 1, 1, 0, 0>(d, wgmma_sA, wgmma_sB);
            }
        }
        warpgroup_commit_batch_v2();
        warpgroup_wait_v2<0>();

        if (block_k == num_block_k - 1) {
            int tid = threadIdx.x;
            int warp_id = tid / 32;
            int lane_id = tid % 32;

            int row_base = block_m * V2_BM + (warp_id / 2) * 16 + (lane_id / 4);
            int col_base = block_n * V2_BN + (warp_id % 2) * 32 + (lane_id % 4) * 2;

            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 8; j += 2) {
                    int row = row_base + (i / 2) * 8;
                    int col = col_base + (i % 2) * 8 + j;
                    if (row < M && col < N) {
                        C[col * M + row] = __float2bfloat16(d[i][j]);
                    }
                    if (row < M && col + 1 < N) {
                        C[(col + 1) * M + row] = __float2bfloat16(d[i][j + 1]);
                    }
                }
            }
        }
        __syncthreads();
    }
}

static CUtensorMap *v2_tma_map_A_bf16 = nullptr;
static CUtensorMap *v2_tma_map_B_bf16 = nullptr;
static int v2_prev_m_bf16 = 0, v2_prev_n_bf16 = 0, v2_prev_k_bf16 = 0;

void gemm_v2_wgmma_tma_bf16(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
                            int M, int N, int K, char lhs_format, char rhs_format,
                            cudaStream_t stream) {
    if (lhs_format != 'R' || rhs_format != 'C') {
        fprintf(stderr, "ERROR: CUDA V2 BF16 (WGMMA+TMA) only supports RC mode\n");
        return;
    }

    if (M % V2_BM != 0 || N % V2_BN != 0 || K % V2_BK != 0) {
        fprintf(stderr, "ERROR: CUDA V2 BF16 requires M,N,K to be multiples of 64\n");
        fprintf(stderr, "       Got: M=%d, N=%d, K=%d\n", M, N, K);
        return;
    }

    if (!v2_tma_map_A_bf16 || M != v2_prev_m_bf16 || N != v2_prev_n_bf16 || K != v2_prev_k_bf16) {
        if (v2_tma_map_A_bf16) cudaFree(v2_tma_map_A_bf16);
        if (v2_tma_map_B_bf16) cudaFree(v2_tma_map_B_bf16);

        v2_tma_map_A_bf16 = allocate_and_create_tensor_map_v2_bf16<V2_BM, V2_BK>(
            const_cast<__nv_bfloat16*>(A), M / V2_BM, K / V2_BK);
        v2_tma_map_B_bf16 = allocate_and_create_tensor_map_v2_bf16<V2_BN, V2_BK>(
            const_cast<__nv_bfloat16*>(B), N / V2_BN, K / V2_BK);

        v2_prev_m_bf16 = M;
        v2_prev_n_bf16 = N;
        v2_prev_k_bf16 = K;
    }

    dim3 grid((M / V2_BM) * (N / V2_BN));
    gemm_v2_wgmma_tma_kernel_bf16<<<grid, V2_NUM_THREADS, 0, stream>>>(
        M, N, K, const_cast<__nv_bfloat16*>(C), v2_tma_map_A_bf16, v2_tma_map_B_bf16);
}

} // namespace baseline

