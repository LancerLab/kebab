/**
 * @file cuda_gemm_v5_persistent.cu
 * @brief CUDA V5 GEMM with larger tiles and register optimization (based on fast.cu kernel 5)
 *
 * Key features:
 * - Larger tiles: 128×256×64 (double N dimension)
 * - 3 warp-groups: 1 producer + 2 consumers (384 threads)
 * - Dynamic register allocation via setmaxnreg
 * - Multiple consumers share M dimension
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

namespace cde = cuda::device::experimental;
using barrier = cuda::barrier<cuda::thread_scope_block>;

// ============================================================================
// WGMMA Helper Functions
// ============================================================================

__device__ static inline uint64_t matrix_descriptor_encode_v5(uint64_t x) {
    return (((x) & 0x3FFFF) >> 0x4);
}

__device__ uint64_t make_smem_desc_v5(__half* ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0x0000000000000000;
    desc |= matrix_descriptor_encode_v5(addr);
    desc |= matrix_descriptor_encode_v5((uint64_t)16) << 16;
    desc |= matrix_descriptor_encode_v5((uint64_t)1024) << 32;
    desc |= 1llu << 62;
    return desc;
}

__device__ void warpgroup_arrive_v5() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ void warpgroup_commit_batch_v5() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N>
__device__ void warpgroup_wait_v5() {
    static_assert(N >= 0 && N <= 7, "WGMMA wait: N must be in range [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

template <uint32_t RegCount>
__device__ void warpgroup_reg_alloc_v5() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

template <uint32_t RegCount>
__device__ void warpgroup_reg_dealloc_v5() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

// WGMMA 64×256×16 for FP16
template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ void wgmma256_v5(float d[16][8], __half* sA, __half* sB) {
    uint64_t desc_a = make_smem_desc_v5(&sA[0]);
    uint64_t desc_b = make_smem_desc_v5(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
        " %96,  %97,  %98,  %99,  %100, %101, %102, %103,  "
        " %104, %105, %106, %107, %108, %109, %110, %111,  "
        " %112, %113, %114, %115, %116, %117, %118, %119,  "
        " %120, %121, %122, %123, %124, %125, %126, %127},"
        " %128,"
        " %129,"
        " %130, %131, %132, %133, %134;\n"
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
          "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7]),
          "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]),
          "+f"(d[8][4]), "+f"(d[8][5]), "+f"(d[8][6]), "+f"(d[8][7]),
          "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]),
          "+f"(d[9][4]), "+f"(d[9][5]), "+f"(d[9][6]), "+f"(d[9][7]),
          "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]), "+f"(d[10][3]),
          "+f"(d[10][4]), "+f"(d[10][5]), "+f"(d[10][6]), "+f"(d[10][7]),
          "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]), "+f"(d[11][3]),
          "+f"(d[11][4]), "+f"(d[11][5]), "+f"(d[11][6]), "+f"(d[11][7]),
          "+f"(d[12][0]), "+f"(d[12][1]), "+f"(d[12][2]), "+f"(d[12][3]),
          "+f"(d[12][4]), "+f"(d[12][5]), "+f"(d[12][6]), "+f"(d[12][7]),
          "+f"(d[13][0]), "+f"(d[13][1]), "+f"(d[13][2]), "+f"(d[13][3]),
          "+f"(d[13][4]), "+f"(d[13][5]), "+f"(d[13][6]), "+f"(d[13][7]),
          "+f"(d[14][0]), "+f"(d[14][1]), "+f"(d[14][2]), "+f"(d[14][3]),
          "+f"(d[14][4]), "+f"(d[14][5]), "+f"(d[14][6]), "+f"(d[14][7]),
          "+f"(d[15][0]), "+f"(d[15][1]), "+f"(d[15][2]), "+f"(d[15][3]),
          "+f"(d[15][4]), "+f"(d[15][5]), "+f"(d[15][6]), "+f"(d[15][7])
        : "l"(desc_a), "l"(desc_b),
          "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
          "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

// ============================================================================
// TMA Functions
// ============================================================================

template <int BlockMajorSize, int BlockMinorSize>
void create_tensor_map_v5(CUtensorMap *tma_map, __half* gmem_ptr,
                          int blocks_height, int blocks_width) {
    void* gmem_address = (void*)gmem_ptr;
    uint64_t gmem_prob_shape[5] = {
        (uint64_t)BlockMinorSize * blocks_width,
        (uint64_t)BlockMajorSize * blocks_height, 1, 1, 1
    };
    uint64_t gmem_prob_stride[5] = {
        sizeof(__half), sizeof(__half) * BlockMinorSize * blocks_width, 0, 0, 0
    };
    uint32_t smem_box_shape[5] = {
        uint32_t(BlockMinorSize), uint32_t(BlockMajorSize), 1, 1, 1
    };
    uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};

    cuTensorMapEncodeTiled(
        tma_map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, gmem_address, gmem_prob_shape,
        gmem_prob_stride + 1, smem_box_shape, smem_box_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

template<int BlockMajorSize, int BlockMinorSize>
static inline CUtensorMap* allocate_and_create_tensor_map_v5(
    __half* src, int blocks_height, int blocks_width) {
    CUtensorMap *tma_map_d;
    cudaMalloc(&tma_map_d, sizeof(CUtensorMap));
    CUtensorMap tma_map_host;
    create_tensor_map_v5<BlockMajorSize, BlockMinorSize>(
        &tma_map_host, src, blocks_height, blocks_width);
    cudaMemcpy(tma_map_d, &tma_map_host, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    return tma_map_d;
}

// ============================================================================
// V5 Kernel Configuration
// ============================================================================

constexpr int V5_BM = 128;
constexpr int V5_BN = 256;
constexpr int V5_BK = 64;
constexpr int V5_NUM_THREADS = 384;  // 3 warp-groups: 1 producer + 2 consumers
constexpr int V5_QSIZE = 3;

template <int BM, int BN, int BK, int QSIZE>
struct SMemV5 {
    alignas(128) __half A[BM * BK * QSIZE];
    alignas(128) __half B[BK * BN * QSIZE];
};

// ============================================================================
// V5 Kernel: Multiple Consumers with Register Optimization
// ============================================================================

template<int BM, int BN, int BK, int NUM_THREADS, int QSIZE>
__global__ void __launch_bounds__(NUM_THREADS)
gemm_v5_persistent_kernel(int M, int N, int K, __half* C,
                           const CUtensorMap* tensorMapA,
                           const CUtensorMap* tensorMapB) {
    constexpr int WGMMA_M = 64, WGMMA_K = 16, WGMMA_N = BN;
    constexpr int num_consumers = (NUM_THREADS / 128) - 1;
    constexpr int B_WG_M = BM / num_consumers;

    extern __shared__ __align__(128) uint8_t smem[];
    SMemV5<BM, BN, BK, QSIZE> &s = *reinterpret_cast<SMemV5<BM, BN, BK, QSIZE>*>(smem);
    __half *sA = s.A;
    __half *sB = s.B;

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier full[QSIZE], empty[QSIZE];

    const int num_blocks_k = K / BK;
    int num_block_n = blockIdx.x % (N / BN);
    int num_block_m = blockIdx.x / (N / BN);
    int wg_idx = threadIdx.x / 128;
    int tid = threadIdx.x % 128;

    if (threadIdx.x == 0) {
        for (int i = 0; i < QSIZE; ++i) {
            init(&full[i], num_consumers * 128 + 1);
            init(&empty[i], num_consumers * 128 + 1);
        }
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    // Producer warp-group
    if (wg_idx == 0) {
        constexpr int num_regs = (num_consumers <= 2 ? 24 : 32);
        warpgroup_reg_dealloc_v5<num_regs>();
        if (tid == 0) {
            int qidx = 0;
            for (int block_k_iter = 0; block_k_iter < num_blocks_k; ++block_k_iter, ++qidx) {
                if (qidx == QSIZE) qidx = 0;
                empty[qidx].wait(empty[qidx].arrive());
                cde::cp_async_bulk_tensor_2d_global_to_shared(
                    &sA[qidx * BK * BM], tensorMapA, block_k_iter * BK, num_block_m * BM, full[qidx]);
                cde::cp_async_bulk_tensor_2d_global_to_shared(
                    &sB[qidx * BK * BN], tensorMapB, block_k_iter * BK, num_block_n * BN, full[qidx]);
                barrier::arrival_token _ = cuda::device::barrier_arrive_tx(
                    full[qidx], 1, (BK * BN + BK * BM) * sizeof(__half));
            }
        }
    }
    // Consumer warp-groups
    else {
        constexpr int num_regs = (num_consumers == 1 ? 256 : (num_consumers == 2 ? 240 : 160));
        warpgroup_reg_alloc_v5<num_regs>();
        --wg_idx;

        for (int i = 0; i < QSIZE; ++i) {
            barrier::arrival_token _ = empty[i].arrive();
        }

        float d[B_WG_M / WGMMA_M][WGMMA_N / 16][8];
        memset(d, 0, sizeof(d));

        int qidx = 0;
        for (int block_k_iter = 0; block_k_iter < num_blocks_k; ++block_k_iter, ++qidx) {
            if (qidx == QSIZE) qidx = 0;
            full[qidx].wait(full[qidx].arrive());

            warpgroup_arrive_v5();
            #pragma unroll
            for (int m_it = 0; m_it < B_WG_M / WGMMA_M; ++m_it) {
                __half *wgmma_sA = sA + qidx * BK * BM + BK * (m_it + wg_idx * B_WG_M / WGMMA_M) * WGMMA_M;
                #pragma unroll
                for (int k_it = 0; k_it < BK / WGMMA_K; ++k_it) {
                    wgmma256_v5<1, 1, 1, 0, 0>(
                        d[m_it], &wgmma_sA[k_it * WGMMA_K], &sB[qidx * BK * BN + k_it * WGMMA_K]);
                }
            }
            warpgroup_commit_batch_v5();
            warpgroup_wait_v5<0>();
            barrier::arrival_token _ = empty[qidx].arrive();
        }

        // Store results
        tid = threadIdx.x % 128;
        int lane = tid % 32;
        int warp = tid / 32;
        int row = warp * 16 + lane / 4;
        __half *block_C = C + num_block_n * BN * M + num_block_m * BM;

        #pragma unroll
        for (int m_it = 0; m_it < B_WG_M / WGMMA_M; ++m_it) {
            int yo = m_it * WGMMA_M + wg_idx * B_WG_M;
            #pragma unroll
            for (int w = 0; w < WGMMA_N / 16; ++w) {
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

static CUtensorMap *v5_tma_map_A = nullptr;
static CUtensorMap *v5_tma_map_B = nullptr;
static int v5_prev_m = 0, v5_prev_n = 0, v5_prev_k = 0;

void gemm_v5_persistent_fp16(const __half* A, const __half* B, __half* C,
                             int M, int N, int K, char lhs_format, char rhs_format,
                             cudaStream_t stream) {
    if (lhs_format != 'R' || rhs_format != 'C') {
        fprintf(stderr, "ERROR: CUDA V5 only supports RC mode\n");
        return;
    }
    if (M % V5_BM != 0 || N % V5_BN != 0 || K % V5_BK != 0) {
        fprintf(stderr, "ERROR: CUDA V5 requires M%%128==0, N%%256==0, K%%64==0\n");
        return;
    }

    if (!v5_tma_map_A || M != v5_prev_m || N != v5_prev_n || K != v5_prev_k) {
        if (v5_tma_map_A) cudaFree(v5_tma_map_A);
        if (v5_tma_map_B) cudaFree(v5_tma_map_B);
        v5_tma_map_A = allocate_and_create_tensor_map_v5<V5_BM, V5_BK>(
            const_cast<__half*>(A), M / V5_BM, K / V5_BK);
        v5_tma_map_B = allocate_and_create_tensor_map_v5<V5_BN, V5_BK>(
            const_cast<__half*>(B), N / V5_BN, K / V5_BK);
        v5_prev_m = M; v5_prev_n = N; v5_prev_k = K;
    }

    size_t sMemSize = sizeof(SMemV5<V5_BM, V5_BN, V5_BK, V5_QSIZE>);
    auto kernel = gemm_v5_persistent_kernel<V5_BM, V5_BN, V5_BK, V5_NUM_THREADS, V5_QSIZE>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize);

    dim3 grid((M / V5_BM) * (N / V5_BN));
    kernel<<<grid, V5_NUM_THREADS, sMemSize, stream>>>(
        M, N, K, const_cast<__half*>(C), v5_tma_map_A, v5_tma_map_B);
}

} // namespace baseline

