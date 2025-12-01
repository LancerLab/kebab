/**
 * @file cuda_gemm_v8_multicast.cu
 * @brief CUDA V8 GEMM with Thread Block Clusters + TMA Multicast (based on fast.cu kernel 8)
 *
 * Key features:
 * - Thread Block Clusters (CLUSTER_M=2, CLUSTER_N=1)
 * - TMA Multicast for shared memory broadcast across cluster
 * - PTX mbarrier with cluster-level synchronization
 * - 128×256×64 tiles, 3 warp-groups
 *
 * Note: Requires SM90 (Hopper) architecture
 */

#include "kebab/cuda/cuda_gemm.h"
#include <cuda_fp16.h>
#include <cstdio>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>

namespace baseline {

// ============================================================================
// WGMMA Helper Functions (V8)
// ============================================================================

__device__ static inline uint64_t matrix_descriptor_encode_v8(uint64_t x) {
    return (((x) & 0x3FFFF) >> 0x4);
}

__device__ uint64_t make_smem_desc_v8(__half* ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0x0000000000000000;
    desc |= matrix_descriptor_encode_v8(addr);
    desc |= matrix_descriptor_encode_v8((uint64_t)16) << 16;
    desc |= matrix_descriptor_encode_v8((uint64_t)1024) << 32;
    desc |= 1llu << 62;
    return desc;
}

__device__ void warpgroup_arrive_v8() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ void warpgroup_commit_batch_v8() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N>
__device__ void warpgroup_wait_v8() {
    static_assert(N >= 0 && N <= 7, "WGMMA wait: N must be in range [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

template <uint32_t RegCount>
__device__ void warpgroup_reg_alloc_v8() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

template <uint32_t RegCount>
__device__ void warpgroup_reg_dealloc_v8() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

// WGMMA 64×256×16 for FP16
template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ void wgmma256_v8(float d[16][8], __half* sA, __half* sB) {
    uint64_t desc_a = make_smem_desc_v8(&sA[0]);
    uint64_t desc_b = make_smem_desc_v8(&sB[0]);
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
// PTX Barrier Functions (V8 - with cluster support)
// ============================================================================

__device__ static __forceinline__ void init_barrier_v8(uint64_t* bar, int thread_count, int transaction_count) {
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], %1;\n"
        :: "r"(bar_ptr), "r"(thread_count + transaction_count)
    );
}

__device__ static __forceinline__ void expect_bytes_v8(uint64_t* bar, uint32_t bytes) {
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
        :: "r"(bar_ptr), "r"(bytes));
}

__device__ static inline void load_async_v8(__half *dst, void const* const src_tma_map, uint64_t* bar,
                                            int global_col_idx, int global_row_idx) {
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(src_tma_map);
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));

    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4, %5}], [%2];"
        :
        : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr),
          "n"(0), "r"(global_row_idx), "r"(global_col_idx / 64)
        : "memory"
    );
}

__device__ static inline void load_async_multicast_v8(__half *dst, void const* const src_tma_map, uint64_t* bar,
                                                      int global_col_idx, int global_row_idx, uint16_t cluster_mask) {
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(src_tma_map);
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));

    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
        " [%0], [%1, {%3, %4, %5}], [%2], %6;"
        :
        : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr),
          "n"(0), "r"(global_row_idx), "r"(global_col_idx / 64), "h"(cluster_mask)
        : "memory"
    );
}

__device__ static __forceinline__ void wait_v8(uint64_t* bar, int kPhaseBit) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1 bra.uni DONE;\n"
        "bra.uni LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(mbar_ptr), "r"(kPhaseBit)
    );
}

__device__ static __forceinline__ void wait_cluster_v8(uint64_t* bar, int kPhaseBit) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], %1;\n"
        "@P1 bra.uni DONE;\n"
        "bra.uni LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(mbar_ptr), "r"(kPhaseBit)
    );
}

__device__ static __forceinline__ void arrive_v8(uint64_t* bar, uint32_t count = 1) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n"
        :
        : "r"(mbar_ptr), "r"(count)
        : "memory"
    );
}

__device__ void arrive_cluster_v8(uint64_t* bar, uint32_t cta_id, uint32_t count = 1) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "{\n\t"
        ".reg .b32 remAddr32;\n\t"
        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
        "mbarrier.arrive.shared::cluster.b64 _, [remAddr32], %2;\n\t"
        "}"
        :
        : "r"(smem_addr), "r"(cta_id), "r"(count));
}

// ============================================================================
// TMA Functions (3D tensor maps for V8)
// ============================================================================

template <int BlockMajorSize, int BlockMinorSize>
CUtensorMap create_tensor_map_v8(__half* gmem_ptr, int global_height, int global_width) {
    CUtensorMap tma_map;
    void* gmem_address = (void*)gmem_ptr;
    static_assert(BlockMinorSize >= 64);

    uint64_t gmem_prob_shape[5] = {64, (uint64_t)global_height, (uint64_t)global_width / 64, 1, 1};
    uint64_t gmem_prob_stride[5] = {sizeof(__half) * global_width, 64 * sizeof(__half), 0, 0, 0};
    uint32_t smem_box_shape[5] = {64, uint32_t(BlockMajorSize), uint32_t(BlockMinorSize / 64), 1, 1};
    uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};

    cuTensorMapEncodeTiled(
        &tma_map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3, gmem_address, gmem_prob_shape,
        gmem_prob_stride, smem_box_shape, smem_box_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    return tma_map;
}

// ============================================================================
// V8 Kernel Configuration
// ============================================================================

constexpr int V8_BM = 128;
constexpr int V8_BN = 256;
constexpr int V8_BK = 64;
constexpr int V8_NUM_THREADS = 384;
constexpr int V8_QSIZE = 3;
constexpr int V8_NUM_SM = 128;
constexpr int V8_CLUSTER_M = 2;
constexpr int V8_CLUSTER_N = 1;

template <int BM, int BN, int BK, int QSIZE>
struct SMemV8 {
    alignas(128) __half A[BM * BK * QSIZE];
    alignas(128) __half B[BK * BN * QSIZE];
};

#define CEIL_DIV_V8(a, b) (((a) + (b) - 1) / (b))

// Simple linear schedule for cluster-based execution
template<int NUM_SM, int BM, int BN>
struct ScheduleV8 {
    int st, en;
    int total_blocks_n;

    __device__ __forceinline__ ScheduleV8(int M, int N, int block) {
        int total_blocks = (M / BM) * (N / BN);
        total_blocks_n = N / BN;
        int blocks_per_sm = total_blocks / NUM_SM;
        int extra_blocks = total_blocks % NUM_SM;
        if (block < extra_blocks) {
            st = block * (blocks_per_sm + 1);
            en = st + blocks_per_sm + 1;
        } else {
            st = extra_blocks + block * blocks_per_sm;
            en = st + blocks_per_sm;
        }
    }

    __device__ __forceinline__ bool next(int &block_m, int &block_n) {
        if (en == st) return false;
        int num_block = st++;
        block_m = num_block / total_blocks_n;
        block_n = num_block % total_blocks_n;
        return true;
    }
};

// ============================================================================
// V8 Kernel: Thread Block Clusters + TMA Multicast
// ============================================================================

template<int BM, int BN, int BK, int NUM_THREADS, int QSIZE, int NUM_SM, int CLUSTER_M, int CLUSTER_N>
__global__ void __launch_bounds__(NUM_THREADS) __cluster_dims__(CLUSTER_M * CLUSTER_N, 1, 1)
gemm_v8_multicast_kernel(int M, int N, int K, __half* C,
                         const __grid_constant__ CUtensorMap tensorMapA,
                         const __grid_constant__ CUtensorMap tensorMapB) {
    constexpr int WGMMA_M = 64, WGMMA_K = 16, WGMMA_N = BN;
    constexpr int num_consumers = (NUM_THREADS / 128) - 1;
    constexpr int B_WG_M = BM / num_consumers;
    constexpr int CLUSTERS = CLUSTER_M * CLUSTER_N;

    extern __shared__ __align__(128) uint8_t smem[];
    SMemV8<BM, BN, BK, QSIZE> &s = *reinterpret_cast<SMemV8<BM, BN, BK, QSIZE>*>(smem);
    __half *sA = s.A;
    __half *sB = s.B;

    __shared__ __align__(8) uint64_t full[QSIZE], empty[QSIZE];
    uint32_t rank;
    asm volatile("mov.u32 %0, %clusterid.x;\n" : "=r"(rank) :);

    const int num_blocks_k = K / BK;
    int wg_idx = threadIdx.x / 128;
    int tid = threadIdx.x % 128;

    if (threadIdx.x == 0) {
        for (int i = 0; i < QSIZE; ++i) {
            init_barrier_v8(&full[i], 0, 1);
            init_barrier_v8(&empty[i], 0, num_consumers * CLUSTERS);
        }
    }
    asm volatile("barrier.cluster.arrive;\n" : :);
    asm volatile("barrier.cluster.wait;\n" : :);

    ScheduleV8<NUM_SM / CLUSTERS, BM * CLUSTER_M, BN * CLUSTER_N> schedule(M, N, rank);

    asm volatile("mov.u32 %0, %cluster_ctarank;\n" : "=r"(rank) :);
    uint32_t rank_m = rank / CLUSTER_N;
    uint32_t rank_n = rank % CLUSTER_N;

    // Producer warp-group
    if (wg_idx == 0) {
        constexpr int num_regs = (num_consumers <= 2 ? 24 : 32);
        warpgroup_reg_dealloc_v8<num_regs>();
        if (tid == 0) {
            int p = 0;
            int qidx = 0;
            uint32_t col_mask = 0;
            for (int i = 0; i < CLUSTER_M; ++i) {
                col_mask |= (1 << (i * CLUSTER_N));
            }
            int num_block_m, num_block_n;
            while (schedule.next(num_block_m, num_block_n)) {
                num_block_n = num_block_n * CLUSTER_N + rank_n;
                num_block_m = num_block_m * CLUSTER_M + rank_m;

                for (int block_k_iter = 0; block_k_iter < num_blocks_k; ++block_k_iter, ++qidx) {
                    if (qidx == QSIZE) { qidx = 0; p ^= 1; }
                    wait_v8(&empty[qidx], p);

                    expect_bytes_v8(&full[qidx], (BK * BN + BK * BM) * sizeof(__half));
                    if constexpr (CLUSTER_N > 1) {
                        uint32_t mask = ((1 << CLUSTER_N) - 1) << (rank_m * CLUSTER_N);
                        if (rank_n == 0) {
                            load_async_multicast_v8(&sA[qidx * BK * BM], &tensorMapA, &full[qidx],
                                                    block_k_iter * BK, num_block_m * BM, mask);
                        }
                    } else {
                        load_async_v8(&sA[qidx * BK * BM], &tensorMapA, &full[qidx],
                                      block_k_iter * BK, num_block_m * BM);
                    }

                    if constexpr (CLUSTER_M > 1) {
                        if (rank_m == 0) {
                            load_async_multicast_v8(&sB[qidx * BK * BN], &tensorMapB, &full[qidx],
                                                    block_k_iter * BK, num_block_n * BN, col_mask);
                        }
                    } else {
                        load_async_v8(&sB[qidx * BK * BN], &tensorMapB, &full[qidx],
                                      block_k_iter * BK, num_block_n * BN);
                    }
                }
            }
        }
    }
    // Consumer warp-groups
    else {
        constexpr int num_regs = (num_consumers == 1 ? 256 : (num_consumers == 2 ? 240 : 160));
        warpgroup_reg_alloc_v8<num_regs>();
        float d[B_WG_M / WGMMA_M][WGMMA_N / 16][8];
        --wg_idx;

        for (int qidx = 0; qidx < QSIZE; ++qidx) {
            if (tid < CLUSTERS) arrive_cluster_v8(&empty[qidx], tid);
        }

        int p = 0;
        int qidx = 0;
        int lane = tid % 32, warp = tid / 32, row = warp * 16 + lane / 4;
        int num_block_m, num_block_n;
        while (schedule.next(num_block_m, num_block_n)) {
            num_block_n = num_block_n * CLUSTER_N + rank_n;
            num_block_m = num_block_m * CLUSTER_M + rank_m;
            memset(d, 0, sizeof(d));

            for (int block_k_iter = 0; block_k_iter < num_blocks_k; ++block_k_iter, ++qidx) {
                if (qidx == QSIZE) { qidx = 0; p ^= 1; }
                wait_v8(&full[qidx], p);

                warpgroup_arrive_v8();
                #pragma unroll
                for (int m_it = 0; m_it < B_WG_M / WGMMA_M; ++m_it) {
                    __half *wgmma_sA = sA + qidx * BK * BM + 64 * (m_it + wg_idx * B_WG_M / WGMMA_M) * WGMMA_M;
                    __half *wgmma_sB = sB + qidx * BK * BN;
                    #pragma unroll
                    for (int bk = 0; bk < BK; bk += 64) {
                        #pragma unroll
                        for (int k_it = 0; k_it < 64 / WGMMA_K; ++k_it) {
                            wgmma256_v8<1, 1, 1, 0, 0>(d[m_it], &wgmma_sA[k_it * WGMMA_K],
                                                       &wgmma_sB[k_it * WGMMA_K]);
                        }
                        wgmma_sA += 64 * BM;
                        wgmma_sB += 64 * BN;
                    }
                }
                warpgroup_commit_batch_v8();
                warpgroup_wait_v8<0>();
                if (tid < CLUSTERS) arrive_cluster_v8(&empty[qidx], tid);
            }

            // Store results
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
}

// ============================================================================
// Host Function
// ============================================================================

static CUtensorMap v8_tma_map_A;
static CUtensorMap v8_tma_map_B;
static int v8_prev_m = 0, v8_prev_n = 0, v8_prev_k = 0;

void gemm_v8_multicast_fp16(const __half* A, const __half* B, __half* C,
                            int M, int N, int K, char lhs_format, char rhs_format,
                            cudaStream_t stream) {
    if (lhs_format != 'R' || rhs_format != 'C') {
        fprintf(stderr, "ERROR: CUDA V8 only supports RC mode\n");
        return;
    }
    // V8 requires cluster-aligned dimensions
    constexpr int CLUSTER_BM = V8_BM * V8_CLUSTER_M;
    constexpr int CLUSTER_BN = V8_BN * V8_CLUSTER_N;
    if (M % CLUSTER_BM != 0 || N % CLUSTER_BN != 0 || K % V8_BK != 0) {
        fprintf(stderr, "ERROR: CUDA V8 requires M%%256==0, N%%256==0, K%%64==0\n");
        return;
    }

    if (M != v8_prev_m || N != v8_prev_n || K != v8_prev_k) {
        v8_tma_map_A = create_tensor_map_v8<V8_BM, V8_BK>(const_cast<__half*>(A), M, K);
        v8_tma_map_B = create_tensor_map_v8<V8_BN, V8_BK>(const_cast<__half*>(B), N, K);
        v8_prev_m = M; v8_prev_n = N; v8_prev_k = K;
    }

    size_t sMemSize = sizeof(SMemV8<V8_BM, V8_BN, V8_BK, V8_QSIZE>);
    auto kernel = gemm_v8_multicast_kernel<V8_BM, V8_BN, V8_BK, V8_NUM_THREADS, V8_QSIZE, V8_NUM_SM, V8_CLUSTER_M, V8_CLUSTER_N>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize);

    kernel<<<V8_NUM_SM, V8_NUM_THREADS, sMemSize, stream>>>(
        M, N, K, const_cast<__half*>(C), v8_tma_map_A, v8_tma_map_B);
}

} // namespace baseline

