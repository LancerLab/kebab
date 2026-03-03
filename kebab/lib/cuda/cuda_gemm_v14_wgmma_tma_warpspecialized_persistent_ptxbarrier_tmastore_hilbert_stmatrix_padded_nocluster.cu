/**
 * @file cuda_gemm_v14_wgmma_tma_warpspecialized_persistent_ptxbarrier_tmastore_hilbert_stmatrix_padded_nocluster.cu
 * @brief CUDA V14 GEMM with stmatrix + Padded TMA (de-clustered from V12)
 *
 * Key features:
 * - Keeps V12 stmatrix instruction path for efficient shared-memory stores
 * - Keeps V12 padded TMA tensor map for C output (72 instead of 64)
 * - Keeps Hilbert curve tile scheduling
 * - Removes V8-introduced cluster/multicast features
 *
 * Note: Requires SM90 (Hopper) architecture
 */

#include "kebab/cuda/cuda_gemm.h"
#include <cuda_fp16.h>
#include <cstdio>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cstring>

namespace baseline {

__device__ static inline uint64_t matrix_descriptor_encode_v14(uint64_t x) {
    return (((x) & 0x3FFFF) >> 0x4);
}

__device__ uint64_t make_smem_desc_v14(__half* ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0x0000000000000000;
    desc |= matrix_descriptor_encode_v14(addr);
    desc |= matrix_descriptor_encode_v14((uint64_t)16) << 16;
    desc |= matrix_descriptor_encode_v14((uint64_t)1024) << 32;
    desc |= 1llu << 62;
    return desc;
}

__device__ void warpgroup_arrive_v14() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ void warpgroup_commit_batch_v14() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

__device__ void warpgroup_wait_v14() {
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}

template <uint32_t RegCount>
__device__ void warpgroup_reg_alloc_v14() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

template <uint32_t RegCount>
__device__ void warpgroup_reg_dealloc_v14() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ void wgmma256_v14(float d[16][8], __half* sA, __half* sB) {
    uint64_t desc_a = make_smem_desc_v14(&sA[0]);
    uint64_t desc_b = make_smem_desc_v14(&sB[0]);
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

__device__ static __forceinline__ void init_barrier_v14(uint64_t* bar, int thread_count, int transaction_count) {
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(bar_ptr), "r"(thread_count + transaction_count));
}

__device__ static __forceinline__ void expect_bytes_v14(uint64_t* bar, uint32_t bytes) {
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar_ptr), "r"(bytes));
}

__device__ static inline void load_async_v14(__half *dst, void const* const src_tma_map, uint64_t* bar,
                                             int global_col_idx, int global_row_idx) {
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(src_tma_map);
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4, %5}], [%2];"
        : : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr), "n"(0), "r"(global_row_idx), "r"(global_col_idx / 64) : "memory");
}

__device__ static inline void store_async_v14(void const* dst_tma_map, __half *src, int global_col_idx, int global_row_idx) {
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(dst_tma_map);
    uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(src));
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group [%0, {%2, %3, %4}], [%1];"
        : : "l"(tma_ptr), "r"(src_ptr), "n"(0), "r"(global_row_idx), "r"(global_col_idx / 64) : "memory");
}

__device__ static __forceinline__ void wait_v14(uint64_t* bar, int kPhaseBit) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "{\n.reg .pred P1;\nLAB_WAIT:\nmbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1 bra.uni DONE;\nbra.uni LAB_WAIT;\nDONE:\n}\n" :: "r"(mbar_ptr), "r"(kPhaseBit));
}

__device__ static __forceinline__ void arrive_v14(uint64_t* bar, uint32_t count = 1) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n"
                 : : "r"(mbar_ptr), "r"(count) : "memory");
}

template <int BlockMajorSize, int BlockMinorSize, bool swizzle = true, bool padding = false>
CUtensorMap create_tensor_map_v14(__half* gmem_ptr, int global_height, int global_width) {
    CUtensorMap tma_map;
    void* gmem_address = (void*)gmem_ptr;
    uint64_t gmem_prob_shape[5] = {64, (uint64_t)global_height, (uint64_t)global_width / 64, 1, 1};
    uint64_t gmem_prob_stride[5] = {sizeof(__half) * global_width, 64 * sizeof(__half), 0, 0, 0};
    uint32_t smem_box_shape[5] = {padding ? 72u : 64u, uint32_t(BlockMajorSize), uint32_t(BlockMinorSize / 64), 1, 1};
    uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};
    cuTensorMapEncodeTiled(&tma_map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3, gmem_address, gmem_prob_shape,
        gmem_prob_stride, smem_box_shape, smem_box_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return tma_map;
}

static void rot_v14(int n, int& x, int& y, int rx, int ry) {
    if (ry == 0) {
        if (rx == 1) { x = n - 1 - x; y = n - 1 - y; }
        int t = x; x = y; y = t;
    }
}

static void d2xy_v14(int n, int d, int& x, int& y) {
    int rx, ry, s, t = d;
    x = y = 0;
    for (s = 1; s < n; s *= 2) {
        rx = 1 & (t / 2);
        ry = 1 & (t ^ rx);
        rot_v14(s, x, y, rx, ry);
        x += s * rx;
        y += s * ry;
        t /= 4;
    }
}

constexpr int V14_SPACE_LEN = 128;

static void createHilbert_v14(int M, int N, int CORES, int* space) {
    int dim = 1;
    while (dim < M || dim < N) dim *= 2;
    memset(space, -1, sizeof(int) * CORES * V14_SPACE_LEN);
    std::vector<std::vector<int>> pos(CORES);
    int core = 0;
    for (int i = 0; i < dim * dim; ++i) {
        int x, y;
        d2xy_v14(dim, i, x, y);
        if (x < M && y < N) {
            pos[core].push_back((x << 16) | y);
            core = (core + 1) % CORES;
        }
    }
    for (int i = 0; i < CORES; ++i) {
        for (size_t j = 0; j < pos[i].size() && j < V14_SPACE_LEN; ++j) {
            space[i * V14_SPACE_LEN + j] = pos[i][j];
        }
    }
}

constexpr int V14_BM = 128;
constexpr int V14_BN = 256;
constexpr int V14_BK = 64;
constexpr int V14_NUM_THREADS = 384;
constexpr int V14_QSIZE = 3;
constexpr int V14_NUM_SM = 128;
constexpr int V14_NUM_CONSUMERS = (V14_NUM_THREADS / 128) - 1;
constexpr int V14_B_WG_M = V14_BM / V14_NUM_CONSUMERS;

static_assert(V14_NUM_SM > 0, "V14_NUM_SM must be positive");

template <int BM, int BN, int BK, int QSIZE>
struct SMemV14 {
    alignas(128) __half A[BM * BK * QSIZE];
    alignas(128) __half B[BK * BN * QSIZE];
    alignas(128) __half C[BN * (BM + (BM / 64) * 8)];
    alignas(8) uint64_t full[QSIZE], empty[QSIZE];
    int space[V14_SPACE_LEN];
};

struct ScheduleV14 {
    int* space;
    int idx;

    __device__ __forceinline__ ScheduleV14(int* _space) : space(_space), idx(0) {}

    __device__ __forceinline__ bool next(int &block_m, int &block_n) {
        if (idx >= V14_SPACE_LEN) return false;
        int val = space[idx];
        if (val == -1) return false;
        block_m = val >> 16;
        block_n = val & 0xFFFF;
        ++idx;
        return true;
    }
};

template<int BM, int BN, int BK, int NUM_THREADS, int QSIZE>
__global__ void __launch_bounds__(NUM_THREADS)
gemm_v14_stmatrix_nocluster_kernel(int M, int N, int K,
                                   const __grid_constant__ CUtensorMap tensorMapC,
                                   const __grid_constant__ CUtensorMap tensorMapA,
                                   const __grid_constant__ CUtensorMap tensorMapB,
                                   int* dspace) {
    constexpr int WGMMA_M = 64, WGMMA_K = 16, WGMMA_N = BN;
    constexpr int num_consumers = (NUM_THREADS / 128) - 1;
    constexpr int B_WG_M = BM / num_consumers;
    constexpr int B_WG_M_PADDED = B_WG_M + 8;

    extern __shared__ __align__(128) uint8_t smem[];
    SMemV14<BM, BN, BK, QSIZE> &s = *reinterpret_cast<SMemV14<BM, BN, BK, QSIZE>*>(smem);
    __half *sA = s.A, *sB = s.B, *sC = s.C;
    uint64_t *full = s.full, *empty = s.empty;
    int *space = s.space;

    int sm_rank = blockIdx.x;
    if (threadIdx.x < V14_SPACE_LEN) space[threadIdx.x] = dspace[sm_rank * V14_SPACE_LEN + threadIdx.x];
    __syncthreads();

    const int num_blocks_k = K / BK;
    int wg_idx = threadIdx.x / 128;
    int tid = threadIdx.x % 128;

    if (threadIdx.x == 0) {
        for (int i = 0; i < QSIZE; ++i) {
            init_barrier_v14(&full[i], 0, 1);
            init_barrier_v14(&empty[i], 0, num_consumers);
        }
    }
    __syncthreads();

    ScheduleV14 schedule(space);

    if (wg_idx == 0) {
        constexpr int num_regs = (num_consumers <= 2 ? 24 : 32);
        warpgroup_reg_dealloc_v14<num_regs>();
        if (tid == 0) {
            int p = 0, qidx = 0;
            int num_block_m, num_block_n;
            while (schedule.next(num_block_m, num_block_n)) {
                for (int block_k_iter = 0; block_k_iter < num_blocks_k; ++block_k_iter, ++qidx) {
                    if (qidx == QSIZE) { qidx = 0; p ^= 1; }
                    wait_v14(&empty[qidx], p);
                    expect_bytes_v14(&full[qidx], (BK * BN + BK * BM) * sizeof(__half));
                    load_async_v14(&sA[qidx * BK * BM], &tensorMapA, &full[qidx], block_k_iter * BK, num_block_m * BM);
                    load_async_v14(&sB[qidx * BK * BN], &tensorMapB, &full[qidx], block_k_iter * BK, num_block_n * BN);
                }
            }
        }
    } else {
        constexpr int num_regs = (num_consumers == 1 ? 256 : (num_consumers == 2 ? 240 : 160));
        warpgroup_reg_alloc_v14<num_regs>();
        float d[B_WG_M / WGMMA_M][WGMMA_N / 16][8];
        --wg_idx;
        for (int qidx = 0; qidx < QSIZE; ++qidx) if (tid == 0) arrive_v14(&empty[qidx], 1);

        int p = 0, qidx = 0;
        int lane = tid % 32, warp = tid / 32;
        __half* block_sC = sC + wg_idx * B_WG_M_PADDED * BN;
        uint32_t tid_offset = warp * 16 + (lane % 8) * B_WG_M_PADDED;
        tid_offset += (lane / 16) * B_WG_M_PADDED * 8 + (lane & 8);
        uint32_t base_addr = static_cast<uint32_t>(__cvta_generic_to_shared(block_sC)) + tid_offset * sizeof(__half);

        int num_block_m, num_block_n;
        while (schedule.next(num_block_m, num_block_n)) {
            if (qidx == QSIZE) { qidx = 0; p ^= 1; }
            wait_v14(&full[qidx], p);
            warpgroup_arrive_v14();
            #pragma unroll
            for (int m_it = 0; m_it < B_WG_M / WGMMA_M; ++m_it) {
                __half *wgmma_sA = sA + qidx * BK * BM + 64 * (m_it + wg_idx * B_WG_M / WGMMA_M) * WGMMA_M;
                __half *wgmma_sB = sB + qidx * BK * BN;
                wgmma256_v14<0, 1, 1, 0, 0>(d[m_it], &wgmma_sA[0], &wgmma_sB[0]);
                #pragma unroll
                for (int k_it = 1; k_it < 64 / WGMMA_K; ++k_it)
                    wgmma256_v14<1, 1, 1, 0, 0>(d[m_it], &wgmma_sA[k_it * WGMMA_K], &wgmma_sB[k_it * WGMMA_K]);
                wgmma_sA += 64 * BM; wgmma_sB += 64 * BN;
                #pragma unroll
                for (int bk = 64; bk < BK; bk += 64) {
                    #pragma unroll
                    for (int k_it = 0; k_it < 64 / WGMMA_K; ++k_it)
                        wgmma256_v14<1, 1, 1, 0, 0>(d[m_it], &wgmma_sA[k_it * WGMMA_K], &wgmma_sB[k_it * WGMMA_K]);
                    wgmma_sA += 64 * BM; wgmma_sB += 64 * BN;
                }
            }
            warpgroup_commit_batch_v14();
            warpgroup_wait_v14();
            if (tid == 0) arrive_v14(&empty[qidx], 1);
            ++qidx;

            for (int block_k_iter = 1; block_k_iter < num_blocks_k; ++block_k_iter, ++qidx) {
                if (qidx == QSIZE) { qidx = 0; p ^= 1; }
                wait_v14(&full[qidx], p);
                warpgroup_arrive_v14();
                #pragma unroll
                for (int m_it = 0; m_it < B_WG_M / WGMMA_M; ++m_it) {
                    __half *wgmma_sA = sA + qidx * BK * BM + 64 * (m_it + wg_idx * B_WG_M / WGMMA_M) * WGMMA_M;
                    __half *wgmma_sB = sB + qidx * BK * BN;
                    #pragma unroll
                    for (int bk = 0; bk < BK; bk += 64) {
                        #pragma unroll
                        for (int k_it = 0; k_it < 64 / WGMMA_K; ++k_it)
                            wgmma256_v14<1, 1, 1, 0, 0>(d[m_it], &wgmma_sA[k_it * WGMMA_K], &wgmma_sB[k_it * WGMMA_K]);
                        wgmma_sA += 64 * BM; wgmma_sB += 64 * BN;
                    }
                }
                warpgroup_commit_batch_v14();
                warpgroup_wait_v14();
                if (tid == 0) arrive_v14(&empty[qidx], 1);
            }

            asm volatile("cp.async.bulk.wait_group 0;");

            __half d_fp16[8];
            int* data_ptr = (int*)d_fp16;
            for (int m_it = 0; m_it < B_WG_M / WGMMA_M; ++m_it) {
                for (int w = 0; w < WGMMA_N; w += 16) {
                    uint32_t addr = base_addr + (w * B_WG_M_PADDED + m_it * WGMMA_M) * sizeof(__half);
                    for (int k = 0; k < 8; k++) d_fp16[k] = __float2half(d[m_it][w / 16][k]);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared::cta.b16 [%0], {%1, %2, %3, %4};"
                                :: "r"(addr), "r"(data_ptr[0]), "r"(data_ptr[1]), "r"(data_ptr[2]), "r"(data_ptr[3]));
                }
            }
            asm volatile("bar.sync %0, 128;\n" :: "r"(wg_idx + 2) : "memory");
            if (tid == 0) {
                store_async_v14(&tensorMapC, block_sC, num_block_m * BM + wg_idx * B_WG_M, num_block_n * BN);
                asm volatile("cp.async.bulk.commit_group;");
            }
        }
    }
}

static CUtensorMap v14_tma_map_A, v14_tma_map_B, v14_tma_map_C;
static int v14_prev_m = 0, v14_prev_n = 0, v14_prev_k = 0;
static int* v14_dspace = nullptr;

void gemm_v14_wgmma_tma_warpspecialized_persistent_ptxbarrier_tmastore_hilbert_stmatrix_padded_nocluster_fp16(
    const __half* A, const __half* B, __half* C,
    int M, int N, int K, char lhs_format, char rhs_format,
    cudaStream_t stream) {
    if (lhs_format != 'R' || rhs_format != 'C') {
        fprintf(stderr, "ERROR: CUDA V14 only supports RC mode\n");
        return;
    }
    if (M % V14_BM != 0 || N % V14_BN != 0 || K % V14_BK != 0) {
        fprintf(stderr, "ERROR: CUDA V14 requires M%%128==0, N%%256==0, K%%64==0\n");
        return;
    }

    if (M != v14_prev_m || N != v14_prev_n || K != v14_prev_k) {
        v14_tma_map_A = create_tensor_map_v14<V14_BM, V14_BK>(const_cast<__half*>(A), M, K);
        v14_tma_map_B = create_tensor_map_v14<V14_BN, V14_BK>(const_cast<__half*>(B), N, K);
        v14_tma_map_C = create_tensor_map_v14<V14_BN, V14_B_WG_M, false, true>(const_cast<__half*>(C), N, M);
        v14_prev_m = M;
        v14_prev_n = N;
        v14_prev_k = K;

        int* space = (int*)malloc(sizeof(int) * V14_NUM_SM * V14_SPACE_LEN);
        createHilbert_v14(M / V14_BM, N / V14_BN, V14_NUM_SM, space);
        if (v14_dspace) cudaFree(v14_dspace);
        cudaMalloc((void**)&v14_dspace, sizeof(int) * V14_NUM_SM * V14_SPACE_LEN);
        cudaMemcpy(v14_dspace, space, sizeof(int) * V14_NUM_SM * V14_SPACE_LEN, cudaMemcpyHostToDevice);
        free(space);
    }

    size_t sMemSize = sizeof(SMemV14<V14_BM, V14_BN, V14_BK, V14_QSIZE>);
    auto kernel = gemm_v14_stmatrix_nocluster_kernel<V14_BM, V14_BN, V14_BK, V14_NUM_THREADS, V14_QSIZE>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize);
    cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    kernel<<<V14_NUM_SM, V14_NUM_THREADS, sMemSize, stream>>>(
        M, N, K, v14_tma_map_C, v14_tma_map_A, v14_tma_map_B, v14_dspace);
}

} // namespace baseline
