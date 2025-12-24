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

// ============================================================================
// VERBOSE DEBUG MODE - Set to 1 to enable detailed debug output
// ============================================================================
#define V2_VERBOSE_DEBUG 1

#include "kebab/cuda/cuda_gemm.h"
#include "kebab/cuda/cuda_kernel_utils.h"
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
using namespace cuda_kernel;

// ============================================================================
// TMA Tensor Map Creation (using utilities from cuda_kernel_utils.h)
// ============================================================================
// Note: create_tensor_map_tma and allocate_and_create_tensor_map_tma are now
// defined in cuda_kernel_utils.h and can be reused by all kernels

// Kernel configuration
constexpr int V2_BM = 64;
constexpr int V2_BN = 64;
constexpr int V2_WGMMA_M = 64;
constexpr int V2_WGMMA_N = 64;
constexpr int V2_WGMMA_K = 16;
constexpr int V2_NUM_THREADS = 128;

// ============================================================================
// WGMMA+TMA Kernel
// ============================================================================

template<WGMMA_MajorOrder MajorOrderA, WGMMA_MajorOrder MajorOrderB,
         WGMMA_Swizzle SwizzleA, WGMMA_Swizzle SwizzleB, typename T>
__global__ void __launch_bounds__(V2_NUM_THREADS)
gemm_v2_wgmma_tma_kernel(int M, int N, int K, T* C,
                          const CUtensorMap* tensorMapA,
                          const CUtensorMap* tensorMapB) {

    // Compute V2_BK based on swizzle pattern
    // V2_BK * sizeof(T) must equal swizzle width in bytes
    // B32: 32 bytes, B64: 64 bytes, B128: 128 bytes
    constexpr int V2_BK =
        (SwizzleA == WGMMA_Swizzle::B32) ? (32 / sizeof(T)) :
        (SwizzleA == WGMMA_Swizzle::B64) ? (64 / sizeof(T)) :
        (SwizzleA == WGMMA_Swizzle::B128) ? (128 / sizeof(T)) :
        (16 / sizeof(T)); // default to NS

    // Shared memory for A and B tiles
    // Alignment must be at least 128 bytes for TMA
    __shared__ alignas(256) T sA[V2_BM * V2_BK];
    __shared__ alignas(256) T sB[V2_BK * V2_BN];

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

            if (MajorOrderB == WGMMA_MajorOrder::K_MAJOR) {
                cde::cp_async_bulk_tensor_2d_global_to_shared(
                    &sB[0], tensorMapB, block_k_iter * V2_BK, num_block_n * V2_BN, barB);
            } else {
                cde::cp_async_bulk_tensor_2d_global_to_shared(
                    &sB[0], tensorMapB, num_block_n * V2_BN, block_k_iter * V2_BK, barB);
            }
            tokenB = cuda::device::barrier_arrive_tx(barB, 1, sizeof(sB));
        } else {
            tokenA = barA.arrive();
            tokenB = barB.arrive();
        }
        barA.wait(std::move(tokenA));
        barB.wait(std::move(tokenB));
        __syncthreads();

        // VERBOSE: Print shared memory data after TMA load
        #if V2_VERBOSE_DEBUG
        if (threadIdx.x == 0 && block_k_iter == 0) {
            printf("\n=== Block %d: Shared Memory After TMA Load (K-iter %d) ===\n", blockIdx.x, block_k_iter);
            printf("sA (first 16x16):\n");
            for (int i = 0; i < 16; ++i) {
                for (int j = 0; j < 16; ++j) {
                    printf("%.2f ", static_cast<float>(sA[i * V2_BK + j]));
                }
                printf("\n");
            }
            printf("\nsB (first 16x16):\n");
            for (int i = 0; i < 16; ++i) {
                for (int j = 0; j < 16; ++j) {
                    printf("%.2f ", static_cast<float>(sB[i * V2_BN + j]));
                }
                printf("\n");
            }
        }
        __syncthreads();
        #endif

        // WGMMA Compute: 4 iterations of K=16 each (total BK=64)
        warpgroup_arrive();
        #pragma unroll
        for (int m_iter = 0; m_iter < V2_BM / V2_WGMMA_M; ++m_iter) {
            #pragma unroll
            for (int n_iter = 0; n_iter < V2_BN / V2_WGMMA_N; ++n_iter) {
                #pragma unroll
                for (int k_iter = 0; k_iter < V2_BK / V2_WGMMA_K; ++k_iter) {
                    // Use unified wgmma64 API for both FP16 and BF16
                    if constexpr (MajorOrderB == WGMMA_MajorOrder::K_MAJOR) {
                        wgmma<WGMMA_MMAShape::M64N64K16, MajorOrderA, MajorOrderB, SwizzleA, SwizzleB, T>(
                            d[m_iter][n_iter],
                            &sA[m_iter * V2_WGMMA_M * V2_BK + k_iter * V2_WGMMA_K],
                            &sB[n_iter * V2_WGMMA_N * V2_BK + k_iter * V2_WGMMA_K]);
                    } else {
                        wgmma<WGMMA_MMAShape::M64N64K16, MajorOrderA, MajorOrderB, SwizzleA, SwizzleB, T>(
                            d[m_iter][n_iter],
                            &sA[m_iter * V2_WGMMA_M * V2_BK + k_iter * V2_WGMMA_K],
                            &sB[k_iter * V2_WGMMA_K + n_iter * V2_WGMMA_N * V2_BK]);
                    }
                }
            }
        }
        warpgroup_commit_batch();
        warpgroup_wait<0>();

        // VERBOSE: Print accumulator after WGMMA (only first K-iter)
        #if V2_VERBOSE_DEBUG
        if (threadIdx.x == 0 && block_k_iter == 0) {
            printf("\n=== Block %d: Accumulator After WGMMA (K-iter %d) ===\n", blockIdx.x, block_k_iter);
            printf("d[0][0] (first 16 values):\n");
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    printf("%.2f ", d[0][0][0][i*4+j]);
                }
                printf("\n");
            }
        }
        __syncthreads();
        #endif
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
        T *block_C = C + num_block_n * V2_BN * M + num_block_m * V2_BM;

        for (int m_it = 0; m_it < V2_BM / V2_WGMMA_M; ++m_it) {
            for (int n_it = 0; n_it < V2_BN / V2_WGMMA_N; ++n_it) {
                for (int w = 0; w < V2_WGMMA_N / 16; ++w) {
                    int col = 16 * w + 2 * (tid % 4);
                    #define IDX(i, j) ((j + n_it * V2_WGMMA_N) * M + ((i) + m_it * V2_WGMMA_M))

                    if constexpr (std::is_same_v<T, __half>) {
                        block_C[IDX(row, col)] = __float2half(d[m_it][n_it][w][0]);
                        block_C[IDX(row, col + 1)] = __float2half(d[m_it][n_it][w][1]);
                        block_C[IDX(row + 8, col)] = __float2half(d[m_it][n_it][w][2]);
                        block_C[IDX(row + 8, col + 1)] = __float2half(d[m_it][n_it][w][3]);
                        block_C[IDX(row, col + 8)] = __float2half(d[m_it][n_it][w][4]);
                        block_C[IDX(row, col + 9)] = __float2half(d[m_it][n_it][w][5]);
                        block_C[IDX(row + 8, col + 8)] = __float2half(d[m_it][n_it][w][6]);
                        block_C[IDX(row + 8, col + 9)] = __float2half(d[m_it][n_it][w][7]);
                    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
                        block_C[IDX(row, col)] = __float2bfloat16(d[m_it][n_it][w][0]);
                        block_C[IDX(row, col + 1)] = __float2bfloat16(d[m_it][n_it][w][1]);
                        block_C[IDX(row + 8, col)] = __float2bfloat16(d[m_it][n_it][w][2]);
                        block_C[IDX(row + 8, col + 1)] = __float2bfloat16(d[m_it][n_it][w][3]);
                        block_C[IDX(row, col + 8)] = __float2bfloat16(d[m_it][n_it][w][4]);
                        block_C[IDX(row, col + 9)] = __float2bfloat16(d[m_it][n_it][w][5]);
                        block_C[IDX(row + 8, col + 8)] = __float2bfloat16(d[m_it][n_it][w][6]);
                        block_C[IDX(row + 8, col + 9)] = __float2bfloat16(d[m_it][n_it][w][7]);
                    }

                    #undef IDX
                }
            }
        }

        // VERBOSE: Print final result (column-major C, first 16x16)
        #if V2_VERBOSE_DEBUG
        if (threadIdx.x == 0) {
            printf("\n=== Block %d: Final Result C (first 16x16, column-major) ===\n", blockIdx.x);
            for (int i = 0; i < 16; ++i) {
                for (int j = 0; j < 16; ++j) {
                    // Column-major indexing: C[i,j] = block_C[j*M + i]
                    printf("%.2f ", static_cast<float>(block_C[j * M + i]));
                }
                printf("\n");
            }
            printf("=== END RESULT ===\n\n");
        }
        #endif
    }
}


// ============================================================================
// Host Function
// ============================================================================

// Cached TMA maps (for efficiency across calls with same dimensions)
// Note: We need to track swizzle pattern as well, since different swizzles require different TMA maps
static CUtensorMap *v2_tma_map_A = nullptr;
static CUtensorMap *v2_tma_map_B = nullptr;
static int v2_prev_m = 0, v2_prev_n = 0, v2_prev_k = 0;
static int v2_prev_swizzle_a = -1, v2_prev_swizzle_b = -1;

template<WGMMA_Swizzle SwizzleA, WGMMA_Swizzle SwizzleB, typename T>
void gemm_v2_wgmma_tma(const T* A, const T* B, T* C,
                            int M, int N, int K, char lhs_format, char rhs_format,
                            cudaStream_t stream) {
    // Compute V2_BK based on swizzle pattern
    // V2_BK * sizeof(T) must equal swizzle width in bytes
    // B32: 32 bytes, B64: 64 bytes, B128: 128 bytes
    constexpr int V2_BK =
        (SwizzleA == WGMMA_Swizzle::B32) ? (32 / sizeof(T)) :
        (SwizzleA == WGMMA_Swizzle::B64) ? (64 / sizeof(T)) :
        (SwizzleA == WGMMA_Swizzle::B128) ? (128 / sizeof(T)) :
        (16 / sizeof(T)); // default to NS

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

    // Create or update TMA maps if dimensions or swizzle pattern changed
    int swizzle_a_int = static_cast<int>(SwizzleA);
    int swizzle_b_int = static_cast<int>(SwizzleB);

    if (!v2_tma_map_A || M != v2_prev_m || N != v2_prev_n || K != v2_prev_k ||
        swizzle_a_int != v2_prev_swizzle_a || swizzle_b_int != v2_prev_swizzle_b) {
        if (v2_tma_map_A) free_tensor_map_tma(v2_tma_map_A);
        if (v2_tma_map_B) free_tensor_map_tma(v2_tma_map_B);

        if (lhs_format == 'R' && rhs_format == 'C') {
            v2_tma_map_A = allocate_and_create_tensor_map_tma<SwizzleA, V2_BM, V2_BK, T>(
                const_cast<T*>(A), M / V2_BM, K / V2_BK);
            v2_tma_map_B = allocate_and_create_tensor_map_tma<SwizzleB, V2_BN, V2_BK, T>(
                const_cast<T*>(B), N / V2_BN, K / V2_BK);
        } else if (lhs_format == 'R' && rhs_format == 'R') {
            v2_tma_map_A = allocate_and_create_tensor_map_tma<SwizzleA, V2_BM, V2_BK, T>(
                const_cast<T*>(A), M / V2_BM, K / V2_BK);
            v2_tma_map_B = allocate_and_create_tensor_map_tma<SwizzleB, V2_BK, V2_BN, T>(
                const_cast<T*>(B), K / V2_BK, N / V2_BN);
        }

        v2_prev_m = M;
        v2_prev_n = N;
        v2_prev_k = K;
        v2_prev_swizzle_a = swizzle_a_int;
        v2_prev_swizzle_b = swizzle_b_int;
    }

    dim3 grid((M / V2_BM) * (N / V2_BN));
    if (lhs_format == 'R' && rhs_format == 'R') {
        // RR mode: A row-major, B row-major
        gemm_v2_wgmma_tma_kernel<WGMMA_MajorOrder::K_MAJOR, WGMMA_MajorOrder::MN_MAJOR, SwizzleA, SwizzleB, T><<<grid, V2_NUM_THREADS, 0, stream>>>(
            M, N, K, const_cast<T*>(C), v2_tma_map_A, v2_tma_map_B);
    } else if (lhs_format == 'R' && rhs_format == 'C') {
        // RC mode: A row-major, B column-major
        gemm_v2_wgmma_tma_kernel<WGMMA_MajorOrder::K_MAJOR, WGMMA_MajorOrder::K_MAJOR, SwizzleA, SwizzleB, T><<<grid, V2_NUM_THREADS, 0, stream>>>(
            M, N, K, const_cast<T*>(C), v2_tma_map_A, v2_tma_map_B);
    }
}

// ============================================================================
// Explicit Template Instantiations
// ============================================================================

// FP16 instantiations with B32 swizzle (default)
template void gemm_v2_wgmma_tma<cuda_kernel::WGMMA_Swizzle::B32, cuda_kernel::WGMMA_Swizzle::B32, __half>(
    const __half* A, const __half* B, __half* C,
    int M, int N, int K, char lhs_format, char rhs_format, cudaStream_t stream);

// FP16 instantiations with B64 swizzle
template void gemm_v2_wgmma_tma<cuda_kernel::WGMMA_Swizzle::B64, cuda_kernel::WGMMA_Swizzle::B64, __half>(
    const __half* A, const __half* B, __half* C,
    int M, int N, int K, char lhs_format, char rhs_format, cudaStream_t stream);

// FP16 instantiations with B128 swizzle
template void gemm_v2_wgmma_tma<cuda_kernel::WGMMA_Swizzle::B128, cuda_kernel::WGMMA_Swizzle::B128, __half>(
    const __half* A, const __half* B, __half* C,
    int M, int N, int K, char lhs_format, char rhs_format, cudaStream_t stream);

template void gemm_v2_wgmma_tma<cuda_kernel::WGMMA_Swizzle::NS, cuda_kernel::WGMMA_Swizzle::NS, __half>(
    const __half* A, const __half* B, __half* C,
    int M, int N, int K, char lhs_format, char rhs_format, cudaStream_t stream);

// BF16 instantiations with B32 swizzle (default)
template void gemm_v2_wgmma_tma<cuda_kernel::WGMMA_Swizzle::B32, cuda_kernel::WGMMA_Swizzle::B32, __nv_bfloat16>(
    const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
    int M, int N, int K, char lhs_format, char rhs_format, cudaStream_t stream);

// BF16 instantiations with B64 swizzle
template void gemm_v2_wgmma_tma<cuda_kernel::WGMMA_Swizzle::B64, cuda_kernel::WGMMA_Swizzle::B64, __nv_bfloat16>(
    const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
    int M, int N, int K, char lhs_format, char rhs_format, cudaStream_t stream);

// BF16 instantiations with B128 swizzle
template void gemm_v2_wgmma_tma<cuda_kernel::WGMMA_Swizzle::B128, cuda_kernel::WGMMA_Swizzle::B128, __nv_bfloat16>(
    const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
    int M, int N, int K, char lhs_format, char rhs_format, cudaStream_t stream);

// BF16 instantiations with NS swizzle
template void gemm_v2_wgmma_tma<cuda_kernel::WGMMA_Swizzle::NS, cuda_kernel::WGMMA_Swizzle::NS, __nv_bfloat16>(
    const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
    int M, int N, int K, char lhs_format, char rhs_format, cudaStream_t stream);

} // namespace baseline

