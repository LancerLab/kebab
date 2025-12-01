/**
 * @file cuda_gemm_v1_warptiling.cu
 * @brief CUDA V1 GEMM using Warp Tiling (based on fast.cu kernel 1)
 *
 * Hierarchical tiling architecture:
 * - Block tile: 128×128 with BK=16
 * - Warp tile: 64×64 per warp (2×2 warp grid)
 * - Thread tile: 8×4 per thread
 *
 * RC mode layout (matches cuBLAS CUBLAS_OP_T, CUBLAS_OP_N):
 * - A: M×K row-major (ldA=K)
 * - B: K×N column-major (ldB=K)
 * - C: M×N column-major (ldC=M)
 */

#include "kebab/cuda/cuda_gemm.h"
#include <cuda_fp16.h>
#include <cstdio>

namespace baseline {

// Kernel configuration
constexpr int BM = 128, BN = 128, BK = 16;
constexpr int WM = 64, WN = 64, WNITER = 4;
constexpr int TM = 8, TN = 4;
constexpr int NUM_THREADS = 128;
constexpr int WMITER = (WM * WN) / (32 * TM * TN * WNITER);  // = 2
constexpr int WSUBM = WM / WMITER;  // = 32
constexpr int WSUBN = WN / WNITER;  // = 16

__global__ void __launch_bounds__(NUM_THREADS)
gemm_v1_warptiling_rc_kernel(int M, int N, int K,
                              const __half* __restrict__ A,
                              const __half* __restrict__ B,
                              __half* __restrict__ C) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    // Warp and thread placement
    const uint warpIdx = threadIdx.x / 32;
    const uint warpCol = warpIdx % (BN / WN);
    const uint warpRow = warpIdx / (BN / WN);
    const uint threadIdxInWarp = threadIdx.x % 32;
    const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN);
    const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN);

    __shared__ __half As[BM * BK];
    __shared__ __half Bs[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN * K;

    // A loading indices (row-major): load 4 consecutive k values
    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;

    // B loading indices (column-major): load 4 consecutive k values (coalesced!)
    const uint innerColB_cm = threadIdx.x / (BK / 4);  // n index
    const uint innerRowB_cm = threadIdx.x % (BK / 4);  // k index / 4
    constexpr uint colStrideB = (NUM_THREADS * 4) / BK;

    float threadResults[WMITER * TM * WNITER * TN] = {0.0f};
    __half regM[WMITER * TM];
    __half regN[WNITER * TN];

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // Load A (row-major) -> As (transposed)
        for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
            __half tmp[4];
            const float2 x = *reinterpret_cast<const float2*>(
                &A[(innerRowA + offset) * K + innerColA * 4]);
            memcpy(&tmp[0], &x, sizeof(__half) * 4);
            As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp[0];
            As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp[1];
            As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp[2];
            As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp[3];
        }

        // Load B (column-major: B[k,n] = B[n*K + k]) -> Bs
        // Coalesced: consecutive threads load consecutive k values
        for (uint offset = 0; offset + colStrideB <= BN; offset += colStrideB) {
            __half tmp[4];
            uint n_idx = innerColB_cm + offset;
            uint k_base = innerRowB_cm * 4;
            const float2 x = *reinterpret_cast<const float2*>(&B[n_idx * K + k_base]);
            memcpy(&tmp[0], &x, sizeof(__half) * 4);
            Bs[(k_base + 0) * BN + n_idx] = tmp[0];
            Bs[(k_base + 1) * BN + n_idx] = tmp[1];
            Bs[(k_base + 2) * BN + n_idx] = tmp[2];
            Bs[(k_base + 3) * BN + n_idx] = tmp[3];
        }
        __syncthreads();

        // Compute tile
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                for (uint i = 0; i < TM; ++i) {
                    regM[wSubRowIdx * TM + i] = As[(dotIdx * BM) + warpRow * WM +
                        wSubRowIdx * WSUBM + threadRowInWarp * TM + i];
                }
            }
            for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                for (uint i = 0; i < TN; ++i) {
                    regN[wSubColIdx * TN + i] = Bs[(dotIdx * BN) + warpCol * WN +
                        wSubColIdx * WSUBN + threadColInWarp * TN + i];
                }
            }
            for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                            threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                                          (wSubColIdx * TN) + resIdxN] +=
                                __half2float(regM[wSubRowIdx * TM + resIdxM]) *
                                __half2float(regN[wSubColIdx * TN + resIdxN]);
                        }
                    }
                }
            }
        }
        A += BK;
        B += BK;
        __syncthreads();
    }

    // Store to column-major C: C[m,n] = C[n*M + m]
    const uint globalRowBase = cRow * BM + warpRow * WM;
    const uint globalColBase = cCol * BN + warpCol * WN;
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    const uint globalRow = globalRowBase + wSubRowIdx * WSUBM +
                                           threadRowInWarp * TM + resIdxM;
                    const uint globalCol = globalColBase + wSubColIdx * WSUBN +
                                           threadColInWarp * TN + resIdxN;
                    const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                                  wSubColIdx * TN + resIdxN;
                    C[globalCol * M + globalRow] = __float2half(threadResults[i]);
                }
            }
        }
    }
}

void gemm_v1_warptiling_fp16(const __half* A, const __half* B, __half* C,
                              int M, int N, int K, char lhs_format, char rhs_format,
                              cudaStream_t stream) {
    if (lhs_format != 'R' || rhs_format != 'C') {
        fprintf(stderr, "ERROR: CUDA V1 (warp tiling) only supports RC mode\n");
        return;
    }
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_v1_warptiling_rc_kernel<<<gridDim, NUM_THREADS, 0, stream>>>(M, N, K, A, B, C);
}

} // namespace baseline

