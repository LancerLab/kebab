#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace baseline {

/**
 * @brief CUDA baseline GEMM with version dispatch
 *
 * Version 1: Warp Tiling (based on fast.cu kernel 1, RC mode only)
 * Version 2: WGMMA + TMA (future, based on fast.cu kernel 2)
 *
 * @param A Input matrix A (M x K)
 * @param B Input matrix B (K x N)
 * @param C Output matrix C (M x N, column-major)
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @param opmode Storage format: "RC" (A row-major, B col-major)
 * @param version Kernel variant ID (default: 1)
 * @param stream CUDA stream for asynchronous execution (default: 0)
 */
void gemm(const float* A, const float* B, float* C, int M, int N, int K,
          const char* opmode = "RC", int version = 1, cudaStream_t stream = 0);
void gemm(const __half* A, const __half* B, __half* C, int M, int N, int K,
          const char* opmode = "RC", int version = 1, cudaStream_t stream = 0);

// ============================================================================
// Internal kernel dispatchers (called by gemm())
// ============================================================================

/**
 * @brief V1: Warp Tiling kernel (RC mode)
 * A: row-major, B: column-major, C: column-major
 */
void gemm_v1_warptiling_fp16(const __half* A, const __half* B, __half* C,
                              int M, int N, int K, char lhs_format, char rhs_format,
                              cudaStream_t stream);

/**
 * @brief V2: WGMMA + TMA kernel (RC mode, SM90 Hopper required)
 * A: row-major, B: column-major, C: column-major
 * Uses TMA for efficient global memory loads
 * Uses WGMMA 64x64x16 for Tensor Core computation
 */
void gemm_v2_wgmma_tma_fp16(const __half* A, const __half* B, __half* C,
                            int M, int N, int K, char lhs_format, char rhs_format,
                            cudaStream_t stream);

/**
 * @brief V3: Warp Group kernel with larger tiles (RC mode, SM90 Hopper required)
 * A: row-major, B: column-major, C: column-major
 * Uses larger 128x128x64 tiles
 * Uses WGMMA 64xNx16 with full N dimension per WGMMA
 */
void gemm_v3_warpgroup_fp16(const __half* A, const __half* B, __half* C,
                            int M, int N, int K, char lhs_format, char rhs_format,
                            cudaStream_t stream);

/**
 * @brief V4: Warp Specialization kernel with multi-stage pipeline (RC mode, SM90 Hopper required)
 * A: row-major, B: column-major, C: column-major
 * Producer-Consumer pattern: warp-group 0 does TMA loads, warp-group 1 does WGMMA compute
 * 5-stage pipeline with full/empty barriers
 */
void gemm_v4_warpspec_fp16(const __half* A, const __half* B, __half* C,
                           int M, int N, int K, char lhs_format, char rhs_format,
                           cudaStream_t stream);

/**
 * @brief V5: Larger tiles + register optimization (RC mode, SM90 Hopper required)
 * 128×256×64 tiles, 3 warp-groups (1 producer + 2 consumers), dynamic register allocation
 */
void gemm_v5_persistent_fp16(const __half* A, const __half* B, __half* C,
                             int M, int N, int K, char lhs_format, char rhs_format,
                             cudaStream_t stream);

/**
 * @brief V6: Persistent kernel + tile scheduling (RC mode, SM90 Hopper required)
 * 128×256×64 tiles, grid-constant TMA, 16×8 tile schedule pattern
 */
void gemm_v6_ptxbarrier_fp16(const __half* A, const __half* B, __half* C,
                              int M, int N, int K, char lhs_format, char rhs_format,
                              cudaStream_t stream);

} // namespace baseline