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

} // namespace baseline