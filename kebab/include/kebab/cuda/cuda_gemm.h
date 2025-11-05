#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace baseline {

/**
 * @brief CUDA baseline GEMM with version dispatch
 * 
 * Version 1: Naive implementation (no tiling, no optimization)
 * Version 2: Shared memory tiling (future)
 * Version 3: WMMA Tensor Cores (future)
 * 
 * @param A Input matrix A (M x K, storage format depends on opmode)
 * @param B Input matrix B (K x N, storage format depends on opmode)
 * @param C Output matrix C (M x N, row-major)
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @param opmode Storage format: "RC" (A row-major, B col-major), "CR" (A col-major, B row-major)
 * @param version Kernel variant ID (default: 1)
 * @param stream CUDA stream for asynchronous execution (default: 0)
 */
void gemm(const float* A, const float* B, float* C, int M, int N, int K, const char* opmode = "RR", int version = 1, cudaStream_t stream = 0);
void gemm(const __half* A, const __half* B, __half* C, int M, int N, int K, const char* opmode = "RR", int version = 1, cudaStream_t stream = 0);

/**
 * @brief GEMM with alpha and beta scaling: C = alpha * A * B + beta * C
 * 
 * @param A Input matrix A (M x K, row-major)
 * @param B Input matrix B (K x N, row-major)
 * @param C Input/Output matrix C (M x N, row-major)
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @param alpha Scaling factor for A*B
 * @param beta Scaling factor for C
 * @param stream CUDA stream for asynchronous execution (default: 0)
 */
void gemm_scaled(const float* A, const float* B, float* C, int M, int N, int K, 
                 float alpha, float beta, cudaStream_t stream = 0);
void gemm_scaled(const __half* A, const __half* B, __half* C, int M, int N, int K, 
                 __half alpha, __half beta, cudaStream_t stream = 0);

} // namespace baseline