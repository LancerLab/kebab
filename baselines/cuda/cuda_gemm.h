#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace baseline {

/**
 * @brief Hand-optimized CUDA baseline for GEMM: C = A * B
 * 
 * This implementation uses the same API signature as the CuTe version
 * for fair performance comparison.
 * 
 * Optimizations:
 * - Tensor Core utilization via wmma/mma.sync instructions
 * - Shared memory tiling (128x128 tiles with 32x32 sub-tiles)
 * - Double buffering for overlapping compute and memory
 * - Warp-level matrix operations for maximum throughput
 * - Memory coalescing for optimal bandwidth utilization
 * - Register blocking to maximize occupancy
 * 
 * Performance targets:
 * - Competitive with cuBLAS performance
 * - High Tensor Core utilization (>80%)
 * - Optimal memory bandwidth usage
 * 
 * @param A Input matrix A (M x K, row-major)
 * @param B Input matrix B (K x N, row-major)
 * @param C Output matrix C (M x N, row-major)
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @param stream CUDA stream for asynchronous execution (default: 0)
 */
void gemm(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream = 0);
void gemm(const __half* A, const __half* B, __half* C, int M, int N, int K, cudaStream_t stream = 0);

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