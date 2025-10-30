#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>

namespace cutekernellib {

// CUDA error checking macro (reuse from elementwise_add.h)
#ifndef CUDA_CHECK
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            fprintf(stderr, "Error code: %d\n", err); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
#endif

// Macro for kernel launch error checking
#ifndef CUDA_CHECK_KERNEL
#define CUDA_CHECK_KERNEL() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "Kernel launch error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
        err = cudaDeviceSynchronize(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "Kernel execution error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
#endif

/**
 * @brief General Matrix Multiplication: C = A * B
 * 
 * Performs C = A * B where:
 * - A is M x K matrix (row-major)
 * - B is K x N matrix (row-major) 
 * - C is M x N matrix (row-major)
 * 
 * This implementation uses CuTe MMA atoms and Tensor Cores for maximum performance.
 * Targets ≥90% of cuBLAS performance through:
 * - Tensor Core utilization via MMA atoms (SM80_16x8x16_F32F16F16F32_TN for Ampere)
 * - TiledMMA for thread block organization
 * - Software pipelining with async copy (cp.async) for data loading
 * - Shared memory tiling for A and B matrices
 * 
 * @tparam T Data type (float or half)
 * @param A Input matrix A (M x K)
 * @param B Input matrix B (K x N)
 * @param C Output matrix C (M x N)
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @param stream CUDA stream for asynchronous execution (default: 0)
 */
template<typename T>
void gemm(const T* A, const T* B, T* C, int M, int N, int K, cudaStream_t stream = 0);

/**
 * @brief GEMM with alpha and beta scaling: C = alpha * A * B + beta * C
 * 
 * @tparam T Data type (float or half)
 * @param A Input matrix A (M x K)
 * @param B Input matrix B (K x N)
 * @param C Input/Output matrix C (M x N)
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @param alpha Scaling factor for A*B
 * @param beta Scaling factor for C
 * @param stream CUDA stream for asynchronous execution (default: 0)
 */
template<typename T>
void gemm_scaled(const T* A, const T* B, T* C, int M, int N, int K, 
                 T alpha, T beta, cudaStream_t stream = 0);

} // namespace cutekernellib