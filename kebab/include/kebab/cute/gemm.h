#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace kebab {
namespace cute {

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
 * @brief General Matrix Multiplication with version dispatch: C = A * B
 * 
 * Version 1: WGMMA without TMA (Hopper SM90+)
 * Version 2: WGMMA with TMA (future)
 * Version 3: Optimized tile sizes (future)
 * 
 * @tparam T Data type (float or half)
 * @param A Input matrix A (M x K)
 * @param B Input matrix B (K x N)
 * @param C Output matrix C (M x N)
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @param opmode Operation mode (default: "default")
 * @param version Kernel variant ID (default: 1)
 * @param stream CUDA stream for asynchronous execution (default: 0)
 */
template<typename T>
void gemm(const T* A, const T* B, T* C, int M, int N, int K, const char* opmode = "default", int version = 1, cudaStream_t stream = 0);

/**
 * @brief Complete WGMMA-based GEMM dispatch for FP16 (Hopper SM90+)
 * 
 * @param A Input matrix A (M x K, row-major)
 * @param B Input matrix B (K x N, row-major)
 * @param C Output matrix C (M x N, row-major)
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @param lhs_format Storage format for A ('R' for row-major, 'C' for column-major)
 * @param rhs_format Storage format for B ('R' for row-major, 'C' for column-major)
 * @param tile_M Tile size for M dimension
 * @param tile_N Tile size for N dimension
 * @param tile_K Tile size for K dimension
 * @param stream CUDA stream for asynchronous execution
 */
void gemm_wgmma_fp16_dispatch(const void* A, const void* B, void* C,
                              int M, int N, int K, 
                              char lhs_format, char rhs_format,
                              int tile_M, int tile_N, int tile_K,
                              cudaStream_t stream = 0);

/**
 * @brief WGMMA-based GEMM for FP16 with configurable tile sizes from config.yaml
 * 
 * @param A Input matrix A
 * @param B Input matrix B
 * @param C Output matrix C
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @param lhs_format Storage format for A ('R' for row-major, 'C' for column-major)
 * @param rhs_format Storage format for B ('R' for row-major, 'C' for column-major)
 * @param stream CUDA stream for asynchronous execution
 */
void gemm_wgmma_fp16(const void* A, const void* B, void* C,
                     int M, int N, int K, 
                     char lhs_format, char rhs_format,
                     cudaStream_t stream = 0);

// Version 2: WGMMA with TMA
void gemm_wgmma_tma_fp16_dispatch(const void* A, const void* B, void* C,
                                  int M, int N, int K, 
                                  char lhs_format, char rhs_format,
                                  int tile_M, int tile_N, int tile_K,
                                  cudaStream_t stream);

// Version 2 with config support
void gemm_wgmma_tma_fp16(const void* A, const void* B, void* C,
                         int M, int N, int K,
                         char lhs_format, char rhs_format,
                         cudaStream_t stream);

// Version 3: Optimized multi-stage pipeline with TMA
void gemm_wgmma_tma_v3_fp16_dispatch(const void* A, const void* B, void* C,
                                     int M, int N, int K,
                                     char lhs_format, char rhs_format,
                                     int tile_M, int tile_N, int tile_K,
                                     cudaStream_t stream);

// Version 3 with config support
void gemm_wgmma_tma_v3_fp16(const void* A, const void* B, void* C,
                            int M, int N, int K,
                            char lhs_format, char rhs_format,
                            cudaStream_t stream);


} // namespace cute
} // namespace kebab
