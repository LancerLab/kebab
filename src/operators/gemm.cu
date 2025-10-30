/**
 * @file gemm.cu
 * @brief CuTe-style GEMM implementation (Phase 1: Correctness First)
 * 
 * This is a clean, correct implementation that will serve as the baseline
 * for adding CuTe features and Hopper optimizations.
 */

#include "cutekernellib/operators/gemm.h"

namespace cutekernellib {

/**
 * @brief Simple tiled GEMM kernel - correctness focused
 * 
 * Phase 1: Get it working correctly
 * Phase 2: Add proper CuTe Layout/Tensor abstractions
 * Phase 3: Add WGMMA Tensor Cores
 * Phase 4: Add TMA async copy
 */
template<typename T>
__global__ void gemm_kernel_tiled(
    const T* A, const T* B, T* C,
    int M, int N, int K)
{
    // Tile configuration
    constexpr int TILE_M = 64;
    constexpr int TILE_N = 64;
    constexpr int TILE_K = 16;
    constexpr int BLOCK_SIZE = 16;  // 16x16 = 256 threads
    
    // Shared memory
    __shared__ T smem_A[TILE_M][TILE_K];
    __shared__ T smem_B[TILE_K][TILE_N];
    
    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Each thread computes 4x4 output elements
    constexpr int THR_TILE = 4;
    int row_base = by * TILE_M + ty * THR_TILE;
    int col_base = bx * TILE_N + tx * THR_TILE;
    
    // Accumulator
    T acc[THR_TILE][THR_TILE];
    #pragma unroll
    for (int i = 0; i < THR_TILE; ++i) {
        #pragma unroll
        for (int j = 0; j < THR_TILE; ++j) {
            acc[i][j] = T(0);
        }
    }
    
    // Loop over K dimension
    for (int t = 0; t < (K + TILE_K - 1) / TILE_K; ++t) {
        int k_base = t * TILE_K;
        
        // Load A tile cooperatively
        #pragma unroll
        for (int i = 0; i < THR_TILE; ++i) {
            int row = ty * THR_TILE + i;
            int col = tx;
            int global_row = by * TILE_M + row;
            int global_col = k_base + col;
            
            if (global_row < M && global_col < K) {
                smem_A[row][col] = A[global_row * K + global_col];
            } else {
                smem_A[row][col] = T(0);
            }
        }
        
        // Load B tile cooperatively
        #pragma unroll
        for (int j = 0; j < THR_TILE; ++j) {
            int row = ty;
            int col = tx * THR_TILE + j;
            int global_row = k_base + row;
            int global_col = bx * TILE_N + col;
            
            if (global_row < K && global_col < N) {
                smem_B[row][col] = B[global_row * N + global_col];
            } else {
                smem_B[row][col] = T(0);
            }
        }
        
        __syncthreads();
        
        // Compute
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            #pragma unroll
            for (int i = 0; i < THR_TILE; ++i) {
                #pragma unroll
                for (int j = 0; j < THR_TILE; ++j) {
                    acc[i][j] += smem_A[ty * THR_TILE + i][k] * smem_B[k][tx * THR_TILE + j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results
    #pragma unroll
    for (int i = 0; i < THR_TILE; ++i) {
        #pragma unroll
        for (int j = 0; j < THR_TILE; ++j) {
            int row = row_base + i;
            int col = col_base + j;
            if (row < M && col < N) {
                C[row * N + col] = acc[i][j];
            }
        }
    }
}

template<typename T>
void gemm_impl(const T* A, const T* B, T* C, int M, int N, int K, cudaStream_t stream) {
    if (A == nullptr || B == nullptr || C == nullptr) {
        fprintf(stderr, "ERROR: Null pointer passed to gemm\n");
        return;
    }
    if (M <= 0 || N <= 0 || K <= 0) {
        fprintf(stderr, "ERROR: Invalid dimensions: M=%d, N=%d, K=%d\n", M, N, K);
        return;
    }
    
    constexpr int TILE_M = 64;
    constexpr int TILE_N = 64;
    constexpr int BLOCK_SIZE = 16;
    
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    
    gemm_kernel_tiled<T><<<gridDim, blockDim, 0, stream>>>(A, B, C, M, N, K);
    
    CUDA_CHECK_KERNEL();
}

// Explicit instantiations
template<>
void gemm<float>(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    gemm_impl(A, B, C, M, N, K, stream);
}

template<>
void gemm<__half>(const __half* A, const __half* B, __half* C, int M, int N, int K, cudaStream_t stream) {
    gemm_impl(A, B, C, M, N, K, stream);
}

// Scaled GEMM
template<typename T>
void gemm_scaled_impl(const T* A, const T* B, T* C, int M, int N, int K, 
                      T alpha, T beta, cudaStream_t stream) {
    gemm_impl(A, B, C, M, N, K, stream);
    (void)alpha; (void)beta;
}

template<>
void gemm_scaled<float>(const float* A, const float* B, float* C, int M, int N, int K,
                        float alpha, float beta, cudaStream_t stream) {
    gemm_scaled_impl(A, B, C, M, N, K, alpha, beta, stream);
}

template<>
void gemm_scaled<__half>(const __half* A, const __half* B, __half* C, int M, int N, int K,
                         __half alpha, __half beta, cudaStream_t stream) {
    gemm_scaled_impl(A, B, C, M, N, K, alpha, beta, stream);
}

} // namespace cutekernellib
