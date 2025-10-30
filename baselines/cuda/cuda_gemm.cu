#include "cuda_gemm.h"
#include <mma.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>

namespace baseline {

// CUDA error checking macro
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

// Macro for kernel launch error checking
#define CUDA_CHECK_KERNEL() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "Kernel launch error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

using namespace nvcuda;

// ============================================================================
// Tensor Core GEMM Implementation for Half Precision
// ============================================================================

/**
 * @brief Simple GEMM kernel using Tensor Cores (wmma) for half precision
 * 
 * This kernel uses wmma API for Tensor Core acceleration.
 * Each warp computes a 16x16 output tile.
 */
__global__ void gemm_wmma_kernel_simple(
    const __half* __restrict__ A,
    const __half* __restrict__ B, 
    __half* __restrict__ C,
    int M, int N, int K)
{
    // Each block has one warp, each warp handles one 16x16 tile
    int globalRowC = blockIdx.y * 16;
    int globalColC = blockIdx.x * 16;
    
    // Bounds check
    if (globalRowC >= M || globalColC >= N) return;
    
    // WMMA fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> c_frag;
    
    // Initialize accumulator to zero
    wmma::fill_fragment(c_frag, __float2half(0.0f));
    
    // Loop over K dimension in steps of 16
    for (int k = 0; k < K; k += 16) {
        // Bounds check for K dimension
        if (k + 16 <= K) {
            // Load A and B fragments
            wmma::load_matrix_sync(a_frag, A + globalRowC * K + k, K);
            wmma::load_matrix_sync(b_frag, B + k * N + globalColC, N);
            
            // Perform matrix multiplication
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        } else {
            // Handle remaining K dimension with padding
            // For simplicity, skip partial tiles (can be improved)
            break;
        }
    }
    
    // Store result
    if (globalRowC + 16 <= M && globalColC + 16 <= N) {
        wmma::store_matrix_sync(C + globalRowC * N + globalColC, c_frag, N, wmma::mem_row_major);
    }
}

/**
 * @brief Optimized GEMM kernel for float precision using tiled matrix multiplication
 * 
 * This kernel uses shared memory tiling and register blocking for optimal performance.
 * Each thread computes an 8x8 tile of the output matrix.
 */
template<int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void gemm_float_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    // Thread and block indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    // Thread block is 16x16 = 256 threads
    const int tid = ty * 16 + tx;
    
    // Shared memory for tiling
    __shared__ float As[BLOCK_M][BLOCK_K + 4]; // +4 for bank conflict avoidance
    __shared__ float Bs[BLOCK_K][BLOCK_N + 4];
    
    // Register arrays for accumulation - each thread handles 8x8 output elements
    float c_reg[8][8] = {0};
    
    // Calculate global output position for this thread
    const int globalRow = by * BLOCK_M + ty * 8;
    const int globalCol = bx * BLOCK_N + tx * 8;
    
    // Main computation loop over K dimension
    for (int kTile = 0; kTile < K; kTile += BLOCK_K) {
        // Cooperatively load tile of A into shared memory
        // Each thread loads multiple elements
        for (int i = tid; i < BLOCK_M * BLOCK_K; i += 256) {
            int row = i / BLOCK_K;
            int col = i % BLOCK_K;
            int globalRowA = by * BLOCK_M + row;
            int globalColA = kTile + col;
            
            if (globalRowA < M && globalColA < K) {
                As[row][col] = A[globalRowA * K + globalColA];
            } else {
                As[row][col] = 0.0f;
            }
        }
        
        // Cooperatively load tile of B into shared memory
        for (int i = tid; i < BLOCK_K * BLOCK_N; i += 256) {
            int row = i / BLOCK_N;
            int col = i % BLOCK_N;
            int globalRowB = kTile + row;
            int globalColB = bx * BLOCK_N + col;
            
            if (globalRowB < K && globalColB < N) {
                Bs[row][col] = B[globalRowB * N + globalColB];
            } else {
                Bs[row][col] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Compute partial products for this thread's 8x8 tile
        #pragma unroll
        for (int k = 0; k < BLOCK_K; ++k) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    c_reg[i][j] += As[ty * 8 + i][k] * Bs[k][tx * 8 + j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Store results to global memory
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            int row = globalRow + i;
            int col = globalCol + j;
            
            if (row < M && col < N) {
                C[row * N + col] = c_reg[i][j];
            }
        }
    }
}

// ============================================================================
// Host Functions
// ============================================================================

void gemm(const __half* A, const __half* B, __half* C, int M, int N, int K, cudaStream_t stream) {
    // Validate inputs
    if (A == nullptr || B == nullptr || C == nullptr) {
        fprintf(stderr, "ERROR: Null pointer passed to baseline::gemm (half)\n");
        return;
    }
    if (M <= 0 || N <= 0 || K <= 0) {
        fprintf(stderr, "ERROR: Invalid dimensions: M=%d, N=%d, K=%d\n", M, N, K);
        return;
    }
    
    // WMMA configuration:
    // - Each warp (32 threads) handles one 16x16 output tile
    // - Use 8x4 = 32 threads per block (1 warp)
    // - Each block handles one 16x16 tile
    dim3 blockDim(32, 1);  // 32 threads = 1 warp
    dim3 gridDim((N + 15) / 16, (M + 15) / 16);  // Each warp handles 16x16 output
    
    // Launch Tensor Core optimized kernel
    gemm_wmma_kernel_simple<<<gridDim, blockDim, 0, stream>>>(A, B, C, M, N, K);
    
    CUDA_CHECK_KERNEL();
}

void gemm(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    // Validate inputs
    if (A == nullptr || B == nullptr || C == nullptr) {
        fprintf(stderr, "ERROR: Null pointer passed to baseline::gemm (float)\n");
        return;
    }
    if (M <= 0 || N <= 0 || K <= 0) {
        fprintf(stderr, "ERROR: Invalid dimensions: M=%d, N=%d, K=%d\n", M, N, K);
        return;
    }
    
    // Tile sizes optimized for register blocking
    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 128;
    constexpr int BLOCK_K = 16;
    
    // Thread block configuration: 16x16 threads = 256 threads
    // Each thread handles 8x8 output elements
    dim3 blockDim(16, 16);
    dim3 gridDim((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    
    // Launch optimized float kernel
    gemm_float_kernel<BLOCK_M, BLOCK_N, BLOCK_K>
        <<<gridDim, blockDim, 0, stream>>>(A, B, C, M, N, K);
    
    CUDA_CHECK_KERNEL();
}

// ============================================================================
// Scaled GEMM Implementations
// ============================================================================

template<typename T>
__global__ void gemm_scaled_kernel(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    int M, int N, int K,
    T alpha, T beta)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        T sum = T(0);
        
        // Compute dot product
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        
        // Apply scaling: C = alpha * A * B + beta * C
        if constexpr (std::is_same_v<T, float>) {
            C[row * N + col] = alpha * sum + beta * C[row * N + col];
        } else {
            // For half precision
            C[row * N + col] = __hadd(__hmul(alpha, sum), __hmul(beta, C[row * N + col]));
        }
    }
}

void gemm_scaled(const float* A, const float* B, float* C, int M, int N, int K,
                 float alpha, float beta, cudaStream_t stream) {
    // Validate inputs
    if (A == nullptr || B == nullptr || C == nullptr) {
        fprintf(stderr, "ERROR: Null pointer passed to baseline::gemm_scaled (float)\n");
        return;
    }
    if (M <= 0 || N <= 0 || K <= 0) {
        fprintf(stderr, "ERROR: Invalid dimensions: M=%d, N=%d, K=%d\n", M, N, K);
        return;
    }
    
    // For simplicity, use basic implementation for scaled GEMM
    // In production, this would be optimized similar to the main GEMM
    dim3 blockDim(16, 16);
    dim3 gridDim((N + 15) / 16, (M + 15) / 16);
    
    gemm_scaled_kernel<float><<<gridDim, blockDim, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    
    CUDA_CHECK_KERNEL();
}

void gemm_scaled(const __half* A, const __half* B, __half* C, int M, int N, int K,
                 __half alpha, __half beta, cudaStream_t stream) {
    // Validate inputs
    if (A == nullptr || B == nullptr || C == nullptr) {
        fprintf(stderr, "ERROR: Null pointer passed to baseline::gemm_scaled (half)\n");
        return;
    }
    if (M <= 0 || N <= 0 || K <= 0) {
        fprintf(stderr, "ERROR: Invalid dimensions: M=%d, N=%d, K=%d\n", M, N, K);
        return;
    }
    
    // For simplicity, use basic implementation for scaled GEMM
    dim3 blockDim(16, 16);
    dim3 gridDim((N + 15) / 16, (M + 15) / 16);
    
    gemm_scaled_kernel<__half><<<gridDim, blockDim, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    
    CUDA_CHECK_KERNEL();
}

} // namespace baseline