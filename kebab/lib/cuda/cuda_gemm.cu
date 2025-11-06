#include "kebab/cuda/cuda_gemm.h"
#include <mma.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <string>

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
/**
 * @brief Naive GEMM kernel for RC mode: A row-major, B column-major
 * Matches cuBLAS behavior: C[m,n] = sum_k A[m,k] * B[k,n]
 * A is stored row-major (M×K): A[m,k] at A[m*K + k]
 * B is stored column-major (K×N): B[k,n] at B[n*K + k]
 * C is stored row-major (M×N): C[m,n] at C[m*N + n]
 */
__global__ void gemm_naive_kernel_rc(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    int M, int N, int K)
{
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m < M && n < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            // A[m,k] at A[m*K + k] (row-major)
            // B[k,n] at B[n*K + k] (column-major: K×N matrix stored column-major)
            sum += __half2float(A[m * K + k]) * __half2float(B[n * K + k]);
        }
        // C[m,n] at C[m*N + n] (row-major)
        C[m * N + n] = __float2half(sum);
    }
}

/**
 * @brief Naive GEMM kernel for CR mode: A column-major, B row-major
 * Matches cuBLAS behavior: C[m,n] = sum_k A[m,k] * B[k,n]
 * A is stored column-major: A[m,k] at A[m + k*M]  (note: column-major means A[m,k] is at A[m + k*M])
 * B is stored row-major: B[k,n] at B[k*N + n]
 * C is stored row-major: C[m,n] at C[m*N + n]
 */
__global__ void gemm_naive_kernel_cr(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    int M, int N, int K)
{
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m < M && n < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            // A[m,k] at A[m + k*M] (column-major)
            // B[k,n] at B[k*N + n] (row-major)
            sum += __half2float(A[m + k * M]) * __half2float(B[k * N + n]);
        }
        // C[m,n] at C[m*N + n] (row-major)
        C[m * N + n] = __float2half(sum);
    }
}

/**
 * @brief Optimized GEMM kernel for float precision aligned with cuBLAS NT
 *
 * Computes: C[n,m] = sum_k B[k,n] * A[m,k]
 * Stored at: C[n + m*N]
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
    __shared__ float Bs[BLOCK_K][BLOCK_N + 4]; // B is K×N, transposed in smem

    // Register arrays for accumulation - each thread handles 8x8 output elements
    float c_reg[8][8] = {0};

    // Calculate global output position for this thread
    // Note: globalRow is m index, globalCol is n index
    const int globalRowM = by * BLOCK_M + ty * 8;
    const int globalColN = bx * BLOCK_N + tx * 8;

    // Main computation loop over K dimension
    for (int kTile = 0; kTile < K; kTile += BLOCK_K) {
        // Cooperatively load tile of A into shared memory
        // A[m,k] at A[m*K + k]
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
        // B[k,n] at B[k*N + n]
        for (int i = tid; i < BLOCK_K * BLOCK_N; i += 256) {
            int row = i / BLOCK_N;  // k index
            int col = i % BLOCK_N;  // n index
            int globalRowB = kTile + row;         // k
            int globalColB = bx * BLOCK_N + col;  // n

            if (globalRowB < K && globalColB < N) {
                Bs[row][col] = B[globalRowB * N + globalColB];
            } else {
                Bs[row][col] = 0.0f;
            }
        }

        __syncthreads();

        // Compute partial products for this thread's 8x8 tile
        // C[n,m] = sum_k B[k,n] * A[m,k]
        #pragma unroll
        for (int k = 0; k < BLOCK_K; ++k) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {  // m index
                #pragma unroll
                for (int j = 0; j < 8; ++j) {  // n index
                    c_reg[i][j] += Bs[k][tx * 8 + j] * As[ty * 8 + i][k];
                }
            }
        }

        __syncthreads();
    }

    // Store results to global memory
    // C[n,m] at C[n + m*N]
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            int m = globalRowM + i;
            int n = globalColN + j;

            if (m < M && n < N) {
                C[n + m * N] = c_reg[i][j];
            }
        }
    }
}

// ============================================================================
// Host Functions
// ============================================================================

void gemm(const __half* A, const __half* B, __half* C, int M, int N, int K, const char* opmode, int version, cudaStream_t stream) {
    // Validate inputs
    if (A == nullptr || B == nullptr || C == nullptr) {
        fprintf(stderr, "ERROR: Null pointer passed to baseline::gemm (half)\n");
        return;
    }
    if (M <= 0 || N <= 0 || K <= 0) {
        fprintf(stderr, "ERROR: Invalid dimensions: M=%d, N=%d, K=%d\n", M, N, K);
        return;
    }
    
    // Parse opmode to determine storage formats
    std::string opmode_str(opmode ? opmode : "RR");
    char lhs_format = (opmode_str.length() >= 1) ? opmode_str[0] : 'R';
    char rhs_format = (opmode_str.length() >= 2) ? opmode_str[1] : 'R';
    
    // Version dispatch
    switch (version) {
        case 1: {
            // Version 1: Naive implementation
            dim3 blockDim(16, 16);
            dim3 gridDim((N + 15) / 16, (M + 15) / 16);
            
            if (lhs_format == 'R' && rhs_format == 'C') {
                gemm_naive_kernel_rc<<<gridDim, blockDim, 0, stream>>>(A, B, C, M, N, K);
            } else if (lhs_format == 'C' && rhs_format == 'R') {
                gemm_naive_kernel_cr<<<gridDim, blockDim, 0, stream>>>(A, B, C, M, N, K);
            } else {
                fprintf(stderr, "ERROR: Unsupported opmode '%s' for CUDA baseline\n", opmode);
                fprintf(stderr, "       Only RC and CR modes are supported\n");
                return;
            }
            break;
        }
        default:
            fprintf(stderr, "ERROR: Unsupported CUDA baseline version %d\n", version);
            fprintf(stderr, "       Available versions: 1 (naive)\n");
            return;
    }
    
    CUDA_CHECK_KERNEL();
}

void gemm(const float* A, const float* B, float* C, int M, int N, int K, const char* opmode, int version, cudaStream_t stream) {
    // Validate inputs
    if (A == nullptr || B == nullptr || C == nullptr) {
        fprintf(stderr, "ERROR: Null pointer passed to baseline::gemm (float)\n");
        return;
    }
    if (M <= 0 || N <= 0 || K <= 0) {
        fprintf(stderr, "ERROR: Invalid dimensions: M=%d, N=%d, K=%d\n", M, N, K);
        return;
    }
    
    fprintf(stderr, "ERROR: Float precision CUDA baseline not implemented\n");
    fprintf(stderr, "       Only half precision is supported for CUDA baseline\n");
}

// ============================================================================
// Scaled GEMM Implementations
// ============================================================================

/**
 * @brief Scaled GEMM kernel aligned with cuBLAS NT
 *
 * Computes: C[n,m] = alpha * sum_k B[k,n] * A[m,k] + beta * C[n,m]
 * Stored at: C[n + m*N]
 */
template<typename T>
__global__ void gemm_scaled_kernel(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    int M, int N, int K,
    T alpha, T beta)
{
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m < M && n < N) {
        T sum = T(0);

        // Compute: sum_k B[k,n] * A[m,k]
        for (int k = 0; k < K; ++k) {
            sum += B[k * N + n] * A[m * K + k];
        }

        // Apply scaling: C = alpha * sum + beta * C
        int idx = n + m * N;
        if constexpr (std::is_same_v<T, float>) {
            C[idx] = alpha * sum + beta * C[idx];
        } else {
            // For half precision
            C[idx] = __hadd(__hmul(alpha, sum), __hmul(beta, C[idx]));
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