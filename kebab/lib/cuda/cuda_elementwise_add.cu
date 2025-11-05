#include "kebab/cuda/cuda_elementwise_add.h"
#include <cuda_runtime.h>
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

/**
 * @brief Hand-optimized CUDA kernel for element-wise addition using vectorized loads/stores
 * 
 * Optimizations:
 * - Vectorized memory access (float4 for float, half2 for half)
 * - Grid-stride loop for handling arbitrary sizes
 * - Proper block sizing (256 threads)
 * - Memory coalescing
 */
template<typename T, typename VecT, int VEC_SIZE>
__global__ void elementwise_add_kernel_vectorized(const T* __restrict__ A, 
                                                   const T* __restrict__ B, 
                                                   T* __restrict__ C, 
                                                   int N) {
    // Grid-stride loop for better scalability
    int vec_elements = N / VEC_SIZE;
    
    for (int vec_idx = blockIdx.x * blockDim.x + threadIdx.x; 
         vec_idx < vec_elements; 
         vec_idx += blockDim.x * gridDim.x) {
        
        // Vectorized load
        VecT a = reinterpret_cast<const VecT*>(A)[vec_idx];
        VecT b = reinterpret_cast<const VecT*>(B)[vec_idx];
        VecT c;
        
        // Perform addition based on vector type
        if constexpr (sizeof(VecT) == sizeof(float4)) {
            c.x = a.x + b.x;
            c.y = a.y + b.y;
            c.z = a.z + b.z;
            c.w = a.w + b.w;
        } else if constexpr (sizeof(VecT) == sizeof(half2)) {
            c = __hadd2(a, b);
        }
        
        // Vectorized store
        reinterpret_cast<VecT*>(C)[vec_idx] = c;
    }
}

/**
 * @brief Scalar kernel for handling remainder elements
 */
template<typename T>
__global__ void elementwise_add_kernel_scalar(const T* __restrict__ A, 
                                               const T* __restrict__ B, 
                                               T* __restrict__ C, 
                                               int N, 
                                               int start_idx) {
    // Grid-stride loop
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x + start_idx; 
         idx < N; 
         idx += blockDim.x * gridDim.x) {
        
        if constexpr (sizeof(T) == sizeof(float)) {
            C[idx] = A[idx] + B[idx];
        } else if constexpr (sizeof(T) == sizeof(__half)) {
            C[idx] = __hadd(A[idx], B[idx]);
        }
    }
}

/**
 * @brief Element-wise addition for float arrays
 * 
 * @param A Input array A
 * @param B Input array B
 * @param C Output array C = A + B
 * @param N Number of elements
 * @param stream CUDA stream for asynchronous execution
 */
void elementwise_add(const float* A, const float* B, float* C, int N, cudaStream_t stream) {
    // Validate inputs
    if (A == nullptr || B == nullptr || C == nullptr) {
        fprintf(stderr, "ERROR: Null pointer passed to baseline::elementwise_add\n");
        return;
    }
    if (N <= 0) {
        fprintf(stderr, "ERROR: Invalid size: N=%d\n", N);
        return;
    }
    
    // Optimal block size for modern GPUs
    const int threads_per_block = 256;
    
    // Process vectorized portion (4 elements per thread using float4)
    constexpr int VEC_SIZE = 4;
    int vec_elements = (N / VEC_SIZE) * VEC_SIZE;
    int vec_threads = N / VEC_SIZE;
    
    if (vec_threads > 0) {
        // Calculate grid size with proper occupancy
        int num_blocks = (vec_threads + threads_per_block - 1) / threads_per_block;
        
        elementwise_add_kernel_vectorized<float, float4, VEC_SIZE>
            <<<num_blocks, threads_per_block, 0, stream>>>(A, B, C, N);
        CUDA_CHECK(cudaGetLastError());
    }
    
    // Handle remainder elements with scalar kernel
    int remainder = N - vec_elements;
    if (remainder > 0) {
        int num_blocks = (remainder + threads_per_block - 1) / threads_per_block;
        
        elementwise_add_kernel_scalar<float>
            <<<num_blocks, threads_per_block, 0, stream>>>(A, B, C, N, vec_elements);
        CUDA_CHECK(cudaGetLastError());
    }
}

/**
 * @brief Element-wise addition for half arrays
 * 
 * @param A Input array A
 * @param B Input array B
 * @param C Output array C = A + B
 * @param N Number of elements
 * @param stream CUDA stream for asynchronous execution
 */
void elementwise_add(const __half* A, const __half* B, __half* C, int N, cudaStream_t stream) {
    // Validate inputs
    if (A == nullptr || B == nullptr || C == nullptr) {
        fprintf(stderr, "ERROR: Null pointer passed to baseline::elementwise_add\n");
        return;
    }
    if (N <= 0) {
        fprintf(stderr, "ERROR: Invalid size: N=%d\n", N);
        return;
    }
    
    // Optimal block size for modern GPUs
    const int threads_per_block = 256;
    
    // Process vectorized portion (2 elements per thread using half2)
    constexpr int VEC_SIZE = 2;
    int vec_elements = (N / VEC_SIZE) * VEC_SIZE;
    int vec_threads = N / VEC_SIZE;
    
    if (vec_threads > 0) {
        // Calculate grid size with proper occupancy
        int num_blocks = (vec_threads + threads_per_block - 1) / threads_per_block;
        
        elementwise_add_kernel_vectorized<__half, half2, VEC_SIZE>
            <<<num_blocks, threads_per_block, 0, stream>>>(A, B, C, N);
        CUDA_CHECK(cudaGetLastError());
    }
    
    // Handle remainder elements with scalar kernel
    int remainder = N - vec_elements;
    if (remainder > 0) {
        int num_blocks = (remainder + threads_per_block - 1) / threads_per_block;
        
        elementwise_add_kernel_scalar<__half>
            <<<num_blocks, threads_per_block, 0, stream>>>(A, B, C, N, vec_elements);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace baseline
