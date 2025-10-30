#include "cutekernellib/operators/elementwise_add.h"

namespace cutekernellib {

// Kernel for float using float4 vectorization
template<typename T>
__global__ void elementwise_add_kernel_vectorized(const T* A, const T* B, T* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process 4 elements per thread for float
    if constexpr (sizeof(T) == sizeof(float)) {
        int vec_idx = idx * 4;
        if (vec_idx + 3 < N) {
            // Vectorized load using float4
            float4 a = reinterpret_cast<const float4*>(A)[idx];
            float4 b = reinterpret_cast<const float4*>(B)[idx];
            
            float4 c;
            c.x = a.x + b.x;
            c.y = a.y + b.y;
            c.z = a.z + b.z;
            c.w = a.w + b.w;
            
            // Vectorized store
            reinterpret_cast<float4*>(C)[idx] = c;
        }
    }
    // Process 2 elements per thread for half
    else if constexpr (sizeof(T) == sizeof(__half)) {
        int vec_idx = idx * 2;
        if (vec_idx + 1 < N) {
            // Vectorized load using half2
            half2 a = reinterpret_cast<const half2*>(A)[idx];
            half2 b = reinterpret_cast<const half2*>(B)[idx];
            
            // Use half2 arithmetic
            half2 c = __hadd2(a, b);
            
            // Vectorized store
            reinterpret_cast<half2*>(C)[idx] = c;
        }
    }
}

// Scalar kernel for handling remainder elements
template<typename T>
__global__ void elementwise_add_kernel_scalar(const T* A, const T* B, T* C, int N, int start_idx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + start_idx;
    
    if (idx < N) {
        if constexpr (sizeof(T) == sizeof(float)) {
            C[idx] = A[idx] + B[idx];
        } else if constexpr (sizeof(T) == sizeof(__half)) {
            C[idx] = __hadd(A[idx], B[idx]);
        }
    }
}

// Host function implementation for float
template<>
void elementwise_add<float>(const float* A, const float* B, float* C, int N, cudaStream_t stream) {
    // Validate inputs
    if (A == nullptr || B == nullptr || C == nullptr) {
        fprintf(stderr, "ERROR: Null pointer passed to elementwise_add\n");
        return;
    }
    if (N <= 0) {
        fprintf(stderr, "ERROR: Invalid size: N=%d\n", N);
        return;
    }
    
    // Calculate vectorized portion (process 4 elements per thread)
    int vec_elements = (N / 4) * 4;
    int vec_threads = N / 4;
    
    if (vec_threads > 0) {
        int threads_per_block = 256;
        int num_blocks = (vec_threads + threads_per_block - 1) / threads_per_block;
        
        elementwise_add_kernel_vectorized<float><<<num_blocks, threads_per_block, 0, stream>>>(
            A, B, C, N
        );
        CUDA_CHECK(cudaGetLastError());
    }
    
    // Handle remainder elements with scalar kernel
    int remainder = N - vec_elements;
    if (remainder > 0) {
        int threads_per_block = 256;
        int num_blocks = (remainder + threads_per_block - 1) / threads_per_block;
        
        elementwise_add_kernel_scalar<float><<<num_blocks, threads_per_block, 0, stream>>>(
            A, B, C, N, vec_elements
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

// Host function implementation for half
template<>
void elementwise_add<__half>(const __half* A, const __half* B, __half* C, int N, cudaStream_t stream) {
    // Validate inputs
    if (A == nullptr || B == nullptr || C == nullptr) {
        fprintf(stderr, "ERROR: Null pointer passed to elementwise_add\n");
        return;
    }
    if (N <= 0) {
        fprintf(stderr, "ERROR: Invalid size: N=%d\n", N);
        return;
    }
    
    // Calculate vectorized portion (process 2 elements per thread)
    int vec_elements = (N / 2) * 2;
    int vec_threads = N / 2;
    
    if (vec_threads > 0) {
        int threads_per_block = 256;
        int num_blocks = (vec_threads + threads_per_block - 1) / threads_per_block;
        
        elementwise_add_kernel_vectorized<__half><<<num_blocks, threads_per_block, 0, stream>>>(
            A, B, C, N
        );
        CUDA_CHECK(cudaGetLastError());
    }
    
    // Handle remainder elements with scalar kernel
    int remainder = N - vec_elements;
    if (remainder > 0) {
        int threads_per_block = 256;
        int num_blocks = (remainder + threads_per_block - 1) / threads_per_block;
        
        elementwise_add_kernel_scalar<__half><<<num_blocks, threads_per_block, 0, stream>>>(
            A, B, C, N, vec_elements
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace cutekernellib
