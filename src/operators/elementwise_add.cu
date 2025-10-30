/**
 * @file elementwise_add.cu
 * @brief CuTe implementation of element-wise addition
 * 
 * This implementation uses CuTe's Layout and Tensor abstractions for
 * type-safe, dimension-aware element-wise operations.
 */

#include "cutekernellib/operators/elementwise_add.h"
#include <cute/tensor.hpp>

using namespace cute;

namespace cutekernellib {

/**
 * @brief CuTe element-wise addition kernel
 * 
 * Uses CuTe features:
 * - Layout for memory pattern description
 * - Tensor for multi-dimensional array abstraction
 * - Automatic vectorization through layout
 */
template<typename T>
__global__ void elementwise_add_kernel_cute(
    const T* A_ptr, const T* B_ptr, T* C_ptr, int N)
{
    // Create 1D layout for the arrays
    auto layout = make_layout(make_shape(N));
    
    // Create CuTe tensors from raw pointers
    Tensor gA = make_tensor(make_gmem_ptr(A_ptr), layout);
    Tensor gB = make_tensor(make_gmem_ptr(B_ptr), layout);
    Tensor gC = make_tensor(make_gmem_ptr(C_ptr), layout);
    
    // Calculate thread's work range
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    // Process elements assigned to this thread
    // CuTe's tensor access handles bounds checking and layout
    for (int i = tid; i < N; i += stride) {
        if (i < size(gC)) {
            gC(i) = gA(i) + gB(i);
        }
    }
}

/**
 * @brief CuTe vectorized element-wise addition kernel
 * 
 * Uses CuTe's copy atoms for automatic vectorization
 */
template<typename T, int VecSize>
__global__ void elementwise_add_kernel_cute_vectorized(
    const T* A_ptr, const T* B_ptr, T* C_ptr, int N)
{
    // Create layout with vectorization hint
    auto layout = make_layout(make_shape(N / VecSize));
    
    // Create tensors with vectorized access pattern
    using VecType = cute::array<T, VecSize>;
    Tensor gA = make_tensor(make_gmem_ptr(reinterpret_cast<const VecType*>(A_ptr)), layout);
    Tensor gB = make_tensor(make_gmem_ptr(reinterpret_cast<const VecType*>(B_ptr)), layout);
    Tensor gC = make_tensor(make_gmem_ptr(reinterpret_cast<VecType*>(C_ptr)), layout);
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    // Process vectorized elements
    for (int i = tid; i < size(gC); i += stride) {
        VecType a = gA(i);
        VecType b = gB(i);
        VecType c;
        
        #pragma unroll
        for (int j = 0; j < VecSize; ++j) {
            c[j] = a[j] + b[j];
        }
        
        gC(i) = c;
    }
}

/**
 * @brief Host function for float element-wise addition
 */
template<>
void elementwise_add<float>(const float* A, const float* B, float* C, int N, cudaStream_t stream) {
    if (A == nullptr || B == nullptr || C == nullptr) {
        fprintf(stderr, "ERROR: Null pointer passed to elementwise_add\n");
        return;
    }
    if (N <= 0) {
        fprintf(stderr, "ERROR: Invalid size: N=%d\n", N);
        return;
    }
    
    constexpr int VecSize = 4;  // float4 vectorization
    constexpr int threads_per_block = 256;
    
    // Process vectorized portion
    int vec_elements = (N / VecSize) * VecSize;
    if (vec_elements > 0) {
        int vec_count = vec_elements / VecSize;
        int num_blocks = (vec_count + threads_per_block - 1) / threads_per_block;
        
        elementwise_add_kernel_cute_vectorized<float, VecSize>
            <<<num_blocks, threads_per_block, 0, stream>>>(A, B, C, vec_elements);
        
        CUDA_CHECK(cudaGetLastError());
    }
    
    // Process remainder with scalar kernel
    int remainder = N - vec_elements;
    if (remainder > 0) {
        int num_blocks = 1;
        
        elementwise_add_kernel_cute
            <<<num_blocks, threads_per_block, 0, stream>>>(
                A + vec_elements, B + vec_elements, C + vec_elements, remainder);
        
        CUDA_CHECK(cudaGetLastError());
    }
}

/**
 * @brief Host function for half element-wise addition
 */
template<>
void elementwise_add<__half>(const __half* A, const __half* B, __half* C, int N, cudaStream_t stream) {
    if (A == nullptr || B == nullptr || C == nullptr) {
        fprintf(stderr, "ERROR: Null pointer passed to elementwise_add\n");
        return;
    }
    if (N <= 0) {
        fprintf(stderr, "ERROR: Invalid size: N=%d\n", N);
        return;
    }
    
    constexpr int VecSize = 2;  // half2 vectorization
    constexpr int threads_per_block = 256;
    
    // Process vectorized portion
    int vec_elements = (N / VecSize) * VecSize;
    if (vec_elements > 0) {
        int vec_count = vec_elements / VecSize;
        int num_blocks = (vec_count + threads_per_block - 1) / threads_per_block;
        
        elementwise_add_kernel_cute_vectorized<__half, VecSize>
            <<<num_blocks, threads_per_block, 0, stream>>>(A, B, C, vec_elements);
        
        CUDA_CHECK(cudaGetLastError());
    }
    
    // Process remainder with scalar kernel
    int remainder = N - vec_elements;
    if (remainder > 0) {
        int num_blocks = 1;
        
        elementwise_add_kernel_cute
            <<<num_blocks, threads_per_block, 0, stream>>>(
                A + vec_elements, B + vec_elements, C + vec_elements, remainder);
        
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace cutekernellib
